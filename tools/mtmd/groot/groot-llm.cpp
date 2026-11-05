#include "groot-llm.h"

#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "llm_fused.inc"

using Half = _Float16;

namespace {
// The model stores this as the callback user pointer; it times the fused prefill
// kernel and then delegates to the baseline's callback.
void timed_callback(ggml_tensor * dst, const ggml_tensor * input, const ggml_tensor * pos, int ith,
                    int nth, void * opaque) {
    auto * self = static_cast<GrootLlm *>(opaque);
    const auto t0 = std::chrono::steady_clock::now();
    groot_fused::Decoder::callback(dst, input, pos, ith, nth, self->fused_decoder());
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ith == 0) {
        self->add_kernel_ms(ms);
    }
}
}  // namespace

GrootLlm::GrootLlm() = default;

GrootLlm::~GrootLlm() {
    close();
}

void GrootLlm::close() {
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
    fused_.reset();
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
}

bool GrootLlm::load(const std::string & gguf, const std::string & weights_dir, int threads,
                    int capacity, int n_gpu_layers, std::string * error) {
    auto fail = [&](const std::string & message) {
        error_ = message;
        if (error) {
            *error = message;
        }
        close();
        return false;
    };
    if (threads < 1 || threads > 8 || capacity < 1 || capacity > 4096) {
        return fail("LLM config: threads must be 1..8 and capacity 1..4096");
    }
    capacity_ = capacity;

    auto model_params = llama_model_default_params();
    // 0 keeps every layer on the host; a large value offloads to the first
    // accelerator backend (the SpacemiT device when GGML_SPACEMIT is built in).
    model_params.n_gpu_layers = n_gpu_layers;
    model_ = llama_model_load_from_file(gguf.c_str(), model_params);
    if (!model_) {
        return fail("llama_model_load_from_file failed: " + gguf);
    }
    if (llama_model_n_embd(model_) != 2048 || llama_model_n_embd_inp(model_) != 8192) {
        return fail("expected E2048 with three deepstack inputs");
    }
    n_embd_ = llama_model_n_embd(model_);

    if (model_->groot_fused_prefill_only) {
        if (capacity > 512) {
            return fail("fused prefill capacity exceeds 512");
        }
        try {
            fused_ = std::make_unique<groot_fused::Decoder>(weights_dir, threads, capacity);
        } catch (const std::exception & e) {
            return fail(std::string("fused prefill init: ") + e.what());
        }
        model_->groot_fused_callback = timed_callback;
        model_->groot_fused_user = this;
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = capacity;
    ctx_params.n_batch = capacity;
    ctx_params.n_ubatch = capacity;
    ctx_params.n_threads = fused_ ? 1 : threads;
    ctx_params.n_threads_batch = fused_ ? 1 : threads;
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.no_perf = true;
    context_ = llama_init_from_model(model_, ctx_params);
    if (!context_) {
        return fail("llama context creation failed");
    }
    vocab_ = llama_model_get_vocab(model_);
    return true;
}

int GrootLlm::tokenize(const std::string & text, int32_t * out, int capacity) const {
    if (!vocab_ || !out) {
        return -1;
    }
    return llama_tokenize(vocab_, text.data(), int(text.size()), out, capacity,
                          /*add_special=*/false, /*parse_special=*/true);
}

int GrootLlm::image_token_id() const {
    int32_t token = -1;
    if (tokenize("<|image_pad|>", &token, 1) != 1) {
        return -1;
    }
    return token;
}

bool GrootLlm::embed_tokens(const int32_t * tokens, int n, uint16_t * out) {
    if (!model_ || !tokens || !out || n < 1 || n > capacity_) {
        error_ = "token embedding inputs";
        return false;
    }
    const ggml_tensor * table = model_->tok_embd;
    if (!table || table->type != GGML_TYPE_F16 || table->ne[0] != 2048) {
        error_ = "F16 GGUF token table required";
        return false;
    }
    for (int r = 0; r < n; ++r) {
        if (tokens[r] < 0 || tokens[r] >= table->ne[1]) {
            error_ = "token id out of range";
            return false;
        }
        ggml_backend_tensor_get(table, out + size_t(r) * 2048, size_t(tokens[r]) * table->nb[1],
                                2048 * 2);
    }
    return true;
}

bool GrootLlm::run(const uint16_t * emb, const uint16_t * deepstack, const uint8_t * visual_mask,
                   const int32_t * positions, int sequence, int visual_tokens, uint16_t * hidden,
                   double * elapsed_ms) {
    if (!context_ || !emb || !deepstack || !visual_mask || !positions || !hidden || sequence < 1 ||
        sequence > capacity_ || visual_tokens < 1 || visual_tokens > sequence) {
        error_ = "LLM inputs";
        return false;
    }
    std::vector<float> input(size_t(sequence) * 8192);
    std::vector<llama_pos> pos4(size_t(sequence) * 4);

    int seen = 0;
    for (int r = 0; r < sequence; ++r) {
        for (int c = 0; c < 2048; ++c) {
            input[size_t(r) * 8192 + c] = float(((const Half *) emb)[size_t(r) * 2048 + c]);
        }
        if (visual_mask[r]) {
            for (int d = 0; d < 3; ++d) {
                const Half * source = (const Half *) deepstack + (size_t(d) * visual_tokens + seen) * 2048;
                for (int c = 0; c < 2048; ++c) {
                    input[size_t(r) * 8192 + (d + 1) * 2048 + c] = float(source[c]);
                }
            }
            ++seen;
        }
        for (int d = 0; d < 3; ++d) {
            pos4[size_t(d) * sequence + r] = positions[size_t(d) * sequence + r];
        }
        pos4[size_t(3) * sequence + r] = positions[r];
    }
    if (seen != visual_tokens) {
        error_ = "visual token count mismatch";
        return false;
    }

    llama_memory_clear(llama_get_memory(context_), false);

    // Embedding-only batch: a token array would make every entry carry a bogus
    // token id next to its embedding, which the batch validator rejects.
    llama_batch batch = {};
    batch.n_tokens = sequence;
    batch.token    = nullptr;
    batch.embd     = input.data();
    batch.pos      = pos4.data();

    kernel_ms_ = 0.0;
    const auto start = std::chrono::steady_clock::now();
    const int rc = llama_decode(context_, batch);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    last_decode_ms_ = ms;
    last_kernel_ms_ = kernel_ms_;

    if (rc != 0) {
        error_ = "llama_decode failed: " + std::to_string(rc);
        return false;
    }
    if (fused_ && !fused_->error.empty()) {
        error_ = fused_->error;
        return false;
    }
    for (int r = 0; r < sequence; ++r) {
        const float * row = llama_get_embeddings_ith(context_, r);
        if (!row) {
            error_ = "missing llama hidden state";
            return false;
        }
        for (int c = 0; c < 2048; ++c) {
            ((Half *) hidden)[size_t(r) * 2048 + c] = Half(row[c]);
        }
    }
    if (elapsed_ms) {
        *elapsed_ms = ms;
    }
    return true;
}
