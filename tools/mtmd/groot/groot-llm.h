// llama.cpp-hosted GR00T prefill.
//
// The GR00T GGUF is a feature-extractor header: it carries the token table and
// the metadata, while the 16 decoder layers live in a packed INT8 weight set.
// llama.cpp still owns the model, tokenizer and llama_decode; the graph builder
// routes the decode graph through the private prefill kernel
// (src/models/qwen3vl.cpp + groot_fused_callback). That is the same split the
// baseline runtime uses.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

struct llama_model;
struct llama_context;
struct llama_vocab;
namespace groot_fused {
struct Decoder;
}

class GrootLlm {
public:
    GrootLlm();
    ~GrootLlm();
    GrootLlm(const GrootLlm &) = delete;
    GrootLlm & operator=(const GrootLlm &) = delete;

    bool load(const std::string & gguf, const std::string & weights_dir, int threads, int capacity,
              std::string * error);
    void close();

    int n_embd() const { return n_embd_; }
    int capacity() const { return capacity_; }
    bool fused() const { return fused_ != nullptr; }

    const std::string & error() const { return error_; }

    // Tokenizer surface, no vocabulary projection anywhere in this path.
    int tokenize(const std::string & text, int32_t * out, int capacity) const;
    int image_token_id() const;
    bool embed_tokens(const int32_t * tokens, int n, uint16_t * out);

    // emb: fp16 [S,2048]; deepstack: fp16 [3,V,2048]; mask: [S];
    // positions: int32 [3,S] (mrope t,h,w). Output: fp16 [S,2048].
    bool run(const uint16_t * emb, const uint16_t * deepstack, const uint8_t * visual_mask,
             const int32_t * positions, int sequence, int visual_tokens, uint16_t * hidden,
             double * elapsed_ms);

private:
    llama_model * model_ = nullptr;
    llama_context * context_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    std::unique_ptr<groot_fused::Decoder> fused_;
    std::string error_;
    int n_embd_ = 0;
    int capacity_ = 0;
};
