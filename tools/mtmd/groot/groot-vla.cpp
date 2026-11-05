// GR00T N1.7 LIBERO end-to-end pipeline on llama.cpp + mtmd + SpacemiT EP.
//
//   images ──> ViT (ONNX/EP) ──> image embeds + deepstack ─┐
//   text ────> llama tokenizer ──> token embeds ───────────┴─> llama prefill ─> hidden
//   hidden ──> VLLN ──> VL (EP) ──> Cross-KV (EP) ──┐
//   state ──> state_encoder (EP) ───────────────────┴─> DiT x4 (EP) ──> actions
//
// llama.cpp owns the LLM backbone (model, tokenizer, llama_decode). Everything
// else is an external ONNX graph executed by the SpacemiT plugin EP; mtmd is the
// multimodal front-end for image decoding.
#include "groot-ep.h"
#include "groot-llm.h"
#include "groot-preprocess.h"

#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

extern "C" {
int  groot_layer_norm(const uint16_t * input, const uint16_t * weight, const uint16_t * bias,
                      uint16_t * output, size_t rows, size_t width, float eps);
void groot_add_fp16(const uint16_t * a, const uint16_t * b, uint16_t * y, size_t n);
void groot_euler_fp16(uint16_t * actions, const uint16_t * velocity, size_t n);
}

namespace {

using Clock = std::chrono::steady_clock;

double since_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// One thread owns llama_decode. The fused prefill kernel and the spert worker set
// it spawns inherit this thread's affinity, so it must be the AI cluster while the
// orchestrating thread stays on the general-purpose cores.
class AiThread {
public:
    AiThread() {
        thread_ = std::thread([this] { loop(); });
    }
    ~AiThread() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    template <class F>
    auto call(F && fn) -> decltype(fn()) {
        using Result = decltype(fn());
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(fn));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = [task] { (*task)(); };
            has_job_ = true;
        }
        cv_.notify_one();
        return future.get();
    }

private:
    void loop() {
#if defined(__linux__)
        // Allow the whole AI cluster: the prefill kernel's workers inherit this
        // mask, so pinning to a single core would serialize them.
        const char * env = std::getenv("GROOT_AI_CPUS");
        const std::string cpus = env ? env : std::string("8-15");
        bool flagged = false;
        {
            std::ofstream flag("/proc/set_ai_thread");
            if (flag) {
                flag << "0\n";
                flag.flush();
                flagged = bool(flag);
            }
        }
        if (!flagged) {
            std::fprintf(stderr, "[groot] cannot write /proc/set_ai_thread\n");
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        {
            // "8-15" or a comma separated list.
            const char * c = cpus.c_str();
            while (*c) {
                char * end = nullptr;
                long lo = std::strtol(c, &end, 10);
                long hi = lo;
                c = end;
                if (*c == '-') {
                    hi = std::strtol(c + 1, &end, 10);
                    c = end;
                }
                for (long cpu = lo; cpu <= hi && cpu < CPU_SETSIZE; ++cpu) {
                    CPU_SET(int(cpu), &set);
                }
                if (*c == ',') {
                    ++c;
                }
            }
        }
        // The VLEN switch is applied on the next scheduling point, so the pin can
        // briefly race with it.
        bool pinned = false;
        for (int attempt = 0; attempt < 200 && !pinned; ++attempt) {
            pinned = sched_setaffinity(0, sizeof(set), &set) == 0;
            if (!pinned) {
                usleep(1000);
            }
        }
        if (!pinned) {
            std::fprintf(stderr, "[groot] cannot pin the LLM thread to %s\n", cpus.c_str());
        }
#endif
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return has_job_ || stop_; });
                if (stop_) {
                    return;
                }
                job = std::move(job_);
                has_job_ = false;
            }
            job();
        }
    }

    std::thread                thread_;
    std::mutex                 mutex_;
    std::condition_variable    cv_;
    std::function<void()>      job_;
    bool                       has_job_ = false;
    bool                       stop_    = false;
};

struct Contract {
    float state_lo[8] = {};
    float state_hi[8] = {};
    float action_lo[7] = {};  // x y z roll pitch yaw gripper
    float action_hi[7] = {};
    std::vector<uint16_t> vlln_weight;   // 2048
    std::vector<uint16_t> vlln_bias;     // 2048
    std::vector<uint16_t> time_embedding;  // 4 x 40 x 1536
    std::vector<uint16_t> position_embedding;  // 40 x 1536
    std::vector<uint16_t> adaln;         // 4 x 32 x 3072
    std::vector<uint16_t> output_mod;    // 4 x 3072
};

bool read_exact(const std::string & path, void * data, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[groot] cannot open %s\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char *>(data), std::streamsize(bytes));
    if (size_t(f.gcount()) != bytes) {
        std::fprintf(stderr, "[groot] short read %s (%zu of %zu bytes)\n", path.c_str(),
                     size_t(f.gcount()), bytes);
        return false;
    }
    return true;
}

bool read_contract(const std::string & path, Contract * contract, std::string * error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *error = "cannot open contract " + path;
        return false;
    }
    char magic[8] = {};
    int32_t version = 0;
    f.read(magic, 8);
    f.read(reinterpret_cast<char *>(&version), 4);
    if (std::memcmp(magic, "GRTCTR01", 8) != 0 || version != 1) {
        *error = "bad contract header";
        return false;
    }
    auto read_vector = [&](std::vector<uint16_t> & v, size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char *>(v.data()), std::streamsize(n * 2));
    };
    f.read(reinterpret_cast<char *>(contract->state_lo), sizeof(contract->state_lo));
    f.read(reinterpret_cast<char *>(contract->state_hi), sizeof(contract->state_hi));
    f.read(reinterpret_cast<char *>(contract->action_lo), sizeof(contract->action_lo));
    f.read(reinterpret_cast<char *>(contract->action_hi), sizeof(contract->action_hi));
    read_vector(contract->vlln_weight, 2048);
    read_vector(contract->vlln_bias, 2048);
    read_vector(contract->time_embedding, 4 * 40 * 1536);
    read_vector(contract->position_embedding, 40 * 1536);
    read_vector(contract->adaln, 4 * 32 * 3072);
    read_vector(contract->output_mod, 4 * 3072);
    if (!f) {
        *error = "truncated contract " + path;
        return false;
    }
    return true;
}

struct Layout {
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;  // 3 x S
    std::vector<uint8_t> visual;
};

bool build_layout(GrootLlm & llm, const std::string & instruction, Layout * layout,
                  std::string * error) {
    const std::string prompt = groot::build_prompt(groot::clean_instruction(instruction));
    std::vector<int32_t> tokens(4096);
    const int n = llm.tokenize(prompt, tokens.data(), int(tokens.size()));
    if (n < 1) {
        *error = "tokenize failed or prompt exceeds capacity";
        return false;
    }
    tokens.resize(n);
    const int image_token = llm.image_token_id();
    if (image_token < 0) {
        *error = "cannot tokenize the image placeholder";
        return false;
    }
    layout->tokens = tokens;
    layout->visual.assign(n, 0);
    layout->positions.assign(size_t(3) * n, 0);

    int offset = 0;
    int images = 0;
    for (int i = 0; i < n;) {
        if (tokens[i] != image_token) {
            layout->positions[i] = offset;
            layout->positions[size_t(n) + i] = offset;
            layout->positions[size_t(2) * n + i] = offset;
            ++offset;
            ++i;
            continue;
        }
        if (i + 64 > n) {
            *error = "truncated visual span";
            return false;
        }
        for (int j = 0; j < 64; ++j) {
            if (tokens[i + j] != image_token) {
                *error = "visual span is not 64 consecutive image tokens";
                return false;
            }
            layout->visual[i + j] = 1;
            layout->positions[i + j] = offset;
            layout->positions[size_t(n) + i + j] = offset + j / 8;
            layout->positions[size_t(2) * n + i + j] = offset + j % 8;
        }
        offset += 8;
        i += 64;
        ++images;
    }
    if (images != 2) {
        *error = "expected exactly two visual spans, got " + std::to_string(images);
        return false;
    }
    return true;
}

struct Options {
    std::string model_dir;
    std::string aux_dir;
    std::string gguf;
    std::string llm_weights;
    std::string ep_lib;
    std::string contract;
    std::string case_dir;
    std::string out_dir;
    int         threads  = 8;
    int         repeats  = 1;
    bool        use_ep   = true;
    bool        dump     = false;
};

bool parse(int argc, char ** argv, Options * o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--model-dir") o->model_dir = next();
        else if (a == "--aux-dir") o->aux_dir = next();
        else if (a == "--gguf") o->gguf = next();
        else if (a == "--llm-weights") o->llm_weights = next();
        else if (a == "--ep-lib") o->ep_lib = next();
        else if (a == "--contract") o->contract = next();
        else if (a == "--case") o->case_dir = next();
        else if (a == "--out") o->out_dir = next();
        else if (a == "--threads") o->threads = std::atoi(next().c_str());
        else if (a == "--repeats") o->repeats = std::atoi(next().c_str());
        else if (a == "--no-ep") o->use_ep = false;
        else if (a == "--dump") o->dump = true;
        else {
            std::fprintf(stderr, "[groot] unknown argument %s\n", a.c_str());
            return false;
        }
    }
    return !o->model_dir.empty() && !o->aux_dir.empty() && !o->gguf.empty() &&
           !o->llm_weights.empty() && !o->contract.empty() && !o->case_dir.empty();
}

bool read_images(const std::string & dir, std::vector<uint8_t> * front,
                 std::vector<uint8_t> * wrist, std::string * error) {
    front->assign(256 * 256 * 3, 0);
    wrist->assign(256 * 256 * 3, 0);
    if (!read_exact(dir + "/front.u8", front->data(), front->size()) ||
        !read_exact(dir + "/wrist.u8", wrist->data(), wrist->size())) {
        *error = "cannot read the raw camera inputs";
        return false;
    }
    // Images enter the pipeline as mtmd bitmaps, the multimodal surface mtmd
    // defines; GR00T has no mmproj, so the encoder behind that surface is the
    // external EP ViT instead of clip.
    for (auto * image : { front, wrist }) {
        mtmd_bitmap * bitmap = mtmd_bitmap_init(256, 256, image->data());
        if (!bitmap || mtmd_bitmap_get_n_bytes(bitmap) != image->size() ||
            mtmd_bitmap_is_audio(bitmap)) {
            if (bitmap) {
                mtmd_bitmap_free(bitmap);
            }
            *error = "mtmd rejected an RGB camera frame";
            return false;
        }
        std::memcpy(image->data(), mtmd_bitmap_get_data(bitmap), image->size());
        mtmd_bitmap_free(bitmap);
    }
    return true;
}

void write_f16(const std::string & path, const std::vector<uint16_t> & data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return;
    }
    f.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * 2));
}

void decode_actions(const std::vector<uint16_t> & normalized, const Contract & contract,
                    std::vector<float> * physical) {
    const int dims[7] = { 1, 1, 1, 1, 1, 1, 1 };
    physical->assign(size_t(16) * 7, 0.0f);
    int offset = 0;
    for (int k = 0; k < 7; ++k) {
        for (int step = 0; step < 16; ++step) {
            for (int j = 0; j < dims[k]; ++j) {
                _Float16 half;
                std::memcpy(&half, &normalized[size_t(step) * 132 + offset + j], 2);
                float v = float(half);
                v = std::min(1.0f, std::max(-1.0f, v));
                (*physical)[size_t(step) * 7 + offset + j] =
                    (v + 1.0f) * 0.5f * (contract.action_hi[k] - contract.action_lo[k]) +
                    contract.action_lo[k];
            }
        }
        offset += dims[k];
    }
}

}  // namespace

int main(int argc, char ** argv) {
    Options opt;
    if (!parse(argc, argv, &opt)) {
        std::fprintf(stderr,
                     "usage: %s --model-dir D --aux-dir D --gguf F --llm-weights D --contract F "
                     "--case D [--out D] [--ep-lib F] [--threads N] [--repeats N] [--no-ep] "
                     "[--dump]\n",
                     argv[0]);
        return 2;
    }
    if (!opt.out_dir.empty()) {
        std::filesystem::create_directories(opt.out_dir);
    }

    std::string error;
    Contract contract;
    if (!read_contract(opt.contract, &contract, &error)) {
        std::fprintf(stderr, "[groot] %s\n", error.c_str());
        return 2;
    }

    std::vector<uint8_t> front, wrist;
    if (!read_images(opt.case_dir, &front, &wrist, &error)) {
        std::fprintf(stderr, "[groot] %s\n", error.c_str());
        return 2;
    }
    float state[8] = {};
    if (!read_exact(opt.case_dir + "/state.f32", state, sizeof(state))) {
        return 2;
    }
    std::vector<float> initial(size_t(40) * 132, 0.0f);
    if (!read_exact(opt.case_dir + "/initial_actions.f32", initial.data(),
                    initial.size() * sizeof(float))) {
        return 2;
    }
    std::string instruction;
    {
        std::ifstream f(opt.case_dir + "/instruction.txt");
        if (!f) {
            std::fprintf(stderr, "[groot] missing instruction.txt\n");
            return 2;
        }
        std::getline(f, instruction);
    }

    // ---- external EP stages -------------------------------------------------
    groot::EpEngine::Config ep_config;
    ep_config.ep_lib         = opt.ep_lib;
    ep_config.vit_model      = opt.model_dir + "/vit_libero_qdq.onnx";
    ep_config.vl_model       = opt.model_dir + "/stage_vl.onnx";
    ep_config.cross_kv_model = opt.model_dir + "/stage_cross_kv.onnx";
    ep_config.dit_model      = opt.model_dir + "/dit_full.onnx";
    ep_config.aux_dir        = opt.aux_dir;
    ep_config.threads        = opt.threads;
    ep_config.use_ep         = opt.use_ep;

    groot::EpEngine engine;
    const auto t_engine = Clock::now();
    if (!engine.init(ep_config, &error)) {
        std::fprintf(stderr, "[groot] EP engine: %s\n", error.c_str());
        return 2;
    }
    const double engine_ms = since_ms(t_engine);

    // ---- llama backbone -----------------------------------------------------
    llama_backend_init();
    AiThread ai;
    GrootLlm llm;
    const auto t_llm_load = Clock::now();
    const bool loaded = ai.call([&] {
        return llm.load(opt.gguf, opt.llm_weights, opt.threads, 512, &error);
    });
    if (!loaded) {
        std::fprintf(stderr, "[groot] LLM: %s\n", error.c_str());
        return 2;
    }
    const double llm_load_ms = since_ms(t_llm_load);
    if (!llm.fused()) {
        std::fprintf(stderr, "[groot] this GGUF is not a GR00T feature-extractor header\n");
        return 2;
    }

    Layout layout;
    if (!build_layout(llm, instruction, &layout, &error)) {
        std::fprintf(stderr, "[groot] layout: %s\n", error.c_str());
        return 2;
    }
    const int S = int(layout.tokens.size());
    if (S > 512) {
        std::fprintf(stderr, "[groot] sequence %d exceeds the prefill capacity\n", S);
        return 2;
    }

    // ---- preprocessing ------------------------------------------------------
    std::vector<uint16_t> pixels(groot::kPatchTokens * groot::kPatchWidth);
    std::vector<uint16_t> state_norm(132);
    const auto t_preprocess = Clock::now();
    groot::images_to_patches(front.data(), wrist.data(), pixels.data());
    groot::normalize_state(state, contract.state_lo, contract.state_hi, state_norm.data());
    const double preprocess_ms = since_ms(t_preprocess);

    std::vector<uint16_t> image_embeds(size_t(groot::kImageTokens) * 2048);
    std::vector<uint16_t> deepstack(size_t(3) * groot::kImageTokens * 2048);
    std::vector<uint16_t> hidden(size_t(S) * 2048);
    std::vector<uint16_t> vlln(size_t(S) * 2048);
    std::vector<uint16_t> vl(size_t(S) * 2048);
    std::vector<uint16_t> keys(size_t(16) * S * 1536);
    std::vector<uint16_t> values(size_t(16) * S * 1536);
    std::vector<uint16_t> state_feature(1536);
    std::vector<uint16_t> action_feature(size_t(40) * 1536);
    std::vector<uint16_t> sa(size_t(41) * 1536);
    std::vector<uint16_t> dit_out(size_t(41) * 1024);
    std::vector<uint16_t> velocity(size_t(41) * 132);
    std::vector<uint16_t> actions(size_t(40) * 132);
    std::vector<uint16_t> embeddings(size_t(S) * 2048);

    std::map<std::string, double> best;
    double best_total = 1e18;
    int    measured   = 0;

    for (int run = 0; run <= opt.repeats; ++run) {
        const bool record = run > 0;  // the first pass warms caches
        const auto t_total = Clock::now();
        std::map<std::string, double> t;
        double ms = 0.0;

        const auto t_vit = Clock::now();
        if (!engine.run_vit(pixels.data(), image_embeds.data(), deepstack.data(), &ms)) {
            std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
            return 2;
        }
        t["vit"] = std::max(ms, since_ms(t_vit));

        const auto t_embed = Clock::now();
        if (!llm.embed_tokens(layout.tokens.data(), S, embeddings.data())) {
            std::fprintf(stderr, "[groot] %s\n", llm.error().c_str());
            return 2;
        }
        int visual_index = 0;
        for (int i = 0; i < S; ++i) {
            if (!layout.visual[i]) {
                continue;
            }
            std::memcpy(embeddings.data() + size_t(i) * 2048,
                        image_embeds.data() + size_t(visual_index) * 2048, 2048 * 2);
            ++visual_index;
        }
        t["embedding"] = since_ms(t_embed);

        const bool llm_ok = ai.call([&] {
            return llm.run(embeddings.data(), deepstack.data(), layout.visual.data(),
                           layout.positions.data(), S, groot::kImageTokens, hidden.data(), &ms);
        });
        if (!llm_ok) {
            std::fprintf(stderr, "[groot] %s\n", llm.error().c_str());
            return 2;
        }
        t["llm"] = ms;

        const auto t_vlln = Clock::now();
        if (groot_layer_norm(hidden.data(), contract.vlln_weight.data(), contract.vlln_bias.data(),
                             vlln.data(), size_t(S), 2048, 1e-5f) != 0) {
            std::fprintf(stderr, "[groot] VLLN failed\n");
            return 2;
        }
        t["vlln"] = since_ms(t_vlln);

        if (!engine.run_vl(vlln.data(), S, vl.data(), &ms)) {
            std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
            return 2;
        }
        t["vl"] = std::max(ms, 0.0);

        if (!engine.run_cross_kv(vl.data(), S, keys.data(), values.data(), &ms)) {
            std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
            return 2;
        }
        t["cross_kv"] = std::max(ms, 0.0);

        if (!engine.run_state_encoder(state_norm.data(), state_feature.data(), &ms)) {
            std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
            return 2;
        }
        t["state_encoder"] = std::max(ms, 0.0);

        // The replayed noise goes through the same fp32 -> fp16 conversion the
        // baseline's infer() applies; unlike the state vector it is not bf16-rounded.
        for (size_t i = 0; i < actions.size(); ++i) {
            const _Float16 half = static_cast<_Float16>(initial[i]);
            std::memcpy(&actions[i], &half, 2);
        }
        // No padding: the baseline strips it before llama_decode, so every backbone
        // position participates in cross attention.
        std::vector<uint8_t> backbone_mask(S, 1);

        for (int step = 0; step < 4; ++step) {
            const std::string suffix = std::to_string(step);
            if (!engine.run_action_encoder(actions.data(),
                                           contract.time_embedding.data() + size_t(step) * 40 * 1536,
                                           action_feature.data(), &ms)) {
                std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
                return 2;
            }
            t["step" + suffix + "_encoder"] = std::max(ms, 0.0);
            groot_add_fp16(action_feature.data(), contract.position_embedding.data(),
                           action_feature.data(), size_t(40) * 1536);
            std::memcpy(sa.data(), state_feature.data(), 1536 * 2);
            std::memcpy(sa.data() + 1536, action_feature.data(), size_t(40) * 1536 * 2);

            if (!engine.run_dit(sa.data(), keys.data(), values.data(), S,
                                contract.adaln.data() + size_t(step) * 32 * 3072,
                                contract.output_mod.data() + size_t(step) * 3072,
                                layout.visual.data(), backbone_mask.data(), dit_out.data(), &ms)) {
                std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
                return 2;
            }
            t["step" + suffix + "_dit"] = std::max(ms, 0.0);

            if (!engine.run_action_decoder(dit_out.data(), velocity.data(), &ms)) {
                std::fprintf(stderr, "[groot] %s\n", engine.error().c_str());
                return 2;
            }
            t["step" + suffix + "_decoder"] = std::max(ms, 0.0);
            groot_euler_fp16(actions.data(), velocity.data() + 132, size_t(40) * 132);

            if (record && opt.dump) {
                char name[64];
                snprintf(name, sizeof(name), "/step%d_dit.f16", step);
                write_f16(opt.out_dir + name, dit_out);
                snprintf(name, sizeof(name), "/step%d_velocity.f16", step);
                write_f16(opt.out_dir + name, velocity);
            }
        }

        const double total = since_ms(t_total);
        if (record) {
            ++measured;
            for (const auto & kv : t) {
                auto it = best.find(kv.first);
                if (it == best.end() || kv.second < it->second) {
                    best[kv.first] = kv.second;
                }
            }
            best_total = std::min(best_total, total);
            if (opt.dump) {
                write_f16(opt.out_dir + "/pixels.f16", pixels);
                write_f16(opt.out_dir + "/state_norm.f16", state_norm);
                {
                    std::ofstream tf(opt.out_dir + "/tokens.i32", std::ios::binary);
                    tf.write(reinterpret_cast<const char *>(layout.tokens.data()),
                             std::streamsize(layout.tokens.size() * 4));
                    std::ofstream pf(opt.out_dir + "/positions.i32", std::ios::binary);
                    pf.write(reinterpret_cast<const char *>(layout.positions.data()),
                             std::streamsize(layout.positions.size() * 4));
                }
                write_f16(opt.out_dir + "/image_embeds.f16", image_embeds);
                write_f16(opt.out_dir + "/deepstack.f16", deepstack);
                write_f16(opt.out_dir + "/llm.f16", hidden);
                write_f16(opt.out_dir + "/vlln.f16", vlln);
                write_f16(opt.out_dir + "/vl.f16", vl);
                write_f16(opt.out_dir + "/keys.f16", keys);
                write_f16(opt.out_dir + "/values.f16", values);
                write_f16(opt.out_dir + "/state_feature.f16", state_feature);
                write_f16(opt.out_dir + "/normalized_actions.f16", actions);
                std::vector<float> physical;
                decode_actions(actions, contract, &physical);
                std::ofstream pf(opt.out_dir + "/physical_actions.f32", std::ios::binary);
                pf.write(reinterpret_cast<const char *>(physical.data()),
                         std::streamsize(physical.size() * sizeof(float)));
            }
        }
    }

    std::printf("[groot] sequence=%d visual=%d engine_init_ms=%.1f llm_load_ms=%.1f\n", S,
                groot::kImageTokens, engine_ms, llm_load_ms);
    const char * order[] = { "vit", "embedding", "llm", "vlln", "vl", "cross_kv",
                             "state_encoder" };
    for (const char * name : order) {
        std::printf("[groot] %-14s %.2f ms\n", name, best[name]);
    }
    for (int step = 0; step < 4; ++step) {
        const std::string s = std::to_string(step);
        std::printf("[groot] step%d encoder=%.2f dit=%.2f decoder=%.2f ms\n", step,
                    best["step" + s + "_encoder"], best["step" + s + "_dit"],
                    best["step" + s + "_decoder"]);
    }
    std::printf("[groot] preprocess=%.2f ms\n", preprocess_ms);
    std::printf("[groot] total=%.2f ms (best of %d), end_to_end=%.2f ms\n", best_total, measured,
                best_total + preprocess_ms);

    if (!opt.out_dir.empty()) {
        std::ofstream f(opt.out_dir + "/timings.txt");
        f << "sequence " << S << "\n";
        for (const auto & kv : best) {
            f << kv.first << " " << kv.second << "\n";
        }
        f << "preprocess " << preprocess_ms << "\n";
        f << "total " << best_total << "\n";
        f << "end_to_end " << best_total + preprocess_ms << "\n";
    }

    llm.close();
    engine.close();
    llama_backend_free();
    return 0;
}
