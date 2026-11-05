// ONNX Runtime + SpacemiT plugin EP engine for the non-LLM GR00T stages.
//
// Every stage except the llama backbone is an ONNX graph executed by the
// external SpacemiT execution provider (ViT, VL self-attention, Cross-KV, the 32
// DiT blocks and the three AUX networks). The EP is a plugin library registered
// through the ORT 1.31 plugin ABI; nothing in this file knows its internals.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace groot {

class EpEngine {
public:
    struct Config {
        std::string ep_lib;
        std::string vit_model;
        std::string vl_model;
        std::string cross_kv_model;
        std::string dit_model;
        std::string aux_dir;
        int         threads = 8;
        bool        use_ep  = true;
    };

    EpEngine() = default;
    ~EpEngine();
    EpEngine(const EpEngine &) = delete;
    EpEngine & operator=(const EpEngine &) = delete;

    bool init(const Config & config, std::string * error);
    void close();

    // ViT: pixels [512,1536] fp16 -> image_embeds [128,2048], deepstack [3,128,2048].
    bool run_vit(const uint16_t * pixels, uint16_t * image_embeds, uint16_t * deepstack,
                 double * ms);

    // VL self-attention over the sequence: x [T,2048] -> y [T,2048].
    bool run_vl(const uint16_t * x, int tokens, uint16_t * y, double * ms);

    // Cross-KV: vl [T,2048] -> keys/values [16,T,1536].
    bool run_cross_kv(const uint16_t * vl, int tokens, uint16_t * keys, uint16_t * values,
                      double * ms);

    // One denoising step: x [41,1536], keys/values [16,C,1536] -> y [41,1024].
    bool run_dit(const uint16_t * x, const uint16_t * keys, const uint16_t * values, int cross,
                 const uint16_t * mods, const uint16_t * outmod, const uint8_t * image_mask,
                 const uint8_t * backbone_mask, uint16_t * y, double * ms);

    bool run_state_encoder(const uint16_t * state, uint16_t * out, double * ms);
    bool run_action_encoder(const uint16_t * actions, const uint16_t * time_embedding,
                            uint16_t * out, double * ms);
    bool run_action_decoder(const uint16_t * model_output, uint16_t * out, double * ms);

    const std::string & error() const { return error_; }

private:
    struct Session {
        std::unique_ptr<Ort::Session> session;
        std::vector<std::string>      input_names;
        std::vector<std::string>      output_names;
        std::vector<const char *>     input_ptrs;
        std::vector<const char *>     output_ptrs;

        bool open(Ort::Env & env, const Ort::SessionOptions & options, const std::string & path,
                  std::string * error);
        bool run(const std::vector<Ort::Value> & inputs, std::vector<Ort::Value> & outputs);
    };

    bool open_session(Session & session, const std::string & path, bool vit_style,
                      bool stage_domain, std::string * error);
    bool make_options(Ort::SessionOptions & options, bool vit_style, bool stage_domain);

    Ort::Env                      env_{ ORT_LOGGING_LEVEL_WARNING, "groot-vla" };
    std::unique_ptr<Ort::CustomOpDomain> stage_domain_;
    std::vector<Ort::ConstEpDevice> ep_devices_;
    Config                        config_;
    Session                       vit_, vl_, cross_, dit_;
    Session                       state_encoder_, action_encoder_, action_decoder_;
    std::string                   error_;
    bool                          initialized_ = false;
};

}  // namespace groot
