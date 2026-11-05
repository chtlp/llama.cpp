#include "groot-ep.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace groot {
namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// The EP consumes the private stage nodes before ORT executes them; these
// placeholder kernels only make the custom domain resolvable at load time.
struct NullKernel {
    NullKernel(const OrtApi &, const OrtKernelInfo *) {}
    void Compute(OrtKernelContext *) { std::abort(); }
};

template <const char * OpName, size_t Inputs, size_t Outputs>
struct StageOp : Ort::CustomOpBase<StageOp<OpName, Inputs, Outputs>, NullKernel> {
    void * CreateKernel(const OrtApi & api, const OrtKernelInfo * info) const {
        return new NullKernel(api, info);
    }
    const char * GetName() const { return OpName; }
    const char * GetExecutionProviderType() const { return ""; }
    size_t GetInputTypeCount() const { return Inputs; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; }
    size_t GetOutputTypeCount() const { return Outputs; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; }
};

constexpr char kVlEncoder[]  = "VlEncoder";
constexpr char kDitCrossKv[] = "DitCrossKV";
constexpr char kDitTail[]    = "DitTail";

StageOp<kVlEncoder, 1, 1>  g_vl_op;
StageOp<kDitCrossKv, 1, 2> g_cross_op;
StageOp<kDitTail, 2, 1>    g_tail_op;

Ort::Value make_f16(const Ort::MemoryInfo & mem, const uint16_t * data,
                    const std::vector<int64_t> & shape) {
    size_t count = 1;
    for (auto d : shape) {
        count *= size_t(d);
    }
    return Ort::Value::CreateTensor<Ort::Float16_t>(
        mem, reinterpret_cast<Ort::Float16_t *>(const_cast<uint16_t *>(data)), count, shape.data(),
        shape.size());
}

Ort::Value make_u8(const Ort::MemoryInfo & mem, const uint8_t * data,
                   const std::vector<int64_t> & shape) {
    size_t count = 1;
    for (auto d : shape) {
        count *= size_t(d);
    }
    return Ort::Value::CreateTensor<uint8_t>(mem, const_cast<uint8_t *>(data), count, shape.data(),
                                             shape.size());
}

const void * output_data(const Ort::Value & value) {
    return value.GetTensorData<void>();
}

size_t output_count(const Ort::Value & value) {
    return value.GetTensorTypeAndShapeInfo().GetElementCount();
}

bool copy_output(const std::vector<Ort::Value> & outputs, size_t index, uint16_t * dst,
                 size_t expected) {
    if (index >= outputs.size()) {
        return false;
    }
    const size_t n = output_count(outputs[index]);
    if (n != expected) {
        fprintf(stderr, "[groot] output %zu has %zu elements, expected %zu\n", index, n, expected);
        return false;
    }
    std::memcpy(dst, output_data(outputs[index]), n * 2);
    return true;
}

}  // namespace

EpEngine::~EpEngine() {
    close();
}

void EpEngine::close() {
    state_encoder_ = Session{};
    action_encoder_ = Session{};
    action_decoder_ = Session{};
    vit_ = Session{};
    vl_ = Session{};
    cross_ = Session{};
    dit_ = Session{};
    ep_devices_.clear();
    initialized_ = false;
}

bool EpEngine::Session::open(Ort::Env & env, const Ort::SessionOptions & options,
                             const std::string & path, std::string * error) {
    try {
        session = std::make_unique<Ort::Session>(env, path.c_str(), options);
    } catch (const std::exception & e) {
        if (error) {
            *error = e.what();
        }
        return false;
    }
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n_in  = session->GetInputCount();
    const size_t n_out = session->GetOutputCount();
    for (size_t i = 0; i < n_in; ++i) {
        input_names.emplace_back(session->GetInputNameAllocated(i, alloc).get());
    }
    for (size_t i = 0; i < n_out; ++i) {
        output_names.emplace_back(session->GetOutputNameAllocated(i, alloc).get());
    }
    for (auto & s : input_names) {
        input_ptrs.push_back(s.c_str());
    }
    for (auto & s : output_names) {
        output_ptrs.push_back(s.c_str());
    }
    return true;
}

bool EpEngine::Session::run(const std::vector<Ort::Value> & inputs,
                            std::vector<Ort::Value> & outputs) {
    outputs = session->Run(Ort::RunOptions{ nullptr }, input_ptrs.data(), inputs.data(),
                           inputs.size(), output_ptrs.data(), output_ptrs.size());
    return true;
}

bool EpEngine::make_options(Ort::SessionOptions & options, bool vit_style,
                            bool stage_domain) {
    options.SetGraphOptimizationLevel(vit_style ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                                : GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetIntraOpNumThreads(vit_style ? 1 : config_.threads);
    options.SetInterOpNumThreads(1);
    if (config_.use_ep && !ep_devices_.empty()) {
        std::unordered_map<std::string, std::string> ep_options;
        ep_options["ep.spacemitexecutionprovider.SPACEMIT_EP_INTRA_THREAD_NUM"] =
            std::to_string(config_.threads);
        options.AppendExecutionProvider_V2(env_, ep_devices_, ep_options);
    }
    if (stage_domain) {
        if (!stage_domain_) {
            stage_domain_ = std::make_unique<Ort::CustomOpDomain>("com.spacemit.stage");
            stage_domain_->Add(&g_vl_op);
            stage_domain_->Add(&g_cross_op);
            stage_domain_->Add(&g_tail_op);
        }
        // The domain has to outlive every session created with these options.
        options.Add(*stage_domain_);
    }
    return true;
}

bool EpEngine::open_session(Session & session, const std::string & path, bool vit_style,
                            bool stage_domain, std::string * error) {
    Ort::SessionOptions options;
    make_options(options, vit_style, stage_domain);
    return session.open(env_, options, path, error);
}

bool EpEngine::init(const Config & config, std::string * error) {
    config_ = config;
    auto fail = [&](const std::string & message) {
        error_ = message;
        if (error) {
            *error = message;
        }
        return false;
    };

    try {
        if (config_.use_ep) {
            const std::string lib = config_.ep_lib.empty() ? "libspacemit_ep.so" : config_.ep_lib;
            Ort::ThrowOnError(
                Ort::GetApi().RegisterExecutionProviderLibrary(env_, "spacemit_ep", lib.c_str()));
            for (auto & device : env_.GetEpDevices()) {
                if (std::string(device.EpName()) == "SpacemiTExecutionProvider") {
                    ep_devices_.push_back(device);
                }
            }
            if (ep_devices_.empty()) {
                return fail("SpacemiTExecutionProvider device not found after registering " + lib);
            }
        }

        // ViT fuses into whole-block operators, so it uses the same private
        // stage domain as the rest of the pipeline.
        if (!open_session(vit_, config_.vit_model, true, true, error)) {
            return fail("ViT session: " + (error ? *error : std::string("failed")));
        }
        if (!open_session(vl_, config_.vl_model, false, true, error)) {
            return fail("VL session: " + (error ? *error : std::string("failed")));
        }
        if (!open_session(cross_, config_.cross_kv_model, false, true, error)) {
            return fail("Cross-KV session: " + (error ? *error : std::string("failed")));
        }
        if (!open_session(dit_, config_.dit_model, false, true, error)) {
            return fail("DiT session: " + (error ? *error : std::string("failed")));
        }
        if (!open_session(state_encoder_, config_.aux_dir + "/state_encoder.onnx", false, false,
                          error)) {
            return fail("state_encoder session: " + (error ? *error : std::string("failed")));
        }
        if (!open_session(action_encoder_, config_.aux_dir + "/action_encoder.onnx", false, false,
                          error)) {
            return fail("action_encoder session: " + (error ? *error : std::string("failed")));
        }
        if (!open_session(action_decoder_, config_.aux_dir + "/action_decoder.onnx", false, false,
                          error)) {
            return fail("action_decoder session: " + (error ? *error : std::string("failed")));
        }
    } catch (const std::exception & e) {
        return fail(std::string("EP engine init: ") + e.what());
    }
    initialized_ = true;
    return true;
}

bool EpEngine::run_vit(const uint16_t * pixels, uint16_t * image_embeds, uint16_t * deepstack,
                       double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, pixels, { 512, 1536 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        vit_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        if (outputs.size() < 2) {
            error_ = "ViT must return image_embeds and deepstack";
            return false;
        }
        return copy_output(outputs, 0, image_embeds, 128 * 2048) &&
               copy_output(outputs, 1, deepstack, 3 * 128 * 2048);
    } catch (const std::exception & e) {
        error_ = std::string("ViT run: ") + e.what();
        return false;
    }
}

bool EpEngine::run_vl(const uint16_t * x, int tokens, uint16_t * y, double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, x, { tokens, 2048 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        vl_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        return copy_output(outputs, 0, y, size_t(tokens) * 2048);
    } catch (const std::exception & e) {
        error_ = std::string("VL run: ") + e.what();
        return false;
    }
}

bool EpEngine::run_cross_kv(const uint16_t * vl, int tokens, uint16_t * keys, uint16_t * values,
                            double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, vl, { tokens, 2048 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        cross_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        const size_t n = size_t(16) * tokens * 1536;
        return copy_output(outputs, 0, keys, n) && copy_output(outputs, 1, values, n);
    } catch (const std::exception & e) {
        error_ = std::string("Cross-KV run: ") + e.what();
        return false;
    }
}

bool EpEngine::run_dit(const uint16_t * x, const uint16_t * keys, const uint16_t * values,
                       int cross, const uint16_t * mods, const uint16_t * outmod,
                       const uint8_t * image_mask, const uint8_t * backbone_mask, uint16_t * y,
                       double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, outmod, { 3072 }));
        inputs.push_back(make_f16(mem, x, { 41, 1536 }));
        inputs.push_back(make_f16(mem, mods, { 32, 3072 }));
        inputs.push_back(make_f16(mem, keys, { 16, cross, 1536 }));
        inputs.push_back(make_f16(mem, values, { 16, cross, 1536 }));
        inputs.push_back(make_u8(mem, image_mask, { 41 }));
        inputs.push_back(make_u8(mem, backbone_mask, { 41 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        dit_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        return copy_output(outputs, 0, y, size_t(41) * 1024);
    } catch (const std::exception & e) {
        error_ = std::string("DiT run: ") + e.what();
        return false;
    }
}

bool EpEngine::run_state_encoder(const uint16_t * state, uint16_t * out, double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, state, { 1, 1, 132 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        state_encoder_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        return copy_output(outputs, 0, out, 1536);
    } catch (const std::exception & e) {
        error_ = std::string("state_encoder run: ") + e.what();
        return false;
    }
}

bool EpEngine::run_action_encoder(const uint16_t * actions, const uint16_t * time_embedding,
                                  uint16_t * out, double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, actions, { 1, 40, 132 }));
        inputs.push_back(make_f16(mem, time_embedding, { 1, 40, 1536 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        action_encoder_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        return copy_output(outputs, 0, out, size_t(40) * 1536);
    } catch (const std::exception & e) {
        error_ = std::string("action_encoder run: ") + e.what();
        return false;
    }
}

bool EpEngine::run_action_decoder(const uint16_t * model_output, uint16_t * out, double * ms) {
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputs;
        inputs.push_back(make_f16(mem, model_output, { 1, 41, 1024 }));
        std::vector<Ort::Value> outputs;
        const double t0 = now_ms();
        action_decoder_.run(inputs, outputs);
        if (ms) {
            *ms = now_ms() - t0;
        }
        return copy_output(outputs, 0, out, size_t(41) * 132);
    } catch (const std::exception & e) {
        error_ = std::string("action_decoder run: ") + e.what();
        return false;
    }
}

}  // namespace groot
