#pragma once

// Format-accuracy experiment: fake-quantize (quantize, then dequantize back) MoE expert weights at
// load time and the activations feeding ggml_mul_mat_id at graph build time, so that one BF16 model
// can emulate W4A16 / W4A8 / W4A4 formats that the CPU backend does not run natively.
//
//   LLAMA_FAKEQ_W     = q4_0 | mxfp4 | mxfp4b | nvfp4   weights (BF16/F32 tensors, needs --no-mmap)
//   LLAMA_FAKEQ_W_PAT = substring of tensor names to touch (default "_exps.weight")
//   LLAMA_FAKEQ_A     = mxfp8 | mxfp4 | nvfp4           activations into expert matmuls
//
// mxfp4  : E2M1, 32-block, UE8M0 scale 2^(floor(log2 amax) - 2)   (= ggml MXFP4 / OCP)
// mxfp4b : as mxfp4, but per block keep the better of that scale and 2x it (min SSE)
// nvfp4  : E2M1, 16-block, UE4M3 scale, plus an FP32 scale per matrix (weights) or per row (acts)
// mxfp8  : E4M3FN, 32-block, UE8M0 scale 2^(floor(log2 amax) - 8), saturating
// q4_0   : ggml quantize_row_q4_0_ref (RTN, fp16 scale)

struct ggml_context;
struct ggml_tensor;

// fake-quantize a loaded weight tensor in place if enabled and its name matches
void llama_fakeq_weight(ggml_tensor * t);

// returns cur, or a fake-quantized copy of cur if activation fake-quant is enabled
ggml_tensor * llama_fakeq_act(ggml_context * ctx, ggml_tensor * cur);

bool llama_fakeq_weight_enabled();
void llama_fakeq_print_summary();
