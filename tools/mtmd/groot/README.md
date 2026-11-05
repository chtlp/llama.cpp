# GR00T N1.7 LIBERO on llama.cpp + mtmd + SpacemiT EP

`llama-groot-vla` runs the complete GR00T N1.7 LIBERO policy in one process:

```
2 cameras ──> ViT (ONNX / SpacemiT EP) ──> image embeds + 3 deepstack features ─┐
text ──────> llama tokenizer ──> token embeds ─────────────────────────────────┤
state ─────> state_encoder (ONNX / EP) ────────────────────────────────────────┤
                                                                               v
                                                    llama_decode(prefill) ──> hidden states
                                                                               │
                        VLLN (RVV) ──> VL self-attention (EP) ──> Cross-KV (EP)
                                                                               │
        action_encoder (EP) + position embedding ──> 32-block DiT (EP) ──> action_decoder (EP)
                                                                               │
                                                                    Euler x4 ──> actions
```

* **The LLM backbone is llama.cpp.** The GGUF (`llm_libero_fused.gguf`) is a
  feature-extractor header: it carries the token table and the metadata, while the
  16 decoder layers live in `llm_fused/*.bin`. `llama_decode` still owns the call;
  `src/models/qwen3vl.cpp` routes the decode graph through the baseline's fused
  prefill kernel via `groot_fused_callback` and returns the hidden states *before*
  the final RMSNorm, which is what GR00T consumes.
* **Every other stage is an external ONNX graph on the SpacemiT plugin EP.**
  `groot-ep.cpp` registers the EP library through ORT 1.31's plugin ABI and opens
  seven sessions: ViT, VL self-attention, Cross-KV, the 32-block DiT + tail, and
  the three AUX networks.
* **mtmd is the multimodal surface.** Camera frames enter as `mtmd_bitmap`s. GR00T
  has no CLIP mmproj, so the encoder behind that surface is the external EP ViT,
  the same substitution pattern as the SMT vision wrapper.

## Build

```bash
export RISCV_ROOT_PATH=/path/to/riscv-toolchain
cmake -S . -B build/rv64 \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF \
    -DLLAMA_GROOT_VLA=ON \
    -DSPACEMIT_ORT_DIR=/path/to/onnxruntime-install
cmake --build build/rv64 --target llama-groot-vla -j8
```

`SPACEMIT_ORT_DIR` must contain `include/onnxruntime/onnxruntime_cxx_api.h` and
`lib/libonnxruntime.so*`. The fused prefill file is compiled with the IME
extension and `-ffp-contract=off`, matching the baseline kernel build.

## Runtime layout

```
groot_vla/
  bin/     llama-groot-vla, libllama/libggml/libmtmd, libspacemit_ep.so,
           libspacemit_core.so, and a libonnxruntime.so.1.31.1
  models/  vit_libero_qdq.onnx, stage_vl.onnx, stage_cross_kv.onnx, dit_full.onnx
  aux/     state_encoder.onnx, action_encoder.onnx, action_decoder.onnx
  llm/     llm_libero_fused.gguf, llm_fused/{layerNN.bin,norm.f32}
  assets/  contract.bin (normalization contract + FP16 glue constants)
  cases/<name>/{front.u8,wrist.u8,state.f32,instruction.txt,initial_actions.f32}
```

`SPACEMIT_EP_DIT_WEIGHTS` must point at the packed DiT weights (`dit_w8a8`); the
VL encoder is loaded from its sibling `../vl`, as in the baseline.

### Affinity

The orchestrating thread stays on the general-purpose cores (X100, 0–7). The
thread that calls `llama_decode` writes `/proc/set_ai_thread` for itself and pins
to the AI cluster (`GROOT_AI_CPUS=8-15` by default). The EP worker threads pin
themselves through `SPACEMIT_EP_INTRA_THREAD_AFFINITY="8;9;...;15"`. **Do not**
pre-write `/proc/set_ai_thread` in the shell: it applies to the shell thread and
children inherit it, which pulls the orchestrator onto the AI cluster.

## Run

```bash
export LD_LIBRARY_PATH=/path/to/ort131:$PWD/bin:/path/to/ep_yyx   # ep_yyx for libspine_tcm
export SPERT_BACKEND=spacemit-k3
export SPACEMIT_EP_DIT_WEIGHTS=/path/to/dit_w8a8
export SPACEMIT_EP_INTRA_THREAD_AFFINITY="8;9;10;11;12;13;14;15" GROOT_AI_CPUS=8-15

./bin/llama-groot-vla \
    --model-dir models --aux-dir aux \
    --gguf llm/llm_libero_fused.gguf --llm-weights llm/llm_fused \
    --ep-lib $PWD/bin/libspacemit_ep.so.2.1.0-a0 \
    --contract assets/contract.bin --case cases/episode00_step0000 \
    --out out/episode00_step0000 --repeats 1 --dump
```

`--repeats N` runs one warm-up pass and then N measured passes; the reported
stage times are the best observation. `--dump` writes every stage tensor and the
decoded actions. `--no-ep` runs the same graphs on ORT's CPU provider.

## Measured results (K3, 8 threads, 2026-11-05)

Six replayed LIBERO observations (two per token length 149/155/156), one warm-up
pass, best of one measured pass. The baseline column is the delivered
`groot-n17-k3` runtime on the same board and inputs; its timing excludes
preprocessing (~16 ms in the original report).

| case | tokens | pipeline e2e | baseline infer | action cos (ours) | action cos (baseline) |
|---|---:|---:|---:|---:|---:|
| episode00_step0000 | 156 | **635.3 ms** | 673.8 ms | 0.999926 | 0.999968 |
| episode00_step0008 | 156 | **636.2 ms** | 674.8 ms | 0.999400 | 0.999703 |
| episode01_step0000 | 155 | **631.7 ms** | 675.2 ms | 0.999916 | 0.999675 |
| episode01_step0008 | 155 | **636.9 ms** | 672.3 ms | 0.995456 | 0.997724 |
| episode02_step0000 | 149 | **629.3 ms** | 669.9 ms | 0.995005 | 0.999584 |
| episode02_step0008 | 149 | **632.7 ms** | 671.1 ms | 0.999938 | 0.999912 |
| **mean** | | **633.7 ms** | **672.8 ms** | **0.99800** | **0.99943** |

Action cosine is measured against the BF16 PyTorch reference in the replay pack,
on the physical action vector (16 steps × 7 dimensions).

Per-stage (mean of the six runs):

| stage | pipeline | baseline | notes |
|---|---:|---:|---|
| ViT | 177 ms | 142.6 ms | QDQ whole-block fusion; known gap |
| LLM prefill | 158 ms | 140.6 ms | same fused kernel |
| VLLN | 0.7 ms | 1.3 ms | RVV |
| VL self-attention | 50 ms | 52.8 ms | EP |
| Cross-KV | 11.9 ms | 28.5 ms | EP, dynamic sequence length |
| DiT (x4) | 52 ms each | 72.7 ms each | EP, dynamic cross length |
| AUX (x4 encoder, x4 decoder) | 0.4 / 1.7 ms | 0.5 / 1.9 ms | EP |
| preprocessing | 5.4 ms | ~16 ms | scalar C++ |

Session creation costs ~10.5 s per process (ViT and DiT fusion); it is outside
the steady-state timing, like the baseline's model load.

## Accuracy: where the gap comes from

Per-stage cosine against the baseline for the same inputs (mean of the six
cases): ViT 0.968, LLM 0.9996, VLLN 0.965, VL 0.951, DiT step0 0.9992, step3
0.9973. The VLLN and VL rows amplify the ViT difference because they normalise
per row; the LLM and DiT chains stay above 0.997.

The ViT gap is the known one from the EP migration: the exported QDQ graph uses
per-tensor activation scales, while the baseline packs per-row activation scales
with INT8 weights. Closing it needs a per-row QDQ export or feeding the baseline
`vit_fp8.weights.bin` into the fused `VitBlock` kernel; swapping in the FP16 ViT
graph would raise accuracy but costs ~430 ms for ViT alone.

## Correctness invariants

1. The orchestrator stays on X100 (0–7); only the LLM thread and EP workers use
   the AI cluster (8–15). Misplacing the LLM thread costs ~12% on prefill.
2. Never pre-write `/proc/set_ai_thread` before launching the process.
3. The `stage_vl`, `stage_cross_kv` and `dit_full` graphs carry the sequence
   length as a symbol; `spine_dit_block.cc` recompiles its cached block when the
   cross length changes, so one graph serves all LIBERO instruction lengths.
4. `initial_actions` are converted fp32→fp16 directly; the state vector is
   round-to-bf16 then fp16, matching the baseline's `bf16_as_half` helper.
