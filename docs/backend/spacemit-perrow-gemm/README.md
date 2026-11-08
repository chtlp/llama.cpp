# K3 per-row INT8 GEMM：不做融合，GR00T LLM prefill 追平 fused kernel

日期 2026-11-08，分支 `linpengt/perrow-gemm`（基于 spacemit `855a53bed`）。

## 结论

GR00T N1.7 的 LLM backbone（Cosmos-Reason2-2B，16 层，prefill 156 token），在
llama.cpp 原生计算图上**只替换 Q8_0 的 MUL_MAT kernel**，不做任何算子融合，每层耗时
从 36.7 ms 降到 9.0 ms，16 层约 144 ms，和 `groot-vla` 分支里手写的 fused prefill
kernel（同板实测 148–157 ms）持平。

此前通用算子路径（约 18 ms/层）和 fused（约 9 ms/层）差了一倍，差距基本都在 GEMM：
非 GEMM 算子在通用路径上反而更快。也就是说，问题出在 GEMM 的 kernel 和调度上，不在
算子之间的同步，不需要融合也能追上。

## 结果

测试环境：K3 板子，A100 8 核空闲，`llama-bench -p 156 -n 0 -fa 1 -r 10`，两轮交替跑
（日志见 `results/depth-sweep.log`）。权重是按行量化的 INT8（每个输出行一个 scale），
存成普通 Q8_0。为了扣掉 embedding、lm_head 等固定开销，分别跑 2/4/8/12/16 层，取 prefill
时间对层数拟合的斜率作为每层耗时。

| 路径 | 每层 | 16 层 |
|---|---:|---:|
| 主线 spacemit `855a53bed`（Q8_0 32x32 路径） | 36.66 ms | 587 ms |
| 通用算子分支（llm-backbone-generic-ops，去掉其 ROPE 提交） | 18.08 ms | 289 ms |
| **本分支** | **8.99 ms** | **144 ms** |
| `groot-vla` fused prefill kernel，同板同时段 | — | 最快 148–150，中位 154–157 ms |

pp156 吞吐（t/s，两轮）：

| 层数 | 本分支 | 通用算子分支 | 主线 |
|---:|---:|---:|---:|
| 2 | 5388 / 4678 | 3510 / 3625 | 2293 / 2442 |
| 4 | 3030 / 3112 | 1984 / 1950 | 1149 / 1118 |
| 8 | 1853 / 1798 | 1017 / 1026 | 548 / 544 |
| 12 | 1284 / 1280 | 698 / 694 | 361 / 363 |
| 16 | 997 / 981 | 加载失败 | 271 / 268 |

16 层模型的其他形状（`results/shapes-ppl.log`，t/s）：

| | 本分支 | 主线 32x32 路径 |
|---|---:|---:|
| pp156 | 976 | 270 |
| pp512 | 541 | 262 |
| tg32 | 16.3 | 17.9 |

精度：完整 Qwen3-1.7B（28 层）转成按行量化后，wikitext-2 PPL（`-c 512 --chunks 8`）为
16.53；同一份权重走主线路径是 16.66，原始 Q8_0 是 16.84，差别在误差范围内。

## 改动

三个提交，图结构不变（每层还是 17 个 opnode）：

1. `fix(llama): keep norm and KV cache on the accelerator for ACCEL backends`：从通用算子
   分支 cherry-pick 的图切分修复。llama-bench 下加不加都是 3 个 split，这里没有影响，
   主要作用于 llama-groot-vla。
2. `ggml: export the op profile hooks`：让 `GGML_BUILD_PROFILE` 能链接，只用于调试。
3. `spacemit: per-row INT8 prefill GEMM for Q8_0 weights with one scale per row`：主要
   改动在 `ggml/src/ggml-spacemit/ime.cpp` 的 `namespace perrow_q8` 里。

kernel 的做法：

- **识别**：repack 时检查 2D Q8_0 权重，如果每一行的所有 32-block 用的是同一个 fp16
  scale，就把 `tensor->extra` 换成 `tensor_traits_q8_0_perrow`，否则走原来的路径。不新增
  ggml 类型，GGUF 还是普通 Q8_0。设 `GGML_SPACEMIT_PERROW=0` 可以关掉。
- **布局**：权重按 32 列一条 strip 排，`[K/16][32 列][16 B]`，后面跟每行一个 f32 scale。
  激活按 16 行一组，`[K/16][16 行][16 B]`。
- **tile**：和 fused kernel 的 `ime_i8_tile` 一样，16x32 int32 累加器一直累加到 k 结束。
  每个 16-k block 做 8 条 vmadot，A/B 寄存器组软件流水。单核热态约 1 TOPS。
- **调度**：以（列 strip × 16 行 tile）为单位，按 strip 优先均分给 8 个 worker。每个
  worker 把自己的权重 strip 从 DRAM 拷进 TCM，只拷一次，然后在 TCM 里跑完所有 M tile。
  k 太长放不下时分段，段间用 int32 累加。
- **激活量化**：每行动态量化成 INT8 一次，放在共享 workspace 之外的 buffer 里。连续几个
  matmul 读的是同一个输入张量时（Q/K/V、gate/up）直接复用。

16 层 prefill 中每个 worker 的时间分布（`GGML_SPACEMIT_PERROW_PROF=1`，加激活复用之前）：

| 阶段 | 时间 | 说明 |
|---|---:|---|
| 权重拷贝 | 约 26 ms | 接近约 30 GB/s 的 DRAM 上限 |
| vmadot tile | 约 25 ms | |
| epilogue | 约 21 ms | 其中输出写回约 10 ms |
| 激活量化 | 约 15 ms | 复用后约 10 ms |

## 试过但没用的

| 尝试 | 结果 |
|---|---|
| 在 TCM 里给 tile 输出留 2 KB，导致 k 被切成 1984+64 两段 | 挪到栈上之后从 700 涨到 942 t/s |
| 激活用 strided `vlse64/vsse64` 散开存放 | 改成 16 B 连续 store 后快一倍 |
| 软件预取下一条 strip 和输出行 | 900 降到 672 t/s，已删 |
| tile 直接从 DRAM 读权重，不拷进 TCM | M=156 慢 20%，decode 慢 7%，已删 |
| 7 个 worker | 880–898，8 个是 940 t/s；按列切本来就均匀，保留 8 个 |
| M ≤ 16 时每个 worker 自己量化激活 | 没有可测的变化，已删 |

## 通用算子分支的两个问题

- ROPE 那个提交（`f022aa96b`，contiguous ROPE factors）跑普通 Qwen3 会段错误：在
  `legacy_small_f32` 路径上（NEOX、head ≤ 128），`cos_cache`/`sin_cache` 没有分配就被写入。
  它只在 Qwen3-VL（M-RoPE）上测过。本文测试时去掉了这个提交。
- 16 层时会申请一个 38 GB 的 SPACEMIT buffer 然后加载失败（2–12 层能加载），可能和
  Q8_0 repack size 的改动有关，没有细查。所以上表里它的斜率用的是 2–12 层。

## 限制

- 测试模型是截成 16 层的 Qwen3-1.7B，每层形状和 GR00T 的 LLM 一样，但没覆盖 M-RoPE
  和 3 个 deepstack add。
- fused kernel 用的是合成权重，格式和 `layerNN.bin` 完全一致（`fused/make_layers.py`），
  全零和随机权重测出来都是 149 ms 左右。
- 没做 SmoothQuant。GR00T 的 per-row INT8 权重需要在导出时把 smooth 因子折进去：
  input/ffn 的折进 norm gamma，down 的折进 up_proj 的行，o 的只有在一个 KV group 的两个
  q head 相同时才能折进 v_proj。GR00T 动作输出的精度还没测。
- decode 慢 8%。主线的 32x32 路径是两个 worker 一组，一个拷贝一个计算交替进行；这里是
  同一个核先拷后算。GR00T 只有 prefill，所以没继续优化。
- 复用的激活 buffer 是进程全局的，同一时间只能有一个 spacemit backend。

## 复现

构建（K3 本机，需要 spine-runtime）：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_SPACEMIT=ON -DSPERT_DIR=<spine-runtime> \
  -DGGML_CPU_REPACK=OFF -DLLAMA_OPENSSL=OFF -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/riscv64-spacemit-linux-gnu-gcc.cmake
cmake --build build --parallel 8 --target llama-bench llama-perplexity
```

测试模型（从 `Qwen3-1.7B-Q8_0.gguf` 生成；`--keep-q8` 保留原始 Q8_0 作对照）：

```bash
for L in 2 4 8 12 16; do
  python3 make_perrow.py gguf-py Qwen3-1.7B-Q8_0.gguf models/pr_L$L.gguf --layers $L
done
python3 make_perrow.py gguf-py Qwen3-1.7B-Q8_0.gguf models/pr_L28.gguf --layers 28
```

层数扫描和 PPL：

```bash
OURS=build/bin BASE=<主线 build/bin> MODELS=models ./sweep.sh
GGML_SPACEMIT_BACKEND=1 build/bin/llama-perplexity -m models/pr_L28.gguf --device SPACEMIT0 \
  -f wiki.test.raw -c 512 --chunks 8 -fa on
```

fused kernel 对照：把 `groot-vla` 分支的 `tools/mtmd/groot/llm_fused.inc` 和
`llm_fused_kernel.inc` 放到 `fused/` 下，然后：

```bash
cd fused && python3 make_layers.py w
g++ -O2 -std=c++17 -march=rv64gcv_zfh_zvfh_zba_xsmtvdotii -mabi=lp64d -ffp-contract=off \
  -I../../../../ggml/include fused_bench.cpp -o fused_bench -L../../../../build/bin -lggml-base -ldl -lpthread
LD_LIBRARY_PATH=../../../../build/bin ./fused_bench w 156 8 10
```

`fused_bench` 和 llama-groot-vla 的设置一样：先写 `/proc/set_ai_thread`，再绑到 A100
核（8–15）。
