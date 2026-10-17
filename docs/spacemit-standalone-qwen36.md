# ggml-spacemit（独立后端）：Qwen3.5/3.6 MoE 在 K3 上的优化

分支 `linpengt/spacemit-standalone-qwen36`，基于 spacemit-com `mtmd-backend` `6df9f7504`（PR #48）。

- 改动范围：`ggml/src/ggml-spacemit/`，以及 `src/models/` 里 9 行图构建改动（第 1 个提交）。
- 不涉及 legacy `ggml-cpu/spacemit` 路径。

这些优化最初在 legacy 路径上开发和验证（分支 `linpengt/qwen35-opt`），这里移植到独立后端。独立后端的 `ime2_kernels.cpp`、`repack.cpp`、`ime_kernels.h` 与 legacy 原版逐字节相同，矩阵乘部分也只有接口不同（`ggml_compute_params` → `ggml::spacemit::context`），所以大部分是原样移植。

## 结果

测试条件：
- 硬件：K3（`k3-dev-006`），8×A100 @1.8 GHz。
- 模型：bartowski `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`，原始 GGUF，不需要重新量化。
- 参数：`GGML_SPACEMIT_BACKEND=1 -dev SPACEMIT0 -ngl 99 -t 8 -fa 1 -lm none -b 2048 -ub 512`。

| | 本分支基线（`6df9f7504` + Q5_0 分派修复） | 本分支 |
|---|---:|---:|
| pp512（t/s） | 35.7 | **127.4** |
| pp2048 | 23.5 | **115.1** |
| tg32 | 7.4 | **9.0** |
| tg32，上下文 4K / 8K | 6.9 / 5.2（其余移植后、split-KV 前） | **7.9 / 7.2** |
| MTP（`draft-mtp`，n-max 1），4K / 8K | — | **9.9 / 9.0** |

- 不修 Q5_0 分派时，`6df9f7504` 原样运行这个模型会中止：`failed to dispatch op MUL_MAT (f32)`。
- MTP 一行用了 `GGML_SPACEMIT_LM_Q4=1`。
- 精度：每个提交都用同一构建里的 legacy 路径（`GGML_SPACEMIT_BACKEND=0 -ngl 0`）对比。
  - 96 token 贪心输出每步都逐字相同。
  - `-t 2` 与 `-t 8` 逐字相同。
  - 最终 KLD：`-c 512` 是 0.0066，`-c 2048` 是 0.0074（top-1 96.6%）。两条路径的 RMS_NORM、softmax、RoPE、FA 实现不同，这个量级是正常的数值差。

## 提交

| 提交 | 内容 | pp512 |
|---|---|---:|
| qwen35/qwen3next: keep the conv-state concat inputs alive until ssm_conv | 把 concat 的两个输入挂到 SSM_CONV 的额外 src 上，让分配器保留到 conv 为止，后端才能直接读它们（下面的 concat → ssm_conv 融合依赖这一点；不读额外 src 的后端不受影响） | — |
| dispatch Q5_0 MUL_MAT / MUL_MAT_ID | `compute_forward` 漏了 Q5_0（`supports_op` 和 repack 都接受），Qwen3.6 的共享专家是 Q5_0 | 35.7 |
| RVV transposed concat and ssm_conv | conv 状态的 concat 原来是标量逐元素拷贝（4.9 ms/token），ssm_conv 是标量（1.8 ms/token） | 44.6 |
| flash attention for head_dim up to 256 | RVV FA 只支持 head_dim ≤ 128；Qwen3.6 是 256，退回标量 FA（3.3 ms/token） | 51.7 |
| chunkwise gated delta net for prefill | prefill 原来逐 token 递推（4.3 ms/token），改为 chunk 16 的分块形式，全部是 4 行分块的 RVV fp32 矩阵乘 | 61.0 |
| port the matmul work | 见下一节 | 102.6 |
| graph-level fusions | concat → ssm_conv；MoE 路由权重并入 MUL_MAT_ID；RMS_NORM + MUL。在 `graph_compute` launch 之前由主机决定 | 127.3 |
| ARGSORT and CLAMP on the device | MoE 路由的 top-k 和权重和截断原来在 CPU 后端，每层都切图（163 段 → 3 段）。与 CPU 算法相同（`std::sort`），专家选择不变 | 127.0 |
| split-KV flash-attention decode | 1～4 个 query token（decode、MTP 验证）时按 KV 切分给各线程，每行 K/V 只读一次，再做 log-sum-exp 合并 | 126.3 |

矩阵乘这个提交包括：
- Q4_0 hp kernel 的 4 行 A 对齐布局 + 有符号 int4 B；
- 8 行 kernel（m8，MoE 与 dense）；
- MoE `MUL_MAT_ID` 的 m4/m8 分组、余数补零、gate/up 共用量化；
- Q8_0 权重按 `vmadot` 读取顺序预排（去掉 `vupack`）；
- dense M 分块。

还有三种加载时的权重转换：
- Q5_0 → Q8_0：精确转换，默认开启。原 Q5_0 IME kernel 改为 opt-in（`GGML_SPACEMIT_Q5_0_IME=1`）。
- F32 MoE 路由矩阵 → Q8_0：默认开启。
- lm_head Q6_K → Q4_0：默认关闭。

保留了独立后端自己的一处上游改动：decode 的直连 IME 路径只给 Q4_0 用。

每个提交的说明里有对应的 legacy 提交和数字。

## 运行时开关（均默认开，`=0` 关闭，除注明外）

| 开关 | 作用 |
|---|---|
| `GGML_SPACEMIT_HP_ALIGNED` | Q4_0 hp 路径：A 对齐 + 有符号 int4 B（关掉后 m8、Q8_PRE 也关） |
| `GGML_SPACEMIT_HP_M8` | 8 行 kernel |
| `GGML_SPACEMIT_Q8_PRE` | Q8_0 权重预排 |
| `GGML_SPACEMIT_MOE_M4` / `_MOE_PAD8` / `_MOE_REUSEQ` / `_MOE_STORE1` | MoE 分组、余数补零、gate/up 复用、单遍写出 |
| `GGML_SPACEMIT_DENSE_MBLK` | dense M 分块 |
| `GGML_SPACEMIT_Q5_0_AS_Q8` | Q5_0 加载时转 Q8_0 |
| `GGML_SPACEMIT_ROUTER_Q8` | 路由矩阵转 Q8_0（有损：top-1 一致率略低，KLD 与其他改动同级） |
| `GGML_SPACEMIT_LM_Q4` | lm_head 转 Q4_0（**默认关**，`=1` 打开；有损，decode +7%） |
| `GGML_SPACEMIT_FUSE_CONV` / `_FUSE_MOE_W` / `_RVV_RMS_MUL` | 三种融合 |
| `GGML_SPACEMIT_GDN_CHUNK`、`_GDN_CHUNK_SIZE` | 分块 GDN、块大小（默认 16） |
| `GGML_SPACEMIT_FA_SPLITKV` | split-KV decode FA |

## 已知问题

- **MTP 输出与非 MTP 不逐字相同。** 上下文 0 时会出现，legacy 路径也一样，10-15 的旧 upstream 上是相同的。关掉本分支的全部开关仍然如此。可能来自 rebase 进来的上游 speculative 改动，或 2-token 验证 batch 与单 token 的数值差。接受率（80～90%）和速度都正常。
- 上游 trace 分析器（`GGML_BUILD_PROFILE`）只能用静态库构建：`ggml-profile.h` 的函数没有 `GGML_API`。静态构建不能同时开 legacy 和独立后端，因为两边的 `spacemit_kernels::*` 符号同名。
- 没有移植：`GGML_OP_STATS` 子阶段计时（在 ggml-cpu 里，这里是桩函数），GDN 的 IME 版本（legacy 上默认关）。

## 复现

```bash
cmake -B build-k3 -DCMAKE_BUILD_TYPE=Release -DGGML_SPACEMIT=ON -DSPERT_DIR=<spine-runtime> \
  -DGGML_CPU_REPACK=OFF -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZICBOP=ON \
  -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/riscv64-spacemit-linux-gnu-gcc.cmake
cmake --build build-k3 --target llama-bench llama-completion
GGML_SPACEMIT_BACKEND=1 GGML_SPACEMIT_WORKERS=8 build-k3/bin/llama-bench -m Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
  -dev SPACEMIT0 -ngl 99 -t 8 -fa 1 -lm none -b 2048 -ub 512 -p 512,2048 -n 32
```

完整的测试脚本和原始输出：[chtlp/edge-ai-exp](https://github.com/chtlp/edge-ai-exp) `bench-results/k3/20261017-k3-dev006-qwen36-standalone-port/`（逐步对比、最终对比、decode/MTP）和 `20261017-k3-dev006-qwen36-standalone-backend/`（移植前的逐算子 trace）。
