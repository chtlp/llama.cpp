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

| | 基线（`6df9f7504` + Q5_0 分派修复） | 本分支（`37acff6dd`） |
|---|---:|---:|
| pp512（t/s） | 34.29 ± 0.17 | **126.37** ± 0.12 |
| pp2048 | 23.50 ± 0.01 | **115.27** ± 0.18 |
| tg32 | 7.40 ± 0.15 | **8.85** ± 0.15 |
| tg32，上下文 4096 / 8192 | — | **8.26 / 7.27** |

- 不修 Q5_0 分派时，`6df9f7504` 原样运行这个模型会中止：`failed to dispatch op MUL_MAT (f32)`。
- 基线没有测长上下文。作为参考，除 split-KV 外全部移植后、`GGML_SPACEMIT_FA_SPLITKV=0` 时是 6.85 / 5.24。
- 两列来自不同时间的运行，每项 2 次（深度测试 1 次）。

MTP（llama-server，`--spec-type draft-mtp --spec-draft-n-max 1`，贪心 192 token，`GGML_SPACEMIT_LM_Q4=1`；t/s；在同时开两个后端的构建上测，独立后端代码与本分支相同）：

| 上下文（token） | 不开 MTP | MTP（接受 / 草稿） |
|---|---:|---:|
| 30 | 9.58 | 10.16（82/109） |
| 3979 | 8.64 | 9.86（88/102） |
| 8154 | 7.67 | 8.95（89/102） |

- 精度：每个提交都用同一构建里的 legacy 路径（`GGML_SPACEMIT_BACKEND=0 -ngl 0`）对比。
  - 96 token 贪心输出每步都逐字相同。
  - `-t 2` 与 `-t 8` 逐字相同。
  - 最终 KLD：`-c 512` 是 0.0066（top-1 92.4%），`-c 2048 -ub 512` 是 0.0074（top-1 96.6%）。两条路径的 RMS_NORM、softmax、RoPE、FA 实现不同，这个量级是正常的数值差。

## 提交

| 提交 | 内容 | pp512 |
|---|---|---:|
| qwen35/qwen3next: keep the conv-state concat inputs alive until ssm_conv | 把 concat 的两个输入挂到 SSM_CONV 的额外 src 上，让分配器保留到 conv 为止，后端才能直接读它们（下面的 concat → ssm_conv 融合依赖这一点；不读额外 src 的后端不受影响） | — |
| dispatch Q5_0 MUL_MAT / MUL_MAT_ID | `compute_forward` 漏了 Q5_0（`supports_op` 和 repack 都接受），Qwen3.6 的共享专家是 Q5_0 | 34.3～35.7 |
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
SPERT=/home/linpengt/spine-runtime.riscv64.0.6.3
M=/home/linpengt/models/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
cmake -B build-k3 -DCMAKE_BUILD_TYPE=Release -DGGML_SPACEMIT=ON -DSPERT_DIR=$SPERT \
  -DGGML_CPU_REPACK=OFF -DLLAMA_OPENSSL=OFF -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZBA=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/riscv64-spacemit-linux-gnu-gcc.cmake
cmake --build build-k3 --parallel 8 --target llama-bench llama-completion llama-perplexity
export LD_LIBRARY_PATH=$SPERT/lib GGML_SPACEMIT_WORKERS=8 GGML_SPACEMIT_BACKEND=1
build-k3/bin/llama-bench -m $M -dev SPACEMIT0 -ngl 99 -t 8 -fa 1 -lm none -b 2048 -ub 512 -p 512,2048 -n 32 -r 2 -o jsonl
build-k3/bin/llama-bench -m $M -dev SPACEMIT0 -ngl 99 -t 8 -fa 1 -lm none -b 2048 -ub 512 -p 0 -n 32 -d 4096,8192 -r 1 -o jsonl
```

以上是板上实际运行的命令。正确性（贪心输出、KLD）和 MTP 的命令及原始日志见 PR 说明（chtlp/llama.cpp#1）。

结果在 `37acff6dd` 上测得。之后的 `common : drop the stale duplicate -lm/--load-mode option` 只改 `common/arg.cpp`（`mtmd-backend` 合并上游时留下了两份 `-lm` 定义，`test-arg-parser` 失败），没有重跑板上测试。

完整的测试脚本和原始输出：[chtlp/edge-ai-exp](https://github.com/chtlp/edge-ai-exp) `bench-results/k3/20261017-k3-dev006-qwen36-standalone-port/`（逐步对比、最终对比、decode/MTP）和 `20261017-k3-dev006-qwen36-standalone-backend/`（移植前的逐算子 trace）。
