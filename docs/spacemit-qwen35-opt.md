# SpacemiT K3：Qwen3.5/3.6 MoE 优化分支（`linpengt/qwen35-opt`）

本文汇总分支 `linpengt/qwen35-opt` 相对基线 `origin/mtmd-backend`（`51c81f90f`，即 spacemit-com/llama.cpp 默认分支）的全部改动和 K3 上的测试结果。

- 测试环境：`k3-dev-006`，8×A100 @1.8 GHz，legacy CPU IME 路径（`GGML_SPACEMIT_BACKEND=0 -ngl 0`），FP16 KV，`-t 8 -fa on -mmp 0 -b 2048 -ub 512`。
- 主要模型：bartowski `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`（`qwen35moe`：40 层，30 层 Gated DeltaNet + 10 层全注意力，head_dim 256，256 专家激活 8 个，带 1 层 MTP）。部分改动也在 Qwen3-30B-A3B Q4_0 上测过。
- 第 1～4 节的改动都在 legacy CPU 路径（`ggml-cpu/spacemit`）上。
- **2026-10-17 更新**：分支 `linpengt/qwen35-opt-rebase` 是本分支 rebase 到 `mtmd-backend` `6df9f7504` 的版本，并把这些优化移植到了独立 `ggml-spacemit` 后端（`-dev SPACEMIT0 -ngl 99`），见第 0 节。

## 结论

| 指标（Qwen3.6-35B-A3B，原始 Q4_0 GGUF） | 基线 | 本分支 | 变化 |
|---|---:|---:|---:|
| 正确性 | **输出乱码**（`3333…`） | 正确 | — |
| pp512（t/s） | 16.9¹ | **121.6** | 7.2× |
| pp2048（t/s） | 16.1¹ | **110.6** | 6.9× |
| tg（短上下文，t/s） | 6.2¹ | **8.3** | 1.34× |
| decode 4K（t/s） | — | **7.5** | |
| decode 4K / 8K，MTP 投机解码（t/s） | — | **9.6 / 8.9**² | |

¹ 基线直接运行会乱码，此处是仅关闭 Q5_0 IME（第一个修复）后的数字。
² MTP 数字使用 lm_head 改为 Q4_0 的模型变体（`-shexpq8-lmq4`），用 llama-server `--spec-type draft-mtp --spec-draft-n-max 1` 测得。

Qwen3-30B-A3B（只覆盖 Qwen3 阶段的改动）：pp512 54.0 → **76.9**（+42%），pp2048 39.4 → **57.9**（+47%），decode 不变（tg64 约 12.1）。

## 0. rebase 与独立后端（`linpengt/qwen35-opt-rebase`，2026-10-17）

**rebase**：32 个提交 rebase 到 `mtmd-backend` `6df9f7504`（1594 个上游提交）。

- 冲突：`repack.cpp` 两处（上游新增 X60 的 Q8_0 IME1 repack，两边都保留）；`compute_forward` 里的 Q5_0 case 和上游 #26792 重复，已在引入它的提交里改掉。
- legacy 路径 rebase 前后对比：

  | | rebase 前 | rebase 后 |
  |---|---:|---:|
  | pp512（t/s） | 121.1 | 124.2 |
  | pp2048 | 110.9 | 112.6 |
  | tg32 | 8.82 | 8.82 |

- 贪心输出逐字相同。KLD 0.0051 全部来自上游 #28068：GDN 的 q/k 归一化改为 eps 在根号内的 `rms_norm` + `scale`，与 FLA 参考实现一致；把这一行改回后 KLD 是 0.0001。这个改动让 RVV L2_NORM 对 Qwen3.6 不再生效，归一化多约 0.04 ms/token。

**新增 `7f47630a9`**：lm_head Q6_K → Q4_0 在加载时转换（`GGML_SPACEMIT_LM_Q4=1`，默认关）。用原始 GGUF 就能得到 `-shexpq8-lmq4` 文件的 decode 速度（tg32 9.41 t/s；Q6_K 的 lm_head 是 8.82），贪心输出与该文件相同。

**独立后端**：上游原版加 Q5_0 分派修复后，pp512 是 35.7 t/s，与我们优化之前的 legacy（35.3）相当。trace 显示瓶颈全部在 kernel：

| 算子 | ms/token |
|---|---:|
| 转置 concat（标量） | 4.9 |
| 逐 token 递推的 GDN | 4.3 |
| 标量 FA（head_dim 256 不支持） | 3.3 |
| 原版 MoE / dense 矩阵乘 | 11.2 |
| 标量 ssm_conv | 1.8 |

图被 MoE 路由的 ARGSORT/CLAMP 切成 163 段，但这部分只占约 0.6 ms/token，不是主因。

独立后端的 `ime2_kernels.cpp`、`repack.cpp`、`ime_kernels.h` 与 legacy 原版逐字节相同，`ime.cpp` 的矩阵乘类只是接口不同（`params` → `ctx`），所以按以下 7 个提交移植。每步都在同一 binary 上与 legacy 对比，贪心输出都逐字相同：

| 提交 | 内容 | pp512 |
|---|---|---:|
| `28d9ecbc3` | Q5_0 分派（不补就跑不起来） | 35.7 |
| `da18597d0` | RVV 转置 concat、ssm_conv | 44.6 |
| `6b6506a87` | FA head_dim ≤ 256 | 51.7 |
| `de2e9ad1d` | 分块 GDN | 61.0 |
| `5013fc6fb` | legacy 矩阵乘类整体移植（m8、A 对齐、Q8_0 预排、MoE m4/m8、路由 Q8_0、Q5_0 → Q8_0、lm_head Q4_0） | 102.6 |
| `5bc2583a3` | 图级融合（concat → ssm_conv、MoE 路由权重、RMS_NORM + MUL），在主机上于 launch 之前决定 | 127.3 |
| `a6f7d2365` | ARGSORT、CLAMP 放到设备上（切图从 163 段降到 3 段，pp512 不变） | 127.0 |

最终对比（同一 binary，各 2 次）：

| | legacy | 独立后端 |
|---|---:|---:|
| pp512（t/s） | 123.8 | **127.4** |
| pp2048 | 111.8 | **115.1** |
| tg32 | 8.66 | **8.98** |
| tg32，LM_Q4 | 9.35 | **9.52** |

精度：独立后端对 legacy 的 KLD，`-c 512` 是 0.0066，`-c 2048` 是 0.0074（top-1 96.6%）。

尚未移植：
- split-KV decode FA 和 MTP 验证（`9f1f5f143`、`cac4ebc65`），长上下文 decode 和 MTP 要单独比较；
- `GGML_OP_STATS` 子阶段计时；
- GDN 的 IME 版本（legacy 上默认关）。

另外，上游的 trace 分析器（`GGML_BUILD_PROFILE`）只能用静态库构建：`ggml-profile.h` 的函数没有 `GGML_API`。静态构建又不能同时开两个后端，因为两边的 `spacemit_kernels::*` 符号同名。

## 1. 改动

26 个代码提交，19 个文件，+5014/−226（`git diff --stat origin/mtmd-backend...linpengt/qwen35-opt -- . ':!docs'`）。

### 1.1 提交列表

| 提交 | 内容 | 主要影响 |
|---|---|---|
| `bcede787a` | `GGML_OP_STATS` 算子级计时（spert 整图路径，按线程/节点/形状统计；`MUL_MAT_ID` 子阶段计时） | 工具 |
| `214fefb17` | op stats 从 embedding `GET_ROWS` 识别图的 token 数 | 工具 |
| `c7f1d0102` | MoE `MUL_MAT_ID` 的 hp 路径改用 4 行 kernel（协同 4 行量化 + m4 kernel + scatter）；修复 `moe_gemm_kernel_m2` 未初始化 | PP |
| `028883791` | tiled FA 中用向量比较对 f16 mask 块分类（全 `-inf` 跳过、全 0 省略加法） | PP（长输入） |
| `a91229c8b` | Q5_0 IME repack 改为 opt-in（修复乱码）；新增 `GGML_SPACEMIT_RVV_OFF` / `REPACK_OFF` 调试开关 | 正确性 |
| `50119f7bf` | op stats 按权重类型拆分矩阵乘 | 工具 |
| `7f30b5626` | RVV kernel：`GATED_DELTA_NET` 单遍融合、`SSM_CONV` 寄存器滑窗、转置输入 `CONCAT` 连续读 | PP、TG |
| `3ca97beb9` | flash attention 支持 head_dim ≤ 256（VLEN 1024，decode 按 128 分段） | TG |
| `9f1f5f143` | GQA split-KV decode（flash-decoding，一个 KV 头的所有 q 头一起算，K/V 只读一次） | TG（长上下文） |
| `cac4ebc65` | MTP 验证：split-KV 支持 1～4 个 query token 单遍处理；GDN kernel 支持回滚快照；op stats 表扩到 16384 项 | MTP |
| `4d99b2c1d` | Q4_0 hp kernel：A 的 4 行布局 128B 对齐、B 存成有符号 int4、m4 kernel 软件流水；单元测试 `tests/test-spacemit-hp-aligned.cpp` | PP |
| `7e4078126` | Q5_0 权重在 repack 时无损转成 Q8_0，走 Q8_0 IME kernel | 直接使用原始 GGUF |
| `cd4b405c2` | 图级融合：concat→ssm_conv、MoE 路由权重并入 MUL_MAT_ID；prefill GDN 分块（chunkwise）实现 | PP |
| `939d0eb6d` | `build_ssm_conv`：把 concat 输入挂为 ssm_conv 的额外 src，保证融合时不被提前回收（qwen35/qwen35moe/qwen3next） | 正确性 |
| `c6a63fc5c` | 分块 GDN 默认分块 16 | PP |
| `4c3b707f7`、`de8dbe2f2` | m8 kernel：A 按 8 行一组量化并预拆 nibble，用满 `vmadot.hp` 的 8 行；MoE 先跑 8 行组 | PP |
| `dc353a5d8` | dense Q4_0 hp 矩阵乘也用 m8 | PP |
| `2d71832db`、`b089d9baa`、`3e6186a6b` | 分块 GDN 的矩阵乘上 IME FP16（`vfwmadot`），默认关闭 | 实验 |
| `deb515e90` | op stats 拆分 MoE / dense 子阶段并记录字节数和形状；`GGML_SPACEMIT_DENSE_MBLK` 实验开关 | 工具 |
| `2399d559a` | 软件优化第一批：dense M 分块默认 8；RMS_NORM+MUL 融合、SIGMOID、L2_NORM 改 RVV；MoE 输出写出时一并乘路由权重 | PP |
| `59fc85716` | 第二批：MoE 每个专家 1～7 行的余数补零走一次 m8；相邻 up/gate 共用路由表和 8 行量化结果 | PP |
| `d232ff1c5` | 第三批：MoE 路由（F32 `ffn_gate_inp`）加载时量化为 Q8_0，走 IME（有损，见 2.5） | PP |
| `b181952e9` | 第四批：Q8_0 权重在 repack 时排成 `vmadot` i8 需要的布局（去掉每块 4 条 `vupack`）；dense 4 行 A 改为 64B 对齐的 192B 块；单元测试 `tests/test-spacemit-q8-pre.cpp` | PP |

### 1.2 改动说明

**正确性**

- 基线乱码的原因是 legacy IME 的 **Q5_0 矩阵乘 kernel 结果错误**。Qwen3.6 的 20 层共享专家（60 个张量）是 Q5_0，Qwen3-30B 没有 Q5_0，所以之前没暴露。`a91229c8b` 把它改为默认关闭（`GGML_SPACEMIT_Q5_0_IME=1` 才开）。
- 关闭后 Q5_0 走通用实现，慢约 10 倍。`7e4078126` 在 repack 时把 Q5_0（`d × (q − 16)`，`q ∈ [0, 31]`）直接存成 Q8_0（`qs = q − 16`，`d` 不变），**完全无损**，之后走现有 Q8_0 IME kernel，只多约 24 MB 内存。不再需要离线转换模型。

**Prefill**

- **MoE 4 行 kernel**（`c7f1d0102`）：原来 Q4_0 专家的 hp 路径只有 M=1 kernel，96% 的行逐行 GEMV。改为每个专家的行按 4 行一组走 m4 kernel，不满 4 行的余数仍走 m1。
- **FA fastmask**（`028883791`）：原 tiled FA 把每个 64×64 mask 块解包成 f32 再逐元素判断，占 FA 时间的 43.5%；改后降到 3.8%。数值逐位不变。
- **DeltaNet RVV kernel**（`7f30b5626`）：`GATED_DELTA_NET` 原来 memcpy + 4 遍扫状态，改为单遍；`CONCAT` 原来标量 gather。Qwen3.6 pp512 35.3 → 58.0 t/s。
- **hp m4 对齐 kernel**（`4d99b2c1d`）：A100 上非 64B 对齐的整寄存器加载慢 2.7～7.7 倍，而原 4 行 A 布局几乎每次加载都不对齐。新布局每个 K256 块 = `[8×128B int8][8×8B scale][72B 尾部][补齐]` = 1216B，数值与原布局逐位相同；B 在 repack 时存成有符号 int4（`qs ^ 0x88`），去掉零点补偿；m4 kernel 展开 ×2 + 软件流水，hi/lo nibble 分开 fp16 累加。m4 每个 4×32×K32 子块 153～170 → 59～66 周期（2.5 倍）。m1 kernel 也改用有符号 B。

- **图级融合**（`cd4b405c2`、`939d0eb6d`）：每次图计算前由 `ggml_backend_cpu_riscv64_spacemit_graph_prepare()` 检查消费者，结果记在节点 op_params 的空闲槽。
  - conv：SSM_CONV 直接读 concat 的两个输入（conv state 和未转置的 qkv，按通道连续），CONCAT 只写回被 conv state 更新读取的末尾几列。原 CONCAT 是 strided 写、原 SSM_CONV 是 strided 读，两者都去掉。模型侧 `build_ssm_conv` 把 concat 输入挂为 ssm_conv 的 src[2]/src[3]，否则分配器在 CONCAT 后就回收它们（第一版因此输出乱码）。
  - moe_w：MUL_MAT_ID 写每行输出时乘上该行的路由权重，后面的 `MUL:ffn_moe_weighted` 变成 no-op。
  - 两者浮点运算和顺序不变，输出逐位相同。
- **分块 GDN**（`cd4b405c2`、`c6a63fc5c`）：prefill（K=1、标量 gate、≥16 token）改为分块 WY 形式：块内 `U = (I+L)^-1 β(V − Γ K S0)`（前向代入），`O = scale(Γ Q S0 + (E∘QKᵀ) U)`，`S ← Γ_C S + K̃ᵀU`，所有乘积用 4 行分块 RVV fp32 矩阵乘。decode 和 MTP（K>1）仍走递推 kernel。A100 的 FP32 向量峰值只有约 4.2 MAC/周期/核，分块越小 C² 部分越少、越快，默认 16。

- **m8 kernel**（`4c3b707f7`、`de8dbe2f2`、`dc353a5d8`）：一条 `vmadot.hp` 算 8 行 × 8 列 × K32，原 m4 只用其中 4 行。实测 A 寄存器第 r 行为字节 16r～16r+15（k 低 nibble 在前）。A 按 8 行一组量化（每个 K32 子块 8 行共用一个 scale），量化时就拆成 lo（u4）/ hi（s4）两个寄存器镜像（每 K256 2176B），kernel 每子块只做两次对齐加载、无需拆 nibble，成对软件流水。微基准比两次 m4 快 1.55～1.65 倍。MoE 每个专家先跑 8 行组、再最多一个 4 行组；dense 按 8 行一组、尾部 4 行 / 1 行。
- **软件优化第一批**（`2399d559a`）：
  - dense M 分块：dense Q4_0 矩阵乘每个线程一次处理连续 R 个行组（默认 R=8），B 的每个 tile 每 R 组读一次，而不是每组读一次。
  - RMS_NORM+MUL：ggml-cpu 已把 `RMS_NORM` 和后面的 `MUL(weight)` 融合，但融合后走的是标量通用代码；改为一个 spacemit RVV kernel 直接写 MUL 的输出。
  - SIGMOID、L2_NORM：原来走标量通用实现，改为 RVV。
  - MoE 输出写出：m8/m4 每行结果原来先 memcpy 到 dst 再原地乘路由权重（两遍），改为拷贝时一并乘（一遍）。
  - 全部开启 pp512 90.0 → 105.8 t/s（+17.6%），与 kernel 级预期（合计省 1.54 ms/token）基本一致；单项：只开 M 分块 93.0，只开 RMS_NORM+MUL 94.9。贪心生成与全关逐字相同。
- **第二批**（`59fc85716`）：
  - MoE 余数补零走 m8：每个专家在整 8 行组之后剩下的 1～7 行（包括总共不足 8 行的专家），原来拆成一个 4 行组加若干单行（单行走 m1 GEMV），改为补零凑成一个 8 行组走 m8，只写回真实行。
  - gate/up 共用量化：同一层相邻的 `ffn_moe_up` 和 `ffn_moe_gate` 读同一份激活、同一组专家 id，第二个直接复用第一个留在 work buffer 里的路由表和 8 行量化结果。
  - pp512 105.9 → 110.9 t/s（只开补零：107.5），省 0.42 ms/token，kernel 级预估 0.53。
- **第三批**（`d232ff1c5`）：路由矩阵（F32，2048×256）原来走通用 RVV fp32，加载时逐行量化为 Q8_0 后走 Q8_0 IME kernel，路由 matmul 0.73 → 0.08 ms/token。pp512 111.0 → 118.9 t/s（+7.2%）。有损：KLD 0.0061 与其他小数值改动同级，但 top-1 一致率 92.0%，低于常见的 94.7%（路由 logits 的误差会改变少数 token 的专家选择）。
- **第四批**（`b181952e9`）：
  - `gemm_kernel_i8i8` 每个 K32 块要用 4 条 `vupack.vv` 把 B 现场排成 i8 模式的 K16 布局；实测 `vupack.vv vd, vs2, vs1, 1`（e32）把每个 256B 寄存器对的第 `(o % 8) * 2 + o / 8` 个 16B 块放到第 `o` 块，repack 时直接按这个顺序存，kernel 只剩加载。Q5_0、F32 路由转成的 Q8_0 也一样。
  - 原 4 行 A 块长 152B，几乎每次 `vl1r` 都跨 64B 边界；改为 192B（数据在 +64），每行 48B；单行仍是连续的 38B 块，放在同样的行槽里。work buffer 按新尺寸计算并 128B 对齐。
  - kernel 单核 225～245 → 137～147 周期/块（结果逐位相同）；模型里 Q8_0 matmul −20～25%，pp512 118.3 → 121.6 t/s。kernel 的 40% 只兑现一半，剩下的时间在 A 量化、TCM 拷贝和 m1 行上。
  - 开关在加载线程上决定，加载线程在 X100 核（VLEN 256），不能在开关里判断 `vlenb`（第一版因此整条路径没有生效）。
- **GDN 上 IME FP16**（`2d71832db`、`b089d9baa`，默认关）：分块 GDN 的 6 个矩阵乘改成 `vfwmadot`（fp16 输入、fp32 累加），状态在 head 内以 fp32 8×8 tile 保存。单线程比 RVV fp32 快 28%，但 8 线程下两个 hart 共享 Tensor Core，GDN 反而 1.43 对 1.29 ms/token，故默认关闭（`GGML_SPACEMIT_GDN_IME=1` 打开）。

**Decode / MTP**

- **FA head_dim 256**（`3ca97beb9`）：10 层全注意力原来走通用 FA。decode 上下文斜率 19.3 → 7.2 µs/token。
- **split-KV decode**（`9f1f5f143`）：decode 4K 6.46 → 7.48 t/s。
- **MTP 验证**（`cac4ebc65`）：原始 MTP 实现比不开还慢（4K 4.60 t/s）。多 token 验证改为单遍 split-KV，GDN 回滚快照走 RVV kernel，4K 4.60 → 8.84 t/s。

### 1.3 运行时开关

都在 legacy 路径。性能相关开关默认开，设为 `0` 关闭，方便同一 binary 内 A/B。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `GGML_SPACEMIT_MOE_M4` | 开 | MoE 4 行 kernel |
| `GGML_SPACEMIT_FA_FASTMASK` | 开 | FA mask 向量化分类 |
| `GGML_SPACEMIT_FA_SPLITKV` | 开 | split-KV decode |
| `GGML_SPACEMIT_HP_ALIGNED` | 开 | Q4_0 hp：对齐 A 布局 + 有符号 int4 B + 流水 m4 |
| `GGML_SPACEMIT_Q5_0_AS_Q8` | 开 | Q5_0 加载时转 Q8_0 |
| `GGML_SPACEMIT_FUSE_CONV` | 开 | concat→ssm_conv 融合 |
| `GGML_SPACEMIT_FUSE_MOE_W` | 开 | MoE 路由权重并入 MUL_MAT_ID |
| `GGML_SPACEMIT_GDN_CHUNK`、`GGML_SPACEMIT_GDN_CHUNK_SIZE` | 开、16 | prefill 分块 GDN 及分块大小（8～128） |
| `GGML_SPACEMIT_HP_M8` | 开 | Q4_0 hp 的 8 行 kernel（MoE 与 dense） |
| `GGML_SPACEMIT_DENSE_MBLK` | 8 | dense 矩阵乘每线程连续处理的行组数（1 = 原行为） |
| `GGML_SPACEMIT_RVV_RMS_MUL` | 开 | 融合的 RMS_NORM+MUL 走 RVV |
| `GGML_SPACEMIT_RVV_SIGMOID`、`GGML_SPACEMIT_RVV_L2` | 开 | SIGMOID、L2_NORM 走 RVV |
| `GGML_SPACEMIT_MOE_STORE1` | 开 | MoE 输出写出时一并乘路由权重 |
| `GGML_SPACEMIT_MOE_PAD8` | 开 | MoE 余数补零走 m8 |
| `GGML_SPACEMIT_MOE_REUSEQ` | 开 | 相邻 up/gate 共用路由表和量化结果 |
| `GGML_SPACEMIT_ROUTER_Q8` | 开 | MoE 路由加载时转 Q8_0 走 IME（有损） |
| `GGML_SPACEMIT_Q8_PRE` | 开 | Q8_0 权重预排 + 对齐的 4 行 A |
| `GGML_SPACEMIT_DUMP_IDS=<file>` | — | 调试：导出 prefill 时 `ffn_moe_up` 的专家 id |
| `GGML_SPACEMIT_GDN_IME` | **关** | 分块 GDN 的矩阵乘走 IME FP16 |
| `GGML_SPACEMIT_Q5_0_IME` | 关 | 打开有问题的 Q5_0 IME kernel（仅调试） |
| `GGML_SPACEMIT_RVV_OFF=op,...` / `ALL` | — | 指定 op 走通用实现 |
| `GGML_SPACEMIT_REPACK_OFF=type,...` | — | 指定量化类型不走 IME（如 `q4_0`，可用作精度参考） |
| `GGML_SPACEMIT_FA_DEBUG=N` | 0 | 打印前 N 次小 batch FA 的路径与形状 |
| `GGML_OP_STATS=1`、`GGML_OP_STATS_FILE=<tsv>` | 关 | 算子级计时（pp 开销约 1%，decode 约 8%） |

## 2. Benchmark 结果

### 2.1 Prefill（Qwen3.6）

逐步累计（pp512 / pp2048，t/s）：

| 阶段 | 模型 | pp512 | pp2048 |
|---|---|---:|---:|
| 基线 + Q5_0 IME 关闭 | 原始 | 16.9 | 16.1 |
| + 共享专家 Q8_0（离线转换） | `-shexpq8` | 35.3 | 32.1 |
| + DeltaNet RVV kernel | `-shexpq8` | 58.0 | — |
| + FA head 256（`3ca97beb9`） | `-shexpq8` | 61.4 / 62.2 | 59.0 / 58.9 |
| + hp m4 对齐 kernel（`4d99b2c1d`） | `-shexpq8` | 69.7 | 65.4 |
| + Q5_0 加载时转 Q8_0（`7e4078126`） | 原始 | 69.7 | 65.8 |
| + concat→ssm_conv、MoE 权重融合（`939d0eb6d`） | 原始 | 75.7 | 71.1 |
| + 分块 GDN，分块 16（`c6a63fc5c`） | 原始 | 83.2 | 77.7 |
| + m8 kernel（MoE + dense，`dc353a5d8`） | 原始 | 90.4 | 83.9 |
| + 软件优化第一批（`2399d559a`）⁴ | 原始 | 105.8 | 96.8 |
| + 第二批：MoE 补零 m8、gate/up 共用量化（`59fc85716`） | 原始 | 110.9 | 101.2 |
| + 第三批：路由 Q8_0（`d232ff1c5`） | 原始 | 118.9 | 108.2 |
| + 第四批：Q8_0 预排（`b181952e9`） | **原始** | **121.6** | **110.6** |

MoE m4 和 fastmask 在以上各阶段都开着。

⁴ 第一到第四批都在同一 binary 内用开关 A/B，每项 2 次，板上无其他负载（edge-ai-exp `bench-results/k3/20261016-k3-dev006-qwen36-sw-integrate`）。第一批全关为 90.0 / 83.5；每批的"关"与上一批的"开"相差不超过 0.6 t/s。最后一行与上一行速度相同，说明加载时转换等价于离线转换。

长输入（`3ca97beb9`，hp 对齐之前）：4K 55.4、8K 49.4 t/s；llama-server 实测 4K 输入约 72 s、8K 约 2.7 min。对齐 kernel 后长输入未重测。

hp 对齐 kernel 的 op stats（pp512，ms/token）：

| 项 | 对齐前 | 对齐后 |
|---|---:|---:|
| 整图 | 16.15 | 14.33 |
| MoE `MUL_MAT_ID` | 4.40 | 2.94 |
| MoE kernel 吞吐 | 601 GOP/s | 1081 GOP/s |
| dense Q4_0 `MUL_MAT` | 2.46 | 2.11 |

### 2.2 Decode（Qwen3.6，单 token，t/s）

| 上下文 | 基线¹ | + 共享专家 Q8_0 | + FA head 256 | + split-KV | 本分支最终 |
|---:|---:|---:|---:|---:|---:|
| 短 | 6.21 | 7.51 | 8.28 | 8.33 | 8.31～8.54 |
| 2K | — | — | 7.38 | — | |
| 4K | — | — | 6.46 | **7.48** | |
| 8K | — | — | 5.12 | 6.87（server） | |
| 16K | — | — | 3.66 | 未测 | |

hp 对齐 kernel 对 decode 影响很小（tg32 8.27 → 8.54，单次，在波动范围内）。

### 2.3 MTP 投机解码（Qwen3.6，llama-server，贪心 192 token）

| 上下文 | 不开 MTP | MTP（Q6_K lm_head） | MTP（Q4_0 lm_head） | 草稿接受率 |
|---:|---:|---:|---:|---:|
| 短 | 8.33 | 9.42 | — | 74% |
| 4K | 7.62 | 8.84 | **9.58** | 78～83% |
| 8K | 6.87 | 8.49 | **8.93** | 85～88% |

MTP 输出与不开 MTP 逐字一致（4K、8K）。Q4_0 lm_head 是模型变体（离线重量化 `output.weight`），不是代码改动。

### 2.4 Qwen3-30B-A3B（Qwen3 阶段改动）

| 配置 | pp512 | pp2048 | tg64 |
|---|---:|---:|---:|
| 基线 | 54.0 | 39.4 | 12.0～12.2 |
| + MoE m4 | 67.9 | 46.1 | 不变 |
| + FA fastmask | **76.9** | **57.9** | 不变 |
| + head 256 / split-KV（回归检查） | — | — | 12.10 |

hp 对齐 kernel 和 split-KV 在 Qwen3-30B 上没有做端到端 A/B（模型已不在板上）。

### 2.5 精度

| 改动 | 对照 | Mean KLD | Top-1 一致 | PPL 比 |
|---|---|---:|---:|---:|
| MoE m4（Qwen3-30B） | 原路径 | 0.0038 | 97.0% | 1.004 ± 0.005 |
| FA fastmask | 原路径 | ≈ 0 | 100% | 1.000 |
| DeltaNet RVV kernel | 通用实现 | ≈ 0（max 6e-5） | 100% | 1.000 |
| FA head 256 | 通用 FA | 0.0096 | 95.1% | 0.996 ± 0.004 |
| hp 对齐 kernel | 高精度参考³ | **0.0049** | 93.5% | 0.998 ± 0.005 |
| （对照）原 hp kernel | 高精度参考³ | 0.0229 | 88.0% | 1.009 ± 0.009 |
| Q5_0 加载时转 Q8_0 | 离线 `-shexpq8` | 0.0035 | 93.7% | 1.003 ± 0.004 |
| conv / MoE 权重融合 | 未融合 | ≈ 0（max 4.8e-5） | 100% | 1.000 |
| 分块 GDN（分块 16） | 递推 | 0.0046 | 92.9% | 1.004 ± 0.005 |
| m8（MoE + dense） | m4 | 0.0054 | 95.3% | 0.995 ± 0.004 |
| GDN IME FP16（默认关） | RVV fp32 分块 | 0.0055 | 94.9% | 0.999 ± 0.005 |
| 软件优化第一批 | 第一批全关 | 0.0058 | 94.7% | 0.996 ± 0.005 |
| 第一 + 第二批 | 第一批全关 | 0.0057 | 94.7% | 1.001 ± 0.004 |
| 第三批：路由 Q8_0 | 第三批关 | 0.0061 | **92.0%** | 0.999 ± 0.005 |
| 第四批：Q8_0 预排 | 第四批关 | ≈ 0（max 5.5e-5） | 100% | 1.000 |

³ `GGML_SPACEMIT_REPACK_OFF=q4_0`：Q4_0 不 repack，走通用 Q4_0×Q8_0 点积，fp32 累加。

- hp 对齐 kernel 比原 kernel **更准**：原 kernel 用 fp16 累加无符号 0～15 权重再减零点，误差更大。
- FA head 256 的差异来自 fork 原有设计（Q×scale 转 f16 后做 f16 乘加），不是本分支引入的。
- 分块 GDN 的 KLD 与分块大小无关（16～128 都是 0.004～0.005），推导在 fp64 下与递推一致到 1e-16；该 MoE 模型对任意微小数值差异都表现为约 0.0035 的 KLD（见 Q5_0 一行），贪心生成与递推逐字相同。
- Qwen3.6 精度测试只有 2×512 token。

## 3. 当前瓶颈与 TODO

### 3.1 Prefill（pp512 8.22 ms/token = 121.6 t/s；软件优化前 11.1 ms/token = 90.4 t/s）

第四批之后的 op stats（pp512，ms/token，含约 1% 统计开销）：

| 类别 | 软件优化前 | 现在 |
|---|---:|---:|
| MoE `MUL_MAT_ID` | 3.07 | 2.41 |
| dense Q4_0 `MUL_MAT` | 1.36 | 1.11 |
| Q8_0 / Q5_0（转 Q8_0）`MUL_MAT` | 约 1.1 | 0.79 |
| F32 router | 0.73 | 0.08 |
| 其他 `MUL_MAT`（约 70 个小 F32 matmul 等） | — | 0.42 |
| `GATED_DELTA_NET`（分块） | 1.29 | 1.29 |
| attention（FA） | 0.32 | 0.31 |
| 其余（norm、逐元素、SSM_CONV、非线性等） | 约 2.7 | 1.84 |

下表是软件优化前的估算与当前状态：

INT8 口径 MAC 利用率约 4.0%（11 TOPS 实测峰值）。8 核扩展、各算子扩展性与对 K5 的分析见 edge-ai-exp 仓库 `docs/k3_qwen36_prefill_bottleneck_and_k5_implications_20261015.md`。TODO 按预计收益排序（均未实测）：

| # | 方向 | 预计省 ms/token | 状态 |
|---|---|---:|---|
| — | concat→ssm_conv、MoE 权重融合 | 1.1 | **已完成**（实测 −1.1） |
| — | GDN 分块 | 1.5～2.5 | **已完成，RVV fp32 版**（实测 −1.27）；受 A100 FP32 向量峰值限制 |
| 1 | GDN 块内矩阵乘改用 IME FP16（`vfwmadot`，fp32 累加） | 约 1 | **已实现，默认关**：单线程 −28%，8 线程反而 +0.14 ms（Tensor Core / L2 共享）；剩余开销在 expf、读入、打包 |
| 2 | MoE kernel 叠加 B 复用和批量 scale；去掉 gate/up 对同一 token 的重复 4 行量化（0.46） | 约 1 | **部分完成**（`59fc85716`）：gate/up 共用量化、余数补零走 m8；B 复用与批量 scale 未做 |
| 3 | m8 kernel（用满 IME 8 行） | 待定 | **已完成**：dense Q4_0 2.16 → 1.36 ms，MoE −0.08 ms；pp512 +8.3% |
| 4 | router 改 fp16 / IME（扩展只有 3.65×，cluster 内带宽受限；decode 也受益） | 约 0.6 | **已完成，Q8_0 版**（`d232ff1c5`）：实测 −0.60；有损，见 2.5 |
| 5 | Q8_0 kernel 做同样的对齐 + 流水 | 约 0.5 | **已完成对齐 + B 预排**（`b181952e9`）：实测 −0.23；流水和 8 行未做 |
| 6 | RMS_NORM / ADD 等融合（逐元素算子不随线程扩展） | 0.5～1 | **部分完成**（`2399d559a`）：RMS_NORM+MUL、SIGMOID、L2_NORM 改 RVV；ADD 等未做 |
| — | dense M 分块、MoE 单遍写出 | — | **已完成**（`2399d559a`），与上一行合计 pp512 +17.6% |
| 7 | 默认 ub=2048（实测 pp4096 +3.6%，来自 TCM 拷贝摊薄和 m1 行减少） | 约 0.5 | |
| 8 | dense 投影改循环顺序（原假设"受 DDR 限制"未被扩展实测支持） | ≤ 0.3 | 降级 |

长输入下 FA 的 QK^T / PV 仍按 FP32 向量速率执行，可考虑块内 FP16 累加或上 IME。

### 3.2 Decode（目标：4～8K 下 ≥ 10 t/s）

单 token decode 的带宽上限约 12.3 t/s（每 token 读约 2.03 GB 权重，STREAM 25 GB/s）。现在短上下文约 120 ms/token，比下限多约 40 ms。按优先级：

1. **草稿采样放 CPU**（`--no-spec-draft-backend-sampling`，已有参数，未测）：MTP 约 −5 ms/步。
2. **lm_head 用 imatrix 重量化或改 Q4_K**：Q4_0 lm_head 的 top-1 一致率只有 92.6%。
3. **DeltaNet 状态原地更新**（现在每层 2 MiB × 30 层 `GET_ROWS` + `CPY`）：约 −10 ms/token。
4. **小算子融合 / RVV 化**（RMS_NORM、SIGMOID、SILU、ARGSORT 等）：约 −8 ms/token。
5. router 权重 f16：约 −2 ms/token。

估计 1～3 完成后 MTP 在 4～8K 可到 10 t/s 以上。

### 3.3 验证与工程

- [ ] Qwen3-30B-A3B 上补测 hp 对齐 kernel 和 split-KV（pp512/pp2048/tg + KLD）。
- [ ] qwen35（dense）、qwen3next 上验证 `build_ssm_conv` 融合（只在 Qwen3.6 上测过）。
- [ ] 对齐 kernel 后重测长输入 prefill（4K/8K/16K）和 16K decode。
- [ ] 重复次数补到 2 次以上，KLD 用 4×512 token。
- [ ] `quantize_a_4row_i8_hp_al` 的 vlenb≠128 分支在其他 VLEN 上测试。
- [ ] 向 SpacemiT 报告：Q5_0 IME kernel 结果错误；原 hp kernel 的 A 布局导致非对齐加载；独立 `ggml-spacemit` 后端 `tensor_traits::compute_forward` 的 MUL_MAT / MUL_MAT_ID 漏了 Q5_0（repack 和 `supports_op` 都接受它），Qwen3.6 因此报 `failed to dispatch op MUL_MAT (f32)`，补上后能跑（6df9f7504 上 pp512 34.3、pp2048 23.5、tg32 7.4 t/s），此前误读为不支持 F32 权重；`gemm_kernel_i8i8_m4` 把只读输入 `k_blks` 当计数器递减，`count_n > 32` 时第二个 tile 起只算 1 个 K 块（现有调用方每次只传 32 列，所以没出错）。
- [ ] 路由 Q8_0 的 top-1 一致率（92.0%）用更长文本复核；必要时改 fp16 路由。
- [ ] 第二到第四批只在 Qwen3.6 上测过 prefill；decode、长输入、Qwen3-30B 未重测。

## 4. 复现

```bash
# 板上构建后（cmake 参数见 docs/build-riscv64-spacemit.md）
export GGML_SPACEMIT_BACKEND=0
./build/bin/llama-bench -m Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
    -ngl 0 -t 8 -fa 1 -mmp 0 -b 2048 -ub 512 -p 512,2048 -n 32

# A/B：关闭某项优化
GGML_SPACEMIT_HP_ALIGNED=0 ./build/bin/llama-bench ...
GGML_SPACEMIT_FUSE_CONV=0 GGML_SPACEMIT_FUSE_MOE_W=0 GGML_SPACEMIT_GDN_CHUNK=0 ./build/bin/llama-bench ...

# 算子级计时
GGML_OP_STATS=1 GGML_OP_STATS_FILE=stats.tsv ./build/bin/llama-bench ...

# kernel 单元测试（不在 CMake 构建中，编译命令见源文件头部注释）
./test-spacemit-hp-aligned 200 8   # reps=200, cpu=8
```

## 局限

- 所有数据都在单块 `k3-dev-006`、1.8 GHz 固定频率下测得，板卡同时有其他用户；Qwen3.6 的多数数据每项只跑 1～2 次。
- 精度参考本身有 Q8_0 激活量化，不是 FP32 真值。
- roofline 和 TODO 收益均为估算。
