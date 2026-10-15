# SpacemiT K3：Qwen3.5/3.6 MoE 优化分支（`linpengt/qwen35-opt`）

本文汇总分支 `linpengt/qwen35-opt` 相对基线 `origin/mtmd-backend`（`51c81f90f`，即 spacemit-com/llama.cpp 默认分支）的全部改动和 K3 上的测试结果。

- 测试环境：`k3-dev-006`，8×A100 @1.8 GHz，legacy CPU IME 路径（`GGML_SPACEMIT_BACKEND=0 -ngl 0`），FP16 KV，`-t 8 -fa on -mmp 0 -b 2048 -ub 512`。
- 主要模型：bartowski `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf`（`qwen35moe`：40 层，30 层 Gated DeltaNet + 10 层全注意力，head_dim 256，256 专家激活 8 个，带 1 层 MTP）。部分改动也在 Qwen3-30B-A3B Q4_0 上测过。
- 所有改动只在 legacy CPU 路径（`ggml-cpu/spacemit`）上，独立 `ggml-spacemit` 后端未改动。

## 结论

| 指标（Qwen3.6-35B-A3B，原始 Q4_0 GGUF） | 基线 | 本分支 | 变化 |
|---|---:|---:|---:|
| 正确性 | **输出乱码**（`3333…`） | 正确 | — |
| pp512（t/s） | 16.9¹ | **69.7** | 4.1× |
| pp2048（t/s） | 16.1¹ | **65.8** | 4.1× |
| tg（短上下文，t/s） | 6.2¹ | **8.3** | 1.34× |
| decode 4K（t/s） | — | **7.5** | |
| decode 4K / 8K，MTP 投机解码（t/s） | — | **9.6 / 8.9**² | |

¹ 基线直接运行会乱码，此处是仅关闭 Q5_0 IME（第一个修复）后的数字。
² MTP 数字使用 lm_head 改为 Q4_0 的模型变体（`-shexpq8-lmq4`），用 llama-server `--spec-type draft-mtp --spec-draft-n-max 1` 测得。

Qwen3-30B-A3B（只覆盖 Qwen3 阶段的改动）：pp512 54.0 → **76.9**（+42%），pp2048 39.4 → **57.9**（+47%），decode 不变（tg64 约 12.1）。

## 1. 改动

12 个提交，12 个文件，+2464/−207（`git diff --stat origin/mtmd-backend...linpengt/qwen35-opt`）。

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

### 1.2 改动说明

**正确性**

- 基线乱码的原因是 legacy IME 的 **Q5_0 矩阵乘 kernel 结果错误**。Qwen3.6 的 20 层共享专家（60 个张量）是 Q5_0，Qwen3-30B 没有 Q5_0，所以之前没暴露。`a91229c8b` 把它改为默认关闭（`GGML_SPACEMIT_Q5_0_IME=1` 才开）。
- 关闭后 Q5_0 走通用实现，慢约 10 倍。`7e4078126` 在 repack 时把 Q5_0（`d × (q − 16)`，`q ∈ [0, 31]`）直接存成 Q8_0（`qs = q − 16`，`d` 不变），**完全无损**，之后走现有 Q8_0 IME kernel，只多约 24 MB 内存。不再需要离线转换模型。

**Prefill**

- **MoE 4 行 kernel**（`c7f1d0102`）：原来 Q4_0 专家的 hp 路径只有 M=1 kernel，96% 的行逐行 GEMV。改为每个专家的行按 4 行一组走 m4 kernel，不满 4 行的余数仍走 m1。
- **FA fastmask**（`028883791`）：原 tiled FA 把每个 64×64 mask 块解包成 f32 再逐元素判断，占 FA 时间的 43.5%；改后降到 3.8%。数值逐位不变。
- **DeltaNet RVV kernel**（`7f30b5626`）：`GATED_DELTA_NET` 原来 memcpy + 4 遍扫状态，改为单遍；`CONCAT` 原来标量 gather。Qwen3.6 pp512 35.3 → 58.0 t/s。
- **hp m4 对齐 kernel**（`4d99b2c1d`）：A100 上非 64B 对齐的整寄存器加载慢 2.7～7.7 倍，而原 4 行 A 布局几乎每次加载都不对齐。新布局每个 K256 块 = `[8×128B int8][8×8B scale][72B 尾部][补齐]` = 1216B，数值与原布局逐位相同；B 在 repack 时存成有符号 int4（`qs ^ 0x88`），去掉零点补偿；m4 kernel 展开 ×2 + 软件流水，hi/lo nibble 分开 fp16 累加。m4 每个 4×32×K32 子块 153～170 → 59～66 周期（2.5 倍）。m1 kernel 也改用有符号 B。

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
| + Q5_0 加载时转 Q8_0（`7e4078126`） | **原始** | **69.7** | **65.8** |

MoE m4 和 fastmask 在以上各阶段都开着。最后一行与上一行速度相同，说明加载时转换等价于离线转换。

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

³ `GGML_SPACEMIT_REPACK_OFF=q4_0`：Q4_0 不 repack，走通用 Q4_0×Q8_0 点积，fp32 累加。

- hp 对齐 kernel 比原 kernel **更准**：原 kernel 用 fp16 累加无符号 0～15 权重再减零点，误差更大。
- FA head 256 的差异来自 fork 原有设计（Q×scale 转 f16 后做 f16 乘加），不是本分支引入的。
- Qwen3.6 精度测试只有 2×512 token。

## 3. 当前瓶颈与 TODO

### 3.1 Prefill（pp512 14.3 ms/token = 69.7 t/s）

| 类别 | ms/token | 占比 |
|---|---:|---:|
| 矩阵乘（MoE 2.94、dense Q4_0 2.10、Q8_0 1.11、F32 router 0.73、其他 0.46） | 7.34 | 51% |
| DeltaNet 相关（`GATED_DELTA_NET` 2.56，`CONCAT`/`SSM_CONV`/norm 等 1.99） | 4.55 | 32% |
| 逐元素 / norm | 2.0 | 14% |
| attention（FA） | 0.31 | 2% |

DDR 带宽 roofline 约 1.5 ms/token（估算），目前约达到 10%。TODO 按预计收益排序（均未实测）：

| # | 方向 | 预计省 ms/token |
|---|---|---:|
| 1 | GDN 改为分块（chunkwise）算法，状态读写从每 token 一次降为每块一次，计算交给 IME | 1.5～2.5 |
| 2 | MoE kernel 叠加 B 复用（`mblock2`）和批量 scale；去掉 gate/up 对同一 token 的重复 4 行量化（0.46） | 约 1 |
| 3 | dense 投影改循环顺序，B 读一次供多个 4 行组使用（先确认是否受 DDR 限制） | 0.5～0.8 |
| 4 | router 改 fp16 / IME（当前 F32，58 GOP/s；decode 也受益） | 约 0.6 |
| 5 | Q8_0 kernel 做同样的对齐 + 流水（当前 478 GOP/s） | 约 0.5 |
| 6 | RMS_NORM / ADD / MUL 融合 | 0.5～1 |

全部完成估计 7～8 ms/token（pp512 约 120～140 t/s）。长输入下 FA 的 QK^T / PV 仍按 FP32 向量速率执行，可考虑块内 FP16 累加。

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
- [ ] 对齐 kernel 后重测长输入 prefill（4K/8K/16K）和 16K decode。
- [ ] 重复次数补到 2 次以上，KLD 用 4×512 token。
- [ ] `quantize_a_4row_i8_hp_al` 的 vlenb≠128 分支在其他 VLEN 上测试。
- [ ] 向 SpacemiT 报告：Q5_0 IME kernel 结果错误；原 hp kernel 的 A 布局导致非对齐加载；独立 `ggml-spacemit` 后端不支持 F32 权重矩阵乘（Qwen3.5/3.6 无法运行）。

## 4. 复现

```bash
# 板上构建后（cmake 参数见 docs/build-riscv64-spacemit.md）
export GGML_SPACEMIT_BACKEND=0
./build/bin/llama-bench -m Qwen_Qwen3.6-35B-A3B-Q4_0.gguf \
    -ngl 0 -t 8 -fa 1 -mmp 0 -b 2048 -ub 512 -p 512,2048 -n 32

# A/B：关闭某项优化
GGML_SPACEMIT_HP_ALIGNED=0 ./build/bin/llama-bench ...

# 算子级计时
GGML_OP_STATS=1 GGML_OP_STATS_FILE=stats.tsv ./build/bin/llama-bench ...

# kernel 单元测试（不在 CMake 构建中，编译命令见源文件头部注释）
./test-spacemit-hp-aligned 200 8   # reps=200, cpu=8
```

## 局限

- 所有数据都在单块 `k3-dev-006`、1.8 GHz 固定频率下测得，板卡同时有其他用户；Qwen3.6 的多数数据每项只跑 1～2 次。
- 精度参考本身有 Q8_0 激活量化，不是 FP32 真值。
- roofline 和 TODO 收益均为估算。
