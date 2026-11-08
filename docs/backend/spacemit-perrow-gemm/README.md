# K3 per-row INT8 GEMM：不做融合，GR00T LLM prefill 追平 fused kernel

日期 2026-11-08，分支 `linpengt/perrow-gemm`（基于 spacemit `855a53bed`）。

## 结论

1. **速度**：GR00T N1.7 的 LLM backbone（Cosmos-Reason2-2B，16 层，prefill 156 token）在
   llama.cpp 原生计算图上**只替换 MUL_MAT kernel**，不做任何算子融合，每层耗时从 36.7 ms
   降到 9.0–9.5 ms，16 层约 144–152 ms，和 `groot-vla` 分支里手写的 fused prefill kernel
   （同板同时段实测最快 147–150 ms，中位 154–159 ms）持平。此前通用算子路径（约 18 ms/层）
   比 fused 慢一倍，差距基本都在 GEMM 的 kernel 和调度上，不在算子之间的同步。
2. **精度**：激活每行只用一个 INT8 scale 时，LLM 激活里的离群通道会让精度明显下降
   （Qwen3-1.7B 上 KL 散度 0.047，标准 Q8_0 是 0.0039）。加载时对权重、运行时对激活做同一个
   Hadamard 旋转之后，从 FP16 权重量化的 KL 散度是 0.0045，和标准 Q8_0 基本一样。旋转让
   prefill 慢 2.8–3.7%。

## 速度

测试环境：K3 板子，A100 8 核空闲，`llama-bench -p 156 -n 0 -fa 1 -r 10`，两轮交替跑。权重
是按行量化的 INT8（每个输出行一个 scale），存成普通 Q8_0。为了扣掉 embedding、lm_head 等
固定开销，分别跑 2/4/8/12/16 层，取 prefill 时间对层数拟合的斜率作为每层耗时。

第一组（`results/depth-sweep.log`，不旋转，和主线、通用算子分支对比）：

| 路径 | 每层 | 16 层 |
|---|---:|---:|
| 主线 spacemit `855a53bed`（Q8_0 32x32 路径） | 36.66 ms | 587 ms |
| 通用算子分支（llm-backbone-generic-ops，去掉其 ROPE 提交） | 18.08 ms | 289 ms |
| **本分支，不旋转** | **8.99 ms** | **144 ms** |
| `groot-vla` fused prefill kernel，同时段 | — | 最快 148–150，中位 154–157 ms |

第二组（`results/depth-sweep-rotation.log`，另一个时段，旋转开关对比）：

| 路径 | 每层 | 16 层 |
|---|---:|---:|
| 本分支，不旋转（`GGML_SPACEMIT_PERROW_ROT=0`） | 9.25 ms | 148 ms |
| **本分支，全量旋转（默认）** | **9.51 ms** | **152 ms** |
| `groot-vla` fused prefill kernel，同时段 | — | 最快 147，中位 155–159 ms |

两组之间不旋转的结果差 3%，是不同时段板子状态的波动。比较时请看同一组内的数字。

第一组的 pp156 吞吐（t/s，两轮）：

| 层数 | 本分支 | 通用算子分支 | 主线 |
|---:|---:|---:|---:|
| 2 | 5388 / 4678 | 3510 / 3625 | 2293 / 2442 |
| 4 | 3030 / 3112 | 1984 / 1950 | 1149 / 1118 |
| 8 | 1853 / 1798 | 1017 / 1026 | 548 / 544 |
| 12 | 1284 / 1280 | 698 / 694 | 361 / 363 |
| 16 | 997 / 981 | 加载失败 | 271 / 268 |

16 层模型的其他形状（t/s）：

| | 不旋转 | 分块旋转 | 全量旋转 | 主线 32x32 路径 |
|---|---:|---:|---:|---:|
| pp156 | 986 | 979 | 949 | 270 |
| pp512 | 501–541 | 513 | 517 | 262 |
| tg32 | 16.3–16.5 | 16.5 | 16.4 | 17.9 |

旋转让模型加载多花约 5 秒（28 层，X100 上 8 线程）。

## 精度

### KL 散度

参考 logits 由 x86 CPU 版 llama.cpp 跑 FP16 模型得到（PPL 14.527）。测试用完整的
Qwen3-1.7B（28 层），wikitext-2，`-c 512 --chunks 32`（`results/kld-vs-f16.log`）。

| 配置 | 平均 KLD | 中位 KLD | 99% KLD | top-1 一致率 | PPL |
|---|---:|---:|---:|---:|---:|
| 标准 Q8_0（原 32x32 路径，对照） | **0.0039** | 0.0014 | 0.037 | **97.4%** | 14.63 |
| 按行 INT8 权重，不旋转 | 0.0467 | 0.0162 | 0.468 | 91.1% | 14.65 |
| 按行 INT8 权重，分块旋转 | 0.0119 | 0.0042 | 0.109 | 95.6% | 14.62 |
| 按行 INT8 权重，全量旋转 | 0.0107 | 0.0035 | 0.123 | 95.6% | 14.65 |
| 原始 Q8_0 → 全量旋转 → 按行量化 | 0.0071 | 0.0025 | 0.073 | 96.5% | 14.70 |
| **FP16 → 全量旋转 → 按行量化** | **0.0045** | 0.0014 | 0.047 | **97.2%** | 14.50 |
| FP16 → 分块旋转 → 按行量化 | 0.0053 | 0.0019 | 0.056 | 97.0% | 14.47 |
| FP16 → 不旋转 → 按行量化 | 0.0436 | 0.0151 | 0.441 | 91.6% | 14.46 |

- **问题出在激活量化**：即使权重来自 FP16，不旋转时 KLD 也有 0.044。旋转之后降到约十分之一。
- **从 FP16 量化最好**：从已经量化过的 INT8 或 Q8_0 来，权重要多量化一次，误差更大。
- **全量旋转比分块旋转稍好**，代价是多约 3% 的 prefill 时间。
- **PPL 看不出这些差别**：不旋转的配置 PPL 反而最低，判断精度要看 KL 散度和 top-1。
  本文早先版本只看了 8 段 PPL，写了"差别在误差范围内"，这个结论不对。
- **注意**：FP16 模型的 lm_head（和 embedding 共用）仍是 FP16，比其他配置的 Q8_0 lm_head
  略准，所以 FP16 那几行稍占便宜。

### 每个 matmul 的误差

在 kernel 里加了一个标量检查（`GGML_SPACEMIT_PERROW_CHECK`），用真实的 512 token prefill，
对每次 matmul 重算 3 行。下表是输出相对误差的平均值，以"存储的权重 × float 激活"为基准，
也就是只算激活量化引入的误差（`results/check-*.log`）：

| 矩阵 | 激活按行 INT8 | 按每 32 个元素量化（原 Q8_0 路径） | 全量旋转后按行 INT8 |
|---|---:|---:|---:|
| q / k | 3.3% | 0.76% | 0.59% |
| v | 6.0% | 1.4% | 1.1% |
| o_proj | 2.0% | 0.71% | 0.99% |
| gate | 2.5% | 0.54% | 0.47% |
| up | 4.7%（最大 12.8%） | 0.98% | 0.83% |
| down | 5.5% | 0.98% | 0.90% |
| **平均** | **3.9%** | **0.87%** | **0.78%** |

- **原因是离群值**：一行激活的最大值是 RMS 的 31 倍（平均），最大 78 倍。旋转之后降到
  3.9 倍，最大 5.8 倍。
- **kernel 本身没有误差问题**：kernel 输出和精确整数结果的相对误差是 4e-8。向量版旋转的
  结果和标量参考版一致。

## 技术说明

### 为什么激活每行一个 scale 会出问题

kernel 的快，来自 int32 累加器一直累加到 k 结束、最后只乘一次 scale。这要求 scale 只能跟行
（token）和列（输出通道）有关，不能跟 k 有关。原来的 Q8_0 路径每 32 个 k 就换一个 scale，
所以要在 k 循环里一边累加一边换算，这正是它慢的原因。

但 LLM 的激活有离群通道：少数几个 k 通道比其余的大几十倍，而且每个 token 都是同样那几个。
一行只有一个 scale 时，scale 被这几个大值撑大，其余大部分元素只能用到 INT8 很少的几个格子，
精度就丢了。

### 两种解决思路

两种方法都基于同一个恒等式：在 X 和 W 之间插一个可逆矩阵，结果不变。

```
Y = X·Wᵀ = (X·T) · (W·T⁻ᵀ)ᵀ
```

- **SmoothQuant**：T 是对角矩阵。离群通道的激活除以一个大系数，对应的权重乘回去，把难度
  从激活挪到权重上。系数要用校准数据统计，然后折进前面的 norm。GR00T 的 fused kernel 用的
  就是这个方法，smooth 因子在它导出的 `layerNN.bin` 里。
- **旋转（QuaRot / SpinQuant 的思路）**：T 是正交矩阵，这里取 Hadamard 矩阵（元素全是 ±1/√n，
  T⁻ᵀ = T）。每个输出元素都是所有输入的一种正负混合，一个特别大的通道会被均匀摊到整行。
  比如一行里有一个 100、其余在 1 左右，旋转后每个元素约为 ±100/√2048 ≈ ±2.2，再加上其余
  值的混合，整行变得很平，一个 scale 就够用。好处是**不需要校准数据**，也不用改模型文件。

QuaRot 的完整做法是离线把残差流的旋转折进 embedding 和各层权重，只在 down_proj 前和
attention 里做少量在线旋转。SpinQuant 则用数据学出旋转矩阵。本分支用的是更简单的"全部在线"
版本：每个 matmul 的输入在运行时各旋转一次，所以加载任意 GGUF 时就能完成，不需要改模型结构。

### 本分支的实现

- **权重，加载时**：每个权重行（长度 K）先反量化成 float（来源可以是 Q8_0 或 F16），做一次
  分块 Walsh-Hadamard 变换，再按行量化成 INT8，写回内存里的布局。GGUF 文件不变。块大小 B
  取能整除 K 的最大 2 的幂，最大 2048：K=2048 时是一整块，K=6144 时分成 3 块。转换在 X100
  上用 8 个线程并行。
- **激活，运行时**：每行在量化之前先做同样的变换。归一化系数 1/√B 直接乘进这一行的 scale，
  不需要额外的计算。实现针对 A100（VLEN 1024，e32m8 一组 256 个 float）：
  - 每 256 个元素读一次内存，距离 < 256 的 8 个阶段都在寄存器里完成：
    - 距离 1–16：用 `vslidedown`/`vslideup` 加 mask merge
    - 距离 32/64/128：在同一组的 8 个向量寄存器之间直接做
  - 距离 256–1024 的 3 个阶段在内存里，每次处理 256 个元素。
- **开销**：第一版全量旋转全部在内存里做，prefill 慢 8%。改成大部分阶段在寄存器里完成后，
  量化阶段每次 prefill 从 9.9 ms 涨到 14.7 ms（每个 worker），整体慢 2.8–3.7%。分块旋转
  （模式 1）只做距离 ≥ 32 的阶段，几乎不花时间，但精度稍差。

### 环境变量

| 变量 | 取值 |
|---|---|
| `GGML_SPACEMIT_PERROW` | 0 关闭；1（默认）只用于每行所有 32-block scale 相同的 Q8_0 权重；2 所有 `blk.*` 下的 Q8_0 和 F16 二维权重都在加载时按行量化 |
| `GGML_SPACEMIT_PERROW_ROT` | 0 不旋转；1 分块旋转（只在 32 宽的行之间做）；2（默认）全量 Walsh-Hadamard |
| `GGML_SPACEMIT_PERROW_CHECK=<n>` | 对前 n 次 matmul 做标量检查，打印误差 |
| `GGML_SPACEMIT_PERROW_PROF=1` | 退出时打印每个 worker 各阶段的耗时 |

F16 权重在 `PERROW=2` 下才会走这个 kernel。因为 backend buffer 是按 F16 大小分配的，内存
占用是 INT8 的两倍。

### GEMM kernel（`ggml/src/ggml-spacemit/ime.cpp`，`namespace perrow_q8`）

- **识别**：repack 时决定。Q8_0 权重看每行的 scale 是否相同（模式 1），或者看名字是否在
  `blk.*` 下（模式 2）；F16 只看名字和形状。命中后 `tensor->extra` 换成
  `tensor_traits_q8_0_perrow`，其余权重走原来的路径。不新增 ggml 类型。
- **布局**：权重按 32 列一条 strip 排，`[K/16][32 列][16 B]`，后面跟每行一个 f32 scale。
  激活按 16 行一组，`[K/16][16 行][16 B]`。
- **tile**：和 fused kernel 的 `ime_i8_tile` 一样，16x32 int32 累加器一直累加到 k 结束。
  每个 16-k block 做 8 条 vmadot，A/B 寄存器组软件流水。单核热态约 1 TOPS。
- **调度**：以（列 strip × 16 行 tile）为单位，按 strip 优先均分给 8 个 worker。每个
  worker 把自己的权重 strip 从 DRAM 拷进 TCM，只拷一次，然后在 TCM 里跑完所有 M tile。
  k 太长放不下时分段，段间用 int32 累加。
- **激活量化**：每行旋转并动态量化成 INT8 一次，放在共享 workspace 之外的 buffer 里。连续
  几个 matmul 读的是同一个输入张量时（Q/K/V、gate/up）直接复用。

16 层 prefill 中每个 worker 的时间（`GGML_SPACEMIT_PERROW_PROF=1`，每次 prefill，不旋转）：

| 阶段 | 时间 | 说明 |
|---|---:|---|
| 权重拷贝 | 约 24 ms | 接近约 30 GB/s 的 DRAM 上限 |
| vmadot tile | 约 25 ms | |
| epilogue | 约 20 ms | 其中输出写回约 10 ms |
| 激活量化 | 约 10 ms | 全量旋转后约 15 ms |

### 提交

基于 `855a53bed`：

1. `fix(llama): keep norm and KV cache on the accelerator for ACCEL backends`：从通用算子
   分支 cherry-pick 的图切分修复。llama-bench 下加不加都是 3 个 split，这里没有影响，
   主要作用于 llama-groot-vla。
2. `ggml: export the op profile hooks`：让 `GGML_BUILD_PROFILE` 能链接，只用于调试。
3. `spacemit: per-row INT8 prefill GEMM for Q8_0 weights with one scale per row`：GEMM kernel。
4. `spacemit: Hadamard rotation and F16/Q8_0 sources for the per-row INT8 GEMM`：旋转、
   F16/Q8_0 来源、标量检查。
5. 文档、脚本和日志（本目录）。

## 试过但没用的

| 尝试 | 结果 |
|---|---|
| 在 TCM 里给 tile 输出留 2 KB，导致 k 被切成 1984+64 两段 | 挪到栈上之后从 700 涨到 942 t/s |
| 激活用 strided `vlse64/vsse64` 散开存放 | 改成 16 B 连续 store 后快一倍 |
| 软件预取下一条 strip 和输出行 | 900 降到 672 t/s，已删 |
| tile 直接从 DRAM 读权重，不拷进 TCM | M=156 慢 20%，decode 慢 7%，已删 |
| 7 个 worker | 880–898，8 个是 940 t/s；按列切本来就均匀，保留 8 个 |
| M ≤ 16 时每个 worker 自己量化激活 | 没有可测的变化，已删 |
| 旋转全部在内存里逐阶段做 | prefill 慢 8%；改成寄存器内完成大部分阶段后慢 3.7% |
| 旋转前加固定的随机符号翻转（随机化 Hadamard） | 误差没有变化（0.77% 对 0.78%），没采用 |

## 通用算子分支的两个问题

- ROPE 那个提交（`f022aa96b`，contiguous ROPE factors）跑普通 Qwen3 会段错误：在
  `legacy_small_f32` 路径上（NEOX、head ≤ 128），`cos_cache`/`sin_cache` 没有分配就被写入。
  它只在 Qwen3-VL（M-RoPE）上测过。本文测试时去掉了这个提交。
- 16 层时会申请一个 38 GB 的 SPACEMIT buffer 然后加载失败（2–12 层能加载），可能和
  Q8_0 repack size 的改动有关，没有细查。所以第一组里它的斜率用的是 2–12 层。

## 对 GR00T 的建议和限制

- **还没在 GR00T 上端到端验证**：测试模型是截成 16 层的 Qwen3-1.7B，每层形状和 GR00T 的
  LLM 一样，但没覆盖 M-RoPE 和 3 个 deepstack add，也没测 GR00T 的动作输出。
- **推荐用 FP16 权重**：GR00T 的打包脚本本来就从 checkpoint 的浮点权重出发，可以直接输出
  FP16 权重（SmoothQuant 折不折都可以），加载时用 `GGML_SPACEMIT_PERROW=2` 做旋转和量化，
  只量化一次。
- **想和 fused 的数值完全一致**：用现有的 per-row INT8 GGUF 加 `GGML_SPACEMIT_PERROW_ROT=0`，
  保留导出时的 INT8 权重。这需要 smooth 因子已经折进权重，否则精度会像上表"不旋转"那几行
  一样明显下降。
- **fused 用的是合成权重**：格式和 `layerNN.bin` 完全一致（`fused/make_layers.py`），全零和
  随机权重测出来都是 149 ms 左右。
- **decode 比 32x32 路径慢 8%**：主线是两个 worker 一组，一个拷贝一个计算交替进行；这里是
  同一个核先拷后算。GR00T 只有 prefill，所以没继续优化。
- **激活 buffer 是进程全局的**：同一时间只能有一个 spacemit backend。
- **可以继续优化旋转开销**：像 QuaRot 那样在加载时把残差流的旋转折进 embedding 和权重，
  只留 down_proj 前的在线旋转，大部分开销可以省掉。

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
for L in 2 4 8 12 16 28; do
  python3 make_perrow.py gguf-py Qwen3-1.7B-Q8_0.gguf models/pr_L$L.gguf --layers $L
done
```

FP16 模型：从 Qwen/Qwen3-1.7B 的 safetensors 转换：

```bash
python3 convert_hf_to_gguf.py Qwen3-1.7B --outtype f16 --outfile Qwen3-1.7B-F16.gguf
```

层数扫描：

```bash
OURS=build/bin BASE=<主线 build/bin> MODELS=models ./sweep.sh
```

旋转开关对比：在命令前加 `GGML_SPACEMIT_PERROW_ROT=0` 再跑一次。

KL 散度：先在 x86 上用 FP16 模型生成参考 logits，再在板上跑各个配置（`kld.sh` 里有完整命令）：

```bash
llama-perplexity -m Qwen3-1.7B-F16.gguf -f wiki.test.raw -c 512 --chunks 32 --kl-divergence-base f16ref.kld
BIN=build/bin REF=f16ref.kld TXT=wiki.test.raw Q8=Qwen3-1.7B-Q8_0.gguf PR=models/pr_L28.gguf \
  F16=Qwen3-1.7B-F16.gguf ./kld.sh
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
