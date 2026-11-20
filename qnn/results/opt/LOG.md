# ACE-Step 1.5 性能优化日志

- 分支：子仓库 `qnn-perf`（基于 `qnn-analysis`）
- 环境：qiao-ubuntu，RTX 4060 8 GB（实测 bf16 GEMM 约 29 TFLOPS、显存带宽 251 GB/s），PCIe 实测约 10 GB/s，torch 2.10.0+cu128，flash-attn 2.8.3，triton 3.6.0，12 核 CPU / 15 GB 内存
- 场景：text2music，30 s 歌曲，batch 1，0.6B 5Hz LM（CoT + 语义码）+ turbo DiT 8 步（shift 3）+ Oobleck VAE；固定全局种子，输出可复现
- 指标：端到端墙钟（`qnn/perf/bench_e2e.py`，1 次预热后测 N 次）。下表“稳态”取第 3 次及以后测量的中位数，以排除 LM 头两次生成偏慢（见候选 #5）
- 正确性：
  - 不改数值的改动要求输出 md5 不变；
  - 改数值的改动与参照比较 SNR，或用仓库自带的 PMI 奖励做质量比较
- 环境说明：`pytorch_wavelets` 未安装，DCW 在所有配置下都静默退化为 no-op，对比公平

## 结果

| # | 层次 | 改动 | 墙钟（稳态） | 显存峰值（reserved） | 正确性 | 结论 | 提交 |
| - | ---- | ---- | ------------ | -------------------- | ------ | ---- | ---- |
| B0 | — | 基线：pt LM + offload + SDPA | 11.50 s | 6.16 GB | 参照 | — | e90ad93 |
| B1 | — | 官方 8 GB 默认：vllm + 全卸载 + int8 + compile + flash | 17.56 s | 7.16 GB | — | 仅作参照（int8 张量不能走快速卸载，搬运 9.66 s） | e90ad93 |
| O1 | 1 系统 | offload 热路径去掉 `gc.collect`（每次约 0.2 s） | 9.47 s | 6.17 GB | md5 不变 | 保留 | 5eee133 |
| O2 | 1 系统 | pinned 主副本：异步 H2D，卸载不做 D2H，按版本号检测修改 | 6.81 s | 6.17 GB | md5 不变 | 保留 | 71bc296 |
| fix | — | pinned 快路径只对 CUDA 目标启用（O2 引入的单测回归） | — | — | 单测 | 保留 | 2eb36c1 |
| O3 | 2 引擎 | LM 改用 nano-vllm，DiT bf16 走 pinned 卸载（配置） | 6.21 s | 7.16 GB | 码改变，质量见下 | **冻结** | （配置） |
| O4 | 2 算法 | 码转 hint 只加载 FSQ + detokenizer（0.2 GB），同码只解一次；DiT 上传 3 次→1 次 | 4.87 s | 6.90 GB | md5 不变 | 保留 | 918cc60 |
| O5 | 1 系统 | DiT 流程中剩余 3 处 GC 去掉（非 offload 模式保留预检前 GC） | 4.17 s | 6.90 GB | md5 不变 | 保留 | 396caea |
| O6 | 4 布局 | VAE 窗口按实测空闲显存决定（替代固定 128 帧）；VAE 解码期间开 cudnn.deterministic | 3.71 s | 6.90 GB | 与 128 窗口 SNR 39.6 dB，逐次一致 | 保留，**冻结** | 79d5713 |
| O7 | 6 kernel | VAE Snake 激活融合为单个 Triton kernel | 3.52 s | 6.89 GB | 对 fp32 参照 SNR 36.9→39.6 dB | 保留 | cd99a62 |
| O8 | 5 并行 | LM 运行时在副 stream 预取 DiT 的 3.33 GB（decoder/tokenizer/detokenizer） | **3.38 s** | 6.93 GB | md5 不变 | 保留 | fba659b |

总计：11.50 → **3.38 s（3.4×）**，显存峰值 6.16 → 6.93 GB（< 7.6 GB 可用），RSS 5.4 → 6.0 GB（pinned 主副本常驻）。

### 冻结的结构性决策

| 层次 | 决策 | 依据 |
| ---- | ---- | ---- |
| 引擎 | LM 用 nano-vllm（CUDA Graph） | 同等条件下 LM 4.41→1.87 s；节点级 nsys 显示 LM 段 GPU 忙碌约 85%，已转为访存受限 |
| 精度 | 全程 bf16，不量化 | int8 DiT 无法走 pinned 快速卸载（B1 搬运 9.66 s），且改变数值 |
| 卸载 | DiT 也卸载，用 pinned 主副本 + 预取 | 8 GB 卡上 DiT 4.46 GB、LM 2.1 GB（含 KV）、文本编码器 1.11 GB、VAE 0.31 GB 无法同时常驻 |
| 布局 | VAE 显存允许时整段解码 | 128 帧窗口重叠浪费 50%；整段解码是无分块误差的真值 |

## 稳态耗时分布（O8，3.38 s）

| 部分 | 耗时 | 占比 | 状态 |
| ---- | ---- | ---- | ---- |
| LM Phase 1（CoT，约 105 token） | 0.81 s | 24% | 访存受限，约 70% 带宽 roofline |
| LM Phase 2（150 码，CFG 双路） | 1.06 s | 31% | 同上 |
| DiT 8 步 | 0.48 s | 14% | GEMM 达实测峰值约 80%（23/29 TFLOPS），逐元素约 21% |
| VAE 解码（融合后） | 0.34 s | 10% | 卷积约 190 ms，Snake 已融合 |
| 搬运（条件编码器、文本编码器、VAE 上传） | 0.11 s | 3% | — |
| 预处理、文本编码、条件编码、其他 | 约 0.58 s | 17% | — |

## 剖析记录

| 时间 | 工具 | 结论 | 文件 |
| ---- | ---- | ---- | ---- |
| B0 | cProfile | `.to()` 搬运 3.3 s、`gc.collect` 13 次 2.7 s；文本编码 2.0 s（其中计算只有 12 ms） | `B0_pt_offload_sdpa/cprofile_*.txt` |
| O3 | nsys（graph 级） | 一次生成 H2D 16.6 GB，DiT 被上传 3 次 | `O3_vllm_offloaddit/nsys_run.*` |
| O4 | `--phases` 同步计时 | 剩余 0.49 s 来自 `generate_music.py` 的 2 次 GC | `O4_phases/phases.json` |
| O6 | nsys（`--cuda-graph-trace=node`） | LM 段 kernel 忙碌约 85%（GPU 受限）；DiT 上传期间 GPU 计算空闲约 0.7 s | `O6_vae_chunk_mem/nsys_node.*` |
| O6 | torch.profiler | DiT 单步 65 ms（GEMM 48.7）；VAE 466 ms 中逐元素 288 ms | `kernel_profile/*` |
| O6 | microbenchmark | VAE 窗口 128/256/384/512/整段：0.88/0.86/0.62/0.55/0.47 s；整段在默认 cuDNN 下不确定（最大差 0.074），deterministic 后一致且不降速 | `vae_chunk_bench/`、`vae_determinism/` |

## LM 后端质量（pt 对 vllm）

每个后端 48 个样本（4 个提示词 × 12 个 seed），用同一个 pt 打分模型计算 PMI 奖励：

| 后端 | PMI 奖励 | LM 耗时 |
| --- | --- | --- |
| pt | 0.679 ± 0.227 | 4.14 s |
| vllm | 0.645 ± 0.246 | 2.08 s |

- 差 −0.034，标准误 0.048，95% 置信区间 [−0.13, +0.06]，统计上检测不到差异；
- 码统计与元数据完整率一致；
- 近贪心解码在第 0～10 个 token 分叉：码分布熵高、前几名 logit 接近，bf16 kernel 差异会翻转 argmax，所以序列一致性不是灵敏的检验；
- 文件：`lm_backend_quality/{summary.json,greedy_compare.json}`，`lm_backend_quality*.log`。

## 下一轮候选池

| # | 候选 | 预期收益上限 | 成本 | 阻塞原因 |
| - | ---- | ------------ | ---- | -------- |
| 1 | Phase 2 的 lm_head 只算 64000 个码 token 对应的行（Phase 1 排除码 token） | 每步读取 1.23→0.92 GB，约 −0.3 s | 中高：改第三方 nano-vllm 的 logits 计算和采样 | 只是成本高 |
| 2 | LM 权重 INT8/FP8 weight-only | LM 约 −40%（约 −0.7 s） | 高：nano-vllm 需要量化 GEMV kernel | 需要突破“bf16 不量化”的约束，并做质量评估 |
| 3 | DiT 用 torch.compile 融合逐元素算子（每步约 13 ms） | 约 −0.08 s | 低（但冷启动要编译，按形状分桶） | 需先测冷启动成本 |
| 4 | 条件编码器 / 文本编码器也预取或常驻 | 约 −0.1 s | 中 | 显存余量约 0.6 GB，需要重新规划 |
| 5 | 前 1–2 次生成 LM Phase 1 偏慢（1.5～3.5 s 对稳态 0.8 s） | 首次生成 −0.7～2.7 s | 低～中：初始化时预热 | 需先定位原因（推测是 pinned 内存增长引起的页缓存回收） |
| 6 | Phase 2 复用 Phase 1 的 KV（目前重新 prefill） | < 0.05 s | 中 | 收益低 |
| 7 | PyTorch 路径下 `prepare_condition` 重复计算；无 LM 时 tokenizer 对静音空转 | 约 0.06 s | 低 | 收益低 |
| 8 | 调整官方 tier3 默认值（现为 int8 + compile + 全卸载，17.6 s）为本分支配置 | 默认用户 17.6→3.4 s | 低（改配置） | 涉及默认行为，需要用户决定 |

已放弃的方案：
- 官方 int8 量化配置：量化后的张量走旧的卸载路径，太慢；
- VAE 256 帧窗口：几乎没有收益（0.86 s 对 0.88 s）。
