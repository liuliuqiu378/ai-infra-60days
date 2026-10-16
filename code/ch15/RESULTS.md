# ch15 数据源与结果登记（并行策略：TP/PP/SP/EP 与通信量）

新风格：**无性能跑测**。外部公开数据（论文/官方文档/源码）+ 零成本公式代入，本文件登记正文中所有关键数字的出处，可复核。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 版本/日期 | 许可 | 用途 |
|---|---|---|---|
| 论文 *Megatron-LM*（arXiv:1909.08053） | 2019-09，v4 PDF | arXiv 非独占分发 | TP 机制出处：列/行并行、按头切、few comm ops |
| 论文 *Efficient Large-Scale Language Model Training… Megatron-LM*（arXiv:2104.04473，SC '21 Best Paper） | 2021-04 | arXiv 非独占分发 | **TP 通信量公式 8bsh(t−1)/t**、PTD-P、52% peak |
| 论文 *GPipe*（arXiv:1811.06965） | 2018-11 | arXiv 非独占分发 | **气泡率 O((K−1)/(M+K−1))、M≥4K negligible** |
| 论文 *Reducing Activation Recomputation…*（arXiv:2205.05198） | 2022-05 | arXiv 非独占分发 | SP：激活显存 5×、开销 4.4%/1.5% |
| 论文 *Switch Transformers*（arXiv:2101.03961） | 2021-01 | arXiv 非独占分发 | EP 出处：all-to-all 形态、top-1 路由 |
| 论文 *DeepSpeed Inference*（arXiv:2207.00032） | 2022-07 | arXiv 非独占分发 | tensor-slicing/expert-slicing、"few GPUs" 警告 |
| 论文 *Speculative MoE*（arXiv:2503.04398） | 2025-03 | arXiv 非独占分发 | EP 是 DeepSpeed-MoE 瓶颈的批评口径 |
| 论文 *Efficiently Scaling Transformer Inference*（arXiv:2211.05102） | 2022-11 | arXiv 非独占分发 | TPU v4 多维切分解析模型、prefill/decode 口径 |
| 论文 *Speculative Sampling*（arXiv:2302.01318） | 2023-01 | arXiv 非独占分发 | all-reduce "latency bound" 论断 |
| MVAPICH MUG '25 *Characterizing Communication Patterns in Distributed LLM Inference*（arXiv:2507.14392 同源） | 2025 | 公开讲义 | PP 通信量：prefill 2Sp·h / decode 2hb |
| vLLM 文档 Parallelism and Scaling / optimization | docs.vllm.ai，2026-10-16 检索 | Apache-2.0 | TP=单机卡数、PP=机器数官方配方 |
| vLLM 博客 *Serving LLMs on AMD MI300X* | 2024-10-23 | Apache-2.0 | TP4→TP8 收益非线性论断 |
| vLLM 源码 `github.com/vllm-project/vllm` | commit `642c18d`（2026-10-15 clone） | Apache-2.0 | 四把刀刀口行号与默认值 |
| 本书 ch03/ch04/ch05/ch08/ch10/ch12/ch13/ch14 | 2026-10-15/16 | 本仓库 | 带宽表、roofline、并发账回扣 |

## 一、外部论文关键数字（2026-10-16 检索）

| 数字 | 原文位置 | 口径提醒 |
|---|---|---|
| TP 每层前向 2 次 all-reduce | Megatron-LM §2.2/§3（1909.08053） | 每 block：attention 出口 1 次 + MLP 出口 1 次；训练含反向为每层每 microbatch 4 次 |
| 每卡每层每 microbatch 通信 **8bsh(t−1)/t** | 2104.04473 原文 | **训练口径（含反向）**；本章前向账取系数 4（2 次 all-reduce × ring 2(t−1)/t） |
| 气泡 **O((K−1)/(M+K−1))**、**M ≥ 4K negligible** | GPipe §2.3 | O() 为渐近上界常数未定；"negligible" 是端到端体感，非公式值归零 |
| SP **激活显存 5×**、开销 **4.4%（GPT-3）/1.5%（1T）** | 2205.05198 摘要 | 训练口径；与选择性重计算联合报告 |
| EP all-to-all 大小 [E, C, d_model] | Switch Transformer §2 | C=每卡 token 数；top-1 路由 |
| TP "can only scale efficiently to a few GPUs" | DeepSpeed Inference 附录 A | 原文警告，非数字 |
| all-reduce "typically latency bound" | 2302.01318 正文 | 小消息（少量 token 激活）场景论断 |
| PP decode 每 token 每边界 **2hb 字节** | MUG'25 分析模型 | b=batch、h=hidden；prefill 为 2Sp·h |
| TP4→TP8 "gains aren't always linear" | vLLM AMD 博客 2024-10-23 | 定性论断，无单一加速比数字 |
| TP=每机卡数、PP=机器数（16 卡 2 机 → 8×2） | vLLM 官方文档 | 官方配方原话 |

## 二、公式代入算账（零成本，正文全部数字的计算过程）

**算账 1：TP 每 decode 步通信字节（正文 2.2 表、图 1 左）**
公式：每步全模型发送 = 4·b·s·h·(t−1)/t × 2 次? → 精确口径：每层 2 次 all-reduce，每次 ring 每卡发 2(t−1)/t·bsh·dtypeB，故每步 = 2 × 2(t−1)/t × bsh·2 × L。
代入 b=32、s=1、h=8192、dtype=2B、L=80：
- t=2：2×2×0.5×32×8192×2×80 = **83,886,080 B ≈ 80 MB（2^20）**
- t=4：2×2×0.75×… = **125,829,120 B ≈ 120 MB**
- t=8：2×2×0.875×… = **146,800,640 B ≈ 140 MB**
（t=16：**157,286,400 B ≈ 150 MB**，正文 2.2 表未列）

**算账 2：同一批字节走不同链路的时间（正文 2.2、6.1、任务 A）**
- 80 MB ÷ NVLink4 单向 450 GB/s = **0.19 ms**（第 8 章规格折半口径）
- 80 MB ÷ PCIe4 实测 18.6 GB/s = **4.5 ms**
- 120 MB ÷ 18.6 GB/s = **6.8 ms**；若一步 decode 取 40 ms（保守估计），占 **约 17%**
（图 1 与正文均按 2^20 字节记 MB，单位一致）

**算账 3：GPipe 气泡率（正文 3.1 表、图 2）**
- K=4, M=4：(4−1)/(4+4−1) = 3/7 = **42.9%**
- K=4, M=16：3/19 = **15.8%**；<10% 需 (3)/(M+3) < 0.1 → M > 27，即 **M ≥ 28**
- K=8, M=8：7/15 = **46.7%**；K=8, M=32：7/39 = **17.9%**；<10% → 7/(M+7)<0.1 → **M ≥ 64**
- 图 2 左格：总格 4×7=28，填 4×4=16，空白 12 → 12/28 = 42.9%（几何一致）

**算账 4：EP all-to-all 每卡发送（正文 5.1）**
每卡把发往其余 P−1 卡的 token 推出：每卡发送 ≈ 总量×(P−1)/P（与 ring 同形），热点专家使实际 ≥ 此值——正文只给形态与"线性涨"定性结论，未给具体 MB 数。

## 三、vLLM 源码行号（commit 642c18d，grep 逐一核验）

### TP（`vllm/model_executor/layers/linear.py` 等）

| 行号 | 内容 |
|---|---|
| `linear.py:436` | `class ColumnParallelLinear(LinearBase)` |
| `linear.py:616` | Column 的 `def forward` |
| `linear.py:627` | `output = tensor_model_parallel_all_gather(output_parallel)` |
| `linear.py:1639` | `class RowParallelLinear(LinearBase)` |
| `linear.py:1694` | `self.input_size_per_partition = divide(input_size, self.tp_size)`（整除校验） |
| `linear.py:1785` | Row 的 `def forward` |
| `linear.py:1804` | `if self.reduce_results and self.tp_size > 1: output = tensor_model_parallel_all_reduce(...)` |
| `communication_op.py:12` / `:17` / `:24` | all-reduce / all-gather / reduce-scatter 三个原语 |
| `device_communicators/cuda_communicator.py:57` | `if unique_name.split(":")[0] != "tp": use_custom_allreduce = False`（custom AR 只给 TP） |
| `parallel_state.py:1713` | `_ENABLE_CUSTOM_ALL_REDUCE = True`（`:1717-1718` setter） |
| `pynccl.py:91` / `:268` | `class PyNcclCommunicator` / `def all_reduce` |
| `custom_all_reduce.py:120` / `:516` | `class CustomAllreduce` / `def all_reduce` |

### PP

| 行号 | 内容 |
|---|---|
| `config/parallel.py:140` | `pipeline_parallel_size: int = Field(default=1, ge=1)` |
| `config/parallel.py:295` | PP×TP 乘积校验注释 |
| `config/parallel.py:574-594` | tp/pcp/dcp/dp 整除族校验（`:581` 报错例 "but got TP=..."） |
| `v1/engine/core.py:216-219` | **"Setup batch queue for pipeline parallelism … required by PP to eliminate pipeline bubbles"** 注释（原话） |
| `v1/engine/core.py:220` | `self.batch_queue_size = vllm_config.max_concurrent_batches` |
| `v1/core/sched/scheduler.py:362` | `self.use_pp = self.parallel_config.pipeline_parallel_size > 1` |
| `v1/executor/multiproc_executor.py:289-290` | `pp_size = self.parallel_config.pipeline_parallel_size` |

### SP

| 行号 | 内容 |
|---|---|
| `config/parallel.py:760-767` | SP 给 EP 配套的注释（原话："replicated tokens results in useless duplicate computation"） |
| `config/parallel.py:768` | `def use_sequence_parallel_moe` 属性 |
| `config/parallel.py:791` | `use_sequence_parallel_moe` 的三条件（EP 开 ∧ TP>1 ∧ DP>1 ∧ 后端匹配，`:770-790` 列表） |

### EP

| 行号 | 内容 |
|---|---|
| `config/parallel.py:190` | `enable_expert_parallel: bool = False`（**默认关**） |
| `config/parallel.py:220` | `all2all_backend: All2AllBackend = "allgather_reducescatter"` |
| `config/parallel.py:45-60` | `All2AllBackend` Literal 14 种后端 |
| `config/parallel.py:65` / `:569` | `class EPLBConfig` / EPLB 要求 EP 必开的校验 |
| `fused_moe/config.py:1015` | `class FusedMoEParallelConfig`（`:1016-1029` 字段：tp/pcp/dp/ep/sp） |
| `fused_moe/config.py:1035-1039` | `def use_all2all_kernels` = `use_ep and (dp>1 or pcp>1 or sp>1)` |
| `fused_moe/config.py:1439` / `:1467` | `ep_size` / `use_ep` 属性 |
| `fused_moe/all2all_utils.py:128-137` | `get_ep_all2all_manager`（取 `ep_group.device_communicator.all2all_manager`） |
| `fused_moe/layer.py:246-251` | `skip_final_all_reduce = (not reduce_results and not use_all2all_kernels and not is_sequence_parallel ...)` |
| `parallel_state.py:1659` | `def get_ep_group()` |

## 四、本章配图

| 图 | 文件 | 数据 |
|---|---|---|
| 图 1 | `images/figures/ch15/fig1_comm_ledger.png` | 左=算账 1 曲线（t=2/4/8，MB=2^20）；右=频率×尺寸定性象限（非数字图） |
| 图 2 | `images/figures/ch15/fig2_pipeline_bubble.png` | 算账 3 的几何版（K=4，M=4/16 甘特图与格数核算） |

均由 `code/ch15/make_figures.py` 生成，零外部图片。图 2 的气泡格数（28 总格/16 实格/12 空格）与正文 3.1 表格（42.9%）互证。
