# ch17 数据源与结果登记（核心项目：mini-vLLM 从零实现）

新风格：**无性能跑测**。本章是项目说明书，不含本机引擎跑测；外部公开数据（官方 README/benchmark/源码）+ 零成本公式代入在此登记，可复核。你自己的 benchmark 结果按 6.3 表模板回填到本文"回填区"。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 版本/日期 | 许可 | 用途 |
|---|---|---|---|
| nano-vllm（github.com/GeeeekExplorer/nano-vllm） | 2026-10-16 检索 README | Apache-2.0 | 1,200 行规模、特性清单、官方 benchmark 表（1361.84 vs 1434.13 tok/s） |
| Dong Wang, *Understanding LLM Inference Through Nano-vLLM* | 2026-04-11 | 博客 | 架构解读、约 1,200 行口径、Qwen3-4B profile（仅作背景，正文未引数字） |
| vLLM 源码 `github.com/vllm-project/vllm` | commit `642c18d`（2026-10-15 clone） | Apache-2.0 | 模块对照表全部行号 |
| 本书 ch03/ch04/ch05/ch06/ch13/ch14/ch16 | 2026-10-15/16 | 本仓库 | 对齐方法（maxdiff 0.00）、batch16 ×15.5、4.2×、block=16、Orca/Sarathi 数字回扣 |

## 一、外部公开数据（全部原文口径，2026-10-16 检索）

### nano-vllm 官方 README benchmark

| 项 | 原文口径 | 口径提醒 |
|---|---|---|
| 代码规模 **约 1,200 行 Python** | README "Readable codebase" | 含前缀缓存 / TP / torch.compile / CUDA graph |
| 硬件 **RTX 4070 Laptop (8GB)**、模型 **Qwen3-0.6B** | README Test Configuration | 作者自测，非第三方 |
| 负载 **256 请求，ISL/OSL 随机 100–1024 token** | README 同上 | 受控：三引擎同负载 |
| vLLM **133,966 tok / 98.37 s / 1361.84 tok/s** | README Performance Results | 输出 token 总数相同是受控对比的证据 |
| nano-vllm **133,966 tok / 93.41 s / 1434.13 tok/s** | 同上 | 同上 |

### 本书内已实测锚点（引用于 6.2/6.3，均为前 8 章登记值）

| 数字 | 出处 | 用途 |
|---|---|---|
| 与 HF 逐 token 对齐 **maxdiff 0.00e+00**（96 token + 93 步 logits） | ch06 RESULTS（2026-10-15 实测） | L1/L2 对齐方法可行性 |
| **batch 1→16 聚合吞吐 ×15.5** | ch06 RESULTS（同上） | B3 同构锚点 |
| 手写 KV 比 HF 快 **10%**；S0=2048 无 cache/KV **4.2×** | ch06 RESULTS（同上） | B1/B2 量级参照 |
| block_size=16 消融口径 | ch13（论文 arXiv:2309.06180 原文） | block_size 选 16 的依据 |

## 二、零成本算账（正文数字的计算过程）

**算账 1：nano-vllm 官方 benchmark 的倍数（正文 1.2）**
1434.13 / 1361.84 = **1.0531 ≈ 1.05×**；耗时比 93.41 / 98.37 = 0.9496 ≈ 省 5.0%。输出 token 总数 133,966 = 133,966（受控成立）。

**算账 2：模块行数预算（正文 2.1）**
250（engine）+ 150（scheduler）+ 180（kv_cache）+ 150（model）+ 80（sampler）+ 100（bench）= **910 行**，位于 500–1000 行区间内（预算值，实际以你 `cloc` 输出为准）。

**算账 3：7B 每块 KV 尺寸与 2 GiB 容量（正文 3.1）**
Qwen2.5-7B：GQA 8 KV head、head_dim=128、bf16（2 B）。
单层单块：K+V = 2 × 16 token × 8 head × 128 dim × 2 B = **65,536 B = 64 KiB**。
2 GiB / 64 KiB = **32,768 块**；可容纳 token = 32,768 × 16 = **524,288 token**；按每请求 16,384 token 计 = **32 个请求**。
（口径提醒：ch04 的 56 KiB/token 是 70 层模型全层每 token 账，本算账是**每层每块**账，二者不可混用。）

## 三、vLLM 源码行号（commit 642c18d，grep 逐一核验，2026-10-16）

| 行号 | 内容 |
|---|---|
| `vllm/v1/core/sched/scheduler.py:84` | `class Scheduler(SchedulerInterface)` |
| `vllm/v1/core/sched/scheduler.py:621` | `def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput` |
| `vllm/v1/core/sched/scheduler.py:2700` | `def add_request(self, request: Request) -> None` |
| `vllm/v1/core/sched/scheduler.py:1649` | `_preempt_request`（第 14 章已核验，本章引用为"不做的机制"） |
| `vllm/v1/core/kv_cache_manager.py:371` | `def allocate_slots(...)` |
| `vllm/v1/core/kv_cache_utils.py:707` | `def hash_block_tokens(...)`（第 16 章已核验） |
| `vllm/v1/core/kv_cache_coordinator.py:460/513/574` | `def find_longest_cache_hit`（三处重载） |
| `vllm/v1/core/block_pool.py:811` | `evict_blocks`（第 16 章已核验） |
| `vllm/v1/worker/gpu_model_runner.py:3935` | `def execute_model(...)` |
| `vllm/v1/worker/gpu_model_runner.py:4299` | `def sample_tokens(...)` |
| `vllm/v1/sample/sampler.py:308` | `def sample(...)` |
| `vllm/engine/arg_utils.py:2952-2980` | `max_num_batched_tokens` 按硬件档默认 2048/8192/16384（第 14 章已核验） |

## 四、你的实测回填区（D40 后填写，禁止预填）

本机配置：GPU ______ / 模型 ______ / 精度 ______ / 代码 commit ______ / 日期 ______

| 对照 | 输出 token 总数 | 耗时 (s) | 吞吐 (tok/s) | 倍数 |
|---|---|---|---|---|
| HF generate（无 cache） | | | | — |
| HF generate（StaticCache） | | | | |
| mini-vLLM（同负载） | | | | = /StaticCache 行 |
| mini-vLLM batch 1 → 16 | | | | |

**回填前自检**：前三行输出 token 总数必须完全相同（不同 = 对齐未过，表格作废）；每个倍数附口径。
