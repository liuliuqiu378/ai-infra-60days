# ch16 数据源与结果登记（前缀缓存、结构化输出与压测实战）

新风格：**无性能跑测**。外部公开数据（论文/官方文档/源码）+ 零成本公式代入，本文件登记正文中所有关键数字的出处，可复核。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 版本/日期 | 许可 | 用途 |
|---|---|---|---|
| 论文 *SGLang: Efficient Execution of Structured Language Model Programs*（arXiv:2312.07104） | v2, 2024-06-06 | arXiv 非独占分发 | RadixAttention 机制、6.4×/3.7×、命中率、生产 74.1%、开销 <0.3%、压缩 FSM 1.6× |
| 论文 *XGrammar: Flexible and Efficient Structured Generation Engine for LLMs*（arXiv:2411.15100，MLSys '25） | v3, 2025-05-12 | CC BY-SA 4.0 | mask 延迟 100×、端到端 80×、44.2→6.8ms、消融表、160MB→0.46MB |
| 论文 *Efficient Memory Management for LLM Serving with PagedAttention*（arXiv:2309.06180） | 2023-09 | arXiv 非独占分发 | 前缀共享 "OS handles shared library" 类比（第 13 章已核验） |
| vLLM 源码 `github.com/vllm-project/vllm` | commit `642c18d`（2026-10-15 clone） | Apache-2.0 | APC 与结构化输出全部行号、压测参数行号 |
| vLLM 文档 / OpenAI API 规范 | 2026-10-16 检索 | Apache-2.0 / MIT | response_format 三型、OpenAI 兼容参数语义 |
| 本书 ch03/ch05/ch06/ch10/ch12/ch13/ch14/ch15 | 2026-10-15/16 | 本仓库 | prefill/decode 属性、崩塌曲线、Goodput、三道墙等回扣 |

## 一、外部论文关键数字（全部原文口径，2026-10-16 检索）

### SGLang（arXiv:2312.07104 v2）

| 数字 | 原文位置 | 口径提醒 |
|---|---|---|
| 吞吐最高 **6.4×**、延迟最高降 **3.7×** | 摘要 / §6.2 | **最高值**，对 Guidance/vLLM/LMQL 弱基线，整套系统功劳（非单开 RadixAttention） |
| 各 benchmark 缓存命中率 **50%–99%** | §6.2 | 覆盖 few-shot/ReAct/RAG/多轮等多类负载 |
| cache-aware 调度达到最优命中率 **96%**（均值） | §6.2（Fig. 13） | 对"最优命中率"的比值，非绝对命中率 |
| Chatbot Arena 生产一月：Vicuna-33B **74.1%**、LLaVA-NeXT-34B **52.4%** | §6.2 Production deployment | **真实生产锚点数字** |
| Vicuna-33B 首 token 延迟平均降 **1.7×** | §6.2 | 生产口径 TTFT 收益 |
| 无复用开销 **0.2s / 74.3s（<0.3%）** | §6.3 | ShareGPT 100 请求，RadixAttention 管理结构耗时 |
| 压缩 FSM 使 JSON 吞吐 **+1.6×**；不复用预处理则吞吐 **低 2.4×** | §6.3 | 两个方向同一机制的两端 |
| 多模态最高 **6×**（0.18→1.15 image/s；0.02→0.10 frame/s） | Table 2 | 基线为模型作者原始 HF 实现 |
| Guidance "lacks batching and parallelism support" | §6.2 | 解释 6.4× 为何含弱基线成分 |

### XGrammar（arXiv:2411.15100，MLSys '25）

| 数字 | 原文位置 | 口径提醒 |
|---|---|---|
| CFG 每 token 延迟**最高降 100×**；端到端输出 token 率**最高 80×** | 摘要 / §4.2 | H100 + Llama-3.1；对当时最佳基线 |
| JSON Schema mask **<40 µs/token**；XML/Python DSL **<200 µs** | §4.1 | Llama-3.1-8B、7950X + RTX 4090 |
| 对最好基线 JSON Schema **3×**、CFG **100×+** | §4.1 | mask 单项延迟对比 |
| SGLang+Outlines **44.2 → 6.8 ms/token**（Llama-3.1 8B）；DeepSeek-V2-Lite 15.8 → 4.8 | Table 1 | **同引擎同任务换后端**，最有说服力的一组 |
| 消融 65.776 → 38.280 → 0.154 → 0.035 → 0.018 ms/token | Table 3 | node merging / adaptive mask cache（248.6×↓）/ rule inlining / context expansion |
| 词表掩码存储 **160 MB → 0.46 MB（0.2%）** | §3 | Llama-3.1 + JSON 语法，自适应 token mask 缓存 |
| MLC-LLM 开/关 XGrammar TPOT **6.2 → 6.3 ms**（batch 1，JSON Schema） | Table 2 | 近零端到端开销 |
| 核心 C++ **12,000 行** | §4 | 实现规模 |
| batch 越大 CPU 语法压力越大（vLLM 旧版 TPOT 劣化） | §4.2 | "a larger batch size … greater pressure for grammar processing on the CPU side" |

### PagedAttention（arXiv:2309.06180）

| 论断 | 原文位置 | 口径提醒 |
|---|---|---|
| 共享前缀 KV 预存 = "as how OS handles shared library across processes" | §3（共享内存一节） | 与第 13 章 OS 对照总表同源 |

## 二、零成本算账（正文数字的计算过程）

**算账 1：多轮对话的重算比例（正文 1.1）**
第 10 轮对话：每轮用户输入 100 token，上一轮结束时 prompt 已 4000 token → 本轮 prompt 总长 4100 token，其中新信息 100 token。
新信息占比 = 100 / 4100 = **2.44% ≈ 2.4%**；无前缀缓存时重算比例 = 1 − 2.44% = **97.56% ≈ 97.6%**（与正文一致）。

**算账 2：APC 命中粒度（正文 2.1）**
block_size 默认 16（第 13 章消融口径），整块才入 hash → 前缀对齐误差上界 = 16 − 1 = **15 token**；4096 token 公共前缀命中后，剩余 prefill 上界 = 原 prompt − ⌊4096/16⌋×16 = **0**（整除时全命中），非整除时 ≤15 token。

**算账 3：fig3 概念曲线（程序生成，非实测）**
- 吞吐：`thr(c) = 1900·(1−e^(−c/7)) / (1+(c/90)^4)` —— 1900 tok/s 锚点取自第 5 章 4090D 实测封顶值（形态参照，非复测）
- TTFT：`ttft(c) = 0.15 + 0.004·max(c−6,0)²` —— 与 SLO=2s 的交叉点程序解出 c≈28
- TPOT：`tpot(c) = 0.030 + 0.0004c + 0.00035·max(c−18,0)^1.8`
- 吞吐峰值（拐点）程序解出 c≈31；图中两条竖线与标注均由 `np.argmax` 计算，非硬编码

## 三、vLLM 源码行号（commit 642c18d，grep 逐一核验）

### 前缀缓存（APC）

| 行号 | 内容 |
|---|---|
| `vllm/config/cache.py:142` | `enable_prefix_caching: bool = True`（默认开） |
| `vllm/config/cache.py:144` | `prefix_caching_hash_algo = "sha256"` |
| `vllm/engine/arg_utils.py:3020-3024` | `_set_default_chunked_prefill_and_prefix_caching_args`；`default_prefix_caching = model_config.is_prefix_caching_supported` |
| `vllm/engine/arg_utils.py:3054-3060` | `if self.enable_prefix_caching is None:` 套模型默认 |
| `vllm/v1/core/kv_cache_utils.py:707` | `def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None)` |
| `vllm/v1/core/kv_cache_utils.py:723` | docstring: "The current block is assumed to be full."（整块才 hash） |
| `vllm/v1/core/kv_cache_utils.py:637/668-669` | `generate_block_hash_extra_keys`；extra_keys = LoRA + 多模态 + cache_salt + prompt_embeds（**无采样参数**） |
| `vllm/v1/core/kv_cache_manager.py:264` | `def get_computed_blocks(self, request)` |
| `vllm/v1/core/kv_cache_manager.py:297` | `self.coordinator.find_longest_cache_hit(...)` |
| `vllm/v1/core/kv_cache_manager.py:371` | `def allocate_slots(...)` |
| `vllm/v1/core/block_pool.py:139` | "free_block_queue stores the free blocks in eviction order" |
| `vllm/v1/core/block_pool.py:785/801` | "Identify blocks with hash (LRU cache)"；"FIFO reuse of cached blocks for LRU eviction behavior" |
| `vllm/v1/core/block_pool.py:811` | `def evict_blocks(self, block_ids)` |
| `vllm/v1/core/sched/scheduler.py:3009` | `prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()` |

### 结构化输出（guided decoding）

| 行号 | 内容 |
|---|---|
| `vllm/config/structured_outputs.py:11-13` | `StructuredOutputsBackend = Literal["auto","xgrammar","guidance","outlines","lm-format-enforcer"]` |
| `vllm/config/structured_outputs.py:20` | `backend: ... = "auto"`（docstring 注明逐版本可能变） |
| `vllm/v1/structured_output/__init__.py:47` | `class StructuredOutputManager` |
| `vllm/v1/structured_output/__init__.py:126` | `def grammar_init(self, request)`（懒编译入口） |
| `vllm/v1/structured_output/__init__.py:138-140` | "We only support a single backend"（服务级单后端） |
| `vllm/v1/structured_output/__init__.py:141-176` | 四后端 if/elif 选择链（xgrammar/guidance/outlines/lm-format-enforcer） |
| `vllm/v1/structured_output/__init__.py:330` | `def grammar_bitmask`（property）；`:219 fill_bitmask`、`:350 allocate_token_bitmask` |
| `vllm/v1/structured_output/request.py:21/70` | `class StructuredOutputRequest`；`structured_output_key` |
| `vllm/v1/structured_output/utils.py:101` | `def apply_grammar_bitmask(...)`；`:141-160` CPU 侧重排，`:160` non_blocking 上 GPU |
| `vllm/v1/structured_output/utils.py:177/187/191` | `xgr.apply_token_bitmask_inplace(logits, ...)` |
| `vllm/v1/worker/gpu_model_runner.py:193` | 导入 `apply_grammar_bitmask` |
| `vllm/v1/worker/gpu_model_runner.py:4328-4332` | 注释 "Apply structured output bitmasks if present." + 调用（forward 后、`_sample` 前） |
| `vllm/sampling_params.py:112/114` | `_backend`、`_backend_was_auto` 字段 |
| `vllm/sampling_params.py:1244-1261` | 请求级后端切换报错 "Request-level structured output backend selection is not supported" |
| `vllm/sampling_params.py:1288` 附近 | 深嵌套 schema 拒绝（防编译爆炸） |
| `vllm/entrypoints/openai/chat_completion/protocol.py:237/238` | `response_format`、`seed` 字段 |
| `vllm/entrypoints/openai/chat_completion/protocol.py:757` | seed 透传 |
| `vllm/entrypoints/openai/chat_completion/protocol.py:803-812` | json_schema 必填校验 |

### 压测（vllm bench serve）

| 行号 | 内容 |
|---|---|
| `vllm/benchmarks/serve.py:17-18` | 用法 docstring：`--request-rate <…> Default inf`、`--num-prompts <…> Default 1000` |
| `vllm/benchmarks/serve.py:1641` | `--max-concurrency` |
| `vllm/benchmarks/serve.py:1708` | `--request-rate` |
| `vllm/benchmarks/serve.py:1717` | `--burstiness` |
| `vllm/benchmarks/datasets/datasets.py:1587` | `--num-prompts` |
| `vllm/benchmarks/datasets/datasets.py:1888/1894/1900` | `--random-input-len` / `--random-output-len` / `--random-range-ratio` |
| `vllm/benchmarks/datasets/datasets.py:1909` | `--random-prefix-len`（默认 0，注入固定共享前缀） |

## 四、配图登记

| 图 | 脚本 | 性质 |
|---|---|---|
| `images/figures/ch16/fig1_hash_vs_radix.png` | `code/ch16/make_figures.py` | 程序生成结构示意（非实测） |
| `images/figures/ch16/fig2_structured_mask.png` | 同上 | 程序生成机制示意（非实测） |
| `images/figures/ch16/fig3_load_curve.png` | 同上 | 概念曲线（公式见算账 3，1900 tok/s 锚点引自第 5 章实测，曲线本身非实测） |
