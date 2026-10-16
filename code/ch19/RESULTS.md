# ch19 数据源与结果登记（同模型双引擎对打实验）

新风格：**无性能跑测**。本章为方法论章：第三方对打数字全部联网核实（检索日期 2026-10-16）并标注"第三方口径，以自家实测为准"；vLLM/SGLang 压测工具字段以官方文档为准；vLLM 参数以 /tmp/vllm 源码（commit 642c18d）核验。对打实验本身留给读者在 Day 46–47 执行，结果回填至第四节。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 类型/日期 | 许可 | 用途 |
|---|---|---|---|
| jarvislabs.ai《SGLang vs vLLM: H100 Benchmarks, with TensorRT-LLM》 | 第三方评测 | 博客 | 第 1.1 节：vLLM 默认最稳、SGLang decode 低但 TTFT 慢 |
| agentnative.dev《vLLM vs SGLang for Agent Inference (2026)》（引 spheron.network） | 第三方评测 | 博客 | 10 并发 p50 TTFT 112 vs 120 ms；无前缀差异 2–5% 在噪声内 |
| CSDN《SGLang:为什么 40 万张 GPU 选择了它而不是 vLLM》 | 第三方文章 | 博客 | TTFT 快 23%（79 vs 103 ms）、前缀场景 2.4–4.1×（**未核到原始出处，仅量级参考**） |
| arXiv:2511.16682《Bench360—Benchmarking Local LLM inference from 360°》 | 论文 | arXiv | vLLM A30 扩到 C=64 达标、A10/L4 C=16–32 饱和、调度最稳（论文口径） |
| docs.vllm.ai Benchmark CLI / vllm.benchmarks.serve | 官方文档 | vLLM 文档 | TTFT/TPOT/ITL 字段名与示例值（Mean 71.54 / Median 73.88 / P99 79.49 ms；TPOT excl. 1st token） |
| docs.sglang.io Bench Serving Guide | 官方文档 | SGLang 文档 | bench_serving 四指标；num-prompts ≥ 5×max-concurrency；--flush-cache/--profile |
| arXiv:2507.09019《On Evaluating Performance of LLM Inference Systems》 | 论文 | arXiv | 公平评测方法论引用（baseline fairness / evaluation setup / metric design） |
| DigitalOcean《How to Benchmark LLM Inference Yourself》 | 第三方设计标准 | 博客 | 等长负载坑、未报告采样参数坑（第 2 节⑦、【坑 2】依据） |
| /tmp/vllm（commit 642c18d） | 源码核验 | Apache-2.0 | --no-enable-prefix-caching（benchmarks/kv_cache_watermark.sh）；POST /reset_prefix_cache 路由 rust/src/server/src/routes.rs:140 |

## 一、外部公开数据（原文口径，2026-10-16 检索）

### 第三方对打关键数字（全部为第三方口径）

| 项 | 原文口径 | 出处 |
|---|---|---|
| p50 TTFT @10 并发 | SGLang 112 ms vs vLLM 120 ms（TRT-LLM 105 ms） | agentnative.dev 引 spheron.network |
| 无共享前缀差异 | ~2–5%，落在 run-to-run variance 内 | 同上 |
| TTFT 快 23% | 79 ms vs 103 ms | CSDN（未核原始出处） |
| 共享前缀吞吐 | 2.4–4.1× | 同上（仅量级参考） |
| 调度稳定性 | vLLM QoS 满足到 C=32、A30 扩至 C=64 | arXiv:2511.16682（论文口径） |

### 官方工具口径

| 项 | 口径 | 出处 |
|---|---|---|
| vLLM TTFT 示例 | Mean 71.54 / Median 73.88 / P99 79.49 ms | docs.vllm.ai Benchmark CLI |
| vLLM TPOT 定义 | "Time per Output Token (excl. 1st token)" | docs.vllm.ai |
| SGLang 稳态规则 | num-prompts ≥ 5 × max-concurrency | docs.sglang.io |
| SGLang 冷启动 | --flush-cache 可与 --profile 同用 | docs.sglang.io |

## 二、本书内已实测锚点（仅引用登记值）

| 数字 | 出处 | 本章用途 |
|---|---|---|
| 1408 并发成功率 40.9%、吞吐 C=32 封顶 ~1900 tok/s | ch05 RESULTS | L1/L3 判定线回扣（5.1 表） |
| 压测五条控制变量 | ch16 | 扩成七条对打清单（第 2 节） |
| SGLang 论文命中 74.1%、吞吐 6.4× | ch16 核验（arXiv:2312.07104） | 负载 B 设计基准与 H2 |
| maxdiff 0.00 对齐方法 | ch06 | 报告"可复现"的迁移思路参照 |
| 七引擎横评与决策树 | ch18 | 双雄分支的终审工具 |

## 三、算账区（零成本推演）

1. **控制变量覆盖核算**：第 16 章五条（负载/参数/工具/数据/采样系）扩为七条 = 前五条管负载一致性 + ⑥缓存显式 + ⑦采样固定，覆盖第 1 节四份第三方评测间结论矛盾的三个主要来源（负载前缀结构、参数/硬件、计时口径）→ 7/7（正文第 2 节）。
2. **三条判定线覆盖核算**：L1 错误（成功率口径）/ L2 延迟（p99 SLO）/ L3 吞吐（环比增幅<5%）分别对应第 5 章指标体系的三个维度，任一崩溃模式至少触一条 → 崩溃定义无盲区（正文 5.1）。
3. **噪声地板取值核算**：第三方无前缀差异测量 2–5% 取上限 5% 作判据 → 与第三方口径对齐且偏保守（正文 2.1）。
4. **稳态请求数核算**：num-prompts ≥ 5×C，扫 C=1/4/16/32/64 → 最低请求数 320，按三组负载 × 3 次重复 = 每引擎约 2880+ 请求量级（正文 3.3/5.2 推荐下限）。

## 四、你的实测回填区（任务 A/B/C 完成后填写）

| 项 | 你的结果 |
|---|---|
| 硬件/驱动/CUDA、两引擎版本、权重 commit | 【待回填】 |
| 负载 A（无前缀）：两引擎 TTFT/TPOT/吞吐中位 ± 极差 | 【待回填】 |
| 负载 B（前缀 70%）：差距倍数与方向 | 【待回填】 |
| 负载 C（混合）：综合排名 | 【待回填】 |
| 三条判定线崩溃档（A/B 引擎各三个数） | 【待回填】 |
| H1/H2/H3 支持与否 + 依据表格行号 | 【待回填】 |
| 报告仓库 URL | 【待回填】 |

## 五、配图登记

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| images/figures/ch19/fig1_experiment_design.png | code/ch19/make_figures.py | 实验设计总图：三负载×三假设×七控制变量（程序生成示意，中文渲染已核验；零外部图片，无需 CREDITS） |
| images/figures/ch19/fig2_collapse_duel.png | code/ch19/make_figures.py | 崩溃点对比示意：双引擎并发-延迟曲线+三条判定线（**形态示意，非实测数据**，图题已注明） |
