# ch20 数据源与结果登记（投机解码：小模型打草稿，大模型批量验收）

新风格：**无性能跑测**。论文数字取原文口径并标 arXiv 编号；生产案例与引擎现状标来源与检索日期（2026-10-16）；加速比账为零成本算术（由 code/ch20/make_figures.py 的 print_table() 输出核对）；接受率实测留给动手任务回填至第四节。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 类型/日期 | 许可 | 用途 |
|---|---|---|---|
| arXiv:2211.17192 Leviathan et al.（ICML 2023） | 论文原文 | arXiv | 2×–3×（T5-XXL）、identical outputs、无重训无结构改动、拒绝采样证明（App. A.1） |
| arXiv:2302.01318 Chen et al.（DeepMind） | 论文原文 | arXiv | speculative sampling，把无损从贪心推广到任意采样 |
| arXiv:2401.10774 Medusa（ICML 2024） | 论文原文 | arXiv | Medusa-1 >2.2×、Medusa-2 2.3–2.8×（摘要原文） |
| arXiv:2401.15077 EAGLE | 论文原文 | arXiv | LLaMA2-70B 延迟 2.7–3.5×、吞吐翻倍 |
| arXiv:2401.07851 survey | 论文原文 | arXiv | 验收规则 min(1, p/q) 与 norm(max(0,p−q)) 修正分布原文 |
| arXiv:2607.17283 | 论文 | arXiv | Proposition 1：一轮输出分布严格等于目标分布（2026 实证复核） |
| arXiv:2604.19767 PayPal 案例 | 论文 | arXiv | γ=3、α≈35.5%（跨并发/温度稳定）、吞吐 22–49%、延迟 −18~33%、零额外硬件 |
| docs.vllm.ai/features/speculative_decoding | 官方文档 | vLLM 文档 | 方法矩阵（EAGLE/EAGLE-3/MTP/draft/PARD/MLP vs ngram/suffix）、低 QPS/高 QPS 分档 |
| nvidia.github.io/TensorRT-LLM DeepSeek R1 MTP 博客 | 官方博客 | NVIDIA | MTP/MTP Eagle 实现、官方检查点仅 1 个 MTP 模块 |
| blog.google《Multi-token-prediction in Gemma 4》 | 官方博客 | Google | bonus token 口径（一次 forward 收整段+1 token） |
| SyncSoft 博客引 DeepSeek-V3 Technical Report | 第三方转述 | 博客 | MTP-1 开箱接受率 >80%、约 1.8×（**建议以技术报告原文为准**） |

## 一、外部公开数据（原文口径，2026-10-16 检索）

| 数字 | 原文口径 | 口径 |
|---|---|---|
| 2×–3× | T5-XXL vs 标准 T5X，identical outputs | 论文原文（Leviathan） |
| >2.2× / 2.3–2.8× | Medusa-1（冻 backbone）/ Medusa-2（联合微调） | 论文摘要（Medusa） |
| 2.7–3.5× | LLaMA2-Chat 70B 延迟加速、吞吐翻倍 | 论文原文（EAGLE） |
| γ=3、α≈35.5%、吞吐 22–49% | PayPal 生产，跨并发档与温度 | 论文（PayPal） |
| α>80%、约 1.8× | DeepSeek-V3 MTP 开箱 | 第三方转述，待核原文 |
| ngram/suffix 适合高峰流量 | "modest speedups without increasing workload during peak traffic" | 官方文档（vLLM） |

## 二、算账区（零成本算术，脚本 print_table() 输出）

### E(α,k) 三点检查（第 2.1 节表）

| α | k | E | 核对 |
|---|---|---|---|
| 1.0 | 4 | 5.000 | = k+1（全收+bonus）✓ |
| 0.0 | 4 | 1.000 | 首位即拒，仅重采 1 个 ✓ |
| 0.8 | 4 | 3.362 | (1−0.8⁵)/(1−0.8) ✓ |

### 验收题代入（第 2.2 节）

α=0.8、k=4、c=0.1 → E=3.362，成本 1+0.4=1.4，**S=2.40×** ✓

### S=E/(1+0.1k) 代入表（第 2.3 节表，脚本实算）

| α \ k | k=1 | k=3 | k=4 | k=7 |
|---|---|---|---|---|
| 0.35 | 1.23× | 1.17× | 1.09× | 0.90× |
| 0.5 | 1.36× | 1.44× | 1.38× | 1.17× |
| 0.7 | 1.55× | 1.95× | 1.98× | 1.85× |
| 0.8 | 1.64× | 2.27× | 2.40× | 2.45× |
| 0.9 | 1.73× | 2.65× | 2.93× | 3.35× |

### 生产反推对账（第 2.4 节）

- PayPal：α=0.355、k=3、c=0.10 → S=1.174（+17%）；c=0.05 → S=1.327（+33%）→ 落在实测 22–49% 区间内，反推得该案例 c≈0.05–0.06 量级。
- c 敏感度（α=0.8、k=4）：c=0.01 → 3.23×；c=0.10 → 2.40×；c=0.30 → 1.53×。

## 三、本书内已实测锚点（仅引用锚点，不重跑）

| 锚点 | 出处 | 本章用途 |
|---|---|---|
| decode 访存 bound、算术强度推导、batch 聚合 ×29 | ch03 | "批改 k 个≈批改 1 个"的依据与高 batch 衰减论证 |
| KV 生命周期与释放 | ch04 | 拒绝位 KV 回滚（【坑 5】） |
| 手写 generate + maxdiff 0.00 逐位对齐 | ch06 | 动手任务 L1 的验收方法与代码底座 |
| 显存账方法 | ch10 | 草稿+目标双模型驻留预算（【坑 3】） |
| chunked prefill/并行按工作区选 | ch14/ch15 | 低 QPS vs 高 QPS 优化选型 |
| 低 QPS 测试纪律与七条控制变量 | ch19 | 任务 D 的实验前提 |

## 四、你的实测回填区（动手任务 A/B/C/D 完成后填写）

| 项 | 你的结果 |
|---|---|
| L1 贪心逐位一致（spec vs ch06，含强制早期拒绝用例） | 【待回填】 |
| 总接受率 ᾱ（≥100 轮） | 【待回填】 |
| 按位置分桶 α（第 1/2/3/4 位条件接受率） | 【待回填】 |
| 按内容分桶 α（代码 vs 闲聊） | 【待回填】 |
| 实测 c（草稿单 token 时间 / 目标前向时间） | 【待回填】 |
| 预期 S（公式） vs 实测 S（墙钟），偏差解释 | 【待回填】 |
| （选）vLLM ngram/MTP 低 QPS TTFT/TPOT 对比 | 【待回填】 |

## 五、配图登记

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| images/figures/ch20/fig1_spec_round.png | code/ch20/make_figures.py | 一轮五步流程+双分支（程序生成示意，中文与字形渲染已核验；零外部图片，无需 CREDITS） |
| images/figures/ch20/fig2_speedup_curve.png | code/ch20/make_figures.py | S(α,k) 曲线+两个生产坐标（**公式代入算账，非实测**，图题已注明） |
