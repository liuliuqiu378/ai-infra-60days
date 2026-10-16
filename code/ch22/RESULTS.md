# ch22 数据源与算账登记（Strata 双案精讲）

> 本章零性能跑测。所有数字为：① 论文摘要原文（标 arXiv 与检索日期 2026-10-16）；② 项目 README 官方口径（自测数据单独标"自报口径"）；③ 本书分析推论（如 1.77× 为简单除法）。

## 一、数据源总表

### 案一：论文 Strata（arXiv:2508.18572）

| 数据 | 值 | 来源 | 检索日期 | 口径 |
|---|---|---|---|---|
| TTFT 收益（vs vLLM+LMCache） | 最高 5×（up to） | 论文 Abstract 原文 | 2026-10-16 | 论文口径，长上下文基准，峰值非均值 |
| 加速（vs TensorRT-LLM） | 3.75× | 论文 Abstract 原文 | 2026-10-16 | 论文口径，long-context benchmarks |
| 短上下文 | 不劣化（without degrading） | 论文 Abstract 原文 | 2026-10-16 | 论文口径 |
| 部署状态 | Built on SGLang, deployed in production | 论文 Abstract 原文 | 2026-10-16 | 论文口径 |
| 问题陈述 | Low Bandwidth Utilization in KV Cache Transfers；loading-bound not compute-bound | 论文 §3 标题与摘要 | 2026-10-16 | 论文口径 |
| 三项设计 | GPU-assisted I/O（引 Min et al. 2020 / Qureshi et al. 2023）；解耦 GPU/CPU 布局；cache-aware 调度（Deferral on Delay Hit / Balanced Batch Formation / Bubble Filling） | 论文 §4 目录核验 | 2026-10-16 | 论文口径 |
| 发表信息 | 2025-08-26 提交；OSDI'26 版本（osdi26-xie-zhiqiang.pdf） | arXiv 页 + usenix.org | 2026-10-16 | 官方口径 |

### 案二：开源 Strata（github.com/Niko1221/Strata）

| 数据 | 值 | 来源 | 检索日期 | 口径 |
|---|---|---|---|---|
| 模型 | Qwen3.8-Flash-Next（125B MoE） | README | 2026-10-16 | 官方口径 |
| 专家结构 | 24,576 个专家，每 token 用 10 个 | README "How does it work?" | 2026-10-16 | 官方口径 |
| 5070 生成速度 | Q2_0 94 / IQ2_XS 79 / IQ3_XXS 62 / IQ3_S 53 / Coder 55 tok/s | README 实测表 | 2026-10-16 | **作者自测（自报口径）**；Ryzen 5 7600 + 64 GB RAM；4K 输出/32K 输入；Q2_0 用引擎 0.1.36、其余 0.1.26 |
| 5070 读 prompt | 2,650 / 2,090 / 1,750 / 1,620 / 2,180 tok/s | 同上 | 2026-10-16 | 自报口径 |
| AMD 9070 XT (16 GB) | Q2_0 60 tok/s 生成 / 1,160 tok/s 读取 | README 实测表 | 2026-10-16 | 自报口径；47 GB RAM；Windows 读图不可用 |
| RTX 3090 预测 | should write about 100–140 tok/s | README | 2026-10-16 | **预测口径（非实测）** |
| 投机解码 | same answer, 1.6–1.8× sooner | README | 2026-10-16 | 官方口径 |
| 大块 prefill | up to 8,192 tokens/chunk, >1,000 tok/s 读取 | README | 2026-10-16 | 官方口径 |
| Coder 版 | 去掉一半专家，达全模型 SWE-bench Verified 91%（作者自测） | README | 2026-10-16 | 自测口径 |
| RAM 不足 SSD 档 | UD-Q4_K_XL 在 64 GB 机器 7–8.5 tok/s | README | 2026-10-16 | 自报口径（换入慢于计算的实证） |
| 启动行为 | 首启加载 35–55 GB 进 RAM 并锁部分给显卡，卡 1–3 分钟 | README | 2026-10-16 | 官方口径 |
| 硬件门槛 | 12 GB+ 显存、≥32 GB RAM（64 GB 跑全尺寸）、约 80 GB 磁盘 | README | 2026-10-16 | 官方口径 |
| 并发 | 默认一次一个请求；`"parallel": 2` 且 12 GB 卡会更慢 | README + BATCHING.md 引用 | 2026-10-16 | 官方口径 |
| 许可 | MIT（模型另有许可） | README | 2026-10-16 | 官方口径 |

## 二、算账区（零成本算术）

- **94/53 = 1.77×**：Q2_0 相对 IQ3_S 的速度差（正文 2.2/2.3 节口径；make_figures.py::print_card() 打印核对）。
- **10/24,576 ≈ 0.041%**：每 token 激活专家占比（README 两数字相除，说明"算少拿多"的结构性矛盾）。
- **案二 SSD 档 7–8.5 tok/s vs Q2_0 94 tok/s ≈ 11–13×**：换入链路（SSD）跟不上时的速度塌陷，与 ch12 offload 悬崖同构（本书分析）。
- 案一无本地算账输入（论文数字全为引用，不重算）。

## 三、口径分级（本章方法论核心）

| 级别 | 本章例子 | 引用姿势 |
|---|---|---|
| A 论文原文 | 5×/3.75×/不劣化短上下文 | 数值+对比对象+基准类型 |
| B 官方文档 | 24,576×10 专家、1.6–1.8×、8K chunk | 数值+项目+检索日期 |
| C 自报实测 | 94/53 tok/s、91% SWE-bench | 数值+硬件三要素+"作者自测口径" |
| D 预测量级 | 3090 "should 100–140" | 只作量级参考，不入结论 |

## 四、你的实测回填区（Day 52 动手任务）

- [ ] 任务 A：双案数字核验卡（7 行，任一数字 30 秒答出"对谁/条件/口径"）：____
- [ ] 任务 A：指出"最经不起追问的数字"及原因（参考答案：94 tok/s，自测+五重绑定）：____
- [ ] 任务 B：案二 200 字诚实评审 + 引入判据：____
- [ ] 任务 C：实测 `BENCH.bat` 对照表（硬件/版本/量化三要素）或 32K KV 搬运纸上算账（PCIe 16 GB/s 口径）：____

## 五、配图登记

| 图 | 文件 | 生成脚本 | 说明 |
|---|---|---|---|
| 图 1 | images/figures/ch22/fig1_dual_strata.png | code/ch22/make_figures.py::fig1 | 双案合流全景结构示意（原理→双案三段→合流→边界），零外部图片 |

无外部图片，无需 CREDITS.md。
