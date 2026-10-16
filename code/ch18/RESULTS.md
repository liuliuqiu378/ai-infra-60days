# ch18 数据源与结果登记（七大引擎横评与选型决策树）

新风格：**无性能跑测**。本章全部为联网核实（检索日期 2026-10-16）+ 本书内已实测锚点的引用 + 零成本决策推演。所有外部数字按"官方口径 / 第三方口径"分栏登记，可复核。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 类型/日期 | 许可 | 用途 |
|---|---|---|---|
| github.com/huggingface/text-generation-inference README | 官方仓库，2026-10-16 抓取 | Apache-2.0 | 第 0 节 Caution 原文（maintenance mode）与官方推荐继任者名单（任务 C 验收对象） |
| docs.ollama.com Modelfile 参考 | 官方文档，2026-10-16 抓取 | 官方文档 | Ollama 默认参数表（num_ctx 2048、temperature 0.8、top_k 40、top_p 0.9、min_p 0.05、repeat_last_n 64 等） |
| martianlee.github.io《Ollama Architecture Analysis》 | 第三方架构分析，2026-03-15 | 博客 | Go 调度器 /server/sched.go、CGO llama.cpp 集成、ollamarunner 纯 Go 引擎、Runner 抽象 |
| docs.nvidia.com/dynamo 兼容性/发布页 | 官方文档，2026-10-16 | NVIDIA 文档 | vLLM 后端 0.28.0；Dynamo 1.3–1.5 三大一等后端（vLLM/SGLang/TensorRT-LLM） |
| pypi.org/project/lmdeploy | 官方项目页，2026-10-16 | Apache-2.0 | LMDeploy 2026-09-28 发版；TurboMind 论文入选 EuroSys 2027 |
| github.com/MoonshotAI/Kimi-K2 部署指南 | 官方文档，2026-10-16 | 官方文档 | KTransformers 列为旗舰 MoE 部署路径之一 |
| 新浪财经（KTransformers-FineTune 报道） | 第三方报道，2026-10-16 | 媒体 | KT-FT 接入 LlamaFactory（第三方口径） |
| inferenceengineering.tech / iotdigitaltwinplm.com 等 2026 评测 | 第三方评测，2026-10-16 | 博客 | TRT-LLM vs vLLM 吞吐量级"15–25%"（**量级参考，非精确值**） |
| promptquorum / dailyaiworld（MLC-LLM/WebLLM） | 第三方，2026-09 文章，2026-10-16 检索 | 博客 | MLC-LLM 跨端现状、WebLLM 浏览器推理（第三方口径） |
| TGI v3.0 发布数字（3×/13×，2024-12） | 官方口径经 AI Wiki 转述 | 转述 | 1.5 节历史高光（标注官方口径） |
| "TGI 2025-12 维护模式 / 2026-03-21 归档" | 第三方（dreaming.press、markaicode） | 博客 | 时间线日期（**以 GitHub README 现状为准**） |
| SGLang 论文 arXiv:2312.07104 / XGrammar | 论文原文（第 16 章已核验） | arXiv | RadixAttention 74.1%、6.4×、1.7×、<0.3%；6.8 ms/token |

## 一、外部公开数据（原文口径，2026-10-16 检索）

### TGI 退场（第 0 节核心证据）

| 项 | 原文口径 | 口径提醒 |
|---|---|---|
| 维护模式 | README Caution: "text-generation-inference is now in maintenance mode. Going forward, we will accept pull requests for minor bug fixes, documentation improvements and lightweight maintenance tasks." | GitHub README 一手抓取 |
| 推荐继任者 | "we contribute to and recommend using going forward: vllm, SGLang, as well as local engines ... llama.cpp or MLX" | 同上 |
| v3.0 数字 | 长提示词 3× tokens、13× faster than vLLM | **官方口径**，2024-12 发布，经第三方转述 |
| 归档日期 2026-03-21 | 第三方文章 | 以 README 现状为准，正文已标注 |

### Ollama 官方默认参数（第 3/4 节依据）

| 参数 | 官方默认值 | 正文用途 |
|---|---|---|
| num_ctx | **2048** | 根因③（长对话静默截断） |
| temperature | 0.8 | 根因②（发散采样） |
| top_k / top_p / min_p | 40 / 0.9 / 0.05 | 根因② |
| repeat_last_n | 64 | 根因②（repeat_penalty 的作用窗口） |
| repeat_penalty | 文档表 1.0，但 API 参考/实例普遍注入 1.1（分歧） | 正文 3.2 已标注分歧，以本机实测为准 |

### 引擎现状关键点（第 1 节七张名片）

| 引擎 | 关键数字/事实 | 口径 |
|---|---|---|
| vLLM | Dynamo 兼容表后端版本 0.28.0 | 官方文档 |
| Dynamo | 1.3–1.5 三大一等后端 vLLM/SGLang/TRT-LLM | 官方文档 |
| LMDeploy | 2026-09-28 发版；EuroSys 2027 论文接收 | 官方（PyPI） |
| TensorRT-LLM | vs vLLM 吞吐"15–25% higher"量级 | **第三方**，仅量级参考 |
| KTransformers | Kimi-K2 官方部署路径列其方案；KT-FT 接入 LlamaFactory | 官方文档 + 第三方报道 |
| MLC-LLM | WebLLM 浏览器推理、跨端编译 | 第三方 |

## 二、本书内已实测锚点（仅引用登记值）

| 数字 | 出处 | 本章用途 |
|---|---|---|
| Ollama/llama.cpp 32 槽压测：并发 1408 成功率 40.9%、吞吐 ~1900 tok/s 封顶（C=32） | ch05 RESULTS（2026-10-15 本机实测） | 根因④（勿当并发服务）、坑 7 |
| 量化档账、Q2 探智 AIME -63%（联通论文） | ch09/ch10 | 根因① |
| KV 显存账、num_ctx 与 KV 成正比 | ch04/ch12 | 根因③纠正动作 |
| APC 默认开启、链式 hash、压测五控制变量 | ch16（vLLM commit 642c18d 行号核验） | Q5 依据与终审方法 |
| 与 HF 逐 token 对齐 maxdiff 0.00 | ch06 | 应急方案迁移一致性对齐 |

## 三、算账区（零成本推演）

1. **决策树分支覆盖核算**：叶节点共 6 个（Ollama / MLC-LLM / KTransformers / 非NVIDIA→vLLM或SGLang+LMDeploy侧枝 / TRT-LLM / SGLang / vLLM），覆盖七引擎全部且 TGI 显式排除 → 覆盖率 7/7，排除 1/1（正文 5.1）。
2. **"变笨"四根因修复成本核算**：根因②③为传参级修复（0 成本）、根因①为换档（下载成本）、根因④为架构级（换引擎）→ 排序建议"先②③后①"依据修复成本递增（正文 4.2）。

## 四、你的实测回填区（任务 A/B 完成后填写）

| 项 | 你的结果 |
|---|---|
| 模型与量化档（ollama show） | 【待回填】 |
| 默认参数 vs 显式参数（temperature 0/top_k 1/repeat_penalty 1.0/num_ctx 8192）输出差异 | 【待回填】 |
| 同 prompt 10 连发一致性 | 【待回填】 |
| 目标场景决策树路径（Q1→Q5 每问答案 + 最终引擎） | 【待回填】 |
| TGI README Caution 截图 | 【待回填路径】 |

## 五、配图登记

| 文件 | 生成脚本 | 说明 |
|---|---|---|
| images/figures/ch18/fig1_decision_tree.png | code/ch18/make_figures.py | 程序生成选型决策树（中文渲染已核验；零外部图片，无需 CREDITS） |
