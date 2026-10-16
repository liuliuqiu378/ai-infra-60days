# 60 天，从零上手大模型推理 Infra（AI Infra 工程师自学教程）

> 让一个**懂基础 AI 算法（知道 Transformer、跑过 PyTorch、训过或调过小模型）但没做过 Infra** 的人，60 天后具备**独立上手推理引擎开发、性能调优、服务化部署**的能力，并且**懂原理、懂方法、懂硬件、懂行业、懂落地**，能扛住 AI Infra 工程师岗位的三轮技术面。
>
> 这不是论文综述，不是源码注释，不是产品说明书。这是一份"拿起来就能学、学完就能干、面试能通过、能给别人讲明白"的实战指南。

- **版本**：v0.13（第 1–12 章已完成，逐章写作中）
- **写作方式**：先目录、后逐章；每章完成后在仓库提交，并等待验收“继续”后再写下一章；自第 9 章起改为文字理论 + 论文/官方规范/权威评测公开数据为主的写法，减少本机性能跑测
- **数字纪律**：所有关键数字必须可溯源；本机数字须与运行结果一致，外部数字须标注来源与检索日期；无法核实的标注【请联网检索最新实测数据补充】，绝不想当然

---

## 一、读者画像（写作时时刻对准的人）

- ✅ 会 Python，写过 PyTorch 训练/推理脚本，知道 attention、KV Cache 的概念
- ✅ 懂操作系统、内存、进程线程等计算机基础，但**没写过 CUDA**
- ❌ 没读过推理引擎源码，不清楚 prefill / decode 两阶段的工程差异
- ❌ 不懂 GPU 显存层次、NVLink、RDMA 等硬件概念
- ❌ 没做过高并发服务，不知道 TTFT / TPOT / 吞吐 / SLO 怎么权衡
- 🎯 目标：转岗/跳槽 AI Infra 方向，需要作品集与面试弹药

---

## 二、60 天总地图

| 阶段 | 天数 | 主题 | 主线项目 | 学完的标志 |
|---|---|---|---|---|
| 0 破冰 | 1–2 | 建立推理服务体感 + 行业认知 | ollama | 能说清"一句话进去，服务器做了哪几件事"，并知道行业要什么人 |
| 1 地基 | 3–10 | 重新理解"推理"这件事 | 手写 mini 推理 | 能手写一个 1B 模型的 forward，算清 KV Cache 显存账 |
| 2 硬件 | 11–15 | GPU / 显存 / 互联 / 成本 | 硬件实验 + profile | 能算出"这个模型要几张卡、多少钱"，读懂 ncu 带宽指标 |
| 3 物理形态 | 16–19 | 模型文件格式与量化 | **项目一：量化实验室** | 能手工解析权重、产出四档量化选型报告 |
| 4 扎马步 | 20–30 | llama.cpp 精读 | llama.cpp | 能画出 GGUF→token 的完整数据流 |
| 5 上战场 | 31–42 | vLLM 精读 + mini-vLLM | **项目二：mini-vLLM** | 能改 scheduler、能讲清 PagedAttention、能提 PR |
| 6 横向 | 43–47 | SGLang / TensorRT-LLM / LMDeploy 等 | 双引擎对比实验 | 能说清各引擎选型边界并拿数据支撑 |
| 7 前沿 | 48–52 | 投机解码 / PD 分离 / 异构调度 / Strata | 前沿复现 | 能讲透至少一个前沿方向的技术细节 |
| 8 工程化 | 53–55 | 服务化 / K8S / 监控 / 容量 | **项目三：生产级部署** | 能独立搭一套可观测的推理服务 |
| 9 求职 | 56–60 | 简历 / 面试 / 项目包装 | 作品集收尾 | 能扛住三轮技术面 |

### 学习心法（贴墙上）

1. **先总后分，逐个击破**：先搭"业务目标 → 量化指标 → 优化手段"的框架，再钻细节。**禁止第一天就啃 CUDA kernel。**
2. **看源码不算会，能改才算**：每章结束必须动手，且有明确验收标准。
3. **不要贪多**：前沿方向挑一个扎透，比十个都浅尝辄止强十倍。
4. **并发曲线是核心战场**：Infra 工程师的全部工作，就是把"延迟-并发病塌曲线"往右推。
5. **系统经验是资产**：PagedAttention 是虚拟内存的翻版，Continuous Batching 是餐厅翻台，CUDA Stream 对标 CPU 多线程——你是在**把系统思维迁移到新战场**，不是学一门新学科。
6. **Ollama 只用一天**，**Strata 只当窗口不当地基**（原因在第 1、22 章讲透）。

---

## 三、完整章节目录（含天数与预估字数）

> 字数为预估中文字符数（含代码与图表），合计正文约 **25.6 万字**，附录约 **4.5 万字**，全书约 **30 万字**。
> 状态：⏳ 未开始 / 🔄 写作中 / ✅ 已完成并推送

### 阶段 0｜破冰与行业（Day 1–2）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 1 | 1 | [破冰：一小时跑起第一个推理服务](chapters/ch01-icebreaker-ollama.md) | ollama 安装与体感；"一句话进去，服务器做了哪几件事"全链路拆解（含 llama-server 进程实证）；冷启动/prefill/decode/显存实测；4 并发阶梯实测；为什么 Ollama 只配学一天 | 6,000 | ✅ |
| 2 | 2 | [行业认知与岗位地图](chapters/ch02-industry-job-map.md) | 推理 Infra 在干什么；训练 Infra vs 推理 Infra；2026 真实 JD 与薪资表（11 条猎聘/校招已联网核实）；JD 黑话逐句翻译与章节索引；产业链地图+国产化支线；三种落地面貌；职业路径 | 9,000 | ✅ |

### 阶段 1｜认知地基：重新理解"推理"（Day 3–10）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 3 | 3–5 | [训练 vs 推理：prefill 与 decode 是两个世界](chapters/ch03-train-vs-infer-prefill-decode.md) | 吞吐-bound vs 延迟-bound；算术强度 AI 推导；4090D 实测 Roofline（146 TFLOPS/920 GB·s/拐点159）；prefill 82 TFLOPS vs decode 141 tok/s；batch 1→32 聚合吞吐×29；全书优化技术地图 | 10,000 | ✅ |
| 4 | 6–7 | [自回归解码与 KV Cache 全生命周期](chapters/ch04-autoregression-kv-cache.md) | 自回归五步循环；KV 写入/读取/释放生命周期；7B 56KiB/token·seq2048=0.11GiB（官方StaticCache实测，公式偏差0.00%）；无cache计算量142× vs 墙钟1.6×反直觉实测；显存三巨头；GQA/MQA | 10,000 | ✅ |
| 5 | 8–9 | [四大指标与并发崩塌曲线](chapters/ch05-metrics-collapse-curve.md) | TTFT/TPOT/E2E/吞吐/成功率+分位数/Goodput 口径；4090D 32槽实测三阶段崩塌（C=1→1408，成功率100%→40.9%，吞吐 C=32 封顶 ~1900 tok/s）；Little 定律容量规划；与 4B 锚点（200→59.8%/300→46.2%）形态对照；埋点口径与 open/closed model | 11,000 | ✅ |
| 6 | 10–11 | [动手：手写朴素 generate 循环](chapters/ch06-handwritten-generate.md) | 100 行手写 prefill/decode 主循环+temp/top-k/top-p 采样器+流式+左 padding 批量；与 HF generate 逐 token/逐步 logits 对齐（**maxdiff 0.00e+00**）；手写 KV 比 HF 快 10%，长 prompt 2048 无cache/KV 4.2×，B=1→16 聚合吞吐 ×15.5；100 行 vs 真实引擎差距地图 | 11,000 | ✅ |

### 阶段 2｜硬件层（Day 11–15）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 7 | 11–13 | [GPU 架构与显存层次](chapters/ch07-gpu-arch-memory-hierarchy.md) | 4090D 身份证（114 SM/72MiB L2/GDDR6X 384-bit）+Triton 首写；实测显存 918–948 GB/s（91–94%规格）、L2 悬崖 3563→923、PCIe 18.6；bf16 Tensor Core 140 TFLOPS；真实 lm_head 形 roofline 落点（M=1 仅 0.94 TFLOPS=峰值 0.7%），拐点 153 与第 3 章 159 互证；coalescing 14.7×、occupancy 7.7×、launch 3.04µs vs Graph 0.9µs；t_token=P·b/BW 延迟下限与八大优化的总根源表 | 11,000 | ✅ |
| 8 | 14–15 | [互联、国产硬件与选型实操](chapters/ch08-interconnect-hardware-selection.md) | PCIe/NVLink/NVSwitch/IB/RoCE/RDMA 带宽表；为什么 TP 必须 NVLink、PP 能走网络；昇腾 910/CANN/寒武纪生态；“模型→量化→显存→硬件→成本”速查表 + 3 个真实配置；nvidia-smi/nsys/ncu 抓第一次 profile | 13,700 | ✅ |

### 阶段 3｜模型的物理形态：格式与量化（Day 16–19）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 9 | 16 | [模型文件格式与量化](chapters/ch09-weight-formats.md) | safetensors 字节级结构（8B+JSON+数据区，本地实测头部 0.003%）vs GGUF 自包含 vs pickle；mmap 与分词器打包；数值格式全表（FP32→FP8→INT4、粒度）；GPTQ/AWQ/SmoothQuant/QLoRA 论文核心数字；llama.cpp 官方 k-quant 评测；vLLM 支持矩阵；格式×精度×kernel 选型流程 | 14,000 | ✅ |
| 10 | 17–19 | [量化原理与量化实验室（项目一）](chapters/ch10-quantization-lab.md) | 三本账（体积/质量/速度）+ **有效 bpw 决策指标**（HF 官方字节实测：Q4_K_M 跨模型 4.87–5.00bpw、预测误差≤1.7%、名义 bit 低估 1.6GB）；质量三把尺（llama.cpp PPL / Unsloth KLD / Llama-3.1-8B 统一评测）；**Q2 探智用联通论文结案**（Q4 掉幅 0%、Q2 AIME -63%）；量化与验收操作闭环；交付《8GB 老机器模型选型建议》 | 11,500 | ✅ |

### 阶段 4｜扎马步：llama.cpp 精读（Day 20–30）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 11 | 20–24 | [llama.cpp 源码精读：从 GGUF 到计算图](chapters/ch11-llamacpp-source-reading.md) | 项目定位与 GGUF 设计哲学；ggml 张量排布与计算图；后端抽象（CUDA/Metal/Vulkan/CPU）；**文件级源码导航表**；main→load→graph→backend→sample 走读 | 12,000 | ✅ |
| 12 | 25–30 | [混合推理、offload 与并发天花板](chapters/ch12-llamacpp-offload-bench.md) | `-ngl` 分层卸载原理；**offload 悬崖实测（14B 83% 上卡仅 29% 吞吐）+ 三时延公式算账吻合**；并发天花板三道墙（KV/SLO/引擎）与官方 batched-bench 样例表逐行解读；多卡 split-mode 四挡；**GGUF→token 数据流大图** | 12,000 | ✅ |

### 阶段 5｜上战场：vLLM 精读 + mini-vLLM（Day 31–42）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 13 | 31–34 | [PagedAttention：把虚拟内存搬进显存](chapters/ch13-pagedattention.md) | 显存碎片老痛点；页/块/Block Table；为什么消碎片、为什么能前缀共享；**OS 概念↔LLM 推理概念对照总表**；vLLM block manager 源码走读；官方架构实图解读 | 13,000 | ⏳ |
| 14 | 35–38 | [Continuous Batching 与调度器](chapters/ch14-continuous-batching-scheduler.md) | 餐厅翻台类比；Static batching 浪费在哪；请求队列/调度循环/抢占(preemption)/换入换出/Chunked Prefill；v1 引擎 scheduler 源码精读；改参数实验 | 12,000 | ⏳ |
| 15 | 39–40 | [并行策略：TP/PP/SP/EP 与通信量](chapters/ch15-parallel-strategies.md) | 各切什么、通信多大、何时用；all-reduce/all-gather 代价；TP 加速比为什么不线性；MoE 的 EP；切分示意图 + 通信量对照表；多卡实测 | 11,000 | ⏳ |
| 16 | 41–42 | [前缀缓存、结构化输出与压测实战](chapters/ch16-prefix-cache-structured-output.md) | APC vs SGLang RadixAttention；agent 场景为什么要 JSON 约束、引擎层怎么做（guided decoding/grammar/xgrammar）；vLLM OpenAI 服务参数全解；压测出自家并发-延迟曲线 | 10,000 | ⏳ |
| 17 | 33–42 贯穿 | [**核心项目：mini-vLLM 从零实现**](projects/p2-mini-vllm.md) | 500–1000 行 PyTorch：分页 KV 管理 + Continuous Batching + 简单调度器；benchmark 对比 HF generate 吞吐倍数；GitHub 仓库 + README 架构图 + 技术博客（**求职一号作品**） | 11,000 | ⏳ |

### 阶段 6｜横向对比与选型（Day 43–47）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 18 | 43–45 | [七大引擎横评与选型决策树](chapters/ch18-engine-comparison.md) | vLLM/SGLang/TensorRT-LLM/LMDeploy/TGI/MLC-LLM/KTransformers 七维横评（联网更新）；**Ollama 真相：Go 封装层做了哪三件事、为什么会"变笨"**；选型决策流程图 | 11,000 | ⏳ |
| 19 | 46–47 | [同模型双引擎对打实验](chapters/ch19-engine-benchmark-duel.md) | 同硬件同模型同 prompt 集，两套引擎出 TTFT/TPOT/吞吐/崩溃点对比；控制变量方法论；产出一份可放进作品集的对比报告 | 7,000 | ⏳ |

### 阶段 7｜前沿方向（Day 48–52）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 20 | 48–49 | [投机解码：小模型打草稿，大模型批量验收](chapters/ch20-speculative-decoding.md) | 接受率与提速 1.6–2.8× 的数学；draft/target 匹配条件与失效场景；MTP/EAGLE 谱系；**动手实现极简投机解码并测接受率** | 10,000 | ⏳ |
| 21 | 50–51 | [PD 分离、分层 KV Cache 与 MoE 推理](chapters/ch21-pd-disagg-kv-tiering-moe.md) | prefill/decode 为什么该分家；KV 跨节点传输（RDMA/NVLink）的收益代价；GPU→CPU→SSD 三级缓存与长上下文 I/O；MoE 路由/负载均衡/异构调度为什么成新战场 | 11,000 | ⏳ |
| 22 | 52 | [Strata 双案精讲：窗口与边界](chapters/ch22-strata-deep-dive.md) | 两个 Strata 澄清（斯坦福+英伟达分层缓存论文 vs 开源消费级 MoE 引擎）；专家三层调度（GPU 约 4000 专家/内存全量/SSD 仓库）；5070 12G+64G 约 94 tok/s(Q2)/53 tok/s(IQ3_S) 实测核实；**诚实局限：单模型/新项目/匿名/AMD 超时/启动 OOM——当窗口不当地基** | 9,000 | ⏳ |

### 阶段 8｜工程化（Day 53–55）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 23 | 53–55 | [从引擎到服务：API/K8S/可观测/容量/稳定性](chapters/ch23-serving-k8s-observability.md) | OpenAI 兼容 API、排队、限流熔断、负载均衡；Device Plugin/GPU Operator/Volcano/扩缩容；Prometheus+Grafana 监控指标与告警；容量规划与在线离线混部；三级降级预案；升华：传统 Infra 能力为何是底座 | 12,000 | ⏳ |

### 贯穿项目三 + 阶段 9｜求职（Day 56–60）

| 章 | 天 | 标题 | 核心内容 | 字数 | 状态 |
|---|---|---|---|---|---|
| 24 | 53–58 | [**项目三：生产级部署与调优实战**](projects/p3-production-serving.md) | 多卡 TP 加速比；前缀缓存+Chunked Prefill 前后对比；压测到崩溃点并用 nsys/ncu 取证；调优报告；断网/OOM/掉卡故障演练；部署文档+面板截图（**求职二号作品**） | 10,000 | ⏳ |
| 25 | 56–60 | [求职篇：简历、题库、模拟面试、作品集](chapters/ch25-job-hunting.md) | 5 组简历措辞对照；原理/实操/场景/前沿四类高频题；三轮模拟面试（技术深挖/系统设计/项目追问）参考答案框架；README/博客/现场白板架构图怎么展示 | 13,000 | ⏳ |

---

## 四、三大实战项目（最终交付物 = 面试作品集）

| # | 项目 | 时间 | 场景 | 核心交付物 |
|---|---|---|---|---|
| 一 | [量化实验室](projects/p1-quantization-lab.md) | Day 16–19 | 8GB 老机器跑 7B，给出量化选型 | Q2_K/Q4_K_M/Q6_K/Q8_0 四维对照表（体积/显存/速度/质量）+ partial offload 掉速记录 +《选型建议》文档 |
| 二 | [mini-vLLM 从零实现](projects/p2-mini-vllm.md) | Day 31–42 | 证明你懂引擎而非只会调 API | 500–1000 行 PyTorch（分页 KV + Continuous Batching + 调度器）+ benchmark + 仓库 README + 技术博客 |
| 三 | [生产级部署与调优](projects/p3-production-serving.md) | Day 53–58 | vLLM 可对外服务、可观测、扛并发 | TP 部署、压测到崩、nsys/ncu 取证、调优报告、故障演练、Grafana 面板截图 |

---

## 五、附录（必备速查）

| 附录 | 标题 | 内容 | 字数 | 状态 |
|---|---|---|---|---|
| A | [心智模型总表：OS ↔ LLM 推理概念对照](appendix/appendix-a-mental-models.md) | 虚拟内存↔PagedAttention、页表↔Block Table、缺页↔cache miss、swap↔offload、CUDA Stream↔线程等全集 | 5,000 | ⏳ |
| B | [显存计算速查](appendix/appendix-b-vram-estimation.md) | 给定参数量/精度/batch/seq_len 的权重+KV+激活估算公式与速算表 | 4,000 | ⏳ |
| C | [主流引擎横评速查表](appendix/appendix-c-engine-cheatsheet.md) | 七引擎一表流：技术/优势/局限/场景/硬件/社区 | 5,000 | ⏳ |
| D | [硬件选型与成本速查表](appendix/appendix-d-hardware-cost.md) | 消费级/数据中心卡规格、带宽、时价、配置模板与月度成本换算 | 5,000 | ⏳ |
| E | [性能分析工具清单](appendix/appendix-e-profiling-tools.md) | nvidia-smi/nsys/ncu/py-spy/torch profiler/vllm metrics 用法与读图指南 | 6,000 | ⏳ |
| F | [核心论文阅读路线](appendix/appendix-f-papers.md) | FlashAttention→PagedAttention→SGLang→SpecDec→PD 分离→Strata，每篇"读哪几节、记住哪句话" | 7,000 | ⏳ |
| G | [60 天打卡表](appendix/appendix-g-checklist-60days.md) | 每天任务、验收标准、打卡框 | 4,000 | ⏳ |
| H | [自检题库 100 题](appendix/appendix-h-100-questions.md) | 判断 + 简答，专治"看时全都会，做时全不会" | 9,000 | ⏳ |

---

## 六、每章固定结构（硬性写作规范）

1. **开头大白话**：这章讲什么 / 为什么重要 / 学完能干什么 / 在真实工作里对应什么活
2. **正文**：原理 → 方法 → 实战 → 坑。每个技术点必答五问：**是什么 / 为什么需要 / 怎么做 / 坑在哪 / 怎么验证**
3. **结尾**：动手任务（可执行 + 验收标准）+ 自检三问

### 标注规范

| 标注 | 含义 |
|---|---|
| `【配图建议：xxx】` | 需要自绘的示意图 |
| `【实图检索：xxx】` | 联网检索的真实截图/架构图/监控面板，需注明来源 |
| `【案例：xxx】` | 真实项目/真实事故 |
| `【坑：xxx】` | 具体踩坑与解法 |
| `【类比：xxx】` | 生活化比喻（停车场/图书馆书架/餐厅翻台） |
| `【自检：xxx】` | 读者自测 |
| 公式 | 一律 LaTeX，写完紧跟一句大白话 |

### 图表纪律

- 能用表不用文字，能用图不用表；所有对比一律做表
- 每张真实图下必写"**从这张图里你能看到什么**"，禁止只贴图不解释
- 真实图片优先公有领域/CC 授权，**下载到 `images/real/` 本地存储**（防止热链失效），作者/来源/许可登记到 `images/real/CREDITS.md`；授权不明确的官方文档图改以"来源页链接 + 正文引用块"方式呈现
- 程序生成的教学示意图统一放 `images/figures/`，与真实照片明确区分

### 数字纪律

- 凡讲性能必给量级（显存 GB / 延迟 ms / 吞吐 tok/s / 带宽 GB/s / 价格 元）
- 所有关键数字必须与实际运行脚本或可溯源资料一致；无法核实的标【请联网检索最新实测数据补充】
- 配套实测脚本放 `code/`，输出结果回填正文，不允许虚构

---

## 七、仓库目录结构

```
ai-infra-60days/
├── README.md                 # 本文件：总地图 + 完整目录
├── chapters/                 # 25 章正文（ch01–ch25）
├── projects/                 # 三大项目说明书与交付模板（p1–p3）
├── appendix/                 # 附录 A–H
├── code/                     # 全书可运行脚本（按章编号，如 ch06/、ch17/）
└── images/
    ├── figures/              # 自绘教学示意图
    └── real/                 # 联网获取的真实图片 + CREDITS.md 版权登记
```

---

## 八、进度记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-15 | v0.1 | 仓库建立，60 天总地图与 25 章 + 8 附录完整目录定稿 |
| 2026-10-15 | v0.2 | 第 1 章完成（RTX 4090 D 实测）：code/ch01 四脚本 + RESULTS 实测记录 + 3 张数据图 |
| 2026-10-15 | v0.3 | 第 2 章完成：2026 真实 JD/薪资联网核实（智遨通/群核/智象/清昴+猎聘大盘）、JD 黑话→章节索引表、产业链与国产化支线、薪资图与产业链图 |
| 2026-10-15 | v0.4 | 第 3 章完成：Roofline 微基准+两阶段实测（含 CUDA Graph 对照、batch×29）、3 张实测图、全书优化技术地图 |
| 2026-10-15 | v0.5 | 第 4 章完成：KV 显存账官方 StaticCache 真实分配（7B 多场景含 OOM 实证、0.5B 公式偏差 0.00%）、有无 cache 计算量/墙钟双口径实测（142× vs 1.6×）、3 张实测图 |
| 2026-10-15 | v0.6 | 第 5 章完成：16 档 closed-model 流式压测（32槽 ollama/llama.cpp，1→1408 并发），三阶段崩塌曲线+Goodput 饱和图+4B 锚点形态对照，3 张实测图；服务级 KV 预分配 12.6GiB 回扣第 4 章公式 |
| 2026-10-15 | v0.7 | 第 6 章完成：手写 generate 全栈（greedy/采样/流式/批量），与 HF 逐位零误差对齐（96 token + 93 步 logits maxdiff 0.00）；速度实测（手写 KV 比 HF 快 10%、S0=2048 4.2×、batch16 ×15.5、TTFT 扫描），3 张实测图 |
| 2026-10-15 | v0.8 | 第 7 章完成：GPU 架构与显存层次。全部数字本机实测（带宽/L2/PCIe/四精度 GEMM 峰值/瘦矩阵 roofline/Triton coalescing+occupancy/launch vs CUDA Graph），4 张实测图；给出 decode 延迟下限公式与优化技术第一性原理索引表 |
| 2026-10-16 | v0.9 | 第 8 章完成：nsys 首次正式 profile（decode 一步 980 kernel/GPU busy 22.7%/kernel 静态画像 FA 仅 2 block）、α+β 通信建模（TP 必须 NVLink·PP 能走网络全部算成数字）、昇腾 910B/C/HCCS 与寒武纪生态对位表、选型速算器 3 组真实配置（$0.15/百万 token 起），4 张图；ncu 权限坑留档 |
| 2026-10-16 | v0.10 | 第 9 章完成 + 全书写作风格调整（文字/理论/公开数据为主）：safetensors 字节级解析（Qwen2.5-0.5B：头部 0.003%、数据区=参数×2B 零开销）、GGUF 六段结构、数值格式与粒度全表、GPTQ/AWQ/SmoothQuant/QLoRA 论文核心数字、llama.cpp 官方 k-quant 评测（L3-8B 与 L2-70B 双表）、vLLM 格式×硬件支持矩阵；本章零性能跑测 |
| 2026-10-16 | v0.11 | 第 10 章完成（项目一）+ ch03/ch04 补业界对照卡片：体积账以 HF 官方 API 字节为准（三模型七档实测 + 有效 bpw 表 + 跨模型预测误差 1%）、质量以公开评测为准（统一评测 Q4_0<Q4_K_M 反直觉点、联通论文 Q2 探智 -8.91%/AIME -63%）、速度以论文 llama-bench 为准（CPU decode 3.5×、prefill 非单调）；3 张文献数据图；交付 8GB 选型建议；本章零性能跑测 |
| 2026-10-16 | v0.12 | 第 11 章完成（llama.cpp 源码精读）：基于 commit e707790c 的文件级源码导航表（src/ggml/tools/common 四层、行数实测）、入口→GGUF 加载→计算图→后端调度→采样链五段走读（20+ 处 文件:行号 引用逐条核验）、"CLI 本地 spawn server"反直觉事实、ggml_tensor/cgraph/backend_i 三件套结构体逐字段解读、grep/read 动手任务 4 则；零性能跑测、零配图 |
| 2026-10-16 | v0.13 | 第 12 章完成（混合推理、offload 与并发天花板）：offload 悬崖（InventiveHQ 公开基准 14B Q4 -ngl 扫描 2.89→12.5→43.18）+ **三时延公式**算账三档吻合（算账 13 vs 实测 12.5）；源码定位分层安放/流水线并行 6 条件；并发天花板三道墙公式 + 官方 batched-bench 样例表逐行解读（每路 41.57→14.54、TTFT 涨 31×）；多卡 split-mode 四挡与"流水线吞吐/张量延迟"官方口径；8GB 跑 14B 三路对质回扣 ch10；**数据流大图 + 悬崖图 2 张**；零性能跑测 |
