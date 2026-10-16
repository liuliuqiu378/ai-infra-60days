# ch21 数据源与算账登记（PD 分离 / 分层 KV / MoE）

> 本章零性能跑测。所有数字为：① 论文/官方文档引用（标来源与检索日期 2026-10-16）；② 零成本算术（输入全部来自本书登记值）；③ 第三方转述（单独标注）。

## 一、数据源总表

| 数据 | 值 | 来源 | 检索日期 | 口径 |
|---|---|---|---|---|
| PD 分离 SLO 吞吐收益 | 7.4× / 收紧 12.6× | DistServe arXiv:2401.09670 / OSDI'24 | 2026-10-16 | 论文口径，ch14 已核验沿用 |
| Dynamo 官方三步 | prefill→传 KV→decode | docs.nvidia.com/dynamo《Disaggregated Serving》 | 2026-10-16 | 官方口径 |
| NIXL 传输 | VRAM→VRAM 直写、64–128 token/块、非阻塞 | 同上 | 2026-10-16 | 官方口径 |
| 三家传输语义 | SGLang `bootstrap_info` 可流水 / vLLM `kv_transfer_params` 同步 / TRT-LLM `opaque_state` 同步 | 同上 | 2026-10-16 | 官方口径 |
| HiCache 分层原话 | L1 显存 / L2 主机内存 / L3 分布式存储（受 CPU 三级缓存启发） | kvcache-ai.github.io Mooncake×SGLang 设计文档 | 2026-10-16 | 官方口径 |
| G1–G4 命名 | NVIDIA context hierarchy | arXiv:2607.18141 | 2026-10-16 | 论文口径 |
| G2 容量/延迟量级 | 50–500 GB / 50–200 µs | hivebook 汇总表 | 2026-10-16 | **第三方口径，量级参考** |
| layer-wise streaming | 传第 L 层时算第 L+1 层 | zhongzhuzhou.org Mooncake 技术评论（2026-05-28） | 2026-10-16 | 第三方口径 |
| DeepSeek-V3 均衡 | auxiliary-loss-free strategy（原文引用见正文 4.2） | arXiv:2412.19437 | 2026-10-16 | 论文原文 |
| EPLB 机制 | redundant experts + group-limited routing | DeepSeek 官方 EPLB README | 2026-10-16 | 官方口径 |
| 推理期负载偏斜 | "in practice the distribution … can be highly skewed" | docs.vllm.ai Expert Parallel Deployment（`--enable-eplb`） | 2026-10-16 | 官方口径 |
| Mooncake 预测式拒绝 | decode 过载时 HTTP 429 拒在 prefill 前 | 论文摘要 + 第三方转述 | 2026-10-16 | 论文口径 + 转述 |

## 二、外部公开数据（DistServe / Mooncake）

- DistServe（OSDI'24）：满足 SLO 请求吞吐最高 7.4×、SLO 收紧下 12.6×（ch14 已核验原文口径，本章沿用不重复核）。
- Mooncake（arXiv:2407.00079）：KVCache-centric 调度、预测式 early rejection、长上下文见长（论文摘要口径）。
- Mooncake 第三方解读传输量："200 token ≈ 64 MB、~1 ms"（zhongzhuzhou.org，2026-05-28，检索 2026-10-16，第三方口径）——与本书 11 MB / 0.2 ms 同量级交叉印证（差异源于其按 dense 口径计，本章 56 KiB/token 为 GQA 口径）。

## 三、算账区（零成本算术，可由 make_figures.py print_account() 复现）

**输入（本书登记值）**：KV/token = 56 KiB（ch04，7B GQA 口径）；NVLink 50 GB/s、PCIe 16 GB/s（ch07/ch08）；prefill 82 TFLOPS、7B bf16 权重 13.5 GB（ch03/ch09）。

**2.2 节传输表**（KiB 数值按 MB 记的既定口径，与正文一致）：

| 场景 | KV 量 | NVLink (50 GB/s) | PCIe (16 GB/s) |
|---|---|---|---|
| 200 token | 11 MB | 0.2 ms | 0.7 ms |
| 4k token | 224 MB | 4.5 ms | 14 ms |
| 32k token | 1800 MB | 36 ms | 113 ms |

**3.3 节下沉判据参数**（建模口径，需向读者交代假设）：

- 搬运时间 `T_move = L0 + 2N·b/BW`；重算时间 `T_re = F + N·(1/r)`
- `b = 57344 B`（56 KiB/token）；`r = 1/170.7 µs ≈ 5859 tok/s`（14 GFLOPs/token ÷ 82 TFLOPS，ch03 登记口径）
- `F = 15 ms`（重算固定开销：13.5 GB ÷ 900 GB/s，权重搬运主导）；`L0 = 0.5 ms`（每块握手固定开销，**建模假设**，量级参考非实测）
- **交叉带宽**（两条线斜率相等点）：`2b/r = 2 × 57344 / 170.7e-6 = 0.672 GB/s` —— 现实链路（≥16 GB/s）远高于此，**斜率意义上任何块长搬运都更快**；固定开销才产生分界：PCIe 下 N* ≈ 28k token、0.1 GB/s 慢链路下 N* ≈ 70 token
- 图 2 右图分界曲线即由上述参数绘制；`L0`、`F` 为建模假设，换实测值后结论方向不变（交叉带宽由 b/r 决定）

## 四、你的实测回填区（Day 50–51 动手任务）

- [ ] 任务 A：`kv_transfer_account.py` 输出与 2.2 节表格逐格一致：____
- [ ] 任务 A：GQA/dense 双口径计算结果：____
- [ ] 任务 B：512 token 块交叉带宽（你的参数）：____ GB/s；结论：____
- [ ] 任务 C（选）：该不该 PD 分离的判据结论与三要素（引擎版本/配比/传输介质）：____

## 五、配图登记

| 图 | 文件 | 生成脚本 | 说明 |
|---|---|---|---|
| 图 1 | images/figures/ch21/fig1_pd_pipeline.png | code/ch21/make_figures.py::fig1 | PD 分离三步流水线与三家传输语义结构示意，零外部图片 |
| 图 2 | images/figures/ch21/fig2_transfer_tiering.png | code/ch21/make_figures.py::fig2 | 左=2.2 节传输时间柱状（算术）；右=3.3 下沉判据分界曲线（参数见第三节） |

无外部图片，无需 CREDITS.md。
