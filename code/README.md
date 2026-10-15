# code/ —— 全书可运行脚本

按章编号：`code/chXX/`。每个脚本文件头都写明了前置条件、用法与对应章节。
所有"正文里出现的关键数字"都来自这些脚本的真实输出，实测记录见各章目录下的 `RESULTS.md`。

## 环境基线

- Ubuntu 24.04 / NVIDIA GPU + 驱动（本书实测机：RTX 4090 D 24G，驱动 580）
- Python 3.10+（仅用标准库的脚本无需装任何依赖；需要绘图时 `pip install matplotlib`）
- ollama 0.34+（服务默认监听 127.0.0.1:11434）

## ch01（对应[第 1 章](../chapters/ch01-icebreaker-ollama.md)）

| 脚本 | 作用 | 前置 | 预期输出 |
|---|---|---|---|
| `01_ollama_timing.py` | 拆 total/load/prefill/decode 计时 | `ollama pull qwen2.5:0.5b` | 两次请求的 tok/s 与显存 |
| `02_cold_start.py` | 冷加载耗时 + `/api/ps` 驻留信息 | 同上 | 0.5B 冷加载约 1.3s（4090D 实测） |
| `03_concurrency.py` | 1 vs 4 并发阶梯 | 12G+ 显存可换 27B，否则用 0.5B+长 num_predict | 完成时刻呈等差阶梯，聚合吞吐≈单请求 |
| `make_figures.py` | 用实测数据生成 3 张配图 | matplotlib + 中文字体 | 输出到 `images/figures/ch01/` |
| `RESULTS.md` | 2026-10-15 实测原始记录（数字溯源） | — | — |

运行顺序：`01 → 02 → 03 → make_figures`。前三个脚本只依赖 Python 标准库。

## ch02（对应[第 2 章](../chapters/ch02-industry-job-map.md)）

| 脚本 | 作用 | 前置 | 预期输出 |
|---|---|---|---|
| `salary_calc.py` | 把 JD 的"30-60k·15薪"换算成年现金包（万元） | 无 | 清昴 28-49 万、瑞芯微 45-90 万等样例 + 交互输入 |
| `make_figures.py` | 生成真实薪资区间图、产业链地图 | matplotlib + 中文字体 | 输出到 `images/figures/ch02/` |

## ch03（对应[第 3 章](../chapters/ch03-train-vs-infer-prefill-decode.md)）

需要 torch + transformers + 一张 CUDA 卡（模型 Qwen2.5-0.5B，HF 镜像可下）。

| 脚本 | 作用 | 预期输出 |
|---|---|---|
| `02_roofline_microbench.py` | 量出本卡 bf16 算力墙与 copy 带宽墙 | 4090D：146 TFLOPS / 920 GB/s / 拐点 159 |
| `01_prefill_decode_probe.py` | prefill TFLOPS、decode eager/CUDA-Graph 水位、batch 1→32 扩展 | 见 RESULTS.md；输出 probe_results.json |
| `make_figures.py` | Roofline 打点图、batch 扩展图、实现水位阶梯图 | 输出到 `images/figures/ch03/` |
| `RESULTS.md` | 2026-10-15 实测原始记录 | — |

运行顺序：`02（量屋顶）→ 01（量模型）→ make_figures`。

## ch04（对应[第 4 章](../chapters/ch04-autoregression-kv-cache.md)）

需要 torch + transformers 4.5x（模型 Qwen2.5-0.5B 必须有；7B **无需下载权重**，
脚本只用本地/官方 config 与 safetensors 索引 + 官方 StaticCache 布局真实分配 KV 显存）。

| 脚本 | 作用 | 预期输出 |
|---|---|---|
| `01_kv_accounting.py` | KV 公式 vs 官方 StaticCache 真实分配（0.5B 交叉验证+7B 六场景+int8） | 7B：56 KiB/token、2048=0.11 GiB、B32×32K 触发 OOM；偏差 0.00%；kv_results.json |
| `02_with_without_cache.py` | 逐 token 对比有无 KV Cache（n=64/128/256） | 计算量 43/77/142× vs 墙钟 ~1.6×；nocache_results.json |
| `make_figures.py` | KV 增长曲线、无cache单步成本+双倍数图、显存三巨头图 | 输出到 `images/figures/ch04/` |
| `RESULTS.md` | 2026-10-15 实测原始记录（含方法学与数字溯源） | — |

运行顺序：`01 → 02 → make_figures`。注意 transformers 4.56 的 StaticCache 为 lazy 分配。

## ch05（对应[第 5 章](../chapters/ch05-metrics-collapse-curve.md)）

零三方依赖（标准库 http.client + 线程 Barrier）。需要 ollama；多槽实例需另起：
`OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=32 OLLAMA_MODELS=<模型根> ollama serve`
（默认 11434 的系统服务通常 -np 1，压不出本章曲线；模型根需可写，manifests 可从
/data/ollama/models 复制、blobs 软链）。

| 脚本 | 作用 | 预期输出 |
|---|---|---|
| `load_test.py` | closed-model 流式压测：逐 chunk 量 TTFT/E2E/eval_count，30s deadline | 见 RESULTS.md；sweep_results.json |
| `make_figures.py` | 崩塌双轴图、Goodput 饱和图、4B 锚点形态对照 | 输出到 `images/figures/ch05/` |
| `RESULTS.md` / `sweep_console.txt` | 2026-10-15 16 档实测原始记录与控制台全文 | — |

复现：`python load_test.py --warmup` 后 `--levels 1,8,...,1408 --tokens 96 --timeout 30`，
再 `python make_figures.py`。

## ch06（对应[第 6 章](../chapters/ch06-handwritten-generate.md)）

| 脚本 | 作用 | 预期输出 |
|---|---|---|
| `mini_generate.py` | 核心库：`sample_token`（temp/top-k/top-p/multinomial）、`naive_generate`（无cache重算）、`kv_generate`（DynamicCache）、`kv_stream`（生成器）、`batched_generate`（左padding） | 被下列脚本 import |
| `01_alignment.py` | greedy 序列+逐步 logits 对齐 HF generate（含 output_scores 形状坑）；采样器 5 项单测 | alignment_results.json：maxdiff 0.00 |
| `02_benchmark.py` | 三实现总时（n=32/64/128）、batch B=1..16、流式 TTFT 扫描 | bench_results.json |
| `03_long_context_shape.py` | S0=512/2048 逐步耗时形状（重算爬升 vs KV 水平） | longctx_results.json |
| `make_figures.py` | 速度对比+逐步形状、batch 扩展、正确性证据卡+TTFT | 输出到 `images/figures/ch06/` |

运行顺序：`01_alignment → 02_benchmark → 03_long_context_shape → make_figures`。
无需联网/ollama；首次运行自动加载本地缓存的 Qwen2.5-0.5B（bf16）。

## ch07（对应[第 7 章](../chapters/ch07-gpu-arch-memory-hierarchy.md)）

需要 `triton`（unsloth 环境自带 3.6）。单卡即可，无需模型/联网。

| 脚本 | 作用 | 预期输出 |
|---|---|---|
| `01_specs.py` | GPU 身份证：SM/L2/时钟 + 推导显存带宽与 FP32 理论（区分实测/【规格】） | gpu_specs.json |
| `02_bandwidth.py` | 512MiB 大张量 copy/add/sum 带宽、L2 悬崖 1MiB→1GiB 扫描、PCIe H2D/D2H、20000 次小 kernel eager vs CUDA Graph | bandwidth_results.json |
| `03_gemm_roofline.py` | fp32/tf32/bf16/fp16 8192³ 峰值；真实 lm_head(896×151936)/MLP(896×4864) 瘦 GEMM M=1..1024 落 roofline | gemm_results.json |
| `04_triton_warps.py` | Triton vadd：跨步访存 coalescing 扫描 + grid=114 隔离 num_warps occupancy | triton_results.json |
| `make_figures.py` | 硬件层次示意图、Roofline、L2+PCIe、coalescing+occupancy 共 4 张 | 输出到 `images/figures/ch07/` |

运行顺序：`01_specs → 02_bandwidth → 03_gemm_roofline → 04_triton_warps → make_figures`。
注意 04 的 coalescing 扫描 stride=32 时瞬时占用约 8 GiB 显存。
