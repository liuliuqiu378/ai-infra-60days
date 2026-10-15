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
