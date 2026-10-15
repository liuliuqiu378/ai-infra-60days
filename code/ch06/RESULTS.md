# 第 6 章 实测记录（2026-10-15，RTX 4090 D，Qwen2.5-0.5B bf16，torch 2.10.0+cu128，transformers 4.56.2）

## 0. 交付物

| 文件 | 内容 |
|---|---|
| `mini_generate.py` | 核心库：`sample_token` / `naive_generate` / `kv_generate` / `kv_stream` / `batched_generate` |
| `01_alignment.py` | A/B：greedy 序列+逐步 logits 对齐 HF generate；C：采样器 5 项单测 |
| `02_benchmark.py` | 总时对比 / 逐步形状 / batch 扩展 / 流式 TTFT |
| `03_long_context_shape.py` | 长 prompt（512/2048）下重算爬升 vs KV 水平 |
| `*_results.json` | 全部原始数字 |

## 1. 正确性对齐（`01_alignment.py` → `alignment_results.json`）

3 个 prompt × 32 个生成 token（共 96 token + 93 个 decode 步 logits）：

| 证据 | 结果 |
|---|---|
| token 序列（含标点） | **逐 token 完全相同** |
| 每步最后位置 logits 最大绝对差 | **0.00e+00**（bf16 逐位精确，非近似） |
| temperature=0 | == argmax ✓ |
| top_k=1 | 等价贪心 ✓ |
| top_p 截断 | 极端分布 50 次全落头部；多峰 200 次只落保留前缀（3 个）✓ |
| 固定 generator 种子 | 两次采样序列一致 ✓ |

踩坑记录：初版对齐显示 logits 差 22.9——原因是 `torch.stack(out.scores)` 形状 (N,**1**,V)，
漏了 batch 维索引导致广播错位；修正 `[:,0]` 后差 0.00。另：transformers 4.56 DynamicCache
不再暴露 `key_cache`，但作为不透明对象传入 forward 仍自动维护，无需手动管位置（显式
`cache_position` 与自动推断结果 maxdiff 仅 0.25 的 bf16 噪声）。

## 2. 速度对比（prompt≈10 token，3 次中位；`bench_results.json`）

| 生成 | 手写·无cache重算 | 手写·KV | HF generate | naive/KV |
|---:|---:|---:|---:|---:|
| 32  | 371 ms (11.6/tok) | **231 ms (7.2/tok)** | 262 ms | 1.6× |
| 64  | 747 ms (11.7/tok) | **464 ms (7.2/tok)** | 519 ms | 1.6× |
| 128 | 1261 ms (9.8/tok) | **932 ms (7.3/tok)** | 1041 ms | 1.4× |

- **手写 KV 比 HF generate 快约 10%**（932 vs 1041ms）：HF 把循环抽象成 LogitsProcessorList、
  StoppingCriteria、各种 wrapper，每层都是 Python 调度；引擎用 C++/CUDA Graph 会再快一个量级。
- 短 prompt 下 naive 也在第 4 章的 ~14ms 固定平台（kernel 启动+大词表 lm_head），1.4–1.6×
  与第 5 章墙钟倍数一致；倍数不在短对话，在长上下文。

## 3. 长 prompt 逐步形状（`longctx_results.json`）

| 初始 S0 | 生成 | naive 总时（首步→末步） | KV 总时（中位） | 倍数 |
|---:|---:|---|---:|---:|
| 512 | 128 | 1639 ms（13.4→12.4ms，全在平台） | 985 ms（7.4ms） | 1.7× |
| **2048** | 128 | **4011 ms（26.9→30.8ms 爬升）** | **953 ms（7.5ms 恒定）** | **4.2×** |

## 4. batch 扩展（96 token，bf16 SDPA，3 次中位）

| B | 整批总时 | 聚合 tok/s | 每 token 步时 |
|---:|---:|---:|---:|
| 1 | 696 ms | 138 | 7.25 ms |
| 2 | 702 ms | 273 | 7.32 |
| 4 | 665 ms | 578 | 6.93 |
| 8 | 710 ms | 1082 | 7.39 |
| 16 | 717 ms | **2143** | 7.47 |

总时几乎不动（±4%），聚合吞吐 ×15.5；这与第 3 章 Roofline、第 5 章服务压测同源。

## 5. 流式 TTFT（prefill+采样到首 token；5 次中位）

| prompt S | 16 | 64 | 256 | 1024 |
|---:|---:|---:|---:|---:|
| TTFT ms | 7.6 | 7.9 | 8.2 | 11.9 |

S≤256 被固定开销钉在 ~8ms；S=1024 才开始抬头——与第 4 章 prefill 扫描同一形状。
