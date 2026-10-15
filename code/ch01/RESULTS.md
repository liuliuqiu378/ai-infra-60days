# 第 1 章 实测原始记录（数字溯源）

- **日期**：2026-10-15
- **机器**：Intel i9-14900K（32 线程）/ 125 GiB 内存 / Ubuntu 24.04.3
- **GPU**：NVIDIA GeForce RTX 4090 D，24564 MiB 显存，驱动 580.126.09
- **软件**：ollama 0.34.0（官方 deb，服务以 systemd 常驻，`ollama serve`）
- **模型**：
  - `qwen2.5:0.5b`（Q4_K_M，494.03M 参数，blob 743,754,955 字节 ≈ 709.3 MiB / 0.69 GiB）
  - `qwen3.8:27b-mtp-q4_K_M`（Q4_K_M，带 MTP 多 token 预测，size 17.40 GB ≈ 16.21 GiB）

## 1. 0.5B 单请求计时（/api/generate，prompt 37 token）

| 调用 | total | load | prefill | decode |
|---|---|---|---|---|
| 第 1 次（热） | 0.089 s | 0.001 s | 37 tok / 0.003 s ≈ **13,093 tok/s** | 43 tok / 0.068 s ≈ **633 tok/s** |
| 第 2 次（热） | 0.069 s | 0.001 s | 37 tok / 0.002 s ≈ **22,424 tok/s** | 43 tok / 0.066 s ≈ **653 tok/s** |
| 长文复现 | — | — | — | 583 tok / 0.90 s ≈ **648 tok/s** |

> 短 prompt 的 prefill 速率波动大，只看数量级（万级 tok/s）。

## 2. 冷启动（先 `ollama stop` 再请求）

| 模型 | 墙钟 | load_duration（权重磁盘→显存） |
|---|---|---|
| qwen2.5:0.5b | 1.378 s | **1.263 s** |
| qwen3.8:27b | 6.78 s（含生成 200 tok） | **4.08 s** |

## 3. 显存驻留（/api/ps + nvidia-smi，模型加载后）

| 模型 | 权重 size | size_vram | nvidia-smi 进程驻留 | 额外开销 |
|---|---|---|---|---|
| 0.5B Q4 | 0.69 GiB | 0.69 GiB（全在显存） | **1.13 GiB**（1155 MiB） | ≈ 0.44 GiB |
| 27B Q4 | 16.21 GiB（17.40 GB） | 17.40 GB 全在显存 | **18.83 GiB**（19287 MiB） | ≈ 2.6 GiB |

额外开销 = CUDA 上下文 + KV Cache（默认 32768 长度，K/V 均 q8_0）+ 计算 buffer。

## 4. 27B 单请求速度

- 热请求生成 200 tok：墙钟 2.74 s，decode **79.9 tok/s**；多次测量 79–83 tok/s
- GPU 利用率 decode 期间约 **98%**（小模型 0.5B 仅约 61%，整卡功耗 93–129 W / TDP 425W）

## 5. 并发阶梯实验（27B，每请求 num_predict=150，temperature=0）

| 场景 | 整批墙钟 | 各请求完成时刻 | 总 token | 按墙钟聚合吞吐 |
|---|---|---|---|---|
| n=1 | 1.95 s | 1.94 s | 150 | **77.1 tok/s** |
| n=4 | 7.70 s | 1.92 / 3.85 / 5.77 / 7.70 s | 600 | **77.9 tok/s** |

四个请求各自的 `eval_duration` 口径 decode 均 ≈ 83 tok/s（不含排队）；按墙钟聚合吞吐与单请求持平 → 默认 `-np 1` 严格串行。

## 6. ollama 实际拉起的内核进程与参数（ps + /proc/<pid>/cmdline 实证）

```
/usr/local/lib/ollama/llama-server
  --model /data/ollama/models/blobs/sha256-c5396e06...   # GGUF blob
  --port 42775 --host 127.0.0.1 --offline
  -c 32768            # 上下文长度 32k
  -np 1                 # 并行槽 = 1（并发串行的直接原因）
  --cache-type-k q8_0 --cache-type-v q8_0   # KV Cache 默认 int8 量化
  --flash-attn on      # 默认开 FlashAttention
  -b 1024 -ub 1024     # 逻辑/物理 batch
  --no-jinja --chat-template chatml           # 强制 chatml 模板
  --context-shift --keep 4                    # 超长时滑动窗口，保留前 4 个 token
```

结论：ollama 的推理内核就是 llama.cpp 官方的 `llama-server`，ollama 自身是 Go 写的 API/模型管理封装层。
