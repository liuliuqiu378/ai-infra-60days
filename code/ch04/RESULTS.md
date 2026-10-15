# 第 4 章 实测记录（2026-10-15，RTX 4090 D 24G，torch 2.10.0+cu128，transformers 4.56.2）

## 环境说明

- 模型：Qwen2.5-7B-In bf16（config 与 safetensors 索引直接取自官方文件；HF 镜像与 ModelScope
  均被限速到约 1 MB/s，15 GB 权重 4 小时下不完，故权重取官方索引精确字节，KV 用官方
  `StaticCache`/`StaticLayer` 的**真实 GPU 分配**测量——KV 分配只依赖 config，不需要权重）
- 0.5B（Qwen2.5-0.5B，L=24 / Q头=14 / KV头=2 / d=64）完整加载，用于交叉验证公式与量激活
- dtype bf16（2 字节/元素），GiB = 1024³ 字节（行业说"GB"通常指 GiB）

## 实验 1：公式 vs 官方真实分配（`01_kv_accounting.py`，数据 `kv_results.json`）

公式（GQA/MQA/MHA 通用）：`KV字节 = 2(K,V) × L × n_kv × d × seq × batch × 2字节`

### 0.5B 交叉验证（完整模型 + 真实 prefill）
| 项目 | 数值 |
|---|---|
| 每 token KV | 12,288 字节 = 12 KiB |
| seq=2048 公式 | 0.0234 GiB |
| seq=2048 官方 `StaticCache` KV 张量实测 | **0.0234 GiB，偏差 +0.00%**，形状 (1,2,2048,64) |
| 权重 / KV / 激活（prefill 2048） | 0.920 / 0.0234 / **0.600 GiB**，峰值 1.544 GiB |
| 激活主体 | lm_head logits（vocab≈152k×2048×2B≈0.58G，模型大小几乎无关）

### 7B KV 速查（官方布局真实分配，config: L=28/28Q/4KV/d=128）
| 场景 | KV 实测 | 占 24G 卡 |
|---|---|---|
| batch=1, seq=2048  | **0.110 GiB**（56 KiB×2048） | 0.5% |
| batch=1, seq=32768 | 1.75 GiB | 7.4% |
| batch=1, seq=131072 | 7.00 GiB | 29.8% |
| batch=8, seq=8192 | 3.50 GiB | 14.9% |
| batch=32, seq=8192 | **14.0 GiB**（32×0.4375） | 59.6% |
| batch=32, seq=32768 | 56 GiB → **直接 OOM** | >100% |
| KV int8 b=1 s=32768 | 0.875 GiB（bf16 一半） | 3.7% |

- 权重 bf16 = 15,231,233,024 字节 = 15.23 GB = **14.19 GiB**（官方 index total_size，7.616B 参数）
- 若为 MHA（KV头=28）：392 KiB/token，GQA 把 KV 压到 **1/7**
- transformers 4.56 的 `StaticCache` 为 lazy 分配，源码 `StaticLayer.lazy_initialization`
  确认每层 K/V 各 `torch.zeros((B, n_kv, max_len, d))`，共 2L 个，与公式逐字节一致。

## 实验 2：无 KV Cache 的代价（`02_with_without_cache.py`，数据 `nocache_results.json`）

0.5B，初始 prompt 32，逐 token 生成（每步只计一次 forward，CUDA sync 后 wall ms）：

| n | 有cache 处理 token | 无cache 处理 token | 墙钟倍数 | FLOPs 倍数 |
|---|---|---|---|---|
| 64  | 96 | 4,096 | 1.61× | 42.7× |
| 128 | 160 | 12,256 | 1.57× | 76.6× |
| 256 | 288 | 40,864 | 1.59× | **141.9×** |

- **反直觉但真实**：n=256 计算量 142×，墙钟只 1.6×。因为重算走 prefill 路径（算力-bound，
  短序列效率低但长序列快），有 cache 走 decode 路径（访存-bound，7.6 ms/步恒定）。
- 纯 prefill 扫描：S=32…512 恒定约 14 ms（kernel 启动+大词表 lm_head 固定开销），
  S=1024/2048/4096/8192 → 16.6 / 28.0 / 59.8 / 152.6 ms（大矩阵区线性增长）。
- decode 单 token 中位 7.62 ms（batch=1）。
- 无 cache 的 FLOPs 倍数随 n 线性增长（≈n/2+1）；墙钟的 n² 拐点随模型/输出变大才会到来，
  但电费/云账单/大并发不可行性立刻显现；高并发下每个请求都重算整段历史，prefill 激活（本次实测 0.6 GiB/请求级）也会迅速放大显存压力。
