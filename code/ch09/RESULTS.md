# 第 9 章 数据记录（2026-10-16，新风格：文献/官方规范为主 + 一次零成本结构观察）

> 本章起教程改为"文字+理论+公开来源数据"为主，不做性能跑测。所有外部数字均
> 标注来源与检索日期；本地只做了一件秒级、不加载权重、不碰 GPU 的结构观察。

## 0. 方法学声明

| 类别 | 内容 | 说明 |
|---|---|---|
| 官方规范 | safetensors 格式（huggingface/safetensors README + PyTorch projects 页）；GGUF 格式（ggml-org/ggml docs/gguf.md）；FP8（OCP OFP8 Rev 1.0, 2023-12-01）；PCIe/NVLink 等沿用 ch08 | 2026-10-16 检索 |
| 论文数据 | GPTQ (arXiv:2210.17323)、AWQ (arXiv:2306.00978v4)、SmoothQuant (arXiv:2211.10438)、QLoRA (arXiv:2305.14314, NeurIPS'23) | 数字核对论文原文表 |
| 官方评测 | llama.cpp `tools/perplexity/README.md`（LLaMA 3 8B，CUDA backend，Wikitext-2，按 KLD 排序） | 2026-10-16 抓取 |
| 框架文档 | vLLM features/quantization（格式与硬件支持矩阵） | 2026-10-16 抓取 |
| 本地结构观察 | `parse_safetensors.py` 解析 Qwen2.5-0.5B 文件头 | 2026-10-16 实跑 |

## 1. 本地结构观察实测（`parse_safetensors.py`，Qwen/Qwen2.5-0.5B bf16）

| 指标 | 实测值 |
|---|---|
| 文件大小 | 988.1 MB |
| 头部 8 字节 + JSON | header=32,280 字节，占全文件 **0.003%** |
| 张量数 | 290（全部 BF16） |
| 总参数 | 494,032,768（=0.5B，与 HF 卡片一致） |
| data_offsets 连续性 | True（张量紧密排列，可直接 mmap） |
| 数据区 | 988,065,536 字节 = 494,032,768 × 2B（BF16 全展开，零 padding） |
| `__metadata__` | `{"format": "pt"}` |
| 首/末张量 | embed_tokens (151936,896) / model.norm.weight (896,) |

推论：读文件头只需 32KB 即可拿到完整张量清单（懒加载/mmap 的基础）；988MB
数据区无任何开销，"494M 参数 × 2 字节"严丝合缝。

## 2. safetensors 格式（官方规范）

`[8B little-endian uint64 = JSON 头长度][UTF-8 JSON 头][纯二进制数据区]`。
JSON 每张量条目：`{dtype, shape, data_offsets:[start,end]}`；全文件小端序。
安全：仅存数值张量，反序列化不可能执行任意代码（替代 pickle/`torch.save`）；
快：near zero-copy，可 mmap 后按需取张量。来源：huggingface/safetensors README、
PyTorch projects 页，2026-10-16。

## 3. GGUF 格式（官方规范 ggml/docs/gguf.md）

魔数 `GGUF` + 版本(u32) + 张量数(u64) + KV 元数据数(u64) → KV 对（值自带类型，
覆盖 arch/name/quantization_version/tokenizer 全套分词器与 chat template）→ 对齐
（`general.alignment`，默认 32，用 0x00 填充）→ 张量描述符（name/ndims/shape/
ggml_type/data offset）→ 数据区。2023-08-21 由 Georgi Gerganov 发布，GGML/GGUF
为 llama.cpp 生态，单文件自包含（权重+分词器+模板）。来源：ggml 官方 docs、
llama.cpp 文档，2026-10-16。

## 4. 数据类型速查（OCP OFP8 Rev1.0 2023-12-01 / NVIDIA 官方文档）

| 格式 | 结构 | 可表示范围 | 定位 |
|---|---|---|---|
| FP32 | 1/8/23 | ±3.4e38 | 训练基准 |
| TF32 | 1/8/19（无尾数压缩） | 同 FP32 | Ampere+ 矩阵乘 |
| FP16 | 1/5/10 | ±65504 | 推理主力 |
| BF16 | 1/8/7 | 同 FP32 量级 | 训练/推理稳态 |
| **FP8 E4M3** | 1/4/3 | **±448**（bias 7，emax 8） | 前向权重/激活 |
| **FP8 E5M2** | 1/5/2 | **±57344**（bias 15，emax 15） | 梯度（动态范围同 FP16） |
| INT8 | 对称 ±127 / 非对称 0-255 | — | W8A8/AWQ |
| INT4 | 0-15 | — | GPTQ/AWQ/NF4 |

## 5. 论文关键数字

**GPTQ（arXiv:2210.17323，ICLR'23）**：一次成形（one-shot）二阶信息（OBQ）优化；
**175B 模型 4 GPU·小时**量化到 3/4 bit，精度近乎无损；逐层独立、低秩列分块 +
Cholesky 重构 + lazy batch 更新；校准集小且鲁棒（128 条样本即可，数据分布不敏感）。
LLaMA 各档 WikiText2 PPL：【请联网检索 GPTQ 论文 Table 2 原值】（二手转引 5.85/6.13 冲突，按数字纪律不写入）

**AWQ（arXiv:2306.00978v4，MLSys'24）**：激活分布找到 ~1% 关键通道，按激活
幅度逐通道缩放保护（**不是**按权重大小选，Table 1 证明按权重/随机选几乎无效）；
无需校准集反传（数据高效、分布鲁棒）。INT3/g128 核心表（WikiText PPL）：
OPT-1.3B FP16 14.62 / RTN 119.00 → 激活选 1% 保护 16.68；OPT-6.7B 10.86 /
23.54 → 11.36；OPT-13B 10.13 / 46.04 → 10.42（激活保护几乎回到 FP16；按权重
保护仅 48.96/22.37 等，随机更差）。TinyChat 系统层：4-bit 相对 FP16 边缘端
**3–4× 加速**，核心是 on-the-fly 反量化 + SIMD 打包 + kernel 融合（decode 是
访存主导，权重流量占大头——与 ch03/ch07 结论互证）。

**SmoothQuant（arXiv:2211.10438，ICLR'23）**：激活离群值导致 W8A8 难做；用
s_j = max|X_j|^α / max|W_j|^(1-α)（α≈0.5）把"激活难量化"平滑进权重，
实现 W8A8；**最高 1.56× 加速、2× 显存**，精度损失可忽略；单卡部署 OPT-175B
（此前需两卡）；覆盖 OPT/BLOOM/GLM/Llama-1,2/Falcon/Mistral/Mixtral。

**QLoRA（arXiv:2305.14314，NeurIPS'23）**：**NF4**（4-bit NormalFloat，信息论
最优的正态分布 4-bit 码本）+ 双重量化（量化后的 scale 再量化成 FP8，每参数省
0.137 bit）+ paged optimizer（防长序列 OOM）；权重 blocksize 64、scale blocksize 256；
梯度经 LoRA adapter 以 BF16 反传，**冻结 4-bit 基座**。显存：65B = 32.5GB(4bit)
+ ~1GB(LoRA r64) + 激活 ≈ **48GB 单卡可微调**；33B ≈ 17.5GB。

## 6. llama.cpp 官方 k-quant 评测（tools/perplexity/README.md，2026-10-16 抓取）

**当前 LLaMA 3 8B 表**（CUDA backend，Wikitext-2，f16 基线 14.97 GiB / PPL
6.2332；imatrix=WT 10m 表示 Wikitext 1000 万 token 重要性矩阵）：

| 量化 | imatrix | 体积 GiB | PPL | ΔPPL | KLD | Mean Δp |
|---|---|---|---|---|---|---|
| f16 | — | 14.97 | 6.2332 | 0.0015 | 0.000551 | 0.001% |
| q8_0 | — | 7.96 | 6.2343 | 0.0027 | 0.001355 | -0.019% |
| q6_K | — | 6.14 | 6.2534 | 0.0217 | 0.005452 | -0.007% |
| q5_K_M | — | 5.33 | 6.2886 | 0.0570 | 0.010762 | -0.114% |
| q4_K_M | WT 10m | 4.58 | 6.3829 | 0.1513 | 0.028152 | -0.389% |
| q4_K_M | 无 | 4.58 | 6.4071 | 0.1755 | 0.031273 | -0.596% |
| q4_K_S | WT 10m | 4.37 | 6.4097 | 0.1781 | 0.031951 | -0.531% |
| iq4_NL | WT 10m | 4.35 | 6.4556 | 0.2240 | 0.035742 | -0.590% |

（注意 imatrix 让 q4_K_M 的 PPL 从 6.4071 降到 6.3829、KLD 从 0.0313 降到
0.0282——重要性矩阵在 4-bit 是免费午餐；q8_0/q6_K 与 f16 差异已进入不确定度
区间，即"几乎无损"。）

**旧 LLaMA 2 70B 表**（体积/ΔPPL vs fp16 128.5GiB/3.4313）：
Q2_K 27.27GiB +8.82%；Q3_K_S 27.86 +7.89%；Q3_K_M 30.83 +4.72%；Q3_K_L
33.67 +3.80%；Q4_K_S 36.39 +1.57%；**Q4_K_M 38.54 +1.20%**；Q5_K_S 44.20
+0.50%；Q5_K_M 45.41 +0.40%；Q6_K 52.70 +0.16%；Q4_0 36.20 +3.61%；Q4_1
40.20 +2.37%（同体积 K-混合精度明显优于老式 Q4_0/Q4_1）。
官方 caveat：llama.cpp 数字与其他项目不可直接比（依赖实现细节）；困惑度非
模型间可比指标（分词器不同）；微调模型困惑度常升高但人类评分反而更好。

## 7. vLLM 量化格式与硬件支持（docs.vllm.ai features/quantization，2026-10-16）

格式：AutoAWQ、BitsAndBytes、GPTQModel、Intel Neural Compressor、LLM Compressor
（FP8 W8A8 / INT4 W4A16 / INT8 W4A8 / INT8 W8A8）、NVIDIA Model Optimizer、
Online Quantization、AMD Quark、Quantized KV Cache、TorchAO、FP8 ViT Attention。
后端选择：`--linear-backend`（全局）/ `linear_backend_per_quant`（按方案覆盖），
如 nvfp4_w4a16 用 Humming、其余用 CUTLASS。

支持矩阵（摘）：AWQ：Turing/Ampere/Ada/Hopper ✅、Volta ❌；GPTQ：Volta→Hopper
全 ✅；Marlin（GPTQ/AWQ/FP8/FP4 高效 kernel）：Turing* /Ampere/Ada/Hopper ✅、
Volta ❌、AMD/CPU ❌；FP8 W8A8：**仅 Ada+Hopper**（AMD ✅）；GGUF：Volta→Hopper
+AMD ✅、Intel/CPU ❌；bitsandbytes：Volta→Hopper ✅、AMD ❌。

## 8. 体积速算（理论值，第 10 章量化实验室将与实测对照）

7B fp16 ≈ 13.0 GiB；Q8≈8.0、Q6≈6.0、Q5≈5.0、Q4≈4.0、Q3≈3.0、Q2≈2.5 GiB
（bits/8 × 参数 + 分词器/头开销）。llama.cpp 社区口径 7B 体积与通用推荐
（Q4_K_M 4.1GB / Q8_0 7.0GB 等）与上式一致；通用场景 Q4_K_M 默认推荐，
代码/医疗建议 Q5+，Q3 以下仅极限内存场景。

## 9. 环境

- 本地观察：RTX 4090D 单卡（未用 GPU），conda unsloth（python 3.11 + safetensors）
- 全部外部数据：2026-10-16 检索；引用时已逐一标注来源，无本机性能跑分
- 本章唯一脚本 `parse_safetensors.py`：纯 stdlib，秒级，不联网
