# 第 8 章 实测记录（2026-10-16，RTX 4090 D 单卡，nsys 2025.3.1 / CUDA 12.8）

> 本章三件事：① 第一次用 Nsight Systems 给第 6 章的 KV decode 做正式 profile；
> ② 用 α+β 通信模型把"TP 必须 NVLink、PP 能走网络"算成数字（纯建模，非多卡实测，
> 假设全部列明）；③ 写一个"模型→量化→显存→硬件→成本"选型速算器跑 3 组真实配置。
> ncu 因驱动权限不可用（见 §4），这是云上的常态，替代方案在正文讲透。

## 0. 环境与脚本

| 文件 | 内容 |
|---|---|
| `01_profile_target.py` | nsys 目标：Qwen2.5-0.5B **base**（bf16），1 条 prompt，prefill 后 decode 64 步；NVTX 标注 `prefill` 与 `decode_i`（i=0..63） |
| `02_comm_model.py` | 纯计算：α+β 模型算 TP all-reduce（decode/prefill）与 PP 边界传输，5 种链路 |
| `02_ncu_target.py` / `run_ncu.sh` | ncu 目标与封装脚本（最终因权限失败，报错留档见 §4） |
| `03_selection.py` | 选型速算器：4 款卡 × 3 组场景，输出每 token 墙钟与 $/百万 token |
| `make_figures.py` | 4 张图 → `../../images/figures/ch08/` |
| `decode_profile.nsys-rep/.sqlite` | nsys 原始 trace（3.1MB / 8.3MB） |
| `kern_sum.csv` / `api_sum.csv` | nsys stats 导出的 kernel/CUDA API 汇总（头部各 7/3 行） |
| `nsys_results.json` / `kernel_portraits.json` | 下述 §1/§2 的结构化结论 |

复现：
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate unsloth
# 1) 采 trace（不要加 -c cudaProfilerApi，本机死锁，见 §5）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /usr/local/cuda-12.8/bin/nsys profile -o decode_profile -f true --trace=cuda,nvtx \
  python -u 01_profile_target.py
# 2) 导 sqlite + stats
/usr/local/cuda-12.8/bin/nsys export --type sqlite --force-overwrite=true decode_profile.nsys-rep
/usr/local/cuda-12.8/bin/nsys stats --report cuda_gpu_kern_sum --format csv decode_profile.nsys-rep > kern_sum.csv
/usr/local/cuda-12.8/bin/nsys stats --report cuda_api_sum   --format csv decode_profile.nsys-rep > api_sum.csv
# 3) 建模 / 选型 / 出图（不需要 GPU）
python 02_comm_model.py && python 03_selection.py && python make_figures.py
```

## 1. nsys：一个 decode 步的解剖（Qwen2.5-0.5B，bf16，单流，KV cache）

64 个 NVTX range：[0]=prefill，[1:64]=decode（sqlite 里 `NVTX_EVENTS.textId` 要 JOIN
`StringIds` 取名）。**每步中位 980 个 kernel**（p10=980，极其稳定）。

| 指标 | prefill | decode（每步中位） |
|---|---|---|
| kernel 数 | 1003 | **980** |
| GPU 上 kernel 总时长 | 3009.5 µs | **2477.6 µs** |
| NVTX range 墙钟 | 294204 µs（含首次编译/加载） | **10928.1 µs** |
| GPU busy 占比 | — | **22.7%**（gap 8449 µs/步） |

**插账膨胀声明（诚实）**：nsys 下墙钟 10.9 ms，而第 6 章裸跑同模型同长度只有约
7.2 ms——插桩与 trace 落盘本身让每步膨胀约 50%。**因此本章只引用"结构性事实"**
（kernel 个数、分类占比、单 kernel GPU 时长、grid/block 形状），**绝对墙钟请以
ch06 裸跑为准**。GPU busy 22.7% 也是被插账放大的下限画面，但"上千个碎 kernel、
launch 间隙巨大"这一定性结论真实成立。

### 每步 GPU 时间分类（2477.6 µs 的去向）

| 类别 | kernel 数 | GPU 时长 µs | 占比 |
|---|---|---|---|
| GEMV/GEMM（cublas，QKV/O/gate/up/down/lm_head 投影） | 169 | 1171.4 | **47%** |
| 逐元素（silu/add/mul/激活） | 439 | 589.0 | 24% |
| copy/cat/index（DynamicCache 追加 KV） | 200 | 326.5 | 13% |
| norm（reduce/pow/rsqrt/mean，RMSNorm 族） | 148 | 221.4 | 9% |
| FlashAttention fwd | 24 | 170.3 | 7% |
| argmax/sample | （在 NVTX range 外执行） | — | — |

980 个 kernel 里只有 169 个在干"读权重的正事"，其余 800+ 个是几十微秒级的碎活；
439 个逐元素 kernel 只贡献 24% 时间却占了 45% 的 launch 次数——这就是 eager 模式
decode 的真实形状，也是 CUDA Graph / kernel fusion 存在的全部理由（第 12 章）。

### 全窗口 CUDA API（62,808 个 kernel）

| API | 次数 | 总计 ms | 单次中位 |
|---|---|---|---|
| `cudaLaunchKernel` | 62,639 | 282.9 | 2.24 µs |
| `cudaMemcpyAsync` | 4,955 | 93.7 | 3.24 µs |
| `cuLibraryLoadData` | 14 | 27.7 | 1865.9 µs（首次库加载） |
| `cudaDeviceSynchronize` | 128（=2×64 步） | 16.7 | 9.74 µs |
| `cudaMalloc` | 48 | 3.3 | 62.7 µs |

每步 2 次 `cudaDeviceSynchronize`（transformers generate 的逐 token 同步），每次
malloc 62.7 µs——这两项在手写引擎里分别被 CUDA Graph 和显存池消灭。

## 2. kernel 静态画像：decode 铺不满 114 个 SM 的铁证

直接从 sqlite 的 CUPTI kernel 事件读 grid/block/寄存器/shared memory（nsys 无需
ncu 权限即可拿到）：

| kernel | 每步次数 | 均时 µs | grid 形状（最大） | 寄存器/线程 | shared mem |
|---|---|---|---|---|---|
| FlashAttention fwd | 24 | 7.1 | **(1,1,2) = 2 block** | 255 | 48 KB |
| RMSNorm reduce(mean) | 49 | 2.1 | **(1,1,1) = 1 block** | 32 | 528 B |
| 逐元素(silu/add/mul/rsqrt) | 395 | 1.17 | (5,1,1) = 5 block | 22–32 | 0 |
| KV cat 拷贝 | 97 | 1.65 | (228,2,1)（样本多为 1–5 block） | 22–39 | 0 |
| cublas GEMV 投影 | 169 | 8.6 | **(18992,1,1)**（lm_head）；中间层 32–1216 | 162–168 | 272–528 B |

读法：FlashAttention decode 一个序列只有 2 个 query 头（paged/2 路 split），
grid=(1,1,2) 意味着 114 个 SM 只上 2 个；RMSNorm 对 896 维做 reduce 只用 1 个
block、128 线程；逐元素算子最多 5 个 block。**单流 decode 的 SM 利用率在结构上
就是个位数百分比**，这不是驱动没调好，是 batch=1 的数学形状。唯一铺得开的是
lm_head 的 18,992 个 block（词表 151,936 行），所以第 6 章观察到 lm_head 占比扎眼。
FA kernel 255 寄存器/线程 + 48KB shared mem 是它 occupancy 上不去的第二重原因。

## 3. 通信建模：α+β（假设全部列明，非多卡实测）

单次 collective 时间 `t = α + 流量 / (带宽 × 0.8效率)`；TP ring all-reduce 每卡
流量 `2(N-1)/N × payload`；decode 每层 2 次 AR。**α 是 NCCL launch+跳数的典型
拟合值（8–25 µs），不是本机实测**，落集群后必须用 `nccl-tests all_reduce_perf`
替换。带宽项：PCIe4 18.6 GB/s 单向为 ch07 本机实测；其余取公开规格的单向有效值。

| 链路 | 单向有效 GB/s | α µs | 来源 |
|---|---|---|---|
| PCIe4 x16 | 18.6 | 25 | ch07 实测 |
| PCIe5 x16 | 64 | 20 | PCIe-SIG 规格 128Gb/s 双向折半 |
| NVLink3（A100） | 300 | 10 | NVIDIA 规格 600GB/s 双向折半 |
| NVLink4（H100 SXM） | 450 | 8 | NVIDIA 规格 900GB/s 双向折半 |
| IB NDR 400Gb | 50 | 15 | IBTA：400Gb/s = 50 GB/s |

### 3.1 7B decode：TP 通信时间 / 单卡权重读取时间（>100% 即通信比算还慢）

| TP | NVLink4 | NVLink3 | IB NDR | PCIe5 | PCIe4 |
|---|---|---|---|---|---|
| 2 | **21.5%** | 26.9% | 40.7% | 54.0% | 68.3% |
| 4 | 43.0% | 53.8% | 81.8% | 108.3% | 137.9% |
| 8 | 86.1% | 107.8% | 164.2% | 217.0% | **277.0%** |

（参考：0.5B TP2 时因权重读取仅 0.545ms，即使 NVLink4 通信占比也有 70.5%、
PCIe4 221%——小模型做 TP 纯亏，别做。）

### 3.2 PP8 decode 全程边界传输（7B，d=3584，bf16，逐卡传隐状态）

每跳 7168 B ×(PP-1) 跳累计：NVLink4 **28.1 µs** / NVLink3 35.2 / IB 53.5 /
PCIe5 70.8 / PCIe4 90.2。对比 7B 每 token 墙钟 ~7000 µs，**最差链路也只占 ~1.3%**。
这就是"PP 能走以太网/IB、TP 不能"的量化根据：PP 的 payload 是一个隐状态向量，
TP 的 payload 量级相同但每层 2 次×28 层、且卡在每层计算的关键路径上。

### 3.3 prefill TP8 单层 all-reduce（带宽主导，payload∝S）

S=2048：NVLink4 0.079ms / IB 0.657 / PCIe4 1.751；S=128：0.012 / 0.055 / 0.133。
长上下文 prefill 时 TP 对带宽更敏感（第 11 章讲 CP/SP 的原因）。

## 4. ncu 权限坑（ERR_NVGPUCTRPERM）

`ncu` 一启动即报 `ERR_NVGPUCTRPERM`，根因：`/proc/driver/nvidia/params` 中
`RmProfilingAdminOnly: 1`，且本机无 sudo（`sudo -n` 要密码），沙箱外运行也被拒。
解法（按优先级）：① 让管理员写 `/etc/modprobe.d/nvidia.conf` 设
`NVreg_RestrictProfilingToAdminUsers=0` 后 reload 驱动；② 用有特权的容器
`--cap-add=SYS_ADMIN`；③ 云上选放行 profiling 的镜像/机型。
**替代分析路径（本章实际采用）**：nsys 的 sqlite 里已含每个 kernel 的
grid/block/registersPerThread/staticSharedMemory，配合 ch07 的带宽公式自算利用率，
大多数"铺不铺得满"的问题不需要 ncu。

## 5. 其他复现坑

- 模型只能用 **base** 版 `Qwen/Qwen2.5-0.5B`；Instruct 缓存里的权重被删过，
  加载报 `'NoneType' has no attribute 'endswith'`。transformers 4.56 用
  `dtype=` 而非 `torch_dtype=`。
- nsys **不要**用 `-c cudaProfilerApi` + `cudaProfilerStart/Stop`：本机 futex
  死锁 9 分钟；直接全窗 `--trace=cuda,nvtx` 即可。
- `nsys stats` 的 csv 前 3/7 行是头部，Med 列是字符串需 float()。

## 6. 选型速算器（`selection_results.json`）

卡库价格为 **2026-09/10 海外现货/云租中位时价，行情波动大，仅供教学**：
4090D 24G/918GB/s/$0.35h（PCIe4 无 NVLink）、L40S 48G/864/$0.79（PCIe4）、
A100-80G SXM/2039/$1.20（NVLink3 600）、H100-80G SXM/3350/$2.80（NVLink4 900）。

| 场景（ctx4096） | 配置 | 需显存 GiB | t_token 墙 ms | $/百万 tok |
|---|---|---|---|---|
| 7B-Q4 conc32（权重3.26+KV3.5） | 4090D ×1 | 8.26 | 3.81 | **0.15** |
| 同上 | L40S ×1 | 8.26 | 4.05 | 0.36 |
| 同上 | A100-80G ×1 | 8.26 | 1.72 | 0.23 |
| 同上 | H100-80G ×1 | 8.26 | 1.04 | 0.33 |
| 32B-bf16 conc16（59.6+32） | A100-80G ×2 TP | 93.1 | 15.69 | 8.47 |
| 同上 | H100-80G ×2 TP | 93.1 | 9.55 | 12.04 |
| 70B-Q4 conc16（32.6+16） | A100-80G ×1 | 50.1 | 17.17 | 4.63 |
| 同上 | H100-80G ×1 | 50.1 | 10.45 | 6.58 |

反直觉结论（正文展开）：7B-Q4 单流吃不满大卡，**按 token 成本算 4090D 最便宜**
（$0.15/M），H100 只在延迟敏感或要打满高并发时才值回票价。
**局限**：多卡墙钟只粗算 TP 通信（α+β），对无 NVLink 的卡（4090D/L40S）未做
外推；实际并发吞吐要用 vLLM 压测（第 14 章），本器只用于选型第一版估算。
