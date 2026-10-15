"""01: 把 RTX 4090 D 的"身份证"与理论上限全部落盘。
区分【实测】(torch/nvidia-smi) 与【规格】(AD102 公开规格)。"""
import json, subprocess, torch

p = torch.cuda.get_device_properties(0)

def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out)

gmax_mhz   = smi("clocks.max.graphics")   # 3105 MHz（max boost）
mem_mhz    = smi("clocks.max.memory")     # 10501 MHz（DDR 时钟，×2 为数据率）
# 【规格】AD102/4090 D 显存位宽 384-bit；每 SM 128 个 FP32 CUDA Core
BUS_BITS, CORE_PER_SM = 384, 128
cuda_cores = p.multi_processor_count * CORE_PER_SM
# 【规格】GDDR6X：数据率 = 2× 时钟；带宽 = 数据率 × 位宽/8
bw_spec = 2 * mem_mhz*1e6 * BUS_BITS / 8 / 1e9          # GB/s
fp32_spec_maxboost = cuda_cores * 2 * gmax_mhz*1e6 / 1e12  # TFLOPS（max boost 瞬时）
# 持续 boost 取 2520 MHz（4090 D 典型游戏/计算负载稳态，【规格·典型值】）
fp32_spec_sustained = cuda_cores * 2 * 2520e6 / 1e12
ridge_spec = fp32_spec_sustained*1e3 / bw_spec          # FLOP/byte（FP32/CUDA core 口径）

info = {
    "device": p.name, "compute_capability": f"{p.major}.{p.minor}",
    "sm_count_measured": p.multi_processor_count,
    "l2_mb_measured": round(p.L2_cache_size/2**20, 1),
    "hbm_gib_measured": round(p.total_memory/2**30, 1),
    "cuda_cores_spec": cuda_cores,
    "graphics_max_boost_mhz_measured": gmax_mhz,
    "mem_clock_mhz_measured": mem_mhz,
    "bus_bits_spec": BUS_BITS,
    "bw_spec_gbs": round(bw_spec, 1),
    "fp32_tflops_spec_at_maxboost": round(fp32_spec_maxboost, 1),
    "fp32_tflops_spec_sustained_2520mhz": round(fp32_spec_sustained, 1),
    "ridge_fp32_spec_flop_per_byte": round(ridge_spec, 1),
    "threads_per_warp": 32,
    "note": "BF16 Tensor Core 实测峰值见 03_gemm_roofline.json；规格表请联网核实 Ada 白皮书"
}
print(json.dumps(info, indent=2, ensure_ascii=False))
json.dump(info, open("gpu_specs.json", "w"), indent=2, ensure_ascii=False)
