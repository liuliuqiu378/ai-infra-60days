"""03: Roofline 在本机坐实。
A. 大 GEMM：fp32(CUDA Core) / tf32 / bf16 / fp16(Tensor Core) 实测峰值
B. decode 形瘦 GEMM：Qwen2.5-0.5B 真实 lm_head 形状 (M,896)@(896,151936)，
   M=1..256，逐点算算术强度 AI 与实际落点
C. MLP 形 (M,896)@(896,4864) 作对照
"""
import json, torch

torch.backends.cuda.matmul.allow_tf32 = False
def time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

def gemm_tflops(dtype, S=8192, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(S, S, device="cuda", dtype=dtype)
    b = torch.randn(S, S, device="cuda", dtype=dtype)
    for _ in range(5): a@b
    torch.cuda.synchronize()
    t = time_ms(lambda: a@b, iters=30, warmup=5)
    return 2*S**3/t/1e9  # TFLOPS

peak = {
    "fp32_cudacore_8192": round(gemm_tflops(torch.float32), 1),
    "tf32_tensor_8192":  round(gemm_tflops(torch.float32, tf32=True), 1),
    "bf16_tensor_8192":  round(gemm_tflops(torch.bfloat16), 1),
    "fp16_tensor_8192":  round(gemm_tflops(torch.float16), 1),
}

def skinny(K, N, Ms, dtype=torch.bfloat16):
    w = torch.randn(K, N, device="cuda", dtype=dtype)
    out = []
    for M in Ms:
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        t = time_ms(lambda: x@w, iters=100, warmup=20)
        flop = 2*M*K*N
        nbytes = 2*(M*K + K*N + M*N)          # bf16：x + w + y
        out.append({"M": M, "ms": round(t,4),
                    "tflops": round(flop/t/1e9, 2),
                    "ai_flop_per_byte": round(flop/nbytes, 1),
                    "eff_gbs": round(nbytes/t/1e6, 1)})
    return out

Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
lmhead = skinny(896, 151936, Ms)      # 真实词表投影
mlp    = skinny(896, 4864, Ms)        # 真实 MLP 投影

res = {"peak_gemm_tflops": peak,
       "bw_roof_gbs": 918.0,         # 取 02 实测保守值
       "lmhead_M896_N151936_bf16": lmhead,
       "mlp_M896_N4864_bf16": mlp}
json.dump(res, open("gemm_results.json", "w"), indent=2)
print(json.dumps(peak, indent=2))
for r in lmhead: print(r)
