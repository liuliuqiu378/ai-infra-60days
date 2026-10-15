#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 3 章 实验 2：亲手量出这张 4090 D 的两堵"墙"
  1) 算力墙：大矩阵乘 (bf16, Tensor Core) 峰值 TFLOPS
  2) 带宽墙：大张量 copy 的有效 GB/s（读1份+写1份=2·N 字节）
量完就能在 Roofline 图上给 prefill / decode 定位。
运行：python 02_roofline_microbench.py
"""
import json
import os
import torch

DEV = "cuda:0"
DT = torch.bfloat16


def time_ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    # ---- 算力墙：不同尺寸的 GEMM ----
    print("=== 算力墙：C=A·B (bf16) ===")
    gemm = []
    for K in (1024, 2048, 4096, 8192, 12288):
        M = N = K
        a = torch.randn(M, K, device=DEV, dtype=DT)
        b = torch.randn(K, N, device=DEV, dtype=DT)
        ms = time_ms(lambda: torch.matmul(a, b), iters=20, warmup=5)
        tflops = 2 * M * N * K / (ms/1000) / 1e12
        print(f"{K:5d}³: {ms:7.2f} ms  {tflops:6.1f} TFLOPS")
        gemm.append(dict(K=K, ms=ms, tflops=tflops))
        del a, b
    peak = max(g["tflops"] for g in gemm)

    # ---- 带宽墙：大张量 copy（1GB+, 读1写1）----
    print("\n=== 带宽墙：大张量 copy（读1份+写1份）===")
    copy_bench = []
    for GB in (0.25, 1, 4):
        n = int(GB * 1e9 / 2)  # bf16 2 bytes
        a = torch.randn(n, device=DEV, dtype=DT)
        c = torch.empty_like(a)
        ms = time_ms(lambda: c.copy_(a), iters=50, warmup=10)
        gbs = 2 * a.numel() * 2 / (ms/1000) / 1e9
        print(f"{GB:4.2f} GB 张量: {ms:6.3f} ms  {gbs:7.0f} GB/s")
        copy_bench.append(dict(GB=GB, ms=ms, gbs=gbs))
        del a, c
    bw = max(b["gbs"] for b in copy_bench)

    print(f"\n实测屋顶：算力 {peak:.1f} TFLOPS，带宽 {bw:.0f} GB/s")
    print(f"实测拐点 AI* = 算力/带宽 = {peak*1e12/(bw*1e9):.1f} FLOP/byte")
    with open(os.path.join(os.path.dirname(__file__), "roofline_results.json"), "w") as f:
        json.dump(dict(gemm=gemm, copy=copy_bench, peak_tflops=peak, bw_gbs=bw,
                       ai_star=peak*1e12/(bw*1e9)), f, indent=2)


if __name__ == "__main__":
    main()
