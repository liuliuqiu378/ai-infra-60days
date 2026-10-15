#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 3 章 核心实验：prefill 与 decode 分别在"饿"什么资源？
=====================================================================
硬件：RTX 4090 D（BF16 dense 标称 82.6 TFLOPS，显存带宽 1008 GB/s）
模型：Qwen2.5-0.5B，bf16（N≈0.494B，权重约 0.99 GB）

四组测量：
  A. prefill：一次喂入 S 个 token，achieved TFLOPS（FLOPs≈2·N·S，忽略 attn 的 O(S^2)）
  B. decode eager（batch=1）：朴素逐 token 前向，反推有效带宽
  C. decode + CUDA Graph（batch=1）：同一份 kernel，只消除 launch 开销，看带宽差多少
  D. decode batch 扩展：B=1/4/8/16/32，聚合 tok/s 是否近线性（batching 的物理地基）

运行：python 01_prefill_decode_probe.py（结果存 probe_results.json）
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoConfig, StaticCache

MODEL = "Qwen/Qwen2.5-0.5B"
DTYPE = torch.bfloat16
DEV = "cuda:0"
PEAK_TFLOPS = 146.0  # 4090D dense bf16 Tensor Core 实测峰值(标称约147；82.6是FP32指标，勿混用)
BW_GBS = 920.0   # 大张量copy实测；标称1008(实测约91%)


def cuda_time_ms(fn, iters=10, warmup=3):
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
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
    n_params = sum(p.numel() for p in model.parameters())
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ai_star = PEAK_TFLOPS * 1e12 / (BW_GBS * 1e9)
    print(f"模型 {MODEL}: N={n_params/1e9:.3f}B, 权重 {w_bytes/1e9:.3f} GB")
    print(f"硬件: {PEAK_TFLOPS} TFLOPS, {BW_GBS:.0f} GB/s, Roofline 拐点 AI*={ai_star:.1f} FLOP/byte")
    print(f"decode(b=1) 理论上限(纯搬权重) ≈ {BW_GBS*1e9/w_bytes:,.0f} tok/s\n")
    results = {"n_params_B": n_params/1e9, "weight_GB": w_bytes/1e9}

    # ---------- A. prefill ----------
    print("=== A. prefill：一次前向 S 个 token（AI≈S FLOP/byte）===")
    prefill = []
    for S in (128, 512, 2048, 4096):
        ids = torch.randint(0, cfg.vocab_size, (1, S), device=DEV)
        ms = cuda_time_ms(lambda: model(ids, use_cache=False), iters=5, warmup=2)
        tflops = 2 * n_params * S / (ms/1000) / 1e12
        print(f"S={S:5d}: {ms:7.2f} ms  均摊 {ms/S:.3f} ms/token  "
              f"{tflops:6.1f} TFLOPS ({tflops/PEAK_TFLOPS*100:5.1f}% 峰值)  AI≈{S}")
        prefill.append(dict(S=S, ms=ms, ms_per_tok=ms/S, tflops=tflops, ai=S))
    results["prefill"] = prefill

    S = 2048
    ids = torch.randint(0, cfg.vocab_size, (1, S), device=DEV)

    # ---------- B. decode eager ----------
    print("\n=== B. decode eager：朴素实现，每步几百个小 kernel 各自启动 ===")
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        pkv, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        ms_e = cuda_time_ms(lambda: model(nxt, past_key_values=pkv, use_cache=True),
                            iters=30, warmup=5)
    bw_e = w_bytes / (ms_e/1000) / 1e9
    print(f"每步 {ms_e:.3f} ms → {1000/ms_e:,.0f} tok/s，有效带宽 {bw_e:,.0f} GB/s "
          f"({bw_e/BW_GBS*100:.0f}% 峰值)")
    results["decode_eager"] = dict(ms=ms_e, toks_s=1000/ms_e, eff_bw_gbs=bw_e)
    del pkv, out
    torch.cuda.empty_cache()

    # ---------- C. decode + CUDA Graph ----------
    print("\n=== C. decode + CUDA Graph：同一份 kernel，只消除 launch 开销 ===")
    sc = StaticCache(config=cfg, max_batch_size=1, max_cache_len=S+4, device=DEV, dtype=DTYPE)
    with torch.inference_mode():
        model(ids, past_key_values=sc, use_cache=True,
              cache_position=torch.arange(S, device=DEV))
    x = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    pos = torch.tensor([S], device=DEV)
    def one_step():
        with torch.inference_mode():
            model(x, past_key_values=sc, use_cache=True, cache_position=pos)
    for _ in range(3):
        one_step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()
    ms_g = cuda_time_ms(g.replay, iters=100, warmup=10)
    bw_g = w_bytes / (ms_g/1000) / 1e9
    print(f"每步 {ms_g:.3f} ms → {1000/ms_g:,.0f} tok/s，有效带宽 {bw_g:,.0f} GB/s "
          f"({bw_g/BW_GBS*100:.0f}% 峰值)")
    print(f"→ 仅消除 launch 开销，速度提升 {ms_e/ms_g:.1f}×")
    results["decode_graph"] = dict(ms=ms_g, toks_s=1000/ms_g, eff_bw_gbs=bw_g,
                                   speedup=ms_e/ms_g)
    del sc
    torch.cuda.empty_cache()

    # ---------- D. decode batch 扩展（S 缩短到 1024 控制 KV 显存）---------
    print("\n=== D. decode batch 扩展：同一批权重服务 B 个请求 ===")
    Sb = 1024
    batches = []
    for B in (1, 4, 8, 16, 32):
        ids = torch.randint(0, cfg.vocab_size, (B, Sb), device=DEV)
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            ms = cuda_time_ms(lambda: model(nxt, past_key_values=pkv, use_cache=True),
                              iters=20, warmup=3)
        del pkv, out
        torch.cuda.empty_cache()
        print(f"B={B:2d}: 每步 {ms:6.2f} ms  每请求 {1000/ms:6.1f} tok/s  "
              f"聚合 {B*1000/ms:7,.0f} tok/s")
        batches.append(dict(B=B, ms=ms, per_req=1000/ms, agg=B*1000/ms))
    results["decode_batch"] = batches

    with open(os.path.join(os.path.dirname(__file__), "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n已保存 probe_results.json")


if __name__ == "__main__":
    main()
