#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 4 章 实验 2：没有 KV Cache 会怎样？——O(n²) 之痛（严谨版）
================================================================
每一步只做一次 forward（计时+推进用同一次输出），记录每步耗时：
  有 KV Cache：第 t 步处理 1 个新 token，逐步耗时恒定，处理 token 总数 ∝ n
  无 KV Cache：第 t 步重算全部 t 个历史 token，逐步耗时线性爬升，处理量 ∝ n²
对 n=64/128/256 分别测量，看墙钟倍数随 n 怎么涨。
注意一个反直觉但真实的点：墙钟倍数远小于 FLOPs 倍数——
因为重算走的是高效 prefill（算力-bound，8万tok/s），而 KV decode 走低效访存（百级tok/s）。
运行：python 02_with_without_cache.py
"""
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoConfig

MODEL = "Qwen/Qwen2.5-0.5B"
DEV = "cuda:0"
PREFILL = 32
N_LIST = (64, 128, 256)


def timed(fn):
    torch.cuda.synchronize(); s = time.time()
    r = fn()
    torch.cuda.synchronize(); return (time.time()-s)*1000, r


def run_with_cache(model, seed, n):
    with torch.inference_mode():
        _, o = timed(lambda: model(seed, use_cache=True))
        pkv, x = o.past_key_values, o.logits[:, -1:].argmax(-1)
        steps, cum = [], 0.0
        for _ in range(n):
            ms, o = timed(lambda: model(x, past_key_values=pkv, use_cache=True))
            pkv, x = o.past_key_values, o.logits[:, -1:].argmax(-1)
            steps.append(ms); cum += ms
    tokens = PREFILL + n
    return steps, cum, tokens, None


def run_no_cache(model, seed, n):
    hist = seed
    steps, cum, seqs = [], 0.0, []
    with torch.inference_mode():
        for _ in range(n):
            L = hist.shape[1]
            ms, o = timed(lambda: model(hist, use_cache=False))
            hist = torch.cat([hist, o.logits[:, -1:].argmax(-1)], dim=1)
            steps.append(ms); cum += ms; seqs.append(L)
    tokens = PREFILL + sum(seqs)
    return steps, cum, tokens, seqs


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    Np = sum(p.numel() for p in model.parameters())
    torch.manual_seed(0)
    seed = torch.randint(0, cfg.vocab_size, (1, PREFILL), device=DEV)
    summary = []
    detail = {}
    for n in N_LIST:
        on, ton, tok_on, _ = run_with_cache(model, seed, n)
        torch.cuda.empty_cache()
        off, toff, tok_off, seqs = run_no_cache(model, seed, n)
        torch.cuda.empty_cache()
        flop_ratio = tok_off / tok_on
        print(f"n={n:3d}: 有cache {ton:8.0f}ms(处理{tok_on:5d}tok) | "
              f"无cache {toff:9.0f}ms(处理{tok_off:7d}tok) | "
              f"墙钟 {toff/ton:5.2f}× | 计算量(FLOPs) {flop_ratio:6.1f}×")
        summary.append(dict(n=n, ms_on=ton, ms_off=toff, tokens_on=tok_on,
                            tokens_off=tok_off, wall_ratio=toff/ton,
                            flop_ratio=flop_ratio))
        if n == 256:
            detail = dict(steps_on=on, steps_off=off, seq_lens=seqs)
    with open(os.path.join(os.path.dirname(__file__), "nocache_results.json"), "w") as f:
        json.dump(dict(summary=summary, detail256=detail, n_params=Np), f, indent=2)
    print("已保存 nocache_results.json")


if __name__ == "__main__":
    main()
