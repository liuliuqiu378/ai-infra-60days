#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 6 章 性能实测：naive(重算) vs KV(手写) vs HF generate
========================================================
A. 三个版本在 32/64/128 生成 token 下的总时、每 token 耗时
B. naive 逐步耗时爬升 vs KV 逐步耗时水平（128 步全量记录）
C. KV 版批量生成 B=1/2/4/8/16：单请求变慢 vs 聚合 tok/s
D. 流式 TTFT：prefill 结束到吐出第 1 个 token 的端到端首字时间
运行：python 02_benchmark.py
"""
import json
import os
import statistics
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_generate import naive_generate, kv_generate, kv_stream

MODEL, DEV = "Qwen/Qwen2.5-0.5B", "cuda:0"
HERE = os.path.dirname(__file__)


def sync_time(fn):
    torch.cuda.synchronize(); t0 = time.time()
    r = fn(); torch.cuda.synchronize()
    return (time.time()-t0)*1000, r


def hf_generate_time(model, ids, n):
    def f():
        return model.generate(ids, max_new_tokens=n, do_sample=False,
                              pad_token_id=model.config.eos_token_id,
                              use_cache=True)
    ms, out = sync_time(f)
    return ms, out.shape[1]-ids.shape[1]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    ids = tok("请用三句话介绍一下长城：", return_tensors="pt").input_ids.to(DEV)
    res = {}

    # 热身
    kv_generate(model, ids, max_new_tokens=8)
    model.generate(ids, max_new_tokens=8, do_sample=False,
                   pad_token_id=model.config.eos_token_id)

    # ---- A. 总时对比（每个 n 跑 3 次取中位）----
    print("=== A. 总时对比（ms，3 次中位）===")
    rows = []
    for n in (32, 64, 128):
        ms_n = statistics.median([
            sync_time(lambda: naive_generate(model, ids, n))[0] for _ in range(3)])
        ms_k, _ = sync_time(lambda: kv_generate(model, ids, n))
        ms_k = statistics.median([
            sync_time(lambda: kv_generate(model, ids, n))[0] for _ in range(3)])
        ms_h = statistics.median([hf_generate_time(model, ids, n)[0] for _ in range(3)])
        rows.append(dict(n=n, naive_ms=ms_n, kv_ms=ms_k, hf_ms=ms_h,
                         naive_per_tok=ms_n/n, kv_per_tok=ms_k/n, hf_per_tok=ms_h/n))
        print(f"n={n:3d}: naive {ms_n:8.0f} ({ms_n/n:5.1f}/tok) | "
              f"手写KV {ms_k:7.0f} ({ms_k/n:5.1f}/tok) | HF {ms_h:7.0f} ({ms_h/n:5.1f}/tok) | "
              f"naive/KV {ms_n/ms_k:5.1f}×")
    res["A_total"] = rows

    # ---- B. 逐步耗时形状（128 步，只跑 1 次拿形状）----
    print("\n=== B. 128 步逐步耗时 ===")
    _, steps_n = naive_generate(model, ids, 128)
    _, steps_k, _ = kv_generate(model, ids, 128)
    # naive 第一步含 prompt 长度 ~10；两端统计
    res["B_steps"] = dict(naive=steps_n, kv=steps_k,
                          naive_first=steps_n[0], naive_last=steps_n[-1],
                          kv_median=statistics.median(steps_k))
    print(f"naive 首步 {steps_n[0]:.2f} → 末步 {steps_n[-1]:.2f} ms；"
          f"KV 中位 {statistics.median(steps_k):.2f} ms，"
          f"末段(最后16步均值) {statistics.mean(steps_n[-16:]):.2f} vs KV "
          f"{statistics.mean(steps_k[-16:]):.2f} ms")

    # ---- C. batch 扩展（相同 prompt 复制 B 份，等长 → 无需 mask）----
    print("\n=== C. KV 批量生成（96 token）===")
    brows = []
    base = ids
    for B in (1, 2, 4, 8, 16):
        bid = base.repeat(B, 1)
        ms_list = [sync_time(lambda: kv_generate(model, bid, 96))[0] for _ in range(3)]
        ms = statistics.median(ms_list)
        agg = B*96/(ms/1000)
        brows.append(dict(B=B, batch_ms=ms, per_req_ms=ms, agg_tok_s=agg,
                         per_tok_ms=ms/96))
        print(f"B={B:2d}: 总时 {ms:7.0f} ms | 单请求 {ms:7.0f} ms | "
              f"聚合 {agg:6.0f} tok/s | 每token {ms/96:5.2f} ms")
    res["C_batch"] = brows

    # ---- D. 流式 TTFT（prefill + 采样到第一个 token）----
    print("\n=== D. 流式首 token 延迟（prompt 长度扫描）===")
    drows = []
    for S in (16, 64, 256, 1024):
        pids = torch.randint(0, model.config.vocab_size, (1, S), device=DEV)
        ts = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.time()
            it = kv_stream(model, pids, max_new_tokens=4, temperature=0.0)
            first = next(it)
            torch.cuda.synchronize(); ts.append((time.time()-t0)*1000)
        med = statistics.median(ts[1:])
        drows.append(dict(S=S, ttft_ms=med))
        print(f"S={S:5d}: TTFT {med:7.1f} ms")
    res["D_ttft"] = drows

    json.dump(res, open(os.path.join(HERE, "bench_results.json"), "w"), indent=2)
    print("\nsaved bench_results.json")


if __name__ == "__main__":
    main()
