#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 6 章 补充实验：长 prompt（2048）下 naive 重算爬升 vs KV 恒定。
短 prompt 时两者都在第 4 章发现的 ~14ms 固定平台，形状差异被掩盖；
把起点推到 S=2048（线性区），naive 每步重算越来越长的序列，爬升才显形。"""
import json, os, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_generate import kv_generate

DEV = "cuda:0"
HERE = os.path.dirname(__file__)


def timed_naive_steps(model, ids, n_new):
    """每步重算 ids 全序列，记录逐步 ms"""
    steps = []
    with torch.inference_mode():
        for _ in range(n_new):
            torch.cuda.synchronize(); t0 = time.time()
            out = model(ids, use_cache=False)
            torch.cuda.synchronize(); steps.append((time.time()-t0)*1000)
            nxt = out.logits[:, -1:].argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
    return steps


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16).to(DEV).eval()
    out = {"settings": []}
    for S0, N in ((512, 128), (2048, 128)):
        ids = torch.randint(0, model.config.vocab_size, (1, S0), device=DEV)
        # 热身
        with torch.inference_mode():
            model(ids[:, :64], use_cache=False)
        naive = timed_naive_steps(model, ids, N)
        _, kv_steps, _ = kv_generate(model, ids, max_new_tokens=N)
        tot_n, tot_k = sum(naive), sum(kv_steps)
        print(f"S0={S0:4d}, 生成{N}: naive {tot_n:7.0f}ms (首{naive[0]:.1f}→末{naive[-1]:.1f}) | "
              f"KV {tot_k:6.0f}ms (中位{statistics.median(kv_steps):.1f}) | 总时 {tot_n/tot_k:.1f}×")
        out["settings"].append(dict(S0=S0, N=N, naive_steps=naive, kv_steps=kv_steps,
                                    naive_total=tot_n, kv_total=tot_k))
    json.dump(out, open(os.path.join(HERE, "longctx_results.json"), "w"), indent=2)
    print("saved longctx_results.json")


if __name__ == "__main__":
    main()
