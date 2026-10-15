#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 6 章 对齐验证：手写 kv_generate vs HF 官方 model.generate()
=============================================================
三道证据：
  A. greedy token 序列逐 token 完全相同
  B. 每一步最后位置的 logits 与 HF generate(return scores) 逐元素 allclose
  C. 采样器单测：greedy / top_k=1 等价 / top_p 截断正确性 / 固定种子可复现
运行：python 01_alignment.py
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_generate import kv_generate, sample_token

MODEL, DEV = "Qwen/Qwen2.5-0.5B", "cuda:0"
HERE = os.path.dirname(__file__)
PROMPTS = [
    "用一句话解释 KV Cache：",
    "Python 里列表和元组最大的区别是",
    "写一首关于 GPU 的五言绝句：",
]


def part_ab(model, tok):
    results = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        N = 32
        # 我方：KV 版，trace 每步 decode 的最后位置 logits
        mine, _, traced = kv_generate(model, ids, max_new_tokens=N,
                                      greedy=True, trace_logits=True)
        # HF：greedy + 返回每步 scores
        out = model.generate(ids, max_new_tokens=N, do_sample=False,
                             return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tok.eos_token_id)
        hf_ids = out.sequences[0, ids.shape[1]:].tolist()
        hf_scores = torch.stack(out.scores, dim=0)[:,0].float().cpu()  # (N,V)
        # hf_scores[i] 是"第 i 个生成 token"的 logits；
        # 我方 traced[i] 是喂入第 i 个 token 后得到的 logits => 对应 hf_scores[i+1]
        match_tokens = mine[0][:len(hf_ids)] == hf_ids[:len(mine[0])]
        cmp_mine, cmp_hf = traced[:N-1], hf_scores[1:N]
        maxdiff = (cmp_mine - cmp_hf).abs().max().item()
        meandiff = (cmp_mine - cmp_hf).abs().mean().item()
        results.append(dict(prompt=p, n_compared=min(len(mine[0]), len(hf_ids)),
                            tokens_identical=bool(match_tokens),
                            logits_max_absdiff=maxdiff, logits_mean_absdiff=meandiff,
                            mine_text=tok.decode(mine[0], skip_special_tokens=True)[:60]))
        print(f"「{p}」 token全同={bool(match_tokens)}  "
              f"logits 最大差 {maxdiff:.2e} 均值差 {meandiff:.2e}")
    return results


def part_c():
    """采样器纯 CPU 单测（构造分布，不依赖模型）"""
    cases = []
    # C1: temperature=0 => argmax
    lg = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    assert sample_token(lg, temperature=0).item() == 1
    cases.append("temperature=0 贪心选 argmax ✓")
    # C2: top_k=1 等价贪心
    assert sample_token(lg, temperature=1.0, top_k=1).item() == 1
    cases.append("top_k=1 等价贪心 ✓")
    # C3: top_p 截断：极端分布 [10,0,0,0]，top_p=0.5 只能选到 token0
    lg2 = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    draws = torch.cat([sample_token(lg2, temperature=1.0, top_p=0.5, generator=g)
                       for _ in range(50)])
    assert (draws == 0).all(), draws.unique()
    cases.append("top_p=0.5 极端分布 50 次采样全部落在头部 ✓")
    # C4: top_p=0.9 + 多峰分布，采样只会落在累计质量覆盖 0.9 的前缀
    lg3 = torch.tensor([[3.0, 2.5, 2.0, 0.1, -5.0]])
    # 手算 softmax 累计质量
    pr = torch.softmax(lg3, -1)
    cum = pr.cumsum(-1)
    # top_p=0.9 应保留前 3 个（前2累计 <0.9 时第3个含边界）；后 2 个概率须为 0
    kept = (pr[cum - pr <= 0.9]).numel()  # 等价实现的边界检验
    g = torch.Generator().manual_seed(1)
    draws = torch.cat([sample_token(lg3, temperature=1.0, top_p=0.9, generator=g)
                       for _ in range(200)])
    assert draws.max() <= kept - 1, (draws.unique(), kept)
    cases.append(f"top_p=0.9 200 次采样只落在保留前缀（{kept} 个候选）✓")
    # C5: 固定种子可复现
    def draw(seed):
        gg = torch.Generator().manual_seed(seed)
        return [sample_token(lg3, temperature=0.8, top_k=3, generator=gg).item()
                for _ in range(20)]
    assert draw(42) == draw(42)
    cases.append("固定 generator 种子两次采样序列完全一致 ✓")
    for c in cases:
        print(c)
    return cases


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    print("=== A/B: 与 HF generate 对齐（3 个 prompt × 32 token）===")
    ab = part_ab(model, tok)
    print("\n=== C: 采样器单测 ===")
    cc = part_c()
    json.dump(dict(alignment=ab, sampler_tests=cc),
              open(os.path.join(HERE, "alignment_results.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\nsaved alignment_results.json")
