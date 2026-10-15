#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 6 章：手写 generate —— 100 行看穿文本生成
=============================================
三个版本，同一模型：
  1. naive_generate ：不用 KV Cache，每步把整段历史重算（第 4 章的 O(n²) 现场）
  2. kv_generate    ：用 DynamicCache，每步只喂 1 个新 token（引擎真实做法）
  3. kv_stream      ：kv 版的流式形态（yield 每个 token，对应 SSE 流式 API）
外加：
  - sample_token：temperature / top-k / top-p（纯 torch 实现，可独立单测）
  - batched_generate：一次批量生成 B 个请求（continuous batching 的静态原型）
对齐目标：greedy 下与 model.generate() 的 token 序列、逐步 logits 完全一致。
"""
import torch
import torch.nn.functional as F
from transformers import DynamicCache


# ---------------- 采样：从 logits 到 1 个 token ----------------
def sample_token(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """logits: (V,) 或 (B,V)。返回选中的 token id。
    temperature=0 走贪心；top_k=0 关闭；top_p=1.0 关闭核采样。"""
    if temperature == 0.0:
        return logits.argmax(-1, keepdim=True)          # (B,1) 贪心
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth,
                             torch.full_like(logits, float("-inf")), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # 保留"使累计概率首次超过 top_p"的那个 token（含边界），其余置 -inf
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)  # (B,1)


# ---------------- 版本 1：无 cache，每步重算 ----------------
@torch.inference_mode()
def naive_generate(model, ids, max_new_tokens=64, greedy=True):
    """ids: (1,S)。每步把 ids[:,:t] 整段重喂。返回 (token id 列表, 每步ms)。"""
    import time
    gen, step_ms = [], []
    for _ in range(max_new_tokens):
        torch.cuda.synchronize(); t0 = time.time()
        out = model(ids, use_cache=False)
        torch.cuda.synchronize(); step_ms.append((time.time()-t0)*1000)
        nxt = out.logits[:, -1].argmax(-1) if greedy else \
              sample_token(out.logits[:, -1], temperature=1.0)[0]
        gen.append(nxt.item())
        ids = torch.cat([ids, nxt[None]], dim=1)
        if nxt.item() == getattr(model.config, "eos_token_id", None):
            break
    return gen, step_ms


# ---------------- 版本 2：KV Cache ----------------
@torch.inference_mode()
def kv_generate(model, ids, max_new_tokens=64, greedy=True,
                temperature=1.0, top_k=0, top_p=1.0, trace_logits=False):
    """ids: (B,S)。prefill 一次，之后每步 1 token。
    trace_logits=True 时额外返回每步最后位置 logits（供对齐用）。"""
    import time
    cache = DynamicCache()
    out = model(ids, past_key_values=cache, use_cache=True)
    nxt = (out.logits[:, -1].argmax(-1, keepdim=True) if greedy else
           sample_token(out.logits[:, -1], temperature, top_k, top_p))
    gens = [[t.item()] for t in nxt]
    step_ms, traced = [], []
    alive = [True] * ids.shape[0]
    for _ in range(max_new_tokens - 1):
        torch.cuda.synchronize(); t0 = time.time()
        out = model(nxt, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(); step_ms.append((time.time()-t0)*1000)
        if trace_logits:
            traced.append(out.logits[:, -1].float().cpu())
        nxt = (out.logits[:, -1].argmax(-1, keepdim=True) if greedy else
               sample_token(out.logits[:, -1], temperature, top_k, top_p))
        for i, t in enumerate(nxt):
            if alive[i]:
                gens[i].append(t.item())
        if all(t.item() == model.config.eos_token_id for t in nxt):
            break
    return gens, step_ms, (torch.cat(traced) if trace_logits else None)


# ---------------- 版本 3：流式 ----------------
@torch.inference_mode()
def kv_stream(model, ids, max_new_tokens=64, **sample_kw):
    """yield 每个新 token id（单条请求）。生产 SSE 服务的最小内核。
    sample_kw 透传给 sample_token（temperature/top_k/top_p）。"""
    cache = DynamicCache()
    out = model(ids, past_key_values=cache, use_cache=True)
    nxt = sample_token(out.logits[:, -1], **sample_kw)
    yield nxt.item()
    for _ in range(max_new_tokens - 1):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = sample_token(out.logits[:, -1], **sample_kw)
        yield nxt.item()
        if nxt.item() == model.config.eos_token_id:
            break


# ---------------- 批量生成（静态 batch 原型） ----------------
@torch.inference_mode()
def batched_generate(model, tokenizer, prompts, max_new_tokens=32, pad_token_id=None):
    """多条等长/不等长 prompt 左 padding 后一次 prefill、一起 decode。
    返回每 prompt 的 token id 列表。"""
    pad = pad_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id
    tok = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    # 因果模型左 padding 才能保证最后一个位置对齐各自 prompt 结尾
    input_ids, attn = tok.input_ids, tok.attention_mask
    cache = DynamicCache()
    out = model(input_ids, attention_mask=attn, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gens = [[t.item()] for t in nxt]
    attn = torch.cat([attn, torch.ones(attn.shape[0], 1, device=attn.device,
                                       dtype=attn.dtype)], dim=1)
    for _ in range(max_new_tokens - 1):
        out = model(nxt, attention_mask=attn, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for i, t in enumerate(nxt):
            gens[i].append(t.item())
        attn = torch.cat([attn, torch.ones(attn.shape[0], 1, device=attn.device,
                                           dtype=attn.dtype)], dim=1)
    return gens


if __name__ == "__main__":
    # 冒烟测试，正式实验在 01_benchmark.py / 02_alignment.py
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
                                             dtype=torch.bfloat16).to("cuda:0").eval()
    ids = tok("用一句话解释 KV Cache：", return_tensors="pt").input_ids.to("cuda:0")
    g, _, _ = kv_generate(m, ids, max_new_tokens=40)
    print("kv greedy:", tok.decode(g[0], skip_special_tokens=True))
    print("stream:", "".join(
        tok.decode([t]) for t in kv_stream(m, ids, max_new_tokens=20,
                                          temperature=0.8, top_p=0.95)))
