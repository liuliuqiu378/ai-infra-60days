#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 4 章 实验 1：KV Cache 显存账——公式、官方 StaticCache 真实分配、三巨头
=============================================================================
方法学（重要）：
  * 0.5B：完整加载权重，StaticCache 实测 vs 公式，并真实跑一次 prefill 量"激活"
  * 7B  ：权重 15.23 GB 取自官方 safetensors index 精确字节；
          KV 不下载权重也能"实测"——transformers 的 StaticCache 只依赖 config，
          它内部就是按真实形状 torch.zeros 出 K/V 张量，我们量显存 delta。
KV 字节公式（GQA/MQA 兼容；MHA 时 n_kv=n_heads）：
  bytes = 2(K,V) × L × n_kv × head_dim × seq × batch × bpe
运行：python 01_kv_accounting.py
"""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig, StaticCache

DEV = "cuda:0"
HERE = os.path.dirname(__file__)
M05 = "Qwen/Qwen2.5-0.5B"
CFG7 = "/home/hmn-cjy/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
W7_BYTES = 15_231_233_024          # 官方 model.safetensors.index.json metadata.total_size


def gib(x):
    return x / 1024**3


def per_tok_bytes(cfg, bpe=2):
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * d * bpe, d


def alloc_kv_delta(cfg, B, S, dtype=torch.bfloat16):
    """按 transformers 4.56 官方 StaticLayer.lazy_initialization 的精确布局在 GPU 上真实分配：
    每层 keys/values 各一个 zeros((B, n_kv, S, d))，共 2×L 个；量显存增量。"""
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    nkv = cfg.num_key_value_heads; L = cfg.num_hidden_layers
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    bufs = [torch.zeros((B, nkv, S, d), dtype=dtype, device=DEV)
            for _ in range(2 * L)]
    after = torch.cuda.memory_allocated()
    delta = after - before
    shape = tuple(bufs[0].shape)
    del bufs; torch.cuda.empty_cache()
    return delta, shape


def main():
    out = {}

    # ============ 0.5B 全量实测 ============
    cfg5 = AutoConfig.from_pretrained(M05)
    pt5, d5 = per_tok_bytes(cfg5)
    print(f"=== 0.5B（Qwen2.5-0.5B）L={cfg5.num_hidden_layers} "
          f"Q头={cfg5.num_attention_heads} KV头={cfg5.num_key_value_heads} d={d5} ===")
    print(f"每token KV = {pt5:,} 字节 = {pt5/1024:.1f} KiB")
    model = AutoModelForCausalLM.from_pretrained(M05, dtype=torch.bfloat16).to(DEV).eval()
    w5 = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"权重 {gib(w5):.3f} GB")

    f5, shape5 = alloc_kv_delta(cfg5, 1, 2048)
    formula5 = pt5 * 2048
    print(f"seq=2048 KV：公式 {gib(formula5):.4f} GB / StaticCache 实测 {gib(f5):.4f} GB，"
          f"形状 {shape5}，偏差 {(f5-formula5)/formula5*100:+.2f}%\n")

    # 真实 prefill 量激活
    S = 2048
    sc = StaticCache(config=cfg5, max_batch_size=1, max_cache_len=S, device=DEV, dtype=torch.bfloat16)
    ids = torch.randint(0, cfg5.vocab_size, (1, S), device=DEV)
    pos = torch.arange(S, device=DEV)
    m_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(ids, past_key_values=sc, use_cache=True, cache_position=pos)
    peak5 = torch.cuda.max_memory_allocated()
    # 官方 StaticCache 自己 lazy 分配出来的 KV 形状与大小（直接对真实张量求和，避开 mask 等干扰）
    real_shape = tuple(sc.layers[0].keys.shape)
    real_kv = sum(lay.keys.numel()*lay.keys.element_size()
                  + lay.values.numel()*lay.values.element_size() for lay in sc.layers)
    print(f"官方 StaticCache forward 后 lazy 分配：每层 {real_shape}，"
          f"KV张量合计 {gib(real_kv):.4f} GB（公式 {gib(f5):.4f} GB，"
          f"偏差 {(real_kv-formula5)/formula5*100:+.2f}%）")
    act5 = peak5 - w5 - f5
    print(f"0.5B prefill(seq=2048) 三巨头：权重 {gib(w5):.3f} + KV {gib(f5):.4f} "
          f"+ 激活 {gib(act5):.4f} = 峰值 {gib(peak5):.3f} GB\n")
    out["m05"] = dict(L=cfg5.num_hidden_layers, nh=cfg5.num_attention_heads,
                      nkv=cfg5.num_key_value_heads, d=d5, per_tok=pt5,
                      weight_gib=gib(w5), kv_formula_gib=gib(formula5),
                      kv_actual_gib=gib(f5), shape=list(shape5),
                      static_real_shape=list(real_shape), static_real_kv_gib=gib(real_kv),
                      act_gib=gib(act5), peak_gib=gib(peak5))
    del model, sc; torch.cuda.empty_cache()

    # ============ 7B：config + 官方 StaticCache 真实分配 ============
    cfg7 = AutoConfig.from_pretrained(CFG7)
    pt7, d7 = per_tok_bytes(cfg7)
    print(f"=== 7B（Qwen2.5-7B-Instruct）L={cfg7.num_hidden_layers} "
          f"Q头={cfg7.num_attention_heads} KV头={cfg7.num_key_value_heads} d={d7} ===")
    print(f"每token KV = 2×{cfg7.num_hidden_layers}×{cfg7.num_key_value_heads}×{d7}×2 "
          f"= {pt7:,} 字节 = {pt7/1024:.0f} KiB")
    mha_pt = 2*cfg7.num_hidden_layers*cfg7.num_attention_heads*d7*2
    print(f"若是 MHA(KV头={cfg7.num_attention_heads})：{mha_pt/1024:.0f} KiB/token，"
          f"GQA 把 KV 压到 1/{cfg7.num_attention_heads//cfg7.num_key_value_heads}")
    print(f"权重 bf16（官方index）= {W7_BYTES/1e9:.2f} GB\n")

    cases = [("batch=1 seq=2048", 1, 2048), ("batch=1 seq=32768", 1, 32768),
             ("batch=1 seq=131072", 1, 131072), ("batch=8 seq=8192", 8, 8192),
             ("batch=32 seq=8192", 32, 8192), ("batch=32 seq=32768", 32, 32768)]
    rows = []
    for name, B, S in cases:
        formula = pt7 * B * S
        try:
            delta, shape = alloc_kv_delta(cfg7, B, S)
            print(f"{name:22s}: KV 实测 {gib(delta):8.3f} GB (公式 {gib(formula):.3f})  "
                  f"占24G卡 {gib(delta)/23.5*100:5.1f}%")
            rows.append(dict(name=name, B=B, seq=S, kv_gib=gib(delta),
                             formula_gib=gib(formula), pct_of_24g=gib(delta)/23.5, oom=False))
        except torch.OutOfMemoryError:
            print(f"{name:22s}: 需要 {gib(formula):.1f} GB KV —— 单卡 24G 直接 OOM（这本身就是结论）")
            rows.append(dict(name=name, B=B, seq=S, kv_gib=None,
                             formula_gib=gib(formula), pct_of_24g=None, oom=True))
        torch.cuda.empty_cache()

    # KV int8 同形状真实分配（dtype=float16 无法 int8，改用 torch.int8 直接测）
    b, s = 1, 32768
    before = torch.cuda.memory_allocated()
    t = torch.zeros(2*cfg7.num_hidden_layers, b, cfg7.num_key_value_heads, s, d7,
                    dtype=torch.int8, device=DEV)
    i8 = torch.cuda.memory_allocated() - before
    print(f"\nKV int8 量化 batch=1 seq=32768：{gib(i8):.3f} GB（bf16 的一半）")
    del t; torch.cuda.empty_cache()

    out["m7"] = dict(L=cfg7.num_hidden_layers, nh=cfg7.num_attention_heads,
                     nkv=cfg7.num_key_value_heads, d=d7, per_tok=pt7,
                     mha_per_tok=mha_pt, weight_gib=W7_BYTES/1024**3,
                     cases=rows, kv_i8_32k_gib=gib(i8))
    with open(os.path.join(HERE, "kv_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n已保存 kv_results.json")


if __name__ == "__main__":
    main()
