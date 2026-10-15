#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 6 章配图：速度对比+逐步形状 / batch 扩展 / 正确性证据"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "images", "figures", "ch06")
os.makedirs(OUT, exist_ok=True)
B = json.load(open(os.path.join(HERE, "bench_results.json")))
LC = json.load(open(os.path.join(HERE, "longctx_results.json")))
AL = json.load(open(os.path.join(HERE, "alignment_results.json")))
BLUE, ORANGE, GREEN, RED, PURPLE = "#2f6fb0", "#e08a1e", "#3a9b6e", "#c0504d", "#7d5ba6"


def fig1_speed():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    rows = B["A_total"]
    ns = [r["n"] for r in rows]
    x = np.arange(len(ns)); w = 0.26
    nv = [r["naive_ms"] for r in rows]
    kv = [r["kv_ms"] for r in rows]
    hf = [r["hf_ms"] for r in rows]
    ax.bar(x-w, nv, w, label="手写·无cache重算", color=RED, alpha=.88)
    ax.bar(x, kv, w, label="手写·KV Cache", color=BLUE, alpha=.88)
    ax.bar(x+w, hf, w, label="HF generate()", color=GREEN, alpha=.88)
    for i in range(len(ns)):
        for j, v in enumerate((nv[i], kv[i], hf[i])):
            ax.text(i+(j-1)*w, v+15, f"{v:.0f}", ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([f"生成 {n} token" for n in ns])
    ax.set_ylabel("总耗时（ms）"); ax.set_ylim(0, 1450)
    ax.set_title("三个实现总耗时（0.5B bf16，prompt≈10 token）", fontsize=11.5)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    s = LC["settings"][1]  # S0=2048
    L = min(len(s["naive_steps"]), len(s["kv_steps"]))
    t = np.arange(1, L+1)
    raw_n = np.array(s["naive_steps"][:L])
    # 3 点滑动中值去除单次 GPU 抖动毛刺（首末点保留原值）
    sm_n = raw_n.copy()
    sm_n[1:-1] = np.median(np.stack([raw_n[:-2], raw_n[1:-1], raw_n[2:]]), 0)
    ax.plot(t, sm_n, color=RED, lw=1.8,
            label=f"无cache：每步重算 2048→2176 token")
    ax.plot(t, s["kv_steps"][:L], color=BLUE, lw=1.8, label="KV Cache：每步只算 1 token")
    nm, km = np.mean(sm_n), np.mean(s["kv_steps"])
    ax.axhline(nm, color=RED, ls="--", lw=1, alpha=.6)
    ax.axhline(km, color=BLUE, ls="--", lw=1, alpha=.6)
    ax.text(100, nm+0.6, f"均值 {nm:.1f} ms", color=RED, fontsize=10)
    ax.text(100, km-1.4, f"均值 {km:.1f} ms", color=BLUE, fontsize=10)
    ax.annotate(f"26.9→30.8 ms\n随历史长度缓慢爬升", (128, sm_n[L-1]),
                xytext=(52, 35), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("第 t 个生成 token"); ax.set_ylabel("该步耗时（ms）")
    ax.set_title("逐步耗时（prompt=2048，生成128）：总时 4.2×，\n重算线在上且向上走，KV 线水平", fontsize=11.5)
    ax.set_ylim(0, 40); ax.legend(fontsize=10, loc="center right"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_speed.png"), dpi=150)
    plt.close(fig)


def fig2_batch():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    rows = B["C_batch"]
    bs = [r["B"] for r in rows]
    agg = [r["agg_tok_s"] for r in rows]
    tot = [r["batch_ms"] for r in rows]
    x = np.arange(len(bs))
    bars = ax.bar(x, agg, 0.55, color=BLUE, alpha=.85, label="聚合吞吐 tok/s")
    for i, v in enumerate(agg):
        ax.text(i, v+40, f"{v:.0f}", ha="center", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([f"B={b}" for b in bs])
    ax.set_ylabel("聚合吞吐（tok/s）"); ax.set_ylim(0, 2500)
    ax2 = ax.twinx()
    ax2.plot(x, tot, "o-", color=ORANGE, lw=2.4, ms=8, label="整批总耗时 ms")
    for i, v in enumerate(tot):
        ax2.annotate(f"{v:.0f}ms", (i, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9.5, color=ORANGE)
    ax2.set_ylabel("整批 96 token 总耗时（ms）", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE); ax2.set_ylim(0, 1300)
    ax.set_title("手写 KV 循环批量生成：B=1→16 总时几乎不变（696→717ms），\n"
                 "聚合吞吐 138→2143 tok/s（15.5×）——和官方引擎同一物理", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left", fontsize=10.5)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_batch.png"), dpi=150)
    plt.close(fig)


def fig3_correctness():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    # 左：正确性证据卡
    ax = axes[0]; ax.axis("off")
    ax.set_title("正确性证据：手写 KV 循环 vs HF generate（greedy）", fontsize=12)
    y = 0.86
    ax.text(0.02, y, "验证项", fontsize=11, weight="bold"); ax.text(0.62, y, "结果", fontsize=11, weight="bold")
    y -= 0.12
    items = [
        ("生成 token 序列（3 prompt×32=96 个）", "逐 token 完全相同 ✓"),
        ("每步最后位置 logits", "最大绝对差 0.00e+00 ✓"),
        ("temperature=0 贪心", "== argmax ✓"),
        ("top_k=1", "等价贪心 ✓"),
        ("top_p 核采样（250 次抽样）", "只落保留前缀 ✓"),
        ("固定随机种子", "两次采样序列一致 ✓"),
    ]
    for k, v in items:
        ax.text(0.02, y, k, fontsize=10.5)
        ax.text(0.62, y, v, fontsize=10.5, color=GREEN, weight="bold")
        y -= 0.125
    fig.text(0.035, 0.03, "意义：我们的 100 行循环与 HF 共享同一套数学；\n"
             "vLLM/llama.cpp 在做的只是把它做得更快、更省、更能并发。",
             fontsize=9.5, color="#555")
    # 右：TTFT 随 prompt 长度
    ax = axes[1]
    d = B["D_ttft"]
    S = [r["S"] for r in d]; ms = [r["ttft_ms"] for r in d]
    ax.plot(S, ms, "o-", color=PURPLE, lw=2.4, ms=8)
    for s_, m_ in zip(S, ms):
        ax.annotate(f"{m_:.1f}ms", (s_, m_), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt 长度（token，对数轴）"); ax.set_ylabel("流式首 token 延迟 TTFT（ms）")
    ax.set_title("流式 TTFT 扫描：S≤256 被固定开销钉在 ~8ms，\n"
                 "S=1024 才开始抬头（同第 4 章 prefill 平台）", fontsize=11)
    ax.set_ylim(0, 16); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_correctness.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_speed(); fig2_batch(); fig3_correctness()
    print("written to", os.path.abspath(OUT))
