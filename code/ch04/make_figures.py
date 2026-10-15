#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 4 章配图：KV 增长曲线 / 有无cache逐步耗时 / 显存三巨头（数据读 json）"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "images", "figures", "ch04")
os.makedirs(OUT, exist_ok=True)
KV = json.load(open(os.path.join(HERE, "kv_results.json")))
NC = json.load(open(os.path.join(HERE, "nocache_results.json")))
M7 = KV["m7"]; M05 = KV["m05"]
BLUE, ORANGE, GREEN, RED, PURPLE = "#2f6fb0", "#e08a1e", "#3a9b6e", "#c0504d", "#7d5ba6"


def fig1_growth():
    seq = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    pt = M7["per_tok"]
    gqa = pt * seq / 1024**3
    gqa_i8 = pt/2 * seq / 1024**3
    mha_pt = M7["mha_per_tok"]
    mha = mha_pt * seq / 1024**3
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(seq, gqa, "o-", color=BLUE, lw=2.5, ms=7,
            label=f"GQA bf16（Qwen2.5-7B，每 token {pt/1024:.0f} KiB）")
    ax.plot(seq, gqa_i8, "s--", color=GREEN, lw=2, ms=6, label="KV int8 量化（字节减半）")
    ax.plot(seq, mha, "^-", color=RED, lw=2, ms=6,
            label=f"假设 MHA bf16（每 token {mha_pt/1024:.0f} KiB，{mha_pt//pt} 倍）")
    ax.axhline(M7["weight_gib"], color=ORANGE, ls="--", lw=2)
    ax.text(620, 6.2, "7B bf16 权重 15.23 GB（=14.19 GiB）",
            color=ORANGE, fontsize=11, ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    for s, v in ((2048, pt*2048/1024**3), (32768, pt*32768/1024**3),
                 (131072, pt*131072/1024**3)):
        ax.annotate(f"{v:.2f} GB", (s, v), textcoords="offset points", xytext=(7, 8),
                    fontsize=10, color=BLUE)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("序列长度 seq_len（token，对数轴）")
    ax.set_ylabel("单请求 KV Cache 占用（GB，对数轴）")
    ax.set_title("KV Cache 随序列长度线性增长：7B@2048 仅 0.11 GB，128k 长上下文达 7.0 GB\n"
                 "（batch=1；线上还要 ×并发数——这才是它被称为显存杀手的原因）", fontsize=12)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_kv_growth.png"), dpi=150)
    plt.close(fig)


def fig2_nocache():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    sw = NC["prefill_sweep"]
    S = [r["S"] for r in sw]
    ms = [r["ms"] for r in sw]
    dec = NC["decode_median_ms"]
    ax.plot(S, ms, "o-", color=RED, lw=2.2, ms=7,
            label="无 cache 第 t 步：重算整段（长 S 的 prefill 前向）")
    ax.axhline(dec, color=GREEN, lw=2.2)
    ax.text(60, dec*0.86, f"有 cache 第 t 步：decode 1 token，恒定 {dec:.1f} ms",
            color=GREEN, fontsize=10.5)
    ax.axvspan(32, 700, color=ORANGE, alpha=0.08)
    ax.text(150, 118, "固定开销平台\nS≤512 都约 14 ms\n(kernel 启动/大词表投影)",
            fontsize=9.5, color="#a85d10", ha="center")
    ax.annotate("S=8192 时重算一次要 152.6 ms", (8192, 152.6), textcoords="offset points",
                xytext=(-175, -26), fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("第 t 步要重算的历史长度 S（token，对数轴）")
    ax.set_ylabel("该步前向耗时（ms）")
    ax.set_title("无 KV Cache 的单步成本：短序列 14ms 打底，长序列线性涨到 153ms（0.5B 实测）",
                 fontsize=11)
    ax.set_ylim(0, 170); ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    summ = NC["summary"]
    ns = [r["n"] for r in summ]
    ax.plot(ns, [r["flop_ratio"] for r in summ], "o-", color=ORANGE, lw=2.5, ms=9,
            label="计算量 FLOPs 倍数（随 n 线性涨）")
    ax.plot(ns, [r["wall_ratio"] for r in summ], "s-", color=BLUE, lw=2.5, ms=9,
            label="墙钟耗时倍数（n=256 仍只 ~1.6×）")
    for r in summ:
        ax.annotate(f"{r['flop_ratio']:.0f}×", (r["n"], r["flop_ratio"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=10.5, color=ORANGE)
    ax.annotate("恒定 ~1.6×：多数步落在 14ms 平台\n重算走高效算力，decode 走低效访存",
                (220, 1.6), xytext=(100, 38), fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_xlabel("生成 token 数 n"); ax.set_ylabel("无 cache / 有 cache")
    ax.set_title("计算量 n=256 时已 142×，墙钟才 1.6×——账单(FLOPs/电)先爆，\n"
                 "墙钟的 n² 拐点要等输出足够长才来", fontsize=11)
    ax.set_ylim(0, 160); ax.legend(fontsize=10.5, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_nocache.png"), dpi=150)
    plt.close(fig)


def fig3_trio():
    # 权重 7B 实测(index)；KV 官方 StaticCache 实测；激活：2048 prefill 为 0.5B 同条件实测
    # （主体是 lm_head logits，vocab 几乎相同故可迁移），大并发/超长场景为工程估算并注明
    W = M7["weight_gib"]
    def kv(b, s): return M7["per_tok"] * b * s / 1024**3
    scenarios = [
        ("7B 单请求\nseq=2048\n(本章主菜)", [W, kv(1, 2048), M05["act_gib"]]),
        ("7B 32并发\n各 seq=8192\n(线上典型)", [W, kv(32, 8192), 0.8]),
        ("7B 单请求\nseq=131072\n(长文档)", [W, kv(1, 131072), 2.0]),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(scenarios)); w = 0.25
    wts = [s[1][0] for s in scenarios]
    kvs = [s[1][1] for s in scenarios]
    acts = [s[1][2] for s in scenarios]
    ax.bar(x-w, wts, w, label="权重（固定税）", color=ORANGE, alpha=0.88)
    ax.bar(x, kvs, w, label="KV Cache（随并发×上下文膨胀）", color=BLUE, alpha=0.88)
    ax.bar(x+w, acts, w, label="激活/临时", color=GREEN, alpha=0.88)
    for i in range(len(scenarios)):
        for j, v in enumerate((wts[i], kvs[i], acts[i])):
            ax.text(i+(j-1)*w, v+0.4, f"{v:.2f}", ha="center", fontsize=10.5)
    ax.axhline(23.5, color=RED, ls="--", lw=1.8)
    ax.text(2.42, 24.2, "RTX 4090 D 显存 23.5 GB", color=RED, fontsize=10.5, ha="right")
    tot = W + kv(32, 8192) + 0.8
    ax.annotate("", xy=(1, tot), xytext=(1, 23.5),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(1.07, tot+0.7, f"{tot:.1f} GB\n权重还没算完就 OOM", color=RED, fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in scenarios], fontsize=11)
    ax.set_ylabel("GB"); ax.set_ylim(0, 33)
    ax.set_title("显存三巨头：权重不动，KV 才是变量（7B bf16；权重/KV 为实测，激活见脚本注）",
                 fontsize=12)
    ax.legend(fontsize=11, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_vram_trio.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_growth(); fig2_nocache(); fig3_trio()
    print("written to", os.path.abspath(OUT))
