#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 章配图：数据读自 probe_results.json / roofline_results.json（4090D 实测）"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "images", "figures", "ch03")
os.makedirs(OUT, exist_ok=True)
P = json.load(open(os.path.join(HERE, "probe_results.json")))
R = json.load(open(os.path.join(HERE, "roofline_results.json")))
BLUE, ORANGE, GREEN, RED, PURPLE = "#2f6fb0", "#e08a1e", "#3a9b6e", "#c0504d", "#7d5ba6"
PEAK = R["peak_tflops"]; BW = R["bw_gbs"]; AIS = R["ai_star"]
N2 = 2 * P["n_params_B"]           # 每 token GFLOP 系数 = 2N(单位B) -> ×1e9 FLOP


def fig1_roofline():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    x = np.logspace(0, 3.6, 200)
    roof = np.minimum(BW/1000 * x, PEAK)     # 920 GB/s -> 0.92 TFLOPS at AI=1
    ax.plot(x, roof, color="#333", lw=2.5, label="Roofline 屋顶（实测 920 GB/s / 146 TFLOPS）")
    ax.fill_between(x, 0.05, roof, color="#2f6fb0", alpha=0.07)
    ax.axvline(AIS, color=RED, ls="--", lw=1.5)
    ax.text(AIS*1.08, 0.07, f"拐点 AI*≈{AIS:.0f}\n(FLOP/byte)", color=RED, fontsize=11)
    ax.text(2.2, 3.2, "带宽区（memory-bound）\n提速靠少搬数据/拼batch", color="#8a5a00", fontsize=12, rotation=33)
    ax.text(5200, 170, "算力区（compute-bound）\n提速靠 Tensor Core/融合", color="#1d5c40", fontsize=12, ha="right")

    # prefill 点
    pf_x = [d["ai"] for d in P["prefill"]]
    pf_y = [d["tflops"] for d in P["prefill"]]
    ax.scatter(pf_x, pf_y, s=130, color=GREEN, zorder=5, label="prefill（一次喂 S 个 token）")
    for d in P["prefill"]:
        ax.annotate(f"S={d['S']}\n{d['tflops']:.0f} TF", (d["ai"], d["tflops"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10, color=GREEN)

    # decode 点（AI≈B，聚合算力=聚合tok/s×2N/1e3）
    db = [(d["B"], d["agg"]*N2/1000) for d in P["decode_batch"]]
    dx = [max(b, 1) for b, _ in db]; dy = [y for _, y in db]
    ax.scatter(dx, dy, s=130, color=ORANGE, marker="D", zorder=5,
               label="decode（batch=1/4/8/16/32，AI≈B）")
    for b, y in db:
        ax.annotate(f"B={b}", (max(b,1), y), textcoords="offset points",
                    xytext=(6, -14), fontsize=10, color=ORANGE)

    ax.scatter([1], [P["decode_eager"]["toks_s"]*N2/1000], s=160, color=RED, marker="x", zorder=6)
    ax.annotate("HF 朴素 decode B=1\n(141 tok/s, 0.14 TFLOPS)", (1, 0.14),
                textcoords="offset points", xytext=(14, -4), fontsize=10, color=RED)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.7, 7000); ax.set_ylim(0.05, 300)
    ax.set_xlabel("算术强度 AI = 每字节搬运换来的浮点运算数（FLOP/byte）")
    ax.set_ylabel("实测性能（TFLOPS）")
    ax.set_title("RTX 4090 D 实测 Roofline：prefill 贴着算力顶，decode 全趴在带宽坡上\n"
                 "（Qwen2.5-0.5B bf16，2026-10-15）", fontsize=13)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_roofline.png"), dpi=150)
    plt.close(fig)


def fig2_batch():
    bs = P["decode_batch"]
    B = [d["B"] for d in bs]
    agg = [d["agg"] for d in bs]
    per = [d["per_req"] for d in bs]
    fig, ax1 = plt.subplots(figsize=(10, 5.6))
    bars = ax1.bar([str(b) for b in B], agg, width=0.55, color=BLUE, alpha=0.85, label="聚合吞吐 tok/s")
    for b, v in zip(bars, agg):
        ax1.text(b.get_x()+b.get_width()/2, v+60, f"{v:,.0f}", ha="center", fontsize=11)
    ax1.set_xlabel("同时拼在一个 batch 里的请求数 B")
    ax1.set_ylabel("聚合吞吐（tok/s）", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(0, 4800)
    ax2 = ax1.twinx()
    ax2.plot([str(b) for b in B], per, "o-", color=RED, lw=2.5, ms=9, label="每个请求分到的 tok/s")
    for i, v in enumerate(per):
        ax2.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10.5, color=RED)
    ax2.set_ylabel("单请求速度（tok/s）", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, 200)
    ax1.set_title("decode batch 扩展（0.5B，4090D 实测）：B 从 1→32，每步耗时 7.07→7.73ms 几乎不变\n"
                  "聚合吞吐 141→4,138 tok/s（29×），单请求几乎不掉速 —— Continuous Batching 的物理地基",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_batch_scaling.png"), dpi=150)
    plt.close(fig)


def fig3_ladder():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    rows = [
        ("理论带宽墙\n(920GB/s÷0.988GB权重)", 931, "#8a8f98", "硬件物理上限，任何实现越不过去"),
        ("llama.cpp/ollama Q4\n(融合kernel,第1章实测)", 648, GREEN, "摸到Q4口径带宽墙的~47%"),
        ("HF transformers + CUDA Graph\n(本章实测)", 210, ORANGE, "消除launch开销,1.5x"),
        ("HF transformers 朴素eager\n(本章实测)", 141, RED, "启动开销大,15%带宽"),
    ]
    for i, (name, v, c, note) in enumerate(rows):
        ax.barh(i, v, height=0.6, color=c, alpha=0.88)
        ax.text(v+12, i, f"{v} tok/s", va="center", fontsize=12)
        ax.text(18, i, note, va="center", fontsize=10, color="white")
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1080)
    ax.set_xlabel("Qwen2.5-0.5B 单请求 decode 速度（tok/s，4090 D）")
    ax.set_title("同一堵带宽墙下的四种水位：模型没变、硬件没变，差距全是系统工程\n"
                 "（llama.cpp 为 Q4 量化口径，权重更小，与 bf16 仅作量级对照）", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_decode_ladder.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_roofline(); fig2_batch(); fig3_ladder()
    print("written to", os.path.abspath(OUT))
