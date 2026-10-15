#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 5 章配图：崩塌曲线双轴 / 吞吐饱和 / 锚点形态对照（数据读 sweep_results.json）"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "images", "figures", "ch05")
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(HERE, "sweep_results.json")))
R = D["rows"]
C = [r["concurrency"] for r in R]
SR = [r["success"]*100 for r in R]
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#2f6fb0", "#e08a1e", "#3a9b6e", "#c0504d", "#7d5ba6", "#8a8f98"

# 用户提供的锚点：某 4B 模型服务实测（不同机型，只对照形态）
ANCHOR_C = [1, 50, 100, 200, 300]
ANCHOR_LAT = [0.53, 7.27, 13.45, 22.3, 30.7]
ANCHOR_SR = [100, 100, 100, 59.8, 46.2]


def fig1_collapse():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xscale("log", base=2)
    ok = [r for r in R if r["n_ok"] > 0]
    cx = [r["concurrency"] for r in ok]
    ax.plot(cx, [r["ttft_p50"]/1000 for r in ok], "o-", color=PURPLE, lw=2, ms=5,
            label="TTFT 首token p50（排队+prefill）")
    ax.plot(cx, [r["total_p50"]/1000 for r in ok], "s-", color=BLUE, lw=2.4, ms=6,
            label="E2E 总延迟 p50")
    ax.plot(cx, [r["total_p90"]/1000 for r in ok], "^-", color=ORANGE, lw=2.2, ms=6,
            label="E2E 总延迟 p90")
    ax2 = ax.twinx()
    ax2.plot(C, SR, "o-", color=RED, lw=2.4, ms=5, label="成功率（右轴）")
    ax2.set_ylabel("成功率（%）", color=RED, fontsize=12)
    ax2.set_ylim(0, 105); ax2.tick_params(axis="y", colors=RED)
    ax2.axhline(99, color=RED, ls=":", lw=1, alpha=0.6)

    # 三区背景
    ax.axvspan(0.8, 32, color=GREEN, alpha=0.07)
    ax.axvspan(32, 544, color=ORANGE, alpha=0.07)
    ax.axvspan(544, 1600, color=RED, alpha=0.07)
    bbox_w = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2)
    ax.text(5.5, 24.5, "舒适区\nC≤32 槽位\n即来即算", color=GREEN, fontsize=10,
            ha="center", bbox=bbox_w)
    ax.text(150, 22.2, "排队区：32<C≤544（32槽+512队列）\n吞吐已封顶，延迟随队列线性涨",
            color="#a85d10", fontsize=10, ha="center", bbox=bbox_w)
    ax.text(960, 24.5, "崩塌区\n拒入+30s超时", color=RED, fontsize=10,
            ha="center", bbox=bbox_w)
    ax.axhline(30, color=GRAY, ls="--", lw=1.5)
    ax.text(1.05, 30.8, "客户端超时 30 s", color=GRAY, fontsize=10)

    ax.set_xlabel("同时涌入的并发请求数 C（对数轴）")
    ax.set_ylabel("延迟（秒）")
    ax.set_title("并发崩塌全景（4090D + ollama -np 32，0.5B Q4，每请求 96 token）\n"
                 "延迟在排队区线性爬升，越过容量/超时线后成功率跳水", fontsize=12.5)
    ax.set_ylim(0, 33); ax.set_xlim(0.9, 1600); ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_collapse.png"), dpi=150)
    plt.close(fig)


def fig2_throughput():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xscale("log", base=2)
    # 有效吞吐 = 成功 token / 全轮墙钟（成功轮=最后完成时刻；失败轮=超时30.2s）
    good = [r["tokens"]/(r["total_max"]/1000 if r["success"] == 1 else 30.2) for r in R]
    ax.plot(C, good, "o-", color=BLUE, lw=2.5, ms=7,
            label="有效吞吐 tok/s（成功token/全轮墙钟）")
    sat = sum(r["agg_tok_s"] for r in R if 32 <= r["concurrency"] <= 512) /           len([r for r in R if 32 <= r["concurrency"] <= 512])
    ax.axhline(sat, color=ORANGE, ls="--", lw=2)
    ax.text(1.1, sat*1.04, f"服务吞吐天花板 ≈ {sat:.0f} tok/s（C=32 即触顶，±5%噪声）", color=ORANGE, fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(C, [r["n_ok"] for r in R], "s-", color=GREEN, lw=2, ms=5,
             label="成功完成请求数（右轴）")
    ax2.set_ylabel("本轮成功请求数", color=GREEN, fontsize=12)
    ax2.tick_params(axis="y", colors=GREEN)
    for r, g in zip(R, good):
        if r["concurrency"] in (1, 32, 128, 512, 1408):
            ax.annotate(f"{g:.0f}", (r["concurrency"], g),
                        textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5)
    ax.axvline(32, color=GRAY, ls=":", lw=1.5)
    ax.text(36, 300, "C=32：槽位填满\n之后再加并发\n吞吐零收益", color=GRAY, fontsize=10)
    ax.annotate("崩塌区成功完成数封顶\n30s 内最多消化 ~18-20 波", xy=(1000, 580),
                xytext=(120, 480), fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("并发请求数 C（对数轴）"); ax.set_ylabel("聚合吞吐（tok/s）")
    ax.set_ylim(0, 2400); ax.set_xlim(0.9, 1600); ax.grid(alpha=0.3)
    ax.set_title("Little 定律现场：吞吐约在 C=32 触顶后永久走平，崩塌区连'成功请求数'都开始停滞",
                 fontsize=12.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="right", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_throughput.png"), dpi=150)
    plt.close(fig)


def fig3_anchor():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.set_xscale("log")
    ok = [r for r in R if r["n_ok"] > 0]
    ax.plot([r["concurrency"] for r in ok], [r["total_p90"]/1000 for r in ok],
            "^-", color=BLUE, lw=2.2, ms=6, label="本机 0.5B Q4 / 4090D：E2E p90")
    ax.plot(ANCHOR_C, ANCHOR_LAT, "o-", color=RED, lw=2.2, ms=7,
            label="锚点 4B 服务：平均总延迟")
    for x, y in zip(ANCHOR_C, ANCHOR_LAT):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(6, 7),
                    fontsize=9.5, color=RED)
    ax.set_xlabel("并发数 C（对数轴）"); ax.set_ylabel("总延迟（秒）")
    ax.set_title("形态对照①：延迟都是'先平后线性'，拐点位置由\n槽位数×单请求成本决定（两图不可横比数值）", fontsize=11.5)
    ax.set_ylim(-1, 34); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="upper left")

    ax = axes[1]
    ax.set_xscale("log")
    ax.plot(C, SR, "^-", color=BLUE, lw=2.2, ms=6, label="本机 0.5B：成功率")
    ax.plot(ANCHOR_C, ANCHOR_SR, "o-", color=RED, lw=2.2, ms=7, label="锚点 4B：成功率")
    for x, y in zip(ANCHOR_C, ANCHOR_SR):
        if y < 100:
            ax.annotate(f"{y}%", (x, y), textcoords="offset points", xytext=(6, 7),
                        fontsize=9.5, color=RED)
    ax.set_xlabel("并发数 C（对数轴）"); ax.set_ylabel("成功率（%）"); ax.set_ylim(0, 108)
    ax.set_title("形态对照②：越过容量线后成功率断崖（锚点 200→59.8%、300→46.2%；\n本机需 C>544 才触发，因模型小 8 倍、槽位机制不同）", fontsize=11.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    fig.suptitle("同一出戏，不同舞台：容量耗尽 → 排队 → 超时/拒入 → 成功率跳水", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_anchor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_collapse(); fig2_throughput(); fig3_anchor()
    print("written to", os.path.abspath(OUT))
