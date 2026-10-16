# ch16 配图生成脚本：3 张程序生成示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch16/make_figures.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch16")
os.makedirs(OUT, exist_ok=True)


def fig1():
    """链式 block hash（vLLM APC） vs 基数树（RadixAttention）结构对照"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle("图 1 程序示意：链式块哈希（vLLM APC） vs 基数树（RadixAttention）", fontsize=13)

    # ---- 左：链式 hash ----
    ax1.set_title("vLLM APC：每请求一条 hash 链（隐式共享）", fontsize=11)
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")

    def block(ax, x, y, w, h, text, fc, ec="black", fs=8.5, bold=False):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    # 请求 A
    ax1.text(0.2, 9.2, "请求 A：[system 3000 tok | user 100 tok]", fontsize=9)
    ya = 7.6
    for i in range(4):
        block(ax1, 0.5 + i * 2.1, ya, 1.7, 1.1, f"B0..B{i}\nh=H(父,B{i})", "#cfe2ff")
        if i < 3:
            ax1.add_patch(FancyArrowPatch((0.5 + i * 2.1 + 1.7, ya + 0.55),
                                          (0.5 + (i + 1) * 2.1, ya + 0.55),
                                          arrowstyle="-|>", mutation_scale=10))
    block(ax1, 0.5 + 4 * 2.1, ya, 1.7, 1.1, "B4\n(变化部分)", "#ffe0b2")

    # 请求 B（共享前 4 块）
    ax1.text(0.2, 5.6, "请求 B：同一 system 前缀，user 不同 → hash 相等才发现可复用", fontsize=9)
    yb = 4.0
    for i in range(4):
        block(ax1, 0.5 + i * 2.1, yb, 1.7, 1.1, f"B0..B{i}\n(hash 相同)", "#d9f2d9", ec="green")
        if i < 3:
            ax1.add_patch(FancyArrowPatch((0.5 + i * 2.1 + 1.7, yb + 0.55),
                                          (0.5 + (i + 1) * 2.1, yb + 0.55),
                                          arrowstyle="-|>", mutation_scale=10))
    block(ax1, 0.5 + 4 * 2.1, yb, 1.7, 1.1, "B4'\n(不同)", "#ffe0b2")

    # 虚线框表示"全局 hash→block 表"
    ax1.add_patch(Rectangle((0.3, 3.6), 9.4, 1.9, fill=False, edgecolor="green",
                            lw=1.5, linestyle="--"))
    ax1.annotate("全局 hash→block 表：hash 相等 = 复用（账各记各的）",
                 xy=(5, 3.3), fontsize=9, ha="center", color="green")
    ax1.annotate("链：改任意前面一个 token\n其后所有块全部失效", xy=(5, 1.6),
                 fontsize=9, ha="center", color="#b26a00")
    ax1.annotate("整块（16 tok）才入 hash 账；部分块 B4 不算", xy=(5, 0.5),
                 fontsize=9, ha="center", color="#b26a00")

    # ---- 右：基数树 ----
    ax2.set_title("SGLang RadixAttention：全部请求一棵树（显式共享 + LRU）", fontsize=11)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")

    # 根
    def node(x, y, w, h, text, fc):
        ax2.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.06", facecolor=fc, edgecolor="black", lw=1.2))
        ax2.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    node(3.9, 8.4, 2.4, 1.1, "root", "#e8e8e8")
    node(3.4, 6.6, 3.4, 1.1, "边：system prompt 3000 tok\n（物理 KV：所有请求共享一份）", "#cfe2ff")
    node(3.0, 4.8, 4.2, 1.1, "边：工具 schema / 多轮历史（多级共享）", "#d9f2d9")
    # 分叉
    node(1.0, 2.8, 3.4, 1.1, "分支 A：user-1 的问题\n→ 请求 A 独享 KV", "#ffe0b2")
    node(5.6, 2.8, 3.4, 1.1, "分支 B：user-2 的问题\n→ 请求 B 独享 KV", "#ffe0b2")
    node(3.0, 0.7, 4.0, 1.1, "树 = LRU 缓存；cache-aware 调度重排请求迁就树", "#f3e5f5")

    for (x1, y1, x2, y2) in [(5.1, 8.4, 5.1, 7.7), (5.1, 6.6, 5.1, 5.9),
                             (5.1, 4.8, 2.7, 3.9), (5.1, 4.8, 7.3, 3.9),
                             (5.1, 2.8, 5.0, 1.8)]:
        ax2.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                      mutation_scale=12, lw=1.5))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig1_hash_vs_radix.png"), dpi=150)
    plt.close(fig)


def fig2():
    """constrained decoding 每 token mask 原理"""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_title("图 2 程序示意：constrained decoding —— mask 在采样之前锁死合法候选（示意，非实测）",
                 fontsize=12)
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    # 1. schema
    ax.add_patch(Rectangle((0.2, 4.6), 2.6, 1.9, facecolor="#e3f2fd", edgecolor="black"))
    ax.text(1.5, 6.2, "JSON Schema", ha="center", fontsize=10, fontweight="bold")
    ax.text(1.5, 5.35, '{"label": enum,\n "confidence": number}', ha="center", fontsize=8)

    ax.add_patch(FancyArrowPatch((2.9, 5.5), (3.6, 5.5), arrowstyle="-|>", mutation_scale=14))

    # 2. 编译成状态机
    ax.add_patch(Rectangle((3.7, 4.6), 2.6, 1.9, facecolor="#fff3e0", edgecolor="black"))
    ax.text(5.0, 6.2, "语法状态机（编译一次，", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(5.0, 5.8, "整 batch 复用）", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(5.0, 5.1, "xgrammar/outlines/\nguidance/lm-format-enforcer", ha="center", fontsize=7.5)

    ax.add_patch(FancyArrowPatch((6.4, 5.5), (7.1, 5.5), arrowstyle="-|>", mutation_scale=14))

    # 3. 每步 bitmask
    ax.add_patch(Rectangle((7.2, 4.6), 4.5, 1.9, facecolor="#f3e5f5", edgecolor="black"))
    ax.text(9.45, 6.2, "每步 decode：算一个词表级 bitmask", ha="center", fontsize=9.5, fontweight="bold")
    # 小词表示意
    words = ['{"', '"la', 'bei', 'la"', '}', '好', '的', ':', ',', '9']
    colors = ["#4caf50"] * 6 + ["#e53935"] * 4
    for i, (w, c) in enumerate(zip(words, colors)):
        ax.add_patch(Rectangle((7.5 + i * 0.42, 5.0), 0.38, 0.55,
                               facecolor=c, edgecolor="black", lw=0.6))
        ax.text(7.69 + i * 0.42, 5.27, w, ha="center", va="center", fontsize=7, color="white")
    ax.text(9.45, 4.8, "绿 = 合法候选，红 = logit 置 −∞", ha="center", fontsize=7.5)

    # 4. logits 应用
    ax.add_patch(FancyArrowPatch((9.45, 4.5), (9.45, 3.7), arrowstyle="-|>", mutation_scale=14))
    ax.add_patch(Rectangle((7.2, 2.2), 4.5, 1.5, facecolor="#e8f5e9", edgecolor="black"))
    ax.text(9.45, 3.4, "采样前应用 mask", ha="center", fontsize=10, fontweight="bold")
    ax.text(9.45, 2.85, "apply_grammar_bitmask →\nxgr.apply_token_bitmask_inplace(logits)",
            ha="center", fontsize=8)

    # 5. 正常采样
    ax.add_patch(FancyArrowPatch((7.1, 2.95), (6.4, 2.95), arrowstyle="-|>", mutation_scale=14))
    ax.add_patch(Rectangle((3.7, 2.2), 2.6, 1.5, facecolor="#e8eaf6", edgecolor="black"))
    ax.text(5.0, 3.4, "temperature/top_p", ha="center", fontsize=10, fontweight="bold")
    ax.text(5.0, 2.75, "正常采样（第 6 章\n手写采样器同款）", ha="center", fontsize=8)

    ax.add_patch(FancyArrowPatch((3.6, 2.95), (2.9, 2.95), arrowstyle="-|>", mutation_scale=14))
    ax.add_patch(Rectangle((0.2, 2.2), 2.6, 1.5, facecolor="#cfe2ff", edgecolor="black"))
    ax.text(1.5, 3.4, "输出 token", ha="center", fontsize=10, fontweight="bold")
    ax.text(1.5, 2.75, "格式必合法\n（by construction）", ha="center", fontsize=8)

    # 注释
    ax.text(6.0, 1.5, "关键时序：mask 必须在“采样之前、logits 之后”——早了没有 logits，晚了已经采错",
            ha="center", fontsize=9.5, color="#b26a00")
    ax.text(6.0, 0.9, "约束保证语法（格式/取值域），不保证语义正确；XGrammar 论文口径 mask <40 µs/token（检索 2026-10-16）",
            ha="center", fontsize=9, color="#555555")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_structured_mask.png"), dpi=150)
    plt.close(fig)


def fig3():
    """并发阶梯压测曲线示意（概念图，非实测）"""
    fig, ax1 = plt.subplots(figsize=(10, 5.6))
    conc = np.array([1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100])
    # 概念曲线（非实测）：吞吐先涨后压平；TTFT/TPOT 先平后翘
    thr = 1900 * (1 - np.exp(-conc / 7.0)) / (1 + (conc / 90.0) ** 4)
    ttft = 0.15 + 0.004 * np.maximum(conc - 6, 0) ** 2      # c≈27.5 处越过 2s
    tpot = 0.030 + 0.0004 * conc + 0.00035 * np.maximum(conc - 18, 0) ** 1.8

    # 关键点：SLO 交叉点（TTFT p99 = 2s）与吞吐峰值（拐点），程序计算而非硬编码
    cf = np.linspace(1, 100, 2000)
    thr_f = 1900 * (1 - np.exp(-cf / 7.0)) / (1 + (cf / 90.0) ** 4)
    ttft_f = 0.15 + 0.004 * np.maximum(cf - 6, 0) ** 2
    c_slo = cf[np.argmax(ttft_f >= 2.0)]
    c_peak = cf[np.argmax(thr_f)]

    ax2 = ax1.twinx()
    l1, = ax1.plot(conc, ttft, "o-", color="#d32f2f", label="TTFT p99（s）")
    l2, = ax1.plot(conc, tpot, "s-", color="#1976d2", label="TPOT p99（s）")
    l3, = ax2.plot(conc, thr, "^-", color="#388e3c", label="有效吞吐（tok/s）")

    ax1.axhline(2.0, color="gray", linestyle="--", lw=1)
    ax1.axvline(c_slo, color="gray", linestyle="--", lw=1)
    ax1.annotate(f"SLO 容量点（TTFT p99 = 2s，约并发 {c_slo:.0f}）\n→ 给业务报这个数",
                 xy=(c_slo, 2.0), xytext=(c_slo + 8, 12), fontsize=9, color="#444444",
                 arrowprops=dict(arrowstyle="->", color="#444444"))
    ax1.axvline(c_peak, color="#b26a00", linestyle=":", lw=1.2)
    ax1.annotate(f"拐点（约并发 {c_peak:.0f}）：吞吐压平 + 延迟翘头\n= 引擎容量点（第 12 章 B_max）",
                 xy=(c_peak, ttft[np.argmin(np.abs(conc - c_peak))]),
                 xytext=(c_peak + 6, 22), fontsize=9, color="#b26a00",
                 arrowprops=dict(arrowstyle="->", color="#b26a00"))

    ax1.set_xlabel("并发（阶梯扫描 1→100）")
    ax1.set_ylabel("延迟 p99（秒）")
    ax2.set_ylabel("有效吞吐（tok/s）")
    ax1.set_title("图 3 程序示意：阶梯压测读出“吞吐压平 + 延迟翘头”拐点（概念曲线，非实测数据）",
                  fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
               loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_load_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("已生成 3 张图到", os.path.abspath(OUT))
