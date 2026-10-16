# ch20 配图生成脚本：2 张程序图（零外部图片、零性能跑测）
# 运行：python3 code/ch20/make_figures.py
# 同时打印第 2 节加速比账的代入表（数字登记进 RESULTS.md）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch20")
os.makedirs(OUT, exist_ok=True)


def E(alpha, k):
    """一轮期望产出：(1 - alpha^(k+1)) / (1 - alpha)，alpha=1 时退化为 k+1"""
    if alpha == 1.0:
        return k + 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def S(alpha, k, c=0.1):
    return E(alpha, k) / (1 + c * k)


def box(ax, x, y, w, h, text, fc="#cfe2ff", ec="black", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, text="", fs=8.5, color="black"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.3))
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, text, ha="center", va="bottom", fontsize=fs, color=color)


def fig1():
    """一轮投机解码五步流程图"""
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.set_title("图 1 程序示意：一轮投机解码的五步流程——打草稿/一次验收/逐位判/覆盖或 bonus（非实测）", fontsize=12.5)

    box(ax, 4.3, 7.7, 4.9, 0.7, "本轮起点：target 已有前文 KV（位置 i 起）", fc="#e0e0e0", fs=9.5, bold=True)

    steps = [
        (4.3, 6.4, 4.9, 0.8, "① 打草稿：小模型/头/ngram 串行出 k 个\nd1…dk，同时记下草稿概率 q(dj)", "#cfe2ff"),
        (4.3, 5.0, 4.9, 0.85, "② 一次 forward：target 吃下 前文+d1…dk\n出 k+1 个位置的 p(·)（访存搭便车）", "#c8e6c9"),
        (4.3, 3.5, 4.9, 0.9, "③ 逐位验收 j=1..k：以 min(1, p/q) 接受\n（贪心版：草稿 == target 的 argmax 即收）", "#fff9c4"),
    ]
    for x, y, w, h, t, c in steps:
        box(ax, x, y, w, h, t, fc=c, fs=9)
    arrow(ax, (6.75, 7.7), (6.75, 7.25))
    arrow(ax, (6.75, 6.4), (6.75, 5.9))
    arrow(ax, (6.75, 5.0), (6.75, 4.4))

    # 分支
    box(ax, 0.4, 1.9, 5.6, 1.0, "④a 有拒绝位 j*：\n按 norm(max(0,p−q)) 重采 1 个 token\n（此位用 target 自己的答案覆盖），本轮结束",
        fc="#ffcdd2", ec="#c62828", fs=8.8, bold=True)
    arrow(ax, (4.3, 3.75), (3.4, 2.9), "某位被拒", fs=9, color="#c62828")

    box(ax, 7.5, 1.9, 5.6, 1.0, "④b 全部 k 位都收：\n用位置 k+1 的 logits 再贪心 1 个\n= bonus token，本轮收 k+1 个",
        fc="#c8e6c9", fs=8.8, bold=True)
    arrow(ax, (9.2, 3.5), (10.3, 2.9), "全收（概率 α^k）", fs=9, color="#2e7d32")

    box(ax, 0.4, 0.4, 12.7, 0.85,
        "⑤ 拼接本轮产出（≤k+1 个 token），拒绝位之后的草稿 KV 回滚（ch04 生命周期），回到 ① 循环\n"
        "无损红线：与「关闭投机、逐 token 解码」逐位一致（贪心）/ 同分布（采样，拒绝采样证明 Leviathan App.A.1）",
        fc="#fffde7", ec="#b26a00", fs=8.8, bold=True)
    arrow(ax, (3.2, 1.9), (3.2, 1.3), color="#b26a00")
    arrow(ax, (10.3, 1.9), (10.3, 1.3), color="#b26a00")

    fig.savefig(os.path.join(OUT, "fig1_spec_round.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2():
    """加速比 S(alpha, k) 曲线（c=0.1 算账，非实测）"""
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.set_title("图 2 程序算账：加速比 S=E(α,k)/(1+0.1k) 随接受率 α 的变化（公式代入，非实测数据）",
                 fontsize=11.5)

    a = np.linspace(0.05, 1.0, 200)
    for k, style in [(1, "-"), (3, "--"), (4, "-."), (7, ":")]:
        ys = [S(ai, k) for ai in a]
        ax.plot(a, ys, style, lw=2, label=f"k = {k}")

    # 生产锚点（第三方/论文口径，标注在图上）
    ax.axvline(0.355, color="#c62828", ls="--", lw=1.4)
    ax.text(0.365, 1.15, "PayPal 生产实测\nα≈35.5%（γ=3，arXiv:2604.19767）\n吞吐 +22–49%",
            fontsize=8.6, color="#c62828", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#c62828", alpha=0.9))
    ax.axvline(0.80, color="#2e7d32", ls="--", lw=1.4)
    ax.text(0.79, 2.6, "DeepSeek-V3 MTP\nα>80%（第三方转述）", fontsize=8.6,
            color="#2e7d32", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff", ec="#2e7d32", alpha=0.9))

    ax.axhline(1.0, color="gray", lw=1)
    ax.set_xlabel("接受率 α")
    ax.set_ylabel("加速比 S（目标单 token 时间为单位）")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0.9, 3.8)
    ax.set_xlim(0.05, 1.0)
    ax.text(0.52, 0.97,
            "E(α,k)=(1−α^(k+1))/(1−α)，S=E/(1+ck)，本图 c=0.1（草稿=target 的 10% 成本）\n"
            "α 是主宰：α=0.35 时 k=7 反而 0.90×（比不开还慢）；S 随 c 升高而下降——公式代入算账，零性能跑测",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffde7", ec="#bbb"))

    fig.savefig(os.path.join(OUT, "fig2_speedup_curve.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_table():
    """打印第 2.3 节的代入表（供 RESULTS.md 登记）"""
    print("=== S = E/(1+0.1k) ===")
    for alpha in (0.35, 0.5, 0.7, 0.8, 0.9):
        row = [f"{S(alpha, k):.2f}x" for k in (1, 3, 4, 7)]
        print(f"alpha={alpha:<5}: " + "  ".join(row))
    print("=== 验收题 alpha=0.8 k=4 c=0.1 ===")
    print(f"E={E(0.8, 4):.3f}  cost={1+0.1*4}  S={S(0.8, 4):.2f}x")
    print("=== PayPal 反推 ===")
    print(f"a=0.355 k=3 c=0.10 -> {S(0.355,3,0.10):.3f}  c=0.05 -> {S(0.355,3,0.05):.3f}")
    print("=== c 敏感度 (a=0.8 k=4) ===")
    for c in (0.01, 0.1, 0.3):
        print(f"c={c}: {S(0.8,4,c):.2f}x")


if __name__ == "__main__":
    fig1()
    fig2()
    print_table()
    print("ch20 figures saved to", OUT)
