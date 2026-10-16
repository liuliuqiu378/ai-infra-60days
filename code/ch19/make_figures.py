# ch19 配图生成脚本：2 张程序示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch19/make_figures.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch19")
os.makedirs(OUT, exist_ok=True)


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
    """对打实验设计总图：三个负载 × 三个假设 × 七条控制变量"""
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.set_title("图 1 程序示意：双引擎对打实验设计——三负载验证三假设，七条控制变量托底（非实测数据）", fontsize=12.5)

    box(ax, 3.8, 7.7, 5.9, 0.7,
        "冻结 prompt 集（落盘 jsonl + 固定 seed + md5 登记）", fc="#ffe082", bold=True, fs=10)

    loads = [
        (0.5, "负载 A：无共享前缀\n随机主题 / 长度按分布采样", "验证 H1：无前缀差异≤5%\n（第三方口径 2–5% 在噪声内）", "#cfe2ff"),
        (5.0, "负载 B：高共享前缀\n长 system/文档块 + 短后缀\n前缀共享率 ≈70–75%", "验证 H2：SGLang 占优\n（论文命中 74.1%、吞吐 6.4× 口径）", "#c8e6c9"),
        (9.5, "负载 C：混合生产型\n长短按真实业务比例混合", "验证 H3：崩溃点与综合排名\n（等长负载测的是不存在的业务）", "#f8bbd0"),
    ]
    for x, t1, t2, c in loads:
        box(ax, x, 6.1, 3.5, 0.85, t1, fc=c, fs=8.8, bold=True)
        box(ax, x, 4.75, 3.5, 0.95, t2, fc="#fffde7", fs=8.2)
        arrow(ax, (x + 1.75, 7.7), (x + 1.75, 6.98))

    box(ax, 3.3, 3.3, 3.4, 0.85, "vLLM 服务\n（参数对等 + 缓存显式）", fc="#cfe2ff", fs=9.5, bold=True)
    box(ax, 6.9, 3.3, 3.4, 0.85, "SGLang 服务\n（参数对等 + 缓存显式）", fc="#c8e6c9", fs=9.5, bold=True)
    box(ax, 0.4, 3.3, 2.4, 0.85, "同一压测器\n同一台机同一张卡", fc="#ffe082", fs=8.8, bold=True)
    arrow(ax, (2.8, 3.72), (3.3, 3.72))
    arrow(ax, (6.7, 3.72), (6.9, 3.72), fs=8)
    box(ax, 10.7, 3.3, 2.4, 0.85, "五档并发扫描\n每档≥3次取中位\nnum-prompts≥5×C", fc="#e0e0e0", fs=8.5, bold=True)
    arrow(ax, (10.3, 3.72), (10.7, 3.72), fs=8)

    box(ax, 0.4, 1.9, 12.7, 0.75,
        "七条控制变量：①同一权重 ②硬件独占 ③引擎参数对等 ④同一压测器口径 ⑤prompt集冻结 ⑥缓存策略显式 ⑦采样固定(temp=0+max_tokens)",
        fc="#ffcdd2", ec="#c62828", fs=9, bold=True)
    for x in (1.6, 5.0, 8.5, 11.5):
        arrow(ax, (x, 1.9), (x, 1.66), color="#c62828")

    box(ax, 0.4, 0.35, 12.7, 0.85,
        "产出：报告八段（结论/前提/七条勾选/负载/口径/数据表+极差/主图/边界）+ 三条崩溃判定线（L1错误>1% · L2 p99超SLO · L3吞吐封顶）\n"
        "判定纪律：差异<5% = 打平在噪声内，不硬找赢家",
        fc="#fffde7", ec="#b26a00", fs=8.8, bold=True)

    fig.savefig(os.path.join(OUT, "fig1_experiment_design.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2():
    """双引擎并发-延迟病塌曲线示意（含三条判定线，纯示意非实测）"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_title("图 2 程序示意：崩溃点对比——双引擎并发-延迟曲线与三条判定线（形态为方法示意，非实测数据）",
                 fontsize=11.5)

    C = np.array([1, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96])
    # 示意曲线：构造为 A 恰在 C=40、B 恰在 C=64 触达 SLO=1000ms（纯形态示意）
    ttft_a = 60 + 940 * (C / 40.0) ** 3.5
    ttft_b = 58 + 942 * (C / 64.0) ** 3.5

    ax.plot(C, ttft_a, "o-", color="#1565c0", lw=2, ms=5, label="引擎 A（示意）")
    ax.plot(C, ttft_b, "s-", color="#2e7d32", lw=2, ms=5, label="引擎 B（示意）")

    slo = 1000
    ax.axhline(slo, color="#c62828", ls="--", lw=1.6)
    ax.text(2, slo * 1.15, "L2 延迟线：p99 TTFT 超 SLO（如 1 s）", ha="left",
            fontsize=9, color="#c62828")
    ax.plot([40], [slo], "v", color="#1565c0", ms=12)
    ax.plot([64], [slo], "v", color="#2e7d32", ms=12)
    ax.annotate("A 触延迟线\n崩溃点 C=40", (40, slo),
                xytext=(8, 4000), fontsize=9, color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0"))
    ax.annotate("B 触延迟线\n崩溃点 C=64", (64, slo),
                xytext=(72, 5000), fontsize=9, color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    ax.set_ylim(40, 200000)

    ax.text(0.98, 0.04,
            "L1 错误线：5xx/超时率>1%　　L3 吞吐线：吞吐环比增幅<5%（封顶，如 ch05 实测 C=32 封顶 ~1900 tok/s）\n"
            "每档并发 ≥3 次取中位并报极差；两引擎显存占用对齐后才开档",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffde7", ec="#bbb"))

    ax.set_xlabel("并发档 C")
    ax.set_ylabel("p99 TTFT（ms，示意值）")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xticks(C)

    fig.savefig(os.path.join(OUT, "fig2_collapse_duel.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("ch19 figures saved to", OUT)
