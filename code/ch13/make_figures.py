#!/usr/bin/env python3
"""ch13 配图生成（PagedAttention）

前置：pip install matplotlib（中文字体 WenQuanYi Micro Hei）
用法：python make_figures.py
输出：../../images/figures/ch13/fig1_waste.png, fig2_block_table.png

数据口径：
- fig1 三种内存浪费的示意比例来自论文 arXiv:2309.06180 §3.1/§6.2
  （existing systems 实际有效 KV 内存仅 20.4%–38.2%），为教学示意而非精确复刻
- fig2 块号/引用计数为教学示例（对应论文 Fig.7/Fig.8 的双请求与 fork 场景）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "../../images/figures/ch13"

# ---------------------------------------------------------------- fig1
def make_fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # 左：传统整段预留的三种浪费（示意堆叠条）
    segs = [
        ("实际 token KV（有效）", 0.204, "#4CAF50"),
        ("预留未用 reserved", 0.30, "#FFB74D"),
        ("内部碎片 internal", 0.34, "#EF5350"),
        ("外部碎片 external", 0.156, "#8E5A9B"),
    ]
    left = 0.0
    for name, w, c in segs:
        ax1.barh([0], [w], left=left, color=c, edgecolor="white", height=0.5)
        ax1.text(left + w / 2, 0, f"{w*100:.1f}%", ha="center", va="center",
                 fontsize=11, color="white", weight="bold")
        left += w
    ax1.text(0.204 / 2, -0.42, "实际 token KV\n20.4%", ha="center", fontsize=9.5,
             color="#2E7D32")
    ax1.text(0.504, 0.42, "预留未用 30%", ha="center", fontsize=9.5, color="#E65100")
    ax1.text(0.824 - 0.17, -0.42, "内部碎片 34%", ha="center", fontsize=9.5, color="#B71C1C")
    ax1.text(0.922, 0.42, "外部碎片 15.6%", ha="center", fontsize=9.5, color="#4A148C")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(-0.7, 0.7)
    ax1.set_yticks([])
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax1.set_title("传统系统：整段按 max_len 预留 KV（论文实测有效率最低 20.4%）",
                  fontsize=12, weight="bold", pad=12)
    for s in ["top", "right", "left"]:
        ax1.spines[s].set_visible(False)

    # 右：PagedAttention 固定块按需分配，浪费锁进一个块
    ax2.add_patch(Rectangle((0, 0.30), 1.0, 0.34, fc="#ECEFF1", ec="#90A4AE",
                            lw=1.2, hatch="//"))
    for i in range(16):
        x = i * 0.06
        ax2.add_patch(Rectangle((x + 0.005, 0.345), 0.05, 0.25,
                                fc="#4CAF50" if i < 15 else "#BDBDBD",
                                ec="white", lw=1))
    ax2.text(0.5, 0.80, "物理池：固定 16-token 块，等大 ⇒ 无外部碎片",
             ha="center", fontsize=11, weight="bold")
    ax2.annotate("最后一块没填满\n（内部碎片 ≤ 1 块）", xy=(0.98, 0.47),
                 xytext=(0.72, 0.05), fontsize=10, color="#B71C1C",
                 arrowprops=dict(arrowstyle="->", color="#B71C1C"))
    ax2.text(0.46, 0.14, "按需逐块增长：用多少占多少", ha="center", fontsize=10.5,
             color="#2E7D32", weight="bold")
    ax2.set_xlim(-0.02, 1.08)
    ax2.set_ylim(0, 1.0)
    ax2.axis("off")
    ax2.set_title("PagedAttention：逻辑块→物理块映射，浪费被锁进单个块内",
                  fontsize=12, weight="bold", pad=12)

    fig.suptitle("分页前后对比：显存浪费去哪了（示意，比例为论文实测下限）",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig1_waste.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- fig2
def make_fig2():
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # 请求 A 的 block table
    def table(x, y, title, rows, color):
        ax.add_patch(FancyBboxPatch((x, y), 24, 4.2, boxstyle="round,pad=0.4",
                                    fc=color, ec="#37474F", lw=1.4))
        ax.text(x + 12, y + 2.1, title, ha="center", va="center", fontsize=11,
                weight="bold", color="white")
        for i, (lb, pb, ref) in enumerate(rows):
            ry = y - 5.4 - i * 5.4
            ax.add_patch(Rectangle((x, ry), 8, 4.6, fc="#ECEFF1", ec="#78909C"))
            ax.add_patch(Rectangle((x + 8.5, ry), 8, 4.6, fc="#BBDEFB", ec="#78909C"))
            ax.add_patch(Rectangle((x + 17, ry), 7, 4.6, fc="#FFE0B2", ec="#78909C"))
            ax.text(x + 4, ry + 2.3, lb, ha="center", va="center", fontsize=9.5)
            ax.text(x + 12.5, ry + 2.3, pb, ha="center", va="center", fontsize=9.5,
                    weight="bold")
            ax.text(x + 20.5, ry + 2.3, str(ref), ha="center", va="center",
                    fontsize=10, weight="bold",
                    color="#C62828" if ref > 1 else "#37474F")
        return y - 5.4 - len(rows) * 5.4

    rowsA = [("逻辑块 0", "物理 7", 2), ("逻辑块 1", "物理 1", 2),
             ("逻辑块 2", "物理 3", 1), ("逻辑块 3", "物理 9", 1)]
    rowsB = [("逻辑块 0", "物理 7", 2), ("逻辑块 1", "物理 1", 2),
             ("逻辑块 2", "物理 4", 1)]
    table(3, 48, "请求 A（样本 A1）的 block table", rowsA, "#1565C0")
    table(66, 48, "请求 B（样本 A2）的 block table", rowsB, "#00838F")

    # 物理块池
    ax.text(50, 22, "GPU 物理块池（等大、不连续）", ha="center", fontsize=11.5,
            weight="bold")
    phys = [(0, "7"), (1, "1"), (2, "3"), (3, "4"), (4, "9")]
    for j, (_, pid) in enumerate(phys):
        x = 20 + j * 13
        ref = {"7": 2, "1": 2, "3": 1, "4": 1, "9": 1}[pid]
        fc = "#FFCC80" if ref > 1 else "#C5E1A5"
        ax.add_patch(FancyBboxPatch((x, 13), 10, 7.5, boxstyle="round,pad=0.3",
                                    fc=fc, ec="#37474F", lw=1.3))
        ax.text(x + 5, 17.6, f"物理 {pid}", ha="center", va="center", fontsize=10,
                weight="bold")
        ax.text(x + 5, 15.0, f"ref={ref}", ha="center", va="center", fontsize=9.5)

    # 引用计数说明
    ax.add_patch(FancyBboxPatch((2, 2), 96, 7.4, boxstyle="round,pad=0.4",
                                fc="#FFF8E1", ec="#F9A825", lw=1.4))
    ax.text(50, 7.6,
            "fork 后共享：物理块 7、1 被两个样本共用（ref=2），prompt 不复制；"
            "写新块（物理 3、4、9）ref=1 直接写，\n"
            "只有写最后一个共享块才触发 copy-on-write——OS 的 fork() + COW 原样重现",
            ha="center", va="center", fontsize=10.5, linespacing=1.5)

    ax.text(50, 59.5, "block table：逻辑块号 → 物理块号 + 引用计数（共享与 CoW 的全部机关）",
            ha="center", fontsize=13.5, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_block_table.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    print(f"done -> {OUT}/fig1_waste.png, fig2_block_table.png")
