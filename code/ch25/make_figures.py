# ch25 配图生成脚本：1 张程序示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch25/make_figures.py
# 图1：求职篇全章结构——题库框架 / 三轮面试 / 作品集展示 / 五天计划
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch25")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#cfe2ff", ec="black", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, color="black"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.3))


def fig1():
    fig, ax = plt.subplots(figsize=(14.2, 9.8))
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 9.8); ax.axis("off")
    ax.set_title("图 1 程序示意：求职篇全章结构——把知识网络转成三页纸、两小时与一个仓库（结构示意，零预填数字）",
                 fontsize=12.5)

    # 顶部总纲
    box(ax, 0.35, 8.7, 13.5, 0.85,
        "不变的锚：JD 黑话表（ch02，20 行）每行 = 一道待考题 · 简历每行 = 你主动递出的题干 · 证据 = RESULTS 表行 / 机制推导 / CI 绿灯",
        fc="#fff3c4", ec="#b26a00", fs=9.3, bold=True)

    # 左列：四类题框架
    ax.text(2.5, 8.25, "① 题库四类框架（第 2 节）", ha="center", fontsize=10, fontweight="bold")
    left = [
        (7.0, "原理类\n结论 -> 机制 -> 边界\n三档伸缩 30s/90s/5min", "#cfe2ff"),
        (5.5, "实操类\n证据链五段（ch24）\n现象->取证->归因\n->动作->验证", "#c8e6c9"),
        (4.0, "场景类（系统设计）\n澄清四问 -> Little 定律\n-> 架构 -> 风险\n（ch23 五模块倒用）", "#ffe0b2"),
        (2.5, "前沿类\n现状 -> 代表工作\n-> 边界 -> 我的判断\n（一手口径 + 分级）", "#f8bbd0"),
    ]
    for y, t, c in left:
        box(ax, 0.5, y, 4.0, 1.3, t, fc=c, fs=8.3, bold=True)

    # 中列：三轮面试
    ax.text(7.3, 8.25, "② 三轮模拟面试（第 3 节）", ha="center", fontsize=10, fontweight="bold")
    mid = [
        (7.0, "第一轮 技术深挖\n考：这是不是你做的\n框架：名词->机制->我的证据\n露馅：追到实测就没数字", "#e1f5fe"),
        (5.3, "第二轮 系统设计\n考：这活交给你敢不敢接\n框架：90 秒四问开场 + 容量账\n露馅：全程没写一个数字", "#e8f5e9"),
        (3.6, "第三轮 项目追问\n考：过程真不真、学到什么\n框架：五段 + 真货反思\n露馅：说不出文件名/命令", "#fff9c4"),
        (1.9, "模拟方法\n录音计时 90s 只听前 30s\n三层追问扛不住就回炉\n一票否决题五问书面化", "#ede7f6"),
    ]
    for y, t, c in mid:
        box(ax, 5.2, y, 4.2, 1.6, t, fc=c, fs=8.3, bold=True)

    # 右列：作品集四件
    ax.text(12.0, 8.25, "③ 作品集展示（第 4 节）", ha="center", fontsize=10, fontweight="bold")
    right = [
        (7.0, "README 30 秒法则\n定位 / 架构图第一屏\n3 行快速开始 / 结果表\n+ Limitations（诚实边界）", "#dcedc8"),
        (5.3, "技术博客\n只写一个决策：\n问题->机制->关键 30 行\n->实测表->局限", "#f0f4c3"),
        (3.6, "白板 3 分钟\n五层：网关/调度/KV/GPU\n必须写三数：\nSLO / 容量账 / 崩溃三线", "#ffe0e0"),
        (1.9, "演示 5 分钟脚本\n定位->架构->活跑->\n一个数带口径->边界\n离线备份，只演必成功", "#e0f2f1"),
    ]
    for y, t, c in right:
        box(ax, 9.7, y, 4.15, 1.5, t, fc=c, fs=8.3, bold=True)

    # 列间箭头
    arrow(ax, (4.6, 6.6), (5.15, 6.6))
    arrow(ax, (9.5, 6.6), (9.65, 6.6))

    # 底部五天计划
    days = ["D56\n简历三版\n+对位表", "D57\n四类题\n书面答案", "D58\n三轮模拟\n录音", "D59\nREADME 体检\n+博客初稿", "D60\n三公司投递\n白板 x3"]
    for i, d in enumerate(days):
        x = 0.5 + i * 2.8
        box(ax, x, 0.15, 2.4, 1.1, d, fc="#e0e0e0", fs=8.5, bold=True)
        if i < 4:
            arrow(ax, (x + 2.4, 0.7), (x + 2.8, 0.7))
    ax.text(7.1, 1.5, "五天执行计划（第 5 节）：每天一个可检查的交付，明细回填 code/ch25/RESULTS.md",
            ha="center", fontsize=9, style="italic", color="#444444")

    fig.savefig(os.path.join(OUT, "fig1_job_map.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print(f"已生成 -> {OUT}")
