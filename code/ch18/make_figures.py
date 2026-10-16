# ch18 配图生成脚本：1 张选型决策树（零外部图片、零性能跑测）
# 运行：python3 code/ch18/make_figures.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch18")
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
        ax.text(mx, my + 0.18, text, ha="center", va="bottom", fontsize=fs, color=color)


def fig1():
    """七大引擎选型决策树（四步分叉）"""
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.set_title("图 1 程序示意：推理引擎选型决策树（四步先删后选，2026-10-16 版图）", fontsize=13)

    # 根节点
    box(ax, 4.6, 7.6, 4.3, 0.75,
        "Q1 本地开发/体验/单用户体验？", fc="#ffe082", bold=True, fs=10)
    box(ax, 0.4, 7.7, 3.4, 0.55, "Ollama（Go 封装 llama.cpp）", fc="#e0e0e0", fs=9.5, bold=True)
    arrow(ax, (4.6, 7.9), (3.8, 7.95), "是 → 到此为止", fs=8.5, color="#b26a00")
    ax.text(2.1, 7.5, "第 4 节四根因三连体检后使用", fontsize=7.5, color="#b26a00", ha="center")

    # Q2
    box(ax, 4.6, 6.5, 4.3, 0.7, "Q2 部署形态？", fc="#ffe082", bold=True, fs=10)
    arrow(ax, (6.75, 7.6), (6.75, 7.25), "否", fs=8.5)

    box(ax, 9.9, 7.7, 3.2, 0.55, "端侧（手机/浏览器）\n→ MLC-LLM (WebLLM)", fc="#cfe2ff", fs=8.5)
    arrow(ax, (8.9, 7.0), (9.9, 7.9), "端侧", fs=8.5)
    box(ax, 9.9, 6.5, 3.2, 0.7, "消费级显卡+大内存\n跑大 MoE → KTransformers", fc="#cfe2ff", fs=8.5)
    arrow(ax, (8.9, 6.85), (9.9, 6.85), "单机 MoE", fs=8.5)

    # Q3
    box(ax, 4.6, 5.3, 4.3, 0.7, "Q3 硬件是 NVIDIA 吗？", fc="#ffe082", bold=True, fs=10)
    arrow(ax, (6.75, 6.5), (6.75, 6.05), "服务端", fs=8.5)

    box(ax, 9.9, 5.35, 3.2, 0.6, "vLLM / SGLang 按覆盖选\n国产模型链路优先 LMDeploy", fc="#c8e6c9", fs=8.5)
    arrow(ax, (8.9, 5.65), (9.9, 5.65), "否", fs=8.5)

    # Q4
    box(ax, 4.6, 4.1, 4.3, 0.7,
        "Q4 模型/精度半年不变\n且吞吐=真金白银？", fc="#ffe082", bold=True, fs=9.5)
    arrow(ax, (6.75, 5.3), (6.75, 4.85), "是", fs=8.5)

    box(ax, 9.9, 4.15, 3.2, 0.6, "TensorRT-LLM\n（编译税 + NVIDIA 绑定）", fc="#c8e6c9", fs=9)
    arrow(ax, (8.9, 4.45), (9.9, 4.45), "是", fs=8.5)

    # Q5
    box(ax, 4.6, 2.7, 4.3, 0.75,
        "Q5 共享前缀重不重？\n(多轮/共享system/RAG块)", fc="#ffe082", bold=True, fs=9.5)
    arrow(ax, (6.75, 4.1), (6.75, 3.5), "否", fs=8.5)

    box(ax, 9.9, 3.1, 3.2, 0.7, "SGLang\nRadixAttention（论文 74.1%）", fc="#c8e6c9", fs=9)
    arrow(ax, (8.9, 3.15), (9.9, 3.4), "重", fs=8.5)
    box(ax, 9.9, 2.2, 3.2, 0.7, "vLLM（通用默认\n生态+人才供给）", fc="#c8e6c9", fs=9)
    arrow(ax, (8.9, 3.05), (9.9, 2.55), "轻/无", fs=8.5)

    # TGI 红牌
    box(ax, 0.4, 4.1, 3.4, 0.9, "TGI：任何新项目勿选\n（GitHub README 官方\nmaintenance mode 警示）",
        fc="#ffcdd2", ec="#c62828", fs=8.5, bold=True)
    ax.text(2.1, 3.85, "进树之前就被 ⑦风险维 删除（第 0 节）", fontsize=7.5,
            color="#c62828", ha="center")

    ax.text(6.75, 1.9,
            "先删后选：⑦风险（TGI）→ ②硬件（MLC/KTransformers/非NVIDIA）→ ⑥场景（TRT-LLM）→ ③前缀（SGLang vs vLLM）\n"
            "双雄分支选错代价有限：先跑起来，第 19 章同机同模型对打后终审；外部数字只做量级参考",
            fontsize=8.6, ha="center", va="top", color="#444444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fffde7", ec="#bbb"))

    fig.savefig(os.path.join(OUT, "fig1_decision_tree.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print("ch18 figure saved to", OUT)
