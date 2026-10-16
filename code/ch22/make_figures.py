# ch22 配图生成脚本：1 张程序示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch22/make_figures.py
# 图1：Strata 双案合流全景——同一套"分层+调度"方法论的两个考场
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch22")
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
    fig, ax = plt.subplots(figsize=(14.0, 9.2))
    ax.set_xlim(0, 14.0); ax.set_ylim(0, 9.2); ax.axis("off")
    ax.set_title("图 1 程序示意：Strata 双案合流——同一套分层+调度方法论的两个考场（结构示意，非实测数据）",
                 fontsize=12.5)

    # 顶部共用原理
    box(ax, 1.6, 8.35, 10.8, 0.7,
        "共用原理：最快介质贵且小 → 热数据画像决定放置 + 流水/重叠/预测藏起慢介质延迟（CPU 缓存的翻版，ch21）",
        fc="#ffe082", bold=True, fs=9.8)

    # 左：案一
    box(ax, 0.3, 7.1, 6.4, 0.95,
        "案一：论文 Strata（arXiv:2508.18572，OSDI'26）\n长上下文服务 · 分层 KV 缓存 · 建于 SGLang · 生产部署",
        fc="#cfe2ff", ec="#1565c0", fs=9.5, bold=True)
    # 右：案二
    box(ax, 7.3, 7.1, 6.4, 0.95,
        "案二：开源 Strata（github.com/Niko1221/Strata，MIT）\n消费级单卡 · 125B MoE（Qwen3.8-Flash-Next）· 0.1.x",
        fc="#c8e6c9", ec="#2e7d32", fs=9.5, bold=True)
    arrow(ax, (4.6, 8.35), (3.9, 8.06), color="#1565c0")
    arrow(ax, (9.4, 8.35), (10.1, 8.06), color="#2e7d32")

    # 左：案一细节
    box(ax, 0.3, 5.55, 6.4, 1.3,
        "问题（论文 §3）：① 分页碎片 I/O 打不满带宽\n② 调度器不知道缓存 → loading-bound（delay hit）\n"
        "③ 分页布局跨介质不友好",
        fc="#fffde7", ec="#b26a00", fs=8.8)
    arrow(ax, (3.5, 5.5), (3.5, 5.16), color="#b26a00")
    box(ax, 0.3, 3.95, 6.4, 1.2,
        "三项设计（论文 §4）：GPU-assisted I/O · 解耦 GPU/CPU 布局\ncache-aware 调度：延迟发车 + 均衡组批 + Bubble Filling",
        fc="#e3f2fd", ec="#1565c0", fs=8.8, bold=True)
    arrow(ax, (3.5, 3.9), (3.5, 3.56), color="#1565c0")
    box(ax, 0.3, 2.6, 6.4, 1.2,
        "论文口径（摘要原文）：长上下文 TTFT 最高 5×（对 vLLM+LMCache）\n3.75×（对 TRT-LLM）· 短上下文不劣化 · deployed in production",
        fc="#ffe082", ec="#b26a00", fs=8.8, bold=True)

    # 右：案二细节
    box(ax, 7.3, 5.55, 6.4, 1.3,
        "问题：125B 权重 vs 12 GB 显存；每 token 只激活 10/24,576 专家\n慢的不是算，是拿（weight streaming）",
        fc="#fffde7", ec="#b26a00", fs=8.8)
    arrow(ax, (10.5, 5.5), (10.5, 5.16), color="#b26a00")
    box(ax, 7.3, 3.95, 6.4, 1.2,
        "专家三层：显存=热专家数千个 · RAM=全量 24,576 · SSD=查表\n藏延迟两板斧：8K 大块 prefill + 猜-验投机 1.6-1.8×",
        fc="#e8f5e9", ec="#2e7d32", fs=8.8, bold=True)
    arrow(ax, (10.5, 3.9), (10.5, 3.56), color="#2e7d32")
    box(ax, 7.3, 2.6, 6.4, 1.2,
        "README 自测口径（5070/64GB，4K 输出 32K 输入）：\nQ2_0 94 tok/s · IQ3_S 53 tok/s（≈1.77×）· SSD 档掉到 7-8.5 tok/s",
        fc="#ffe082", ec="#b26a00", fs=8.8, bold=True)

    # 中间合流点
    box(ax, 0.3, 1.55, 13.4, 0.75,
        "合流：都是「放置（谁在哪层）× 搬运（多快）× 调度（何时搬/何时算）」三件事 —— 案一证明调度层单独能拿最高 5× TTFT（论文口径）",
        fc="#f3e5f5", ec="#6a1b9a", fs=9.2, bold=True)
    arrow(ax, (3.5, 2.6), (3.5, 2.36), color="#6a1b9a")
    arrow(ax, (10.5, 2.6), (10.5, 2.36), color="#6a1b9a")

    # 底部边界
    box(ax, 0.3, 0.3, 13.4, 0.95,
        "边界（窗口不当地基）：案二五条 = 单模型绑定 / 0.1.x 新项目 / 自测无第三方口径 / AMD 半受限（Win 读图不可用）/ 首启锁 35-55 GB RAM\n"
        "引用三步：分口径（论文/官方/自报）→ 拆条件（硬件·模型·负载·版本）→ 对回自家尺子（Roofline/传输账/控制变量）",
        fc="#ffcdd2", ec="#c62828", fs=8.8, bold=True)

    fig.savefig(os.path.join(OUT, "fig1_dual_strata.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_card():
    """任务 A 数字核验卡的口径清单（零成本打印）"""
    rows = [
        ("案一 5x", "论文摘要", "对 vLLM+LMCache, long-context, up-to 峰值", "arXiv:2508.18572"),
        ("案一 3.75x", "论文摘要", "对 TensorRT-LLM, long-context", "arXiv:2508.18572"),
        ("案一 不劣化", "论文摘要", "short-context performance", "arXiv:2508.18572"),
        ("案二 94 tok/s", "README 自测", "5070/64GB/Q2_0/引擎0.1.36/4K出32K入", "GitHub README"),
        ("案二 53 tok/s", "README 自测", "5070/64GB/IQ3_S/引擎0.1.26/4K出32K入", "GitHub README"),
        ("案二 1.6-1.8x", "README 口径", "猜-验投机解码, same answer", "GitHub README"),
        ("案二 7-8.5 tok/s", "README 自测", "UD-Q4_K_XL 在 64GB 机器从 SSD 边读边答", "GitHub README"),
    ]
    print(f"{'数字':<16}{'口径类型':<12}{'条件':<46}{'来源'}")
    for r in rows:
        print(f"{r[0]:<16}{r[1]:<12}{r[2]:<46}{r[3]}")
    print("\n最经不起追问的数字：94 tok/s（自测 + 五重绑定条件）；引用必须带硬件三要素。")
    print(f"算术核对：94/53 = {94/53:.2f}x（正文中 1.77x 口径）")


if __name__ == "__main__":
    print_card()
    fig1()
    print(f"\n已生成 -> {OUT}")
