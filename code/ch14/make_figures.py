#!/usr/bin/env python3
"""ch14 配图生成（Continuous Batching 与调度器）

前置：pip install matplotlib（中文字体 WenQuanYi Micro Hei）
用法：python make_figures.py
输出：../../images/figures/ch14/fig1_static_vs_continuous.png,
      ../../images/figures/ch14/fig2_schedule_loop.png

数据口径（零性能跑测，全部为结构性示意）：
- fig1 请求长度与到达时刻为教学示例；浪费比例由示例时序直接算出（算账，非实测），
  用于说明 Orca 论文（OSDI '22）"request-level scheduling 的批次只能整批进出"的机制
- fig2 调度循环结构与行号来自 vLLM commit 642c18d 的
  vllm/v1/core/sched/scheduler.py（schedule():621、token_budget:641、
  running 循环:714、waiting 循环:967、SchedulerOutput:1579、
  _preempt_request:1649），逐条 grep 核验（检索 2026-10-16）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "../../images/figures/ch14"


# ---------------------------------------------------------------- fig1
def make_fig1():
    """static batching vs continuous batching 的批次时序对比（算账示意）。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 7.6))

    # 教学示例：4 个请求，先到 A、B；C、D 在第 5 步到达；每步每请求生成 1 token
    # 每步 1 token 只为画图简洁，机制与真实步长无关
    reqs = [
        # (name, 到达步, 总输出步, 颜色)
        ("A", 0, 10, "#4CAF50"),
        ("B", 0, 6, "#2196F3"),
        ("C", 5, 8, "#FF9800"),
        ("D", 5, 12, "#9C27B0"),
    ]

    def draw(ax, title, mode):
        ax.set_title(title, fontsize=12.5, weight="bold", pad=10)
        ax.set_xlim(-0.6, 23)
        ax.set_ylim(-1.0, 4.6)
        ax.set_yticks([])
        ax.set_xticks(range(0, 23, 2))
        ax.set_xlabel("调度步（iteration）", fontsize=10)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        row = 0
        for name, arrive, length, color in reqs[::-1]:
            y = row
            if mode == "static":
                # 静态：批次 A/B 整批跑完（10 步，以最长者为准），C/D 只能等；
                # B 第 6 步就结束，仍占位到批次结束
                if name in ("A", "B"):
                    start, end = 0, 10
                else:
                    start, end = 10, 22
                ax.add_patch(Rectangle((start, y - 0.3), end - start, 0.6,
                                       fc=color, ec="white", alpha=0.85))
                if start > 0:
                    ax.add_patch(Rectangle((0, y - 0.3), start, 0.6,
                                           fc="#E0E0E0", ec="#BDBDBD",
                                           hatch="//", alpha=0.9))
                    ax.text(start / 2, y, "排队 10 步\n零产出", ha="center",
                            va="center", fontsize=8, color="#616161")
                ax.text(-0.4, y, name, ha="right", va="center",
                        fontsize=11, weight="bold", color=color)
                if name == "B":
                    ax.add_patch(Rectangle((6, y - 0.3), 4, 0.6,
                                           fc="white", ec=color,
                                           linestyle="--", lw=1.4))
                    ax.text(8, y, "已结束\n仍占位", ha="center", va="center",
                            fontsize=8, color=color)
            else:
                # 连续：谁到谁进，谁完谁走
                start, end = arrive, arrive + length
                ax.add_patch(Rectangle((start, y - 0.3), end - start, 0.6,
                                       fc=color, ec="white", alpha=0.85))
                if arrive > 0:
                    ax.add_patch(Rectangle((0, y - 0.3), arrive, 0.6,
                                           fc="#E0E0E0", ec="#BDBDBD",
                                           hatch="//", alpha=0.9))
                    ax.text(arrive / 2, y, f"等 {arrive} 步", ha="center",
                            va="center", fontsize=8, color="#616161")
                ax.text(-0.4, y, name, ha="right", va="center",
                        fontsize=11, weight="bold", color=color)
            row += 1

        if mode == "static":
            ax.axvline(10, color="#B71C1C", lw=1.6, ls="--")
            ax.text(10.2, 4.2, "批 1 到这一步才整批结束，批 2 才开始",
                    fontsize=10, color="#B71C1C", weight="bold")
        else:
            ax.axvline(5, color="#1B5E20", lw=1.6, ls="--")
            ax.text(5.2, 4.2, "C/D 第 5 步到达即入批，与 A/B 同批执行",
                    fontsize=10, color="#1B5E20", weight="bold")

    draw(ax1, "Static batching（request-level）：批次只能整批进出，先到的被后到的拖住",
         "static")
    draw(ax2, "Continuous batching（iteration-level，Orca OSDI '22）：每步重排，随到随进、完事即走",
         "continuous")

    fig.suptitle("请求级调度 vs 迭代级调度：浪费到底在哪（示例时序，算账非实测）",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/fig1_static_vs_continuous.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- fig2
def make_fig2():
    """vLLM v1 schedule() 单步调度循环结构（按源码流程绘制）。"""
    fig, ax = plt.subplots(figsize=(13.2, 8.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=10, tc="black", weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06",
                                    fc=fc, ec="#455A64", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, weight=weight, linespacing=1.4)

    def arrow(x1, y1, x2, y2, label="", color="#37474F"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=color, lw=1.6))
        if label:
            ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, label,
                    fontsize=8.6, color=color, weight="bold")

    box(3.4, 9.1, 3.2, 0.7,
        "schedule() 入口（scheduler.py:621）\n本步配额 token_budget = 2048（OpenAI 服务默认）",
        "#37474F", 9, "white", "bold")

    box(0.4, 7.6, 2.9, 1.1,
        "① 先排 RUNNING（:714）\n每个在跑请求要几个 token\n= num_tokens − num_computed",
        "#BBDEFB", 9)
    box(3.55, 7.6, 2.9, 1.1,
        "② 再排 WAITING（:967）\nFIFO 队头 peek\n够预算才真正入批",
        "#C8E6C9", 9)
    box(6.7, 7.6, 2.9, 1.1,
        "③ 打包 SchedulerOutput（:1579）\n每个请求各几个 token\n交给 worker 执行",
        "#FFE0B2", 9)

    arrow(4.5, 9.1, 1.85, 8.7)
    arrow(5.5, 9.1, 4.7, 8.72)
    ax.text(5.68, 8.88, "running 排完\n仍有预算", fontsize=8,
            color="#37474F", weight="bold", ha="left", va="center",
            linespacing=1.3)
    arrow(6.45, 8.15, 6.7, 8.15)
    ax.text(6.575, 7.32, "预算=0 或队空", fontsize=8.6, color="#37474F",
            weight="bold", ha="center", va="center")

    box(0.4, 5.5, 2.9, 1.2,
        "allocate_slots 拿不到块\n（kv_cache_manager.py:371）",
        "#FFCDD2", 9)
    box(0.4, 3.9, 2.9, 1.35,
        "抢占受害者（:849-857）\nFCFS：running[-1]（最后到的）\npriority：最低优先级者\n_preempt_request（:1649）",
        "#FFCDD2", 9)
    arrow(1.85, 7.6, 1.85, 6.7, "拿不到")

    box(3.55, 3.9, 2.9, 1.35,
        "释放全部块 + 状态 PREEMPTED\nnum_computed_tokens = 0（:1671）\nprepend 回 waiting（:1691）\nnum_preemptions += 1（:1686）",
        "#F8BBD0", 9)
    arrow(3.3, 4.6, 3.55, 4.6)
    ax.add_patch(FancyArrowPatch((5.0, 5.28), (5.0, 7.55),
                                 arrowstyle="-|>",
                                 mutation_scale=16,
                                 color="#AD1457", lw=1.6))
    ax.text(4.85, 6.3, "回 waiting 重新 prefill（v1 只重算不 swap）",
            fontsize=8.4, color="#AD1457", weight="bold", ha="center",
            va="center", rotation=90, linespacing=1.4)

    box(3.4, 1.5, 6.2, 1.6,
        "引擎主循环 core.py step(:657)\n"
        "schedule → execute_model → update_from_output(:2107)\n"
        "→ 结束的请求 free 块、新请求下步补进 → 回到 schedule",
        "#ECEFF1", 9.5)

    box(0.4, 1.5, 2.6, 1.6,
        "两条预算闸门\n"
        "token_budget（max_num_scheduled_tokens）\n"
        "input_budget（max_num_batched_tokens）",
        "#ECEFF1", 9)
    ax.add_patch(FancyArrowPatch((3.0, 2.3), (3.4, 2.3),
                                 arrowstyle="-|>",
                                 mutation_scale=16,
                                 color="#37474F", lw=1.6))

    ax.set_title("vLLM v1 单步调度循环：两张队列 + 一条预算 + 一个抢占出口（行号 @642c18d）",
                 fontsize=13.5, weight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_schedule_loop.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    make_fig1()
    make_fig2()
    print(f"done -> {OUT}/fig1_static_vs_continuous.png, fig2_schedule_loop.png")
