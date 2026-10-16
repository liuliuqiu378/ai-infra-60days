# ch17 配图生成脚本：3 张程序生成示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch17/make_figures.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch17")
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#cfe2ff", ec="black", fs=9, bold=False, style="round,pad=0.05"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, text="", fs=8.5, color="black", style="-|>", conn="arc3,rad=0",
          dx=0.0, dy=0.25, va="bottom"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 color=color, connectionstyle=conn, lw=1.3))
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + dx, my + dy, text, ha="center", va=va, fontsize=fs, color=color)


def fig1():
    """mini-vLLM 四模块架构与主循环数据流"""
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("图 1 程序示意：mini-vLLM 四模块架构与主循环数据流（约 900 行）", fontsize=13)

    # 左：调度器（只做决策）
    box(ax, 0.4, 3.4, 3.0, 2.2, "", fc="#f4f4f4")
    ax.text(1.9, 5.25, "scheduler.py（约150行）", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 0.7, 4.3, 2.4, 0.7, "请求状态机 NEW→WAITING\n→PREFILL→DECODING→DONE", fc="#cfe2ff", fs=8)
    box(ax, 0.7, 3.5, 2.4, 0.7, "schedule(token_budget)\n先 decode 后 prefill（FCFS）", fc="#cfe2ff", fs=8)
    ax.text(0.4, 3.15, "◀ 只做决策，不碰显存", fontsize=8.5, color="#b26a00")

    # 中：组批层
    box(ax, 4.4, 3.7, 2.4, 1.6, "", fc="#fff3e0")
    ax.text(5.6, 5.0, "engine.py collate()", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 4.6, 3.9, 2.0, 0.9, "拼 prompt 张量\n位置索引 + block table\n→ model_input", fc="#ffe0b2", fs=8)

    # 右：引擎 forward + 块池
    box(ax, 7.6, 3.4, 4.5, 2.4, "", fc="#e8f5e9")
    ax.text(9.85, 5.6, "model.py + kv_cache.py（约330行）", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 7.9, 4.35, 1.9, 1.1, "一次 forward\nprefill 全 logits\ndecode 末位 logits", fc="#d9f2d9", fs=8)
    box(ax, 10.0, 4.35, 1.9, 1.1, "gather(block_table)\n按表取回连续 KV\n→ SDPA attention", fc="#d9f2d9", fs=8)
    box(ax, 8.4, 3.55, 3.3, 0.6, "共享块池 K/V：(num_blocks,16,n_kv,d)", fc="#d9f2d9", fs=8)
    ax.text(4.4, 3.15, "◀ 决策与执行的边界：SchedulerOutput ↔ model_input", fontsize=8.5, color="#b26a00")

    # 下：采样与写回
    box(ax, 4.4, 1.2, 3.4, 1.4, "", fc="#fce4ec")
    ax.text(6.1, 2.3, "sampler.py + engine 写回", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 4.6, 1.35, 3.0, 0.75, "greedy / temperature / top-k\n逐请求采样 + EOS/max_tokens 停判", fc="#f8bbd0", fs=8)
    box(ax, 8.6, 1.2, 3.5, 1.4, "", fc="#f4f4f4")
    ax.text(10.35, 2.3, "完成即回收", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 8.8, 1.35, 3.1, 0.75, "FINISHED → free(全部块)\n新请求 add_request 随时入队", fc="#e0e0e0", fs=8)

    # 数据流箭头
    arrow(ax, (3.4, 4.5), (4.4, 4.5), "sched")
    arrow(ax, (6.8, 4.5), (7.6, 4.5), "model_input")
    arrow(ax, (9.85, 4.3), (6.1, 2.6), "logits", color="#b26a00", conn="arc3,rad=0.25")
    arrow(ax, (4.4, 1.7), (1.9, 3.4), "next_tokens", color="#b26a00", conn="arc3,rad=0.25")
    arrow(ax, (10.35, 2.6), (10.35, 3.4), "free blocks", dy=-0.45, va="top")
    arrow(ax, (8.6, 1.9), (3.4, 1.9), "", conn="arc3,rad=0")
    arrow(ax, (1.9, 1.9), (1.9, 3.4), "advance", color="#b26a00", dx=-0.75, dy=0.1)

    ax.text(6.25, 0.45, "循环五件事：①schedule 决策 → ②collate 组批 → ③forward 写 KV → ④逐请求采样 → ⑤写回/释放/新请求入队",
            ha="center", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=150)
    plt.close(fig)


def fig2():
    """请求生命周期状态机 + 一步调度决策顺序"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.6))
    fig.suptitle("图 2 程序示意：请求生命周期状态机（上）与一步调度的决策顺序（下）", fontsize=13)

    # ---- 上：状态机 ----
    ax1.set_xlim(0, 12.5); ax1.set_ylim(0, 3.4); ax1.axis("off")
    ax1.set_title("状态机只有四个状态：只有 FINISHED 才释放块", fontsize=11)
    states = {"NEW": (0.5, 1.2, "#e0e0e0"), "WAITING": (3.0, 1.2, "#cfe2ff"),
              "PREFILL": (6.0, 1.2, "#ffe0b2"), "DECODING": (9.0, 1.2, "#d9f2d9"),
              "FINISHED": (6.0, 0.2, "#f8bbd0")}
    for name, (x, y, c) in states.items():
        box(ax1, x, y, 2.0, 0.9, name, fc=c, fs=10, bold=True)
    arrow(ax1, (2.5, 1.65), (3.0, 1.65), "add_request()", fs=8, dx=-0.2, dy=0.4)
    arrow(ax1, (5.0, 1.65), (6.0, 1.65), "首次被选中 +\nKV 分配成功", fs=8, dx=-0.55, dy=0.4)
    arrow(ax1, (8.0, 1.65), (9.0, 1.65), "首批前向完成", fs=8, dx=-0.5, dy=0.4)
    arrow(ax1, (9.5, 2.15), (10.5, 2.15), "每步被选中 + 1 token（token 预算内）", fs=8,
          conn="arc3,rad=-1.1", dx=0.0, dy=0.62)
    arrow(ax1, (11.0, 1.65), (11.5, 1.65), "", fs=8)
    arrow(ax1, (11.5, 1.65), (11.5, 0.65), "", conn="arc3,rad=0.35")
    arrow(ax1, (11.5, 0.65), (8.0, 0.65), "EOS 或 max_tokens", fs=8, dx=0.0, dy=0.28)
    arrow(ax1, (6.1, 1.25), (4.9, 1.25), "KV 装不下 → 留在 WAITING",
          fs=8, color="#b26a00", conn="arc3,rad=-0.45", dx=0.0, dy=-0.5, va="top")
    ax1.text(7.0, 0.0, "FINISHED：释放该请求全部 block（每轮后 assert 空闲块数==总块数，抓泄漏）",
             fontsize=9, ha="center", color="#c2185b")

    # ---- 下：一步调度决策 ----
    ax2.set_xlim(0, 12.5); ax2.set_ylim(0, 3.4); ax2.axis("off")
    ax2.set_title("一步（iteration）调度的决策顺序：先 decode 后 prefill", fontsize=11)
    steps = [
        (0.3, "① schedule()\n先装 decode 批\n（每请求1 token）", "#cfe2ff"),
        (2.7, "② 剩余 token 预算\n给 waiting 的 prefill\n（装不下→FCFS 截断）", "#ffe0b2"),
        (5.1, "③ 跨 16 边界的请求\n分配新 block\n（入批时刻之一）", "#d9f2d9"),
        (7.5, "④ 一次 forward\nprefill+decode 混批\nKV 写入块池", "#e1bee7"),
        (9.9, "⑤ 采样 + 停判\n完成请求释放块\n（释放时刻之二）", "#f8bbd0"),
    ]
    for x, t, c in steps:
        box(ax2, x, 1.0, 2.2, 1.5, t, fc=c, fs=8.5)
    for i in range(4):
        arrow(ax2, (steps[i][0] + 2.2, 1.75), (steps[i + 1][0], 1.75))
    ax2.text(6.25, 0.45, "块池只在两个时刻变化：入批分配（prefill 首次 / decode 跨块）与完成释放 —— 面试白板题原型",
             ha="center", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_lifecycle.png"), dpi=150)
    plt.close(fig)


def fig3():
    """一步调度时序：张量形态变化与显存动作"""
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.set_title("图 3 程序示意：一步调度时序——张量形态（上排）与显存动作（下排）", fontsize=13)

    phases = [
        (0.3, "① 调度决策", "每请求: token序列\nblock_table\n状态", "#cfe2ff",
         "无显存动作\n（纯 Python 决策）", "#f4f4f4"),
        (2.8, "② 组批 collate", "batch 张量\ninput_ids (B,)\npositions (B,)", "#ffe0b2",
         "prefill 首次:\nallocate ceil(L/16) 块\n（分配时刻 1）", "#fff3e0"),
        (5.3, "③ forward + KV", "logits\nprefill:(L,vocab)\ndecode:(1,vocab)", "#d9f2d9",
         "KV 写入 k/v_cache\n[block_id, slot]\n（跨16边界→追加块）", "#e8f5e9"),
        (7.8, "④ 采样 + 停判", "next_tokens (B',)\nEOS/max_tokens\n→ finished 标记", "#f8bbd0",
         "无显存动作\n（读 logits 写 token）", "#fce4ec"),
        (10.3, "⑤ 写回 + 回收", "序列拼接\nhash_chain 更新\n新请求入队", "#e0e0e0",
         "FINISHED:\nfree(全部块)\n（释放时刻 2）", "#f4f4f4"),
    ]
    for x, title, tensor, tc, mem, mc in phases:
        ax.add_patch(FancyBboxPatch((x, 3.6), 2.2, 1.9, boxstyle="round,pad=0.05",
                                    facecolor=tc, edgecolor="black", lw=1.2))
        ax.text(x + 1.1, 5.25, title, ha="center", fontsize=9.5, fontweight="bold")
        ax.text(x + 1.1, 4.4, tensor, ha="center", va="center", fontsize=8.5)
        ax.add_patch(FancyBboxPatch((x, 1.0), 2.2, 1.7, boxstyle="round,pad=0.05",
                                    facecolor=mc, edgecolor="#b26a00", lw=1.2))
        ax.text(x + 1.1, 1.85, mem, ha="center", va="center", fontsize=8.5, color="#7a4a00")
        ax.plot([x + 1.1, x + 1.1], [3.6, 2.7], color="#b26a00", lw=1.1, ls="--")
    for i in range(4):
        arrow(ax, (phases[i][0] + 2.2, 4.55), (phases[i + 1][0], 4.55))
        arrow(ax, (phases[i][0] + 2.2, 1.85), (phases[i + 1][0], 1.85), color="#b26a00")

    ax.text(6.25, 3.0, "张量流（上排箭头）", ha="center", fontsize=9, color="#555")
    ax.text(6.25, 0.45, "回环：⑤ 的 next_tokens 成为下一轮 ① 的状态；块池分配/释放只发生在 ② 与 ⑤ 两个时刻",
            ha="center", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_step_timeline.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("ch17 3 张图已生成至", OUT)
