# -*- coding: utf-8 -*-
"""第 15 章配图：程序生成，零外部图片。
图1：TP 通信量随度数增长（公式代入）+ 四把刀通信形态象限
图2：PP 流水线气泡甘特图（M=K vs M=4K）+ 气泡率曲线
数据口径：Megatron SC'21 通信公式、GPipe 气泡公式，均为公式代入，非实测。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "../../images/figures/ch15/"

# ---------- 图 1 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

# 左：TP 每 decode 步全模型发送字节（公式 4·b·s·h·(t-1)/t × 层系数；s=1,b=32,h=8192,2字节,80层,前向2次）
t = np.array([2, 4, 8, 16])
b, s, h, L = 32, 1, 8192, 80
bytes_per_step = 2 * (2 * b * s * h * 2) * (t - 1) / t * L  # 每层2次all-reduce×ring系数×层
mb = bytes_per_step / (1024 ** 2)  # MB=2^20B，与正文 80/120/140 口径一致
ax1.plot(t, mb, "o-", color="#2b6cb0", lw=2.2, ms=9)
for xi, yi in zip(t, mb):
    ax1.annotate(f"{yi:.0f} MB", (xi, yi), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=11, color="#1a4f8b")
ax1.set_xlabel("TP 度 t（卡数）")
ax1.set_ylabel("每 decode 步全模型 all-reduce 发送（MB=2^20B / 卡）")
ax1.set_title("TP 通信量随度数温和增长\n（Llama-3 70B：80层×2次×b=32,s=1,h=8192,bf16，公式代入）", fontsize=12)
ax1.set_xticks(t)
ax1.grid(alpha=0.3)
ax1.axvspan(4.5, 9, color="#fed7d7", alpha=0.45)
ax1.text(6.5, mb.min() + 3, "单机 NVLink\n可承受区", ha="center", fontsize=10, color="#9b2c2c")

# 右：四象限——通信频率 × 单次消息大小
quadrant = {
    "TP": (9.0, 1.6, "#2b6cb0"),   # 高频(每层2次)、消息中等
    "PP (decode)": (3.0, 0.8, "#2f855a"),  # 低频(每token每边界)、消息小
    "SP": (7.5, 1.2, "#805ad5"),   # 与TP同量级（搬家长宽一致）
    "EP": (6.0, 3.2, "#c05621"),   # 每MoE层2次、消息变长（路由波动）
}
for name, (x, y, c) in quadrant.items():
    ax2.scatter(x, y, s=520, color=c, alpha=0.85, zorder=3)
    ax2.annotate(name, (x, y), textcoords="offset points", xytext=(0, -26),
                 ha="center", fontsize=12, fontweight="bold", color=c)
ax2.annotate("路由热点 → 消息变长", (6.0, 3.2), textcoords="offset points",
             xytext=(18, 10), fontsize=9, color="#7b341e")
ax2.set_xlabel("通信频率（高 →）  每层/每token/每MoE层")
ax2.set_ylabel("单次消息大小（小 → 大）")
ax2.set_xlim(1, 11); ax2.set_ylim(0, 4.2)
ax2.set_title("四把刀的通信形态：选刀 = 选频率 × 消息大小", fontsize=12)
ax2.grid(alpha=0.3)
ax2.axhline(2.0, color="#a0aec0", ls="--", lw=1)
ax2.axvline(6.5, color="#a0aec0", ls="--", lw=1)

fig.suptitle("第 15 章 图 1｜四把刀通信账（左：公式代入字节账；右：频率×尺寸定性象限）", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(OUT + "fig1_comm_ledger.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- 图 2 ----------
fig, gs = plt.subplots(1, 2, figsize=(13.5, 6.2))

# 左：气泡甘特图 K=4，M=4 vs M=16
def draw_pipeline(ax, K, M, title, color_alive, color_bubble):
    # 第 m 个 micro-batch 在 stage k 的开始时间 = m + k（单位步），时长 1
    for m in range(M):
        for k in range(K):
            t0 = m + k
            ax.add_patch(Rectangle((t0, K - 1 - k), 1, 1,
                                   facecolor=color_alive, edgecolor="white", lw=1.2))
    span = M + K - 1
    ax.set_xlim(0, span); ax.set_ylim(0, K)
    ax.set_xlabel("时间（步）")
    ax.set_ylabel("Stage")
    ax.set_yticks([K - 1 - k for k in range(K)], [f"卡{k+1}(层段)" for k in range(K)])
    bubble_frac = (K - 1) / (M + K - 1)
    ax.set_title(f"{title}\n气泡率=(K−1)/(M+K−1)={bubble_frac:.1%}", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    # 空白处即气泡：示意填充
    ax.add_patch(Rectangle((0, 0), span, K, facecolor="none", edgecolor="#c53030",
                           ls=":", lw=1.5))

ax_a, ax_b = gs
draw_pipeline(ax_a, 4, 4, "K=4 段, M=4 micro-batch（M=K）", "#2b6cb0", "#c53030")
draw_pipeline(ax_b, 4, 16, "K=4 段, M=16 micro-batch（M=4K）", "#2f855a", "#c53030")
# 手动标气泡（左上/右下三角为空白）
ax_a.text(5.5, 3.4, "气泡\n(空闲)", color="#c53030", fontsize=11, fontweight="bold", ha="center")
ax_a.text(1.5, 0.5, "气泡\n(空闲)", color="#c53030", fontsize=11, fontweight="bold", ha="center")
ax_b.text(17.0, 3.4, "气泡区（占比小了很多）", color="#c53030", fontsize=10, ha="left")
ax_b.text(1.2, 0.4, "气泡", color="#c53030", fontsize=10, ha="center")

fig.suptitle("第 15 章 图 2｜PP 流水线气泡：几何视图与 GPipe 公式（K=阶段数, M=micro-batch 数）", fontsize=13)
plt.tight_layout()
fig.savefig(OUT + "fig2_pipeline_bubble.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("figures written to", OUT)
