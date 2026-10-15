# -*- coding: utf-8 -*-
"""ch07 配图：fig0 层次示意 / fig1 roofline / fig2 L2+PCIe / fig3 coalescing+occupancy"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
BLUE, RED, GREEN, ORANGE, PURPLE, GRAY = "#4C78A8","#D85C5C","#54A273","#E8943A","#8467A8","#8A8A8A"
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch07"))
os.makedirs(OUT, exist_ok=True)

bw  = json.load(open("bandwidth_results.json"))
gm  = json.load(open("gemm_results.json"))
tr  = json.load(open("triton_results.json"))
sp  = json.load(open("gpu_specs.json"))

# ============ fig0：硬件层次示意图 ============
def fig0():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.6))
    axL.set_xlim(0,10); axL.set_ylim(0,10); axL.axis("off")
    axR.set_xlim(0,10); axR.set_ylim(0,10); axR.axis("off")

    # 左：计算层次（倒金字塔/嵌套盒）
    boxes = [
        (1.0, 7.6, 8.0, 1.9, "GPU 芯片（Ada AD102 阉割版，cc 8.9）", BLUE),
        (1.7, 5.4, 6.6, 1.9, "114 个 SM（流多处理器）——114 条并行流水线", "#6E93BC"),
        (2.4, 3.2, 5.2, 1.9, "每 SM：128 CUDA Core + 4 Tensor Core\n128 个 FP32 核 / 第4代张量核", "#8FAECB"),
        (3.1, 1.0, 3.8, 1.9, "Warp = 32 线程\n（GPU 调度的最小舰队）", PURPLE),
    ]
    for x,y,w,h,t,c in boxes:
        axL.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",
                     facecolor=c, edgecolor="white", lw=1.5))
        axL.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=10.5,
                 color="white", weight="bold")
    axL.text(5, 9.75, "计算侧：车间 → 流水线 → 工位 → 32人班组",
             ha="center", fontsize=13, weight="bold")
    axL.annotate("", xy=(5,0.9), xytext=(5,9.3),
                 arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0))

    # 右：存储金字塔（越上越快越贵越小）
    levels = [
        ("寄存器 Registers", "每线程私有 · 0 等待 · 【规格】64K×32bit/SM", 0.6, PURPLE),
        ("共享内存 Shared Memory / SRAM", "SM 内 100+ KB 级，~数 TB/s（Triton 里的 tile 住这）", 2.3, "#9A7FB8"),
        ("L2 Cache", f"全片共享 {sp['l2_mb_measured']:.0f} MiB · 实测最高 3,563 GB/s", 4.2, ORANGE),
        ("GDDR6X 显存", f"24 GiB · 规格 1,008 GB/s · 实测 918 GB/s（91%）", 6.4, BLUE),
        ("主机内存（经 PCIe 4.0 x16）", "实测 H2D 18.6 GB/s —— 比显存慢 49 倍", 8.4, GRAY),
    ]
    for name, sub, y, c in levels:
        axR.add_patch(FancyBboxPatch((0.55,y),8.25,1.35,boxstyle="round,pad=0.05",
                     facecolor=c, edgecolor="white", lw=1.2))
        axR.text(1.0, y+0.88, name, fontsize=11.5, weight="bold", color="white", va="center")
        axR.text(1.0, y+0.38, sub, fontsize=9.3, color="white", va="center")
    axR.text(5, 9.75, "存储侧：越往上越快、越小、越贵", ha="center", fontsize=13, weight="bold")
    axR.annotate("", xy=(9.55,9.15), xytext=(9.55,0.85),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
    axR.text(9.55, 0.35, "快 ↑  小 ↑", ha="center", fontsize=10.5)
    fig.suptitle("RTX 4090 D 硬件地图：本章所有数字都挂在这张图上", fontsize=14.5, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(f"{OUT}/fig0_hierarchy.png", dpi=140, bbox_inches="tight"); plt.close()

# ============ fig1：Roofline ============
def fig1():
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    BW, P = 918.0, gm["peak_gemm_tflops"]["bf16_tensor_8192"]
    ridge = P*1e3/BW
    ai = np.logspace(-0.3, 3.0, 400)
    perf = np.minimum(BW*ai/1e3, P)
    ax.plot(ai, perf, color="black", lw=2.2)
    ax.fill_between(ai, 0.01, perf, color="#4C78A8", alpha=0.08)
    ax.axvline(ridge, color=GRAY, ls="--", lw=1.2)
    ax.text(ridge+6, 42, f"拐点 ≈ {ridge:.0f} FLOP/Byte\n（与第 3 章 159 互证）",
            fontsize=9.8, color="#333", rotation=90, va="center")
    ax.text(300, P*1.22, f"算力顶 bf16 Tensor Core：{P:.0f} TFLOPS（8192³ 实测）",
            fontsize=11, color=RED, weight="bold", ha="center")
    ax.text(0.62, 0.34, "带宽顶：918 GB/s（512MiB copy 实测）",
            fontsize=10.5, color=BLUE, weight="bold", rotation=29)

    def pts(rows, c, m, label, annotate_ms=()):
        x = [r["ai_flop_per_byte"] for r in rows]
        y = [r["tflops"] for r in rows]
        ax.plot(x, y, f"{m}-", color=c, lw=1.6, ms=8, label=label, zorder=5)
        for r in rows:
            if r["M"] in annotate_ms:
                ax.annotate(f"M={r['M']}", (r["ai_flop_per_byte"], r["tflops"]),
                            textcoords="offset points", xytext=(0,-14), fontsize=8.5,
                            ha="center", color=c)
    pts(gm["lmhead_M896_N151936_bf16"], RED, "o",
        "lm_head 形 (M,896)×(896,151936)：decode 主犯", annotate_ms=(1,4,16,64,256))
    pts(gm["mlp_M896_N4864_bf16"], GREEN, "s", "MLP 投影形 (M,896)×(896,4864)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 1000); ax.set_ylim(0.3, 300)
    ax.set_xlabel("算术强度 AI（FLOP / Byte）", fontsize=12)
    ax.set_ylabel("实测性能（TFLOPS）", fontsize=12)
    ax.set_title("Roofline 本机坐实：decode 点全贴在带宽斜坡上，M≥512 才摸到算力顶",
                 fontsize=13.5, weight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10.5)
    r0 = gm["lmhead_M896_N151936_bf16"][0]
    ax.annotate(f"单请求 decode：{r0['tflops']} TFLOPS\n仅用峰值 {r0['tflops']/P*100:.1f}%，带宽 {r0['eff_gbs']:.0f} GB/s 顶满",
                xy=(r0["ai_flop_per_byte"], r0["tflops"]), xytext=(3.2, 0.55),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_roofline.png", dpi=140); plt.close()

# ============ fig2：L2 悬崖 + 各级带宽柱 ============
def fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8),
                                   gridspec_kw={"width_ratios":[1.35,1]})
    rows = bw["l2_sweep"]
    x = [r["size_mib"] for r in rows]; y = [r["gbs"] for r in rows]
    ax1.plot(x, y, "o-", color=ORANGE, lw=2.2, ms=7)
    ax1.set_xscale("log", base=2)
    ax1.axvspan(1, sp["l2_mb_measured"], color=ORANGE, alpha=0.10)
    ax1.axvline(sp["l2_mb_measured"], color=GRAY, ls="--")
    ax1.text(sp["l2_mb_measured"]*1.18, 3000, "L2 容量 72 MiB", fontsize=10.5, color="#444")
    for xi, yi in [(x[5],y[5]),(x[7],y[7])]:
        ax1.annotate(f"{yi:,} GB/s", (xi,yi), textcoords="offset points",
                     xytext=(0,10), ha="center", fontsize=10.5, weight="bold")
    ax1.text(2.2, 250, "工作集住得进 L2\n（最快 3,563 GB/s）", fontsize=10.5, color=ORANGE)
    ax1.text(260, 250, "溢出到 GDDR6X\n稳定 ~920 GB/s", fontsize=10.5, color=BLUE)
    ax1.set_xlabel("工作集大小（MiB，log2 轴）", fontsize=11.5)
    ax1.set_ylabel("实测有效带宽（GB/s）", fontsize=11.5)
    ax1.set_title("L2 悬崖：32MiB 还在 3,563，128MiB 直接掉到 923", fontsize=12.5, weight="bold")
    ax1.grid(True, ls=":", alpha=0.5); ax1.set_ylim(0, 4000)

    b = bw["big_tensor"]; pc = bw["pcie"]
    names = ["sum\n(只读)", "copy\n读1写1", "a+b\n读2写1", "PCIe\nH2D"]
    vals = [b["sum_read_gbs"], b["copy_gbs"], b["add_gbs"], pc["h2d_gbs"]]
    cols = [GREEN, BLUE, "#6E93BC", GRAY]
    bars = ax2.bar(names, vals, color=cols, width=0.62)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, v+12, f"{v:g}",
                 ha="center", fontsize=11, weight="bold")
    ax2.axhline(1008.1, color=RED, ls="--", lw=1.3)
    ax2.text(3.35, 1025, "规格顶 1,008", color=RED, fontsize=9.5, ha="right")
    ax2.set_ylabel("GB/s", fontsize=11.5)
    ax2.set_title("大张量（512MiB）达规格 91–94%；\n过 PCIe 只剩 18.6（慢 49×）",
                  fontsize=12.5, weight="bold")
    ax2.set_ylim(0, 1150)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_l2_pcie.png", dpi=140); plt.close()

# ============ fig3：coalescing + occupancy + launch ============
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8))
    co = tr["coalescing_32M_elems"]
    x = [r["byte_gap"] for r in co]; y = [r["useful_gbs"] for r in co]
    ax1.plot(x, y, "o-", color=PURPLE, lw=2.2, ms=8)
    for r in co:
        ax1.annotate(f"{r['useful_gbs']:g}", (r["byte_gap"], r["useful_gbs"]),
                     textcoords="offset points", xytext=(0,9),
                     ha="center", fontsize=9.5)
    ax1.set_xscale("log", base=2); ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in x])
    ax1.axhline(128, color=GRAY, ls="--", lw=1)
    ax1.text(128, 150, "128B = 一条缓存行（warp 合并访问的目标）", fontsize=9.3, color="#555")
    ax1.set_xlabel("同 warp 相邻线程访问地址间隔（Byte）", fontsize=11)
    ax1.set_ylabel("有用带宽（GB/s）", fontsize=11)
    ax1.set_title("访存合并：连续 917 GB/s → 跨步 128B 只剩 62（14.7×）",
                  fontsize=12.3, weight="bold")
    ax1.grid(True, ls=":", alpha=0.5); ax1.set_ylim(0, 1050)

    oc = tr["occupancy_114blocks_thread1elem"]
    xw = [r["num_warps"] for r in oc]; yw = [r["gbs"] for r in oc]
    ax2.plot(xw, yw, "s-", color=GREEN, lw=2.2, ms=8, label="114 块各占一个 SM（每线程每拍只取1个元素）")
    for r in oc:
        ax2.annotate(f"{r['gbs']:g}", (r["num_warps"], r["gbs"]),
                     textcoords="offset points", xytext=(0,9), ha="center", fontsize=9.5)
    ax2.set_xscale("log", base=2); ax2.set_xticks(xw); ax2.set_xticklabels([str(v) for v in xw])
    ax2.set_xlabel("每个 Triton program 的 warp 数", fontsize=11)
    ax2.set_ylabel("有效带宽（GB/s）", fontsize=11)
    ax2.set_title("占用率：warp 是等显存时的替补——1 个 warp 仅 111，16 个补到 825",
                  fontsize=12.3, weight="bold")
    ax2.grid(True, ls=":", alpha=0.5); ax2.set_ylim(0, 1050); ax2.legend(loc="lower right", fontsize=9.2)

    ln = bw["launch"]
    fig.text(0.5, -0.02,
             f"另一项固定税：单个小 kernel eager launch {ln['eager_per_launch_us']}µs（CPU 发射）"
             f" vs CUDA Graph 重放 {ln['cuda_graph_per_launch_us']}µs（{ln['speedup']}×）——第 8 章的提速主牌",
             ha="center", fontsize=10.8, color="#333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#F2EDE3", ec="#C9BFA8"))
    fig.tight_layout(rect=[0,0.04,1,1]); fig.savefig(f"{OUT}/fig3_coalesce_occ.png", dpi=140, bbox_inches="tight"); plt.close()

fig0(); fig1(); fig2(); fig3()
print("written to", OUT)
