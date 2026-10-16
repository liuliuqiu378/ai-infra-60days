# ch21 配图生成脚本：2 张程序示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch21/make_figures.py
# 图1：PD 分离流水线与三家传输语义（结构示意）
# 图2：双面板——左=三场景 KV 传输时间（2.2 节零成本算术），右=下沉判据分界曲线
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch21")
os.makedirs(OUT, exist_ok=True)

# ---------- 零成本算账参数（全部来自本书登记值，见 RESULTS.md） ----------
KIB = 1024
KV_PER_TOKEN = 56 * KIB          # 7B GQA 口径，ch04 登记值
BW = {"NVLink": 50e9, "PCIe": 16e9}   # ch07/ch08 登记口径
# 正文 2.1/2.2 节登记的 KV 量（MiB 数值按 MB 记的既定口径，四舍五入到正文所列值）
PROMPTS = [("200 token", 200, 11), ("4k token", 4096, 224), ("32k token", 32768, 1800)]
# 下沉判据参数：搬运 = L0 + 2N·b/BW；重算 = F + N·(1/r)
L0 = 0.5e-3        # 每块握手/启动固定开销（建模假设，见 RESULTS）
F = 15e-3          # 重算固定开销：7B bf16 权重 13.5 GB ÷ 900 GB/s ≈ 15 ms
PER_TOKEN = 170.7e-6   # 7B prefill 计算：14 GFLOPs ÷ 82 TFLOPS（ch03 登记口径）
BW_CROSS = 2 * KV_PER_TOKEN / PER_TOKEN   # 斜率相等的交叉带宽


def print_account():
    print("=== 2.2 节传输表（零成本算术，口径与正文一致） ===")
    print(f"{'场景':<10}{'KV量':>10}{'NVLink 50GB/s':>16}{'PCIe 16GB/s':>14}")
    for name, _, kv in PROMPTS:  # kv 单位 MB（正文所列值）
        t_n = kv * 1e6 / BW["NVLink"] * 1e3
        t_p = kv * 1e6 / BW["PCIe"] * 1e3
        print(f"{name:<10}{kv:>8.0f} MB{t_n:>14.1f} ms{t_p:>12.0f} ms")
    print(f"\n=== 下沉判据交叉带宽（斜率相等点） ===\n"
          f"2b/r = 2 x {KV_PER_TOKEN} B x (1/{PER_TOKEN*1e3:.1f} ms) "
          f"= {BW_CROSS/1e9:.3f} GB/s")
    print("链路高于该值：任何块长搬运都更便宜；低于该值：超过 N* 的块重算更快。")


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
    """PD 分离流水线：编排层 → prefill 池 → NIXL → decode 池 + 三家传输语义"""
    fig, ax = plt.subplots(figsize=(13.5, 8.8))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.8); ax.axis("off")
    ax.set_title("图 1 程序示意：PD 分离三步流水线与三家后端传输语义（结构示意，非实测数据）",
                 fontsize=12.5)

    # 顶部编排层
    box(ax, 2.6, 7.7, 8.3, 0.75,
        "编排层：Dynamo PrefillRouter（KV-aware 路由）/ Mooncake 调度 / DistServe 资源配比",
        fc="#ffe082", bold=True, fs=9.5)

    # 请求入口
    box(ax, 0.3, 6.1, 2.1, 0.9, "请求到达\n（长 prompt + 流式输出）", fc="#e0e0e0", fs=8.8, bold=True)
    arrow(ax, (2.4, 6.55), (3.0, 6.55), fs=8.5)

    # 第 1 步 prefill 池
    box(ax, 3.0, 5.75, 3.6, 1.6,
        "① Prefill 池\n小 TP · compute-bound\n管 TTFT（ch03）\n产出 KV Cache",
        fc="#cfe2ff", fs=9.2, bold=True)

    # 第 2 步 NIXL 传输
    arrow(ax, (6.6, 6.55), (7.6, 6.55), fs=8.5)
    box(ax, 7.6, 5.75, 3.4, 1.6,
        "② NIXL 传输 KV\nVRAM→VRAM 直写\n64–128 token/块\n非阻塞 · NVLink/RDMA",
        fc="#fff9c4", ec="#b26a00", fs=9.2, bold=True)

    # 第 3 步 decode 池
    arrow(ax, (11.0, 6.55), (11.6, 6.55), fs=8.5)
    box(ax, 11.6, 5.75, 1.7, 1.6, "③ Decode\n池\n大 TP\n管 TPOT",
        fc="#c8e6c9", fs=8.8, bold=True)

    # 传输语义三家
    box(ax, 0.3, 3.55, 4.1, 1.5,
        "SGLang：bootstrap_info\nprefill 作后台任务\ndecode 与传输并行开工 → 可流水",
        fc="#c8e6c9", ec="#2e7d32", fs=8.8, bold=True)
    box(ax, 4.7, 3.55, 4.1, 1.5,
        "vLLM：kv_transfer_params\nprefill 同步执行\ndecode 等 prefill 完成 → 串行",
        fc="#ffe082", ec="#b26a00", fs=8.8, bold=True)
    box(ax, 9.1, 3.55, 4.1, 1.5,
        "TRT-LLM：opaque_state\n元数据序列化传输\n同样同步等待",
        fc="#ffe082", ec="#b26a00", fs=8.8, bold=True)
    ax.text(6.75, 5.25, "三家后端传输语义不同——决定传输是否计入 TTFT 串行段：",
            ha="center", fontsize=9, style="italic")
    for x in (2.35, 6.75, 11.15):
        arrow(ax, (x, 5.05), (x, 5.05 + 0.66), color="#555555")

    # 重叠手段
    box(ax, 0.3, 2.0, 12.9, 0.85,
        "把传输藏进计算：逐层流水（layer-wise streaming，传第 L 层时算第 L+1 层）"
        " + NIXL 非阻塞 + 传输与 prefill 尾部重叠",
        fc="#e3f2fd", ec="#1565c0", fs=9, bold=True)
    arrow(ax, (6.75, 2.85), (6.75, 3.5), color="#1565c0")

    box(ax, 0.3, 0.35, 12.9, 1.1,
        "代价：请求跨卡 → KV 必须搬运。传输时间 = KV 量 ÷ 链路带宽（本章 2.2 节账）："
        "200 token 0.2 ms 可忽略，32k token 走 NVLink 36 ms / 走 PCIe 113 ms。\n"
        "判据：长 prompt 占比高 + TTFT/TPOT 分开盯 + NVLink/RDMA 打底 → 才值得拆",
        fc="#ffcdd2", ec="#c62828", fs=9, bold=True)

    fig.savefig(os.path.join(OUT, "fig1_pd_pipeline.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2():
    """双面板：左=三场景传输时间（算术），右=下沉判据分界曲线（不等式可视化）"""
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.4))
    fig.suptitle("图 2 程序示意：左=KV 传输时间三场景账（零成本算术）；右=下沉判据搬 vs 算分界（7B GQA 口径）",
                 fontsize=12)

    # ---- 左：传输时间柱状图 ----
    xs = range(len(PROMPTS))
    t_nl = [kv * 1e6 / BW["NVLink"] * 1e3 for _, _, kv in PROMPTS]
    t_pc = [kv * 1e6 / BW["PCIe"] * 1e3 for _, _, kv in PROMPTS]
    w = 0.36
    b1 = ax1.bar([x - w / 2 for x in xs], t_nl, w, label="NVLink (50 GB/s)", color="#1565c0")
    b2 = ax1.bar([x + w / 2 for x in xs], t_pc, w, label="PCIe (16 GB/s)", color="#ef6c00")
    ax1.set_yscale("log")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([n for n, _, _ in PROMPTS])
    ax1.set_ylabel("KV 传输时间 (ms，对数轴)")
    ax1.set_title("三场景 KV 传输量 × 两种链路（2.2 节表）", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.bar_label(b1, fmt="%.1f ms", fontsize=8.5, padding=2)
    # 0.7 ms 量级保留一位小数，10 ms 以上四舍五入取整（112.5→113，与正文 2.2 表一致）
    ax1.bar_label(b2, labels=[f"{v:.1f} ms" if v < 10 else f"{v + 0.5:.0f} ms" for v in t_pc],
                  fontsize=8.5, padding=2)
    ax1.set_ylim(0.1, 400)
    ax1.axhline(10, color="#c62828", ls="--", lw=1.2)
    ax1.text(2.42, 11, "10 ms 线", color="#c62828", fontsize=8.5, ha="right")
    ax1.text(0.02, 0.92, "短 prompt 可忽略；32k 跨 PCIe 上 100 ms 级\n→ 长上下文才是 PD 主战场",
             transform=ax1.transAxes, fontsize=8.8, va="top",
             bbox=dict(boxstyle="round", fc="#fffde7", ec="#b26a00"))

    # ---- 右：下沉判据分界曲线 ----
    # 搬运 T = L0 + 2N·b/BW；重算 T = F + N·per_token。相等解出 N*(BW)
    bw = np.logspace(np.log10(0.05), np.log10(100), 600) * 1e9
    slope_t = 2 * KV_PER_TOKEN / bw
    denom = slope_t - PER_TOKEN
    n_star = np.where(denom > 0, (F - L0) / np.where(denom > 0, denom, 1), np.inf)
    n_cap = np.minimum(n_star, 4e5)
    ax2.fill_between(bw / 1e9, 4, n_cap, color="#c8e6c9", label="搬运分层更快（2×KV÷带宽 < 重算）")
    ax2.fill_between(bw / 1e9, n_star, 4e5, where=(n_star < 4e5),
                     color="#ffcdd2", label="重算更快（慢链路 × 超长块）")
    ax2.axvline(BW_CROSS / 1e9, color="#c62828", ls="--", lw=1.4)
    ax2.text(BW_CROSS / 1e9 * 0.93, 20, f"交叉带宽 ≈ {BW_CROSS/1e9:.2f} GB/s",
             color="#c62828", fontsize=9, ha="right", rotation=90, va="bottom")
    for x, c, t in [(50, "#1565c0", "NVLink 50"), (16, "#6a1b9a", "PCIe 16")]:
        ax2.axvline(x, color=c, ls=":", lw=1.2)
        ax2.text(x * 1.06, 40, t, color=c, fontsize=9, rotation=90, va="bottom")
    ax2.annotate("N*：慢链路上块长超过该线\n就该丢弃重算而非搬运",
                 xy=(0.3, 120), xytext=(1.2, 800), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(0.05, 100); ax2.set_ylim(4, 4e5)
    ax2.set_xlabel("链路带宽 (GB/s)")
    ax2.set_ylabel("KV 块大小 (token)")
    ax2.set_title("下沉判据：2×KV÷带宽 vs 重算时间（3.3 节不等式）", fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig2_transfer_tiering.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    print_account()
    fig1()
    fig2()
    print(f"\n已生成 -> {OUT}")
