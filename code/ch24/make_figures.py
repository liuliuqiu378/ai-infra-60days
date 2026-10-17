# ch24 配图生成脚本：1 张程序示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch24/make_figures.py
# 图1：项目三交付物地图与六天流水线
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch24")
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
    fig, ax = plt.subplots(figsize=(14.2, 9.6))
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.set_title("图 1 程序示意：项目三交付物地图与六天流水线——每天一张表，四份交付进作品集（结构示意，非预填数据）",
                 fontsize=12.5)

    # 顶部四份交付物
    dels = [
        (0.35, "交付① 部署文档\nYAML+命令+硬件清单\n（含 NVLink 拓扑验证）", "#ffe082"),
        (3.85, "交付② 调优报告\nTP 加速比 + A/B 四档\n证据链五段", "#cfe2ff"),
        (7.35, "交付③ 压测报告\n三崩溃线扫描 + 取证\nnsys/ncu 图", "#c8e6c9"),
        (10.85, "交付④ 演练记录+面板\n三故障恢复时间\nGrafana 四排截图", "#f8bbd0"),
    ]
    for x, t, c in dels:
        box(ax, x, 8.3, 3.3, 1.0, t, fc=c, bold=True, fs=9)
    ax.text(7.1, 7.95, "求职二号作品 = 四份交付物（GitHub 仓库 + README + 调优博客）",
            ha="center", fontsize=9.5, style="italic", color="#444444")

    # 中排五模块
    mods = [
        (0.35, "① TP 多卡部署\n--tensor-parallel-size 2\ntopo 验证 NV1/PIX（ch15）\nTP=1/2 受控对照", "#cfe2ff"),
        (3.2, "② 优化前后对比\n前缀缓存 x Chunked Prefill\nD0-D3 四档，锁 md5\n分负载归因（ch16）", "#fffde7"),
        (6.05, "③ 压测到崩溃+取证\n阶梯扫 6-8 档\nL1/L2/L3 三判定线\nnsys NVTX + /metrics 快照", "#e8f5e9"),
        (8.9, "④ 故障演练\n断网->L1 · OOM->L3\n掉卡->L3 切备胎\n生效/恢复双时间", "#ffebee"),
        (11.75, "⑤ 面板与文档\n/metrics -> Grafana\n四排面板（ch23）\nREADME 五段", "#f3e5f5"),
    ]
    for x, t, c in mods:
        box(ax, x, 6.0, 2.7, 1.7, t, fc=c, fs=8.3, bold=True)
    for x in (2.0, 5.5, 7.4, 10.3, 13.1):
        arrow(ax, (x, 7.95), (x, 7.76))

    # 六天流水线
    days = ["D53\nTP 对照表", "D54\nA/B 四档表", "D55\n崩溃曲线", "D56\n取证图", "D57\n演练记录", "D58\n面板+合稿"]
    for i, d in enumerate(days):
        x = 0.6 + i * 2.3
        box(ax, x, 4.2, 2.0, 1.0, d, fc="#e0e0e0", fs=8.8, bold=True)
        if i < 5:
            arrow(ax, (x + 2.0, 4.7), (x + 2.3, 4.7))
    ax.text(7.1, 5.4, "六天节奏：每天只交付一张表（防攒成最后一天的大报告）",
            ha="center", fontsize=9.5, style="italic", color="#444444")

    # 下部：数字纪律 + 验收锚
    box(ax, 0.35, 2.3, 13.5, 1.35,
        "数字纪律（项目一/二/三同一铁律）：正文零预填性能数字 —— 每个数字必须来自你自己的一次运行并登记 RESULTS.md\n"
        "对照表三要素：硬件/模型/并发档（或负载 md5）· 口径备注缺一不进报告",
        fc="#fffde7", ec="#b26a00", fs=9, bold=True)

    box(ax, 0.35, 0.35, 13.5, 1.5,
        "验收锚：限流档 <= 崩溃档（限流档来自 L1/L2 线）· 每结论配一张取证图 · 每故障带生效+恢复两个秒级数字 · 每次演练后至少一处改进 · README 必写已知边界\n"
        "证据链五段：现象 -> 取证 -> 归因 -> 动作 -> 验证（全项目三的论文格式，④⑤ 也按它写）",
        fc="#ffcdd2", ec="#c62828", fs=9, bold=True)

    fig.savefig(os.path.join(OUT, "fig1_p3_deliverables.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print(f"已生成 -> {OUT}")
