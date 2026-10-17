# ch23 配图生成脚本：1 张程序示意图（零外部图片、零性能跑测）
# 运行：python3 code/ch23/make_figures.py
# 图1：服务化五模块全景——接入层→K8s调度→引擎→可观测→容量与降级
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch23")
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
    fig, ax = plt.subplots(figsize=(14.2, 9.4))
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 9.4); ax.axis("off")
    ax.set_title("图 1 程序示意：服务化五模块全景——从引擎能力到可承诺的服务（结构示意，非实测数据）",
                 fontsize=12.5)

    # 顶部流量入口
    box(ax, 3.6, 8.5, 7.0, 0.65, "业务流量 → OpenAI 兼容 API（对外协议稳定，对内引擎可换）",
        fc="#ffe082", bold=True, fs=10)
    arrow(ax, (7.1, 8.5), (7.1, 8.18))

    # 模块一：接入层四手段
    box(ax, 0.3, 6.5, 6.7, 1.65,
        "模块一 · 接入层四手段（挡在门外）\n"
        "① 排队吸收秒级突发 → ② 限流挡持续过载（429）\n"
        "→ ③ 熔断止血（下游已坏）→ ④ 降级兜底\n"
        "限流点 = 满足 SLO 最大并发档 − 余量（ch16/19 数据）",
        fc="#fffde7", ec="#b26a00", fs=8.8, bold=True)
    # 模块二：K8s 调度
    box(ax, 7.2, 6.5, 6.7, 1.65,
        "模块二 · K8s GPU 调度\n"
        "GPU 只写 limits（K8s 官方，整数设备）\n"
        "共享三路：整卡优先 / MIG（A100+ 硬隔离）/ 时间片（拿隔离换共享）\n"
        "Gang：Volcano PodGroup minMember —— 多副本要么全要么无",
        fc="#e8f5e9", ec="#2e7d32", fs=8.8, bold=True)
    arrow(ax, (3.65, 6.45), (3.65, 6.1), color="#b26a00")
    arrow(ax, (10.55, 6.45), (10.55, 6.1), color="#2e7d32")

    # 模块：引擎 Pod（ch14-22 优化干活处）
    box(ax, 0.3, 4.6, 6.7, 1.4,
        "引擎 Pod（2 副本 + maxUnavailable 0 滚动更新）\n"
        "ch14–22 的优化在这里干活：\n"
        "chunked prefill / 前缀缓存 / PD 分离 / 投机解码 …\n"
        "readiness 等模型加载（冷启动分钟级）",
        fc="#cfe2ff", ec="#1565c0", fs=8.8, bold=True)
    # 模块三：可观测
    box(ax, 7.2, 4.6, 6.7, 1.4,
        "模块三 · 可观测三支柱（回采）\n"
        "Metrics：/metrics → Prometheus → Grafana（告警靠它）\n"
        "Logging：结构化日志（排障）· Tracing：链路打点（优化）\n"
        "vllm:num_requests_waiting · vllm:gpu_cache_usage_perc",
        fc="#f3e5f5", ec="#6a1b9a", fs=8.8, bold=True)
    arrow(ax, (7.0, 5.3), (7.2, 5.3), color="#6a1b9a")

    # 模块四：容量
    box(ax, 0.3, 2.6, 6.7, 1.45,
        "模块四 · 容量与扩缩容\n"
        "Little 定律 N = X × R：代入 SLO 算在途容量\n"
        "实例数 = 峰值 QPS 账 ÷ N + 冗余（滚动+故障各一份）\n"
        "KEDA：队列/TTFT 信号；activation 阈值与目标值分开；冷启动进账",
        fc="#e3f2fd", ec="#1565c0", fs=8.8, bold=True)
    # 模块五：降级
    box(ax, 7.2, 2.6, 6.7, 1.45,
        "模块五 · 三级降级预案（永远先 L1）\n"
        "L1 参数级（秒-分钟）：限流收紧 / max_num_seqs 调小\n"
        "L2 算法级（分钟）：截断上下文 / 降 max_tokens / 关加速开关\n"
        "L3 架构级（分钟-小时）：切备胎小模型 / 关非核心功能",
        fc="#ffebee", ec="#c62828", fs=8.8, bold=True)
    arrow(ax, (3.65, 4.55), (3.65, 4.1), color="#1565c0")
    arrow(ax, (10.55, 4.55), (10.55, 4.1), color="#c62828")

    # 底部锚
    box(ax, 0.3, 0.4, 13.6, 1.5,
        "贯穿全章的锚：一切服务化决策都落在 ch05「并发-延迟崩塌曲线」上——\n"
        "限流守住崩溃点不被越过 · 容量把点位算准 · 降级是点位被越过时的逃生门\n"
        "告警三件套 = ch19 三条崩溃线：L1 错误 >1% · L2 p99 超 SLO · L3 积压/吞吐封顶",
        fc="#ffcdd2", ec="#c62828", fs=9.2, bold=True)

    fig.savefig(os.path.join(OUT, "fig1_serving_stack.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_capacity():
    """任务 B 容量账（零成本算术，演示口径）"""
    slo_e2e = 10.0      # s，p99 E2E SLO
    x_slo = 8.0         # req/s，满足 SLO 的单实例吞吐（演示值，回填自家压测）
    peak_qps = 40.0     # req/s，业务峰值（演示值）
    n_inflight = x_slo * slo_e2e
    n_inst = -(-peak_qps // x_slo)          # 向上取整
    n_final = n_inst + 2                     # +滚动更新1 +故障冗余1
    print("=== 任务 B：Little 定律容量账（演示口径，回填自家数据） ===")
    print(f"N = X x R = {x_slo} req/s x {slo_e2e} s = {n_inflight:.0f} 在途请求/实例")
    print(f"实例数 = ceil({peak_qps}/{x_slo}) = {n_inst:.0f}，+2 余量 = {n_final:.0f} 实例")
    print("\n=== 告警三件套（绑定 ch19 三条崩溃线） ===")
    print("L1: rate(vllm:request_success_total{finish_reason=\"error\"}[2m]) / rate(...total[2m]) > 0.01")
    print("L2: histogram_quantile(0.99, ttft_bucket) > SLO  持续 2 分钟")
    print("L3: vllm:num_requests_waiting > 阈值  或  vllm:gpu_cache_usage_perc > 0.9")


if __name__ == "__main__":
    print_capacity()
    fig1()
    print(f"\n已生成 -> {OUT}")
