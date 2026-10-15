#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 章 配图生成（数据全部来自 2026-10-15 RTX 4090 D 实测）
输出到 ../../images/figures/ch01/：
  fig1_request_lifecycle.png   一次请求的全链路（自绘示意图）
  fig2_concurrency_staircase.png 4 并发"阶梯式"完成时刻（实测）
  fig3_model_compare.png       0.5B vs 27B：显存与吐字速度（实测）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch01")
os.makedirs(OUT, exist_ok=True)

C_BLUE, C_ORANGE, C_GREEN, C_GRAY, C_RED = "#2f6fb0", "#e08a1e", "#3a9b6e", "#8a8f98", "#c0504d"


def fig1_lifecycle():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.3, 3.2, 1.7, 1.0, "你的程序\ncurl / SDK", C_GRAY),
        (2.4, 3.2, 2.0, 1.0, "ollama serve\nAPI 服务(:11434)\n收发 HTTP/排队", C_BLUE),
        (4.8, 3.2, 2.2, 1.0, "llama-server\n(= llama.cpp 内核)\n- np 1 单并行槽", C_ORANGE),
        (7.4, 3.2, 1.9, 1.0, "分词器\n文字→token id", C_GREEN),
        (9.7, 3.2, 2.0, 1.0, "GPU 前向计算\n逐层读权重+KV", C_RED),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=c, ec="none", alpha=0.85))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                color="white", fontsize=10.5, weight="bold")
    for i in range(len(boxes) - 1):
        x = boxes[i][0] + boxes[i][2]
        ax.add_patch(FancyArrowPatch((x, 3.7), (boxes[i + 1][0], 3.7),
                                     arrowstyle="-|>", mutation_scale=18, color="#444"))

    # 回程：采样 + 解码 + 流式返回
    back = [
        (9.7, 1.2, 2.0, 1.0, "采样器\n按概率选下一个 token", C_GREEN),
        (7.4, 1.2, 1.9, 1.0, "token→文字\n(chatml 模板)", C_GREEN),
        (4.0, 1.2, 2.4, 1.0, "流式回传 JSON\n一个 token 一帧", C_BLUE),
        (0.3, 1.2, 2.6, 1.0, "用户看到文字一个一个蹦出来", C_GRAY),
    ]
    for x, y, w, h, t, c in back:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=c, ec="none", alpha=0.85))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                color="white", fontsize=10.5, weight="bold")
    for i in range(len(back) - 1):
        x = back[i][0]
        ax.add_patch(FancyArrowPatch((x, 1.7), (back[i + 1][0] + back[i + 1][2], 1.7),
                                     arrowstyle="-|>", mutation_scale=18, color="#444"))
    # decode 自循环
    ax.add_patch(FancyArrowPatch((10.7, 3.2), (10.7, 2.25),
                                 connectionstyle="arc3,rad=0.5",
                                 arrowstyle="-|>", mutation_scale=16, color="#c0504d", linestyle="--"))
    ax.text(11.55, 2.72, "decode\n循环 N 次", ha="center", va="center",
            fontsize=9.5, color=C_RED, weight="bold")
    ax.text(6, 4.72, "去程：一句话进服务器，依次经历的 5 个环节",
            ha="center", fontsize=12, color="#333", weight="bold")
    ax.text(6, 0.35, "回程：每生成一个 token 都要在「GPU 计算 → 采样 → 转回文字 → 推给客户端」之间转一圈",
            ha="center", fontsize=11, color="#333")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_request_lifecycle.png"), dpi=150)
    plt.close(fig)


def fig2_staircase():
    # 实测：4 个请求的完成时刻（秒）
    fins = [1.92, 3.85, 5.77, 7.70]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, f in enumerate(fins):
        ax.barh(i, f, left=0, height=0.55, color=C_BLUE, alpha=0.8)
        ax.barh(i, i * 1.92, left=0, height=0.55, color=C_RED, alpha=0.75, label="排队等待" if i == 0 else "")
        ax.barh(i, f - i * 1.92, left=i * 1.92, height=0.55, color=C_GREEN, alpha=0.8,
                label="真正轮到我跑(~1.9s)" if i == 0 else "")
        ax.text(f + 0.12, i, f"{f:.2f}s", va="center", fontsize=11, weight="bold")
    ax.set_yticks(range(4)); ax.set_yticklabels([f"请求{i}" for i in range(4)])
    ax.set_xlabel("墙钟时间（秒，从 4 个请求同时发出开始）")
    ax.set_title("默认配置下 4 个并发请求呈严格阶梯：一个一个排队，总吞吐≈单请求（实测 77.9 vs 77.1 tok/s）",
                 fontsize=11.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, 9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_concurrency_staircase.png"), dpi=150)
    plt.close(fig)


def fig3_compare():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    names = ["qwen2.5:0.5b\nQ4_K_M", "qwen3.8:27b\nQ4_K_M(MTP)"]
    vram = [1155 / 1024, 19287 / 1024]          # MiB -> GiB，nvidia-smi 实测驻留
    weights = [743754955 / 1024**3, 17.40 / 1.0737]  # 0.5B 权重 GiB；27B 17.4GB->约16.2GiB
    speed = [648, 83]                            # decode tok/s 实测
    cold = [1.26, 4.08]                          # 冷加载秒

    ax = axes[0]
    x = range(2); w = 0.38
    ax.bar([i - w / 2 for i in x], weights, w, label="权重占用", color=C_ORANGE, alpha=0.85)
    ax.bar([i + w / 2 for i in x], vram, w, label="进程实际驻留显存", color=C_BLUE, alpha=0.85)
    for i in x:
        ax.text(i - w / 2, weights[i] + 0.2, f"{weights[i]:.2f}", ha="center", fontsize=10)
        ax.text(i + w / 2, vram[i] + 0.2, f"{vram[i]:.2f}", ha="center", fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("GiB"); ax.set_ylim(0, 21)
    ax.set_title("显存：驻留 = 权重 + KV Cache + 计算缓冲 + CUDA 上下文", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(names, speed, width=0.5, color=[C_GREEN, C_RED], alpha=0.85)
    for b, s, c in zip(bars, speed, cold):
        ax.text(b.get_x() + b.get_width() / 2, s + 12, f"{s} tok/s",
                ha="center", fontsize=12, color="#222")
        ax.text(b.get_x() + b.get_width() / 2, s + 42, f"冷加载 {c:.2f}s",
                ha="center", fontsize=10, color="#555")
    ax.set_ylabel("单请求 decode 速度 (tok/s)")
    ax.set_title("吐字速度：模型大 55 倍，decode 慢约 8 倍（访存瓶颈，第 7 章）", fontsize=11)
    ax.set_ylim(0, 720); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("同一张 RTX 4090 D(24G) 上两个模型的实测对比 · ollama 0.34.0 · 2026-10-15",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_model_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_lifecycle()
    fig2_staircase()
    fig3_compare()
    print("figures written to", os.path.abspath(OUT))
