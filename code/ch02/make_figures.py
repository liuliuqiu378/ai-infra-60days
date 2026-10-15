#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第2章配图：数据来自2026年8-10月公开招聘信息，来源登记在 ch02 资料来源。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "images", "figures", "ch02")
os.makedirs(OUT, exist_ok=True)
C = dict(blue="#2f6fb0", orange="#e08a1e", green="#3a9b6e", gray="#8a8f98",
         red="#c0504d", purple="#7d5ba6")


def fig1_salary():
    rows = [
        ("智遨通(天津,紫光系) 算子工程师·本科", 16, 24, C["green"], "校招/初级"),
        ("群核信息(杭州) AI Infra·本科以上", 20, 30, C["green"], "JD写20k起"),
        ("欧立腾(上海) 推理部署·3-5年", 20, 25, C["blue"], "x13薪"),
        ("清昴智能(北京) 高级推理工程师", 20, 35, C["blue"], "x14薪"),
        ("中科类脑(北京) 推理服务·2年+硕", 28, 55, C["blue"], "集群部署方向"),
        ("瑞芯微(上海) 推理部署·3年+硕", 30, 60, C["orange"], "x15薪,上市芯片厂"),
        ("猎聘 推理部署(北京,A轮)·3-5年", 40, 70, C["orange"], "x15薪"),
        ("国产芯片框架适配(深圳)·3-5年", 25, 45, C["orange"], "昇腾/寒武纪/海光等"),
        ("猎聘 推理方向专家(深圳)·3-5年", 50, 80, C["red"], "技术专家/首席架构师"),
        ("跳脱网络(沪/穗) vLLM/SGLang·3年+硕", 60, 90, C["red"], "x17薪"),
        ("整车厂 智驾大模型部署专家(北京)", 90, 120, C["purple"], "稀缺方向顶价"),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    for i, (label, lo, hi, color, note) in enumerate(rows):
        ax.barh(i, hi - lo, left=lo, height=0.58, color=color, alpha=0.85)
        ax.text(hi + 1.5, i, f"{lo}-{hi}k", va="center", fontsize=10.5)
        ax.text(2, i, note, va="center", fontsize=9, color="white")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlabel("月薪区间（千元/月，不含年终奖倍数；注意 x13/14/15/17 薪差异）")
    ax.set_xlim(0, 135)
    ax.set_title("2026年8-10月公开在招的推理 Infra 岗位月薪区间（猎聘/高校就业网/企业简章真实JD）",
                 fontsize=12.5, pad=12)
    legend = [Patch(fc=C["green"], label="初级/校招(16-30k)"),
              Patch(fc=C["blue"], label="1-5年 主力工程师(20-55k)"),
              Patch(fc=C["orange"], label="资深/芯片厂(25-70k)"),
              Patch(fc=C["red"], label="专家/架构师(50-90k)"),
              Patch(fc=C["purple"], label="稀缺顶价(智驾等,90k+)")]
    ax.legend(handles=legend, loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_salary_ranges.png"), dpi=150)
    plt.close(fig)


def fig2_chain():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, title, items, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    fc=color, ec="none", alpha=0.88))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
                color="white", fontsize=12)
        ax.text(x + w / 2, y + h / 2 - 0.25, items, ha="center", va="center",
                color="white", fontsize=9.8, linespacing=1.5)

    box(0.2, 4.2, 2.7, 2.2, "① 芯片层",
        "NVIDIA H200/B200/RTX4090\nAMD MI 系列\n国产:昇腾910/寒武纪\n海光/摩尔线程/沐曦", C["red"])
    box(3.3, 4.2, 2.9, 2.2, "② 引擎/框架层",
        "vLLM · SGLang\nTensorRT-LLM · lmdeploy\nTGI · llama.cpp\n国产:MindIE等", C["orange"])
    box(6.6, 4.2, 2.9, 2.2, "③ 云与平台层",
        "火山/百炼/腾讯云\n华为云/运营商智算中心\n推理网关·K8S调度\nMaaS·监控·计费", C["blue"])
    box(9.9, 4.2, 2.9, 2.2, "④ 应用层",
        "C端:豆包/元宝/夸克\nB端:Agent/客服/编程\n私有化:政务/金融/能源\n端侧:车/PC/手机", C["green"])
    for x in (2.9, 6.2, 9.5):
        ax.add_patch(FancyArrowPatch((x, 5.3), (x + 0.4, 5.3),
                                     arrowstyle="-|>", mutation_scale=20, color="#444"))
    box(0.2, 0.7, 12.6, 2.0, "⑤ 国产化替代支线（政策驱动，2026年招聘增量明显）",
        "昇腾 CANN + MindIE ｜ 寒武纪 NeuWare ｜ 海光 DTK ｜ 摩尔线程 MUSA\n"
        "典型工作:把 vLLM/SGLang 的算子与并行策略移植到国产卡 → 框架适配工程师 25-55k\n"
        "卡点:生态(CUDA护城河)、算子完备度、通信库、PyTorch生态兼容性", C["purple"])
    ax.text(6.5, 3.55, "越往右离业务越近，越往左离硬件越近；推理Infra工程师主要活跃在②③两层，上下各伸一只手",
            ha="center", fontsize=11, color="#333")
    ax.text(6.5, 6.72, "大模型推理产业链地图（2026）", ha="center", fontsize=14, color="#222")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_industry_chain.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_salary()
    fig2_chain()
    print("figures written to", os.path.abspath(OUT))
