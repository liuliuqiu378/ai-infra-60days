#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 2 章 动手任务 B：把 JD 上的 "30-60k·15薪" 翻译成年现金包
用法：python3 salary_calc.py
数字仅为招聘信息口径的现金年化（月薪×薪数），不含期权/签字费，税前。
"""

def annual(lo_k, hi_k, months):
    return lo_k * months / 10, hi_k * months / 10  # k*月 -> 万元

EXAMPLES = [
    ("清昴智能 高级推理工程师 20-35k×14", 20, 35, 14),
    ("瑞芯微 推理部署 30-60k×15", 30, 60, 15),
    ("跳脱网络 vLLM/SGLang 60-90k×17", 60, 90, 17),
    ("国央企 推理加速 29-40k×14", 29, 40, 14),
    ("校招白菜(参考牛客27届) 30k×15", 30, 30, 15),
]

if __name__ == "__main__":
    print(f"{'岗位':<38}{'年现金包(万)':>14}")
    for name, lo, hi, m in EXAMPLES:
        a, b = annual(lo, hi, m)
        rng = f"{a:.0f}" if a == b else f"{a:.0f}-{b:.0f}"
        print(f"{name:<38}{rng:>12} 万")
    print("\n你也可以交互式计算：")
    try:
        lo = float(input("月薪下限(k)，直接回车退出：") or 0)
        if lo:
            hi = float(input("月薪上限(k)：") or lo)
            m = int(input("一年几薪(默认12)：") or 12)
            a, b = annual(lo, hi, m)
            print(f"年现金包约 {a:.0f}-{b:.0f} 万元（税前，不含期权）")
    except (KeyboardInterrupt, EOFError):
        pass
