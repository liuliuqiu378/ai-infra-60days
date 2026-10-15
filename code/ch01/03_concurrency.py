#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 章 实验 3（本章最重要的一个实验）：4 个请求一起来会怎样？
==============================================================
同时向 ollama 发 1 个和 4 个相同请求，记录每个请求的墙钟耗时。

4090 D + 27B Q4 实测：
  n=1 : 整批 1.95 s，按墙钟聚合吞吐 77.1 tok/s
  n=4 : 完成时刻呈阶梯 1.92 / 3.85 / 5.77 / 7.70 s，
        按墙钟聚合吞吐 77.9 tok/s —— 和单请求几乎一样！
说明默认配置下 ollama 背后 llama-server 的并行槽 -np 1，
请求是"一个一个排队"跑的，并发没有带来任何总吞吐提升。
这不是 ollama 坏，而是它的定位（本地单用户）决定的。
真正的高并发推理引擎（vLLM 等）如何把这堵墙拆掉，是第 14 章的主题。

注意：decode tok/s 字段只统计"轮到我之后"的速度，不含排队等待，
所以四个请求各自仍显示 ~83 tok/s——不要被它骗了，看墙钟。
"""
import concurrent.futures as cf
import json
import time
import urllib.request

API = "http://localhost:11434"
MODEL = "qwen3.8:27b-mtp-q4_K_M"   # 没有这个模型就换成 qwen2.5:0.5b，现象一样
PROMPT = "用150字介绍杭州。"


def one(i: int):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "stream": False,
                       "options": {"temperature": 0, "num_predict": 150}}).encode()
    t0 = time.time()
    with urllib.request.urlopen(
            urllib.request.Request(f"{API}/api/generate", data=body), timeout=600) as r:
        d = json.load(r)
    return i, time.time() - t0, d["eval_count"], d["eval_count"] / (d["eval_duration"] / 1e9)


if __name__ == "__main__":
    for n in (1, 4):
        t0 = time.time()
        with cf.ThreadPoolExecutor(n) as ex:
            res = sorted(ex.map(one, range(n)))
        wall = time.time() - t0
        toks = sum(x[2] for x in res)
        print(f"同时发 {n} 个请求：整批墙钟 {wall:.2f} s，"
              f"总 token {toks}，按墙钟聚合吞吐 {toks / wall:.1f} tok/s")
        for i, w, c, rate in res:
            print(f"  请求{i}: 完成于 {w:5.2f} s, {c} token, decode {rate:5.1f} tok/s")
        print()
