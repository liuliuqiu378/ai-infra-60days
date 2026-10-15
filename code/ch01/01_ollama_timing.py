#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 章 实验 1：拆开一次推理请求的耗时
=====================================
前置：本机已安装 ollama（https://ollama.com），服务在 11434 端口；
      并已 `ollama pull qwen2.5:0.5b`。

用法：python3 01_ollama_timing.py

它会：
  1. 连续调用 2 次 /api/generate（第 1 次可能含冷加载）；
  2. 把 ollama 返回的计时字段拆成：加载 / prefill（读 prompt）/ decode（逐字生成）；
  3. 调用 nvidia-smi 打印模型驻留时的显存。

所有数字均为 2026-10-15 在 RTX 4090 D(24G) + ollama 0.34.0 上实测，
你在自己机器上跑出来的数字会不同——这很正常，重点是学会"看哪几个字段"。
"""
import json
import subprocess
import urllib.request

API = "http://localhost:11434"
MODEL = "qwen2.5:0.5b"
PROMPT = "用一句话解释什么是大模型推理。"


def generate(prompt: str) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{API}/api/generate", data=body)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def report(tag: str, d: dict) -> None:
    ns = 1_000_000_000
    pe_n, pe_d = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 1)
    e_n, e_d = d.get("eval_count", 0), d.get("eval_duration", 1)
    print(f"--- {tag} ---")
    print(f"回答前 80 字: {d['response'][:80].strip()!r}")
    print(f"总耗时 total_duration      : {d['total_duration'] / ns:6.3f} s")
    print(f"  其中 模型加载 load_duration: {d['load_duration'] / ns:6.3f} s")
    print(f"  prefill: {pe_n:4d} prompt token / {pe_d / ns:6.3f} s "
          f"= {pe_n / (pe_d / ns):8.1f} tok/s")
    print(f"  decode : {e_n:4d} 输出 token   / {e_d / ns:6.3f} s "
          f"= {e_n / (e_d / ns):8.1f} tok/s")
    print()


if __name__ == "__main__":
    report("第 1 次调用（可能含冷加载）", generate(PROMPT))
    report("第 2 次调用（权重已在显存）", generate(PROMPT))
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True)
    print("模型驻留时的 GPU:", smi.stdout.strip())
    print("\n看图要点：decode 的 tok/s 才是用户感知的'吐字速度'；"
          "prefill 快得多（一次并行算一整段），两者是两个世界——第 3 章讲透。")
