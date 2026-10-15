#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 1 章 实验 2：冷启动到底慢在哪
================================
1. `ollama stop` 把模型请出显存；
2. 立刻发一个最短请求，用墙钟掐表，同时读 load_duration；
3. 查 /api/ps 看权重文件大小、有多少在显存里、量化档位。

结论（4090 D 实测）：0.5B(Q4, 710 MiB) 冷加载约 1.3 s；
27B(Q4, 17.4 GB) 冷加载约 4.1 s。权重越大，首次等待越久——
这正是生产服务要搞"模型预热/常驻"的原因（第 23 章）。
"""
import json
import subprocess
import time
import urllib.request

API = "http://localhost:11434"
MODEL = "qwen2.5:0.5b"


def post(path: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=body)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


if __name__ == "__main__":
    print(f"先把 {MODEL} 请出显存 ...")
    subprocess.run(["ollama", "stop", MODEL], check=False, capture_output=True)
    time.sleep(2)
    print("停掉后显存:", subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip())

    t0 = time.time()
    d = post("/api/generate", {"model": MODEL, "prompt": "你好",
                               "stream": False, "options": {"temperature": 0}})
    print(f"冷请求墙钟             : {time.time() - t0:.3f} s")
    print(f"其中 load_duration     : {d['load_duration'] / 1e9:.3f} s  <-- 权重从磁盘搬进显存")
    print(f"decode {d['eval_count']} token : {d['eval_duration'] / 1e9:.3f} s")

    ps = json.load(urllib.request.urlopen(f"{API}/api/ps"))
    if ps.get("models"):
        m = ps["models"][0]
        print("\n/api/ps 驻留信息:")
        print(f"  权重文件大小 size     : {m['size'] / 1e9:.2f} GB")
        print(f"  其中在显存 size_vram  : {m['size_vram'] / 1e9:.2f} GB  "
              f"(小于 size 就说明有层被卸到 CPU，第 12 章讲)")
        print(f"  量化档位              : {m['details']['quantization_level']}")
        print(f"  上下文长度            : {m['context_length']}")
