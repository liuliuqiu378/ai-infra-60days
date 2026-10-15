#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 5 章 流式并发压测脚本
========================
对多并行槽 ollama（底层 llama.cpp continuous batching）做等长输出压测：
  * C 个客户端用 Barrier 在同一瞬间发起请求（closed-model：同一时刻涌入 C 个新请求）
  * 每个请求 stream:true，逐行读 NDJSON，量三个时间点：
      t0 发起 / ttft 收到第一个 token chunk / tend 收到 done=true
  * 客户端超时 TIMEOUT 秒，超时/异常记失败（对应线上网关的 504/超时丢弃）
指标：
  TTFT  首 token 延迟（prefill+排队）；TPOT = (tend-ttft)/(eval_count-1)
  E2E   总延迟；QPS/聚合吞吐 = Σeval_count / 墙钟窗口；成功率 = 完成数/C
用法：
  python load_test.py --levels 1,4,8,16,32,48,64,96,128 --tokens 96 --timeout 30
  python load_test.py --single 32          # 单点标定
"""
import argparse
import http.client
import json
import os
import statistics
import threading
import time

HOST, PORT = "127.0.0.1", 11435
MODEL = "qwen2.5:0.5b"
# 固定 prompt（约 20+ token），所有请求相同；temperature=0 保证可重复
PROMPT = "请用流畅的中文，围绕『春天的公园』写一小段描写，要求有画面感。"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs)-1, int(round((q/100)*(len(xs)-1)))))
    return xs[k]


def one_request(n_tokens, timeout, rec, idx):
    """单请求；rec 是每个线程写入的 dict"""
    body = json.dumps({
        "model": MODEL, "prompt": PROMPT, "stream": True,
        "options": {"num_predict": n_tokens, "temperature": 0},
    })
    t0 = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
        conn.request("POST", "/api/generate", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        ttft = None; ntok = 0; t_first_byte = None
        deadline = t0 + timeout
        for raw in resp:
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError("deadline exceeded while streaming")
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            if t_first_byte is None:
                t_first_byte = now
            ch = json.loads(line)
            if ch.get("response"):
                if ttft is None and ch["response"].strip() != "":
                    ttft = now
                ntok = ch.get("eval_count", ntok)
            if ch.get("done"):
                ntok = ch.get("eval_count", ntok)
                tend = time.perf_counter()
                rec.update(ok=True, t0=t0, ttft=(ttft or tend)-t0,
                           total=tend-t0, tokens=ntok)
                conn.close(); return
        rec.update(ok=False, reason="stream ended without done")
    except Exception as e:
        rec.update(ok=False, reason=type(e).__name__)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_level(concurrency, n_tokens, timeout):
    recs = [dict(ok=False) for _ in range(concurrency)]
    barrier = threading.Barrier(concurrency + 1)

    def worker(i):
        barrier.wait()
        one_request(n_tokens, timeout, recs[i], i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    t_start = time.perf_counter()
    barrier.wait()           # 同时放行
    for t in threads:
        t.join(timeout + 15)
    t_end = time.perf_counter()

    ok = [r for r in recs if r["ok"]]
    bad = [r.get("reason", "?") for r in recs if not r["ok"]]
    n_ok = len(ok)
    sr = n_ok / concurrency
    if ok:
        ttfts = [r["ttft"]*1000 for r in ok]
        totals = [r["total"]*1000 for r in ok]
        toks = sum(r["tokens"] for r in ok)
        # TPOT：用成功请求的 (total-ttft)/(tokens-1)
        tpots = [(r["total"]-r["ttft"])/(r["tokens"]-1)*1000
                 for r in ok if r["tokens"] > 1]
        window = max(r["t0"]+r["total"] for r in ok) - min(r["t0"] for r in ok)
        agg = toks / window
        res = dict(concurrency=concurrency, success=sr, n_ok=n_ok,
                   ttft_p50=pct(ttfts, 50), ttft_p90=pct(ttfts, 90),
                   total_p50=pct(totals, 50), total_p90=pct(totals, 90),
                   total_max=max(totals), tpot_ms=statistics.median(tpots),
                   agg_tok_s=agg, tokens=toks,
                   fail_reasons={k: bad.count(k) for k in set(bad)})
    else:
        res = dict(concurrency=concurrency, success=0.0, n_ok=0,
                   fail_reasons={k: bad.count(k) for k in set(bad)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,8,16,24,32,48,64,96,128")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--single", type=int, default=0)
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()

    if args.warmup:
        print("warmup..."); run_level(1, 8, 60)

    levels = [args.single] if args.single else [int(x) for x in args.levels.split(",")]
    rows = []
    for C in levels:
        r = run_level(C, args.tokens, args.timeout)
        rows.append(r)
        if r["n_ok"]:
            print(f"C={C:4d} | 成功率 {r['success']*100:5.1f}% | "
                  f"TTFT p50/p90 {r['ttft_p50']:7.0f}/{r['ttft_p90']:7.0f} ms | "
                  f"E2E p50/p90 {r['total_p50']:7.0f}/{r['total_p90']:7.0f} ms | "
                  f"TPOT {r['tpot_ms']:5.1f} ms | 聚合 {r['agg_tok_s']:7.0f} tok/s")
        else:
            print(f"C={C:4d} | 成功率   0.0% | 失败原因 {r['fail_reasons']}")

    if not args.single:
        out = os.path.join(os.path.dirname(__file__), "sweep_results.json")
        old = []
        if os.path.exists(out):
            old = json.load(open(out)).get("runs", [])
        payload = dict(ts=time.strftime("%F %T"), model=MODEL,
                       n_tokens=args.tokens, timeout=args.timeout,
                       num_parallel=32, rows=rows)
        json.dump(payload, open(out, "w"), indent=2, ensure_ascii=False)
        print("saved", out)


if __name__ == "__main__":
    main()
