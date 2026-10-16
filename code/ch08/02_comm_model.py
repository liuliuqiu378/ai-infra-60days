"""02: 多卡互联通信建模（纯计算）。
TP：每层 2 次 all-reduce；ring 每卡流量 2(N-1)/N·payload。
单次 collective 时间 = 固定延迟 α(NCCL launch+跳数，典型拟合值) + 流量/有效带宽 β。
decode 的 payload 只有 d 个元素 → 延迟项主导；这是"TP 必须低延迟 NVLink"的根因。
"""
import json

# 单向有效 GB/s：ch07 本机实测 / NVIDIA·IBTA 公开规格（双向值折半）
LINKS = {
    "PCIe4_x16(4090D实测)": dict(bw=18.6,  alpha_us=25.0),
    "PCIe5_x16(规格)":      dict(bw=64.0,  alpha_us=20.0),
    "NVLink3(A100规格)":    dict(bw=300.0, alpha_us=10.0),
    "NVLink4(H100规格)":    dict(bw=450.0, alpha_us=8.0),
    "IB_NDR_400Gb(规格)":   dict(bw=50.0,  alpha_us=15.0),
}

def ar_us(payload_bytes, n, bw, alpha, eff=0.8):
    flow = 2*(n-1)/n * payload_bytes
    return alpha + flow/(bw*eff*1e9)*1e6

def hbm_read_ms(P_b, bpe, n, hbm_tbs):
    return P_b*1e9*bpe/(n*hbm_tbs*1e12)*1e3

rows = []
for tag, d, L, P_b, hbm in [("Qwen2.5-0.5B",896,24,0.5,0.918),
                             ("Qwen2.5-7B",3584,28,7.0,3.35)]:
    ns = [2,4,8] if "7B" in tag else [2]
    for link, cfg in LINKS.items():
        for n in ns:
            payload = d*2                                   # bf16 一个隐状态
            ar = ar_us(payload, n, cfg["bw"], cfg["alpha_us"])
            comm_ms = 2*L*ar/1e3
            comp_ms = hbm_read_ms(P_b, 2, n, hbm)
            rows.append(dict(model=tag, link=link, tp=n,
                weight_read_ms=round(comp_ms,3),
                comm_ms_per_token=round(comm_ms,3),
                comm_over_compute_pct=round(comm_ms/comp_ms*100,1)))

# PP：每 token 仅在 stage 边界传 1 个隐状态，共 (N-1) 次 P2P
pp = []
for link, cfg in LINKS.items():
    for n in [2,4,8]:
        b = (n-1)*3584*2
        us = (n-1)*cfg["alpha_us"]/2 + b/(cfg["bw"]*1e9)*1e6  # P2P 延迟按 α/2
        pp.append(dict(link=link, pp=n, bytes=b, us=round(us,1)))

# prefill：TP8 单层 AR，payload = S·d·2，带宽项主导
pre = []
for S in [128, 2048]:
    for link in ["NVLink4(H100规格)","IB_NDR_400Gb(规格)","PCIe4_x16(4090D实测)"]:
        cfg = LINKS[link]
        pre.append(dict(seq=S, link=link,
                        ar_ms=round(ar_us(S*3584*2,8,cfg["bw"],cfg["alpha_us"])/1e3,3)))

res = dict(links=LINKS, tp_decode=rows, pp_decode=pp, tp_prefill_single_ar=pre,
    alpha_note="α 为单次 NCCL collective 的典型固定延迟拟合值（launch+跳数，非本机实测），"
               "请以各集群 nccl-tests all_reduce_perf 实测替换；带宽项用 ch07 实测/公开规格",
    sources="NVLink3 600GB/s、NVLink4 900GB/s、PCIe5 128GB/s 双向=2024 NVIDIA datasheet；"
            "IB NDR 400Gb/s=50GB/s=IBTA 规格；PCIe4 18.6GB/s=ch07 本机实测")
json.dump(res, open("comm_results.json","w"), indent=2, ensure_ascii=False)

print("== 7B decode：权重读 vs TP 通信/ token ==")
for r in rows:
    if r["model"]=="Qwen2.5-7B":
        print(f'TP{r["tp"]} {r["link"]:<22} 读权重 {r["weight_read_ms"]:>5.2f}ms  '
              f'通信 {r["comm_ms_per_token"]:>6.2f}ms  通信/计算 {r["comm_over_compute_pct"]:>7.1f}%')
print("\n== 7B PP decode 边界传输（全部 stage）==")
for r in pp:
    if r["pp"]==8: print(f'{r["link"]:<22} {r["bytes"]} B  {r["us"]} µs')
print("\n== 7B TP8 prefill 单层 AR ==")
for r in pre: print(r)
