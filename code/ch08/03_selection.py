"""03: 选型速算器。模型→量化字节→KV→卡数→单 token 成本。
价格：2026-09/10 海外现货区间（联网核实，随行情波动，落地前必须重查）；
      国内合规/昇腾价格【请联网检索最新报价补充】。"""
import json

CARDS = {  # 显存 GiB，HBM/GDDR 实测或规格带宽 GB/s，每小时美元（现货区间中位）
    "RTX4090D_24G":      dict(mem=24,  bw=918,  usd_hr=0.35, kind="PCIe4, 无 NVLink"),
    "L40S_48G":          dict(mem=48,  bw=864,  usd_hr=0.79, kind="PCIe4, 数据中心"),
    "A100_80G_SXM":      dict(mem=80,  bw=2039, usd_hr=1.20, kind="NVLink3 600"),
    "H100_80G_SXM":      dict(mem=80,  bw=3350, usd_hr=2.80, kind="NVLink4 900"),
}
# 价格来源：runpod/thunder/cloudgpu 2026-09~10 行情：H100 $2.1~3.5（企业 OD 可至 $4.5+），
# A100 80G ~$1.09~1.5，L40 ~$0.79，4090 ~$0.3~0.4。取区间中位仅用于教学估算。

def plan(model_b, bpe, kv_per_token_kib, ctx, concurrency, card, util=0.85, reserve=1.5):
    """reserve：激活/碎片/CUDA context 余量系数（1.5GiB 基线 + 经验冗余）。"""
    c = CARDS[card]
    w_gib = model_b*1e9*bpe/2**30
    kv_gib = concurrency * kv_per_token_kib*ctx/2**20
    need = w_gib + kv_gib + reserve
    n = max(1, -(-int(need*100)//int(c["mem"]*util*100)))  # ceil
    # 聚合 decode 墙：n 卡 TP（只在同构 NVLink 卡成立）；多卡无 NVLink 只给数据并行口径
    wall_ms = w_gib*2**30/(min(n,1)*c["bw"]*1e9)*1e3 if "NVLink" not in c["kind"] or n==1 \
              else w_gib*2**30/(n*c["bw"]*1e9)*1e3
    agg_tok_s = concurrency / (wall_ms/1e3)
    hourly = n*c["usd_hr"]
    return dict(card=card, cards=n, weight_gib=round(w_gib,2), kv_gib=round(kv_gib,2),
                total_need_gib=round(need,2), per_token_wall_ms=round(wall_ms,2),
                usd_per_mtok=round(hourly*3600/(agg_tok_s)/1e6*1e6,2))

scenarios = []
# 7B Q4：kv/token 用 ch04 口径 INT4 后约 28KiB/token（56KiB bf16 减半）
for card in CARDS:
    scenarios.append(dict(model="7B-Q4", ctx=4096, conc=32,
        **plan(7, 0.5, 28, 4096, 32, card)))
# 32B bf16：kv/token ≈ 512KiB（L64/d5120/GQA8 粗算），只跑 80G 卡
for card in ["A100_80G_SXM","H100_80G_SXM"]:
    scenarios.append(dict(model="32B-bf16", ctx=4096, conc=16,
        **plan(32, 2.0, 512, 4096, 16, card)))
# 70B Q4：35GiB 权重
for card in ["A100_80G_SXM","H100_80G_SXM"]:
    scenarios.append(dict(model="70B-Q4", ctx=4096, conc=16,
        **plan(70, 0.5, 256, 4096, 16, card)))

json.dump(dict(cards={k:{kk:vv for kk,vv in v.items() if kk!="kind"} for k,v in CARDS.items()},
               scenarios=scenarios,
               price_note="2026-09/10 海外现货中位，行情波动大；国内/昇腾报价需另行联网核实",
               card_kinds={k:v["kind"] for k,v in CARDS.items()}),
          open("selection_results.json","w"), indent=2, ensure_ascii=False)
for s in scenarios:
    print(f'{s["model"]:<8} {s["card"]:<16} 卡数 {s["cards"]}  '
          f'权重 {s["weight_gib"]}GiB KV {s["kv_gib"]}GiB 墙 {s["per_token_wall_ms"]}ms  '
          f'${s["usd_per_mtok"]}/百万tok')
