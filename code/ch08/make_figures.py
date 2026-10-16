# -*- coding: utf-8 -*-
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"]=False
BLUE,RED,GREEN,ORANGE,PURPLE,GRAY="#4C78A8","#D85C5C","#54A273","#E8943A","#8467A8","#8A8A8A"
OUT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","images","figures","ch08"))
os.makedirs(OUT,exist_ok=True)
nsys=json.load(open("nsys_results.json"))
comm=json.load(open("comm_results.json"))
sel=json.load(open("selection_results.json"))
kp=json.load(open("kernel_portraits.json"))

# ===== fig1 互联带宽阶梯 =====
def fig1():
    fig,(axL,axR)=plt.subplots(1,2,figsize=(13.4,6.0),gridspec_kw={"width_ratios":[1.25,1]})
    items=[("PCIe4 x16\n(4090D实测)",18.6,"机内·GPU↔CPU",GRAY),
           ("IB NDR 400Gb\n(规格)",50.0,"机间·RDMA网络",PURPLE),
           ("PCIe5 x16\n(规格)",64.0,"机内·GPU↔GPU",BLUE),
           ("NVLink3 A100\n(规格)",300.0,"机内·8卡全互联",GREEN),
           ("NVLink4 H100\n(规格)",450.0,"机内·NVSwitch",ORANGE)]
    names=[i[0] for i in items]; vals=[i[1] for i in items]; cols=[i[3] for i in items]
    bars=axL.bar(names,vals,color=cols,width=0.62)
    for b,v in zip(bars,vals): axL.text(b.get_x()+b.get_width()/2,v+8,f"{v:g}",ha="center",fontsize=11,weight="bold")
    axL.set_ylabel("单向有效带宽（GB/s）",fontsize=11.5)
    axL.set_title("互联带宽阶梯：机间网络比机内 NVLink 慢一个数量级",fontsize=12.5,weight="bold")
    axL.set_ylim(0,520); axL.tick_params(axis="x",labelsize=9.5)
    axL.annotate("",xy=(4,450),xytext=(1,50),arrowprops=dict(arrowstyle="-|>",color="black",lw=1.4))
    axL.text(2.2,200,"24×",fontsize=13,weight="bold",ha="center")
    for i,(axis,lab) in enumerate([(axL,None)]): pass
    axL.text(0.02,0.96,"49× 于本机显存带宽 918 GB/s\n（过 PCIe 就掉档）",transform=axL.transAxes,
             fontsize=9.3,va="top",color="#555")

    # 右：延迟量级（典型值）
    lat=[("NVLink P2P",1.5,"sub-µs~2µs",ORANGE),
         ("IB NDR RDMA",3.0,"数 µs（软件栈）",PURPLE),
         ("PCIe P2P",8.0,"经 CPU/PCIe Switch",GRAY),
         ("普通以太网(TCP)",40.0,"几十~上百 µs","#B07146")]
    y=np.arange(len(lat))[::-1]
    axR.barh(y,[x[1] for x in lat],color=[x[3] for x in lat],height=0.55)
    for yi,(n,v,note,_) in zip(y,lat):
        axR.text(61.5,yi,note,va="center",ha="right",fontsize=9,color="#444")
    for yi,(n,v,note,_) in zip(y,lat):
        axR.text(v+0.8,yi,f"~{v:g}µs",va="center",fontsize=11,weight="bold")
    axR.set_yticks(y); axR.set_yticklabels([x[0] for x in lat],fontsize=10.5)
    axR.set_xlabel("单次点对点延迟（典型量级，非本机实测）",fontsize=10.5)
    axR.set_title("延迟阶梯：decode 小消息对延迟更敏感",fontsize=12.5,weight="bold")
    axR.set_xlim(0,62)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_links.png",dpi=140); plt.close()

# ===== fig2 TP vs PP =====
def fig2():
    fig,(axL,axR)=plt.subplots(1,2,figsize=(13.4,5.8))
    links=["PCIe4_x16(4090D实测)","PCIe5_x16(规格)","IB_NDR_400Gb(规格)","NVLink3(A100规格)","NVLink4(H100规格)"]
    marks={"PCIe4_x16(4090D实测)":("o",GRAY),"PCIe5_x16(规格)":("s",BLUE),
           "IB_NDR_400Gb(规格)":("^",PURPLE),"NVLink3(A100规格)":("D",GREEN),"NVLink4(H100规格)":("*",ORANGE)}
    for lk in links:
        rs=[r for r in comm["tp_decode"] if r["model"]=="Qwen2.5-7B" and r["link"]==lk]
        rs.sort(key=lambda r:r["tp"])
        m,c=marks[lk]
        axL.plot([r["tp"] for r in rs],[r["comm_over_compute_pct"] for r in rs],
                 marker=m,color=c,lw=1.8,ms=9,label=lk.replace("(4090D实测)","").replace("_规格","").replace("(规格)",""))
    axL.axhline(100,color=RED,ls="--",lw=1.2); axL.text(2.05,108,"100%：通信吃掉全部权重读取红利",color=RED,fontsize=9.5)
    axL.set_xlabel("张量并行度 TP=N",fontsize=11.5); axL.set_ylabel("TP 通信时间 / 单卡权重读取时间（%）",fontsize=10.8)
    axL.set_xticks([2,4,8]); axL.set_ylim(0,300)
    axL.set_title("7B decode：TP 通信占比（建模，α/β 模型）",fontsize=12.5,weight="bold")
    axL.grid(ls=":",alpha=0.5); axL.legend(fontsize=8.6,loc="upper left")

    pp8=[r for r in comm["pp_decode"] if r["pp"]==8]
    names=["PCIe4\n实测","PCIe5","IB NDR","NVLink3","NVLink4"]
    vals=[r["us"] for r in pp8]
    axR.bar(names,vals,color=[GRAY,BLUE,PURPLE,GREEN,ORANGE],width=0.6)
    for i,v in enumerate(vals): axR.text(i,v+1.5,f"{v:g}",ha="center",fontsize=10.5,weight="bold")
    axR.axhline(7000,color=RED,ls="--")
    axR.text(2.0,7200,"7B 单 token 墙钟量级 ~7,000µs（H100 单卡口径）",color=RED,fontsize=9,ha="center")
    axR.set_ylabel("PP8 全程边界传输（µs）",fontsize=11)
    axR.set_title("流水并行：每 token 只传 7 次隐状态（共 50KB），\n28~90µs ≈ 墙钟的 1%——走网络都够",fontsize=12.3,weight="bold")
    axR.set_ylim(0,8200)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_tp_pp.png",dpi=140); plt.close()

# ===== fig3 decode 步解剖 =====
def fig3():
    fig,(axL,axR)=plt.subplots(1,2,figsize=(13.4,5.9),gridspec_kw={"width_ratios":[1.15,1]})
    cat=nsys["category_per_step_avg"]
    order=["gemv/gemm(cublas)","elementwise/激活","copy/cat/index","norm(reduce/pow/rsqrt/mean)","flash_attention"]
    labels=["GEMV/GEMM 投影","逐元素(silu/mul/add…)","KV cat/copy/index","RMSNorm 族","FlashAttention"]
    us=[cat[k][1] for k in order]; cnt=[cat[k][0] for k in order]
    cols=[RED,"#D98B7A",ORANGE,"#C7B26B",PURPLE]
    wed,_,_=axL.pie(us,labels=[f"{l}\n{c:.0f}个/{v:.0f}µs" for l,c,v in zip(labels,cnt,us)],
                  colors=cols,autopct="%1.0f%%",startangle=90,
                  textprops={"fontsize":9.6},wedgeprops={"edgecolor":"white"})
    axL.set_title("一个 decode 步的 GPU 2,478µs 花在哪\n（980 个小 kernel，nsys 实测分类）",fontsize=12.3,weight="bold")

    # 右：代表 kernel 的 block 总数 vs 114 SM
    items=[("FlashAttention\n(decode)",1*1*2),("RMSNorm reduce",1),("逐元素 silu 等",5),
           ("KV cat(最大)",228*2),("lm_head GEMV",18992)]
    names=[i[0] for i in items]; blocks=[min(i[1],2600) for i in items]
    raw=[i[1] for i in items]
    yp=np.arange(len(items))[::-1]
    bars=axR.barh(yp,blocks,color=[PURPLE,"#C7B26B","#D98B7A",ORANGE,RED],height=0.58)
    for yi,v,rawv in zip(yp,blocks,raw):
        t=f"{rawv:,} block" if rawv<=2600 else f"{rawv:,} block（截顶）"
        axR.text(v+180,yi,t,va="center",fontsize=9.8)
    axR.axvline(114,color=GREEN,ls="--",lw=1.6)
    axR.text(2050,2.55,"绿色虚线 = 114 SM\n（超过它才算铺满）",color=GREEN,fontsize=9.5,ha="center",va="center")
    axR.set_yticks(yp); axR.set_yticklabels(names,fontsize=10)
    axR.set_xlabel("kernel launch 的总 block 数",fontsize=11)
    axR.set_xlim(0,2600)
    axR.set_title("第二个痛点：小 kernel 根本铺不满 114 SM\n（FA 仅 2 块、norm 仅 1 块在跑）",fontsize=12.3,weight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_decode_anatomy.png",dpi=140); plt.close()

# ===== fig4 选型成本 =====
def fig4():
    fig,(axL,axR)=plt.subplots(1,2,figsize=(13.4,5.8),gridspec_kw={"width_ratios":[1,1.15]})
    s7=[s for s in sel["scenarios"] if s["model"]=="7B-Q4"]
    names=["4090D","L40S","A100-80G","H100-80G"]
    cost=[s["usd_per_mtok"] for s in s7]
    wall=[s["per_token_wall_ms"] for s in s7]
    x=np.arange(4)
    b1=axL.bar(x-0.2,cost,width=0.38,color=BLUE,label="$ / 百万 token")
    axL.set_xticks(x); axL.set_xticklabels(names,fontsize=10)
    axL.set_ylabel("美元 / 百万输出 token（32并发·教学估算）",color=BLUE,fontsize=9.8)
    axL.tick_params(axis="y",labelcolor=BLUE)
    for xi,v in zip(x,cost): axL.text(xi-0.2,v+0.01,f"${v:g}",ha="center",fontsize=9.5)
    ax2=axL.twinx()
    ax2.bar(x+0.2,wall,width=0.38,color=ORANGE,label="单token墙(ms)")
    ax2.set_ylabel("t_token 带宽墙（ms）",color=ORANGE,fontsize=10)
    ax2.tick_params(axis="y",labelcolor=ORANGE); ax2.set_ylim(0,6)
    for xi,v in zip(x,wall): ax2.text(xi+0.2,v+0.1,f"{v}",ha="center",fontsize=9.5,color=ORANGE)
    axL.set_title("7B-Q4：便宜卡按 token 反而划算\n（单流延迟敏感才上 A100/H100）",fontsize=12,weight="bold")

    # 右：三模型在 H100 的显存堆叠
    rows=[("7B-Q4×32并发",3.26,3.5),("32B-bf16×16并发",59.6,32.0),("70B-Q4×16并发",32.6,16.0)]
    yp=np.arange(3)[::-1]
    w=np.array([r[1] for r in rows]); kv=np.array([r[2] for r in rows]); other=1.5
    axR.barh(yp,w,color=BLUE,label="权重")
    axR.barh(yp,kv,left=w,color=ORANGE,label="KV（ctx 4096）")
    axR.barh(yp,other,left=w+kv,color=GRAY,label="余量/激活")
    for yi,(n,a,b) in zip(yp,rows):
        axR.text(a+b+2.2,yi,f"{a+b+other:.1f} GiB",va="center",fontsize=10,weight="bold")
    axR.axvline(80,color=GREEN,ls="--",lw=1.5); axR.text(80,-0.72,"单卡 80GB",color=GREEN,ha="center",fontsize=9.5)
    axR.set_yticks(yp); axR.set_yticklabels([r[0] for r in rows],fontsize=10)
    axR.set_xlabel("显存需求（GiB）",fontsize=11); axR.set_xlim(0,105)
    axR.set_title("显存账：32B-bf16 权重 59.6GiB 已逼满单卡\n（+KV 必须多卡/量化）",fontsize=12,weight="bold")
    axR.legend(loc="lower right",fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_selection.png",dpi=140); plt.close()

fig1();fig2();fig3();fig4()
print("written to",OUT)
