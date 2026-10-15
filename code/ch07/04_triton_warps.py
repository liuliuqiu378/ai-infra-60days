"""04: 第一个 Triton kernel，亲眼看见两个硬件事实。
A. 访存合并(coalescing)：warp 内 32 线程连续地址→一次 128B 事务；跨步→带宽崩
B. 占用率(occupancy)：grid 恰好 114 块(每 SM 一块)时，num_warps 决定
   单个 block 能拿出多少 warp 掩盖显存延迟
"""
import json, torch, triton, triton.language as tl

@triton.jit
def vadd_stride(x_ptr, o_ptr, n_logical, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid*BLOCK + tl.arange(0, BLOCK)
    addr = i * STRIDE
    m = addr < n_logical
    x = tl.load(x_ptr + addr, mask=m)
    tl.store(o_ptr + addr, x + 1.0, mask=m)

@triton.jit
def vadd_chunk(x_ptr, o_ptr, chunk, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * chunk
    off = 0
    while off < chunk:
        i = base + off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + i)
        tl.store(o_ptr + i, x + 1.0)
        off += BLOCK

SM = 114
def timed(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

# ---------- A. coalescing：固定 64M 个【有用】元素，跨步从 1→32 ----------
NLOG = 32*2**20
BLOCK = 1024
coalesce = []
for stride in [1, 2, 4, 8, 16, 32]:
    a = torch.zeros(NLOG*stride, device="cuda")
    o = torch.empty_like(a)
    grid = lambda meta, s=stride: (triton.cdiv(NLOG, BLOCK),)
    fn = lambda g=grid: vadd_stride[g](a, o, NLOG*stride, STRIDE=stride, BLOCK=BLOCK)
    t = timed(fn)
    useful_gbs = 2*NLOG*4/t/1e6          # 只按【有用字节】算，实际搬运远不止
    coalesce.append({"stride_floats": stride, "byte_gap": stride*4,
                     "footprint_mib": round(a.numel()*4/2**20,1),
                     "ms": round(t,4), "useful_gbs": round(useful_gbs,1)})
    del a, o; torch.cuda.empty_cache()

# ---------- B. occupancy：114 块，每块吃 2 MiB 连续数据，内部循环 ----------
CHUNK = 2*2**20//4                     # 524288 floats = 2MiB
a = torch.zeros(SM*CHUNK, device="cuda")
o = torch.empty_like(a)
occ = []
for w in [1, 2, 4, 8, 16, 32]:
    BLOCK = 32*w                        # 每 tile 每线程恰好 1 个元素：w 小就没别的 warp 可换
    fn = lambda w=w, B=BLOCK: vadd_chunk[(SM,)](a, o, CHUNK, BLOCK=B, num_warps=w)
    t = timed(fn, iters=30)
    occ.append({"num_warps": w, "ms": round(t,4),
                "gbs": round(2*a.numel()*4/t/1e6,1)})

res = {"coalescing_32M_elems": coalesce,
       "occupancy_114blocks_thread1elem": occ}
json.dump(res, open("triton_results.json","w"), indent=2)
print(json.dumps(res, indent=2))
