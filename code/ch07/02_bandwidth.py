"""02: 显存层次实测。
A. HBM/GDDR6X 大张量带宽：copy(读1写1)/add(读2写1)/sum(只读)
B. L2 悬崖：工作集 1MB→1GB 扫描，小到能住进 72MB L2 时带宽跳变
C. PCIe：锁页内存 H2D / D2H
D. kernel launch 固定开销：eager 20000 次 vs CUDA Graph 重放
"""
import json, torch, numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"

def time_ms(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# ---------- A. 大张量显存带宽（512 MiB fp32 = 128M 元素）----------
N = 128 * 2**20
a = torch.randn(N, device=dev)
b = torch.randn(N, device=dev)
c = torch.empty_like(a)
giB = N * 4 / 1e9
t_copy = time_ms(lambda: c.copy_(a))                    # 流量 2*size
t_add  = time_ms(lambda: torch.add(a, b, out=c))       # 流量 3*size
t_sum  = time_ms(lambda: torch.sum(a))                  # 流量 1*size（reduce）
t_scale= time_ms(lambda: torch.mul(a, 1.0001, out=c))  # 流量 2*size
big = {
    "size_mib": round(N*4/2**20, 0),
    "copy_ms": round(t_copy,4),  "copy_gbs": round(2*giB/t_copy*1e3, 1),
    "add_ms":  round(t_add,4),   "add_gbs":  round(3*giB/t_add*1e3, 1),
    "sum_ms":  round(t_sum,4),   "sum_read_gbs": round(giB/t_sum*1e3, 1),
    "scale_ms":round(t_scale,4), "scale_gbs":round(2*giB/t_scale*1e3, 1),
}

# ---------- B. L2 悬崖：总流量恒定 4 GiB，按工作集大小决定重复次数 ----------
TARGET_BYTES = 4 * 2**30
sizes = [2**k for k in range(20, 31)]   # 1 MiB .. 1024 MiB
l2 = []
for s in sizes:
    n = s // 4
    x = torch.randn(n, device=dev)
    reps = max(1, TARGET_BYTES // (2*s))
    fn = lambda x=x, reps=reps: [torch.mul(x, 1.0001, out=x) for _ in range(reps)] if False else None
    # 用 Python 循环重复 launch；每 kernel 读+写各一次
    def run(x=x, reps=reps):
        for _ in range(reps):
            torch.mul(x, 1.0001, out=x)
    t = time_ms(run, iters=3, warmup=1) / reps
    l2.append({"size_mib": round(s/2**20, 1), "reps": reps,
               "per_kernel_ms": round(t, 5),
               "gbs": round(2*s/t/1e6, 1)})

# ---------- C. PCIe：256 MiB 锁页内存 ----------
nbytes = 256 * 2**20
host = torch.empty(nbytes//4, dtype=torch.float32).pin_memory()
dst  = torch.empty(nbytes//4, dtype=torch.float32, device=dev)
src  = torch.randn(nbytes//4, device=dev)
t_h2d = time_ms(lambda: dst.copy_(host, non_blocking=False))
t_d2h = time_ms(lambda: host.copy_(src, non_blocking=False))
pcie = {"size_mib": 256,
        "h2d_gbs": round(nbytes/t_h2d/1e6, 1),
        "d2h_gbs": round(nbytes/t_d2h/1e6, 1)}

# ---------- D. launch 开销：1 元素小 kernel ×20000，eager vs CUDA Graph ----------
K = 20000
tiny = torch.ones(1, device=dev)
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(K):
    tiny.add_(1.0)
e.record(); torch.cuda.synchronize()
eager_us = s.elapsed_time(e)*1000/K

g = torch.cuda.CUDAGraph()
for _ in range(3): tiny.add_(1.0)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    for _ in range(K):
        tiny.add_(1.0)
s.record(); g.replay(); e.record(); torch.cuda.synchronize()
graph_us = s.elapsed_time(e)*1000/K

launch = {"launches": K,
          "eager_per_launch_us": round(eager_us, 2),
          "cuda_graph_per_launch_us": round(graph_us, 3),
          "speedup": round(eager_us/graph_us, 1)}

res = {"big_tensor": big, "l2_sweep": l2, "pcie": pcie, "launch": launch}
json.dump(res, open("bandwidth_results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
