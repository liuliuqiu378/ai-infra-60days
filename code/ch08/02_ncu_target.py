"""ncu 目标：真实 lm_head 形状 (M,896)@(896,151936) bf16，命令行选 M=1 / 128。
只跑一次目标 GEMM（ncu --launch-count 1 抓它）。"""
import sys, torch
M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
torch.manual_seed(0)
w = torch.randn(896, 151936, device="cuda", dtype=torch.bfloat16)
x = torch.randn(M, 896, device="cuda", dtype=torch.bfloat16)
for _ in range(5):
    x @ w
torch.cuda.synchronize()
y = x @ w
torch.cuda.synchronize()
print("M =", M, "y sum:", float(y.float().sum()))
