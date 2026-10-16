"""nsys 被测目标：ch06 的 KV decode 循环（64 步），NVTX 标注 prefill 与每个 decode 步。"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch06"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_generate import sample_token
from transformers import DynamicCache

name = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
ids = tok("用一句话解释 KV Cache：", return_tensors="pt").input_ids.cuda()

NSTEPS = 64
torch.cuda.synchronize()
with torch.no_grad():
    cache = DynamicCache()
    torch.cuda.nvtx.range_push("prefill")
    out = model(ids, past_key_values=cache, use_cache=True)
    torch.cuda.nvtx.range_pop()
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    for i in range(NSTEPS - 1):
        torch.cuda.nvtx.range_push(f"decode_{i}")
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("done", NSTEPS)
