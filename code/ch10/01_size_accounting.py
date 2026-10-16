"""10-01: 体积核算器——零成本、纯标准库、不下载权重不碰 GPU。

数据来源（2026-10-16 抓取，均已核实）：
- 文件体积: HuggingFace API /tree (bartowski/Qwen2.5-{3,7,14}B-Instruct-GGUF)
- 官方参数: HF API safetensors 元数据 (Qwen/Qwen2.5-* 官方仓库)
输出: 有效 bpw = 文件字节×8÷总参数，与名义 bit 的偏差；并做跨模型预测误差验证。
用法: python 01_size_accounting.py
"""

# (GB, bytes 实测: HF 文件树)
SIZES_GB = {
    "14B": {"f16": 29.55, "Q8_0": 15.70, "Q6_K": 12.12, "Q5_K_M": 10.51,
            "Q4_K_M": 8.99, "Q3_K_M": 7.34, "Q2_K": 5.77},
    "7B":  {"f16": 15.24, "Q8_0": 8.10, "Q6_K": 6.25, "Q5_K_M": 5.44,
            "Q4_K_M": 4.68, "Q3_K_M": 3.81, "Q2_K": 3.02},
    "3B":  {"f16": 6.18, "Q8_0": 3.29, "Q6_K": 2.54, "Q5_K_M": 2.22,
            "Q4_K_M": 1.93, "Q3_K_M": 1.59, "Q2_K": 1.27},
}
# 官方总参数 (HF safetensors metadata)
PARAMS = {"14B": 14_770_033_664, "7B": 7_615_616_512, "3B": 3_085_938_688}
NOMINAL = {"f16": 16, "Q8_0": 8, "Q6_K": 6, "Q5_K_M": 5, "Q4_K_M": 4,
           "Q3_K_M": 3, "Q2_K": 2}
QUANTS = list(NOMINAL)


def main() -> None:
    print("== 有效 bpw = GB×1e9×8 ÷ 官方总参数（1GB=1e9B，与 HF 一致）==")
    print(f"{'档位':7s} {'名义':>4s} " + " ".join(f"{m+' 实测':>12s}" for m in SIZES_GB))
    bpw: dict[str, dict[str, float]] = {m: {} for m in SIZES_GB}
    for q in QUANTS:
        cells = []
        for m in SIZES_GB:
            bits = SIZES_GB[m][q] * 1e9 * 8 / PARAMS[m]
            bpw[m][q] = bits
            cells.append(f"{bits:12.2f}")
        print(f"{q:7s} {NOMINAL[q]:>4d} " + " ".join(cells))

    print("\n== 跨模型预测验证（用 7B 的 bpw 预测 14B/3B 体积）==")
    for m in ("14B", "3B"):
        q = "Q4_K_M"
        pred = bpw["7B"][q] * PARAMS[m] / 8 / 1e9
        real = SIZES_GB[m][q]
        print(f"{m} {q}: 预测 {pred:.2f} GB vs 实测 {real:.2f} GB"
              f"  误差 {abs(pred-real)/real*100:.1f}%")

    print("\n== 只用名义 bit 做预算的偏差（14B Q4_K_M）==")
    naive = NOMINAL["Q4_K_M"] / 8 * PARAMS["14B"] / 1e9
    real = SIZES_GB["14B"]["Q4_K_M"]
    print(f"名义估算 {naive:.2f} GB vs 实测 {real:.2f} GB，低估 {real-naive:.2f} GB")


if __name__ == "__main__":
    main()
