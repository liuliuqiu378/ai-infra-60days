#!/bin/bash
# ncu 单 kernel 深度指标（replay 模式较慢，每个 M 约 1-3 分钟）
set -e
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate unsloth
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum"
for M in 1 128; do
  echo "===== M=$M ====="
  ncu --launch-count 1 --kernel-name "s16816gemm" --metrics $METRICS \
      python 02_ncu_target.py $M 2>/dev/null | grep -E "s16816gemm|Duration|Compute \(SM\)|Memory Throughput|Achieved Occupancy|DRAM Frequency|dram__bytes" 
done
