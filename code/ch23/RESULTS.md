# ch23 数据源与算账登记（服务化：API/K8S/可观测/容量/稳定性）

> 本章零性能跑测。所有数字为：① 官方文档原文（标来源与检索日期 **2026-10-17**）；② 本书锚点引用（ch05/ch16/ch19 登记值）；③ 零成本算术（演示口径，回填自家压测数据）。

## 一、数据源总表（官方口径，检索 2026-10-17）

| 数据 | 值/原文 | 来源 | 口径 |
|---|---|---|---|
| GPU 只写 limits | "GPUs are only supposed to be specified in the `limits` section"；只写 limit 时 K8s 以 limit 作 request | kubernetes.io《Schedule GPUs》 | 官方口径 |
| Device Plugin | 厂商插件 DaemonSet 上报资源名 `nvidia.com/gpu` / `amd.com/gpu` | kubernetes.io《Device Plugins》 | 官方口径 |
| 时间片 vs MIG | "time-slicing trades the memory and fault-isolation that is provided by MIG for the ability to share a GPU by a larger number of users" | NVIDIA GPU Operator《Time-Slicing》 | 官方口径（原话） |
| MIG | Ampere+（A100/H100）硬件切分，显存+故障隔离 | NVIDIA GPU Operator《MIG》 | 官方口径 |
| Volcano 混部 | "在离线混部——统一调度、动态资源超卖、CPU Burst、资源隔离……保障在线业务 QoS" | volcano.sh 首页 | 官方口径（原话） |
| Volcano Gang | PodGroup `minMember`（v1.12.0：annotation `scheduling.volcano.sh/group-min-member`） | Volcano v1.12.0 发布说明 | 官方口径 |
| Kthena（Volcano 2026） | PD 独立扩缩容与比例、ServingGroup Gang 防死锁、拓扑感知 | volcano.sh 官方博客 | 官方口径 |
| vLLM 指标端点 | `/metrics`（OpenAI 兼容 API server）；`vllm:num_requests_waiting`、`vllm:gpu_cache_usage_perc`、`vllm:cpu_cache_usage_perc`、`vllm:num_requests_swapped`（Gauge）等 | docs.vllm.ai Production Metrics / v1 Metrics | 官方口径（v0.8.5 与 v1 文档均核到指标名） |
| KEDA scale-to-zero | `minReplicaCount: 0` + prometheus trigger | docs.vllm.ai production-stack《Autoscaling with KEDA》 | 官方口径 |
| KEDA activation threshold | "separates the threshold that activates scaling from the target threshold used to determine replica count" | AWS EKS《Autoscale AI inference with HPA and KEDA》 | 官方口径（原话） |
| Dynamo KEDA 示例 | decode 端点用 **TTFT 信号**的 ScaledObject | docs.nvidia.com/dynamo K8s 文档 | 官方口径 |

本书锚点（只引用不重测）：ch05 四大指标与崩溃曲线；ch16 压测五条控制变量；ch19 三条崩溃线（L1 错误>1%、L2 p99 超 SLO、L3 吞吐封顶/积压）；ch13 v1 无 swap 路径（`vllm:num_requests_swapped` 恒 0 属正常）。

## 二、算账区（零成本算术，print_capacity() 可复现，演示口径）

**Little 定律容量账**：
- 输入（演示值，回填自家数据）：SLO p99 E2E = 10 s；满足 SLO 的单实例吞吐 X = 8 req/s；业务峰值 = 40 req/s
- N = X × R = 8 × 10 = **80 在途请求/实例**
- 实例数 = ceil(40/8) = 5，+2（滚动更新 1 + 故障冗余 1）= **7 实例**
- 静态核对：峰值在途 = 40 req/s × 10 s = 400；400/80 = 5 → 与上式一致

**告警三件套（绑定 ch19 三条崩溃线）**：
- L1：错误率 `rate(finish_reason="error")/rate(total) > 0.01` 持续窗口
- L2：p99 TTFT `histogram_quantile(0.99, ...)` > SLO 持续 2 分钟
- L3：`vllm:num_requests_waiting > 阈值` 或 `vllm:gpu_cache_usage_perc > 0.9`

## 三、你的实测回填区（Day 53–55 动手任务）

- [ ] 任务 A：YAML 五项自查（GPU 只 limits / readiness 等加载 / 2 副本 maxUnavailable 0 / 探针时长依据 / 扩缩信号来自 /metrics）：____
- [ ] 任务 B：自家 SLO 三档容量表（SLO、X、峰值 → N、实例数）：____
- [ ] 任务 B：三条 PromQL 告警表达式（阈值来源标注 ch16/19 压测拐点）：____
- [ ] 任务 C：三级降级桌面演练记录（每级生效时间、指标变化、用户感知、哪级最快/代价最大）：____
- [ ] 冷启动实测：模型加载秒数（readiness initialDelaySeconds 依据）：____

## 四、配图登记

| 图 | 文件 | 生成脚本 | 说明 |
|---|---|---|---|
| 图 1 | images/figures/ch23/fig1_serving_stack.png | code/ch23/make_figures.py::fig1 | 服务化五模块全景结构示意（接入层→K8s→引擎→可观测→容量/降级 + 崩塌曲线锚），零外部图片 |

无外部图片，无需 CREDITS.md。
