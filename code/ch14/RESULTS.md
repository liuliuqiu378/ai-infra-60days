# ch14 数据源与结果登记（Continuous Batching 与调度器）

新风格：**无性能跑测**（本机无 vllm 环境，也不拿单机跑分冒充行业结论）。外部公开数据（论文/官方文档/源码）+ 零成本结构性观察，本文件登记正文中所有关键数字的出处，可复核。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 版本/日期 | 许可 | 用途 |
|---|---|---|---|
| 论文 *Orca: A Distributed Serving System for Transformer-Based Generative Models*（OSDI '22, pp.521–538） | USNIX Open Access | USNIX Open Access | 迭代级调度、选择性批处理、36.9× |
| 论文 *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*（arXiv:2403.02310v3，OSDI '24） | v3，2024-06-17 修订 | arXiv 非独占分发 | chunked prefill / stall-free、2.6×/3.7×/5.6× |
| 论文 *DistServe*（arXiv:2401.09670v3，OSDI '24） | v3，2024-06-06 修订 | CC BY-SA 4.0 | PD 分离 7.4×/12.6×（详见 ch16/ch21） |
| 论文 *PagedAttention*（arXiv:2309.06180，SOSP '23） | v1，2023-09-12 | CC BY 4.0 | 回扣：Oracle/Max 口径、2.2×/4.3× |
| vLLM 源码 `github.com/vllm-project/vllm` | commit `642c18d`（2026-10-15 clone） | Apache-2.0 | v1 调度器走读行号与默认值 |
| 本书 ch03/ch05/ch10/ch12/ch13 | 2026-10-15/16 | 本仓库 | 回扣与算账基线 |

## 一、外部论文关键数字（2026-10-16 检索）

| 数字 | 原文位置 | 口径提醒 |
|---|---|---|
| 吞吐 **36.9×** | Orca 摘要 | GPT-3 175B vs NVIDIA FasterTransformer，**同一延迟水平**；含 selective batching 与分布式实现的综合贡献，非单一技术 |
| 新请求"最多等一个迭代" | Orca §3 | 机制描述，非性能数字 |
| 服务容量 **2.6×** | Sarathi-Serve 摘要 | Mistral-7B，1×A100，尾延迟 SLO 约束下 vs vLLM |
| 服务容量最高 **3.7×** | Sarathi-Serve 摘要 | Yi-34B，2×A100（TP2），不同 SLO 目标 |
| 端到端容量最高 **5.6×** | Sarathi-Serve 摘要 | Falcon-180B，PP2×TP4，8×A100 over commodity ethernet（流水线气泡减少） |
| **7.4× 更多请求 / 12.6× 更严 SLO** | DistServe 摘要 | PD 分离，>90% 请求满足 TTFT+TPOT 约束 |
| PagedAttention 同批 **2.2×/4.3×**、beam 省 44%–66% | arXiv:2309.06180 §6.2/§6.3 | 回扣第 13 章，本章只引用不重推 |

## 二、vLLM 源码行号（commit 642c18d，grep 逐一核验）

### 调度主循环 `vllm/v1/core/sched/scheduler.py`

| 行号 | 内容 |
|---|---|
| `:84-85` | `class Scheduler` / `__init__` |
| `:129-141` | `max_num_running_reqs = max_num_seqs`、`max_num_scheduled_tokens` 取值 |
| `:223` / `:227` | `self.waiting` / `self.kv_holding_waiting`（`create_request_queue(policy)`） |
| `:621` | `def schedule(self, throttle_prefills=False)` |
| `:623-632` | **"没有 decoding phase 也没有 prefill phase"** 核心注释（原话） |
| `:641` / `:644` | `token_budget = max_num_scheduled_tokens` / `input_budget = max_num_batched_tokens` |
| `:678-688` | `long_prefill_token_threshold` 生效条件与自适应下限 `input_budget // num` |
| `:714` | 第一段：`while req_index < len(self.running) and token_budget > 0` |
| `:758-767` | `num_new_tokens` = 进度差 + 截断到预算 |
| `:763-764` | 超长 prefill 单步截断到 threshold |
| `:833-841` | `allocate_slots` 循环与判空 |
| `:849-857` | **抢占受害者选择**：PRIORITY 挑 `(priority, arrival_time)` 最大者；否则 `self.running[-1]` |
| `:860` | `_request_blocks_can_be_freed` 不可回收则 `break` |
| `:901-903` | `new_blocks is None` → 本步停止排产 |
| `:955` | 第二段条件：本步无抢占才收新请求 |
| `:967` | 第二段：`while token_budget > 0` |
| `:970-972` | 先出 `kv_holding_waiting`，再出 `waiting` |
| `:978-979` | `max_num_active_reqs` 上限检查 |
| `:981` | `peek_request()` 只看队头 |
| `:1022` | `_get_local_prefix_cache_hit`（前缀命中，回扣 ch13） |
| `:1184` | `num_new_tokens = request.num_tokens - num_computed_tokens` |
| `:1215-1222` | **chunked prefill 关闭时的 `break`** |
| `:1224` | **`min(num_new_tokens, request_token_budget)` ← chunked prefill 一行实现** |
| `:1311-1337` | `allocate_slots`（含 `full_sequence_must_fit=scheduler_reserve_full_isl`）与判空 |
| `:1404-1407` | `pop_request()` + `running.append` 正式入批 |
| `:1426-1430` | 扣预算 + 状态 RUNNING + 进度落账 |
| `:1446-1447` | 未 prefill 完的请求登记 `_inflight_prefills` |
| `:1579-1600` | `SchedulerOutput` 组装 |
| `:1649` | `def _preempt_request(...)` |
| `:1667` | `_free_request_blocks`（释放全部块） |
| `:1670-1671` | `status = PREEMPTED`、**`num_computed_tokens = 0`** |
| `:1686` | `num_preemptions += 1` |
| `:1691` | `waiting.prepend_request(request)` 插回队头 |
| `:1694-1725` | `_update_after_schedule`：`num_computed_tokens += n`（`:1707`）、`is_prefill_chunk` 刷新（`:1712`） |
| `:2107` | `def update_from_output(...)` |
| `:2207` | 投机解码拒绝回滚 `num_computed_tokens -= num_rejected` |
| `:2696-2698` | `get_kv_cache_usage()` |

### 配置 `vllm/config/scheduler.py`

| 行号 | 内容 |
|---|---|
| `:41` | `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` |
| `:43` | `DEFAULT_MAX_NUM_SEQS = 128` |
| `:48` / `:62` | 两参数的 Field 定义 |
| `:79` | `long_prefill_token_threshold` 默认 **0** |
| `:134` | `enable_chunked_prefill: bool = True` |
| `:159` | `policy: SchedulerPolicy = "fcfs"` |
| `:190` | `scheduler_reserve_full_isl: bool = True` |
| `:196` | `watermark` 默认 **0.0**（回扣 ch13） |

### 按硬件/用法分档默认 `vllm/engine/arg_utils.py`

| 行号 | 档位 | `max_num_batched_tokens` | `max_num_seqs` |
|---|---|---|---|
| `:2952-2960` | 显存 ≥160GB（B200/B300 档） | 16384 | 1024 |
| `:2963-2975` | 显存 ≥70GB 且非 A100（H100/H200 档） | 离线 16384 / API **8192** | 1024 |
| `:2977-2984` | 其余（4090/L40S 等消费级与中端卡） | 离线 8192 / API **2048** | 256 |
| `:3130-3160` | 未显式指定时按上表落地；`performance_mode=throughput` 全部 ×2（`:3163-3166`） | — | — |
| `_set_default_chunked_prefill_and_prefix_caching_args` | `enable_chunked_prefill=None` 时由 `model_config.is_chunked_prefill_supported` 决定（`vllm/config/model.py:2171` 按 attention 类型） | — | — |

### 引擎主循环与指标

| 位置 | 内容 |
|---|---|
| `vllm/v1/engine/core.py:657` | `def step(...)` |
| `core.py:667` / `:681` | `schedule()` → `update_from_output()` 三拍子 |
| `vllm/v1/core/sched/request_queue.py:13` | `SchedulingPolicy`（FCFS/PRIORITY） |
| `request_queue.py:75` / `:131` / `:200` | `FCFSRequestQueue`（deque，`popleft` `:82`）/ `PriorityRequestQueue`（堆）/ `create_request_queue` |
| `vllm/v1/metrics/loggers.py:739-740` | Prometheus 计数器 **`vllm:num_preemptions`**（"Cumulative number of preemption from the engine."） |

## 三、图 1 算账口径（教学示例，非实测负载）

示例：4 个请求（A 到达步 0 长 10 步、B 到达步 0 长 6 步、C/D 第 5 步到达长 8/12 步），每步每请求 1 token（只为画图简洁，机制与步长无关）。

| 口径 | static（分两批：A+B，C+D） | continuous |
|---|---|---|
| 批 1 时长 | 10 步（由最长者 A 决定） | A 至第 10 步、B 第 6 步即退出 |
| C/D 等待 | 各等 10 步（共 20 请求·步零产出） | 各等 5 步（共 10 请求·步） |
| B 完成后空占 | 第 6–10 步共 4 步（虚线框） | 0 步 |
| **合计浪费（请求·步）** | **24** | **10** |
| 全部完成时刻 | 第 22 步 | 第 17 步 |

以上为示例时序直接相减的算账结果，图题已注明"算账非实测"；真实负载请求长度方差越大差距越大（ShareGPT 特性回扣 ch13 §1.2）。

## 四、零成本结构性观察

1. 本机 **无 vllm 模块**（`python3 -c "import vllm"` → `ModuleNotFoundError`），故全章不做任何引擎跑测——与第 9 章起的写法一致。
2. 全部源码行号在 /tmp/vllm（commit `642c18d`）中以 `grep -n` 核验，函数名与默认值为逐条读取源码确认。
3. 配图 2 张由本目录 `make_figures.py` 生成，输出至 `../../images/figures/ch14/`，不含外部图片。
