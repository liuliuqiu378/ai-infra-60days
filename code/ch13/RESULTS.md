# ch13 数据源与结果登记（PagedAttention）

新风格：**无性能跑测**。外部公开数据（论文/官方文档/源码）+ 零成本结构性观察，
本文件登记正文中所有关键数字的出处，可复核。

## 数据源总表（检索日期：2026-10-16）

| 来源 | 版本/日期 | 许可 | 用途 |
|---|---|---|---|
| 论文 *Efficient Memory Management for LLM Serving with PagedAttention*（arXiv:2309.06180，SOSP '23） | v1，2023-09-12 提交；DOI 10.1145/3600006.3613165 | CC BY 4.0 | 碎片浪费、吞吐/内存节省全部数字 |
| vLLM 源码 `github.com/vllm-project/vllm` | commit `642c18d`（2026-10-15 clone） | Apache-2.0 | v1 引擎 block manager 走读行号 |
| vLLM 官方文档 `docs.vllm.ai`（engine args / paged_attention / prefix_caching） | latest | Apache-2.0 | block size 默认值、历史文档警告 |
| 本书 ch04（7B KV 56 KiB/token 实测）、ch10、ch12 | 2026-10-15/16 | 本仓库 | 回扣与算账基线 |

## 一、论文关键数字（arXiv:2309.06180，2026-10-16 检索）

| 数字 | 原文位置 | 说明 |
|---|---|---|
| 现有系统 KV 内存实际有效率仅 **20.4%–38.2%** | §3.1 + Fig.2 | 三种浪费：预留 reserved、内部碎片 internal、外部碎片 external |
| OPT-13B 单 token KV = **800 KB**（2×5120×40×2B = 819,200 B = 800 KiB），2048 token 单请求 = **1.6 GB** | §2.1/§3 | 原文公式原样 |
| 吞吐相对 SOTA（FasterTransformer/Orca）提升 **2–4×**，同延迟水平 | Abstract/§6 | 总口径 |
| ShareGPT 上可承受请求率：vs Orca(Oracle) **1.7×–2.7×**、vs Orca(Max) **2.7×–8×**、vs FasterTransformer 最高 **22×** | §6.2 | 同延迟 |
| OPT-13B 同时在批请求数：**2.2×** vs Orca(Oracle)、**4.3×** vs Orca(Max) | §6.2 Fig.13(a) | ShareGPT 2 reqs/s |
| parallel sampling 共享节省：Alpaca **6.1%–9.8%**、ShareGPT **16.2%–30.5%** | §6.3 Fig.15 | 仅 prompt 共享 |
| beam search 共享节省：Alpaca **37.6%–55.2%**、ShareGPT **44.3%–66.3%** | §6.3 Fig.15 | 分支共享更多 |
| 共享前缀吞吐：1 example(80 tok) **1.67×**、5 examples(341 tok) **3.58×** vs Orca(Oracle) | §6.3 Fig.16 | 翻译任务 |
| prompt 占 KV 总量 **12%**（其 parallel sampling 实验） | §3 | 共享收益的量级来源 |
| block size **16–128** 在 ShareGPT 最优；16 在多数负载足够，vLLM 默认 **16** | §7.2 | 论文自己定的默认值 |
| recompute 开销**永不超过 swap 延迟的 20%**；block 16–64 两者端到端相当 | §7.3 | 为什么偏向 recompute |
| 13B 模型在 A100 40GB 的显存分布图 | Fig.1 | 权重常驻 + KV 每请求 + 激活临时 |

## 二、vLLM v1 源码事实（commit 642c18d，2026-10-16 grep 核验）

| 事实 | 文件:行号 |
|---|---|
| `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` | `vllm/config/cache.py:72` |
| `enable_prefix_caching: bool = True`（v1 默认开） | `vllm/config/cache.py:142` |
| `block_size is None → 16`（validator） | `vllm/config/cache.py:344` |
| `watermark: float = Field(default=0.0, …)`（默认关） | `vllm/config/scheduler.py:196` |
| `KVCacheBlock` 结构体（`block_id`/`ref_cnt`/`_block_hash`/空闲链表指针） | `vllm/v1/core/kv_cache_utils.py:162-184` |
| `FreeKVCacheBlockQueue`（空闲块双向链表） | `vllm/v1/core/kv_cache_utils.py:258` |
| `class KVCacheManager` | `vllm/v1/core/kv_cache_manager.py:131` |
| `watermark_blocks = int(watermark * num_blocks)` | `vllm/v1/core/kv_cache_manager.py:208` |
| `allocate_slots()`（含 prefix 命中分段图 ASCII 注释 `Blocks layout`） | `vllm/v1/core/kv_cache_manager.py:371`（分段图 :420 起） |
| watermark 仅对 WAITING/PREEMPTED 生效 | `vllm/v1/core/kv_cache_manager.py:509-516` |
| `class Scheduler`；抢占策略（PRIORITY/最晚到者 `running[-1]`） | `vllm/v1/core/sched/scheduler.py:84`、`:848-857` |
| `_preempt_request()`：释放块、`num_computed_tokens = 0`、`num_preemptions += 1` | `vllm/v1/core/sched/scheduler.py:1649`、`:1667`、`:1671`、`:1686` |
| `class BlockPool`；`free_blocks()` `ref_cnt -= 1` 归零回空闲链表 | `vllm/v1/core/block_pool.py:136`、`:776`、`:795-796` |
| `BlockTable.compute_slot_mapping()`（kernel 取数的地址翻译） | `vllm/v1/worker/block_table.py:153` |
| `commit_block_table()`（CPU→GPU 拷贝 block table） | `vllm/v1/worker/block_table.py:181` |
| v1 调度器无 swap 路径（`grep -c swap` = 0；抢占即 free + 重算） | `vllm/v1/core/sched/scheduler.py` |
| CPU MLA 后端强制 block_size=16 | `vllm/platforms/cpu.py:276` |

## 三、配图清单（程序生成 + 1 张论文真实图）

| 图 | 类型 | 数据口径 |
|---|---|---|
| `images/figures/ch13/fig1_waste.png` | 程序生成（`code/ch13/make_figures.py`） | 左条比例示意取论文 20.4% 下限为基准（20.4/30/34/15.6 为教学示意拆分，标题已注明"示意"）；右为按需分配示意 |
| `images/figures/ch13/fig2_block_table.png` | 程序生成（同上） | 块号/引用计数为论文 Fig.7/8 场景的教学复刻 |
| `images/real/vllm-pagedattention-system-overview.png` | 论文原图 Fig.4 | CC BY 4.0，登记于 `images/real/CREDITS.md` |

## 四、算账示例（正文用，非实测）

7B（ch04 实测 56 KiB/token）在 24GB 卡上的"有效 KV 容量放大"账：

- 传统：有效内存最低 20.4%（论文下限）→ 若 KV 池 16 GiB，实际可用 ≈ 3.3 GiB
- 分页：浪费锁进单块（≥96% 有效，对应"near-zero waste"口径）→ 同池可用 ≈ 15.4 GiB
- 比值 ≈ 4.7×——与论文"2–4× 吞吐"同量级（吞吐还受算力/SLO 墙约束，见 ch12 `B_max = min(...)`）

> 口径提醒：4.7× 是**容量比**，不是吞吐承诺；吞吐受三道墙中最小者约束（ch12 §4）。
