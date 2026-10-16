# 第 11 章｜llama.cpp 源码精读：从 GGUF 到计算图（Day 20–24）

> **这章讲什么**：把 llama.cpp 的源码当成一张地铁线路图，从 `main()` 出发，走完 **入口 → GGUF 加载 → 计算图构建 → 后端调度执行 → 采样出 token** 五个换乘站。本章不跑任何性能实验，只做一件事：给你一张**文件级源码导航表**和 10 处带行号的一手引用，让你打开任何一台机器上的 llama.cpp 源码都不会迷路。
>
> **为什么重要**：前面 10 章你学的概念——量化格式、KV Cache、Roofline、prefill/decode——在工程里最终都要落到"某一个项目的某一个文件的某一段代码"。面试官问"llama.cpp 是怎么把 Q4_K_M 算成浮点的"，背名词的人答"用 dequant 函数"，读过源码的人能说出"`ggml-quants.c` 里每个 block 一个 `dequant`，被 `ggml.c` 的 `GGML_OP_MUL_MAT` 分发调用"。**源码精读是把"知识"变成"工程直觉"的最后一道工序。**
>
> **学完能干什么**：① 给你一个陌生的 GGUF 和一份 llama.cpp 源码，5 分钟内定位"打开文件→读头→映射→建图→执行→采样"各自的代码位置；② 讲清楚 ggml 的张量/计算图/后端三层设计为什么能一套代码跑 CPU/CUDA/Metal/Vulkan 等 20 多种后端；③ 面试时把"LLM 推理引擎的架构"讲成一个有文件、有函数、有数据结构的实体，而不是一堆名词。
>
> **本章素材**：llama.cpp 源码 commit `e707790c`（2026-10-15 浅克隆），所有行号以该 commit 为准；仓库内自带文档（docs/）为辅。**引用行号随版本漂移，你复现时若对不上，用函数名 grep 定位即可。**

---

## 0. 大白话开场：源码精读到底在"读"什么

新手对源码精读有两个极端恐惧：一是"几万行 C++，我从哪看起"，二是"我看懂了这一段，下一段又不认识了"。两个问题的解药是同一个：**先有地图，再有路线**。

llama.cpp 是个庞然大物，但它的复杂度是**分层的**，每一层只关心一件事：

| 层 | 目录 | 一句话职责 | 对应你学过的概念 |
|---|---|---|---|
| 应用层 | `tools/`、`common/` | 解析命令行、组织交互循环、采样参数 | CLI 用法、采样参数（第 6 章） |
| 推理框架层 | `src/` | 加载模型、管理 KV、构建计算图、生命周期 | 模型加载、KV Cache、prefill/decode（第 3、4 章） |
| 张量计算层 | `ggml/src/` | 张量抽象、计算图执行、后端调度、算子实现 | 张量、算子融合、后端（本章） |
| 硬件层 | `ggml/src/ggml-cuda/` 等 20+ 目录 | 把算子翻译成 CUDA/Metal/Vulkan 内核 | Kernel、显存搬运（第 7 章） |

【类比】读 llama.cpp 像读一座商场：`tools/` 是大堂和导览牌，`src/` 是物业管理层（决定哪家店开在哪），`ggml/src/` 是建筑结构（承重墙、电梯系统），`ggml-cuda/` 等是各家店铺的装修队。你不会一天看完每一块瓷砖，但你必须知道**电梯系统（计算图 + 后端调度）在哪一层**——那是全楼的主动脉。

**本章验收题**（做完本章你应该能张口回答）：

> 给你一个从没见过的 `model.gguf`，`llama-cli -m model.gguf -p "你好"` 之后，程序从打开文件到吐出第一个 token，依次经过了哪几个函数？每个函数在哪个文件？

答案会在本章第 3–7 节走完后自动拼出来，最后一节我们对答案。

---

## 1. 项目定位与 GGUF 设计哲学：为什么它长成这样

### 1.1 llama.cpp 是"为边缘而生"的推理引擎

llama.cpp 的设计目标从第一天起就和 vLLM、TensorRT-LLM 不同：**在消费级硬件（甚至手机、树莓派）上把模型跑起来**。这个定位决定了它的一切工程选择：

| 设计目标 | vLLM/TensorRT-LLM（服务端） | llama.cpp（边缘优先） |
|---|---|---|
| 核心 KPI | 吞吐、并发、SLO | **单请求能跑、内存占用最小** |
| 依赖 | CUDA 生态、Python | 纯 C/C++，零强制依赖，CPU 必须能跑 |
| 权重格式 | safetensors（需框架加载） | **GGUF（自解析、可 mmap）** |
| 精度路线 | FP8/INT8 服务端量化 | 4/5/6/8 bit 全家桶 k-quant |
| 执行模型 | 动态 batch 调度器 | 单 batch 精雕细琢（新版本已加 server） |

这三个差异是连锁的：**因为要零依赖，所以自己定义 GGUF 格式、自己写解析器；因为要在 CPU 上跑，所以张量库（ggml）从一开始就把"内存排布和步长"当成一等公民；因为要省内存，所以全程 mmap、按需分页。**

### 1.2 GGUF 的三个设计决定（对照第 9 章）

第 9 章我们已经从格式角度拆过 GGUF，现在换成"如果你是设计者"的视角，看它为什么这么定：

1. **自描述的键值对头（KV metadata）**。模型架构、词表、超参（层数/头数/RoPE 参数）全部以"字符串 key → 类型化 value"的形式写在文件头。这带来的直接好处是：**加载器不需要认识"这是 Llama 还是 Qwen"就能先读出全部超参**，再决定用哪个架构的建图代码。对比 pickle：pickle 是"反序列化即执行"，模型结构藏在 Python 类里；GGUF 是"先读数据，再查表选代码"——**数据和代码解耦**。
2. **张量按名索引，不做结构化嵌套**。文件里所有张量是扁平列表，靠 `blk.0.attn_q.weight` 这种命名约定归属到层。好处：加载器用"名称 + 形状"双重校验（下一节 `create_tensor` 会看到），错配当场报错而不是算出 NaN。
3. **数据区零拷贝可 mmap**。头解析完后，张量数据区的每个张量只是"offset + 字节数"的记录。操作系统把整页映射进进程地址空间，**第一次真正读某个张量的数据，是计算时的缺页中断**。这就是为什么 llama.cpp 启动快（对量化模型而言）且内存占用接近文件大小——第 9 章实测的"数据区 = 参数 × 每参数字节、零开销"在这里获得了运行时意义。

【坑：把 GGUF 当成"更好的 pickle"】GGUF 和 safetensors 解决的是**存储格式**问题，而"模型怎么算"完全在代码里。你拿到一个 GGUF 不会知道它怎么前向——必须靠文件头里的 `general.architecture` 字段去 `src/models/` 里查对应的建图代码。这也是"转换脚本 convert_hf_to_gguf.py 必须写对架构名"的底层原因：**架构名是数据通向代码的唯一钥匙**。

### 1.3 本章的"宪法"：三层数据结构

整个 ggml 的世界观只有三个结构体，后面所有走读都围绕它们：

- `ggml_tensor`：**一个张量**（形状、步长、类型、数据指针、它是什么运算的输出）；
- `ggml_cgraph`：**一张计算图**（一批张量的有向无环图，节点 = 会算的张量，叶子 = 常量）；
- `ggml_backend_i`：**一种后端**（一组函数指针，告诉调度器"我这种硬件怎么算图"）。

源码精读的正确姿势是：**把每个函数读成"对这三个结构体的读写操作"**。`build_graph` 是往 `cgraph` 里塞 `tensor`；`sched_compute` 是让某个 `backend` 执行这个 `cgraph`；`sampler` 是读最后一个 `tensor`（logits）吐出 token。

---

## 2. 文件级源码导航表（本章核心交付）

以下所有行数为 commit `e707790c` 实测（`wc -l`），是本章的"地图页"。**建议你现在就把这张表存成书签**——后续 30 天遇到任何 llama.cpp 问题，先来这里定位文件，再进去读。

### 2.1 应用层 `tools/` 与 `common/`

| 文件 | 行数 | 职责 | 何时来读 |
|---|---:|---|---|
| `tools/cli/main.cpp` | 5 | 真正的 `main()`，只调 `llama_cli` | 找入口 |
| `tools/cli/cli.cpp` | 68 | `llama_cli()`：解析参数 → 初始化后端 → 起交互 | 看启动顺序 |
| `tools/cli/cli-context.cpp` | 682 | 交互上下文：**本地模式起 server + HTTP client 对话** | 查 CLI 行为 |
| `tools/server/` | — | HTTP 服务（OpenAI 兼容 API） | 部署相关 |
| `common/arg.cpp` | 4,813 | **全部命令行参数定义与解析** | 查任何 `--xxx` 参数含义 |
| `common/common.cpp` | 2,459 | 通用工具（分词、格式化、公共状态） | 查通用行为 |
| `common/sampling.cpp` | 1,091 | **采样链装配**（common_sampler_init） | 调采样参数 |
| `common/speculative.cpp` | 3,115 | 投机解码实现 | 第 20 章 |

> 读参数文档别只看 `--help`：`common/arg.cpp` 是命令行的唯一事实来源，每个参数的类型、默认值、合法范围、帮助文本都在里面。

### 2.2 推理框架层 `src/`

| 文件 | 行数 | 职责 | 何时来读 |
|---|---:|---|---|
| `src/llama.cpp` | 621 | 公共 C API 实现（load_model / new_context 等入口） | 从 API 层进 |
| `src/llama-context.cpp` | 4,719 | **推理主循环**：建图、预留显存、逐 batch 执行 | 理解 prefill/decode 执行 |
| `src/llama-graph.cpp` | 4,251 | **算子级建图**（norm/qkv/ffn/attention 的图构造） | 看前向怎么拼 |
| `src/llama-model.cpp` | — | 模型对象：`build_graph`、超参、张量注册总装 | 看架构装配 |
| `src/llama-model-loader.cpp` | 1,876 | **加载链**：读 GGUF、建张量映射、灌数据 | 排查加载错误 |
| `src/llama-mmap.cpp/.h` | — | 文件内存映射（mmap/mlock） | 查内存映射问题 |
| `src/llama-sampler.cpp` | 4,391 | **采样器实现**（每种 sampler 一个工厂 + 接口表） | 查采样算法 |
| `src/llama-quant.cpp` | 1,546 | 量化工具链（convert/quantize 命令的后端） | 第 10 章量化实操 |
| `src/llama-batch.cpp` | 1,493 | batch/ubatch 切分（决定一次算多少 token） | 深入调度 |
| `src/llama-kv-cache*.cpp` | 系列 | KV Cache 各变体（GQA/recurrent/hybrid/DSA） | 第 4、13 章 |
| `src/llama-vocab.cpp` | — | 分词器（BPE/SPM/WordPiece…） | tokenizer 问题 |
| `src/llama-arch.cpp` | — | 架构/张量命名注册表（KV key 与 tensor 名的字典） | 查名字约定 |
| `src/models/*.cpp` | **162 个文件** | 每个模型架构一个文件（建图 + 张量注册） | 看具体模型前向 |
| `src/llama.cpp` 头文件 `include/llama.h` | — | **对外 C API 唯一接口** | 集成到自己项目 |

### 2.3 张量计算层 `ggml/src/`

| 文件 | 行数 | 职责 | 何时来读 |
|---|---:|---|---|
| `ggml/include/ggml.h` | — | 张量结构、算子枚举、图操作 API | 查结构体/算子定义 |
| `ggml/src/ggml.c` | 8,190 | **CPU 参考实现**：全部算子的 C 实现 + 图执行 | 看算子数学 |
| `ggml/src/ggml-quants.c` | 5,639 | **量化/反量化**（block 级，Q4_K 等全在这） | 量化原理落地 |
| `ggml/src/ggml-common.h` | — | 量化 block 结构定义（`block_q4_K` 等 + 静态断言） | 查块排布 |
| `ggml/src/ggml-backend.cpp` | 2,630 | **后端调度器**（split/reserve/compute） | 多后端协同 |
| `ggml/src/ggml-backend-impl.h` | — | 后端接口 `ggml_backend_i` 定义 | 写新后端 |
| `ggml/src/ggml-backend-reg.cpp` | — | 后端注册/动态加载（dlopen 探测） | 后端发现机制 |
| `ggml/src/ggml-alloc.c` | 1,173 | 图内显存复用分配（算完的 buffer 给下一个节点用） | 内存优化 |
| `ggml/src/gguf.cpp` | 1,731 | **GGUF 读写解析器** | 格式细节 |
| `ggml/src/ggml-cuda/`（含 `ggml-cuda.cu` 6,153 行） | 大 | CUDA 后端：kernel 实现 | GPU 算子 |
| `ggml/src/ggml-metal/` `ggml-vulkan/` `ggml-sycl/` `ggml-hip/` `ggml-cann/` `ggml-rpc/` … | 各自独立 | 其余 20+ 后端 | 移植/跨平台 |

**读表方法**：遇到"现象"先判断在哪层——**参数报错查 `common/arg.cpp`，加载失败查 `llama-model-loader.cpp`+`gguf.cpp`，算错/慢查 `llama-graph.cpp`+`ggml.c`，多卡/多后端问题查 `ggml-backend.cpp`，采样行为查 `llama-sampler.cpp`。**

---

## 3. 走读一：入口链——一个反直觉的事实，CLI 其实是 client

### 3.1 main 只有 5 行

```c
// tools/cli/main.cpp（全文）
int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
    return llama_cli(argc, argv);
}
```

`main()` 干净到没有存在感——这是刻意的：真正的启动逻辑在 `tools/cli/cli.cpp` 的 `llama_cli()`，它做了四件事，顺序固定：

```c
int llama_cli(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) return 1;
    llama_backend_init();               // 初始化后端（含 ggml_backend_load_all）
    llama_numa_init(params.numa);       // NUMA 拓扑
    // ... 注册 SIGINT 信号处理（第二次 Ctrl+C 立即退出）
    cli_context ctx_cli(params);
    if (!ctx_cli.init()) return 1;
    return ctx_cli.run();
}
```

注意 `common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)` 这一行：**4,813 行的 `common/arg.cpp` 就是为这一行服务的**。最后一个参数 `LLAMA_EXAMPLE_CLI` 是"场景过滤器"——同一个仓库里有 CLI/server/quantize 等多个入口，每个入口只允许自己相关的参数子集，防止你给 server 指定一个只有 CLI 才认的参数。

### 3.2 反直觉点：你在终端敲的 llama-cli，本地偷偷起了一个 server

`cli_context::init()`（`tools/cli/cli-context.cpp:94`）的核心逻辑：

```c
bool cli_context::init() {
    // ...
    bool use_external_server = !params.server_base.empty();
    if (use_external_server) {
        // 模式 B：连一个已经在跑的 llama-server
        client.server_base = base;
    } else {
        // 模式 A：本地模式
        server.emplace();
        if (!server->start(params)) { /* 报错 */ }
        if (!server->wait_ready(should_stop)) { /* 报错 */ }
        client.server_base = server->address();   // 关键：client 指向刚起的本地 server
    }
    healthy = client.wait_health(is_aborted);
    // ...
}
```

**也就是说：新版 llama.cpp 的交互式 CLI 不直接调用 llama.cpp 的 C API，而是 spawn 一个内嵌的 llama-server，然后自己作为 HTTP 客户端和它对话。** 推理路径变成 `cli → HTTP → server → llama API → ggml`。

【类比】以前的 CLI 是"厨师直接端盘子上桌"，现在柜台和后厨之间加了一条传送带（HTTP），柜台（cli）甚至可以对接**别家后厨**（`--server-base http://…` 连远程 server）。代价是一点点 HTTP 开销，收益是 **CLI 和 server 永远行为一致**——你在 CLI 里看到的参数行为、流式输出、工具调用，和 API 上完全同源，不会出现"CLI 能跑、server 上变了样"的分裂。

【坑：还在用 2024 年的眼光找 `tools/main/`】历史教程常提 `tools/main` 或 `examples/chat`，2026 年的仓库里入口已改为 `tools/cli/`（main.cpp 只有 5 行），旧路径全部不存在。**找入口的万能方法：`grep -rn "int main(" tools/`**，一行定位。这也是本章反复强调"带 commit 看源码"的原因：行号会漂移，**入口函数和数据结构的名字非常稳定**。

---

## 4. 走读二：加载链——从打开文件到张量就位

加载链回答一个问题：**磁盘上的一坨字节，怎么变成内存里可以参与计算的张量？** 四步：打开解析 → 建元数据 → 映射文件 → 匹配张量。

### 4.1 调用链总览

```
llama_load_model_from_file()            src/llama.cpp:460
  └─ llama_model_load()                 src/llama.cpp:316
       ├─ llama_model_loader ml(...)    src/llama-model-loader.cpp:533   // 构造：读头
       │    └─ gguf_init_from_file()    ggml/src/gguf.cpp:1017
       │         └─ gguf_init_from_reader()  ggml/src/gguf.cpp:460       // 逐字段解析
       ├─ ml.init_mappings()            src/llama-model-loader.cpp:1411  // mmap
       ├─ 架构注册：load_arch_hparams / load_arch_tensors   src/models/*.cpp
       └─ ml.load_all_data()            src/llama-model-loader.cpp:1553  // 校验并填充
```

### 4.2 GGUF 解析：一个字节一个字节的强迫症

`gguf_init_from_reader()`（gguf.cpp:460 起）把文件头按顺序读出来，每一步都有防御性检查，值得逐个看：

| 步骤 | 代码位置 | 检查内容 | 为什么 |
|---|---|---|---|
| 魔数 | gguf.cpp:466-487 | 前 4 字节必须是 `'G','G','U','F'` | 一眼识别"这不是别的格式" |
| 版本 | :493-519 | v1 直接拒载；版本 > 支持上限拒载 | 格式演进的兼容闸门 |
| 端序 | :506-509 | `(version & 0x0000FFFF)==0` 判定大小端不符 | 大端机器读小端文件会把 3 读成 0x30000000 |
| 张量数 | :524-530 | 上限检查（不能超过 `SIZE_MAX/sizeof(info)`） | 防恶意/损坏文件撑爆内存 |
| KV 数 | :535 起 | 同上 | 同上 |

之后是"KV 段"（元数据键值对）和"张量信息段"（每个张量的 name/type/shape/offset），最后记录数据区起点。**这个顺序和第 9 章手读文件头的实验完全一致——当时我们用 Python `struct.unpack` 复刻的就是这段 C 代码的逻辑。**

【类比】解析 GGUF 像入关安检：先看护照封皮（魔数）、再核对签证有效期（版本）、确认护照没拿反（端序）、清点申报物品数量（张量数）。任何一步不对，**当场遣返（返回 nullptr + 错误日志）**，绝不带着可疑行李过关。

### 4.3 mmap：把"读文件"变成"访问内存"

`llama_model_loader::init_mappings()`（llama-model-loader.cpp:1411）为每个模型文件建立 `llama_mmap`（src/llama-mmap.h:47）。要点：

- 数据不 `fread` 进堆内存，而是把文件映射进地址空间；`load_all_data()`（:1553）只负责**把张量指针指向映射区 + 校验形状**，字节搬运交给操作系统缺页；
- 可选 `mlock`：锁定页面防止被换出（`--mlock`），避免推理中途被 swap 卡住；
- 可选 prefetch：预读热数据；
- 多文件（split GGUF）时用 `mapping_ranges`（:1450）管理每个张量落在哪个文件的哪段区间。

**这就是第 9 章"数据区零拷贝"的运行时兑现**：加载 4 GB 的量化模型，实际驻留内存的只有被访问过的部分页。

### 4.4 create_tensor：名称 + 形状双重校验

每个架构的 `load_arch_tensors()` 做的事非常朴素，以 Llama（`src/models/llama.cpp`）为例：

```c
tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
if (output == NULL) {           // 没有独立 lm_head：权重与输入嵌入共享
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
}
for (int i = 0; i < n_layer; ++i) {
    layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
    create_tensor_qkv(layer, i, ...);
    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
    // ...
}
```

`create_tensor`（llama-model-loader.cpp:1117）拿着**规范化名称**和**期望形状**去 GGUF 的张量表里查：

- 名字不存在且标记 `0`（required）→ 报错退出；
- 名字存在但**形状不符** → 报错退出；
- `TENSOR_NOT_REQUIRED` → 允许缺失（可选张量，比如带 bias 的变体）；
- `TENSOR_DUPLICATED` → 同一物理张量被两个逻辑名引用（tied embeddings）。

两个直接推论：**① 转换脚本写错张量名，报错一定发生在这里；② "我的模型是不是共享 lm_head"不用猜，看这里是否走了 DUPLICATED 分支。** 第 10 章量化时若遇到 "missing tensor" 报错，病根就是这行检查在替你踩刹车。

---

## 5. 走读三：计算图——ggml 的心脏

### 5.1 `ggml_tensor`：一个结构体看懂张量的一切（ggml.h:685）

```c
struct ggml_tensor {
    enum ggml_type type;                 // 数据类型（含 Q4_K 等量化类型）
    struct ggml_backend_buffer * buffer; // 数据放在哪个后端 buffer
    int64_t ne[GGML_MAX_DIMS];           // 各维元素数（ne[0] 是最内层维）
    size_t  nb[GGML_MAX_DIMS];           // 各维字节步长：nb[0]=类型大小, nb[i]=nb[i-1]*ne[i-1]
    enum ggml_op op;                     // 这个张量是什么运算的输出（GGML_OP_ADD…）
    int32_t op_params[...];              // 算子参数（如 softmax 的掩码标志）
    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入（图的边）
    struct ggml_tensor * view_src;       // 若是视图：源张量
    size_t view_offs;
    void * data;                         // 数据指针
    char name[GGML_MAX_NAME];            // 名字（报错定位全靠它）
    void * extra;                        // 后端私有数据（如 CUDA 的 kernel 选择信息）
};
```

三个最有教学价值的字段：

1. **`ne`/`nb` 分离**：形状和**内存步长**分开存。这意味着张量可以是连续的，也可以是别人的一部分（view/转置/量化切片），算子只需按 `nb` 跳着读，不用管布局怎么来的。这是 ggml 能高效支持"权重不复制直接参与计算"的基础。
2. **`op` + `src` = 图的节点与边**：张量自己携带"我是什么运算、我吃哪些张量"。**图不需要额外的 Node 类，张量即节点**——这是 ggml 和 PyTorch autograd 图、TensorFlow GraphDef 的根本差异（后两者是独立的图对象包着张量）。
3. **`type` 是量化的一等公民**：`GGML_TYPE_Q4_K` 和 `GGML_TYPE_F32` 平级。所以第 10 章的量化模型在计算图里**没有任何特殊节点**，只是乘法算子看到输入类型是 Q4_K 时走"块反量化"分支——格式（第 9 章）和执行（本章）在这里接头。

### 5.2 `ggml_cgraph`：节点数组 + 哈希去重（ggml-impl.h:341）

```c
struct ggml_cgraph {
    int size;        // 预分配容量
    int n_nodes;     // 当前节点数（会被求值的张量）
    int n_leafs;     // 叶子数（常量，如权重、掩码）
    struct ggml_tensor ** nodes;
    struct ggml_tensor ** leafs;
    struct ggml_hash_set visited_hash_set;  // 去重：同一张量只入图一次
    enum ggml_cgraph_eval_order order;      // 求值顺序
    uint64_t uid;    // 图指纹：两次 uid 相同 → 可视为同一张图
};
```

建图用 `ggml_build_forward_expand(gf, tensor)`：**从最终张量递归回溯所有 `src`，整棵依赖树挂进 nodes**（相同张量靠 `visited_hash_set` 只挂一次）。注意 `ggml.c` 里的 `GGML_OP_NONE` 叶子——权重是叶子，永远不"被算"，只被读。

### 5.3 build_graph：从"层"到"图"的总装

每种架构一个 `src/models/xxx.cpp`，但总装逻辑在 `llama_model::build_graph()`（llama-model.cpp:3307）：

```c
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params); // ① 架构专属的 Transformer 层堆叠
    llm->build_pooling(...);      // ② 池化（embedding 类模型）
    llm->build_logits();          // ③ lm_head → logits
    llm->build_sampling();        // ④ 可选：把采样也放进图（GPU 端采样，--sampling-gpu）
    llm->res->set_outputs(params);
    return llm->res->get_gf();
}
```

第 ①步进入 `src/models/llama.cpp` 这类文件，里面调用的是 `llama-graph.cpp` 提供的**算子级积木**（这就是为什么 162 个模型文件能保持不臃肿——它们只是积木的排列组合）：

| 积木函数 | llama-graph.cpp 行号 | 拼出什么 |
|---|---:|---|
| `build_inp_embd` | 2561 | 图的入口：token ids → 嵌入查表 |
| `build_norm` | 1706 | RMSNorm/LayerNorm |
| `build_qkv` | 1742 | Q/K/V 投影（含 fused QKV 与 LoRA 分支） |
| `build_ffn` | 1871 | SwiGLU 等前馈网络 |
| `build_attn_mha` | 2882 | 注意力核心（QKᵀ·V 的图） |
| `build_moe_ffn` | 2090 | MoE 专家路由 |

`llm_graph_result`（:1365）包着产物 `gf` 和输入输出张量表，还有一个关键方法 **`can_reuse(params)`（:1464）**——本次请求形状和上次相同就**复用已构建的图**，只换输入数据。这是"每步 decode 都要重建图吗？"的标准答案：**建图可以免，但要看参数匹配**。

### 5.4 graph_reserve + graph_compute：一次预留，每步执行

`llama_context::graph_reserve`（llama-context.cpp:2621）是"开工前的场地布置"：

```c
auto * res = gf_res_reserve.get();
const auto gparams = graph_params(res, ubatch, mctx, ...);
res->reset();
auto * gf = model.build_graph(gparams);              // 构建一张样板图
// ...
if (!ggml_backend_sched_reserve(sched.get(), gf)) {   // 让调度器按样板图预分配全部工作内存
    LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
    return nullptr;
}
```

**样板图（measure graph）不执行，只用来量出"运行这张图需要多少中间显存"**——这就是第 4 章说的"激活/工作区预分配"的代码出处。之后每个 decode step 走 `graph_compute`（:2731）：

```c
// 设置各后端线程数（prefill/decode 可用不同线程池）
auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);  // :2752
```

一行调度，下面第 6 节展开。

【类比】`graph_reserve` 是剧院开演前的安全演练和道具摆位：先拿一张"今晚节目单"（样板图）排一遍，确定灯光音响占多大地方；正式演出（每步 decode）就只管按固定走位演——**走位不变，每场的成本才可预测**。

---

## 6. 走读四：后端抽象——一张图怎么跑在 20 种硬件上

### 6.1 `ggml_backend_i`：C 语言手写的"虚函数表"（ggml-backend-impl.h:122）

```c
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t);
    void (*free)(ggml_backend_t);
    // 异步数据搬运
    void (*set_tensor_async)(...);
    void (*get_tensor_async)(...);
    bool (*cpy_tensor_async)(ggml_backend_t src, ggml_backend_t dst, ...);
    void (*synchronize)(ggml_backend_t);
    // 图执行（两种模式：带计划 / 直接算）
    ggml_backend_graph_plan_t (*graph_plan_create)(...);
    enum ggml_status (*graph_plan_compute)(ggml_backend_t, ggml_backend_graph_plan_t);
    enum ggml_status (*graph_compute)(ggml_backend_t, struct ggml_cgraph *);   // 核心
    // 事件同步、图优化
    void (*event_record)(...);
    void (*graph_optimize)(...);
};
```

这就是 C++ 虚函数表的手写版：**后端 = 一个填满函数指针的结构体 + 一个 context 指针**（`struct ggml_backend`，:159）。CUDA 后端填 CUDA 版本，CPU 后端填 C 版本，调度器只认接口。关键理解：

- **`graph_compute` 是唯一必需品**：任何后端只要能"给我一张图，我把它算完"，就能接入；
- `set/get_tensor_async` + `cpy_tensor_async`：权重和激活在不同设备间搬运的统一通道——**多卡、CPU+GPU 混跑的本质就是这些异步拷贝节点被插进图里**（第 12 章 offload 的地基）；
- `graph_optimize`：后端可以自行重排节点（如融合、按 kernel 特性排序）。

### 6.2 调度器 split：按张量归属切图（ggml-backend.cpp）

调度器 `ggml_backend_sched`（`:1949 ggml_backend_sched_new`）持有多个后端，执行分三步：

1. **定归属**：`ggml_backend_sched_backend_id_from_cur`（:1069）给每个节点/叶子/源张量分配后端 id（规则：张量所在 buffer 属于谁就归谁；CPU 是兜底）。GPU 层设置为 0（全 CPU）时，整张图只有一个 split——**这就是 `-ngl 0` 与 `-ngl 99` 的结构差异：split 数量不同**。
2. **切图**：`ggml_backend_sched_split_graph`（:1214）把连续归属相同的节点切成子图，在**跨后端边界插入搬运节点**。
3. **分配 + 执行**：`ggml_backend_sched_reserve`（:2076）/ `alloc_graph`（:2093）为每个 split 分配工作内存（跨 split 的 buffer 复用，`ggml-alloc.c` 负责"算完这块、下个节点接着用这块"），然后逐 split 调各自 `graph_compute`，用 event/`cpy_tensor_async` 做衔接。

### 6.3 注册与动态加载：后端是可插拔的（ggml-backend-reg.cpp）

- 静态注册：`ggml_backend_register`（:298）——编译期带进来的后端（CUDA 编译了就有）；
- 动态加载：`ggml_backend_load_best`（:480）按名字（`blas`、`zendnn`、`cann`…）在库里找 `ggml_backend_init` 入口符号，dlopen 载入；版本号不符直接拒绝（:252）；
- 启动总入口 `ggml_backend_load_all`（:567）：`llama_backend_init()` 会走到它，依次探测 blas/zendnn/cann/…。

对照 `ggml/src/` 下的 20+ 后端目录（cuda、hip、metal、vulkan、sycl、cann、opencl、openvino、rpc、webgpu、cpu、blas…），你可以得出本章最重要的架构结论：

> **ggml = 一张中立的计算图 + 一个认图不认硬件的调度器 + 一堆只会回答"我会不会算这张图"的后端。** 新硬件（比如国产 NPU）的接入成本被压缩到"实现 `ggml_backend_i` + 目录挂进 CMake"，这正是 llama.cpp 能比任何 Python 推理框架都更快铺满新硬件的原因。

【坑：RPC 后端常被忽视，但面试很加分】`ggml-rpc` 是把"图序列化后发给远端机器执行"的后端——它证明了后端抽象的边界之清晰：**连"远程机器"都可以假装是一个能算图的本地设备**。被问"llama.cpp 支持分布式吗"时，这是比"支持多卡 split"更有深度的答案。

---

## 7. 走读五：采样链——logits 变成 token 的流水线

### 7.1 接口：每个 sampler 都是同一张表（llama-sampler.cpp）

所有采样器实现同一组函数指针 `struct llama_sampler_i`（名字、apply、accept、reset、clone…），工厂函数成对出现：

| 工厂 | llama-sampler.cpp 行号 | 干什么 |
|---|---:|---|
| `llama_sampler_chain_init` | 870 | 建一条空链 |
| `llama_sampler_init_greedy` | 1113 | argmax（确定性） |
| `llama_sampler_init_dist` | 1405 | 按分布随机采样 |
| `llama_sampler_init_top_k` | 1525 | 截断到前 K |
| `llama_sampler_init_top_p` | 1725 | 核采样 |
| `llama_sampler_init_min_p` | 1888 | 相对概率阈值 |
| `llama_sampler_init_temp_ext` | 2313 | 温度（支持动态温度） |
| `llama_sampler_init_grammar` | 2694 | 语法约束（GBNF） |

**链 = 容器**（`llama_sampler_chain_i`，:855）：`chain_apply` 顺着链依次让每个 sampler 改写候选分布，最后一个 `dist`/`greedy` 出最终 token。

### 7.2 装配顺序：`common_sampler_init`（common/sampling.cpp:191）

装配代码（简化，行号见括号）：

```c
llama_sampler * chain = llama_sampler_chain_init(lparams);          // :212
// ① 先做"与分布无关"的准备：语法 grmr、logit_bias（含模型自带 suppress tokens）  :340
// ② 按 params.samplers 的用户顺序追加：
for (const auto & cnstr : params.samplers) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:          ... init_dry(...)          // :357
        case COMMON_SAMPLER_TYPE_TOP_K:        ... init_top_k(...)         // :361
        case COMMON_SAMPLER_TYPE_TOP_P:        ... init_top_p(...)         // :364
        case COMMON_SAMPLER_TYPE_MIN_P:        ... init_min_p(...)         // :370
        case COMMON_SAMPLER_TYPE_TEMPERATURE:  ... init_temp_ext(...)      // :379
        case COMMON_SAMPLER_TYPE_PENALTIES:    ... init_penalties(...)     // :385
    }
}
// ③ 收尾二选一（greedy 判定，:403-406）：
const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
    ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && ...) ||
     (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
// ④ 全部上链（:419）
for (auto * smpl : samplers) llama_sampler_chain_add(chain, smpl);
```

三个可直接用于面试的观察：

1. **链的顺序就是你 `-s top_k,top_p,temp` 命令行的顺序**——llama.cpp 不擅自重排用户指定的采样器（默认顺序在参数解析层给出），因为截断和温度的先后会影响分布（先 temp 后 top_p 与反过来结果不同）。
2. **greedy 不是一个命令，而是"退化条件"的判定**：温度为 0 且排在末位，或 top_k=1 → 尾部换成 `greedy`；否则尾部必须是 `dist` 才能出随机 token。**如果你设置 temp=0 却得到随机输出，先检查是不是链尾被别的采样器盖住了。**
3. **概率输出（`n_probs`）会抑制 greedy 判定**——因为 argmax 没有分布可以返回。

### 7.3 与第 6 章 mini_generate 的对照

第 6 章我们 100 行手写的 `sample()` 只有 softmax→temperature→top-k→top-p→argmax 这条主干，与本章对齐后，差距地图清晰了：

| 能力 | mini_generate | llama.cpp |
|---|---|---|
| 链式组合 | 硬编码顺序 | 任意顺序、动态装配 |
| 分布修正 | 无 | penalties、DRY、logit_bias、XTC |
| 约束解码 | 无 | GBNF/llguidance 语法 sampler |
| 自适应熵控制 | 无 | mirostat v1/v2 |
| 复现性 | 全局种子 | 每条链独立 seed、可 clone/保存状态 |
| GPU 内采样 | 不支持 | `build_sampling()` 进图（图 5.3 节 ④） |

**这不是功能堆砌的差异，而是"采样被当成流水线还是当成函数"的架构差异。**

---

## 8. 源码全局地图：162 个模型文件意味着什么

`src/models/` 下有 **162 个 `.cpp` 文件**，每个对应一个架构。用本章地图复述它的插件化设计：

| 关注点 | 由谁负责 | 文件 |
|---|---|---|
| "这个架构的超参怎么从 GGUF 读" | `load_arch_hparams` | `src/models/<arch>.cpp` |
| "这个架构有哪些张量、各叫什么、多大" | `load_arch_tensors` + `create_tensor` | 同上（命名字典在 `llama-arch.cpp`） |
| "这个架构的前向怎么拼" | `build_arch_graph` 调积木 | 同上（积木在 `llama-graph.cpp`） |
| "所有架构共通的收尾" | logits/sampling/pooling | `llama-model.cpp:3307` |

所以"支持一个新模型"在 llama.cpp 里是**一个文件的增量**（外加转换脚本的一个分支），不需要动加载器、调度器、后端任何一层——这就是第 9 章说的"数据（GGUF）与代码（models/）解耦"在工程组织上的形态。

行数分布也印证了分层（commit e707790c 实测）：

| 文件 | 行数 | 解读 |
|---|---:|---|
| `ggml/src/ggml.c` | 8,190 | 全部 CPU 算子参考实现——**数学的唯一权威版本** |
| `ggml/src/ggml-cuda.cu` | 6,153 | CUDA 后端（不含同目录其余文件） |
| `ggml/src/ggml-quants.c` | 5,639 | 量化/反量化——第 9、10 章的代码落点 |
| `src/llama-context.cpp` | 4,719 | 推理主循环 |
| `src/llama-sampler.cpp` | 4,391 | 采样器全家 |
| `src/llama-graph.cpp` | 4,251 | 算子积木 |
| `common/arg.cpp` | 4,813 | 命令行参数 |

【类比】这像一家餐厅的员工结构：`ggml.c` 是总厨（所有菜的标准做法），`ggml-cuda.cu` 是各分店的炉灶规格，`llama-graph.cpp` 是摆盘手册，`models/` 的 162 个文件是 162 道菜的菜谱，`common/arg.cpp` 是前厅点单系统。**换炉灶（新后端）不动菜谱，新菜（新模型）不动总厨。**

---

## 9. 坑清单、动手任务与面试话术

### 9.1 【坑】清单（源码阅读特供版）

1. **【坑】照着旧教程找文件**：`tools/main`、`examples/` 已消失，入口是 `tools/cli/`；行号更是随版本漂移。**定位靠函数名 grep，引用必带 commit。**
2. **【坑】以为图每步都重建**：`llm_graph_result::can_reuse`（llama-graph.cpp:1464）会在参数匹配时复用图；真正每步变的只是输入 ubatch 和 KV 写入位置。追问自己"哪些输入变了、哪些没变"，才算真懂 prefill/decode 复用。
3. **【坑】以为量化模型在图里有特殊分支**：量化只是 `ggml_tensor.type` 的取值；分叉发生在 `ggml.c` 的 `GGML_OP_MUL_MAT` 内部按 src 类型选 dequant 路径，**图结构与 FP16 模型完全同构**。
4. **【坑】把 CLI 行为等同于 API 行为**：新版 CLI 走 HTTP（3.2 节），CLI 的流式、中断、多模态行为由 server 实现；排查"CLI 好好的、我自己调 API 不对"时，先对齐你和 CLI 的实际调用路径。
5. **【坑】改了采样顺序却发现"没效果"**：链尾必须是 greedy/dist 才会真正选 token；`temp=0` 若不在末位（或被 mirostat 分支覆盖，sampling.cpp:408-413 会整体换链），行为完全不同。
6. **【坑】忽略 `extra` 字段和 buffer 归属**：同一个逻辑张量在不同 buffer（CPU/GPU）意味着 split 边界；多卡调试时"张量在谁的 buffer 上"是第一问。

### 9.2 动手任务（纯源码探索，零编译、零下载）

> 环境：`git clone --depth 50 https://github.com/ggml-org/llama.cpp && cd llama.cpp`（浅克隆即可；不必与本章同 commit，**以你本地版本为准并记录它**）。

**任务 A：画出你本地版本的入口链（15 分钟）**

1. `grep -rn "int main(" tools/` 找到入口；
2. 读入口文件，写下它调用的 3–5 个关键函数及所在文件；
3. 找到参数解析函数，确认某个你用过的参数（如 `--temp`）定义在哪个文件。

*验收标准*：能填出下表并保证每格有 `文件:行号`——

| 步骤 | 函数 | 文件:行号 |
|---|---|---|
| 真正的 main | | |
| 参数解析入口 | | |
| 后端初始化 | | |
| 交互/服务主体 | | |

**任务 B：复现 GGUF 解析顺序（20 分钟）**

1. `grep -n "GGUF_MAGIC" ggml/src/gguf.cpp`，读解析函数前 80 行；
2. 按代码顺序列出它读文件的步骤（魔数/版本/端序/计数/KV/张量表），每步抄一句它的错误信息；
3. 用任意 `.gguf` 文件（可用第 10 章量化产物）对照：`xxd -l 64 model.gguf`，亲眼看到魔数 `47 47 55 46` 和版本号。

*验收标准*：步骤表 ≥6 行且错误信息为原文；能解释小端机器读到的版本号 3 与"0x30000000 报错"的联系。

**任务 C：给采样链"接一个假想采样器"（25 分钟）**

1. 读 `common/sampling.cpp` 的 `common_sampler_init`，抄下你默认参数下链的真实顺序（哪个在前、哪个是链尾）；
2. `grep -n "llama_sampler_init_top_k" src/llama-sampler.cpp`，打开工厂函数，确认参数非法（k≤0）时它返回什么（提示：`llama_sampler_init_empty`）；
3. 推演：把 `--top-k 0` 和 `--temp 0` 同时设置，链尾会是什么？（写推理过程，不许运行）

*验收标准*：写出链顺序表（含链尾的 greedy/dist 判定依据行号）；解释"空采样器"这种防御式设计的意图。

**任务 D（加分）：数一数 split**

`grep -n "ggml_backend_sched_get_n_splits" -r src/` 找到被注释的调试输出（llama-context.cpp:2731 附近 `// fprintf(stderr, "splits: ...")`），说明作者为什么把它注释掉、在什么场景下你会取消注释。

*验收标准*：能说出"splits 数 = 跨后端边界的子图数，调试多卡/offload 时打开"。

### 9.3 自检三问（闭卷作答）

1. 一个 GGUF 从 `open()` 到第一个 token，主链路上依次经过哪 5 个关键函数？分别在哪一层（应用/框架/张量/硬件）？
2. `ggml_tensor` 用哪两个字段描述"我是什么运算、我依赖谁"？这种设计与"图是独立对象"的框架（TF GraphDef）相比，建图时多了什么便利？
3. 同一张图能跑 CPU 和 CUDA，靠的是哪个结构体的哪组函数指针？跨后端边界时谁负责插搬运节点？

### 9.4 面试话术（可直接背）

**话术 1：被问"介绍一下 llama.cpp 的架构"**

> "它分四层：`tools/common` 是应用层，负责参数和交互——顺带一提新版 CLI 其实是本地起了个 server 自己当 client；`src/` 是推理框架层，核心是 llama-context 的主循环和 llama-model-loader 的 GGUF 加载；`ggml/src` 是张量层，抽象出 ggml_tensor、计算图和后端接口三件套；最底下是 20 多个后端目录。关键是层间契约：数据用 GGUF 自描述头和代码解耦，执行用 ggml_backend_i 函数表和硬件解耦，所以加一个新模型只用加一个 `src/models/` 文件，加一种新硬件只用实现一张函数表。"

**话术 2：被问"GGUF 加载为什么快 / 怎么保证不加载错模型"**

> "解析是逐字段防御式的：魔数、版本、端序、张量计数各有检查，错文件当场拒载；数据区走 mmap，字节按缺页拉入，不做整体 fread。正确性靠 create_tensor 的名称加形状双重校验——转换脚本把张量名写错或形状不符会立刻报 missing/shape mismatch，而不是默默算出垃圾。这也是我第 10 章量化排错时的第一定位点。"

**话术 3：被问"推理引擎里计算图和后端调度怎么配合"（进阶）**

> "llama.cpp 先用一张样板图做 graph_reserve，让调度器预分配全部工作内存，正式执行只灌数据；调度器 split_graph 时按张量所在 buffer 给每个节点定后端归属，切成子图、在边界插异步拷贝，再逐段调用各后端的 graph_compute 函数指针。所以多 GPU 或 CPU offload 不是特殊模式，只是 split 数量变多——我 grep 过它的 splits 调试输出，这个视角看 `-ngl 0` 和 `-ngl 99` 的区别就是 split 从 1 变成几十。"

---

## 参考与版本说明

- 源码事实：llama.cpp commit `e707790c`（2026-10-15 克隆，浅克隆于 2026-10-16 走读），行号以该 commit 为准；仓库自带 `docs/`（如 docs/build.md、docs/backend/）为辅助材料。
- 本章遵循第 9 章起的写作风格：**不跑性能实验，只做零成本结构性观察**（grep 行数、读结构体、解析文件头），所有行号与结构体字段均来自对该 commit 源码的直接读取，未引用任何未经核实的数字。
- 上一章（第 10 章）的量化实测与本章 `ggml-quants.c`/`GGML_OP_MUL_MAT` 形成闭环：格式在文件里（第 9 章）、选择在三本账里（第 10 章）、执行在算子里（本章）。
