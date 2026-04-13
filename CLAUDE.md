# Mini-SGLang: 算法工程师入门指南

> 本文档面向大模型算法工程师，帮助你快速理解 SGLang 推理引擎的核心设计。Mini-SGLang 是 SGLang 的教学简化版，保留了所有关键加速机制，去除了工程噪音。

---

## 一、项目结构总览

```
python/minisgl/
├── server/           # HTTP 服务层 (FastAPI + ZMQ 进程通信)
│   ├── launch.py         # 启动入口：拉起 tokenizer/scheduler/detokenizer 进程
│   ├── api_server.py     # FastAPI 路由 (/v1/chat/completions, /generate)
│   └── args.py           # CLI 参数定义
├── scheduler/        # 调度核心 (请求编排、批次组装)
│   ├── scheduler.py      # 主循环 (normal_loop / overlap_loop)
│   ├── prefill.py        # Prefill 调度器 (chunked prefill)
│   ├── decode.py         # Decode 调度器
│   ├── cache.py          # 物理页分配 + 驱逐策略
│   ├── table.py          # page_table / token_pool 槽位管理
│   └── io.py             # ZMQ 消息收发 + TP 广播
├── engine/           # 推理引擎 (模型执行层)
│   ├── engine.py         # forward_batch：模型前向 + 采样
│   ├── config.py         # EngineConfig
│   ├── graph.py          # CUDA Graph 捕获与回放
│   └── sample.py         # 采样策略 (greedy / top-k / top-p)
├── models/           # 模型实现
│   ├── llama.py          # LlamaForCausalLM
│   ├── qwen2.py          # Qwen2ForCausalLM
│   ├── qwen3.py          # Qwen3ForCausalLM
│   ├── qwen3_moe.py      # Qwen3MoeForCausalLM (MoE)
│   ├── utils.py          # GatedMLP, MoEMLP, RopeAttn
│   └── weight.py         # HuggingFace 权重加载 + TP 切分
├── layers/           # 通用算子层
│   ├── attention.py      # AttentionLayer (QKV split → RoPE → backend dispatch)
│   ├── linear.py         # TP 线性层 (ColumnParallel / RowParallel / QKV / O-proj)
│   ├── embedding.py      # VocabParallelEmbedding, ParallelLMHead
│   ├── norm.py           # RMSNorm / RMSNormFused (fused add + norm)
│   └── rotary.py         # RotaryEmbedding (含 Llama3 scaling)
├── attention/        # Attention 后端
│   ├── fi.py             # FlashInfer 后端 (FA2, paged KV)
│   ├── fa.py             # FlashAttention3 后端 (Hopper)
│   ├── base.py           # 基类 + HybridBackend (prefill/decode 用不同后端)
│   └── __init__.py       # 自动后端选择 (SM90/SM100)
├── kvcache/          # KV Cache 管理
│   ├── mha_pool.py       # MHAKVCache：物理 KV 显存池
│   ├── radix_manager.py  # RadixCacheManager：基数树前缀缓存
│   └── naive_manager.py  # NaiveCacheManager：无前缀复用基线
├── moe/              # Mixture of Experts
│   └── fused.py          # FusedMoe (topk_softmax + Triton GEMM)
├── kernel/           # 自定义算子
│   ├── triton/fused_moe.py   # Triton MoE kernel
│   ├── csrc/jit/store.cu     # KV cache scatter-write (CUDA, JIT)
│   ├── csrc/jit/index.cu     # Embedding gather (CUDA, JIT)
│   ├── csrc/src/radix.cpp    # fast_compare_key (C++, 基数树加速)
│   └── csrc/src/pynccl.cu    # PyNCCL 通信 (绕过 torch.distributed)
├── distributed/      # 分布式通信
│   └── impl.py           # DistributedCommunicator (PyNCCL / TorchDistributed)
├── tokenizer/        # Tokenizer 进程
│   └── server.py         # tokenize_worker / detokenize_worker
├── llm/              # 离线推理 API
│   └── llm.py            # LLM 类 (继承 Scheduler，用于 benchmark)
├── core.py           # 核心数据结构 (Req, Batch, Context, SamplingParams)
├── env.py            # 环境变量配置
└── __main__.py       # python -m minisgl 入口
```

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | Python 打包 | `pyproject.toml` + `setuptools`，`pip install -e .` 的含义 | `setuptools` |
> | 异步 Web 框架 | FastAPI 的路由、中间件、SSE 流式响应 | `fastapi`, `uvicorn`, `sse-starlette` |
> | 进程间通信 | ZeroMQ 的 PUSH/PULL、PUB/SUB 模式；`msgpack` 序列化 | `pyzmq`, `msgpack` |
> | GPU 编程基础 | CUDA stream、kernel launch、显存分配的基本概念 | `torch.cuda`, CUDA Toolkit |
> | Tokenizer | HuggingFace `transformers.AutoTokenizer` 的用法、chat_template | `transformers`, `tokenizers` |
>
> **自行扩展方向**: 如果你想给项目加 gRPC 接口或 WebSocket 推送，主要改 `server/api_server.py`；如果想换序列化协议（如 protobuf），改 `server/` 和 `scheduler/io.py` 中的 msgpack 部分。

---

## 二、一个请求的完整生命周期

```
用户 HTTP 请求
  │
  ▼
┌──────────────────┐   ZMQ    ┌──────────────────┐   ZMQ    ┌──────────────────┐
│   API Server     │ ───────► │   Tokenizer      │ ───────► │   Scheduler      │
│ (api_server.py)  │          │ (tokenizer/)      │          │ (scheduler/)     │
│ FastAPI + SSE    │          │ HF tokenize       │          │ 主循环            │
└──────────────────┘          └──────────────────┘          └──────┬───────────┘
                                                                   │
                                                    ┌──────────────┴──────────────┐
                                                    │  Schedule: prefill → decode  │
                                                    │  Prepare: page alloc, meta  │
                                                    │  Forward: model + sample    │
                                                    └──────────────┬──────────────┘
                                                                   │
  ┌──────────────────┐   ZMQ    ┌──────────────────┐   ZMQ        │
  │   API Server     │ ◄─────── │   Detokenizer    │ ◄────────────┘
  │ 流式返回给用户     │          │ HF detokenize    │
  └──────────────────┘          └──────────────────┘
```

**关键步骤详解：**

1. **Tokenize**: API Server 收到请求，经 ZMQ 发给 Tokenizer 进程，得到 `input_ids`
2. **加入等待队列**: Scheduler 收到 `UserMsg`，加入 `PrefillManager.pending_list`
3. **Prefix Match**: `RadixCacheManager.match_prefix()` 在基数树中查找最长公共前缀，跳过已缓存的 KV 计算
4. **Prefill 调度**: 在 token 预算内（默认 8192）组装 prefill 批次，超长请求自动分块 (Chunked Prefill)
5. **页分配**: `CacheManager.allocate()` 从空闲页池分配物理页，写入 `page_table`
6. **模型前向**: `engine.forward_batch()` → 模型前向 → 采样 next_token
7. **异步回写**: next_token 异步 GPU→CPU 拷贝，同时写入 `token_pool`
8. **Decode 循环**: 请求进入 `DecodeManager.running_reqs`，每轮生成一个 token，直到 EOS 或 max_tokens
9. **完成回收**: 释放 page_table 槽位，将 KV 页插入基数树供后续请求复用

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | 自回归生成 | Prefill (prompt 一次算完) vs Decode (逐 token 生成) 的两阶段本质 | — |
> | CUDA 异步语义 | `non_blocking=True` 的 GPU↔CPU 拷贝、`torch.cuda.Event` 做同步 | `torch.cuda` |
> | ZMQ 消息模式 | 理解 5 条 IPC socket 的拓扑 (PUSH/PULL + PUB/SUB)，见 `server/launch.py` | `pyzmq` |
> | 多进程模型 | `multiprocessing.Process`、进程间无共享 GPU 显存 | Python `multiprocessing` |
>
> **自行扩展方向**: 如果你想加请求优先级队列、基于长度的调度策略、或请求超时取消机制，核心修改点在步骤 2-4，即 `scheduler/prefill.py` 的 `pending_list` 和 `PrefillAdder` 逻辑。

---

## 三、核心加速机制

### 3.1 PagedAttention (页式 KV Cache)

**解决什么问题**: 传统实现为每个请求预分配最大长度的连续 KV 显存，浪费严重。

**怎么做的**: 类似操作系统虚拟内存。每个 token 的 KV 占一个物理"页"(page_size=1)，通过 `page_table[req_idx, pos]` 间接映射。

- 物理池: `MHAKVCache` (`kvcache/mha_pool.py`) — 预分配 `(2, num_layers, num_pages, 1, kv_heads, head_dim)` 的连续 buffer
- 页表: `page_table[req_idx, pos] = physical_page_idx`
- 分配/释放: `CacheManager` (`scheduler/cache.py`) 管理空闲页栈 `_free_slots`
- 写入: 自定义 CUDA kernel `store_cache` (`kernel/csrc/jit/store.cu`) scatter-write KV 到物理页
- 读取: FlashInfer/FA3 原生支持 paged KV cache

**算法工程师关注点**: 如果你修改了模型的 attention 结构（如 GQA→MQA），需要关注 `kvcache/mha_pool.py` 中的 shape 计算和 `layers/attention.py` 中的 head 切分逻辑。

### 3.2 RadixAttention (基数树前缀缓存)

**解决什么问题**: 多轮对话、共享 system prompt 时，相同前缀的 KV 被重复计算。

**怎么做的**: 用压缩字典树 (Radix Tree) 索引已计算的 KV 前缀：

```
文件: kvcache/radix_manager.py

              root
             /    \
      [system prompt]  [另一个前缀]
           /     \
    [用户问题A]  [用户问题B]  ← 共享 system prompt 的 KV
```

- `match_prefix(input_ids)`: 沿树查找最长匹配，返回已缓存长度 → 跳过这些 token 的 prefill
- `insert_prefix(input_ids, indices)`: 请求完成后，将其 KV 页索引插入树中
- `evict(size)`: 内存不足时，LRU 驱逐叶子节点（按 timestamp 排序）
- 引用计数: `ref_count` 防止正在使用的节点被驱逐

**算法工程师关注点**: 如果你在做 prompt 工程优化（如固定 system prompt），RadixAttention 会自动帮你缓存公共前缀。理解这个机制有助于设计更高效的 prompt 结构。

### 3.3 Continuous Batching + Chunked Prefill

**解决什么问题**: 
- Static batching: 必须等一批请求全部完成才能处理下一批，GPU 利用率低
- 长 prefill: 一个超长请求会阻塞所有 decode 请求

**怎么做的**:
- **Continuous Batching**: `Scheduler` 主循环每轮重新组装批次 — 完成的请求立即退出，新请求随时加入
- **Chunked Prefill** (`scheduler/prefill.py`): 每轮 prefill 的 token 总数有预算 (`max_extend_tokens=8192`)。超长请求被拆分为 `ChunkedReq`，分多轮 prefill，中间可以穿插 decode 批次

**调度优先级** (`scheduler.py:_schedule_next_batch`): prefill 优先于 decode — 新请求尽快进入 decode 阶段。

### 3.4 Overlap Scheduling (CPU-GPU 重叠调度)

**解决什么问题**: 每轮迭代中，CPU 端的调度/元数据准备 与 GPU 端的模型计算串行执行，CPU 成为瓶颈。

**怎么做的** (`scheduler.py:overlap_loop`):

```
时间 ──────────────────────────────────►

GPU:  ┃ forward(batch_N) ┃         ┃ forward(batch_N+1) ┃
CPU:  ┃                  ┃ process_last(N) + schedule(N+1) ┃ process_last(N+1) ...
                         ▲ 重叠区域：GPU 在跑 N+1 的同时，CPU 在处理 N 的结果
```

- 使用两个 CUDA stream: `self.stream`（CPU 调度）和 `self.engine.stream`（GPU 计算）
- `engine.stream.wait_stream(self.stream)` 确保 page_table 写入完成后 GPU 才读取
- `copy_done_event` 用于 CPU 端同步 GPU→CPU 的 token 拷贝

### 3.5 CUDA Graph

**解决什么问题**: Decode 阶段每个 token 的计算量很小，但 kernel launch 开销占比很高。

**怎么做的** (`engine/graph.py`):
- 启动时对常见 decode batch size (1,2,4,8,...,max_bs) 预录制 CUDA Graph
- 运行时 `replay()`: 只需更新输入数据，然后重放整个计算图，避免反复 launch kernel
- 所有 graph 共享一个 CUDA memory pool，减少显存开销
- 仅用于 decode 阶段（prefill 的 seq_len 变化太大，不适合 graph）

### 3.6 FlashInfer / FlashAttention3 后端

**自动选择逻辑** (`attention/__init__.py:resolve_auto_backend`):

| GPU 架构 | Prefill 后端 | Decode 后端 |
|----------|-------------|-------------|
| SM100 (Blackwell) | FlashInfer (FA2) | FlashInfer (FA2) |
| SM90 (Hopper H100/H200) | FlashAttention3 | FlashInfer (FA2) |
| Pre-Hopper (A100 等) | FlashInfer (FA2) | FlashInfer (FA2) |

- `HybridBackend` (`attention/base.py`): 将 prefill 和 decode 委托给不同后端
- FlashInfer decode 在 GQA ratio >= 4 时自动启用 tensor core

### 3.7 Fused 算子

| 算子 | 文件 | 作用 |
|------|------|------|
| `fused_add_rmsnorm` | `layers/norm.py` | 残差加法 + RMSNorm 融合为一个 kernel |
| `apply_rope_inplace` | `layers/attention.py` | RoPE 原地计算，无额外显存分配 |
| `store_cache` | `kernel/csrc/jit/store.cu` | KV scatter-write，JIT 编译 |
| `indexing` | `kernel/csrc/jit/index.cu` | Embedding gather with TP masking |
| `fused_moe_kernel` | `kernel/triton/fused_moe.py` | Triton MoE batched GEMM |

### 3.8 Tensor Parallelism (TP)

**通信层** (`distributed/impl.py`):
- 默认使用 PyNCCL (`kernel/csrc/src/pynccl.cu`) 而非 `torch.distributed`，延迟更低
- 权重加载时自动按 TP 切分 (`models/weight.py`)
- 线性层: `ColumnParallelLinear` (输出维度切分) / `RowParallelLinear` (输入维度切分)
- Embedding/LMHead: vocab 按 TP 切分，前向后 all_reduce / all_gather

> **前置知识 & 依赖 (覆盖 3.1 ~ 3.8)**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | OS 虚拟内存 | 页表 (page table)、按需分配、碎片整理的基本概念 — PagedAttention 直接类比 | — |
> | 数据结构 | 压缩字典树 (Radix/Patricia Trie)、LRU 驱逐、引用计数 | — |
> | CUDA Stream & Event | 多 stream 并发、`stream.wait_stream()`、`Event.record/synchronize` | `torch.cuda` |
> | CUDA Graph | `torch.cuda.CUDAGraph` 的 capture/replay 原理、memory pool 共享 | `torch.cuda`, CUDA 12+ |
> | FlashAttention 原理 | tiling + online softmax 避免 O(N^2) 显存；FA2 vs FA3 (Hopper warp-specialization) | `flashinfer`, `sgl_kernel` |
> | Kernel Fusion | 为什么 fused_add_rmsnorm、inplace RoPE 能减少 HBM 访问 | — |
> | 并行策略 | Tensor Parallelism 中 Column/Row 切分 + all_reduce/all_gather 的通信开销分析 | `nccl`, `torch.distributed` |
> | NCCL 通信 | NCCL allreduce/allgather 的调用方式；PyNCCL 绕过 torch.distributed 降低延迟的原因 | `pynccl.cu`, NCCL 2.x |
>
> **自行扩展方向**:
> - **新的缓存驱逐策略** (如 LFU、ARC): 改 `kvcache/radix_manager.py` 的 `evict()` 方法
> - **Speculative Decoding**: 需要在 `engine/engine.py` 的 `forward_batch` 中加 draft model 前向 + verify 逻辑，同时修改 `scheduler/decode.py` 支持一次验证多个 token
> - **Pipeline Parallelism**: 需要拆分模型层到不同 GPU，改 `engine/engine.py` 的 forward 路径和 `distributed/impl.py` 的通信原语
> - **自定义 Attention 变体** (如 Sliding Window、Cross Attention): 在 `attention/` 下新建后端，继承 `BaseAttnBackend`

---

## 四、MoE (Mixture of Experts) 支持

MoE 模型（如 Qwen3-MoE）的核心路径:

```
hidden → gate (路由器) → topk_softmax → fused_moe_kernel (Triton) → reduce
```

- `MoEMLP` (`models/utils.py`): 路由器 + MoELayer 的组合
- `MoELayer` (`layers/moe.py`): 持有 `gate_up_proj` 和 `down_proj` 权重，调用 `moe_backend.forward()`
- `FusedMoe` (`moe/fused.py`): 
  1. `fused_topk()` — 用 `sgl_kernel.topk_softmax` 选出 top-k 专家
  2. `moe_align_block_size()` — 将 token 按专家分组并对齐到 block 边界
  3. 两次 `fused_moe_kernel_triton` — gate_up_proj → activation → down_proj
  4. `moe_sum_reduce_triton()` — 合并 top-k 专家的输出

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | MoE 基础 | 稀疏激活原理：每个 token 只激活 top-k 个专家，而非全部 | — |
> | 路由机制 | Softmax 路由 → top-k 选择 → load balancing loss 的基本概念 | — |
> | Triton 编程 | `@triton.jit` 装饰器、block 级 tiling、shared memory、L2 cache 分组优化 | `triton` (OpenAI Triton) |
> | GPU GEMM 优化 | 分块矩阵乘 (tiled GEMM)、block size 对 occupancy 和 cache 命中率的影响 | — |
> | Activation 函数 | SiLU (Swish)、GeGLU 等 gated activation 的 fused 实现 | `sgl_kernel` |
>
> **自行扩展方向**:
> - **Expert Parallelism (EP)**: 当前 MoE 权重在每张卡上完整存放，如果专家数很多可以改为跨卡切分，需改 `layers/moe.py` 的权重布局和 `distributed/impl.py` 加 all-to-all 通信
> - **新的路由策略** (如 Expert Choice、Hash Routing): 替换 `moe/fused.py` 中 `fused_topk()` 的 topk_softmax 逻辑
> - **Triton kernel 调优**: `kernel/triton/fused_moe.py` 中的 `BLOCK_SIZE_M/N/K` 和 `GROUP_SIZE_M` 可以通过 auto-tuning 搜索最优配置

---

## 五、核心数据结构速查

```python
# core.py

class Req:
    """一个推理请求的完整状态"""
    input_ids: Tensor       # CPU, shape=(seq_len,), 完整 token 序列
    table_idx: int          # page_table / token_pool 中的槽位
    cached_len: int         # 已在 KV cache 中的长度 (来自 radix 匹配 + 已 decode)
    device_len: int         # 已上传到 GPU 的长度
    # extend_len = device_len - cached_len  → 本轮需要计算的 token 数
    # remain_len = max_device_len - device_len  → 还需生成的 token 数

class Batch:
    """一个前向批次"""
    reqs: List[Req]
    phase: str              # "prefill" 或 "decode"
    input_ids: Tensor       # GPU, 本轮输入 token (gather 自 token_pool)
    positions: Tensor       # GPU, 位置编码用的 position ids
    out_loc: Tensor         # GPU, KV 写入的物理页索引
    attn_metadata: Any      # FlashInfer/FA3 的注意力元数据

class Context:
    """全局单例，模型层通过 get_global_ctx() 访问当前批次信息"""
    page_size: int
    attn_backend: BaseAttnBackend
    moe_backend: BaseMoE
    batch: Batch            # 当前正在前向的批次 (forward_batch 上下文管理器设置)
```

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | Python dataclass | `dataclass`、`__slots__`、property 的用法 | Python stdlib |
> | 上下文管理器 | `contextmanager` 装饰器、`with` 语句的 enter/exit 语义 — `Context.forward_batch()` 用此模式 | Python stdlib |
> | Tensor 索引 | PyTorch 高级索引 (fancy indexing)、scatter/gather 操作 — `page_table[req_idx, pos]` 的核心 | `torch` |
>
> **自行扩展方向**: 如果你要加新的请求级状态 (如 beam search 的 beam_width、repetition penalty 的历史 token)，在 `Req` 中加字段，然后在 `BatchSamplingArgs.prepare()` 中收集到 GPU tensor 即可。

---

## 六、常用修改位置指南

### 添加新模型

1. 在 `models/` 下新建文件，参考 `llama.py` 的结构
2. 实现 `XXXForCausalLM`，继承 `BaseOP`
3. 在 `models/__init__.py` 的 `MODEL_MAPPING` 中注册
4. 在 `models/weight.py` 的 `_get_weight_rules()` 中添加权重映射规则
5. 如有特殊 attention（如 sliding window），可能需修改 `attention/` 后端

### 修改采样策略

- `engine/sample.py`: `Sampler.sample()` — 添加新的采样方法
- `core.py`: `SamplingParams` — 添加新参数字段
- `scheduler/scheduler.py`: `BatchSamplingArgs.prepare()` — 将新参数收集到 GPU tensor

### 修改调度策略

- `scheduler/prefill.py`: `PrefillManager.schedule_next_batch()` — 修改 prefill 组批逻辑
- `scheduler/decode.py`: `DecodeManager.schedule_next_batch()` — 修改 decode 组批逻辑
- `scheduler/scheduler.py`: `_schedule_next_batch()` — 修改 prefill/decode 优先级

### 修改 KV Cache / 缓存策略

- `kvcache/radix_manager.py`: 修改前缀匹配/驱逐策略
- `kvcache/mha_pool.py`: 修改 KV 存储布局
- `scheduler/cache.py`: 修改页分配/回收逻辑

### 添加新的 Attention 后端

1. 在 `attention/` 下新建文件，继承 `BaseAttnBackend`
2. 实现 `prepare_metadata()` 和 `forward()` 方法
3. 在 `attention/__init__.py` 中注册

### 添加自定义 CUDA Kernel

- JIT 编译的 CUDA: 放在 `kernel/csrc/jit/`，通过 tvm-ffi 注册
- Triton kernel: 放在 `kernel/triton/`
- 编译时 C++/CUDA: 放在 `kernel/csrc/src/`

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | HuggingFace 模型结构 | `config.json` 中各字段含义 (num_layers, hidden_size, num_attention_heads 等) | `transformers` |
> | 权重映射 | HF checkpoint 的 state_dict key 命名规则，以及如何映射到自定义模型 | `safetensors` |
> | TP 切分规则 | 哪些权重按行切、哪些按列切 — 参考 Megatron-LM 的 Column/Row 并行论文 | — |
> | CUDA C++ 编程 | 写自定义 kernel 需要：grid/block 配置、shared memory、warp 操作 | CUDA Toolkit, `nvcc` |
> | Triton 编程 | 写 fused kernel 需要：`@triton.jit`、block-level tiling、auto-tuning | `triton` |
> | tvm-ffi | JIT kernel 注册机制 — `kernel/csrc/jit/` 下的 .cu 文件通过 tvm-ffi 暴露给 Python | `tvm` |
>
> **自行扩展方向**:
> - **添加新模型架构** (如 Mixtral、DeepSeek-V2): 参考 `models/qwen3_moe.py`，重点是 attention 变体 (GQA/MQA/MLA) 和 MLP 结构
> - **自定义采样** (如 min_p, repetition penalty, beam search): 改 `engine/sample.py` + `core.py:SamplingParams`
> - **新的 KV Cache 布局** (如 multi-query 共享 KV): 改 `kvcache/mha_pool.py` 的 shape 和 `layers/attention.py` 的 head 切分

---

## 七、快速启动

```bash
# 安装
pip install -e .

# 启动服务 (单卡)
python -m minisgl --model-path Qwen/Qwen2.5-7B-Instruct --port 1919

# 启动服务 (2卡 TP)
python -m minisgl --model-path Qwen/Qwen2.5-72B-Instruct --tp 2

# 交互式 shell
python -m minisgl --model-path Qwen/Qwen2.5-7B-Instruct shell

# 离线 benchmark
python benchmark/offline/bench.py --model-path Qwen/Qwen2.5-7B-Instruct

# 关键 CLI 参数
--attn auto|fi|fa|fa,fi    # Attention 后端
--cache radix|naive        # 缓存策略
--max-prefill-length 8192  # Chunked prefill 预算
--cuda-graph-max-bs 128    # CUDA Graph 最大 batch size
--memory-ratio 0.9         # KV cache 占 GPU 显存比例
```

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | Python 可编辑安装 | `pip install -e .` 的工作原理、`pyproject.toml` 中 `[project.scripts]` 定义 CLI 入口 | `pip`, `setuptools` |
> | GPU 环境 | `nvidia-smi` 查看 GPU 型号/显存、`CUDA_VISIBLE_DEVICES` 控制可见卡号 | CUDA Toolkit |
> | HTTP 调试 | `curl` 发送 POST 请求、`jq` 解析 JSON 响应、SSE 流式输出的观察 | `curl`, `jq` |
> | 性能分析 | `torch.profiler` 或 `nsys profile` 分析 kernel 耗时和 GPU 利用率 | `nsight-systems`, `torch.profiler` |
>
> **自行扩展方向**: 如果你想做 A/B 性能对比 (如 radix vs naive、overlap vs normal)，可以用 `benchmark/offline/bench.py` 配合不同 `--cache` / 环境变量跑离线 benchmark，对比 throughput 和 latency。

---

## 八、环境变量

| 变量名 | 默认 | 说明 |
|--------|------|------|
| `MINISGL_DISABLE_OVERLAP_SCHEDULING` | `false` | 禁用 CPU-GPU 重叠调度 (用于 ablation) |
| `MINISGL_OVERLAP_EXTRA_SYNC` | `false` | 额外 stream 同步 (调试用) |
| `MINISGL_FLASHINFER_USE_TENSOR_CORES` | auto | 强制 FlashInfer decode 使用 tensor core |
| `MINISGL_PYNCCL_MAX_BUFFER_SIZE` | `1GB` | PyNCCL 通信 buffer 大小 |

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | 环境变量机制 | `os.environ` 读取、`bool`/`int`/内存大小的解析方式 | Python stdlib |
> | Ablation 实验 | 控制变量法：禁用单个优化 (如 overlap scheduling) 对比性能差异 | — |
>
> **自行扩展方向**: 如果你要加新的环境变量开关 (如禁用 CUDA Graph、强制使用某个 attention 后端)，在 `env.py` 的 `ENV` 类中添加属性，然后在对应模块中读取即可。

---

## 九、架构设计哲学（对比理解）

如果你熟悉 vLLM 或 HuggingFace Transformers，以下对比有助于理解 SGLang 的设计选择：

| 维度 | HF Transformers | vLLM | SGLang (mini-sglang) |
|------|----------------|------|---------------------|
| 目标 | 易用性 | 高吞吐 | 高吞吐 + 低延迟 |
| Batching | Static | Continuous | Continuous + Chunked Prefill |
| KV Cache | 预分配连续 | PagedAttention | PagedAttention + RadixAttention |
| 前缀复用 | 无 | 有限 | 基数树自动前缀缓存 |
| CPU-GPU 重叠 | 无 | 有限 | Overlap Scheduling |
| CUDA Graph | 无 | 支持 | 支持 (decode only) |
| 进程模型 | 单进程 | 单进程 + Ray | 多进程 + ZMQ |

**SGLang 的核心洞察**: 推理引擎的瓶颈不只在 GPU 计算，还在 CPU 调度开销和显存碎片。RadixAttention 解决前缀复用，Overlap Scheduling 隐藏 CPU 开销，PagedAttention 消除显存碎片。

> **前置知识 & 依赖**
>
> | 领域 | 需要了解 | 对应依赖 / 工具 |
> |------|---------|----------------|
> | vLLM 架构 | PagedAttention 原始论文 (Kwon et al. 2023)、vLLM 的 Scheduler + Worker 设计 | — |
> | HF Transformers | `model.generate()` 的 static batching 行为、`GenerationConfig` | `transformers` |
> | Ray 分布式 | vLLM 用 Ray 做 TP 编排 vs SGLang 用 ZMQ + multiprocessing 的取舍 | — |
> | 推理优化论文 | Orca (continuous batching)、SGLang (RadixAttention)、NanoFlow (overlap scheduling) | — |
>
> **推荐阅读顺序** (面向算法工程师):
> 1. 先读 `core.py` 理解 Req/Batch/Context 三个核心抽象
> 2. 再读 `scheduler/scheduler.py` 的 `normal_loop` (非 overlap 版本更易理解)
> 3. 然后读 `engine/engine.py:forward_batch` 理解单次前向的完整流程
> 4. 最后根据兴趣深入某个加速机制 (radix → `kvcache/radix_manager.py`，CUDA graph → `engine/graph.py` 等)
