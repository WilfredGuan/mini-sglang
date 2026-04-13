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

---

## 八、环境变量

| 变量名 | 默认 | 说明 |
|--------|------|------|
| `MINISGL_DISABLE_OVERLAP_SCHEDULING` | `false` | 禁用 CPU-GPU 重叠调度 (用于 ablation) |
| `MINISGL_OVERLAP_EXTRA_SYNC` | `false` | 额外 stream 同步 (调试用) |
| `MINISGL_FLASHINFER_USE_TENSOR_CORES` | auto | 强制 FlashInfer decode 使用 tensor core |
| `MINISGL_PYNCCL_MAX_BUFFER_SIZE` | `1GB` | PyNCCL 通信 buffer 大小 |

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
