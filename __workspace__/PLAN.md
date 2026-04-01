# vLLM Throughput Optimization Plan

**Objective**: Maximize tokens/second throughput on a vLLM serving deployment.

---

## How the 5 Core Techniques Work

### 1. Paged Attention

KV cache is divided into fixed-size blocks (e.g., 16 tokens each). Requests hold a *block table* — a list of block IDs — rather than contiguous memory. Attention kernels use block tables to gather K,V from scattered physical memory.

```text
Request → [block_id_1, block_id_2, ...] → CUDA kernel gathers K,V → fused attention
```

**Throughput impact**: Eliminates memory fragmentation. More requests fit in GPU memory simultaneously → higher batch sizes → better GPU utilization.

### 2. Optimized KV Caching (Prefix Caching)

Blocks are content-addressed via hash. When a new request shares a prefix (system prompt, few-shot examples) with a cached request, vLLM reuses those blocks — skipping their recomputation entirely.

```text
Hash(tokens[0:block_size]) → lookup → cache hit → skip prefill for that block
```

**Throughput impact**: Repeated system prompts / shared prefixes become free. Reduces effective prefill compute for common workloads.

### 3. Optimized CUDA Kernels

Three kernel families:

- **Prefill kernel** (Triton unified): dense Q @ K^T @ V over all prompt tokens
- **Decode kernel** (paged attention v1/v2): indirect-indexed attention over cached K,V blocks
- **Cache write kernel** (reshape_and_cache): writes new K,V into allocated slots with optional FP8 quantization

All kernels are fused (no separate softmax pass), and the decode kernel uses multi-partition reduction for long contexts.

**Throughput impact**: Minimizes kernel launch overhead and memory bandwidth. FP8 cache halves KV memory footprint.

### 4. Speculative Decoding

A fast *draft model* (EAGLE, Medusa, NGram, or a small LLM) proposes K tokens. The target model verifies all K+1 positions in a single forward pass. Accepted tokens are kept; the first rejection reverts to the target's sample.

```text
Draft → [t1, t2, t3, t4]
Target → verify all 4 in one pass → accept [t1, t2], reject t3 → output 3 tokens
```

**Throughput impact**: Each target forward pass produces 1–K tokens instead of 1. Effective throughput multiplier is the *acceptance rate* × *num_speculative_tokens*.

### 5. Chunked Prefill

Long prompts are split into chunks (`max_num_batched_tokens`). Chunks from multiple prefill requests — and decode steps — are batched together per iteration.

```text
Iteration: [chunk_A(1024), chunk_B(512), decode_C(1), decode_D(1)] → single forward pass
```

**Throughput impact**: Decode requests are not stalled behind a long prefill. GPU is kept busy across both prefill and decode work simultaneously.

---

## Optimization Schema

The following is the layered optimization strategy, ordered by impact and independence.

```text
Layer 0: Hardware baseline
  └── FP8/BF16 compute dtype, tensor parallelism (--tensor-parallel-size)

Layer 1: Memory maximization
  └── --gpu-memory-utilization 0.95
  └── --kv-cache-dtype fp8  (halves KV memory → doubles live batch size)
  └── --max-model-len (set to actual workload max, not model default)

Layer 2: Batching
  └── --max-num-batched-tokens  (tuned to GPU SRAM / compute balance)
  └── --max-num-seqs            (max concurrent requests)
  └── enable_chunked_prefill=True

Layer 3: Prefix cache
  └── --enable-prefix-caching   (automatic KV reuse for shared prefixes)

Layer 4: Speculative decoding (output-length dependent)
  └── --speculative-model / --num-speculative-tokens
  └── Effective only when acceptance rate > 0.7 and output is long
```

Each layer is independent — they stack multiplicatively when all conditions are met.

---

## Key Configuration Tradeoffs

| Knob | Higher value | Lower value |
| ---- | ----------- | ----------- |
| `gpu_memory_utilization` | More KV blocks, larger batches | Risk of OOM |
| `max_num_batched_tokens` | Higher arithmetic intensity | More latency per step |
| `num_speculative_tokens` | More tokens/step when accepted | More wasted compute on rejection |
| `kv_cache_dtype=fp8` | 2× KV capacity | Small accuracy loss |
| `max_num_seqs` | More concurrency | Memory pressure |

---

## Dataset

`main.py` uses **LongBench-E** (`longbench_lcc_e`) via `lm_eval`, which provides real long-context code completion documents. Prompts exceeding `MAX_MODEL_LEN - OUTPUT_LEN` tokens are filtered at load time.

For production-representative workloads, replace with `ShareGPTDataset` or `BurstGPTDataset` from `vllm.benchmarks.datasets`.

---

## Measurement

### Timing

Use `torch.cuda.Event` (GPU hardware counters) instead of `time.perf_counter()` (CPU wall clock). CUDA ops are async — the host returns from a kernel launch before the GPU finishes, making wall clock unreliable.

### Profiling

Set `PROFILER = "torch"` or `"cuda"` in `main.py`. The engine wraps PyTorch's kineto profiler; traces reveal per-kernel GPU timelines, CPU↔GPU sync points, and memory copy events.

### Metrics

- **Primary**: output tokens / GPU time (tok/s)
- **Secondary**: TTFT (time to first token), TPOT (time per output token)
- **Diagnostic**: GPU utilization (`nvidia-smi`), KV cache utilization, prefix cache hit rate
