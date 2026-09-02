---
name: debug-distributed-hang
description: Debug hanging issues in vLLM distributed inference (TP/PP/DP/EP). Covers locating the hang via py-spy/NCCL logs/CUDA coredump, vLLM debug env vars, per-rank logging to find state divergence, and binary-search methodology for the first diverge point. Use when a multi-GPU vLLM run hangs, freezes, or times out during collective operations.
---

# Debugging Distributed Hangs in vLLM

## Overview

Hangs in distributed inference happen when ranks diverge in state, causing
collective operations (AllGather, AllReduce, Broadcast, Barrier) to deadlock.
Common causes:

- **Size mismatch**: ranks pass different tensor sizes to a collective
- **Branch divergence**: one rank enters a collective, another skips it
- **Cascading state drift**: a small non-determinism (e.g., floating-point)
  propagates into different batch structures
- **Resource exhaustion**: one rank OOMs or crashes, others wait forever
- **Broken hardware / network setup**: NCCL can't establish communication at all

First rule out setup problems: `docs/usage/troubleshooting.md` has a standalone
NCCL/GLOO/pynccl sanity script to run with `torchrun` — if that hangs too, the
problem is drivers/network, not vLLM.

## Prerequisites

- **py-spy**: `uv pip install py-spy`. Needs root or `CAP_SYS_PTRACE` to attach.
- **cuda-gdb**: ships with the CUDA toolkit.

## Step 1: Confirm and Locate the Hang

### 1a. py-spy every process

vLLM runs the engine core and workers as separate processes
(`EngineCore_0`, per-rank workers; with Ray, ray worker processes). Dump all
of them:

```bash
ps aux | grep -iE "vllm|EngineCore" | grep -v grep
py-spy dump --pid <pid>          # repeat per process
```

The hanging thread is typically blocked in `cuStreamSynchronize` or an NCCL
collective; note which Python frame issued it (e.g.
`tensor_model_parallel_all_gather`). Comparing dumps *across ranks* is the
point: one rank stuck in AllGather while another waits in the scheduler already
tells you who missed the collective.

For single-process debugging (stock `pdb` works), set
`VLLM_ENABLE_V1_MULTIPROCESSING=0`.

### 1b. Debug logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_STATS_INTERVAL=1
export NCCL_DEBUG=INFO           # TRACE for even more
export NCCL_DEBUG_SUBSYS=COLL
```

Look for the last collective logged before the hang; a size mismatch shows up
as one rank waiting on a collective the others never entered.

Last resort: `export VLLM_TRACE_FUNCTION=1` records every Python function call
per rank into log files (path printed at startup). It slows execution >100×,
but the tail of each rank's trace is exactly where that rank is stuck.

If ranks pick a wrong IP/interface (multi-node), fix with `VLLM_HOST_IP`,
`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`.

### 1c. CUDA Coredump

To see which GPU kernel is stuck, set before launching:

```bash
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_PIPE="/tmp/cuda_pipe_%h_%p"
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h_%p"
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
```

While hanging, write into the pipe to trigger the dump (or `kill -SIGABRT`
if the process doesn't need to survive):

```bash
ls -la /proc/<pid>/fd/ | grep cuda_pipe
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_pipe_<hostname>_<pid>
```

Open with `cuda-gdb --batch -ex "target cudacore <file>"` — the focus line
names the stuck kernel, e.g. `ncclDevKernel_AllGather_RING_LL<<<...>>>`, which
distinguishes "NCCL collective mismatch" from "compute kernel wedged".

### 1d. Identify the Collective

From stacks and logs, pin down: which collective hangs, which code path invokes
it (e.g. logits gather, custom all-reduce, PP send/recv in
`vllm/distributed/`), and whether it's a size mismatch or a missing
participant.

## Step 2: Per-Rank Logging

The key technique: each rank writes its own log file so you can diff them.

```python
import os

_debug_files = {}

def get_debug_file(rank):
    if rank not in _debug_files:
        _debug_files[rank] = open(f"/tmp/debug_rank{rank}.log", "w")
    return _debug_files[rank]
```

Gate it behind an ad-hoc env var. Use a **non-`VLLM_` prefix** (e.g.
`DEBUG_HANG=1`) — vLLM warns/fails on unknown `VLLM_*` vars
(`envs.check_unknown_env_vars`), and this is throwaway instrumentation:

```python
if os.environ.get("DEBUG_HANG"):
    f = get_debug_file(rank)
    f.write(f"EVENT_NAME key1={val1} key2={val2}\n")
    f.flush()
```

Get the rank from `vllm.distributed.parallel_state` (e.g.
`get_tensor_model_parallel_rank()`), or `os.environ["LOCAL_RANK"]` early in
startup.

### What to Log

Structured events at state-mutation points, with consistent uppercase prefixes
for easy grep/diff:

```python
f.write(f"SCHED step={step} num_reqs={n} num_tokens={lens}\n")
f.write(f"SAMPLE token_hash={h} \n")
f.write(f"KV_ALLOC req={rid} blocks={n}\n")
```

### Hash Large Tensors

```python
import hashlib
h = hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()[:8]
```

### Avoid Implicit Synchronization

`tensor.cpu()` / `.tolist()` / `.numpy()` synchronize CUDA, which can shift or
mask the hang — or deadlock if placed between two back-to-back collectives.
Prefer values already on CPU (Python ints, lengths, request IDs). Hash GPU
tensors only where the GPU is already idle (between scheduler steps, not inside
the model forward).

## Step 3: Diff to Find the Diverge Point

```bash
grep "^SCHED" /tmp/debug_rank0.log > /tmp/s0.txt
grep "^SCHED" /tmp/debug_rank1.log > /tmp/s1.txt
diff /tmp/s0.txt /tmp/s1.txt | head -20

grep -c "^SCHED" /tmp/debug_rank*.log   # differing counts = one rank ran more steps
```

The first differing line is the step where ranks diverge; the root cause is at
or before it.

## Step 4: Binary-Search the Root Cause

1. **Identify inputs** of the diverging operation; log a hash for each.
2. **Diff the hashes across ranks** — the non-matching input is where
   divergence entered.
3. **Recurse**: trace where that input was produced, hash *its* inputs, diff
   again, until you reach the origin.

## Step 5: Common Root Causes and Fixes

### Floating-Point Non-Determinism

All "logical" inputs identical, but derived FP values (softmax, renormalized
probabilities) differ slightly per GPU. Typical chain in speculative decoding:
per-rank probability differences → different sampled/accepted tokens →
different prefix-cache state → different batch shapes → collective size
mismatch → hang. Fix: compute on one rank and broadcast, or make the kernel
deterministic.

### Random Number Divergence

Ops using `torch.rand` produce different values per rank. Fix: generate on
rank 0 and broadcast, or use a shared seed.

### Conditional Code Paths

A condition (memory headroom, queue length) evaluates differently across
ranks, so one rank enters a collective the other skips. Fix: synchronize the
condition value before branching, or restructure so all ranks take the same
path.

### Pipeline Parallel Send/Recv Mismatch

One stage `send`s what the next never `recv`s (or vice versa) — point-to-point,
unlike TP's collective mismatches. Fix: ensure all stages agree on microbatch
count and the send/recv sequence per microbatch.

## Step 6: Verify the Fix

Intermittent hangs need many clean runs: a ~30%-of-the-time hang needs at
least 10 consecutive passes before you can claim it fixed. Then remove all the
debug env vars — `VLLM_TRACE_FUNCTION` and `NCCL_DEBUG` left on will wreck
performance.

## Quick Reference

| Technique | When to Use |
| --- | --- |
| py-spy dump per process | First step — see where each rank is stuck |
| `NCCL_DEBUG=INFO` + `SUBSYS=COLL` | Identify which collective and sizes |
| `VLLM_TRACE_FUNCTION=1` | Tail of per-rank trace = hang location (very slow) |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Single process, stock pdb works |
| CUDA coredump + `cuda-gdb` | See which GPU kernel is blocked |
| Per-rank log files + `diff` | Find the exact step of divergence |
| Tensor hashes | Compare large tensors across ranks cheaply |
| troubleshooting.md sanity script | Rule out broken drivers/network first |
