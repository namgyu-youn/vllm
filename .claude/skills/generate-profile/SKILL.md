---
name: generate-profile
description: Generate an e2e torch.profiler trace of a vLLM server run. Launches vllm serve with profiling enabled, validates the server, captures a Chrome/Perfetto-compatible trace, and returns the profile path.
---

# Generate an E2E Profile of a vLLM Server Run

Launch a vLLM server with the torch profiler enabled, sanity-check it, capture
a trace, and report the trace path. Reference: `docs/contributing/profiling.md`.

Profiling adds significant overhead — send only a few requests, and never
leave a profiling-enabled server running for real workloads.

## Step-by-step Workflow

### Step 1: Launch the server with profiling enabled

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> vllm serve <model> --port 8000 \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "<trace_dir>"}' &
```

- Default model choice: an 8B-class instruct model (e.g. `Qwen/Qwen3-8B`,
  `meta-llama/Llama-3.1-8B-Instruct`) unless the user names one.
- Check memory files for the user's GPU preferences before picking a device.
- Save the PID for cleanup.
- Useful extra config keys (all inside `--profiler-config`):
  `torch_profiler_record_shapes`, `torch_profiler_with_memory`,
  `torch_profiler_with_stack` (on by default), `torch_profiler_with_flops`,
  `torch_profiler_use_gzip` (on by default).

### Step 2: Wait for server readiness

```bash
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && { echo ready; break; }
  sleep 5
done
```

Startup takes ~1–5 minutes depending on model size, compilation, and CUDA
graph capture.

### Step 3: Sanity-check the server

Send one real request and verify the output is coherent before profiling:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 32}'
```

If the response is garbage or errors, stop — do not profile a broken setup.

### Step 4: Capture the profile

Preferred — `vllm bench serve` drives start/stop automatically:

```bash
vllm bench serve --backend vllm --model <model> --port 8000 \
    --dataset-name random --num-prompts 4 --profile
```

Manual alternative:

```bash
curl -X POST http://127.0.0.1:8000/start_profile
# ... send a few requests ...
curl -X POST http://127.0.0.1:8000/stop_profile
```

**The stop call flushes traces to `torch_profiler_dir` and can take minutes for
large captures** — let it run to completion; the engine client waits without
timing out.

### Step 5: Kill the server and report

```bash
kill <server_pid>; sleep 5
pgrep -af "vllm serve" || echo "server down"
ls -la <trace_dir>
```

One `.trace.json(.gz)` file is written per worker rank (with `--tensor-parallel-size N`
you get N files). Report the directory and file list.

## Viewing the Trace

- **Perfetto UI**: <https://ui.perfetto.dev/> (drag and drop; reads `.gz` directly)
- `chrome://tracing` also works.

## Nsight Systems alternative (lower overhead)

For kernel-level detail without torch-profiler overhead:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # required for nsys + fork issues
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    vllm bench latency --model <model> --num-iters 1 --batch-size 16
```

For server mode add `--capture-range=cudaProfilerApi --capture-range-end repeat`
to `nsys` and launch `vllm serve <model> --profiler-config.profiler cuda`; then
drive it with `vllm bench serve ... --profile`. Analyze with
`nsys stats <report>` or the Nsight GUI.

## Analyzing the trace

Hand the resulting `trace.json(.gz)` to the
[llm-torch-profiler-analysis skill](../llm-torch-profiler-analysis/SKILL.md)
for the kernel/overlap/fusion triage tables.
