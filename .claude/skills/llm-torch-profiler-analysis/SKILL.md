---
name: llm-torch-profiler-analysis
description: "vLLM torch-profiler triage skill. Use it to inspect an existing `trace.json(.gz)` or profile directory, or to drive live profiling against a running vLLM server, and return one three-table report with kernel, overlap-opportunity, and fuse-pattern tables."
---

# vLLM Torch Profiler Analysis

## Overview

Use this skill for `torch.profiler` analysis of vLLM runs.

There is only one public workflow:

- `triage`

Preferred unified entrypoint:

- [scripts/analyze_llm_torch_profile.py](scripts/analyze_llm_torch_profile.py)

Markdown bundling helper:

- [scripts/render_triage_markdown_bundle.py](scripts/render_triage_markdown_bundle.py)

`triage` always prints the same three tables:

- kernel table
- overlap-opportunity table
- fuse-pattern table

By default, all three tables only render rows at or above `1.0%` cumulative GPU-time share.
Rows below that are hidden by default unless the user asks for a lower cutoff.

Keep the fuse-pattern table source-backed and deterministic.
Do not turn it into a fuzzy matcher.

If exact source-backed matching is weak but a kernel cluster is still close to a known family,
add one short note after the tables with exactly one of:

- `high`
- `medium`
- `low`

## Scope note

The analyzer core (`profile_common.py` and the triage helpers) retains
framework auto-detection branches for SGLang, TensorRT-LLM, and TokenSpeed
traces. That code is kept working but is not the supported workflow here:
this skill targets vLLM. If handed a non-vLLM trace, the analyzer will still
classify it, but treat the output as best-effort.

## When To Use It

- inspect a `torch.profiler` trace or profile directory from vLLM
- profile a live vLLM serving endpoint and analyze the result
- summarize which kernel families dominate prefill or decode
- map kernels back to Python code paths
- judge whether a code path still leaves overlap opportunity
- check whether an already-known fusion or overlap path should have applied
  (e.g. a torch.compile fusion pass that was expected to fire)

## Stage-Separated Live Capture Contract

Live capture must not use one mixed prompt as the default.
By default, `analyze_llm_torch_profile.py --url ...` captures two labeled
workloads and then renders the same three tables with separate stage sections:

- prefill: synthetic input length `4090`, output length `1`
- decode: synthetic input length `1`, output length `2048`

Every live profiler path warms up `10` steps before arming the profiler and then
captures `5` active steps by default. Keep this warmup/active split identical
between any two runs you intend to compare.

Use these options to override the contract when the benchmark workload is known:

```bash
--profile-workload both \
--warmup-steps 10 --num-steps 5 \
--prefill-input-len 4090 --prefill-output-len 1 \
--decode-input-len 1 --decode-output-len 2048
```

Allowed `--profile-workload` values:

- `both`: default; capture prefill and decode separately
- `prefill`: capture only the long-input / one-token workload
- `decode`: capture only the one-input / long-output workload
- `legacy`: keep the old `--probe-prompt` / `--probe-max-new-tokens` behavior

If the benchmark under investigation has a known input/output distribution,
set the profiler lengths from it instead of the defaults: prefill uses the
representative input length with output `1`, decode uses input `1` with the
representative output length. For a mixed dataset, profile the slowest
representative bucket (p50 or p95 input/output pair) and record the bucket in
the artifact notes.

## Main Flows

### 1. Single-trace triage from an existing profile dir or trace

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --input /path/to/profile_dir_or_trace.json.gz
```

Use this when one trace is enough.
The overlap table stays conservative in single-trace mode and will tell you when a
mapping/formal pair is needed.

### 2. Single-trace live capture from vLLM

Launch vLLM with the torch profiler enabled, for example:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/path/to/vllm_profile"}'
```

Then run:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --url http://127.0.0.1:8000 \
  --output-dir /path/to/vllm_profile \
  --num-steps 5 \
  --warmup-steps 10 \
  --no-profile-by-stage \
  --profile-workload both
```

Notes:

- `--output-dir` must point to the same `torch_profiler_dir` the server uses;
  the script drives `POST /start_profile` / `POST /stop_profile`
  (`vllm/entrypoints/serve/profile/api_router.py`) and then reads the trace
  files the worker wrote there.
- vLLM's profiler config defaults `torch_profiler_with_stack=true`, so the
  runner only needs to set `torch_profiler_dir`.
- Always pass `--no-profile-by-stage`; vLLM has no internal stage profiler.
  Stage separation comes from the workload-separated capture above.
- If the server runs in a container, the trace dir must be a shared mount
  visible to both the server and this script.

Alternative capture paths when this script cannot drive the server:

- `vllm bench serve --profile ...` arms the profiler around the benchmark run.
- Offline scripts can call `LLM.start_profile()` / `LLM.stop_profile()`
  (see `examples/features/profiling/`).
- The repo-level `generate-profile` skill launches a fresh instrumented
  server end-to-end.

### 3. Two-trace triage from existing profile dirs or traces

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --mapping-input /path/to/low_fusion_profile_dir \
  --formal-input /path/to/optimized_profile_dir
```

Use this when you need stronger overlap attribution and kernel-to-source mapping.

### 4. Two-trace triage from running servers

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --mapping-url http://127.0.0.1:8000 \
  --formal-url http://127.0.0.1:8001 \
  --mapping-output-dir /path/to/mapping_profile \
  --formal-output-dir /path/to/formal_profile \
  --num-steps 5 \
  --no-profile-by-stage
```

## How To Choose The Triage Shape

### Single-trace triage

Use when you want the lowest-friction report:

- one trace is already available
- you mainly want kernel share and fusion clues
- you are comparing two runs side by side by running triage once per trace

Prefer this by default.

### Two-trace triage

Use when you need:

- a stronger overlap answer
- source mapping from a low-fusion run plus final behavior from the optimized run
- more trustworthy overlap recommendations in the middle table

1. mapping trace with `--enforce-eager` or the lower-fusion / more-readable
   compilation config
2. formal trace with the real serving optimizations enabled

Do not call the mapping pass a "fast profile".
It exists to recover `kernel -> cpu_op -> python scope`.

For vLLM specifically, the mapping/formal split maps naturally onto
compilation config: e.g. mapping = default config, formal = with a fusion
pass enabled (`--compilation-config '{"pass_config":{"fuse_norm_quant":true}}'`).
When comparing fusion arms, first confirm from the server debug log that the
fusion pass actually fired ("Replaced N patterns"), otherwise the two arms
are the same configuration.

## Workflow

### Single-trace workflow

1. If the user only wants a diagnosis, one trace is enough.
2. Prefer one-rank traces over merged traces whenever the profiler emitted both.
3. For a live server, let the script drive the profiler only when
   `torch_profiler_dir` is already configured and shared.
4. Prefer `--profile-workload both`; use `legacy` only when reproducing an old
   trace contract.

### Two-trace workflow

1. Produce a mapping trace first with the lower-fusion configuration.
2. Produce a formal trace second with the real serving optimizations enabled.
3. Run `triage` for the three-table report.
4. Read the results in this order:
   - kernel table
   - overlap-opportunity table
   - fuse-pattern table
5. Before calling something a "new" optimization idea, compare the top rows against both [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) and [references/overlap-catalog.md](references/overlap-catalog.md). Check mainline rows first, then the `PR-backed / in-flight` sections. Prefer reporting:
   - an existing fused or overlap path that should already apply here
   - an existing path that appears disabled, unsupported, or regressed in this trace
   - an upstream pattern that is mainline elsewhere but missing locally, or still open upstream
   - a truly new opportunity only when no catalog entry fits
6. If no exact pattern fully matches but the trace is still close to a known family, add one flat similarity note after the tables.
   Use `high`, `medium`, or `low` only.
   Base that note on the full pattern shape, not on one kernel name alone.
   Prefer semantic cues such as producer-consumer chain, source locations, CPU op names, TP context, and model-specific structure.
   Do not rewrite the script table itself to include these heuristic judgments.

## References

Load these only when needed:

- [references/source-map.md](references/source-map.md)
    - vLLM profiler entrypoints and trace-writing paths, for source follow-up
- [references/heuristics.md](references/heuristics.md)
    - overlap labels, dependency-risk interpretation, and limits
- [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md)
    - source-backed catalog of known fuse and overlap pattern families; rows
    were verified against multiple frameworks' source, so treat non-vLLM
    `Primary code` pointers as pattern references and re-verify against vLLM
    source before citing them
- [references/vllm-torch-compile-fusions.md](references/vllm-torch-compile-fusions.md)
    - current vLLM torch.compile fusion passes and the source patterns they target
- [references/overlap-catalog.md](references/overlap-catalog.md)
    - overlap-only lookup table across LLM, VLM, diffusion, disaggregation, and
    speculative scheduling; same caveat on non-vLLM code pointers

## Output Contract

Return:

- trace path or generated profile path
- framework
- model/server args when available
- kernel table
- overlap-opportunity table
- fuse-pattern table
- optional similarity note with `high` / `medium` / `low` when exact matching is inconclusive
- one short summary of what dominates the run
- whether the overlap read came from single-trace triage or mapping/formal two-trace triage
