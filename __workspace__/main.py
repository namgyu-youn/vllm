# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM throughput benchmark. See __workspace__/PLAN.md for strategy."""

import torch
from lm_eval.tasks import TaskManager, get_task_dict

from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"

# Set to e.g. "eagle" or a HF repo to enable speculative decoding.
SPECULATIVE_MODEL = None
NUM_SPECULATIVE_TOKENS = 5

TENSOR_PARALLEL_SIZE = 1

GPU_MEMORY_UTILIZATION = 0.93
MAX_NUM_SEQS = 32  # long docs are memory-heavy; keep concurrency modest
OUTPUT_LEN = 256  # max tokens to generate per request

# Profiling
# Two modes (set one, leave the other None):
#   "torch" → TensorBoard trace.  View: tensorboard --logdir=./traces
#             (requires: pip install torch-tb-profiler)
#   "cuda"  → CUDA marker trace.  View: nsys profile python main.py
#             (requires: NVIDIA Nsight Systems)
# Set to None to disable profiling entirely.
PROFILER = None  # "torch" | "cuda" | None
PROFILE_DIR = "./traces"  # only used when PROFILER == "torch"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_prompts(model: str) -> list[str]:
    """
    Load LongBench-E docs via lm_eval, render each doc into a prompt string
    using the task's own doc_to_text() template, then filter out any prompt
    that would overflow MAX_MODEL_LEN (after leaving room for OUTPUT_LEN).
    """
    task_manager = TaskManager()
    task_dict = get_task_dict("longbench_lcc_e", task_manager)

    prompts = []
    for task_name, task in task_dict.items():
        docs = list(task.test_docs())
        for doc in docs:
            prompt = task.doc_to_text(doc)
            prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts from {list(task_dict.keys())}")
    return prompts


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def build_engine() -> LLM:
    kwargs = dict(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        kv_cache_dtype="fp8",
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )
    if SPECULATIVE_MODEL is not None:
        kwargs["speculative_model"] = SPECULATIVE_MODEL
        kwargs["num_speculative_tokens"] = NUM_SPECULATIVE_TOKENS
    if PROFILER is not None:
        # profiler_config must be set at engine init — llm.start_profile()
        # will raise RuntimeError if it is missing.
        kwargs["profiler_config"] = ProfilerConfig(
            profiler=PROFILER,
            torch_profiler_dir=PROFILE_DIR,  # ignored for "cuda" profiler
        )
    return LLM(**kwargs)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(llm: LLM, prompts: list[str]) -> None:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LEN)

    # Warm-up: compile Triton kernels and prime the prefix cache
    print("Warming up...")
    llm.generate(prompts[:4], sampling_params, use_tqdm=False)

    torch.accelerator.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Running {len(prompts)} requests...")
    start_event.record()

    if PROFILER is not None:
        llm.start_profile()

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    if PROFILER is not None:
        llm.stop_profile()
        if PROFILER == "torch":
            print(
                f"Trace written to: {PROFILE_DIR}"
                f"  (view: tensorboard --logdir={PROFILE_DIR})"
            )
        else:
            print("CUDA markers recorded — run: nsys profile python main.py")

    end_event.record()
    torch.accelerator.synchronize()

    elapsed_s = start_event.elapsed_time(end_event) / 1000.0

    total_prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
    total_output_tokens = sum(len(o.token_ids) for out in outputs for o in out.outputs)

    print("\n=== Throughput Results ===")
    print(f"  Requests       : {len(outputs)}")
    print(f"  Prompt tokens  : {total_prompt_tokens:,}")
    print(f"  Output tokens  : {total_output_tokens:,}")
    print(f"  GPU time       : {elapsed_s:.2f}s")
    print(f"  Output tok/s   : {total_output_tokens / elapsed_s:,.1f}")
    print(
        f"  Total tok/s    : "
        f"{(total_prompt_tokens + total_output_tokens) / elapsed_s:,.1f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prompts = load_prompts(MODEL)
    llm = build_engine()
    run_benchmark(llm, prompts)
