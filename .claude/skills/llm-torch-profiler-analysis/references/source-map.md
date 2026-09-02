# Source Map

Use these vLLM files when the workflow or behavior needs to be justified from
source.

## Profiler configuration

- `vllm/config/profiler.py`
    - `ProfilerConfig` (`--profiler-config`): `profiler` backend selection,
    `torch_profiler_dir`, stack/shape recording flags
    - validation rules (`torch_profiler_dir` required when `profiler` is
    `torch`)

## Profiler control entrypoints

- `vllm/entrypoints/serve/profile/api_router.py`
    - `POST /start_profile` and `POST /stop_profile` HTTP routes on the
    OpenAI-compatible server

- `vllm/entrypoints/llm.py`
    - `LLM.start_profile()` / `LLM.stop_profile()` for offline runs

- `vllm/benchmarks/serve.py`
    - `vllm bench serve --profile` arms the server profiler around the
    benchmark workload via the HTTP routes

## Worker-side trace writing

- `vllm/v1/worker/gpu_worker.py`
    - `Worker.profile(...)`: creates and drives the `torch.profiler` instance,
    writes per-rank trace files under `torch_profiler_dir`

## Documentation and examples

- `docs/contributing/profiling.md`
    - canonical profiling docs, including GUI/visualization guidance

- `examples/features/profiling/`
    - minimal offline profiling examples (`simple_profiling_offline.py` etc.)
