---
name: write-vllm-test
description: Guide for writing vLLM tests and registering them in Buildkite CI. Covers pytest conventions, conftest fixtures (vllm_runner/hf_runner), RemoteOpenAIServer, multi-GPU decorators, markers, and .buildkite/test_areas registration. Use when creating new tests, adding CI coverage, or reviewing test placement for vLLM features.
---

# Writing vLLM Tests

vLLM tests are plain `pytest` under `tests/`. This skill covers how to write a
test and how it gets picked up by Buildkite CI. For diagnosing CI failures, see
the [ci-fails-buildkite skill](../ci-fails-buildkite/SKILL.md).

## Core Rules

1. **pytest, not unittest** — functions + fixtures + `pytest.mark.parametrize`.
2. **Place the test in the existing `tests/<area>/` that mirrors the feature**
   (`tests/kernels/`, `tests/entrypoints/`, `tests/v1/`, `tests/distributed/`,
   `tests/models/`, `tests/quantization/`, `tests/lora/`, ...). Grep for
   sibling tests of the code you're touching and follow their pattern.
3. **Smallest model that exercises the feature.** Prefer tiny models already
   used in the suite (e.g. `Qwen/Qwen3-0.6B`, `facebook/opt-125m`) — grep
   `tests/` for precedent before introducing a new checkpoint.
4. **Prefer no-engine tests when possible** — pure-logic code (parsing, config
   validation, scheduling math) should be tested directly or with mocks, not by
   booting an engine or server.
5. **A test that isn't reachable from a `.buildkite/test_areas/*.yaml` step does
   not run in CI** — see Registration below.

Run locally (per `AGENTS.md`, use the venv python):

```bash
.venv/bin/python -m pytest -s -v tests/path/to/test_file.py
```

## Fixtures and Utilities

### Correctness vs HuggingFace — `vllm_runner` / `hf_runner`

`tests/conftest.py` provides context-manager factories that handle engine
setup/teardown and GPU cleanup:

```python
def test_my_model(vllm_runner, hf_runner, example_prompts):
    with hf_runner(MODEL, dtype="half") as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens=32)

    with vllm_runner(MODEL, dtype="half") as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=32)

    check_outputs_equal(outputs_0_lst=hf_outputs, outputs_1_lst=vllm_outputs,
                        name_0="hf", name_1="vllm")
```

`VllmRunner` also exposes `generate_greedy_logprobs`, sampling variants, and
encoder/multimodal helpers — read `tests/conftest.py` for the full surface.
Comparison helpers (`check_outputs_equal`, `check_logprobs_close`) live in
`tests/models/utils.py`.

### API-server tests — `RemoteOpenAIServer`

For anything that needs the OpenAI-compatible frontend, launch a real server
subprocess via `tests/utils.py`:

```python
from tests.utils import RemoteOpenAIServer

with RemoteOpenAIServer(MODEL, ["--max-model-len", "2048"]) as server:
    client = server.get_client()          # sync openai client
    # server.get_async_client(), server.url_for("health"), ...
```

It sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and kills the whole process tree
on exit — don't hand-roll `subprocess.Popen("vllm serve ...")` in tests.

### Multi-GPU and process isolation

```python
from tests.utils import multi_gpu_test, create_new_process_for_each_test, large_gpu_mark

@multi_gpu_test(num_gpus=2)      # distributed mark + skipif + fresh process
def test_tp2(...): ...

@create_new_process_for_each_test()   # for tests that initialize CUDA state
def test_needs_clean_process(...): ...

@large_gpu_mark(min_gb=80)       # skip on small cards
def test_big_model(...): ...
```

### Markers (defined in `pyproject.toml`)

`core_model` (run per-PR instead of nightly-only), `cpu_model` / `cpu_test`,
`distributed`, `optional` (skipped unless `--optional`), `slow_test`,
`skip_global_cleanup`, `hybrid_model`, `split`.

## CI Registration — `.buildkite/test_areas/`

CI steps are declared in `.buildkite/test_areas/<area>.yaml` (the old
`test-pipeline.yaml` is deprecated). A step looks like:

```yaml
- label: Samplers Test
  key: samplers-test
  device: h200_35gb            # or num_devices: 2 for multi-GPU steps
  timeout_in_minutes: 75
  source_file_dependencies:    # step runs when a PR touches these paths
  - vllm/model_executor/layers
  - tests/samplers
  commands:
  - pytest -v -s samplers      # cwd is /vllm-workspace/tests
  mirror:
    amd:                       # optional AMD mirror of the step
      device: mi250_1
      commands:
      - pytest -v -s samplers
```

When adding a test file, check in this order:

1. **Covered already?** If an existing step's `commands` runs your whole
   directory (e.g. `pytest -v -s samplers`), a new file there is picked up
   automatically. Most new tests should land this way.
2. **Trigger paths.** Make sure the source files your test guards are listed in
   that step's `source_file_dependencies`; otherwise the step won't run on PRs
   that touch them.
3. **New step** only if no existing step fits (new hardware need, heavy runtime,
   new area). Copy the schema above, pick a realistic `timeout_in_minutes`, and
   only add an `amd` mirror when the test exercises ROCm-specific paths —
   mirroring everything wastes CI capacity.

Hardware-specific suites live in `.buildkite/hardware_tests/`; pipeline-wide
config in `.buildkite/ci_config.yaml`.

## Checklist

- [ ] Test lives in the `tests/<area>/` matching the feature, follows sibling style
- [ ] Uses conftest fixtures / `RemoteOpenAIServer` instead of hand-rolled setup
- [ ] Smallest viable model; no new checkpoint without precedent
- [ ] Multi-GPU tests use `@multi_gpu_test(num_gpus=N)`
- [ ] Reachable from a `.buildkite/test_areas/*.yaml` step, and
      `source_file_dependencies` covers the code under test
- [ ] Passes locally: `.venv/bin/python -m pytest -s -v <file>`
