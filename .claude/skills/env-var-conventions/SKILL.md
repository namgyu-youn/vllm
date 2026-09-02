---
name: env-var-conventions
description: Conventions for vLLM environment variables — where to define, how to access, naming, docs extraction, and the torch.compile cache-factor implication. Use when adding, renaming, or reviewing any `VLLM_*` environment variable, or when touching `vllm/envs.py`.
---

# Environment Variables — Conventions

Apply this skill when adding, renaming, or reviewing any vLLM-owned environment
variable (`VLLM_*`), or when touching `vllm/envs.py`.

## Rule 1 — Every `VLLM_*` var is declared in `vllm/envs.py`, in two places

A new env var requires **both** of these edits, kept in sync:

1. **Typed declaration** in the `if TYPE_CHECKING:` block at the top of the file
   (name, type annotation, default value). This is what IDEs and type checkers see.
2. **Entry in the `environment_variables: dict[str, Callable[[], Any]]` dict**
   (between the `--8<-- [start:env-vars-definition]` / `[end:env-vars-definition]`
   markers), as a lambda that parses `os.getenv`, with a `# comment` directly above it.

```python
# In the TYPE_CHECKING block:
VLLM_MY_FLAG: bool = False

# In environment_variables:
# One-line description of what the flag does — this comment is extracted
# into the public docs, write it for users.
"VLLM_MY_FLAG": lambda: bool(int(os.getenv("VLLM_MY_FLAG", "0"))),
```

The comment above the dict entry is **user-facing documentation**: the docs
generator extracts the marked section into
`https://docs.vllm.ai/en/latest/configuration/env_vars.html`.

Never add a raw `os.getenv("VLLM_...")` call site elsewhere in the codebase.
`envs.check_unknown_env_vars()` warns (or hard-fails) on any `VLLM_*` key in the
environment that is not declared in `environment_variables`, so undeclared vars
are treated as typos.

Non-`VLLM_*` vars set by external tooling (`CUDA_*`, `NCCL_*`, `HF_*`, `TORCH_*`)
are read raw where consumed. A handful of external keys (`LOCAL_RANK`,
`CUDA_VISIBLE_DEVICES`, `S3_*`, `MAX_JOBS`, `LD_LIBRARY_PATH`, ...) are
centralized in `envs.py` because vLLM consumes them in many places — follow that
precedent only when there are multiple consumers.

## Rule 2 — Parsing patterns

| Kind | Pattern |
| --- | --- |
| bool flag | `lambda: bool(int(os.getenv("VLLM_X", "0")))` |
| int / float | `lambda: int(os.getenv("VLLM_X", "60"))` |
| optional string | `lambda: os.getenv("VLLM_X")` (default `None`) |
| enumerated choice | `env_with_choices("VLLM_X", "auto", ["auto", "nccl", "shm"])` — validates and raises on bad values; annotate the TYPE_CHECKING entry with `Literal[...]` |
| comma-separated list / set | `env_list_with_choices(...)` / `env_set_with_choices(...)` |
| generate-once default (e.g. random ID) | `get_env_or_set_default("VLLM_X", factory)` — sets the value back into `os.environ` so child processes inherit it |

For a truly dynamic default (depends on torch version, other envs), define a
module-level helper function next to the other ones near the top of the file
(see `use_aot_compile()` / `use_mega_aot_artifact()`) and reference it from the
lambda.

## Rule 3 — Access via `envs.VLLM_FOO`, never raw `os.environ`

```python
import vllm.envs as envs

if envs.VLLM_MY_FLAG:
    ...
```

Module-level `__getattr__` evaluates the lambda lazily. Two caveats:

- **`envs.is_set("VLLM_FOO")`** tells you whether the user explicitly set the
  var (vs. the default applying). Use it when "unset" must behave differently
  from "set to the default value".
- **Caching**: `enable_envs_cache()` is called after service initialization and
  memoizes every lookup. Env vars must therefore be treated as **immutable after
  startup**. Tests that mutate env vars (e.g. `monkeypatch.setenv`) and then
  re-read them through `envs.*` may need `envs.disable_envs_cache()` to avoid
  stale values leaking across tests.

## Rule 4 — Naming

Prefix is always `VLLM_`. The next token signals intent — match existing
precedent in the file:

| Pattern | Meaning | Examples |
| --- | --- | --- |
| `VLLM_USE_FOO` | selects an implementation / backend | `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_USE_AOT_COMPILE` |
| `VLLM_ENABLE_FOO` / `VLLM_DISABLE_FOO` | feature on/off switch | `VLLM_ENABLE_V1_MULTIPROCESSING`, `VLLM_DISABLE_PYNCCL` |
| `VLLM_FORCE_FOO` | overrides autodetection | `VLLM_FORCE_AOT_LOAD` |
| `VLLM_SKIP_FOO` | skips a check/step | `VLLM_SKIP_P2P_CHECK` |
| `VLLM_LOGGING_FOO` / `VLLM_LOG_FOO` | logging-only knob | `VLLM_LOGGING_LEVEL`, `VLLM_LOG_STATS_INTERVAL` |
| `VLLM_DEBUG_FOO` | debug-only instrumentation | `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` |

Avoid `VLLM_DISABLE_FOO` defaulting to disabled-by-default `True` — it produces
double-negative call sites. Hardware-specific knobs carry the platform in the
name (`VLLM_ROCM_*`, `VLLM_CPU_*`, `VLLM_XLA_*`, `VLLM_TPU_*`).

## Rule 5 — torch.compile cache factors

`compile_factors()` in `envs.py` hashes **every declared env var** into the
torch.compile cache key, except those listed in its `ignored_factors` set.

- If the new var can change compiled-graph behavior (kernel selection, fusion,
  dtype, backend choice): do nothing — being a factor is correct and required.
- If it is a purely operational knob (timeout, logging, path, network address,
  fetch limit): add it to `ignored_factors`, otherwise flipping it needlessly
  invalidates users' compile caches.

Reviewers should treat a new operational var missing from `ignored_factors` as
a (minor) defect, and a new behavioral var listed in it as a correctness bug.

## Rule 6 — Env var vs CLI argument

| If the knob is… | Goes in |
| --- | --- |
| User-facing, expected to be set per deployment | CLI arg / config field (`vllm/engine/arg_utils.py`, `vllm/config/`) |
| Expert toggle, kill-switch, platform/vendor integration | `envs.py` env var |
| Debug / CI-only hook | `envs.py` env var with `VLLM_DEBUG_*` / CI naming |

Don't add both surfaces for the same knob. Note that CLI/config fields have
their own compile-cache story (`compute_hash` on config classes) — a config
field is generally preferable for anything documented.

## Renames and removals

There is no built-in alias machinery in `envs.py`. For a rename, keep reading
the old name as a fallback inside the new lambda, emit a deprecation warning at
the consuming site, and note the removal target release; follow
`docs/contributing/deprecation_policy.md`. Never silently flip a default during
a rename — that's a separate behavior change that must be called out in the PR.
