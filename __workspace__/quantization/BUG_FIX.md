# KV-Cache Quantization — Applied Fixes

## BUG-01: Removed redundant `get_quant_method()` call

**File:** `vllm/model_executor/layers/attention/attention.py`

`_init_kv_cache_quant` called `get_quant_method()` twice. The first result was unused — it was immediately discarded before `set_default_quant_scales()` ran, then recomputed. Removed the dead first call.

---

## BUG-02: Typo in error message

**File:** `vllm/model_executor/layers/quantization/kv_cache.py`

`"scaling factorfor"` → `"scaling factor for"` in the per-tensor scale validation error.

---

## BUG-04: Dead `q_scale` fallback block replaced with correct implementation

**File:** `vllm/model_executor/layers/quantization/kv_cache.py`

**Root cause:** The fallback block (inside the `is_quantized_kv_cache` branch) set `layer._q_scale.copy_(k_scale)` to proxy a missing `q_scale`. But the unconditional block below always overwrote `_q_scale` — with `q_scale = 1.0` when no checkpoint q_scale was present. The fallback was completely dead.

**Fix:**

- Removed the dead fallback block entirely.
- Added an `elif` branch in the unconditional `q_scale` resolution block: when `q_scale` is not in the checkpoint and kv_cache is quantized, fall back to `layer._k_scale_float` (already resolved and FNUZ-adjusted) instead of `1.0`.
- FNUZ correctness is now automatic — `_k_scale_float` already carries the ×2 factor set earlier in the same function.

---

## BUG-06: `KVCacheQuantSchema` accepts `float8_e4m3fnuz`

**File:** `vllm/model_executor/layers/quantization/schema.py`

The dtype validator hardcoded `"float8_e4m3fn"` as the only valid value, rejecting checkpoints from ROCm platforms that store `"float8_e4m3fnuz"`. Expanded the check to `valid_dtypes = ("float8_e4m3fn", "float8_e4m3fnuz")`.

---

## Skipped / Deferred

| Bug | Reason |
|-----|--------|
| BUG-03: `prob_scale` missing FNUZ ×2 | No ROCm hardware available |
| BUG-05: `kv_cache_scheme` silent override | Complex — unclear correct behavior; deferred |
| BUG-07: Triple scale storage forms | Architectural refactor, out of scope |
| BUG-08: `calculate_kv_scales` deprecated in hot-path | Requires coordinated removal with deprecation cycle |
