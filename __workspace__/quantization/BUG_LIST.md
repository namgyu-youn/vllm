# KV-Cache Quantization — Bug / Anti-Pattern List

Ordered from easiest to fix (top) to most involved (bottom).

---

## BUG-01: Redundant `get_quant_method()` call (dead code)

**File:** `vllm/model_executor/layers/attention/attention.py:133-157`  
**Severity:** Low (no runtime impact, correctness risk if diverged)

`_init_kv_cache_quant` calls `get_quant_method()` twice. The result of the first call (line 133) is thrown away; the second call (line 156) is actually used.

```python
# Line 133 — result unused
quant_method = (
    quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
)

set_default_quant_scales(layer, register_buffer=True)   # ... lines 150
layer._o_scale_float = None

# Line 156 — redundant re-computation
quant_method = (
    quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
)
```

**Fix:** Delete lines 133-135. The second assignment at line 156 is the operative one.

---

## BUG-02: Typo in error message — missing space

**File:** `vllm/model_executor/layers/quantization/kv_cache.py:136`  
**Severity:** Low (cosmetic)

```python
raise ValueError(
    "Only support per-tensor scaling factorfor fp8-quantized Q/prob"
    #                                    ^ missing space
)
```

**Fix:** Change `"scaling factorfor"` → `"scaling factor for"`.

---

## BUG-03: `prob_scale` missing FNUZ adjustment

**File:** `vllm/model_executor/layers/quantization/kv_cache.py:121-126`  
**Severity:** Medium (accuracy on ROCm FNUZ platforms)

`k_scale` and `v_scale` are multiplied by 2 on FNUZ platforms (lines 68-70, 84-86), and `q_scale` is also adjusted (line 116-117). But `prob_scale` is never multiplied:

```python
if layer.prob_scale > 0.0:
    prob_scale = layer.prob_scale
    if current_platform.is_fp8_fnuz():
        prob_scale *= 2   # <-- this line is MISSING
else:
    prob_scale = 1.0
```

If a checkpoint provides a `prob_scale` and the platform is ROCm FNUZ, the scale will be off by 2×, causing incorrect attention output.

**Fix:** Add `prob_scale *= 2` inside the `is_fp8_fnuz()` branch for `prob_scale`.

---

## BUG-04: `q_scale` fallback silently uses `k_scale` without a warning for FNUZ

**File:** `vllm/model_executor/layers/quantization/kv_cache.py:93-100`  
**Severity:** Low-Medium

When `q_scale` is missing from the checkpoint, it falls back to `k_scale` (line 99). However, when `q_scale` is loaded from the checkpoint it gets the FNUZ ×2 adjustment (line 116-117). The fallback path at line 99 copies the **already-adjusted** `k_scale` into `_q_scale`, which means on FNUZ `q_scale` ends up being `k_scale * 2` — correct. But the warning message (lines 94-98) doesn't mention FNUZ behavior, potentially confusing users debugging scale issues.

---

## BUG-05: `kv_cache_scheme` silently overrides `cache_config`

**File:** `vllm/model_executor/layers/attention/attention.py:228-235`  
**Severity:** Medium (hidden behavior)

```python
kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
if kv_cache_scheme is not None:
    kv_cache_dtype = "fp8"
    calculate_kv_scales = False
    if cache_config is not None:
        cache_config.cache_dtype = "fp8"          # mutates shared config object
        cache_config.calculate_kv_scales = False
```

If a user explicitly sets `--kv-cache-dtype auto` but loads a `compressed-tensors` or `modelopt` checkpoint with a `kv_cache_scheme`, this silently forces FP8 and mutates the `cache_config` object. There is no log warning emitted.

---

## BUG-06: `KVCacheQuantSchema` hardcodes dtype to `float8_e4m3fn`

**File:** `vllm/model_executor/layers/quantization/schema.py:28-33`  
**Severity:** Medium (blocks loading valid checkpoints)

```python
assert self.dtype == "float8_e4m3fn", (...)
```

`CacheConfig` supports `fp8_e5m2`, `fp8_inc`, `fp8_ds_mla` etc., but the schema validator rejects any checkpoint whose metadata says anything other than `float8_e4m3fn`. A checkpoint saved with dtype `float8_e4m3fnuz` (valid ROCm variant) would be rejected.

---

## BUG-07: Scale storage in three inconsistent forms

**File:** `vllm/model_executor/layers/quantization/kv_cache.py` + `attention/attention.py`  
**Severity:** Medium (maintenance / correctness risk)

Each scale exists in three forms simultaneously:

1. `nn.Parameter` (e.g., `k_scale`) — temporary, for checkpoint loading
2. `nn.Buffer` (e.g., `_k_scale`) — persistent, for kernel dispatch
3. Python `float` (e.g., `_k_scale_float`) — CPU-side copy for backends like FlashInfer

There is no canonical update path; callers must remember to update all three. A missed update (e.g., updating `_k_scale` but not `_k_scale_float`) causes subtle accuracy bugs only visible on specific backends.

---

## BUG-08: `calculate_kv_scales` deprecated but still in forward hot-path

**File:** `vllm/model_executor/layers/attention/attention.py:417-418`  
**Severity:** Low-Medium (tech debt becoming a maintenance liability)

`CacheConfig.calculate_kv_scales` is documented as deprecated (to be removed in v0.19), but `Attention.forward()` still gates a `maybe_calc_kv_scales` call on it. If removed from `CacheConfig` without removing this branch, it will raise `AttributeError` at inference time.
