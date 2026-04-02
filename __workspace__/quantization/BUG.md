# KV-Cache Quantization — New Bug Report

---

## BUG-B: Per-head query quantization permanently dead (init/load ordering violation)

**File:** `vllm/model_executor/layers/attention/attention.py:389-398`

`is_per_head` is decided in `__init__` by inspecting `q_scale.numel()`, but `q_scale`
is always a scalar sentinel (`-1.0`) at that point — weight loading hasn't run yet.
For any model with `num_kv_heads > 1`, `is_per_head` is always False, so `query_quant`
is always built with `GroupShape.PER_TENSOR`. The per-head path (`GroupShape(-1, block_size)`)
is permanently dead.

```python
# attention.py:389 — q_scale.numel() is always 1 here (pre-load sentinel)
is_per_head = (
    hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
)
```

**Reproduce:**

```python
# After __init__, with num_kv_heads=8 and a per-head quant checkpoint:
assert attn.q_scale.numel() == 1          # always — weights not loaded yet
assert attn.query_quant.group_shape == GroupShape.PER_TENSOR  # always, even for per-head
```

**Fix:** Move `is_per_head` detection and `query_quant` construction into
`process_weights_after_loading`, where `_q_scale` has been resolved. In `__init__`,
initialize `self.query_quant = None` unconditionally.

---

## BUG-C: `q_scale` presence silently disables `calculate_kv_scales` (wrong flag ownership)

**File:** `vllm/model_executor/layers/quantization/kv_cache.py:109`

`calculate_kv_scales` governs **K/V** dynamic scale calculation in the forward pass.
But when a checkpoint provides `q_scale > 0`, the weight-loading path forces this flag off:

```python
if layer.q_scale > 0.0:
    q_scale = layer.q_scale
    ...
    layer.calculate_kv_scales = False   # kills K/V dynamic calc based on Q-scale presence
```

A user who explicitly requested dynamic K/V quantization (`--calculate-kv-scales`) loses it
silently if the checkpoint happens to include a static Q scale. Q-scale ownership and K/V-scale
dynamic calculation are orthogonal concerns — coupling them through one flag is a design error.

**Reproduce:**

```python
layer.calculate_kv_scales = True    # user-requested dynamic K/V calc
layer.q_scale = torch.tensor(0.25)  # checkpoint has a static Q scale

method.process_weights_after_loading(layer)

assert layer.calculate_kv_scales == False  # silently suppressed
```

**Fix:** Remove `layer.calculate_kv_scales = False` from the `q_scale > 0` branch.
The static Q scale should be stored and used for Q without touching K/V dynamic
calculation. If static Q + dynamic K/V is semantically invalid, emit a warning and
document the constraint — don't silently override.
