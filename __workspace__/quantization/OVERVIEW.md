# KV-Cache Quantization in vLLM — Overview

## 1. Core Abstractions

Two base classes define the quantization contract:

- **`QuantizationConfig`** (`quantization/base_config.py`) — loaded from checkpoint metadata. Acts as a factory via `get_quant_method(layer, prefix)`.
- **`QuantizeMethodBase`** (`quantization/base_config.py`) — the worker. Implements `create_weights()`, `apply()`, `process_weights_after_loading()`.

KV-cache quantization reuses the same interface but subclasses from **`BaseKVCacheMethod`** (`quantization/kv_cache.py`) which extends `QuantizeMethodBase` specifically for Attention layers.

---

## 2. Class Hierarchy

```
QuantizeMethodBase (ABC)
├── LinearMethodBase           — weight/activation quantization for linear layers
├── FusedMoEMethodBase         — MoE expert quantization
└── BaseKVCacheMethod          — KV cache quantization for Attention layers
    ├── Fp8KVCacheMethod                 (fp8.py)
    ├── ModelOptFp8KVCacheMethod         (modelopt.py)
    ├── CompressedTensorsKVCacheMethod   (compressed_tensors/)
    ├── QuarkKVCacheMethod               (quark/)
    └── PetitFp8KVCacheMethod            (petit.py)
```

---

## 3. End-to-End Flow

### Phase 1: Config Loading (startup)

```
ModelConfig detects quantization from checkpoint
    └─ VllmConfig._get_quantization_config()
         └─ get_quant_config() [weight_utils.py]
              └─ Quant_cls.from_config(hf_quant_config)
                   └─ Returns typed QuantizationConfig instance
```

### Phase 2: Attention Layer Init (`Attention.__init__`)

```
Attention.__init__(quant_config, cache_config, ...)
    │
    ├─ 1. Determine kv_cache_dtype
    │       Priority: kv_cache_scheme (quant_config) > cache_config.cache_dtype > "auto"
    │       Valid values: "auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla"
    │
    ├─ 2. Select attention backend
    │       get_attn_backend(kv_cache_dtype, ...) → AttentionBackend subclass
    │
    ├─ 3. Init KV cache quant (_init_kv_cache_quant)
    │       set_default_quant_scales(layer, register_buffer=True)
    │           → registers _k_scale, _v_scale, _q_scale, _prob_scale as buffers (=1.0)
    │           → registers _k/v/q_scale_float as Python floats (=1.0)
    │       if quant_method exists (BaseKVCacheMethod):
    │           quant_method.create_weights(layer)
    │               → adds k_scale, v_scale, q_scale, prob_scale as Parameters (=-1.0)
    │                 (sentinel value; overwritten on checkpoint load)
    │
    └─ 4. Setup query quantization (QuantFP8) if kv_cache_dtype is fp8
```

### Phase 3: Weight Loading (`process_weights_after_loading`)

```
BaseKVCacheMethod.process_weights_after_loading(layer)
    │
    ├─ Scale resolution (3 branches):
    │     k_scale > 0 AND v_scale > 0  →  use checkpoint scales
    │     k_scale < 0 AND v_scale < 0  →  default to 1.0
    │     mixed (one valid, one not)   →  duplicate max(k_scale, v_scale) to both
    │
    ├─ ROCm FNUZ adjustment: scale *= 2
    │
    ├─ Copy resolved values → layer._k_scale / _v_scale / _q_scale / _prob_scale (buffers)
    │   Copy as float      → layer._k_scale_float / _v_scale_float / _q_scale_float
    │
    └─ Delete temporary Parameters: del layer.k_scale, v_scale, q_scale, prob_scale
```

### Phase 4: Forward Pass (`Attention.forward`)

```
Attention.forward(query, key, value, kv_cache, ...)
    │
    ├─ (optional) maybe_calc_kv_scales — dynamic scale calculation if calculate_kv_scales=True
    │
    ├─ (optional) query_quant.quantize(query, q_scale) — quantize Q to FP8
    │
    └─ self.impl.forward(query, key, value, kv_cache, ...)  ← attention backend
            Kernel handles:
            ├─ Quantize K/V to fp8 (k_scale/v_scale) before storing in KV cache
            ├─ Dequantize K/V from cache during attention compute
            └─ (optional) quantize softmax output (prob_scale)
```

---

## 4. Scale Tensor Lifecycle

| Name | Type | Initial Value | Purpose |
|------|------|---------------|---------|
| `k_scale`, `v_scale`, `q_scale`, `prob_scale` | `nn.Parameter` | `-1.0` (sentinel) | Temporary: loaded from checkpoint, deleted after |
| `_k_scale`, `_v_scale`, `_q_scale`, `_prob_scale` | `nn.Buffer` | `1.0` | Persistent: used in forward pass, moves with model.to(device) |
| `_k_scale_float`, `_v_scale_float`, `_q_scale_float` | Python `float` | `1.0` | CPU-side value for backends requiring host memory (e.g., FlashInfer) |

The two-stage design (Parameter → Buffer) exists so that scales are loadable via the state dict during checkpoint loading but are stored as buffers (not parameters) at runtime.

---

## 5. KVCacheQuantSchema

Located in `quantization/schema.py`. Validates the quantization JSON file shipped alongside some checkpoints (primarily ROCm FP8 models):

```
QuantParamSchema
└─ model_type: str | None
└─ kv_cache: KVCacheQuantSchema
       dtype: must be "float8_e4m3fn"
       scaling_factor: { tp_rank: { layer_idx: float } }
       Validators:
         - All TP ranks 0..tp_size-1 must be present
         - Each TP rank must have all num_hidden_layers entries
         - Current rank must have all layer scales
```

---

## 6. Big Picture

```
Checkpoint metadata
     │
     ▼
QuantizationConfig.from_config()
     │
     ├──► get_quant_method(LinearBase)    → LinearMethodBase
     │         create_weights()  → quantized params
     │         apply()           → quantized matmul
     │
     ├──► get_quant_method(Attention)     → BaseKVCacheMethod
     │         create_weights()  → scale parameters (temporary)
     │         process_weights_after_loading() → resolves → scale buffers
     │         (quant/dequant in attention backend kernels at forward time)
     │
     └──► get_quant_method(FusedMoE)      → FusedMoEMethodBase
               create_weights()  → expert quantized params
```

**Key design principles:**

- Layers are method-agnostic — they delegate to whatever `get_quant_method()` returns.
- KV-cache quant reuses the `QuantizeMethodBase` interface but **does not** do compute in `apply()` — actual quant/dequant is inside the attention backend kernels.
- Scale tensors are registered as buffers so they move with `model.to(device)`.
