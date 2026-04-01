# vLLM Quantization Overview

## Model Quantization

### Abstractions (`quantization/base_config.py`)

| Class | Role |
| --- | --- |
| `QuantizationConfig` | Loaded from checkpoint. Factory via `get_quant_method(layer, prefix)` |
| `QuantizeMethodBase` | Worker: `create_weights()`, `apply()`, `process_weights_after_loading()` |

### Config → Layer Flow

```text
Checkpoint metadata
  └─ QuantizationConfig.from_config()
       └─ LinearBase.__init__(quant_config)
            └─ quant_config.get_quant_method(self, prefix)   # factory
                 └─ LinearMethodBase subclass
                      ├─ create_weights()                    # register quantized params
                      └─ process_weights_after_loading()     # post-load transforms
```

### Method Hierarchy

```text
QuantizeMethodBase
├── LinearMethodBase
│   ├── UnquantizedLinearMethod     (default)
│   ├── Fp8LinearMethod             (static/dynamic activation, block-wise weights)
│   ├── AWQLinearMethod             (4-bit weight-only)
│   ├── GPTQMarlinLinearMethod      (Marlin kernel-accelerated)
│   └── ... (25+ methods: bitsandbytes, torchao, compressed-tensors, etc.)
│
└── FusedMoEMethodBase
    ├── Fp8MoEMethod
    └── AWQMoEMarlinMethod
```

> Layers are **method-agnostic** — they delegate entirely to whatever `get_quant_method()` returns.

### Example: TorchAO Quantized Checkpoint

**1. Config loading** — `torchao.py::TorchAOConfig.from_config()` (line 140)

```python
# Parses checkpoint's quantization_config dict
ao_config = config_from_dict(hf_config["default"])   # e.g. Int8WeightOnlyConfig
return cls(ao_config, skip_modules, is_checkpoint_torchao_serialized=True)
```

**2. Weight creation** — `torchao.py::TorchAOLinearMethod.create_weights()` (line 300)

```python
weight = Parameter(torch.empty(out, in, dtype=params_dtype))
if self.quant_config.is_checkpoint_torchao_serialized:
    weight = torchao_quantize_param_data(weight, self.quant_config.torchao_config)
layer.register_parameter("weight", weight)
```

**3. Post-load packing** — `torchao.py::TorchAOLinearMethod.process_weights_after_loading()` (line 336)

```python
# If pre-quantized checkpoint: just convert to hardware-optimal packed layout
layer.weight = Parameter(convert_to_packed_tensor_based_on_current_hardware(layer.weight))

# If non-quantized checkpoint (online quant path):
weight = torchao_quantize_param_data(layer.weight, self.quant_config.torchao_config)
layer.register_parameter("weight", convert_to_packed_tensor_based_on_current_hardware(weight))
```

**4. Inference** — `torchao.py::TorchAOLinearMethod.apply()` (line 328)

```python
return F.linear(x, layer.weight, bias)   # weight is a quantized AQTTensor; dispatch handled by torchao
```

> `torchao_quantize_param_data()` (line 258) wraps a dummy `nn.Linear`, calls `torchao.quantize_()` on it, and returns the quantized weight — enabling reuse of torchao's own quantization logic.

---

## KV-Cache Quantization

### Key Files

- `quantization/kv_cache.py` — `BaseKVCacheMethod`
- `quantization/schema.py` — `KVCacheQuantSchema` (Pydantic, validates checkpoint scales)
- `layers/attention/attention.py` — integration point

### Attention Layer Flow

```text
Attention.__init__()
  └─ quant_config.get_quant_method(self, prefix)
       └─ Fp8KVCacheMethod
            └─ Registers scale buffers: q_scale, k_scale, v_scale, prob_scale
                 └─ process_weights_after_loading()   # loads checkpoint scales

Attention.forward()
  └─ Backend kernel (FlashInfer / FlashAttn / Triton)
       ├─ Quantize Q/K/V before caching
       ├─ Dequantize K/V during attention compute
       └─ (quant/dequant is kernel-level, not Python-level)
```

`kv_cache_dtype` is set from either `CacheConfig.cache_dtype` (user config) or `quant_config.kv_cache_scheme` (checkpoint). Supported: `auto`, `fp8`, `fp8_e4m3fn`, `fp8_e4m3fnuz`, `int8`.

### Example: FP8 KV-Cache

**1. Attention init** — `attention.py::_init_kv_cache_quant()` (line 116)

```python
set_default_quant_scales(layer, register_buffer=True)  # _k_scale, _v_scale = 1.0
quant_method = quant_config.get_quant_method(layer, prefix)  # → Fp8KVCacheMethod
layer.quant_method = quant_method
layer.quant_method.create_weights(layer)   # registers k_scale, v_scale as Parameters
```

**2. Scale loading from checkpoint** — `kv_cache.py::BaseKVCacheMethod.process_weights_after_loading()` (line 48)

```python
k_scale = layer.k_scale.to("cpu").tolist()   # loaded from checkpoint file
v_scale = layer.v_scale.to("cpu").tolist()
if current_platform.is_fp8_fnuz():           # ROCm: FNUZ format needs 2× adjustment
    k_scale *= 2;  v_scale *= 2
layer._k_scale.copy_(k_scale)               # buffer used by attention kernel
layer._k_scale_float = k_scale              # host-side float for backends that need it
```

**3. Dequant at inference** — `v1/attention/backends/flashinfer.py` (line 88, Triton kernel)

```python
# Inside _trtllm_prefill_attn_kvfp8_dequant Triton kernel:
k_scale_val = tl.load(k_scale_ptr)                              # layer._k_scale
dequant_k   = (fp8_k.to(tl.float32) * k_scale_val).to(dtype)  # fp8 → float32 → bf16
tl.store(mock_kv_cache_ptr + dst_k, dequant_k)                 # write to temp buffer
# ... same for v_scale / V cache
```

Called from the Python-level forward via `trtllm_prefill_attn_kvfp8_dequant(kv_cache, ..., layer._k_scale, layer._v_scale, ...)` (line 1510).

---

## Big Picture

```text
Checkpoint
  ├─► QuantizationConfig.from_config()
  │     ├─ get_quant_method(LinearBase)  → quantized matmul
  │     ├─ get_quant_method(Attention)   → KV cache scale management
  │     └─ get_quant_method(FusedMoE)   → expert quantized params
  │
  └─► KVCacheQuantSchema
        └─ { tp_rank → { layer_idx → scale } }   # per-layer, per-TP-rank scales
```
