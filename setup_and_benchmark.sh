#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-0.6B"
OUTPUT_DIR="./qwen3-0.6b-nvfp4"
NUM_PROMPTS=200
INPUT_LEN=2048
OUTPUT_LEN=256

# --- 1. Install dependencies ---
pip install llmcompressor datasets

# --- 2. Quantize to NVFP4 ---
python - <<'EOF'
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer
import os

model = os.environ["MODEL"]
output = os.environ["OUTPUT_DIR"]

tokenizer = AutoTokenizer.from_pretrained(model)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")
ds = ds.map(lambda x: {"text": tokenizer.apply_chat_template(
    x["messages"], tokenize=False, add_generation_prompt=False
)})

oneshot(
    model=model,
    dataset=ds,
    recipe=QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"]),
    output_dir=output,
    max_seq_length=512,
    num_calibration_samples=512,
)
print(f"Saved to {output}")
EOF

# --- 3. Benchmark: NVFP4 with CUDA graph ---
echo "=== NVFP4 + CUDA Graph ==="
python -m vllm.entrypoints.cli.main bench throughput \
    --model "$OUTPUT_DIR" \
    --quantization compressed-tensors \
    --num-prompts $NUM_PROMPTS \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN

# --- 4. Benchmark: NVFP4 eager (baseline) ---
echo "=== NVFP4 Eager (baseline) ==="
python -m vllm.entrypoints.cli.main bench throughput \
    --model "$OUTPUT_DIR" \
    --quantization compressed-tensors \
    --enforce-eager \
    --num-prompts $NUM_PROMPTS \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN
