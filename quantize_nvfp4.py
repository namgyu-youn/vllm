"""
Quantize Qwen3-0.6B to NVFP4 (W4A4) using llmcompressor.

Requirements:
  - Blackwell GPU (SM100+)
  - pip install llmcompressor datasets

Usage:
  python quantize_nvfp4.py
"""
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
OUTPUT = "./qwen3-0.6b-nvfp4"
NUM_SAMPLES = 512
MAX_SEQ_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_SAMPLES}]")
ds = ds.map(
    lambda x: {"text": tokenizer.apply_chat_template(
        x["messages"], tokenize=False, add_generation_prompt=False
    )},
)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"],
)

oneshot(
    model=MODEL,
    dataset=ds,
    recipe=recipe,
    output_dir=OUTPUT,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_SAMPLES,
)
print(f"Saved to {OUTPUT}")
