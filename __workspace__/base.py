# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Naive HuggingFace throughput baseline (no vLLM). Compare with main.py."""

import torch
from main import MODEL, OUTPUT_LEN, load_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    prompts = load_prompts(MODEL)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print("Warming up...")
    for prompt in prompts[:4]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=OUTPUT_LEN, do_sample=False)

    torch.accelerator.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_prompt_tokens = 0
    total_output_tokens = 0

    print(f"Running {len(prompts)} requests...")
    start_event.record()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=OUTPUT_LEN, do_sample=False
            )
        total_prompt_tokens += input_len
        total_output_tokens += output_ids.shape[1] - input_len
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)}")

    end_event.record()
    torch.accelerator.synchronize()

    elapsed_s = start_event.elapsed_time(end_event) / 1000.0

    print("\n=== Throughput Results ===")
    print(f"  Requests       : {len(prompts)}")
    print(f"  Prompt tokens  : {total_prompt_tokens:,}")
    print(f"  Output tokens  : {total_output_tokens:,}")
    print(f"  GPU time       : {elapsed_s:.2f}s")
    print(f"  Output tok/s   : {total_output_tokens / elapsed_s:,.1f}")
    print(
        f"  Total tok/s    : "
        f"{(total_prompt_tokens + total_output_tokens) / elapsed_s:,.1f}"
    )
