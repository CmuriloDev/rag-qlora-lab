import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from utils import benchmark_generation
from config import MODEL_ID, MAX_NEW_TOKENS

DEVICE = "cpu"

print(f"Running on: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)

model.config.use_cache = False

with open("data/massive_context.txt", "r", encoding="utf-8") as f:
    context = f.read()

max_context = min(tokenizer.model_max_length, 4096)

inputs = tokenizer(
    context,
    return_tensors="pt",
    truncation=True,
    max_length=max_context
)

def generate():

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

result = benchmark_generation(generate)

print("\n=== WITHOUT KV CACHE ===")
print(f"Generation Time: {result['time_seconds']:.2f} sec")
print(f"Peak VRAM Usage: {result['vram_mb']:.2f} MB")