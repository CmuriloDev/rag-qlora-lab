import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from utils import benchmark_generation
from config import MODEL_ID, MAX_NEW_TOKENS

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

try:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    print("FlashAttention-2 enabled.")

except Exception:

    print("FlashAttention-2 unavailable. Using default attention.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

model.config.use_cache = True

with open("data/massive_context.txt", "r", encoding="utf-8") as f:
    context = f.read()

inputs = tokenizer(
    context,
    return_tensors="pt",
    truncation=True,
    max_length=12000
).to("cuda")

def generate():

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

result = benchmark_generation(generate)

print("\n=== OPTIMIZED INFERENCE ===")
print(f"Generation Time: {result['time_seconds']:.2f} sec")
print(f"Peak VRAM Usage: {result['vram_mb']:.2f} MB")