import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"