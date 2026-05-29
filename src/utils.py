import torch
import time

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0

def benchmark_generation(generate_function):
    torch.cuda.reset_peak_memory_stats()

    start = time.time()

    output = generate_function()

    end = time.time()

    memory = get_vram_usage()

    return {
        "output": output,
        "time_seconds": end - start,
        "vram_mb": memory
    }