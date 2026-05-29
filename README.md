# RAG + QLoRA Optimization Lab

**Institution:** ICEV
**Course:** Artificial Intelligence
**Professor:** Dimmy Magalhães
**Student:** Carlos Murilo Nogueira Portela

---

## Project Overview

This laboratory explores optimization techniques used in modern Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems. The objective is to analyze how memory-efficient methods can reduce GPU memory consumption and improve inference performance when processing long-context inputs.

The project evaluates and discusses the impact of three important optimization strategies:

* **QLoRA (Quantized Low-Rank Adaptation)**
* **KV Cache (Key-Value Cache)**
* **FlashAttention**

These techniques are widely adopted in the industry to make Transformer-based architectures more efficient during training and inference.

---

## Laboratory Objectives

The main goals of this laboratory are:

1. Understand the memory bottlenecks of Transformer architectures.
2. Evaluate optimization methods used in modern LLM systems.
3. Compare the effect of KV Cache on inference performance.
4. Analyze the role of quantization through QLoRA.
5. Investigate the impact of FlashAttention on memory efficiency.
6. Discuss the limitations of Transformers when processing extremely long contexts.

---

## Technologies Used

* Python 3.x
* PyTorch
* Hugging Face Transformers
* Accelerate
* BitsAndBytes
* TinyLlama 1.1B Chat
* CUDA (when available)

---

## Project Structure

```text
project/
│
├── data/
│   └── massive_context.txt
│
├── src/
│   ├── benchmark_no_cache.py
│   ├── benchmark_optimized.py
│   ├── config.py
│   └── utils.py
│
└── README.md
```

---

## Execution

Activate the virtual environment:

```bash
source .venv/bin/activate
```

or on Windows:

```powershell
.venv\Scripts\activate
```

Run the baseline benchmark:

```bash
python src/benchmark_no_cache.py
```

Run the optimized benchmark:

```bash
python src/benchmark_optimized.py
```

---

## Optimization Techniques Evaluated

### QLoRA

QLoRA reduces memory consumption by storing model weights in low-bit precision formats, typically 4-bit quantization, while maintaining acceptable model performance.

### KV Cache

KV Cache stores previously computed key and value tensors during autoregressive generation, preventing redundant calculations and accelerating inference.

### FlashAttention

FlashAttention optimizes the attention computation process by reducing memory transfers and improving GPU utilization, allowing larger contexts to be processed more efficiently.

---

## Technical Opinion

### 1 — How QLoRA, KV Cache, and FlashAttention Prevented VRAM Collapse

Traditional Transformer architectures suffer from high memory consumption because self-attention requires storing large intermediate matrices whose size grows quadratically with the input sequence length. In this laboratory, the combination of QLoRA, KV Cache, and FlashAttention significantly reduced this bottleneck. QLoRA enabled the model to operate with 4-bit quantized weights, drastically reducing VRAM requirements while preserving most of the model's capabilities. KV Cache avoided recomputing key and value tensors during autoregressive generation, reducing redundant operations and improving inference speed. FlashAttention further optimized the attention mechanism by minimizing memory transfers and computing attention blocks more efficiently. Together, these techniques allowed the Transformer architecture to process long contexts with substantially lower memory consumption and better performance than a traditional implementation.

### 2 — Why FlashAttention Would Not Be Enough for 2 Million Tokens

Although FlashAttention greatly improves memory efficiency, it does not eliminate the fundamental quadratic complexity of the self-attention mechanism. As sequence lengths increase to extreme scales, such as 2 million tokens, the amount of computation and memory required by attention operations becomes impractical even with state-of-the-art optimizations. In this scenario, the industry would likely need to adopt alternative architectures such as Mamba and other State Space Models (SSMs). Unlike Transformers, these architectures process sequences using recurrent state representations whose memory complexity is approximately O(1) with respect to sequence length. As a result, memory consumption remains stable even for extremely large contexts, making them more suitable for applications involving millions of tokens. Therefore, while FlashAttention extends the practical limits of Transformer-based models, architectures such as Mamba represent a more scalable long-term solution for ultra-long-context processing.

---

## Conclusion

This laboratory demonstrates that modern optimization techniques can significantly extend the practical limits of Transformer-based models. However, it also highlights that these optimizations address the symptoms rather than the underlying scalability limitations of self-attention. As context windows continue to grow, alternative architectures such as State Space Models may become increasingly important for the next generation of large-scale language models.

---

## AI Usage Disclosure

Portions of the source code, documentation, technical explanations, and project structure were generated with the assistance of Artificial Intelligence tools. All generated material was reviewed, validated, adapted, and revised by **Carlos Murilo Nogueira Portela**, who assumes full responsibility for the final submitted version of this project.
