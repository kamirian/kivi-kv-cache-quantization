# KIVI: 2-Bit KV-Cache Quantization for LLaMA (7B & 13B)

Implements and evaluates **KIVI** — a training-free, 2-bit KV-cache quantization scheme applied to LLaMA-2 7B and 13B.

**Key idea:** Replace full-precision (FP16) attention key-value tensors with 2-bit group-quantized representations, keeping a small residual buffer of recent full-precision tokens. No retraining required.

This was my individual contribution to a team course project on KV-cache efficiency methods (CMSC 723, Graduate NLP). Team: Saketh Akella, Kiyan Amirian, Nicholas Forman, Helia Hosseini, Hengyuan Qi. The full team report, which also covers H2O, Streaming-LLM, ZipCache, and StreamingSliding, is included as `report.pdf`.

---

## Implementation

- `quantize_per_token` / `quantize_per_channel` — 2-bit asymmetric quantization with configurable group size
- `KIVICache` — quantized KV-cache manager with residual buffer and on-the-fly dequantization
- `LlamaAttentionWithKIVI` — drop-in replacement for LLaMA self-attention
- `replace_llama_attention_with_kivi` — applies KIVI to every attention layer in a loaded model

## Benchmarks evaluated

| Benchmark | Model | Metric |
|-----------|-------|--------|
| CNN/DailyMail | LLaMA-2 7B | ROUGE-L, BERTScore, token match rate |
| GSM8K | LLaMA-2 13B | Exact match accuracy |
| CoQA | LLaMA-2 7B | F1 (raw and robust), ROUGE-L, BERTScore |

Memory analysis includes theoretical KV-cache reduction (~8× for 2-bit vs FP16) and empirical peak GPU memory profiling.

Pre-computed results are stored in `results/`.

---

## Repository structure

```
kivi-kv-cache-quantization/
├── kivi/
│   ├── quantization.py, cache.py, attention.py, memory.py
│   └── benchmarks/ (cnn_dm.py, gsm8k.py, coqa.py)
├── utils.py                  JSON helper and token-match metric
├── kivi_llama_7b_13b.ipynb   unit tests, memory profiling, and the three benchmarks
├── results/                  {examples|memory|results}_{cnn|coqa|gsm8k_13b}_Kiyan.json
├── report.pdf                team report
└── requirements.txt
```

## Running

Requires a CUDA GPU and a Hugging Face token with access to the LLaMA-2 weights.

```bash
pip install -r requirements.txt
```

Then open `kivi_llama_7b_13b.ipynb` in Jupyter from the repository root, so it can import `kivi` and `utils`.
