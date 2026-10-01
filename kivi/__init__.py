"""
kivi — 2-bit KV-cache quantization for LLaMA.

Key symbols re-exported for convenience:
    quantize_per_token, quantize_per_channel, dequantize  (quantization.py)
    KIVICache                                              (cache.py)
    LlamaAttentionWithKIVI, replace_llama_attention_with_kivi (attention.py)
    get_kivi_memory_stats, get_model_kv_config,
    measure_baseline_memory, print_all_memory_results     (memory.py)
"""

from .quantization import quantize_per_token, quantize_per_channel, dequantize
from .cache import KIVICache
from .attention import (
    LlamaAttentionWithKIVI,
    replace_llama_attention_with_kivi,
    reset_all_kivi_caches,
)
from .memory import (
    generate_baseline_with_output_and_kv,
    get_baseline_kv_memory_stats,
    theoretical_baseline_kv_mb,
    get_kivi_memory_stats,
    get_model_kv_config,
    measure_baseline_memory,
    print_per_prompt_kv,
    print_aggregate_kv,
    print_full_run_gpu_memory,
    print_all_memory_results,
    print_kivi_memory_stats,
)

__all__ = [
    # quantization
    "quantize_per_token",
    "quantize_per_channel",
    "dequantize",
    # cache
    "KIVICache",
    # attention
    "LlamaAttentionWithKIVI",
    "replace_llama_attention_with_kivi",
    "reset_all_kivi_caches",
    # memory
    "generate_baseline_with_output_and_kv",
    "get_baseline_kv_memory_stats",
    "theoretical_baseline_kv_mb",
    "get_kivi_memory_stats",
    "get_model_kv_config",
    "measure_baseline_memory",
    "print_per_prompt_kv",
    "print_aggregate_kv",
    "print_full_run_gpu_memory",
    "print_all_memory_results",
    "print_kivi_memory_stats",
]
