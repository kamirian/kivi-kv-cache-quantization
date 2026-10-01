"""
Memory profiling utilities for baseline (FP16) and KIVI KV caches.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Baseline KV memory helpers
# ---------------------------------------------------------------------------

def generate_baseline_with_output_and_kv(
    model,
    tokenizer,
    prompt,
    max_new_tokens=40,
):
    """
    Run autoregressive generation with the baseline model, capturing:
      - generated text (prompt tokens excluded)
      - past_key_values (HuggingFace DynamicCache or tuple of (k, v) tuples)

    Returns:
        (text: str, past_key_values)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    past_key_values = None
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            input_ids = next_token

    continuation_ids = generated_ids[:, prompt_len:]
    text = tokenizer.decode(continuation_ids[0], skip_special_tokens=True)

    return text, past_key_values


def get_baseline_kv_memory_stats(past_key_values, dtype_bytes=2):
    """
    Compute memory usage of a baseline FP16 KV cache returned by HuggingFace.

    Args:
        past_key_values: tuple of (k, v) pairs from HF model generation
        dtype_bytes: bytes per element (fp16 = 2)

    Returns:
        dict with key_bytes, value_bytes, total_bytes, total_mb, total_gb
    """
    stats = {
        "key_bytes": 0,
        "value_bytes": 0,
        "total_bytes": 0,
    }

    for k, v in past_key_values:
        # k, v shape: [batch, num_heads, seq_len, head_dim]
        stats["key_bytes"]   += k.numel() * dtype_bytes
        stats["value_bytes"] += v.numel() * dtype_bytes

    stats["total_bytes"] = stats["key_bytes"] + stats["value_bytes"]
    stats["total_mb"]    = stats["total_bytes"] / (1024 ** 2)
    stats["total_gb"]    = stats["total_bytes"] / (1024 ** 3)

    return stats


def theoretical_baseline_kv_mb(
    num_prompt_tokens: int,
    num_generated_tokens: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # fp16
):
    """
    Compute theoretical FP16 KV-cache memory for baseline attention.

    KV shape per layer:
        [num_heads, total_tokens, head_dim] for K and V

    Returns:
        dict with total_tokens, total_bytes, total_mb, total_gb
    """
    total_tokens = num_prompt_tokens + num_generated_tokens

    total_bytes = (
        total_tokens
        * num_layers
        * num_heads
        * head_dim
        * dtype_bytes
        * 2  # K + V
    )

    return {
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
        "total_gb": total_bytes / (1024 ** 3),
    }


# ---------------------------------------------------------------------------
# KIVI memory helpers
# ---------------------------------------------------------------------------

def get_kivi_memory_stats(model, verbose=False):
    """
    Compute total theoretical memory usage across ALL KIVI caches in the model.

    Args:
        model: LlamaForCausalLM with KIVI attention layers
        verbose: If True, print per-layer/per-head stats

    Returns:
        dict with aggregated byte counts + total_mb + total_gb
    """
    totals = {
        "total_bytes": 0,
        "key_residual_bytes": 0,
        "value_residual_bytes": 0,
        "key_quant_bytes": 0,
        "value_quant_bytes": 0,
        "key_metadata_bytes": 0,
        "value_metadata_bytes": 0,
        "layers_with_kivi": 0,
        "heads_per_layer": [],
    }

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not hasattr(attn, "kivi_cache") or attn.kivi_cache is None:
            continue

        totals["layers_with_kivi"] += 1
        totals["heads_per_layer"].append(len(attn.kivi_cache))

        for head_idx, cache in enumerate(attn.kivi_cache):
            stats = cache.get_memory_stats()

            totals["total_bytes"]          += stats["total_bytes"]
            totals["key_residual_bytes"]   += stats["key_residual_bytes"]
            totals["value_residual_bytes"] += stats["value_residual_bytes"]
            totals["key_quant_bytes"]      += stats["key_quant_bytes"]
            totals["value_quant_bytes"]    += stats["value_quant_bytes"]
            totals["key_metadata_bytes"]   += stats["key_metadata_bytes"]
            totals["value_metadata_bytes"] += stats["value_metadata_bytes"]

            if verbose:
                print(f"[Layer {layer_idx}][Head {head_idx}] {stats}")

    totals["total_mb"] = totals["total_bytes"] / (1024 ** 2)
    totals["total_gb"] = totals["total_bytes"] / (1024 ** 3)

    return totals


def get_model_kv_config(model):
    """
    Extract KV-cache configuration from a model's config object.

    Returns:
        dict with num_layers, num_heads, head_dim
    """
    cfg = model.config
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_heads": cfg.num_attention_heads,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
    }


# ---------------------------------------------------------------------------
# measure_baseline_memory  (GPU peak + theoretical KV)
# ---------------------------------------------------------------------------

def measure_baseline_memory(
    model,
    tokenizer,
    prompt,
    max_new_tokens=40,
):
    """
    Measure GPU peak memory and theoretical KV-cache size for a baseline
    (unquantized) forward pass.

    Returns:
        dict with output text, gpu_peak_mb, kv_theoretical_mb, kv_stats
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    text, past_kv = generate_baseline_with_output_and_kv(
        model, tokenizer, prompt, max_new_tokens
    )

    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    kv_stats = get_baseline_kv_memory_stats(past_kv)

    return {
        "output": text,
        "gpu_peak_mb": gpu_peak_mb,
        "kv_theoretical_mb": kv_stats["total_mb"],
        "kv_stats": kv_stats,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_per_prompt_kv(memory_results):
    """
    Print a table of per-prompt theoretical KV cache sizes (MB)
    for baseline and KIVI.
    """
    baseline = memory_results["baseline"]["per_prompt_kv"]
    kivi     = memory_results["kivi"]["per_prompt_kv"]

    print("\nPer-Prompt Theoretical KV Cache Memory")
    print("=" * 70)
    print(f"{'Prompt':<8} {'Baseline KV (MB)':>20} {'KIVI KV (MB)':>20}")
    print("-" * 70)

    for i in range(len(baseline)):
        b_mb = baseline[i]["total_mb"]
        k_mb = kivi[i]["total_mb"]
        print(f"{i:<8} {b_mb:>20.2f} {k_mb:>20.2f}")


def print_aggregate_kv(memory_results):
    """
    Print aggregate (mean/max/min) theoretical KV cache sizes and
    compression ratio.
    """
    def summarize(per_prompt):
        mbs = [x["total_mb"] for x in per_prompt]
        return {
            "mean": np.mean(mbs),
            "max":  np.max(mbs),
            "min":  np.min(mbs),
        }

    base = summarize(memory_results["baseline"]["per_prompt_kv"])
    kivi = summarize(memory_results["kivi"]["per_prompt_kv"])

    compression = base["mean"] / kivi["mean"]

    print("\nAggregate Theoretical KV Cache (MB)")
    print("=" * 70)
    print(f"{'':<20} {'Baseline':>15} {'KIVI':>15}")
    print("-" * 70)
    print(f"{'Mean KV':<20} {base['mean']:>15.2f} {kivi['mean']:>15.2f}")
    print(f"{'Max KV':<20} {base['max']:>15.2f}  {kivi['max']:>15.2f}")
    print(f"{'Min KV':<20} {base['min']:>15.2f}  {kivi['min']:>15.2f}")
    print("-" * 70)
    print(f"Compression ratio (Baseline / KIVI): {compression:.2f}x")


def print_full_run_gpu_memory(memory_results):
    """Print empirical full-run peak GPU memory for baseline and KIVI."""
    print("\nFull-Run Peak GPU Memory (empirical)")
    print("=" * 70)
    print(f"Baseline peak GPU memory: {memory_results['baseline']['peak_gpu_mb']:.2f} MB")
    print(f"KIVI peak GPU memory:     {memory_results['kivi']['peak_gpu_mb']:.2f} MB")


def print_all_memory_results(memory_results):
    """Print per-prompt KV, aggregate KV, and full-run GPU memory."""
    print_per_prompt_kv(memory_results)
    print_aggregate_kv(memory_results)
    print_full_run_gpu_memory(memory_results)


def print_kivi_memory_stats(stats):
    """Pretty-print the output of get_kivi_memory_stats."""
    print("=" * 70)
    print("KIVI Cache Memory Statistics")
    print("=" * 70)
    print(f"Total memory:          {stats['total_mb']:.2f} MB ({stats['total_gb']:.4f} GB)")
    print(f"Layers with KIVI:      {stats['layers_with_kivi']}")
    print(f"Heads per layer:       {stats['heads_per_layer']}")
    print("-" * 70)
    print(f"Key residual:          {stats['key_residual_bytes'] / (1024**2):.2f} MB")
    print(f"Value residual:        {stats['value_residual_bytes'] / (1024**2):.2f} MB")
    print(f"Key quantized:         {stats['key_quant_bytes'] / (1024**2):.2f} MB")
    print(f"Value quantized:       {stats['value_quant_bytes'] / (1024**2):.2f} MB")
    print(f"Key metadata:          {stats['key_metadata_bytes'] / (1024**2):.2f} MB")
    print(f"Value metadata:        {stats['value_metadata_bytes'] / (1024**2):.2f} MB")
    print("=" * 70)
