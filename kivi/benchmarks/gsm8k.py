"""
GSM8K math reasoning benchmark.

Covers dataset loading, prompt construction, generation, answer extraction,
exact-match evaluation, results table, and JSON saving.

Dependencies (install once):
    pip install datasets
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from utils import to_json_safe, token_level_match_rate
from kivi.memory import (
    get_model_kv_config,
    get_kivi_memory_stats,
    theoretical_baseline_kv_mb,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_gsm8k(split="test", num_samples=None):
    """
    Load the GSM8K dataset.

    Args:
        split:       'train' or 'test'
        num_samples: optional int to subsample

    Returns:
        HuggingFace dataset (or slice)
    """
    ds = load_dataset("gsm8k", "main", split=split)
    if num_samples is not None:
        ds = ds.select(range(num_samples))
    return ds


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_gsm8k_prompt(question):
    return (
        "Solve the following math problem step by step.\n"
        "Provide the final answer as a number.\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_final_answer(text):
    """
    Extract the final numeric answer from a GSM8K-style output string.
    Returns the last number found, or None if no number is present.
    """
    matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_gsm8k_answer(
    model,
    tokenizer,
    prompt,
    label,
    max_new_tokens=256,
    verbose=False,
):
    """
    Greedy generation for a single GSM8K prompt.
    Returns only the newly generated tokens (not the prompt).
    """
    if verbose:
        print(f"\n=========== {label} PROMPT ===========")
        print(prompt)
        print("=====================================")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if verbose:
        print(f"\n=========== {label} OUTPUT ===========")
        print(text)
        print("=====================================")

    return text


def generate_all_gsm8k_answers(
    model,
    tokenizer,
    dataset,
    label,
    max_new_tokens=256,
    num_print=3,
):
    """
    Run generation over the full GSM8K dataset split and collect:
      - generated outputs
      - prompt strings
      - ground-truth answers (final numeric)
      - per-prompt theoretical KV memory
      - full-run empirical peak GPU memory

    Returns:
        (outputs, prompts, gt_answers, memory_dict)
    """
    outputs = []
    prompts = []
    gt_answers = []
    kv_theoretical = []

    kv_cfg = get_model_kv_config(model)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for i, sample in enumerate(dataset):
        prompt = build_gsm8k_prompt(sample["question"])

        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        num_prompt_tokens = prompt_ids.shape[-1]

        text = generate_gsm8k_answer(
            model,
            tokenizer,
            prompt,
            label,
            max_new_tokens=max_new_tokens,
            verbose=(i < num_print),
        )

        gen_ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        num_gen_tokens = gen_ids.shape[-1]

        if label == "BASELINE":
            kv_stats = theoretical_baseline_kv_mb(
                num_prompt_tokens=num_prompt_tokens,
                num_generated_tokens=num_gen_tokens,
                **kv_cfg,
            )
        else:
            kv_stats = get_kivi_memory_stats(model)

        outputs.append(text)
        prompts.append(prompt)
        gt_answers.append(extract_final_answer(sample["answer"]))
        kv_theoretical.append(kv_stats)

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return outputs, prompts, gt_answers, {
        "per_prompt_kv": kv_theoretical,
        "peak_gpu_mb":   peak_gpu_mb,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_gsm8k(
    dataset,
    tokenizer,
    baseline_model,
    kivi_model,
    max_new_tokens=256,
    num_print=3,
):
    """
    Run both models on the GSM8K dataset simultaneously and compute
    exact-match accuracy, consistency, and token match rate.

    Returns:
        (per_sample: list[dict], aggregate: dict)
    """
    stats = {
        "baseline_correct":      [],
        "kivi_correct":          [],
        "consistency_rate":      [],
        "baseline_to_kivi_fail": [],
        "token_match":           [],
    }
    per_sample = []

    for i, sample in enumerate(dataset):
        verbose = i < num_print
        if verbose:
            print(f"\n================ Example {i} =================")

        prompt    = build_gsm8k_prompt(sample["question"])
        gt_answer = extract_final_answer(sample["answer"])

        baseline_text = generate_gsm8k_answer(
            baseline_model, tokenizer, prompt, "BASELINE", max_new_tokens, verbose
        )
        kivi_text = generate_gsm8k_answer(
            kivi_model, tokenizer, prompt, "KIVI", max_new_tokens, verbose
        )

        base_ans = extract_final_answer(baseline_text)
        kivi_ans = extract_final_answer(kivi_text)

        base_correct = int(base_ans == gt_answer)
        kivi_correct = int(kivi_ans == gt_answer)
        consistency  = int(base_correct == kivi_correct)
        btk_fail     = int(base_correct == 1 and kivi_correct == 0)

        token_stats = token_level_match_rate(baseline_text, kivi_text, tokenizer)

        per_sample.append({
            "idx":                  i,
            "baseline_correct":     base_correct,
            "kivi_correct":         kivi_correct,
            "token_match_rate":     token_stats["token_match_rate"],
            "baseline_to_kivi_fail":bool(base_correct and not kivi_correct),
            "both_correct":         bool(base_correct and kivi_correct),
        })

        stats["baseline_correct"].append(base_correct)
        stats["kivi_correct"].append(kivi_correct)
        stats["consistency_rate"].append(consistency)
        stats["baseline_to_kivi_fail"].append(btk_fail)
        stats["token_match"].append(token_stats["token_match_rate"])

        if verbose:
            print(f"\n=== GSM8K Example {i} ===")
            print(f"GT answer:        {gt_answer}")
            print(f"Baseline answer:  {base_ans} ({'correct' if base_correct else 'wrong'})")
            print(f"KIVI answer:      {kivi_ans} ({'correct' if kivi_correct else 'wrong'})")
            print(f"Token match rate: {token_stats['token_match_rate']:.3f}")

    return per_sample, {
        "baseline_EM":          np.mean(stats["baseline_correct"]),
        "kivi_EM":              np.mean(stats["kivi_correct"]),
        "consistency":          np.mean(stats["consistency_rate"]),
        "baseline_to_kivi_fail":np.mean(stats["baseline_to_kivi_fail"]),
        "token_match": {
            "mean": np.mean(stats["token_match"]),
            "std":  np.std(stats["token_match"]),
        },
    }


def evaluate_on_gsm8k_from_texts(
    prompts,
    gt_answers,
    baseline_outputs,
    kivi_outputs,
    tokenizer,
):
    """
    Compute GSM8K metrics from pre-generated text outputs (no model needed).

    Returns:
        (per_sample: list[dict], aggregate: dict)
    """
    stats = {
        "baseline_correct":      [],
        "kivi_correct":          [],
        "baseline_to_kivi_fail": [],
        "consistency":           [],
        "token_match":           [],
    }
    per_sample = []

    for i in range(len(prompts)):
        gt       = gt_answers[i]
        base_text = baseline_outputs[i]
        kivi_text = kivi_outputs[i]

        base_ans = extract_final_answer(base_text)
        kivi_ans = extract_final_answer(kivi_text)

        base_correct = int(base_ans == gt)
        kivi_correct = int(kivi_ans == gt)

        token_stats = token_level_match_rate(base_text, kivi_text, tokenizer)

        row = {
            "idx":            i,
            "prompt":         prompts[i],
            "ground_truth":   gt,
            "baseline_output": base_text,
            "kivi_output":    kivi_text,
            "scores": {
                "baseline_correct":      base_correct,
                "kivi_correct":          kivi_correct,
                "baseline_to_kivi_fail": int(base_correct and not kivi_correct),
                "consistency":           int(base_correct == kivi_correct),
                "token_match_rate":      token_stats["token_match_rate"],
            },
        }

        per_sample.append(row)

        stats["baseline_correct"].append(base_correct)
        stats["kivi_correct"].append(kivi_correct)
        stats["baseline_to_kivi_fail"].append(base_correct and not kivi_correct)
        stats["consistency"].append(base_correct == kivi_correct)
        stats["token_match"].append(token_stats["token_match_rate"])

    aggregate = {
        "baseline_EM":          np.mean(stats["baseline_correct"]),
        "kivi_EM":              np.mean(stats["kivi_correct"]),
        "consistency":          np.mean(stats["consistency"]),
        "baseline_to_kivi_fail":np.mean(stats["baseline_to_kivi_fail"]),
        "token_match": {
            "mean": np.mean(stats["token_match"]),
            "std":  np.std(stats["token_match"]),
        },
    }

    return per_sample, aggregate


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_gsm8k_aggregate_table(results):
    """Build a formatted results table dict from GSM8K aggregate metrics."""
    def f(x):
        return float(x)

    return {
        "Baseline EM":            f"{f(results['baseline_EM']):.2f}",
        "KIVI EM":                f"{f(results['kivi_EM']):.2f}",
        "Consistency rate":       f"{f(results['consistency']):.2f}",
        "Baseline -> KIVI failure":f"{f(results['baseline_to_kivi_fail']):.2f}",
        "Token match rate": (
            f"{f(results['token_match']['mean']):.2f} +/- "
            f"{f(results['token_match']['std']):.2f}"
        ),
    }


def print_gsm8k_aggregate_table(table):
    """Pretty-print the formatted GSM8K results table."""
    print("\nGSM8K AGGREGATE RESULTS")
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 47)

    for metric, value in table.items():
        print(f"{metric:<30} {value:>15}")


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_gsm8k_results_json(
    per_sample,
    aggregate,
    memory_results,
    out_dir="results",
    prefix="gsm8k",
):
    """
    Save GSM8K per-sample results, aggregate metrics, and memory results
    to three JSON files under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    examples_path = f"{out_dir}/examples_{prefix}.json"
    results_path  = f"{out_dir}/results_{prefix}.json"
    memory_path   = f"{out_dir}/memory_{prefix}.json"

    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(per_sample), f, indent=2, ensure_ascii=False)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(aggregate), f, indent=2, ensure_ascii=False)

    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(memory_results), f, indent=2, ensure_ascii=False)

    print("Saved GSM8K results:")
    print(" -", examples_path)
    print(" -", results_path)
    print(" -", memory_path)
