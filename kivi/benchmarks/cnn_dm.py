"""
CNN/DailyMail summarization benchmark.

Covers dataset loading, prompt construction, generation, full evaluation
(ROUGE, BERTScore, token match, sentence compliance, entity hallucination),
results table construction, and JSON saving.

Dependencies (install once):
    pip install rouge-score bert-score datasets
"""

import json
import os
import re
import math
from collections import defaultdict

import numpy as np
import torch

from rouge_score import rouge_scorer as _rouge_scorer
from bert_score import score as _bertscore
from datasets import load_dataset
from transformers import pipeline

from utils import to_json_safe, token_level_match_rate
from kivi.memory import (
    get_model_kv_config,
    get_kivi_memory_stats,
    theoretical_baseline_kv_mb,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_cnn_dm(split="validation", num_samples=10):
    """Load the CNN/DailyMail dataset and return the first num_samples examples."""
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
    return dataset.select(range(num_samples))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_summarization_prompt(tokenizer, article):
    """
    Build a chat-templated summarization prompt for LLaMA-2 chat models.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that writes concise summaries.",
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following news article in 3 sentences.\n\n"
                f"{article}\n\nSUMMARY:"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(
    model,
    tokenizer,
    prompt,
    label,
    max_new_tokens=40,
    verbose=False,
):
    """
    Generate text with greedy decoding; returns only the newly generated tokens
    (not the prompt).
    """
    if verbose:
        print("\n" + "=" * 50)
        print(f"[{label}] Prompt: {prompt}")
        print("=" * 50)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if verbose:
        print(f"[{label}] Generated continuation:\n{text}\n")

    return text


@torch.no_grad()
def generate_all_summaries(
    model,
    tokenizer,
    dataset,
    label,
    max_new_tokens=256,
    num_print=3,
):
    """
    Run generation over an entire dataset split, collecting:
      - generated texts
      - prompt strings
      - reference highlights
      - per-prompt theoretical KV memory stats
      - empirical per-sample peak GPU memory

    Returns:
        (outputs, prompts, references, memory_dict)
    """
    outputs = []
    prompts = []
    references = []
    kv_theoretical = []
    peak_gpu_mb_per_sample = []

    kv_cfg = get_model_kv_config(model)

    for i, sample in enumerate(dataset):
        article = sample["article"]
        prompt = build_summarization_prompt(tokenizer, article)

        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        num_prompt_tokens = prompt_ids.shape[-1]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        text = generate_text(
            model,
            tokenizer,
            prompt,
            label,
            max_new_tokens=max_new_tokens,
            verbose=(i < num_print),
        )

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_gpu_mb_per_sample.append(peak_mb)

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
        references.append(sample["highlights"])
        kv_theoretical.append(kv_stats)

        if i < num_print:
            print(
                f"[{label}] Example {i} | "
                f"KV theo: {kv_stats['total_mb']:.2f} MB | "
                f"Empirical peak GPU: {peak_mb:.2f} MB"
            )

        del text, prompt_ids, gen_ids
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return outputs, prompts, references, {
        "per_prompt_kv": kv_theoretical,
        "peak_gpu_mb_per_sample": peak_gpu_mb_per_sample,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rouge_l_score(reference, prediction):
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def bert_score_f1(
    reference_text,
    prediction_text,
    model_type="roberta-large",
    device="cpu",
):
    """Compute BERTScore F1 between reference and prediction."""
    if device is None:
        device = "cpu"
    P, R, F1 = _bertscore(
        [prediction_text],
        [reference_text],
        lang="en",
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return F1.mean().item()


# ---- Sentence compliance ----

def _split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def sentence_compliance_score(generated: str, requested: int = 3, max_penalty: int = 3) -> float:
    n_gen = len(_split_sentences(generated))
    diff = abs(n_gen - requested)
    return max(0.0, 1.0 - diff / max_penalty)


# ---- Entity hallucination rate ----

def _normalize_for_match(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower()).strip()


def _extract_summary_entities(summary: str):
    ents = set()
    for m in re.findall(r'\b\d+(?:[\-–]\d+)?\b', summary):
        ents.add(m)
    cap_spans = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\b', summary)
    for s in cap_spans:
        if s not in {"The", "A", "An", "This", "That", "In", "On", "But", "And", "Some"}:
            ents.add(s.strip())
    return ents


def entity_hallucination_rate(article: str, summary: str):
    """
    Fraction of extracted entities in the summary that do NOT appear in the article.
    Returns: (rate, hallucinated_list, matched_list)
    """
    art = _normalize_for_match(article)
    ents = _extract_summary_entities(summary)
    if len(ents) == 0:
        return 0.0, [], []

    hallucinated = []
    matched = []
    for e in ents:
        if _normalize_for_match(e) in art:
            matched.append(e)
        else:
            hallucinated.append(e)

    rate = len(hallucinated) / max(1, len(ents))
    return rate, hallucinated, matched


# ---------------------------------------------------------------------------
# Full evaluation (from pre-generated texts)
# ---------------------------------------------------------------------------

def evaluate_on_cnn_dm_from_texts(
    articles,
    references,
    prompts,
    baseline_outputs,
    kivi_outputs,
    tokenizer,
):
    """
    Compute all metrics comparing baseline and KIVI outputs against ground truth
    and against each other.

    Returns:
        (per_sample: list[dict], aggregate_results: dict)
    """
    metrics = defaultdict(list)
    per_sample = []

    for i in range(len(articles)):
        article        = articles[i]
        reference      = references[i]
        baseline_text  = baseline_outputs[i]
        kivi_text      = kivi_outputs[i]

        token_stats = token_level_match_rate(baseline_text, kivi_text, tokenizer)

        rouge_bk = rouge_l_score(baseline_text, kivi_text)
        bert_bk  = bert_score_f1(baseline_text, kivi_text)

        rouge_bg = rouge_l_score(reference, baseline_text)
        bert_bg  = bert_score_f1(reference, baseline_text)

        rouge_kg = rouge_l_score(reference, kivi_text)
        bert_kg  = bert_score_f1(reference, kivi_text)

        sce_b = sentence_compliance_score(baseline_text, 3)
        sce_k = sentence_compliance_score(kivi_text, 3)

        hall_b, _, _ = entity_hallucination_rate(article, baseline_text)
        hall_k, _, _ = entity_hallucination_rate(article, kivi_text)

        row = {
            "idx": i,
            "prompt": prompts[i],
            "article": article[:2000],
            "reference": reference,
            "baseline_output": baseline_text,
            "kivi_output": kivi_text,
            "scores": {
                "baseline_vs_gt": {
                    "rouge_l": rouge_bg,
                    "bert": bert_bg,
                    "sce": sce_b,
                    "scr": int(sce_b == 0),
                    "ent_hall_rate": hall_b,
                },
                "kivi_vs_gt": {
                    "rouge_l": rouge_kg,
                    "bert": bert_kg,
                    "sce": sce_k,
                    "scr": int(sce_k == 0),
                    "ent_hall_rate": hall_k,
                },
                "baseline_vs_kivi": {
                    "token_match_rate": token_stats["token_match_rate"],
                    "rouge_l": rouge_bk,
                    "bert": bert_bk,
                },
            },
        }

        per_sample.append(row)

        metrics["rouge_bg"].append(rouge_bg)
        metrics["bert_bg"].append(bert_bg)
        metrics["rouge_kg"].append(rouge_kg)
        metrics["bert_kg"].append(bert_kg)
        metrics["rouge_bk"].append(rouge_bk)
        metrics["bert_bk"].append(bert_bk)
        metrics["token_bk"].append(token_stats["token_match_rate"])
        metrics["sce_baseline"].append(sce_b)
        metrics["sce_kivi"].append(sce_k)
        metrics["scr_baseline"].append(int(sce_b == 0))
        metrics["scr_kivi"].append(int(sce_k == 0))
        metrics["ent_hall_rate_baseline"].append(hall_b)
        metrics["ent_hall_rate_kivi"].append(hall_k)

    def summarize(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x))}

    results = {
        "rouge_bg": summarize(metrics["rouge_bg"]),
        "bert_bg":  summarize(metrics["bert_bg"]),
        "rouge_kg": summarize(metrics["rouge_kg"]),
        "bert_kg":  summarize(metrics["bert_kg"]),
        "rouge_bk": summarize(metrics["rouge_bk"]),
        "bert_bk":  summarize(metrics["bert_bk"]),
        "token_bk": summarize(metrics["token_bk"]),
        "sce_baseline": summarize(metrics["sce_baseline"]),
        "sce_kivi":     summarize(metrics["sce_kivi"]),
        "scr_baseline": float(np.mean(metrics["scr_baseline"])),
        "scr_kivi":     float(np.mean(metrics["scr_kivi"])),
        "ent_hall_rate_baseline": summarize(metrics["ent_hall_rate_baseline"]),
        "ent_hall_rate_kivi":     summarize(metrics["ent_hall_rate_kivi"]),
    }

    return per_sample, results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_results_table(results, decimals=4):
    """Build a formatted results table dict from the aggregate results dict."""
    def fmt(x):
        if x == "—":
            return "—"
        if isinstance(x, dict):
            return f"{x['mean']:.{decimals}f} ± {x['std']:.{decimals}f}"
        if isinstance(x, float):
            return f"{x:.{decimals}f}"
        return str(x)

    table = {
        "ROUGE-L F1": {
            "Baseline vs GT":   results["rouge_bg"],
            "KIVI vs GT":       results["rouge_kg"],
            "Baseline vs KIVI": results["rouge_bk"],
        },
        "BERTScore F1": {
            "Baseline vs GT":   results["bert_bg"],
            "KIVI vs GT":       results["bert_kg"],
            "Baseline vs KIVI": results["bert_bk"],
        },
        "Token match rate": {
            "Baseline vs GT":   "—",
            "KIVI vs GT":       "—",
            "Baseline vs KIVI": results["token_bk"],
        },
        "Sentence error (SCE)": {
            "Baseline vs GT":   results["sce_baseline"],
            "KIVI vs GT":       results["sce_kivi"],
            "Baseline vs KIVI": "—",
        },
        "Sentence compliance (strict)": {
            "Baseline vs GT":   results["scr_baseline"],
            "KIVI vs GT":       results["scr_kivi"],
            "Baseline vs KIVI": "—",
        },
        "Entity hallucination rate": {
            "Baseline vs GT":   results["ent_hall_rate_baseline"],
            "KIVI vs GT":       results["ent_hall_rate_kivi"],
            "Baseline vs KIVI": "—",
        },
    }

    return {metric: {k: fmt(v) for k, v in cols.items()} for metric, cols in table.items()}


def print_results_table(table):
    """Pretty-print the formatted CNN/DM results table."""
    header = (
        "Metric".ljust(30)
        + "Baseline vs GT".rjust(22)
        + "KIVI vs GT".rjust(22)
        + "Baseline vs KIVI".rjust(24)
    )
    print("\n" + header)
    print("-" * len(header))

    for metric, cols in table.items():
        print(
            metric.ljust(30)
            + cols["Baseline vs GT"].rjust(22)
            + cols["KIVI vs GT"].rjust(22)
            + cols["Baseline vs KIVI"].rjust(24)
        )


# ---------------------------------------------------------------------------
# Memory transform
# ---------------------------------------------------------------------------

def transform_memory_results_cnn(memory_results_cnn):
    """
    Transform the in-memory memory_results_cnn dict into an explicit
    prompt-indexed KV schema suitable for JSON serialisation.
    """
    baseline_kv = memory_results_cnn["baseline"]["per_prompt_kv"]
    kivi_kv     = memory_results_cnn["kivi"]["per_prompt_kv"]

    assert len(baseline_kv) == len(kivi_kv), \
        "Mismatch: baseline and kivi prompt counts differ"

    prompts = []
    for prompt_id, (b, k) in enumerate(zip(baseline_kv, kivi_kv)):
        entry = {
            "prompt_id": prompt_id,
            "tokens": {"total": b.get("total_tokens", None)},
            "baseline": {"kv_mb": float(b["total_mb"])},
            "kivi":     {"kv_mb": float(k["total_mb"])},
        }

        if "key_quant_bytes" in k:
            entry["kivi"]["breakdown"] = {
                "key_quant_mb":     k.get("key_quant_bytes",     0) / (1024 ** 2),
                "value_quant_mb":   k.get("value_quant_bytes",   0) / (1024 ** 2),
                "key_residual_mb":  k.get("key_residual_bytes",  0) / (1024 ** 2),
                "value_residual_mb":k.get("value_residual_bytes",0) / (1024 ** 2),
                "key_metadata_mb":  k.get("key_metadata_bytes",  0) / (1024 ** 2),
                "value_metadata_mb":k.get("value_metadata_bytes",0) / (1024 ** 2),
            }

        prompts.append(entry)

    baseline_mbs = [p["baseline"]["kv_mb"] for p in prompts]
    kivi_mbs     = [p["kivi"]["kv_mb"]     for p in prompts]

    return {
        "metadata": {
            "measurement": "terminal_per_prompt_kv",
            "kv_unit": "MB",
            "dataset": "CNN/DailyMail",
            "notes": (
                "Each entry corresponds to KV cache memory after "
                "full autoregressive generation for one prompt."
            ),
        },
        "prompts": prompts,
        "run_summary": {
            "baseline_peak_gpu_mb": memory_results_cnn["baseline"].get("peak_gpu_mb"),
            "kivi_peak_gpu_mb":     memory_results_cnn["kivi"].get("peak_gpu_mb"),
            "mean_kv_mb": {
                "baseline": float(np.mean(baseline_mbs)),
                "kivi":     float(np.mean(kivi_mbs)),
            },
            "compression_ratio_mean": float(
                np.mean(baseline_mbs) / np.mean(kivi_mbs)
            ),
        },
    }


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_cnn_results(
    examples,
    results,
    memory_results,
    out_dir="cnn_results",
):
    """
    Save CNN/DM per-example records, aggregate results, and memory results
    to three JSON files under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    examples_path = f"{out_dir}/examples_cnn_100.json"
    results_path  = f"{out_dir}/results_cnn_100.json"
    memory_path   = f"{out_dir}/memory_results_cnn_100.json"

    with open(examples_path, "w") as f:
        json.dump(to_json_safe(examples), f, indent=2)

    with open(results_path, "w") as f:
        json.dump(to_json_safe(results), f, indent=2)

    with open(memory_path, "w") as f:
        json.dump(to_json_safe(memory_results), f, indent=2)

    print("Saved CNN results:")
    print(" -", examples_path)
    print(" -", results_path)
    print(" -", memory_path)
