"""
CoQA conversational QA benchmark.

Covers dataset loading, prompt construction, generation, CoQA F1 scoring,
full evaluation with all metrics, results table, and saving.

Dependencies (install once):
    pip install rouge-score bert-score datasets
"""

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer as _rouge_scorer
from bert_score import score as _bertscore

from utils import to_json_safe, token_level_match_rate
from kivi.memory import (
    get_model_kv_config,
    get_kivi_memory_stats,
    theoretical_baseline_kv_mb,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_coqa(n=200, split="validation"):
    """
    Load CoQA dataset and flatten into a list of (context, question, answer) dicts.
    Returns at most n examples.
    """
    ds = load_dataset("coqa", split=split)
    data = []

    for ex in ds:
        context = ex["story"]
        for i in range(len(ex["questions"])):
            data.append({
                "context":  context,
                "question": ex["questions"][i],
                "answer":   ex["answers"]["input_text"][i],
            })
            if len(data) >= n:
                return data

    return data


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_coqa_prompt(context, question):
    return (
        "<s>[INST]\n"
        "Read the story and answer the question using a short phrase.\n\n"
        f"Story:\n{context}\n\n"
        f"Question:\n{question}\n"
        "[/INST]"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_coqa(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy generation; returns only the newly generated text (no prompt)."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CoQA F1 scoring
# ---------------------------------------------------------------------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def coqa_f1(pred, ref):
    """Token-level F1 between predicted and reference answer strings."""
    pred = normalize(pred)
    ref  = normalize(ref)

    if pred == "" and ref == "":
        return 1.0
    if pred == "" or ref == "":
        return 0.0

    pred_tokens = pred.split()
    ref_tokens  = ref.split()

    common  = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _strip_answer_prefix(text):
    patterns = [
        r"^the answer to the question is[:\s]*",
        r"^answer[:\s]*",
    ]
    text = text.lower().strip()
    for p in patterns:
        text = re.sub(p, "", text)
    return text.strip()


def _canonicalize_yes_no(text):
    text = text.lower()
    if re.search(r"\b(yes|yeah|yep)\b", text):
        return "yes"
    if re.search(r"\b(no|not|didn't|did not)\b", text):
        return "no"
    return text


def coqa_f1_robust(pred, ref):
    """CoQA F1 with answer-prefix stripping and yes/no canonicalization."""
    pred = _strip_answer_prefix(pred)
    pred = _canonicalize_yes_no(pred)
    ref  = _canonicalize_yes_no(ref)
    return coqa_f1(pred, ref)


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
    P, R, F1 = _bertscore(
        [prediction_text],
        [reference_text],
        lang="en",
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return F1.mean().item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_model(
    model,
    tokenizer,
    data,
    debug_n=3,
    max_new_tokens=32,
):
    """
    Run a single model over all CoQA examples and return mean F1.
    Prints debug output for the first debug_n examples.
    """
    f1_scores = []

    for idx, ex in enumerate(data):
        prompt = build_coqa_prompt(ex["context"], ex["question"])
        output = generate_coqa(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        score  = coqa_f1(output, ex["answer"])
        f1_scores.append(score)

        if idx < debug_n:
            print("=" * 80)
            print(f"Example {idx}")
            print("- Prompt:")
            print(prompt)
            print("- Ground Truth:", ex["answer"])
            print("- Model Output:", output)
            print(f"- CoQA F1: {score:.3f}")

    return sum(f1_scores) / len(f1_scores)


def evaluate_coqa(baseline, kivi, tokenizer, data, debug_n=3):
    """
    Run both baseline and KIVI models over CoQA data.
    Returns (mean_baseline_f1, mean_kivi_f1).
    """
    baseline_f1 = []
    kivi_f1 = []

    for idx, ex in enumerate(data):
        prompt = build_coqa_prompt(ex["context"], ex["question"])

        base_out = generate_coqa(baseline, tokenizer, prompt)
        kivi_out = generate_coqa(kivi,     tokenizer, prompt)

        base_score = coqa_f1(base_out, ex["answer"])
        kivi_score = coqa_f1(kivi_out, ex["answer"])

        baseline_f1.append(base_score)
        kivi_f1.append(kivi_score)

        if idx < debug_n:
            print("=" * 80)
            print(f"Example {idx}")
            print("- Prompt:")
            print(prompt)
            print("- GT Answer:",  ex["answer"])
            print("- Baseline:",   base_out)
            print("- KIVI:",       kivi_out)
            print(f"- F1 (Baseline / KIVI): {base_score:.3f} / {kivi_score:.3f}")

    return sum(baseline_f1) / len(baseline_f1), sum(kivi_f1) / len(kivi_f1)


def evaluate_all_metrics(
    baseline_outputs,
    kivi_outputs,
    ground_truths,
    tokenizer,
    device="cpu",
):
    """
    Compute all CoQA metrics (F1, robust F1, ROUGE-L, BERTScore, token match)
    for baseline and KIVI outputs vs. ground truth and vs. each other.

    Returns:
        dict with 'aggregate' (mean scores) and 'per_example' (per-example dicts)
    """
    per_example = []
    agg = defaultdict(list)

    for base_out, kivi_out, gt in zip(baseline_outputs, kivi_outputs, ground_truths):
        ex_scores = {
            "baseline_vs_gt": {
                "f1_raw":    coqa_f1(base_out, gt),
                "f1_robust": coqa_f1_robust(base_out, gt),
                "rouge_l":   rouge_l_score(gt, base_out),
                "bert":      bert_score_f1(gt, base_out, device=device),
            },
            "kivi_vs_gt": {
                "f1_raw":    coqa_f1(kivi_out, gt),
                "f1_robust": coqa_f1_robust(kivi_out, gt),
                "rouge_l":   rouge_l_score(gt, kivi_out),
                "bert":      bert_score_f1(gt, kivi_out, device=device),
            },
            "baseline_vs_kivi": {
                "token_match": token_level_match_rate(base_out, kivi_out, tokenizer),
                "rouge_l":     rouge_l_score(base_out, kivi_out),
                "bert":        bert_score_f1(base_out, kivi_out, device=device),
            },
        }

        for side in ["baseline_vs_gt", "kivi_vs_gt"]:
            agg[f"{side}_f1_raw"].append(ex_scores[side]["f1_raw"])
            agg[f"{side}_f1_robust"].append(ex_scores[side]["f1_robust"])
            agg[f"{side}_rouge"].append(ex_scores[side]["rouge_l"])
            agg[f"{side}_bert"].append(ex_scores[side]["bert"])

        agg["bk_token_match"].append(ex_scores["baseline_vs_kivi"]["token_match"])
        agg["bk_rouge"].append(ex_scores["baseline_vs_kivi"]["rouge_l"])
        agg["bk_bert"].append(ex_scores["baseline_vs_kivi"]["bert"])

        per_example.append(ex_scores)

    aggregate = {k: sum(v) / len(v) for k, v in agg.items()}

    return {
        "aggregate": aggregate,
        "per_example": per_example,
    }


@torch.no_grad()
def run_model_and_collect_outputs_with_kv(
    model,
    tokenizer,
    data,
    label,
    max_new_tokens=32,
    debug_n=3,
):
    """
    Run one model (baseline or KIVI) over the CoQA data, tracking:
      - generated outputs
      - ground truths
      - per-prompt theoretical KV memory
      - per-prompt empirical GPU peak memory

    Returns:
        (outputs, ground_truths, memory_dict)
    """
    outputs = []
    ground_truths = []
    kv_theoretical = []
    peak_gpu_mb_per_prompt = []

    kv_cfg = get_model_kv_config(model)

    for idx, ex in enumerate(data):
        prompt = build_coqa_prompt(ex["context"], ex["question"])

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        num_prompt_tokens = prompt_ids.shape[-1]

        out = generate_coqa(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_gpu_mb_per_prompt.append(peak_mb)

        gen_ids = tokenizer(
            out, return_tensors="pt", add_special_tokens=False
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

        outputs.append(out)
        ground_truths.append(ex["answer"])
        kv_theoretical.append(kv_stats)

        if idx < debug_n:
            print("=" * 80)
            print(f"[{label}] Example {idx}")
            print("- Prompt:\n", prompt)
            print("- GT:",    ex["answer"])
            print("- Output:", out)
            print(f"- KV theoretical (MB): {kv_stats['total_mb']:.2f}")
            print(f"- Empirical GPU peak (MB): {peak_mb:.2f}")

        del out, prompt_ids, gen_ids
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return outputs, ground_truths, {
        "per_prompt_kv":              kv_theoretical,
        "per_prompt_empirical_gpu_mb": peak_gpu_mb_per_prompt,
        "theoretical_peak_kv_mb":     max(kv["total_mb"] for kv in kv_theoretical),
        "empirical_peak_gpu_mb":      max(peak_gpu_mb_per_prompt),
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_coqa_aggregate_table(agg, decimals=3):
    """Build a formatted results table dict from CoQA aggregate metrics."""
    def fmt(x):
        return f"{x:.{decimals}f}"

    return {
        "Token match rate": {
            "Baseline -> GT":   "—",
            "KIVI -> GT":       "—",
            "Baseline <-> KIVI": fmt(agg["bk_token_match"]),
        },
        "Token F1 (raw)": {
            "Baseline -> GT":   fmt(agg["baseline_vs_gt_f1_raw"]),
            "KIVI -> GT":       fmt(agg["kivi_vs_gt_f1_raw"]),
            "Baseline <-> KIVI": "—",
        },
        "Token F1 (robust)": {
            "Baseline -> GT":   fmt(agg["baseline_vs_gt_f1_robust"]),
            "KIVI -> GT":       fmt(agg["kivi_vs_gt_f1_robust"]),
            "Baseline <-> KIVI": "—",
        },
        "ROUGE-L": {
            "Baseline -> GT":   fmt(agg["baseline_vs_gt_rouge"]),
            "KIVI -> GT":       fmt(agg["kivi_vs_gt_rouge"]),
            "Baseline <-> KIVI": fmt(agg["bk_rouge"]),
        },
        "BERTScore": {
            "Baseline -> GT":   fmt(agg["baseline_vs_gt_bert"]),
            "KIVI -> GT":       fmt(agg["kivi_vs_gt_bert"]),
            "Baseline <-> KIVI": fmt(agg["bk_bert"]),
        },
    }


def print_coqa_aggregate_table(table):
    """Pretty-print the formatted CoQA results table."""
    print("\nFINAL RESULTS (CoQA)")
    print(
        "Metric".ljust(22),
        "Baseline->GT".rjust(14),
        "KIVI->GT".rjust(12),
        "Baseline<->KIVI".rjust(18),
    )
    print("-" * 68)

    for metric, cols in table.items():
        print(
            metric.ljust(22),
            cols["Baseline -> GT"].rjust(14),
            cols["KIVI -> GT"].rjust(12),
            cols["Baseline <-> KIVI"].rjust(18),
        )


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_coqa_results(
    examples,
    results,
    memory_results,
    out_dir="coqa_results",
    n_samples=None,
):
    """
    Save CoQA per-example records, aggregate results, and memory results
    to three JSON files under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{n_samples}" if n_samples else ""

    examples_path = f"{out_dir}/examples_coqa{suffix}.json"
    results_path  = f"{out_dir}/results_coqa{suffix}.json"
    memory_path   = f"{out_dir}/memory_results_coqa{suffix}.json"

    with open(examples_path, "w") as f:
        json.dump(to_json_safe(examples), f, indent=2)

    with open(results_path, "w") as f:
        json.dump(to_json_safe(results), f, indent=2)

    with open(memory_path, "w") as f:
        json.dump(to_json_safe(memory_results), f, indent=2)

    print("Saved CoQA results:")
    print(" -", examples_path)
    print(" -", results_path)
    print(" -", memory_path)
