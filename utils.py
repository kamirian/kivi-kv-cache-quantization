"""
Shared utility functions used across benchmarks.
"""

import numpy as np


def to_json_safe(obj):
    """
    Recursively convert numpy types to native Python types
    so json.dump won't crash.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]  # tuples -> lists
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def token_level_match_rate(baseline_text, kivi_text, tokenizer):
    """
    Compute token-level match rate between baseline and KIVI outputs.

    Tokenizes both texts (without special tokens) and counts how many tokens
    at identical positions agree.

    Returns:
        dict with token_match_rate, matched_tokens, compared_tokens,
        baseline_len, kivi_len.
    """
    base_tokens = tokenizer.encode(baseline_text, add_special_tokens=False)
    kivi_tokens = tokenizer.encode(kivi_text, add_special_tokens=False)

    min_len = min(len(base_tokens), len(kivi_tokens))

    matches = sum(
        base_tokens[i] == kivi_tokens[i]
        for i in range(min_len)
    )

    return {
        "token_match_rate": matches / min_len if min_len > 0 else 0.0,
        "matched_tokens": matches,
        "compared_tokens": min_len,
        "baseline_len": len(base_tokens),
        "kivi_len": len(kivi_tokens),
    }
