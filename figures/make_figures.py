"""
Make the README figures.

    python figures/make_figures.py

Run from the repository root. The results figure reads only the saved files in
results/; the other three figures are diagrams.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

# KIVI settings, from kivi/attention.py (kivi_group_size, kivi_residual_length)
G = 32
R = 128

FP16 = "#4C72B0"
QUANT = "#F2A65A"
QUANT_DARK = "#C9711E"
EVICT = "#E3E3E3"
HIGH = "#B0413E"
BASE = "#7F7F7F"
TEXT = "#222222"

plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "svg.fonttype": "none",
})


def load(name):
    return json.loads((RESULTS / name).read_text())


# ---------------------------------------------------------------------------
# Figure 1: KIVI schematic
# ---------------------------------------------------------------------------

def draw_grid(ax, x0, y0, w, h, n_tok, n_ch, color, edge="white", lw=0.6):
    """Token x channel grid; tokens along x, channels along y."""
    dx, dy = w / n_tok, h / n_ch
    for i in range(n_tok):
        for j in range(n_ch):
            ax.add_patch(Rectangle((x0 + i * dx, y0 + j * dy), dx, dy,
                                   facecolor=color, edgecolor=edge, lw=lw))
    return dx, dy


def cache_panel(ax, kind):
    """One cache (key or value) as a token x channel grid with its grouping."""
    ax.set_xlim(-1.2, 20.6)
    ax.set_ylim(-2.9, 9.6)
    ax.axis("off")

    # 12 quantized token columns + 4 full-precision columns, 8 channel rows
    n_q, n_r, n_ch = 12, 4, 8
    cw, ch = 1.0, 0.85
    y0 = 0.6
    xq = 0.6
    xr = xq + n_q * cw + 0.35

    draw_grid(ax, xq, y0, n_q * cw, n_ch * ch, n_q, n_ch, QUANT)
    draw_grid(ax, xr, y0, n_r * cw, n_ch * ch, n_r, n_ch, FP16)

    top = y0 + n_ch * ch
    if kind == "key":
        # token groups of G along the token axis
        for k in range(1, 3):
            xb = xq + 4 * k * cw
            ax.plot([xb, xb], [y0 - 0.1, top + 0.1], color=TEXT, lw=2.2)
        for k in range(3):
            xc = xq + (4 * k + 2) * cw
            ax.annotate("", xy=(xq + 4 * k * cw + 0.08, top + 0.45),
                        xytext=(xq + 4 * (k + 1) * cw - 0.08, top + 0.45),
                        arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.2))
            ax.text(xc, top + 0.75, f"G = {G}", ha="center", va="bottom", fontsize=10.5)
        # one channel within one token group
        hy = y0 + 5 * ch
        ax.add_patch(Rectangle((xq + 4 * cw, hy), 4 * cw, ch, facecolor="none",
                               edgecolor=HIGH, lw=2.8, zorder=5))
        ax.annotate("one scale + zero-point\nper channel, per token group",
                    xy=(xq + 8 * cw, hy + ch / 2), xytext=(xq + 8.4 * cw, -2.1),
                    ha="center", va="center", fontsize=10.5, color=HIGH,
                    arrowprops=dict(arrowstyle="->", color=HIGH, lw=1.4,
                                    connectionstyle="arc3,rad=-0.25"))
        res_label = "residual: < R tokens\n(full precision)"
        title = "Key cache: per-channel quantization"
    else:
        # 4 channel groups of G = 32 along the 128-channel head dimension
        for k in range(1, 4):
            yb = y0 + 2 * k * ch
            ax.plot([xq - 0.1, xq + n_q * cw + 0.1], [yb, yb], color=TEXT, lw=2.2)
        for k in range(4):
            ax.annotate("", xy=(xq - 0.3, y0 + 2 * k * ch + 0.06),
                        xytext=(xq - 0.3, y0 + 2 * (k + 1) * ch - 0.06),
                        arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.1))
            ax.text(xq - 0.5, y0 + (2 * k + 1) * ch, f"G = {G}", ha="right", va="center", fontsize=10)
        ax.text(xq - 0.5, top + 0.1, "channels", ha="right", va="bottom", fontsize=10)
        # one token within one channel group
        hx = xq + 6 * cw
        ax.add_patch(Rectangle((hx, y0 + 4 * ch), cw, 2 * ch, facecolor="none",
                               edgecolor=HIGH, lw=2.8, zorder=5))
        ax.annotate("one scale + zero-point\nper token, per channel group",
                    xy=(hx + cw / 2, y0 + 4 * ch), xytext=(xq + 8.4 * cw, -2.1),
                    ha="center", va="center", fontsize=10.5, color=HIGH,
                    arrowprops=dict(arrowstyle="->", color=HIGH, lw=1.4,
                                    connectionstyle="arc3,rad=0.25"))
        res_label = "residual: last R = 128 tokens\n(full precision)"
        title = "Value cache: per-token quantization"

    # axes
    ax.annotate("", xy=(xr + n_r * cw + 0.6, -0.15), xytext=(xq, -0.15),
                arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4))
    ax.text(xr + n_r * cw + 0.75, -0.15, "token\n(time)", ha="left", va="center", fontsize=11)
    ax.annotate("", xy=(xr + n_r * cw + 0.6, top), xytext=(xr + n_r * cw + 0.6, y0),
                arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.4))
    ax.text(xr + n_r * cw + 0.85, (y0 + top) / 2, "channel\n(head_dim\n= 128)",
            ha="left", va="center", fontsize=11)

    ax.text(xq + n_q * cw / 2, top + (1.55 if kind == "key" else 0.5),
            "quantized part: 2-bit levels", ha="center", va="bottom", fontsize=11,
            color=QUANT_DARK, weight="bold")
    ax.text(xr + n_r * cw / 2, top + 0.5, res_label, ha="center", va="bottom", fontsize=10.5,
            color=FP16, weight="bold")
    ax.set_title(title, loc="left", weight="bold", pad=2)


def bar(ax, y, n_q, n_r, total, label, note, x0=3.6, width=13.5, h=0.62):
    """One cache state drawn as a token bar: quantized tokens, then full-precision residual."""
    sc = width / total
    if n_q:
        ax.add_patch(Rectangle((x0, y), n_q * sc, h, facecolor=QUANT, edgecolor="white", lw=0.8))
        # group boundaries every G tokens (keys) are drawn by the caller
    if n_r:
        ax.add_patch(Rectangle((x0 + n_q * sc, y), n_r * sc, h, facecolor=FP16, edgecolor="white", lw=0.8))
    ax.text(x0 - 0.15, y + h / 2, label, ha="right", va="center", fontsize=11)
    ax.text(x0 + (n_q + n_r) * sc + 0.15, y + h / 2, note, ha="left", va="center", fontsize=10.5)
    return sc


def timeline_panel(ax):
    ax.set_xlim(-0.2, 24.5)
    ax.set_ylim(-0.6, 8.6)
    ax.axis("off")
    ax.set_title("How the caches change during generation (one attention head, kivi/cache.py)",
                 loc="left", weight="bold", pad=2)

    total = 416  # longest state drawn: prompt of 300 tokens + 116 generated tokens
    rows = [
        # (step label, key quantized, key residual, value quantized, value residual)
        ("after prefill\n(prompt l = 300)", 288, 12, 172, 128,
         "keys: l - (l mod G) = 288 quantized, l mod G = 12 kept\n(Algorithm 1 of the paper keeps l mod R here)",
         "values: all but the last R quantized (172), last 128 kept"),
        ("after 115 new\ntokens", 288, 127, 287, 128,
         "keys: new tokens append to the residual (127 < R)",
         "values: each new token pushes the oldest residual token\ninto the quantized part (per-token); residual stays at R"),
        ("after 116 new\ntokens", 416, 0, 288, 128,
         "keys: residual reached R = 128, so all 128 are quantized\nper channel (4 groups of G) and the residual is emptied",
         "values: same rule as every step; residual stays at R"),
    ]
    y = 7.4
    for lab, kq, kr, vq, vr, kn, vn in rows:
        ax.text(0.0, y - 0.55, lab, ha="left", va="center", fontsize=11, weight="bold")
        sc = bar(ax, y, kq, kr, total, "K", "")
        for g in range(G, kq + 1, G):
            ax.plot([3.6 + g * sc] * 2, [y, y + 0.62], color="white", lw=0.9)
        ax.text(3.6 + 13.5 + 0.3, y + 0.31, kn, ha="left", va="center", fontsize=10)
        bar(ax, y - 1.0, vq, vr, total, "V", "")
        ax.text(3.6 + 13.5 + 0.3, y - 0.69, vn, ha="left", va="center", fontsize=10)
        y -= 2.65

    ax.add_patch(Rectangle((3.6, -0.45), 0.45, 0.35, facecolor=QUANT))
    ax.text(4.15, -0.27, "2-bit levels (grouped, scale + zero-point per group)", va="center", fontsize=10.5)
    ax.add_patch(Rectangle((13.2, -0.45), 0.45, 0.35, facecolor=FP16))
    ax.text(13.75, -0.27, "full precision (FP16) residual", va="center", fontsize=10.5)


def formula_panel(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0.01",
                                facecolor="#F7F7F7", edgecolor="#BBBBBB"))
    ax.text(0.05, 0.86, "Asymmetric min-max quantization, B = 2 bits (kivi/quantization.py)",
            fontsize=12, weight="bold", va="center")
    lines = [
        r"zero-point:   $z = \min(X_{\mathrm{group}})$",
        r"scale:   $s = (\max(X_{\mathrm{group}}) - \min(X_{\mathrm{group}})) \;/\; (2^{2} - 1)$",
        r"quantize:   $Q(X) = \mathrm{round}((X - z)\,/\,s) \in \{0, 1, 2, 3\}$",
        r"dequantize:   $X' = Q(X)\cdot s + z$",
    ]
    for i, t in enumerate(lines):
        ax.text(0.07, 0.66 - i * 0.17, t, fontsize=12, va="center")
    ax.text(0.55, 0.47, f"G = {G}, R = {R} (kivi/attention.py; same as the KIVI paper)\n"
            "one cache per layer and per KV head\n"
            "prefill attention uses the exact FP16 keys and values;\n"
            "decoding dequantizes the cache, then runs standard attention",
            fontsize=11, va="center", ha="left", color=TEXT)


def figure_kivi_schematic():
    fig = plt.figure(figsize=(15, 12.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 0.42, 1.05], hspace=0.1, wspace=0.08)
    cache_panel(fig.add_subplot(gs[0, 0]), "key")
    cache_panel(fig.add_subplot(gs[0, 1]), "value")
    formula_panel(fig.add_subplot(gs[1, :]))
    timeline_panel(fig.add_subplot(gs[2, :]))
    fig.savefig(OUT / "kivi_schematic.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: KIVI vs StreamingLLM vs ZipCache
# ---------------------------------------------------------------------------

def figure_method_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.6))
    n = 40
    w = 0.25
    x0 = 0.3

    def cells(ax, colors, hatches=None):
        for i, c in enumerate(colors):
            ax.add_patch(Rectangle((x0 + i * w, 0), w, 1, facecolor=c, edgecolor="white", lw=1.0,
                                   hatch=(hatches[i] if hatches else None)))
        ax.set_xlim(0, x0 + n * w + 3.6)
        ax.set_ylim(-0.75, 1.9)
        ax.axis("off")
        ax.annotate("", xy=(x0 + n * w + 0.25, -0.35), xytext=(x0, -0.35),
                    arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2))
        ax.text(x0 + n * w + 0.35, -0.35, "token (time)", va="center", fontsize=11)

    # KIVI: every token kept; older tokens 2-bit, the recent R tokens full precision
    ax = axes[0]
    cells(ax, [QUANT] * 30 + [FP16] * 10)
    ax.set_title("KIVI (Liu et al., 2024): keep every token, lower the numerical precision",
                 loc="left", weight="bold")
    ax.text(x0 + 15 * w, 1.25, "older tokens: 2-bit (keys per channel, values per token)",
            ha="center", fontsize=10.5, color=QUANT_DARK)
    ax.text(x0 + 35 * w, 1.25, "recent tokens (up to R): full precision", ha="center", fontsize=10.5, color=FP16)

    # StreamingLLM: attention sinks + rolling window; middle evicted
    ax = axes[1]
    cols = [FP16] * 4 + [EVICT] * 24 + [FP16] * 12
    hat = [None] * 4 + ["xx"] * 24 + [None] * 12
    cells(ax, cols, hat)
    ax.set_title("StreamingLLM (Xiao et al., 2023): keep attention sinks + recent window, evict the middle",
                 loc="left", weight="bold")
    ax.text(x0 + 2 * w, 1.25, "initial tokens\n(attention sinks)", ha="center", fontsize=10.5, color=FP16)
    ax.text(x0 + 16 * w, 1.25, "evicted (removed from the cache)", ha="center", fontsize=10.5, color=BASE)
    ax.text(x0 + 34 * w, 1.25, "rolling window of recent tokens", ha="center", fontsize=10.5, color=FP16)

    # ZipCache: per-token bit-width from normalized-attention saliency
    ax = axes[2]
    salient = {2, 3, 9, 15, 16, 22, 30, 31, 35, 36, 37, 38, 39}
    cells(ax, [HIGH if i in salient else QUANT for i in range(n)])
    ax.set_title("ZipCache (He et al., 2024): keep every token, higher precision for salient tokens",
                 loc="left", weight="bold")
    ax.text(x0 + n * w / 2, 1.25,
            "saliency = normalized attention score; salient tokens get a higher bit-width, the rest a lower one",
            ha="center", fontsize=10.5)

    handles = [Rectangle((0, 0), 1, 1, facecolor=FP16), Rectangle((0, 0), 1, 1, facecolor=QUANT),
               Rectangle((0, 0), 1, 1, facecolor=HIGH),
               Rectangle((0, 0), 1, 1, facecolor=EVICT, hatch="xx", edgecolor="white")]
    fig.legend(handles, ["full precision", "low-bit", "higher-bit (salient)", "evicted"],
               loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "method_comparison.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: baseline vs KIVI results (saved files only)
# ---------------------------------------------------------------------------

def mean_kv_mb(memory, side):
    kv = memory[side]["per_prompt_kv"]
    return sum(x["total_mb"] for x in kv) / len(kv)


def figure_results():
    cnn = load("results_cnn_dm_7b_n100.json")
    gsm = load("results_gsm8k_13b_n50.json")
    coqa = load("results_coqa_7b_n500.json")
    n_cnn = len(load("examples_cnn_dm_7b_n100.json"))
    n_gsm = len(load("examples_gsm8k_13b_n50.json"))
    n_coqa = len(load("examples_coqa_7b_n500.json"))
    mem = {
        "CNN/DM": load("memory_cnn_dm_7b_n100.json"),
        "GSM8K": load("memory_gsm8k_13b_n50.json"),
        "CoQA": load("memory_coqa_7b_n500.json"),
    }

    panels = [
        ("Summarization: CNN/DailyMail", f"LLaMA-2-7B-chat, n = {n_cnn}", "ROUGE-L F1 vs reference",
         cnn["rouge_bg"]["mean"], cnn["rouge_kg"]["mean"], "{:.3f}"),
        ("Math reasoning: GSM8K", f"LLaMA-2-13B-chat, n = {n_gsm}", "exact-match accuracy",
         gsm["baseline_EM"], gsm["kivi_EM"], "{:.2f}"),
        ("Question answering: CoQA", f"LLaMA-2-7B-chat, n = {n_coqa}", "token F1 vs reference",
         coqa["baseline_vs_gt_f1_raw"], coqa["kivi_vs_gt_f1_raw"], "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for ax, (title, sub, metric, b, k, fmt) in zip(axes, panels):
        bars = ax.bar(["FP16\nbaseline", "KIVI\n2-bit"], [b, k], color=[BASE, QUANT], width=0.6)
        for r, v in zip(bars, [b, k]):
            ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=12)
        ax.set_title(f"{title}\n{sub}", fontsize=12)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(b, k) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[3]
    names = list(mem)
    xs = range(len(names))
    bvals = [mean_kv_mb(mem[n], "baseline") for n in names]
    kvals = [mean_kv_mb(mem[n], "kivi") for n in names]
    ax.bar([x - 0.2 for x in xs], bvals, 0.4, color=BASE, label="FP16 baseline")
    ax.bar([x + 0.2 for x in xs], kvals, 0.4, color=QUANT, label="KIVI 2-bit")
    for x, b, k in zip(xs, bvals, kvals):
        ax.text(x - 0.2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=11)
        ax.text(x + 0.2, k, f"{k:.0f}", ha="center", va="bottom", fontsize=11)
        ax.text(x, max(b, k) * 1.12 + 25, f"{b / k:.1f}x smaller", ha="center", fontsize=11, weight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{n}\n({'13B' if n == 'GSM8K' else '7B'})" for n in names])
    ax.set_ylabel("MiB per prompt (mean)")
    ax.set_ylim(0, max(bvals) * 1.3 + 50)
    ax.set_title("Theoretical packed KV-cache size\n(not measured GPU memory)", fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "results_summary.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: evaluation workflow
# ---------------------------------------------------------------------------

def figure_workflow():
    fig, ax = plt.subplots(figsize=(16.5, 3.6))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0.3, 3.6)
    ax.axis("off")

    boxes = [
        ("1. Dataset selection",
         "first N examples:\nCNN/DM 3.0.0 validation\nGSM8K test\nCoQA validation\n(kivi/benchmarks/)"),
        ("2. Prompt construction",
         "LLaMA-2 chat format\n(CNN/DM, CoQA)\nplain step-by-step\nprompt (GSM8K)"),
        ("3. Generation",
         "greedy decoding, batch 1\nFP16 baseline model,\nthen KIVI model\n(same weights, attention\nlayers replaced)"),
        ("4. Scoring",
         "each output vs reference\n(ROUGE-L, BERTScore,\nF1, exact match)\nKIVI vs baseline (token\nmatch, ROUGE-L, BERTScore)\nadded: sentence compliance,\nentity flag rate (CNN/DM),\nrobust F1 (CoQA)"),
        ("5. Saved outputs",
         "results/\nexamples_*: per-example\nresults_*: aggregates\nmemory_*: KV size and\npeak GPU memory"),
    ]
    w, h, gap = 2.85, 2.9, 0.5
    x = 0.15
    for i, (title, body) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 0.45), w, h, boxstyle="round,pad=0.04",
                                    facecolor="#F4F7FB" if i != 2 else "#FFF4E8",
                                    edgecolor=FP16 if i != 2 else QUANT_DARK, lw=1.6))
        ax.text(x + w / 2, 0.45 + h - 0.35, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, 0.45 + h / 2 - 0.25, body, ha="center", va="center", fontsize=10.5,
                linespacing=1.35)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, 0.45 + h / 2), (x + w + gap - 0.05, 0.45 + h / 2),
                                         arrowstyle="-|>", mutation_scale=18, color=TEXT, lw=1.5))
        x += w + gap

    fig.savefig(OUT / "evaluation_workflow.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    figure_kivi_schematic()
    figure_method_comparison()
    figure_results()
    figure_workflow()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))
