"""
Generate publication-quality figures for the IRC-Bench paper.

Reads experiment results from experiments/results/ and produces
six figures saved as both PDF and PNG to article/figures/.

Usage:
    python article/generate_results_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DATA_DIR = PROJECT_ROOT / "data" / "benchmark"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Layout constants ──────────────────────────────────────────────────────
DPI = 300
SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.0   # inches

# ── Colorblind-friendly palette (Okabe-Ito inspired) ─────────────────────
PAL = {
    "emb_base":   "#56B4E9",  # sky blue
    "dpr":        "#0072B2",  # dark blue
    "llm_zs":     "#E69F00",  # orange
    "llm_fs":     "#D55E00",  # vermillion
    "cot":        "#CC79A7",  # pink
    "rag":        "#009E73",  # green
    "qlora":      "#F0E442",  # yellow
}

# ── Global matplotlib style ──────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    try:
        plt.style.use("seaborn-paper")
    except Exception:
        pass  # fall back to default

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,       # TrueType fonts in PDF
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ═════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Load all individual metrics files
metrics = {}
for p in sorted(RESULTS_DIR.glob("*_metrics.json")):
    d = load_json(p)
    metrics[d["exp_id"]] = d

# Load breakdown files
type_breakdown = load_json(RESULTS_DIR / "type_breakdown.json")
domain_breakdown = load_json(RESULTS_DIR / "domain_breakdown.json")
alias_eval = load_json(RESULTS_DIR / "alias_eval_all.json")
split_meta = load_json(DATA_DIR / "split_metadata.json")


# ═════════════════════════════════════════════════════════════════════════
#  EXPERIMENT REGISTRY
# ═════════════════════════════════════════════════════════════════════════
# Each entry: (exp_id, short_label, family, model_display, approx_params_B,
#              mode)
# "mode" is zs/fs/cot/rag for scatter plot markers.

EXPERIMENTS = [
    # Closed-world embedding baselines
    ("C1", "BGE name",        "Emb. Baseline", "BGE-base",      0.11, "base"),
    ("C2", "BGE desc.",       "Emb. Baseline", "BGE-base",      0.11, "base"),
    ("C3", "BGE wiki",        "Emb. Baseline", "BGE-base",      0.11, "base"),
    # Closed-world DPR fine-tuned
    ("C4", "DPR name",        "DPR Fine-tuned","BGE-base+DPR",  0.11, "ft"),
    ("C5", "DPR desc.",       "DPR Fine-tuned","BGE-base+DPR",  0.11, "ft"),
    ("C6", "DPR wiki",        "DPR Fine-tuned","BGE-base+DPR",  0.11, "ft"),
    # Open-world LLM zero-shot
    ("O1", "GPT-4o ZS",       "LLM ZS",       "GPT-4o",       200,  "zs"),
    ("O3", "GPT-4.1m ZS",     "LLM ZS",       "GPT-4.1-mini",   8,  "zs"),
    ("O5", "Llama-8B ZS",     "LLM ZS",       "Llama 3.1 8B",   8,  "zs"),
    # Open-world LLM few-shot
    ("O2", "GPT-4o FS",       "LLM FS",       "GPT-4o",       200,  "fs"),
    ("O4", "GPT-4.1m FS",     "LLM FS",       "GPT-4.1-mini",   8,  "fs"),
    ("O6", "Llama-8B FS",     "LLM FS",       "Llama 3.1 8B",   8,  "fs"),
    # Chain-of-Thought
    ("O11","GPT-4.1m CoT",    "CoT",          "GPT-4.1-mini",   8,  "cot"),
    ("O12","GPT-4o CoT",      "CoT",          "GPT-4o",       200,  "cot"),
    # RAG
    ("RAG1","BGE+GPT-4.1m",   "RAG",          "GPT-4.1-mini",   8,  "rag"),
    # QLoRA (Llama 8B)
    ("O10","Llama-8B QLoRA",  "QLoRA",        "Llama 3.1 8B",   8,  "qlora"),
]

# Build look-up dicts
EXP_BY_ID = {e[0]: e for e in EXPERIMENTS}
FAMILY_ORDER = [
    "Emb. Baseline", "DPR Fine-tuned",
    "LLM ZS", "LLM FS", "CoT", "RAG", "QLoRA",
]
FAMILY_COLORS = {
    "Emb. Baseline":  PAL["emb_base"],
    "DPR Fine-tuned": PAL["dpr"],
    "LLM ZS":        PAL["llm_zs"],
    "LLM FS":        PAL["llm_fs"],
    "CoT":           PAL["cot"],
    "RAG":           PAL["rag"],
    "QLoRA":         PAL["qlora"],
}


def get_exact(exp_id: str) -> float:
    """Return exact match (or hit@1 for closed-world) as a fraction."""
    if exp_id in metrics:
        m = metrics[exp_id]
        if "exact_match" in m:
            return m["exact_match"]
        if "hit_at_1" in m:
            return m["hit_at_1"]
    # Fall back to alias_eval_all (for O7/O8 which lack individual files)
    if exp_id in alias_eval:
        return alias_eval[exp_id]["hit_at_1_pct"] / 100.0
    return 0.0


def get_alias(exp_id: str) -> float:
    """Return alias match (or alias-aware hit@1) as a fraction."""
    if exp_id in metrics:
        m = metrics[exp_id]
        if "alias_match" in m:
            return m["alias_match"]
    if exp_id in alias_eval:
        return alias_eval[exp_id]["hit_at_1_pct"] / 100.0
    # For closed-world, alias eval has separate hit@1 values
    if exp_id in alias_eval and "hit_at_1_pct" in alias_eval[exp_id]:
        return alias_eval[exp_id]["hit_at_1_pct"] / 100.0
    return get_exact(exp_id)


def save(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(OUT_DIR / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Main results bar chart
# ═════════════════════════════════════════════════════════════════════════

def fig1_main_results():
    """Horizontal grouped bar chart: exact match and alias match by method."""
    print("Figure 1: Main results bar chart")

    # Collect data, sorted by exact match within each family
    labels, exact_vals, alias_vals, colors = [], [], [], []
    for fam in FAMILY_ORDER:
        entries = [(e[0], e[1]) for e in EXPERIMENTS if e[2] == fam]
        # Sort by exact match ascending (bars go bottom to top)
        entries.sort(key=lambda x: get_exact(x[0]))
        for eid, lbl in entries:
            labels.append(lbl)
            exact_vals.append(get_exact(eid))
            alias_vals.append(get_alias(eid))
            colors.append(FAMILY_COLORS[fam])

    y = np.arange(len(labels))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.38 * len(labels) + 0.8))

    bars_alias = ax.barh(y + bar_h / 2, alias_vals, bar_h,
                         color=[c + "88" for c in colors],
                         edgecolor=colors, linewidth=0.6,
                         label="Alias Match")
    bars_exact = ax.barh(y - bar_h / 2, exact_vals, bar_h,
                         color=colors, edgecolor=colors, linewidth=0.6,
                         label="Exact Match")

    # Value annotations
    for bar in bars_exact:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.1%}", va="center", fontsize=7)
    for bar in bars_alias:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.1%}", va="center", fontsize=7, fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, max(alias_vals) * 1.18)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("IRC-Bench: All Methods Comparison")
    ax.invert_yaxis()

    fig.tight_layout(pad=1.5)
    save(fig, "fig1_main_results")


# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Closed-world Hit@K curves
# ═════════════════════════════════════════════════════════════════════════

def fig2_hitk_curves():
    """Line plot of Hit@1/3/5/10 for C1-C6."""
    print("Figure 2: Closed-world Hit@K curves")

    k_values = [1, 3, 5, 10]
    closed_exps = ["C1", "C2", "C3", "C4", "C5", "C6"]

    repr_labels = {"name": "Name", "description": "Desc.", "wiki": "Wiki"}
    base_style = {"linestyle": "--", "marker": "o", "markersize": 5}
    dpr_style  = {"linestyle": "-",  "marker": "s", "markersize": 5}

    # Colors per entity representation
    repr_colors = {
        "name":        "#0072B2",
        "description": "#E69F00",
        "wiki":        "#009E73",
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.1))

    for eid in closed_exps:
        m = metrics[eid]
        vals = [m[f"hit_at_{k}"] for k in k_values]
        rep = m["entity_repr"]
        is_dpr = "dpr" in m.get("model", "").lower() or eid in ("C4","C5","C6")
        style = dpr_style if is_dpr else base_style
        prefix = "DPR" if is_dpr else "BGE"
        ax.plot(k_values, vals, **style,
                color=repr_colors[rep],
                label=f"{prefix} {repr_labels[rep]}")

    ax.set_xlabel("K")
    ax.set_ylabel("Hit@K")
    ax.set_xticks(k_values)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 0.80)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right", framealpha=0.9)
    ax.set_title("Closed-World Retrieval: Hit@K")

    fig.tight_layout(pad=1.5)
    save(fig, "fig2_hitk_curves")


# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 3: Per-entity-type heatmap
# ═════════════════════════════════════════════════════════════════════════

def fig3_entity_type_heatmap():
    """Heatmap: rows = experiments (best from each family), cols = entity types."""
    print("Figure 3: Per-entity-type heatmap")

    entity_types = ["Place", "Organization", "Person", "Event",
                    "Work", "Military_Unit"]
    type_labels  = ["Place", "Org.", "Person", "Event", "Work", "Mil. Unit"]

    # Pick representative experiments (top performers + variety)
    heatmap_exps = [
        "C1", "C5",      # best baseline, best DPR
        "O1", "O2",      # GPT-4o ZS, FS
        "O3", "O4",      # GPT-4.1-mini ZS, FS
        "O5", "O6",      # Llama 8B ZS, FS
        "O11", "O12",    # CoT
        "RAG1",          # RAG
        "O10",           # QLoRA
    ]

    # Filter to only those present in type_breakdown
    heatmap_exps = [e for e in heatmap_exps if e in type_breakdown]

    exp_labels = []
    for eid in heatmap_exps:
        if eid in EXP_BY_ID:
            exp_labels.append(EXP_BY_ID[eid][1])
        else:
            exp_labels.append(eid)

    mat = np.zeros((len(heatmap_exps), len(entity_types)))
    for i, eid in enumerate(heatmap_exps):
        tb = type_breakdown[eid]
        for j, et in enumerate(entity_types):
            if et in tb:
                mat[i, j] = tb[et]["hit_at_1_pct"]
            else:
                mat[i, j] = 0.0

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL * 0.75, 0.4 * len(heatmap_exps) + 1.0))

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=60)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    ax.set_xticks(range(len(type_labels)))
    ax.set_xticklabels(type_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels)
    ax.set_title("Accuracy (%) by Entity Type")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=9)

    fig.tight_layout(pad=1.5)
    save(fig, "fig3_entity_type_heatmap")


# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Dataset composition
# ═════════════════════════════════════════════════════════════════════════

def fig4_dataset_composition():
    """Two subplots: (a) entity type donut chart, (b) split distribution bar."""
    print("Figure 4: Dataset composition")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0))

    # (a) Entity type distribution (donut chart)
    by_type = dict(split_meta["by_type"])
    # Merge small categories into "Other"
    other_val = 0
    for key in ["Military_Unit", "Unknown", ""]:
        if key in by_type:
            other_val += by_type.pop(key)
    if other_val > 0:
        by_type["Other"] = other_val

    type_order = ["Place", "Organization", "Person", "Event", "Work", "Other"]
    type_order = [t for t in type_order if t in by_type]
    type_vals = [by_type[t] for t in type_order]
    type_labels = {
        "Place": "Place", "Organization": "Org.",
        "Person": "Person", "Event": "Event",
        "Work": "Work", "Other": "Other",
    }
    labels = [type_labels.get(t, t) for t in type_order]

    donut_colors = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#999999",
    ]

    wedges, texts, autotexts = ax1.pie(
        type_vals, labels=labels, autopct="%1.1f%%",
        colors=donut_colors[:len(type_vals)],
        pctdistance=0.78, labeldistance=1.22, startangle=140,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(7.5)
    ax1.set_title("(a) Entity Types", fontsize=11, pad=12)

    # Center text
    ax1.text(0, 0, f"N={split_meta['total_samples']:,}",
             ha="center", va="center", fontsize=9, fontweight="bold")

    # (b) Split distribution bar chart
    splits = ["train", "dev", "test"]
    split_samples = [split_meta[s]["samples"] for s in splits]
    split_entities = [split_meta[s]["entities"] for s in splits]
    split_labels = ["Train", "Dev", "Test"]

    x = np.arange(len(splits))
    w = 0.35
    bars1 = ax2.bar(x - w / 2, split_samples, w,
                    color="#0072B2", label="Samples", edgecolor="white")
    bars2 = ax2.bar(x + w / 2, split_entities, w,
                    color="#E69F00", label="Entities", edgecolor="white")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 150,
                 f"{int(h):,}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(split_labels)
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Dataset Splits", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, max(split_samples) * 1.15)

    fig.tight_layout(pad=1.5)
    save(fig, "fig4_dataset_composition")


# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 5: DPR improvement (baseline vs fine-tuned)
# ═════════════════════════════════════════════════════════════════════════

def fig5_dpr_improvement():
    """Paired bar chart: BGE baseline vs DPR fine-tuned per representation."""
    print("Figure 5: DPR improvement")

    reprs = ["name", "description", "wiki"]
    repr_labels = ["Name", "Description", "Wiki"]
    baseline_ids = ["C1", "C2", "C3"]
    dpr_ids = ["C4", "C5", "C6"]

    hit1_base = [metrics[e]["hit_at_1"] for e in baseline_ids]
    hit1_dpr  = [metrics[e]["hit_at_1"] for e in dpr_ids]
    mrr_base  = [metrics[e]["mrr"] for e in baseline_ids]
    mrr_dpr   = [metrics[e]["mrr"] for e in dpr_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 1.15, 3.2),
                                   sharey=False)

    x = np.arange(len(reprs))
    w = 0.32

    # Hit@1 subplot
    b1 = ax1.bar(x - w / 2, hit1_base, w, color=PAL["emb_base"],
                 label="BGE Baseline", edgecolor="white")
    b2 = ax1.bar(x + w / 2, hit1_dpr, w, color=PAL["dpr"],
                 label="DPR Fine-tuned", edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f"{h:.1%}", ha="center", va="bottom", fontsize=7)

    # Add improvement arrows/annotations
    for i in range(len(reprs)):
        diff = hit1_dpr[i] - hit1_base[i]
        mid_y = (hit1_base[i] + hit1_dpr[i]) / 2
        ax1.annotate(
            f"+{diff:.1%}",
            xy=(x[i] + w / 2, hit1_dpr[i]),
            xytext=(x[i] + w * 1.3, mid_y + 0.04),
            fontsize=6.5, color="#D55E00", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(repr_labels)
    ax1.set_ylabel("Hit@1")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_title("(a) Hit@1")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.set_ylim(0, max(hit1_dpr) * 1.25)

    # MRR subplot
    b3 = ax2.bar(x - w / 2, mrr_base, w, color=PAL["emb_base"],
                 label="BGE Baseline", edgecolor="white")
    b4 = ax2.bar(x + w / 2, mrr_dpr, w, color=PAL["dpr"],
                 label="DPR Fine-tuned", edgecolor="white")

    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    for i in range(len(reprs)):
        diff = mrr_dpr[i] - mrr_base[i]
        mid_y = (mrr_base[i] + mrr_dpr[i]) / 2
        ax2.annotate(
            f"+{diff:.3f}",
            xy=(x[i] + w / 2, mrr_dpr[i]),
            xytext=(x[i] + w * 1.3, mid_y + 0.04),
            fontsize=6.5, color="#D55E00", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8),
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(repr_labels)
    ax2.set_ylabel("MRR")
    ax2.set_title("(b) MRR")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_ylim(0, max(mrr_dpr) * 1.25)

    fig.suptitle("DPR Fine-tuning Improvement over BGE Baseline",
                 fontsize=12, y=1.02)
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(top=0.88, right=0.95)
    save(fig, "fig5_dpr_improvement")


# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 6: Model size vs accuracy scatter
# ═════════════════════════════════════════════════════════════════════════

def fig6_model_size_scatter():
    """Scatter plot: X = approx model params (log), Y = exact match."""
    print("Figure 6: Model size vs accuracy scatter")

    # Only include open-world experiments with known param counts
    scatter_exps = [
        ("O1",  "GPT-4o ZS",     200, "zs"),
        ("O2",  "GPT-4o FS",     200, "fs"),
        ("O3",  "GPT-4.1m ZS",     8, "zs"),
        ("O4",  "GPT-4.1m FS",     8, "fs"),
        ("O5",  "Llama-8B ZS",     8, "zs"),
        ("O6",  "Llama-8B FS",     8, "fs"),
        ("O11", "GPT-4.1m CoT",    8, "cot"),
        ("O12", "GPT-4o CoT",    200, "cot"),
        ("RAG1","BGE+4.1m RAG",    8, "rag"),
        ("O10", "Llama-8B QLoRA",  8, "qlora"),
    ]

    mode_markers = {
        "zs":    ("o", PAL["llm_zs"]),
        "fs":    ("s", PAL["llm_fs"]),
        "cot":   ("^", PAL["cot"]),
        "rag":   ("D", PAL["rag"]),
        "qlora": ("P", PAL["qlora"]),
    }
    mode_labels = {
        "zs": "Zero-shot",
        "fs": "Few-shot",
        "cot": "Chain-of-Thought",
        "rag": "RAG",
        "qlora": "QLoRA",
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.2))

    # Plot each mode group for legend
    plotted_modes = set()
    for eid, label, params, mode in scatter_exps:
        marker, color = mode_markers[mode]
        acc = get_exact(eid)
        show_label = mode not in plotted_modes
        ax.scatter(params, acc, marker=marker, color=color, s=70,
                   edgecolors="black", linewidth=0.5, zorder=5,
                   label=mode_labels[mode] if show_label else None)
        plotted_modes.add(mode)

        # Position annotations to avoid overlap
        # Use small offsets based on direction
        x_off, y_off = 0.15, 0.008
        ha = "left"
        # Special adjustments for crowded points
        if eid == "O4":
            y_off = -0.012
        elif eid == "O6":
            y_off = -0.018
        elif eid == "O5":
            y_off = 0.012
        elif eid == "O11":
            y_off = -0.008
            x_off = -0.1
            ha = "right"
        elif eid == "RAG1":
            y_off = 0.015
        elif eid == "O2":
            y_off = 0.01
        elif eid == "O12":
            y_off = -0.015

        ax.annotate(label,
                    xy=(params, acc),
                    xytext=(params * (1 + x_off), acc + y_off),
                    fontsize=6.5, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="gray",
                                   lw=0.4, alpha=0.6))

    ax.set_xscale("log")
    ax.set_xlabel("Approx. Parameters (B)")
    ax.set_ylabel("Exact Match")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0.5, 600)
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_title("Model Size vs. Accuracy")

    # Custom x ticks
    ax.set_xticks([1, 8, 200])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which="minor", size=0)

    fig.tight_layout(pad=1.5)
    save(fig, "fig6_model_size_scatter")


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results dir:  {RESULTS_DIR}")
    print(f"Output dir:   {OUT_DIR}")
    print(f"Loaded {len(metrics)} metric files, "
          f"{len(type_breakdown)} type breakdowns, "
          f"{len(alias_eval)} alias evals")
    print()

    fig1_main_results()
    fig2_hitk_curves()
    fig3_entity_type_heatmap()
    fig4_dataset_composition()
    fig5_dpr_improvement()
    fig6_model_size_scatter()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
