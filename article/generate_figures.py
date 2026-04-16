"""
Generate publication-quality figures for IRC Recovery paper.
Uses matplotlib + seaborn with ACL-compatible styling.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
DPI = 300
SINGLE_COL = 3.25   # inches (ACL single column)
DOUBLE_COL = 6.75   # inches (ACL double column)

# Colorblind-safe palette
COLORS = {
    "LLM":       "#0173B2",  # blue
    "Hybrid RAG":"#DE8F05",  # orange
    "Embedding": "#029E73",  # green
}
METHOD_ORDER = ["LLM", "Hybrid RAG", "Embedding"]

DATASET_LABELS = {
    "veterans_text_to_entity": "Vet-T2E",
    "veterans_entity_to_text": "Vet-E2T",
    "twitter_baseline":        "Tweet-BL",
    "twitter_entity_to_text":  "Tweet-E2T",
}
DATASET_ORDER = [
    "veterans_text_to_entity",
    "veterans_entity_to_text",
    "twitter_baseline",
    "twitter_entity_to_text",
]

# ── Load data ──────────────────────────────────────────────────────────────
with open(Path(__file__).parent / "results.json") as f:
    data = json.load(f)

stats = data["dataset_stats"]
results = data["recognition_results"]

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Figure 1: Entity Type Distribution ─────────────────────────────────────
def fig1_entity_distribution():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.2))

    types = ["Person", "Place", "Event"]
    x = np.arange(len(DATASET_ORDER))
    width = 0.25
    type_colors = ["#0173B2", "#DE8F05", "#029E73"]

    for i, (etype, col) in enumerate(zip(types, type_colors)):
        key = f"pct_{etype.lower()}"
        vals = [stats[d][key] for d in DATASET_ORDER]
        bars = ax.bar(x + i * width, vals, width, label=etype, color=col, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    fig.savefig(OUT_DIR / "fig1_entity_distribution.png")
    plt.close(fig)
    print("  [OK] fig1_entity_distribution.png")


# ── Figure 2: Hit@K Comparison (4-panel) ───────────────────────────────────
def fig2_hitk_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, 2.4), sharey=True)
    ks = ["hit1", "hit3", "hit5"]
    k_labels = ["Hit@1", "Hit@3", "Hit@5"]

    for idx, ds in enumerate(DATASET_ORDER):
        ax = axes[idx]
        ds_results = [r for r in results if r["dataset"] == ds]

        x = np.arange(len(ks))
        width = 0.25

        for mi, method in enumerate(METHOD_ORDER):
            row = next(r for r in ds_results if r["method"] == method)
            vals = [row[k] for k in ks]
            ax.bar(x + mi * width, vals, width, label=method, color=COLORS[method],
                   edgecolor="white", linewidth=0.5)

        ax.set_title(DATASET_LABELS[ds], fontsize=8.5, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(k_labels, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        if idx == 0:
            ax.set_ylabel("Score")

    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT_DIR / "fig2_hitk_comparison.png")
    plt.close(fig)
    print("  [OK] fig2_hitk_comparison.png")


# ── Figure 3: Global MRR Heatmap ──────────────────────────────────────────
def fig3_global_mrr_heatmap():
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))

    matrix = np.zeros((len(METHOD_ORDER), len(DATASET_ORDER)))
    for mi, method in enumerate(METHOD_ORDER):
        for di, ds in enumerate(DATASET_ORDER):
            row = next(r for r in results if r["dataset"] == ds and r["method"] == method)
            matrix[mi, di] = row["global_mrr"]

    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[DATASET_LABELS[d] for d in DATASET_ORDER],
                yticklabels=METHOD_ORDER,
                ax=ax, cbar_kws={"shrink": 0.8, "label": "Global MRR"},
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 7.5})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.savefig(OUT_DIR / "fig3_global_mrr_heatmap.png")
    plt.close(fig)
    print("  [OK] fig3_global_mrr_heatmap.png")


# ── Figure 4: Filtered vs Global MRR Gap ──────────────────────────────────
def fig4_mrr_gap_analysis():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.4))

    labels = []
    global_vals = []
    filtered_vals = []
    colors_list = []

    for ds in DATASET_ORDER:
        for method in METHOD_ORDER:
            row = next(r for r in results if r["dataset"] == ds and r["method"] == method)
            labels.append(f"{DATASET_LABELS[ds]}\n{method}")
            global_vals.append(row["global_mrr"])
            filtered_vals.append(row["filtered_mrr"])
            colors_list.append(COLORS[method])

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, global_vals, width, label="Global MRR",
                   color=[c + "99" for c in colors_list], edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + width/2, filtered_vals, width, label="Filtered MRR",
                   color=colors_list, edgecolor="white", linewidth=0.3)

    ax.set_ylabel("MRR Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5, ha="center")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # Add gap annotations for selected pairs
    for i in range(len(x)):
        gap = filtered_vals[i] - global_vals[i]
        if gap > 0.3:
            mid_y = (global_vals[i] + filtered_vals[i]) / 2
            ax.annotate(f"+{gap:.2f}", xy=(x[i], mid_y), fontsize=5,
                        ha="center", va="center", color="#666666")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_mrr_gap_analysis.png")
    plt.close(fig)
    print("  [OK] fig4_mrr_gap_analysis.png")


# ── Figure 5: Radar Chart (Method Comparison) ─────────────────────────────
def fig5_method_radar():
    metrics = ["hit1", "hit3", "hit5", "global_mrr", "filtered_mrr"]
    metric_labels = ["Hit@1", "Hit@3", "Hit@5", "Global\nMRR", "Filtered\nMRR"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for method in METHOD_ORDER:
        method_results = [r for r in results if r["method"] == method]
        avg_vals = []
        for m in metrics:
            avg_vals.append(np.mean([r[m] for r in method_results]))
        avg_vals += avg_vals[:1]

        ax.plot(angles, avg_vals, "o-", linewidth=1.5, markersize=3,
                label=method, color=COLORS[method])
        ax.fill(angles, avg_vals, alpha=0.1, color=COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=6, color="gray")
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, linewidth=0.3)

    fig.savefig(OUT_DIR / "fig5_method_radar.png")
    plt.close(fig)
    print("  [OK] fig5_method_radar.png")


# ── Figure 6: Per-Entity-Type Global MRR (from recovered figure data) ──────
def fig6_entity_type_mrr():
    """Recreate report Figure 4: Global MRR per entity type per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0), sharey=True)

    etype_data = data.get("per_entity_type_global_mrr", {})
    ds_list = [
        ("twitter_baseline", "Tweet-BL"),
        ("twitter_entity_to_text", "Tweet-E2T"),
        ("veterans_text_to_entity", "Vet-T2E"),
        ("veterans_entity_to_text", "Vet-E2T"),
    ]

    for idx, (ds_key, ds_label) in enumerate(ds_list):
        ax = axes[idx // 2][idx % 2]
        ds_data = etype_data.get(ds_key, {})
        if not ds_data:
            ax.set_title(ds_label, fontsize=8.5, fontweight="bold")
            continue

        # Collect all entity types across methods
        all_types = sorted(set(t for m in ds_data.values() for t in m.keys()))
        x = np.arange(len(all_types))
        width = 0.25

        for mi, method in enumerate(METHOD_ORDER):
            method_data = ds_data.get(method, {})
            vals = [method_data.get(t, 0) for t in all_types]
            ax.bar(x + mi * width, vals, width, label=method,
                   color=COLORS[method], edgecolor="white", linewidth=0.5)

        ax.set_title(ds_label, fontsize=8.5, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_types, fontsize=7, rotation=15)
        ax.set_ylim(0, 1.0)
        if idx % 2 == 0:
            ax.set_ylabel("Global MRR", fontsize=8)

    axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    fig.savefig(OUT_DIR / "fig6_entity_type_mrr.png")
    plt.close(fig)
    print("  [OK] fig6_entity_type_mrr.png")


# ── Figure 7: Hit@K Saturation (including K=10) ──────────────────────────
def fig7_hitk_saturation():
    """Line plot showing Hit@K across K=1,3,5,10 to visualize saturation."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2), sharey=True)
    ks = [1, 3, 5, 10]
    k_fields = ["hit1", "hit3", "hit5", "hit10"]

    for mi, method in enumerate(METHOD_ORDER):
        ax = axes[mi]
        for ds in DATASET_ORDER:
            row = next(r for r in results if r["dataset"] == ds and r["method"] == method)
            vals = [row.get(f, row.get(f, None)) for f in k_fields]
            if None in vals:
                continue
            ax.plot(ks, vals, "o-", linewidth=1.3, markersize=3,
                    label=DATASET_LABELS[ds])
        ax.set_title(method, fontsize=9, fontweight="bold")
        ax.set_xlabel("K", fontsize=8)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.0)
        if mi == 0:
            ax.set_ylabel("Hit@K")
            ax.legend(frameon=False, fontsize=6, loc="lower right")

    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT_DIR / "fig7_hitk_saturation.png")
    plt.close(fig)
    print("  [OK] fig7_hitk_saturation.png")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig1_entity_distribution()
    fig2_hitk_comparison()
    fig3_global_mrr_heatmap()
    fig4_mrr_gap_analysis()
    fig5_method_radar()
    fig6_entity_type_mrr()
    fig7_hitk_saturation()
    print("All figures generated in:", OUT_DIR)
