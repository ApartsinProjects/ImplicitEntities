"""
Post-hoc analysis of implicit entity recognition experiment results.

Processes prediction CSVs and metrics JSONs produced by run_experiments.py,
computes comprehensive analyses, and saves outputs to experiments/analysis/.

Analyses:
  1. Per-entity-type breakdown (Hit@1, Hit@3, MRR by entity_type)
  2. Matching layer contribution (% of hits from each tier)
  3. Inter-method agreement (Cohen's kappa between method pairs)
  4. Ensemble methods (majority vote, RRF, cascading)
  5. Text length effects (short/medium/long bins)
  6. Per-narrator variance (veterans only)
  7. Bootstrap confidence intervals (95% CI on Hit@1 and MRR)
  8. nDCG@K (using Jaccard similarity as graded relevance)
  9. Dataset difficulty (embedding similarity vs. recognition success)
  10. Summary tables (CSV + Markdown)

Usage:
    python analyze_results.py                          # analyze latest results
    python analyze_results.py --timestamp 20260416_... # analyze specific run
    python analyze_results.py --results-dir experiments/results/
"""

import argparse
import json
import math
import os
import re
import string
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Optional: sentence-transformers for dataset difficulty analysis
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"


# ============================================================================
#  UTILITIES
# ============================================================================

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def jaccard_token_similarity(a: str, b: str) -> float:
    """Jaccard similarity over word tokens of normalized strings."""
    tokens_a = set(normalize(a).split())
    tokens_b = set(normalize(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def token_count(text: str) -> int:
    """Rough token count by whitespace splitting."""
    return len(text.split())


def reciprocal_rank(match_rank: int) -> float:
    """Return 1/rank if matched, else 0."""
    if match_rank > 0:
        return 1.0 / match_rank
    return 0.0


# ============================================================================
#  FILE DISCOVERY
# ============================================================================

def discover_result_files(results_dir: Path) -> dict:
    """
    Scan results_dir and group files by timestamp.

    Returns dict keyed by timestamp, each value is a dict:
      {
        "predictions": {(dataset, method, model): Path, ...},
        "metrics": {(dataset, method, model): Path, ...},
        "summary": Path or None,
      }
    """
    groups = defaultdict(lambda: {"predictions": {}, "metrics": {}, "summary": None})

    for f in sorted(results_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name

        # Summary files: {timestamp}_summary.json
        m = re.match(r"^(\d{8}_\d{6})_summary\.json$", name)
        if m:
            ts = m.group(1)
            groups[ts]["summary"] = f
            continue

        # Prediction CSVs: {timestamp}_{dataset}_{method}_{model}_predictions.csv
        m = re.match(r"^(\d{8}_\d{6})_(.+?)_(llm|embedding|hybrid)_(.+?)_predictions\.csv$", name)
        if m:
            ts, dataset, method, model = m.groups()
            groups[ts]["predictions"][(dataset, method, model)] = f
            continue

        # Metrics JSONs: {timestamp}_{dataset}_{method}_{model}_metrics.json
        m = re.match(r"^(\d{8}_\d{6})_(.+?)_(llm|embedding|hybrid)_(.+?)_metrics\.json$", name)
        if m:
            ts, dataset, method, model = m.groups()
            groups[ts]["metrics"][(dataset, method, model)] = f
            continue

    return dict(groups)


def pick_timestamp(groups: dict, requested: str = "") -> str:
    """Pick the timestamp to analyze. If requested is given, use it; otherwise use latest."""
    if requested:
        if requested in groups:
            return requested
        # Try prefix match
        matches = [ts for ts in groups if ts.startswith(requested)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Ambiguous timestamp prefix '{requested}', matches: {matches}")
            return matches[-1]
        print(f"  WARNING: Timestamp '{requested}' not found. Available: {sorted(groups.keys())}")
        sys.exit(1)
    if not groups:
        print("  ERROR: No result files found.")
        sys.exit(1)
    return sorted(groups.keys())[-1]


# ============================================================================
#  DATA LOADING
# ============================================================================

def load_run_data(group: dict) -> tuple:
    """
    Load all prediction DataFrames and metrics dicts for one timestamp.

    Returns:
        pred_frames: dict mapping (dataset, method, model) to DataFrame
        metrics_dicts: dict mapping (dataset, method, model) to metrics dict
    """
    pred_frames = {}
    for key, path in group["predictions"].items():
        df = pd.read_csv(path, dtype=str).fillna("")
        # Ensure numeric columns
        df["match_rank"] = pd.to_numeric(df["match_rank"], errors="coerce").fillna(0).astype(int)
        pred_frames[key] = df

    metrics_dicts = {}
    for key, path in group["metrics"].items():
        with open(path, "r", encoding="utf-8") as f:
            metrics_dicts[key] = json.load(f)

    return pred_frames, metrics_dicts


# ============================================================================
#  ANALYSIS 1: Per-entity-type breakdown
# ============================================================================

def analysis_entity_type_breakdown(pred_frames: dict) -> pd.DataFrame:
    """Compute Hit@1, Hit@3, MRR by entity_type for each dataset-method pair."""
    rows = []
    for (dataset, method, model), df in sorted(pred_frames.items()):
        if df.empty:
            continue
        for etype, grp in df.groupby("entity_type"):
            if not etype:
                etype = "(unknown)"
            n = len(grp)
            hit1 = (grp["match_rank"].between(1, 1)).sum() / n
            hit3 = (grp["match_rank"].between(1, 3)).sum() / n
            rr = grp["match_rank"].apply(reciprocal_rank)
            mrr = rr.mean()
            rows.append({
                "dataset": dataset,
                "method": method,
                "model": model,
                "entity_type": etype,
                "n_samples": n,
                "Hit@1": round(hit1, 4),
                "Hit@3": round(hit3, 4),
                "MRR": round(mrr, 4),
            })
    return pd.DataFrame(rows)


# ============================================================================
#  ANALYSIS 2: Matching layer contribution
# ============================================================================

def analysis_matching_layers(pred_frames: dict) -> pd.DataFrame:
    """For each dataset-method, compute % of hits from each match tier."""
    rows = []
    for (dataset, method, model), df in sorted(pred_frames.items()):
        if df.empty:
            continue
        n = len(df)
        matched = df[df["match_rank"].astype(int) > 0]
        n_matched = len(matched)

        tier_counts = df["match_tier"].value_counts().to_dict()
        row = {
            "dataset": dataset,
            "method": method,
            "model": model,
            "n_samples": n,
            "n_matched": n_matched,
        }
        for tier in ["exact", "alias", "jaccard", "none"]:
            count = tier_counts.get(tier, 0)
            row[f"{tier}_count"] = count
            row[f"{tier}_pct"] = round(100.0 * count / n, 2) if n > 0 else 0.0
            if n_matched > 0 and tier != "none":
                row[f"{tier}_pct_of_hits"] = round(100.0 * count / n_matched, 2)
            else:
                row[f"{tier}_pct_of_hits"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
#  ANALYSIS 3: Inter-method agreement (Cohen's kappa)
# ============================================================================

def analysis_inter_method_agreement(pred_frames: dict) -> pd.DataFrame:
    """For each dataset, compute Cohen's kappa between each pair of methods on Hit@1."""
    # Group by dataset
    by_dataset = defaultdict(dict)
    for (dataset, method, model), df in pred_frames.items():
        by_dataset[dataset][(method, model)] = df

    rows = []
    for dataset, method_frames in sorted(by_dataset.items()):
        keys = sorted(method_frames.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_a = keys[i]
                key_b = keys[j]
                df_a = method_frames[key_a]
                df_b = method_frames[key_b]

                # Merge on uid
                merged = df_a[["uid", "match_rank"]].merge(
                    df_b[["uid", "match_rank"]],
                    on="uid", suffixes=("_a", "_b"),
                    how="inner",
                )
                if merged.empty:
                    continue

                # Binary: did each method get Hit@1?
                y_a = (merged["match_rank_a"].astype(int) == 1).astype(int).values
                y_b = (merged["match_rank_b"].astype(int) == 1).astype(int).values

                # Cohen's kappa
                if len(set(y_a)) < 2 and len(set(y_b)) < 2:
                    kappa = 1.0 if np.array_equal(y_a, y_b) else 0.0
                else:
                    try:
                        kappa = _cohens_kappa(y_a, y_b)
                    except Exception:
                        kappa = float("nan")

                # Also compute simple agreement %
                agreement = np.mean(y_a == y_b)

                rows.append({
                    "dataset": dataset,
                    "method_a": key_a[0],
                    "method_b": key_b[0],
                    "model_a": key_a[1],
                    "model_b": key_b[1],
                    "n_common": len(merged),
                    "cohens_kappa": round(kappa, 4),
                    "agreement_pct": round(100.0 * agreement, 2),
                })
    return pd.DataFrame(rows)


def _cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two binary arrays."""
    # Confusion matrix
    n = len(y1)
    a = np.sum((y1 == 1) & (y2 == 1))
    b = np.sum((y1 == 1) & (y2 == 0))
    c = np.sum((y1 == 0) & (y2 == 1))
    d = np.sum((y1 == 0) & (y2 == 0))

    po = (a + d) / n  # observed agreement
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)  # expected agreement
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ============================================================================
#  ANALYSIS 4: Ensemble methods
# ============================================================================

def analysis_ensembles(pred_frames: dict) -> tuple:
    """
    Compute ensemble predictions and metrics.

    Returns:
        ensemble_df: DataFrame with ensemble predictions per sample
        ensemble_metrics_df: DataFrame with ensemble metrics
    """
    # Group by dataset (combine all methods for the same dataset)
    by_dataset = defaultdict(dict)
    for (dataset, method, model), df in pred_frames.items():
        by_dataset[dataset][(method, model)] = df

    all_ensemble_rows = []
    all_metrics_rows = []

    for dataset, method_frames in sorted(by_dataset.items()):
        if len(method_frames) < 2:
            continue  # Need at least 2 methods for ensembles

        keys = sorted(method_frames.keys())
        # Find common UIDs
        uid_sets = [set(method_frames[k]["uid"].values) for k in keys]
        common_uids = uid_sets[0]
        for s in uid_sets[1:]:
            common_uids = common_uids & s
        common_uids = sorted(common_uids)

        if not common_uids:
            continue

        # Build lookup: uid -> (method -> row)
        uid_lookup = {}
        for uid in common_uids:
            uid_lookup[uid] = {}
            for key in keys:
                df = method_frames[key]
                row = df[df["uid"] == uid].iloc[0]
                uid_lookup[uid][key] = row

        # --- Majority Vote ---
        mv_rows = _ensemble_majority_vote(uid_lookup, keys, dataset)
        all_ensemble_rows.extend(mv_rows)
        all_metrics_rows.append(_compute_ensemble_metrics(mv_rows, dataset, "majority_vote"))

        # --- Reciprocal Rank Fusion (RRF) ---
        rrf_rows = _ensemble_rrf(uid_lookup, keys, dataset, k=60)
        all_ensemble_rows.extend(rrf_rows)
        all_metrics_rows.append(_compute_ensemble_metrics(rrf_rows, dataset, "rrf"))

        # --- Cascading (embedding first, fallback to LLM) ---
        cascade_rows = _ensemble_cascading(uid_lookup, keys, dataset)
        all_ensemble_rows.extend(cascade_rows)
        all_metrics_rows.append(_compute_ensemble_metrics(cascade_rows, dataset, "cascading"))

    ensemble_df = pd.DataFrame(all_ensemble_rows) if all_ensemble_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics_rows) if all_metrics_rows else pd.DataFrame()
    return ensemble_df, metrics_df


def _ensemble_majority_vote(uid_lookup, keys, dataset):
    """Majority vote: take the entity predicted most often at rank 1 across methods."""
    rows = []
    for uid, method_rows in uid_lookup.items():
        gold = method_rows[keys[0]]["gold_entity"]
        entity_type = method_rows[keys[0]]["entity_type"]

        # Collect rank-1 predictions from each method
        votes = []
        for key in keys:
            pred_1 = method_rows[key].get("pred_1", "")
            if pred_1:
                votes.append(pred_1)

        # Count votes (normalized)
        vote_counts = defaultdict(int)
        vote_original = {}
        for v in votes:
            nv = normalize(v)
            vote_counts[nv] += 1
            if nv not in vote_original:
                vote_original[nv] = v

        # Pick the entity with most votes (ties broken by first occurrence)
        best_pred = ""
        best_count = 0
        for nv, count in vote_counts.items():
            if count > best_count:
                best_count = count
                best_pred = vote_original[nv]

        # Determine if it matches
        match_rank, match_tier = _check_match(best_pred, gold)

        rows.append({
            "uid": uid,
            "dataset": dataset,
            "ensemble": "majority_vote",
            "gold_entity": gold,
            "entity_type": entity_type,
            "pred_1": best_pred,
            "match_rank": match_rank,
            "match_tier": match_tier,
            "vote_count": best_count,
            "n_methods": len(keys),
        })
    return rows


def _ensemble_rrf(uid_lookup, keys, dataset, k=60):
    """Reciprocal Rank Fusion: score(e) = sum(1/(k + rank_in_method))."""
    rows = []
    for uid, method_rows in uid_lookup.items():
        gold = method_rows[keys[0]]["gold_entity"]
        entity_type = method_rows[keys[0]]["entity_type"]

        # Accumulate RRF scores across methods
        rrf_scores = defaultdict(float)
        original_names = {}
        for key in keys:
            row = method_rows[key]
            for rank_idx in range(1, 11):
                pred_col = f"pred_{rank_idx}"
                pred = row.get(pred_col, "")
                if not pred:
                    continue
                npred = normalize(pred)
                rrf_scores[npred] += 1.0 / (k + rank_idx)
                if npred not in original_names:
                    original_names[npred] = pred

        # Sort by RRF score descending
        sorted_entities = sorted(rrf_scores.items(), key=lambda x: -x[1])

        # Build ranked prediction list
        preds = [original_names[ne] for ne, _ in sorted_entities[:10]]
        best_pred = preds[0] if preds else ""

        # Check match for top prediction
        match_rank = 0
        match_tier = "none"
        for i, p in enumerate(preds, 1):
            mr, mt = _check_match(p, gold)
            if mr > 0:
                match_rank = i
                match_tier = mt
                break

        row_out = {
            "uid": uid,
            "dataset": dataset,
            "ensemble": "rrf",
            "gold_entity": gold,
            "entity_type": entity_type,
            "pred_1": best_pred,
            "match_rank": match_rank,
            "match_tier": match_tier,
        }
        for i, p in enumerate(preds[:10], 1):
            row_out[f"pred_{i}"] = p
        rows.append(row_out)
    return rows


def _ensemble_cascading(uid_lookup, keys, dataset):
    """Cascading: use embedding first, fall back to LLM on misses."""
    # Determine priority: embedding > hybrid > llm
    priority_order = ["embedding", "hybrid", "llm"]
    sorted_keys = sorted(keys, key=lambda k: (
        priority_order.index(k[0]) if k[0] in priority_order else 99
    ))

    rows = []
    for uid, method_rows in uid_lookup.items():
        gold = method_rows[keys[0]]["gold_entity"]
        entity_type = method_rows[keys[0]]["entity_type"]

        # Try each method in priority order
        chosen_pred = ""
        chosen_rank = 0
        chosen_tier = "none"
        chosen_method = ""

        for key in sorted_keys:
            row = method_rows[key]
            rank = int(row.get("match_rank", 0) or 0)
            if rank == 1:
                chosen_pred = row.get("pred_1", "")
                chosen_rank = 1
                chosen_tier = row.get("match_tier", "none")
                chosen_method = key[0]
                break

        # If no method got rank 1, use first method's predictions
        if chosen_rank == 0:
            first_row = method_rows[sorted_keys[0]]
            chosen_pred = first_row.get("pred_1", "")
            chosen_method = sorted_keys[0][0]
            # Check match
            chosen_rank_check, chosen_tier_check = _check_match(chosen_pred, gold)
            if chosen_rank_check > 0:
                chosen_rank = chosen_rank_check
                chosen_tier = chosen_tier_check

        rows.append({
            "uid": uid,
            "dataset": dataset,
            "ensemble": "cascading",
            "gold_entity": gold,
            "entity_type": entity_type,
            "pred_1": chosen_pred,
            "match_rank": chosen_rank,
            "match_tier": chosen_tier,
            "chosen_method": chosen_method,
        })
    return rows


def _check_match(pred: str, gold: str) -> tuple:
    """Simple check: exact or jaccard match. Returns (rank, tier). Rank is 1 or 0."""
    if not pred or not gold:
        return 0, "none"
    if normalize(pred) == normalize(gold):
        return 1, "exact"
    if jaccard_token_similarity(pred, gold) >= 0.60:
        return 1, "jaccard"
    return 0, "none"


def _compute_ensemble_metrics(rows: list, dataset: str, ensemble_name: str) -> dict:
    """Compute basic metrics for ensemble rows."""
    n = len(rows)
    if n == 0:
        return {"dataset": dataset, "ensemble": ensemble_name}

    hit1 = sum(1 for r in rows if r["match_rank"] == 1) / n
    hit3 = sum(1 for r in rows if 0 < r["match_rank"] <= 3) / n
    mrr_vals = [reciprocal_rank(r["match_rank"]) for r in rows]
    mrr = np.mean(mrr_vals)

    return {
        "dataset": dataset,
        "ensemble": ensemble_name,
        "n_samples": n,
        "Hit@1": round(hit1, 4),
        "Hit@3": round(hit3, 4),
        "Global_MRR": round(mrr, 4),
    }


# ============================================================================
#  ANALYSIS 5: Text length effects
# ============================================================================

def analysis_text_length(pred_frames: dict) -> pd.DataFrame:
    """Bin samples into short/medium/long by token count, report Hit@1 and MRR per bin."""
    rows = []
    for (dataset, method, model), df in sorted(pred_frames.items()):
        if df.empty:
            continue

        df = df.copy()
        df["token_count"] = df["text"].apply(token_count)

        # Compute tercile boundaries
        try:
            df["length_bin"] = pd.qcut(
                df["token_count"], q=3, labels=["short", "medium", "long"],
                duplicates="drop",
            )
        except ValueError:
            # If too few unique values, use simple cut
            df["length_bin"] = pd.cut(
                df["token_count"], bins=3, labels=["short", "medium", "long"],
            )

        for bin_label, grp in df.groupby("length_bin", observed=True):
            n = len(grp)
            if n == 0:
                continue
            hit1 = (grp["match_rank"].between(1, 1)).sum() / n
            rr = grp["match_rank"].apply(reciprocal_rank)
            mrr = rr.mean()
            avg_tokens = grp["token_count"].mean()
            rows.append({
                "dataset": dataset,
                "method": method,
                "model": model,
                "length_bin": str(bin_label),
                "n_samples": n,
                "avg_tokens": round(avg_tokens, 1),
                "Hit@1": round(hit1, 4),
                "MRR": round(mrr, 4),
            })
    return pd.DataFrame(rows)


# ============================================================================
#  ANALYSIS 6: Per-narrator variance (veterans only)
# ============================================================================

def analysis_narrator_variance(pred_frames: dict) -> pd.DataFrame:
    """
    For veterans datasets, load source dataset to get narrator (source column),
    then compute Hit@1 per narrator per method.
    """
    # Load the veterans dataset for narrator info
    veterans_path = SCRIPT_DIR.parent / "data" / "implicit_reference_veterans_dataset.csv"
    if not veterans_path.exists():
        print("  [Analysis 6] Veterans dataset not found; skipping narrator analysis.")
        return pd.DataFrame()

    vet_df = pd.read_csv(veterans_path, dtype=str).fillna("")
    uid_to_narrator = dict(zip(vet_df["uid"].astype(str), vet_df["source"].astype(str)))

    rows = []
    for (dataset, method, model), df in sorted(pred_frames.items()):
        if "veteran" not in dataset.lower():
            continue
        if df.empty:
            continue

        df = df.copy()
        df["narrator"] = df["uid"].map(uid_to_narrator).fillna("(unknown)")

        narrator_stats = []
        for narrator, grp in df.groupby("narrator"):
            if not narrator or narrator == "(unknown)":
                continue
            n = len(grp)
            if n < 2:
                continue
            hit1 = (grp["match_rank"].between(1, 1)).sum() / n
            narrator_stats.append({
                "narrator": narrator,
                "n_samples": n,
                "Hit@1": hit1,
            })

        if not narrator_stats:
            continue

        hit1_values = [s["Hit@1"] for s in narrator_stats]
        rows.append({
            "dataset": dataset,
            "method": method,
            "model": model,
            "n_narrators": len(narrator_stats),
            "mean_Hit@1": round(np.mean(hit1_values), 4),
            "std_Hit@1": round(np.std(hit1_values, ddof=1) if len(hit1_values) > 1 else 0.0, 4),
            "min_Hit@1": round(np.min(hit1_values), 4),
            "max_Hit@1": round(np.max(hit1_values), 4),
            "median_Hit@1": round(np.median(hit1_values), 4),
        })

        # Also save per-narrator detail
        for ns in narrator_stats:
            rows.append({
                "dataset": dataset,
                "method": method,
                "model": model,
                "narrator": ns["narrator"],
                "n_samples": ns["n_samples"],
                "Hit@1": round(ns["Hit@1"], 4),
            })

    return pd.DataFrame(rows)


# ============================================================================
#  ANALYSIS 7: Bootstrap confidence intervals
# ============================================================================

def analysis_bootstrap_ci(pred_frames: dict, n_bootstrap: int = 1000, ci: float = 0.95) -> pd.DataFrame:
    """Compute 95% CI on Hit@1 and MRR using bootstrap resampling."""
    rng = np.random.default_rng(seed=42)
    alpha = 1.0 - ci
    rows = []

    for (dataset, method, model), df in sorted(pred_frames.items()):
        if df.empty:
            continue

        n = len(df)
        ranks = df["match_rank"].astype(int).values

        # Compute bootstrap distributions
        hit1_boots = np.zeros(n_bootstrap)
        mrr_boots = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            sampled_ranks = ranks[indices]
            hit1_boots[b] = np.mean(sampled_ranks == 1)
            rrs = np.where(sampled_ranks > 0, 1.0 / sampled_ranks, 0.0)
            mrr_boots[b] = np.mean(rrs)

        hit1_lo, hit1_hi = np.percentile(hit1_boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        mrr_lo, mrr_hi = np.percentile(mrr_boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        # Point estimates
        hit1_point = np.mean(ranks == 1)
        rr_vals = np.where(ranks > 0, 1.0 / ranks, 0.0)
        mrr_point = np.mean(rr_vals)

        rows.append({
            "dataset": dataset,
            "method": method,
            "model": model,
            "n_samples": n,
            "Hit@1": round(hit1_point, 4),
            "Hit@1_CI_lo": round(hit1_lo, 4),
            "Hit@1_CI_hi": round(hit1_hi, 4),
            "MRR": round(mrr_point, 4),
            "MRR_CI_lo": round(mrr_lo, 4),
            "MRR_CI_hi": round(mrr_hi, 4),
            "n_bootstrap": n_bootstrap,
            "ci_level": ci,
        })

    return pd.DataFrame(rows)


# ============================================================================
#  ANALYSIS 8: nDCG@K
# ============================================================================

def analysis_ndcg(pred_frames: dict) -> pd.DataFrame:
    """Compute nDCG@1,3,5,10 using Jaccard similarity as graded relevance."""
    rows = []
    for (dataset, method, model), df in sorted(pred_frames.items()):
        if df.empty:
            continue

        ndcg_sums = {k: 0.0 for k in [1, 3, 5, 10]}
        n = len(df)

        for _, row in df.iterrows():
            gold = str(row["gold_entity"])
            # Gather predictions
            preds = []
            for i in range(1, 11):
                p = str(row.get(f"pred_{i}", ""))
                if p:
                    preds.append(p)

            # Compute relevance scores using Jaccard similarity
            relevances = [jaccard_token_similarity(p, gold) for p in preds]

            for k in [1, 3, 5, 10]:
                rels_at_k = relevances[:k]
                if not rels_at_k:
                    continue
                dcg = _dcg(rels_at_k)
                # Ideal: sort relevances descending
                ideal_rels = sorted(relevances[:k], reverse=True)
                idcg = _dcg(ideal_rels)
                if idcg > 0:
                    ndcg_sums[k] += dcg / idcg
                elif dcg == 0:
                    ndcg_sums[k] += 0.0
                # If all relevances are 0, both dcg and idcg are 0: count as 0

        row_out = {
            "dataset": dataset,
            "method": method,
            "model": model,
            "n_samples": n,
        }
        for k in [1, 3, 5, 10]:
            row_out[f"nDCG@{k}"] = round(ndcg_sums[k] / n, 4) if n > 0 else 0.0
        rows.append(row_out)

    return pd.DataFrame(rows)


def _dcg(relevances: list) -> float:
    """Compute DCG for a list of relevance scores."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


# ============================================================================
#  ANALYSIS 9: Dataset difficulty (embedding similarity)
# ============================================================================

def analysis_dataset_difficulty(pred_frames: dict) -> pd.DataFrame:
    """
    For each sample, compute cosine similarity between text embedding and entity name
    embedding. Correlate with recognition success.

    Requires sentence-transformers; returns empty DataFrame if unavailable.
    """
    if not _SBERT_AVAILABLE:
        print("  [Analysis 9] sentence-transformers not available; skipping dataset difficulty.")
        return pd.DataFrame()

    print("  [Analysis 9] Loading sentence-transformers model for difficulty analysis...")
    try:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"  [Analysis 9] Failed to load model: {e}")
        return pd.DataFrame()

    rows = []
    for (dataset, method, model), df in sorted(pred_frames.items()):
        if df.empty:
            continue

        texts = df["text"].tolist()
        golds = df["gold_entity"].tolist()
        ranks = df["match_rank"].astype(int).values

        print(f"  [Analysis 9] Encoding {len(texts)} texts for {dataset}/{method}...")
        text_embs = sbert.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        gold_embs = sbert.encode(golds, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

        # Cosine similarity (already normalized)
        cosine_sims = np.sum(text_embs * gold_embs, axis=1)

        # Binary success: Hit@1
        success = (ranks == 1).astype(int)

        # Correlation
        if np.std(cosine_sims) > 0 and np.std(success) > 0:
            corr, p_value = scipy_stats.pearsonr(cosine_sims, success)
        else:
            corr, p_value = 0.0, 1.0

        # Bin by difficulty
        try:
            bins = pd.qcut(cosine_sims, q=4, labels=["low_sim", "med_low_sim", "med_high_sim", "high_sim"], duplicates="drop")
        except ValueError:
            bins = pd.cut(cosine_sims, bins=4, labels=["low_sim", "med_low_sim", "med_high_sim", "high_sim"])

        for bin_label in ["low_sim", "med_low_sim", "med_high_sim", "high_sim"]:
            mask = bins == bin_label
            if mask.sum() == 0:
                continue
            bin_hit1 = success[mask].mean()
            bin_sim = cosine_sims[mask].mean()
            rows.append({
                "dataset": dataset,
                "method": method,
                "model": model,
                "similarity_bin": str(bin_label),
                "n_samples": int(mask.sum()),
                "avg_cosine_sim": round(float(bin_sim), 4),
                "Hit@1": round(float(bin_hit1), 4),
                "pearson_r": round(float(corr), 4),
                "pearson_p": round(float(p_value), 6),
            })

    return pd.DataFrame(rows)


# ============================================================================
#  ANALYSIS 10: Summary tables
# ============================================================================

def generate_summary_tables(
    metrics_dicts: dict,
    entity_type_df: pd.DataFrame,
    matching_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    ndcg_df: pd.DataFrame,
    ensemble_metrics_df: pd.DataFrame,
) -> tuple:
    """Produce combined results table in CSV and formatted Markdown."""

    # --- Main results CSV ---
    main_rows = []
    for (dataset, method, model), m in sorted(metrics_dicts.items()):
        main_rows.append({
            "dataset": dataset,
            "method": method,
            "model": model,
            "n_samples": m.get("n_samples", 0),
            "Hit@1": m.get("Hit@1", 0),
            "Hit@3": m.get("Hit@3", 0),
            "Hit@5": m.get("Hit@5", 0),
            "Hit@10": m.get("Hit@10", 0),
            "Global_MRR": m.get("Global_MRR", 0),
            "Filtered_MRR": m.get("Filtered_MRR", 0),
            "elapsed_seconds": m.get("elapsed_seconds", 0),
        })
    main_df = pd.DataFrame(main_rows)

    # --- Markdown ---
    lines = []
    lines.append("# Implicit Entity Recognition: Experiment Results")
    lines.append("")

    # Main results table
    lines.append("## Main Results")
    lines.append("")
    if not main_df.empty:
        lines.append(_df_to_markdown(main_df, float_fmt=".4f"))
    lines.append("")

    # Bootstrap CIs
    if not bootstrap_df.empty:
        lines.append("## Bootstrap 95% Confidence Intervals")
        lines.append("")
        cols = ["dataset", "method", "Hit@1", "Hit@1_CI_lo", "Hit@1_CI_hi", "MRR", "MRR_CI_lo", "MRR_CI_hi"]
        available_cols = [c for c in cols if c in bootstrap_df.columns]
        lines.append(_df_to_markdown(bootstrap_df[available_cols], float_fmt=".4f"))
        lines.append("")

    # nDCG
    if not ndcg_df.empty:
        lines.append("## nDCG Scores")
        lines.append("")
        lines.append(_df_to_markdown(ndcg_df, float_fmt=".4f"))
        lines.append("")

    # Matching layers
    if not matching_df.empty:
        lines.append("## Matching Layer Contribution")
        lines.append("")
        cols = ["dataset", "method", "exact_pct_of_hits", "alias_pct_of_hits", "jaccard_pct_of_hits"]
        available_cols = [c for c in cols if c in matching_df.columns]
        if available_cols:
            lines.append(_df_to_markdown(matching_df[available_cols], float_fmt=".2f"))
        lines.append("")

    # Ensemble results
    if not ensemble_metrics_df.empty:
        lines.append("## Ensemble Methods")
        lines.append("")
        lines.append(_df_to_markdown(ensemble_metrics_df, float_fmt=".4f"))
        lines.append("")

    # Entity type breakdown
    if not entity_type_df.empty:
        lines.append("## Per-Entity-Type Breakdown")
        lines.append("")
        lines.append(_df_to_markdown(entity_type_df, float_fmt=".4f"))
        lines.append("")

    markdown_text = "\n".join(lines)
    return main_df, markdown_text


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Convert a DataFrame to a Markdown table string."""
    if df.empty:
        return "(no data)"

    cols = list(df.columns)
    # Header
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:{float_fmt}}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of implicit entity recognition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--timestamp", default="",
        help="Analyze a specific experiment run by timestamp prefix (default: latest)",
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help=f"Directory containing result files (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for confidence intervals (default: 1000)",
    )
    parser.add_argument(
        "--skip-difficulty", action="store_true",
        help="Skip dataset difficulty analysis (analysis 9) even if sentence-transformers is available",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"  ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    print("=" * 65)
    print("  Implicit Entity Recognition: Post-Hoc Analysis")
    print("=" * 65)

    # Discover files
    groups = discover_result_files(results_dir)
    if not groups:
        print("  ERROR: No result files found in", results_dir)
        sys.exit(1)

    print(f"\n  Found {len(groups)} experiment run(s): {sorted(groups.keys())}")

    timestamp = pick_timestamp(groups, args.timestamp)
    group = groups[timestamp]

    n_pred = len(group["predictions"])
    n_met = len(group["metrics"])
    print(f"\n  Analyzing run: {timestamp}")
    print(f"  Prediction CSVs:  {n_pred}")
    print(f"  Metrics JSONs:    {n_met}")

    if n_pred == 0:
        print("  ERROR: No prediction files for this timestamp.")
        sys.exit(1)

    # Load data
    pred_frames, metrics_dicts = load_run_data(group)

    # Create output directory
    out_dir = ANALYSIS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory:  {out_dir}")

    # List dataset-method combos
    combos = sorted(pred_frames.keys())
    print(f"\n  Dataset-method combinations ({len(combos)}):")
    for ds, meth, mdl in combos:
        n = len(pred_frames[(ds, meth, mdl)])
        print(f"    {ds:20s} / {meth:10s} / {mdl}  ({n} samples)")

    # ---- Run analyses ----

    print("\n" + "-" * 65)
    print("  [1/10] Per-entity-type breakdown")
    entity_type_df = analysis_entity_type_breakdown(pred_frames)
    if not entity_type_df.empty:
        entity_type_df.to_csv(out_dir / "entity_type_breakdown.csv", index=False)
        print(f"         Saved: entity_type_breakdown.csv ({len(entity_type_df)} rows)")
    else:
        print("         (no data)")

    print("  [2/10] Matching layer contribution")
    matching_df = analysis_matching_layers(pred_frames)
    if not matching_df.empty:
        matching_df.to_csv(out_dir / "matching_layer_contribution.csv", index=False)
        print(f"         Saved: matching_layer_contribution.csv ({len(matching_df)} rows)")
    else:
        print("         (no data)")

    print("  [3/10] Inter-method agreement")
    agreement_df = analysis_inter_method_agreement(pred_frames)
    if not agreement_df.empty:
        agreement_df.to_csv(out_dir / "inter_method_agreement.csv", index=False)
        print(f"         Saved: inter_method_agreement.csv ({len(agreement_df)} rows)")
    else:
        print("         (no data, need 2+ methods for same dataset)")

    print("  [4/10] Ensemble methods")
    ensemble_pred_df, ensemble_metrics_df = analysis_ensembles(pred_frames)
    if not ensemble_pred_df.empty:
        ensemble_pred_df.to_csv(out_dir / "ensemble_predictions.csv", index=False)
        print(f"         Saved: ensemble_predictions.csv ({len(ensemble_pred_df)} rows)")
    if not ensemble_metrics_df.empty:
        ensemble_metrics_df.to_csv(out_dir / "ensemble_metrics.csv", index=False)
        print(f"         Saved: ensemble_metrics.csv ({len(ensemble_metrics_df)} rows)")
    if ensemble_pred_df.empty:
        print("         (no data, need 2+ methods for same dataset)")

    print("  [5/10] Text length effects")
    length_df = analysis_text_length(pred_frames)
    if not length_df.empty:
        length_df.to_csv(out_dir / "text_length_effects.csv", index=False)
        print(f"         Saved: text_length_effects.csv ({len(length_df)} rows)")
    else:
        print("         (no data)")

    print("  [6/10] Per-narrator variance (veterans)")
    narrator_df = analysis_narrator_variance(pred_frames)
    if not narrator_df.empty:
        narrator_df.to_csv(out_dir / "narrator_variance.csv", index=False)
        print(f"         Saved: narrator_variance.csv ({len(narrator_df)} rows)")
    else:
        print("         (no veterans data or dataset not found)")

    print(f"  [7/10] Bootstrap confidence intervals (n={args.n_bootstrap})")
    bootstrap_df = analysis_bootstrap_ci(pred_frames, n_bootstrap=args.n_bootstrap)
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(out_dir / "bootstrap_ci.csv", index=False)
        print(f"         Saved: bootstrap_ci.csv ({len(bootstrap_df)} rows)")
    else:
        print("         (no data)")

    print("  [8/10] nDCG@K")
    ndcg_df = analysis_ndcg(pred_frames)
    if not ndcg_df.empty:
        ndcg_df.to_csv(out_dir / "ndcg_scores.csv", index=False)
        print(f"         Saved: ndcg_scores.csv ({len(ndcg_df)} rows)")
    else:
        print("         (no data)")

    print("  [9/10] Dataset difficulty (embedding similarity)")
    if args.skip_difficulty:
        difficulty_df = pd.DataFrame()
        print("         Skipped (--skip-difficulty)")
    else:
        difficulty_df = analysis_dataset_difficulty(pred_frames)
        if not difficulty_df.empty:
            difficulty_df.to_csv(out_dir / "dataset_difficulty.csv", index=False)
            print(f"         Saved: dataset_difficulty.csv ({len(difficulty_df)} rows)")
        else:
            print("         (no data or sentence-transformers unavailable)")

    print("  [10/10] Summary tables")
    main_results_df, markdown_text = generate_summary_tables(
        metrics_dicts, entity_type_df, matching_df, bootstrap_df, ndcg_df, ensemble_metrics_df,
    )
    if not main_results_df.empty:
        main_results_df.to_csv(out_dir / "summary_results.csv", index=False)
        print(f"          Saved: summary_results.csv ({len(main_results_df)} rows)")
    with open(out_dir / "summary_results.md", "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"          Saved: summary_results.md")

    # ---- Print summary to stdout ----

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    if not main_results_df.empty:
        header = f"  {'Dataset':<20s} {'Method':<12s} {'Hit@1':>7s} {'Hit@3':>7s} {'MRR':>7s} {'Time':>7s}"
        print(header)
        print(f"  {'-' * 60}")
        for _, row in main_results_df.iterrows():
            print(f"  {row['dataset']:<20s} {row['method']:<12s} "
                  f"{row['Hit@1']:>7.3f} {row['Hit@3']:>7.3f} "
                  f"{row['Global_MRR']:>7.3f} {row['elapsed_seconds']:>6.0f}s")

    if not ensemble_metrics_df.empty:
        print(f"\n  Ensemble results:")
        print(f"  {'-' * 50}")
        for _, row in ensemble_metrics_df.iterrows():
            ds = row.get("dataset", "")
            ens = row.get("ensemble", "")
            h1 = row.get("Hit@1", 0)
            mrr = row.get("Global_MRR", 0)
            print(f"  {ds:<20s} {ens:<16s} Hit@1={h1:.3f}  MRR={mrr:.3f}")

    if not bootstrap_df.empty:
        print(f"\n  95% Confidence intervals:")
        print(f"  {'-' * 60}")
        for _, row in bootstrap_df.iterrows():
            ds = row.get("dataset", "")
            meth = row.get("method", "")
            h1 = row.get("Hit@1", 0)
            h1_lo = row.get("Hit@1_CI_lo", 0)
            h1_hi = row.get("Hit@1_CI_hi", 0)
            mrr = row.get("MRR", 0)
            mrr_lo = row.get("MRR_CI_lo", 0)
            mrr_hi = row.get("MRR_CI_hi", 0)
            print(f"  {ds:<20s} {meth:<10s}  "
                  f"Hit@1={h1:.3f} [{h1_lo:.3f}, {h1_hi:.3f}]  "
                  f"MRR={mrr:.3f} [{mrr_lo:.3f}, {mrr_hi:.3f}]")

    print(f"\n  All outputs saved to: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
