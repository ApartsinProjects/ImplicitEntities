"""
C3: Ensemble methods from existing predictions (NO new API calls).

Combines LLM, embedding, and hybrid predictions using:
  1. Union: take all unique predictions from all methods
  2. Majority vote: entity appears in 2+ methods' top-3
  3. Reciprocal Rank Fusion (RRF): score = sum(1/(k + rank_in_method))
  4. Cascading: LLM first, fall back to hybrid on LLM misses

Usage:
    python run_ensemble.py --dataset veterans_t2e
    python run_ensemble.py --dataset all
"""
import asyncio
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    Sample, load_dataset, evaluate_predictions, compute_metrics,
    print_metrics, save_results, RESULTS_DIR,
)

TIMESTAMP_PREFIX = "20260416_092603"  # main experiment run


def load_predictions(dataset: str, method: str, timestamp: str = TIMESTAMP_PREFIX) -> dict:
    """Load predictions CSV for a dataset-method pair. Returns uid -> [pred_1..pred_10]."""
    pattern = f"{timestamp}_{dataset}_{method}_*_predictions.csv"
    files = list(RESULTS_DIR.glob(pattern))
    if not files:
        return {}
    pred_map = {}
    with open(files[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds = [row.get(f"pred_{i}", "").strip() for i in range(1, 11)]
            preds = [p for p in preds if p]
            pred_map[row["uid"]] = preds
    return pred_map


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank) if rank > 0 else 0.0


def ensemble_rrf(method_preds: dict[str, dict], uids: list[str], top_k: int = 10) -> dict:
    """Reciprocal Rank Fusion across methods."""
    results = {}
    for uid in uids:
        entity_scores = defaultdict(float)
        for method, preds_map in method_preds.items():
            preds = preds_map.get(uid, [])
            for rank, entity in enumerate(preds, 1):
                entity_scores[entity.lower()] += rrf_score(rank)
                # Also store original casing
                entity_scores[f"_orig_{entity.lower()}"] = entity

        # Sort by score, return top_k
        scored = [(s, e) for e, s in entity_scores.items() if not e.startswith("_orig_")]
        scored.sort(key=lambda x: -x[0])
        # Get original casing back
        ranked = []
        for score, entity_lower in scored[:top_k]:
            orig = entity_scores.get(f"_orig_{entity_lower}", entity_lower)
            ranked.append(orig)
        results[uid] = ranked
    return results


def ensemble_majority(method_preds: dict[str, dict], uids: list[str], top_k: int = 10) -> dict:
    """Majority vote: entities that appear in 2+ methods' predictions."""
    results = {}
    n_methods = len(method_preds)
    for uid in uids:
        entity_votes = Counter()
        entity_best_rank = {}
        for method, preds_map in method_preds.items():
            preds = preds_map.get(uid, [])
            for rank, entity in enumerate(preds, 1):
                key = entity.lower()
                entity_votes[key] += 1
                if key not in entity_best_rank or rank < entity_best_rank[key]:
                    entity_best_rank[key] = rank
                    entity_votes[f"_orig_{key}"] = entity

        # Sort by votes (desc), then by best rank (asc)
        scored = [(entity_votes[e], -entity_best_rank.get(e, 999), e)
                  for e in entity_votes if not e.startswith("_orig_")]
        scored.sort(key=lambda x: (-x[0], x[1]))
        ranked = [entity_votes.get(f"_orig_{e}", e) for _, _, e in scored[:top_k]]
        results[uid] = ranked
    return results


def ensemble_cascade(primary_preds: dict, fallback_preds: dict, uids: list[str],
                     samples_by_uid: dict, top_k: int = 10) -> dict:
    """Cascading: use primary method, fall back to secondary on misses."""
    results = {}
    for uid in uids:
        primary = primary_preds.get(uid, [])
        if primary:
            results[uid] = primary[:top_k]
        else:
            results[uid] = fallback_preds.get(uid, [])[:top_k]
    return results


def ensemble_union(method_preds: dict[str, dict], uids: list[str], top_k: int = 10) -> dict:
    """Union: all unique predictions from all methods, ordered by first appearance."""
    results = {}
    for uid in uids:
        seen = set()
        ordered = []
        for method in ["llm", "hybrid", "embedding"]:
            preds_map = method_preds.get(method, {})
            for entity in preds_map.get(uid, []):
                key = entity.lower()
                if key not in seen:
                    seen.add(key)
                    ordered.append(entity)
        results[uid] = ordered[:top_k]
    return results


async def run_ensemble_experiments(dataset_name: str):
    """Run all ensemble methods for a dataset."""
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE EXPERIMENTS: {dataset_name}")
    print(f"{'='*60}")

    # Load base predictions
    llm_preds = load_predictions(dataset_name, "llm")
    emb_preds = load_predictions(dataset_name, "embedding")
    hyb_preds = load_predictions(dataset_name, "hybrid")

    if not llm_preds:
        print(f"  ERROR: No LLM predictions found for {dataset_name}")
        return {}

    print(f"  Loaded predictions: LLM={len(llm_preds)}, Embedding={len(emb_preds)}, Hybrid={len(hyb_preds)}")

    # Load samples for evaluation
    samples, unique_entities = load_dataset(dataset_name)
    if not samples:
        print(f"  ERROR: Could not load dataset {dataset_name}")
        return {}

    samples_by_uid = {s.uid: s for s in samples}
    uids = [s.uid for s in samples]

    method_preds = {"llm": llm_preds, "embedding": emb_preds, "hybrid": hyb_preds}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics = {}

    # Run each ensemble strategy
    ensembles = {
        "ensemble_rrf": lambda: ensemble_rrf(method_preds, uids),
        "ensemble_majority": lambda: ensemble_majority(method_preds, uids),
        "ensemble_union": lambda: ensemble_union(method_preds, uids),
        "ensemble_cascade_llm_hybrid": lambda: ensemble_cascade(llm_preds, hyb_preds, uids, samples_by_uid),
    }

    for name, ensemble_fn in ensembles.items():
        print(f"\n  --- {name} ---")
        pred_map = ensemble_fn()

        # Convert to list of predictions aligned with samples
        predictions = [pred_map.get(s.uid, []) for s in samples]

        # Evaluate
        eval_results = await evaluate_predictions(
            samples, predictions, model="ensemble", concurrency=15,
        )
        metrics = compute_metrics(eval_results)
        print_metrics(metrics, f"{dataset_name} / {name}")

        save_results(eval_results, metrics, dataset_name, name, "ensemble", timestamp)
        all_metrics[name] = metrics

    # Print comparison
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE COMPARISON: {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Method':<35s} {'Hit@1':>7s} {'Hit@3':>7s} {'MRR':>7s}")
    print(f"  {'-'*56}")

    # Include base methods for reference
    for method_name, preds_map in [("llm (baseline)", llm_preds), ("embedding", emb_preds), ("hybrid", hyb_preds)]:
        predictions = [preds_map.get(s.uid, []) for s in samples]
        eval_results = await evaluate_predictions(samples, predictions, model="ref", concurrency=15)
        m = compute_metrics(eval_results)
        print(f"  {method_name:<35s} {m['Hit@1']:>7.3f} {m['Hit@3']:>7.3f} {m['Global_MRR']:>7.3f}")

    for name, m in all_metrics.items():
        print(f"  {name:<35s} {m['Hit@1']:>7.3f} {m['Hit@3']:>7.3f} {m['Global_MRR']:>7.3f}")
    print(f"{'='*70}")

    # Save summary
    summary_path = RESULTS_DIR / f"{timestamp}_ensemble_summary_{dataset_name}.json"
    with open(summary_path, "w") as f:
        json.dump({"dataset": dataset_name, "results": all_metrics}, f, indent=2)

    return all_metrics


async def main():
    parser = argparse.ArgumentParser(description="Run ensemble experiments from existing predictions")
    parser.add_argument("--dataset", default="veterans_t2e",
                        help="Dataset name or 'all'")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["veterans_t2e", "twitter", "e2t_veterans", "e2t_twitter"]
    else:
        datasets = [args.dataset]

    for ds in datasets:
        await run_ensemble_experiments(ds)


if __name__ == "__main__":
    asyncio.run(main())
