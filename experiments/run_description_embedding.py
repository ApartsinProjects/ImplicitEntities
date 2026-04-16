"""
C1: Entity description embedding experiment.

Instead of comparing text embeddings to bare entity NAME embeddings (which gives
terrible results, ~0.027 Hit@1), compare to entity DESCRIPTION embeddings.

Uses pre-generated descriptions from experiments/entity_descriptions/veterans_v2_descriptions.json

Usage:
    python run_description_embedding.py --dataset veterans_t2e_v2
    python run_description_embedding.py --dataset veterans_t2e_v2 --max-samples 50
    python run_description_embedding.py --dataset veterans_t2e_v2 --compare-baseline
"""

import asyncio
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_experiments import (
    load_dataset,
    Sample,
    evaluate_predictions,
    compute_metrics,
    print_metrics,
    save_results,
    compute_embeddings,
    embedding_rank_entities,
    sbert_available,
    RESULTS_DIR,
)

DESCRIPTIONS_DIR = Path(__file__).parent / "entity_descriptions"
DEFAULT_DESCRIPTIONS = DESCRIPTIONS_DIR / "veterans_v2_descriptions.json"


def load_descriptions(path: Path) -> dict[str, dict]:
    """
    Load entity descriptions JSON.
    Returns dict mapping normalized entity name -> {entity, entity_type, description}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entity descriptions from {path.name}")
    return data


def build_description_lookup(
    descriptions: dict[str, dict],
    unique_entities: list[str],
) -> tuple[list[str], list[str]]:
    """
    For each unique entity in the dataset, find its description.
    Returns (entity_names, description_texts) as parallel lists.
    Entities without descriptions fall back to their bare name.
    """
    entity_names = []
    desc_texts = []
    matched = 0
    for entity in unique_entities:
        key = entity.lower().strip()
        if key in descriptions and descriptions[key].get("description"):
            entity_names.append(entity)
            desc_texts.append(descriptions[key]["description"])
            matched += 1
        else:
            # Fallback: use entity name as its own description
            entity_names.append(entity)
            desc_texts.append(entity)

    print(f"  Description coverage: {matched}/{len(unique_entities)} entities "
          f"({100 * matched / max(len(unique_entities), 1):.1f}%)")
    return entity_names, desc_texts


async def run_description_embedding(
    samples: list[Sample],
    unique_entities: list[str],
    descriptions: dict[str, dict],
    model_name: str,
    dataset_name: str,
    timestamp: str,
    eval_model: str = "google/gemini-2.0-flash-001:floor",
    concurrency: int = 10,
) -> dict:
    """
    Run C1: description-based embedding ranking.
    Returns metrics dict.
    """
    print(f"\n{'#' * 60}")
    print(f"  C1: Entity Description Embeddings")
    print(f"  Dataset: {dataset_name} ({len(samples)} samples)")
    print(f"  Embedding model: {model_name}")
    print(f"{'#' * 60}")

    t0 = time.time()

    # Build description texts for entities
    entity_names, desc_texts = build_description_lookup(descriptions, unique_entities)

    # Compute text embeddings
    print("\n  Computing text embeddings...")
    text_strings = [s.text for s in samples]
    text_embs = compute_embeddings(text_strings)

    # Compute entity DESCRIPTION embeddings
    print("  Computing entity description embeddings...")
    desc_embs = compute_embeddings(desc_texts)

    # Rank entities by cosine similarity
    print("  Ranking entities by cosine similarity (text vs description)...")
    predictions = embedding_rank_entities(text_embs, desc_embs, entity_names, top_k=10)

    # Evaluate
    eval_results = await evaluate_predictions(
        samples, predictions, model=eval_model, concurrency=concurrency,
    )

    metrics = compute_metrics(eval_results)
    label = f"{dataset_name} / C1 description embedding / {model_name}"
    print_metrics(metrics, label)

    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["method_detail"] = "C1_description_embedding"
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    method_name = "embedding_description"
    save_results(eval_results, metrics, dataset_name, method_name, model_name, timestamp)

    return metrics


async def run_baseline_embedding(
    samples: list[Sample],
    unique_entities: list[str],
    model_name: str,
    dataset_name: str,
    timestamp: str,
    eval_model: str = "google/gemini-2.0-flash-001:floor",
    concurrency: int = 10,
) -> dict:
    """
    Run baseline: bare entity name embedding ranking (for comparison).
    Returns metrics dict.
    """
    print(f"\n{'#' * 60}")
    print(f"  BASELINE: Bare Entity Name Embeddings")
    print(f"  Dataset: {dataset_name} ({len(samples)} samples)")
    print(f"  Embedding model: {model_name}")
    print(f"{'#' * 60}")

    t0 = time.time()

    # Compute text embeddings
    print("\n  Computing text embeddings...")
    text_strings = [s.text for s in samples]
    text_embs = compute_embeddings(text_strings)

    # Compute entity NAME embeddings (bare names)
    print("  Computing entity name embeddings (bare names)...")
    entity_embs = compute_embeddings(unique_entities)

    # Rank
    print("  Ranking entities by cosine similarity (text vs bare name)...")
    predictions = embedding_rank_entities(text_embs, entity_embs, unique_entities, top_k=10)

    # Evaluate
    eval_results = await evaluate_predictions(
        samples, predictions, model=eval_model, concurrency=concurrency,
    )

    metrics = compute_metrics(eval_results)
    label = f"{dataset_name} / baseline bare-name embedding / {model_name}"
    print_metrics(metrics, label)

    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["method_detail"] = "baseline_bare_name_embedding"
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    method_name = "embedding_bare_name"
    save_results(eval_results, metrics, dataset_name, method_name, model_name, timestamp)

    return metrics


def print_comparison(desc_metrics: dict, baseline_metrics: dict | None):
    """Print side-by-side comparison of description vs baseline embeddings."""
    print(f"\n\n{'=' * 70}")
    print(f"  EMBEDDING COMPARISON: Description vs Bare Name")
    print(f"{'=' * 70}")
    header = f"  {'Method':<35s} {'Hit@1':>7s} {'Hit@3':>7s} {'Hit@10':>8s} {'MRR':>7s}"
    print(header)
    print(f"  {'-' * 65}")

    for label, m in [
        ("C1: Description Embedding", desc_metrics),
        ("Baseline: Bare Name Embedding", baseline_metrics),
    ]:
        if m is None:
            continue
        h1 = f"{m.get('Hit@1', 0):.3f}"
        h3 = f"{m.get('Hit@3', 0):.3f}"
        h10 = f"{m.get('Hit@10', 0):.3f}"
        mrr = f"{m.get('Global_MRR', 0):.3f}"
        print(f"  {label:<35s} {h1:>7s} {h3:>7s} {h10:>8s} {mrr:>7s}")

    if baseline_metrics:
        # Compute deltas
        delta_h1 = desc_metrics.get("Hit@1", 0) - baseline_metrics.get("Hit@1", 0)
        delta_mrr = desc_metrics.get("Global_MRR", 0) - baseline_metrics.get("Global_MRR", 0)
        print(f"  {'-' * 65}")
        sign_h1 = "+" if delta_h1 >= 0 else ""
        sign_mrr = "+" if delta_mrr >= 0 else ""
        print(f"  {'Delta (desc - baseline)':<35s} {sign_h1}{delta_h1:.3f}{'':>1s} "
              f"{'':>7s} {'':>8s} {sign_mrr}{delta_mrr:.3f}")

    print(f"{'=' * 70}")


async def main():
    parser = argparse.ArgumentParser(
        description="C1: Entity description embedding experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_description_embedding.py --dataset veterans_t2e_v2\n"
            "  python run_description_embedding.py --dataset veterans_t2e_v2 --compare-baseline\n"
            "  python run_description_embedding.py --dataset veterans_t2e_v2 --max-samples 100\n"
        ),
    )
    parser.add_argument(
        "--dataset", default="veterans_t2e_v2",
        help="Dataset name (default: veterans_t2e_v2)",
    )
    parser.add_argument(
        "--descriptions", default=str(DEFAULT_DESCRIPTIONS),
        help="Path to entity descriptions JSON",
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--eval-model", default="google/gemini-2.0-flash-001:floor",
        help="LLM model for alias matching evaluation (default: google/gemini-2.0-flash-001:floor)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit number of samples, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests for evaluation (default: 10)",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Also run baseline bare-name embedding for comparison",
    )
    args = parser.parse_args()

    # Check sentence-transformers
    if not sbert_available():
        print("  ERROR: sentence-transformers is not installed.")
        print("  Install with: pip install sentence-transformers")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    print(f"\n  Loading dataset: {args.dataset}")
    samples, unique_entities = load_dataset(args.dataset)
    if not samples:
        print("  ERROR: No samples loaded.")
        sys.exit(1)

    if args.max_samples and args.max_samples < len(samples):
        print(f"  Limiting to {args.max_samples} samples (of {len(samples)})")
        samples = samples[:args.max_samples]

    # Load descriptions
    descriptions = load_descriptions(Path(args.descriptions))

    print(f"\n{'=' * 60}")
    print(f"  C1: Description Embedding Experiment")
    print(f"  Dataset: {args.dataset} ({len(samples)} samples, {len(unique_entities)} entities)")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Descriptions: {len(descriptions)} loaded")
    print(f"  Compare baseline: {args.compare_baseline}")
    print(f"  Time: {timestamp}")
    print(f"{'=' * 60}")

    # Run C1: description embeddings
    desc_metrics = await run_description_embedding(
        samples=samples,
        unique_entities=unique_entities,
        descriptions=descriptions,
        model_name=args.embedding_model,
        dataset_name=args.dataset,
        timestamp=timestamp,
        eval_model=args.eval_model,
        concurrency=args.concurrency,
    )

    # Optionally run baseline for comparison
    baseline_metrics = None
    if args.compare_baseline:
        baseline_metrics = await run_baseline_embedding(
            samples=samples,
            unique_entities=unique_entities,
            model_name=args.embedding_model,
            dataset_name=args.dataset,
            timestamp=timestamp,
            eval_model=args.eval_model,
            concurrency=args.concurrency,
        )

    # Print comparison
    print_comparison(desc_metrics, baseline_metrics)

    # Save combined summary
    summary = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "embedding_model": args.embedding_model,
        "n_samples": len(samples),
        "n_entities": len(unique_entities),
        "n_descriptions": len(descriptions),
        "description_embedding": desc_metrics,
    }
    if baseline_metrics:
        summary["bare_name_embedding"] = baseline_metrics

    summary_path = RESULTS_DIR / f"{timestamp}_description_embedding_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
