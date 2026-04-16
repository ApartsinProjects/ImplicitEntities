"""
Embedding model comparison experiment for implicit entity recognition.

Compares multiple sentence-transformer embedding models on the embedding-based
and hybrid methods to assess sensitivity to the choice of embedding model.

Models tested:
  - sentence-transformers/all-MiniLM-L6-v2       (baseline, already used)
  - sentence-transformers/all-mpnet-base-v2       (larger, higher quality)
  - sentence-transformers/multi-qa-MiniLM-L6-cos-v1  (optimized for QA/retrieval)
  - BAAI/bge-small-en-v1.5                        (recent, strong for its size)

Usage:
    python run_embedding_compare.py --dataset veterans_t2e --models all
    python run_embedding_compare.py --dataset veterans_t2e --models "all-MiniLM-L6-v2,all-mpnet-base-v2"
    python run_embedding_compare.py --dry-run --dataset veterans_t2e
    python run_embedding_compare.py --dataset veterans_t2e --max-samples 50
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    Sample,
    load_dataset,
    get_all_dataset_names,
    embedding_rank_entities,
    evaluate_predictions,
    compute_metrics,
    print_metrics,
    save_results,
    RESULTS_DIR,
)

# ── Embedding model registry ─────────────────────────────────────────────

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "full_name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "MiniLM-L6-v2 (baseline)",
        "dim": 384,
        "params": "22M",
        "note": "Baseline model, fast and compact",
    },
    "all-mpnet-base-v2": {
        "full_name": "sentence-transformers/all-mpnet-base-v2",
        "label": "MPNet-base-v2",
        "dim": 768,
        "params": "109M",
        "note": "Larger, higher quality embeddings",
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "full_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "label": "MultiQA-MiniLM-L6",
        "dim": 384,
        "params": "22M",
        "note": "Optimized for question-answering and retrieval",
    },
    "bge-small-en-v1.5": {
        "full_name": "BAAI/bge-small-en-v1.5",
        "label": "BGE-small-en-v1.5",
        "dim": 384,
        "params": "33M",
        "note": "Strong recent model for its size",
    },
}

MODEL_ORDER = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "bge-small-en-v1.5",
]

# Default LLM model for evaluation (alias matching)
DEFAULT_LLM_MODEL = "google/gemini-2.0-flash-001"


def load_embedding_model(model_key: str):
    """Load a sentence-transformers model by short name."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: sentence-transformers is not installed.")
        print("  Install with: pip install sentence-transformers")
        return None

    info = EMBEDDING_MODELS.get(model_key, {})
    full_name = info.get("full_name", model_key)
    label = info.get("label", model_key)

    print(f"  Loading embedding model: {label} [{full_name}]...")
    try:
        model = SentenceTransformer(full_name)
        print(f"  Model loaded successfully. Dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"  ERROR loading model '{full_name}': {e}")
        return None


def compute_embeddings_with_model(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Compute embeddings using a specific sentence-transformer model."""
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
    )
    return np.array(embeddings)


async def run_embedding_with_model(
    model_key: str,
    samples: list[Sample],
    unique_entities: list[str],
    llm_model: str,
    concurrency: int = 10,
) -> list[list[str]]:
    """
    Run the embedding method with a specific embedding model.
    Returns ranked entity lists per sample.
    """
    emb_model = load_embedding_model(model_key)
    if emb_model is None:
        raise RuntimeError(f"Failed to load embedding model: {model_key}")

    print(f"  Computing text embeddings ({len(samples)} texts)...")
    text_strings = [s.text for s in samples]
    text_embs = compute_embeddings_with_model(emb_model, text_strings)

    print(f"  Computing entity embeddings ({len(unique_entities)} entities)...")
    entity_embs = compute_embeddings_with_model(emb_model, unique_entities)

    print("  Ranking entities by cosine similarity...")
    predictions = embedding_rank_entities(text_embs, entity_embs, unique_entities, top_k=10)

    # Free memory
    del emb_model, text_embs, entity_embs

    return predictions


def estimate_embedding_cost(n_samples: int, n_entities: int) -> dict:
    """
    Estimate resources for an embedding comparison run.
    Embedding inference is local (free), but evaluation uses the LLM for alias matching.
    """
    # Evaluation cost: ~50 input + ~5 output tokens per alias check
    # Assume ~80% need alias check, ~3 guesses each
    alias_checks = int(n_samples * 0.8 * 3)
    eval_input = alias_checks * 50
    eval_output = alias_checks * 5

    # Default model pricing (Gemini Flash)
    input_per_m = 0.10
    output_per_m = 0.40
    eval_cost = (eval_input / 1_000_000) * input_per_m + (eval_output / 1_000_000) * output_per_m

    return {
        "n_samples": n_samples,
        "n_entities": n_entities,
        "alias_checks": alias_checks,
        "est_eval_input_tokens": eval_input,
        "est_eval_output_tokens": eval_output,
        "est_eval_cost_usd": eval_cost,
        "note": "Embedding computation is local (free). Cost is for LLM-based evaluation only.",
    }


def print_embedding_cost_table(dataset_names: list[str], model_keys: list[str], max_samples: int):
    """Print cost estimate table."""
    print(f"\n{'=' * 70}")
    print("  COST ESTIMATE (dry run)")
    print(f"  NOTE: Embedding computation is local (free).")
    print(f"  Cost shown is for LLM-based evaluation (alias matching) only.")
    print(f"{'=' * 70}")

    total_cost = 0.0
    for ds_name in dataset_names:
        samples, unique_entities = load_dataset(ds_name)
        n = min(len(samples), max_samples) if max_samples > 0 else len(samples)
        est = estimate_embedding_cost(n, len(unique_entities))

        print(f"\n  Dataset: {ds_name}")
        print(f"    Samples: {n}, Unique entities: {len(unique_entities)}")
        print(f"    Embedding models to test: {len(model_keys)}")
        for mk in model_keys:
            info = EMBEDDING_MODELS.get(mk, {})
            label = info.get("label", mk)
            params = info.get("params", "?")
            dim = info.get("dim", "?")
            print(f"      - {label:<30s} (dim={dim}, params={params})")

        per_model_cost = est["est_eval_cost_usd"]
        ds_total = per_model_cost * len(model_keys)
        total_cost += ds_total
        print(f"    Eval cost per model: ${per_model_cost:.4f}")
        print(f"    Total for dataset:   ${ds_total:.4f} ({len(model_keys)} models)")

    print(f"\n  {'=' * 40}")
    print(f"  TOTAL ESTIMATED COST: ${total_cost:.4f}")
    print(f"  {'=' * 40}")
    return total_cost


def print_embedding_comparison(all_results: dict):
    """Print comparison table across embedding models."""
    print(f"\n\n{'=' * 85}")
    print("  EMBEDDING MODEL COMPARISON")
    print(f"{'=' * 85}")
    header = (f"  {'Dataset':<18s} {'Model':<28s} {'Hit@1':>7s} {'Hit@3':>7s} "
              f"{'Hit@5':>7s} {'MRR':>7s} {'Time':>7s}")
    print(header)
    print(f"  {'-' * 82}")

    for key, m in all_results.items():
        ds, model_key = key
        info = EMBEDDING_MODELS.get(model_key, {})
        label = info.get("label", model_key)
        if "error" in m:
            print(f"  {ds:<18s} {label:<28s} ERROR: {m['error'][:25]}")
        else:
            h1 = f"{m.get('Hit@1', 0):.3f}"
            h3 = f"{m.get('Hit@3', 0):.3f}"
            h5 = f"{m.get('Hit@5', 0):.3f}"
            mrr = f"{m.get('Global_MRR', 0):.3f}"
            t = f"{m.get('elapsed_seconds', 0):.0f}s"
            print(f"  {ds:<18s} {label:<28s} {h1:>7s} {h3:>7s} {h5:>7s} {mrr:>7s} {t:>7s}")

    print(f"{'=' * 85}")

    # Delta table vs baseline
    baselines = {
        ds: m for (ds, mk), m in all_results.items()
        if mk == "all-MiniLM-L6-v2" and "error" not in m
    }
    if baselines:
        print(f"\n  DELTA vs BASELINE (all-MiniLM-L6-v2)")
        print(f"  {'-' * 55}")
        for (ds, mk), m in all_results.items():
            if mk == "all-MiniLM-L6-v2" or "error" in m:
                continue
            if ds in baselines:
                label = EMBEDDING_MODELS.get(mk, {}).get("label", mk)
                base_h1 = baselines[ds].get("Hit@1", 0)
                this_h1 = m.get("Hit@1", 0)
                delta = this_h1 - base_h1
                sign = "+" if delta >= 0 else ""
                base_mrr = baselines[ds].get("Global_MRR", 0)
                this_mrr = m.get("Global_MRR", 0)
                d_mrr = this_mrr - base_mrr
                sign_mrr = "+" if d_mrr >= 0 else ""
                print(f"  {ds:<18s} {label:<22s} Hit@1: {sign}{delta:.3f}  MRR: {sign_mrr}{d_mrr:.3f}")
        print(f"  {'-' * 55}")


def parse_models_arg(models_str: str) -> list[str]:
    """Parse the --models argument."""
    if models_str.lower() == "all":
        return list(MODEL_ORDER)
    parts = [m.strip() for m in models_str.split(",") if m.strip()]
    validated = []
    for m in parts:
        if m in EMBEDDING_MODELS:
            validated.append(m)
        else:
            # Try matching by full name or partial match
            matched = False
            for key, info in EMBEDDING_MODELS.items():
                if m == info["full_name"] or m in key or key in m:
                    validated.append(key)
                    matched = True
                    break
            if not matched:
                print(f"  WARNING: Unknown embedding model '{m}'. Known models:")
                for key, info in EMBEDDING_MODELS.items():
                    print(f"    {key:<35s}  ({info['full_name']})")
                print(f"  Skipping '{m}'.")
    return validated


async def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding models for implicit entity recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models:\n"
            "  all-MiniLM-L6-v2           Baseline (already used)\n"
            "  all-mpnet-base-v2          Larger, higher quality\n"
            "  multi-qa-MiniLM-L6-cos-v1  Optimized for QA/retrieval\n"
            "  bge-small-en-v1.5          Strong recent model\n\n"
            "Examples:\n"
            "  python run_embedding_compare.py --dataset veterans_t2e --models all\n"
            '  python run_embedding_compare.py --dataset veterans_t2e --models "all-MiniLM-L6-v2,all-mpnet-base-v2"\n'
            "  python run_embedding_compare.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--dataset", default="veterans_t2e",
        help="Dataset name or 'all' (default: veterans_t2e)",
    )
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model short names or 'all' (default: all)",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_LLM_MODEL,
        help=f"LLM model for evaluation alias matching (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests for evaluation (default: 10)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit number of samples for testing, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate cost and list models without running experiments",
    )
    args = parser.parse_args()

    # Resolve datasets
    if args.dataset == "all":
        dataset_names = get_all_dataset_names()
    else:
        dataset_names = [args.dataset]

    # Resolve models
    model_keys = parse_models_arg(args.models)
    if not model_keys:
        print("  ERROR: No valid embedding models selected.")
        return

    print(f"\n{'=' * 60}")
    print("  Embedding Model Comparison Experiment")
    print(f"  Datasets: {dataset_names}")
    print(f"  Embedding models: {len(model_keys)}")
    for mk in model_keys:
        info = EMBEDDING_MODELS[mk]
        print(f"    - {info['label']:<30s} [{info['full_name']}]")
        print(f"      {info['note']}")
    print(f"  LLM for eval: {args.llm_model}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    print(f"{'=' * 60}")

    # Check sentence-transformers availability
    try:
        from sentence_transformers import SentenceTransformer
        print("  sentence-transformers: available")
    except ImportError:
        print("  ERROR: sentence-transformers is not installed.")
        print("  Install with: pip install sentence-transformers")
        return

    # Cost estimation
    total_est = print_embedding_cost_table(dataset_names, model_keys, args.max_samples)

    if args.dry_run:
        print("\n  DRY RUN complete. No experiments run.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for ds_name in dataset_names:
        samples, unique_entities = load_dataset(ds_name)
        if not samples:
            print(f"  ERROR: No samples in dataset '{ds_name}', skipping.")
            continue

        if args.max_samples and args.max_samples < len(samples):
            print(f"  Limiting to {args.max_samples} samples (of {len(samples)})")
            samples = samples[:args.max_samples]

        for model_key in model_keys:
            info = EMBEDDING_MODELS[model_key]
            label = info["label"]

            print(f"\n{'#' * 60}")
            print(f"  EMBEDDING MODEL: {label}")
            print(f"  DATASET: {ds_name} ({len(samples)} samples, {len(unique_entities)} entities)")
            print(f"{'#' * 60}")

            t0 = time.time()
            try:
                predictions = await run_embedding_with_model(
                    model_key, samples, unique_entities,
                    llm_model=args.llm_model, concurrency=args.concurrency,
                )
                eval_results = await evaluate_predictions(
                    samples, predictions, model=args.llm_model,
                    concurrency=args.concurrency,
                )
                metrics = compute_metrics(eval_results)
                elapsed = time.time() - t0
                metrics["elapsed_seconds"] = round(elapsed, 1)
                metrics["embedding_model"] = model_key
                metrics["embedding_model_full"] = info["full_name"]
                metrics["embedding_dim"] = info["dim"]

                print_metrics(metrics, f"{ds_name} / embedding / {label}")

                save_results(
                    eval_results, metrics, ds_name,
                    f"emb_{model_key}", args.llm_model, timestamp,
                )
                all_results[(ds_name, model_key)] = metrics

            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  ERROR running {label} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[(ds_name, model_key)] = {
                    "error": str(e),
                    "elapsed_seconds": round(elapsed, 1),
                }

    # Comparison table
    if all_results:
        print_embedding_comparison(all_results)

    # Save combined summary
    summary_path = RESULTS_DIR / f"{timestamp}_embedding_compare_summary.json"
    serializable = {}
    for (ds, mk), m in all_results.items():
        serializable[f"{ds}/{mk}"] = m
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "experiment": "embedding_model_comparison",
            "llm_model": args.llm_model,
            "datasets": dataset_names,
            "embedding_models": model_keys,
            "max_samples": args.max_samples,
            "results": serializable,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
