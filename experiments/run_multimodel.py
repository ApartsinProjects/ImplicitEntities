"""
Multi-model comparison experiment for implicit entity recognition.

Runs the LLM-based method across multiple models via OpenRouter to compare
performance. This is the highest-impact Tier 2 experiment for the paper.

Usage:
    python run_multimodel.py --dataset veterans_t2e --models all
    python run_multimodel.py --dataset all --models "google/gemini-2.0-flash-001,meta-llama/llama-3.1-8b-instruct"
    python run_multimodel.py --dry-run --dataset veterans_t2e
    python run_multimodel.py --dataset veterans_t2e --models all --max-samples 20
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    load_dataset,
    get_all_dataset_names,
    run_llm_method,
    evaluate_predictions,
    compute_metrics,
    print_metrics,
    save_results,
    RESULTS_DIR,
)

# ── Model registry with cost estimates ────────────────────────────────────
# Prices are per 1M tokens (input/output) from OpenRouter as of early 2026.
# These are approximate and may change.
MODELS = {
    "google/gemini-2.0-flash-001": {
        "label": "Gemini 2.0 Flash",
        "input_per_m": 0.10,
        "output_per_m": 0.40,
        "tier": "baseline",
        "note": "Already run, baseline model",
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "label": "Llama 3.1 8B",
        "input_per_m": 0.055,
        "output_per_m": 0.055,
        "tier": "cheap",
        "note": "Very cheap, small model",
    },
    "mistralai/mistral-7b-instruct": {
        "label": "Mistral 7B",
        "input_per_m": 0.055,
        "output_per_m": 0.055,
        "tier": "cheap",
        "note": "Very cheap, small model",
    },
    "google/gemini-2.0-flash-thinking-exp-01-21": {
        "label": "Gemini Flash Thinking",
        "input_per_m": 0.0,
        "output_per_m": 0.0,
        "tier": "free",
        "note": "Free experimental model on OpenRouter",
    },
    "anthropic/claude-3.5-haiku": {
        "label": "Claude 3.5 Haiku",
        "input_per_m": 0.80,
        "output_per_m": 4.00,
        "tier": "moderate",
        "note": "Moderate cost, strong performance",
    },
    "openai/gpt-4o-mini": {
        "label": "GPT-4o Mini",
        "input_per_m": 0.15,
        "output_per_m": 0.60,
        "tier": "moderate",
        "note": "Moderate cost, good baseline",
    },
}

# Order for running: cheapest/free first
MODEL_ORDER = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-thinking-exp-01-21",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
]


def estimate_cost(n_samples: int, model_id: str) -> dict:
    """
    Estimate token usage and cost for the LLM method on a given number of samples.

    The LLM method has two steps:
      Step 1 (context): ~300 input tokens, ~150 output tokens per sample
      Step 2 (inference): ~500 input tokens, ~100 output tokens per sample
      Evaluation (alias matching): ~50 input tokens, ~5 output tokens per unmatched pair
        Assume ~80% of samples need alias check with ~3 guesses each = ~2.4 checks/sample
    """
    info = MODELS.get(model_id, {})
    input_per_m = info.get("input_per_m", 0.10)
    output_per_m = info.get("output_per_m", 0.40)

    # Token estimates per sample
    input_tokens_per_sample = 300 + 500 + (50 * 2.4)  # ~920
    output_tokens_per_sample = 150 + 100 + (5 * 2.4)   # ~262

    total_input = int(n_samples * input_tokens_per_sample)
    total_output = int(n_samples * output_tokens_per_sample)
    total_cost = (total_input / 1_000_000) * input_per_m + (total_output / 1_000_000) * output_per_m

    return {
        "n_samples": n_samples,
        "est_input_tokens": total_input,
        "est_output_tokens": total_output,
        "est_cost_usd": total_cost,
        "input_per_m": input_per_m,
        "output_per_m": output_per_m,
    }


def print_cost_table(dataset_names: list[str], model_ids: list[str], max_samples: int):
    """Print a table of estimated costs for all dataset/model combinations."""
    print(f"\n{'=' * 75}")
    print("  COST ESTIMATE (dry run)")
    print(f"{'=' * 75}")

    total_cost = 0.0
    total_samples = 0

    header = f"  {'Model':<42s} {'Samples':>8s} {'Input':>10s} {'Output':>10s} {'Cost':>10s}"
    print(header)
    print(f"  {'-' * 72}")

    for ds_name in dataset_names:
        samples, _ = load_dataset(ds_name)
        n = min(len(samples), max_samples) if max_samples > 0 else len(samples)
        print(f"  Dataset: {ds_name} ({n} samples)")

        for model_id in model_ids:
            est = estimate_cost(n, model_id)
            info = MODELS.get(model_id, {})
            label = info.get("label", model_id.split("/")[-1])
            total_cost += est["est_cost_usd"]
            total_samples += n
            print(f"    {label:<40s} {n:>8d} {est['est_input_tokens']:>10,} "
                  f"{est['est_output_tokens']:>10,} ${est['est_cost_usd']:>8.4f}")

    print(f"  {'-' * 72}")
    print(f"  {'TOTAL':<42s} {total_samples:>8d} {'':>10s} {'':>10s} ${total_cost:>8.4f}")
    print(f"{'=' * 75}")
    return total_cost


def print_comparison_table(all_results: dict):
    """Print a comparison table across models."""
    print(f"\n\n{'=' * 85}")
    print("  MULTI-MODEL COMPARISON")
    print(f"{'=' * 85}")
    header = (f"  {'Dataset':<18s} {'Model':<28s} {'Hit@1':>7s} {'Hit@3':>7s} "
              f"{'Hit@5':>7s} {'MRR':>7s} {'Time':>7s}")
    print(header)
    print(f"  {'-' * 82}")

    for key, m in all_results.items():
        ds, model_id = key
        label = MODELS.get(model_id, {}).get("label", model_id.split("/")[-1])
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


def parse_models(models_str: str) -> list[str]:
    """Parse the --models argument into a list of model IDs."""
    if models_str.lower() == "all":
        return list(MODEL_ORDER)
    parts = [m.strip() for m in models_str.split(",") if m.strip()]
    # Validate
    for m in parts:
        if m not in MODELS:
            print(f"  WARNING: Unknown model '{m}'. Known models:")
            for mid, info in MODELS.items():
                print(f"    {mid:<50s}  ({info['label']})")
            print(f"  Proceeding anyway (OpenRouter may still support it).")
    return parts


async def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model comparison for implicit entity recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_multimodel.py --dataset veterans_t2e --models all\n"
            '  python run_multimodel.py --dataset all --models "meta-llama/llama-3.1-8b-instruct"\n'
            "  python run_multimodel.py --dry-run --dataset veterans_t2e\n"
            "  python run_multimodel.py --dataset veterans_t2e --max-samples 20\n"
        ),
    )
    parser.add_argument(
        "--dataset", default="veterans_t2e",
        help="Dataset name or 'all' (default: veterans_t2e)",
    )
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model IDs or 'all' (default: all)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit number of samples for testing, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate token count and cost without making API calls",
    )
    args = parser.parse_args()

    # Resolve datasets
    if args.dataset == "all":
        dataset_names = get_all_dataset_names()
    else:
        dataset_names = [args.dataset]

    # Resolve models
    model_ids = parse_models(args.models)

    print(f"\n{'=' * 60}")
    print("  Multi-Model Comparison Experiment")
    print(f"  Datasets: {dataset_names}")
    print(f"  Models:   {len(model_ids)} models")
    for mid in model_ids:
        info = MODELS.get(mid, {})
        label = info.get("label", mid.split("/")[-1])
        note = info.get("note", "")
        print(f"    - {label:<30s} [{mid}] {note}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    print(f"{'=' * 60}")

    # Cost estimation
    total_est = print_cost_table(dataset_names, model_ids, args.max_samples)

    if args.dry_run:
        print("\n  DRY RUN complete. No API calls made.")
        return

    # Cost warning for expensive runs
    if total_est > 0.50:
        print(f"\n  *** WARNING: Estimated total cost is ${total_est:.4f}. ***")
        print("  Consider using --max-samples to test with a subset first.")
        print("  Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n  Aborted by user.")
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

        for model_id in model_ids:
            info = MODELS.get(model_id, {})
            label = info.get("label", model_id.split("/")[-1])

            print(f"\n{'#' * 60}")
            print(f"  MODEL: {label} [{model_id}]")
            print(f"  DATASET: {ds_name} ({len(samples)} samples)")

            est = estimate_cost(len(samples), model_id)
            print(f"  Estimated cost: ${est['est_cost_usd']:.4f}")
            print(f"{'#' * 60}")

            t0 = time.time()
            try:
                predictions = await run_llm_method(
                    samples, model_id, args.concurrency, batch_size=0,
                )
                eval_results = await evaluate_predictions(
                    samples, predictions, model=model_id, concurrency=args.concurrency,
                )
                metrics = compute_metrics(eval_results)
                elapsed = time.time() - t0
                metrics["elapsed_seconds"] = round(elapsed, 1)
                metrics["model"] = model_id
                metrics["model_label"] = label

                print_metrics(metrics, f"{ds_name} / LLM / {label}")

                save_results(
                    eval_results, metrics, ds_name,
                    f"llm_multimodel", model_id, timestamp,
                )
                all_results[(ds_name, model_id)] = metrics

            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  ERROR running {label} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[(ds_name, model_id)] = {
                    "error": str(e),
                    "elapsed_seconds": round(elapsed, 1),
                }

    # Comparison table
    if all_results:
        print_comparison_table(all_results)

    # Save combined summary
    summary_path = RESULTS_DIR / f"{timestamp}_multimodel_summary.json"
    serializable = {}
    for (ds, mid), m in all_results.items():
        serializable[f"{ds}/{mid}"] = m
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "experiment": "multimodel_comparison",
            "datasets": dataset_names,
            "models": model_ids,
            "max_samples": args.max_samples,
            "results": serializable,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
