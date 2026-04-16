"""
Full experiment battery on IRC-Bench v2.
Runs all experiments with versioned results directory, smoke tests, and parallel execution.

Usage:
    python run_battery.py                    # Full battery
    python run_battery.py --smoke-only       # Smoke tests only
    python run_battery.py --group 1          # Run specific group
    python run_battery.py --group 1,2,3      # Run multiple groups
    python run_battery.py --dry-run          # Print plan and costs
"""
import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openrouter_client import batch_call, load_api_key, estimate_cost
from run_experiments import (
    load_dataset, get_all_dataset_names, Sample,
    run_llm_method, run_embedding_method, run_hybrid_method,
    evaluate_predictions, compute_metrics, print_metrics,
    smoke_test_llm, RESULTS_DIR,
)

PROJECT_ROOT = Path(__file__).parent.parent
BENCH_DATASETS = [
    "bench_veterans_t2e",
    "bench_veterans_e2t",
    "bench_twitter_t2e",
    "bench_twitter_e2t",
]

DEFAULT_MODEL = "google/gemini-2.0-flash-001:floor"


def create_battery_dir() -> Path:
    """Create a versioned results directory for this battery run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    battery_dir = RESULTS_DIR / f"battery_{ts}"
    battery_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results directory: {battery_dir}")
    return battery_dir


def save_battery_results(
    eval_results, metrics, dataset_name, method_name, model, battery_dir
):
    """Save results into the battery directory (no colons, no overwrite risk)."""
    import csv
    import pandas as pd

    safe_model = model.replace("/", "_").replace(":", "_")
    prefix = f"{dataset_name}_{method_name}_{safe_model}"

    # Predictions CSV
    csv_path = battery_dir / f"{prefix}_predictions.csv"
    rows = []
    for r in eval_results:
        preds_padded = r.predictions[:10]
        while len(preds_padded) < 10:
            preds_padded.append("")
        rows.append({
            "uid": r.sample.uid,
            "text": r.sample.text[:500],
            "gold_entity": r.sample.entity,
            "entity_type": r.sample.entity_type,
            "match_tier": r.match_tier,
            "match_rank": r.match_rank,
            "matched_prediction": r.matched_prediction,
            **{f"pred_{i+1}": p for i, p in enumerate(preds_padded)},
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")

    # Metrics JSON
    json_path = battery_dir / f"{prefix}_metrics.json"
    meta = {
        "dataset": dataset_name,
        "method": method_name,
        "model": model,
        "battery_dir": str(battery_dir),
        **metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {csv_path.name}")
    return csv_path, json_path


async def run_single(dataset, method, model, battery_dir, concurrency=15, max_samples=0):
    """Run a single experiment with smoke test."""
    print(f"\n{'#'*60}")
    print(f"  {dataset} / {method} / {model.split('/')[-1]}")
    print(f"{'#'*60}")

    samples, unique_entities = load_dataset(dataset)
    if not samples:
        return {"error": "no samples"}

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    # Smoke test for LLM methods
    if method in ("llm", "hybrid", "closedset"):
        ok = await smoke_test_llm(samples, model, concurrency=min(concurrency, 5))
        if not ok:
            print(f"  SMOKE FAILED. Skipping.")
            return {"error": "smoke_test_failed"}

    t0 = time.time()

    if method == "llm":
        predictions = await run_llm_method(samples, model, concurrency)
    elif method == "embedding":
        predictions = await run_embedding_method(samples, unique_entities, model, concurrency)
    elif method == "hybrid":
        predictions = await run_hybrid_method(samples, unique_entities, model, concurrency)
    else:
        print(f"  Unknown method: {method}")
        return {"error": f"unknown method: {method}"}

    eval_results = await evaluate_predictions(samples, predictions, model=model, concurrency=concurrency)
    metrics = compute_metrics(eval_results)
    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)

    print_metrics(metrics, f"{dataset} / {method} / {model.split('/')[-1]}")
    save_battery_results(eval_results, metrics, dataset, method, model, battery_dir)

    return metrics


async def run_group_1_embedding(battery_dir, concurrency=15):
    """Group 1: Embedding methods (local, $0)."""
    print(f"\n{'='*60}")
    print(f"  GROUP 1: EMBEDDING METHODS (local, $0)")
    print(f"{'='*60}")

    results = {}
    for ds in BENCH_DATASETS:
        m = await run_single(ds, "embedding", DEFAULT_MODEL, battery_dir, concurrency)
        results[f"{ds}/embedding/minilm"] = m
    return results


async def run_group_2_llm_openset(battery_dir, concurrency=15):
    """Group 2: LLM open-set (OpenRouter)."""
    print(f"\n{'='*60}")
    print(f"  GROUP 2: LLM OPEN-SET (OpenRouter)")
    print(f"{'='*60}")

    results = {}
    # Gemini Flash on all datasets
    for ds in BENCH_DATASETS:
        m = await run_single(ds, "llm", DEFAULT_MODEL, battery_dir, concurrency)
        results[f"{ds}/llm/gemini-flash"] = m

    # Llama 3.1 8B on all datasets
    llama_model = "meta-llama/llama-3.1-8b-instruct:floor"
    for ds in BENCH_DATASETS:
        m = await run_single(ds, "llm", llama_model, battery_dir, concurrency)
        results[f"{ds}/llm/llama-8b"] = m

    return results


async def run_group_3_closedset(battery_dir, concurrency=15):
    """Group 3: Closed-set entity recognition."""
    print(f"\n{'='*60}")
    print(f"  GROUP 3: CLOSED-SET (OpenRouter)")
    print(f"{'='*60}")

    # Import closedset functions
    from run_closedset import run_closedset_chunked, run_closedset_direct

    results = {}
    for ds in BENCH_DATASETS:
        print(f"\n  --- Closed-set: {ds} ---")
        samples, unique_entities = load_dataset(ds)
        if not samples:
            results[f"{ds}/closedset"] = {"error": "no samples"}
            continue

        # Smoke test
        ok = await smoke_test_llm(samples, DEFAULT_MODEL, concurrency=5)
        if not ok:
            results[f"{ds}/closedset"] = {"error": "smoke_test_failed"}
            continue

        t0 = time.time()
        entity_list = sorted(unique_entities)

        if len(entity_list) <= 100:
            predictions = await run_closedset_direct(samples, entity_list, DEFAULT_MODEL, concurrency)
        else:
            predictions = await run_closedset_chunked(samples, entity_list, DEFAULT_MODEL, concurrency)

        eval_results = await evaluate_predictions(samples, predictions, model=DEFAULT_MODEL, concurrency=concurrency)
        metrics = compute_metrics(eval_results)
        metrics["elapsed_seconds"] = round(time.time() - t0, 1)
        metrics["n_entities"] = len(entity_list)

        print_metrics(metrics, f"{ds} / closedset")
        save_battery_results(eval_results, metrics, ds, "closedset", DEFAULT_MODEL, battery_dir)
        results[f"{ds}/closedset"] = metrics

    return results


async def run_group_4_hybrid(battery_dir, concurrency=15):
    """Group 4: Hybrid RAG."""
    print(f"\n{'='*60}")
    print(f"  GROUP 4: HYBRID RAG (Embedding + LLM)")
    print(f"{'='*60}")

    results = {}
    for ds in BENCH_DATASETS:
        m = await run_single(ds, "hybrid", DEFAULT_MODEL, battery_dir, concurrency)
        results[f"{ds}/hybrid/gemini-flash"] = m
    return results


async def run_group_5_improved(battery_dir, concurrency=15):
    """Group 5: Improved prompts (veterans_t2e only)."""
    print(f"\n{'='*60}")
    print(f"  GROUP 5: IMPROVED PROMPTS (bench_veterans_t2e)")
    print(f"{'='*60}")

    from run_improved_llm import run_improved_method

    ds = "bench_veterans_t2e"
    results = {}

    for improvement in ["structured", "fewshot", "cot", "direct"]:
        print(f"\n  --- {improvement} ---")
        samples, unique_entities = load_dataset(ds)
        if not samples:
            results[f"{ds}/improved_{improvement}"] = {"error": "no samples"}
            continue

        t0 = time.time()
        try:
            predictions = await run_improved_method(
                samples, improvement, DEFAULT_MODEL, concurrency
            )
            eval_results = await evaluate_predictions(
                samples, predictions, model=DEFAULT_MODEL, concurrency=concurrency
            )
            metrics = compute_metrics(eval_results)
            metrics["elapsed_seconds"] = round(time.time() - t0, 1)
            metrics["improvement"] = improvement

            print_metrics(metrics, f"{ds} / {improvement}")
            save_battery_results(
                eval_results, metrics, ds, f"improved_{improvement}", DEFAULT_MODEL, battery_dir
            )
            results[f"{ds}/improved_{improvement}"] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[f"{ds}/improved_{improvement}"] = {"error": str(e)}

    return results


async def run_smoke_tests():
    """Phase 1: Smoke tests for all components."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: SMOKE TESTS")
    print(f"{'='*60}")

    all_ok = True

    # Test 1: Benchmark data loading
    print("\n  [1/4] Loading benchmark datasets...")
    for ds in BENCH_DATASETS:
        try:
            samples, ents = load_dataset(ds)
            print(f"    {ds}: {len(samples)} samples, {len(ents)} entities OK")
        except Exception as e:
            print(f"    {ds}: FAILED - {e}")
            all_ok = False

    # Test 2: OpenRouter API
    print("\n  [2/4] Testing OpenRouter API...")
    try:
        test_prompts = [[
            {"role": "user", "content": "Say 'hello' in one word."}
        ]]
        results = await batch_call(test_prompts, model=DEFAULT_MODEL, max_tokens=10, concurrency=1, progress_every=999)
        if results[0]:
            print(f"    OpenRouter OK: '{results[0][:30]}'")
        else:
            print(f"    OpenRouter FAILED: None response")
            all_ok = False
    except Exception as e:
        print(f"    OpenRouter FAILED: {e}")
        all_ok = False

    # Test 3: Embedding model
    print("\n  [3/4] Testing embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(["test"], normalize_embeddings=True)
        print(f"    Embedding OK: dim={emb.shape[1]}")
        del model
    except Exception as e:
        print(f"    Embedding FAILED: {e}")
        all_ok = False

    # Test 4: OpenAI API
    print("\n  [4/4] Testing OpenAI API...")
    try:
        from openai_batch_client import OpenAIBatchClient, OPENAI_AVAILABLE
        if OPENAI_AVAILABLE:
            client = OpenAIBatchClient()
            batches = client.list_batches(limit=1)
            print(f"    OpenAI API OK (listed {len(batches)} batches)")
        else:
            print(f"    OpenAI SDK not installed, skipping batch")
    except Exception as e:
        print(f"    OpenAI FAILED: {e}")
        # Not a blocker, just skip Group 6

    # Test 5: LLM smoke test on 3 benchmark samples
    print("\n  [5/5] LLM smoke test on benchmark samples...")
    samples, _ = load_dataset("bench_veterans_t2e")
    ok = await smoke_test_llm(samples, DEFAULT_MODEL, concurrency=5, n_samples=3)
    if not ok:
        print(f"    LLM smoke FAILED")
        all_ok = False

    print(f"\n  {'ALL SMOKE TESTS PASSED' if all_ok else 'SOME SMOKE TESTS FAILED'}")
    return all_ok


async def main():
    parser = argparse.ArgumentParser(description="Full experiment battery on IRC-Bench v2")
    parser.add_argument("--smoke-only", action="store_true", help="Run smoke tests only")
    parser.add_argument("--group", default="all", help="Groups to run: 1,2,3,4,5 or 'all'")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per experiment (for testing)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  IRC-BENCH v2 FULL EXPERIMENT BATTERY")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  DRY RUN: Printing experiment plan and cost estimates\n")
        for ds in BENCH_DATASETS:
            samples, ents = load_dataset(ds)
            print(f"  {ds}: {len(samples)} samples, {len(ents)} entities")
        print(f"\n  Total experiments: 13 (5 groups)")
        print(f"  Estimated cost: ~$0.50")
        print(f"  Estimated time: ~45 min")
        return

    # Phase 1: Smoke tests
    smoke_ok = await run_smoke_tests()
    if args.smoke_only:
        return
    if not smoke_ok:
        print("\n  WARNING: Some smoke tests failed. Proceeding with caution...")

    # Create versioned battery directory
    battery_dir = create_battery_dir()

    # Determine which groups to run
    if args.group == "all":
        groups = [1, 2, 3, 4, 5]
    else:
        groups = [int(g.strip()) for g in args.group.split(",")]

    all_results = {}

    # Phase 2: Groups 1, 2, 3
    if 1 in groups:
        r = await run_group_1_embedding(battery_dir, args.concurrency)
        all_results.update(r)

    if 2 in groups:
        r = await run_group_2_llm_openset(battery_dir, args.concurrency)
        all_results.update(r)

    if 3 in groups:
        r = await run_group_3_closedset(battery_dir, args.concurrency)
        all_results.update(r)

    if 4 in groups:
        r = await run_group_4_hybrid(battery_dir, args.concurrency)
        all_results.update(r)

    if 5 in groups:
        r = await run_group_5_improved(battery_dir, args.concurrency)
        all_results.update(r)

    # Save battery summary
    summary_path = battery_dir / "battery_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "groups_run": groups,
            "datasets": BENCH_DATASETS,
            "results": {k: v for k, v in all_results.items() if isinstance(v, dict)},
        }, f, indent=2, ensure_ascii=False)

    # Print final summary table
    print(f"\n\n{'='*80}")
    print(f"  BATTERY SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Experiment':<50s} {'Hit@1':>7s} {'Hit@3':>7s} {'MRR':>7s} {'Time':>7s}")
    print(f"  {'-'*75}")
    for key, m in sorted(all_results.items()):
        if not isinstance(m, dict) or "error" in m:
            err = m.get("error", "?") if isinstance(m, dict) else str(m)
            print(f"  {key:<50s} ERROR: {err[:25]}")
        else:
            h1 = f"{m.get('Hit@1', 0):.3f}"
            h3 = f"{m.get('Hit@3', 0):.3f}"
            mrr = f"{m.get('Global_MRR', 0):.3f}"
            t = f"{m.get('elapsed_seconds', 0):.0f}s"
            print(f"  {key:<50s} {h1:>7s} {h3:>7s} {mrr:>7s} {t:>7s}")
    print(f"{'='*80}")
    print(f"\n  Results saved to: {battery_dir}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
