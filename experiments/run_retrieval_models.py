"""
Run all retrieval model experiments: zero-shot embedding + DPR fine-tuning.
Tests BGE and E5 models with and without fine-tuning, with different entity representations.

Usage:
    python run_retrieval_models.py --phase zeroshot        # Zero-shot only (no training)
    python run_retrieval_models.py --phase finetune        # DPR fine-tuning only
    python run_retrieval_models.py --phase all             # Both
    python run_retrieval_models.py --dry-run               # Print plan
    python run_retrieval_models.py --models bge-base       # Single model
"""
import asyncio
import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (
    load_dataset, evaluate_predictions, compute_metrics, print_metrics,
    RESULTS_DIR,
)
from run_battery import save_battery_results

BENCH_DATASETS = ["bench_veterans_t2e", "bench_twitter_t2e"]  # real data only for comparison
DESCRIPTIONS_DIR = Path(__file__).parent / "entity_descriptions"

# Models to test
MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "hf_id": "all-MiniLM-L6-v2",
        "params": "22M", "dims": 384,
        "query_prefix": "", "entity_prefix": "",
    },
    "bge-base": {
        "name": "BGE-base-en-v1.5",
        "hf_id": "BAAI/bge-base-en-v1.5",
        "params": "109M", "dims": 768,
        "query_prefix": "Represent this sentence for retrieving relevant entities: ",
        "entity_prefix": "",
    },
    "bge-large": {
        "name": "BGE-large-en-v1.5",
        "hf_id": "BAAI/bge-large-en-v1.5",
        "params": "335M", "dims": 1024,
        "query_prefix": "Represent this sentence for retrieving relevant entities: ",
        "entity_prefix": "",
    },
    "e5-base": {
        "name": "E5-base-v2",
        "hf_id": "intfloat/e5-base-v2",
        "params": "109M", "dims": 768,
        "query_prefix": "query: ",
        "entity_prefix": "passage: ",
    },
    "e5-large": {
        "name": "E5-large-v2",
        "hf_id": "intfloat/e5-large-v2",
        "params": "335M", "dims": 1024,
        "query_prefix": "query: ",
        "entity_prefix": "passage: ",
    },
}

# Entity representation modes
ENTITY_REPR_MODES = ["name", "description", "type_prefix"]


def load_all_descriptions() -> dict:
    """Load all entity descriptions (veterans + twitter)."""
    all_desc = {}
    for f in DESCRIPTIONS_DIR.glob("*_descriptions.json"):
        data = json.load(open(f, encoding="utf-8"))
        for key, val in data.items():
            all_desc[key.lower().strip()] = val
    print(f"  Loaded {len(all_desc)} entity descriptions")
    return all_desc


def get_entity_text(entity: str, entity_type: str, mode: str, descriptions: dict) -> str:
    """Get entity text for a given representation mode."""
    if mode == "name":
        return entity
    elif mode == "type_prefix":
        return f"{entity_type}: {entity}" if entity_type else entity
    elif mode == "description":
        key = entity.lower().strip()
        desc = descriptions.get(key, {}).get("description", "")
        return desc if desc else entity
    return entity


def run_embedding_experiment(
    model_key: str,
    dataset_name: str,
    entity_repr: str,
    descriptions: dict,
    results_dir: Path,
) -> dict:
    """Run a single zero-shot embedding experiment."""
    from sentence_transformers import SentenceTransformer
    import torch

    model_info = MODELS[model_key]
    print(f"\n  --- {model_info['name']} / {entity_repr} / {dataset_name} ---")

    samples, unique_entities = load_dataset(dataset_name)
    if not samples:
        return {"error": "no samples"}

    # Load model on GPU
    model = SentenceTransformer(model_info["hf_id"], trust_remote_code=True)
    print(f"  Model device: {model.device}, dim: {model.get_sentence_embedding_dimension()}")

    # Prepare texts with prefixes
    q_prefix = model_info["query_prefix"]
    e_prefix = model_info["entity_prefix"]

    text_inputs = [q_prefix + s.text for s in samples]

    # Prepare entity representations
    entity_texts = []
    for ent in unique_entities:
        etype = ""
        for s in samples:
            if s.entity == ent:
                etype = s.entity_type
                break
        entity_texts.append(e_prefix + get_entity_text(ent, etype, entity_repr, descriptions))

    # Encode
    t0 = time.time()
    text_embs = model.encode(text_inputs, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    ent_embs = model.encode(entity_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    # Rank
    sim = np.array(text_embs) @ np.array(ent_embs).T
    predictions = []
    for i in range(len(samples)):
        top_indices = np.argsort(sim[i])[::-1][:10]
        predictions.append([unique_entities[j] for j in top_indices])

    encode_time = time.time() - t0

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return predictions, samples, unique_entities, encode_time


async def run_zeroshot_experiments(results_dir: Path, model_keys: list, concurrency: int = 15):
    """Run all zero-shot embedding experiments."""
    print(f"\n{'='*60}")
    print(f"  ZERO-SHOT EMBEDDING EXPERIMENTS")
    print(f"{'='*60}")

    descriptions = load_all_descriptions()
    all_results = {}

    for model_key in model_keys:
        for dataset_name in BENCH_DATASETS:
            for entity_repr in ENTITY_REPR_MODES:
                try:
                    result = run_embedding_experiment(
                        model_key, dataset_name, entity_repr, descriptions, results_dir
                    )
                    if isinstance(result, dict) and "error" in result:
                        all_results[f"{dataset_name}/{model_key}/{entity_repr}"] = result
                        continue

                    predictions, samples, unique_entities, encode_time = result

                    # Evaluate
                    eval_results = await evaluate_predictions(
                        samples, predictions, model=f"emb_{model_key}",
                        concurrency=concurrency,
                    )
                    metrics = compute_metrics(eval_results)
                    metrics["elapsed_seconds"] = round(encode_time, 1)
                    metrics["model"] = MODELS[model_key]["hf_id"]
                    metrics["entity_repr"] = entity_repr

                    key = f"{dataset_name}/{model_key}/{entity_repr}"
                    print_metrics(metrics, key)
                    save_battery_results(
                        eval_results, metrics, dataset_name,
                        f"emb_{model_key}_{entity_repr}",
                        MODELS[model_key]["hf_id"], results_dir,
                    )
                    all_results[key] = metrics

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()
                    all_results[f"{dataset_name}/{model_key}/{entity_repr}"] = {"error": str(e)}

    return all_results


def run_dpr_finetune(model_key: str, entity_repr: str):
    """Launch DPR fine-tuning for a specific model and entity repr mode."""
    import subprocess
    model_info = MODELS[model_key]

    # Map entity_repr to DPR mode
    dpr_mode = "name" if entity_repr == "name" else "description"
    if entity_repr == "type_prefix":
        dpr_mode = "name"  # type prefix is handled via --type-prefix flag

    cmd = [
        sys.executable, "experiments/train_dpr.py",
        "--base-model", model_info["hf_id"],
        "--mode", dpr_mode,
        "--epochs", "10",
        "--batch-size", "64",
    ]

    if entity_repr != "type_prefix":
        cmd.append("--no-type-prefix")

    print(f"\n  Running: {' '.join(cmd)}")
    return cmd


async def main():
    parser = argparse.ArgumentParser(description="Run retrieval model experiments")
    parser.add_argument("--phase", default="zeroshot", choices=["zeroshot", "finetune", "all"])
    parser.add_argument("--models", default="all",
                        help="Comma-separated model keys or 'all'. Available: " + ",".join(MODELS.keys()))
    parser.add_argument("--entity-repr", default="all",
                        help="Comma-separated repr modes or 'all'. Available: " + ",".join(ENTITY_REPR_MODES))
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Parse model keys
    if args.models == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [k.strip() for k in args.models.split(",")]

    if args.entity_repr == "all":
        repr_modes = ENTITY_REPR_MODES
    else:
        repr_modes = [r.strip() for r in args.entity_repr.split(",")]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"retrieval_{ts}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"\n  DRY RUN: Retrieval Model Experiments")
        print(f"  Models: {model_keys}")
        print(f"  Entity representations: {repr_modes}")
        print(f"  Datasets: {BENCH_DATASETS}")
        print(f"  Phase: {args.phase}")
        n_zs = len(model_keys) * len(BENCH_DATASETS) * len(repr_modes)
        n_ft = len(model_keys) * len(repr_modes)
        print(f"\n  Zero-shot experiments: {n_zs}")
        print(f"  DPR fine-tuning runs: {n_ft}")
        print(f"\n  Models:")
        for k in model_keys:
            m = MODELS[k]
            print(f"    {m['name']}: {m['params']} params, {m['dims']} dims")
        print(f"\n  Cost: $0 (all local GPU)")
        return

    all_results = {}

    if args.phase in ("zeroshot", "all"):
        zs_results = await run_zeroshot_experiments(results_dir, model_keys, args.concurrency)
        all_results.update(zs_results)

    if args.phase in ("finetune", "all"):
        print(f"\n{'='*60}")
        print(f"  DPR FINE-TUNING EXPERIMENTS")
        print(f"{'='*60}")
        for model_key in model_keys:
            for entity_repr in repr_modes:
                cmd = run_dpr_finetune(model_key, entity_repr)
                print(f"  Queued: {MODELS[model_key]['name']} / {entity_repr}")
                # Run sequentially (GPU memory constraint)
                import subprocess
                result = subprocess.run(cmd, capture_output=False)
                if result.returncode != 0:
                    print(f"  FAILED: return code {result.returncode}")

    # Summary
    summary_path = results_dir / "retrieval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "models": model_keys,
            "entity_representations": repr_modes,
            "datasets": BENCH_DATASETS,
            "results": {k: v for k, v in all_results.items() if isinstance(v, dict)},
        }, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  RETRIEVAL MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<20s} {'Repr':<15s} {'Dataset':<25s} {'Hit@1':>7s} {'Hit@3':>7s} {'MRR':>7s}")
    print(f"  {'-'*75}")
    for key, m in sorted(all_results.items()):
        if isinstance(m, dict) and "error" not in m:
            parts = key.split("/")
            ds = parts[0].replace("bench_", "")
            model = parts[1] if len(parts) > 1 else "?"
            repr_mode = parts[2] if len(parts) > 2 else "?"
            h1 = f"{m.get('Hit@1', 0):.3f}"
            h3 = f"{m.get('Hit@3', 0):.3f}"
            mrr = f"{m.get('Global_MRR', 0):.3f}"
            print(f"  {model:<20s} {repr_mode:<15s} {ds:<25s} {h1:>7s} {h3:>7s} {mrr:>7s}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
