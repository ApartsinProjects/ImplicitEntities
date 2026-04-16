#!/usr/bin/env python
"""
DPR (Dense Passage Retrieval) fine-tuning pipeline for IRC implicit entity recognition.

Fine-tunes a bi-encoder to map implicit reference texts close to their correct
entities in embedding space, using Multiple Negatives Ranking Loss with in-batch
negatives.

Training modes:
  --mode name         Train to match text -> entity name (baseline)
  --mode description  Train to match text -> entity description
  --mode both         Train on both name and description pairs

Usage:
  python train_dpr.py --mode name --epochs 10 --batch-size 64
  python train_dpr.py --mode description --base-model all-mpnet-base-v2
  python train_dpr.py --dry-run --mode both
  python train_dpr.py --eval-only --model-path experiments/trained_models/dpr_name_all-MiniLM-L6-v2
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sentence_transformers import (
        InputExample,
        SentenceTransformer,
        losses,
        evaluation,
    )
    from sentence_transformers.util import cos_sim
    from torch.utils.data import DataLoader
except ImportError:
    print(
        "ERROR: sentence-transformers is not installed.\n"
        "Install with: pip install sentence-transformers\n"
        "Or: /c/Python314/python -m pip install sentence-transformers"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmark_v2"
BENCH_V3 = DATA_DIR / "IRC_Bench_v3.csv"
# Train/test loaded from partition column in IRC_Bench_v3.csv
DESCRIPTIONS_JSON = SCRIPT_DIR / "entity_descriptions" / "veterans_v2_descriptions.json"
MODELS_DIR = SCRIPT_DIR / "trained_models"
RESULTS_DIR = SCRIPT_DIR / "results"

BASE_MODEL_MAP = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2": "intfloat/e5-base-v2",
    "e5-base": "intfloat/e5-base-v2",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> list[dict]:
    """Load a CSV file and return a list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_descriptions(path: Path) -> dict[str, dict]:
    """Load entity descriptions JSON. Keys are lowercase entity names."""
    if not path.exists():
        logger.warning("Descriptions file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_entity_text(
    entity: str,
    entity_type: str,
    mode: str,
    descriptions: dict[str, dict],
    use_type_prefix: bool = True,
) -> str:
    """
    Build the target text for an entity depending on mode.

    For 'name' mode:  "Place: Pearl Harbor"
    For 'description' mode: "Place: Pearl Harbor was a surprise military strike..."
    For 'both' mode: caller handles by generating two examples.
    """
    entity_lower = entity.strip().lower()
    prefix = f"{entity_type}: " if use_type_prefix and entity_type else ""

    if mode == "name":
        return f"{prefix}{entity}"

    # description or both: try to find a description, fall back to name
    desc_entry = descriptions.get(entity_lower)
    if desc_entry and desc_entry.get("description"):
        return f"{prefix}{desc_entry['description']}"
    # Fallback to entity name
    return f"{prefix}{entity}"


def build_training_examples(
    rows: list[dict],
    mode: str,
    descriptions: dict[str, dict],
    use_type_prefix: bool = True,
) -> list[InputExample]:
    """
    Build InputExample pairs for sentence-transformers training.

    Each example is (query_text, entity_target) where:
      - query_text is the implicit reference text
      - entity_target is the entity name, description, or both
    """
    examples = []
    for row in rows:
        text = row["text"].strip()
        entity = row["entity"].strip()
        entity_type = row.get("entity_type", "").strip()

        if not text or not entity:
            continue

        if mode == "both":
            # Add a pair for the name
            name_target = get_entity_text(
                entity, entity_type, "name", descriptions, use_type_prefix
            )
            examples.append(InputExample(texts=[text, name_target]))
            # Add a pair for the description
            desc_target = get_entity_text(
                entity, entity_type, "description", descriptions, use_type_prefix
            )
            # Only add description pair if it differs from name pair
            if desc_target != name_target:
                examples.append(InputExample(texts=[text, desc_target]))
        else:
            target = get_entity_text(
                entity, entity_type, mode, descriptions, use_type_prefix
            )
            examples.append(InputExample(texts=[text, target]))

    return examples


def build_entity_corpus(
    train_rows: list[dict],
    test_rows: list[dict],
    mode: str,
    descriptions: dict[str, dict],
    use_type_prefix: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Build a complete corpus of entity representations for retrieval evaluation.

    Returns (entity_names, entity_texts) where entity_texts are the encoded
    representations and entity_names are the canonical entity strings.
    """
    # Collect all unique entities from both train and test
    entity_info = {}  # entity_lower -> (entity, entity_type)
    for row in train_rows + test_rows:
        entity = row["entity"].strip()
        entity_type = row.get("entity_type", "").strip()
        entity_lower = entity.lower()
        if entity_lower not in entity_info:
            entity_info[entity_lower] = (entity, entity_type)

    entity_names = []
    entity_texts = []
    for entity_lower in sorted(entity_info.keys()):
        entity, entity_type = entity_info[entity_lower]
        entity_names.append(entity)
        # For eval, use the description mode if available; for 'both' prefer description
        eval_mode = "description" if mode in ("description", "both") else "name"
        target = get_entity_text(
            entity, entity_type, eval_mode, descriptions, use_type_prefix
        )
        entity_texts.append(target)

    return entity_names, entity_texts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
try:
    from sentence_transformers.sentence_transformer.evaluation import BaseEvaluator
except ImportError:
    try:
        from sentence_transformers.evaluation import SentenceEvaluator as BaseEvaluator
    except ImportError:
        BaseEvaluator = object  # fallback: plain class


class IRCRetrievalEvaluator(BaseEvaluator):
    """
    Custom evaluator that computes Hit@k and MRR on a retrieval task.

    Encodes all test texts and all entities, then ranks entities by cosine
    similarity for each test text.
    """

    def __init__(
        self,
        test_rows: list[dict],
        entity_names: list[str],
        entity_texts: list[str],
        name: str = "irc_retrieval",
        batch_size: int = 128,
    ):
        self.test_texts = [r["text"].strip() for r in test_rows]
        self.test_entities = [r["entity"].strip().lower() for r in test_rows]
        self.entity_names = entity_names
        self.entity_names_lower = [e.lower() for e in entity_names]
        self.entity_texts = entity_texts
        self.name = name
        self.batch_size = batch_size
        self.best_mrr = 0.0

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """Run evaluation and return the primary metric (MRR) for model selection."""
        metrics = self.compute_metrics(model)

        logger.info(
            "Eval [epoch=%d, steps=%d]: Hit@1=%.4f  Hit@3=%.4f  Hit@5=%.4f  "
            "Hit@10=%.4f  MRR=%.4f",
            epoch,
            steps,
            metrics["hit_at_1"],
            metrics["hit_at_3"],
            metrics["hit_at_5"],
            metrics["hit_at_10"],
            metrics["mrr"],
        )

        if output_path and metrics["mrr"] > self.best_mrr:
            self.best_mrr = metrics["mrr"]
            metrics_path = os.path.join(output_path, f"{self.name}_best_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        # sentence-transformers uses the returned float for best-model selection
        return metrics["mrr"]

    def compute_metrics(
        self, model, return_predictions: bool = False
    ) -> dict:
        """Encode everything and compute retrieval metrics."""
        logger.info(
            "Encoding %d test texts and %d entities...",
            len(self.test_texts),
            len(self.entity_texts),
        )

        query_embeddings = model.encode(
            self.test_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        entity_embeddings = model.encode(
            self.entity_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Compute cosine similarity matrix: (num_queries, num_entities)
        # Normalize for cosine similarity
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        entity_norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / np.clip(query_norms, 1e-8, None)
        entity_embeddings = entity_embeddings / np.clip(entity_norms, 1e-8, None)

        sim_matrix = query_embeddings @ entity_embeddings.T  # (Q, E)

        # Rank entities for each query (descending similarity)
        ranked_indices = np.argsort(-sim_matrix, axis=1)

        hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
        reciprocal_ranks = []
        predictions = []

        for i, gold_entity in enumerate(self.test_entities):
            # Find the rank of the gold entity
            ranking = ranked_indices[i]
            rank = None
            for r, idx in enumerate(ranking):
                if self.entity_names_lower[idx] == gold_entity:
                    rank = r + 1  # 1-indexed
                    break

            if rank is None:
                # Gold entity not in corpus (should not happen if built correctly)
                reciprocal_ranks.append(0.0)
                for k in hits_at:
                    pass  # no hit
            else:
                reciprocal_ranks.append(1.0 / rank)
                for k in hits_at:
                    if rank <= k:
                        hits_at[k] += 1

            if return_predictions:
                top10_entities = [
                    self.entity_names[ranked_indices[i][j]] for j in range(min(10, len(self.entity_names)))
                ]
                top10_scores = [
                    float(sim_matrix[i, ranked_indices[i][j]]) for j in range(min(10, len(self.entity_names)))
                ]
                predictions.append(
                    {
                        "text": self.test_texts[i],
                        "gold_entity": self.test_entities[i],
                        "rank": rank,
                        "top10_entities": top10_entities,
                        "top10_scores": top10_scores,
                    }
                )

        n = len(self.test_entities)
        metrics = {
            "hit_at_1": hits_at[1] / n,
            "hit_at_3": hits_at[3] / n,
            "hit_at_5": hits_at[5] / n,
            "hit_at_10": hits_at[10] / n,
            "mrr": float(np.mean(reciprocal_ranks)),
            "num_queries": n,
            "num_entities": len(self.entity_names),
        }

        if return_predictions:
            metrics["predictions"] = predictions

        return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    """Main training loop."""
    logger.info("Loading data...")
    train_rows = load_csv(TRAIN_CSV)
    test_rows = load_csv(TEST_CSV)
    descriptions = load_descriptions(DESCRIPTIONS_JSON)

    logger.info("Train samples: %d", len(train_rows))
    logger.info("Test samples: %d", len(test_rows))
    logger.info("Entity descriptions loaded: %d", len(descriptions))

    # Compute entity statistics
    train_entities = set(r["entity"].strip().lower() for r in train_rows)
    test_entities = set(r["entity"].strip().lower() for r in test_rows)
    all_entities = train_entities | test_entities
    unseen_entities = test_entities - train_entities

    logger.info("Unique train entities: %d", len(train_entities))
    logger.info("Unique test entities: %d", len(test_entities))
    logger.info("Total unique entities: %d", len(all_entities))
    logger.info("Unseen test entities: %d", len(unseen_entities))

    # Count how many entities have descriptions
    desc_coverage_train = sum(
        1 for e in train_entities if e in descriptions and descriptions[e].get("description")
    )
    desc_coverage_test = sum(
        1 for e in test_entities if e in descriptions and descriptions[e].get("description")
    )
    logger.info(
        "Description coverage: train=%d/%d (%.1f%%), test=%d/%d (%.1f%%)",
        desc_coverage_train,
        len(train_entities),
        100 * desc_coverage_train / max(len(train_entities), 1),
        desc_coverage_test,
        len(test_entities),
        100 * desc_coverage_test / max(len(test_entities), 1),
    )

    # Build training examples
    examples = build_training_examples(
        train_rows, args.mode, descriptions, use_type_prefix=not args.no_type_prefix
    )
    logger.info("Training examples built: %d (mode=%s)", len(examples), args.mode)

    # Build entity corpus for evaluation
    entity_names, entity_texts = build_entity_corpus(
        train_rows, test_rows, args.mode, descriptions, use_type_prefix=not args.no_type_prefix
    )
    logger.info("Entity corpus size: %d", len(entity_names))

    # Resolve base model
    base_model_name = args.base_model
    model_id = BASE_MODEL_MAP.get(base_model_name, base_model_name)
    safe_model_name = base_model_name.replace("/", "_")

    # Output directory
    output_dir = MODELS_DIR / f"dpr_{args.mode}_{safe_model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Estimate training info
    steps_per_epoch = len(examples) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    logger.info("Base model: %s", model_id)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Epochs: %d", args.epochs)
    logger.info("Steps per epoch: %d", steps_per_epoch)
    logger.info("Total training steps: %d", total_steps)
    logger.info("Learning rate: %s", args.lr)
    logger.info("Warmup steps: %d", args.warmup_steps)
    logger.info("Output directory: %s", output_dir)

    if args.dry_run:
        logger.info("=== DRY RUN: stopping before model loading ===")
        # Print some example pairs
        logger.info("Sample training pairs:")
        for ex in examples[:5]:
            logger.info("  Query:  %s", ex.texts[0][:100])
            logger.info("  Target: %s", ex.texts[1][:100])
            logger.info("  ---")
        # Rough estimate: ~0.5s per step for MiniLM on GPU, ~2s on CPU
        est_gpu = total_steps * 0.5 / 60
        est_cpu = total_steps * 2.0 / 60
        logger.info(
            "Estimated training time: %.1f min (GPU) / %.1f min (CPU)",
            est_gpu,
            est_cpu,
        )
        return

    # Load model
    logger.info("Loading model: %s", model_id)
    model = SentenceTransformer(model_id)

    # Create DataLoader
    train_dataloader = DataLoader(
        examples, shuffle=True, batch_size=args.batch_size
    )

    # Loss: Multiple Negatives Ranking Loss (in-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
    evaluator = IRCRetrievalEvaluator(
        test_rows=test_rows,
        entity_names=entity_names,
        entity_texts=entity_texts,
        name="open_set_test",
        batch_size=args.eval_batch_size,
    )

    # Evaluate baseline before training
    logger.info("=== Baseline evaluation (before training) ===")
    baseline_metrics = evaluator.compute_metrics(model, return_predictions=False)
    logger.info(
        "Baseline: Hit@1=%.4f  Hit@3=%.4f  Hit@5=%.4f  Hit@10=%.4f  MRR=%.4f",
        baseline_metrics["hit_at_1"],
        baseline_metrics["hit_at_3"],
        baseline_metrics["hit_at_5"],
        baseline_metrics["hit_at_10"],
        baseline_metrics["mrr"],
    )

    # Determine evaluation strategy
    eval_steps = max(steps_per_epoch // 2, 1)  # evaluate twice per epoch
    logger.info("Evaluation every %d steps", eval_steps)

    # Train
    logger.info("=== Starting training ===")
    start_time = time.time()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=eval_steps,
        output_path=str(output_dir),
        optimizer_params={"lr": args.lr},
        save_best_model=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Load best model and do final evaluation
    logger.info("=== Final evaluation with best model ===")
    best_model = SentenceTransformer(str(output_dir))
    final_metrics = evaluator.compute_metrics(best_model, return_predictions=True)
    logger.info(
        "Final: Hit@1=%.4f  Hit@3=%.4f  Hit@5=%.4f  Hit@10=%.4f  MRR=%.4f",
        final_metrics["hit_at_1"],
        final_metrics["hit_at_3"],
        final_metrics["hit_at_5"],
        final_metrics["hit_at_10"],
        final_metrics["mrr"],
    )

    # Save results
    save_results(args, baseline_metrics, final_metrics, safe_model_name, elapsed)

    # Save training config
    config = {
        "mode": args.mode,
        "base_model": model_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "use_type_prefix": not args.no_type_prefix,
        "num_train_examples": len(examples),
        "num_entities": len(entity_names),
        "training_time_seconds": elapsed,
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Training config saved to %s", config_path)


def evaluate_only(args):
    """Evaluate a previously trained model without training."""
    model_path = args.model_path
    if not model_path:
        # Try to infer from mode and base model
        safe_model_name = args.base_model.replace("/", "_")
        model_path = str(MODELS_DIR / f"dpr_{args.mode}_{safe_model_name}")

    if not os.path.isdir(model_path):
        logger.error("Model not found at %s", model_path)
        sys.exit(1)

    logger.info("Loading data...")
    train_rows = load_csv(TRAIN_CSV)
    test_rows = load_csv(TEST_CSV)
    descriptions = load_descriptions(DESCRIPTIONS_JSON)

    entity_names, entity_texts = build_entity_corpus(
        train_rows, test_rows, args.mode, descriptions, use_type_prefix=not args.no_type_prefix
    )

    # Evaluate trained model
    logger.info("Loading trained model from %s", model_path)
    trained_model = SentenceTransformer(model_path)

    evaluator = IRCRetrievalEvaluator(
        test_rows=test_rows,
        entity_names=entity_names,
        entity_texts=entity_texts,
        name="open_set_test",
        batch_size=args.eval_batch_size,
    )

    trained_metrics = evaluator.compute_metrics(trained_model, return_predictions=True)
    logger.info(
        "Trained model: Hit@1=%.4f  Hit@3=%.4f  Hit@5=%.4f  Hit@10=%.4f  MRR=%.4f",
        trained_metrics["hit_at_1"],
        trained_metrics["hit_at_3"],
        trained_metrics["hit_at_5"],
        trained_metrics["hit_at_10"],
        trained_metrics["mrr"],
    )

    # Also evaluate baseline for comparison
    base_model_name = args.base_model
    model_id = BASE_MODEL_MAP.get(base_model_name, base_model_name)
    logger.info("Loading baseline model: %s", model_id)
    baseline_model = SentenceTransformer(model_id)

    baseline_metrics = evaluator.compute_metrics(baseline_model, return_predictions=False)
    logger.info(
        "Baseline: Hit@1=%.4f  Hit@3=%.4f  Hit@5=%.4f  Hit@10=%.4f  MRR=%.4f",
        baseline_metrics["hit_at_1"],
        baseline_metrics["hit_at_3"],
        baseline_metrics["hit_at_5"],
        baseline_metrics["hit_at_10"],
        baseline_metrics["mrr"],
    )

    safe_model_name = args.base_model.replace("/", "_")
    save_results(args, baseline_metrics, trained_metrics, safe_model_name, elapsed=0)


def save_results(
    args,
    baseline_metrics: dict,
    final_metrics: dict,
    safe_model_name: str,
    elapsed: float,
):
    """Save metrics JSON and predictions CSV to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{timestamp}_dpr_{args.mode}_{safe_model_name}"

    # Save metrics
    combined_metrics = {
        "timestamp": timestamp,
        "mode": args.mode,
        "base_model": args.base_model,
        "use_type_prefix": not args.no_type_prefix,
        "training_time_seconds": elapsed,
        "baseline": {
            k: v for k, v in baseline_metrics.items() if k != "predictions"
        },
        "trained": {
            k: v for k, v in final_metrics.items() if k != "predictions"
        },
        "improvement": {
            "hit_at_1": final_metrics["hit_at_1"] - baseline_metrics["hit_at_1"],
            "hit_at_3": final_metrics["hit_at_3"] - baseline_metrics["hit_at_3"],
            "hit_at_5": final_metrics["hit_at_5"] - baseline_metrics["hit_at_5"],
            "hit_at_10": final_metrics["hit_at_10"] - baseline_metrics["hit_at_10"],
            "mrr": final_metrics["mrr"] - baseline_metrics["mrr"],
        },
    }

    metrics_path = RESULTS_DIR / f"{prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save predictions CSV
    predictions = final_metrics.get("predictions", [])
    if predictions:
        preds_path = RESULTS_DIR / f"{prefix}_predictions.csv"
        with open(preds_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "text",
                "gold_entity",
                "rank",
                "pred_1",
                "score_1",
                "pred_2",
                "score_2",
                "pred_3",
                "score_3",
                "pred_4",
                "score_4",
                "pred_5",
                "score_5",
            ])
            for p in predictions:
                row = [p["text"], p["gold_entity"], p["rank"]]
                for j in range(5):
                    if j < len(p["top10_entities"]):
                        row.append(p["top10_entities"][j])
                        row.append(f"{p['top10_scores'][j]:.4f}")
                    else:
                        row.extend(["", ""])
                writer.writerow(row)
        logger.info("Predictions saved to %s", preds_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  DPR Fine-Tuning Results: mode={args.mode}, model={args.base_model}")
    print("=" * 70)
    print(f"  {'Metric':<12} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
    print("  " + "-" * 44)
    for metric in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr"]:
        b = baseline_metrics[metric]
        t = final_metrics[metric]
        d = t - b
        sign = "+" if d >= 0 else ""
        print(f"  {metric:<12} {b:>10.4f} {t:>10.4f} {sign}{d:>9.4f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPR fine-tuning for IRC implicit entity recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["name", "description", "both"],
        default="description",
        help="Training mode: match text to entity name, description, or both (default: description)",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base model name or path (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Evaluation batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)",
    )
    parser.add_argument(
        "--no-type-prefix",
        action="store_true",
        help="Disable entity-type prefixing (e.g., 'Place: Pearl Harbor')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dataset stats and estimated training time without actually training",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a previously trained model without training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a trained model directory (for --eval-only)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("IRC DPR Fine-Tuning Pipeline")
    logger.info("Mode: %s", args.mode)
    logger.info("Base model: %s", args.base_model)

    if args.eval_only:
        evaluate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
