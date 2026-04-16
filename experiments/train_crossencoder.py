"""
Cross-encoder re-ranker training for IRC implicit entity recognition.

Trains a cross-encoder that scores (text, entity_candidate) pairs for relevance.
Used as a second-stage re-ranker after bi-encoder retrieval of top-K candidates.

Training data construction:
  - Positive pair: (text, correct_entity) with label 1.0
  - Hard negatives: 4-7 wrong entities from the same entity_type
  - Easy negatives: 2-3 random entities from other types
  - ~8 training pairs per sample

Supported base models:
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, decent)
  - cross-encoder/ms-marco-electra-base    (better quality)
  - BAAI/bge-reranker-base                 (state-of-art reranker)

Usage:
    python train_crossencoder.py --mode train --base-model cross-encoder/ms-marco-MiniLM-L-6-v2
    python train_crossencoder.py --mode train --base-model BAAI/bge-reranker-base --epochs 5
    python train_crossencoder.py --mode eval --model-path experiments/trained_models/crossencoder_xxx
    python train_crossencoder.py --dry-run
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmark_v2"
DESCRIPTIONS_DIR = EXPERIMENTS_DIR / "entity_descriptions"
MODELS_DIR = EXPERIMENTS_DIR / "trained_models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

TRAIN_CSV = DATA_DIR / "irc_benchmark_v2_train_entity_split.csv"
TEST_CSV = DATA_DIR / "irc_benchmark_v2_test_open_set.csv"
DESCRIPTIONS_JSON = DESCRIPTIONS_DIR / "veterans_v2_descriptions.json"

# ── Base model registry ──────────────────────────────────────────────────────

BASE_MODELS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {
        "label": "MiniLM-L6 (MMarco)",
        "note": "Fast, 22M params, good baseline",
    },
    "cross-encoder/ms-marco-electra-base": {
        "label": "ELECTRA-base (MMarco)",
        "note": "Better quality, 110M params",
    },
    "BAAI/bge-reranker-base": {
        "label": "BGE-reranker-base",
        "note": "State-of-art reranker, 110M params",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════


def load_csv_samples(csv_path: Path) -> pd.DataFrame:
    """Load benchmark CSV, return DataFrame with required columns."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    required = {"text", "entity", "entity_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path.name} missing columns: {missing}")
    df = df[df["text"].str.strip().astype(bool) & df["entity"].str.strip().astype(bool)].copy()
    df["text"] = df["text"].str.strip()
    df["entity"] = df["entity"].str.strip()
    df["entity_type"] = df["entity_type"].str.strip()
    print(f"  Loaded {len(df)} samples from {csv_path.name}")
    return df


def load_descriptions(path: Path) -> dict[str, dict]:
    """Load entity descriptions JSON. Keys are lowercased entity names."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entity descriptions from {path.name}")
    return data


def get_entity_repr(
    entity: str,
    descriptions: dict[str, dict],
    mode: str = "both",
) -> str:
    """
    Build text representation of an entity for the cross-encoder input.

    mode:
      - "name": just the entity name
      - "description": description only (falls back to name)
      - "both": "entity_name: description" (falls back to name)
    """
    if mode == "name":
        return entity

    key = entity.lower().strip()
    desc_entry = descriptions.get(key, {})
    desc_text = desc_entry.get("description", "").strip()

    if mode == "description":
        return desc_text if desc_text else entity

    # mode == "both"
    if desc_text:
        return f"{entity}: {desc_text}"
    return entity


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING PAIR CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════


def build_entity_type_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build mapping from entity_type to list of unique entities."""
    type_index: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        etype = row["entity_type"]
        entity = row["entity"]
        if entity not in type_index[etype]:
            type_index[etype].append(entity)
    for etype in type_index:
        type_index[etype] = sorted(set(type_index[etype]))
    print(f"  Entity type index: {len(type_index)} types, "
          f"{sum(len(v) for v in type_index.values())} total entity slots")
    return dict(type_index)


def build_training_pairs(
    df: pd.DataFrame,
    descriptions: dict[str, dict],
    entity_repr_mode: str = "both",
    hard_neg_count: int = 5,
    easy_neg_count: int = 3,
    max_samples: int = 0,
    seed: int = 42,
) -> list[tuple[str, str, float]]:
    """
    Build (text, entity_repr, label) training triples.

    For each sample:
      - 1 positive: (text, correct_entity_repr) with label 1.0
      - hard_neg_count negatives from same entity_type (harder)
      - easy_neg_count negatives from random other types (easier)
      - All negatives get label 0.0

    Returns list of (text, entity_repr, label) tuples.
    """
    rng = random.Random(seed)

    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {max_samples} samples")

    type_index = build_entity_type_index(df)
    all_entities = sorted(set(df["entity"].tolist()))

    pairs = []
    skipped_hard = 0
    skipped_easy = 0

    for _, row in df.iterrows():
        text = row["text"]
        gold_entity = row["entity"]
        gold_type = row["entity_type"]

        # Positive pair
        gold_repr = get_entity_repr(gold_entity, descriptions, entity_repr_mode)
        pairs.append((text, gold_repr, 1.0))

        # Hard negatives: same entity_type, different entity
        same_type_entities = [
            e for e in type_index.get(gold_type, [])
            if e.lower() != gold_entity.lower()
        ]
        if len(same_type_entities) >= hard_neg_count:
            hard_negs = rng.sample(same_type_entities, hard_neg_count)
        else:
            hard_negs = same_type_entities[:]
            skipped_hard += hard_neg_count - len(hard_negs)

        for neg_entity in hard_negs:
            neg_repr = get_entity_repr(neg_entity, descriptions, entity_repr_mode)
            pairs.append((text, neg_repr, 0.0))

        # Easy negatives: random entities from other types
        other_entities = [
            e for e in all_entities
            if e.lower() != gold_entity.lower()
            and e not in same_type_entities
        ]
        if len(other_entities) >= easy_neg_count:
            easy_negs = rng.sample(other_entities, easy_neg_count)
        else:
            easy_negs = other_entities[:]
            skipped_easy += easy_neg_count - len(easy_negs)

        for neg_entity in easy_negs:
            neg_repr = get_entity_repr(neg_entity, descriptions, entity_repr_mode)
            pairs.append((text, neg_repr, 0.0))

    n_pos = sum(1 for _, _, l in pairs if l > 0.5)
    n_neg = len(pairs) - n_pos
    print(f"  Training pairs: {len(pairs)} total ({n_pos} pos, {n_neg} neg)")
    print(f"  Ratio: 1:{n_neg / max(n_pos, 1):.1f} (pos:neg)")
    if skipped_hard > 0:
        print(f"  Note: {skipped_hard} hard negatives skipped (not enough same-type entities)")
    if skipped_easy > 0:
        print(f"  Note: {skipped_easy} easy negatives skipped (not enough other-type entities)")

    # Shuffle
    rng.shuffle(pairs)
    return pairs


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════════════


def train_crossencoder(
    base_model: str,
    train_pairs: list[tuple[str, str, float]],
    val_pairs: list[tuple[str, str, float]] | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    output_dir: Path | None = None,
    seed: int = 42,
) -> "CrossEncoder":
    """
    Fine-tune a cross-encoder on the given training pairs.

    Returns the trained CrossEncoder instance.
    """
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import (
        CEBinaryClassificationEvaluator,
    )
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  Training cross-encoder")
    print(f"  Base model:  {base_model}")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs) if val_pairs else 0}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Warmup:      {warmup_ratio:.0%}")
    print(f"{'=' * 60}\n")

    # Build model
    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=512,
        default_activation_function=None,
    )

    # Build training data
    train_examples = [
        InputExample(texts=[text, entity], label=label)
        for text, entity, label in train_pairs
    ]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size
    )

    # Evaluator (if val set provided)
    evaluator = None
    if val_pairs and len(val_pairs) > 0:
        val_texts_a = [p[0] for p in val_pairs]
        val_texts_b = [p[1] for p in val_pairs]
        val_labels = [int(p[2] > 0.5) for p in val_pairs]
        evaluator = CEBinaryClassificationEvaluator(
            sentence_pairs=list(zip(val_texts_a, val_texts_b)),
            labels=val_labels,
            name="val",
        )

    # Warmup steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = base_model.split("/")[-1].replace(" ", "_")
        output_dir = MODELS_DIR / f"crossencoder_{model_short}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output: {output_dir}")
    print()

    t0 = time.time()

    # Train
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        evaluation_steps=max(len(train_dataloader) // 2, 100),
        save_best_model=evaluator is not None,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"  Model saved to: {output_dir}")

    # Save training config alongside model
    config = {
        "base_model": base_model,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs) if val_pairs else 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "training_time_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION (RE-RANKING)
# ═════════════════════════════════════════════════════════════════════════════


def get_biencoder_candidates(
    texts: list[str],
    all_entities: list[str],
    descriptions: dict[str, dict],
    entity_repr_mode: str = "both",
    top_k: int = 20,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> list[list[str]]:
    """
    Use a bi-encoder (sentence-transformers) to retrieve top-K entity candidates
    for each text. Returns list of lists of entity names (not representations).
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading bi-encoder: {embedding_model_name} ...")
    biencoder = SentenceTransformer(embedding_model_name)

    # Build entity representations for embedding
    entity_reprs = [
        get_entity_repr(e, descriptions, entity_repr_mode)
        for e in all_entities
    ]

    print(f"  Encoding {len(all_entities)} entities ...")
    entity_embeddings = biencoder.encode(
        entity_reprs, show_progress_bar=True, convert_to_numpy=True, batch_size=128,
    )

    print(f"  Encoding {len(texts)} texts ...")
    text_embeddings = biencoder.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64,
    )

    # Normalize for cosine similarity
    entity_embeddings = entity_embeddings / (
        np.linalg.norm(entity_embeddings, axis=1, keepdims=True) + 1e-9
    )
    text_embeddings = text_embeddings / (
        np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-9
    )

    # Compute similarities and get top-K
    candidates_per_text = []
    for i, text_emb in enumerate(text_embeddings):
        sims = text_emb @ entity_embeddings.T
        top_indices = np.argsort(sims)[::-1][:top_k]
        candidates_per_text.append([all_entities[idx] for idx in top_indices])

    print(f"  Retrieved top-{top_k} candidates for {len(texts)} texts")
    return candidates_per_text


def rerank_with_crossencoder(
    model,
    texts: list[str],
    candidates_per_text: list[list[str]],
    descriptions: dict[str, dict],
    entity_repr_mode: str = "both",
    batch_size: int = 64,
) -> list[list[str]]:
    """
    Re-rank candidate entities for each text using the cross-encoder.

    Args:
        model: CrossEncoder instance (trained or loaded)
        texts: list of input texts
        candidates_per_text: list of candidate entity lists (one per text)
        descriptions: entity description lookup
        entity_repr_mode: how to represent entities
        batch_size: scoring batch size

    Returns:
        list of re-ranked entity lists (highest score first)
    """
    reranked = []

    for i, (text, candidates) in enumerate(zip(texts, candidates_per_text)):
        if not candidates:
            reranked.append([])
            continue

        # Build pairs for scoring
        pairs = [
            [text, get_entity_repr(c, descriptions, entity_repr_mode)]
            for c in candidates
        ]

        # Score all pairs
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Sort candidates by score (descending)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        reranked.append([entity for entity, _ in scored])

    return reranked


def match_entity(prediction: str, gold: str) -> tuple[str, bool]:
    """
    Simple entity matching. Returns (tier, is_match).
    Tiers: "exact", "jaccard", "none".
    """
    pred_norm = prediction.lower().strip()
    gold_norm = gold.lower().strip()

    # Exact match
    if pred_norm == gold_norm:
        return "exact", True

    # Substring containment
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return "exact", True

    # Jaccard token overlap
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if pred_tokens and gold_tokens:
        jaccard = len(pred_tokens & gold_tokens) / len(pred_tokens | gold_tokens)
        if jaccard >= 0.5:
            return "jaccard", True

    return "none", False


def evaluate_reranking(
    texts: list[str],
    gold_entities: list[str],
    ranked_candidates: list[list[str]],
    label: str = "",
) -> dict:
    """
    Evaluate re-ranked candidates against gold entities.
    Computes Hit@1, Hit@3, Hit@5, MRR.
    """
    n = len(texts)
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    match_details = []

    for i in range(n):
        gold = gold_entities[i]
        candidates = ranked_candidates[i]

        found_rank = 0
        found_tier = "none"
        for rank, cand in enumerate(candidates, start=1):
            tier, is_match = match_entity(cand, gold)
            if is_match:
                found_rank = rank
                found_tier = tier
                break

        for k in hits:
            if 0 < found_rank <= k:
                hits[k] += 1

        if found_rank > 0:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        match_details.append({
            "text": texts[i][:120],
            "gold_entity": gold,
            "match_rank": found_rank,
            "match_tier": found_tier,
            "top3_candidates": candidates[:3],
        })

    metrics = {
        "n_samples": n,
        "n_matched": sum(1 for rr in reciprocal_ranks if rr > 0),
    }
    for k in hits:
        metrics[f"Hit@{k}"] = hits[k] / max(n, 1)
    metrics["MRR"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # Print
    print(f"\n  {'=' * 55}")
    if label:
        print(f"  RESULTS: {label}")
    print(f"  {'=' * 55}")
    print(f"  Samples: {n}  |  "
          f"Matched: {metrics['n_matched']} "
          f"({100 * metrics['n_matched'] / max(n, 1):.1f}%)")
    print()
    for k in [1, 3, 5]:
        print(f"  {'Hit@' + str(k):12s}: {metrics[f'Hit@{k}']:.4f}  "
              f"({metrics[f'Hit@{k}'] * 100:.1f}%)")
    print()
    print(f"  {'MRR':12s}: {metrics['MRR']:.4f}")
    print(f"  {'=' * 55}")

    return metrics, match_details


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN MODES
# ═════════════════════════════════════════════════════════════════════════════


def run_train(args):
    """Full training pipeline."""
    print("\n[1/4] Loading data ...")
    train_df = load_csv_samples(TRAIN_CSV)
    descriptions = load_descriptions(DESCRIPTIONS_JSON)

    print("\n[2/4] Building training pairs ...")
    all_pairs = build_training_pairs(
        train_df,
        descriptions,
        entity_repr_mode=args.entity_repr,
        hard_neg_count=max(args.neg_per_positive - 3, 2),
        easy_neg_count=min(args.neg_per_positive, 3),
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Split off a validation set (10%)
    val_size = max(int(len(all_pairs) * 0.1), 100)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]
    print(f"  Train/Val split: {len(train_pairs)} train, {len(val_pairs)} val")

    print("\n[3/4] Training cross-encoder ...")
    model = train_crossencoder(
        base_model=args.base_model,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    print("\n[4/4] Quick validation check ...")
    # Score a few positive and negative pairs from the val set
    pos_pairs = [(a, b) for a, b, l in val_pairs[:50] if l > 0.5]
    neg_pairs = [(a, b) for a, b, l in val_pairs[:200] if l < 0.5]

    if pos_pairs and neg_pairs:
        pos_scores = model.predict([[a, b] for a, b in pos_pairs[:20]])
        neg_scores = model.predict([[a, b] for a, b in neg_pairs[:20]])
        print(f"  Positive scores: mean={np.mean(pos_scores):.3f}, "
              f"min={np.min(pos_scores):.3f}, max={np.max(pos_scores):.3f}")
        print(f"  Negative scores: mean={np.mean(neg_scores):.3f}, "
              f"min={np.min(neg_scores):.3f}, max={np.max(neg_scores):.3f}")
        separation = np.mean(pos_scores) - np.mean(neg_scores)
        print(f"  Score separation: {separation:.3f}")

    print("\nTraining complete.")


def run_eval(args):
    """Evaluate cross-encoder as a re-ranker on the test set."""
    from sentence_transformers.cross_encoder import CrossEncoder

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"  ERROR: Model path not found: {model_path}")
        sys.exit(1)

    print("\n[1/5] Loading model and data ...")
    ce_model = CrossEncoder(str(model_path), max_length=512)

    # Load training config if available
    config_path = model_path / "training_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            train_config = json.load(f)
        print(f"  Loaded model trained on: {train_config.get('base_model', 'unknown')}")
        print(f"  Training time: {train_config.get('training_time_s', '?')}s, "
              f"epochs: {train_config.get('epochs', '?')}")

    test_df = load_csv_samples(TEST_CSV)
    descriptions = load_descriptions(DESCRIPTIONS_JSON)

    if args.max_samples > 0 and len(test_df) > args.max_samples:
        test_df = test_df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"  Subsampled test set to {args.max_samples} samples")

    texts = test_df["text"].tolist()
    gold_entities = test_df["entity"].tolist()

    # Collect all known entities (train + test) for the candidate pool
    print("\n[2/5] Building entity candidate pool ...")
    train_df = load_csv_samples(TRAIN_CSV)
    train_entities = set(train_df["entity"].tolist())
    test_entities = set(test_df["entity"].tolist())
    all_entities = sorted(train_entities | test_entities)
    print(f"  Entity pool: {len(all_entities)} entities "
          f"({len(train_entities)} train, {len(test_entities)} test, "
          f"{len(train_entities & test_entities)} overlap)")

    # Step 1: Bi-encoder retrieval
    print("\n[3/5] Bi-encoder retrieval (top-20) ...")
    biencoder_candidates = get_biencoder_candidates(
        texts=texts,
        all_entities=all_entities,
        descriptions=descriptions,
        entity_repr_mode=args.entity_repr,
        top_k=args.top_k,
        embedding_model_name=args.biencoder_model,
    )

    # Evaluate bi-encoder alone
    print("\n[4/5] Evaluating bi-encoder only ...")
    biencoder_metrics, _ = evaluate_reranking(
        texts, gold_entities, biencoder_candidates,
        label=f"Bi-encoder only ({args.biencoder_model})",
    )

    # Step 2: Cross-encoder re-ranking
    print("\n[5/5] Cross-encoder re-ranking ...")
    t0 = time.time()
    reranked_candidates = rerank_with_crossencoder(
        model=ce_model,
        texts=texts,
        candidates_per_text=biencoder_candidates,
        descriptions=descriptions,
        entity_repr_mode=args.entity_repr,
        batch_size=args.batch_size,
    )
    rerank_time = time.time() - t0
    print(f"  Re-ranking took {rerank_time:.1f}s "
          f"({rerank_time / max(len(texts), 1) * 1000:.0f}ms per sample)")

    reranked_metrics, match_details = evaluate_reranking(
        texts, gold_entities, reranked_candidates,
        label=f"Cross-encoder re-ranked ({model_path.name})",
    )

    # Comparison summary
    print(f"\n  {'=' * 55}")
    print(f"  COMPARISON SUMMARY")
    print(f"  {'=' * 55}")
    print(f"  {'Metric':15s}  {'Bi-encoder':>12s}  {'+ Re-ranker':>12s}  {'Delta':>10s}")
    print(f"  {'-' * 55}")
    for k in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
        bi_val = biencoder_metrics.get(k, 0)
        re_val = reranked_metrics.get(k, 0)
        delta = re_val - bi_val
        sign = "+" if delta >= 0 else ""
        print(f"  {k:15s}  {bi_val:12.4f}  {re_val:12.4f}  {sign}{delta:9.4f}")
    print(f"  {'=' * 55}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"{timestamp}_crossencoder_eval_{model_path.name}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_output = {
        "model_path": str(model_path),
        "biencoder_model": args.biencoder_model,
        "entity_repr_mode": args.entity_repr,
        "top_k": args.top_k,
        "n_test_samples": len(texts),
        "n_entities": len(all_entities),
        "biencoder_metrics": biencoder_metrics,
        "reranked_metrics": reranked_metrics,
        "rerank_time_s": round(rerank_time, 2),
        "timestamp": timestamp,
        "match_details_sample": match_details[:20],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")


def run_dry_run(args):
    """Show what would happen without actually training."""
    print("\n[DRY RUN] Checking data and configuration ...\n")

    # Check files exist
    for label, path in [
        ("Train CSV", TRAIN_CSV),
        ("Test CSV", TEST_CSV),
        ("Descriptions", DESCRIPTIONS_JSON),
    ]:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")

    if not TRAIN_CSV.exists():
        print("\n  Cannot continue without training data.")
        return

    print()
    train_df = load_csv_samples(TRAIN_CSV)
    descriptions = load_descriptions(DESCRIPTIONS_JSON)

    # Show entity type distribution
    type_counts = train_df["entity_type"].value_counts()
    print(f"\n  Entity types in training data:")
    for etype, count in type_counts.items():
        n_unique = train_df[train_df["entity_type"] == etype]["entity"].nunique()
        print(f"    {etype:20s}: {count:5d} samples, {n_unique:4d} unique entities")

    # Show what training would produce
    total_entities = train_df["entity"].nunique()
    est_pairs = len(train_df) * (1 + args.neg_per_positive)
    if args.max_samples > 0:
        est_pairs = min(args.max_samples, len(train_df)) * (1 + args.neg_per_positive)

    print(f"\n  Training configuration:")
    print(f"    Base model:       {args.base_model}")
    print(f"    Entity repr:      {args.entity_repr}")
    print(f"    Epochs:           {args.epochs}")
    print(f"    Batch size:       {args.batch_size}")
    print(f"    Learning rate:    {args.lr}")
    print(f"    Neg per positive: {args.neg_per_positive}")
    print(f"    Max samples:      {args.max_samples or 'all'}")
    print(f"    Seed:             {args.seed}")
    print(f"\n  Estimated training pairs: ~{est_pairs:,}")
    print(f"  Unique entities:          {total_entities}")

    # Description coverage
    matched = sum(
        1 for e in train_df["entity"].unique()
        if e.lower().strip() in descriptions
    )
    print(f"  Description coverage:     {matched}/{total_entities} "
          f"({100 * matched / max(total_entities, 1):.1f}%)")

    # Check dependencies
    print(f"\n  Dependency check:")
    try:
        import torch
        print(f"    [OK] torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"         CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print(f"         CUDA: not available (will use CPU)")
    except ImportError:
        print(f"    [MISSING] torch")

    try:
        import sentence_transformers
        print(f"    [OK] sentence-transformers {sentence_transformers.__version__}")
    except ImportError:
        print(f"    [MISSING] sentence-transformers (pip install sentence-transformers)")

    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        print(f"    [OK] CrossEncoder class available")
    except ImportError:
        print(f"    [MISSING] CrossEncoder (update sentence-transformers)")

    print(f"\n  Base model: {args.base_model}")
    if args.base_model in BASE_MODELS:
        info = BASE_MODELS[args.base_model]
        print(f"    Label: {info['label']}")
        print(f"    Note:  {info['note']}")
    else:
        print(f"    (Custom model, not in registry)")

    print("\n[DRY RUN] Complete. Use --mode train to start training.")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-encoder re-ranker for IRC implicit entity recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode train --base-model cross-encoder/ms-marco-MiniLM-L-6-v2
  %(prog)s --mode train --base-model BAAI/bge-reranker-base --epochs 5 --lr 1e-5
  %(prog)s --mode eval --model-path experiments/trained_models/crossencoder_xxx
  %(prog)s --dry-run
  %(prog)s --dry-run --base-model BAAI/bge-reranker-base --neg-per-positive 10
        """,
    )

    parser.add_argument(
        "--mode", choices=["train", "eval"], default=None,
        help="Operation mode: train a new model or eval an existing one",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration and data stats without training",
    )

    # Model selection
    parser.add_argument(
        "--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Base model for training (default: cross-encoder/ms-marco-MiniLM-L-6-v2)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to trained model directory (for eval mode)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument(
        "--neg-per-positive", type=int, default=8,
        help="Total negatives per positive sample (default: 8, split ~5 hard + 3 easy)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit training/eval samples (0 = use all, default: 0)",
    )
    parser.add_argument(
        "--entity-repr", choices=["name", "description", "both"], default="both",
        help="Entity representation: name only, description only, or both (default: both)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Eval-specific
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of bi-encoder candidates to re-rank (default: 20)",
    )
    parser.add_argument(
        "--biencoder-model", default="all-MiniLM-L6-v2",
        help="Bi-encoder model for candidate retrieval in eval (default: all-MiniLM-L6-v2)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'#' * 60}")
    print(f"  IRC Cross-Encoder Re-ranker")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    if args.dry_run:
        run_dry_run(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        if not args.model_path:
            print("  ERROR: --model-path is required for eval mode")
            sys.exit(1)
        run_eval(args)
    else:
        print("  ERROR: Specify --mode train, --mode eval, or --dry-run")
        sys.exit(1)


if __name__ == "__main__":
    main()
