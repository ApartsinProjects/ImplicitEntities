"""
IRC-Bench v5: DPR Fine-tuning (C4-C6)
======================================
Fine-tune BGE-base bi-encoder on train split for entity retrieval.

Usage:
  python train_dpr.py --repr name         # C4
  python train_dpr.py --repr description  # C5
  python train_dpr.py --repr wiki         # C6
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

random.seed(42)

V5_DIR = Path(__file__).parent
DATA_DIR = V5_DIR.parent / "data" / "benchmark"
MODELS_DIR = V5_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 48
EPOCHS = 3
LR = 2e-5


def get_entity_text(entity_info, repr_mode):
    name = entity_info.get("entity", "")
    if repr_mode == "name":
        return name
    elif repr_mode == "description":
        desc = entity_info.get("description", "")
        return f"{name}: {desc}" if desc else name
    elif repr_mode == "wiki":
        wiki = entity_info.get("description_wiki", "")
        return wiki if wiki else entity_info.get("description", name)
    return name


def build_training_data(repr_mode):
    """Build training examples: (implicit_text, entity_repr) pairs."""
    with open(DATA_DIR / "irc_bench_v5_train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "entity_kb.json", encoding="utf-8") as f:
        kb = json.load(f)

    examples = []
    for sample in train_data:
        entity = sample["entity"]
        if entity not in kb:
            continue
        entity_text = get_entity_text(kb[entity], repr_mode)
        examples.append(InputExample(
            texts=[sample["implicit_text"], entity_text],
        ))

    random.shuffle(examples)
    print(f"Training examples: {len(examples)}")
    return examples


def build_dev_evaluator(repr_mode):
    """Build IR evaluator from dev split."""
    with open(DATA_DIR / "irc_bench_v5_dev.json", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(DATA_DIR / "entity_kb.json", encoding="utf-8") as f:
        kb = json.load(f)

    queries = {}
    corpus = {}
    relevant = {}

    # Build corpus from all entities
    for name, info in kb.items():
        cid = f"e_{name}"
        corpus[cid] = get_entity_text(info, repr_mode)

    # Build queries from dev samples
    for i, sample in enumerate(dev_data):
        qid = f"q_{i}"
        queries[qid] = sample["implicit_text"]
        entity = sample["entity"]
        cid = f"e_{entity}"
        if cid in corpus:
            relevant[qid] = {cid: 1}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name=f"dev_{repr_mode}",
        show_progress_bar=True,
    )
    return evaluator


def train(repr_mode):
    exp_map = {"name": "C4", "description": "C5", "wiki": "C6"}
    exp_id = exp_map[repr_mode]
    print(f"\n{'='*60}")
    print(f"  DPR Training: {exp_id} (repr={repr_mode})")
    print(f"{'='*60}")

    # Load model
    model = SentenceTransformer(BASE_MODEL)
    print(f"Model: {BASE_MODEL}, device: {model.device}")

    # Build data
    examples = build_training_data(repr_mode)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
    evaluator = build_dev_evaluator(repr_mode)

    # Train
    output_path = str(MODELS_DIR / f"dpr_{repr_mode}")
    print(f"Output: {output_path}")
    print(f"Training: {len(examples)} examples, {EPOCHS} epochs, batch={BATCH_SIZE}")

    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,  # Skip eval during training (eval on test after)
        epochs=EPOCHS,
        evaluation_steps=0,
        warmup_steps=100,
        output_path=output_path,
        optimizer_params={"lr": LR},
        show_progress_bar=True,
        use_amp=True,  # Mixed precision fp16
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.0f}s")

    # Save training info
    info = {
        "exp_id": exp_id,
        "repr_mode": repr_mode,
        "base_model": BASE_MODEL,
        "n_train": len(examples),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "training_time": round(elapsed, 1),
        "output_path": output_path,
    }
    with open(MODELS_DIR / f"dpr_{repr_mode}_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr", type=str, required=True,
                        choices=["name", "description", "wiki"])
    args = parser.parse_args()
    train(args.repr)


if __name__ == "__main__":
    main()
