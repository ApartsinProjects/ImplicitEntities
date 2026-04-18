"""
IRC-Bench v5: Closed-World Experiments (C1-C6)
===============================================
Rank entities from the knowledge base using embedding similarity.

C1: BGE-base → entity name
C2: BGE-base → entity description
C3: BGE-base → entity wiki sentence
C4: DPR fine-tuned → entity name
C5: DPR fine-tuned → entity description
C6: DPR fine-tuned → entity wiki sentence

Usage:
  python run_closed_world.py --exp C1        # Single experiment
  python run_closed_world.py --exp C1,C2,C3  # Multiple
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

V5_DIR = Path(__file__).parent
DATA_DIR = V5_DIR.parent / "data" / "benchmark"
RESULTS_DIR = V5_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

ENTITY_REPR = {
    "C1": "name",
    "C2": "description",
    "C3": "wiki",
    "C4": "name",       # Same repr, but model will be fine-tuned
    "C5": "description",
    "C6": "wiki",
}


def load_test_data():
    with open(DATA_DIR / "irc_bench_v5_test.json", encoding="utf-8") as f:
        return json.load(f)


def load_entity_kb():
    with open(DATA_DIR / "entity_kb.json", encoding="utf-8") as f:
        return json.load(f)


def get_entity_text(entity_info, repr_mode):
    """Get entity representation text for embedding."""
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


def run_retrieval(exp_id, model_path=None):
    """Run embedding retrieval experiment."""
    repr_mode = ENTITY_REPR[exp_id]
    print(f"\nExperiment {exp_id}: BGE-base → entity {repr_mode}")

    # Load data
    test_data = load_test_data()
    entity_kb = load_entity_kb()
    print(f"Test samples: {len(test_data)}")
    print(f"Entity KB: {len(entity_kb)} entities")

    # Load model
    model_name = model_path or MODEL_NAME
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode all entities
    print("Encoding entities...")
    entity_names = list(entity_kb.keys())
    entity_texts = [get_entity_text(entity_kb[name], repr_mode) for name in entity_names]
    entity_embeddings = model.encode(entity_texts, show_progress_bar=True, batch_size=64)
    entity_embeddings = entity_embeddings / np.linalg.norm(entity_embeddings, axis=1, keepdims=True)

    # Encode test queries
    print("Encoding test queries...")
    query_texts = [s["implicit_text"] for s in test_data]
    query_embeddings = model.encode(query_texts, show_progress_bar=True, batch_size=64)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Compute similarities and rank
    print("Computing rankings...")
    predictions = []
    hit_at = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0

    for i, sample in enumerate(test_data):
        scores = query_embeddings[i] @ entity_embeddings.T
        ranked_indices = np.argsort(scores)[::-1]

        gold = sample["entity"]
        top_k_names = [entity_names[idx] for idx in ranked_indices[:10]]

        # Find rank of gold entity
        rank = -1
        for r, idx in enumerate(ranked_indices):
            if entity_names[idx] == gold:
                rank = r + 1
                break

        # Metrics
        for k in hit_at:
            if 0 < rank <= k:
                hit_at[k] += 1
        if rank > 0:
            mrr_sum += 1.0 / rank

        predictions.append({
            "uid": sample["uid"],
            "gold_entity": gold,
            "gold_type": sample.get("entity_type", ""),
            "prediction": top_k_names[0] if top_k_names else "",
            "top_10": top_k_names,
            "gold_rank": rank,
            "implicit_text": sample["implicit_text"],
        })

    n = len(test_data)
    metrics = {
        "exp_id": exp_id,
        "model": model_name,
        "entity_repr": repr_mode,
        "n_test": n,
        "n_entities": len(entity_kb),
        "hit_at_1": round(hit_at[1] / n, 4),
        "hit_at_3": round(hit_at[3] / n, 4),
        "hit_at_5": round(hit_at[5] / n, 4),
        "hit_at_10": round(hit_at[10] / n, 4),
        "mrr": round(mrr_sum / n, 4),
    }

    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save
    pred_path = RESULTS_DIR / f"{exp_id}_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    metrics_path = RESULTS_DIR / f"{exp_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {pred_path}, {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment IDs (comma-separated)")
    parser.add_argument("--model-path", type=str, help="Path to fine-tuned model (for C4-C6)")
    args = parser.parse_args()

    exp_ids = [e.strip() for e in args.exp.split(",")]

    for exp_id in exp_ids:
        if exp_id in ("C4", "C5", "C6") and not args.model_path:
            print(f"Skipping {exp_id}: needs --model-path for fine-tuned model")
            continue
        run_retrieval(exp_id, model_path=args.model_path)


if __name__ == "__main__":
    main()
