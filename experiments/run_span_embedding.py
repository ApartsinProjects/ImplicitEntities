"""
Mention-span extraction + embedding: BLINK-inspired baseline.

Instead of embedding the full narrative text, first extract the key
descriptive phrase (mention span) that refers to the implicit entity,
then embed only that span for entity matching.

Pipeline:
  1. LLM extracts mention span from full text
  2. Encode mention span with sentence-transformer
  3. Compare to entity name/description embeddings
  4. Rank by cosine similarity

This tests whether focused span extraction improves embedding retrieval
compared to full-text embedding (our current approach).

Usage:
    python run_span_embedding.py --dataset bench_veterans_t2e
    python run_span_embedding.py --dataset bench_veterans_t2e --entity-repr description
    python run_span_embedding.py --dry-run
"""
import asyncio
import argparse
import csv
import json
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call, estimate_cost
from run_experiments import (
    load_dataset, evaluate_predictions, compute_metrics, print_metrics,
    Sample, RESULTS_DIR,
)
from run_battery import save_battery_results

DESCRIPTIONS_DIR = Path(__file__).parent / "entity_descriptions"
DEFAULT_MODEL = "google/gemini-2.0-flash-001:floor"


def load_all_descriptions() -> dict:
    all_desc = {}
    for f in DESCRIPTIONS_DIR.glob("*_descriptions.json"):
        data = json.load(open(f, encoding="utf-8"))
        for key, val in data.items():
            all_desc[key.lower().strip()] = val
    return all_desc


async def extract_mention_spans(
    samples: list[Sample],
    model: str = DEFAULT_MODEL,
    concurrency: int = 15,
) -> list[str]:
    """Use LLM to extract the key descriptive phrase from each text."""
    print(f"  Extracting mention spans from {len(samples)} texts...")

    prompts = []
    for s in samples:
        prompts.append([
            {"role": "system", "content": (
                "Extract the key phrase from the text that most directly describes "
                "an implicitly referenced entity. The entity is NOT named in the text. "
                "Output ONLY the descriptive phrase (5-30 words), nothing else. "
                "Do not name the entity, just extract the describing phrase."
            )},
            {"role": "user", "content": f"Text: \"{s.text[:500]}\"\n\nDescriptive phrase:"},
        ])

    responses = await batch_call(
        prompts, model=model, temperature=0.1, max_tokens=60,
        concurrency=concurrency, progress_every=100,
    )

    spans = []
    for resp in responses:
        span = (resp or "").strip().strip('"\'')
        spans.append(span if span else "")

    extracted = sum(1 for s in spans if s)
    print(f"  Extracted {extracted}/{len(samples)} spans")
    return spans


async def run_span_embedding(
    dataset_name: str,
    entity_repr: str = "name",
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str = DEFAULT_MODEL,
    concurrency: int = 15,
    results_dir: Path = None,
):
    """Run the full span-extraction + embedding pipeline."""
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"\n{'='*60}")
    print(f"  SPAN EMBEDDING: {dataset_name}")
    print(f"  Entity repr: {entity_repr}, Embedding: {embedding_model}")
    print(f"{'='*60}")

    samples, unique_entities = load_dataset(dataset_name)
    if not samples:
        return {}

    # Step 1: Extract mention spans via LLM
    spans = await extract_mention_spans(samples, llm_model, concurrency)

    # Step 2: Load embedding model
    print(f"  Loading {embedding_model}...")
    sbert = SentenceTransformer(embedding_model, trust_remote_code=True)
    print(f"  Device: {sbert.device}")

    # Step 3: Prepare entity representations
    descriptions = load_all_descriptions()
    entity_texts = []
    for ent in unique_entities:
        if entity_repr == "description":
            key = ent.lower().strip()
            desc = descriptions.get(key, {}).get("description", "")
            entity_texts.append(desc if desc else ent)
        else:
            entity_texts.append(ent)

    # Step 4: Encode spans and entities
    print(f"  Encoding {len(spans)} spans + {len(entity_texts)} entities...")
    span_embs = sbert.encode(spans, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    entity_embs = sbert.encode(entity_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    # Also encode full texts for comparison
    full_text_embs = sbert.encode(
        [s.text[:500] for s in samples], batch_size=128,
        show_progress_bar=False, normalize_embeddings=True,
    )

    # Step 5: Rank entities for each span
    sim_span = np.array(span_embs) @ np.array(entity_embs).T
    sim_full = np.array(full_text_embs) @ np.array(entity_embs).T

    predictions_span = []
    predictions_full = []
    for i in range(len(samples)):
        top_span = np.argsort(sim_span[i])[::-1][:10]
        predictions_span.append([unique_entities[j] for j in top_span])
        top_full = np.argsort(sim_full[i])[::-1][:10]
        predictions_full.append([unique_entities[j] for j in top_full])

    # Free GPU
    del sbert
    torch.cuda.empty_cache()

    # Step 6: Evaluate both
    if results_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = RESULTS_DIR / f"span_embedding_{ts}"
        results_dir.mkdir(parents=True, exist_ok=True)

    for method_name, preds in [("span_embedding", predictions_span), ("fulltext_embedding", predictions_full)]:
        eval_results = await evaluate_predictions(
            samples, preds, model=f"{method_name}_{embedding_model}",
            concurrency=concurrency,
        )
        metrics = compute_metrics(eval_results)
        print_metrics(metrics, f"{dataset_name} / {method_name} / {entity_repr}")
        save_battery_results(
            eval_results, metrics, dataset_name,
            f"{method_name}_{entity_repr}", embedding_model, results_dir,
        )

    return {"span": predictions_span, "full": predictions_full}


async def main():
    parser = argparse.ArgumentParser(description="Mention-span embedding experiment")
    parser.add_argument("--dataset", default="bench_veterans_t2e")
    parser.add_argument("--entity-repr", default="name", choices=["name", "description"])
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        samples, ents = load_dataset(args.dataset)
        est = estimate_cost(len(samples), avg_prompt_tokens=100, avg_completion_tokens=30, model=args.llm_model)
        print(f"  Dataset: {args.dataset} ({len(samples)} samples)")
        print(f"  LLM cost for span extraction: ${est['estimated_cost_usd']}")
        print(f"  Embedding: {args.embedding_model} (local, free)")
        return

    await run_span_embedding(
        args.dataset, args.entity_repr, args.embedding_model,
        args.llm_model, args.concurrency,
    )


if __name__ == "__main__":
    asyncio.run(main())
