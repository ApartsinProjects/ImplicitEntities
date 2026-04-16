"""
B4: Closed-set entity recognition.

Instead of asking the LLM to freely generate entity names (open-set),
provide the full entity list and ask it to PICK from the list.

This converts the task from open-ended generation to closed-set ranking,
dramatically reducing hallucination.

For large entity lists (500+), we chunk the list into groups and use
a two-pass approach:
  Pass 1: Score each chunk of ~50 entities against the text
  Pass 2: Re-rank the top candidates from all chunks

Usage:
    python run_closedset.py --dataset veterans_t2e_v2
    python run_closedset.py --dataset twitter --max-samples 50
    python run_closedset.py --dry-run --dataset veterans_t2e
"""
import asyncio
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call, estimate_cost
from run_experiments import (
    Sample, load_dataset, evaluate_predictions, compute_metrics,
    print_metrics, save_results, parse_ranked_guesses, smoke_test_llm,
    RESULTS_DIR,
)

CHUNK_SIZE = 50  # entities per chunk


def chunk_entities(entities: list[str], chunk_size: int = CHUNK_SIZE) -> list[list[str]]:
    """Split entity list into chunks."""
    return [entities[i:i+chunk_size] for i in range(0, len(entities), chunk_size)]


def build_closedset_prompts(
    samples: list[Sample],
    entity_list: list[str],
    include_type: bool = True,
) -> list[list[dict]]:
    """Build prompts with the full entity list for closed-set recognition."""
    # Format entity list as numbered list
    entity_str = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(entity_list))

    prompts = []
    for s in samples:
        type_hint = f" (entity type: {s.entity_type})" if include_type and s.entity_type else ""

        messages = [
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities. "
                "Given a text and a list of candidate entities, determine which entity "
                "is being implicitly described. Output your top 3 guesses from the list, "
                "one per line, most likely first. Output ONLY entity names from the list."
            )},
            {"role": "user", "content": (
                f"Text{type_hint}:\n\"{s.text}\"\n\n"
                f"Candidate entities:\n{entity_str}\n\n"
                f"Which entity from the list above is implicitly referenced? "
                f"Give your top 3 guesses, one per line:"
            )},
        ]
        prompts.append(messages)
    return prompts


def build_chunked_pass1_prompts(
    sample: Sample,
    entity_chunks: list[list[str]],
    include_type: bool = True,
) -> list[list[dict]]:
    """Build pass-1 prompts: one per chunk, asking for best match from that chunk."""
    type_hint = f" (entity type: {sample.entity_type})" if include_type and sample.entity_type else ""
    prompts = []
    for chunk in entity_chunks:
        entity_str = ", ".join(chunk)
        messages = [
            {"role": "system", "content": (
                "You are identifying implicitly referenced entities. "
                "Given a text and a list of candidates, pick the single best match. "
                "Output ONLY the entity name, nothing else. If none match, output NONE."
            )},
            {"role": "user", "content": (
                f"Text{type_hint}: \"{sample.text}\"\n\n"
                f"Candidates: {entity_str}\n\n"
                f"Best match:"
            )},
        ]
        prompts.append(messages)
    return prompts


def build_rerank_prompt(
    sample: Sample,
    candidates: list[str],
    include_type: bool = True,
) -> list[dict]:
    """Build pass-2 re-ranking prompt from shortlisted candidates."""
    type_hint = f" (entity type: {sample.entity_type})" if include_type and sample.entity_type else ""
    candidate_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    return [
        {"role": "system", "content": (
            "You are an expert at identifying implicitly referenced entities. "
            "Given a text and a shortlist of candidates, rank the top 3 most likely matches. "
            "Output exactly 3 entity names from the list, one per line, most likely first."
        )},
        {"role": "user", "content": (
            f"Text{type_hint}: \"{sample.text}\"\n\n"
            f"Shortlisted candidates:\n{candidate_str}\n\n"
            f"Top 3 matches (most likely first):"
        )},
    ]


async def run_closedset_direct(
    samples: list[Sample],
    entity_list: list[str],
    model: str,
    concurrency: int = 10,
) -> list[list[str]]:
    """Direct closed-set: full entity list in every prompt (for small lists)."""
    print(f"\n  [Closed-set direct] {len(entity_list)} entities in prompt")
    prompts = build_closedset_prompts(samples, entity_list)

    responses = await batch_call(
        prompts, model=model, temperature=0.2,
        max_tokens=100, concurrency=concurrency, progress_every=50,
    )

    all_predictions = []
    for resp in responses:
        guesses = parse_ranked_guesses(resp or "")
        all_predictions.append(guesses)
    return all_predictions


async def run_closedset_chunked(
    samples: list[Sample],
    entity_list: list[str],
    model: str,
    concurrency: int = 10,
    chunk_size: int = CHUNK_SIZE,
) -> list[list[str]]:
    """
    Two-pass chunked closed-set for large entity lists.
    Pass 1: Find best match in each chunk of ~50 entities
    Pass 2: Re-rank top candidates from all chunks
    """
    chunks = chunk_entities(entity_list, chunk_size)
    n_chunks = len(chunks)
    print(f"\n  [Closed-set chunked] {len(entity_list)} entities in {n_chunks} chunks of ~{chunk_size}")

    # Pass 1: score each chunk for each sample
    print(f"  Pass 1: {len(samples) * n_chunks} prompts ({len(samples)} samples x {n_chunks} chunks)...")
    all_pass1_prompts = []
    prompt_map = []  # (sample_idx, chunk_idx)

    for si, sample in enumerate(samples):
        for ci, chunk in enumerate(chunks):
            prompt = build_chunked_pass1_prompts(sample, [chunk])[0]
            all_pass1_prompts.append(prompt)
            prompt_map.append((si, ci))

    pass1_responses = await batch_call(
        all_pass1_prompts, model=model, temperature=0.1,
        max_tokens=50, concurrency=concurrency, progress_every=100,
    )

    # Collect candidates per sample
    sample_candidates = defaultdict(list)
    for idx, resp in enumerate(pass1_responses):
        si, ci = prompt_map[idx]
        if resp and resp.strip().upper() != "NONE":
            candidate = resp.strip().strip("\"'")
            if candidate and len(candidate) > 1:
                sample_candidates[si].append(candidate)

    # Pass 2: re-rank shortlisted candidates
    print(f"  Pass 2: Re-ranking {len(samples)} samples...")
    rerank_prompts = []
    rerank_indices = []
    for si, sample in enumerate(samples):
        candidates = sample_candidates.get(si, [])
        if len(candidates) > 1:
            rerank_prompts.append(build_rerank_prompt(sample, candidates))
            rerank_indices.append(si)
        # If 0 or 1 candidates, no re-ranking needed

    if rerank_prompts:
        rerank_responses = await batch_call(
            rerank_prompts, model=model, temperature=0.1,
            max_tokens=100, concurrency=concurrency, progress_every=50,
        )
    else:
        rerank_responses = []

    # Build final predictions
    all_predictions = [[] for _ in samples]

    # Samples that went through re-ranking
    for idx, resp in enumerate(rerank_responses):
        si = rerank_indices[idx]
        guesses = parse_ranked_guesses(resp or "")
        if guesses:
            all_predictions[si] = guesses
        else:
            all_predictions[si] = sample_candidates.get(si, [])

    # Samples that skipped re-ranking (0-1 candidates)
    for si in range(len(samples)):
        if not all_predictions[si]:
            all_predictions[si] = sample_candidates.get(si, [])

    parsed_ok = sum(1 for p in all_predictions if p)
    print(f"  Results: {parsed_ok}/{len(samples)} samples have predictions")

    return all_predictions


async def main():
    parser = argparse.ArgumentParser(description="Closed-set entity recognition")
    parser.add_argument("--dataset", default="veterans_t2e_v2")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001:floor")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    samples, unique_entities = load_dataset(args.dataset)
    if not samples:
        print("ERROR: Could not load dataset")
        return

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]

    entity_list = sorted(unique_entities)
    print(f"\n  Dataset: {args.dataset}")
    print(f"  Samples: {len(samples)}")
    print(f"  Unique entities: {len(entity_list)}")

    # Decide approach based on entity list size
    if len(entity_list) <= 100:
        approach = "direct"
        n_prompts = len(samples)
    else:
        approach = "chunked"
        n_chunks = len(chunk_entities(entity_list, args.chunk_size))
        n_prompts = len(samples) * n_chunks + len(samples)  # pass1 + pass2

    print(f"  Approach: {approach}")
    print(f"  Estimated prompts: {n_prompts}")

    est = estimate_cost(n_prompts, avg_prompt_tokens=200, avg_completion_tokens=30, model=args.model)
    print(f"  Estimated cost: ${est['estimated_cost_usd']}")

    if args.dry_run:
        print("\n  DRY RUN. No API calls made.")
        return

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if approach == "direct":
        predictions = await run_closedset_direct(
            samples, entity_list, args.model, args.concurrency)
    else:
        predictions = await run_closedset_chunked(
            samples, entity_list, args.model, args.concurrency, args.chunk_size)

    # Evaluate
    eval_results = await evaluate_predictions(
        samples, predictions, model=args.model, concurrency=args.concurrency)
    metrics = compute_metrics(eval_results)
    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["approach"] = approach
    metrics["n_entities"] = len(entity_list)

    print_metrics(metrics, f"{args.dataset} / closed-set ({approach})")
    save_results(eval_results, metrics, args.dataset, f"closedset_{approach}",
                 args.model, timestamp)


if __name__ == "__main__":
    asyncio.run(main())
