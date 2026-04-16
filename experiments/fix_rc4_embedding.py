"""
RC4: Fix embedding method by using entity DESCRIPTIONS instead of bare names.

Problem: Cosine similarity between a 150-word narrative and a 2-word entity name
(e.g., "Pearl Harbor") is semantically meaningless. The text describes emotions
and experiences; the entity name is a proper noun.

Fix: Build an entity description index where each entity is represented by a
short descriptive sentence (like a Wikipedia first line), then compare text
embeddings to description embeddings.

Two approaches:
  A) Use LLM to generate descriptions for each entity (cheap, ~500 entities)
  B) Use Wikipedia API to fetch first sentences (free, but not all entities exist)

This script implements approach A (LLM-generated descriptions) with B as fallback.
"""
import asyncio
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call, estimate_cost

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "entity_descriptions"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_entities(dataset_path: Path) -> list[dict]:
    """Extract unique entities with types from a dataset."""
    entities = {}
    with open(dataset_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ent = row["entity"].strip()
            etype = row.get("entity_type", row.get("entity_type_original", "")).strip()
            if ent and ent not in entities:
                entities[ent] = {"entity": ent, "entity_type": etype}
    return list(entities.values())


def build_description_prompts(entities: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Build prompts to generate entity descriptions."""
    system_msg = (
        "You are building a knowledge base for entity matching. "
        "Given an entity name and type, write a single concise sentence (15-30 words) "
        "describing the entity. Include key identifying attributes: "
        "for places, mention location and what it's known for; "
        "for people, mention their role and achievements; "
        "for events, mention when, where, and what happened; "
        "for organizations, mention what they do and where. "
        "Output ONLY the description sentence, nothing else."
    )
    prompts = []
    for ent_info in entities:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Entity: {ent_info['entity']}\nType: {ent_info['entity_type']}"},
        ]
        prompts.append((ent_info, messages))
    return prompts


async def generate_descriptions(
    dataset_name: str,
    dataset_path: Path,
    model: str = "google/gemini-2.0-flash-001:floor",
    concurrency: int = 15,
):
    """Generate entity descriptions for a dataset."""
    print(f"\n{'='*60}")
    print(f"  Generating entity descriptions: {dataset_name}")
    print(f"{'='*60}")

    entities = load_entities(dataset_path)
    print(f"  Loaded {len(entities)} unique entities")

    # Cost estimate
    est = estimate_cost(len(entities), avg_prompt_tokens=80, avg_completion_tokens=40, model=model)
    print(f"  Estimated cost: ${est['estimated_cost_usd']}")

    # Build prompts
    prompt_pairs = build_description_prompts(entities)
    all_messages = [msgs for _, msgs in prompt_pairs]
    all_meta = [meta for meta, _ in prompt_pairs]

    # Generate
    print(f"  Calling OpenRouter ({concurrency} concurrent)...")
    responses = await batch_call(
        all_messages, model=model, temperature=0.3, max_tokens=60,
        concurrency=concurrency, progress_every=100,
    )

    # Build description index
    descriptions = {}
    failed = 0
    for meta, response in zip(all_meta, responses):
        if response:
            descriptions[meta["entity"]] = {
                "entity": meta["entity"],
                "entity_type": meta["entity_type"],
                "description": response.strip(),
            }
        else:
            failed += 1
            descriptions[meta["entity"]] = {
                "entity": meta["entity"],
                "entity_type": meta["entity_type"],
                "description": meta["entity"],  # fallback to name
            }

    # Save
    out_path = OUTPUT_DIR / f"{dataset_name}_descriptions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    # Also save as CSV for easy inspection
    csv_path = OUTPUT_DIR / f"{dataset_name}_descriptions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "entity_type", "description"])
        writer.writeheader()
        for d in descriptions.values():
            writer.writerow(d)

    print(f"\n  Results:")
    print(f"    Entities: {len(entities)}")
    print(f"    Descriptions generated: {len(descriptions) - failed}")
    print(f"    Failed: {failed}")
    print(f"    Saved to: {out_path}")
    print(f"    CSV: {csv_path}")

    # Show samples
    print(f"\n  Sample descriptions:")
    for d in list(descriptions.values())[:10]:
        print(f"    {d['entity']}: {d['description']}")

    return descriptions


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate entity descriptions for RC4 fix")
    parser.add_argument("--dataset", default="veterans_v2",
                        choices=["veterans_v2", "veterans_original", "twitter", "all"])
    parser.add_argument("--model", default="google/gemini-2.0-flash-001:floor")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = {
        "veterans_v2": DATA_DIR / "veterans_t2e_v2.csv",
        "veterans_original": DATA_DIR / "implicit_reference_veterans_dataset.csv",
        "twitter": DATA_DIR / "twitter_implicit_dataset.csv",
    }

    if args.dataset == "all":
        targets = datasets
    else:
        targets = {args.dataset: datasets[args.dataset]}

    if args.dry_run:
        for name, path in targets.items():
            entities = load_entities(path)
            est = estimate_cost(len(entities), avg_prompt_tokens=80, avg_completion_tokens=40, model=args.model)
            print(f"  {name}: {len(entities)} entities, est. cost: ${est['estimated_cost_usd']}")
        print(f"\n  DRY RUN. No API calls made.")
        return

    for name, path in targets.items():
        await generate_descriptions(name, path, args.model, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
