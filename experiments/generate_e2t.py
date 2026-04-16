"""
Generate Entity-to-Text (E2T) implicit reference datasets.
Takes entity lists from veterans and Twitter datasets, prompts an LLM to generate
implicit narratives for each entity.

Usage:
    python generate_e2t.py --domain veterans --model google/gemini-2.0-flash-001
    python generate_e2t.py --domain twitter --model google/gemini-2.0-flash-001
    python generate_e2t.py --domain both --model google/gemini-2.0-flash-001
"""
import asyncio
import argparse
import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)

# ── Prompt Templates ────────────────────────────────────────────────────────

VETERAN_SYSTEM = """You are helping create a dataset for implicit entity recognition research.
Given an entity name and its type, write a short first-person reminiscence narrative (2-4 sentences)
that implicitly references the entity WITHOUT ever naming it directly.

Rules:
- Do NOT use the entity's name or any common aliases
- Use descriptive cues: roles, attributes, geography, timeframes, cultural markers
- Write in the voice of an elderly American veteran recalling the past
- Keep it natural and emotionally grounded
- Output ONLY the narrative text, nothing else"""

TWEET_SYSTEM = """You are helping create a dataset for implicit entity recognition research.
Given an entity name and its type, write a short tweet-style post (1-2 sentences, under 280 chars)
that implicitly references the entity WITHOUT ever naming it directly.

Rules:
- Do NOT use the entity's name or any common aliases
- Use indirect descriptions, cultural references, or context clues
- Write in casual social media style
- Output ONLY the tweet text, nothing else"""


def load_veterans_entities() -> list[dict]:
    """Extract unique entities from the veterans dataset."""
    path = DATA_DIR / "implicit_reference_veterans_dataset.csv"
    entities = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ent = row["entity"].strip()
            etype = row["entity_type"].strip()
            if ent and ent not in entities:
                entities[ent] = {"entity": ent, "entity_type": etype}
    return list(entities.values())


def load_twitter_entities() -> list[dict]:
    """Extract unique entities from the full Twitter implicit dataset."""
    path = DATA_DIR / "twitter_implicit_dataset.csv"
    entities = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ent = row["entity"].strip()
            etype = row.get("entity_type", "").strip()
            if ent and ent not in entities:
                entities[ent] = {"entity": ent, "entity_type": etype}
    return list(entities.values())


def build_prompts(entities: list[dict], domain: str, n_variants: int = 3) -> list[tuple[dict, list[dict]]]:
    """
    Build prompt messages for each entity.
    Returns list of (entity_info, messages) tuples.
    """
    system_msg = VETERAN_SYSTEM if domain == "veterans" else TWEET_SYSTEM
    prompts = []

    for ent_info in entities:
        entity = ent_info["entity"]
        etype = ent_info["entity_type"]

        for variant_idx in range(n_variants):
            # Vary the prompt slightly for diversity
            if variant_idx == 0:
                user_msg = f"Entity: {entity}\nType: {etype}\n\nWrite an implicit reference."
            elif variant_idx == 1:
                user_msg = f"Entity: {entity}\nType: {etype}\n\nWrite a different implicit reference, using emotional or sensory details."
            else:
                user_msg = f"Entity: {entity}\nType: {etype}\n\nWrite another implicit reference, focusing on historical or cultural context."

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            prompts.append((
                {**ent_info, "variant": variant_idx, "domain": domain},
                messages,
            ))

    return prompts


def validate_no_leak(text: str, entity: str) -> bool:
    """Check that entity name doesn't appear in the generated text."""
    text_lower = text.lower()
    entity_lower = entity.lower()

    # Check full name
    if entity_lower in text_lower:
        return False

    # Check individual words (for multi-word entities, skip short words)
    words = entity_lower.split()
    if len(words) > 1:
        for w in words:
            if len(w) > 3 and w in text_lower:
                return False

    return True


async def generate_e2t(domain: str, model: str, n_variants: int = 3, concurrency: int = 10):
    """Generate E2T dataset for a given domain."""
    print(f"\n{'='*60}")
    print(f"  Generating E2T: domain={domain}, model={model}")
    print(f"  Variants per entity: {n_variants}")
    print(f"{'='*60}")

    # Load entities
    if domain == "veterans":
        entities = load_veterans_entities()
    else:
        entities = load_twitter_entities()

    print(f"  Loaded {len(entities)} unique entities")

    # Build prompts
    prompt_pairs = build_prompts(entities, domain, n_variants)
    print(f"  Built {len(prompt_pairs)} prompts ({len(entities)} entities x {n_variants} variants)")

    # Extract just the messages for the batch call
    all_messages = [msgs for _, msgs in prompt_pairs]
    all_meta = [meta for meta, _ in prompt_pairs]

    # Smoke test: verify API + generation quality with 3 samples
    max_tokens = 200 if domain == "veterans" else 100
    print(f"  [SMOKE TEST] Testing 3 samples...")
    test_responses = await batch_call(
        all_messages[:3], model=model, temperature=0.7, max_tokens=max_tokens,
        concurrency=3, progress_every=999,
    )
    smoke_pass = 0
    for i, resp in enumerate(test_responses):
        entity = all_meta[i]["entity"]
        ok = resp is not None and len(resp) > 10
        leak = ok and not validate_no_leak(resp, entity)
        status = "LEAK" if leak else ("OK" if ok else "FAIL")
        print(f"    [{status}] Entity: \"{entity}\" -> \"{(resp or '')[:80]}\"")
        if ok and not leak:
            smoke_pass += 1
    if smoke_pass == 0:
        print(f"  [SMOKE TEST] FAIL: 0/3 passed. Check model and prompts.")
        return []
    print(f"  [SMOKE TEST] PASS: {smoke_pass}/3\n")

    # Full batch call
    print(f"  Calling OpenRouter ({concurrency} concurrent requests)...")
    responses = await batch_call(
        all_messages,
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
        concurrency=concurrency,
        progress_every=50,
    )

    # Process results
    results = []
    leaked = 0
    failed = 0

    for meta, response in zip(all_meta, responses):
        if response is None:
            failed += 1
            continue

        # Validate no name leak
        if not validate_no_leak(response, meta["entity"]):
            leaked += 1
            continue

        results.append({
            "uid": f"e2t_{domain}_{len(results):05d}",
            "source": "generated",
            "SQN": meta["variant"],
            "text": response,
            "origin": "",
            "entity": meta["entity"],
            "entity_type": meta["entity_type"],
            "probability_entity_is_related_to_text": "Generated",
            "links": "",
            "model": model,
            "domain": domain,
        })

    # Save
    out_path = OUT_DIR / f"e2t_{domain}_{model.replace('/', '_')}.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n  Results:")
    print(f"    Total prompts:  {len(prompt_pairs)}")
    print(f"    Successful:     {len(results)}")
    print(f"    Name leaked:    {leaked}")
    print(f"    API failures:   {failed}")
    print(f"    Saved to:       {out_path}")

    # Also save as JSON for easy loading
    json_path = OUT_DIR / f"e2t_{domain}_{model.replace('/', '_')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"    JSON backup:    {json_path}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Generate E2T implicit reference datasets")
    parser.add_argument("--domain", choices=["veterans", "twitter", "both"], default="both",
                        help="Which domain to generate for")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001",
                        help="OpenRouter model to use (default: gemini-2.0-flash-001, very cheap)")
    parser.add_argument("--variants", type=int, default=3,
                        help="Number of implicit variants per entity")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API requests")
    args = parser.parse_args()

    domains = ["veterans", "twitter"] if args.domain == "both" else [args.domain]

    all_results = {}
    for domain in domains:
        results = await generate_e2t(domain, args.model, args.variants, args.concurrency)
        all_results[domain] = results

    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    for domain, results in all_results.items():
        print(f"  {domain}: {len(results)} samples")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
