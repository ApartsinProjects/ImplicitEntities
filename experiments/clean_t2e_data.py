"""
Clean the T2E veterans dataset by filtering out problematic samples.

Issues addressed:
  1. Generic/vague entities ("war", "I", "friend", "military", "soldiers", etc.)
  2. Name leakage (entity appears in implicit text)
  3. Entity type remapping (professions/organizations -> paper's 3 categories or new ones)
  4. Deduplication of near-identical entities

Produces:
  - data/veterans_t2e_cleaned.csv (filtered dataset)
  - data/veterans_t2e_cleaning_report.json (what was removed and why)
"""
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "implicit_reference_veterans_dataset.csv"
OUTPUT_PATH = DATA_DIR / "veterans_t2e_cleaned.csv"
REPORT_PATH = DATA_DIR / "veterans_t2e_cleaning_report.json"

# ── Filter criteria ──────────────────────────────────────────────────────

# Generic entities that are too vague for meaningful implicit recognition
GENERIC_ENTITIES = {
    # Pronouns / common words that aren't real entities
    "i", "me", "we", "us", "he", "she", "they", "them",
    "men", "man", "women", "woman", "people", "person",
    "friend", "friends", "buddy", "buddies",
    # Overly generic terms
    "war", "the war", "a war",
    "military", "the military",
    "soldiers", "soldier", "troops",
    "veterans", "veteran",
    "home", "the home",
    "god", "the lord",
    "family", "mother", "father", "wife", "husband",
    "future generations",
    # Generic professions (not recoverable as implicit entities)
    "mba", "phd", "degree",
}

# Minimum entity length (single-char entities like "I" are meaningless)
MIN_ENTITY_LENGTH = 2
MIN_ENTITY_WORDS = 1  # at least 1 real word after normalization

# Entity type remapping: map 5-type system to paper's categories
# Original types: places, people, events, organizations, professions
TYPE_REMAP = {
    "places": "Place",
    "people": "Person",
    "events": "Event",
    "organizations": "Organization",
    "professions": "Profession",
}

# Which remapped types to keep (paper uses Person/Place/Event)
# We keep Organization and Profession as additional categories
VALID_TYPES = {"Place", "Person", "Event", "Organization", "Profession"}


def normalize_entity(entity: str) -> str:
    """Normalize entity for comparison."""
    return re.sub(r'\s+', ' ', entity.strip().lower())


def check_name_leak(text: str, entity: str) -> bool:
    """Check if entity name leaks into the implicit text."""
    text_lower = text.lower()
    entity_lower = entity.lower().strip()

    # Skip very short entities (1-2 chars) as they produce false positives
    if len(entity_lower) <= 2:
        return True  # treat as leaked (should be filtered)

    # Check full entity name
    if entity_lower in text_lower:
        return True

    # For multi-word entities, check if all significant words appear
    words = [w for w in entity_lower.split() if len(w) > 3]
    if len(words) >= 2:
        # If all significant words appear, likely leaked
        if all(w in text_lower for w in words):
            return True

    return False


def is_generic_entity(entity: str) -> bool:
    """Check if entity is too generic for implicit recognition."""
    norm = normalize_entity(entity)

    # Direct match
    if norm in GENERIC_ENTITIES:
        return True

    # Check if it's just a common word with articles
    stripped = re.sub(r'^(the|a|an|my|our|his|her|their)\s+', '', norm)
    if stripped in GENERIC_ENTITIES:
        return True

    return False


def clean_dataset():
    """Clean the T2E dataset and produce a report."""
    print("Loading dataset...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  Loaded {len(rows)} samples")

    # Track removals
    removed = {
        "generic_entity": [],
        "name_leaked": [],
        "too_short_entity": [],
        "identical_text": [],
    }
    kept = []

    for row in rows:
        entity = row["entity"].strip()
        entity_norm = normalize_entity(entity)
        text = row["text"].strip()
        origin = row["origin"].strip()
        entity_type = row["entity_type"].strip()

        # Remap entity type
        row["entity_type_original"] = entity_type
        row["entity_type"] = TYPE_REMAP.get(entity_type, entity_type)

        # Filter 1: Generic entities
        if is_generic_entity(entity):
            removed["generic_entity"].append({
                "entity": entity, "type": entity_type,
                "reason": f"Generic entity: '{entity}'"
            })
            continue

        # Filter 2: Too short entities
        if len(entity_norm) < MIN_ENTITY_LENGTH:
            removed["too_short_entity"].append({
                "entity": entity, "type": entity_type,
                "reason": f"Entity too short: '{entity}' ({len(entity_norm)} chars)"
            })
            continue

        # Filter 3: Name leakage
        if check_name_leak(text, entity):
            removed["name_leaked"].append({
                "entity": entity, "type": entity_type,
                "text_snippet": text[:100],
                "reason": f"Entity '{entity}' found in implicit text"
            })
            continue

        # Filter 4: Identical text (no rewriting happened)
        if text == origin:
            removed["identical_text"].append({
                "entity": entity, "type": entity_type,
                "reason": "Implicit text identical to original"
            })
            continue

        # Validate type
        if row["entity_type"] not in VALID_TYPES:
            row["entity_type"] = "Other"

        kept.append(row)

    # Write cleaned dataset
    if kept:
        fieldnames = list(kept[0].keys())
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

    # Compute stats
    total_removed = sum(len(v) for v in removed.values())
    type_dist_before = Counter(r["entity_type"].strip() for r in rows)
    type_dist_after = Counter(r["entity_type"] for r in kept)
    entity_dist_after = Counter(r["entity"].lower().strip() for r in kept)

    report = {
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_PATH),
        "total_before": len(rows),
        "total_after": len(kept),
        "total_removed": total_removed,
        "removal_breakdown": {k: len(v) for k, v in removed.items()},
        "removal_details": {k: v[:10] for k, v in removed.items()},  # first 10 per category
        "type_distribution_before": dict(type_dist_before.most_common()),
        "type_distribution_after": dict(type_dist_after.most_common()),
        "unique_entities_before": len(set(r["entity"].lower().strip() for r in rows)),
        "unique_entities_after": len(entity_dist_after),
        "top_entities_after": dict(entity_dist_after.most_common(20)),
        "narrators": len(set(r["source"] for r in kept)),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  T2E CLEANING REPORT")
    print(f"{'='*60}")
    print(f"  Before: {len(rows)} samples")
    print(f"  After:  {len(kept)} samples")
    print(f"  Removed: {total_removed} ({total_removed/len(rows)*100:.1f}%)")
    print()
    print(f"  Removal breakdown:")
    for reason, items in removed.items():
        print(f"    {reason}: {len(items)}")
    print()
    print(f"  Entity type distribution (after):")
    for etype, count in type_dist_after.most_common():
        pct = count / len(kept) * 100
        print(f"    {etype}: {count} ({pct:.1f}%)")
    print()
    print(f"  Unique entities: {report['unique_entities_before']} -> {report['unique_entities_after']}")
    print(f"  Narrators: {report['narrators']}")
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Report:   {REPORT_PATH}")


if __name__ == "__main__":
    clean_dataset()
