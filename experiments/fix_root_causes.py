"""
Fix root causes of poor experiment results on veterans T2E data.

RC1: De-duplicate texts with multiple gold entities (keep most specific entity)
RC2: Filter common-word entities (keep only proper nouns / named entities)
RC3: Separate professions/organizations from core Person/Place/Event types

Produces: data/veterans_t2e_v2.csv (publication-quality dataset)
"""
import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "implicit_reference_veterans_dataset.csv"
OUTPUT_PATH = DATA_DIR / "veterans_t2e_v2.csv"
REPORT_PATH = DATA_DIR / "veterans_t2e_v2_report.json"

# ── RC2: Common words that are NOT real named entities ─────────────────────
# These will never be recoverable as implicit entities
COMMON_WORD_ENTITIES = {
    # Pronouns and generic people references
    "i", "me", "we", "he", "she", "they", "man", "men", "women", "woman",
    "people", "person", "friend", "friends", "buddy", "buddies",
    "boy", "boys", "girl", "girls", "kid", "kids",
    # Generic military terms (not proper nouns)
    "war", "the war", "a war", "wars",
    "military", "the military",
    "soldiers", "soldier", "troops", "troop",
    "veterans", "veteran",
    "officer", "officers",
    "enlisted men", "enlisted",
    "recruits", "recruit",
    "pilots", "pilot",
    "medic", "medics",
    "gunner", "gunners",
    # Generic ranks (not named entities unless tied to a person)
    "lieutenant", "captain", "sergeant", "corporal", "colonel",
    "major", "general", "admiral", "private", "commander",
    "second lieutenant", "first lieutenant", "staff sergeant",
    # Generic professions
    "military training", "combat medic", "infantry", "infantry man",
    "engineer", "clerk", "typist", "crane", "mechanic",
    "mba", "phd", "degree",
    # Generic family / relationships
    "family", "mother", "father", "wife", "husband", "brother", "sister",
    "son", "daughter", "parents", "children", "relatives",
    "future generations",
    # Generic concepts
    "home", "the home", "god", "the lord", "lord",
    "communists", "enemy", "allies",
    "draft", "draft law",
    "battles", "battle", "fighting",
    "hurricanes", "hurricane", "storms",
    "depression", "the depression",
    # Too short / ambiguous
    "us", "uk",
}

# Named entities that LOOK generic but are actually proper (keep these)
KEEP_DESPITE_COMMON = {
    "pearl harbor", "d-day", "vietnam", "korea", "japan", "germany",
    "france", "england", "italy", "russia", "china", "philippines",
    "okinawa", "iwo jima", "normandy", "berlin", "tokyo", "paris",
    "manhattan", "brooklyn", "new york", "new york city",
    "101st airborne", "82nd airborne",
    "red cross", "ford foundation", "united nations",
    "martin luther king", "eisenhower", "macarthur", "patton", "truman",
    "april fool's day", "christmas", "new year's day",
    "korean war", "vietnam war", "world war ii", "world war i",
    "tet offensive", "battle of the bulge",
    "gi bill",
}


def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def is_common_word_entity(entity: str) -> bool:
    """Check if entity is a common word, not a proper named entity."""
    norm = normalize(entity)

    # Check keeplist first
    if norm in KEEP_DESPITE_COMMON:
        return False
    for keep in KEEP_DESPITE_COMMON:
        if keep in norm or norm in keep:
            return False

    # Check against common words
    if norm in COMMON_WORD_ENTITIES:
        return True

    # Strip articles and check again
    stripped = re.sub(r'^(the|a|an|my|our|his|her|their)\s+', '', norm)
    if stripped in COMMON_WORD_ENTITIES:
        return True

    return False


def entity_specificity_score(entity: str, entity_type: str) -> float:
    """Score entity specificity for dedup ranking. Higher = more specific."""
    norm = normalize(entity)
    score = 0.0

    # Length bonus (longer = more specific)
    score += len(norm.split()) * 2.0

    # Named entity bonus (capitalized words in original)
    caps = sum(1 for w in entity.split() if w[0:1].isupper())
    score += caps * 3.0

    # Type bonus (places/events > people > professions/organizations)
    type_scores = {
        "places": 5.0, "events": 5.0,
        "people": 4.0,
        "organizations": 3.0,
        "professions": 1.0,
    }
    score += type_scores.get(entity_type, 2.0)

    # Penalize common words
    if is_common_word_entity(entity):
        score -= 10.0

    return score


def fix_dataset():
    """Apply all root cause fixes."""
    print("Loading dataset...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  Loaded {len(rows)} samples")

    # ── RC2: Filter common-word entities ──────────────────────────────────
    print("\n--- RC2: Filtering common-word entities ---")
    rc2_removed = []
    rc2_kept = []
    for r in rows:
        entity = r["entity"].strip()
        if is_common_word_entity(entity):
            rc2_removed.append({"entity": entity, "type": r["entity_type"]})
        else:
            rc2_kept.append(r)
    print(f"  Removed {len(rc2_removed)} common-word entities")
    print(f"  Kept {len(rc2_kept)} samples")

    # ── RC1: De-duplicate texts (keep most specific entity per text) ──────
    print("\n--- RC1: De-duplicating texts ---")
    text_groups = defaultdict(list)
    for r in rc2_kept:
        # Use first 200 chars of text as key (handles minor whitespace diffs)
        key = normalize(r["text"][:200])
        text_groups[key].append(r)

    rc1_deduped = []
    rc1_removed = []
    for key, group in text_groups.items():
        if len(group) == 1:
            rc1_deduped.append(group[0])
        else:
            # Pick the most specific entity
            scored = [(entity_specificity_score(r["entity"], r["entity_type"]), r) for r in group]
            scored.sort(key=lambda x: -x[0])
            best = scored[0][1]
            rc1_deduped.append(best)
            for _, r in scored[1:]:
                rc1_removed.append({
                    "entity": r["entity"],
                    "type": r["entity_type"],
                    "kept_instead": best["entity"],
                })

    print(f"  Duplicate text groups: {sum(1 for g in text_groups.values() if len(g) > 1)}")
    print(f"  Removed {len(rc1_removed)} duplicate-text entries")
    print(f"  Kept {len(rc1_deduped)} samples")

    # ── RC3: Remap entity types ───────────────────────────────────────────
    print("\n--- RC3: Remapping entity types ---")
    TYPE_MAP = {
        "places": "Place",
        "people": "Person",
        "events": "Event",
        "organizations": "Organization",
        "professions": "Profession",
    }
    for r in rc1_deduped:
        r["entity_type_original"] = r["entity_type"]
        r["entity_type"] = TYPE_MAP.get(r["entity_type"], r["entity_type"])

    # ── Check for name leakage one more time ──────────────────────────────
    print("\n--- Final leak check ---")
    final = []
    leak_removed = []
    for r in rc1_deduped:
        entity_lower = r["entity"].strip().lower()
        text_lower = r["text"].lower()
        if len(entity_lower) > 2 and entity_lower in text_lower:
            leak_removed.append({"entity": r["entity"], "type": r["entity_type"]})
        else:
            final.append(r)
    print(f"  Removed {len(leak_removed)} leaked entities")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n--- Saving {len(final)} samples ---")
    fieldnames = list(final[0].keys()) if final else []
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final)

    # Stats
    type_dist = Counter(r["entity_type"] for r in final)
    unique_ents = len(set(normalize(r["entity"]) for r in final))
    narrators = len(set(r["source"] for r in final))

    # Core types only (for paper comparison)
    core = [r for r in final if r["entity_type"] in ("Place", "Person", "Event")]

    report = {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "original_count": len(rows),
        "after_rc2_filter": len(rc2_kept),
        "after_rc1_dedup": len(rc1_deduped),
        "after_leak_check": len(final),
        "rc2_removed_count": len(rc2_removed),
        "rc1_removed_count": len(rc1_removed),
        "leak_removed_count": len(leak_removed),
        "total_removed": len(rows) - len(final),
        "removal_pct": round((len(rows) - len(final)) / len(rows) * 100, 1),
        "unique_entities": unique_ents,
        "narrators": narrators,
        "type_distribution": dict(type_dist.most_common()),
        "core_types_only": len(core),
        "rc2_examples": rc2_removed[:15],
        "rc1_examples": rc1_removed[:15],
        "leak_examples": leak_removed[:5],
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VETERANS T2E V2 - ROOT CAUSE FIXES APPLIED")
    print(f"{'='*60}")
    print(f"  Original:           {len(rows)} samples")
    print(f"  After RC2 (filter): {len(rc2_kept)} (-{len(rc2_removed)} common words)")
    print(f"  After RC1 (dedup):  {len(rc1_deduped)} (-{len(rc1_removed)} duplicate texts)")
    print(f"  After leak check:   {len(final)} (-{len(leak_removed)} leaks)")
    print(f"  Total removed:      {len(rows) - len(final)} ({report['removal_pct']}%)")
    print()
    print(f"  Entity types (all):")
    for t, c in type_dist.most_common():
        print(f"    {t}: {c} ({c/len(final)*100:.1f}%)")
    print(f"  Core types (Place/Person/Event): {len(core)} samples")
    print(f"  Unique entities: {unique_ents}")
    print(f"  Narrators: {narrators}")
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Report:   {REPORT_PATH}")


if __name__ == "__main__":
    fix_dataset()
