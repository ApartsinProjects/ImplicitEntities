"""
Create a standardized IRC benchmark dataset for reproducible evaluation.

The benchmark combines cleaned veterans and Twitter data into a single,
deduplicated, well-documented dataset with consistent schema and splits.

Produces:
  data/benchmark/
    irc_benchmark_full.csv        (all samples)
    irc_benchmark_train.csv       (80% for development)
    irc_benchmark_test.csv        (20% held out for evaluation)
    irc_benchmark_metadata.json   (dataset statistics and documentation)
    entity_index.csv              (all unique entities with descriptions)
"""
import csv
import json
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
BENCH_DIR = DATA_DIR / "benchmark"
BENCH_DIR.mkdir(exist_ok=True)
DESC_DIR = Path(__file__).parent / "entity_descriptions"

RANDOM_SEED = 42
TEST_RATIO = 0.20

# Standardized entity type mapping
TYPE_MAP = {
    # Veterans types
    "places": "Place", "Place": "Place",
    "people": "Person", "Person": "Person",
    "events": "Event", "Event": "Event",
    "organizations": "Organization", "Organization": "Organization",
    "professions": "Profession", "Profession": "Profession",
    # Twitter types (map fine-grained to coarse)
    "Movie": "Work", "Book": "Work", "Film": "Work", "WrittenWork": "Work",
    "Country": "Place", "City": "Place", "PopulatedPlace": "Place",
    "HistoricalPlace": "Place", "ArchitecturalStructure": "Place",
    "ReligiousBuilding": "Place", "Building": "Place", "Skyscraper": "Place",
    "Actor": "Person", "Writer": "Person", "Athlete": "Person",
    "Politician": "Person", "Businessperson": "Person", "Celebrity": "Person",
    "Royalty": "Person", "Scientist": "Person", "Director": "Person",
    "MusicalArtist": "Person", "Rapper": "Person", "Model": "Person",
    "Comedian": "Person", "Cricketer": "Person", "MusicalPerformer": "Person",
    "CEO": "Person", "President": "Person", "PrimeMinister": "Person",
    "VicePresident": "Person", "Leader": "Person", "Founder": "Person",
    "Executive": "Person", "Co-founder": "Person", "Entrepreneur": "Person",
    "OfficeHolder": "Person", "Author": "Person",
    "Company": "Organization", "Organisation": "Organization",
    "PoliticalParty": "Organization", "Charity": "Organization",
    "Public_company": "Organization",
    "Team": "Organization", "CricketTeam": "Organization",
    "HockeyTeam": "Organization", "BasketballTeam": "Organization",
    "SoccerClub": "Organization", "BaseballTeam": "Organization",
    "Band": "Organization", "MusicGroup": "Organization", "Group": "Organization",
    "Event": "Event", "SocietalEvent": "Event", "SportsEvent": "Event",
    "SoccerTournament": "Event", "Festival": "Event",
    "ReligiousEvent": "Event", "Awards": "Event",
    "CellularTelephone": "Product", "MobilePhone": "Product",
    "Software": "Product", "Instrument": "Product",
    "MusicalInstrument": "Product",
}

# Generic entities to exclude (not recoverable as implicit references)
EXCLUDE_ENTITIES = {
    "i", "me", "we", "he", "she", "they", "man", "men", "women", "woman",
    "people", "person", "friend", "friends", "buddy", "buddies",
    "war", "the war", "military", "the military",
    "soldiers", "soldier", "troops", "veterans", "veteran",
    "officer", "officers", "enlisted men",
    "lieutenant", "captain", "sergeant", "corporal", "colonel",
    "major", "general", "admiral", "private", "commander",
    "second lieutenant", "first lieutenant", "staff sergeant",
    "military training", "combat medic", "infantry",
    "mba", "phd", "degree", "family", "mother", "father",
    "wife", "husband", "brother", "sister", "home", "god", "the lord",
    "future generations", "communists", "enemy", "allies",
    "draft", "draft law", "battles", "battle", "fighting",
    "hurricanes", "depression", "the depression",
    "winter", "spring", "summer", "fall",
}


def uid_hash(text: str, entity: str) -> str:
    """Generate a deterministic UID from text + entity."""
    raw = f"{text[:200]}|{entity}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def normalize_entity(e: str) -> str:
    return e.strip().lower()


def load_veterans_v2():
    """Load cleaned veterans T2E dataset."""
    path = DATA_DIR / "veterans_t2e_v2.csv"
    if not path.exists():
        path = DATA_DIR / "implicit_reference_veterans_dataset.csv"
    samples = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["entity"].strip()
            if normalize_entity(entity) in EXCLUDE_ENTITIES:
                continue
            if len(entity) < 2:
                continue
            samples.append({
                "domain": "veterans",
                "text": row["text"].strip(),
                "origin": row.get("origin", "").strip(),
                "entity": entity,
                "entity_type_original": row.get("entity_type_original", row.get("entity_type", "")).strip(),
                "entity_type": TYPE_MAP.get(row.get("entity_type", "").strip(), "Other"),
                "source": row.get("source", "").strip(),
                "links": row.get("links", "").strip(),
            })
    return samples


def load_twitter():
    """Load full Twitter implicit dataset."""
    path = DATA_DIR / "twitter_implicit_dataset.csv"
    samples = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["entity"].strip()
            if normalize_entity(entity) in EXCLUDE_ENTITIES:
                continue
            if len(entity) < 2:
                continue
            orig_type = row.get("entity_type", "").strip()
            samples.append({
                "domain": "twitter",
                "text": row["text"].strip(),
                "origin": "",
                "entity": entity,
                "entity_type_original": orig_type,
                "entity_type": TYPE_MAP.get(orig_type, "Other"),
                "source": row.get("source", "").strip(),
                "links": row.get("links", "").strip(),
            })
    return samples


def deduplicate(samples):
    """Remove duplicate (text, entity) pairs."""
    seen = set()
    deduped = []
    for s in samples:
        key = (s["text"][:200].lower(), normalize_entity(s["entity"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def create_benchmark():
    print("Loading datasets...")
    veterans = load_veterans_v2()
    twitter = load_twitter()
    print(f"  Veterans: {len(veterans)} samples")
    print(f"  Twitter:  {len(twitter)} samples")

    # Combine
    all_samples = veterans + twitter
    print(f"  Combined: {len(all_samples)} samples")

    # Deduplicate
    all_samples = deduplicate(all_samples)
    print(f"  After dedup: {len(all_samples)} samples")

    # Assign UIDs
    for s in all_samples:
        s["uid"] = uid_hash(s["text"], s["entity"])

    # Verify no duplicate UIDs
    uid_counts = Counter(s["uid"] for s in all_samples)
    dupes = {uid: c for uid, c in uid_counts.items() if c > 1}
    if dupes:
        print(f"  WARNING: {len(dupes)} duplicate UIDs, adding suffix...")
        uid_seen = Counter()
        for s in all_samples:
            uid_seen[s["uid"]] += 1
            if uid_counts[s["uid"]] > 1:
                s["uid"] = f"{s['uid']}_{uid_seen[s['uid']]}"

    # Split: stratified by domain and entity_type
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    # Group by (domain, entity_type) for stratified split
    groups = defaultdict(list)
    for s in all_samples:
        groups[(s["domain"], s["entity_type"])].append(s)

    train, test = [], []
    for key, group_samples in groups.items():
        n_test = max(1, int(len(group_samples) * TEST_RATIO))
        test.extend(group_samples[:n_test])
        train.extend(group_samples[n_test:])

    random.shuffle(train)
    random.shuffle(test)

    print(f"\n  Train: {len(train)} samples")
    print(f"  Test:  {len(test)} samples")

    # Build entity index
    entity_index = {}
    for s in all_samples:
        ent_key = normalize_entity(s["entity"])
        if ent_key not in entity_index:
            entity_index[ent_key] = {
                "entity": s["entity"],
                "entity_type": s["entity_type"],
                "domain": s["domain"],
                "count": 0,
            }
        entity_index[ent_key]["count"] += 1

    # Load descriptions if available
    desc_path = DESC_DIR / "veterans_v2_descriptions.json"
    if desc_path.exists():
        descriptions = json.load(open(desc_path, encoding="utf-8"))
        for ent_key, info in entity_index.items():
            desc_entry = descriptions.get(info["entity"], descriptions.get(ent_key, {}))
            info["description"] = desc_entry.get("description", "")

    # Save
    fieldnames = ["uid", "domain", "text", "origin", "entity", "entity_type",
                  "entity_type_original", "source", "links"]

    for name, data in [("full", all_samples), ("train", train), ("test", test)]:
        path = BENCH_DIR / f"irc_benchmark_{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {path.name} ({len(data)} rows)")

    # Entity index
    ent_path = BENCH_DIR / "entity_index.csv"
    with open(ent_path, "w", encoding="utf-8", newline="") as f:
        ent_fields = ["entity", "entity_type", "domain", "count", "description"]
        writer = csv.DictWriter(f, fieldnames=ent_fields, extrasaction="ignore")
        writer.writeheader()
        for info in sorted(entity_index.values(), key=lambda x: -x["count"]):
            writer.writerow(info)

    # Statistics
    type_dist = Counter(s["entity_type"] for s in all_samples)
    domain_dist = Counter(s["domain"] for s in all_samples)
    type_by_domain = defaultdict(Counter)
    for s in all_samples:
        type_by_domain[s["domain"]][s["entity_type"]] += 1

    metadata = {
        "name": "IRC Benchmark v1.0",
        "description": "Implicit Reminiscence Context recovery benchmark for evaluating implicit entity recognition",
        "version": "1.0",
        "total_samples": len(all_samples),
        "train_samples": len(train),
        "test_samples": len(test),
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "unique_entities": len(entity_index),
        "domains": dict(domain_dist),
        "entity_types": dict(type_dist.most_common()),
        "entity_types_by_domain": {d: dict(c.most_common()) for d, c in type_by_domain.items()},
        "sources": {
            "veterans": "Library of Congress Veterans History Project, cleaned and deduplicated",
            "twitter": "Hosseini (2022) implicit entity benchmark, full 856-tweet dataset",
        },
        "excluded": "Generic entities (pronouns, common military terms, family roles) filtered out",
        "schema": {
            "uid": "Deterministic hash ID",
            "domain": "veterans or twitter",
            "text": "Implicit reference text (entity not named)",
            "origin": "Original text before rewriting (veterans only)",
            "entity": "Gold-standard entity name",
            "entity_type": "Coarse type: Person, Place, Event, Organization, Work, Product, Profession, Other",
            "entity_type_original": "Original fine-grained type from source dataset",
            "source": "Narrator name (veterans) or tweet source",
            "links": "Source URL",
        },
    }

    meta_path = BENCH_DIR / "irc_benchmark_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  IRC BENCHMARK v1.0")
    print(f"{'='*60}")
    print(f"  Total: {len(all_samples)} samples ({len(train)} train / {len(test)} test)")
    print(f"  Unique entities: {len(entity_index)}")
    print(f"\n  By domain:")
    for d, c in domain_dist.most_common():
        print(f"    {d}: {c}")
    print(f"\n  By entity type:")
    for t, c in type_dist.most_common():
        print(f"    {t}: {c} ({c/len(all_samples)*100:.1f}%)")
    print(f"\n  Saved to: {BENCH_DIR}")


if __name__ == "__main__":
    create_benchmark()
