"""
Create IRC Benchmark v2.0 with all 4 dataset variants.

The benchmark includes:
  1. Veterans T2E (real interviews, text-to-entity rewriting) - REAL
  2. Veterans E2T (entity-driven LLM generation) - SYNTHETIC
  3. Twitter T2E (real tweets, Hosseini 2022 benchmark) - REAL
  4. Twitter E2T (entity-driven LLM generation) - SYNTHETIC

Each variant is preserved as a distinct subset with clear provenance.
Train/test split is stratified by (variant, entity_type).

Produces:
  data/benchmark_v2/
    irc_benchmark_v2_full.csv
    irc_benchmark_v2_train.csv
    irc_benchmark_v2_test.csv
    irc_benchmark_v2_metadata.json
    entity_index.csv
    variants/
      veterans_t2e.csv
      veterans_e2t.csv
      twitter_t2e.csv
      twitter_e2t.csv
"""
import csv
import json
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
GENERATED_DIR = Path(__file__).parent / "generated"
BENCH_DIR = DATA_DIR / "benchmark_v2"
BENCH_DIR.mkdir(exist_ok=True)
(BENCH_DIR / "variants").mkdir(exist_ok=True)
DESC_DIR = Path(__file__).parent / "entity_descriptions"

RANDOM_SEED = 42
TEST_RATIO = 0.20

# Coarse type mapping
TYPE_MAP = {
    "places": "Place", "Place": "Place",
    "people": "Person", "Person": "Person",
    "events": "Event", "Event": "Event",
    "organizations": "Organization", "Organization": "Organization",
    "professions": "Profession", "Profession": "Profession",
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
    "SocietalEvent": "Event", "SportsEvent": "Event",
    "SoccerTournament": "Event", "Festival": "Event",
    "ReligiousEvent": "Event", "Awards": "Event",
    "CellularTelephone": "Product", "MobilePhone": "Product",
    "Software": "Product", "Instrument": "Product",
    "MusicalInstrument": "Product",
}

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

FIELDNAMES = [
    "uid", "variant", "domain", "generation", "text", "origin",
    "entity", "entity_type", "entity_type_original", "source", "links",
]


def uid_hash(text: str, entity: str, variant: str) -> str:
    raw = f"{variant}|{text[:200]}|{entity}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def norm_ent(e: str) -> str:
    return e.strip().lower()


def should_exclude(entity: str) -> bool:
    return norm_ent(entity) in EXCLUDE_ENTITIES or len(entity.strip()) < 2


def load_veterans_t2e():
    """Veterans text-to-entity: real interviews with implicit rewrites."""
    path = DATA_DIR / "veterans_t2e_v2.csv"
    if not path.exists():
        path = DATA_DIR / "implicit_reference_veterans_dataset.csv"
    samples = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["entity"].strip()
            if should_exclude(entity):
                continue
            text = row["text"].strip()
            key = (text[:200].lower(), norm_ent(entity))
            if key in seen:
                continue
            seen.add(key)
            orig_type = row.get("entity_type_original", row.get("entity_type", "")).strip()
            samples.append({
                "variant": "veterans_t2e",
                "domain": "veterans",
                "generation": "real",
                "text": text,
                "origin": row.get("origin", "").strip(),
                "entity": entity,
                "entity_type": TYPE_MAP.get(row.get("entity_type", "").strip(), "Other"),
                "entity_type_original": orig_type,
                "source": row.get("source", "").strip(),
                "links": row.get("links", "").strip(),
            })
    return samples


def load_veterans_e2t():
    """Veterans entity-to-text: LLM-generated implicit narratives."""
    pattern = "e2t_veterans_*.csv"
    files = list(GENERATED_DIR.glob(pattern))
    if not files:
        print(f"  WARNING: No E2T veterans files found in {GENERATED_DIR}")
        return []
    samples = []
    seen = set()
    with open(files[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["entity"].strip()
            if should_exclude(entity):
                continue
            text = row["text"].strip()
            key = (text[:200].lower(), norm_ent(entity))
            if key in seen:
                continue
            seen.add(key)
            samples.append({
                "variant": "veterans_e2t",
                "domain": "veterans",
                "generation": "synthetic",
                "text": text,
                "origin": "",
                "entity": entity,
                "entity_type": TYPE_MAP.get(row.get("entity_type", "").strip(), "Other"),
                "entity_type_original": row.get("entity_type", "").strip(),
                "source": "generated",
                "links": "",
            })
    return samples


def load_twitter_t2e():
    """Twitter text-to-entity: real tweets from Hosseini 2022."""
    path = DATA_DIR / "twitter_implicit_dataset.csv"
    samples = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["entity"].strip()
            if should_exclude(entity):
                continue
            text = row["text"].strip()
            key = (text[:200].lower(), norm_ent(entity))
            if key in seen:
                continue
            seen.add(key)
            orig_type = row.get("entity_type", "").strip()
            samples.append({
                "variant": "twitter_t2e",
                "domain": "twitter",
                "generation": "real",
                "text": text,
                "origin": "",
                "entity": entity,
                "entity_type": TYPE_MAP.get(orig_type, "Other"),
                "entity_type_original": orig_type,
                "source": row.get("source", "").strip(),
                "links": row.get("links", "").strip(),
            })
    return samples


def load_twitter_e2t():
    """Twitter entity-to-text: LLM-generated implicit tweets."""
    pattern = "e2t_twitter_*.csv"
    files = list(GENERATED_DIR.glob(pattern))
    if not files:
        print(f"  WARNING: No E2T twitter files found in {GENERATED_DIR}")
        return []
    samples = []
    seen = set()
    with open(files[0], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["entity"].strip()
            if should_exclude(entity):
                continue
            text = row["text"].strip()
            key = (text[:200].lower(), norm_ent(entity))
            if key in seen:
                continue
            seen.add(key)
            samples.append({
                "variant": "twitter_e2t",
                "domain": "twitter",
                "generation": "synthetic",
                "text": text,
                "origin": "",
                "entity": entity,
                "entity_type": TYPE_MAP.get(row.get("entity_type", "").strip(), "Other"),
                "entity_type_original": row.get("entity_type", "").strip(),
                "source": "generated",
                "links": "",
            })
    return samples


def create_benchmark():
    print("Loading all 4 variants...")

    variants = {
        "veterans_t2e": load_veterans_t2e(),
        "veterans_e2t": load_veterans_e2t(),
        "twitter_t2e": load_twitter_t2e(),
        "twitter_e2t": load_twitter_e2t(),
    }

    for name, samples in variants.items():
        print(f"  {name}: {len(samples)} samples")

    # Assign UIDs
    for name, samples in variants.items():
        for s in samples:
            s["uid"] = uid_hash(s["text"], s["entity"], s["variant"])

    # Save per-variant CSVs
    for name, samples in variants.items():
        path = BENCH_DIR / "variants" / f"{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(samples)
        print(f"  Saved variant: {path.name}")

    # Combine all
    all_samples = []
    for samples in variants.values():
        all_samples.extend(samples)
    print(f"\n  Combined: {len(all_samples)} samples")

    # Stratified train/test split by (variant, entity_type)
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)

    groups = defaultdict(list)
    for s in all_samples:
        groups[(s["variant"], s["entity_type"])].append(s)

    train, test = [], []
    for key, group_samples in groups.items():
        n_test = max(1, int(len(group_samples) * TEST_RATIO))
        test.extend(group_samples[:n_test])
        train.extend(group_samples[n_test:])

    random.shuffle(train)
    random.shuffle(test)

    print(f"  Train: {len(train)}")
    print(f"  Test:  {len(test)}")

    # Save
    for name, data in [("full", all_samples), ("train", train), ("test", test)]:
        path = BENCH_DIR / f"irc_benchmark_v2_{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(data)

    # Entity index with descriptions
    entity_index = {}
    for s in all_samples:
        ent_key = norm_ent(s["entity"])
        if ent_key not in entity_index:
            entity_index[ent_key] = {
                "entity": s["entity"],
                "entity_type": s["entity_type"],
                "domains": set(),
                "variants": set(),
                "count": 0,
            }
        entity_index[ent_key]["domains"].add(s["domain"])
        entity_index[ent_key]["variants"].add(s["variant"])
        entity_index[ent_key]["count"] += 1

    # Load descriptions
    for desc_file in DESC_DIR.glob("*_descriptions.json"):
        descriptions = json.load(open(desc_file, encoding="utf-8"))
        for ent_key, info in entity_index.items():
            desc_entry = descriptions.get(info["entity"], descriptions.get(ent_key, {}))
            if desc_entry.get("description"):
                info["description"] = desc_entry["description"]

    ent_path = BENCH_DIR / "entity_index.csv"
    with open(ent_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entity", "entity_type", "domains", "variants", "count", "description"
        ])
        writer.writeheader()
        for info in sorted(entity_index.values(), key=lambda x: -x["count"]):
            writer.writerow({
                **info,
                "domains": "|".join(sorted(info["domains"])),
                "variants": "|".join(sorted(info["variants"])),
                "description": info.get("description", ""),
            })

    # Statistics
    variant_dist = Counter(s["variant"] for s in all_samples)
    type_dist = Counter(s["entity_type"] for s in all_samples)
    gen_dist = Counter(s["generation"] for s in all_samples)
    domain_dist = Counter(s["domain"] for s in all_samples)

    # Per-variant type distribution
    type_by_variant = {}
    for v_name in variants:
        type_by_variant[v_name] = dict(Counter(
            s["entity_type"] for s in all_samples if s["variant"] == v_name
        ).most_common())

    # Per-split variant distribution
    split_variant = {}
    for split_name, split_data in [("train", train), ("test", test)]:
        split_variant[split_name] = dict(Counter(
            s["variant"] for s in split_data
        ).most_common())

    metadata = {
        "name": "IRC Benchmark v2.0",
        "description": "Implicit Reminiscence Context recovery benchmark with 4 dataset variants (2 real + 2 synthetic)",
        "version": "2.0",
        "total_samples": len(all_samples),
        "train_samples": len(train),
        "test_samples": len(test),
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "unique_entities": len(entity_index),
        "variants": {
            "veterans_t2e": {
                "description": "Real veterans' interview transcripts with implicit entity rewrites",
                "source": "Library of Congress Veterans History Project",
                "generation": "real (text-to-entity rewriting)",
                "samples": variant_dist.get("veterans_t2e", 0),
            },
            "veterans_e2t": {
                "description": "LLM-generated implicit narratives from veteran entity lists",
                "source": "Generated by Gemini 2.0 Flash from veterans entity list",
                "generation": "synthetic (entity-to-text generation)",
                "samples": variant_dist.get("veterans_e2t", 0),
            },
            "twitter_t2e": {
                "description": "Real tweets with implicit entity references",
                "source": "Hosseini (2022), Toronto Metropolitan University",
                "generation": "real (annotated tweets)",
                "samples": variant_dist.get("twitter_t2e", 0),
            },
            "twitter_e2t": {
                "description": "LLM-generated implicit tweets from Twitter entity lists",
                "source": "Generated by Gemini 2.0 Flash from Hosseini entity list",
                "generation": "synthetic (entity-to-text generation)",
                "samples": variant_dist.get("twitter_e2t", 0),
            },
        },
        "by_generation": dict(gen_dist),
        "by_domain": dict(domain_dist),
        "entity_types": dict(type_dist.most_common()),
        "entity_types_by_variant": type_by_variant,
        "split_by_variant": split_variant,
        "schema": {
            "uid": "Deterministic hash ID (variant + text + entity)",
            "variant": "Dataset variant: veterans_t2e, veterans_e2t, twitter_t2e, twitter_e2t",
            "domain": "Source domain: veterans or twitter",
            "generation": "Data origin: real or synthetic",
            "text": "Implicit reference text (entity not named)",
            "origin": "Original text before rewriting (veterans_t2e only)",
            "entity": "Gold-standard entity name",
            "entity_type": "Coarse type: Person, Place, Event, Organization, Work, Product, Profession, Other",
            "entity_type_original": "Original fine-grained type from source dataset",
            "source": "Narrator name / tweet source / 'generated'",
            "links": "Source URL (where available)",
        },
    }

    meta_path = BENCH_DIR / "irc_benchmark_v2_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  IRC BENCHMARK v2.0")
    print(f"{'='*70}")
    print(f"  Total: {len(all_samples)} samples ({len(train)} train / {len(test)} test)")
    print(f"  Unique entities: {len(entity_index)}")
    print(f"\n  By variant:")
    print(f"  {'Variant':<20s} {'Domain':<10s} {'Gen':<12s} {'Samples':>8s} {'Entities':>10s}")
    print(f"  {'-'*60}")
    for v_name in ["veterans_t2e", "veterans_e2t", "twitter_t2e", "twitter_e2t"]:
        v_samples = [s for s in all_samples if s["variant"] == v_name]
        v_ents = len(set(norm_ent(s["entity"]) for s in v_samples))
        domain = "veterans" if "veterans" in v_name else "twitter"
        gen = "real" if "t2e" in v_name else "synthetic"
        print(f"  {v_name:<20s} {domain:<10s} {gen:<12s} {len(v_samples):>8d} {v_ents:>10d}")

    print(f"\n  By generation method:")
    for g, c in gen_dist.most_common():
        print(f"    {g}: {c} ({c/len(all_samples)*100:.1f}%)")

    print(f"\n  By entity type:")
    for t, c in type_dist.most_common():
        print(f"    {t}: {c} ({c/len(all_samples)*100:.1f}%)")

    print(f"\n  Train/test split by variant:")
    for split_name in ["train", "test"]:
        print(f"    {split_name}:")
        for v, c in split_variant[split_name].items():
            print(f"      {v}: {c}")

    print(f"\n  Saved to: {BENCH_DIR}")


if __name__ == "__main__":
    create_benchmark()
