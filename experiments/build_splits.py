"""
Build train/dev/test splits for IRC-Bench v5 experiments.

Entity-level split: entities in test are NEVER seen during training.
All outputs are saved with versioning for reproducibility.
"""

import csv
import json
import os
import random
import hashlib
from pathlib import Path
from collections import Counter

random.seed(42)  # Reproducible

BENCH_DIR = Path(__file__).parent.parent / "data" / "pipeline" / "transcripts"
V5_DIR = Path(__file__).parent
DATA_DIR = V5_DIR.parent / "data" / "benchmark"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_KB_PATH = Path(__file__).parent.parent / "data" / "benchmark" / "entity_kb.json"


def collect_all_samples():
    """Collect all implicit samples from IRC-Bench."""
    samples = []

    for root, dirs, files in os.walk(BENCH_DIR):
        if "_excluded" in root:
            continue
        for f in sorted(files):
            if not f.endswith(".implicit.json"):
                continue

            path = Path(root) / f
            with open(path, encoding="utf-8") as fh:
                items = json.load(fh)

            yaml_rel = str(path.relative_to(BENCH_DIR)).replace(".implicit.json", ".yaml")
            entities_rel = str(path.relative_to(BENCH_DIR)).replace(".implicit.json", ".entities.json")

            # Load entity details for Wikidata info
            entities_path = path.with_suffix("").with_suffix(".entities.json")
            entity_details = {}
            if entities_path.exists():
                with open(entities_path, encoding="utf-8") as ef:
                    for e in json.load(ef):
                        entity_details[e.get("entity", "")] = e

            for item in items:
                entity_name = item.get("entity", "")
                details = entity_details.get(entity_name, {})

                # Generate stable UID from entity + transcript
                uid_seed = f"{entity_name}|{yaml_rel}|{item.get('explicit_text', '')[:50]}"
                uid = hashlib.md5(uid_seed.encode()).hexdigest()[:12]

                samples.append({
                    "uid": uid,
                    "entity": entity_name,
                    "entity_type": details.get("type", ""),
                    "entity_qid": details.get("wikidata_qid", ""),
                    "entity_aliases": "|".join(details.get("aliases", [])),
                    "entity_description": details.get("description", ""),
                    "entity_description_wiki": details.get("description_wiki", ""),
                    "entity_wikipedia_url": details.get("wikipedia_url", ""),
                    "explicit_text": item.get("explicit_text", ""),
                    "implicit_text": item.get("implicit_text", ""),
                    "cues": "|".join(item.get("cues", [])),
                    "transcript_ref": yaml_rel,
                    "collection": yaml_rel.split("/")[0] if "/" in yaml_rel else "unknown",
                })

    return samples


def split_by_entity(samples, train_ratio=0.7, dev_ratio=0.1, test_ratio=0.2):
    """Split samples by entity (no entity overlap between splits)."""
    # Get unique entities
    entities = list(set(s["entity"] for s in samples))
    random.shuffle(entities)

    n = len(entities)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))

    train_entities = set(entities[:train_end])
    dev_entities = set(entities[train_end:dev_end])
    test_entities = set(entities[dev_end:])

    # Assign partition
    for s in samples:
        if s["entity"] in train_entities:
            s["partition"] = "train"
        elif s["entity"] in dev_entities:
            s["partition"] = "dev"
        else:
            s["partition"] = "test"

    return samples, train_entities, dev_entities, test_entities


def build_entity_kb(samples):
    """Build entity knowledge base from all samples."""
    kb = {}
    for s in samples:
        ent = s["entity"]
        if ent not in kb:
            kb[ent] = {
                "entity": ent,
                "type": s["entity_type"],
                "qid": s["entity_qid"],
                "aliases": s["entity_aliases"].split("|") if s["entity_aliases"] else [],
                "description": s["entity_description"],
                "description_wiki": s["entity_description_wiki"],
                "wikipedia_url": s["entity_wikipedia_url"],
                "sample_count": 0,
            }
        kb[ent]["sample_count"] += 1
    return kb


def save_splits(samples, train_entities, dev_entities, test_entities):
    """Save all data files."""

    fieldnames = [
        "uid", "partition", "entity", "entity_type", "entity_qid",
        "entity_aliases", "entity_description", "entity_description_wiki",
        "entity_wikipedia_url", "explicit_text", "implicit_text",
        "cues", "transcript_ref", "collection",
    ]

    # Full dataset
    full_path = DATA_DIR / "irc_bench_v5.csv"
    with open(full_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    # Per-partition files
    for partition in ["train", "dev", "test"]:
        part_samples = [s for s in samples if s["partition"] == partition]

        csv_path = DATA_DIR / f"irc_bench_v5_{partition}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(part_samples)

        json_path = DATA_DIR / f"irc_bench_v5_{partition}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(part_samples, f, indent=2, ensure_ascii=False)

    # Entity KB
    kb = build_entity_kb(samples)
    kb_path = DATA_DIR / "entity_kb.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    # Entity lists per partition
    for name, ents in [("train", train_entities), ("dev", dev_entities), ("test", test_entities)]:
        list_path = DATA_DIR / f"entity_list_{name}.txt"
        list_path.write_text("\n".join(sorted(ents)), encoding="utf-8")

    # Split metadata
    meta = {
        "version": "v5",
        "seed": 42,
        "split_ratios": {"train": 0.7, "dev": 0.1, "test": 0.2},
        "total_samples": len(samples),
        "total_entities": len(kb),
        "train": {"samples": sum(1 for s in samples if s["partition"] == "train"),
                  "entities": len(train_entities)},
        "dev": {"samples": sum(1 for s in samples if s["partition"] == "dev"),
                "entities": len(dev_entities)},
        "test": {"samples": sum(1 for s in samples if s["partition"] == "test"),
                 "entities": len(test_entities)},
        "entity_overlap": {
            "train_dev": len(train_entities & dev_entities),
            "train_test": len(train_entities & test_entities),
            "dev_test": len(dev_entities & test_entities),
        },
        "by_type": dict(Counter(s["entity_type"] for s in samples).most_common()),
        "by_collection": dict(Counter(s["collection"] for s in samples).most_common()),
    }

    meta_path = DATA_DIR / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    print("Collecting samples from IRC-Bench .implicit.json files...")
    samples = collect_all_samples()
    print(f"Total samples: {len(samples)}")
    print(f"Unique entities: {len(set(s['entity'] for s in samples))}")

    print("\nBuilding entity-level splits (70/10/20)...")
    samples, train_ents, dev_ents, test_ents = split_by_entity(samples)

    print("\nSaving...")
    meta = save_splits(samples, train_ents, dev_ents, test_ents)

    print(f"\n{'='*60}")
    print(f"  IRC-Bench v5 SPLITS")
    print(f"  Total: {meta['total_samples']} samples, {meta['total_entities']} entities")
    print(f"  Train: {meta['train']['samples']} samples, {meta['train']['entities']} entities")
    print(f"  Dev:   {meta['dev']['samples']} samples, {meta['dev']['entities']} entities")
    print(f"  Test:  {meta['test']['samples']} samples, {meta['test']['entities']} entities")
    print(f"  Entity overlap: train-test={meta['entity_overlap']['train_test']}, "
          f"train-dev={meta['entity_overlap']['train_dev']}, "
          f"dev-test={meta['entity_overlap']['dev_test']}")
    print(f"  Output: {DATA_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
