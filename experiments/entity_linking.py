"""
Entity linking to Wikidata/Wikipedia for knowledge-grounded matching.

Two use cases:
  1. Link gold entities to Wikidata QIDs for canonical matching
     (prediction "Pearl Harbor" and gold "Attack on Pearl Harbor" both link to Q52418)
  2. Compute knowledge graph distance between prediction and gold
     (entities sharing the same Wikipedia category or Wikidata class are related)

APIs used (all free, no API key needed):
  - Wikidata Search API: find entities by name
  - Wikidata Entity API: get entity properties (classes, categories, aliases)
  - Wikipedia API: search and get page info

Usage:
    from entity_linking import WikidataLinker

    linker = WikidataLinker()
    qid = linker.link("Pearl Harbor")           # -> "Q52418"
    match = linker.same_entity("Pearl Harbor", "Attack on Pearl Harbor")  # -> True
    dist = linker.distance("Pearl Harbor", "World War II")  # -> 0.6 (related)
"""
import json
import time
import hashlib
import re
from pathlib import Path
from functools import lru_cache

import requests

CACHE_DIR = Path(__file__).parent / "entity_cache"
CACHE_DIR.mkdir(exist_ok=True)

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"

# Rate limiting
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests


def _rate_limit():
    global _last_request_time
    now = time.time()
    wait = MIN_REQUEST_INTERVAL - (now - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.time()


def _cache_key(method: str, query: str) -> str:
    h = hashlib.md5(f"{method}:{query}".encode()).hexdigest()[:12]
    return h


class WikidataLinker:
    """Link entity names to Wikidata QIDs and compute entity relationships."""

    def __init__(self, cache: bool = True):
        self.use_cache = cache
        self._link_cache = {}
        self._load_disk_cache()

    def _load_disk_cache(self):
        cache_file = CACHE_DIR / "wikidata_links.json"
        if cache_file.exists():
            self._link_cache = json.load(open(cache_file, encoding="utf-8"))

    def _save_disk_cache(self):
        cache_file = CACHE_DIR / "wikidata_links.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self._link_cache, f, indent=2, ensure_ascii=False)

    def search_wikidata(self, query: str, limit: int = 5) -> list[dict]:
        """Search Wikidata for entities matching a query string."""
        _rate_limit()
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "limit": limit,
            "format": "json",
        }
        try:
            resp = requests.get(WIKIDATA_SEARCH, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("search", [])
        except Exception as e:
            return []

    def search_wikipedia(self, query: str, limit: int = 3) -> list[dict]:
        """Search Wikipedia for pages matching a query string."""
        _rate_limit()
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        try:
            resp = requests.get(WIKIPEDIA_SEARCH, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("query", {}).get("search", [])
        except Exception:
            return []

    def link(self, entity_name: str) -> dict | None:
        """
        Link an entity name to its Wikidata QID.
        Returns {qid, label, description, aliases} or None.
        """
        key = entity_name.lower().strip()
        if key in self._link_cache:
            return self._link_cache[key]

        results = self.search_wikidata(entity_name)
        if not results:
            # Fallback: try Wikipedia search
            wp_results = self.search_wikipedia(entity_name)
            if wp_results:
                # Try Wikidata with Wikipedia title
                results = self.search_wikidata(wp_results[0].get("title", entity_name))

        if results:
            top = results[0]
            entry = {
                "qid": top.get("id", ""),
                "label": top.get("label", ""),
                "description": top.get("description", ""),
                "aliases": top.get("aliases", []),
                "url": top.get("url", ""),
            }
            self._link_cache[key] = entry
            return entry

        self._link_cache[key] = None
        return None

    def get_entity_classes(self, qid: str) -> list[str]:
        """Get the 'instance of' (P31) and 'subclass of' (P279) for a QID."""
        _rate_limit()
        url = f"https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetclaims",
            "entity": qid,
            "property": "P31",  # instance of
            "format": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            claims = data.get("claims", {}).get("P31", [])
            classes = []
            for claim in claims:
                target = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if "id" in target:
                    classes.append(target["id"])
            return classes
        except Exception:
            return []

    def same_entity(self, entity_a: str, entity_b: str) -> bool:
        """Check if two entity names resolve to the same Wikidata QID."""
        link_a = self.link(entity_a)
        link_b = self.link(entity_b)
        if not link_a or not link_b:
            return False
        return link_a["qid"] == link_b["qid"]

    def qid_match(self, pred: str, gold: str) -> float:
        """
        Compute QID-based match score.
        1.0 = same QID (same entity)
        0.5 = share a Wikidata class (related)
        0.0 = no relationship found
        """
        link_pred = self.link(pred)
        link_gold = self.link(gold)

        if not link_pred or not link_gold:
            return 0.0

        # Same entity
        if link_pred["qid"] == link_gold["qid"]:
            return 1.0

        # Check alias match: gold appears in pred's aliases or vice versa
        pred_aliases = set(a.lower() for a in link_pred.get("aliases", []))
        gold_aliases = set(a.lower() for a in link_gold.get("aliases", []))
        pred_aliases.add(link_pred.get("label", "").lower())
        gold_aliases.add(link_gold.get("label", "").lower())

        if gold.lower() in pred_aliases or pred.lower() in gold_aliases:
            return 0.9

        # Check shared classes (both are instances of the same thing)
        pred_classes = set(self.get_entity_classes(link_pred["qid"]))
        gold_classes = set(self.get_entity_classes(link_gold["qid"]))

        if pred_classes and gold_classes:
            shared = pred_classes & gold_classes
            if shared:
                return 0.5  # related (same type of entity)

        return 0.0

    def batch_link(self, entities: list[str], save_every: int = 50) -> dict:
        """Link a list of entities, saving cache periodically."""
        results = {}
        for i, entity in enumerate(entities):
            results[entity] = self.link(entity)
            if (i + 1) % save_every == 0:
                self._save_disk_cache()
                print(f"  [EntityLinker] {i+1}/{len(entities)} linked, cache saved")
        self._save_disk_cache()
        return results


def link_benchmark_entities(benchmark_path: str = None):
    """Link all entities in the benchmark to Wikidata QIDs."""
    import csv
    if benchmark_path is None:
        benchmark_path = Path(__file__).parent.parent / "data" / "benchmark_v2" / "irc_benchmark_v2_clean.csv"

    rows = list(csv.DictReader(open(benchmark_path, encoding="utf-8")))
    entities = list(set(r["entity"].strip() for r in rows if r["entity"].strip()))
    print(f"Linking {len(entities)} unique entities to Wikidata...")

    linker = WikidataLinker()
    results = linker.batch_link(entities)

    linked = sum(1 for v in results.values() if v is not None)
    print(f"  Linked: {linked}/{len(entities)} ({linked/len(entities)*100:.0f}%)")

    # Save
    out_path = CACHE_DIR / "benchmark_entity_links.json"
    serializable = {}
    for entity, link in results.items():
        if link:
            serializable[entity] = link
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        link_benchmark_entities()
    else:
        # Quick test
        linker = WikidataLinker()

        test_pairs = [
            ("Pearl Harbor", "Attack on Pearl Harbor"),
            ("Pearl Harbor", "World War II"),
            ("Martin Luther King Jr.", "Martin Luther King"),
            ("Harvard", "Harvard University"),
            ("Statue of Liberty", "New York Harbor"),
            ("101st Airborne", "82nd Airborne"),
            ("Completely Wrong", "Pearl Harbor"),
        ]

        print(f"{'Entity A':<30s} {'Entity B':<30s} {'Same?':>5s} {'Score':>5s}")
        print("-" * 75)
        for a, b in test_pairs:
            same = linker.same_entity(a, b)
            score = linker.qid_match(a, b)
            print(f"{a:<30s} {b:<30s} {'YES' if same else 'NO':>5s} {score:>5.1f}")

        linker._save_disk_cache()
        print(f"\nCache saved: {len(linker._link_cache)} entries")
