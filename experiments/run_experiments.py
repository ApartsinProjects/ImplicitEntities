"""
Experiment runner for implicit entity recognition.

Implements three recognition methods from the paper:
  1) LLM-based (two-step contextualization + inference)
  2) Embedding-based (sentence-transformers cosine similarity)
  3) Hybrid RAG (embedding shortlist + LLM re-ranking)

Evaluation uses 3-tier matching: exact, LLM alias, Jaccard token overlap.
Metrics: Hit@1, Hit@3, Hit@5, Hit@10, Global MRR, Filtered MRR.

Usage:
    python run_experiments.py --dataset veterans_t2e --method llm --model google/gemini-2.0-flash-001
    python run_experiments.py --dataset all --method all --model google/gemini-2.0-flash-001
    python run_experiments.py --dataset twitter --method embedding --batch-size 100
"""

import asyncio
import argparse
import csv
import json
import os
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = Path(__file__).parent / "generated"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding support ──────────────────────────────────────────────────────
_SBERT_MODEL = None
_SBERT_AVAILABLE = None


def sbert_available() -> bool:
    global _SBERT_AVAILABLE
    if _SBERT_AVAILABLE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_AVAILABLE = True
        except ImportError:
            _SBERT_AVAILABLE = False
    return _SBERT_AVAILABLE


def get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Model loaded.")
    return _SBERT_MODEL


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    uid: str
    text: str
    entity: str
    entity_type: str
    source: str
    origin: str = ""
    sqn: int = 0

    def text_snippet(self, max_len: int = 120) -> str:
        t = self.text[:max_len]
        if len(self.text) > max_len:
            t += "..."
        return t


def load_dataset(name: str) -> tuple[list[Sample], list[str]]:
    """
    Load a dataset by name. Returns (samples, unique_entity_list).

    Supported names:
      - veterans_t2e : Veterans implicit reference dataset (T2E direction)
      - twitter       : Twitter implicit dataset
      - e2t_veterans  : Generated E2T veterans (if exists)
      - e2t_twitter   : Generated E2T twitter (if exists)
    """
    samples: list[Sample] = []

    if name == "veterans_t2e":
        path = DATA_DIR / "implicit_reference_veterans_dataset.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            if not row["text"].strip() or not row["entity"].strip():
                continue
            samples.append(Sample(
                uid=str(row["uid"]),
                text=row["text"].strip(),
                entity=row["entity"].strip(),
                entity_type=row["entity_type"].strip(),
                source=row.get("source", "veterans"),
                origin=row.get("origin", ""),
                sqn=int(row.get("SQN", 0) or 0),
            ))

    elif name == "twitter":
        path = DATA_DIR / "twitter_implicit_dataset.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            if not row["text"].strip() or not row["entity"].strip():
                continue
            samples.append(Sample(
                uid=str(row["uid"]),
                text=row["text"].strip(),
                entity=row["entity"].strip(),
                entity_type=row.get("entity_type", "").strip(),
                source=row.get("source", "twitter"),
                origin=row.get("origin", ""),
                sqn=int(row.get("SQN", 0) or 0),
            ))

    elif name == "veterans_t2e_v2":
        path = DATA_DIR / "veterans_t2e_v2.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            if not row["text"].strip() or not row["entity"].strip():
                continue
            samples.append(Sample(
                uid=str(row["uid"]),
                text=row["text"].strip(),
                entity=row["entity"].strip(),
                entity_type=row["entity_type"].strip(),
                source=row.get("source", "veterans"),
                origin=row.get("origin", ""),
                sqn=int(row.get("SQN", 0) or 0),
            ))

    elif name == "veterans_t2e_core":
        # Core types only: Person, Place, Event (no Profession/Organization)
        path = DATA_DIR / "veterans_t2e_v2.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        core_types = {"Place", "Person", "Event"}
        for _, row in df.iterrows():
            if not row["text"].strip() or not row["entity"].strip():
                continue
            if row["entity_type"].strip() not in core_types:
                continue
            samples.append(Sample(
                uid=str(row["uid"]),
                text=row["text"].strip(),
                entity=row["entity"].strip(),
                entity_type=row["entity_type"].strip(),
                source=row.get("source", "veterans"),
                origin=row.get("origin", ""),
                sqn=int(row.get("SQN", 0) or 0),
            ))

    elif name.startswith("e2t_"):
        domain = name.replace("e2t_", "")
        # Find matching generated files
        pattern = f"e2t_{domain}_*.csv"
        found = list(GENERATED_DIR.glob(pattern))
        if not found:
            print(f"  WARNING: No generated E2T files matching {pattern} in {GENERATED_DIR}")
            return [], []
        # Use the first match
        path = found[0]
        print(f"  Loading E2T from: {path.name}")
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            if not row["text"].strip() or not row["entity"].strip():
                continue
            samples.append(Sample(
                uid=str(row.get("uid", "")),
                text=row["text"].strip(),
                entity=row["entity"].strip(),
                entity_type=row.get("entity_type", "").strip(),
                source="e2t_generated",
                origin="",
                sqn=int(row.get("SQN", 0) or 0),
            ))
    elif name.startswith("bench_"):
        # Load from benchmark_v2 variants or splits
        BENCH_DIR = PROJECT_ROOT / "data" / "benchmark_v2"
        variant_map = {
            "bench_veterans_t2e": BENCH_DIR / "variants" / "veterans_t2e.csv",
            "bench_veterans_e2t": BENCH_DIR / "variants" / "veterans_e2t.csv",
            "bench_twitter_t2e": BENCH_DIR / "variants" / "twitter_t2e.csv",
            "bench_twitter_e2t": BENCH_DIR / "variants" / "twitter_e2t.csv",
            "bench_full": BENCH_DIR / "irc_benchmark_v2_full.csv",
            "bench_train": BENCH_DIR / "irc_benchmark_v2_train.csv",
            "bench_test": BENCH_DIR / "irc_benchmark_v2_test.csv",
            "bench_test_open": BENCH_DIR / "irc_benchmark_v2_test_open_set.csv",
        }
        path = variant_map.get(name)
        if not path or not path.exists():
            raise ValueError(f"Unknown benchmark dataset: {name}. Available: {list(variant_map.keys())}")
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            if not row["text"].strip() or not row["entity"].strip():
                continue
            # Skip explicit baseline if accidentally loaded
            if row.get("eval_mode", "") == "EXPLICIT_BASELINE_ONLY":
                continue
            samples.append(Sample(
                uid=str(row.get("uid", "")),
                text=row["text"].strip(),
                entity=row["entity"].strip(),
                entity_type=row.get("entity_type", "").strip(),
                source=row.get("source", row.get("variant", "")),
                origin=row.get("origin", ""),
                sqn=int(row.get("SQN", 0) or 0),
            ))

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Build unique entity list from the samples
    unique_entities = list(dict.fromkeys(s.entity for s in samples))
    print(f"  Dataset '{name}': {len(samples)} samples, {len(unique_entities)} unique entities")
    return samples, unique_entities


def get_all_dataset_names() -> list[str]:
    """List available datasets including benchmark_v2 variants."""
    names = ["veterans_t2e", "veterans_t2e_v2", "twitter"]
    # Benchmark v2 variants
    BENCH_DIR = PROJECT_ROOT / "data" / "benchmark_v2" / "variants"
    if BENCH_DIR.exists():
        for f in BENCH_DIR.glob("*.csv"):
            if "explicit" not in f.stem:
                names.append(f"bench_{f.stem}")
    # Generated E2T files
    if GENERATED_DIR.exists():
        for f in GENERATED_DIR.glob("e2t_*.csv"):
            stem = f.stem
            parts = stem.split("_", 2)
            if len(parts) >= 2:
                domain = parts[1]
                ds_name = f"e2t_{domain}"
                if ds_name not in names:
                    names.append(ds_name)
    return names


# ═══════════════════════════════════════════════════════════════════════════
#  NORMALIZATION AND MATCHING
# ═══════════════════════════════════════════════════════════════════════════

_ABBREVIATION_MAP = {
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "wwi": "world war i",
    "ww1": "world war i",
    "wwii": "world war ii",
    "ww2": "world war ii",
    "fdr": "franklin d roosevelt",
    "jfk": "john f kennedy",
    "mlk": "martin luther king jr",
    "nyc": "new york city",
    "dc": "washington dc",
    "la": "los angeles",
    "sf": "san francisco",
    "pow": "prisoner of war",
    "va": "veterans affairs",
    "gi": "gi bill",
    "nato": "north atlantic treaty organization",
    "un": "united nations",
    "ussr": "soviet union",
    "dod": "department of defense",
}

# Pattern to strip trailing parenthetical qualifiers like "(film)", "(2014 film)", "(novel)"
_PAREN_SUFFIX_RE = re.compile(r"\s*\((?:\d{4}\s+)?(?:film|movie|tv series|novel|book|song|album|band|play|musical|game|company|organization|event)\)$", re.IGNORECASE)


def normalize(text: str) -> str:
    """
    Normalize text for matching. Steps:
      1. Lowercase and strip
      2. Strip trailing parenthetical qualifiers like "(film)", "(2014 film)"
      3. Remove punctuation and collapse whitespace
      4. Strip leading articles (the/a/an)
      5. Expand common abbreviations
    """
    text = text.lower().strip()

    # Strip trailing parenthetical qualifiers: "(film)", "(2014 film)", etc.
    text = _PAREN_SUFFIX_RE.sub("", text).strip()

    # Check abbreviation map BEFORE removing punctuation (for "u.s." etc.)
    text_for_abbrev = text.strip()
    if text_for_abbrev in _ABBREVIATION_MAP:
        text = _ABBREVIATION_MAP[text_for_abbrev]

    # Remove punctuation and collapse whitespace
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    # Check abbreviation map AFTER removing punctuation too
    if text in _ABBREVIATION_MAP:
        text = _ABBREVIATION_MAP[text]

    # Strip leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)

    return text


def jaccard_token_similarity(a: str, b: str) -> float:
    """Jaccard similarity over word tokens of normalized strings."""
    tokens_a = set(normalize(a).split())
    tokens_b = set(normalize(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def containment_match(pred: str, gold: str) -> bool:
    """Check if one entity contains the other (substring match after normalization)."""
    p, g = normalize(pred), normalize(gold)
    if not p or not g or len(p) < 3 or len(g) < 3:
        return False
    return p in g or g in p


def synonym_match(pred: str, gold: str) -> bool:
    """Check if pred and gold are synonyms via WordNet Wu-Palmer similarity >= 0.9."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return False
    p_norm = normalize(pred)
    g_norm = normalize(gold)
    # For single-word entities, check direct WordNet synonymy
    p_synsets = wn.synsets(p_norm.replace(" ", "_"))
    g_synsets = wn.synsets(g_norm.replace(" ", "_"))
    if not p_synsets or not g_synsets:
        # Try individual words
        p_synsets = [s for w in p_norm.split() for s in wn.synsets(w)[:2]]
        g_synsets = [s for w in g_norm.split() for s in wn.synsets(w)[:2]]
    for ps in p_synsets[:3]:
        for gs in g_synsets[:3]:
            sim = ps.wup_similarity(gs)
            if sim is not None and sim >= 0.90:
                return True
            # Also check if they share any lemma
            p_lemmas = set(l.name().lower() for l in ps.lemmas())
            g_lemmas = set(l.name().lower() for l in gs.lemmas())
            if p_lemmas & g_lemmas:
                return True
    return False


def jaccard_match(pred: str, gold: str, threshold: float = 0.50) -> bool:
    """Jaccard match with slightly lower threshold (0.50 instead of 0.60)."""
    return jaccard_token_similarity(pred, gold) >= threshold


async def llm_alias_match_batch(
    pairs: list[tuple[str, str]],
    model: str,
    concurrency: int = 10,
) -> list[bool]:
    """
    Ask the LLM whether each (prediction, gold) pair refers to the same entity.
    Returns a list of booleans.
    """
    if not pairs:
        return []

    prompts = []
    for pred, gold in pairs:
        prompts.append([
            {"role": "system", "content": (
                "You are an entity resolution judge. "
                "Given two names, determine if they refer to the same real-world entity. "
                "Consider aliases, abbreviations, nicknames, and alternate spellings. "
                "Respond with ONLY 'YES' or 'NO'."
            )},
            {"role": "user", "content": f'Name A: "{pred}"\nName B: "{gold}"\n\nDo these refer to the same entity?'},
        ])

    responses = await batch_call(
        prompts, model=model, temperature=0.0, max_tokens=10,
        concurrency=concurrency, progress_every=200,
    )

    results = []
    for resp in responses:
        if resp is None:
            results.append(False)
        else:
            results.append(resp.strip().upper().startswith("YES"))
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  METHOD 1: LLM-BASED (Two-Step Prompting)
# ═══════════════════════════════════════════════════════════════════════════

def build_llm_context_prompts(samples: list[Sample]) -> list[list[dict]]:
    """Step 1: Generate contextual background for each text."""
    prompts = []
    for s in samples:
        prompts.append([
            {"role": "system", "content": (
                "You are a knowledgeable historian and cultural analyst. "
                "Given a text passage, generate a brief historical and situational background "
                "that could help identify any implicitly referenced entities. "
                "Focus on time periods, locations, events, cultural markers, and roles mentioned. "
                "Be concise (2-4 sentences)."
            )},
            {"role": "user", "content": f"Text: \"{s.text}\"\n\nProvide relevant background context."},
        ])
    return prompts


def build_llm_inference_prompts(
    samples: list[Sample],
    contexts: list[str] = None,
) -> list[list[dict]]:
    """
    Build inference prompts. Direct single-step (no context) by default.
    Applies fixes #1 (strict format), #2 (no context), #4 (negative instruction).
    """
    prompts = []
    for i, s in enumerate(samples):
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying named entities that are implicitly "
                "referenced in text without being named directly. "
                "Output ONLY specific entity names (proper nouns, named places, named events, "
                "named organizations, or named people). "
                "Do NOT output generic descriptions like 'a soldier', 'the narrator', or 'military service'. "
                "If the text does not clearly reference a specific named entity, output NONE."
            )},
            {"role": "user", "content": (
                f"Text: \"{s.text}\"\n\n"
                f"What specific {entity_type_hint} is implicitly described in this text?\n\n"
                f"Output EXACTLY 3 entity names, one per line. "
                f"No descriptions, no numbering, no explanations. Just the bare entity name on each line:"
            )},
        ])
    return prompts


def build_llm_fewshot_prompts(
    samples: list[Sample],
) -> list[list[dict]]:
    """
    Few-shot prompts with 3 worked examples. Fix #3.
    This is a separate method for fair comparison with zero-shot.
    """
    FEW_SHOT_EXAMPLES = (
        'Examples of implicit entity references:\n'
        'Text (Place): "a colossal figure holding a torch high, a beacon of hope and freedom"\n'
        '-> Statue of Liberty\n\n'
        'Text (Person): "he spoke of a dream, his voice rising like a hymn that marched into the streets"\n'
        '-> Martin Luther King Jr.\n\n'
        'Text (Event): "the surprise military strike on the major naval base in the Pacific"\n'
        '-> Pearl Harbor\n\n'
        'Text (Organization): "a prominent philanthropic organization known for its commitment to social justice"\n'
        '-> Ford Foundation\n\n'
        'Text (Work): "the director\'s epic about a sinking ship and a doomed romance won every award"\n'
        '-> Titanic\n\n'
    )
    prompts = []
    for s in samples:
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying named entities that are implicitly "
                "referenced in text without being named directly. "
                "Output ONLY specific entity names (proper nouns). "
                "Do NOT output generic descriptions. "
                "If the text does not clearly reference a specific named entity, output NONE."
            )},
            {"role": "user", "content": (
                f"{FEW_SHOT_EXAMPLES}"
                f"Now identify the {entity_type_hint} implicitly described:\n"
                f"Text: \"{s.text}\"\n\n"
                f"Output exactly 3 entity names, one per line:"
            )},
        ])
    return prompts


def parse_ranked_guesses(response: str, max_guesses: int = 10) -> list[str]:
    """
    Parse numbered guesses from LLM response.
    Handles formats like "1. Entity Name" or "1) Entity Name" or "- Entity Name".
    Also handles comma-separated lists and plain text responses.
    """
    if not response:
        return []

    guesses = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering: "1. ", "1) ", "- ", "* "
        cleaned = re.sub(r"^(\d+[\.\)]\s*|[-\*]\s*)", "", line).strip()
        # Remove surrounding quotes
        cleaned = cleaned.strip("\"'")
        if cleaned and len(cleaned) > 1:
            guesses.append(cleaned)
        if len(guesses) >= max_guesses:
            break

    # Fallback: if no guesses parsed from lines, try comma-separated
    if not guesses and response.strip():
        parts = [p.strip().strip("\"'") for p in response.strip().split(",")]
        guesses = [p for p in parts if len(p) > 1][:max_guesses]

    # Fallback: if still nothing, treat whole response as one guess
    if not guesses and response.strip() and len(response.strip()) > 1:
        guesses = [response.strip()[:200]]

    return guesses


async def smoke_test_llm(
    samples: list,
    model: str,
    concurrency: int = 5,
    n_samples: int = 3,
) -> bool:
    """
    Run a quick smoke test with n_samples to verify API connectivity,
    prompt format, and response parsing BEFORE launching the full batch.
    Returns True if smoke test passes, False if it fails.
    """
    test_samples = samples[:n_samples]
    print(f"\n  [SMOKE TEST] Testing {n_samples} samples with {model}...")

    # Direct inference (single-step, no context)
    inf_prompts = build_llm_inference_prompts(test_samples)
    responses = await batch_call(
        inf_prompts, model=model, temperature=0.2, max_tokens=150,
        concurrency=concurrency, progress_every=999,
    )

    null_responses = sum(1 for r in responses if r is None)
    if null_responses == n_samples:
        print(f"  [SMOKE TEST] FAIL: All {n_samples} inference calls returned None.")
        return False

    # Step 3: parse
    parsed_ok = 0
    for i, resp in enumerate(responses):
        guesses = parse_ranked_guesses(resp or "")
        gold = test_samples[i].entity
        status = "OK" if guesses else "EMPTY"
        print(f"    [{status}] Gold: \"{gold}\"")
        print(f"          Response: \"{(resp or '')[:120]}\"")
        print(f"          Parsed: {guesses[:3]}")
        if guesses:
            parsed_ok += 1

    if parsed_ok == 0:
        print(f"  [SMOKE TEST] FAIL: Parsed 0/{n_samples} responses. Response format incompatible.")
        return False

    print(f"  [SMOKE TEST] PASS: {parsed_ok}/{n_samples} parsed successfully.\n")
    return True


async def run_llm_method(
    samples: list[Sample],
    model: str,
    concurrency: int = 10,
    batch_size: int = 0,
) -> list[list[str]]:
    """
    Direct single-step LLM method (no context generation).
    Uses strict output format + negative instruction.
    Returns list of ranked predictions per sample.
    """
    print("\n  [LLM Method] Direct inference (single-step, strict format)...")
    inf_prompts = build_llm_inference_prompts(samples)
    responses = await batch_call(
        inf_prompts, model=model, temperature=0.2, max_tokens=100,
        concurrency=concurrency, progress_every=50,
    )

    all_predictions = []
    for resp in responses:
        guesses = parse_ranked_guesses(resp or "")
        all_predictions.append(guesses)


async def run_llm_fewshot_method(
    samples: list[Sample],
    model: str,
    concurrency: int = 10,
    batch_size: int = 0,
) -> list[list[str]]:
    """
    Few-shot LLM method with 3 worked examples.
    For comparison with zero-shot method.
    """
    print("\n  [LLM Few-shot] Inference with 3 examples...")
    inf_prompts = build_llm_fewshot_prompts(samples)
    responses = await batch_call(
        inf_prompts, model=model, temperature=0.2, max_tokens=100,
        concurrency=concurrency, progress_every=50,
    )

    all_predictions = []
    for resp in responses:
        guesses = parse_ranked_guesses(resp or "")
        all_predictions.append(guesses)

    return all_predictions


# ═══════════════════════════════════════════════════════════════════════════
#  METHOD 2: EMBEDDING-BASED
# ═══════════════════════════════════════════════════════════════════════════

def compute_embeddings(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Compute sentence embeddings using sentence-transformers."""
    model = get_sbert_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)


def embedding_rank_entities(
    text_embeddings: np.ndarray,
    entity_embeddings: np.ndarray,
    entity_names: list[str],
    top_k: int = 10,
) -> list[list[str]]:
    """
    For each text embedding, rank entities by cosine similarity.
    Returns top_k entity names per text.
    """
    # Both are L2-normalized, so dot product = cosine similarity
    sim_matrix = text_embeddings @ entity_embeddings.T  # (n_texts, n_entities)

    all_predictions = []
    for i in range(sim_matrix.shape[0]):
        scores = sim_matrix[i]
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_entities = [entity_names[j] for j in top_indices]
        all_predictions.append(top_entities)
    return all_predictions


async def run_embedding_method_llm_fallback(
    samples: list[Sample],
    unique_entities: list[str],
    model: str,
    concurrency: int = 10,
) -> list[list[str]]:
    """
    Fallback embedding method: use LLM to rank top-5 most similar entities.
    Used when sentence-transformers is not available.
    """
    print("  [Embedding Fallback] Using LLM similarity ranking...")

    # Chunk entity list for prompt (to avoid exceeding token limits)
    MAX_ENTITIES_PER_PROMPT = 100
    prompts = []
    for s in samples:
        # For large entity lists, we pass a randomized subset plus the answer
        if len(unique_entities) > MAX_ENTITIES_PER_PROMPT:
            # Include a random subset
            subset = list(np.random.choice(
                unique_entities, size=min(MAX_ENTITIES_PER_PROMPT, len(unique_entities)), replace=False
            ))
        else:
            subset = unique_entities

        entity_list_str = "\n".join(f"  - {e}" for e in subset)
        prompts.append([
            {"role": "system", "content": (
                "You are an entity matching expert. Given a text and a list of candidate entities, "
                "identify which entities the text most likely implicitly refers to. "
                "Return the top 5 most relevant entity names from the list.\n\n"
                "IMPORTANT: Respond in EXACTLY this format:\n"
                "1. <entity name from list>\n"
                "2. <entity name from list>\n"
                "3. <entity name from list>\n"
                "4. <entity name from list>\n"
                "5. <entity name from list>"
            )},
            {"role": "user", "content": (
                f"Text: \"{s.text}\"\n\n"
                f"Entity type hint: {s.entity_type}\n\n"
                f"Candidate entities:\n{entity_list_str}\n\n"
                "Which 5 entities does this text most likely refer to?"
            )},
        ])

    responses = await batch_call(
        prompts, model=model, temperature=0.1, max_tokens=200,
        concurrency=concurrency, progress_every=50,
    )

    all_predictions = []
    for resp in responses:
        guesses = parse_ranked_guesses(resp or "")
        all_predictions.append(guesses)

    return all_predictions


async def run_embedding_method(
    samples: list[Sample],
    unique_entities: list[str],
    model: str,
    concurrency: int = 10,
    batch_size: int = 0,
) -> list[list[str]]:
    """
    Embedding-based method using sentence-transformers (or LLM fallback).
    Returns ranked entity lists per sample.
    """
    if not sbert_available():
        print("  WARNING: sentence-transformers not available, using LLM fallback.")
        return await run_embedding_method_llm_fallback(
            samples, unique_entities, model, concurrency
        )

    print("  [Embedding Method] Computing text embeddings...")
    text_strings = [s.text for s in samples]
    text_embs = compute_embeddings(text_strings)

    print("  [Embedding Method] Computing entity embeddings...")
    entity_embs = compute_embeddings(unique_entities)

    print("  [Embedding Method] Ranking entities by cosine similarity...")
    predictions = embedding_rank_entities(text_embs, entity_embs, unique_entities, top_k=10)

    return predictions


# ═══════════════════════════════════════════════════════════════════════════
#  METHOD 3: HYBRID RAG (Embedding Shortlist + LLM Re-ranking)
# ═══════════════════════════════════════════════════════════════════════════

async def run_hybrid_method(
    samples: list[Sample],
    unique_entities: list[str],
    model: str,
    concurrency: int = 10,
    batch_size: int = 0,
    shortlist_k: int = 10,
) -> list[list[str]]:
    """
    Hybrid RAG method:
      Step 1: Use embeddings (or LLM fallback) to get top-K candidates
      Step 2: Feed text + candidates to LLM for re-ranking with reasoning
    """
    # Step 1: Get embedding-based shortlist
    print("\n  [Hybrid] Step 1: Generating candidate shortlists...")
    if sbert_available():
        text_strings = [s.text for s in samples]
        text_embs = compute_embeddings(text_strings)
        entity_embs = compute_embeddings(unique_entities)
        shortlists = embedding_rank_entities(text_embs, entity_embs, unique_entities, top_k=shortlist_k)
    else:
        # Use LLM fallback for shortlisting
        shortlists = await run_embedding_method_llm_fallback(
            samples, unique_entities, model, concurrency
        )
        # Pad to shortlist_k if needed
        for i in range(len(shortlists)):
            while len(shortlists[i]) < shortlist_k:
                shortlists[i].append("")

    # Step 2: LLM re-ranking
    print("\n  [Hybrid] Step 2: LLM re-ranking with reasoning...")
    prompts = []
    for s, candidates in zip(samples, shortlists):
        # Filter out empty candidates
        valid_candidates = [c for c in candidates if c.strip()]
        if not valid_candidates:
            prompts.append([
                {"role": "user", "content": "No candidates available. Return empty."}
            ])
            continue

        candidates_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(valid_candidates))
        entity_type_hint = s.entity_type if s.entity_type else "entity"

        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities. "
                "Given a text and a shortlist of candidate entities, re-rank them based on "
                "how well they match the implicit reference in the text. "
                "Consider contextual clues, time periods, geography, roles, and cultural markers.\n\n"
                "IMPORTANT: Respond in EXACTLY this format (re-ranked, best match first):\n"
                "1. <entity name>\n"
                "2. <entity name>\n"
                "3. <entity name>\n"
                "(list as many as are plausible, minimum 3)"
            )},
            {"role": "user", "content": (
                f"Text: \"{s.text}\"\n\n"
                f"Entity type: {entity_type_hint}\n\n"
                f"Candidate entities:\n{candidates_str}\n\n"
                "Re-rank these candidates from most to least likely."
            )},
        ])

    responses = await batch_call(
        prompts, model=model, temperature=0.1, max_tokens=200,
        concurrency=concurrency, progress_every=50,
    )

    all_predictions = []
    for resp, shortlist in zip(responses, shortlists):
        guesses = parse_ranked_guesses(resp or "")
        if not guesses:
            # Fall back to embedding order
            all_predictions.append(shortlist)
        else:
            # Append any shortlist items not in guesses (preserve full ranking)
            seen = set(normalize(g) for g in guesses)
            for s_ent in shortlist:
                if normalize(s_ent) not in seen:
                    guesses.append(s_ent)
                    seen.add(normalize(s_ent))
            all_predictions.append(guesses)

    return all_predictions


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    sample: Sample
    predictions: list[str]
    match_tier: str = "none"       # "exact", "alias", "jaccard", "none"
    match_rank: int = 0            # 1-based rank of first match (0 = no match)
    matched_prediction: str = ""


async def evaluate_predictions(
    samples: list[Sample],
    predictions: list[list[str]],
    model: str,
    concurrency: int = 10,
    max_k: int = 10,
) -> list[EvalResult]:
    """
    Evaluate predictions with 3-tier matching.
    Returns per-sample EvalResult objects.
    """
    print("\n  [Evaluation] Running 5-tier matching...")
    results: list[EvalResult] = []

    # First pass: exact, containment, synonym, jaccard matches
    needs_llm_check: list[tuple[int, int, str, str]] = []  # (result_idx, rank, pred, gold)

    for i, (s, preds) in enumerate(zip(samples, predictions)):
        er = EvalResult(sample=s, predictions=preds[:max_k])
        results.append(er)

        # Tier 1: Exact match (normalized string equality)
        for rank, pred in enumerate(preds[:max_k], start=1):
            if exact_match(pred, s.entity):
                er.match_tier = "exact"
                er.match_rank = rank
                er.matched_prediction = pred
                break

        if er.match_rank > 0:
            continue

        # Tier 2: Containment match (substring in either direction)
        for rank, pred in enumerate(preds[:max_k], start=1):
            if containment_match(pred, s.entity):
                er.match_tier = "containment"
                er.match_rank = rank
                er.matched_prediction = pred
                break

        if er.match_rank > 0:
            continue

        # Tier 3: Jaccard token overlap (>= 0.50)
        for rank, pred in enumerate(preds[:max_k], start=1):
            if jaccard_match(pred, s.entity):
                er.match_tier = "jaccard"
                er.match_rank = rank
                er.matched_prediction = pred
                break

        if er.match_rank > 0:
            continue

        # Tier 4: WordNet synonym match (Wu-Palmer >= 0.90 or shared lemma)
        for rank, pred in enumerate(preds[:max_k], start=1):
            if synonym_match(pred, s.entity):
                er.match_tier = "synonym"
                er.match_rank = rank
                er.matched_prediction = pred
                break

        if er.match_rank > 0:
            continue

        # Tier 5: Need LLM alias check for unmatched predictions
        for rank, pred in enumerate(preds[:max_k], start=1):
            needs_llm_check.append((i, rank, pred, s.entity))

    # Batch LLM alias checks
    if needs_llm_check:
        print(f"  [Evaluation] Running LLM alias matching for {len(needs_llm_check)} pairs...")
        pairs = [(pred, gold) for _, _, pred, gold in needs_llm_check]
        alias_results = await llm_alias_match_batch(pairs, model=model, concurrency=concurrency)

        # Group by result index, take first match
        alias_by_idx: dict[int, list[tuple[int, str, bool]]] = {}
        for (res_idx, rank, pred, _), is_match in zip(needs_llm_check, alias_results):
            if res_idx not in alias_by_idx:
                alias_by_idx[res_idx] = []
            alias_by_idx[res_idx].append((rank, pred, is_match))

        for res_idx, checks in alias_by_idx.items():
            er = results[res_idx]
            if er.match_rank > 0:
                continue  # Already matched
            for rank, pred, is_match in sorted(checks, key=lambda x: x[0]):
                if is_match:
                    er.match_tier = "alias"
                    er.match_rank = rank
                    er.matched_prediction = pred
                    break

    return results


def compute_metrics(results: list[EvalResult], max_k: int = 10) -> dict:
    """
    Compute full evaluation metrics from results.

    Metrics:
      - Hit@K: proportion of queries with correct entity in top K
      - Precision@K: avg fraction of top-K that are correct (single gold: 0 or 1/K)
      - Recall@K: avg fraction of gold entities found in top-K (single gold: same as Hit@K)
      - Global MRR: avg reciprocal rank (0 for misses)
      - Filtered MRR: avg reciprocal rank for hits only
      - nDCG@K: normalized discounted cumulative gain with binary relevance
      - Match tier distribution
    """
    n = len(results)
    if n == 0:
        return {}

    ks = [k for k in [1, 3, 5, 10] if k <= max_k]

    hits_at = {}
    precision_at = {}
    recall_at = {}
    ndcg_at = {}

    for k in ks:
        hit_count = sum(1 for r in results if 0 < r.match_rank <= k)
        hits_at[f"Hit@{k}"] = hit_count / n

        # Precision@K = (# correct in top-K) / K
        # With single gold: either 1/K (if hit) or 0
        prec_sum = 0.0
        for r in results:
            n_correct_in_k = 1.0 if (0 < r.match_rank <= k) else 0.0
            prec_sum += n_correct_in_k / k
        precision_at[f"P@{k}"] = prec_sum / n

        # Recall@K = (# correct in top-K) / (# total relevant)
        # With single gold: same as Hit@K. With multi-reference, divide by n_gold.
        # For now single gold, so Recall@K = Hit@K
        n_gold = 1  # single reference per sample
        recall_sum = 0.0
        for r in results:
            n_correct_in_k = 1.0 if (0 < r.match_rank <= k) else 0.0
            recall_sum += n_correct_in_k / n_gold
        recall_at[f"R@{k}"] = recall_sum / n

        # nDCG@K with binary relevance
        # DCG@K = sum(rel_i / log2(i+1)) for i=1..K
        # Ideal DCG@K = 1/log2(2) = 1.0 (single relevant item at rank 1)
        ideal_dcg = 1.0  # single relevant item
        ndcg_sum = 0.0
        for r in results:
            if 0 < r.match_rank <= k:
                dcg = 1.0 / np.log2(r.match_rank + 1)
                ndcg_sum += dcg / ideal_dcg
            # else: dcg = 0, ndcg contribution = 0
        ndcg_at[f"nDCG@{k}"] = ndcg_sum / n

    # Global MRR: 1/rank for matches, 0 for non-matches
    reciprocal_ranks = []
    for r in results:
        if r.match_rank > 0:
            reciprocal_ranks.append(1.0 / r.match_rank)
        else:
            reciprocal_ranks.append(0.0)
    global_mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Filtered MRR: only over samples that had at least one match
    filtered_rr = [rr for rr in reciprocal_ranks if rr > 0]
    filtered_mrr = np.mean(filtered_rr) if filtered_rr else 0.0

    # Match tier distribution
    tier_counts = {"exact": 0, "containment": 0, "jaccard": 0, "synonym": 0, "alias": 0, "none": 0}
    for r in results:
        tier_counts[r.match_tier] = tier_counts.get(r.match_tier, 0) + 1

    metrics = {
        "n_samples": n,
        "n_matched": sum(1 for r in results if r.match_rank > 0),
        **hits_at,
        **precision_at,
        **recall_at,
        **ndcg_at,
        "Global_MRR": global_mrr,
        "Filtered_MRR": filtered_mrr,
        **{f"tier_{k}": v for k, v in tier_counts.items()},
    }
    return metrics


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print metrics."""
    print(f"\n  {'=' * 55}")
    if label:
        print(f"  RESULTS: {label}")
    print(f"  {'=' * 55}")
    print(f"  Samples: {metrics.get('n_samples', 0)}  |  "
          f"Matched: {metrics.get('n_matched', 0)} "
          f"({100 * metrics.get('n_matched', 0) / max(metrics.get('n_samples', 1), 1):.1f}%)")
    print()

    # Hit@K row
    header = f"  {'K':>4s}  {'Hit@K':>8s}  {'P@K':>8s}  {'R@K':>8s}  {'nDCG@K':>8s}"
    print(header)
    print(f"  {'-' * 42}")
    for k in [1, 3, 5, 10]:
        hk = metrics.get(f"Hit@{k}")
        pk = metrics.get(f"P@{k}")
        rk = metrics.get(f"R@{k}")
        nk = metrics.get(f"nDCG@{k}")
        if hk is not None:
            print(f"  {k:>4d}  {hk:>8.4f}  {pk:>8.4f}  {rk:>8.4f}  {nk:>8.4f}")
    print()
    print(f"  {'Global MRR':12s}: {metrics.get('Global_MRR', 0):.4f}")
    print(f"  {'Filtered MRR':12s}: {metrics.get('Filtered_MRR', 0):.4f}")
    print()
    print(f"  Match tiers: exact={metrics.get('tier_exact', 0)}, "
          f"alias={metrics.get('tier_alias', 0)}, "
          f"jaccard={metrics.get('tier_jaccard', 0)}, "
          f"none={metrics.get('tier_none', 0)}")
    print(f"  {'=' * 55}")


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_results(
    results: list[EvalResult],
    metrics: dict,
    dataset_name: str,
    method_name: str,
    model: str,
    timestamp: str,
):
    """Save per-sample predictions CSV and metrics JSON."""
    prefix = f"{timestamp}_{dataset_name}_{method_name}_{model.replace('/', '_').replace(':', '_')}"

    # Per-sample CSV
    csv_path = RESULTS_DIR / f"{prefix}_predictions.csv"
    rows = []
    for r in results:
        preds_padded = r.predictions[:10]
        while len(preds_padded) < 10:
            preds_padded.append("")
        rows.append({
            "uid": r.sample.uid,
            "text": r.sample.text[:500],
            "gold_entity": r.sample.entity,
            "entity_type": r.sample.entity_type,
            "match_tier": r.match_tier,
            "match_rank": r.match_rank,
            "matched_prediction": r.matched_prediction,
            **{f"pred_{i+1}": p for i, p in enumerate(preds_padded)},
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Predictions saved: {csv_path}")

    # Metrics JSON
    json_path = RESULTS_DIR / f"{prefix}_metrics.json"
    meta = {
        "dataset": dataset_name,
        "method": method_name,
        "model": model,
        "timestamp": timestamp,
        "sbert_available": sbert_available(),
        **metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved:     {json_path}")

    return csv_path, json_path


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

async def run_single_experiment(
    dataset_name: str,
    method_name: str,
    model: str,
    concurrency: int = 10,
    batch_size: int = 0,
    max_samples: int = 0,
    timestamp: str = "",
) -> dict:
    """Run one experiment (dataset x method). Returns metrics dict."""
    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENT: dataset={dataset_name}, method={method_name}")
    print(f"  Model: {model}")
    print(f"{'#' * 60}")

    t0 = time.time()

    # Load data
    samples, unique_entities = load_dataset(dataset_name)
    if not samples:
        print("  ERROR: No samples loaded, skipping.")
        return {}

    # Optionally limit samples for testing
    if max_samples and max_samples < len(samples):
        print(f"  Limiting to {max_samples} samples (of {len(samples)})")
        samples = samples[:max_samples]

    # Smoke test for LLM-based methods (catches API/parsing issues early)
    if method_name in ("llm", "hybrid"):
        smoke_ok = await smoke_test_llm(samples, model, concurrency=min(concurrency, 5))
        if not smoke_ok:
            print(f"  SMOKE TEST FAILED for {method_name}/{model}. Skipping this experiment.")
            print(f"  This usually means the model's response format doesn't match the parser.")
            return {"error": f"smoke_test_failed for {model}", "Hit@1": 0, "Global_MRR": 0}

    # Run method
    if method_name == "llm":
        predictions = await run_llm_method(samples, model, concurrency, batch_size)
    elif method_name == "fewshot":
        predictions = await run_llm_fewshot_method(samples, model, concurrency, batch_size)
    elif method_name == "embedding":
        predictions = await run_embedding_method(
            samples, unique_entities, model, concurrency, batch_size
        )
    elif method_name == "hybrid":
        predictions = await run_hybrid_method(
            samples, unique_entities, model, concurrency, batch_size
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Evaluate
    eval_results = await evaluate_predictions(
        samples, predictions, model=model, concurrency=concurrency,
    )

    # Compute metrics
    metrics = compute_metrics(eval_results)
    label = f"{dataset_name} / {method_name} / {model}"
    print_metrics(metrics, label)

    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save
    save_results(eval_results, metrics, dataset_name, method_name, model, timestamp)

    return metrics


async def main():
    parser = argparse.ArgumentParser(
        description="Run implicit entity recognition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_experiments.py --dataset veterans_t2e --method llm\n"
            "  python run_experiments.py --dataset twitter --method embedding\n"
            "  python run_experiments.py --dataset all --method all\n"
            "  python run_experiments.py --dataset veterans_t2e --method hybrid --max-samples 50\n"
        ),
    )
    parser.add_argument(
        "--dataset", default="veterans_t2e",
        help="Dataset name: veterans_t2e, twitter, e2t_veterans, e2t_twitter, or 'all' (default: veterans_t2e)",
    )
    parser.add_argument(
        "--method", default="llm", choices=["llm", "embedding", "hybrid", "all"],
        help="Recognition method (default: llm)",
    )
    parser.add_argument(
        "--model", default="google/gemini-2.0-flash-001",
        help="OpenRouter model ID (default: google/gemini-2.0-flash-001)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Process samples in batches of this size, 0 for all at once (default: 0)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit number of samples for testing, 0 for all (default: 0)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine datasets
    if args.dataset == "all":
        dataset_names = get_all_dataset_names()
    else:
        dataset_names = [args.dataset]

    # Determine methods
    if args.method == "all":
        method_names = ["llm", "embedding", "hybrid"]
    else:
        method_names = [args.method]

    print(f"\n{'=' * 60}")
    print(f"  Implicit Entity Recognition Experiments")
    print(f"  Datasets: {dataset_names}")
    print(f"  Methods:  {method_names}")
    print(f"  Model:    {args.model}")
    print(f"  SBERT:    {'available' if sbert_available() else 'NOT available (will use LLM fallback)'}")
    print(f"  Time:     {timestamp}")
    print(f"{'=' * 60}")

    all_metrics = {}
    for ds in dataset_names:
        for method in method_names:
            try:
                metrics = await run_single_experiment(
                    dataset_name=ds,
                    method_name=method,
                    model=args.model,
                    concurrency=args.concurrency,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    timestamp=timestamp,
                )
                all_metrics[f"{ds}/{method}"] = metrics
            except Exception as e:
                print(f"\n  ERROR in {ds}/{method}: {e}")
                import traceback
                traceback.print_exc()
                all_metrics[f"{ds}/{method}"] = {"error": str(e)}

    # Summary table
    if len(all_metrics) > 1:
        print(f"\n\n{'=' * 70}")
        print(f"  SUMMARY")
        print(f"{'=' * 70}")
        header = f"  {'Experiment':<35s} {'Hit@1':>7s} {'Hit@3':>7s} {'MRR':>7s} {'Time':>7s}"
        print(header)
        print(f"  {'-' * 65}")
        for key, m in all_metrics.items():
            if "error" in m:
                print(f"  {key:<35s} ERROR: {m['error'][:30]}")
            else:
                h1 = f"{m.get('Hit@1', 0):.3f}"
                h3 = f"{m.get('Hit@3', 0):.3f}"
                mrr = f"{m.get('Global_MRR', 0):.3f}"
                t = f"{m.get('elapsed_seconds', 0):.0f}s"
                print(f"  {key:<35s} {h1:>7s} {h3:>7s} {mrr:>7s} {t:>7s}")
        print(f"{'=' * 70}")

    # Save combined summary
    summary_path = RESULTS_DIR / f"{timestamp}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model": args.model,
            "datasets": dataset_names,
            "methods": method_names,
            "results": all_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Combined summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
