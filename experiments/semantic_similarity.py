"""
Multi-metric semantic similarity between predicted and gold entities.

Metrics:
  1. Exact match (binary)
  2. Jaccard token overlap
  3. Containment (substring match)
  4. Embedding cosine similarity (sentence-transformers)
  5. WordNet Wu-Palmer similarity
  6. Entity type agreement
  7. Combined graded relevance score

Usage:
    from semantic_similarity import SemanticMatcher, compute_graded_metrics

    matcher = SemanticMatcher()  # loads embedding model once
    score = matcher.score("Pearl Harbor", "World War II")
    # Returns: {"exact": 0, "jaccard": 0.0, "containment": 0, "embedding": 0.72,
    #           "wordnet": 0.5, "type_match": 1.0, "combined": 0.45}

    # Graded evaluation of predictions vs gold
    metrics = compute_graded_metrics(eval_results, matcher)
"""
import re
import string
import numpy as np
from pathlib import Path
from functools import lru_cache

# ── Normalization ────────────────────────────────────────────────────────

_ARTICLES = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)
_PAREN = re.compile(r'\s*\([^)]*\)$')


def normalize(text: str) -> str:
    """Normalize entity text for comparison."""
    text = text.lower().strip()
    text = _PAREN.sub('', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = _ARTICLES.sub('', text)
    return text


def tokenize(text: str) -> set:
    return set(normalize(text).split())


# ── Individual metrics ───────────────────────────────────────────────────

def exact_match(pred: str, gold: str) -> float:
    """Binary exact match after normalization."""
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def jaccard_similarity(pred: str, gold: str) -> float:
    """Jaccard token overlap."""
    a, b = tokenize(pred), tokenize(gold)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment_score(pred: str, gold: str) -> float:
    """
    Bidirectional containment: does one contain the other?
    Returns 1.0 if full containment, partial score for partial containment.
    """
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0

    # Full containment
    if p in g or g in p:
        # Score based on how much of the shorter is in the longer
        shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
        return len(shorter) / len(longer)

    # Token-level containment
    p_tokens, g_tokens = tokenize(pred), tokenize(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    shorter_t, longer_t = (p_tokens, g_tokens) if len(p_tokens) <= len(g_tokens) else (g_tokens, p_tokens)
    overlap = len(shorter_t & longer_t)
    return overlap / len(longer_t) if longer_t else 0.0


def wordnet_similarity(pred: str, gold: str) -> float:
    """
    WordNet Wu-Palmer similarity between entity names.
    Uses best matching synset pair across all tokens.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return 0.0

    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    best_scores = []
    for pt in pred_tokens:
        p_synsets = wn.synsets(pt)
        if not p_synsets:
            continue
        for gt in gold_tokens:
            g_synsets = wn.synsets(gt)
            if not g_synsets:
                continue
            for ps in p_synsets[:2]:  # limit to top 2 senses
                for gs in g_synsets[:2]:
                    sim = ps.wup_similarity(gs)
                    if sim is not None:
                        best_scores.append(sim)

    return max(best_scores) if best_scores else 0.0


def type_agreement(pred_type: str, gold_type: str) -> float:
    """Entity type agreement score."""
    if not pred_type or not gold_type:
        return 0.5  # unknown type, neutral
    return 1.0 if pred_type.lower() == gold_type.lower() else 0.0


# ── Embedding-based similarity ───────────────────────────────────────────

class EmbeddingSimilarity:
    """Caches entity embeddings for fast pairwise comparison."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._cache = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        key = normalize(text)
        if key not in self._cache:
            emb = self.model.encode([text], normalize_embeddings=True)[0]
            self._cache[key] = emb
        return self._cache[key]

    def similarity(self, pred: str, gold: str) -> float:
        """Cosine similarity between entity name embeddings."""
        if not pred.strip() or not gold.strip():
            return 0.0
        e1 = self._get_embedding(pred)
        e2 = self._get_embedding(gold)
        return float(np.dot(e1, e2))


# ── Combined Matcher ─────────────────────────────────────────────────────

class SemanticMatcher:
    """
    Multi-metric semantic matcher with combined relevance scoring.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_wordnet: bool = True):
        self.embedding = EmbeddingSimilarity(embedding_model)
        self.use_wordnet = use_wordnet

        # Weights for combined score
        self.weights = {
            "exact": 0.30,
            "containment": 0.15,
            "jaccard": 0.10,
            "embedding": 0.25,
            "wordnet": 0.20,
        }

    def score(self, pred: str, gold: str, pred_type: str = "", gold_type: str = "") -> dict:
        """
        Compute all similarity metrics between prediction and gold.
        Returns dict of individual scores + combined weighted score.
        """
        scores = {
            "exact": exact_match(pred, gold),
            "jaccard": jaccard_similarity(pred, gold),
            "containment": containment_score(pred, gold),
            "embedding": self.embedding.similarity(pred, gold),
            "wordnet": wordnet_similarity(pred, gold) if self.use_wordnet else 0.0,
            "type_match": type_agreement(pred_type, gold_type),
        }

        # Combined weighted score
        combined = sum(scores[k] * self.weights[k] for k in self.weights if k in scores)
        scores["combined"] = min(combined, 1.0)

        # Graded relevance bucket (for nDCG)
        if scores["exact"] == 1.0:
            scores["relevance"] = 3  # perfect match
        elif scores["combined"] >= 0.5:
            scores["relevance"] = 2  # highly relevant
        elif scores["combined"] >= 0.25:
            scores["relevance"] = 1  # partially relevant
        else:
            scores["relevance"] = 0  # irrelevant

        return scores

    def score_predictions(self, predictions: list[str], gold: str,
                          gold_type: str = "") -> list[dict]:
        """Score all predictions against a gold entity."""
        return [self.score(pred, gold, gold_type=gold_type) for pred in predictions]


# ── Graded Evaluation Metrics ────────────────────────────────────────────

def compute_graded_metrics(
    eval_results,  # list of EvalResult from run_experiments
    matcher: SemanticMatcher = None,
    max_k: int = 10,
) -> dict:
    """
    Compute graded metrics using semantic similarity instead of binary matching.

    Returns extended metrics dict with:
      - Graded nDCG@K (using relevance 0-3 instead of binary)
      - Mean embedding similarity of top-1 prediction
      - Mean combined similarity of top-1 prediction
      - Distribution of relevance grades
      - Partial match rate (combined > 0.25 but not exact)
    """
    if matcher is None:
        matcher = SemanticMatcher()

    n = len(eval_results)
    if n == 0:
        return {}

    ks = [k for k in [1, 3, 5, 10] if k <= max_k]

    # Per-sample graded scores
    top1_embedding = []
    top1_combined = []
    top1_relevance = []
    graded_ndcg = {k: [] for k in ks}

    for r in eval_results:
        gold = r.sample.entity
        gold_type = r.sample.entity_type
        preds = r.predictions[:max_k] if r.predictions else []

        if not preds:
            top1_embedding.append(0.0)
            top1_combined.append(0.0)
            top1_relevance.append(0)
            for k in ks:
                graded_ndcg[k].append(0.0)
            continue

        # Score all predictions
        pred_scores = matcher.score_predictions(preds, gold, gold_type)

        # Top-1 scores
        top1_embedding.append(pred_scores[0]["embedding"])
        top1_combined.append(pred_scores[0]["combined"])
        top1_relevance.append(pred_scores[0]["relevance"])

        # Graded nDCG@K
        for k in ks:
            k_scores = pred_scores[:k]
            dcg = sum(s["relevance"] / np.log2(i + 2) for i, s in enumerate(k_scores))
            # Ideal: all relevant items sorted by relevance
            ideal_rels = sorted([s["relevance"] for s in pred_scores[:k]], reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            graded_ndcg[k].append(ndcg)

    # Aggregate
    relevance_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for rel in top1_relevance:
        relevance_dist[rel] = relevance_dist.get(rel, 0) + 1

    metrics = {
        "graded_top1_embedding_sim": float(np.mean(top1_embedding)),
        "graded_top1_combined_sim": float(np.mean(top1_combined)),
        "graded_top1_mean_relevance": float(np.mean(top1_relevance)),
        "graded_partial_match_rate": sum(1 for c in top1_combined if c >= 0.25) / n,
        "graded_strong_match_rate": sum(1 for c in top1_combined if c >= 0.50) / n,
        "graded_relevance_dist": {
            "irrelevant_0": relevance_dist[0],
            "partial_1": relevance_dist[1],
            "relevant_2": relevance_dist[2],
            "exact_3": relevance_dist[3],
        },
    }

    for k in ks:
        metrics[f"graded_nDCG@{k}"] = float(np.mean(graded_ndcg[k]))

    return metrics


# ── CLI for testing ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading SemanticMatcher...")
    matcher = SemanticMatcher()

    test_pairs = [
        ("Pearl Harbor", "World War II"),
        ("Pearl Harbor", "Pearl Harbor"),
        ("New York Harbor", "Statue of Liberty"),
        ("Attack on Pearl Harbor", "Pearl Harbor"),
        ("Anti-Aircraft Gunner", "90mm anti-aircraft gun battery"),
        ("Martin Luther King Jr.", "Civil Rights Movement"),
        ("France", "French"),
        ("Harvard University", "Harvard"),
        ("101st Airborne Division", "331st Infantry"),
        ("World War II", "Vietnam War"),
        ("Completely Wrong Answer", "Pearl Harbor"),
    ]

    print(f"\n{'Prediction':<35s} {'Gold':<35s} {'Exact':>5s} {'Cont':>5s} {'Jacc':>5s} {'Emb':>5s} {'WN':>5s} {'Comb':>5s} {'Rel':>3s}")
    print("-" * 130)
    for pred, gold in test_pairs:
        s = matcher.score(pred, gold)
        print(f"{pred:<35s} {gold:<35s} {s['exact']:>5.2f} {s['containment']:>5.2f} {s['jaccard']:>5.2f} {s['embedding']:>5.2f} {s['wordnet']:>5.2f} {s['combined']:>5.2f} {s['relevance']:>3d}")
