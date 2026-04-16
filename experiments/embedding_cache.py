"""
Embedding cache: encode once, reuse across all experiments.

Caches embeddings by (model_name, text_hash) so the same text is never
encoded twice, even across different experiment runs.

Versioned: stores model name + dimension in cache metadata so stale
embeddings are detected if the model changes.

Usage:
    from embedding_cache import EmbeddingCache

    cache = EmbeddingCache("BAAI/bge-base-en-v1.5")
    embs = cache.encode(texts)          # encodes only uncached texts
    cache.save()                         # persist to disk

    # Later, different experiment:
    cache2 = EmbeddingCache("BAAI/bge-base-en-v1.5")  # loads from disk
    embs = cache2.encode(same_texts)    # instant, all cached
"""
import hashlib
import json
import time
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "embedding_cache_data"
CACHE_DIR.mkdir(exist_ok=True)


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    """
    Persistent embedding cache keyed by (model, text_hash).
    Stores embeddings as numpy arrays on disk.
    """

    def __init__(self, model_name: str, prefix: str = ""):
        self.model_name = model_name
        self.safe_name = model_name.replace("/", "_").replace(":", "_")
        self.prefix = prefix
        self._model = None
        self._dim = None

        # Cache: hash -> embedding (numpy array)
        self._cache = {}
        self._meta_path = CACHE_DIR / f"{self.safe_name}_{prefix}_meta.json"
        self._data_path = CACHE_DIR / f"{self.safe_name}_{prefix}_embeddings.npz"

        self._load_from_disk()

    def _load_from_disk(self):
        if self._meta_path.exists() and self._data_path.exists():
            meta = json.load(open(self._meta_path))
            if meta.get("model_name") != self.model_name:
                print(f"  [Cache] Model mismatch: cached={meta.get('model_name')}, "
                      f"requested={self.model_name}. Ignoring cache.")
                return
            data = np.load(self._data_path)
            hashes = meta.get("hashes", [])
            for i, h in enumerate(hashes):
                if f"emb_{i}" in data:
                    self._cache[h] = data[f"emb_{i}"]
            self._dim = meta.get("dim")
            print(f"  [Cache] Loaded {len(self._cache)} cached embeddings "
                  f"(model={self.model_name}, dim={self._dim})")

    def save(self):
        """Save cache to disk."""
        if not self._cache:
            return
        hashes = list(self._cache.keys())
        arrays = {f"emb_{i}": self._cache[h] for i, h in enumerate(hashes)}
        np.savez_compressed(self._data_path, **arrays)
        meta = {
            "model_name": self.model_name,
            "dim": self._dim,
            "n_entries": len(hashes),
            "hashes": hashes,
        }
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [Cache] Saved {len(hashes)} embeddings to disk")

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self._dim = self._model.get_sentence_embedding_dimension()
            print(f"  [Cache] Loaded model {self.model_name} (dim={self._dim}, device={self._model.device})")
        return self._model

    def encode(self, texts: list[str], batch_size: int = 128,
               show_progress: bool = True, query_prefix: str = "") -> np.ndarray:
        """
        Encode texts, using cache for already-encoded texts.
        Returns (n_texts, dim) numpy array.
        """
        hashes = [text_hash(query_prefix + t) for t in texts]

        # Find uncached
        uncached_indices = [i for i, h in enumerate(hashes) if h not in self._cache]
        cached_count = len(texts) - len(uncached_indices)

        if uncached_indices:
            model = self._get_model()
            uncached_texts = [query_prefix + texts[i] for i in uncached_indices]
            print(f"  [Cache] Encoding {len(uncached_texts)} new texts "
                  f"({cached_count} cached, {len(uncached_texts)} uncached)")
            new_embs = model.encode(
                uncached_texts, batch_size=batch_size,
                show_progress_bar=show_progress, normalize_embeddings=True,
            )
            for idx, emb in zip(uncached_indices, new_embs):
                self._cache[hashes[idx]] = emb
        else:
            print(f"  [Cache] All {len(texts)} texts cached (0 to encode)")

        # Assemble result
        result = np.array([self._cache[h] for h in hashes])
        return result

    def get_dim(self) -> int:
        if self._dim is None:
            self._get_model()
        return self._dim

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self):
        self._cache = {}
        if self._meta_path.exists():
            self._meta_path.unlink()
        if self._data_path.exists():
            self._data_path.unlink()


if __name__ == "__main__":
    # Demo
    cache = EmbeddingCache("all-MiniLM-L6-v2", prefix="demo")

    texts = [
        "A colossal figure holding a torch, a beacon of hope.",
        "He spoke of a dream, his voice rising like a hymn.",
        "The surprise attack on the naval base in the Pacific.",
    ]

    # First call: encodes all
    embs1 = cache.encode(texts)
    print(f"Shape: {embs1.shape}")
    cache.save()

    # Second call: all cached
    embs2 = cache.encode(texts)
    print(f"Same embeddings: {np.allclose(embs1, embs2)}")

    # Add new text: only encodes the new one
    texts2 = texts + ["A new text to encode."]
    embs3 = cache.encode(texts2)
    print(f"Shape with new: {embs3.shape}")
    cache.save()
