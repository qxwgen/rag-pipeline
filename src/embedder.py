"""
src/embedder.py
───────────────
Wrap sentence-transformers for batch embedding of text chunks.
Supports caching to avoid re-computing embeddings.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

from src.ingestion import Chunk


# ──────────────────────────────────────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────────────────────────────────────

class Embedder:
    """
    Encodes text into dense vectors using a Sentence Transformer model.

    Parameters
    ----------
    model_name  : HuggingFace model identifier.
    cache_dir   : If set, embeddings are cached to disk (avoids re-computation).
    batch_size  : Number of texts to encode in one forward pass.
    normalize   : L2-normalise vectors (required for cosine via inner product).
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str | Path] = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed.  Run: pip install sentence-transformers"
            )
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        print(f"   Embedding dim : {self.embedding_dim}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cache_key(self, texts: List[str]) -> str:
        joined = "\n".join(texts) + self.model_name
        return hashlib.md5(joined.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, vectors: np.ndarray) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(vectors, f)

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of strings → float32 matrix of shape (N, D).
        Results are L2-normalised when self.normalize=True.
        """
        key = self._cache_key(texts)
        cached = self._load_cache(key)
        if cached is not None:
            print(f"   ⚡ Loaded {len(cached)} embeddings from cache")
            return cached

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        self._save_cache(key, vectors)
        return vectors.astype("float32")

    def encode_chunks(
        self, chunks: List[Chunk], show_progress: bool = True
    ) -> np.ndarray:
        """Encode a list of Chunk objects. Returns (N, D) float32 array."""
        texts = [c.text for c in chunks]
        return self.encode(texts, show_progress=show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → (1, D) float32 array."""
        return self.encode([query], show_progress=False)
