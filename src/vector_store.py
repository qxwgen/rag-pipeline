"""
src/vector_store.py
────────────────────
FAISS-backed vector store.  Stores chunk vectors and their metadata,
supports save/load, and exposes a fast similarity search interface.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from src.ingestion import Chunk


# ──────────────────────────────────────────────────────────────────────────────
# VectorStore
# ──────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wraps a FAISS index with chunk metadata for RAG retrieval.

    Uses `IndexFlatIP` (exact inner product search).
    When embeddings are L2-normalised, this equals cosine similarity.

    Parameters
    ----------
    dim : Embedding dimensionality (e.g. 384 for MiniLM).
    """

    def __init__(self, dim: int):
        if not _FAISS_AVAILABLE:
            raise ImportError("faiss not installed.  Run: pip install faiss-cpu")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    # ── building ──────────────────────────────────────────────────────────────

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """
        Add chunks and their pre-computed vectors to the store.

        Parameters
        ----------
        chunks  : List of Chunk objects (metadata).
        vectors : float32 array of shape (N, dim).
        """
        assert len(chunks) == len(vectors), "chunks and vectors must have equal length"
        vectors = np.asarray(vectors, dtype="float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        print(f"✅ Added {len(chunks)} vectors.  Index size: {self.index.ntotal}")

    # ── querying ──────────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Return top-k (Chunk, score) pairs ordered by cosine similarity.

        Parameters
        ----------
        query_vector : shape (1, dim) or (dim,)  float32
        top_k        : number of results

        Returns
        -------
        List of (Chunk, similarity_score) tuples, descending order.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty.  Ingest documents first.")

        query_vector = np.asarray(query_vector, dtype="float32")
        if query_vector.ndim == 1:
            query_vector = query_vector[None, :]

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save index + metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(path / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "ntotal": self.index.ntotal}, f)

        print(f"💾 VectorStore saved to {path}  ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a previously saved VectorStore from disk."""
        path = Path(path)

        with open(path / "meta.json") as f:
            meta = json.load(f)

        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)

        print(f"⚡ VectorStore loaded from {path}  ({store.index.ntotal} vectors)")
        return store

    # ── stats ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        sources = {c.source for c in self.chunks}
        return (
            f"VectorStore(dim={self.dim}, ntotal={self.index.ntotal}, "
            f"sources={len(sources)})"
        )
