"""
src/retriever.py
────────────────
Retrieval layer on top of VectorStore.
Implements two strategies:
  1. Plain top-K cosine retrieval
  2. Maximal Marginal Relevance (MMR) — balances relevance vs. diversity
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.ingestion import Chunk
from src.vector_store import VectorStore


# ──────────────────────────────────────────────────────────────────────────────
# MMR helper
# ──────────────────────────────────────────────────────────────────────────────

def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_chunks: List[Chunk],
    top_k: int,
    lambda_: float,
) -> List[Tuple[Chunk, float]]:
    """
    Maximal Marginal Relevance selection.

    At each step picks the candidate that maximises:
        score = λ · sim(query, c) - (1-λ) · max_sim(c, already_selected)

    Parameters
    ----------
    lambda_ : 1.0 = pure relevance, 0.0 = pure diversity
    """
    selected_indices: List[int] = []
    selected_vecs: List[np.ndarray] = []

    # cosine similarities between query and all candidates
    query_sims = (candidate_vecs @ query_vec.T).ravel()

    remaining = list(range(len(candidate_chunks)))

    for _ in range(min(top_k, len(candidate_chunks))):
        if not selected_vecs:
            # First pick: highest relevance
            best = int(np.argmax(query_sims[remaining]))
            best_idx = remaining[best]
        else:
            selected_mat = np.stack(selected_vecs)  # (S, D)
            mmr_scores = []
            for idx in remaining:
                rel = query_sims[idx]
                red = float(np.max(candidate_vecs[idx] @ selected_mat.T))
                mmr_scores.append(lambda_ * rel - (1 - lambda_) * red)
            best = int(np.argmax(mmr_scores))
            best_idx = remaining[best]

        selected_indices.append(best_idx)
        selected_vecs.append(candidate_vecs[best_idx])
        remaining.remove(best_idx)

    return [
        (candidate_chunks[i], float(query_sims[i]))
        for i in selected_indices
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Retrieves relevant chunks from a VectorStore for a query vector.

    Parameters
    ----------
    vector_store : A built/loaded VectorStore.
    top_k        : Number of chunks to return.
    use_mmr      : Enable MMR re-ranking for diversity.
    mmr_lambda   : MMR trade-off (1=relevance, 0=diversity).
    mmr_fetch_k  : Candidates fetched before MMR filtering.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        mmr_fetch_k: int = 20,
    ):
        self.vs = vector_store
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.mmr_fetch_k = mmr_fetch_k

    def retrieve(
        self, query_vector: np.ndarray
    ) -> List[Tuple[Chunk, float]]:
        """
        Return top-k (Chunk, score) pairs for the given query vector.

        Parameters
        ----------
        query_vector : shape (1, D) or (D,) float32

        Returns
        -------
        List of (Chunk, similarity_score), best-first.
        """
        query_vector = np.asarray(query_vector, dtype="float32").ravel()

        if not self.use_mmr:
            return self.vs.search(query_vector[None, :], top_k=self.top_k)

        # Fetch a larger candidate set, then re-rank with MMR
        fetch_k = max(self.top_k, self.mmr_fetch_k)
        candidates = self.vs.search(query_vector[None, :], top_k=fetch_k)

        if not candidates:
            return []

        cand_chunks = [c for c, _ in candidates]
        cand_indices = [self.vs.chunks.index(c) for c in cand_chunks]

        # Reconstruct vectors for the candidate indices
        # (FAISS IndexFlatIP stores them — we reconstruct via reconstruct_batch)
        cand_vecs = np.stack([
            self.vs.index.reconstruct(i) for i in cand_indices
        ]).astype("float32")

        return _mmr(query_vector, cand_vecs, cand_chunks,
                    top_k=self.top_k, lambda_=self.mmr_lambda)

    def format_context(self, results: List[Tuple[Chunk, float]]) -> str:
        """Format retrieved chunks as a numbered context block for prompting."""
        lines = []
        for rank, (chunk, score) in enumerate(results, start=1):
            lines.append(
                f"[{rank}] Source: {chunk.source}"
                + (f", Page {chunk.page}" if chunk.page else "")
                + f" (score={score:.3f})\n{chunk.text}"
            )
        return "\n\n".join(lines)
