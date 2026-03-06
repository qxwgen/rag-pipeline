"""
tests/test_pipeline.py
───────────────────────
Unit tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import Chunk, _sliding_window_chunks, _clean_text
from src.vector_store import VectorStore
from src.retriever import Retriever, _mmr
from src.evaluator import _f1, _faithfulness, QAPair


# ──────────────────────────────────────────────────────────────────────────────
# ingestion tests
# ──────────────────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_clean_text_collapses_whitespace(self):
        raw = "hello   world\n\n\n\nfoo"
        cleaned = _clean_text(raw)
        assert "   " not in cleaned
        assert cleaned.count("\n") <= 2

    def test_sliding_window_returns_chunks(self):
        text = "Hello world. " * 100
        chunks = _sliding_window_chunks(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, str)
            assert len(c) > 0

    def test_sliding_window_overlap_exists(self):
        # First and second chunk should share some words due to overlap
        text = ". ".join([f"sentence number {i}" for i in range(50)])
        chunks = _sliding_window_chunks(text, chunk_size=80, overlap=20)
        assert len(chunks) >= 2

    def test_chunk_repr(self):
        c = Chunk(text="Hello world", source="paper.pdf", chunk_id=0)
        r = repr(c)
        assert "paper.pdf" in r
        assert "Hello world" in r


# ──────────────────────────────────────────────────────────────────────────────
# vector store tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    def _make_store_with_data(self, n=10, dim=16):
        try:
            store = VectorStore(dim=dim)
        except ImportError:
            pytest.skip("faiss not installed")

        chunks  = [Chunk(text=f"text {i}", source="test.txt", chunk_id=i) for i in range(n)]
        vectors = np.random.randn(n, dim).astype("float32")
        # L2-normalise
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store.add(chunks, vectors)
        return store, vectors

    def test_add_and_search(self):
        store, vectors = self._make_store_with_data(n=20, dim=16)
        results = store.search(vectors[0:1], top_k=3)
        assert len(results) == 3
        # top result should be the query itself (score close to 1)
        assert results[0][1] > 0.99

    def test_len(self):
        store, _ = self._make_store_with_data(n=7, dim=16)
        assert len(store) == 7

    def test_save_load(self, tmp_path):
        store, vectors = self._make_store_with_data(n=5, dim=16)
        save_path = tmp_path / "index"
        store.save(str(save_path))

        loaded = VectorStore.load(str(save_path))
        assert len(loaded) == 5
        results = loaded.search(vectors[0:1], top_k=1)
        assert results[0][1] > 0.99


# ──────────────────────────────────────────────────────────────────────────────
# retriever tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRetriever:
    def _make_retriever(self, n=20, dim=16, use_mmr=False):
        try:
            store = VectorStore(dim=dim)
        except ImportError:
            pytest.skip("faiss not installed")

        chunks  = [Chunk(text=f"chunk {i}", source="s.txt", chunk_id=i) for i in range(n)]
        vectors = np.random.randn(n, dim).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store.add(chunks, vectors)
        retriever = Retriever(store, top_k=5, use_mmr=use_mmr)
        return retriever, vectors

    def test_plain_retrieval_count(self):
        retriever, vectors = self._make_retriever(use_mmr=False)
        results = retriever.retrieve(vectors[0])
        assert len(results) == 5

    def test_mmr_retrieval_diversity(self):
        retriever, vectors = self._make_retriever(n=30, use_mmr=True)
        results = retriever.retrieve(vectors[0])
        # MMR should return distinct chunks
        chunk_ids = [c.chunk_id for c, _ in results]
        assert len(set(chunk_ids)) == len(chunk_ids)


# ──────────────────────────────────────────────────────────────────────────────
# evaluator metric tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_f1_identical(self):
        assert _f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_f1_no_overlap(self):
        assert _f1("hello world", "foo bar baz") == pytest.approx(0.0)

    def test_f1_partial(self):
        score = _f1("the cat sat on the mat", "the cat")
        assert 0.0 < score < 1.0

    def test_faithfulness_full(self):
        ans  = "neural networks learn features"
        ctx  = "neural networks can learn hierarchical features from data"
        score = _faithfulness(ans, ctx)
        assert score > 0.5

    def test_faithfulness_zero(self):
        ans  = "zzz qqq xxx"
        ctx  = "hello world foo bar"
        assert _faithfulness(ans, ctx) == pytest.approx(0.0)
