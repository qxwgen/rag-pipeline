"""
src/pipeline.py
───────────────
Orchestrates the full RAG loop:
    query → embed → retrieve → format → generate → return answer + sources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.ingestion import Chunk, ingest_directory
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import build_generator


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    query: str
    answer: str
    sources: List[Tuple[Chunk, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"❓ Query : {self.query}",
            f"\n💬 Answer:\n{self.answer}",
            "\n📎 Retrieved sources:",
        ]
        for rank, (chunk, score) in enumerate(self.sources, start=1):
            page_info = f", p.{chunk.page}" if chunk.page else ""
            preview = chunk.text[:120].replace("\n", " ")
            lines.append(f"  [{rank}] {chunk.source}{page_info}  (score={score:.3f})")
            lines.append(f"       {preview}…")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline.

    Parameters
    ----------
    embedder        : Embedder instance
    vector_store    : VectorStore instance (must be built/loaded before querying)
    retriever       : Retriever instance
    generator       : OpenAIGenerator or HFGenerator
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        retriever: Retriever,
        generator,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator

    # ── building the index ────────────────────────────────────────────────────

    def ingest(
        self,
        directory: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> None:
        """Ingest a directory of papers and build the vector index."""
        print(f"\n📂 Ingesting papers from: {directory}")
        chunks = ingest_directory(directory, chunk_size=chunk_size, overlap=overlap)

        print(f"\n🔢 Embedding {len(chunks)} chunks …")
        vectors = self.embedder.encode_chunks(chunks)

        self.vector_store.add(chunks, vectors)

    # ── querying ──────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Run a full RAG query.

        Parameters
        ----------
        question : Natural-language question.
        top_k    : Override retriever's default top_k.
        """
        if top_k is not None:
            self.retriever.top_k = top_k

        # 1. Embed query
        q_vec = self.embedder.encode_query(question)

        # 2. Retrieve relevant chunks
        retrieved = self.retriever.retrieve(q_vec)

        # 3. Format context for the LLM
        context = self.retriever.format_context(retrieved)

        # 4. Generate answer
        answer = self.generator.generate(query=question, context=context)

        return RAGResult(query=question, answer=answer, sources=retrieved)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        self.vector_store.save(path)

    def load_index(self, path: str) -> None:
        loaded = VectorStore.load(path)
        self.vector_store.index  = loaded.index
        self.vector_store.chunks = loaded.chunks


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    generator_backend: str = "hf",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    use_mmr: bool = True,
    cache_dir: Optional[str] = ".cache/embeddings",
    **generator_kwargs,
) -> RAGPipeline:
    """
    Convenience factory for building a RAGPipeline with sensible defaults.

    Parameters
    ----------
    generator_backend : "hf" (local, free) or "openai" (requires API key)
    embedding_model   : Any sentence-transformers model name.
    top_k             : Chunks to retrieve per query.
    use_mmr           : Enable MMR re-ranking.
    cache_dir         : Where to cache embeddings.
    **generator_kwargs: Passed to the generator (e.g. model="gpt-4").
    """
    embedder = Embedder(
        model_name=embedding_model,
        cache_dir=cache_dir,
    )
    vector_store = VectorStore(dim=embedder.embedding_dim)
    retriever    = Retriever(vector_store, top_k=top_k, use_mmr=use_mmr)
    generator    = build_generator(generator_backend, **generator_kwargs)

    return RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        generator=generator,
    )
