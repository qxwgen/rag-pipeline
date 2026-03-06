"""
src/evaluator.py
────────────────
Evaluate RAG pipeline quality using a JSON QA benchmark.

Metrics
-------
  Recall@K        - fraction of queries where the gold chunk is in top-K
  MRR             - Mean Reciprocal Rank of the gold chunk
  Faithfulness    - rough lexical overlap of answer vs. retrieved context
  Answer F1       - token-level F1 between generated and gold answers
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.pipeline import RAGPipeline


# ──────────────────────────────────────────────────────────────────────────────
# Data model for a QA pair
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QAPair:
    """
    A single evaluation example.

    Fields
    ------
    query         : The question.
    gold_answer   : Reference answer (string).
    gold_source   : Filename of the document that contains the answer.
    gold_chunk_id : (Optional) exact chunk id, if known.
    """
    query: str
    gold_answer: str
    gold_source: str
    gold_chunk_id: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Token-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _f1(pred: str, gold: str) -> float:
    pred_toks  = Counter(_tokenize(pred))
    gold_toks  = Counter(_tokenize(gold))
    common     = sum((pred_toks & gold_toks).values())
    if common == 0:
        return 0.0
    precision  = common / sum(pred_toks.values())
    recall     = common / sum(gold_toks.values())
    return 2 * precision * recall / (precision + recall)


def _faithfulness(answer: str, context: str) -> float:
    """
    Proportion of answer tokens that appear in the retrieved context.
    A rough proxy for grounding (no hallucination).
    """
    answer_toks  = _tokenize(answer)
    context_toks = set(_tokenize(context))
    if not answer_toks:
        return 0.0
    return sum(1 for t in answer_toks if t in context_toks) / len(answer_toks)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    recall_at_k: Dict[int, float]
    mrr: float
    mean_faithfulness: float
    mean_answer_f1: float
    n_queries: int
    per_query: List[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "── Evaluation Results ──────────────────────",
            f"  Queries evaluated : {self.n_queries}",
            f"  MRR               : {self.mrr:.4f}",
        ]
        for k, r in sorted(self.recall_at_k.items()):
            lines.append(f"  Recall@{k:<2}          : {r:.4f}")
        lines += [
            f"  Faithfulness      : {self.mean_faithfulness:.4f}",
            f"  Answer F1         : {self.mean_answer_f1:.4f}",
            "─────────────────────────────────────────────",
        ]
        return "\n".join(lines)


class Evaluator:
    """
    Run quantitative evaluation of a RAGPipeline.

    Parameters
    ----------
    pipeline : A fully built RAGPipeline with a loaded index.
    ks       : List of K values for Recall@K.
    """

    def __init__(self, pipeline: RAGPipeline, ks: List[int] = [1, 3, 5]):
        self.pipeline = pipeline
        self.ks = ks

    # ── data loading ──────────────────────────────────────────────────────────

    @staticmethod
    def load_qa_pairs(path: str | Path) -> List[QAPair]:
        """
        Load QA pairs from a JSON file.

        Expected format:
        [
          {
            "query": "...",
            "gold_answer": "...",
            "gold_source": "paper.pdf"
          },
          ...
        ]
        """
        with open(path) as f:
            raw = json.load(f)
        return [QAPair(**item) for item in raw]

    # ── evaluation loop ───────────────────────────────────────────────────────

    def evaluate(self, qa_pairs: List[QAPair], verbose: bool = True) -> EvalResult:
        """Run the full evaluation and return an EvalResult."""
        max_k = max(self.ks)
        reciprocal_ranks: List[float] = []
        faithfulness_scores: List[float] = []
        answer_f1_scores: List[float] = []
        hits: Dict[int, List[int]] = {k: [] for k in self.ks}
        per_query = []

        for i, qa in enumerate(qa_pairs):
            if verbose:
                print(f"  [{i+1}/{len(qa_pairs)}] {qa.query[:60]}…")

            result = self.pipeline.query(qa.query, top_k=max_k)
            retrieved_sources = [chunk.source for chunk, _ in result.sources]

            # Recall@K and MRR
            rr = 0.0
            for rank, src in enumerate(retrieved_sources, start=1):
                if src == qa.gold_source:
                    rr = 1.0 / rank
                    break

            reciprocal_ranks.append(rr)
            for k in self.ks:
                hits[k].append(
                    int(qa.gold_source in retrieved_sources[:k])
                )

            # Faithfulness
            context_text = " ".join(c.text for c, _ in result.sources)
            faith = _faithfulness(result.answer, context_text)
            faithfulness_scores.append(faith)

            # Answer F1
            f1 = _f1(result.answer, qa.gold_answer)
            answer_f1_scores.append(f1)

            per_query.append({
                "query": qa.query,
                "answer": result.answer,
                "gold_answer": qa.gold_answer,
                "mrr": rr,
                "faithfulness": faith,
                "answer_f1": f1,
            })

        return EvalResult(
            recall_at_k={k: float(np.mean(hits[k])) for k in self.ks},
            mrr=float(np.mean(reciprocal_ranks)),
            mean_faithfulness=float(np.mean(faithfulness_scores)),
            mean_answer_f1=float(np.mean(answer_f1_scores)),
            n_queries=len(qa_pairs),
            per_query=per_query,
        )
