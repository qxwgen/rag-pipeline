"""
scripts/evaluate.py
────────────────────
CLI: run the QA benchmark and report MRR, Recall@K, Faithfulness, F1.

Usage
-----
    python scripts/evaluate.py \
        --index_path  data/index \
        --qa_pairs    data/eval_qa.json \
        --output      results/eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_pipeline
from src.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    p.add_argument("--index_path",  required=True)
    p.add_argument("--qa_pairs",    required=True, help="Path to eval_qa.json")
    p.add_argument("--output",      default=None,  help="Optional: save results JSON")
    p.add_argument("--top_k",       type=int, default=5)
    p.add_argument("--generator",   default="hf", choices=["hf", "openai"])
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline = build_pipeline(
        generator_backend=args.generator,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    pipeline.load_index(args.index_path)

    evaluator = Evaluator(pipeline, ks=[1, 3, 5])
    qa_pairs  = Evaluator.load_qa_pairs(args.qa_pairs)

    print(f"\n📐 Evaluating on {len(qa_pairs)} QA pairs …\n")
    result = evaluator.evaluate(qa_pairs, verbose=True)
    print("\n" + str(result))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "recall_at_k":        result.recall_at_k,
                "mrr":                result.mrr,
                "mean_faithfulness":  result.mean_faithfulness,
                "mean_answer_f1":     result.mean_answer_f1,
                "n_queries":          result.n_queries,
                "per_query":          result.per_query,
            }, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
