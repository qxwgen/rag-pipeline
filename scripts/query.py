"""
scripts/query.py
────────────────
CLI: load a saved FAISS index and answer questions interactively.

Usage
-----
    # Single question
    python scripts/query.py \
        --index_path data/index \
        --query "What is the attention mechanism?"

    # Interactive REPL
    python scripts/query.py --index_path data/index --interactive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_pipeline
from src.vector_store import VectorStore


def parse_args():
    p = argparse.ArgumentParser(description="Query the RAG pipeline")
    p.add_argument("--index_path", required=True)
    p.add_argument("--query",      default=None, help="Single question to answer")
    p.add_argument("--interactive", action="store_true", help="Launch interactive REPL")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--no_mmr", action="store_true")
    p.add_argument("--generator", default="hf", choices=["hf", "openai"])
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def run_query(pipeline, question, top_k):
    result = pipeline.query(question, top_k=top_k)
    print("\n" + str(result))
    print()


def main():
    args = parse_args()

    pipeline = build_pipeline(
        generator_backend=args.generator,
        embedding_model=args.embedding_model,
        use_mmr=not args.no_mmr,
        top_k=args.top_k,
    )
    pipeline.load_index(args.index_path)

    if args.query:
        run_query(pipeline, args.query, args.top_k)

    elif args.interactive:
        print("🔍 RAG Research Assistant — type 'exit' to quit\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            run_query(pipeline, question, args.top_k)

    else:
        print("Provide --query or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
