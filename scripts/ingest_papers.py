"""
scripts/ingest_papers.py
─────────────────────────
CLI: parse a folder of PDFs/text files, embed them, and save a FAISS index.

Usage
-----
    python scripts/ingest_papers.py \
        --input_dir  data/sample_papers/ \
        --index_path data/index \
        --chunk_size 512 \
        --overlap    64
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Ingest research papers into RAG index")
    p.add_argument("--input_dir",  required=True,  help="Folder containing PDFs / .txt files")
    p.add_argument("--index_path", required=True,  help="Where to save the FAISS index")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--overlap",    type=int, default=64)
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--cache_dir", default=".cache/embeddings")
    p.add_argument("--generator", default="hf", choices=["hf", "openai"],
                   help="Generator backend (only used later in query.py)")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline = build_pipeline(
        generator_backend=args.generator,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
    )

    pipeline.ingest(
        directory=args.input_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    pipeline.save_index(args.index_path)
    print(f"\n🎉 Index saved to: {args.index_path}")


if __name__ == "__main__":
    main()
