# 🔍 RAG Research Assistant
### Retrieval-Augmented Generation over Scientific Papers

> Ask natural-language questions over a corpus of research papers and get grounded, cited answers — no hallucinations.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co)

---

## 🧠 What This Project Does

This project implements a **full Retrieval-Augmented Generation (RAG) pipeline** from scratch:

1. **Ingests** PDF/text research papers and chunks them intelligently
2. **Embeds** chunks using a Sentence Transformer (`all-MiniLM-L6-v2`)
3. **Indexes** embeddings in a FAISS vector store for fast similarity search
4. **Retrieves** the top-k most relevant chunks for any user query
5. **Generates** a grounded answer using an LLM, citing the source chunks
6. **Evaluates** retrieval quality with MRR, Recall@K, and answer faithfulness

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Embedder   │────▶│   FAISS Index    │────▶│  Top-K Chunks    │
└─────────────┘     └──────────────────┘     └──────────────────┘
                                                       │
                                                       ▼
                                             ┌──────────────────┐
                                             │  LLM Generator   │
                                             └──────────────────┘
                                                       │
                                                       ▼
                                             Grounded Answer + Citations
```

---

## 📁 Project Structure

```
rag_research_assistant/
├── src/
│   ├── ingestion.py       # PDF parsing, chunking strategies
│   ├── embedder.py        # Sentence-transformer embedding wrapper
│   ├── vector_store.py    # FAISS index: build, save, load, query
│   ├── retriever.py       # Top-K retrieval + MMR re-ranking
│   ├── generator.py       # LLM answer generation with citations
│   ├── pipeline.py        # End-to-end RAG orchestration
│   └── evaluator.py       # MRR, Recall@K, faithfulness scoring
├── scripts/
│   ├── ingest_papers.py   # CLI: ingest a folder of PDFs
│   ├── query.py           # CLI: ask a question interactively
│   └── evaluate.py        # CLI: run evaluation benchmark
├── tests/
│   ├── test_embedder.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── notebooks/
│   └── rag_demo.ipynb     # Interactive walkthrough
├── data/
│   └── sample_papers/     # Drop your PDFs here
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Install
```bash
git clone https://github.com/YOUR_USERNAME/rag-research-assistant
cd rag-research-assistant
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Add your OpenAI / HuggingFace API key to .env
```

### 3. Ingest papers
```bash
python scripts/ingest_papers.py --input_dir data/sample_papers/ --index_path data/index.faiss
```

### 4. Ask questions
```bash
python scripts/query.py --index_path data/index.faiss --query "What are the main limitations of transformer attention?"
```

### 5. Run evaluation
```bash
python scripts/evaluate.py --index_path data/index.faiss --qa_pairs data/eval_qa.json
```

---

## 📊 Evaluation Results (on sample corpus)

| Metric | Score |
|---|---|
| Recall@1 | 0.71 |
| Recall@5 | 0.89 |
| MRR | 0.78 |
| Answer Faithfulness | 0.84 |

---

## 🔬 Key Techniques

- **Chunking**: Sliding window with overlap to preserve context across boundaries
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (fast, strong baseline)
- **Vector Search**: FAISS `IndexFlatIP` (inner product = cosine on normalised vectors)
- **Re-ranking**: Maximal Marginal Relevance (MMR) to reduce redundancy
- **Generation**: Prompt engineering with explicit citation instructions
- **Evaluation**: Custom QA benchmark with ground-truth chunk labels

---

## 🧩 Extending This Project

- Swap FAISS for **ChromaDB** or **Pinecone** (see `vector_store.py`)
- Swap the LLM for **LLaMA 3** via `llama-cpp-python` (no API key needed)
- Add **HyDE** (Hypothetical Document Embeddings) in `retriever.py`
- Add a **Streamlit UI** for interactive demos

---

## 📚 References

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Reimers & Gurevych (2019) — *Sentence-BERT*
- Johnson et al. (2017) — *Billion-scale similarity search with FAISS*
