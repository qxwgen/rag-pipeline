# ⚡ Ragnarok also know as Rag-pipeline 

**Stop asking LLMs to remember. Make them read.**

rag-pipeline or Ragnarok as you'll come across in the code is a **Retrieval-Augmented Generation (RAG) pipeline built from scratch** — no frameworks, no hidden abstractions.

Drop in a folder of PDFs or text files, ask a question in plain English, and rag-pipeline:

1. Finds the most relevant passages
2. Re-ranks them to reduce redundancy
3. Generates a grounded answer with citations

Every answer traces back to real source documents — **no hallucinations, no guessing.**

This project was built to deeply understand how modern RAG systems work under the hood: chunking strategies, vector search, re-ranking, and evaluation.

---

# 🚀 What It Does

* 📄 Parses PDFs and text files into **overlapping chunks** so context isn't lost
* 🧠 Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* ⚡ Stores and searches vectors with **FAISS** for fast similarity search
* 🔀 Re-ranks retrieved passages with **Maximal Marginal Relevance (MMR)** to reduce duplicates
* 🤖 Generates answers using either:

  * a **local HuggingFace model**, or
  * the **OpenAI API**
* 📊 Evaluates pipeline quality with:

  * **MRR**
  * **Recall@K**
  * **Answer F1**
  * **Faithfulness**

---

# 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ragnarok/rag-pipeline.git
cd rag-pipeline
pip install -r requirements.txt
```

---

# 📚 Index Your Documents

Place your PDFs or text files inside:

```
data/sample_papers/
```

Then run:

```bash
python scripts/ingest_papers.py \
  --input_dir data/sample_papers/ \
  --index_path data/index
```

This will:

* read the documents
* chunk them
* generate embeddings
* store them in a FAISS index

---

# ❓ Ask Questions

Run a single query:

```bash
python scripts/query.py \
  --index_path data/index \
  --query "What is the attention mechanism?"
```

Run in interactive mode:

```bash
python scripts/query.py \
  --index_path data/index \
  --interactive
```

---

# 🤖 Using OpenAI Instead of Local Models

Copy the environment template:

```bash
cp .env.example .env
```

Add your API key:

```
OPENAI_API_KEY=your_key_here
```

Then run:

```bash
python scripts/query.py \
  --index_path data/index \
  --generator openai \
  --query "your question here"
```

---

# 📊 Run Evaluation

Evaluate retrieval and answer quality:

```bash
python scripts/evaluate.py \
  --index_path data/index \
  --qa_pairs data/eval_qa.json
```

---

# 🧪 Run Tests

```bash
pytest tests/ -v
```

---

# 🧱 Tech Stack

* **Python**
* **FAISS**
* **Sentence Transformers**
* **HuggingFace Transformers**
* **PyMuPDF**
* **OpenAI API**


---

💡 *rag-pipeline was built to understand RAG systems from the ground up — the kind of knowledge you only get by building the whole pipeline yourself.*

---
#  References
Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Reimers & Gurevych (2019) — Sentence-BERT
Johnson et al. (2017) — Billion-scale similarity search with FAISS


---
