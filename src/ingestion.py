"""
src/ingestion.py
────────────────
Parse PDF / plain-text research papers into overlapping text chunks
ready for embedding.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except ImportError:
    _PYMUPDF = False


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    text: str
    source: str                    # filename
    page: Optional[int] = None     # page number (PDF only)
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source={self.source!r}, text={preview!r}…)"


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, remove control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\n]+", " ", text)        # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # max two consecutive newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# PDF reader
# ──────────────────────────────────────────────────────────────────────────────

def _read_pdf(path: Path) -> List[tuple[int, str]]:
    """Return list of (page_number, page_text) from a PDF."""
    if not _PYMUPDF:
        raise ImportError("PyMuPDF not installed.  Run: pip install pymupdf")
    pages = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append((page_num, _clean_text(text)))
    return pages


def _read_text(path: Path) -> List[tuple[None, str]]:
    """Return the entire file as a single (None, text) pair."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, _clean_text(text))]


# ──────────────────────────────────────────────────────────────────────────────
# Chunking strategies
# ──────────────────────────────────────────────────────────────────────────────

def _sliding_window_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.
    Splits are made at sentence boundaries when possible.
    """
    # Split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap: drop sentences from the front until under overlap size
            while current and current_len > overlap:
                removed = current.pop(0)
                current_len -= len(removed) + 1
        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def ingest_file(
    path: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """
    Parse a single PDF or .txt file and return a list of Chunks.

    Parameters
    ----------
    path       : Path to the file.
    chunk_size : Target chunk size in characters.
    overlap    : Overlap between successive chunks in characters.

    Returns
    -------
    List of Chunk objects ready for embedding.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _read_pdf(path)
    elif suffix in {".txt", ".md"}:
        pages = _read_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    chunks: List[Chunk] = []
    chunk_id = 0

    for page_num, page_text in pages:
        for raw_chunk in _sliding_window_chunks(page_text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    text=raw_chunk,
                    source=path.name,
                    page=page_num,
                    chunk_id=chunk_id,
                    metadata={"chunk_size": chunk_size, "overlap": overlap},
                )
            )
            chunk_id += 1

    return chunks


def ingest_directory(
    directory: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md"),
) -> List[Chunk]:
    """
    Recursively ingest all matching files in a directory.

    Returns
    -------
    Flat list of all Chunks across all files.
    """
    directory = Path(directory)
    all_chunks: List[Chunk] = []

    files = [p for p in directory.rglob("*") if p.suffix.lower() in extensions]
    if not files:
        raise ValueError(f"No files with extensions {extensions} found in {directory}")

    for file_path in sorted(files):
        print(f"  📄 Ingesting {file_path.name} …", end=" ")
        file_chunks = ingest_file(file_path, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(file_chunks)
        print(f"{len(file_chunks)} chunks")

    print(f"\n✅ Total chunks: {len(all_chunks)} from {len(files)} file(s)")
    return all_chunks
