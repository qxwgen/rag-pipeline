"""
src/generator.py
────────────────
LLM-based answer generator.
Supports:
  - OpenAI API  (gpt-3.5-turbo / gpt-4)
  - HuggingFace local models  (e.g. flan-t5-base — no API key required)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from src.ingestion import Chunk


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise scientific assistant.
Answer questions using ONLY the provided context passages.
After your answer, list the source(s) you used as [1], [2], etc.
If the context does not contain enough information, say so clearly.
Never fabricate information."""

def _build_user_prompt(query: str, context: str) -> str:
    return f"""Context passages:
{context}

Question: {query}

Answer (cite sources with [N] notation):"""


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI generator
# ──────────────────────────────────────────────────────────────────────────────

class OpenAIGenerator:
    """
    Generate answers via OpenAI chat completions API.

    Requires OPENAI_API_KEY in environment or .env file.
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.  Check your .env file.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(query, context)},
            ],
        )
        return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace local generator (no API key needed)
# ──────────────────────────────────────────────────────────────────────────────

class HFGenerator:
    """
    Generate answers using a local HuggingFace seq2seq model.
    Default: google/flan-t5-base  (~250 MB, runs on CPU)

    For better quality use flan-t5-large or flan-t5-xl.
    """

    DEFAULT_MODEL = "google/flan-t5-base"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print(f"🤗 Loading generator model: {model_name}  (device={device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 256,
    ) -> str:
        import torch

        prompt = _build_user_prompt(query, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=4,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_generator(backend: str = "openai", **kwargs):
    """
    Factory function.

    Parameters
    ----------
    backend : "openai" | "hf"
    **kwargs: forwarded to the generator constructor
    """
    if backend == "openai":
        return OpenAIGenerator(**kwargs)
    elif backend == "hf":
        return HFGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator backend: {backend!r}.  Use 'openai' or 'hf'.")
