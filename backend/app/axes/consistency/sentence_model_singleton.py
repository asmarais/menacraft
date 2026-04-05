"""Shared singleton for sentence-transformer embeddings used across Axis 2 modules."""

from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def get_sentence_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL
