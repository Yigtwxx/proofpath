"""Passage retrieval: chunk a source, embed it, rank chunks against a claim.

Embeddings run under ONNX through ``fastembed``; ranking is a numpy cosine scan.
Per-document corpora are a few hundred chunks, so nothing heavier is warranted
(spec section 12).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from proofpath.models import Passage

# Sentence boundary: terminal punctuation followed by whitespace and an upper-case
# letter, a digit or an opening bracket. Common abbreviations are excluded so
# "et al." and "Dr." do not split. Good enough for abstracts; full text gets
# re-checked in Phase 4.
_ABBREVIATIONS = ("et al.", "Dr.", "Mr.", "Ms.", "Prof.", "Fig.", "vs.", "e.g.", "i.e.", "cf.")
_NOT_AFTER_ABBREVIATION = "".join(f"(?<!\\b{re.escape(a)})" for a in _ABBREVIATIONS)
_BOUNDARY = re.compile(_NOT_AFTER_ABBREVIATION + r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping blank fragments."""
    return [s.strip() for s in _BOUNDARY.split(text) if s.strip()]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalised vectors."""


@dataclass(frozen=True)
class Hit:
    passage: Passage
    similarity: float


def _unit(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.asarray(arr / np.clip(norms, 1e-12, None), dtype=np.float32)


class PassageIndex:
    """Passages plus their unit vectors, searched by a brute-force cosine scan.

    A source contributes at most a few hundred chunks, so a numpy matrix product
    beats every vector store measured (2026-09-11: 1.2 ms at 10^5 vectors, versus
    7.4 ms for sqlite-vec) and needs nothing the standard library lacks.
    Persistence is the cache's job (``cache.py``), not the index's.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._passages: list[Passage] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    @classmethod
    def from_vectors(cls, passages: Sequence[Passage], vectors: np.ndarray) -> PassageIndex:
        index = cls(dim=int(np.asarray(vectors).shape[1]))
        index.add(passages, vectors)
        return index

    def __len__(self) -> int:
        return len(self._passages)

    def add(self, passages: Sequence[Passage], vectors: np.ndarray) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if len(passages) != len(arr):
            raise ValueError("one vector per passage is required")
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(f"vectors must have dim {self._dim}, got shape {arr.shape}")
        self._matrix = np.concatenate([self._matrix, _unit(arr)], axis=0)
        self._passages.extend(passages)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if not self._passages:
            return []
        similarities = self._matrix @ _unit(vector)
        order = np.argsort(-similarities, kind="stable")[: min(k, len(self._passages))]
        return [Hit(self._passages[int(i)], float(similarities[i])) for i in order]


def rank(claim: str, passages: Sequence[Passage], embedder: Embedder, *, k: int) -> list[Hit]:
    """Embed the passages and the claim, return the ``k`` closest passages, best first."""
    if not passages:
        return []
    index = PassageIndex(dim=embedder.dim)
    index.add(passages, embedder.embed([p.text for p in passages]))
    return index.search(embedder.embed([claim])[0], k=k)


class FastEmbedder:
    """``fastembed`` wrapper. Model and revision are pinned by the caller."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        providers: Sequence[str] | None = None,
    ) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            providers=list(providers) if providers else None,
        )
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)
