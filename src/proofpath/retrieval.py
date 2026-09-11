"""Passage retrieval: chunk a source, embed it, rank chunks against a claim.

Embeddings run under ONNX through ``fastembed``; vectors live in ``sqlite-vec``.
Per-document corpora are a few hundred chunks, so an in-memory database is the
normal case and a file is only used for the persistent cache.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import sqlite_vec

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


class PassageIndex:
    """A ``sqlite-vec`` table over passages, keyed by insertion order."""

    def __init__(self, dim: int, path: Path | None = None) -> None:
        self._dim = dim
        self._conn = sqlite3.connect(str(path) if path else ":memory:")
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS passages USING vec0(embedding float[{dim}])"
        )
        self._passages: list[Passage] = []

    def add(self, passages: Sequence[Passage], vectors: np.ndarray) -> None:
        if len(passages) != len(vectors):
            raise ValueError("one vector per passage is required")
        start = len(self._passages)
        rows = [
            (start + i, sqlite_vec.serialize_float32(vec.astype(np.float32).tolist()))
            for i, vec in enumerate(vectors)
        ]
        self._conn.executemany("INSERT INTO passages(rowid, embedding) VALUES (?, ?)", rows)
        self._passages.extend(passages)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if not self._passages:
            return []
        query = sqlite_vec.serialize_float32(vector.astype(np.float32).tolist())
        rows = self._conn.execute(
            "SELECT rowid, distance FROM passages WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (query, min(k, len(self._passages))),
        ).fetchall()
        # vec0 distance is L2; on unit vectors cosine = 1 - d^2 / 2.
        return [
            Hit(self._passages[rowid], similarity=1.0 - (dist * dist) / 2.0) for rowid, dist in rows
        ]


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
