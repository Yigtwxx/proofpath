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
from typing import Literal, Protocol

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


Backend = Literal["sqlite-vec", "numpy"]


def _open_vec_connection(path: Path | None) -> sqlite3.Connection | None:
    """Connect and load sqlite-vec, or return None when this Python cannot.

    Some builds (notably python.org macOS installers) ship sqlite3 without
    ``enable_load_extension``; sqlite-vec then cannot be loaded at all.
    """
    conn = sqlite3.connect(str(path) if path else ":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.Error):
        conn.close()
        return None
    return conn


def _unit(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.asarray(arr / np.clip(norms, 1e-12, None), dtype=np.float32)


class PassageIndex:
    """Passages plus their vectors, searchable by cosine similarity.

    Backed by ``sqlite-vec`` when the interpreter can load extensions, otherwise
    by a numpy brute-force scan. Per-document corpora are small, so both are fast;
    the sqlite path matters for the persistent cache, not for speed.
    """

    def __init__(
        self,
        dim: int,
        path: Path | None = None,
        *,
        backend: Backend | None = None,
    ) -> None:
        self._dim = dim
        self._passages: list[Passage] = []
        self._vectors: list[np.ndarray] = []
        self._conn: sqlite3.Connection | None = None
        if backend != "numpy":
            self._conn = _open_vec_connection(path)
        if self._conn is not None:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS passages USING vec0(embedding float[{dim}])"
            )
        self.backend: Backend = "numpy" if self._conn is None else "sqlite-vec"

    def add(self, passages: Sequence[Passage], vectors: np.ndarray) -> None:
        if len(passages) != len(vectors):
            raise ValueError("one vector per passage is required")
        units = _unit(vectors)
        start = len(self._passages)
        if self._conn is not None:
            rows = [
                (start + i, sqlite_vec.serialize_float32(vec.tolist()))
                for i, vec in enumerate(units)
            ]
            self._conn.executemany("INSERT INTO passages(rowid, embedding) VALUES (?, ?)", rows)
        else:
            self._vectors.extend(units)
        self._passages.extend(passages)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if not self._passages:
            return []
        k = min(k, len(self._passages))
        query = _unit(vector)
        if self._conn is not None:
            # ``k = ?`` is the documented vec0 knn form and works on every SQLite
            # version; ``LIMIT ?`` is only recognised by newer builds.
            rows = self._conn.execute(
                "SELECT rowid, distance FROM passages WHERE embedding MATCH ? AND k = ? "
                "ORDER BY distance",
                (sqlite_vec.serialize_float32(query.tolist()), k),
            ).fetchall()
            # vec0 distance is L2; on unit vectors cosine = 1 - d^2 / 2.
            return [
                Hit(self._passages[rowid], similarity=float(1.0 - (dist * dist) / 2.0))
                for rowid, dist in rows
            ]
        matrix = np.stack(self._vectors)
        similarities = matrix @ query
        order = np.argsort(-similarities, kind="stable")[:k]
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
