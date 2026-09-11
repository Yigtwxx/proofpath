"""Deterministic stand-ins for the two models, shared by every test that needs one.

Nothing in here loads a model or touches the network. An embedder whose vectors are
fixed (or read off the words of a sentence) and a scorer that looks its probabilities
up in a table make retrieval and entailment predictable, so a test asserts about the
pipeline rather than about what a real model happened to think that day.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np


class FakeEmbedder:
    """Three fixed axes so nearest-neighbour results are predictable."""

    name = "fake"
    dim = 3

    _table: ClassVar[dict[str, tuple[float, float, float]]] = {
        "cats purr": (1.0, 0.0, 0.0),
        "dogs bark": (0.0, 1.0, 0.0),
        "fish swim": (0.0, 0.0, 1.0),
        "a cat purring": (0.9, 0.1, 0.0),
    }

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._table[t] for t in texts], dtype=np.float32)


class WordEmbedder:
    """Hashed bag of words: any sentence embeds, and word overlap decides the ranking.

    ``FakeEmbedder`` needs every string in its table, which a whole-document test
    cannot supply. This one embeds anything, deterministically (``zlib.crc32`` is
    stable across runs and platforms, unlike ``hash``), so a claim retrieves the
    source sentence it shares its words with. ``calls`` counts the batches, which is
    how a test sees that a cache hit did no work.
    """

    name = "words"
    dim = 64

    _WORD = re.compile(r"[a-z0-9]+")

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        batch = list(texts)
        self.calls.append(batch)
        rows = np.zeros((len(batch), self.dim), dtype=np.float32)
        for row, text in zip(rows, batch, strict=True):
            for word in self._WORD.findall(text.lower()):
                row[zlib.crc32(word.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return np.asarray(rows / np.clip(norms, 1e-12, None), dtype=np.float32)


class TableScorer:
    """Scores looked up by passage text; order is (SUPPORTED, REFUTED, NEI).

    A premise the table does not know raises ``KeyError`` on purpose: a test whose
    retrieval picked the wrong passage must fail loudly rather than score a sentence
    nobody meant to score.
    """

    name = "table"

    def __init__(
        self,
        table: dict[str, tuple[float, float, float]],
        *,
        on_score: Callable[[], None] | None = None,
    ) -> None:
        self._table = table
        self.seen: list[tuple[str, str]] = []
        self.on_score = on_score

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.seen.extend(pairs)
        if self.on_score is not None:
            self.on_score()
        return np.array([self._table[premise] for premise, _ in pairs], dtype=np.float32)


class NeverScorer:
    name = "never"

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        raise AssertionError("scorer must not be called on a numeric mismatch")
