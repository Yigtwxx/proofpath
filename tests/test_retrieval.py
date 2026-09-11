from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest

from proofpath.models import Passage
from proofpath.retrieval import PassageIndex, rank, split_sentences


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


def test_split_sentences_keeps_abbreviations_and_decimals() -> None:
    text = "Results improved by 4.5% (p < 0.01). Dr. Smith et al. disagreed! Why? Because."
    assert split_sentences(text) == [
        "Results improved by 4.5% (p < 0.01).",
        "Dr. Smith et al. disagreed!",
        "Why?",
        "Because.",
    ]


def test_split_sentences_drops_blank_fragments() -> None:
    assert split_sentences("  One.   \n\n Two.  ") == ["One.", "Two."]


def test_index_returns_nearest_passage_first() -> None:
    passages = [
        Passage("cats purr", "d", 0),
        Passage("dogs bark", "d", 1),
        Passage("fish swim", "d", 2),
    ]
    embedder = FakeEmbedder()
    index = PassageIndex(dim=embedder.dim)
    index.add(passages, embedder.embed([p.text for p in passages]))
    hits = index.search(embedder.embed(["a cat purring"])[0], k=2)
    assert [hit.passage.text for hit in hits] == ["cats purr", "dogs bark"]
    assert hits[0].similarity > hits[1].similarity


def test_index_similarity_is_cosine_on_unit_vectors() -> None:
    index = PassageIndex(dim=3)
    index.add([Passage("cats purr", "d", 0)], np.array([[2.0, 0.0, 0.0]], dtype=np.float32))
    hit = index.search(np.array([1.0, 1.0, 0.0], dtype=np.float32), k=1)[0]
    assert hit.similarity == pytest.approx(1 / np.sqrt(2))


def test_index_rejects_wrong_dimension() -> None:
    index = PassageIndex(dim=3)
    with pytest.raises(ValueError, match="dim"):
        index.add([Passage("x", "d", 0)], np.zeros((1, 4), dtype=np.float32))


def test_index_can_be_rebuilt_from_stored_vectors() -> None:
    embedder = FakeEmbedder()
    passages = [Passage("cats purr", "d", 0), Passage("dogs bark", "d", 1)]
    vectors = embedder.embed([p.text for p in passages])
    index = PassageIndex.from_vectors(passages, vectors)
    assert index.search(embedder.embed(["a cat purring"])[0], k=1)[0].passage.text == "cats purr"


def test_rank_wraps_index_and_respects_k() -> None:
    passages = [Passage("dogs bark", "d", 0), Passage("cats purr", "d", 1)]
    ranked = rank("a cat purring", passages, FakeEmbedder(), k=1)
    assert [hit.passage.text for hit in ranked] == ["cats purr"]


def test_rank_with_more_k_than_passages_returns_all() -> None:
    passages = [Passage("dogs bark", "d", 0)]
    assert len(rank("cats purr", passages, FakeEmbedder(), k=5)) == 1
