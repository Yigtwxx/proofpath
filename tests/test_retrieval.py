from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest

from proofpath import retrieval
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


BACKENDS = ["numpy", "sqlite-vec"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_index_returns_nearest_passage_first(backend: str) -> None:
    passages = [
        Passage("cats purr", "d", 0),
        Passage("dogs bark", "d", 1),
        Passage("fish swim", "d", 2),
    ]
    embedder = FakeEmbedder()
    index = PassageIndex(dim=embedder.dim, backend=backend)  # type: ignore[arg-type]
    if index.backend != backend:
        pytest.skip(f"{backend} unavailable on this Python build")
    index.add(passages, embedder.embed([p.text for p in passages]))
    hits = index.search(embedder.embed(["a cat purring"])[0], k=2)
    assert [hit.passage.text for hit in hits] == ["cats purr", "dogs bark"]
    assert hits[0].similarity > hits[1].similarity


def test_index_falls_back_to_numpy_when_sqlite_vec_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval, "_open_vec_connection", lambda path: None)
    index = PassageIndex(dim=3)
    assert index.backend == "numpy"
    index.add([Passage("cats purr", "d", 0)], FakeEmbedder().embed(["cats purr"]))
    assert (
        index.search(FakeEmbedder().embed(["a cat purring"])[0], k=1)[0].passage.text == "cats purr"
    )


def test_both_backends_agree_on_similarity() -> None:
    embedder = FakeEmbedder()
    passages = [Passage("cats purr", "d", 0), Passage("dogs bark", "d", 1)]
    results = []
    for backend in BACKENDS:
        index = PassageIndex(dim=3, backend=backend)  # type: ignore[arg-type]
        if index.backend != backend:
            pytest.skip("sqlite-vec unavailable")
        index.add(passages, embedder.embed([p.text for p in passages]))
        hits = index.search(embedder.embed(["a cat purring"])[0], k=2)
        results.append([(h.passage.text, round(h.similarity, 4)) for h in hits])
    assert results[0] == results[1]


def test_rank_wraps_index_and_respects_k() -> None:
    passages = [Passage("dogs bark", "d", 0), Passage("cats purr", "d", 1)]
    ranked = rank("a cat purring", passages, FakeEmbedder(), k=1)
    assert [hit.passage.text for hit in ranked] == ["cats purr"]


def test_rank_with_more_k_than_passages_returns_all() -> None:
    passages = [Passage("dogs bark", "d", 0)]
    assert len(rank("cats purr", passages, FakeEmbedder(), k=5)) == 1
