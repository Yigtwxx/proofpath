from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from proofpath.models import Passage
from proofpath.retrieval import (
    PassageIndex,
    rank,
    rank_indexed,
    sentence_spans,
    split_sentences,
)
from tests.fakes import FakeEmbedder


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


def test_sentence_spans_round_trip_to_the_split_strings() -> None:
    text = "Results improved by 4.5% (p < 0.01). Dr. Smith et al. disagreed! Why? Because."
    spans = sentence_spans(text)
    assert [text[start:end] for start, end in spans] == split_sentences(text)
    assert spans[0] == (0, len("Results improved by 4.5% (p < 0.01)."))
    assert all(0 <= start < end <= len(text) for start, end in spans)


def test_sentence_spans_are_exact_offsets_after_leading_whitespace() -> None:
    assert sentence_spans("  One.   \n\n Two.  ") == [(2, 6), (12, 16)]


def test_sentence_spans_of_blank_text_are_empty() -> None:
    assert sentence_spans("   \n  ") == []


def test_split_sentences_keeps_equation_and_figure_references() -> None:
    assert split_sentences("See Eq. 3 and Fig. 2. Then more.") == [
        "See Eq. 3 and Fig. 2.",
        "Then more.",
    ]


def test_split_sentences_keeps_author_initials_together() -> None:
    # An initial is never a sentence end, wherever it sits in the sentence.
    assert split_sentences("A study by J. Smith showed an effect. Then Y.") == [
        "A study by J. Smith showed an effect.",
        "Then Y.",
    ]
    assert split_sentences("J. Smith et al. (2020) reported it. Then Y.") == [
        "J. Smith et al. (2020) reported it.",
        "Then Y.",
    ]
    # The cost of that rule: a sentence really ending in a capital letter and a period
    # does not split -- "said X.", "we gave vitamin D.", "tested for hepatitis B." all
    # merge with the sentence after them. Merging two sentences is the safe direction;
    # cutting one in half, which the earlier look-behind did, is not.
    assert split_sentences("J. Smith et al. (2020) said X. Then Y.") == [
        "J. Smith et al. (2020) said X. Then Y."
    ]


def test_split_sentences_does_not_mistake_an_acronym_for_an_initial() -> None:
    assert split_sentences("The trial ran in the USA. Then it stopped.") == [
        "The trial ran in the USA.",
        "Then it stopped.",
    ]


def test_split_sentences_keeps_the_new_abbreviations() -> None:
    text = "Ref. 4 and Refs. 5 agree. No. 7 in Sec. 2 and Tab. 1 too. St. Louis, Jr. did it."
    assert split_sentences(text) == [
        "Ref. 4 and Refs. 5 agree.",
        "No. 7 in Sec. 2 and Tab. 1 too.",
        "St. Louis, Jr. did it.",
    ]
    assert split_sentences("We used approx. 5 ml. Then ca. 3 more. Jr. Smith agreed.") == [
        "We used approx. 5 ml.",
        "Then ca. 3 more.",
        "Jr. Smith agreed.",
    ]


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


def test_rank_indexed_matches_rank_without_re_embedding_the_passages() -> None:
    class CountingEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def embed(self, texts: Sequence[str]) -> np.ndarray:
            self.calls.append(tuple(texts))
            return super().embed(texts)

    passages = [Passage("dogs bark", "d", 0), Passage("cats purr", "d", 1)]
    embedder = CountingEmbedder()
    index = PassageIndex.from_vectors(passages, embedder.embed([p.text for p in passages]))
    embedder.calls.clear()
    ranked = rank_indexed("a cat purring", index, embedder, k=1)
    assert [hit.passage.text for hit in ranked] == ["cats purr"]
    assert embedder.calls == [("a cat purring",)]


def test_rank_indexed_on_an_empty_index_is_empty() -> None:
    assert rank_indexed("a cat purring", PassageIndex(dim=3), FakeEmbedder(), k=3) == []
