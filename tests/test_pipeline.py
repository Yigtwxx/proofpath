from __future__ import annotations

from collections.abc import Sequence

import pytest

from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds, decide, decide_indexed, judge
from proofpath.retrieval import PassageIndex
from tests.fakes import FakeEmbedder, NeverScorer, TableScorer

PASSAGES = [
    Passage("cats purr", "d", 0),
    Passage("dogs bark", "d", 1),
    Passage("fish swim", "d", 2),
]
THRESHOLDS = Thresholds(decide=0.5, high=0.9, medium=0.7)


def test_supported_verdict_carries_the_best_supporting_passage() -> None:
    scorer = TableScorer({"cats purr": (0.95, 0.02, 0.03), "dogs bark": (0.2, 0.1, 0.7)})
    verdict = decide("a cat purring", PASSAGES, FakeEmbedder(), scorer, k=2, thresholds=THRESHOLDS)
    assert verdict.label is Label.SUPPORTED
    assert verdict.passage is not None and verdict.passage.text == "cats purr"
    assert verdict.score == pytest.approx(0.95)
    assert verdict.tier == "high"


def test_premise_is_the_passage_and_hypothesis_is_the_claim() -> None:
    scorer = TableScorer({"cats purr": (0.9, 0.05, 0.05)})
    decide("a cat purring", PASSAGES[:1], FakeEmbedder(), scorer, k=1, thresholds=THRESHOLDS)
    assert scorer.seen == [("cats purr", "a cat purring")]


def test_refuted_wins_when_contradiction_is_strongest_anywhere() -> None:
    scorer = TableScorer({"cats purr": (0.6, 0.3, 0.1), "dogs bark": (0.05, 0.85, 0.1)})
    verdict = decide("a cat purring", PASSAGES, FakeEmbedder(), scorer, k=2, thresholds=THRESHOLDS)
    assert verdict.label is Label.REFUTED
    assert verdict.passage is not None and verdict.passage.text == "dogs bark"
    assert verdict.tier == "medium"


def test_below_decide_threshold_is_nei_without_passage() -> None:
    scorer = TableScorer({"cats purr": (0.45, 0.1, 0.45), "dogs bark": (0.3, 0.3, 0.4)})
    verdict = decide("a cat purring", PASSAGES, FakeEmbedder(), scorer, k=2, thresholds=THRESHOLDS)
    assert verdict.label is Label.NEI
    assert verdict.passage is None
    assert verdict.score == pytest.approx(0.45)
    assert verdict.tier == "low"


def test_no_passages_is_nei() -> None:
    verdict = decide("anything", [], FakeEmbedder(), TableScorer({}), k=3, thresholds=THRESHOLDS)
    assert verdict.label is Label.NEI
    assert verdict.passage is None


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        Thresholds(decide=0.8, high=0.7, medium=0.9)


def _embedder_knowing(*texts: str) -> FakeEmbedder:
    embedder = FakeEmbedder()
    known = dict.fromkeys(texts, (1.0, 0.0, 0.0))
    embedder._table = {**embedder._table, **known}  # type: ignore[misc]
    return embedder


def test_numeric_mismatch_refutes_before_nli_and_names_both_figures() -> None:
    source = [Passage("we observed a 4-8% improvement in throughput", "s", 0)]
    claim = "the method yields a 40% speedup"
    embedder = _embedder_knowing(source[0].text, claim)
    verdict = decide(claim, source, embedder, NeverScorer(), k=1, thresholds=THRESHOLDS)
    assert verdict.label is Label.REFUTED
    assert verdict.passage is source[0]
    assert verdict.tier == "high" and verdict.score == 1.0
    assert verdict.reason == "numeric mismatch: claim says 40%, source says 4-8%"


def test_consistent_numbers_still_go_to_nli() -> None:
    source = [Passage("we observed a 4-8% improvement in throughput", "s", 0)]
    claim = "the method yields a 6% speedup"
    embedder = _embedder_knowing(source[0].text, claim)
    scorer = TableScorer({source[0].text: (0.2, 0.1, 0.7)})
    verdict = decide(claim, source, embedder, scorer, k=1, thresholds=THRESHOLDS)
    assert scorer.seen, "NLI must run when numbers agree"
    assert verdict.label is Label.NEI
    assert verdict.reason == ""


def _index_of(passages: Sequence[Passage], embedder: FakeEmbedder) -> PassageIndex:
    return PassageIndex.from_vectors(passages, embedder.embed([p.text for p in passages]))


def test_decide_indexed_matches_decide_on_the_same_inputs() -> None:
    table = {"cats purr": (0.95, 0.02, 0.03), "dogs bark": (0.2, 0.1, 0.7)}
    embedder = FakeEmbedder()
    direct = decide(
        "a cat purring", PASSAGES, embedder, TableScorer(table), k=2, thresholds=THRESHOLDS
    )
    indexed = decide_indexed(
        "a cat purring",
        _index_of(PASSAGES, embedder),
        embedder,
        TableScorer(table),
        k=2,
        thresholds=THRESHOLDS,
    )
    assert indexed == direct


def test_decide_indexed_refuses_a_numeric_mismatch_before_nli() -> None:
    source = [Passage("we observed a 4-8% improvement in throughput", "s", 0)]
    claim = "the method yields a 40% speedup"
    embedder = _embedder_knowing(source[0].text, claim)
    verdict = decide_indexed(
        claim, _index_of(source, embedder), embedder, NeverScorer(), k=1, thresholds=THRESHOLDS
    )
    assert verdict.label is Label.REFUTED
    assert verdict.reason == "numeric mismatch: claim says 40%, source says 4-8%"


def test_decide_indexed_on_an_empty_index_is_nei() -> None:
    verdict = decide_indexed(
        "anything", PassageIndex(dim=3), FakeEmbedder(), NeverScorer(), k=3, thresholds=THRESHOLDS
    )
    assert verdict.label is Label.NEI
    assert verdict.passage is None
    assert verdict.score == 0.0 and verdict.tier == "low"


def test_judge_is_still_importable_as_an_alias_of_decide() -> None:
    assert judge is decide
