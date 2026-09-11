from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds, judge
from tests.test_retrieval import FakeEmbedder


class TableScorer:
    """Scores looked up by passage text; order is (SUPPORTED, REFUTED, NEI)."""

    name = "table"

    def __init__(self, table: dict[str, tuple[float, float, float]]) -> None:
        self._table = table
        self.seen: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.seen.extend(pairs)
        return np.array([self._table[premise] for premise, _ in pairs], dtype=np.float32)


PASSAGES = [
    Passage("cats purr", "d", 0),
    Passage("dogs bark", "d", 1),
    Passage("fish swim", "d", 2),
]
THRESHOLDS = Thresholds(decide=0.5, high=0.9, medium=0.7)


def test_supported_verdict_carries_the_best_supporting_passage() -> None:
    scorer = TableScorer({"cats purr": (0.95, 0.02, 0.03), "dogs bark": (0.2, 0.1, 0.7)})
    verdict = judge("a cat purring", PASSAGES, FakeEmbedder(), scorer, k=2, thresholds=THRESHOLDS)
    assert verdict.label is Label.SUPPORTED
    assert verdict.passage is not None and verdict.passage.text == "cats purr"
    assert verdict.score == pytest.approx(0.95)
    assert verdict.tier == "high"


def test_premise_is_the_passage_and_hypothesis_is_the_claim() -> None:
    scorer = TableScorer({"cats purr": (0.9, 0.05, 0.05)})
    judge("a cat purring", PASSAGES[:1], FakeEmbedder(), scorer, k=1, thresholds=THRESHOLDS)
    assert scorer.seen == [("cats purr", "a cat purring")]


def test_refuted_wins_when_contradiction_is_strongest_anywhere() -> None:
    scorer = TableScorer({"cats purr": (0.6, 0.3, 0.1), "dogs bark": (0.05, 0.85, 0.1)})
    verdict = judge("a cat purring", PASSAGES, FakeEmbedder(), scorer, k=2, thresholds=THRESHOLDS)
    assert verdict.label is Label.REFUTED
    assert verdict.passage is not None and verdict.passage.text == "dogs bark"
    assert verdict.tier == "medium"


def test_below_decide_threshold_is_nei_without_passage() -> None:
    scorer = TableScorer({"cats purr": (0.45, 0.1, 0.45), "dogs bark": (0.3, 0.3, 0.4)})
    verdict = judge("a cat purring", PASSAGES, FakeEmbedder(), scorer, k=2, thresholds=THRESHOLDS)
    assert verdict.label is Label.NEI
    assert verdict.passage is None
    assert verdict.score == pytest.approx(0.45)
    assert verdict.tier == "low"


def test_no_passages_is_nei() -> None:
    verdict = judge("anything", [], FakeEmbedder(), TableScorer({}), k=3, thresholds=THRESHOLDS)
    assert verdict.label is Label.NEI
    assert verdict.passage is None


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        Thresholds(decide=0.8, high=0.7, medium=0.9)
