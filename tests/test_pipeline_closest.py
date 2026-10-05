"""``pipeline.closest`` and ``decide_closest`` (OPEN-ITEMS 20.13, spec 2026-10-05 §2.1).

The closest passage is the one ``aggregate`` already picks and, for an NEI verdict,
then drops. These tests pin that the helper returns that very passage and that the
product's verdicts do not move: ``tests/test_pipeline.py`` is left as it was, so it
keeps saying the same thing about ``aggregate`` and ``decide`` that it always did.
"""

from __future__ import annotations

import numpy as np
import pytest

from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds, aggregate, closest, decide, decide_closest
from proofpath.retrieval import Hit
from tests.fakes import FakeEmbedder, TableScorer

PASSAGES = [
    Passage("cats purr", "d", 0),
    Passage("dogs bark", "d", 1),
    Passage("fish swim", "d", 2),
]
THRESHOLDS = Thresholds(decide=0.5, high=0.9, medium=0.7)


def _hits(*passages: Passage) -> list[Hit]:
    return [Hit(passage, 1.0) for passage in passages]


def test_closest_is_the_passage_with_the_strongest_supported_or_refuted_probability() -> None:
    probs = np.array([(0.30, 0.10, 0.60), (0.05, 0.42, 0.53)], dtype=np.float32)
    best = closest(_hits(PASSAGES[0], PASSAGES[1]), probs)
    assert best.label is Label.REFUTED
    assert best.score == pytest.approx(0.42)
    assert best.passage == PASSAGES[1]


def test_closest_is_the_passage_aggregate_keeps_when_it_decides() -> None:
    probs = np.array([(0.95, 0.02, 0.03), (0.2, 0.1, 0.7)], dtype=np.float32)
    hits = _hits(PASSAGES[0], PASSAGES[1])
    verdict = aggregate(hits, probs, thresholds=THRESHOLDS)
    best = closest(hits, probs)
    assert verdict.label is best.label is Label.SUPPORTED
    assert verdict.passage == best.passage == PASSAGES[0]
    assert verdict.score == pytest.approx(best.score)


def test_an_nei_verdict_still_drops_its_passage_while_closest_keeps_it() -> None:
    probs = np.array([(0.45, 0.1, 0.45), (0.3, 0.3, 0.4)], dtype=np.float32)
    hits = _hits(PASSAGES[0], PASSAGES[1])
    verdict = aggregate(hits, probs, thresholds=THRESHOLDS)
    assert verdict.label is Label.NEI and verdict.passage is None  # unchanged
    assert verdict.score == pytest.approx(0.45)
    assert closest(hits, probs).passage == PASSAGES[0]


def test_the_first_hit_wins_a_tie_as_aggregate_always_kept_it() -> None:
    probs = np.array([(0.4, 0.1, 0.5), (0.4, 0.1, 0.5)], dtype=np.float32)
    assert closest(_hits(PASSAGES[0], PASSAGES[1]), probs).passage == PASSAGES[0]


def test_no_signal_at_all_has_no_closest_passage() -> None:
    probs = np.array([(0.0, 0.0, 1.0)], dtype=np.float32)
    best = closest(_hits(PASSAGES[0]), probs)
    assert best.label is Label.NEI and best.passage is None and best.score == 0.0


def test_decide_closest_returns_decides_verdict_and_the_closest_passage() -> None:
    table = {"cats purr": (0.45, 0.1, 0.45), "dogs bark": (0.3, 0.3, 0.4)}
    verdict = decide(
        "a cat purring", PASSAGES, FakeEmbedder(), TableScorer(table), k=2, thresholds=THRESHOLDS
    )
    both = decide_closest(
        "a cat purring", PASSAGES, FakeEmbedder(), TableScorer(table), k=2, thresholds=THRESHOLDS
    )
    assert both == (verdict, PASSAGES[0])
    assert verdict.label is Label.NEI and verdict.passage is None


def test_decide_closest_with_no_passages_has_nothing_to_offer() -> None:
    verdict, passage = decide_closest(
        "anything", [], FakeEmbedder(), TableScorer({}), k=3, thresholds=THRESHOLDS
    )
    assert verdict.label is Label.NEI and passage is None
