"""Offline tests for the bake-off's aggregation, calibration and decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from proofpath import pipeline
from proofpath.eval import bakeoff
from proofpath.eval.bakeoff_data import Item, SnapshotClaim, SnapshotSource
from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds
from proofpath.retrieval import Hit

# Rows are (SUPPORTED, REFUTED, NEI), the order of entailment.LABEL_ORDER.
P = [Passage(f"passage {i}", "doc", i) for i in range(3)]


def _probs(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_max_takes_the_loudest_assertion_over_the_hits() -> None:
    probs = _probs((0.6, 0.1, 0.3), (0.1, 0.8, 0.1))
    signal = bakeoff.aggregate_max(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.REFUTED, P[1])
    assert signal.score == pytest.approx(0.8)


def test_max_nei_ignores_an_assertion_its_own_hit_calls_neutral() -> None:
    probs = _probs((0.45, 0.05, 0.50), (0.40, 0.05, 0.30))
    signal = bakeoff.aggregate_max_nei(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.SUPPORTED, P[1])
    assert signal.score == pytest.approx(0.40)


def test_max_nei_proposes_nothing_when_every_hit_is_neutral() -> None:
    probs = _probs((0.2, 0.1, 0.7))
    assert bakeoff.aggregate_max_nei(P[:1], probs) == bakeoff.NOTHING


def test_margin_lets_a_contradicting_passage_cancel_a_supporting_one() -> None:
    probs = _probs((0.90, 0.05, 0.05), (0.05, 0.85, 0.10))
    signal = bakeoff.aggregate_margin(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.SUPPORTED, P[0])
    assert signal.score == pytest.approx(0.05)


def test_margin_with_an_exact_tie_proposes_nothing() -> None:
    probs = _probs((0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    assert bakeoff.aggregate_margin(P[:2], probs) == bakeoff.NOTHING


@pytest.mark.parametrize("name", bakeoff.AGGREGATIONS)
def test_no_hits_propose_nothing(name: str) -> None:
    assert bakeoff.AGGREGATORS[name]([], np.zeros((0, 3), np.float32)) == bakeoff.NOTHING


def test_signal_for_reads_only_the_first_k_hits() -> None:
    probs = _probs((0.2, 0.1, 0.7), (0.95, 0.0, 0.05))
    first = bakeoff.signal_for("a claim", P[:2], probs, k=1, aggregation="max")
    assert first.passage == P[0]
    both = bakeoff.signal_for("a claim", P[:2], probs, k=2, aggregation="max")
    assert both.passage == P[1]


def test_a_numeric_mismatch_decides_before_any_aggregation() -> None:
    passages = [Passage("Mortality fell by 12% in the treated group.", "doc", 0)]
    probs = _probs((0.99, 0.0, 0.01))
    signal = bakeoff.signal_for(
        "Mortality fell by 40% in the treated group.", passages, probs, k=1, aggregation="max"
    )
    assert signal == bakeoff.Signal(Label.REFUTED, 1.0, passages[0])


def test_decide_needs_the_cut_and_a_passage() -> None:
    cuts = Thresholds(decide=0.5, high=0.9, medium=0.7)
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.6, P[0]), cuts) is Label.SUPPORTED
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.4, P[0]), cuts) is Label.NEI
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.9, None), cuts) is Label.NEI


def test_wilson_interval_matches_known_values() -> None:
    low, high = bakeoff.wilson(0, 10)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.2775, abs=1e-4)
    low, high = bakeoff.wilson(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-4)
    assert high == pytest.approx(1.0)


def test_wilson_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError):
        bakeoff.wilson(0, 0)


@pytest.mark.parametrize("name", bakeoff.AGGREGATIONS)
def test_aggregators_refuse_a_passage_and_row_count_mismatch(name: str) -> None:
    probs = _probs((0.5, 0.2, 0.3), (0.1, 0.1, 0.8), (0.2, 0.7, 0.1))
    with pytest.raises(ValueError):
        bakeoff.AGGREGATORS[name](P[:2], probs)
    with pytest.raises(ValueError):
        bakeoff.AGGREGATORS[name](P[:3], probs[:2])


def test_margin_reports_a_refuted_signal_when_the_contradiction_is_louder() -> None:
    probs = _probs((0.10, 0.05, 0.85), (0.05, 0.80, 0.15))
    signal = bakeoff.aggregate_margin(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.REFUTED, P[1])
    assert signal.score == pytest.approx(0.70)


@pytest.mark.parametrize(
    "rows",
    [
        ((0.6, 0.1, 0.3), (0.1, 0.8, 0.1)),
        ((0.7, 0.2, 0.1), (0.3, 0.7, 0.0)),  # tie between SUPPORTED and REFUTED
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),  # nothing asserted
        ((0.4, 0.4, 0.2),),  # tie inside one row
    ],
)
def test_max_agrees_with_the_pipeline_aggregate(
    rows: tuple[tuple[float, float, float], ...],
) -> None:
    probs = _probs(*rows)
    passages = P[: len(rows)]
    hits = [Hit(passage, 0.5) for passage in passages]
    cuts = Thresholds(decide=0.0, high=1.0, medium=1.0)
    verdict = pipeline.aggregate(hits, probs, thresholds=cuts)
    signal = bakeoff.aggregate_max(passages, probs)
    assert (verdict.label, verdict.score, verdict.passage) == (
        signal.label,
        signal.score,
        signal.passage,
    )


def test_max_matches_the_pipeline_when_nothing_is_asserted() -> None:
    probs = _probs((0.0, 0.0, 1.0))
    verdict = pipeline.aggregate(
        [Hit(P[0], 0.5)], probs, thresholds=Thresholds(decide=0.0, high=1.0, medium=1.0)
    )
    assert verdict.label is Label.NEI and verdict.passage is None
    assert bakeoff.aggregate_max(P[:1], probs) == bakeoff.NOTHING


def test_decide_at_the_cut_asserts_and_keeps_a_refuted_signal() -> None:
    cuts = Thresholds(decide=0.5, high=0.9, medium=0.7)
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.5, P[0]), cuts) is Label.SUPPORTED
    assert bakeoff.decide(bakeoff.Signal(Label.REFUTED, 0.5, P[0]), cuts) is Label.REFUTED
    assert bakeoff.decide(bakeoff.Signal(Label.REFUTED, 0.49, P[0]), cuts) is Label.NEI
    assert bakeoff.decide(bakeoff.NOTHING, cuts) is Label.NEI


def _scored(
    key: str,
    dataset: str,
    gold: str,
    row: tuple[float, float, float],
    *,
    group: str | None = None,
    rationale: frozenset[int] = frozenset({0}),
) -> bakeoff.Scored:
    item = Item(key, dataset, group or key, "a claim", gold, (Passage("p", key, 0),), rationale)
    return bakeoff.Scored(item, _probs(row))


def test_calibration_refuses_anything_but_scifact_train() -> None:
    dev = [_scored("d", "scifact-dev", "SUPPORTED", (0.9, 0.0, 0.1))]
    with pytest.raises(ValueError, match="scifact-train"):
        bakeoff.calibration_rows(dev, k=1, aggregation="max")


def test_evaluation_refuses_the_calibration_split() -> None:
    train = [_scored("t", "scifact-train", "SUPPORTED", (0.9, 0.0, 0.1))]
    cuts = Thresholds(decide=0.5, high=0.9, medium=0.7)
    with pytest.raises(ValueError, match="scifact-dev"):
        bakeoff.evaluate_scifact(train, cuts, k=1, aggregation="max")


def test_calibrate_picks_a_cut_that_separates_right_from_wrong() -> None:
    rows = [(Label.SUPPORTED, Label.SUPPORTED, 0.9)] * 5 + [(Label.NEI, Label.SUPPORTED, 0.3)] * 5
    cuts = bakeoff.calibrate(rows)
    assert 0.3 < cuts.decide <= 0.9
    assert cuts.decide <= cuts.medium <= cuts.high


def test_tier_stats_hide_precision_below_the_minimum_n() -> None:
    assert bakeoff.TierStat("high", 19, 19).precision is None
    assert bakeoff.TierStat("high", 19, 19).interval is None
    assert bakeoff.TierStat("high", 20, 15).precision == pytest.approx(0.75)


def test_evaluate_scifact_counts_verdicts_tiers_and_rationale() -> None:
    cuts = Thresholds(decide=0.5, high=0.95, medium=0.7)
    scored = [
        _scored("a", "scifact-dev", "SUPPORTED", (0.97, 0.01, 0.02)),  # high, right
        _scored("b", "scifact-dev", "REFUTED", (0.80, 0.10, 0.10)),  # medium, wrong
        _scored("c", "scifact-dev", "NEI", (0.30, 0.10, 0.60)),  # below decide
    ]
    result = bakeoff.evaluate_scifact(scored, cuts, k=1, aggregation="max")
    assert result.n == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert {t.tier: (t.n, t.correct) for t in result.tiers} == {
        "high": (1, 1),
        "medium": (1, 0),
        "low": (0, 0),
    }
    assert result.asserted_without_passage == 0
    assert 0.0 < result.rationale_f1 <= 1.0


def test_evaluate_averitec_keeps_the_strongest_source_and_both_denominators() -> None:
    cuts = Thresholds(decide=0.5, high=0.95, medium=0.7)
    claims = [
        SnapshotClaim(1, "c1", "Refuted", (SnapshotSource("u1", "ok", "t"),), 0),
        SnapshotClaim(2, "c2", "Supported", (SnapshotSource("u2", "UNVERIFIED (x)", ""),), 0),
        SnapshotClaim(3, "c3", "Conflicting Evidence/Cherrypicking", (), 0),
    ]
    scored = [
        _scored("averitec:1:0", "averitec", "Refuted", (0.6, 0.1, 0.3), group="averitec:1"),
        _scored("averitec:1:1", "averitec", "Refuted", (0.0, 0.9, 0.1), group="averitec:1"),
    ]
    result = bakeoff.evaluate_averitec(claims, scored, cuts, k=1, aggregation="max")
    assert result.n == 3
    assert result.counted == 2  # the Conflicting claim has no 3-way label
    assert result.accuracy_all == pytest.approx(0.5)  # claim 1 right, claim 2 unread → NEI
    assert result.readable == 1
    assert result.accuracy_readable == pytest.approx(1.0)
    assert result.majority_baseline == pytest.approx(0.5)
