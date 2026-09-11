from __future__ import annotations

import pytest

from proofpath.eval import metrics as m
from proofpath.models import Label

S, R, N = Label.SUPPORTED, Label.REFUTED, Label.NEI


def test_accuracy() -> None:
    assert m.accuracy([S, R, N, N], [S, N, N, S]) == pytest.approx(0.5)


def test_macro_f1_averages_over_all_three_labels_even_if_unpredicted() -> None:
    # S: p=1, r=1 -> 1.0 ; R: never predicted -> 0 ; N: p=1/2, r=1 -> 2/3
    assert m.macro_f1([S, R, N], [S, N, N]) == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)


def test_rationale_f1_is_micro_over_sentences() -> None:
    gold = [frozenset({1, 2}), frozenset(), frozenset({0})]
    pred = [frozenset({2}), frozenset(), frozenset({3})]
    # tp=1, fp=1, fn=2 -> p=0.5, r=1/3 -> f1=0.4
    assert m.rationale_f1(gold, pred) == pytest.approx(0.4)


def test_recall_at_k_counts_any_gold_sentence_in_top_hits() -> None:
    gold = [frozenset({1, 2}), frozenset(), frozenset({0})]
    ranked = [[2, 0, 1], [0], [1, 2]]
    assert m.recall_at_k(gold, ranked, k=1) == pytest.approx(0.5)
    assert m.recall_at_k(gold, ranked, k=3) == pytest.approx(0.5)


def test_sweep_picks_decide_threshold_with_best_accuracy() -> None:
    # (gold, strongest non-NEI label, its score)
    rows = [(S, S, 0.9), (N, S, 0.6), (R, R, 0.8), (N, R, 0.55)]
    best = m.sweep_decide(rows, grid=[0.5, 0.7])
    assert best.threshold == 0.7
    assert best.accuracy == pytest.approx(1.0)


def test_tier_cutpoints_come_from_precision_bands() -> None:
    rows = [(S, S, 0.95), (S, S, 0.9), (N, S, 0.8), (S, S, 0.75), (N, R, 0.6), (R, R, 0.55)]
    cuts = m.tier_cutpoints(rows, decide=0.5, high_precision=0.9, medium_precision=0.7)
    # precision >= 0.9 first holds at 0.9 (2/2); >= 0.7 holds down to 0.75 (3/4)
    assert cuts == (0.9, 0.75)
