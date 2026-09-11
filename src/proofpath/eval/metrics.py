"""Per-stage metrics for the evaluation harness. Plain Python, no sklearn."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from proofpath.models import Label

LABELS: tuple[Label, ...] = (Label.SUPPORTED, Label.REFUTED, Label.NEI)

# (gold label, strongest non-NEI label proposed, its score)
Row = tuple[Label, Label, float]


def accuracy(gold: Sequence[Label], pred: Sequence[Label]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def macro_f1(gold: Sequence[Label], pred: Sequence[Label]) -> float:
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        scores.append(_f1(tp, fp, fn))
    return sum(scores) / len(scores)


def rationale_f1(gold: Sequence[frozenset[int]], pred: Sequence[frozenset[int]]) -> float:
    """Micro F1 over evidence sentence ids across all pairs."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return _f1(tp, fp, fn)


def recall_at_k(
    gold: Sequence[frozenset[int]], ranked: Sequence[Sequence[int]], *, k: int
) -> float:
    """Share of pairs with a rationale whose top-k retrieval contains any gold sentence."""
    considered = [(g, r) for g, r in zip(gold, ranked, strict=True) if g]
    if not considered:
        return 0.0
    hits = sum(bool(g & set(r[:k])) for g, r in considered)
    return hits / len(considered)


def _decide(row: Row, threshold: float) -> Label:
    _, label, score = row
    return label if score >= threshold else Label.NEI


@dataclass(frozen=True)
class SweepResult:
    threshold: float
    accuracy: float
    macro_f1: float


def sweep_decide(rows: Sequence[Row], *, grid: Sequence[float]) -> SweepResult:
    """Pick the decide threshold that maximises accuracy; ties go to the higher cut."""
    gold = [g for g, _, _ in rows]
    best: SweepResult | None = None
    for threshold in grid:
        pred = [_decide(r, threshold) for r in rows]
        candidate = SweepResult(threshold, accuracy(gold, pred), macro_f1(gold, pred))
        if best is None or (candidate.accuracy, candidate.threshold) >= (
            best.accuracy,
            best.threshold,
        ):
            best = candidate
    assert best is not None
    return best


def tier_cutpoints(
    rows: Sequence[Row],
    *,
    decide: float,
    high_precision: float,
    medium_precision: float,
) -> tuple[float, float]:
    """Lowest score at which precision of decided verdicts stays above each target.

    Walks the decided rows from the highest score down and records the score at
    which cumulative precision last held the target. Returns ``(high, medium)``.
    """
    decided = sorted((r for r in rows if r[2] >= decide), key=lambda r: -r[2])
    high = medium = 1.0
    correct = 0
    for i, (gold, label, score) in enumerate(decided, start=1):
        correct += gold == label
        precision = correct / i
        if precision >= high_precision:
            high = score
        if precision >= medium_precision:
            medium = score
    return high, max(medium, decide)
