"""Claim -> verdict: retrieval, entailment, aggregation, confidence tier.

This is the part of the product that Phase 1 measures on SciFact. It is
deliberately small: rank passages, score the top ``k`` pairs, and let the strongest
signal decide. Numeric claims are handled before this stage (spec section 10) and
the LLM judge, when enabled, runs after it on low-tier verdicts only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from proofpath.entailment import LABEL_ORDER, Scorer
from proofpath.models import Label, Passage, Tier, Verdict
from proofpath.retrieval import Embedder, Hit, rank


@dataclass(frozen=True)
class Thresholds:
    """Cut-points calibrated on SciFact dev (spec section 14).

    ``decide``: below this, the strongest signal is not trusted and the verdict is
    ``NEI``. ``high`` / ``medium``: tier boundaries for the reported confidence.
    """

    decide: float
    high: float
    medium: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.decide <= self.medium <= self.high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= decide <= medium <= high <= 1")

    def tier(self, score: float) -> Tier:
        if score >= self.high:
            return "high"
        if score >= self.medium:
            return "medium"
        return "low"


# Reasonable starting point until the Phase 1 sweep replaces it.
DEFAULT_THRESHOLDS = Thresholds(decide=0.5, high=0.9, medium=0.7)

_SUPPORTED = LABEL_ORDER.index(Label.SUPPORTED)
_REFUTED = LABEL_ORDER.index(Label.REFUTED)


def aggregate(
    hits: Sequence[Hit],
    probs: np.ndarray,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Let the strongest SUPPORTED / REFUTED signal among the scored hits decide."""
    best_label = Label.NEI
    best_score = 0.0
    best_passage: Passage | None = None
    for hit, row in zip(hits, probs, strict=True):
        for label, column in ((Label.SUPPORTED, _SUPPORTED), (Label.REFUTED, _REFUTED)):
            value = float(row[column])
            if value > best_score:
                best_label, best_score, best_passage = label, value, hit.passage

    if best_score < thresholds.decide:
        return Verdict(Label.NEI, best_score, "low", None)
    return Verdict(best_label, best_score, thresholds.tier(best_score), best_passage)


def judge(
    claim: str,
    passages: Sequence[Passage],
    embedder: Embedder,
    scorer: Scorer,
    *,
    k: int,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Decide one claim against the passages of one source."""
    hits = rank(claim, passages, embedder, k=k)
    if not hits:
        return Verdict(Label.NEI, 0.0, "low", None)
    # NLI convention: premise is the source passage, hypothesis is the claim.
    probs = scorer.score([(hit.passage.text, claim) for hit in hits])
    return aggregate(hits, probs, thresholds=thresholds)
