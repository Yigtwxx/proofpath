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

from proofpath import numerics
from proofpath.entailment import LABEL_ORDER, Scorer
from proofpath.models import Label, Passage, Tier, Verdict
from proofpath.retrieval import Embedder, Hit, PassageIndex, rank_indexed


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


# Decimals a calibrated cut-point is carried with, everywhere it is written down: the
# eval harness prints them, ``cache.model_id`` keys on them. The NLI softmax saturates,
# so the cuts sit against 1.0 and fewer decimals would make 0.999330 and 0.999400 the
# same number — and make one recalibration serve the other's cached verdicts.
CUT_DECIMALS = 6

# Calibrated on SciFact dev, 2026-09-12, at k=1 with the numeric layer on:
# ``decide`` is the sweep's best-accuracy cut, ``high`` and ``medium`` are the lowest
# scores at which the verdicts at or above them stay 85% / 70% precise. The numbers
# are read off the run to six decimals rather than rounded to two, because the NLI
# softmax saturates and 0.99933 is a real cut where 1.00 would mean "never reached".
# Table and reasoning: docs/eval/2026-09-12-tiers.md.
DEFAULT_THRESHOLDS = Thresholds(decide=0.45, high=0.99933, medium=0.457948)

# What a report has to admit when the calibration leaves no reachable ``high`` band.
NO_HIGH_TIER = (
    "this model earns no high tier on SciFact dev, so medium is the strongest confidence shown"
)
# What it has to admit when ``medium`` sits on ``decide``: the ``low`` band between
# them is too thin to hold an asserted verdict, so ``low`` is in practice the tier of
# NEI alone. It names SciFact, not a split, because the two profiles were calibrated
# on different ones (dev for the default, train for the accurate one).
NO_LOW_TIER = (
    "this model has no real low tier on SciFact, so an asserted verdict is almost always "
    "medium or better"
)
# How close ``medium`` may sit to ``decide`` before ``low`` stops meaning anything.
# The same margin ``scripts/eval_scifact.py`` warns at, so the harness and the report
# agree on which calibrations have two tiers rather than three.
LOW_TIER_MARGIN = 0.01


def tier_note(thresholds: Thresholds) -> str:
    """What a run using these thresholds owes its reader about its tiers.

    Derived, never hand-set: a recalibration that loses a tier starts saying so on its
    own, and one that keeps it cannot leave a stale apology in every report. A ``high``
    cut of exactly 1.0 is ``eval.metrics.tier_cutpoints`` reporting that no score below
    1.0 held the precision target — only a rule-decided verdict (spec section 10, which
    scores exactly 1.0) could reach it, and that is not the model earning a tier. A
    ``medium`` cut within ``LOW_TIER_MARGIN`` of ``decide`` is the other end of the same
    story (docs/eval/2026-09-12-tiers.md: ``low`` held 1 of 135 asserted verdicts). Both
    can hold at once, and then both are said, joined with ``"; "``.
    """
    notes: list[str] = []
    if thresholds.high >= 1.0:
        notes.append(NO_HIGH_TIER)
    if thresholds.medium - thresholds.decide < LOW_TIER_MARGIN:
        notes.append(NO_LOW_TIER)
    return "; ".join(notes)


_SUPPORTED = LABEL_ORDER.index(Label.SUPPORTED)
_REFUTED = LABEL_ORDER.index(Label.REFUTED)


@dataclass(frozen=True)
class Closest:
    """The strongest SUPPORTED / REFUTED signal among the scored hits, decided or not.

    Never a verdict: ``aggregate`` turns it into one, and below ``decide`` that verdict
    is NEI and carries no passage (product rule 1). ``passage`` is ``None`` only when
    no hit scored above zero for either label, and ``label`` is then NEI.
    """

    label: Label
    score: float
    passage: Passage | None


def closest(hits: Sequence[Hit], probs: np.ndarray) -> Closest:
    """The passage the models came closest to deciding on, and the label and score
    they came closest with.

    Split out of ``aggregate`` so the AVeriTeC eval can show an NEI's passage to the
    judge (OPEN-ITEMS 20.13) by the very rule that picks a decided verdict's passage:
    a second copy of the loop could drift from this one, and the eval would then probe
    a passage the product would never have chosen. The first hit wins a tie, as it
    always did.
    """
    best = Closest(Label.NEI, 0.0, None)
    for hit, row in zip(hits, probs, strict=True):
        for label, column in ((Label.SUPPORTED, _SUPPORTED), (Label.REFUTED, _REFUTED)):
            value = float(row[column])
            if value > best.score:
                best = Closest(label, value, hit.passage)
    return best


def aggregate(
    hits: Sequence[Hit],
    probs: np.ndarray,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Let the strongest SUPPORTED / REFUTED signal among the scored hits decide."""
    best = closest(hits, probs)
    if best.score < thresholds.decide:
        return Verdict(Label.NEI, best.score, "low", None)
    return Verdict(best.label, best.score, thresholds.tier(best.score), best.passage)


def decide_indexed(
    claim: str,
    index: PassageIndex,
    embedder: Embedder,
    scorer: Scorer,
    *,
    k: int,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Decide one claim against an already-embedded source.

    The index is built once per source and reused for every claim that cites it, so
    a document's passages are embedded once per run, not once per claim.
    """
    verdict, _ = _decide_indexed(claim, index, embedder, scorer, k=k, thresholds=thresholds)
    return verdict


def _decide_indexed(
    claim: str,
    index: PassageIndex,
    embedder: Embedder,
    scorer: Scorer,
    *,
    k: int,
    thresholds: Thresholds,
) -> tuple[Verdict, Passage | None]:
    """``decide_indexed``'s verdict, and the passage the models came closest to
    deciding on (``closest``): the verdict's own when it asserts, the one an NEI
    dropped when it does not. The verdict is the product's, unchanged."""
    hits = rank_indexed(claim, index, embedder, k=k)
    if not hits:
        return Verdict(Label.NEI, 0.0, "low", None), None

    # Numbers first (spec section 10): a contradicting figure is decided by rule,
    # with both figures named, and never reaches the NLI model.
    numeric = numerics.check(claim, [hit.passage for hit in hits])
    if numeric is not None and numeric.mismatch:
        reason = (
            f"numeric mismatch: claim says {numeric.claim_text}, source says {numeric.source_text}"
        )
        verdict = Verdict(Label.REFUTED, 1.0, "high", numeric.passage, reason=reason)
        return verdict, numeric.passage

    # NLI convention: premise is the source passage, hypothesis is the claim.
    probs = scorer.score([(hit.passage.text, claim) for hit in hits])
    return aggregate(hits, probs, thresholds=thresholds), closest(hits, probs).passage


def decide(
    claim: str,
    passages: Sequence[Passage],
    embedder: Embedder,
    scorer: Scorer,
    *,
    k: int,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Decide one claim against the passages of one source."""
    if not passages:
        return Verdict(Label.NEI, 0.0, "low", None)
    index = PassageIndex(dim=embedder.dim)
    index.add(passages, embedder.embed([p.text for p in passages]))
    return decide_indexed(claim, index, embedder, scorer, k=k, thresholds=thresholds)


def decide_closest(
    claim: str,
    passages: Sequence[Passage],
    embedder: Embedder,
    scorer: Scorer,
    *,
    k: int,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[Verdict, Passage | None]:
    """``decide``'s verdict and the closest passage, from one embedding and one scoring.

    For ``scripts/eval_averitec.py --judge-nei`` (OPEN-ITEMS 20.13), which shows an NEI
    source's closest passage to the judge. The product never calls it: an NEI verdict
    there carries no passage, and nothing here changes that.
    """
    if not passages:
        return Verdict(Label.NEI, 0.0, "low", None), None
    index = PassageIndex(dim=embedder.dim)
    index.add(passages, embedder.embed([p.text for p in passages]))
    return _decide_indexed(claim, index, embedder, scorer, k=k, thresholds=thresholds)


# Alias for the Phase 1 harness; removed in Phase 9 when the LLM judge lands.
judge = decide
