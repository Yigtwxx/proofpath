"""NLI bake-off: aggregation variants, calibration, evaluation and the decision rule.

Pure functions over stored NLI probabilities; nothing here touches the network or a
model (docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md). A variant moves into
``pipeline.py`` only if the bake-off picks it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from proofpath import numerics
from proofpath.entailment import LABEL_ORDER
from proofpath.eval import averitec, metrics
from proofpath.eval.bakeoff_data import Item, SnapshotClaim
from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds

_S = LABEL_ORDER.index(Label.SUPPORTED)
_R = LABEL_ORDER.index(Label.REFUTED)
_N = LABEL_ORDER.index(Label.NEI)

# Simplest first: this order is also the tie order of the decision rule.
AGGREGATIONS: tuple[str, ...] = ("max", "max_nei", "margin")
KS: tuple[int, ...] = (1, 2, 3)
# Passages retrieved and scored per item; every k is a prefix of these.
TOP_N = 5


@dataclass(frozen=True)
class Signal:
    """The strongest assertion a combination proposes for one item, before ``decide``.

    ``label`` is SUPPORTED or REFUTED, or NEI when the aggregation proposes nothing;
    ``passage`` is what the assertion rests on.
    """

    label: Label
    score: float
    passage: Passage | None


NOTHING = Signal(Label.NEI, 0.0, None)

Aggregator = Callable[[Sequence[Passage], np.ndarray], Signal]


def aggregate_max(passages: Sequence[Passage], probs: np.ndarray) -> Signal:
    """Today's ``pipeline.aggregate``: the loudest SUPPORTED or REFUTED over the hits."""
    best = NOTHING
    for passage, row in zip(passages, probs, strict=True):
        for label, column in ((Label.SUPPORTED, _S), (Label.REFUTED, _R)):
            value = float(row[column])
            if value > best.score:
                best = Signal(label, value, passage)
    return best


def aggregate_max_nei(passages: Sequence[Passage], probs: np.ndarray) -> Signal:
    """As ``max``, but a hit only counts when its assertion beats its own p(NEI)."""
    best = NOTHING
    for passage, row in zip(passages, probs, strict=True):
        neutral = float(row[_N])
        for label, column in ((Label.SUPPORTED, _S), (Label.REFUTED, _R)):
            value = float(row[column])
            if value > neutral and value > best.score:
                best = Signal(label, value, passage)
    return best


def aggregate_margin(passages: Sequence[Passage], probs: np.ndarray) -> Signal:
    """max p(S) - max p(R): a supporting and a contradicting passage cancel out."""
    if len(passages) != len(probs):
        raise ValueError(f"{len(passages)} passages but {len(probs)} probability rows")
    if len(passages) == 0:
        return NOTHING
    s_at = int(np.argmax(probs[:, _S]))
    r_at = int(np.argmax(probs[:, _R]))
    supported = float(probs[s_at, _S])
    refuted = float(probs[r_at, _R])
    if supported > refuted:
        return Signal(Label.SUPPORTED, supported - refuted, passages[s_at])
    if refuted > supported:
        return Signal(Label.REFUTED, refuted - supported, passages[r_at])
    return NOTHING


AGGREGATORS: Mapping[str, Aggregator] = {
    "max": aggregate_max,
    "max_nei": aggregate_max_nei,
    "margin": aggregate_margin,
}


def signal_for(
    claim: str, passages: Sequence[Passage], probs: np.ndarray, *, k: int, aggregation: str
) -> Signal:
    """What one combination proposes for one item, numeric layer first (spec §10).

    A numeric mismatch decides by rule at score 1.0, as ``pipeline.decide_indexed``
    does, so no candidate model is credited with what the rule decided.
    """
    ruled = _rule_signal(claim, passages, k=k)
    if ruled is not None:
        return ruled
    return AGGREGATORS[aggregation](passages[:k], probs[:k])


def _rule_signal(claim: str, passages: Sequence[Passage], *, k: int) -> Signal | None:
    """The numeric layer's REFUTED at score 1.0 over the first ``k`` hits, if it fires."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    numeric = numerics.check(claim, list(passages[:k]))
    if numeric is not None and numeric.mismatch:
        return Signal(Label.REFUTED, 1.0, numeric.passage)
    return None


def numeric_firings(scored: Sequence[Scored], *, k: int) -> tuple[int, int]:
    """``(fired, correct)``: how often the numeric layer decides by rule at this k.

    The same path as ``signal_for``, so these are exactly the verdicts scored 1.0 that
    no model earned. A firing is correct when the gold label is REFUTED. It depends on
    the items and k only, never on the model or the aggregation.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    fired = correct = 0
    for s in scored:
        if _rule_signal(s.item.claim, s.item.passages, k=k) is not None:
            fired += 1
            correct += Label(s.item.gold) is Label.REFUTED
    return fired, correct


def decide(signal: Signal, thresholds: Thresholds) -> Label:
    """The verdict a signal earns: below the cut, or with no passage, it is NEI."""
    if signal.label is Label.NEI or signal.passage is None:
        return Label.NEI
    return signal.label if signal.score >= thresholds.decide else Label.NEI


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for ``correct`` successes out of ``n``."""
    if n <= 0:
        raise ValueError("a Wilson interval needs at least one observation")
    if not 0 <= correct <= n:
        raise ValueError(f"correct must lie between 0 and n={n}, got {correct}")
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# Wider than eval_scifact's 0.30-0.95 grid: a ``margin`` score is a difference of two
# probabilities and lives much nearer 0. Every combination is swept on the same grid.
GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
HIGH_TARGET = 0.85
MEDIUM_TARGET = 0.70
# Below this many dev verdicts a tier's precision is reported as "too few to judge".
MIN_TIER_N = 20
CUT_DECIMALS = 6


@dataclass(frozen=True)
class Scored:
    item: Item
    probs: np.ndarray  # (len(item.passages), 3) in LABEL_ORDER


def _require(scored: Sequence[Scored], dataset: str) -> None:
    wrong = sorted({s.item.dataset for s in scored if s.item.dataset != dataset})
    if wrong:
        raise ValueError(f"expected only {dataset} items, got {', '.join(wrong)}")


def ship_cut(value: float) -> float:
    """Round a cut up to ``CUT_DECIMALS`` — the rule of ``eval_scifact.ship_cut``.

    Up, never down: a cut rounded down would hand a tier to scores the calibration
    split never showed were that good.
    """
    scale: int = 10**CUT_DECIMALS
    return min(1.0, math.ceil(value * scale) / scale)


def calibration_rows(scored: Sequence[Scored], *, k: int, aggregation: str) -> list[metrics.Row]:
    """``(gold, proposed label, score)`` rows for calibration, SciFact train only."""
    _require(scored, "scifact-train")
    rows: list[metrics.Row] = []
    for s in scored:
        signal = signal_for(s.item.claim, s.item.passages, s.probs, k=k, aggregation=aggregation)
        rows.append((Label(s.item.gold), signal.label, signal.score))
    return rows


def calibrate(rows: Sequence[metrics.Row]) -> Thresholds:
    """Best-accuracy ``decide``, then the 0.85 / 0.70 precision cuts above it."""
    best = metrics.sweep_decide(rows, grid=GRID)
    raw_high, raw_medium = metrics.tier_cutpoints(
        rows,
        decide=best.threshold,
        high_precision=HIGH_TARGET,
        medium_precision=MEDIUM_TARGET,
    )
    high, medium = ship_cut(raw_high), ship_cut(raw_medium)
    return Thresholds(decide=best.threshold, high=max(high, medium), medium=medium)


@dataclass(frozen=True)
class TierStat:
    tier: str
    n: int
    correct: int

    @property
    def precision(self) -> float | None:
        return self.correct / self.n if self.n >= MIN_TIER_N else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.correct, self.n) if self.n >= MIN_TIER_N else None


@dataclass(frozen=True)
class SciFactResult:
    n: int
    accuracy: float
    macro_f1: float
    f1: dict[Label, float]
    rationale_f1: float
    tiers: tuple[TierStat, ...]  # high, medium, low
    asserted_without_passage: int


def per_label_f1(gold: Sequence[Label], pred: Sequence[Label]) -> dict[Label, float]:
    scores: dict[Label, float] = {}
    for label in metrics.LABELS:
        tp = sum(g is label and p is label for g, p in zip(gold, pred, strict=True))
        fp = sum(g is not label and p is label for g, p in zip(gold, pred, strict=True))
        fn = sum(g is label and p is not label for g, p in zip(gold, pred, strict=True))
        scores[label] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return scores


def _unbacked(signal: Signal, thresholds: Thresholds) -> bool:
    """An assertion over the cut with no passage under it (product rule 1)."""
    return (
        signal.label is not Label.NEI
        and signal.score >= thresholds.decide
        and signal.passage is None
    )


def evaluate_scifact(
    scored: Sequence[Scored], thresholds: Thresholds, *, k: int, aggregation: str
) -> SciFactResult:
    _require(scored, "scifact-dev")
    gold: list[Label] = []
    pred: list[Label] = []
    gold_rationale: list[frozenset[int]] = []
    pred_rationale: list[frozenset[int]] = []
    tiers = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}
    unbacked = 0
    for s in scored:
        signal = signal_for(s.item.claim, s.item.passages, s.probs, k=k, aggregation=aggregation)
        unbacked += _unbacked(signal, thresholds)
        label = decide(signal, thresholds)
        truth = Label(s.item.gold)
        gold.append(truth)
        pred.append(label)
        gold_rationale.append(s.item.rationale)
        if label is not Label.NEI and signal.passage is not None:
            pred_rationale.append(frozenset({signal.passage.index}))
            counts = tiers[thresholds.tier(signal.score)]
            counts[0] += 1
            counts[1] += label is truth
        else:
            pred_rationale.append(frozenset())
    return SciFactResult(
        n=len(scored),
        accuracy=metrics.accuracy(gold, pred),
        macro_f1=metrics.macro_f1(gold, pred),
        f1=per_label_f1(gold, pred),
        rationale_f1=metrics.rationale_f1(gold_rationale, pred_rationale),
        tiers=tuple(TierStat(name, n, c) for name, (n, c) in tiers.items()),
        asserted_without_passage=unbacked,
    )


@dataclass(frozen=True)
class AveritecResult:
    n: int
    counted: int  # claims with a 3-way gold label
    accuracy_all: float
    readable: int  # counted claims with a source whose snapshot text is not blank
    accuracy_readable: float
    majority_baseline: float
    per_label: dict[str, tuple[int, int]]  # gold label -> (n, correct)
    asserted_without_passage: int


def evaluate_averitec(
    claims: Sequence[SnapshotClaim],
    scored: Sequence[Scored],
    thresholds: Thresholds,
    *,
    k: int,
    aggregation: str,
) -> AveritecResult:
    """Per claim, the rule of ``eval_averitec._decide_claim``.

    The strongest decided non-NEI verdict across the claim's sources wins. A claim
    with a readable source and no assertion is NEI. A claim with nothing readable is
    unanswered, and is scored as NEI.

    A claim is readable when any of its snapshot sources has non-blank text, as
    ``_decide_claim``'s ``fetched_any`` has it, whether or not scored items exist.
    Scored items whose group matches no snapshot claim are refused: a key-format drift
    must not silently turn every claim into NEI (product rule 6).
    """
    _require(scored, "averitec")
    by_group: dict[str, list[Scored]] = {}
    for s in scored:
        by_group.setdefault(s.item.group, []).append(s)
    known = {f"averitec:{claim.claim_id}" for claim in claims}
    orphans = sorted(set(by_group) - known)
    if orphans:
        raise ValueError(
            f"{len(orphans)} scored groups match no snapshot claim: {', '.join(orphans[:5])}"
        )
    per_label: dict[str, tuple[int, int]] = {}
    golds: list[Label] = []
    correct_all = readable = correct_readable = unbacked = 0
    for claim in claims:
        group = by_group.get(f"averitec:{claim.claim_id}", [])
        best: Signal | None = None
        for s in group:
            signal = signal_for(
                s.item.claim, s.item.passages, s.probs, k=k, aggregation=aggregation
            )
            unbacked += _unbacked(signal, thresholds)
            if decide(signal, thresholds) is Label.NEI:
                continue
            if best is None or signal.score > best.score:
                best = signal
        predicted = best.label if best is not None else Label.NEI
        gold = averitec.to_label(claim.gold)
        correct = gold is not None and predicted is gold
        seen, right = per_label.get(claim.gold, (0, 0))
        per_label[claim.gold] = (seen + 1, right + int(correct))
        if gold is None:
            continue
        golds.append(gold)
        correct_all += correct
        if any(source.text.strip() for source in claim.sources):
            readable += 1
            correct_readable += correct
    majority = max(Counter(golds).values(), default=0)
    return AveritecResult(
        n=len(claims),
        counted=len(golds),
        accuracy_all=correct_all / len(golds) if golds else 0.0,
        readable=readable,
        accuracy_readable=correct_readable / readable if readable else 0.0,
        majority_baseline=majority / len(golds) if golds else 0.0,
        per_label=per_label,
        asserted_without_passage=unbacked,
    )


TIE_MARGIN = 0.01
LARGE_MARGIN = 0.03
NO_CHANGE_MARGIN = 0.01
# Absorbs float representation error only (0.627 - 0.597 is 0.02999999...), not display
# rounding: a gap shown as 0.0295 must still fail a 0.03 margin.
_EPS = 1e-9

DECISION_RULE = """\
The steps run in this order. Each one filters the rows the previous step left.

1. **Eligible:** zero assertions without a passage, and AVeriTeC readable-subset accuracy
   not below the `base` × k=1 × `max` row.
2. **Beats the baseline:** a row stays only if its SciFact dev macro-F1 beats
   `base` × k=1 × `max` by **≥ 0.01**. If no row stays, the result is "no change". It is
   written up as such, and the next package (B, coverage) starts.
3. **Size gate:** a `large*` row stays only if it beats the highest macro-F1 of any
   eligible `base` row by **≥ 0.03**. That is the strongest eligible `base` row, not the
   one the tie rule would pick.
4. **Winner:** the highest SciFact dev macro-F1 among the rows left.
5. **Tie:** rows left within 0.01 of the winner go to the smaller k, then to the simpler
   aggregation, in the order `max`, `max_nei`, `margin`."""  # noqa: RUF001


@dataclass(frozen=True)
class Combo:
    model: str
    k: int
    aggregation: str

    def label(self) -> str:
        return f"`{self.model}` × k={self.k} × `{self.aggregation}`"  # noqa: RUF001


@dataclass(frozen=True)
class Outcome:
    combo: Combo
    large: bool
    macro_f1: float  # SciFact dev
    averitec_readable: float
    asserted_without_passage: int


@dataclass(frozen=True)
class Choice:
    winner: Combo | None  # None means "no change"
    reason: str


def choose(outcomes: Sequence[Outcome], *, baseline: Combo) -> Choice:
    """Apply ``DECISION_RULE`` step by step; each step filters the previous step's rows.

    When the baseline row itself fails step 1, the reason says so: every margin is still
    measured against its numbers, and a reader must not take it for a clean baseline.
    """
    choice = _choose(outcomes, baseline=baseline)
    base = next(row for row in outcomes if row.combo == baseline)
    if base.asserted_without_passage:
        return Choice(
            choice.winner,
            "the baseline itself is not eligible: it asserts without a passage "
            f"{base.asserted_without_passage} times; {choice.reason}",
        )
    return choice


def _choose(outcomes: Sequence[Outcome], *, baseline: Combo) -> Choice:
    by_combo = {row.combo: row for row in outcomes}
    if len(by_combo) != len(outcomes):
        duplicated = sorted(
            combo.label() for combo, n in Counter(row.combo for row in outcomes).items() if n > 1
        )
        raise ValueError(f"duplicate combinations among the outcomes: {', '.join(duplicated)}")
    if baseline not in by_combo:
        raise ValueError(f"the baseline {baseline.label()} is not among the outcomes")
    base = by_combo[baseline]

    # Step 1: eligible.
    eligible = [
        row
        for row in outcomes
        if row.asserted_without_passage == 0
        and row.averitec_readable >= base.averitec_readable - _EPS
    ]
    if not eligible:
        return Choice(None, "no change: no row is eligible under step 1")

    # Step 2: beats the baseline.
    beating = [row for row in eligible if row.macro_f1 - base.macro_f1 >= NO_CHANGE_MARGIN - _EPS]
    if not beating:
        best = max(eligible, key=lambda row: row.macro_f1)
        gain = best.macro_f1 - base.macro_f1
        return Choice(
            None,
            f"no change: the best eligible row, {best.combo.label()}, is {gain:+.3f} "
            f"macro-F1 over the baseline, below {NO_CHANGE_MARGIN}",
        )

    # Step 3: size gate, measured against the best eligible base row (step 1 rows).
    best_base = max((row.macro_f1 for row in eligible if not row.large), default=base.macro_f1)
    left = [
        row for row in beating if not row.large or row.macro_f1 - best_base >= LARGE_MARGIN - _EPS
    ]
    if not left:
        return Choice(
            None,
            "no change: the only rows that beat the baseline were large rows, and none "
            f"beat the best eligible base row ({best_base:.3f}) by {LARGE_MARGIN}",
        )

    # Steps 4 and 5: highest macro-F1, then the tie rule.
    top = max(left, key=lambda row: row.macro_f1)
    contenders = [row for row in left if top.macro_f1 - row.macro_f1 <= TIE_MARGIN + _EPS]
    winner = min(
        contenders,
        key=lambda row: (
            row.combo.k,
            AGGREGATIONS.index(row.combo.aggregation),
            -row.macro_f1,
        ),
    )
    gain = winner.macro_f1 - base.macro_f1
    reason = f"{winner.combo.label()} is {gain:+.3f} macro-F1 over {baseline.label()}"
    if winner is not top:
        reason += (
            f"; the tie rule chose it over {top.combo.label()}, "
            f"the top macro-F1 row ({top.macro_f1:.3f})"
        )
    return Choice(winner.combo, reason)


def best_eligible_base(outcomes: Sequence[Outcome], *, baseline: Combo) -> Outcome | None:
    """The highest-macro-F1 non-large row that passes step 1, or None when none does.

    Step 1 as ``choose`` applies it: no assertion without a passage, and AVeriTeC
    readable accuracy not below the baseline's. The report uses it to state a winner's
    gain over the strongest eligible ``base`` row, the bar of the size gate.
    """
    base = next((row for row in outcomes if row.combo == baseline), None)
    if base is None:
        raise ValueError(f"the baseline {baseline.label()} is not among the outcomes")
    eligible = [
        row
        for row in outcomes
        if not row.large
        and row.asserted_without_passage == 0
        and row.averitec_readable >= base.averitec_readable - _EPS
    ]
    return max(eligible, key=lambda row: row.macro_f1, default=None)
