"""NLI bake-off: aggregation variants, calibration, evaluation and the decision rule.

Pure functions over stored NLI probabilities; nothing here touches the network or a
model (docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md). A variant moves into
``pipeline.py`` only if the bake-off picks it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from proofpath import numerics
from proofpath.entailment import LABEL_ORDER
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
    top = list(passages[:k])
    numeric = numerics.check(claim, top)
    if numeric is not None and numeric.mismatch:
        return Signal(Label.REFUTED, 1.0, numeric.passage)
    return AGGREGATORS[aggregation](top, probs[:k])


def decide(signal: Signal, thresholds: Thresholds) -> Label:
    """The verdict a signal earns: below the cut, or with no passage, it is NEI."""
    if signal.label is Label.NEI or signal.passage is None:
        return Label.NEI
    return signal.label if signal.score >= thresholds.decide else Label.NEI


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for ``correct`` successes out of ``n``."""
    if n <= 0:
        raise ValueError("a Wilson interval needs at least one observation")
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)
