"""Core value types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, get_args

Tier = Literal["high", "medium", "low"]
TIER_VALUES: tuple[str, ...] = get_args(Tier)


class Label(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEI = "NEI"


@dataclass(frozen=True)
class Passage:
    """One quotable unit of a source, addressable enough to be cited back."""

    text: str
    source_id: str
    index: int


@dataclass(frozen=True)
class Verdict:
    """The outcome for one claim.

    A ``SUPPORTED`` or ``REFUTED`` verdict cannot exist without the passage it rests
    on (product rule 1). That is checked here, once, so no later stage can forget.
    """

    label: Label
    score: float
    tier: Tier
    passage: Passage | None
    # Set when a rule, not the model, decided: e.g. "numeric mismatch: ...".
    reason: str = ""

    def __post_init__(self) -> None:
        if self.label is not Label.NEI and self.passage is None:
            raise ValueError(f"{self.label.value} verdict requires a passage")
        if self.tier not in TIER_VALUES:
            raise ValueError(f"tier must be one of {TIER_VALUES}, got {self.tier!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within [0, 1], got {self.score}")
