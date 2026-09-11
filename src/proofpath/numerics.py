"""Numeric claims, checked before NLI (spec section 10).

NLI models do not read numbers: "a 40% speedup" against "a 4-8% improvement"
comes back NEI. This layer extracts quantities from both sides, compares only
what is genuinely comparable (same kind and unit), and refutes only when nothing
in the passage agrees. A matching number never asserts support on its own.

Rules fixed 2026-09-11 (OPEN-ITEMS 4.3): relative tolerance 10 %, applied to the
source side; a point agrees with a range if it lies inside the stretched range;
ranges agree if they overlap; opposite explicit directions disagree.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from proofpath.models import Passage

Kind = Literal["percent", "factor", "unit"]
Direction = Literal["up", "down"]

TOLERANCE = 0.10


@dataclass(frozen=True)
class Quantity:
    kind: Kind
    low: float
    high: float
    unit: str
    text: str
    direction: Direction | None


@dataclass(frozen=True)
class NumericResult:
    """Outcome of comparing one claim quantity against the passages."""

    mismatch: bool
    passage: Passage
    claim: Quantity
    source: Quantity

    @property
    def claim_text(self) -> str:
        return self.claim.text

    @property
    def source_text(self) -> str:
        return self.source.text


# --- extraction ---------------------------------------------------------------

# A number glued to a letter, dot or sign ("H3.3", "+1", "CD4") is a label, not a quantity.
_NUM_RAW = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_NUM = rf"(?<![\w.+-]){_NUM_RAW}"
# Percentages and factors may carry a sign ("-31%"); counts with units may not
# ("+1 nucleosome" is a position label).
_NUM_SIGNED = rf"(?<![\w.+-])(?P<sign>[+-])?{_NUM_RAW}"
_SEP = r"(?:\s*[-–—]\s*|\s+to\s+|\s+and\s+)"  # noqa: RUF001 - real dashes in prose
_SCALE = r"(?:\s*(?P<scale>thousand|million|billion))?"
_PCT = r"(?:\s*%|\s+per\s?cent\b)"
_FACTOR = r"(?:\s*[x×](?![A-Za-z])|\s*-?\s*fold\b|\s+times\b)"  # noqa: RUF001
_UNIT = r"\s+(?P<unit>[A-Za-z][A-Za-z-]*)"

_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_WORD = "|".join(_WORDS)

# Ordered: ranges before singles, suffixed forms before bare unit words.
_PATTERNS: list[tuple[re.Pattern[str], Kind]] = [
    (
        re.compile(
            rf"(?:between\s+)?(?P<a>{_NUM_SIGNED})(?:\s*%)?{_SEP}(?P<b>{_NUM_RAW}){_PCT}", re.I
        ),
        "percent",
    ),
    (re.compile(rf"(?:between\s+)?(?P<a>{_NUM}){_SEP}(?P<b>{_NUM_RAW}){_FACTOR}", re.I), "factor"),
    (
        re.compile(rf"(?:between\s+)?(?P<a>{_NUM}){_SEP}(?P<b>{_NUM_RAW}){_SCALE}{_UNIT}", re.I),
        "unit",
    ),
    (re.compile(rf"(?P<a>{_NUM_SIGNED}){_PCT}", re.I), "percent"),
    (re.compile(rf"(?P<a>{_NUM}){_FACTOR}", re.I), "factor"),
    (re.compile(rf"\b(?P<w>{_WORD})(?:\s*-?\s*fold|\s+times)\b", re.I), "factor"),
    (re.compile(rf"(?P<a>{_NUM}){_SCALE}{_UNIT}", re.I), "unit"),
]

# "95% CI" is a confidence level, not a measured quantity.
_CONFIDENCE_LEVEL = re.compile(r"\s*(?:CI|confidence)\b", re.I)

# Words that follow a number without being its unit.
_NOT_UNITS = {
    "to", "and", "of", "in", "on", "at", "for", "with", "by", "from", "the", "a", "an",
    "or", "per", "is", "was", "were", "are", "than", "vs", "versus", "as", "that",
    "which", "out", "over", "under", "about", "respectively", "but", "while", "if",
    "when", "where", "into", "onto", "each", "using", "after", "before", "between",
    "among", "up", "down", "more", "less", "fewer", "higher", "lower",
}  # fmt: skip

_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}

_UP = {
    "increase", "increased", "increases", "increasing", "gain", "gains", "gained",
    "improvement", "improvements", "improved", "improve", "improves", "speedup",
    "speed-up", "faster", "higher", "rise", "rose", "risen", "growth", "grew", "more",
    "greater", "larger", "up",
}  # fmt: skip
_DOWN = {
    "decrease", "decreased", "decreases", "decreasing", "reduction", "reductions",
    "reduced", "reduce", "reduces", "drop", "dropped", "drops", "decline", "declined",
    "declines", "lower", "slower", "fewer", "less", "loss", "losses", "lost", "down",
    "smaller", "shorter", "cut",
}  # fmt: skip

_WINDOW = 4


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _direction(text: str, start: int, end: int) -> Direction | None:
    before = re.findall(r"[A-Za-z-]+", text[:start])[-_WINDOW:]
    after = re.findall(r"[A-Za-z-]+", text[end:])[:_WINDOW]
    # "less than 10%" / "more than 40%" are bounds on a value, not a change.
    if "than" in [w.lower() for w in before[-2:]]:
        before = []
    for word in [*reversed(before), *after]:
        lowered = word.lower()
        if lowered in _UP:
            return "up"
        if lowered in _DOWN:
            return "down"
    return None


def extract(text: str) -> list[Quantity]:
    """Quantities in ``text``, in order of appearance. Bare numbers are ignored."""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, Quantity]] = []
    for pattern, kind in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(s < end and start < e for s, e in taken):
                continue
            groups = match.groupdict()
            if kind == "unit":
                unit = groups["unit"].lower()
                if unit in _NOT_UNITS:
                    continue
            elif kind == "percent":
                unit = "%"
                if _CONFIDENCE_LEVEL.match(text, end):
                    continue
            else:
                unit = "x"
            sign = groups.get("sign")
            if groups.get("w"):
                low = high = float(_WORDS[groups["w"].lower()])
            else:
                low = _number(groups["a"].lstrip("+-"))
                high = _number(groups["b"]) if groups.get("b") else low
            if groups.get("scale"):
                factor = _SCALES[groups["scale"].lower()]
                low, high = low * factor, high * factor
            taken.append((start, end))
            found.append(
                (
                    start,
                    Quantity(
                        kind=kind,
                        low=min(low, high),
                        high=max(low, high),
                        unit=unit,
                        text=match.group(0).strip(),
                        direction=("down" if sign == "-" else "up" if sign == "+" else None)
                        or _direction(text, start, end),
                    ),
                )
            )
    return [quantity for _, quantity in sorted(found, key=lambda item: item[0])]


# --- comparison ---------------------------------------------------------------

_UNIT_SYNONYMS = {
    "milligram": "mg",
    "gram": "g",
    "kilogram": "kg",
    "microgram": "ug",
    "µg": "ug",
    "μg": "ug",
    "milliliter": "ml",
    "millilitre": "ml",
    "liter": "l",
    "litre": "l",
    "millimeter": "mm",
    "millimetre": "mm",
    "centimeter": "cm",
    "centimetre": "cm",
    "meter": "m",
    "metre": "m",
    "kilometer": "km",
    "kilometre": "km",
    "hour": "h",
    "hr": "h",
    "minute": "min",
    "second": "s",
    "sec": "s",
    "year": "yr",
    "day": "d",
    "person": "people",
}


def _normalize_unit(unit: str) -> str:
    lowered = unit.lower()
    singular = lowered[:-1] if lowered.endswith("s") and len(lowered) > 2 else lowered
    return _UNIT_SYNONYMS.get(singular, _UNIT_SYNONYMS.get(lowered, singular))


def comparable(a: Quantity, b: Quantity) -> bool:
    """Same kind, and for unit quantities the same normalised unit."""
    if a.kind != b.kind:
        return False
    if a.kind == "unit":
        return _normalize_unit(a.unit) == _normalize_unit(b.unit)
    return True


def agrees(claim: Quantity, source: Quantity, *, tolerance: float = TOLERANCE) -> bool:
    """Does the claim's figure fall within the source's figure, stretched by ``tolerance``?

    Asymmetric on purpose: the source interval is widened, the claim's is not, so
    a claim of 40 % against a source of 35 % is a mismatch while 38 % is not.
    Opposite explicit directions disagree regardless of value.
    """
    if not comparable(claim, source):
        return False
    if claim.direction and source.direction and claim.direction != source.direction:
        return False
    low = source.low * (1 - tolerance)
    high = source.high * (1 + tolerance)
    return claim.low <= high and claim.high >= low


def _distance(claim: Quantity, source: Quantity) -> float:
    mid_claim = (claim.low + claim.high) / 2
    mid_source = (source.low + source.high) / 2
    return abs(mid_source - mid_claim) / max(abs(mid_claim), 1e-9)


def check(claim: str, passages: Sequence[Passage]) -> NumericResult | None:
    """Compare every claim quantity against the passages.

    Returns ``None`` when the claim has no quantities or nothing comparable was
    found. A mismatch is reported only when the attribution is unambiguous: the
    claim holds one quantity of that kind, the passage holds exactly one
    comparable figure, and no comparable figure in any passage agrees. With
    several candidate figures the layer cannot tell which one the claim refers to
    and stays silent — measured on SciFact dev, every ambiguous firing was wrong.
    """
    claim_quantities = extract(claim)
    if not claim_quantities:
        return None
    extracted = [(passage, extract(passage.text)) for passage in passages]
    consistent: NumericResult | None = None
    for quantity in claim_quantities:
        if sum(comparable(quantity, other) for other in claim_quantities) > 1:
            continue
        agreed: NumericResult | None = None
        closest: NumericResult | None = None
        for passage, found in extracted:
            candidates = [c for c in found if comparable(quantity, c)]
            for candidate in candidates:
                if agrees(quantity, candidate):
                    agreed = agreed or NumericResult(False, passage, quantity, candidate)
            # A change ("decreased by 10%") and a level ("57% women") are different
            # quantities; only like against like can be called a mismatch.
            if (
                agreed is None
                and len(candidates) == 1
                and (candidates[0].direction is None) == (quantity.direction is None)
            ):
                candidate = candidates[0]
                if closest is None or _distance(quantity, candidate) < _distance(
                    quantity, closest.source
                ):
                    closest = NumericResult(True, passage, quantity, candidate)
        if agreed is not None:
            consistent = consistent or agreed
        elif closest is not None:
            return closest
    return consistent
