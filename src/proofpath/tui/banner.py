"""The ``plain`` raven, and the result type both themes draw into (raven design §5).

The bird is six lines of pure ASCII: no box drawing, no Braille, no emoji, so legacy
conhost and a CJK locale render it exactly as any other terminal does. It stands on
a ground of ``_`` that re-flows to the right edge on every resize -- what is left of
the ferret's path.

This module colours nothing. Each line comes with its colour *runs* -- ``(start,
end, tone)`` column slices -- and the widget asks the theme what a tone is painted
with, because one module owns every colour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: What a run is painted with; the theme maps a tone to a style.
Tone = Literal["dark", "light"]
#: ``(start, end, tone)``: the columns ``[start, end)`` of one line and their tone.
Run = tuple[int, int, Tone]

#: The raven, perched, head top-right, tail bottom-left. Lines 3 and 4 get the text
#: beside them; line 5 is the feet on the ground, which :func:`render` extends.
ART = (
    "       __",
    "      (o >",
    "    _/ /",
    "   /  /",
    "  /__/",
    " ____||",
)
GROUND = "_"
#: The index of the feet-on-the-ground line, the one :func:`render` extends.
GROUND_LINE = len(ART) - 1
#: The index of the hint line. The widget's drop rule reads the drawn hint back by
#: ``wordmark.HINT_ROW`` above the wordmark's floor and by this below it, where this
#: banner is what ``wordmark.render`` drew (``test_tui_wordmark.py``).
HINT_LINE = 4
#: Columns the drawn banner leaves free on the right. Two, not one: the context and
#: the command list end at column 78 of an 80-column terminal, level with each other.
RIGHT_MARGIN = 2
#: Below this the ground is not extended (the bird alone is what fits).
BODY_WIDTH_FLOOR = 40
#: The two text lines start here, clear of the widest art line beside them.
TEXT_INDENT = " " * 14
#: The hint is passed whole (the caller joined the prompt hint and the commands); it
#: is split at the first run of two or more spaces and the rest is right-aligned.
HINT_GAP = re.compile(r" {2,}")


@dataclass(frozen=True)
class Banner:
    """The rendered lines plus, per line, the column runs the theme colours."""

    lines: tuple[str, ...]
    #: ``tones[i]`` are the runs of ``lines[i]``; empty when nothing on it is coloured.
    tones: tuple[tuple[Run, ...], ...]


def render(width: int, *, version: str, context: str, hint: str) -> Banner:
    """Draw the ``plain`` banner for a terminal ``width`` columns wide.

    ``context`` is the run context (provider, online/offline, device), right-aligned
    on line 3; ``hint`` is the prompt hint and the slash-command list joined by the
    caller, split here and right-aligned the same way. Nothing is truncated: a line
    that does not fit overflows by one space.
    """
    lines = list(ART)
    lines[3] = right_align(_beside(ART[3], f"proofpath v{version}"), context, width)
    prompt, commands = split_hint(hint)
    lines[HINT_LINE] = (
        _beside(ART[HINT_LINE], prompt)
        if commands is None
        else right_align(_beside(ART[HINT_LINE], prompt), commands, width)
    )
    if width >= BODY_WIDTH_FLOOR:
        ground = ART[GROUND_LINE]
        lines[GROUND_LINE] = ground + GROUND * (width - RIGHT_MARGIN - len(ground))
    # A run stops where the art stops; on the ground line the extended ground is art too.
    tones = tuple(
        (_run(_first_ink(art), len(lines[index] if index == GROUND_LINE else art)),)
        for index, art in enumerate(ART)
    )
    return Banner(tuple(lines), tones)


def right_align(left: str, right: str, width: int) -> str:
    """Pad ``left`` so ``right`` ends at the right margin, never truncating."""
    padding = width - RIGHT_MARGIN - len(left) - len(right)
    return left + " " * max(padding, 1) + right


def split_hint(hint: str) -> tuple[str, str | None]:
    """The prompt hint and, if the hint has a gap of two or more spaces, the commands."""
    parts = HINT_GAP.split(hint, maxsplit=1)
    return (parts[0], parts[1] if len(parts) == 2 else None)


def _beside(art: str, text: str) -> str:
    return art.ljust(len(TEXT_INDENT)) + text


def _first_ink(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _run(start: int, end: int) -> Run:
    return (start, end, "light")


__all__ = [
    "ART",
    "BODY_WIDTH_FLOOR",
    "GROUND",
    "GROUND_LINE",
    "HINT_GAP",
    "HINT_LINE",
    "RIGHT_MARGIN",
    "TEXT_INDENT",
    "Banner",
    "Run",
    "Tone",
    "render",
    "right_align",
    "split_hint",
]
