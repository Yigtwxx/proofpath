"""What both banner themes share (wordmark design §8.1): the rendered-lines-plus-runs
result type, the right margin and the two text-row helpers. The art lives in
:mod:`proofpath.tui.wordmark`, which is the only renderer.

This module colours nothing. Each line comes with its colour *runs* -- ``(start,
end, tone)`` column slices -- and the widget asks the theme what a tone is painted
with, because one module owns every colour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A tone name the theme maps to a colour -- the wordmark's gradient bands and "shadow".
Tone = str
#: ``(start, end, tone)``: the columns ``[start, end)`` of one line and their tone.
Run = tuple[int, int, Tone]

#: Columns the drawn banner leaves free on the right. Two, not one: the context and
#: the command list end at column 78 of an 80-column terminal, level with each other.
RIGHT_MARGIN = 2
#: The hint is passed whole (the caller joined the prompt hint and the commands); it
#: is split at the first run of two or more spaces and the rest is right-aligned.
HINT_GAP = re.compile(r" {2,}")


@dataclass(frozen=True)
class Banner:
    """The rendered lines plus, per line, the column runs the theme colours."""

    lines: tuple[str, ...]
    #: ``tones[i]`` are the runs of ``lines[i]``; empty when nothing on it is coloured.
    tones: tuple[tuple[Run, ...], ...]


def right_align(left: str, right: str, width: int) -> str:
    """Pad ``left`` so ``right`` ends at the right margin, never truncating."""
    padding = width - RIGHT_MARGIN - len(left) - len(right)
    return left + " " * max(padding, 1) + right


def split_hint(hint: str) -> tuple[str, str | None]:
    """The prompt hint and, if the hint has a gap of two or more spaces, the commands."""
    parts = HINT_GAP.split(hint, maxsplit=1)
    return (parts[0], parts[1] if len(parts) == 2 else None)


__all__ = [
    "HINT_GAP",
    "RIGHT_MARGIN",
    "Banner",
    "Run",
    "Tone",
    "right_align",
    "split_hint",
]
