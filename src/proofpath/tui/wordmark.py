"""The ``rich`` banner: ``ProofPath`` in block letters (wordmark design §3-§4).

:data:`WORDMARK` is the source of truth -- six lines in the block-and-shadow style
of the figlet font *ANSI Shadow*, hand-drawn because that font has no lowercase and
the two ``P`` are the only capitals -- and is kept here verbatim so the mark can be
read without running anything. :func:`render` puts the version line and the hint
line under it; below :data:`RICH_WIDTH_FLOOR` it draws the ``plain`` banner whatever
the theme, as the raven did below its own floor.

Pure: no Textual, and no colour. A block cell is ``light`` and a shadow cell is
``dark``; the result carries colour *runs*, and the widget asks the theme what a tone
is painted with.
"""

from __future__ import annotations

from typing import Protocol

from proofpath.tui import banner
from proofpath.tui.banner import Banner, Run, Tone

#: The mark. ``P`` is six rows; ``r o a`` sit on the baseline four rows tall; ``f h``
#: rise the full six; ``t`` five. Trailing spaces are stripped; only :data:`BLOCK`,
#: :data:`SHADOW` and the space appear (``test_tui_wordmark.py`` reads it back).
WORDMARK: tuple[str, ...] = (
    "██████╗                        █████╗██████╗               ██╗",
    "██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║",
    "██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗",
    "██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗",
    "██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║",
    "╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝",
)
#: The letter body, painted in the light tone.
BLOCK = "█"
#: The shadow to the right of and under each letter, painted in the dark tone.
SHADOW = "╗╔═╝║╚"
#: The two text lines start at the left edge, under the mark, not beside it.
TEXT_COLUMN = 0
VERSION_ROW = len(WORDMARK)
HINT_ROW = VERSION_ROW + 1
#: The mark's width plus the right margin, plus one: narrower than this the mark
#: would touch or cross the margin, so the ``plain`` banner is drawn instead.
RICH_WIDTH_FLOOR = max(len(line) for line in WORDMARK) + banner.RIGHT_MARGIN + 1


class ThemeLike(Protocol):
    """What :func:`render` needs of a theme: its name."""

    name: str


def _tone(character: str) -> Tone | None:
    """The tone one character paints, or ``None`` when it belongs to no run."""
    return "light" if character == BLOCK else "dark" if character in SHADOW else None


def tones(line: str) -> tuple[Run, ...]:
    """The colour runs of one art line: ``light`` over blocks, ``dark`` over shadow.

    Spaces belong to no run, so a run ends at a space and the next begins after it;
    neighbouring cells of one tone are one run.
    """
    runs: list[Run] = []
    start = 0
    tone: Tone | None = None
    for column, character in enumerate(line):
        here = _tone(character)
        if here != tone:
            if tone is not None:
                runs.append((start, column, tone))
            start, tone = column, here
    if tone is not None:
        runs.append((start, len(line), tone))
    return tuple(runs)


def _fits(width: int) -> bool:
    """Whether the mark fits at ``width`` columns -- the shared floor check for
    :func:`render` and :func:`hint_row`, so the two cannot drift apart."""
    return width >= RICH_WIDTH_FLOOR


def render(width: int, theme: ThemeLike, *, version: str, context: str, hint: str) -> Banner:
    """Draw the banner ``theme`` calls for at ``width`` columns.

    In ``rich`` the mark is fixed at its six rows; the version line and the hint
    line follow, each with its right part aligned to the margin. Below
    :data:`RICH_WIDTH_FLOOR` the ``plain`` banner is drawn instead.
    """
    if theme.name != "rich" or not _fits(width):
        return banner.render(width, version=version, context=context, hint=hint)
    lines = list(WORDMARK)
    lines.append(banner.right_align(f"proofpath v{version}", context, width))
    prompt, commands = banner.split_hint(hint)
    lines.append(prompt if commands is None else banner.right_align(prompt, commands, width))
    art_tones = tuple(tones(line) for line in WORDMARK)
    return Banner(tuple(lines), (*art_tones, (), ()))


def hint_row(width: int) -> int:
    """The row :func:`render` puts the hint on at ``width`` in ``rich``: the wordmark's
    own below the mark, or the ``plain`` banner's when the mark does not fit."""
    return HINT_ROW if _fits(width) else banner.HINT_LINE


__all__ = [
    "BLOCK",
    "HINT_ROW",
    "RICH_WIDTH_FLOOR",
    "SHADOW",
    "TEXT_COLUMN",
    "VERSION_ROW",
    "WORDMARK",
    "ThemeLike",
    "hint_row",
    "render",
    "tones",
]
