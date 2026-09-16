"""The raven in ``rich``: a two-tone pixel bitmap drawn in Braille cells (raven design
sections 3 and 4).

:data:`BITMAP` is the source of truth -- 29 x 30 pixels, ``#`` dark, ``+`` light, ``.``
empty -- and is kept here verbatim so the bird can be read without running anything.
:func:`encode` turns it into Braille cells (U+2800-U+28FF, two by four dots each), so
eight rows of text carry thirty rows of pixels, and every glyph is East Asian width
``N``: one cell wide in every locale. :func:`render` puts the two text lines beside the
bird and runs its ground to the right edge; below :data:`RICH_WIDTH_FLOOR` it draws
the ``plain`` raven whatever the theme.

Pure: no Textual, and no colour. A cell takes the majority tone of its dots and the
result carries colour *runs*; the widget asks the theme what a tone is painted with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from proofpath.tui import banner
from proofpath.tui.banner import Banner, Run, Tone

DARK, LIGHT, EMPTY = "#", "+", "."

#: The raven, perched, head top-right, tail bottom-left, on its ground. The eye is
#: the ``..`` hole on row 3. Row 29 is the feet; the ground continues to the right.
BITMAP: tuple[str, ...] = (
    "................#####........",
    "...............#######.......",
    "..............########.......",
    "..............####..###+.....",
    "..............#########+++...",
    "..............########+++++..",
    "..............+#######++++...",
    ".............++++####...++...",
    "............+++++####........",
    "............++++++###........",
    "...........+++++++###........",
    "...........+++++++###........",
    "..........++++++++###........",
    "..........++++++++###........",
    ".........+++++++++###........",
    ".........++++++++####........",
    "........+++++++++###.........",
    "........++++++++####.........",
    ".......+++++++++###..........",
    ".......++++++++####..........",
    "......++++++++####...........",
    "......+++++++####............",
    ".....++++++#####.............",
    "....++++++#####..............",
    "...+++++#####+###............",
    "..++++######..+.#............",
    ".########+++..#.#............",
    "########..++..#.#............",
    "#####.....++..#.#............",
    "##.##.....+++##+##+++++++++++",
)

BRAILLE_BASE = 0x2800
#: ``DOT_BITS[row][column]``: the bit of each dot of a cell, in the standard order
#: (dots 1, 2, 3, 7 down the left column; 4, 5, 6, 8 down the right).
DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
CELL_WIDTH, CELL_HEIGHT = 2, 4
#: The ground beyond the bird: dots 2 and 5, the middle pair, a dotted rule.
GROUND = "⠒"

#: Where the two text lines start, clear of the bird, and which rows they sit on.
TEXT_COLUMN = 18
VERSION_ROW = 3
HINT_ROW = 4
GROUND_ROW = 7
#: Below this the ``plain`` pet is drawn whatever the theme (design section 4).
RICH_WIDTH_FLOOR = 48


class ThemeLike(Protocol):
    """The one field of ``theme.Theme`` the pet reads: which raven to draw."""

    @property
    def name(self) -> str: ...


def encode(pixels: Sequence[str]) -> tuple[tuple[str, ...], tuple[tuple[Run, ...], ...]]:
    """Braille cells and colour runs for a bitmap of ``#``, ``+`` and ``.``.

    Rows are padded to a multiple of four and columns to a multiple of two with
    empty pixels. A cell with no dots is a space; trailing spaces are stripped. A
    cell's tone is the majority tone of its dots, dark on a tie; neighbouring cells
    of one tone are merged into one run.
    """
    width = max((len(row) for row in pixels), default=0)
    width = -(-width // CELL_WIDTH) * CELL_WIDTH
    rows = [row.ljust(width, EMPTY) for row in pixels]
    while len(rows) % CELL_HEIGHT:
        rows.append(EMPTY * width)
    lines: list[str] = []
    runs: list[tuple[Run, ...]] = []
    for top in range(0, len(rows), CELL_HEIGHT):
        cells: list[str] = []
        tones: list[Tone | None] = []
        for left in range(0, width, CELL_WIDTH):
            bits = dark = light = 0
            for dy in range(CELL_HEIGHT):
                for dx in range(CELL_WIDTH):
                    pixel = rows[top + dy][left + dx]
                    if pixel == EMPTY:
                        continue
                    bits |= DOT_BITS[dy][dx]
                    if pixel == DARK:
                        dark += 1
                    else:
                        light += 1
            cells.append(chr(BRAILLE_BASE + bits) if bits else " ")
            tones.append(None if not bits else ("dark" if dark >= light else "light"))
        lines.append("".join(cells).rstrip())
        runs.append(_runs(tones))
    return tuple(lines), tuple(runs)


def _runs(tones: Sequence[Tone | None]) -> tuple[Run, ...]:
    """Merge a row of per-cell tones into ``(start, end, tone)`` runs."""
    out: list[Run] = []
    start = 0
    for column in range(len(tones) + 1):
        tone = tones[column] if column < len(tones) else None
        previous = tones[column - 1] if column else None
        if tone != previous:
            if previous is not None:
                out.append((start, column, previous))
            start = column
    return tuple(out)


def render(width: int, theme: ThemeLike, *, version: str, context: str, hint: str) -> Banner:
    """Draw the raven ``theme`` calls for at ``width`` columns.

    In ``rich`` the bird is fixed at fifteen columns and eight rows; the version line
    (row 3) and the hint line (row 4) start at :data:`TEXT_COLUMN` with their right
    parts aligned to the margin, and the ground (row 7) runs to the margin. Below
    :data:`RICH_WIDTH_FLOOR` the ``plain`` pet is drawn instead.
    """
    if theme.name != "rich" or width < RICH_WIDTH_FLOOR:
        return banner.render(width, version=version, context=context, hint=hint)
    art, art_tones = encode(BITMAP)
    lines, tones = list(art), list(art_tones)
    lines[VERSION_ROW] = banner.right_align(
        _beside(lines[VERSION_ROW], f"proofpath v{version}"), context, width
    )
    prompt, commands = banner.split_hint(hint)
    lines[HINT_ROW] = (
        _beside(lines[HINT_ROW], prompt)
        if commands is None
        else banner.right_align(_beside(lines[HINT_ROW], prompt), commands, width)
    )
    ground = width - banner.RIGHT_MARGIN - len(lines[GROUND_ROW])
    if ground > 0:
        start = len(lines[GROUND_ROW])
        ground_run: Run = (start, start + ground, "light")
        tones[GROUND_ROW] = (*tones[GROUND_ROW], ground_run)
        lines[GROUND_ROW] += GROUND * ground
    return Banner(tuple(lines), tuple(tones))


def _beside(art: str, text: str) -> str:
    return art.ljust(TEXT_COLUMN) + text


__all__ = [
    "BITMAP",
    "BRAILLE_BASE",
    "CELL_HEIGHT",
    "CELL_WIDTH",
    "DARK",
    "DOT_BITS",
    "EMPTY",
    "GROUND",
    "GROUND_ROW",
    "HINT_ROW",
    "LIGHT",
    "RICH_WIDTH_FLOOR",
    "TEXT_COLUMN",
    "VERSION_ROW",
    "ThemeLike",
    "encode",
    "render",
]
