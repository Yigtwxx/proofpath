"""The banner in both themes: ``ProofPath`` in block letters (wordmark design §3, §4,
§8 and §8.1).

:data:`WORDMARK` is the source of truth -- six lines in the block-and-shadow style
of the figlet font *ANSI Shadow*, hand-drawn because that font has no lowercase and
the two ``P`` are the only capitals -- and is kept here verbatim so the mark can be
read without running anything. :func:`render` draws the same thing for both themes:
``rich`` as written, ``plain`` transliterated cell for cell through
:data:`PLAIN_GLYPHS`, so the rows and the colour runs are identical and only the
glyphs differ.

The layout depends on the width alone. When both text rows fit to the right of the
mark they sit beside it, on its middle two rows; otherwise they go under it; and
below :data:`RICH_WIDTH_FLOOR` there is no mark at all, only the two text rows. In
every case a full-width rule is the last row, separating the banner from the log.
:func:`rows` says which of the three happened, so the widget never guesses.

Pure: no Textual, and no colour. The blocks run through five tones left to right
and the shadow glyphs and the rule share one more; the result carries colour
*runs* by tone name, and the widget asks the theme what a tone is painted with.
:func:`band` is the one place the five-band arithmetic lives: the mark applies it
over its own width, the input bar's frame (design section 9) over the terminal's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from proofpath.tui import banner
from proofpath.tui.banner import Banner, Run

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
#: The letter body, painted in the gradient band of its column.
BLOCK = "█"
#: The shadow to the right of and under each letter, painted in the shadow tone.
SHADOW = "╗╔═╝║╚"
#: The rule that closes the banner in both layouts, from column 0 to the margin.
RULE = "─"
#: The ``plain`` theme draws the same mark in seven-bit ASCII, cell for cell, so the
#: rows and the runs are identical in both themes and only the glyphs differ.
PLAIN_GLYPHS = str.maketrans("█╗╔╝╚═║─", "#++++-|-")
MARK_WIDTH = max(len(line) for line in WORDMARK)
#: When the texts sit beside the mark they start two columns past its widest line.
BESIDE_COLUMN = MARK_WIDTH + 2
#: The mark's middle two rows, which the two texts share when they sit beside it.
BESIDE_VERSION_ROW, BESIDE_HINT_ROW = 2, 3
#: Left to right, the blocks run through these tones in equal column bands of the
#: mark; the bands are fixed by the mark's width, never by the terminal's.
GRADIENT_TONES: tuple[str, ...] = ("g0", "g1", "g2", "g3", "g4")
BANDS = len(GRADIENT_TONES)
#: The shadow glyphs and the rule share one tone, darker than the gradient's end.
SHADOW_TONE = "shadow"
#: The mark's width plus the right margin, plus one: narrower than this the mark
#: would touch or cross the margin, so the text rows are drawn without it.
RICH_WIDTH_FLOOR = MARK_WIDTH + banner.RIGHT_MARGIN + 1


class ThemeLike(Protocol):
    """What :func:`render` needs of a theme: its name."""

    name: str


#: The three things :func:`rows` can decide: no mark at all (``narrow``), the texts
#: to the right of the mark (``beside``), or the texts under it (``under``).
Layout = Literal["narrow", "beside", "under"]


@dataclass(frozen=True)
class Rows:
    """Where :func:`render` put the version, the hint and the rule for one width,
    and which of the three layouts that was. :func:`render` branches on ``layout``
    alone, so the width predicates live in :func:`rows` and nowhere else."""

    version: int
    hint: int
    rule: int
    layout: Layout

    @property
    def beside(self) -> bool:
        """Whether the texts sit to the right of the mark."""
        return self.layout == "beside"


def band(column: int, width: int) -> str:
    """The gradient tone of ``column`` in something ``width`` columns wide: five equal
    bands, light at the left, dark at the right. The mark uses it over its own width,
    the bar's frame over the terminal's, so the two run the same way. A width of
    zero -- a frame that has not been laid out yet -- is treated as one, so the
    answer is a tone and not an exception."""
    return GRADIENT_TONES[min(BANDS - 1, column * BANDS // max(width, 1))]


def _tone(column: int, character: str) -> str | None:
    """The tone one cell paints: its band for a block, the shadow tone for a shadow
    glyph, nothing for a space."""
    if character == BLOCK:
        return band(column, MARK_WIDTH)
    return SHADOW_TONE if character in SHADOW else None


def tones(line: str) -> tuple[Run, ...]:
    """The colour runs of one art line. A block run is split where its band changes;
    a shadow run is not, because the shadow has one tone. Spaces belong to no run."""
    runs: list[Run] = []
    start = 0
    tone: str | None = None
    for column, character in enumerate(line):
        here = _tone(column, character)
        if here != tone:
            if tone is not None:
                runs.append((start, column, tone))
            start, tone = column, here
    if tone is not None:
        runs.append((start, len(line), tone))
    return tuple(runs)


def _fits(width: int) -> bool:
    """Whether the mark fits at ``width`` columns: the floor check, asked by
    :func:`rows` alone, so :func:`render` cannot drift from it."""
    return width >= RICH_WIDTH_FLOOR


def _beside_fits(width: int, *, version: str, context: str, hint: str) -> bool:
    """Whether both text rows fit to the right of the mark, each with at least one
    space between its halves, ending at the margin."""
    prompt, commands = banner.split_hint(hint)
    longest = max(
        len(f"proofpath v{version}") + 1 + len(context),
        len(prompt) + (1 + len(commands) if commands is not None else 0),
    )
    return width - banner.RIGHT_MARGIN - BESIDE_COLUMN >= longest


def rows(width: int, *, version: str, context: str, hint: str) -> Rows:
    """Where :func:`render` puts things at ``width``: beside the mark when both texts
    fit to its right, under it otherwise, and with no mark at all below the floor.
    The same for both themes; the widget asks this rather than guessing."""
    if not _fits(width):
        return Rows(version=0, hint=1, rule=2, layout="narrow")
    if _beside_fits(width, version=version, context=context, hint=hint):
        return Rows(
            version=BESIDE_VERSION_ROW, hint=BESIDE_HINT_ROW, rule=len(WORDMARK), layout="beside"
        )
    under = len(WORDMARK)
    return Rows(version=under, hint=under + 1, rule=under + 2, layout="under")


def _text_rows(
    width: int, *, version: str, context: str, hint: str, before: tuple[str, str] = ("", "")
) -> tuple[str, str]:
    """The version row and the hint row, each with its right part aligned to the
    margin. ``before`` is what each row starts with: nothing when the texts stand
    alone, the mark's row padded to :data:`BESIDE_COLUMN` when they sit beside it.
    A hint with no command list is written whole, with nothing to align."""
    version_line = banner.right_align(before[0] + f"proofpath v{version}", context, width)
    prompt, commands = banner.split_hint(hint)
    hint_line = (
        before[1] + prompt
        if commands is None
        else banner.right_align(before[1] + prompt, commands, width)
    )
    return version_line, hint_line


def render(width: int, theme: ThemeLike, *, version: str, context: str, hint: str) -> Banner:
    """Draw the banner for ``theme`` at ``width`` columns: the mark (when it fits),
    the two text rows beside or under it, and the rule. ``plain`` is the same
    drawing through :data:`PLAIN_GLYPHS`. Nothing is ever truncated: a text row
    that does not fit overflows by one space, and the widget deals with that.
    """
    where = rows(width, version=version, context=context, hint=hint)
    span = width - banner.RIGHT_MARGIN
    rule_run: tuple[Run, ...] = ((0, span, SHADOW_TONE),)

    if where.layout == "narrow":
        lines = [
            *_text_rows(width, version=version, context=context, hint=hint),
            RULE * span,
        ]
        runs: list[tuple[Run, ...]] = [(), (), rule_run]
    elif where.layout == "beside":
        # The text rows share the mark's rows, so their runs are the art's: the
        # text part starts past MARK_WIDTH, where no run reaches.
        lines = list(WORDMARK)
        runs = [tones(line) for line in WORDMARK]
        before = (
            WORDMARK[where.version].ljust(BESIDE_COLUMN),
            WORDMARK[where.hint].ljust(BESIDE_COLUMN),
        )
        lines[where.version], lines[where.hint] = _text_rows(
            width, version=version, context=context, hint=hint, before=before
        )
        lines.append(RULE * span)
        runs.append(rule_run)
    else:
        lines = [
            *WORDMARK,
            *_text_rows(width, version=version, context=context, hint=hint),
            RULE * span,
        ]
        runs = [*(tones(line) for line in WORDMARK), (), (), rule_run]
    if theme.name != "rich":
        lines = [line.translate(PLAIN_GLYPHS) for line in lines]
    return Banner(tuple(lines), tuple(runs))


__all__ = [
    "BANDS",
    "BESIDE_COLUMN",
    "BESIDE_HINT_ROW",
    "BESIDE_VERSION_ROW",
    "BLOCK",
    "GRADIENT_TONES",
    "MARK_WIDTH",
    "PLAIN_GLYPHS",
    "RICH_WIDTH_FLOOR",
    "RULE",
    "SHADOW",
    "SHADOW_TONE",
    "WORDMARK",
    "Layout",
    "Rows",
    "ThemeLike",
    "band",
    "render",
    "rows",
    "tones",
]
