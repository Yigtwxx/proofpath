"""Helpers more than one widget module draws with.

Nothing here is a widget: the layout primitives (:func:`_row`, :func:`_elide`), the
hyperlink builder and the accent rotation. They live apart so that ``run_block``,
``finding`` and ``footer`` can share them without importing each other.
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.color import Color

from proofpath import ui
from proofpath.tui.theme import Theme

#: Fallback width for a widget asked to render before it has been laid out.
DEFAULT_WIDTH = 80
#: Terminal width below which RICH stops drawing panels and their fixed-column
#: tables and falls back to PLAIN's rows (design section 4: "< 60 no borders").
#: Measured off the whole terminal, not the widget, so the border and the rows it
#: holds always agree and a border that changed the width cannot flip its own rows.
PANEL_FLOOR = 60
#: Longest reference or title a finding line prints before eliding it, and the
#: shortest it may be squeezed to before the line wraps instead. The verdict and its
#: tier are never the part that gives way: a state word that got cut off is a report
#: lying about what it found (rule 2).
LABEL_LIMIT = 34
MIN_LABEL = 12


def accent_for(run_id: int) -> str:
    """The accent of run ``run_id``: five colours, rotating (spec section 13.3)."""
    return ui.ACCENTS[(run_id - 1) % len(ui.ACCENTS)]


def panelled(theme: Theme, width: int) -> bool:
    """Whether ``theme`` draws panels and tables at a terminal ``width`` columns wide."""
    return theme.glyphs.box != "none" and width >= PANEL_FLOOR


def textual_colour(style: str) -> Color:
    """The colour of a theme style, spelled the way Textual's stylesheet wants it.

    A theme names colours in Rich's vocabulary (``cyan``, ``bold #22c55e``) because
    every line of content is a ``rich.Text``; the few places a colour reaches a
    Textual *style* -- a border, a button, the awaiting tint -- go through here. An
    ANSI name becomes ``ansi_<name>``, the terminal's own palette entry, so a PLAIN
    session keeps looking like its terminal; a truecolor tone is taken as it is.
    """
    colour = Style.parse(style).color
    if colour is None:
        return Color.parse("ansi_default")
    if colour.triplet is not None:
        return Color.parse(colour.triplet.hex)
    return Color.parse(f"ansi_{colour.name}")


def _elide(text: str, limit: int = LABEL_LIMIT) -> str:
    """Collapse whitespace and cut to ``limit`` columns, ellipsis included.

    A limit below two leaves nothing but the ellipsis, so it is floored there: a
    caller that has run out of room gets one character back, never a negative slice
    that would silently return the whole string again.
    """
    flat = " ".join(text.split())
    limit = max(limit, 2)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _link(text: str, url: str) -> Text:
    """``text`` as an OSC 8 hyperlink. Underlined, never coloured (spec section 13.3).

    Kept under ``--no-color`` and ``NO_COLOR`` on purpose: underline is not a colour and
    a hyperlink is not meaning, so switching colour off is not a reason to take the
    address away. A terminal that cannot follow the link shows the text unchanged.
    """
    return Text(text) if not url else Text(text, style=f"underline link {url}")


def _row(left: Text, middle: Text, right: Text, width: int, column: int) -> Text:
    """Lay three cells out: ``left``, ``middle`` from ``column``, ``right`` at the edge.

    Nothing is ever truncated — an attribution or a long path pushes the next cell
    right instead of being cut, exactly as the one-shot stage table behaves. The gaps
    shrink to one space before that happens, so the common case stays on one line.
    """
    line = left.copy()
    spare = width - len(left) - len(middle) - len(right)
    gap = max(column - len(left), 1 if spare < 4 else 2)
    line.append(" " * gap)
    line.append_text(middle)
    if right.plain:
        line.append(" " * max(width - len(line) - len(right), 1))
        line.append_text(right)
    return line
