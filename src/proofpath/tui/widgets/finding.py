"""A finding under the rule: the line, its badge and the passage it rests on."""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import events as tevents
from textual.binding import Binding, BindingType
from textual.geometry import Size
from textual.widgets import Static

from proofpath import ui
from proofpath.report import Finding
from proofpath.resolve import strip_marker
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import (
    DEFAULT_WIDTH,
    MIN_LABEL,
    _elide,
    _link,
    _row,
    panelled,
)

#: PLAIN: column a finding's reference label starts at, under the locator, and where
#: it hangs when the row had to be split in two.
LABEL_COLUMN = 15
LABEL_INDENT = 7
#: RICH (design section 4): the location is a fixed cell after the symbol, so every
#: label starts at one column whatever the page number; the hang is the same seven.
LOCATION_WIDTH = 12
RICH_LABEL_COLUMN = 16
#: RICH: the tier's cell after the badge, so every badge ends on one column whether
#: the finding carries ``high``, ``medium`` or no tier at all.
TIER_WIDTH = 6
#: Where the quoted passage hangs under its finding.
PASSAGE_INDENT = 7
#: PLAIN prints a finding's notes as the CLI does (``report.render_diagnostics``).
NOTE_KEY = "= note: "
#: RICH detail rows name what they quote: the claim, then the passage it was checked
#: against. Both keys are padded to one width so the two quotes line up.
YOU = "you"
SOURCE = "source"
QUOTE_KEY_WIDTH = 8


def _wrap(text: str, width: int) -> list[str]:
    """Break ``text`` into rows of at most ``width`` columns, on spaces where it can.

    A word longer than the whole row is cut rather than allowed to overflow — a URL
    in a quoted passage is the usual one — because the caller's height is computed
    from the rows returned here and a row that overflows would be drawn over.
    """
    rows: list[str] = []
    row = ""
    for word in text.split():
        candidate = f"{row} {word}" if row else word
        if len(candidate) <= width:
            row = candidate
            continue
        if row:
            rows.append(row)
        while len(word) > width:
            rows.append(word[:width])
            word = word[width:]
        row = word
    if row:
        rows.append(row)
    return rows or [""]


def _source_url(source_id: str | None) -> str:
    """The address a source id points at, or ``""`` when it points at nothing.

    Spec section 13.1 makes a source a terminal hyperlink; a cache key like
    ``doi:10.1/x`` is not one until it is turned back into the address it came from.
    """
    if not source_id:
        return ""
    if source_id.startswith(("http://", "https://")):
        return source_id
    if source_id.startswith("doi:"):
        return "https://doi.org/" + source_id[4:]
    if source_id.startswith("arxiv:"):
        return "https://arxiv.org/abs/" + source_id[6:]
    return ""


def level_glyph(theme: Theme, level: str) -> str:
    """The symbol a severity gets on its line: the theme's, never spelled here."""
    return {"error": theme.glyphs.error, "warning": theme.glyphs.warning}.get(
        level, theme.glyphs.note
    )


def state_text(theme: Theme, out: ui.Ui, word: str) -> Text:
    """A state word as the theme draws it: a badge in RICH, a coloured word in PLAIN.

    The meaning comes from ``ui``'s tables through the theme; nothing here decides
    what a word means, only how the decided meaning looks. Without colour the word
    is plain, as ``ui.style_state`` leaves it.
    """
    if not out.color:
        return Text(word)
    badge = theme.badge_style(word)
    if badge:
        return Text(f" {word} ", style=badge)
    text = Text(word)
    if theme.style_for(word):
        text.stylize(theme.style_for(word))
    return text


class Line(Static):
    """A log line the mouse and the keyboard both reach (spec section 13.1).

    The rule the spec sets is that nothing is mouse-only and nothing is
    keyboard-only, so every clickable line is also focusable and carries the two
    bindings that do the same things a click and the copy glyph do. Subclasses say
    what toggling means for them and what there is to copy; a line with nothing to
    copy answers ``""`` and the binding does nothing rather than copying a blank.
    """

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        # Not ``toggle``: ``DOMNode.action_toggle`` already means "flip a reactive"
        # and shadowing it would break every binding that uses the real one.
        Binding("enter", "expand", "expand or collapse"),
        Binding("c", "copy", "copy"),
        # Design section 2.4: the arrows walk the log's lines the way they walk the
        # bar's history; ``shift`` with them scrolls, and that pair is the app's.
        Binding("up", "neighbour(-1)", "previous line", show=False),
        Binding("down", "neighbour(1)", "next line", show=False),
    ]

    #: Whether the line's detail is showing. What "detail" means is the subclass's.
    expanded = True

    def on_click(self, event: tevents.Click) -> None:
        """The one click handler these lines have.

        It is deliberately not overridden anywhere: Textual dispatches a handler once
        per class in the MRO that defines one, so a subclass with its own ``on_click``
        would have this one run as well and undo it. Subclasses say where the copy
        glyph is instead, through :meth:`hit_copy`.
        """
        # Focus follows the click, so the keyboard picks up where the mouse left off
        # and ``enter`` toggles the line the user just pointed at.
        self.focus()
        event.stop()
        if self.hit_copy(event.x, event.y):
            self.action_copy()
        else:
            self.action_expand()

    def hit_copy(self, x: int, y: int) -> bool:
        """Whether ``(x, y)`` landed on this line's copy glyph. Most lines have none."""
        return False

    def action_expand(self) -> None:
        self.expanded = not self.expanded
        self.refresh(layout=True)

    def action_copy(self) -> None:
        text = self.copyable()
        if text:
            self.app.copy_to_clipboard(text)

    def copyable(self) -> str:
        """What ``c`` and the copy glyph put on the clipboard."""
        return ""

    def action_neighbour(self, step: int) -> None:
        """Focus the previous (``-1``) or next (``1``) displayed line; the ends stay put.

        The screen's focus chain already skips everything inside a collapsed block
        (``display: none``), so a folded run is one line to step over, not many.
        """
        lines = [w for w in self.screen.focus_chain if isinstance(w, Line)]
        try:
            index = lines.index(self)
        except ValueError:  # pragma: no cover - a line that is not displayed has no focus
            return
        target = index + step
        if 0 <= target < len(lines):
            lines[target].focus()

    # A printable key on a focused line is typing and goes to the bar; that is the
    # app's ``on_key`` (wordmark design section 11), one rule in one place.


class FindingsRule(Static):
    """The rule the findings sit below (spec section 13.1). Structure, not meaning."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self._theme = theme

    def render(self) -> Text:
        width = self.size.width or self.app.size.width or DEFAULT_WIDTH
        return Text(self._theme.glyphs.rule * max(width, 1), style=self._theme.tone("muted"))


class FindingLine(Line):
    """``x  p.4 L112  [12] Zhang 2021        GHOST REFERENCE  high``, plus the passage.

    The state word is coloured by meaning (``ui``'s tables, through the theme) and
    the passage is printed under it: a finding that asserts something owes its
    evidence (rule 1), and the display is not where that gets dropped. In RICH the
    word is a badge, the location is a fixed cell and the finding's notes and the
    claim it checked are printed under it too (design section 4); PLAIN keeps the
    flat row it has always drawn.
    """

    #: A passage starts elided to one line; a click or ``enter`` opens it in full.
    expanded = False

    def __init__(self, finding: Finding, out: ui.Ui, theme: Theme) -> None:
        super().__init__()
        self.finding = finding
        self._out = out
        self._theme = theme

        #: Where the copy glyph was last drawn, as ``(row, column)``. A click on it
        #: copies the passage instead of toggling it, which is what the glyph is for.
        self._copy_cell: tuple[int, int] | None = None

    def copyable(self) -> str:
        """The quoted passage, which is the evidence and the only thing worth taking."""
        return self._passage() or ""

    def hit_copy(self, x: int, y: int) -> bool:
        cell = self._copy_cell
        if cell is None:
            return False
        row, column = cell
        return y == row and column <= x < column + len(self._theme.glyphs.copy)

    def render(self) -> Text:
        return self._draw(self.size.width or self.app.size.width or DEFAULT_WIDTH)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self._draw(width).plain.count("\n") + 1

    def _draw(self, width: int) -> Text:
        """Lay the line out at ``width``, giving room up in a fixed order.

        The reference label is the only part that shrinks, and when even its floor
        does not fit, the verdict moves to its own row rather than being cut. A
        truncated ``NOT SUPPORTED`` or a missing tier would misstate the finding, and
        no terminal width makes that acceptable (rules 1 and 2).
        """
        item = self.finding
        self._copy_cell = None
        rich = panelled(self._theme, self.app.size.width)
        marker = Text(level_glyph(self._theme, item.level))
        if self._out.color:
            marker.stylize(self._theme.tones[ui.LEVEL_STYLES[item.level]])
        if rich:
            left = Text.assemble(marker, " ", item.locator.label().ljust(LOCATION_WIDTH))
            column = RICH_LABEL_COLUMN
        else:
            left = Text.assemble(marker, "  ", item.locator.label())
            column = LABEL_COLUMN
        right = state_text(self._theme, self._out, item.state)
        if rich:
            right = Text.assemble(right, "  ", (item.tier or "").ljust(TIER_WIDTH))
        elif item.tier is not None:
            right = Text.assemble(right, f"  {item.tier}")

        url = _source_url(item.source_id)
        budget = width - max(column, len(left) + 2) - 2 - len(right)
        if rich and budget < MIN_LABEL and item.tier is None:
            # The tier cell is empty: give its columns to the label before stacking,
            # so a tier-less badge at 60-79 columns keeps its row. The badges no
            # longer end on one column at that width, which is the cheaper loss.
            right = state_text(self._theme, self._out, item.state)
            budget = width - max(column, len(left) + 2) - 2 - len(right)
        if budget >= MIN_LABEL:
            middle = _link(_elide(self._label(), budget), url)
            line = _row(left, middle, right, width, column)
        else:
            # Too narrow for three cells: the verdict keeps the first row, the label
            # takes the second. Nothing is dropped, it is only stacked.
            line = Text.assemble(left, " " * max(width - len(left) - len(right), 1), right)
            label = _elide(self._label(), max(width - LABEL_INDENT, MIN_LABEL))
            line.append("\n" + " " * LABEL_INDENT)
            line.append_text(_link(label, url))
        if rich:
            self._rich_detail(line, width)
        else:
            self._plain_detail(line, width)
        return line

    def _plain_detail(self, line: Text, width: int) -> None:
        """The passage, elided until opened, the copy glyph after; then the notes.

        The notes are the CLI's own ``= note:`` lines: a ghost's reason (the DOI that
        resolves to nothing) is part of the finding, and a display that kept only the
        state word would make the reader take it on trust.
        """
        passage = self._passage()
        if passage is not None:
            self._quote(line, width, "", passage, copyable=True)
        self._notes(line, width, NOTE_KEY)

    def _rich_detail(self, line: Text, width: int) -> None:
        """The notes, then ``you`` / ``source`` (design section 4), all muted."""
        self._notes(line, width, "")
        passage = self._passage()
        if passage is None:
            return
        claim = self.finding.claim
        if claim is not None:
            self._quote(line, width, YOU.ljust(QUOTE_KEY_WIDTH), claim.text, copyable=False)
        self._quote(line, width, SOURCE.ljust(QUOTE_KEY_WIDTH), passage, copyable=True)

    def _notes(self, line: Text, width: int, key: str) -> None:
        """One muted row per note, elided until the finding is opened, wrapped after."""
        muted = self._theme.tone("muted")
        hang = " " * PASSAGE_INDENT + key
        room = max(width - len(hang), MIN_LABEL)
        for note in self.finding.detail:
            rows = _wrap(note, room) if self.expanded else [_elide(note, room)]
            for index, row in enumerate(rows):
                line.append("\n" + (hang if index == 0 else " " * len(hang)) + row, style=muted)

    def _quote(self, line: Text, width: int, key: str, text: str, *, copyable: bool) -> None:
        """One quoted row under the finding, wrapped when opened and elided until then.

        The whole passage when opened, wrapped rather than cut: what the finding rests
        on is the one thing a display may not shorten on the user (rule 1). The copy
        glyph sits after the first row and its cell is remembered for the mouse.
        """
        muted = self._theme.tone("muted")
        glyph = self._theme.glyphs.copy
        hang = " " * PASSAGE_INDENT + key
        room = max(width - len(hang) - len(glyph) - 3, MIN_LABEL)
        rows = _wrap(f'"{text}"', room) if self.expanded else [f'"{_elide(text, room)}"']
        for index, row in enumerate(rows):
            line.append("\n" + hang + row, style=muted)
            if index == 0 and copyable:
                self._copy_cell = (line.plain.count("\n"), len(hang) + len(row) + 1)
                line.append(f" {glyph}", style=muted)
            # Continuation rows hang under the quote, not under its key.
            hang = " " * len(hang)

    def _passage(self) -> str | None:
        """The evidence the finding rests on (rule 1); the display never drops it."""
        verdict = self.finding.verdict
        if verdict is None or verdict.passage is None:
            return None
        return verdict.passage.text

    def _label(self) -> str:
        reference = self.finding.reference
        if reference is not None:
            # ``raw`` keeps the entry's printed marker (Phase 5); the row prints the
            # number itself, so the marker is stripped or the label reads ``[7] [7] …``.
            return f"[{reference.number}] {_elide(strip_marker(reference.raw))}"
        return _elide(self.finding.title)
