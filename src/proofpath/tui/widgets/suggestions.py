"""The list above the bar: what a ``/`` line may still become (wordmark design section 10)."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from rich.text import Text
from textual.content import Content
from textual.style import Style
from textual.widgets import Static

from proofpath.tui.commands import DESCRIPTIONS, NEEDS_ARGUMENT
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import DEFAULT_WIDTH
from proofpath.tui.widgets.prompt import tinted

#: Columns between the completion and the placeholder, and between the placeholder
#: and the description: each column is as wide as its widest entry plus this.
GAP = 2
#: What ``plain`` puts in front of the selected row and in front of the others. Two
#: cells each, so the rows line up whichever one is selected; the marker sits under
#: the caret's column and the completion under the bar's text.
MARKER = "> "
NO_MARKER = "  "
#: How many lines an indicator costs the window: always one, whichever theme draws
#: it (its text is far short of wrapping at any width a terminal offers).
INDICATOR_LINES = 1


class Suggestions(Static):
    """One row per candidate, the selected one tinted (RICH) or marked (PLAIN).

    Fed by the app, never by the bar: :meth:`show` takes the candidates the bar's
    cycle holds and which of them is selected, and :meth:`hide` takes the list away.
    The widget decides nothing about *what* is listed -- ``commands.complete`` does,
    through the bar -- it only lays the three columns out: the completion as ``Tab``
    would type it, the verb's placeholder (empty for an answer such as ``/allow
    once``) and the verb's one-line description.

    The selected row is the bar's own accent at :data:`~.prompt.TINT_ALPHA`, blended
    over the surface the way the awaiting bar is, so the list and the bar read as one
    thing. Without colour the marker says which row it is, whichever theme is on:
    a tint that cannot be drawn would leave the selection invisible.

    The list takes its rows from the log, never from the banner or the bar: the app
    says how many lines the screen has left (``limit``), and on a terminal too short
    for every row the list shows a run of them around the selected one, sliding as
    the selection moves, the way a list view scrolls. A row is never cut short; one
    that wraps counts for the lines it takes. When the cap hides rows above or below
    the window, an ``... n more`` line takes their place at that end -- itself
    counted against the cap, so the window gives up a row to make room for it.
    """

    def __init__(self, *, theme: Theme, coloured: bool) -> None:
        super().__init__(id="suggestions")
        self._theme = theme
        self._coloured = coloured
        self._accent = theme.accent(0)
        self._selected = 0
        #: What was last shown, so a list that only moved its selection keeps its
        #: window and a new one starts at its top.
        self._candidates: tuple[str, ...] = ()
        #: The held rows, as text, built when they are shown: every candidate, in
        #: full, regardless of what the window ends up drawing. The tint and the
        #: padding to the width are the render's, the words are not.
        self._lines: tuple[Text, ...] = ()
        #: How many lines the list may take, or ``None`` for all it needs; and the
        #: first row on show when it may not take them all.
        self._limit: int | None = None
        self._first = 0
        #: The window as last computed by :meth:`show` or :meth:`on_resize`: never
        #: recomputed by ``render``, which only draws it.
        self._window: tuple[int, int] = (0, 0)
        #: Whether the top and bottom ``... n more`` indicators are drawn for the
        #: current window. Rows are hidden (``first > 0`` / ``last < len(lines)``)
        #: more often than an indicator is actually drawn for them: at ``limit == 1``
        #: there is no room for a row and an indicator together, so neither end gets
        #: one, and at ``limit == 2`` with both ends hidden only one indicator fits,
        #: so the bottom keeps it. Set by :meth:`_settle_window`, read by
        #: :meth:`render` and :attr:`drawn` instead of the raw window bounds.
        self._indicators: tuple[bool, bool] = (False, False)

    @property
    def visible(self) -> bool:
        return self.display

    @property
    def held(self) -> tuple[str, ...]:
        """The text of every candidate row, in full, whether or not it is drawn."""
        return tuple(line.plain for line in self._lines)

    @property
    def drawn(self) -> tuple[str, ...]:
        """The text of every line actually rendered: the window's rows, plus any
        ``... n more`` indicator the cap made room for."""
        first, last = self._window
        show_above, show_below = self._indicators
        lines = [line.plain for line in self._lines[first:last]]
        if show_above:
            lines.insert(0, self._indicator_text(first))
        if show_below:
            lines.append(self._indicator_text(len(self._lines) - last))
        return tuple(lines)

    @property
    def window(self) -> tuple[int, int]:
        """The candidate rows on show, as ``(first, past-the-last)``: every row when
        they all fit. Set by :meth:`show` and :meth:`on_resize`; see there."""
        return self._window

    @property
    def tinted(self) -> bool:
        """Whether the selection is a tint rather than a marker: RICH, with colour."""
        return self._coloured and self._theme.name == "rich"

    def set_accent(self, accent: str) -> None:
        """The tint follows the caret: the current run's accent, set by the app."""
        self._accent = accent
        self.refresh()

    def show(self, candidates: Sequence[str], selected: int, *, limit: int | None = None) -> None:
        """Draw ``candidates`` with ``selected`` marked, in at most ``limit`` lines."""
        if tuple(candidates) != self._candidates:
            self._first = 0  # a new list starts at its top
        self._candidates = tuple(candidates)
        self._selected = selected
        self._limit = limit
        self._lines = tuple(
            self._line(cells, index == selected)
            for index, cells in enumerate(self._columns(candidates))
        )
        self._window = self._settle_window()
        # The limit is also the stylesheet's: a row the count below misjudged (a
        # word wrap the arithmetic did not see) is clipped rather than let push the
        # dock past the screen.
        self.styles.max_height = limit
        self.display = True
        # A layout refresh, not a repaint: the rows change the widget's height and
        # the container above the bar grows or shrinks with it (the log gives way).
        self.refresh(layout=True)

    def hide(self) -> None:
        self._candidates = ()
        self._lines = ()
        self._first = 0
        self._window = (0, 0)
        self._indicators = (False, False)
        self.display = False
        self.refresh(layout=True)

    def _settle_window(self) -> tuple[int, int]:
        """Where the window lands for the rows and the limit :meth:`show` just set.

        Slides only as far as the selection asks: up to it when it went above the
        window, down until it fits when it went below, and not at all otherwise, so
        the rows hold still under a selection that moves inside them. Past the last
        row it slides back up instead, so room that opens (a taller terminal) is
        filled rather than left under the list.

        An indicator line at either end is counted against the same limit: the
        window is first fit assuming neither is needed, and, if that leaves rows
        hidden at an end, refit with the limit cut by one (or two) to make room for
        the line that now says so. Whether an end is hidden cannot get *less* true as
        the limit shrinks, so this settles in at most one more pass per end.

        Two limits are too tight for that scheme to draw what it promises. At
        ``limit == 1`` there is room for the selected row alone -- a row and an
        indicator both would already overrun it -- so neither end gets one however
        many rows sit outside the window; :attr:`drawn` and :meth:`render` agree
        because they read :attr:`_indicators`, not the window bounds. At
        ``limit == 2`` with rows hidden at both ends, only one indicator fits; the
        bottom one stays, so a selection that just scrolled off the top is not the
        one line that vanishes behind it.
        """
        self._indicators = (False, False)
        if self._limit is None or not self._lines:
            return 0, len(self._lines)
        width = self._width()
        heights = [max(1, ceil(line.cell_len / width)) for line in self._lines]
        reserve = 0
        show_above, show_below = False, False
        first, last = 0, len(heights)
        for _ in range(3):  # top and bottom can each add their reserve once
            first, last = self._fit(heights, reserve)
            hidden_above = first > 0
            hidden_below = last < len(heights)
            if self._limit <= 1:
                # No room for the row and either indicator: draw the row alone.
                show_above, show_below = False, False
            elif self._limit == 2 and hidden_above and hidden_below:
                show_above, show_below = False, True
            else:
                show_above, show_below = hidden_above, hidden_below
            needed = (INDICATOR_LINES if show_above else 0) + (INDICATOR_LINES if show_below else 0)
            if needed == reserve:
                break
            reserve = needed
        self._first = first  # where the window settled is where the next slide starts
        self._indicators = (show_above, show_below)
        return first, last

    def _fit(self, heights: list[int], reserve: int) -> tuple[int, int]:
        assert self._limit is not None
        limit = max(self._limit - reserve, 1)
        first = min(self._first, self._selected)
        while first < self._selected and sum(heights[first : self._selected + 1]) > limit:
            first += 1
        last = self._selected + 1
        while last < len(heights) and sum(heights[first : last + 1]) <= limit:
            last += 1
        while last == len(heights) and first > 0 and sum(heights[first - 1 : last]) <= limit:
            first -= 1
        return first, last

    def render(self) -> Content:
        lines: list[Content] = []
        width = self._width()
        first, last = self._window
        show_above, show_below = self._indicators
        if show_above:
            lines.append(self._indicator_content(first))
        for index in range(first, last):
            line = Content.from_rich_text(self._lines[index])
            if self.tinted and index == self._selected:
                # Out to the edge, so the tint is a bar and not a ragged wash; the
                # rows themselves are never cut (a long one wraps, as any line does).
                line = line.pad_right(max(width - len(line), 0))
                line = line.stylize(Style(background=tinted(self._accent)))
            lines.append(line)
        if show_below:
            lines.append(self._indicator_content(len(self._lines) - last))
        return Content("\n").join(lines)

    def _indicator_text(self, hidden: int) -> str:
        # RICH draws its own ellipsis glyph; PLAIN stays ASCII-only, like every other
        # glyph it draws (``theme.unicode`` is what tells the two apart elsewhere).
        ellipsis = "…" if self._theme.unicode else "..."
        return f"{ellipsis} {hidden} more"

    def _indicator_content(self, hidden: int) -> Content:
        text = Text(self._indicator_text(hidden), style=self._theme.tone("muted"))
        return Content.from_rich_text(text)

    def _width(self) -> int:
        # The content width once laid out; before that (the first show, while the
        # widget is still ``display: none``) the container's, less this widget's own
        # gutter, which is what the layout is about to give it.
        width = self.size.width
        if not width and self.parent is not None:
            width = self.parent.size.width - self.styles.gutter.width
        return width or DEFAULT_WIDTH

    def _line(self, cells: tuple[str, str, str], selected: bool) -> Text:
        completion, placeholder, description = cells
        text = Text()
        if not self.tinted:
            text.append(MARKER if selected else NO_MARKER)
        # The completion in the text tone -- no style, the terminal's own -- and
        # everything after it muted, so the eye lands on what ``Tab`` would type.
        text.append(completion)
        text.append(placeholder + description, style=self._theme.tone("muted"))
        return text

    @staticmethod
    def _columns(candidates: Sequence[str]) -> list[tuple[str, str, str]]:
        """Each candidate's three cells, padded so the columns line up.

        The placeholder column goes entirely when no candidate has one (the four
        answers, the cancellable runs), rather than leaving an empty column between
        the completion and the description.
        """
        cells: list[tuple[str, str, str]] = []
        for candidate in candidates:
            verb, _, rest = candidate[1:].partition(" ")
            # ``/check `` is the verb with the space ``Tab`` adds; ``/allow once`` is
            # an answer. Only the bare verb shows its placeholder.
            placeholder = "" if rest else NEEDS_ARGUMENT.get(verb, "")
            cells.append((candidate.rstrip(), placeholder, DESCRIPTIONS.get(verb, "")))
        if not cells:
            return []
        completion_width = max(len(completion) for completion, _, _ in cells) + GAP
        placeholder_width = max(len(placeholder) for _, placeholder, _ in cells)
        if placeholder_width:
            placeholder_width += GAP
        return [
            (completion.ljust(completion_width), placeholder.ljust(placeholder_width), description)
            for completion, placeholder, description in cells
        ]

    def on_resize(self) -> None:
        # The tinted row is padded to the width, so a resize redraws the rows; the
        # layout is asked again because a row that wrapped may no longer, or may now
        # -- which is also why the window is settled again here rather than left to
        # ``render`` (see the note on ``_window``).
        if self._lines:
            self._window = self._settle_window()
        self.refresh(layout=True)
