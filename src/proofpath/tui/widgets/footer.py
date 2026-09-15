"""The docked coverage footer (rule 6) and the coverage cell it shares with a block."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from proofpath import ui
from proofpath.report import BROWSER_SKIPPED_REASON, Footer, Report, no_bibliography, render_footer
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import DEFAULT_WIDTH, _elide

#: Rows the docked footer occupies in each theme. PLAIN: counts and coverage, what
#: the run cost, and the caveats line (empty when the run owes none). RICH puts the
#: coverage bar on a row of its own above those three (design section 4).
FOOTER_HEIGHT = 3
RICH_FOOTER_HEIGHT = 4
#: The coverage bar's cells, and the fewest it is drawn with before it is dropped
#: for the legend alone: a bar too short to show a share would only look like one.
BAR_CELLS = 30
MIN_BAR = 10
#: Every row starts two columns in, matching the run blocks.
INDENT = "  "
#: The three shares, in the order the bar and the legend draw them (``ui`` fixes
#: the colours and the report fixes the wording).
LEGEND = ("full text", "abstract", "unverified")


class CoverageFooter(Static):
    """The latest finished run's counts and coverage. Docked; never scrolls (rule 6)."""

    def __init__(self, out: ui.Ui, theme: Theme) -> None:
        super().__init__(id="footer")
        self._out = out
        self._theme = theme
        self._footer: Footer | None = None

    @property
    def rows(self) -> int:
        """How many rows this theme's footer takes; the app docks it at exactly that."""
        return RICH_FOOTER_HEIGHT if self._theme.name == "rich" else FOOTER_HEIGHT

    def show(self, report: Report) -> None:
        self._footer = render_footer(report)
        self.refresh()

    def render(self) -> Text:
        """Always :attr:`rows` lines, whatever the run had to say.

        Fixed rather than ``auto`` because the footer is docked: a docked widget's
        height is settled when the screen is arranged, and a footer that grew after
        that would be drawn over the prompt instead of pushing it up. Rule 6 is not
        something to leave to a re-layout that may or may not happen.
        """
        width = self.size.width or self.app.size.width or DEFAULT_WIDTH
        if self._footer is None:
            blank = "\n" * (self.rows - 1)
            return Text(
                f"{INDENT}no run yet · coverage is reported per run{blank}",
                style=self._theme.tone("muted"),
            )
        item = self._footer
        # Every line is cut to the width rather than left to wrap: the height is
        # fixed, and a wrapped first line would push the caveats row off the screen.
        if self._theme.name == "rich":
            line = Text(INDENT)
            line.append_text(coverage_bar(item, self._out, self._theme, width - len(INDENT)))
            line.append("\n" + INDENT + _elide(item.counts, width - 4))
        else:
            # The coverage cell is measured first and keeps its room; the counts give way.
            coverage = coverage_text(item, self._out)
            line = Text.assemble(
                INDENT, _elide(item.counts, width - len(coverage) - 8), "      ", coverage
            )
        written = f"{item.written} written" if item.written else "no report written"
        cost = f"{written}  ·  {item.api_calls} API calls  ·  {item.elapsed:.1f}s"
        line.append("\n" + INDENT + _elide(cost, width - 4))
        # Every caveat the run owes the reader, on one line and never dropped: a
        # low-coverage run must not look like a clean one, and the one place that
        # cannot scroll away is this one.
        line.append(
            "\n" + INDENT + _elide(" · ".join(_hints(item)), width - 4),
            style=self._theme.tone("muted"),
        )
        return line


def coverage_text(item: Footer, out: ui.Ui) -> Text:
    """``coverage 62/21/17%``, the three numbers coloured as ``ui`` colours them."""
    text = Text("coverage ")
    for index, (value, style) in enumerate(zip(item.coverage, ui.COVERAGE_STYLES, strict=True)):
        if index:
            text.append("/")
        text.append(str(value), style=style if out.color else "")
    text.append("%")
    return text


def coverage_bar(item: Footer, out: ui.Ui, theme: Theme, width: int) -> Text:
    """``████████▓▓▓░░  █ 62% full text · ▓ 21% abstract · ░ 17% unverified``.

    The bar is :data:`BAR_CELLS` wide when the row has room, shrinks to
    :data:`MIN_BAR` as the width does, and is dropped for the legend alone below
    that; the legend itself gives way last, to ``coverage 62/21/17%``. The numbers
    are the report's own (``Coverage.pct``): the bar rounds its cells to them, never
    the other way round, so what is drawn never disagrees with what is written.
    """
    styles = [theme.tones[name] if out.color else "" for name in ui.COVERAGE_STYLES]
    glyphs = (theme.glyphs.cov_full, theme.glyphs.cov_abstract, theme.glyphs.cov_unverified)
    legend = Text()
    for index, (value, label) in enumerate(zip(item.coverage, LEGEND, strict=True)):
        if index:
            legend.append(f" {theme.glyphs.sep} ")
        # The swatch says which cells are which without a key elsewhere.
        legend.append(f"{glyphs[index]} {value}%", style=styles[index])
        legend.append(f" {label}")
    cells = min(BAR_CELLS, width - len(legend) - 2)
    if cells < MIN_BAR:
        return legend if len(legend) <= width else coverage_text(item, out)
    bar = Text()
    for value, glyph, style in zip(_cells(item.coverage, cells), glyphs, styles, strict=True):
        bar.append(glyph * value, style=style)
    # A run that counted nothing (no bibliography: 0/0/0) leaves the track empty
    # rather than painting one share over the whole bar.
    bar.append(theme.glyphs.rule * (cells - len(bar)), style=theme.tone("muted"))
    return Text.assemble(bar, "  ", legend)


def _cells(shares: tuple[int, int, int], cells: int) -> tuple[int, ...]:
    """Split ``cells`` between the shares in proportion; rounding goes where it hurts least.

    Percentages that add up to 100 fill the bar exactly; a run whose shares do not
    (the 0/0/0 of a document with no bibliography) leaves the rest of it empty.
    """
    total = sum(shares)
    if total <= 0:
        return (0,) * len(shares)
    exact = [cells * share / 100 for share in shares]
    rounded = [int(value) for value in exact]
    # Hand the leftover cells to the shares that lost the most in rounding, so a
    # 100 % total always fills the bar and a smaller one never overfills it.
    leftover = min(cells, round(cells * total / 100)) - sum(rounded)
    for index in sorted(range(len(shares)), key=lambda i: exact[i] - rounded[i], reverse=True):
        if leftover <= 0:
            break
        rounded[index] += 1
        leftover -= 1
    return tuple(rounded)


def _hints(item: Footer) -> list[str]:
    """Everything a run still owes the reader about its own coverage (rule 6)."""
    hints: list[str] = []
    if item.unchecked_markers:
        hints.append(no_bibliography(item.unchecked_markers))
    elif item.weak:
        hints.append(ui.WEAK_COVERAGE)
    if item.browser_skipped:
        hints.append(f"{item.browser_skipped} source(s) {BROWSER_SKIPPED_REASON}")
    if item.note:
        hints.append(item.note)
    return hints
