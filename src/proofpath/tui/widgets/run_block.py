"""One run's block: its header and progress bar, the stage lines, and what came after.

In RICH the block is a panel (design section 4): the run's accent draws a rounded
border, the header's words move onto it -- the command as the title, the state as
the subtitle -- and the stages become a fixed-column table whose active row carries
the progress bar. PLAIN keeps the flat rows it has always drawn, header included, so
nothing regresses where borders cannot draw.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rich.text import Text
from textual import events as tevents
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.geometry import Size
from textual.markup import escape
from textual.widget import AwaitMount, Widget
from textual.widgets import Static

from proofpath import ui
from proofpath.events import Emitted, Event, Note, Progress, Prompted, StageEnd, StageStart
from proofpath.report import Footer, render_footer
from proofpath.tui.runs import Run
from proofpath.tui.theme import Meaning, Theme
from proofpath.tui.widgets._shared import DEFAULT_WIDTH, _elide, _row, panelled, textual_colour
from proofpath.tui.widgets.finding import FindingLine, FindingsRule, Line
from proofpath.tui.widgets.footer import coverage_text
from proofpath.tui.widgets.prompt import PermissionPrompt

#: Column the run header's state word starts at, measured off the spec's own block.
STATE_COLUMN = 47
#: Width of the PLAIN header's progress bar, and the two cells it is drawn from.
#: Twelve, as in the spec's own multi-run block: wide enough to read at a glance and
#: narrow enough that a long path plus a state word plus a bar still fits 80 columns.
BAR_WIDTH = 12
#: RICH (design section 4): the stage name is padded to this after its symbol, the
#: summary starts at the column after, the elapsed time is right-aligned in the last.
NAME_WIDTH = 12
SUMMARY_COLUMN = 16
ELAPSED_WIDTH = 6
#: The active stage's progress bar, in cells.
STAGE_BAR_WIDTH = 18
#: Shortest a stage's summary is squeezed to before the row stacks. The same bargain
#: as a finding's label: the summary elides, the attribution never does.
MIN_SUMMARY = 10
#: Column a stage's summary and its attribution share on the second line.
SUMMARY_INDENT = "  "
#: A summary that says something was *not* established: a count in the engine's own
#: words (``verify._resolve_summary`` and friends), or a stage that was not run at
#: all (``verify.NOT_ATTEMPTED``). A finished stage that reports one keeps the active
#: mark rather than a tick, so a glance down the table finds the stages that left
#: something unverified (rule 2, on screen).
HONESTY_COUNT = re.compile(
    r"\b[1-9]\d* "
    r"(?:unverified|ghost|amb(?:iguous)?|retracted|unavailable|not indexed|unresolved|NEI)\b"
    r"|^not attempted\b"
)
#: The state words a run's border subtitle draws, by the meaning each carries. The
#: two busy states take the run's accent instead (design section 4).
STATE_MEANINGS: dict[str, Meaning] = {
    "queued": "muted",
    "done": "ok",
    "cancelled": "muted",
    "failed": "finding",
}


def _style_of(out: ui.Ui, word: str) -> str:
    """The style ``ui`` gives ``word``, so a lowercase run state can borrow it.

    ``ui.style_state`` returns a coloured ``Text``; the run states are lowercase and
    not in its tables, but ``failed`` means exactly what ``FAILED`` means. Reading the
    style back off the probe keeps one module owning every colour, which copying
    ``"red"`` into this file would not.
    """
    styled = ui.style_state(out, word)
    return str(styled.spans[0].style) if styled.spans else ""


def _right(left: str, right: str, width: int) -> Text:
    """``left``, then ``right`` at the edge; one space between when they collide."""
    return Text(left + " " * max(width - len(left) - len(right), 1) + right)


def progress_bar(event: Progress, theme: Theme, accent: str, cells: int) -> Text:
    """``▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱  51/118``: ``done`` of ``total``, in the run's accent.

    The count is printed beside the bar because a bar alone rounds: seventeen cells
    cannot show 51 of 118, and the number is what the run actually knows.
    """
    total = max(event.total, 1)
    filled = int(cells * min(event.done, total) / total)
    bar = Text(theme.glyphs.bar_full * filled + theme.glyphs.bar_empty * (cells - filled))
    if accent:
        bar.stylize(accent)
    bar.append(f"  {event.done}/{event.total}")
    return bar


class RunHeader(Line):
    """``#n  /check target        state        progress or coverage``.

    The command line keeps the run's accent for the life of the session (spec
    section 13.1), which is what makes a log of several runs read as a sequence of
    coloured headings. In RICH the same words are drawn on the block's border and
    this row is hidden; it still answers for the block's fold and its copy.
    """

    def __init__(self, run: Run, accent: str, out: ui.Ui, theme: Theme) -> None:
        super().__init__()
        self.run = run
        self.accent = accent
        self._out = out
        self._theme = theme
        self._progress: Progress | None = None
        self._footer: Footer | None = None

    def action_expand(self) -> None:
        """A header toggles its whole block, which is what the spec makes it for.

        The header itself always stays: a collapsed run still has to show its state
        and its coverage, or a folded-away run would look like a run that never said
        anything (rule 6).
        """
        block = self.parent
        if isinstance(block, RunBlock):
            block.toggle()

    def copyable(self) -> str:
        return self.render().plain.strip()

    def set_progress(self, event: Progress) -> None:
        self._progress = event
        self.refresh()

    def set_footer(self, footer: Footer) -> None:
        self._footer = footer
        self._progress = None
        self.refresh()

    def render(self) -> Text:
        width = self.size.width or self.app.size.width or DEFAULT_WIDTH
        left = Text(f"#{self.run.id}  {self.run.command}")
        if self._out.color:
            left.stylize(self.accent)
        return _row(left, self._state(), self._tail(), width, STATE_COLUMN)

    def _state(self) -> Text:
        probe = {"failed": "FAILED", "cancelled": "cancelled"}.get(self.run.state)
        style = _style_of(self._out, probe) if probe is not None else ""
        return Text(self.run.state, style=style)

    def _tail(self) -> Text:
        if self.run.state == "failed" and self.run.error:
            return Text(self.run.error)
        if self._footer is not None:
            full, abstract, unverified = self._footer.coverage
            return Text(f"coverage {full}/{abstract}/{unverified}%")
        if self._progress is None or self._progress.total <= 0:
            return Text("")
        return progress_bar(
            self._progress, self._theme, self.accent if self._out.color else "", BAR_WIDTH
        )


class StageLine(Line):
    """One pipeline stage: its name, who produced it, and what it found.

    PLAIN draws two lines, because the spec's block is two lines: the bullet with
    the attribution and the elapsed time, then the summary underneath. RICH draws
    one row of the fixed-column table of design section 4, with the progress bar
    in the summary cell while the stage runs. No number is left unattributable
    (spec section 13.1) in either.
    """

    def __init__(self, start: StageStart, theme: Theme, accent: str, out: ui.Ui) -> None:
        super().__init__()
        self.stage = start.name
        self._by = start.by
        self._theme = theme
        self._accent = accent
        self._out = out
        self._summary = ""
        self._elapsed: float | None = None
        self._progress: Progress | None = None

    def copyable(self) -> str:
        return self.render().plain.strip()

    def set_progress(self, event: Progress) -> None:
        self._progress = event
        self.refresh()

    def finish(self, end: StageEnd) -> None:
        self._by = end.by or self._by
        self._summary = end.summary
        self._elapsed = end.elapsed
        self._progress = None
        self.refresh(layout=True)

    def render(self) -> Text:
        return self._draw(self.size.width or self.app.size.width or DEFAULT_WIDTH)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self._draw(width).plain.count("\n") + 1

    def _draw(self, width: int) -> Text:
        if panelled(self._theme, self.app.size.width):
            return self._draw_rich(width)
        return self._draw_plain(width)

    def _draw_plain(self, width: int) -> Text:
        """The bullet line, then the summary with its attribution beside it.

        The summary elides to make room; the attribution never does, and when there
        is no room left for both they stack. A stage whose provider got cut off is a
        number nobody can attribute, which spec section 13.1 does not allow.
        """
        bullet, _ = self._symbol()
        if self._elapsed is None:
            # Still running: the attribution is all there is to say about it yet.
            return _right(f"{bullet} {self.stage}", self._by, width)
        # Finished: the bullet line takes the elapsed time the spec's block puts
        # there and the attribution moves beside the summary.
        line = _right(f"{bullet} {self.stage}", f"{self._elapsed:.1f}s", width)
        if not self.expanded:
            # Collapsed: the stage, and how long it took. The detail is one click or
            # one ``enter`` away, and nothing about it was thrown away.
            return line
        budget = width - len(SUMMARY_INDENT) - 2 - len(self._by)
        if budget >= MIN_SUMMARY or not self._summary:
            summary = _elide(self._summary, budget) if self._summary else ""
            line.append("\n")
            line.append_text(_right(SUMMARY_INDENT + summary, self._by, width))
        else:
            summary = _elide(self._summary, max(width - len(SUMMARY_INDENT), MIN_SUMMARY))
            line.append("\n" + SUMMARY_INDENT + summary)
            line.append("\n")
            line.append_text(_right("", self._by, width))
        return line

    def _draw_rich(self, width: int) -> Text:
        """``✓ Parsing       24 pages · 42 refs                pymupdf  1.2s``.

        Symbol and name, the summary (or the progress bar while the stage runs), the
        attribution muted at the right, the elapsed time in the last six columns.
        The summary elides first; when even its floor does not fit beside the
        attribution, the attribution takes a row of its own rather than a cut.
        """
        muted = self._theme.tone("muted") if self._out.color else ""
        glyph, style = self._symbol()
        left = Text.assemble((glyph, style), " ", self.stage.ljust(NAME_WIDTH))
        by = Text(self._by, style=muted)
        if self._elapsed is None:
            middle = Text("")
            if self._progress is not None and self._progress.total > 0:
                accent = self._accent if self._out.color else ""
                middle = progress_bar(self._progress, self._theme, accent, STAGE_BAR_WIDTH)
            return _row(left, middle, by, width, SUMMARY_COLUMN)
        right = Text.assemble(by, "  ", f"{self._elapsed:.1f}s".rjust(ELAPSED_WIDTH))
        if not self.expanded:
            return _row(left, Text(""), right, width, SUMMARY_COLUMN)
        summary = self._summary.replace(", ", f" {self._theme.glyphs.sep} ")
        budget = width - max(SUMMARY_COLUMN, len(left) + 2) - 2 - len(right)
        if budget >= MIN_SUMMARY or not summary:
            return _row(left, Text(_elide(summary, budget)), right, width, SUMMARY_COLUMN)
        elapsed = Text(f"{self._elapsed:.1f}s".rjust(ELAPSED_WIDTH))
        budget = width - max(SUMMARY_COLUMN, len(left) + 2) - 2 - len(elapsed)
        summary = _elide(summary, max(budget, MIN_SUMMARY))
        line = _row(left, Text(summary), elapsed, width, SUMMARY_COLUMN)
        line.append("\n")
        line.append_text(_row(Text(""), Text(""), by, width, SUMMARY_COLUMN))
        return line

    def _symbol(self) -> tuple[str, str]:
        """The stage's mark and its style: running, finished, or finished with a caveat."""
        glyphs = self._theme.glyphs
        colour = self._out.color
        if self._elapsed is None:
            return glyphs.stage_active, self._accent if colour else ""
        if HONESTY_COUNT.search(self._summary):
            return glyphs.stage_flag, self._theme.tone("caution") if colour else ""
        return glyphs.stage_done, self._theme.tone("ok") if colour else ""


class NoteLine(Static):
    """Something the run said that is not a finding: a permission, a log line."""

    def __init__(self, text: str, theme: Theme, *, dim: bool = True, style: str = "") -> None:
        super().__init__()
        self._text = text
        self._style = style or (theme.tone("muted") if dim else "")

    def render(self) -> Text:
        return Text(f"  {self._text}", style=self._style)


class CoverageLine(Static):
    """A finished run's own coverage, kept in its block when the next run starts."""

    def __init__(self, footer: Footer, out: ui.Ui) -> None:
        super().__init__()
        self.footer = footer
        self._out = out

    def render(self) -> Text:
        return Text.assemble("  ", coverage_text(self.footer, self._out))


class KvLine(Static):
    """One ``key        value`` line, laid out the way ``ui.kv`` lays it out.

    The mirrored verbs print what the CLI prints, so the ten-column key and the bold
    it carries are read off ``ui`` rather than repeated here.
    """

    def __init__(self, text: Text) -> None:
        super().__init__()
        self._text = text

    def render(self) -> Text:
        return Text.assemble("  ", self._text)


class CommandBlock(Vertical):
    """One mirrored verb and what it answered: a block with no run number.

    ``/resolve`` and ``/fetch`` reach the network, so the block is mounted the moment
    the line is submitted and says it is working; the result replaces that. A block
    that only appeared when the answer did would leave the log looking as if the
    command had been swallowed.
    """

    def __init__(self, command: str, out: ui.Ui, theme: Theme) -> None:
        super().__init__(classes="run-block")
        self.command = command
        self._out = out
        self._working = Static(Text(f"  {command}", style=theme.tone("muted")))
        self._lines = Vertical(classes="stages")

    def compose(self) -> ComposeResult:
        yield self._working
        yield self._lines

    def show(self, lines: Iterable[Text]) -> None:
        """Replace "working" with the answer. Safe to call once."""
        self._working.update(Text(f"  {self.command}"))
        self._lines.mount_all([KvLine(line) for line in lines])

    def ask(self, prompt: PermissionPrompt) -> AwaitMount:
        """Where a ``/fetch`` that hit the section 7.1 wall puts its question. The
        mount is handed back so the app can scroll to it once it is on screen."""
        return self._lines.mount(prompt)


class RunBlock(Vertical):
    """One run's whole block: header, stages, then findings below a rule.

    Findings stream in while stages are still running, so the two live in separate
    containers: a stage line mounted after the first finding still lands above the
    rule, which is where the spec puts it. A finished run's coverage line is mounted
    beside them, not inside, so a folded block still shows it (rule 6).
    """

    def __init__(self, run: Run, out: ui.Ui, theme: Theme) -> None:
        super().__init__(classes=f"run-block {theme.name}")
        self.run = run
        self._theme = theme
        self.accent = theme.accent(run.id - 1)
        self._out = out
        self.header = RunHeader(run, self.accent, out, theme)
        self._stages = Vertical(classes="stages")
        self._findings = Vertical(classes="findings")
        self._rule = FindingsRule(theme)
        self._footer: Footer | None = None
        #: The finished run's coverage row. Mounted once the report is in and kept
        #: for the life of the block; shown only while the block is flat, because the
        #: panel's bottom border says the same thing and a run must say it once.
        self._coverage: CoverageLine | None = None
        self._lines: dict[str, StageLine] = {}
        #: Note lines that describe something still in progress (``loading models …``).
        #: Taken back by the next event that proves it is over -- a stage end, a
        #: finding, a lasting note -- or by the run settling. Never by ``Progress``:
        #: a bar can tick while a model is still being downloaded.
        self._transient: list[NoteLine] = []
        self._pending: list[tuple[Widget, Widget]] = []
        self._ready = False
        self._ruled = False
        self._panelled = False
        #: Whether the block is folded. The header is never part of the fold.
        self.collapsed = False
        #: True once ``add_summary`` has mounted this run's one summary line. One run
        #: gets one paragraph (spec section 11.1); a second would read as a second
        #: opinion about the same finished report.
        self.summarised = False

    @property
    def panelled(self) -> bool:
        """Whether the block is drawn as a bordered panel right now (design section 4)."""
        return self._panelled

    def compose(self) -> ComposeResult:
        yield self.header
        yield self._stages
        yield self._findings

    def on_mount(self) -> None:
        self._fit()
        self._ready = True
        for parent, child in self._pending:
            parent.mount(child)
        self._pending.clear()

    def on_resize(self) -> None:
        self._fit()
        # The title is padded to the frame, so a new width is a new title.
        self.refresh_state()

    def _fit(self) -> None:
        """Panel or flat rows, by the terminal's width. Idempotent, so resizes are cheap.

        The border is the run's accent for the life of the session, exactly as the
        PLAIN header's colour is; the header's words move onto it. Below the floor
        the border goes and the header row comes back, so a narrow terminal reads
        today's rows rather than a panel squeezed around them.
        """
        wide = panelled(self._theme, self.app.size.width)
        if wide == self._panelled:
            return
        self._panelled = wide
        self.header.display = not wide
        if self._coverage is not None:
            self._coverage.display = not wide
        self.set_class(not wide, "narrow")
        if wide:
            # Without colour the frame is drawn in the terminal's default, which is
            # still a frame; a border in the background colour would be an empty box.
            colour = textual_colour(self.accent if self._out.color else "")
            self.styles.border = (self._theme.glyphs.box, colour)
        else:
            self.styles.border = None
        self.refresh_state()

    def on_click(self, event: tevents.Click) -> None:
        """A click on the panel's top border folds the run, as one on the header does.

        The header row is hidden in RICH, so its click has to live somewhere: the
        title it became. Clicks on the lines inside never reach here -- each line
        stops its own -- so this only ever sees the border and the padding.
        """
        if self.panelled and event.screen_y == self.region.y:
            self.toggle()

    # --- what the scheduler's events do to it ---------------------------------------

    def handle(self, event: Event) -> None:
        if isinstance(event, StageStart):
            line = StageLine(event, self._theme, self.accent, self._out)
            self._lines[event.name] = line
            self._add(self._stages, line)
        elif isinstance(event, Progress):
            self.header.set_progress(event)
            active = self._lines.get(event.name)
            if active is not None:
                active.set_progress(event)
        elif isinstance(event, StageEnd):
            self._settle_transient()
            # A stage that ended without ever starting still gets its line: an event
            # stream with a gap in it must not lose what the stage said it found.
            ended = self._lines.get(event.name)
            if ended is None:
                ended = StageLine(
                    StageStart(event.name, event.by), self._theme, self.accent, self._out
                )
                self._lines[event.name] = ended
                self._add(self._stages, ended)
            ended.finish(event)
        elif isinstance(event, Note):
            note = NoteLine(event.text, self._theme)
            if event.transient:
                self._transient.append(note)
            else:
                self._settle_transient()
            self._add(self._stages, note)
        elif isinstance(event, Emitted):
            self._settle_transient()
            if not self._ruled:
                self._ruled = True
                self._add(self._findings, self._rule)
            self._add(self._findings, FindingLine(event.finding, self._out, self._theme))
        elif isinstance(event, Prompted):
            # Informational: the gate has already asked, inline, through ``ask()``.
            # It is still shown, because a run that was asked something and a run that
            # was not are different runs (rule 6).
            self._add(self._stages, NoteLine(event.text, self._theme))

    def ask(self, prompt: PermissionPrompt) -> AwaitMount | None:
        """Mount the section 7.1 question under the stage that ran into it. The mount
        is handed back so the app can scroll to it once it is on screen; ``None``
        when the block itself is not on screen yet and the question is queued."""
        return self._add(self._stages, prompt)

    def toggle(self) -> None:
        """Fold the run away, or open it again. The frame always stays (rule 6).

        The PLAIN header and the RICH border both carry the state and, once the run
        is over, its coverage, so a folded run still says what it found.
        """
        self.collapsed = not self.collapsed
        self._stages.display = not self.collapsed
        self._findings.display = not self.collapsed

    def refresh_state(self) -> None:
        """The run's state changed: redraw the header, and the frame it became.

        The state word sits at the right of the top border (design section 4): the
        title is the command, a fill in the border's colour, then the word, padded to
        the frame's inner width so the word ends where the border's own dash begins.
        The command elides before the word ever does. A finished run's coverage takes
        the bottom border, where the PLAIN header's tail would have put it.
        """
        self.header.refresh()
        if not self.panelled:
            return
        word = self.run.state
        meaning = STATE_MEANINGS.get(word)
        style = self._theme.tone(meaning) if meaning else self.accent
        state = f"[{style}]{escape(word)}[/]" if self._out.color else escape(word)
        # Textual draws the title over the content's columns less one each side.
        inner = self.content_region.width - 2
        command = f"#{self.run.id}  {self.run.command}"
        if inner > 0:
            # Cut, not ``_elide``: the two spaces after ``#n`` are the header's own.
            limit = max(inner - len(word) - 2, 2)
            if len(command) > limit:
                command = command[: limit - 1] + "…"
            fill = self._theme.glyphs.title_fill * max(inner - len(command) - len(word) - 2, 0)
            self.border_title = f"{escape(command)} {fill} {state}"
        else:
            self.border_title = f"{escape(command)} {state}"
        self.border_subtitle = self._coverage_markup()

    def _coverage_markup(self) -> str:
        """``coverage 62/21/17%`` for the bottom border, each share in its meaning."""
        if self._footer is None:
            return ""
        parts = []
        for value, name in zip(self._footer.coverage, ui.COVERAGE_STYLES, strict=True):
            style = self._theme.tones[name]
            parts.append(f"[{style}]{value}[/]" if self._out.color else str(value))
        return "coverage " + "/".join(parts) + "%"

    def settle(self) -> None:
        """The run reached a terminal state: keep its own coverage line in place."""
        # Whatever was still "in progress" is over with the run, however it ended.
        self._settle_transient()
        self.refresh_state()
        if self.panelled and self.run.state == "failed" and self.run.error:
            # The PLAIN header prints the error in its tail; the panel has no header
            # row, so the error gets a line of its own, in the meaning's colour.
            style = self._theme.tone("finding") if self._out.color else ""
            self._add(self._stages, NoteLine(self.run.error, self._theme, dim=False, style=style))
        if self.run.report is None:
            return
        self._footer = render_footer(self.run.report)
        self.header.set_footer(self._footer)
        self.refresh_state()
        # Always mounted, so a resize across the floor in either direction finds the
        # row to show or hide; ``_fit`` keeps it and the border from both showing.
        self._coverage = CoverageLine(self._footer, self._out)
        self._coverage.display = not self.panelled
        self._add(self._findings, self._coverage)

    def add_summary(self, line: Text) -> None:
        """The model-written summary, at the foot of the run it summarises.

        It belongs to this run and nothing else, so it goes in this run's block rather
        than in a block of its own -- and last, where the markdown report puts it.
        """
        self.summarised = True
        self._add(self._findings, KvLine(line))

    def _add(self, parent: Widget, child: Widget) -> AwaitMount | None:
        """Mount ``child``, or queue it when this block is not on screen yet."""
        if self._ready:
            return parent.mount(child)
        self._pending.append((parent, child))
        return None

    def _settle_transient(self) -> None:
        """Take back the transient note lines: what they described is over."""
        for line in self._transient:
            if line.is_attached:
                line.remove()
            else:
                # Queued before the block was mounted: never shown, so never mounted.
                self._pending = [(p, c) for p, c in self._pending if c is not line]
        self._transient.clear()
