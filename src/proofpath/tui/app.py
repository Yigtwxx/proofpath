"""The streaming TUI itself: banner, run blocks, coverage footer (spec section 13.1).

This module holds no pipeline logic (spec section 13.3). It parses a line with
:mod:`~proofpath.tui.commands`, hands the target to
:class:`~proofpath.tui.runs.Scheduler`, and turns the events that come back into
widgets. Every decision about *what* a run found was made before it got here.

Two rules shape the layout rather than taste. Rule 6: the coverage footer is docked
and never scrolls, so a thin run cannot be mistaken for a clean one however far the
log has run on. Rule 4: a run's engine is built with ``interactive=True`` and an
inline prompt of this app's own, so the section 7.1 question is asked *in the block
where it happened* — a gate that fell back to ``ask_terminal`` would read a terminal
Textual is already holding and deadlock the app instead of asking anybody anything.
The answer travels back over a ``concurrent.futures.Future``, which is what lets the
worker thread block on a question the event loop is drawing.

The mirrored verbs (spec section 13.3) hold no more logic than the rest: ``/resolve``,
``/fetch``, ``/config`` and ``/cache`` call :mod:`proofpath.commands`, the same
functions ``cli.py`` calls, on a worker thread, and this module only turns what comes
back into lines.

Colour comes from :mod:`proofpath.ui` and nowhere else. Widget content is built as
``rich.Text`` with ANSI-16 style names (``cyan``, ``bright_magenta``, ``red``), which
is the same vocabulary ``ui``'s tables are written in; the one place a colour becomes
a Textual style — the awaiting bar's tint — spells it ``ansi_<name>``, because that is
what Textual calls the terminal's own palette entry.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import ClassVar, Protocol

from rich.text import Text
from textual import events as tevents
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.color import Color
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.geometry import Size
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from proofpath import __version__, device, ui
from proofpath import commands as library
from proofpath.browser import Answer, ConsentGate, prompt_text
from proofpath.cache import SourceDetail, SourceSummary
from proofpath.config import Config, ConfigError, load_config
from proofpath.events import Emitted, Event, Note, Progress, Prompted, StageEnd, StageStart
from proofpath.fetch import STEP_NAMES, Fetched
from proofpath.judge import JudgeError
from proofpath.paths import config_path
from proofpath.report import (
    BROWSER_SKIPPED_REASON,
    Finding,
    Footer,
    Report,
    no_bibliography,
    render_footer,
)
from proofpath.resolve import strip_marker
from proofpath.tui import banner, commands
from proofpath.tui.runs import EngineFactory as RunsEngineFactory
from proofpath.tui.runs import Run, Scheduler
from proofpath.verify import Engine

#: Line 4 of the banner, joined here because only the app knows which commands it
#: offers. Matches the spec's own 80-column block, down to the spacing.
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"
#: What the input bar says when it is not waiting for a verb's argument.
DEFAULT_PLACEHOLDER = "paste a file path, a URL, or a claim"
#: The spec's "~15-20%" tint, at the quiet end: an accent on a bar, not a highlight.
TINT_ALPHA = 0.15

#: The CLI verbs the TUI mirrors (spec section 13.3): each one calls the same
#: :mod:`proofpath.commands` function ``cli.py`` calls, on a worker thread.
MIRRORED = ("resolve", "fetch", "config", "cache")
#: ``/summarize`` is the one verb that is still nothing but a promise (spec 13.1).
V03_NOTE = "arrives in v0.3"

#: Which verbs hold the bar is the parser's decision, not this module's; it is
#: re-exported here because the app is where the mode is entered and left.
AWAITING_VERBS = commands.AWAITING_VERBS

#: The four answers of spec section 7.1, in the order the prompt draws them, with the
#: label each button carries. A click is worth exactly the ``/allow`` line beside it.
ANSWER_LABELS: dict[Answer, str] = {
    "once": "allow once",
    "always": "always",
    "no": "no",
    "never": "never",
}
#: Prefix of a permission button's id, so one handler reads every one of them.
ANSWER_ID = "allow-"
#: What the mouse copies and ``c`` copies: the glyph the spec puts after a passage.
COPY_GLYPH = "⧉"
#: The run states the ferret watches the path in (spec section 13.1).
BUSY_STATES = frozenset({"running", "verifying"})
#: The eyes, in seconds: how often they are re-read, how long a blink lasts, the
#: window a blink falls in, and how long a finished run's expression holds.
EYES_TICK = 0.05
BLINK_SECONDS = 0.15
BLINK_WINDOW = (6.0, 10.0)
FLASH_SECONDS = 2.0

#: Column the run header's state word starts at, measured off the spec's own block.
STATE_COLUMN = 47
#: Width of the header's progress bar, and the two cells it is drawn from. Twelve,
#: as in the spec's own multi-run block: wide enough to read at a glance and narrow
#: enough that a long path plus a state word plus a bar still fits 80 columns.
BAR_WIDTH = 12
BAR_FULL = "█"
BAR_EMPTY = "░"
#: The prompt glyph of the spec's own block. Deliberately not ">": the bar is a
#: prompt, not a quoted line.
PROMPT_CARET = "\u203a"
#: A stage's bullet, and the marker each severity gets on a finding line.
STAGE_BULLET = "⏺"
LEVEL_MARKERS = {"error": "✗", "warning": "⚠", "note": "·"}
#: Fallback width for a widget asked to render before it has been laid out.
DEFAULT_WIDTH = 80
#: Rows the docked footer always occupies: counts and coverage, what the run cost,
#: and the caveats line (empty when the run owes none).
FOOTER_HEIGHT = 3
#: Longest reference or title a finding line prints before eliding it, and the
#: shortest it may be squeezed to before the line wraps instead. The verdict and its
#: tier are never the part that gives way: a state word that got cut off is a report
#: lying about what it found (rule 2).
LABEL_LIMIT = 34
MIN_LABEL = 12
#: The same bargain on a stage line: the summary elides, the attribution never does.
MIN_SUMMARY = 10
#: Column a finding's reference label starts at, under the locator, and where it
#: hangs when the row had to be split in two.
LABEL_COLUMN = 15
LABEL_INDENT = 7
#: Where the quoted passage hangs under its finding.
PASSAGE_INDENT = 7
#: Column a stage's summary and its attribution share on the second line.
SUMMARY_INDENT = "  "


def accent_for(run_id: int) -> str:
    """The accent of run ``run_id``: five colours, rotating (spec section 13.3)."""
    return ui.ACCENTS[(run_id - 1) % len(ui.ACCENTS)]


def run_context(config: Config) -> str:
    """Line 3 of the banner: provider, whether the network is open, and the device."""
    online = "online" if config.permissions.network == "allow" else "offline"
    return f"academic . {online} . {_device_name()}"


def _device_name() -> str:
    """The ONNX device, or ``cpu`` when the runtime cannot say (CUDA -> CoreML -> CPU)."""
    try:
        return device.onnx_device_name()
    except Exception:  # pragma: no cover - onnxruntime is a hard dependency
        return "cpu"


def _style_of(out: ui.Ui, word: str) -> str:
    """The style ``ui`` gives ``word``, so a lowercase run state can borrow it.

    ``ui.style_state`` returns a coloured ``Text``; the run states are lowercase and
    not in its tables, but ``failed`` means exactly what ``FAILED`` means. Reading the
    style back off the probe keeps one module owning every colour, which copying
    ``"red"`` into this file would not.
    """
    styled = ui.style_state(out, word)
    return str(styled.spans[0].style) if styled.spans else ""


def _elide(text: str, limit: int = LABEL_LIMIT) -> str:
    """Collapse whitespace and cut to ``limit`` columns, ellipsis included.

    A limit below two leaves nothing but the ellipsis, so it is floored there: a
    caller that has run out of room gets one character back, never a negative slice
    that would silently return the whole string again.
    """
    flat = " ".join(text.split())
    limit = max(limit, 2)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


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


def _right(left: str, right: str, width: int) -> Text:
    """``left``, then ``right`` at the edge; one space between when they collide."""
    return Text(left + " " * max(width - len(left) - len(right), 1) + right)


class SchedulerLike(Protocol):
    """What the app needs of a scheduler, so a test can hand it a fake one."""

    @property
    def runs(self) -> tuple[Run, ...]: ...

    def submit(self, target: str, *, command: str) -> Run: ...

    def cancel(self, run_id: int) -> bool: ...

    async def close(self) -> None: ...


SchedulerFactory = Callable[..., SchedulerLike]
#: Re-exported, never redeclared: the scheduler calls the factory with the run it is
#: building for, and a second spelling here would let a zero-argument factory type-check
#: against an app that then hands it a ``Run`` at runtime.
EngineFactory = RunsEngineFactory


# --- widgets ------------------------------------------------------------------------


class Banner(Static):
    """The ferret, pinned above the log and redrawn on every resize.

    The stamp is the only coloured element in it (spec section 13.1) and the red
    comes from ``ui.STAMP_COLOUR``; nothing else here names a colour.

    The eyes are the only thing in the TUI that moves. They are driven by one timer
    that exists at all only when the terminal reports colour and ``-q`` was not
    asked for (spec section 13.1): a log being piped or read by a screen reader gets
    a still banner rather than four characters rewriting themselves forever.
    """

    def __init__(
        self,
        *,
        version: str,
        context: str,
        hint: str,
        coloured: bool,
        animated: bool = False,
    ) -> None:
        super().__init__(id="banner")
        self._version = version
        self._banner_context = context
        self._hint = hint
        self._coloured = coloured
        self._animated = animated
        #: The last drawing, so a caller (and a test) can read the exact lines back.
        self.drawn: banner.Banner | None = None
        #: The eyes as last chosen, and the timer that chooses them (``None`` when the
        #: animation is off -- which is how a test, and a reader, can tell).
        self.eyes = banner.EYES["idle"]
        self.timer: Timer | None = None
        #: Read through an attribute so a test can hand the widget a clock of its own
        #: and need not wait two real seconds for an expression to expire.
        self.clock: Callable[[], float] = time.monotonic
        # Seeded from the wall clock, so two sessions on one screen do not blink in
        # lockstep; the schedule itself is just "somewhere in the next 6-10 seconds".
        self._random = random.Random(time.time())
        self._busy = False
        self._mood: str | None = None
        self._mood_until = 0.0
        self._blink_at = 0.0
        self._blink_until = 0.0

    # --- what the app tells it ------------------------------------------------------

    def start_animation(self) -> None:
        """Begin blinking, if this session animates at all. Idempotent."""
        if not self._animated or self.timer is not None:
            return
        self._blink_at = self.clock() + self._gap()
        self.timer = self.set_interval(EYES_TICK, self.tick)

    def set_context(self, context: str) -> None:
        """Line 3 changed under it — ``permissions.network``, and so online/offline."""
        if context != self._banner_context:
            self._banner_context = context
            self.refresh(layout=True)

    def set_busy(self, busy: bool) -> None:
        """``(>.>)`` while any run is fetching or verifying; idle again after."""
        if not self._animated or busy == self._busy:
            return
        self._busy = busy
        self.tick()

    def flash(self, mood: str) -> None:
        """``(O.O)`` for two seconds on findings, ``(^.^)`` for two on a clean run."""
        if not self._animated:
            return
        self._mood, self._mood_until = mood, self.clock() + FLASH_SECONDS
        self.tick()

    def tick(self) -> None:
        """Re-read the state and repaint only when the eyes actually changed."""
        eyes = self._choose()
        if eyes != self.eyes:
            self.eyes = eyes
            self.refresh()

    def _choose(self) -> str:
        """The expression this instant calls for. Advances the blink schedule.

        Order matters: a finished run's two seconds outrank everything, then a run
        that is still working, and only an idle ferret blinks. A blink owed from a
        stretch where the eyes were busy is dropped rather than fired the moment the
        run ends, which would read as part of the result.
        """
        now = self.clock()
        if self._mood is not None:
            if now < self._mood_until:
                return banner.EYES[self._mood]
            self._mood = None
        if self._busy:
            self._blink_at = now + self._gap()
            return banner.EYES["busy"]
        if now < self._blink_until:
            return banner.EYES["blink"]
        if now >= self._blink_at:
            self._blink_until = now + BLINK_SECONDS
            self._blink_at = now + self._gap()
            return banner.EYES["blink"]
        return banner.EYES["idle"]

    def _gap(self) -> float:
        return self._random.uniform(*BLINK_WINDOW)

    def render(self) -> Text:
        drawn = self._draw(self.size.width or self.app.size.width or DEFAULT_WIDTH)
        text = Text("\n".join(drawn.lines))
        if drawn.stamp_span is not None and self._coloured:
            offset = len(drawn.lines[0]) + 1
            start, end = drawn.stamp_span
            text.stylize(ui.STAMP_COLOUR, offset + start, offset + end)
        return text

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """Four drawn lines, but more than four rows once one of them wraps.

        ``banner.render`` never truncates: below about 60 columns the version and the
        run context stop fitting side by side and the hint line is wider still, so the
        container is sized to what was actually drawn. A banner clipped to four rows
        would hide the line that says which commands exist.
        """
        return sum(max(1, -(-len(line) // max(width, 1))) for line in self._draw(width).lines)

    def _draw(self, width: int) -> banner.Banner:
        self.drawn = banner.render(
            width,
            version=self._version,
            context=self._banner_context,
            hint=self._hint,
            eyes=self.eyes,
        )
        return self.drawn

    def on_mount(self) -> None:
        self.start_animation()

    def on_resize(self) -> None:
        # The height is width-dependent, so a resize is a re-layout and not a repaint.
        self.refresh(layout=True)


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
        """What ``c`` and the ``⧉`` glyph put on the clipboard."""
        return ""


class RunHeader(Line):
    """``#n  /check target        state        progress or coverage``.

    The command line keeps the run's accent for the life of the session (spec
    section 13.1), which is what makes a log of several runs read as a sequence of
    coloured headings.
    """

    def __init__(self, run: Run, accent: str, out: ui.Ui) -> None:
        super().__init__()
        self.run = run
        self.accent = accent
        self._out = out
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
        done = min(self._progress.done, self._progress.total)
        filled = int(BAR_WIDTH * done / self._progress.total)
        bar = Text(BAR_FULL * filled + BAR_EMPTY * (BAR_WIDTH - filled))
        if self._out.color:
            bar.stylize(self.accent)
        bar.append(f"  {self._progress.done}/{self._progress.total}")
        return bar


class StageLine(Line):
    """One pipeline stage: its name, who produced it, and what it found.

    Two lines, because the spec's block is two lines: the bullet with the
    attribution and the elapsed time, then the summary underneath. No number is left
    unattributable (spec section 13.1).
    """

    def __init__(self, start: StageStart) -> None:
        super().__init__()
        self.stage = start.name
        self._by = start.by
        self._summary = ""
        self._elapsed: float | None = None

    def copyable(self) -> str:
        return self.render().plain.strip()

    def finish(self, end: StageEnd) -> None:
        self._by = end.by or self._by
        self._summary = end.summary
        self._elapsed = end.elapsed
        self.refresh()

    def render(self) -> Text:
        return self._draw(self.size.width or self.app.size.width or DEFAULT_WIDTH)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self._draw(width).plain.count("\n") + 1

    def _draw(self, width: int) -> Text:
        """The bullet line, then the summary with its attribution beside it.

        The summary elides to make room; the attribution never does, and when there
        is no room left for both they stack. A stage whose provider got cut off is a
        number nobody can attribute, which spec section 13.1 does not allow.
        """
        if self._elapsed is None:
            # Still running: the attribution is all there is to say about it yet.
            return _right(f"{STAGE_BULLET} {self.stage}", self._by, width)
        # Finished: the bullet line takes the elapsed time the spec's block puts
        # there and the attribution moves beside the summary.
        line = _right(f"{STAGE_BULLET} {self.stage}", f"{self._elapsed:.1f}s", width)
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


class NoteLine(Static):
    """Something the run said that is not a finding: a permission, a log line."""

    def __init__(self, text: str, *, dim: bool = True) -> None:
        super().__init__()
        self._text = text
        self._dim = dim

    def render(self) -> Text:
        return Text(f"  {self._text}", style="dim" if self._dim else "")


class PermissionPrompt(Vertical):
    """The section 7.1 question, asked where it happened (spec section 13.1).

    Never a modal: it is mounted inside the block of the run that hit the wall, under
    the stage that hit it, and the log keeps scrolling around it. The four buttons are
    worth exactly the four ``/allow`` answers, because they resolve the same future —
    a terminal without mouse reporting loses nothing (spec section 13.1's mouse rule).

    The container is vertical although the buttons are a row: the question is the ten
    lines ``browser.prompt_text`` writes, and laying those out *beside* four buttons
    would leave neither readable at 80 columns. The row itself is the ``Horizontal``.

    Nothing here decides anything. The answer goes back to the gate, which is the only
    thing that may act on it, and the install still happens behind it (rule 5).
    """

    def __init__(self, owner: int, host: str, status: int | None, out: ui.Ui) -> None:
        super().__init__(classes="permission")
        #: The run this question belongs to, or the negative id of a ``/fetch`` block.
        self.owner = owner
        #: The section 7.1 block, verbatim: the terminal and the TUI ask the same
        #: question, in the same words, about the same download.
        self.question = prompt_text(host, status)
        self._out = out
        #: The answer once it was given, so a scrolled-back log still says what was
        #: allowed and what was not (rule 6).
        self.answer: Answer | None = None
        self._answered = Static("", classes="answered")

    def compose(self) -> ComposeResult:
        yield Static(self._question())
        with Horizontal(classes="answers"):
            for answer, label in ANSWER_LABELS.items():
                # A ``Text``, not a string: Textual reads ``[...]`` in a string label as
                # its own content markup and the brackets the spec draws would vanish.
                yield Button(Text(f"[{label}]"), id=f"{ANSWER_ID}{answer}", classes="answer")
        yield self._answered

    def on_mount(self) -> None:
        self._answered.display = False
        if not self._out.color:
            return
        # Yellow is what section 13.3 gives the permission prompt: caution, not a
        # finding. ``ui`` names the colour; this only spells it the way Textual does.
        colour = Color.parse(f"ansi_{ui.PROMPT_COLOUR}")
        for button in self.query(Button):
            button.styles.color = colour

    def _question(self) -> Text:
        text = Text(self.question)
        if self._out.color:
            text.stylize(ui.PROMPT_COLOUR)
        return text

    def settle(self, answer: Answer) -> None:
        """Record the answer and take the buttons away: it is asked once per run."""
        self.answer = answer
        for button in self.query(Button):
            button.disabled = True
        self.query_one(".answers", Horizontal).display = False
        self._answered.update(Text(f"    answered: {ANSWER_LABELS[answer]}", style="dim"))
        self._answered.display = True


class FindingsRule(Static):
    """The rule the findings sit below (spec section 13.1). Structure, not meaning."""

    def render(self) -> Text:
        width = self.size.width or self.app.size.width or DEFAULT_WIDTH
        return Text("─" * max(width, 1), style="dim")


class FindingLine(Line):
    """``x  p.4 L112  [12] Zhang 2021        GHOST REFERENCE  high``, plus the passage.

    The state word is coloured by ``ui.style_state`` and the passage is printed under
    it: a finding that asserts something owes its evidence (rule 1), and the display
    is not where that gets dropped.
    """

    #: A passage starts elided to one line; a click or ``enter`` opens it in full.
    expanded = False

    def __init__(self, finding: Finding, out: ui.Ui) -> None:
        super().__init__()
        self.finding = finding
        self._out = out

        #: Where the ``⧉`` glyph was last drawn, as ``(row, column)``. A click there
        #: copies the passage instead of toggling it, which is what the glyph is for.
        self._copy_cell: tuple[int, int] | None = None

    def copyable(self) -> str:
        """The quoted passage, which is the evidence and the only thing worth taking."""
        return self._passage() or ""

    def hit_copy(self, x: int, y: int) -> bool:
        cell = self._copy_cell
        return cell is not None and (y, x) == cell

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
        marker = Text(LEVEL_MARKERS[item.level])
        if self._out.color:
            marker.stylize(ui.LEVEL_STYLES[item.level])
        left = Text.assemble(marker, "  ", item.locator.label())
        right = ui.style_state(self._out, item.state)
        if item.tier is not None:
            right = Text.assemble(right, f"  {item.tier}")

        url = _source_url(item.source_id)
        column = max(LABEL_COLUMN, len(left) + 2)
        budget = width - column - 2 - len(right)
        if budget >= MIN_LABEL:
            middle = _link(_elide(self._label(), budget), url)
            line = _row(left, middle, right, width, LABEL_COLUMN)
        else:
            # Too narrow for three cells: the verdict keeps the first row, the label
            # takes the second. Nothing is dropped, it is only stacked.
            line = Text.assemble(left, " " * max(width - len(left) - len(right), 1), right)
            label = _elide(self._label(), max(width - LABEL_INDENT, MIN_LABEL))
            line.append("\n" + " " * LABEL_INDENT)
            line.append_text(_link(label, url))
        passage = self._passage()
        if passage is not None:
            room = max(width - PASSAGE_INDENT - len(COPY_GLYPH) - 3, MIN_LABEL)
            hang = " " * PASSAGE_INDENT
            if self.expanded:
                # The whole passage, wrapped rather than cut: what the finding rests on
                # is the one thing a display may not shorten on the user (rule 1).
                for index, row in enumerate(_wrap(f'"{passage}"', room)):
                    line.append("\n" + hang + row, style="dim")
                    if index == 0:
                        self._copy_cell = (line.plain.count("\n"), PASSAGE_INDENT + len(row) + 1)
                        line.append(f" {COPY_GLYPH}", style="dim")
            else:
                shown = f'"{_elide(passage, room)}"'
                line.append("\n" + hang + shown, style="dim")
                self._copy_cell = (line.plain.count("\n"), PASSAGE_INDENT + len(shown) + 1)
                line.append(f" {COPY_GLYPH}", style="dim")
        return line

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


class CoverageLine(Static):
    """A finished run's own coverage, kept in its block when the next run starts."""

    def __init__(self, footer: Footer, out: ui.Ui) -> None:
        super().__init__()
        self.footer = footer
        self._out = out

    def render(self) -> Text:
        return Text.assemble("  ", _coverage_text(self.footer, self._out))


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


def _kv(out: ui.Ui, key: str, value: str | Text, *, state: bool = False) -> Text:
    """``ui.kv``'s line, built as a ``Text`` instead of printed to a console."""
    if isinstance(value, str):
        value = ui.style_state(out, value) if state else Text(value)
    return Text.assemble((key.ljust(ui.KEY_WIDTH) + " ", "bold" if out.color else ""), value)


class CommandBlock(Vertical):
    """One mirrored verb and what it answered: a block with no run number.

    ``/resolve`` and ``/fetch`` reach the network, so the block is mounted the moment
    the line is submitted and says it is working; the result replaces that. A block
    that only appeared when the answer did would leave the log looking as if the
    command had been swallowed.
    """

    def __init__(self, command: str, out: ui.Ui) -> None:
        super().__init__(classes="run-block")
        self.command = command
        self._out = out
        self._working = Static(Text(f"  {command}", style="dim"))
        self._lines = Vertical(classes="stages")

    def compose(self) -> ComposeResult:
        yield self._working
        yield self._lines

    def show(self, lines: Iterable[Text]) -> None:
        """Replace "working" with the answer. Safe to call once."""
        self._working.update(Text(f"  {self.command}"))
        self._lines.mount_all([KvLine(line) for line in lines])

    def ask(self, prompt: PermissionPrompt) -> None:
        """Where a ``/fetch`` that hit the section 7.1 wall puts its question."""
        self._lines.mount(prompt)


class RunBlock(Vertical):
    """One run's whole block: header, stages, then findings below a rule.

    Findings stream in while stages are still running, so the two live in separate
    containers: a stage line mounted after the first finding still lands above the
    rule, which is where the spec puts it.
    """

    def __init__(self, run: Run, out: ui.Ui) -> None:
        super().__init__(classes="run-block")
        self.run = run
        self.accent = accent_for(run.id)
        self._out = out
        self.header = RunHeader(run, self.accent, out)
        self._stages = Vertical(classes="stages")
        self._findings = Vertical(classes="findings")
        self._rule = FindingsRule()
        self._lines: dict[str, StageLine] = {}
        self._pending: list[tuple[Widget, Widget]] = []
        self._ready = False
        self._ruled = False
        #: Whether the block is folded. The header is never part of the fold.
        self.collapsed = False

    def compose(self) -> ComposeResult:
        yield self.header
        yield self._stages
        yield self._findings

    def on_mount(self) -> None:
        self._ready = True
        for parent, child in self._pending:
            parent.mount(child)
        self._pending.clear()

    # --- what the scheduler's events do to it ---------------------------------------

    def handle(self, event: Event) -> None:
        if isinstance(event, StageStart):
            line = StageLine(event)
            self._lines[event.name] = line
            self._add(self._stages, line)
        elif isinstance(event, Progress):
            self.header.set_progress(event)
        elif isinstance(event, StageEnd):
            # A stage that ended without ever starting still gets its line: an event
            # stream with a gap in it must not lose what the stage said it found.
            ended = self._lines.get(event.name)
            if ended is None:
                ended = StageLine(StageStart(event.name, event.by))
                self._lines[event.name] = ended
                self._add(self._stages, ended)
            ended.finish(event)
        elif isinstance(event, Note):
            self._add(self._stages, NoteLine(event.text))
        elif isinstance(event, Emitted):
            if not self._ruled:
                self._ruled = True
                self._add(self._findings, self._rule)
            self._add(self._findings, FindingLine(event.finding, self._out))
        elif isinstance(event, Prompted):
            # Informational: the gate has already asked, inline, through ``ask()``.
            # It is still shown, because a run that was asked something and a run that
            # was not are different runs (rule 6).
            self._add(self._stages, NoteLine(event.text))

    def ask(self, prompt: PermissionPrompt) -> None:
        """Mount the section 7.1 question under the stage that ran into it."""
        self._add(self._stages, prompt)

    def toggle(self) -> None:
        """Fold the run away, or open it again. The header always stays (rule 6)."""
        self.collapsed = not self.collapsed
        self._stages.display = not self.collapsed
        self._findings.display = not self.collapsed

    def settle(self) -> None:
        """The run reached a terminal state: keep its own coverage line in place."""
        self.header.refresh()
        if self.run.report is None:
            return
        footer = render_footer(self.run.report)
        self.header.set_footer(footer)
        self._add(self._findings, CoverageLine(footer, self._out))

    def _add(self, parent: Widget, child: Widget) -> None:
        """Mount ``child``, or queue it when this block is not on screen yet."""
        if self._ready:
            parent.mount(child)
        else:
            self._pending.append((parent, child))


class RunLog(VerticalScroll):
    """The scrolling log: one :class:`RunBlock` per run, newest last."""


class CoverageFooter(Static):
    """The latest finished run's counts and coverage. Docked; never scrolls (rule 6)."""

    def __init__(self, out: ui.Ui) -> None:
        super().__init__(id="footer")
        self._out = out
        self._footer: Footer | None = None

    def show(self, report: Report) -> None:
        self._footer = render_footer(report)
        self.refresh()

    def render(self) -> Text:
        """Always :data:`FOOTER_HEIGHT` lines, whatever the run had to say.

        Fixed rather than ``auto`` because the footer is docked: a docked widget's
        height is settled when the screen is arranged, and a footer that grew after
        that would be drawn over the prompt instead of pushing it up. Rule 6 is not
        something to leave to a re-layout that may or may not happen.
        """
        width = self.size.width or self.app.size.width or DEFAULT_WIDTH
        if self._footer is None:
            return Text("  no run yet · coverage is reported per run\n\n", style="dim")
        item = self._footer
        # Every line is cut to the width rather than left to wrap: the height is
        # fixed, and a wrapped first line would push the caveats row off the screen.
        # The coverage cell is measured first and keeps its room; the counts give way.
        coverage = _coverage_text(item, self._out)
        line = Text.assemble(
            "  ", _elide(item.counts, width - len(coverage) - 8), "      ", coverage
        )
        written = f"{item.written} written" if item.written else "no report written"
        cost = f"{written}  ·  {item.api_calls} API calls  ·  {item.elapsed:.1f}s"
        line.append("\n  " + _elide(cost, width - 4))
        # Every caveat the run owes the reader, on one line and never dropped: a
        # low-coverage run must not look like a clean one, and the one place that
        # cannot scroll away is this one.
        line.append("\n  " + _elide(" · ".join(_hints(item)), width - 4), style="dim")
        return line


def _coverage_text(item: Footer, out: ui.Ui) -> Text:
    """``coverage 62/21/17%``, the three numbers coloured as ``ui`` colours them."""
    text = Text("coverage ")
    for index, (value, style) in enumerate(zip(item.coverage, ui.COVERAGE_STYLES, strict=True)):
        if index:
            text.append("/")
        text.append(str(value), style=style if out.color else "")
    text.append("%")
    return text


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


# --- the mirrored verbs: run the library function, render what it answered -----------
#
# Everything below runs on a worker thread and returns lines. The calls go to
# ``proofpath.commands``, which is what ``cli.py`` calls, so the two surfaces cannot
# drift about what a verb does -- only about how it is drawn, which is all these are.


def _error_line(out: ui.Ui, text: str) -> Text:
    """``ui.error``'s line, as a widget's ``Text``. Red is a meaning, and ``ui`` owns it."""
    return Text.assemble(("error: ", ui.LEVEL_STYLES["error"] if out.color else ""), text)


def verb_lines(
    command: commands.Command,
    *,
    out: ui.Ui,
    config: Config,
    prompt: library.PromptFn | None = None,
) -> list[Text]:
    """Run one mirrored verb and render its result. Worker thread; never prints."""
    verb, arg = command.verb, command.arg
    if verb == "resolve":
        return _resolve_lines(out, library.resolve_reference(arg, config=config))
    if verb == "fetch":
        fetched = library.fetch_target(
            arg,
            config=config,
            # The TUI is a terminal and has a prompt of its own, so ``ask`` really can
            # be asked here -- of the block, never of stdin (rule 4).
            interactive=True,
            prompt=prompt,
        )
        return _fetch_lines(out, fetched)
    if verb == "config":
        return _config_lines(out, arg)
    return _cache_lines(out, arg)


def _resolve_lines(out: ui.Ui, resolved: library.Resolved) -> list[Text]:
    """What ``proofpath resolve`` prints, line for line (spec section 13.3)."""
    result, retraction = resolved.result, resolved.retraction
    lines = [_kv(out, "state", result.state.value, state=True)]
    best = result.best
    if best is not None:
        url = ("https://doi.org/" + best.doi) if best.doi else best.url
        lines.append(_kv(out, "record", best.title))
        byline = f"{best.first_author} {best.year or '?'} · {best.venue or '—'}"
        lines.append(_kv(out, "", f"{byline} · via {best.provider}"))
        lines.append(_kv(out, "", _link(url, url)))
        if result.match is not None:
            match = result.match
            lines.append(
                _kv(
                    out,
                    "agreement",
                    f"title {match.title:.2f} · author {'yes' if match.author else 'no'} · "
                    f"year {'yes' if match.year else 'no'}",
                )
            )
        if best.doi and resolved.retraction_error is not None:
            # Nobody answered: neither "not retracted" nor a notice (rule 2).
            lines.append(
                _kv(
                    out,
                    "retraction",
                    Text.assemble(
                        ui.style_state(out, "unavailable"), f" ({resolved.retraction_error})"
                    ),
                )
            )
        elif best.doi and retraction is None:
            lines.append(
                _kv(
                    out,
                    "retraction",
                    Text.assemble(
                        ui.style_state(out, "not retracted"),
                        " (Crossref/Retraction Watch, OpenAlex)",
                    ),
                )
            )
        elif best.doi:
            assert retraction is not None
            lines.append(
                _kv(
                    out,
                    "retraction",
                    Text.assemble(
                        ui.style_state(out, "RETRACTED"),
                        f" {retraction.date or ''} — {retraction.source}",
                    ),
                )
            )
    elif result.candidates:
        lines.append(
            _kv(
                out, "checked", f"{len(result.candidates)} candidate(s), none agrees on the fields:"
            )
        )
        lines.extend(
            _kv(out, "", f"- {c.title[:70]} ({c.first_author} {c.year or '?'}, {c.provider})")
            for c in result.candidates[:5]
        )
    lines.extend(_kv(out, "note", note) for note in result.notes)
    return lines


def _fetch_lines(out: ui.Ui, fetched: library.FetchOutcome) -> list[Text]:
    """What ``proofpath fetch`` prints: the result, then the section 7.1 lines."""
    result = fetched.result
    lines: list[Text] = []
    if isinstance(result, Fetched):
        lines.append(_kv(out, "outcome", result.outcome.value, state=True))
        lines.append(_kv(out, "step", f"{result.step} ({STEP_NAMES[result.step]})"))
        lines.append(_kv(out, "status", str(result.status) if result.status is not None else "—"))
        lines.append(_kv(out, "type", f"{result.content_type or '—'} · {result.kind}"))
        lines.append(_kv(out, "words", str(result.words)))
        lines.append(_kv(out, "url", _link(result.final_url or "—", result.final_url)))
        lines.append(_kv(out, "cached", "yes" if result.from_cache else "no"))
        lines.extend(_kv(out, "note", note) for note in result.notes)
    else:
        if result.state:
            # abstract / none: the state word is the styled one, not the kind.
            lines.append(
                _kv(
                    out,
                    "evidence",
                    Text.assemble(f"{result.kind} — ", ui.style_state(out, result.state)),
                )
            )
        else:
            lines.append(_kv(out, "evidence", result.kind, state=True))
        lines.append(_kv(out, "source", result.source or "—"))
        lines.append(_kv(out, "words", str(result.words)))
        lines.append(_kv(out, "url", _link(result.url or "—", result.url)))
        lines.extend(
            _kv(
                out,
                "attempt",
                f"{attempt.location.label:<13} {attempt.outcome.value:<44} "
                f"step {attempt.step}   {attempt.words} words",
            )
            for attempt in result.attempts
        )
        lines.extend(_kv(out, "note", note) for note in result.notes)
    lines.extend(_gate_lines(out, fetched.gate))
    return lines


def _gate_lines(out: ui.Ui, gate: ConsentGate) -> list[Text]:
    """The section 7.1 permission lines: only when the browser step mattered."""
    lines: list[Text] = []
    if gate.consulted:
        lines.append(_kv(out, "browser", gate.decision.reason))
    if gate.skipped:
        number = Text(f"{gate.skipped} source(s)")
        if out.color:
            number.stylize(ui.BROWSER_SKIPPED_STYLE)
        lines.append(_kv(out, "skipped", Text.assemble(number, f" {BROWSER_SKIPPED_REASON}")))
    lines.extend(_kv(out, "install", line) for line in gate.install_log)
    return lines


def _config_lines(out: ui.Ui, arg: str) -> list[Text]:
    """``/config``, ``/config path|show|set K V|check`` — the CLI group, mirrored."""
    sub, *rest = arg.split() or ["show"]
    if sub == "path":
        return [_kv(out, "config", str(config_path()))]
    if sub == "set":
        if len(rest) != 2:
            return [_error_line(out, "/config set wants a key and a value: SECTION.KEY VALUE")]
        try:
            written = library.config_set(rest[0], rest[1])
        except (ConfigError, JudgeError) as exc:
            return [_error_line(out, str(exc))]
        return [_kv(out, "", f"{key} = {value}  ({config_path()})") for key, value in written]
    if sub not in ("check", "show"):
        return [_error_line(out, f"/config {sub}: try show, path, set or check")]
    try:
        view = library.config_view()
    except ConfigError as exc:
        # An unreadable config file is reported the way the CLI reports it, not raised
        # as a traceback in the block.
        return [_error_line(out, str(exc))]
    if sub == "check":
        return _judge_lines(out, library.config_check(view.config))
    state = "" if view.exists else "  (not written yet, showing defaults)"
    lines = [_kv(out, "config", f"{view.path}{state}"), Text("")]
    lines.extend(Text(row) for row in view.toml.splitlines())
    return lines


def _judge_lines(out: ui.Ui, checked: library.JudgeCheck) -> list[Text]:
    """``/config check``: the provider that would be used, and whether it answered."""
    if checked.result is None:
        lines = [_error_line(out, f"no API key found for {checked.judge.provider}.")]
        lines.append(_kv(out, "", f"Put a line like  {checked.judge.api_key_env}=...  in one of:"))
        lines.extend(_kv(out, "", f"  {path}") for path in checked.dotenv_paths)
        return lines
    lines = [
        _kv(out, "provider", checked.judge.provider),
        _kv(out, "model", checked.result.model),
        _kv(out, "key from", checked.key.source if checked.key else "(none needed)"),
    ]
    if checked.ok:
        word, rest = "ok", f"{checked.result.latency_ms} ms"
    else:
        word, rest = "FAILED", checked.result.detail
    lines.append(_kv(out, "status", Text.assemble(ui.style_state(out, word), f" {rest}")))
    return lines


def _cache_lines(out: ui.Ui, arg: str) -> list[Text]:
    """``/cache``, ``/cache path|ls|show ID|clear [--expired]`` — the CLI group."""
    sub, *rest = arg.split() or [""]
    if sub == "path":
        return [_kv(out, "cache", str(library.cache_file()))]
    if sub == "ls":
        listing = library.cache_list()
        if listing.empty:
            return [_kv(out, "cache", "cache is empty")]
        lines = [
            Text(f"{entry.source_id}  {_entry_line(entry, listing.now)}")
            for entry in listing.entries
        ]
        lines.append(
            Text(f"{listing.resolutions} resolution(s), {listing.retractions} retraction check(s)")
        )
        return lines
    if sub == "show":
        if not rest:
            return [_error_line(out, "/cache show wants a source id, as listed by /cache ls")]
        detail = library.cache_detail(rest[0])
        if detail is None:
            return [_error_line(out, f"no cached source {rest[0]!r}")]
        return _detail_lines(out, detail)
    if sub == "clear":
        expired = "--expired" in rest
        removed = library.cache_clear(expired=expired)
        return [
            _kv(
                out,
                "removed",
                f"{removed.sources} source(s), {removed.resolutions} resolution(s), "
                f"{removed.retractions} retraction check(s)"
                f"{' (expired only)' if expired else ''}",
            )
        ]
    if sub:
        return [_error_line(out, f"/cache {sub}: try path, ls, show or clear")]
    held = library.cache_overview()
    return [
        _kv(out, "cache", str(held.path)),
        _kv(
            out, "holds", f"{held.sources} sources, {held.chunks} chunks, {held.verdicts} verdicts"
        ),
        _kv(
            out, "lookups", f"{held.resolutions} resolutions, {held.retractions} retraction checks"
        ),
    ]


def _entry_line(entry: SourceSummary, now: str) -> str:
    """One ``/cache ls`` row, expiry measured against the listing's single instant."""
    if entry.expires_at is None:
        expiry = "no raw text"
    elif entry.expires_at <= now:
        expiry = "raw text expired"
    else:
        expiry = f"raw text until {entry.expires_at[:10]}"
    return (
        f"{entry.title or '(untitled)'}  [{entry.scheme}/{entry.text_kind}]  "
        f"{entry.chunks} chunk(s), {entry.verdicts} verdict(s), {expiry}"
    )


def _detail_lines(out: ui.Ui, detail: SourceDetail) -> list[Text]:
    summary = detail.summary
    lines = [
        _kv(
            out,
            "source",
            f"{summary.source_id}  {summary.title or '(untitled)'}  "
            f"[{summary.scheme}/{summary.text_kind}]  fetched {summary.fetched_at[:19]}",
        )
    ]
    chunks, verdicts = detail.chunks, detail.verdicts
    lines.append(_kv(out, "chunks", str(len(chunks))))
    lines.extend(
        Text(f"  [{ordinal}] {text if text is not None else '(text expired)'}")
        for ordinal, text in chunks
    )
    lines.append(_kv(out, "verdicts", str(len(verdicts))))
    for verdict in verdicts:
        lines.append(
            Text.assemble(
                "  ",
                ui.style_state(out, verdict.label),
                f"  {verdict.tier:<6} {verdict.score:.2f}  claim {verdict.claim_hash[:12]}…",
            )
        )
        if verdict.passage_text is not None:
            lines.append(Text(f'    "{verdict.passage_text}"  [{verdict.passage_index}]'))
    return lines


class Prompt(Input):
    """The input bar. Awaiting mode tints it with the next run's accent."""

    def tint(self, accent: str) -> None:
        base = Color.parse(f"ansi_{accent}")
        # The ANSI flag is dropped on purpose: a palette entry cannot be blended, and
        # the spec asks for a tint rather than a solid bar. The *name* still comes
        # from ``ui.ACCENTS``; only its RGB is borrowed.
        self.styles.background = Color(base.r, base.g, base.b, TINT_ALPHA)

    def untint(self) -> None:
        self.styles.background = None


# --- the application ----------------------------------------------------------------


class ProofpathApp(App[None]):
    """`proofpath` with no arguments: the streaming prompt of spec section 13.1."""

    CSS = """
    Screen { background: $surface; }
    /* No padding: the banner draws itself against the full terminal width. */
    #banner { dock: top; height: auto; padding: 0; }
    #bottom { dock: bottom; height: 4; }
    #footer { height: 3; padding: 0 1; }
    #prompt-row { height: 1; padding: 0 1; }
    #caret { width: 2; }
    Prompt { border: none; height: 1; padding: 0; background: $surface; }
    RunLog { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    .run-block { height: auto; margin-bottom: 1; }
    .stages, .findings { height: auto; }
    /* The permission prompt is part of the log, not something drawn over it: no
       border, no panel, and buttons one row high, so the block keeps reading as a
       block (spec section 13.1 -- inline, never a modal). */
    .permission { height: auto; padding: 0; }
    .permission .answers { height: 1; padding: 0; }
    .permission Button {
        border: none; height: 1; min-width: 13; margin: 0 1 0 0;
        background: $surface; text-style: none;
    }
    .permission .answered { height: 1; }
    Line:focus { text-style: underline; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "leave_awaiting", "cancel awaiting", priority=True),
        Binding("ctrl+c", "quit_app", "quit", priority=True),
        Binding("ctrl+d", "quit_app", "quit", priority=True),
    ]

    def __init__(
        self,
        config: Config,
        out: ui.Ui,
        *,
        engine_factory: EngineFactory | None = None,
        scheduler_factory: SchedulerFactory | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._out = out
        self._engine_factory = engine_factory or self._default_engine
        self._scheduler_factory: SchedulerFactory = scheduler_factory or Scheduler
        self._scheduler: SchedulerLike | None = None
        self._blocks: dict[int, RunBlock] = {}
        #: The verb whose argument the bar is waiting for, or ``None``.
        self.awaiting: str | None = None
        #: The section 7.1 questions waiting for an answer, by the id of whatever
        #: asked: a run's number, or the negative counter a mirrored ``/fetch`` takes.
        #: One map, so ``/allow`` answers the newest question without a second lookup.
        self.pending: dict[int, Future[Answer]] = {}
        self._command_blocks: dict[int, CommandBlock] = {}
        self._owner = 0  # the negative half of the id space, for ``/fetch``
        #: Set the moment ``_quit`` starts. A question asked after that is answered
        #: ``no`` without ever being drawn: the app is on its way out, and a worker
        #: thread parked on a prompt nobody will see is a thread ``close()`` cannot
        #: join (rule 5 -- and the safe answer installs nothing).
        self._quitting = False

    def _default_engine(self, run: Run) -> Engine:
        """The run's engine, with the section 7.1 question pointed at its own block.

        Rule 4: ``interactive=True`` is the truth here -- a TUI *is* a terminal -- but
        the gate must never reach ``ask_terminal``, which would read the stdin Textual
        is holding and hang the app instead of asking anybody. ``prompt`` is what
        keeps it from ever having to.
        """
        return Engine.default(
            self._config,
            interactive=True,
            browser=None,
            no_cache=False,
            prompt=self._prompt_for(run.id),
        )

    # --- the section 7.1 prompt, inline ---------------------------------------------

    def _prompt_for(self, owner: int) -> Callable[[str, int | None], Answer]:
        """The gate's prompt, for one asker. Runs on that asker's worker thread.

        The thread blocks on a future while the event loop draws the question and
        waits for a button or an ``/allow`` line. An app that is already gone answers
        ``no``: nothing large is installed without an answer (rule 5).
        """

        def ask(host: str, status: int | None) -> Answer:
            if self._quitting:
                return "no"  # asked while the app was already leaving
            answer: Future[Answer] = Future()
            try:
                self.call_from_thread(self._open_prompt, owner, host, status, answer)
            except Exception:
                return "no"
            return answer.result()

        return ask

    def _open_prompt(
        self, owner: int, host: str, status: int | None, answer: Future[Answer]
    ) -> None:
        """Mount the question where it happened. Loop thread only."""
        if self._quitting:
            # The thread reached the loop after ``_quit`` had already answered
            # everything: answer this one too rather than mounting a question into a
            # screen that is going away.
            answer.set_result("no")
            return
        self.pending[owner] = answer
        widget = PermissionPrompt(owner, host, status, self._out)
        home: RunBlock | CommandBlock | None = (
            self._blocks.get(owner) if owner > 0 else self._command_blocks.get(owner)
        )
        if home is None:  # pragma: no cover - the block is mounted before the engine is
            self.query_one(RunLog).mount(widget)
        else:
            home.ask(widget)
        self._scroll_log()

    def _settle_prompt(self, owner: int, answer: Answer) -> bool:
        """Resolve one waiting question. ``False`` when there was none to resolve."""
        waiting = self.pending.pop(owner, None)
        if waiting is None:
            return False
        if not waiting.done():
            waiting.set_result(answer)
        for widget in self.query(PermissionPrompt):
            if widget.owner == owner and widget.answer is None:
                widget.settle(answer)
        return True

    def _settle_all(self, answer: Answer) -> None:
        """Answer everything still waiting -- on quit, so no worker thread is stranded."""
        for owner in list(self.pending):
            self._settle_prompt(owner, answer)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """A permission button: worth exactly the ``/allow`` line it is labelled with."""
        button_id = event.button.id or ""
        if not button_id.startswith(ANSWER_ID):
            return  # pragma: no cover - the prompt's are the only buttons there are
        owner = next(
            (w.owner for w in event.button.ancestors_with_self if isinstance(w, PermissionPrompt)),
            None,
        )
        if owner is not None:
            answer: Answer = button_id[len(ANSWER_ID) :]  # type: ignore[assignment]
            self._answer(owner, answer)

    def _answer(self, owner: int, answer: Answer) -> None:
        if not self._settle_prompt(owner, answer):
            return
        self._note(f"/allow {answer}", dim=False)
        if answer in ("always", "never"):
            # The gate writes the answer to the config file on its own thread, just
            # after this future resolves, so this read may be a moment early -- which is
            # why every use re-reads as well (see :meth:`_reload_config`). It is done
            # here too because it is usually not early, and the banner then says what
            # the session will actually do.
            self._reload_config()

    def _reload_config(self) -> None:
        """Re-read the config file and redraw whatever it decides on screen.

        Called before every run and every mirrored verb, and again after anything
        writes it — an ``always``/``never`` answer through the gate, or ``/config set``.
        Before *every* use rather than only after a write, because the write happens on
        the gate's own thread and because a session that ran for an hour should not keep
        deciding by a file somebody edited at the start of it. It is one small file read.

        A failure is reported and the session keeps the config it already had: deciding
        by a half-read config would be worse than deciding by a stale one.
        """
        try:
            self._config = load_config()
        except ConfigError as exc:
            self._note(f"config not reloaded: {exc}", dim=False)
            return
        self.query_one(Banner).set_context(run_context(self._config))

    # --- layout ---------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Banner(
            version=__version__,
            context=run_context(self._config),
            hint=HINT,
            coloured=self._out.color,
            # Spec section 13.1: no animation without colour, and none under ``-q``.
            animated=self._out.color and not self._out.quiet,
        )
        # One docked container, not two: Textual stacks same-edge docks on top of
        # each other and only reserves room for the tallest, so the footer and the
        # prompt share a single dock and split it between them.
        with Vertical(id="bottom"):
            yield CoverageFooter(self._out)
            with Horizontal(id="prompt-row"):
                yield Static(PROMPT_CARET, id="caret")
                yield Prompt(placeholder=DEFAULT_PLACEHOLDER, id="prompt")
        yield RunLog(id="log")

    def on_mount(self) -> None:
        self._scheduler = self._scheduler_factory(
            self._engine_factory, on_event=self._on_event, on_state=self._on_state
        )
        self.query_one(Prompt).focus()

    # --- the scheduler's two callbacks ----------------------------------------------

    def _on_event(self, run: Run, event: Event) -> None:
        block = self._blocks.get(run.id)
        if block is None:
            # A run whose block never got built still said something; dropping it
            # would make a run that spoke look like a run that did not.
            self._note(f"#{run.id} {run.command}: {type(event).__name__}")
            return
        block.handle(event)
        self._scroll_log()

    def _on_state(self, run: Run) -> None:
        block = self._blocks.get(run.id)
        if block is None:
            block = RunBlock(run, self._out)
            self._blocks[run.id] = block
            self.query_one(RunLog).mount(block)
        block.header.refresh()
        if run.state in ("done", "cancelled", "failed"):
            # A run that ends while its question is still up would strand the worker
            # thread on ``future.result()``: the safe answer is the one that installs
            # nothing (rule 5).
            self._settle_prompt(run.id, "no")
            block.settle()
            if run.report is not None:
                self.query_one(CoverageFooter).show(run.report)
        self._watch_eyes(run)
        self._scroll_log()

    def _watch_eyes(self, run: Run) -> None:
        """The ferret follows the runs: down the path while they work, up when done."""
        runs = self._scheduler.runs if self._scheduler is not None else ()
        ferret = self.query_one(Banner)
        ferret.set_busy(any(other.state in BUSY_STATES for other in runs))
        if run.state == "done" and run.report is not None:
            ferret.flash("findings" if run.report.findings else "clean")
        elif run.state == "failed":
            # A run that fell over is not a clean run; it never gets the clean face.
            ferret.flash("findings")

    def _scroll_log(self) -> None:
        self.query_one(RunLog).scroll_end(animate=False)

    # --- the command line -------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value
        event.input.value = ""
        await self._run_line(line)

    async def _run_line(self, line: str) -> None:
        text = line.strip()
        if self.awaiting is not None and not text.startswith("/"):
            # The bar is holding a verb; whatever was typed is its argument.
            verb = self.awaiting
            self._leave_awaiting()
            if text:
                await self._dispatch(commands.Command(verb, text))
            return
        # Anything else — a fresh slash command, or a blank line — ends the mode
        # before it is read, so the bar's tint and placeholder can never outlive the
        # verb that set them.
        self._leave_awaiting()
        parsed = commands.parse(line)
        if isinstance(parsed, commands.Awaiting):
            self._enter_awaiting(parsed)
        elif isinstance(parsed, commands.Unknown):
            if parsed.verb:
                self._note(f"unknown command: /{parsed.verb} · try /help")
        else:
            await self._dispatch(parsed)

    async def _dispatch(self, command: commands.Command) -> None:
        if command.verb == "check":
            self._check(command.arg)
        elif command.verb == "cancel":
            self._cancel(command.arg)
        elif command.verb == "allow":
            self._allow(command.arg)
        elif command.verb == "help":
            self._help()
        elif command.verb == "summarize":
            self._note(f"/summarize {V03_NOTE}")
        elif command.verb == "quit":
            await self._quit()
        elif command.verb in MIRRORED:
            self._mirror(command)

    # --- the mirrored verbs ----------------------------------------------------------

    def _mirror(self, command: commands.Command) -> None:
        """Run one CLI verb in the log. The work happens off the loop, in a worker.

        Awaiting it here instead would hold the app's own message queue for as long as
        the network takes, which is a frozen TUI rather than a busy one. Each call
        opens and closes its own clients, so the thread pool is safe: unlike a run,
        nothing it opens outlives it.
        """
        self._reload_config()  # the verb decides by it too (contact address, network)
        line = f"/{command.verb} {command.arg}".strip()
        block = CommandBlock(line, self._out)
        self._owner -= 1
        owner = self._owner
        self._command_blocks[owner] = block
        self.query_one(RunLog).mount(block)
        self._scroll_log()
        self.run_worker(self._run_verb(command, block, owner), name=line, group="mirror")

    async def _run_verb(self, command: commands.Command, block: CommandBlock, owner: int) -> None:
        lines: list[Text]
        try:
            lines = await asyncio.to_thread(
                verb_lines,
                command,
                out=self._out,
                config=self._config,
                prompt=self._prompt_for(owner),
            )
        except Exception as exc:
            # Reported, never raised at the loop: a verb that failed still has to say
            # so in the block that asked it.
            lines = [_error_line(self._out, f"{type(exc).__name__}: {exc}")]
        finally:
            self._settle_prompt(owner, "no")
            self._command_blocks.pop(owner, None)
        block.show(lines)
        if command.verb == "config" and command.arg.split()[:1] == ["set"]:
            # The file on disk has just changed; the session must not keep deciding by
            # the config it was launched with (``permissions.network`` is on the banner).
            self._reload_config()
        self._scroll_log()

    def _allow(self, arg: str) -> None:
        """Answer the section 7.1 question from the bar, exactly as the buttons do.

        With nothing waiting there is nothing to allow, and saying so is better than
        silently arming the next question. With a question waiting but no answer
        given, the four answers are listed rather than the bar being held: ``/allow``
        is an answer, not a target, and holding the bar for it would read a pasted
        path as one (the same reason ``/cancel`` does not hold it).
        """
        if not self.pending:
            self._note("nothing to allow")
            return
        owner = next(reversed(self.pending))  # the newest question is the one on screen
        answer = arg.strip().lower()
        if answer not in commands.ALLOW_ANSWERS:
            listed = "  ".join(f"/allow {name}" for name in commands.ALLOW_ANSWERS)
            if answer:
                self._note(f"/allow {arg}: not one of the four answers")
            self._note(listed, dim=False)
            return
        self._answer(owner, answer)  # type: ignore[arg-type]

    def _check(self, target: str) -> None:
        if self._scheduler is None:  # pragma: no cover - mount always runs first
            return
        # The run is about to be decided by the config, so the config is re-read first:
        # this is what makes a "never ask again" answered two runs ago stick.
        self._reload_config()
        self._scheduler.submit(target, command=f"/check {target}")

    def _cancel(self, arg: str) -> None:
        if self._scheduler is None:  # pragma: no cover - mount always runs first
            return
        try:
            run_id = int(arg)
        except ValueError:
            self._note(f"/cancel takes a run number, not {arg!r}")
            return
        if not self._scheduler.cancel(run_id):
            self._note(f"run #{run_id} is not running")

    def _help(self) -> None:
        """Every verb with what it wants, so the list is also the syntax."""
        for verb in commands.VERBS:
            placeholder = commands.NEEDS_ARGUMENT.get(verb, "")
            self._note(f"/{verb} {placeholder}".rstrip(), dim=False)
        self._note("a line that is not a command is checked as a target")

    def _note(self, text: str, *, dim: bool = True) -> None:
        self.query_one(RunLog).mount(NoteLine(text, dim=dim))
        self._scroll_log()

    # --- awaiting mode ----------------------------------------------------------------

    def _enter_awaiting(self, awaiting: commands.Awaiting) -> None:
        """Hold the bar for a verb that starts a run; otherwise just say what is missing.

        ``/cancel`` and ``/allow`` want an argument too, but they are answers rather
        than targets: capturing the next line for them would read a pasted path as
        ``/cancel ~/paper.pdf``. They print what they want and let the next line be
        the command it looks like.
        """
        if awaiting.verb == "allow":
            # ``/allow`` on its own is an answer that has not been chosen yet, not a
            # verb waiting for a target: it lists what may be answered, or says there
            # is nothing to answer.
            self._allow("")
            return
        if awaiting.verb not in AWAITING_VERBS:
            self._note(f"/{awaiting.verb} wants an argument: {awaiting.placeholder}")
            return
        self.awaiting = awaiting.verb
        prompt = self.query_one(Prompt)
        prompt.placeholder = awaiting.placeholder
        prompt.tint(accent_for(len(self._blocks) + 1))

    def _leave_awaiting(self) -> None:
        if self.awaiting is None:
            return
        self.awaiting = None
        prompt = self.query_one(Prompt)
        prompt.placeholder = DEFAULT_PLACEHOLDER
        prompt.untint()

    def action_leave_awaiting(self) -> None:
        self._leave_awaiting()

    async def action_quit(self) -> None:
        """Textual's own quit action, overridden so no route skips the close.

        ``ctrl+q``, the command palette and anything else that quits an ``App`` land
        here; ``ctrl+c``, ``ctrl+d`` and ``/quit`` are routed into it too. A run's
        worker thread holds an open ``Cache`` connection and an ONNX session, and a
        quit that walked past :meth:`Scheduler.close` would leave both behind.
        """
        await self._quit()

    async def action_quit_app(self) -> None:
        await self.action_quit()

    async def _quit(self) -> None:
        # The flag goes up first and stays up: everything already waiting is answered
        # here, and anything that asks *while* ``close()`` is running is answered on the
        # spot. Either way no worker thread is parked on a question when ``close()``
        # comes to join it.
        self._quitting = True
        self._settle_all("no")
        if self._scheduler is not None:
            await self._scheduler.close()
        self.exit()


def run(config: Config, out: ui.Ui) -> None:
    """Open the TUI. The bare ``proofpath`` invocation calls exactly this."""
    ProofpathApp(config, out).run()


__all__ = ["HINT", "ProofpathApp", "accent_for", "run", "run_context"]
