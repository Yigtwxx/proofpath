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
functions ``cli.py`` calls, on a worker thread, and :mod:`proofpath.tui.verbs` only
turns what comes back into lines. The widgets themselves live in
:mod:`proofpath.tui.widgets`; this module composes them, wires the scheduler to them
and reads the slash commands.

Colour comes from :mod:`proofpath.ui` through one :class:`~proofpath.tui.theme.Theme`
and nowhere else. ``run()`` detects the theme from the terminal (design section 2);
a caller that builds the app itself -- a test -- gets ``PLAIN`` unless it says
otherwise, so what a test sees never depends on the terminal it runs in. Widget
content is built as ``rich.Text`` in the theme's styles; the few places a colour
becomes a Textual style (a border, the awaiting bar's tint) go through
``widgets.textual_colour``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import ClassVar, Protocol

from rich.text import Text
from textual import events as tevents
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from proofpath import __version__, device, ui
from proofpath.browser import Answer
from proofpath.config import Config, ConfigError, load_config
from proofpath.events import Event
from proofpath.tui import commands, pet
from proofpath.tui.runs import EngineFactory as RunsEngineFactory
from proofpath.tui.runs import Run, Scheduler
from proofpath.tui.theme import PLAIN, Theme, detect
from proofpath.tui.verbs import error_line, verb_lines
from proofpath.tui.widgets import (
    ANSWER_ID,
    ANSWER_LABELS,
    Banner,
    CommandBlock,
    CoverageFooter,
    FindingLine,
    KvLine,
    NoteLine,
    PermissionPrompt,
    Prompt,
    RunBlock,
    RunHeader,
    StageLine,
    accent_for,
    panelled,
    textual_colour,
)
from proofpath.verify import Engine

#: Line 4 of the banner, joined here because only the app knows which commands it
#: offers. Matches the spec's own 80-column block, down to the spacing.
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"
#: What the input bar says when it is not waiting for a verb's argument.
DEFAULT_PLACEHOLDER = "paste a file path, a URL, or a claim"
#: The CLI verbs the TUI mirrors (spec section 13.3): each one calls the same
#: :mod:`proofpath.commands` function ``cli.py`` calls, on a worker thread.
MIRRORED = ("resolve", "fetch", "config", "cache")
#: ``/summarize`` is the one verb that is still nothing but a promise (spec 13.1).
V03_NOTE = "arrives in v0.3"

#: Which verbs hold the bar is the parser's decision, not this module's; it is
#: re-exported here because the app is where the mode is entered and left.
AWAITING_VERBS = commands.AWAITING_VERBS
#: The run states the ferret watches the path in (spec section 13.1).
BUSY_STATES = frozenset({"running", "verifying"})
#: How long a finished run's expression holds. The pet owns the animation; this is
#: re-exported because the app is where a run ends and the face is asked for.
FLASH_SECONDS = pet.FLASH_SECONDS


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


class RunLog(VerticalScroll):
    """The scrolling log: one :class:`RunBlock` per run, newest last."""


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
    /* RICH (design section 4): the footer gains the coverage bar's row, the prompt a
       rounded border, and a run's panel keeps two columns between border and text.
       The border colours are the run's accent and are set where the run is known. */
    #bottom.rich { height: 7; }
    #bottom.rich #footer { height: 4; }
    #bottom.rich #prompt-row { height: 3; margin: 0 1; padding: 0 1; border: round $surface; }
    .run-block.rich { padding: 0 1; }
    /* Below the panel floor the borders go (design section 4) and the rows come back. */
    #bottom.rich.narrow { height: 5; }
    #bottom.rich.narrow #prompt-row { height: 1; margin: 0; border: none; }
    .run-block.rich.narrow { padding: 0; }
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
        theme: Theme = PLAIN,
    ) -> None:
        super().__init__()
        self._config = config
        self._out = out
        #: The look, fixed for the session. ``PLAIN`` unless told otherwise: ``run()``
        #: detects the terminal's; a test says which one it is testing.
        self._theme = theme
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
        #: Zero-based index of the run whose accent the prompt wears.
        self._prompt_accent = 0
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
        widget = PermissionPrompt(owner, host, status, self._out, self._theme)
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
            theme=self._theme,
            coloured=self._out.color,
            # Spec section 13.1: no animation without colour, and none under ``-q``.
            animated=self._out.color and not self._out.quiet,
        )
        # One docked container, not two: Textual stacks same-edge docks on top of
        # each other and only reserves room for the tallest, so the footer and the
        # prompt share a single dock and split it between them.
        with Vertical(id="bottom", classes=self._theme.name):
            yield CoverageFooter(self._out, self._theme)
            with Horizontal(id="prompt-row"):
                yield Static(self._theme.glyphs.prompt, id="caret")
                yield Prompt(placeholder=DEFAULT_PLACEHOLDER, id="prompt")
        yield RunLog(id="log")

    def on_mount(self) -> None:
        self._scheduler = self._scheduler_factory(
            self._engine_factory, on_event=self._on_event, on_state=self._on_state
        )
        self._accent_prompt(0)
        self.query_one(Prompt).focus()

    def on_resize(self, event: tevents.Resize) -> None:
        # The event's size, not ``self.size``: the app's own is a step behind here.
        self._fit_bottom(event.size.width)

    def _fit_bottom(self, width: int | None = None) -> None:
        """The prompt's border, and the rows it costs, go below the panel floor.

        The border is set here and nowhere else, inline, because its colour is the
        current run's accent and a stylesheet cannot know that; the class only
        moves the heights.
        """
        wide = panelled(self._theme, self.size.width if width is None else width)
        self.query_one("#bottom").set_class(not wide and self._theme.name == "rich", "narrow")
        row = self.query_one("#prompt-row")
        if wide:
            # Without colour the box is drawn in the terminal's default: still a
            # box, where a border in the background colour would be three empty rows.
            accent = self._theme.accent(self._prompt_accent) if self._out.color else ""
            row.styles.border = (self._theme.glyphs.box, textual_colour(accent))
        else:
            row.styles.border = None

    def _accent_prompt(self, index: int) -> None:
        """The prompt in the current run's accent: its caret, and in RICH its border.

        ``index`` is the run's zero-based position, so the bar always matches the
        panel above it and, before any run, wears the accent the first one will.
        """
        self._prompt_accent = index
        if self._out.color:
            accent = self._theme.accent(index)
            caret = Text(self._theme.glyphs.prompt, style=accent)
            self.query_one("#caret", Static).update(caret)
        self._fit_bottom()

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
            block = RunBlock(run, self._out, self._theme)
            self._blocks[run.id] = block
            self.query_one(RunLog).mount(block)
            self._accent_prompt(run.id - 1)
        block.refresh_state()
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
        block = CommandBlock(line, self._out, self._theme)
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
            lines = [error_line(self._out, f"{type(exc).__name__}: {exc}")]
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
        self.query_one(RunLog).mount(NoteLine(text, self._theme, dim=dim))
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
        prompt.tint(self._theme.accent(len(self._blocks)))

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
    """Open the TUI. The bare ``proofpath`` invocation calls exactly this.

    The theme is read off the terminal here and nowhere else (design section 2):
    ``--no-color`` and ``-q`` arrive through ``out``, everything else through the
    environment. ``PROOFPATH_THEME`` overrides the lot.
    """
    theme = detect(os.environ, no_color=not out.color, quiet=out.quiet)
    ProofpathApp(config, out, theme=theme).run()


__all__ = [
    "ANSWER_LABELS",
    "AWAITING_VERBS",
    "FLASH_SECONDS",
    "HINT",
    "Banner",
    "CommandBlock",
    "CoverageFooter",
    "FindingLine",
    "KvLine",
    "NoteLine",
    "PermissionPrompt",
    "Prompt",
    "ProofpathApp",
    "RunBlock",
    "RunHeader",
    "RunLog",
    "StageLine",
    "accent_for",
    "run",
    "run_context",
]
