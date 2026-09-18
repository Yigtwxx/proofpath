"""The Textual application: layout, streaming run blocks, awaiting mode (spec 13.1).

Every test drives the app through ``run_test`` at a fixed 80x24 and a fake
scheduler, so no engine is built, no model is loaded and nothing touches the
network. The fake is the only thing standing in for the real one: the widgets, the
event routing and the command parsing under test are the production ones.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest
from textual import events as tevents
from textual.binding import BindingType
from textual.widgets import Button, Input, Static
from textual.widgets.input import Selection as InputSelection

from proofpath import __version__, browser, ui
from proofpath import commands as library
from proofpath import verify as verify_mod
from proofpath.browser import ConsentGate
from proofpath.config import Config, ConfigError, JudgeConfig, Permissions, load_config
from proofpath.document import Document, Locator, Reference
from proofpath.events import Emitted, Event, Note, Progress, Prompted, StageEnd, StageStart
from proofpath.judge import JudgeCost, JudgeError, provider_defaults
from proofpath.models import Label, Passage, Verdict
from proofpath.paths import config_path
from proofpath.report import Coverage, Finding, Kind, Report, summary_silence
from proofpath.resolve import Candidate, ResolveResult
from proofpath.resolve import State as ResolveState
from proofpath.tui import commands, wordmark
from proofpath.tui.app import (
    ANSWER_LABELS,
    AWAITING_VERBS,
    CLEARED_NOT_SUMMARIZABLE,
    HINT,
    Banner,
    CommandBlock,
    CoverageFooter,
    FindingLine,
    KvLine,
    NoteLine,
    PermissionPrompt,
    Prompt,
    PromptFrame,
    ProofpathApp,
    RunBlock,
    RunHeader,
    RunLog,
    StageLine,
    Suggestions,
    accent_for,
    run_context,
)
from proofpath.tui.banner import split_hint
from proofpath.tui.runs import Run, State
from proofpath.tui.theme import PLAIN, RICH
from proofpath.tui.widgets.config_panel import ConfigPanel, PanelLine
from proofpath.tui.widgets.prompt import HELD_SUMMARY
from proofpath.verify import Engine

SIZE = (80, 24)
HOST = "sciencedirect.com"


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/config`` and ``/cache`` read real files; never the ones this machine uses."""
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PROOFPATH_STATE_DIR", str(tmp_path / "state"))


# --- the fake scheduler ------------------------------------------------------------


class FakeScheduler:
    """Records what the app asked for and lets the test push events back at it.

    It keeps the real :class:`Run` dataclass so the widgets see exactly the object
    the production scheduler hands them; only the threading is gone.
    """

    def __init__(
        self,
        engine_factory: Any,
        *,
        on_event: Any,
        on_state: Any,
        **kwargs: Any,
    ) -> None:
        self.engine_factory = engine_factory
        self.on_event = on_event
        self.on_state = on_state
        self.submitted: list[tuple[str, str]] = []
        self.cancelled: list[int] = []
        self.closed = False
        self._runs: list[Run] = []

    @property
    def runs(self) -> tuple[Run, ...]:
        return tuple(self._runs)

    def get(self, run_id: int) -> Run | None:
        return next((run for run in self._runs if run.id == run_id), None)

    def submit(self, target: str, *, command: str) -> Run:
        run = Run(id=len(self._runs) + 1, command=command, target=target)
        self._runs.append(run)
        self.submitted.append((target, command))
        self.on_state(run)
        return run

    def last_done(self) -> Run | None:
        return next(
            (run for run in reversed(self._runs) if run.state == "done" and run.report is not None),
            None,
        )

    def cancel(self, run_id: int) -> bool:
        self.cancelled.append(run_id)
        run = self.get(run_id)
        if run is None:
            return False
        self.move(run, "cancelled")
        return True

    async def close(self) -> None:
        self.closed = True

    # --- what the test drives ------------------------------------------------------

    def push(self, run: Run, event: Event) -> None:
        self.on_event(run, event)

    def move(self, run: Run, state: State, *, report: Report | None = None) -> None:
        if report is not None:
            run.report = report
        run.state = state
        self.on_state(run)


def build_app(out: ui.Ui | None = None, **kwargs: Any) -> tuple[ProofpathApp, list[FakeScheduler]]:
    """An app wired to a fake scheduler, plus the list the fake lands in."""
    made: list[FakeScheduler] = []

    def factory(engine_factory: Any, **inner: Any) -> FakeScheduler:
        scheduler = FakeScheduler(engine_factory, **inner)
        made.append(scheduler)
        return scheduler

    app = ProofpathApp(
        Config(), out or ui.build(force_terminal=True), scheduler_factory=factory, **kwargs
    )
    return app, made


def _banner_lines(app: ProofpathApp) -> tuple[str, ...]:
    """The banner exactly as it was last drawn."""
    drawn = app.query_one(Banner).drawn
    assert drawn is not None
    return drawn.lines


async def until(pilot: Any, ready: Callable[[], bool], what: str, timeout: float = 15.0) -> None:
    """Pump the loop until ``ready()``; a worker thread is on the other end of these."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if ready():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


async def answered(pilot: Any, answer: Future[Any]) -> Any:
    """Pump the loop until the worker's future is settled, then return its value.

    ``Future.result(timeout=...)`` would block the very event loop that has to deliver
    the button press or the ``/allow`` line, so on a slow runner the answer never comes.
    """
    await until(pilot, answer.done, "the worker's answer")
    return answer.result(timeout=0)


def _laid_out(app: ProofpathApp, target: Any, offset: tuple[int, int]) -> bool:
    """True once ``target`` has a region on screen that ``offset`` falls inside."""
    found = app.query(target)
    if not found:
        return False
    region = found.first().region
    return region.area > 0 and (region.offset + offset) in app.screen.size.region


async def click(pilot: Any, target: Any, offset: tuple[int, int] = (0, 0)) -> None:
    """Press a widget once it is somewhere a press can reach it.

    ``Pilot.click`` reads ``widget.region`` *before* it pumps the loop, and a widget
    that is in the DOM but has not been laid out yet still carries the empty region it
    was born with. The click is then delivered to cell (0, 0) -- the banner -- and
    silently lost: nothing is pressed, the gate's future is never settled, and the test
    fails fifteen seconds later waiting for a worker nobody answered. Mounting is not
    layout, so waiting for the *widget* to exist is not enough; this waits for a region
    a click can land in and then asserts that it did land, so a press that goes astray
    says so at once instead of looking like a hung worker.
    """
    await until(pilot, lambda: _laid_out(pilot.app, target, offset), f"{target} to be laid out")
    landed = await pilot.click(target, offset=offset)
    assert landed, f"the click at {offset} never reached {target}"


def in_a_worker(work: Callable[[], Any]) -> Future[Any]:
    """Run ``work`` on a daemon thread, the way a run's own thread runs its engine.

    Daemon, and never a pool: a test that fails before the question is answered leaves
    the thread parked on ``future.result()``, and a pool would hold the interpreter
    open joining it at exit -- a failing test that hangs the suite instead of failing.
    """
    answer: Future[Any] = Future()

    def call() -> None:
        try:
            answer.set_result(work())
        except BaseException as exc:  # pragma: no cover - reported through the future
            answer.set_exception(exc)

    threading.Thread(target=call, daemon=True).start()
    return answer


def ask_from_a_worker(app: ProofpathApp, owner: int, status: int | None = 403) -> Future[Any]:
    """Call the app's prompt the way a run's engine calls it: from another thread.

    This is the callable :meth:`ProofpathApp._default_engine` hands the consent gate,
    so the future here is the one the gate blocks on.
    """
    ask = app._prompt_for(owner)
    return in_a_worker(lambda: ask(HOST, status))


async def submit(pilot: Any, line: str) -> None:
    """Type ``line`` into the prompt and press enter, the way a user does."""
    prompt = pilot.app.query_one(Prompt)
    prompt.focus()
    prompt.value = line
    await pilot.press("enter")
    await pilot.pause()


# --- fixtures for the streaming tests ----------------------------------------------


def a_finding() -> Finding:
    return Finding(
        kind=Kind.GHOST,
        level="error",
        locator=Locator(line=112, page=4),
        title="cited source does not exist",
        state="GHOST REFERENCE",
        reference=Reference(number=12, raw="Zhang et al. 2021", locator=Locator(line=400)),
        claim=None,
        verdict=None,
        source_id=None,
        fetch_step=None,
        tier=None,
        detail=("DOI 10.1016/j.xxxx.2021.99999 resolves to nothing",),
    )


def a_report(
    *,
    findings: tuple[Finding, ...] = (),
    fulltext: int = 26,
    abstract: int = 9,
    unverified: int = 7,
) -> Report:
    return Report(
        document=Document(name="draft.md", kind="markdown", paragraphs=(), references=(), pages=1),
        claims=1,
        markers=1,
        sources=(),
        results=(),
        findings=findings,
        coverage=Coverage(
            references=42,
            fulltext=fulltext,
            abstract=abstract,
            unverified=unverified,
            reasons={},
            browser_skipped=0,
            network_denied=False,
        ),
        stages=(),
        models={},
        api_calls=0,
        elapsed=38.0,
    )


# --- the banner --------------------------------------------------------------------


async def test_banner_reproduces_the_renderer_at_eighty_columns() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE):
        drawn = app.query_one(Banner).drawn
    assert drawn is not None
    expected = wordmark.render(
        80, PLAIN, version=__version__, context=run_context(Config()), hint=HINT
    )
    assert drawn.lines == expected.lines


async def test_banner_is_pure_ascii() -> None:
    """Assumption 3.3: the banner renders identically in Windows Terminal."""
    app, _ = build_app()
    async with app.run_test(size=SIZE):
        drawn = app.query_one(Banner).drawn
    assert drawn is not None
    for line in drawn.lines:
        assert all(ord(character) < 128 for character in line), line


async def test_banner_context_says_offline_when_the_network_is_denied() -> None:
    config = Config(permissions=Permissions(network="deny"))
    assert " offline " in run_context(config)
    assert " online " in run_context(Config())


async def test_busy_and_flash_are_accepted_and_change_nothing() -> None:
    """The raven does not move (raven design §2); the app's calls must still be safe."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        widget = app.query_one(Banner)
        before = widget.drawn
        widget.set_busy(True)
        widget.flash("findings")
        widget.flash("clean")
        widget.set_busy(False)
        await pilot.pause()
        assert widget.drawn == before


# --- awaiting mode -----------------------------------------------------------------


async def test_bare_check_opens_awaiting_mode() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check")
        prompt = app.query_one(Prompt)
        assert prompt.placeholder == commands.NEEDS_ARGUMENT["check"]
        assert app.awaiting == "check"
        # The bar takes the *next* run's accent, at the spec's ~15%.
        assert prompt.styles.background.a == pytest.approx(0.15)
    assert schedulers[0].submitted == []


async def test_escape_leaves_awaiting_mode() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check")
        await pilot.press("escape")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert app.awaiting is None
        assert prompt.placeholder != commands.NEEDS_ARGUMENT["check"]
        assert prompt.styles.background.a == pytest.approx(0.0)  # the terminal's own, untinted


async def test_the_awaited_argument_starts_the_run() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check")
        await submit(pilot, "draft.md")
        assert app.awaiting is None
    assert schedulers[0].submitted == [("draft.md", "/check draft.md")]


# --- submitting a run --------------------------------------------------------------


async def test_check_with_an_argument_skips_the_mode_and_opens_a_block() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        assert app.awaiting is None
        block = app.query_one(RunBlock)
        header = block.query_one(RunHeader)
        assert header.render().plain.split()[:4] == ["#1", "/check", "draft.md", "queued"]
    assert schedulers[0].submitted == [("draft.md", "/check draft.md")]


async def test_a_bare_line_is_an_implicit_check() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "~/paper.pdf")
    assert schedulers[0].submitted == [("~/paper.pdf", "/check ~/paper.pdf")]


async def test_two_runs_get_different_accents() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.md")
        await submit(pilot, "/check two.md")
        blocks = list(app.query(RunBlock))
    assert [block.accent for block in blocks] == [ui.ACCENTS[0], ui.ACCENTS[1]]
    assert accent_for(6) == ui.ACCENTS[0]  # the palette rotates, five wide


# --- streaming ---------------------------------------------------------------------


async def test_events_render_stages_findings_and_the_footer() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]

        scheduler.move(run, "running")
        scheduler.push(run, StageStart("Parse", "pymupdf"))
        scheduler.push(run, Progress("Parse", 3, 42, "refs"))
        scheduler.push(run, StageEnd("Parse", "pymupdf", "24 pages, 42 refs", 1.2))
        scheduler.push(run, Emitted(a_finding()))
        await pilot.pause()

        stage = app.query_one(StageLine).render().plain
        assert "Parse" in stage
        assert "pymupdf" in stage
        assert "24 pages, 42 refs" in stage
        assert "1.2s" in stage

        finding = app.query_one(FindingLine).render().plain
        assert "p.4 L112" in finding
        assert "[12]" in finding
        assert "GHOST REFERENCE" in finding

        scheduler.move(run, "done", report=a_report(findings=(a_finding(),)))
        await pilot.pause()

        footer = app.query_one(CoverageFooter).render().plain
        assert "42 refs" in footer
        assert "coverage 62/21/17%" in footer
        assert "no report written" in footer
        assert "0 API calls" in footer

        header = app.query_one(RunHeader).render().plain
        assert "done" in header
        assert "62/21/17%" in header


async def test_a_findings_notes_are_printed_under_it_as_the_cli_prints_them() -> None:
    """TUI v2 polish: a ghost's reason is part of the finding, in PLAIN too (rule 2)."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.push(scheduler.runs[0], Emitted(a_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        rendered = line.render()
        rows = rendered.plain.split("\n")
        assert line.region.height == 2
    assert rows[1] == "       = note: DOI 10.1016/j.xxxx.2021.99999 resolves to nothing"
    assert str(rendered.spans[-1].style) == "dim"


async def test_a_finding_prints_the_entrys_marker_once() -> None:
    """``Reference.raw`` keeps its printed marker (Phase 5); the row adds the number
    itself, so the label must strip the marker or read ``[7] [7] Marchetti …``."""
    item = a_finding()
    item = replace(
        item,
        reference=Reference(
            number=7, raw="[7] Marchetti, A. et al. 2019", locator=Locator(line=400)
        ),
    )
    app, schedulers = build_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.push(scheduler.runs[0], Emitted(item))
        await pilot.pause()
        text = app.query_one(FindingLine).render().plain
    assert "[7] Marchetti" in text
    assert "[7] [7]" not in text


async def test_the_state_word_is_coloured_by_meaning() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.push(scheduler.runs[0], Emitted(a_finding()))
        await pilot.pause()
        rendered = app.query_one(FindingLine).render()
    styles = {str(span.style) for span in rendered.spans}
    assert "red" in styles  # GHOST REFERENCE, straight out of ui's table


async def test_a_failed_run_borrows_reds_meaning_from_ui() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        run.error = "OSError: no such file"
        scheduler.move(run, "failed")
        await pilot.pause()
        rendered = app.query_one(RunHeader).render()
    assert "failed" in rendered.plain
    assert "OSError: no such file" in rendered.plain
    assert "red" in {str(span.style) for span in rendered.spans}


async def test_two_hundred_notes_keep_the_footer_and_the_banner_in_place() -> None:
    """Rule 6: the coverage footer never scrolls away, however long the log gets."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.move(run, "done", report=a_report())
        for index in range(200):
            scheduler.push(run, Note(f"note {index}"))
        await pilot.pause()

        assert len(app.query(NoteLine)) == 200
        banner_widget = app.query_one(Banner)
        footer = app.query_one(CoverageFooter)
        assert banner_widget.region.y == 0
        assert banner_widget.region.height == 9  # the mark, the two text rows, the rule
        assert footer.display
        assert footer.region.bottom <= app.size.height
        assert "coverage" in footer.render().plain


# --- the rest of the command surface ------------------------------------------------


async def test_cancel_reaches_the_scheduler() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        await submit(pilot, "/cancel 1")
        await submit(pilot, "/cancel #1")
    assert schedulers[0].cancelled == [1, 1]


async def test_cancel_of_a_word_is_reported_not_crashed() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/cancel nope")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert "nope" in text
    assert schedulers[0].cancelled == []


async def test_help_lists_every_verb() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    for verb in commands.VERBS:
        assert f"/{verb}" in text


async def test_help_prints_each_verbs_description_after_its_placeholder() -> None:
    """``commands.DESCRIPTIONS`` feeds ``/help`` too, not only the list above the
    bar: each row is the verb, its placeholder (if any), the theme's own separator
    glyph, and the description -- so ``/help`` never drifts from what the list
    above the bar says a verb does."""
    app, _ = build_app()  # PLAIN: sep is ","
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        notes = [line.render().plain for line in app.query(NoteLine)]
    for verb in commands.VERBS:
        placeholder = commands.NEEDS_ARGUMENT.get(verb, "")
        expected = (
            f"/{verb} {placeholder}".rstrip()
            + f" {PLAIN.glyphs.sep} "
            + commands.DESCRIPTIONS[verb]
        )
        assert any(note.endswith(expected) for note in notes), expected


class FakeJudge:
    """``Judge`` without a socket: one paragraph, or a provider that never answered."""

    PARAGRAPH = "Three references do not say this."
    name = "fake judge-1"

    def __init__(self, *, text: str = PARAGRAPH, detail: str = "") -> None:
        self._text = text
        self.detail = detail
        self.unavailable = not text
        self.cost = JudgeCost(calls=1 if text else 0, model="judge-1")
        self.seen = ""
        self.closed = False

    def summarize(self, report_markdown: str, *, max_tokens: int = 1500) -> str:
        self.seen = report_markdown
        return self._text

    def close(self) -> None:
        self.closed = True


async def finished_run(pilot: Any, scheduler: FakeScheduler) -> Run:
    """One run in the log, done, with a report to summarise."""
    await submit(pilot, "/check draft.md")
    run = scheduler.runs[-1]
    scheduler.move(run, "done", report=a_report())
    await pilot.pause()
    return run


def _kv_text(app: ProofpathApp) -> str:
    return "\n".join(line.render().plain for line in app.query(KvLine))


def _notes(app: ProofpathApp) -> str:
    return "\n".join(line.render().plain for line in app.query(NoteLine))


async def test_summarize_appends_one_model_written_line_to_the_last_finished_run() -> None:
    """Spec section 11.1: off until asked, labelled when shown, one call over a report."""
    judge = FakeJudge()
    app, schedulers = build_app(judge_factory=lambda: judge)
    async with app.run_test(size=SIZE) as pilot:
        await finished_run(pilot, schedulers[0])
        await submit(pilot, "/summarize")
        await until(pilot, lambda: judge.closed, "the summary worker")
        text = _kv_text(app)
        blocks = len(app.query(CommandBlock))

    assert "summary" in text
    assert f"(model-written, {FakeJudge.name}) {FakeJudge.PARAGRAPH}" in text
    # It reads the finished report and nothing else -- never a source, never a draft.
    assert judge.seen.startswith("# proofpath report")
    assert blocks == 0  # the line lands in the run's own block, not a new one


async def test_summarize_before_any_run_has_finished_says_so() -> None:
    app, _ = build_app(judge_factory=lambda: FakeJudge())
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/summarize")
        text = _notes(app)
    assert "/check" in text


async def test_summarize_without_a_judge_says_how_to_configure_one() -> None:
    def no_judge() -> FakeJudge:
        raise JudgeError("GROQ_API_KEY is not set")

    app, schedulers = build_app(judge_factory=no_judge)
    async with app.run_test(size=SIZE) as pilot:
        await finished_run(pilot, schedulers[0])
        await submit(pilot, "/summarize")
        text = _notes(app)
    assert "/config set judge.provider" in text
    assert ".env" in text


async def test_a_summary_the_judge_never_wrote_is_reported_not_left_blank() -> None:
    """Product rule 2: an empty line would read as a report with nothing to add."""
    down = "HTTP 401 from https://api.test/chat/completions"
    judge = FakeJudge(text="", detail=down)
    app, schedulers = build_app(judge_factory=lambda: judge)
    async with app.run_test(size=SIZE) as pilot:
        await finished_run(pilot, schedulers[0])
        await submit(pilot, "/summarize")
        await until(pilot, lambda: judge.closed, "the summary worker")
        text = _kv_text(app)

    # The CLI's builder, not a wording of its own: one silence, one sentence.
    assert f"summary    {summary_silence(down)}" in text
    assert "0 calls" not in text  # one request was made; ``calls`` counts answers


async def test_summarizing_a_run_twice_adds_one_line_and_says_why() -> None:
    """One summary per run: a second line would read as a second opinion about it."""
    judge = FakeJudge()
    app, schedulers = build_app(judge_factory=lambda: judge)
    async with app.run_test(size=SIZE) as pilot:
        await finished_run(pilot, schedulers[0])
        await submit(pilot, "/summarize")
        await until(pilot, lambda: judge.closed, "the summary worker")
        await submit(pilot, "/summarize")
        await pilot.pause()
        text = _kv_text(app)
        notes = _notes(app)

    assert text.count(FakeJudge.PARAGRAPH) == 1
    assert "already summarised" in notes


async def test_an_unknown_verb_is_reported() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/Check draft.md")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert "/Check" in text


async def test_quit_closes_the_scheduler_and_exits() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/quit")
        await pilot.pause()
    assert schedulers[0].closed
    assert not app.is_running


async def test_the_footer_is_pinned_above_the_prompt_and_keeps_its_height() -> None:
    """Rule 6 again: the footer is docked at a fixed height, so nothing overlaps it."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        footer = app.query_one(CoverageFooter)
        prompt_row = app.query_one("#prompt-row")
        assert footer.region.height == 3
        assert footer.region.bottom <= prompt_row.region.y
        scheduler = schedulers[0]
        run = scheduler.runs[0] if scheduler.runs else None
        assert run is None
        await submit(pilot, "/check draft.md")
        scheduler.move(scheduler.runs[0], "done", report=a_report())
        await pilot.pause()
        assert footer.region.height == 3
        assert footer.region.bottom <= prompt_row.region.y


async def test_a_weak_run_says_so_where_it_cannot_scroll_away() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.move(
            scheduler.runs[0], "done", report=a_report(fulltext=8, abstract=4, unverified=30)
        )
        await pilot.pause()
        footer = app.query_one(CoverageFooter).render().plain
    assert "coverage is weak" in footer
    assert footer.count("\n") == 2  # never more, never fewer: the height is fixed


async def test_ctrl_c_closes_the_scheduler_before_it_exits() -> None:
    """Closing is not optional: a worker thread holds an open cache and an ONNX session."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert schedulers[0].closed
    assert not app.is_running


async def test_escape_outside_awaiting_mode_does_nothing() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        await pilot.press("escape")
        await pilot.pause()
        assert app.awaiting is None
        assert len(app.query(RunBlock)) == 1
    assert schedulers[0].cancelled == []


# --- fix round 1: every quit route closes the scheduler -----------------------------


@pytest.mark.parametrize("keys", [("ctrl+q",), ("ctrl+c",), ("ctrl+d",)])
async def test_every_quit_key_closes_the_scheduler(keys: tuple[str, ...]) -> None:
    """A worker thread holds an open cache and an ONNX session; no route may skip close."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press(*keys)
        await pilot.pause()
    assert schedulers[0].closed
    assert not app.is_running


# --- fix round 1: display truth below 80 columns -------------------------------------


def a_verdict_finding() -> Finding:
    """A finding that asserts, so it carries a tier and a passage (rule 1)."""
    return Finding(
        kind=Kind.NOT_SUPPORTED,
        level="error",
        locator=Locator(line=260, page=9),
        title="the source does not support this",
        state="NOT SUPPORTED",
        reference=Reference(
            number=31,
            raw="Kumar, A. and Patel, S. (2022) A very long bibliography entry indeed",
            locator=Locator(line=800),
        ),
        claim=None,
        verdict=Verdict(
            label=Label.REFUTED,
            score=0.99,
            tier="high",
            passage=Passage(
                text="we observed a 4-8% improvement in throughput", source_id="doi:x", index=0
            ),
        ),
        source_id="doi:x",
        fetch_step=1,
        tier="high",
    )


@pytest.mark.parametrize("width", [80, 60, 46])
async def test_a_finding_never_clips_its_state_word_or_tier(width: int) -> None:
    """Rule 1 and rule 2 are display rules too: the verdict and its tier always show."""
    app, schedulers = build_app()
    async with app.run_test(size=(width, 24)) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.push(scheduler.runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        text = line.render().plain
    assert "NOT SUPPORTED" in text
    assert "high" in text
    assert "p.9 L260" in text
    assert "[31]" in text
    for row in text.split("\n"):
        assert len(row) <= width, row


@pytest.mark.parametrize("width", [80, 60, 46])
async def test_a_stage_never_drops_its_attribution(width: int) -> None:
    """No number a stage produced is left unattributable (spec 13.1)."""
    app, schedulers = build_app()
    async with app.run_test(size=(width, 24)) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.push(run, StageStart("Resolve references", "Crossref, OpenAlex"))
        scheduler.push(
            run,
            StageEnd(
                "Resolve references",
                "Crossref, OpenAlex",
                "38 resolved, 3 ambiguous, 1 ghost, 0 retracted",
                4.4,
            ),
        )
        await pilot.pause()
        text = app.query_one(StageLine).render().plain
    assert "Crossref, OpenAlex" in text
    assert "4.4s" in text
    for row in text.split("\n"):
        assert len(row) <= width, row


async def test_the_plain_banner_drops_the_command_list_when_the_hint_row_overflows() -> None:
    """The overflow rule is the widget's, not the rich theme's: at 46 columns the
    prompt and the command list cannot share a row, so the plain hint row is the
    prompt alone and ``/help`` lists the commands."""
    app, _ = build_app()
    async with app.run_test(size=(46, 24)):
        drawn = app.query_one(Banner).drawn
    assert drawn is not None
    where = wordmark.rows(46, version=__version__, context=run_context(Config()), hint=HINT)
    prompt, _ = split_hint(HINT)
    assert drawn.lines[where.hint] == prompt
    assert "/quit" not in drawn.lines[where.hint]
    assert all(len(line) <= 46 for line in drawn.lines)


@pytest.mark.parametrize("width", [80, 60, 46])
async def test_the_banner_container_fits_every_line_it_drew(width: int) -> None:
    """Below the floor the mark goes and the two text rows stay; narrower still, the
    hint row would wrap. Whatever was drawn, the container holds every row of it
    and the hint is on screen."""
    app, _ = build_app()
    async with app.run_test(size=(width, 24)):
        widget = app.query_one(Banner)
        drawn = widget.drawn
        log = app.query_one(RunLog)
        needed = sum(max(1, -(-len(line) // width)) for line in (drawn.lines if drawn else ()))
        assert widget.region.height >= needed
        assert widget.region.bottom <= log.region.y
        assert HINT.split(".")[0] in widget.render().plain


async def test_resizing_redraws_the_banner_and_keeps_the_footer_docked() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        assert app.query_one(Banner).drawn.lines[-1].endswith("-")
        await pilot.resize_terminal(60, 24)
        await pilot.pause()
        drawn = app.query_one(Banner).drawn
        assert drawn is not None
        assert (
            drawn.lines
            == wordmark.render(
                60, PLAIN, version=__version__, context=run_context(Config()), hint=HINT
            ).lines
        )
        footer = app.query_one(CoverageFooter)
        assert footer.region.bottom <= app.size.height
        assert footer.region.y >= app.query_one(RunLog).region.bottom


# --- fix round 1: a fresh command always leaves awaiting mode ------------------------


async def test_a_fresh_command_leaves_awaiting_mode() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check")
        assert app.awaiting == "check"
        await submit(pilot, "/help")
        prompt = app.query_one(Prompt)
        assert app.awaiting is None
        assert prompt.placeholder != commands.NEEDS_ARGUMENT["check"]
        assert prompt.styles.background.a == pytest.approx(0.0)  # the terminal's own, untinted
    assert schedulers[0].submitted == []


# --- history ------------------------------------------------------------------------


async def test_up_brings_back_the_last_line() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/config show")  # bare ``/config`` opens a panel that takes focus
        prompt = app.query_one(Prompt)
        await pilot.press("up")
        assert prompt.value == "/config show"
        assert prompt.cursor_position == len("/config show")
        await pilot.press("up")
        assert prompt.value == "/help"
        await pilot.press("up")  # at the oldest: stays
        assert prompt.value == "/help"


async def test_down_past_the_newest_restores_the_draft() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        prompt = app.query_one(Prompt)
        prompt.value = "half a th"
        await pilot.press("up")
        assert prompt.value == "/help"
        await pilot.press("down")
        assert prompt.value == "half a th"
        await pilot.press("down")  # nothing newer than the draft
        assert prompt.value == "half a th"


async def test_an_edit_ends_the_walk() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/config show")  # bare ``/config`` opens a panel that takes focus
        prompt = app.query_one(Prompt)
        await pilot.press("up")
        await pilot.press("x")
        assert prompt.value == "/config showx"
        await pilot.press("up")  # a fresh walk: starts at the newest again
        assert prompt.value == "/config show"


async def test_held_up_walks_back_even_when_keys_outrun_changed() -> None:
    """Two ``up`` keys already queued must not read as an edit that ends the walk."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        for line in ("/one", "/two", "/three"):
            await submit(pilot, line)
        prompt = app.query_one(Prompt)
        # ``pilot.press`` pauses between keys; a held key while the loop is busy does
        # not. Posted to the app the way the driver posts them, both are handled
        # before the first ``Changed`` reaches the bar.
        app.post_message(tevents.Key("up", None))
        app.post_message(tevents.Key("up", None))
        await pilot.pause()
        assert prompt.value == "/two"
        await pilot.press("up")
        assert prompt.value == "/one"


async def test_down_from_an_empty_bar_comes_back_empty() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        prompt = app.query_one(Prompt)
        await pilot.press("up")
        assert prompt.value == "/help"
        await pilot.press("down")
        assert prompt.value == ""
        await pilot.press("up")  # a fresh walk starts at the newest again
        assert prompt.value == "/help"


async def test_history_survives_a_restart(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
    assert (tmp_path / "state" / "history").read_text(encoding="utf-8") == "/help\n"
    again, _ = build_app()
    async with again.run_test(size=SIZE) as pilot:
        await pilot.press("up")
        assert again.query_one(Prompt).value == "/help"


# --- a multi-line paste (spec 2026-09-17-multiline-paste) ---------------------------

POST = (
    "The vaccine trial enrolled 40,000 people.\n\n"
    "Source: https://example.org/trial\nSee also doi:10.1000/xyz123"
)


async def paste(pilot: Any, text: str) -> None:
    """Paste ``text`` into the bar the way a bracketed paste arrives: one event."""
    prompt = pilot.app.query_one(Prompt)
    prompt.focus()
    prompt.post_message(tevents.Paste(text))
    await pilot.pause()


async def test_a_multiline_paste_is_held_and_summarised() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        prompt = app.query_one(Prompt)
        assert prompt.held == POST
        # Three non-blank lines of four; the blank one is not counted.
        assert prompt.value == HELD_SUMMARY.format(sep=",", lines=3, chars=len(POST))
    assert schedulers[0].submitted == []


async def test_the_summary_uses_the_rich_separator() -> None:
    app, _ = build_app(theme=RICH)
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        assert app.query_one(Prompt).value == HELD_SUMMARY.format(sep="·", lines=3, chars=len(POST))


async def test_enter_checks_the_held_paste_whole() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("enter")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""
    # The target keeps its line breaks; the label is one line.
    assert schedulers[0].submitted == [(POST, "/check pasted text")]


async def test_a_cr_separated_paste_is_labelled_pasted_text_too() -> None:
    """A bracketed paste's line breaks can arrive as ``\\r``; ``_check``'s label must
    agree with ``_on_paste``'s ``splitlines()``, not just ``"\\n" in target``."""
    text = POST.replace("\n", "\r")
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, text)
        await pilot.press("enter")
        await pilot.pause()
    assert schedulers[0].submitted == [(text, "/check pasted text")]


async def test_a_held_paste_is_the_awaited_argument() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check")
        await paste(pilot, POST)
        await pilot.press("enter")
        await pilot.pause()
        assert app.awaiting is None
    assert schedulers[0].submitted == [(POST, "/check pasted text")]


async def test_a_typed_character_drops_the_held_paste() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("x")
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == "x"


async def test_backspace_drops_the_held_paste_and_empties_the_bar() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("backspace")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""


async def test_escape_drops_the_held_paste() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("escape")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""


async def test_a_history_step_drops_the_held_paste() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await paste(pilot, POST)
        await pilot.press("up")
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == "/help"


async def test_a_one_line_paste_lands_in_the_bar_as_text() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        await paste(pilot, "~/Desktop/paper.pdf")
        assert prompt.held is None
        assert prompt.value == "~/Desktop/paper.pdf"
        prompt.value = ""
        await paste(pilot, "~/Desktop/paper.pdf\n")  # a trailing newline is still one line
        assert prompt.held is None
        assert prompt.value == "~/Desktop/paper.pdf"


async def test_a_blank_paste_holds_nothing() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, "\n\n  \n")
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""


async def test_a_held_paste_stays_out_of_the_history_file(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await paste(pilot, POST)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("up")
        assert app.query_one(Prompt).value == "/help"
    assert (tmp_path / "state" / "history").read_text(encoding="utf-8") == "/help\n"


async def test_a_mirrored_verb_labels_a_held_paste_on_one_line() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/resolve")
        await paste(pilot, "Smith J.\nA title.\nJournal 2020")
        await pilot.press("enter")
        await pilot.pause()
        block = app.query_one(CommandBlock)
        assert block.command == "/resolve Smith J. A title. Journal 2020"


async def test_cancel_never_captures_a_path_as_its_argument() -> None:
    """Only the verbs that start a run hold the bar; a path after /cancel is a check."""
    assert "cancel" not in AWAITING_VERBS
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/cancel")
        assert app.awaiting is None
        await submit(pilot, "~/Desktop/paper.pdf")
    assert schedulers[0].cancelled == []
    assert schedulers[0].submitted == [("~/Desktop/paper.pdf", "/check ~/Desktop/paper.pdf")]


async def test_a_bare_cancel_says_what_it_wants() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/cancel")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert commands.NEEDS_ARGUMENT["cancel"] in text


async def test_an_event_for_an_unknown_run_is_logged_not_dropped() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        stray = Run(id=99, command="/check ghost.md", target="ghost.md")
        schedulers[0].push(stray, Note("something happened"))
        await pilot.pause()
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert "#99" in text


@pytest.mark.parametrize("width", [80, 60, 46])
async def test_the_footer_never_wraps_out_of_its_fixed_height(width: int) -> None:
    """Its height is fixed, so a wrapped line would push the caveats row off screen."""
    app, schedulers = build_app()
    async with app.run_test(size=(width, 24)) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.move(
            scheduler.runs[0], "done", report=a_report(fulltext=8, abstract=4, unverified=30)
        )
        await pilot.pause()
        footer = app.query_one(CoverageFooter)
        rows = footer.render().plain.split("\n")
    assert len(rows) == 3
    for row in rows:
        assert len(row) <= width, row
    assert "coverage 19/10/71%" in rows[0]  # the cell that never gives way


# --- task 8.4: the inline permission prompt (spec 13.1, section 7.1) -----------------


async def test_the_prompt_is_drawn_in_the_block_of_the_run_that_hit_the_wall() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        answer = ask_from_a_worker(app, run.id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt to be mounted")

        prompt = app.query_one(PermissionPrompt)
        assert any(isinstance(parent, RunBlock) for parent in prompt.ancestors_with_self)
        assert [str(button.label) for button in prompt.query(Button)] == [
            f"[{label}]" for label in ANSWER_LABELS.values()
        ]
        # The buttons are the spec's own four, and the text is section 7.1's own block.
        text = prompt.question
        assert HOST in text
        assert "HTTP 403" in text
        assert browser.WHEELS_SIZE in text and browser.BROWSER_SIZE in text
        # The last line names what the bar takes, not the keys the terminal reads.
        assert text.endswith(browser.TUI_ANSWERS)
        assert "[y] yes" not in text

        await click(pilot, "#allow-once")
        assert await answered(pilot, answer) == "once"


async def test_a_button_click_and_an_allow_line_resolve_the_same_future() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        answer = ask_from_a_worker(app, run.id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await submit(pilot, "/allow never")
        assert await answered(pilot, answer) == "never"
        # The prompt stays in the log saying what was answered: rule 6, in the log.
        prompt = app.query_one(PermissionPrompt)
        assert prompt.answer == "never"
        assert all(button.disabled for button in prompt.query(Button))
        assert app.pending == {}


async def test_allow_with_nothing_pending_says_so() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/allow once")
        notes = [line.render().plain for line in app.query(NoteLine)]
    note = next(line for line in notes if line.lstrip().startswith("nothing to allow — "))
    # The note says where the question appears and what answers it (spec 13.1).
    assert "/allow " + "|".join(commands.ALLOW_ANSWERS) in note
    assert "/allow once|always|no|never" in note


def test_the_footer_hint_for_skipped_sources_names_the_setting() -> None:
    """The docked caveat line says which setting lets the browser run next time."""
    from proofpath.report import BROWSER_SKIPPED_REASON, render_footer
    from proofpath.tui.widgets.footer import _hints

    plain = a_report()
    item = render_footer(replace(plain, coverage=replace(plain.coverage, browser_skipped=2)))
    hint = next(line for line in _hints(item) if line.startswith("2 source(s)"))
    assert hint == f"2 source(s) {BROWSER_SKIPPED_REASON} — /config set {ui.BROWSER_SETTING}"
    assert ui.BROWSER_SETTING == "permissions.install_browser ask"


def test_the_fetch_block_skipped_line_names_the_setting() -> None:
    """The ``/fetch`` block's line says which setting lets the browser run next time."""
    from proofpath.report import BROWSER_SKIPPED_REASON
    from proofpath.tui.verbs import _gate_lines

    gate = ConsentGate("deny", interactive=False)
    gate.consulted = True
    gate.skipped = 1
    lines = [line.plain for line in _gate_lines(ui.build(force_terminal=True), gate)]
    skipped = next(line for line in lines if line.startswith("skipped"))
    assert skipped == (
        f"skipped    1 source(s) {BROWSER_SKIPPED_REASON} — /config set {ui.BROWSER_SETTING}"
    )


async def test_a_bare_allow_lists_the_four_answers_without_holding_the_bar() -> None:
    """``/allow`` is an answer, not a target: a path typed after it is still a check."""
    assert "allow" not in AWAITING_VERBS
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        answer = ask_from_a_worker(app, run.id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await submit(pilot, "/allow")
        assert app.awaiting is None
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
        for name in commands.ALLOW_ANSWERS:
            assert f"/allow {name}" in text
        assert not answer.done()
        await submit(pilot, "/allow no")
        assert await answered(pilot, answer) == "no"


async def test_an_answer_that_is_not_one_of_the_four_is_refused() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        answer = ask_from_a_worker(app, schedulers[0].runs[0].id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await submit(pilot, "/allow maybe")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
        assert "maybe" in text
        assert not answer.done()
        await submit(pilot, "/allow no")
        assert await answered(pilot, answer) == "no"


async def test_a_cancelled_run_answers_its_own_question_with_no() -> None:
    """Nothing large is installed for a run that is over (rule 5), and no thread waits."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        answer = ask_from_a_worker(app, schedulers[0].runs[0].id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await submit(pilot, "/cancel 1")
        assert await answered(pilot, answer) == "no"
        assert app.pending == {}


async def test_quitting_answers_every_open_question_before_closing() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        answer = ask_from_a_worker(app, schedulers[0].runs[0].id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await submit(pilot, "/quit")
        await pilot.pause()
    assert answer.result(timeout=15) == "no"
    assert schedulers[0].closed


async def test_a_prompted_event_is_shown_rather_than_dropped() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.push(scheduler.runs[0], Prompted("permission asked for x.test"))
        await pilot.pause()
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert "permission asked for x.test" in text


# --- task 8.4: rule 4 — the gate the app builds asks the TUI, never the terminal ------


async def test_the_engine_is_built_interactive_with_the_tui_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_default(config: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(verify_mod.Engine, "default", staticmethod(fake_default))
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        # Exactly what the scheduler does on the run's own thread.
        app._default_engine(run)
    assert captured["interactive"] is True  # a TUI is a terminal
    assert captured["prompt"] is not None  # ...but not one the gate may read


async def test_the_gate_asks_the_inline_prompt_and_never_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 4: inside Textual, ``ask_terminal`` would read a stdin Textual is holding."""
    captured: dict[str, Any] = {}

    def fake_default(config: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(verify_mod.Engine, "default", staticmethod(fake_default))
    monkeypatch.setattr(
        browser.typer, "prompt", lambda *a, **k: pytest.fail("the gate read the terminal")
    )
    installed: list[str] = []
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        app._default_engine(schedulers[0].runs[0])
        gate = ConsentGate(
            "ask",
            interactive=True,
            prompt=captured["prompt"],
            installer=lambda log: installed.append("install") is None,
        )
        allowed = in_a_worker(lambda: gate.allow(HOST, 403))
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await click(pilot, "#allow-no")
        assert await answered(pilot, allowed) is False
    # Rule 5: the install happens through the gate, after an answer, and "no" is an
    # answer that installs nothing.
    assert gate.decision.reason == "user answered no"
    assert installed == []


# --- task 8.4: the mirrored verbs (spec 13.3) ----------------------------------------


def a_resolved() -> library.Resolved:
    best = Candidate(
        doi="10.1038/s41586-021-03819-2",
        title="Highly accurate protein structure prediction with AlphaFold",
        first_author="Jumper",
        year=2021,
        venue="Nature",
        provider="crossref",
    )
    return library.Resolved(
        ResolveResult(ResolveState.RESOLVED, best, [best], notes=["a note"]), None
    )


async def lines_of(app: ProofpathApp, pilot: Any) -> str:
    await until(
        pilot,
        lambda: bool(app.query(CommandBlock)) and bool(app.query(KvLine)),
        "the command block to be filled in",
    )
    return "\n".join(line.render().plain for line in app.query(KvLine))


async def test_resolve_calls_the_same_library_function_the_cli_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def stub(reference: str, *, config: Config) -> library.Resolved:
        asked.append(reference)
        return a_resolved()

    monkeypatch.setattr(library, "resolve_reference", stub)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/resolve Jumper 2021")
        text = await lines_of(app, pilot)
    assert asked == ["Jumper 2021"]
    assert "RESOLVED" in text
    assert "not retracted" in text
    assert "a note" in text
    assert "https://doi.org/10.1038/s41586-021-03819-2" in text


async def test_resolve_renders_a_failed_retraction_check_like_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror rule (spec 13.3): the CLI's ``retraction  unavailable (...)`` line,
    never ``not retracted`` for a check nobody answered (rule 2)."""
    resolved = a_resolved()
    failed = library.Resolved(
        resolved.result, None, retraction_error="crossref and openalex unavailable (HTTP 503)"
    )
    monkeypatch.setattr(library, "resolve_reference", lambda reference, *, config: failed)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/resolve Jumper 2021")
        text = await lines_of(app, pilot)
    assert "RESOLVED" in text
    assert "retraction unavailable (crossref and openalex unavailable (HTTP 503))" in text
    assert "not retracted" not in text


async def test_a_mirrored_verb_that_fails_says_so_instead_of_dying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(reference: str, *, config: Config) -> library.Resolved:
        raise OSError("no route to host")

    monkeypatch.setattr(library, "resolve_reference", boom)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/resolve Jumper 2021")
        text = await lines_of(app, pilot)
        assert app.is_running  # reported in the block, not raised at the loop
    assert "OSError: no route to host" in text


async def test_fetch_is_wired_with_the_inline_prompt_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``/fetch`` that hits the section 7.1 wall is answered in its own block."""
    seen: dict[str, Any] = {}

    def stub(target: str, **kwargs: Any) -> Any:
        seen.update(kwargs, target=target)
        raise OSError("stopped here")

    monkeypatch.setattr(library, "fetch_target", stub)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/fetch https://x.test/p.html")
        await lines_of(app, pilot)
    assert seen["target"] == "https://x.test/p.html"
    assert seen["interactive"] is True
    assert seen["prompt"] is not None


async def test_config_show_prints_the_path_and_the_toml() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/config show")
        text = await lines_of(app, pilot)
    assert "[permissions]" in text
    assert "install_browser" in text


async def test_config_set_writes_through_the_same_function_the_cli_uses() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/config set permissions.install_browser deny")
        text = await lines_of(app, pilot)
    assert "permissions.install_browser = deny" in text
    assert library.config_view().config.permissions.install_browser == "deny"


async def test_cache_says_what_it_holds() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/cache")
        text = await lines_of(app, pilot)
    assert "0 sources" in text
    await_empty = "0 resolutions"
    assert await_empty in text


async def test_cache_ls_on_an_empty_cache_says_it_is_empty() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/cache ls")
        text = await lines_of(app, pilot)
    assert "cache is empty" in text


async def test_an_unknown_subcommand_is_reported_not_guessed() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/cache nope")
        text = await lines_of(app, pilot)
    assert "nope" in text and "try path, ls, show or clear" in text


async def test_help_lists_every_verb_with_what_it_wants() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    for verb in commands.VERBS:
        assert f"/{verb}" in text
    for verb, placeholder in commands.NEEDS_ARGUMENT.items():
        assert f"/{verb} {placeholder}" in text


async def test_help_lists_the_keys() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        notes = [str(note.render()) for note in app.query(NoteLine)]
        assert any("history" in note and "Tab" in note for note in notes)
        assert any("Ctrl+L" in note for note in notes)


async def test_help_keys_are_ascii_in_plain_theme() -> None:
    app, _ = build_app(theme=PLAIN)
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        notes = [str(note.render()) for note in app.query(NoteLine)]
    assert "up/down" in "\n".join(notes)
    for text in notes:
        assert text.isascii(), f"non-ASCII help note in PLAIN theme: {text!r}"


# --- task 8.4: the mouse, and its keyboard equivalent (spec 13.1) --------------------


async def test_a_click_on_a_finding_opens_the_whole_passage() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        assert not line.expanded
        await click(pilot, FindingLine, offset=(2, 0))
        await pilot.pause()
        assert line.expanded
        assert "we observed a 4-8% improvement in throughput" in line.render().plain


async def test_enter_on_a_focused_finding_does_what_the_click_does() -> None:
    """Nothing is mouse-only and nothing is keyboard-only (spec section 13.1)."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        line.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert line.expanded
        await pilot.press("enter")
        await pilot.pause()
        assert not line.expanded


async def test_c_copies_the_passage_and_the_glyph_click_does_the_same() -> None:
    copied: list[str] = []
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        line.focus()
        await pilot.press("c")
        await pilot.pause()
        assert copied == ["we observed a 4-8% improvement in throughput"]
        row, column = line._copy_cell
        await click(pilot, FindingLine, offset=(column, row))
        await pilot.pause()
    assert copied == ["we observed a 4-8% improvement in throughput"] * 2
    assert not line.expanded  # the glyph copies; it does not also toggle


async def test_a_finding_links_its_source(monkeypatch: pytest.MonkeyPatch) -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        rendered = app.query_one(FindingLine).render()
    styles = " ".join(str(span.style) for span in rendered.spans)
    assert "link https://doi.org/x" in styles


async def test_a_click_on_the_header_folds_the_run_away_but_never_its_state() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.push(run, StageStart("Parse", "pymupdf"))
        scheduler.move(run, "done", report=a_report())
        await pilot.pause()
        block = app.query_one(RunBlock)
        await click(pilot, RunHeader, offset=(2, 0))
        await pilot.pause()
        assert block.collapsed
        assert not block.query_one(".stages").display
        assert not block.query_one(".findings").display
        # Rule 6: the header, its state and its coverage never fold away.
        assert block.header.display
        assert "coverage" in block.header.render().plain
        await click(pilot, RunHeader, offset=(2, 0))
        await pilot.pause()
        assert not block.collapsed


async def test_a_click_on_a_stage_hides_its_detail_and_keeps_its_attribution() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.push(run, StageStart("Parse", "pymupdf"))
        scheduler.push(run, StageEnd("Parse", "pymupdf", "24 pages, 42 refs", 1.2))
        await pilot.pause()
        stage = app.query_one(StageLine)
        await click(pilot, StageLine, offset=(2, 0))
        await pilot.pause()
        text = stage.render().plain
    assert "24 pages, 42 refs" not in text
    assert "Parse" in text and "1.2s" in text


# --- fix round 1: one engine-factory type, the scheduler's ---------------------------


def test_the_app_and_the_scheduler_agree_on_what_an_engine_factory_is() -> None:
    """A second spelling here would let a zero-argument factory type-check (8.4 review).

    The scheduler calls the factory with the run it is building for, because the
    engine's consent gate has to know whose block to ask. An ``EngineFactory`` in
    ``app.py`` that took no arguments would pass mypy and then be handed a ``Run``.
    """
    from typing import get_args, get_type_hints

    import proofpath.tui.app as app_mod
    import proofpath.tui.runs as runs_mod

    assert app_mod.EngineFactory is runs_mod.EngineFactory
    hint = get_type_hints(ProofpathApp.__init__)["engine_factory"]
    factory = next(arg for arg in get_args(hint) if arg is not type(None))
    assert get_args(factory) == ([Run], Engine)


# --- fix round 1: a persisted answer is honoured by the session that gave it ---------


def a_gate(app: ProofpathApp, owner: int) -> ConsentGate:
    """A real gate wired to the app's prompt, the way ``_default_engine`` wires one.

    ``config_path`` is passed rather than left to default, although the ``isolated``
    fixture already points ``PROOFPATH_CONFIG_DIR`` at ``tmp_path``: an "always" or a
    "never" answer makes the gate *write* a config file, and the file it writes is
    pinned here to the temp one this test reads back, not left to an environment
    variable a future edit could stop setting. It is the same path either way, so the
    app still re-reads exactly what the gate wrote.
    """
    return ConsentGate(
        "ask",
        interactive=True,
        prompt=app._prompt_for(owner),
        installer=lambda log: pytest.fail("nothing may be installed"),
        config_path=config_path(),
    )


async def test_never_is_honoured_by_the_run_after_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Never ask again" that asks again on the next run has not been honoured."""
    built: list[Config] = []

    def fake_default(config: Config, **kwargs: Any) -> object:
        built.append(config)
        return object()

    monkeypatch.setattr(verify_mod.Engine, "default", staticmethod(fake_default))
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        assert app._config.permissions.install_browser == "ask"
        gate = a_gate(app, run.id)
        allowed = in_a_worker(lambda: gate.allow(HOST, 403))
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        await click(pilot, "#allow-never")
        assert await answered(pilot, allowed) is False
        await until(
            pilot,
            lambda: library.config_view().config.permissions.install_browser == "deny",
            "the gate to write the answer to the config file",
        )
        # The next run re-reads it, so its engine is built with a config that never asks.
        await submit(pilot, "/check second.md")
        assert app._config.permissions.install_browser == "deny"
        app._default_engine(schedulers[0].runs[1])
    assert built[-1].permissions.install_browser == "deny"
    assert library.config_view().config.permissions.install_browser == "deny"


async def test_config_set_changes_what_the_rest_of_the_session_decides_by() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/config set permissions.install_browser deny")
        await lines_of(app, pilot)
        await until(
            pilot,
            lambda: app._config.permissions.install_browser == "deny",
            "the session to re-read its own config",
        )
        # And a run started afterwards decides by the new value, not the launch one.
        await submit(pilot, "/check draft.md")
        assert app._config.permissions.install_browser == "deny"


async def test_config_set_of_the_network_permission_redraws_the_banner() -> None:
    """The version row says online/offline; a session that changed it must not still say online."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        context = run_context(Config())
        row = wordmark.rows(80, version=__version__, context=context, hint=HINT).version
        assert " online " in app.query_one(Banner).drawn.lines[row]
        await submit(pilot, "/config set permissions.network deny")
        await lines_of(app, pilot)
        await until(
            pilot,
            lambda: " offline " in _banner_lines(app)[row],
            "the banner to be redrawn offline",
        )


# --- fix round 1: the quit race ------------------------------------------------------


async def test_a_question_asked_while_quitting_is_answered_without_being_drawn() -> None:
    """A worker that reaches the gate while ``close()`` is joining it must not park."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        scheduler = schedulers[0]
        original = scheduler.close
        late: list[Any] = []

        async def close_and_ask() -> None:
            # The worker asks *after* the quit began: it must get an answer straight
            # back rather than a widget nobody will ever see.
            late.append(ask_from_a_worker(app, run.id).result(timeout=15))
            await original()

        scheduler.close = close_and_ask  # type: ignore[method-assign]
        await submit(pilot, "/quit")
        await pilot.pause()
    assert late == ["no"]
    assert scheduler.closed
    assert app.pending == {}
    assert not app.query(PermissionPrompt)  # nothing was drawn on the way out


# --- fix round 1: the minors ---------------------------------------------------------


async def test_a_fetch_that_hits_the_wall_is_answered_in_its_own_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers: list[str] = []

    def stub(target: str, **kwargs: Any) -> Any:
        answers.append(kwargs["prompt"](HOST, 403))
        raise OSError("stopped after the question")

    monkeypatch.setattr(library, "fetch_target", stub)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/fetch https://x.test/p.html")
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        prompt = app.query_one(PermissionPrompt)
        assert any(isinstance(parent, CommandBlock) for parent in prompt.ancestors_with_self)
        assert prompt.owner < 0  # a command block, not a run
        await click(pilot, "#allow-no")
        await lines_of(app, pilot)
    assert answers == ["no"]


async def test_a_link_survives_no_color() -> None:
    """Underline is not a colour and an address is not meaning (spec section 13.3)."""
    app, schedulers = build_app(out=ui.build(force_terminal=True, no_color=True))
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        rendered = app.query_one(FindingLine).render()
    assert "link https://doi.org/x" in " ".join(str(span.style) for span in rendered.spans)


async def test_an_unreadable_config_is_reported_not_raised(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        (tmp_path / "conf").mkdir(exist_ok=True)
        (tmp_path / "conf" / "config.toml").write_text("this is not toml [[", encoding="utf-8")
        await submit(pilot, "/config show")
        text = await lines_of(app, pilot)
        assert app.is_running
    assert text.startswith("  error: ")


# --- completion ---------------------------------------------------------------------


async def test_tab_completes_a_verb() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        prompt.focus()
        prompt.value = "/ch"
        prompt.cursor_position = 3
        await pilot.press("tab")
        assert prompt.value == "/check "
        assert app.focused is prompt


async def test_tab_cycles_through_the_allow_answers() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        prompt.focus()
        prompt.value = "/allow "
        prompt.cursor_position = 7
        await pilot.press("tab")
        assert prompt.value == "/allow once"
        await pilot.press("tab")
        assert prompt.value == "/allow always"
        await pilot.press("tab")
        await pilot.press("tab")
        await pilot.press("tab")  # wraps
        assert prompt.value == "/allow once"


async def test_tab_offers_the_runs_that_can_be_cancelled() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await submit(pilot, "/check two.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        prompt.value = "/cancel "
        prompt.cursor_position = 8
        await pilot.press("tab")
        assert prompt.value == "/cancel #2"


async def test_tab_on_a_target_moves_focus_instead() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        prompt = app.query_one(Prompt)
        prompt.value = "paper.pdf"
        await pilot.press("tab")
        assert prompt.value == "paper.pdf"
        assert app.focused is not prompt


async def test_up_after_a_completion_moves_the_selection_not_the_history() -> None:
    """Once ``Tab`` has shown the cycle, ``up`` is the list's (wordmark design section
    10); a history step can only happen once the list is gone, and it then starts a
    fresh cycle over the recalled line."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        prompt = app.query_one(Prompt)
        prompt.value = "/allow "
        prompt.cursor_position = 7
        await pilot.press("tab")
        assert prompt.value == "/allow once"
        await pilot.press("up")  # the last row, not the history
        assert prompt.value == "/allow once"
        assert prompt.selected == len(commands.ALLOW_ANSWERS) - 1
        prompt.value = ""  # an edit: the cycle and the list go
        await pilot.pause()
        await pilot.press("up")
        assert prompt.value == "/check one.pdf"
        await pilot.press("tab")  # a fresh cycle over the recalled line: nothing to complete
        assert prompt.value == "/check one.pdf"


async def test_an_edit_ends_the_completion_cycle() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        prompt.focus()
        prompt.value = "/allow "
        prompt.cursor_position = 7
        await pilot.press("tab")
        assert prompt.value == "/allow once"
        await pilot.press("backspace")
        await pilot.press("tab")  # fresh cycle from "/allow onc": the first match again
        assert prompt.value == "/allow once"


async def test_tab_on_a_slash_line_with_nothing_to_complete_keeps_focus() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        prompt = app.query_one(Prompt)
        prompt.focus()
        prompt.value = "/check paper.pdf"
        prompt.cursor_position = len(prompt.value)
        await pilot.press("tab")
        assert prompt.value == "/check paper.pdf"
        assert app.focused is prompt


# --- slash suggestions (wordmark design section 10) ---------------------------------


async def type_into(pilot: Any, text: str) -> None:
    """Type ``text`` into the bar one key at a time, so ``Changed`` fires per keystroke."""
    prompt = pilot.app.query_one(Prompt)
    prompt.focus()
    await pilot.press(*text)
    await pilot.pause()


def _completion(verb: str) -> str:
    return f"/{verb} " if verb in commands.NEEDS_ARGUMENT else f"/{verb}"


async def test_a_slash_lists_every_verb_in_order_with_the_first_selected() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await type_into(pilot, "/")
        prompt = app.query_one(Prompt)
        suggestions = app.query_one(Suggestions)
        assert suggestions.visible
        assert prompt.value == "/"
        assert prompt.candidates == tuple(_completion(verb) for verb in commands.VERBS)
        assert prompt.selected == 0
        assert len(suggestions.held) == len(commands.VERBS)
        for row, verb in zip(suggestions.held, commands.VERBS, strict=True):
            # ``plain`` marks the selected row and indents the others under it.
            assert row.startswith("> ") if verb == commands.VERBS[0] else row.startswith("  ")
            body = row[2:]
            assert body.startswith(f"/{verb}")
            assert commands.NEEDS_ARGUMENT.get(verb, "") in body
            assert body.endswith(commands.DESCRIPTIONS[verb])
        # Three columns: every completion, placeholder and description starts on the
        # same column as the others of its kind.
        columns = {
            row.index(commands.DESCRIPTIONS[verb])
            for row, verb in zip(suggestions.held, commands.VERBS, strict=True)
        }
        assert len(columns) == 1


async def test_the_list_narrows_per_keystroke_and_goes_when_nothing_matches() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/h")
        assert suggestions.visible
        assert [row[2:].split("  ")[0] for row in suggestions.held] == ["/help"]
        await type_into(pilot, "e")
        assert suggestions.visible
        assert app.query_one(Prompt).candidates == ("/help",)
        await pilot.press("backspace", "backspace")
        await type_into(pilot, "x")
        assert app.query_one(Prompt).value == "/x"
        assert suggestions.visible is False


async def test_a_verb_with_its_space_suggests_nothing_but_allow_and_cancel_do() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/check ")
        assert suggestions.visible is False
        await pilot.press(*(["backspace"] * len("/check ")))
        await type_into(pilot, "/allow ")
        assert suggestions.visible
        assert [row[2:].split("  ")[0] for row in suggestions.held] == [
            f"/allow {answer}" for answer in commands.ALLOW_ANSWERS
        ]
        # An answer has no placeholder of its own; the description is the verb's.
        assert all(row.endswith(commands.DESCRIPTIONS["allow"]) for row in suggestions.held)
        assert all(commands.NEEDS_ARGUMENT["allow"] not in row for row in suggestions.held)
        await pilot.press(*(["backspace"] * len("/allow ")))
        await submit(pilot, "/check one.pdf")
        assert schedulers[0].runs[0].state == "queued"
        await type_into(pilot, "/cancel ")
        assert suggestions.visible
        assert [row[2:].split("  ")[0] for row in suggestions.held] == ["/cancel #1"]


async def test_arrows_move_the_selection_and_tab_takes_the_selected_row() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/")
        await pilot.press("down")
        assert prompt.value == "/"
        assert prompt.selected == 1
        assert suggestions.held[1].startswith("> ") and suggestions.held[0].startswith("  ")
        await pilot.press("up")
        assert prompt.selected == 0
        await pilot.press("up")  # wraps to the last row
        assert prompt.selected == len(commands.VERBS) - 1
        await pilot.press("down", "down", "down")
        chosen = prompt.candidates[prompt.selected]
        assert chosen == _completion("fetch")
        await pilot.press("tab")
        assert prompt.value == chosen
        # The cycle stays on show, with its siblings, so a further ``tab`` steps on.
        assert suggestions.visible
        assert prompt.selected == 2 and suggestions.held[2].startswith("> ")
        await pilot.press("tab")
        assert prompt.value == _completion("config")
        await pilot.press("down")  # moves the selection, leaves the bar alone
        assert prompt.value == _completion("config")
        assert prompt.selected == 4
        await pilot.press("tab")
        assert prompt.value == _completion("cache")


async def test_enter_runs_what_is_in_the_bar_and_the_list_goes() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/help")
        assert suggestions.visible
        await pilot.press("enter")
        await pilot.pause()
        assert f"/check {commands.NEEDS_ARGUMENT['check']}" in _notes(app)
        assert app.query_one(Prompt).value == ""
        assert suggestions.visible is False


async def test_the_selection_never_walks_the_history_and_a_recalled_line_shows_no_list() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/cache")  # bare ``/config`` opens a panel that takes focus
        prompt = app.query_one(Prompt)
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/")
        await pilot.press("down")
        assert prompt.value == "/"  # the list's, not the history's
        await pilot.press("up")
        assert prompt.value == "/"
        await pilot.press("backspace")
        assert suggestions.visible is False
        # With the list hidden the keys are the walk again, and a recalled line does
        # not open the list over itself: the next ``up`` walks on.
        await pilot.press("up")
        assert prompt.value == "/cache"
        assert suggestions.visible is False
        await pilot.press("up")
        assert prompt.value == "/help"
        await pilot.press("down", "down")
        assert prompt.value == ""
        # An edit of a recalled line is typing again: the list comes back.
        await pilot.press("up")
        await type_into(pilot, "x")
        assert prompt.value == "/cachex"
        assert suggestions.visible is False
        await pilot.press("backspace", "backspace", "backspace")
        assert prompt.value == "/cac"
        assert suggestions.visible


async def test_tab_on_a_recalled_slash_line_brings_the_list_back() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/con")  # unknown, but any submitted line is one of history
        prompt = app.query_one(Prompt)
        suggestions = app.query_one(Suggestions)
        await pilot.press("up")
        assert prompt.value == "/con"
        assert suggestions.visible is False  # a recalled line shows no list over itself
        await pilot.press("tab")
        # "con" completes to one candidate; ``tab`` both takes it and reopens the list.
        assert prompt.value == "/config"
        assert suggestions.visible
        assert prompt.candidates == ("/config",)
        assert prompt.selected == 0
        assert suggestions.held[0].endswith(commands.DESCRIPTIONS["config"])


async def test_the_list_never_covers_the_log_and_the_bottom_grows_by_its_rows() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        bottom = app.query_one("#bottom")
        log = app.query_one(RunLog)
        suggestions = app.query_one(Suggestions)
        frame = app.query_one(PromptFrame)
        footer = app.query_one(CoverageFooter)
        assert bottom.region.height == 4
        assert footer.region.height == 3 and frame.region.height == 1
        log_height = log.region.height
        await type_into(pilot, "/c")
        await pilot.pause()
        assert suggestions.visible
        assert len(suggestions.held) == 4
        assert suggestions.drawn == suggestions.held  # four candidates, nothing capped
        assert bottom.region.height == 4 + len(suggestions.drawn)
        assert suggestions.region.height == len(suggestions.drawn)
        assert log.region.height == log_height - len(suggestions.drawn)
        assert log.region.bottom <= suggestions.region.y
        assert footer.region.bottom <= suggestions.region.y
        assert suggestions.region.bottom <= frame.region.y
        await pilot.press("backspace")
        await pilot.pause()
        # ``/`` alone lists every verb, and with the shortened ``/summarize``
        # description (item 1) none of the ten rows wraps at eighty columns: at
        # 80x24 they all fit, so there is nothing to cap and no indicator.
        banner = app.query_one("#banner").region.height
        assert len(suggestions.held) == len(commands.VERBS)
        assert suggestions.window == (0, len(commands.VERBS))
        assert suggestions.drawn == suggestions.held
        assert suggestions.region.height == len(commands.VERBS)
        assert bottom.region.height == 4 + len(commands.VERBS)
        assert log.region.bottom == footer.region.y
        assert footer.region.bottom == suggestions.region.y
        assert suggestions.region.bottom == frame.region.y
        assert not app.screen.show_vertical_scrollbar
        assert bottom.region.width == SIZE[0]

        # A shorter terminal is what forces the cap now (not a wrapped row): the
        # banner, the footer (3) and the bar (1) plus one row of log leave fewer
        # than ten lines at 80x20, so the window shows a run of rows and an
        # ``... n more`` line -- itself counted against the cap -- fills the rest.
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        limit = 20 - banner - footer.rows - frame.rows - 1
        assert 0 < limit < len(commands.VERBS)
        assert suggestions.visible
        first, last = suggestions.window
        assert (first, last) != (0, len(commands.VERBS))
        assert first == 0  # the selection (row 0) is still at the top
        assert len(suggestions.drawn) == limit
        assert suggestions.drawn[:-1] == suggestions.held[first:last]
        assert suggestions.drawn[-1] == f"... {len(commands.VERBS) - last} more"
        assert bottom.region.height == 4 + limit
        assert suggestions.region.height == limit
        assert log.region.height == 1
        assert log.region.bottom <= suggestions.region.y
        assert footer.region.bottom <= suggestions.region.y
        assert suggestions.region.bottom <= frame.region.y
        assert not app.screen.show_vertical_scrollbar
        assert bottom.region.width == 80

        # The window slides with the selection: ``up`` wraps to ``/quit``, off the
        # bottom of the first window, and the top gives way to an indicator instead.
        await pilot.press("up")
        await pilot.pause()
        assert app.query_one(Prompt).selected == len(commands.VERBS) - 1
        first, last = suggestions.window
        assert first > 0 and last == len(commands.VERBS)
        assert len(suggestions.drawn) == limit
        assert suggestions.drawn[0] == f"... {first} more"
        assert suggestions.drawn[1:] == suggestions.held[first:last]
        assert suggestions.drawn[-1].endswith(commands.DESCRIPTIONS["quit"])
        assert bottom.region.height == 4 + limit

        # Taller again: the window and the indicator both give the rows back.
        await pilot.resize_terminal(*SIZE)
        await pilot.pause()
        assert suggestions.window == (0, len(commands.VERBS))
        assert suggestions.drawn == suggestions.held
        assert bottom.region.height == 4 + len(commands.VERBS)
        assert log.region.height == 1
        assert not app.screen.show_vertical_scrollbar

        await pilot.press("backspace")
        await pilot.pause()
        assert suggestions.visible is False
        assert bottom.region.height == 4
        assert log.region.height == log_height


async def test_the_more_indicator_marks_only_the_rows_the_cap_hides() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/")
        # At SIZE (80x24) all ten rows fit, nothing is hidden: no indicator either end.
        assert suggestions.window == (0, len(commands.VERBS))
        assert suggestions.drawn == suggestions.held
        assert not any(row.startswith("...") for row in suggestions.drawn)

        # 80x20 is short enough that fewer than ten rows fit: the bottom indicator
        # takes the last drawn line, muted, with the count of rows below the window.
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        first, last = suggestions.window
        assert first == 0
        assert last < len(commands.VERBS)
        assert suggestions.drawn[-1] == f"... {len(commands.VERBS) - last} more"
        assert suggestions.drawn[0] != suggestions.drawn[-1]  # only the one end hides

        # ``up`` wraps the selection to ``/quit``, off the bottom of that window:
        # the window slides down and the *top* indicator takes the first drawn line.
        await pilot.press("up")
        await pilot.pause()
        first, last = suggestions.window
        assert first > 0
        assert last == len(commands.VERBS)
        assert suggestions.drawn[0] == f"... {first} more"
        assert suggestions.drawn[-1].endswith(commands.DESCRIPTIONS["quit"])

        # Back to SIZE, everything fits again and the indicator goes.
        await pilot.resize_terminal(*SIZE)
        await pilot.pause()
        assert suggestions.window == (0, len(commands.VERBS))
        assert not any(row.startswith("...") for row in suggestions.drawn)


async def test_the_indicator_is_dropped_when_the_cap_leaves_room_for_only_the_row() -> None:
    """At ``limit == 1`` there is room for the selected row and nothing else: an
    ``... n more`` indicator would already overrun the cap on its own, so neither
    end draws one, however many rows sit outside the window."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/")
        await pilot.resize_terminal(80, 15)
        await pilot.pause()
        assert app._suggestion_lines(app.size) == 1
        first, last = suggestions.window
        assert last - first == 1
        assert suggestions.drawn == (suggestions.held[first],)
        assert not any(row.startswith("...") for row in suggestions.drawn)

        # The selection can sit in the middle of the list, hidden on both sides:
        # the row alone still shows, still with no indicator either end.
        for _ in range(4):
            await pilot.press("down")
        await pilot.pause()
        first, last = suggestions.window
        assert first > 0
        assert last < len(commands.VERBS)
        assert last - first == 1
        assert suggestions.drawn == (suggestions.held[first],)
        assert not any(row.startswith("...") for row in suggestions.drawn)


async def test_the_indicator_keeps_the_bottom_row_when_the_cap_leaves_room_for_one() -> None:
    """At ``limit == 2`` there is room for the selected row and one indicator. With
    rows hidden on both sides of the selection, only one indicator fits, and the
    bottom one is the one that stays."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        suggestions = app.query_one(Suggestions)
        await type_into(pilot, "/")
        await pilot.resize_terminal(80, 16)
        await pilot.pause()
        assert app._suggestion_lines(app.size) == 2

        # Move the selection into the middle of the list: rows are hidden above
        # and below it.
        for _ in range(4):
            await pilot.press("down")
        await pilot.pause()
        first, last = suggestions.window
        assert first > 0
        assert last < len(commands.VERBS)
        assert last - first == 1
        assert suggestions.drawn == (
            suggestions.held[first],
            f"... {len(commands.VERBS) - last} more",
        )


async def test_a_held_paste_never_opens_the_list() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, "/help\n/config")
        prompt = app.query_one(Prompt)
        assert prompt.held == "/help\n/config"
        assert prompt.value.startswith("pasted")
        assert HELD_SUMMARY.startswith("pasted")
        assert app.query_one(Suggestions).visible is False


# --- the log from the keyboard ------------------------------------------------------


async def _tall_log(pilot: Any, app: ProofpathApp) -> None:
    """Enough help lines to overflow 24 rows, so there is something to scroll."""
    for _ in range(6):
        await submit(pilot, "/help")
    await pilot.pause()
    assert app.query_one(RunLog).max_scroll_y > 0


async def test_pageup_scrolls_the_log_without_leaving_the_bar() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await _tall_log(pilot, app)
        log = app.query_one(RunLog)
        at_end = log.scroll_y
        await pilot.press("pageup")
        await pilot.pause()
        assert log.scroll_y < at_end
        assert app.focused is app.query_one(Prompt)
        after_pageup = log.scroll_y
        await pilot.press("pagedown")
        await pilot.pause()
        assert log.scroll_y > after_pageup
        await pilot.press("ctrl+end")
        await pilot.pause()
        assert log.scroll_y == at_end
        await pilot.press("ctrl+home")
        await pilot.pause()
        assert log.scroll_y == 0
        await pilot.press("shift+down")
        await pilot.pause()
        assert log.scroll_y == 1
        await pilot.press("shift+up")
        await pilot.pause()
        assert log.scroll_y == 0


async def test_up_and_down_walk_the_lines() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.move(run, "running")
        scheduler.push(run, StageStart("Parse", "pymupdf"))
        scheduler.push(run, StageEnd("Parse", "pymupdf", "24 pages, 42 refs", 1.2))
        scheduler.push(run, StageStart("Resolve references", "Crossref, OpenAlex"))
        await pilot.pause()
        header = app.query_one(RunHeader)
        header.focus()
        await pilot.press("down")
        assert isinstance(app.focused, StageLine)
        assert app.focused.stage == "Parse"
        await pilot.press("down")
        assert isinstance(app.focused, StageLine)
        assert app.focused.stage == "Resolve references"
        await pilot.press("down")  # last line: stays
        assert isinstance(app.focused, StageLine)
        assert app.focused.stage == "Resolve references"
        await pilot.press("up")
        await pilot.press("up")
        assert app.focused is header
        await pilot.press("up")  # first line: stays
        assert app.focused is header


async def test_a_collapsed_block_is_skipped_by_the_walk() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await submit(pilot, "/check two.pdf")
        scheduler = schedulers[0]
        for run in scheduler.runs:
            scheduler.move(run, "running")
            scheduler.push(run, StageStart("Parse", "pymupdf"))
        await pilot.pause()
        first, second = app.query(RunHeader)
        first.focus()
        await pilot.press("enter")  # collapse the first block
        await pilot.press("down")
        assert app.focused is second


async def test_typing_on_a_line_goes_to_the_bar() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await pilot.pause()
        app.query_one(RunHeader).focus()
        await pilot.press("slash")
        prompt = app.query_one(Prompt)
        assert app.focused is prompt
        assert prompt.value == "/"
        await pilot.press("h")
        assert prompt.value == "/h"


async def test_typing_on_a_line_keeps_the_bar_draft() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        prompt.value = "check pa"
        prompt.cursor_position = len(prompt.value)
        app.query_one(RunHeader).focus()
        await pilot.pause()
        await pilot.press("p")
        assert app.focused is prompt
        assert prompt.value == "check pap"


# --- typing reaches the bar (wordmark design section 11) ----------------------------


async def test_a_click_on_the_log_background_keeps_the_bar_focused() -> None:
    """The log is not focusable: a click on its empty background is not a departure."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        await click(pilot, RunLog, offset=(10, 5))
        await pilot.pause()
        assert app.focused is prompt
        await pilot.press("x")
        assert prompt.value == "x"


async def test_a_click_on_the_banner_keeps_the_bar_focused() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        await click(pilot, Banner)
        await pilot.pause()
        assert app.focused is prompt
        await pilot.press("x")
        assert prompt.value == "x"


async def test_typing_after_clicking_a_finding_goes_to_the_bar() -> None:
    """A click on a line still focuses it; the first printable key after that is
    typing and takes focus back to the bar, except ``c``, which is the line's copy."""
    copied: list[str] = []
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        prompt = app.query_one(Prompt)
        await click(pilot, FindingLine, offset=(2, 0))
        await pilot.pause()
        assert app.focused is line
        await pilot.press("h")
        assert app.focused is prompt
        assert prompt.value == "h"
        await click(pilot, FindingLine, offset=(2, 0))
        await pilot.pause()
        assert app.focused is line
        await pilot.press("c")
        await pilot.pause()
        assert prompt.value == "h"
        assert app.focused is line
        assert copied == ["we observed a 4-8% improvement in throughput"]


async def test_a_slash_after_clicking_a_finding_brings_the_list_up() -> None:
    """The forwarded key goes through the bar's own path, so the suggestions follow
    it (wordmark design section 10)."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        await click(pilot, FindingLine, offset=(2, 0))
        await pilot.pause()
        assert app.focused is app.query_one(FindingLine)
        await pilot.press("slash")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert app.focused is prompt
        assert prompt.value == "/"
        assert app.query_one(Suggestions).visible
        assert len(app.query_one(Suggestions).held) == len(commands.VERBS)


async def test_typing_on_a_permission_button_goes_to_the_bar() -> None:
    """A printable key on a button is typing; ``enter`` has no printable character,
    never reaches the forwarding, and is still the button's own."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        answer = ask_from_a_worker(app, run.id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        button = app.query_one("#allow-once", Button)
        prompt = app.query_one(Prompt)
        button.focus()
        await pilot.pause()
        assert app.focused is button
        await pilot.press("y")
        assert app.focused is prompt
        assert prompt.value == "y"
        assert not answer.done()
        button.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await answered(pilot, answer) == "once"
        assert prompt.value == "y"


async def test_space_on_a_permission_button_is_typing_too() -> None:
    """Textual's ``Button`` binds ``enter`` alone and has no key handler, so
    ``space`` never pressed a button; it is a printable key and goes to the bar.
    The binding is asserted so a Textual that starts binding ``space`` fails here
    loudly: the app's rule then keeps it with the button, and this test flips."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        run = schedulers[0].runs[0]
        answer = ask_from_a_worker(app, run.id)
        await until(pilot, lambda: app.query(PermissionPrompt), "the prompt")
        button = app.query_one("#allow-once", Button)
        prompt = app.query_one(Prompt)
        assert set(button._bindings.key_to_bindings) == {"enter"}
        button.focus()
        await pilot.pause()
        assert app.focused is button
        await pilot.press("space")
        assert app.focused is prompt
        assert prompt.value == " "
        assert not answer.done()
        button.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await answered(pilot, answer) == "once"


async def test_a_tuple_bound_comma_separated_key_stays_with_its_widget() -> None:
    """``_bound_keys`` walks ``BINDINGS`` through the public ``Binding.make_bindings``
    route rather than a widget's private ``_bindings`` cache, so it must handle both
    forms ``BINDINGS`` is allowed to hold: a bare ``(key, action, description)``
    tuple, and a single entry whose key is a comma-separated list. Neither ``h`` nor
    ``g`` should reach the bar; an unbound printable key still should."""

    class _CommaBoundWidget(Static, can_focus=True):
        BINDINGS: ClassVar[list[BindingType]] = [("h,g", "noop", "no-op")]

        def action_noop(self) -> None:
            pass

    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        widget = _CommaBoundWidget("custom")
        await app.query_one(RunLog).mount(widget)
        widget.focus()
        await pilot.pause()
        assert app.focused is widget

        await pilot.press("h")
        assert app.focused is widget
        assert prompt.value == ""

        await pilot.press("g")
        assert app.focused is widget
        assert prompt.value == ""

        await pilot.press("x")
        assert app.focused is prompt
        assert prompt.value == "x"


async def test_the_log_is_not_in_the_focus_chain() -> None:
    """``tab`` from the bar lands on the first line, never on the log itself."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        log = app.query_one(RunLog)
        assert not log.can_focus
        assert log not in app.screen.focus_chain
        prompt = app.query_one(Prompt)
        prompt.value = "paper.pdf"  # not a slash line: ``tab`` is Textual's focus-next
        await pilot.press("tab")
        assert app.focused is app.query_one(RunHeader)
        await pilot.press("shift+tab")
        assert app.focused is prompt


# --- clearing and copying -----------------------------------------------------------


async def test_ctrl_l_drops_finished_blocks_and_keeps_live_ones() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/check one.pdf")
        await submit(pilot, "/check two.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done")
        scheduler.move(scheduler.runs[1], "running")
        await pilot.pause()
        assert len(app.query(RunBlock)) == 2
        assert app.query(NoteLine)
        await pilot.press("ctrl+l")
        await pilot.pause()
        blocks = list(app.query(RunBlock))
        assert [block.run.id for block in blocks] == [2]
        assert not app.query(NoteLine)
        # The scheduler still knows the cleared run: /cancel and /summarize see it.
        assert scheduler.get(1) is not None


async def test_ctrl_l_keeps_a_running_blocks_own_notes() -> None:
    """A block's stage notes belong to the block, not to the log: they stay with it."""
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.move(run, "running")
        scheduler.push(run, StageStart("Parse", "pymupdf"))
        scheduler.push(run, Note("still parsing"))
        await pilot.pause()
        inside = len(app.query_one(RunBlock).query(NoteLine))
        assert inside >= 1
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert len(app.query_one(RunBlock).query(NoteLine)) == inside


async def test_ctrl_l_keeps_the_footer() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done", report=a_report())
        await pilot.pause()
        before = app.query_one(CoverageFooter).render().plain
        assert "no run yet" not in before
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert app.query_one(CoverageFooter).render().plain == before


async def test_summarize_after_ctrl_l_says_the_run_was_cleared() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done", report=a_report())
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        await submit(pilot, "/summarize")
        notes = _notes(app)
        assert CLEARED_NOT_SUMMARIZABLE in notes
        assert "finish a /check first" not in notes


async def test_a_late_event_for_a_cleared_run_is_noted_not_crashed() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.move(run, "done", report=a_report())
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert not app.query(RunBlock)
        # A worker's last words arriving after the clear: the run is finished, so
        # they must not raise a block from the dead, and they must not be lost either.
        scheduler.push(run, Note("late words"))
        scheduler.move(run, "done", report=a_report())
        await pilot.pause()
        assert not app.query(RunBlock)
        assert "#1" in _notes(app)


async def test_ctrl_c_with_a_selection_copies_instead_of_quitting() -> None:
    app, schedulers = build_app()
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        line = app.query(NoteLine).first()
        # Textual 8 has no public way to select programmatically; the private helper
        # is what its own mouse handling calls, so it is the closest thing to a drag.
        app.screen._select_all_in_widget(line)
        await pilot.pause()
        assert app.screen.get_selected_text() is not None
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert copied and "/check" in copied[0]
        assert app.is_running
    assert not schedulers[0].closed


async def test_ctrl_c_without_a_selection_quits() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert schedulers[0].closed


async def test_ctrl_d_quits_even_with_a_selection() -> None:
    app, schedulers = build_app()
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        app.screen._select_all_in_widget(app.query(NoteLine).first())
        await pilot.pause()
        await pilot.press("ctrl+d")
        await pilot.pause()
    assert schedulers[0].closed
    assert copied == []


# --- final review: cross-task interactions the per-task reviews could not see ---------


async def test_ctrl_c_quits_after_ctrl_l_removed_the_selected_widget() -> None:
    """A selection on a widget ``ctrl+l`` removed is dead, and dead is not a selection.

    Textual keeps the removed widget in ``screen.selections``, so
    ``get_selected_text`` comes back ``""`` rather than ``None``: without the fix
    ``ctrl+c`` copies nothing, forever, and never quits.
    """
    app, schedulers = build_app()
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        line = app.query(NoteLine).first()
        # Textual 8 has no public way to select programmatically; the private helper
        # is what its own mouse handling calls, so it is the closest thing to a drag.
        app.screen._select_all_in_widget(line)
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert not app.query(NoteLine)
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert copied == []
    assert schedulers[0].closed


class GatedJudge(FakeJudge):
    """A judge whose one call waits for the test to let it answer."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def summarize(self, report_markdown: str, *, max_tokens: int = 1500) -> str:
        self.release.wait(5)
        return super().summarize(report_markdown, max_tokens=max_tokens)


async def test_a_summary_in_flight_when_ctrl_l_fires_is_noted_not_lost() -> None:
    """The paragraph was paid for; a block that left the screen must not swallow it."""
    judge = GatedJudge()
    app, schedulers = build_app(judge_factory=lambda: judge)
    async with app.run_test(size=SIZE) as pilot:
        run = await finished_run(pilot, schedulers[0])
        await submit(pilot, "/summarize")
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert not app.query(RunBlock)
        judge.release.set()
        await until(pilot, lambda: judge.closed, "the summary worker")
        await pilot.pause()
        notes = _notes(app)
        assert f"#{run.id} summary:" in notes
        assert FakeJudge.PARAGRAPH in notes
        assert app.is_running


async def test_escape_clears_a_selection() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        app.screen._select_all_in_widget(app.query(NoteLine).first())
        await pilot.pause()
        assert app.screen.get_selected_text() is not None
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.get_selected_text() is None


async def test_ctrl_c_copies_a_selection_inside_the_bar() -> None:
    """``Input`` keeps its own selection (Shift+Home and friends); it is copied, not quit over."""
    app, schedulers = build_app()
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        prompt.value = "/check paper.pdf"
        prompt.selection = InputSelection(0, len(prompt.value))
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert copied == ["/check paper.pdf"]
        assert app.is_running
    assert not schedulers[0].closed


async def test_an_unknown_verb_note_is_ascii_in_plain_theme() -> None:
    app, _ = build_app(theme=PLAIN)
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/Check draft.md")
        notes = _notes(app)
    assert "/Check" in notes
    assert notes.isascii(), f"non-ASCII note in PLAIN theme: {notes!r}"


# --- the config panel (wordmark design section 12) ----------------------------------


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No API key anywhere the panel looks: not in the environment and not in a
    ``.env`` beside the working directory, which is ``tmp_path`` for the test. The
    config dir's ``.env`` is already the isolated one."""
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _shown(panel: ConfigPanel) -> list[str]:
    """The panel's lines as drawn, top to bottom: a line inside a hidden container
    (the body of a collapsed panel) is not on screen either."""
    return [
        line.render().plain
        for line in panel.query(PanelLine)
        if all(node.display for node in line.ancestors_with_self)
    ]


def _panel_text(panel: ConfigPanel) -> str:
    return "\n".join(_shown(panel))


def _panel_row(panel: ConfigPanel, name: str) -> str:
    """The one row line that names ``name`` (``install_browser``, ``model``...)."""
    return next(line for line in _shown(panel) if line[2:].startswith(f"{name} "))


def _error_lines(app: ProofpathApp) -> str:
    """The error lines mounted loose in the log, under a panel."""
    return "\n".join(
        line.render().plain for line in app.query(KvLine) if isinstance(line.parent, RunLog)
    )


async def open_panel(pilot: Any) -> ConfigPanel:
    """``/config`` with nothing after it, and the panel it opens, focused."""
    await submit(pilot, "/config")
    await pilot.pause()
    panel = pilot.app.query(ConfigPanel).last()  # the newest, below any collapsed one
    await until(pilot, lambda: pilot.app.focused is panel, "the panel to take focus")
    return panel


async def written(pilot: Any, key: str, value: str) -> None:
    """Pump until the note for ``key = value`` is in the log: the write is done."""
    await until(
        pilot,
        lambda: f"{key} = {value}  ({config_path()})" in _notes(pilot.app),
        f"the note for {key} = {value}",
    )


@pytest.mark.usefixtures("no_keys")
async def test_config_opens_the_settings_panel_and_reads_the_file() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        assert isinstance(panel.parent, RunLog)
        assert len(app.query(ConfigPanel)) == 1
        assert not panel.collapsed and not panel.editing
        text = _panel_text(panel)
    lines = text.split("\n")
    assert lines[0].startswith(" config   ")
    assert str(config_path()) in lines[0]
    assert "(not written yet, showing defaults)" in lines[0]
    assert lines[1].startswith(" key      GROQ_API_KEY")
    assert "not set" in lines[1] and "GROQ_API_KEY=" in lines[1]
    for heading in ("permissions", "fetch", "judge", "contact"):
        assert f" {heading}" in lines


async def test_the_key_line_says_where_a_key_was_found_and_never_its_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_keys: None
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret-for-test")
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        from_env = _panel_text(panel)
        await pilot.press("escape")
        monkeypatch.delenv("GROQ_API_KEY")
        (tmp_path / ".env").write_text("GROQ_API_KEY=secret-for-test\n", encoding="utf-8")
        panel = await open_panel(pilot)
        from_file = _panel_text(panel)
        await pilot.press("escape")
        await pilot.pause()
        collapsed = "\n".join(_panel_text(p) for p in app.query(ConfigPanel))
    sep = PLAIN.glyphs.sep
    assert f"GROQ_API_KEY {sep} found in the environment" in from_env
    assert f"GROQ_API_KEY {sep} found in {tmp_path / '.env'}" in from_file
    for text in (from_env, from_file, collapsed):
        assert "secret-for-test" not in text


@pytest.mark.usefixtures("no_keys")
async def test_the_rows_come_in_the_configs_order_and_the_selection_stays_inside() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        assert [row.key for row in panel.rows] == [
            "permissions.install_browser",
            "permissions.network",
            "fetch.respect_robots",
            "judge.provider",
            "judge.model",
            "judge.base_url",
            "judge.api_key_env",
            "contact.email",
        ]
        assert [row.kind for row in panel.rows] == ["choice"] * 4 + ["text"] * 4
        assert panel.selected == 0
        assert _panel_row(panel, "install_browser").startswith("> install_browser")
        await pilot.press(*["down"] * 8)
        assert panel.selected == 7
        assert _panel_row(panel, "email").startswith("> email")
        assert _panel_row(panel, "install_browser").startswith("  install_browser")
        await pilot.press(*["up"] * 9)
        assert panel.selected == 0
    # The file was only read: moving through the rows writes nothing.
    assert not config_path().exists()


@pytest.mark.usefixtures("no_keys")
async def test_a_choice_row_writes_at_once_and_wraps() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("right")
        await written(pilot, "permissions.install_browser", "allow")
        assert load_config().permissions.install_browser == "allow"
        assert app._config.permissions.install_browser == "allow"
        assert panel.value("permissions.install_browser") == "allow"
        assert _panel_row(panel, "install_browser") == "> install_browser   ask   [allow]   deny"
        await pilot.press("right", "right")
        await written(pilot, "permissions.install_browser", "ask")
        assert load_config().permissions.install_browser == "ask"  # deny, then wrapped
        await pilot.press("left")
        await written(pilot, "permissions.install_browser", "deny")
        assert load_config().permissions.install_browser == "deny"
        await pilot.press("enter")  # on a choice row, the same as ``right``
        await until(
            pilot,
            lambda: load_config().permissions.install_browser == "ask",
            "enter to cycle the row",
        )
        assert not panel.editing
        # The note says what ``/config set`` would have said, once per write: five.
        assert _notes(app).count("permissions.install_browser = ") == 5
        assert str(config_path()) in _notes(app)


@pytest.mark.usefixtures("no_keys")
async def test_the_network_row_drives_the_banner() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        row = wordmark.rows(80, version=__version__, context=run_context(Config()), hint=HINT)
        assert " online " in _banner_lines(app)[row.version]
        await open_panel(pilot)
        await pilot.press("down", "right")  # network: allow -> deny
        await written(pilot, "permissions.network", "deny")
        assert app._config.permissions.network == "deny"
        await until(
            pilot,
            lambda: " offline " in _banner_lines(app)[row.version],
            "the banner to be redrawn offline",
        )
        await pilot.press("right")
        await written(pilot, "permissions.network", "ask")
        assert app._config.permissions.network == "ask"
        await until(
            pilot,
            lambda: run_context(app._config) in _banner_lines(app)[row.version],
            "the banner to follow the config again",
        )


@pytest.mark.usefixtures("no_keys")
async def test_a_bool_row_round_trips_through_the_file() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("down", "down", "right")
        await written(pilot, "fetch.respect_robots", "false")
        assert load_config().fetch.respect_robots is False
        assert panel.value("fetch.respect_robots") == "false"
        assert _panel_row(panel, "respect_robots") == "> respect_robots    true   [false]"
        await pilot.press("right")
        await written(pilot, "fetch.respect_robots", "true")
        assert load_config().fetch.respect_robots is True


@pytest.mark.usefixtures("no_keys")
async def test_a_provider_switch_writes_four_and_moves_the_key_line() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        assert panel.rows[3].values == ("gemini", "groq", "ollama")
        await pilot.press("down", "down", "down", "left")  # groq -> gemini
        await written(pilot, "judge.api_key_env", "GEMINI_API_KEY")
        gemini = provider_defaults("gemini")
        assert load_config().judge == gemini
        notes = _notes(app)
        for name in library.JUDGE_PRESET_FIELDS:
            assert f"judge.{name} = {getattr(gemini, name)}  ({config_path()})" in notes
        assert _panel_row(panel, "model").endswith(gemini.model)
        assert _panel_row(panel, "base_url").endswith(gemini.base_url)
        assert _panel_row(panel, "api_key_env").endswith("GEMINI_API_KEY")
        assert " key      GEMINI_API_KEY" in _panel_text(panel)
        assert "GROQ_API_KEY" not in _panel_text(panel)


@pytest.mark.usefixtures("no_keys")
async def test_a_provider_outside_the_known_choices_is_drawn_honestly() -> None:
    """Hand-edited TOML can hold a ``judge.provider`` ``load_config`` never rejects
    (it is typed as a plain ``str``, unlike the permission and bool fields): the row
    must say so rather than silently show none of ``gemini``/``groq``/``ollama`` as
    current."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[judge]\nprovider = "openrouter"\n', encoding="utf-8")
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        row = _panel_row(panel, "provider")
        assert row == "  provider          gemini   groq   ollama   [openrouter]"
        await pilot.press("down", "down", "down")  # install_browser -> ... -> provider
        await pilot.press("right")  # cycles from the first choice, not from "nowhere"
        await written(pilot, "judge.provider", "gemini")


@pytest.mark.usefixtures("no_keys")
async def test_a_provider_without_a_key_variable_says_none_is_needed() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("down", "down", "down", "right")  # groq -> ollama
        await written(pilot, "judge.provider", "ollama")
        text = _panel_text(panel)
    assert " key      not needed" in text
    assert "not set" not in text


@pytest.mark.usefixtures("no_keys")
async def test_a_text_row_is_edited_in_place() -> None:
    app, _ = build_app()
    default = JudgeConfig().model
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("down", "down", "down", "down", "enter")
        await pilot.pause()
        assert panel.editing
        edit = panel.query_one(Input)
        assert edit.value == default
        await until(pilot, lambda: app.focused is edit, "the edit to take focus")
        await pilot.press("-", "x", "enter")
        await written(pilot, "judge.model", f"{default}-x")
        assert load_config().judge.model == f"{default}-x"
        assert not panel.editing
        assert not panel.query(Input)
        assert _panel_row(panel, "model") == f"> model             {default}-x"
        await until(pilot, lambda: app.focused is panel, "focus to come back to the panel")
        # ``Esc`` cancels: nothing written, the row shows what it showed.
        await pilot.press("enter")
        await pilot.pause()
        await until(pilot, lambda: app.focused is panel.query_one(Input), "the second edit")
        await pilot.press("z", "z", "z", "escape")
        await pilot.pause()
        assert not panel.editing
        assert load_config().judge.model == f"{default}-x"
        assert _panel_row(panel, "model") == f"> model             {default}-x"
        assert app.focused is panel
        assert app.query_one(Prompt).value == ""  # nothing leaked into the bar


@pytest.mark.usefixtures("no_keys")
async def test_an_empty_text_row_shows_unset_and_its_note() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        assert _panel_row(panel, "email") == (
            "  email             (unset) , optional, for the Crossref / OpenAlex polite pools"
        )
        await pilot.press(*["down"] * 7, "enter")
        await pilot.pause()
        await until(pilot, lambda: app.focused is panel.query_one(Input), "the edit")
        assert panel.query_one(Input).value == ""
        await pilot.press("a", "@", "b", ".", "c", "enter")
        await written(pilot, "contact.email", "a@b.c")
        assert load_config().contact.email == "a@b.c"
        assert _panel_row(panel, "email").startswith("> email             a@b.c , optional")


@pytest.mark.usefixtures("no_keys")
async def test_backspace_resets_a_row_to_its_default() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("right", "right")
        await written(pilot, "permissions.install_browser", "deny")
        await pilot.press("backspace")
        await written(pilot, "permissions.install_browser", "ask")
        assert load_config().permissions.install_browser == "ask"
        assert panel.value("permissions.install_browser") == "ask"
        await pilot.press("down", "down", "down", "left")
        await written(pilot, "judge.provider", "gemini")
        before = _notes(app).count("judge.")
        await pilot.press("backspace")
        await written(pilot, "judge.provider", "groq")
        assert load_config().judge == JudgeConfig()
        assert _notes(app).count("judge.") == before + 4


async def test_a_failed_write_leaves_the_row_and_the_file_alone(
    monkeypatch: pytest.MonkeyPatch, no_keys: None
) -> None:
    def boom(key: str, value: str, path: Path | None = None) -> tuple[tuple[str, str], ...]:
        raise ConfigError("boom")

    monkeypatch.setattr("proofpath.tui.widgets.config_panel.config_set", boom)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("right")
        await until(pilot, lambda: "boom" in _error_lines(app), "the error line")
        assert _error_lines(app) == "  error: boom"
        assert not config_path().exists()
        assert panel.value("permissions.install_browser") == "ask"
        assert _panel_row(panel, "install_browser") == "> install_browser   [ask]   allow   deny"
        assert "permissions.install_browser = " not in _notes(app)
        assert app.focused is panel and not panel.collapsed


async def test_a_failed_reread_still_reports_what_was_written(
    monkeypatch: pytest.MonkeyPatch, no_keys: None
) -> None:
    """``config_set`` can succeed while the re-read that follows it fails -- the row
    must not be reported ``Failed`` over a file that did in fact change."""
    real_config_view = library.config_view
    calls = {"n": 0}

    def flaky_config_view() -> library.ConfigView:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConfigError("reread-boom")
        return real_config_view()

    monkeypatch.setattr("proofpath.tui.widgets.config_panel.config_view", flaky_config_view)
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await open_panel(pilot)
        await pilot.press("right")
        await until(
            pilot, lambda: "could not be read back" in _error_lines(app), "the re-read error line"
        )
        # The write landed on disk -- config_set already returned before the reread
        # was even attempted -- so it is reported like any other successful write.
        assert "permissions.install_browser = allow" in _notes(app)
        assert load_config().permissions.install_browser == "allow"
        assert "reread-boom" in _error_lines(app)


@pytest.mark.usefixtures("no_keys")
async def test_escape_collapses_the_panel_to_one_line_and_focuses_the_bar() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("down", "right", "right")  # network: allow -> deny -> ask
        await written(pilot, "permissions.network", "ask")
        await pilot.press("escape")
        await pilot.pause()
        assert panel.collapsed
        assert _panel_text(panel) == (
            "  config  install_browser=ask  network=ask  respect_robots=true  judge=groq"
        )
        await until(pilot, lambda: app.focused is app.query_one(Prompt), "the bar to take focus")
        # A collapsed panel is a record: the keys do nothing to it any more.
        await pilot.press("escape")
        assert app.query_one(Prompt).value == ""


@pytest.mark.usefixtures("no_keys")
async def test_typing_on_the_panel_goes_to_the_bar_and_collapses_it() -> None:
    """Wordmark design section 11: a printable key is typing wherever focus is; the
    panel sees the blur and folds."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("x")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert app.focused is prompt
        assert prompt.value == "x"
        assert panel.collapsed
        assert _panel_text(panel).startswith("  config  ")


@pytest.mark.usefixtures("no_keys")
async def test_the_terminal_losing_focus_does_not_fold_an_open_panel() -> None:
    """An ``AppBlur`` (the terminal window itself losing focus) delivers the same
    ``Blur`` to whatever is focused as any other blur would, but nothing inside the
    app moved focus -- the panel is still what would take input back. Only a real
    blur to another widget (typing, a click) should fold it."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        app.post_message(tevents.AppBlur())
        await pilot.pause()
        await pilot.pause()
        assert not panel.collapsed
        # Textual clears ``screen.focused`` for the duration of the blur and restores
        # it on the matching ``AppFocus``, which is what "the panel is still what
        # would take input back" comes down to.
        app.post_message(tevents.AppFocus())
        await pilot.pause()
        await pilot.pause()
        assert app.focused is panel


@pytest.mark.usefixtures("no_keys")
async def test_escape_on_a_collapsed_but_focused_panel_runs_the_apps_own_escape() -> None:
    """A collapsed panel has no ``escape`` of its own (``action_close`` stands down
    once folded); the app's priority ``escape`` must not keep standing aside for it
    either, or ``Esc`` would do nothing until some other event moved focus away."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        await pilot.press("escape")  # folds it, focuses the bar
        await pilot.pause()
        assert panel.collapsed
        await submit(pilot, "/check")  # an argument-taking verb: enters awaiting mode
        assert app.awaiting == "check"
        panel.focus()  # reach a collapsed-but-focused panel however the code allows
        await pilot.pause()
        assert app.focused is panel
        await pilot.press("escape")
        await pilot.pause()
        assert app.awaiting is None


@pytest.mark.usefixtures("no_keys")
async def test_config_again_opens_a_fresh_panel_and_show_still_prints_the_toml() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        first = await open_panel(pilot)
        second = await open_panel(pilot)
        assert first is not second
        assert len(app.query(ConfigPanel)) == 2
        assert first.collapsed and not second.collapsed
        await submit(pilot, "/config show")
        text = await lines_of(app, pilot)
        assert "[permissions]" in text
        assert len(app.query(ConfigPanel)) == 2


@pytest.mark.usefixtures("no_keys")
async def test_the_plain_legend_spells_the_arrows_in_ascii() -> None:
    app, _ = build_app(theme=PLAIN)
    async with app.run_test(size=SIZE) as pilot:
        panel = await open_panel(pilot)
        text = _panel_text(panel)
    lines = text.split("\n")
    assert lines[-1] == (
        " up/down row   left/right change   enter edit text   backspace default   esc close"
    )
    # Everything but the paths is the theme's ASCII (the paths are the machine's).
    for line in lines[2:]:
        assert line.isascii(), f"non-ASCII panel line in PLAIN theme: {line!r}"


@pytest.mark.usefixtures("no_keys")
async def test_an_unreadable_config_opens_no_panel(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        (tmp_path / "conf").mkdir(exist_ok=True)
        (tmp_path / "conf" / "config.toml").write_text("this is not toml [[", encoding="utf-8")
        await submit(pilot, "/config")
        await pilot.pause()
        assert not app.query(ConfigPanel)
        assert _error_lines(app).startswith("  error: ")
        assert app.is_running
        assert app.focused is app.query_one(Prompt)


@pytest.mark.usefixtures("no_keys")
async def test_ctrl_l_clears_a_collapsed_panel_and_keeps_an_open_one() -> None:
    """A collapsed panel is a record, like a finished block; an open one is in use."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await open_panel(pilot)
        await pilot.press("escape")
        await pilot.pause()
        second = await open_panel(pilot)
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert list(app.query(ConfigPanel)) == [second]
        assert not second.collapsed
