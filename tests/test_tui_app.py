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
from typing import Any

import pytest
from textual.widgets import Button

from proofpath import __version__, browser, ui
from proofpath import commands as library
from proofpath import verify as verify_mod
from proofpath.browser import ConsentGate
from proofpath.config import Config, Permissions
from proofpath.document import Document, Locator, Reference
from proofpath.events import Emitted, Event, Note, Progress, Prompted, StageEnd, StageStart
from proofpath.models import Label, Passage, Verdict
from proofpath.report import Coverage, Finding, Kind, Report
from proofpath.resolve import Candidate, ResolveResult
from proofpath.resolve import State as ResolveState
from proofpath.tui import banner, commands
from proofpath.tui.app import (
    ANSWER_LABELS,
    AWAITING_VERBS,
    FLASH_SECONDS,
    HINT,
    Banner,
    CommandBlock,
    CoverageFooter,
    FindingLine,
    KvLine,
    NoteLine,
    PermissionPrompt,
    Prompt,
    ProofpathApp,
    RunBlock,
    RunHeader,
    RunLog,
    StageLine,
    accent_for,
    run_context,
)
from proofpath.tui.runs import Run, State
from proofpath.verify import Engine

SIZE = (80, 24)
HOST = "sciencedirect.com"


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/config`` and ``/cache`` read real files; never the ones this machine uses."""
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))


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


async def until(pilot: Any, ready: Callable[[], bool], what: str, timeout: float = 3.0) -> None:
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
    expected = banner.render(80, version=__version__, context=run_context(Config()), hint=HINT)
    assert drawn.lines == expected.lines
    assert drawn.stamp_span == expected.stamp_span


async def test_banner_is_pure_ascii() -> None:
    """Assumption 3.3: the pet renders identically in Windows Terminal."""
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
        assert prompt.styles.background.a == pytest.approx(1.0)


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
        assert banner_widget.region.height == 4
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


async def test_summarize_is_the_one_verb_still_owed() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/summarize")
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert "v0.3" in text


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


@pytest.mark.parametrize("width", [80, 60, 46])
async def test_the_banner_container_fits_every_line_it_drew(width: int) -> None:
    """Line 3 wraps below about 60 columns; the hint line must still be on screen."""
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
        assert app.query_one(Banner).drawn.lines[1].endswith("[PROOF]")
        await pilot.resize_terminal(60, 24)
        await pilot.pause()
        drawn = app.query_one(Banner).drawn
        assert drawn is not None
        assert (
            drawn.lines
            == banner.render(
                60, version=__version__, context=run_context(Config()), hint=HINT
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
        assert prompt.styles.background.a == pytest.approx(1.0)
    assert schedulers[0].submitted == []


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

        await pilot.click("#allow-once")
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
        text = "\n".join(line.render().plain for line in app.query(NoteLine))
    assert "nothing to allow" in text


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
    assert answer.result(timeout=3) == "no"
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
        await pilot.click("#allow-no")
        assert allowed.result(timeout=3) is False
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


# --- task 8.4: the mouse, and its keyboard equivalent (spec 13.1) --------------------


async def test_a_click_on_a_finding_opens_the_whole_passage() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        schedulers[0].push(schedulers[0].runs[0], Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        assert not line.expanded
        await pilot.click(FindingLine, offset=(2, 0))
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
        await pilot.click(FindingLine, offset=(column, row))
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
        await pilot.click(RunHeader, offset=(2, 0))
        await pilot.pause()
        assert block.collapsed
        assert not block.query_one(".stages").display
        assert not block.query_one(".findings").display
        # Rule 6: the header, its state and its coverage never fold away.
        assert block.header.display
        assert "coverage" in block.header.render().plain
        await pilot.click(RunHeader, offset=(2, 0))
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
        await pilot.click(StageLine, offset=(2, 0))
        await pilot.pause()
        text = stage.render().plain
    assert "24 pages, 42 refs" not in text
    assert "Parse" in text and "1.2s" in text


# --- task 8.4: the eyes (spec 13.1) --------------------------------------------------


async def test_the_eyes_watch_the_path_while_a_run_works_and_settle_after() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        ferret = app.query_one(Banner)
        assert ferret.timer is not None
        assert ferret.eyes == banner.EYES["idle"]

        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.move(run, "running")
        await pilot.pause()
        assert ferret.eyes == banner.EYES["busy"]
        assert app.query_one(Banner).drawn.lines[1].startswith("  " + banner.EYES["busy"])

        scheduler.move(run, "done", report=a_report())
        await pilot.pause()
        assert ferret.eyes == banner.EYES["clean"]

        # Two seconds later the expression is over; the test moves the clock instead.
        later = time.monotonic() + FLASH_SECONDS + 1
        ferret.clock = lambda: later
        ferret.tick()
        assert ferret.eyes == banner.EYES["idle"]


async def test_a_run_with_findings_gets_the_other_face() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done", report=a_report(findings=(a_finding(),)))
        await pilot.pause()
        assert app.query_one(Banner).eyes == banner.EYES["findings"]


async def test_a_failed_run_never_gets_the_clean_face() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check draft.md")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        run.error = "OSError: no such file"
        scheduler.move(run, "failed")
        await pilot.pause()
        assert app.query_one(Banner).eyes == banner.EYES["findings"]


async def test_the_ferret_blinks_on_its_own_schedule() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        ferret = app.query_one(Banner)
        await pilot.pause()
        start = time.monotonic()
        ferret.clock = lambda: start + 11  # past the 6-10s window, whatever it drew
        ferret.tick()
        assert ferret.eyes == banner.EYES["blink"]
        ferret.clock = lambda: start + 11.2  # 150 ms later it is over
        ferret.tick()
        assert ferret.eyes == banner.EYES["idle"]


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [({"no_color": True}, "--no-color"), ({"quiet": True}, "-q")],
)
async def test_no_animation_without_colour_or_under_quiet(kwargs: Any, why: str) -> None:
    app, schedulers = build_app(out=ui.build(force_terminal=True, **kwargs))
    async with app.run_test(size=SIZE) as pilot:
        ferret = app.query_one(Banner)
        assert ferret.timer is None, why
        await submit(pilot, "/check draft.md")
        schedulers[0].move(schedulers[0].runs[0], "running")
        await pilot.pause()
        assert ferret.eyes == banner.EYES["idle"], why


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
    """A real gate wired to the app's prompt, the way ``_default_engine`` wires one."""
    return ConsentGate(
        "ask",
        interactive=True,
        prompt=app._prompt_for(owner),
        installer=lambda log: pytest.fail("nothing may be installed"),
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
        await pilot.click("#allow-never")
        assert allowed.result(timeout=3) is False
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
    """Line 3 says online/offline; a session that changed it must not still say online."""
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        assert " online " in app.query_one(Banner).drawn.lines[2]
        await submit(pilot, "/config set permissions.network deny")
        await lines_of(app, pilot)
        await until(
            pilot,
            lambda: " offline " in _banner_lines(app)[2],
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
            late.append(ask_from_a_worker(app, run.id).result(timeout=3))
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
        await pilot.click("#allow-no")
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


async def test_the_blink_falls_inside_the_six_to_ten_second_window() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        ferret = app.query_one(Banner)
        await pilot.pause()
        start = time.monotonic()
        ferret.clock = lambda: start + 5.5  # never before six seconds
        ferret.tick()
        assert ferret.eyes == banner.EYES["idle"]
        ferret.clock = lambda: start + 10.5  # and never later than ten
        ferret.tick()
        assert ferret.eyes == banner.EYES["blink"]


async def test_an_unreadable_config_is_reported_not_raised(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        (tmp_path / "conf").mkdir(exist_ok=True)
        (tmp_path / "conf" / "config.toml").write_text("this is not toml [[", encoding="utf-8")
        await submit(pilot, "/config show")
        text = await lines_of(app, pilot)
        assert app.is_running
    assert text.startswith("  error: ")
