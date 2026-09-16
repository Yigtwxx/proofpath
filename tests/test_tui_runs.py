"""The TUI's run scheduler (spec section 13.1): several runs per session.

Both halves of the pipeline are faked with blocking stand-ins, so the test — not
the clock — decides when a run leaves a stage: every half parks until the test
releases it, which is what makes "three at a time" and "one verify slot, in arrival
order" assertable rather than hopeful.

The fakes also answer the question the real halves would hide: which thread ran
them. A ``Cache`` connection may not cross threads, so both halves of one run must
happen on the same one, and ``test_both_halves_of_a_run_share_one_thread`` is the
test that says so.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import pytest

from proofpath.events import Cancelled, Event, StageStart
from proofpath.tui.runs import Run, Scheduler

TIMEOUT = 5.0  # generous: every wait below is on an event a test has already set


# --- the fakes --------------------------------------------------------------------


@dataclass
class FakePrepared:
    """What the fake I/O half returns; the fake model half reads its target back."""

    target: str


@dataclass
class FakeReport:
    target: str
    partial: bool = False


class StubEngine:
    """Stands in for ``verify.Engine``: the scheduler only builds and closes it."""

    def __init__(self, number: int) -> None:
        self.number = number
        self.closed = False
        self.built_on = threading.current_thread()

    def close(self) -> None:
        self.closed = True


class Factory:
    """The engine factory, recording every engine it was asked for."""

    def __init__(self) -> None:
        self.engines: list[StubEngine] = []
        self.runs: list[Run] = []

    def __call__(self, run: Run) -> StubEngine:
        engine = StubEngine(len(self.engines) + 1)
        self.engines.append(engine)
        self.runs.append(run)
        return engine


class Half:
    """One blocking half of the pipeline, driven per target by the test.

    ``run`` parks until the test releases that target, honouring ``cancel`` the way
    the real halves do: between units of work rather than instantly.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.entered: dict[str, threading.Event] = defaultdict(threading.Event)
        self.release: dict[str, threading.Event] = defaultdict(threading.Event)
        self.fail: dict[str, Exception] = {}
        self.emit: dict[str, Event] = {}
        self.partial: set[str] = set()
        self.order: list[str] = []
        self.threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def work(self, target: str, cancel: threading.Event | None, on_event: Callable | None) -> None:
        with self._lock:
            self.order.append(target)
            self.threads[target] = threading.current_thread()
        self.entered[target].set()
        if on_event is not None and target in self.emit:
            on_event(self.emit[target])
        while not self.release[target].wait(0.005):
            if cancel is not None and cancel.is_set():
                report = FakeReport(target, partial=True) if target in self.partial else None
                raise Cancelled(f"{self.name} cancelled", report=report)
        failure = self.fail.get(target)
        if failure is not None:
            raise failure

    def let_go(self, *targets: str) -> None:
        for target in targets:
            self.release[target].set()

    def release_everything(self) -> None:
        for event in list(self.release.values()):
            event.set()


@dataclass
class Harness:
    """A scheduler wired to the two fake halves, plus everything it reported."""

    scheduler: Scheduler
    factory: Factory
    io: Half
    model: Half
    states: list[tuple[int, str]] = field(default_factory=list)
    events: list[tuple[int, Event, threading.Thread]] = field(default_factory=list)
    callers: list[threading.Thread] = field(default_factory=list)  # where on_state ran

    def submit(self, target: str) -> Run:
        return self.scheduler.submit(target, command=f"/check {target}")

    def states_of(self, run: Run) -> list[str]:
        return [state for run_id, state in self.states if run_id == run.id]

    def running(self) -> list[str]:
        return [run.target for run in self.scheduler.runs if run.state == "running"]


def make_harness(
    io_limit: int = 3, engine_factory: Callable[[Run], object] | None = None
) -> Harness:
    factory, io, model = Factory(), Half("prepare"), Half("decide_all")
    states: list[tuple[int, str]] = []
    events: list[tuple[int, Event, threading.Thread]] = []
    callers: list[threading.Thread] = []

    def note_state(run: Run) -> None:
        states.append((run.id, run.state))
        callers.append(threading.current_thread())

    def prepare(target, engine, *, on_event=None, cancel=None):
        io.work(target, cancel, on_event)
        return FakePrepared(target)

    def decide_all(prepared, engine, *, on_event=None, cancel=None):
        model.work(prepared.target, cancel, on_event)
        return FakeReport(prepared.target)

    scheduler = Scheduler(
        engine_factory or factory,
        io_limit=io_limit,
        on_event=lambda run, event: events.append((run.id, event, threading.current_thread())),
        on_state=note_state,
        prepare=prepare,
        decide_all=decide_all,
    )
    return Harness(scheduler, factory, io, model, states, events, callers)


@pytest.fixture
async def build() -> AsyncIterator[Callable[..., Harness]]:
    """Hands out harnesses and closes every one of them, blocked halves and all."""
    made: list[Harness] = []

    def make(io_limit: int = 3, engine_factory: Callable[[Run], object] | None = None) -> Harness:
        harness = make_harness(io_limit, engine_factory)
        made.append(harness)
        return harness

    yield make
    for harness in made:
        harness.io.release_everything()
        harness.model.release_everything()
        await harness.scheduler.close()


async def until(predicate: Callable[[], bool], what: str = "", timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what or predicate}")


async def settle() -> None:
    """Let the loop run a while, so that anything wrong has time to happen."""
    await asyncio.sleep(0.05)


# --- the I/O stage ----------------------------------------------------------------


async def test_at_most_three_runs_do_io_at_once(build) -> None:
    harness = build(io_limit=3)
    runs = [harness.submit(f"doc{n}") for n in range(1, 6)]
    await until(lambda: len(harness.running()) == 3, "three runs in the I/O stage")
    # A run is "running" the moment the loop starts its thread; the engine is built on
    # that thread afterwards. Wait for the engines too, or ``settle`` is being asked to
    # guess how long three threads take to start.
    await until(lambda: len(harness.factory.engines) == 3, "three engines to be built")
    await settle()
    assert [run.state for run in runs] == ["running"] * 3 + ["queued"] * 2
    assert len(harness.factory.engines) == 3, "a queued run must not build an engine"


async def test_finishing_one_io_half_starts_the_next_queued_run(build) -> None:
    harness = build(io_limit=3)
    runs = [harness.submit(f"doc{n}") for n in range(1, 6)]
    await until(lambda: len(harness.running()) == 3, "three runs in the I/O stage")
    harness.io.let_go("doc1")
    await until(lambda: runs[3].state == "running", "the fourth run to start")
    await until(lambda: "doc4" in harness.io.order, "the fourth run to reach the I/O half")
    assert runs[4].state == "queued"
    assert set(harness.io.order) == {"doc1", "doc2", "doc3", "doc4"}, "the fifth waits"
    # The claim is that the freed slot is what let the fourth in, not the order three
    # threads happened to be scheduled in: only doc1 is ordered against doc4.
    assert harness.io.order.index("doc1") < harness.io.order.index("doc4")


async def test_a_run_gets_its_own_engine(build) -> None:
    harness = build()
    harness.submit("doc1")
    harness.submit("doc2")
    await until(lambda: len(harness.factory.engines) == 2, "two engines")
    assert harness.factory.engines[0] is not harness.factory.engines[1]


# --- the verify slot --------------------------------------------------------------


async def test_only_one_run_verifies_at_a_time(build) -> None:
    harness = build()
    runs = [harness.submit(f"doc{n}") for n in range(1, 4)]
    harness.io.let_go("doc1", "doc2", "doc3")
    await until(
        lambda: (
            all(r.state in {"waiting for verify", "verifying"} for r in runs)
            and len(harness.model.order) == 1
        ),
        "all three runs past the I/O half, one of them inside the model",
    )
    await settle()
    states = [run.state for run in runs]
    assert states.count("verifying") == 1
    assert states.count("waiting for verify") == 2
    assert len(harness.model.order) == 1


async def test_the_verify_slot_is_granted_in_arrival_order(build) -> None:
    """Arrival order, not id order: the run that queued first verifies first."""
    harness = build()
    runs = {n: harness.submit(f"doc{n}") for n in (1, 2, 3)}
    for number in (3, 2, 1):  # reach the verify queue in reverse submission order
        harness.io.let_go(f"doc{number}")
        await until(
            lambda n=number: runs[n].state in {"waiting for verify", "verifying"},
            f"doc{number} to reach the verify queue",
        )
    # "verifying" is set on the loop before the thread is let into the model half.
    await until(lambda: harness.model.order == ["doc3"], "doc3 to enter the model half")
    harness.model.let_go("doc3")
    await until(lambda: harness.model.order == ["doc3", "doc2"], "doc2 to verify next")
    harness.model.let_go("doc2")
    await until(lambda: harness.model.order == ["doc3", "doc2", "doc1"], "doc1 to verify last")


async def test_a_clean_run_walks_the_five_states_in_order(build) -> None:
    harness = build()
    run = harness.submit("doc1")
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
    assert harness.states_of(run) == [
        "queued",
        "running",
        "waiting for verify",
        "verifying",
        "done",
    ]
    assert run.report == FakeReport("doc1")
    assert run.error is None
    assert run.started is not None and run.finished is not None


async def test_both_halves_of_a_run_share_one_thread(build) -> None:
    """A ``Cache`` connection is thread-bound: one dedicated thread owns the run."""
    harness = build()
    runs = [harness.submit(f"doc{n}") for n in (1, 2)]
    harness.io.let_go("doc1", "doc2")
    harness.model.let_go("doc1", "doc2")
    for run in runs:
        await harness.scheduler.wait(run.id)
    for target in ("doc1", "doc2"):
        thread = harness.io.threads[target]
        assert harness.model.threads[target] is thread
        assert thread is not threading.main_thread()
    assert harness.io.threads["doc1"] is not harness.io.threads["doc2"]
    assert harness.factory.engines[0].built_on is harness.io.threads["doc1"]
    assert all(engine.closed for engine in harness.factory.engines)


# --- cancelling -------------------------------------------------------------------


async def test_cancelling_a_queued_run_is_immediate_and_builds_no_engine(build) -> None:
    harness = build(io_limit=1)
    first, second = harness.submit("doc1"), harness.submit("doc2")
    await until(lambda: first.state == "running", "the first run to start")
    assert harness.scheduler.cancel(second.id) is True
    assert second.state == "cancelled"
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(first.id)
    await settle()
    assert harness.io.order == ["doc1"], "a cancelled queued run never runs a half"
    assert len(harness.factory.engines) == 1
    assert harness.states_of(second) == ["queued", "cancelled"]


async def test_cancelling_a_running_run_stops_it_in_the_io_half(build) -> None:
    harness = build()
    run = harness.submit("doc1")
    await until(lambda: run.state == "running", "the run to start")
    assert harness.scheduler.cancel(run.id) is True
    await harness.scheduler.wait(run.id)
    assert run.state == "cancelled"
    assert harness.model.order == [], "the model must not run after a cancelled I/O half"
    assert harness.factory.engines[0].closed


async def test_cancelling_a_run_waiting_for_verify_skips_the_model(build) -> None:
    harness = build()
    first, second = harness.submit("doc1"), harness.submit("doc2")
    harness.io.let_go("doc1", "doc2")
    await until(lambda: second.state == "waiting for verify", "the second run to queue")
    assert harness.scheduler.cancel(second.id) is True
    assert second.state == "cancelled"
    harness.model.let_go("doc1")
    await harness.scheduler.wait(first.id)
    await harness.scheduler.wait(second.id)
    assert first.state == "done"
    assert harness.model.order == ["doc1"]
    # The cancelled run's task ends before its thread does; the engine is closed on the
    # thread, so this waits for the thread rather than for the task that woke it.
    await until(
        lambda: all(engine.closed for engine in harness.factory.engines),
        "both engines to be closed on their own threads",
    )


async def test_a_run_cancelled_while_waiting_for_verify_leaves_the_queue_at_once(build) -> None:
    """It must not hold an engine open until a slot it will never use comes free."""
    harness = build()
    first, second = harness.submit("doc1"), harness.submit("doc2")
    harness.io.let_go("doc1", "doc2")
    await until(
        lambda: (
            first.state == "verifying"
            and second.state == "waiting for verify"
            and harness.model.order == ["doc1"]
        ),
        "doc1 inside the model half and doc2 queued behind it",
    )
    thread = harness.io.threads["doc2"]
    engine = next(e for e in harness.factory.engines if e.built_on is thread)
    assert harness.scheduler.cancel(second.id) is True
    await until(lambda: not thread.is_alive(), "the cancelled run's thread to finish")
    # All of that while the run holding the verify slot is still inside the model.
    assert engine.closed
    assert first.state == "verifying"
    assert harness.model.order == ["doc1"]
    assert await harness.scheduler.wait(second.id) is second
    harness.model.let_go("doc1")
    await harness.scheduler.wait(first.id)
    assert first.state == "done"


async def test_cancelling_the_task_of_a_running_run_still_settles_it(build) -> None:
    """A task cancelled from outside must not leave a run block frozen mid-stage."""
    harness = build()
    run = harness.submit("doc1")
    await until(lambda: run.state == "running", "the run to start")
    thread = harness.io.threads["doc1"]
    task = next(t for t in asyncio.all_tasks() if t.get_name() == f"proofpath-run-{run.id}")
    task.cancel()
    await until(lambda: run.state == "cancelled", "the cancelled task to settle its run")
    await until(lambda: not thread.is_alive(), "the worker thread to finish")
    assert harness.states_of(run) == ["queued", "running", "cancelled"]
    assert run.finished is not None
    assert harness.factory.engines[0].closed
    assert await harness.scheduler.wait(run.id) is run


async def test_a_cancelled_model_half_keeps_its_partial_report(build) -> None:
    harness = build()
    harness.model.partial.add("doc1")
    run = harness.submit("doc1")
    harness.io.let_go("doc1")
    await until(lambda: run.state == "verifying", "the run to reach the model")
    harness.scheduler.cancel(run.id)
    await harness.scheduler.wait(run.id)
    assert run.state == "cancelled"
    assert run.report == FakeReport("doc1", partial=True)


async def test_cancel_answers_false_for_an_unknown_or_finished_run(build) -> None:
    harness = build()
    run = harness.submit("doc1")
    assert harness.scheduler.cancel(99) is False
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
    assert harness.scheduler.cancel(run.id) is False


# --- failing ----------------------------------------------------------------------


async def test_a_failing_io_half_fails_only_that_run(build) -> None:
    harness = build()
    harness.io.fail["doc2"] = RuntimeError("boom")
    runs = [harness.submit(f"doc{n}") for n in (1, 2, 3)]
    harness.io.let_go("doc1", "doc2", "doc3")
    harness.model.let_go("doc1", "doc3")
    for run in runs:
        await harness.scheduler.wait(run.id)
    assert [run.state for run in runs] == ["done", "failed", "done"]
    assert runs[1].error == "RuntimeError: boom"
    assert runs[1].report is None
    assert harness.states_of(runs[1]) == ["queued", "running", "failed"]
    assert harness.model.order == ["doc1", "doc3"]
    assert all(engine.closed for engine in harness.factory.engines)


async def test_a_failing_model_half_fails_the_run_with_its_message(build) -> None:
    harness = build()
    harness.model.fail["doc1"] = ValueError("no scorer")
    run = harness.submit("doc1")
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
    assert run.state == "failed"
    assert run.error == "ValueError: no scorer"


async def test_a_failing_engine_factory_fails_the_run(build) -> None:
    def explode(run: Run) -> StubEngine:
        raise OSError("no cache dir")

    harness = build(engine_factory=explode)
    run = harness.submit("doc1")
    await harness.scheduler.wait(run.id)
    assert run.state == "failed"
    assert run.error == "OSError: no cache dir"


# --- events and shutdown ----------------------------------------------------------


async def test_events_reach_the_listener_on_the_loop_thread(build) -> None:
    harness = build()
    harness.io.emit["doc1"] = StageStart("Parsing", "pymupdf")
    run = harness.submit("doc1")
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
    assert [(run_id, event) for run_id, event, _ in harness.events] == [
        (run.id, StageStart("Parsing", "pymupdf"))
    ]
    assert harness.events[0][2] is threading.main_thread()
    assert harness.io.threads["doc1"] is not threading.main_thread()


async def test_state_changes_are_reported_on_the_loop_thread(build) -> None:
    harness = build()
    run = harness.submit("doc1")
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
    assert len(harness.callers) == 5
    assert all(thread is threading.main_thread() for thread in harness.callers)


async def test_close_cancels_every_run_and_closes_every_engine(build) -> None:
    harness = build()
    runs = [harness.submit(f"doc{n}") for n in (1, 2, 3)]
    await until(lambda: len(harness.factory.engines) == 3, "three engines")
    await harness.scheduler.close()
    assert [run.state for run in runs] == ["cancelled"] * 3
    assert all(engine.closed for engine in harness.factory.engines)


async def test_close_is_idempotent_and_refuses_later_submissions(build) -> None:
    harness = build()
    run = harness.submit("doc1")
    await until(lambda: run.state == "running", "the run to start")
    await harness.scheduler.close()
    await harness.scheduler.close()
    assert run.state == "cancelled"
    assert harness.scheduler.unstopped == (), "every worker thread stopped in time"
    with pytest.raises(RuntimeError, match="closed"):
        harness.submit("doc2")


async def test_close_leaves_a_finished_run_alone(build) -> None:
    harness = build()
    run = harness.submit("doc1")
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
    await harness.scheduler.close()
    assert run.state == "done"
    assert harness.states_of(run)[-1] == "done"


async def test_runs_are_listed_in_submission_order_and_wait_rejects_unknown_ids(build) -> None:
    harness = build()
    first, second = harness.submit("doc1"), harness.submit("doc2")
    assert harness.scheduler.runs == (first, second)
    assert [run.id for run in harness.scheduler.runs] == [1, 2]
    assert harness.scheduler.get(2) is second
    assert harness.scheduler.get(7) is None
    with pytest.raises(KeyError):
        await harness.scheduler.wait(7)


async def test_the_engine_is_built_for_the_run_that_will_use_it(build) -> None:
    """Task 8.4: the engine's consent gate has to know whose block to ask (spec 13.1)."""
    harness = build()
    run = harness.submit("doc1")
    await until(lambda: harness.factory.runs == [run], "the factory to see the run")
    harness.io.let_go("doc1")
    harness.model.let_go("doc1")
    await harness.scheduler.wait(run.id)
