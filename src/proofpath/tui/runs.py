"""Several runs in one session, scheduled by stage (spec section 13.1).

The TUI lets you start a second check while the first is still going, which the
pipeline itself knows nothing about: ``verify.prepare`` and ``verify.decide_all``
are two blocking halves over one :class:`~proofpath.verify.Engine`. This module is
the only place that knows how many of each may run at once.

The shape of the schedule follows what the two halves cost. ``prepare`` is I/O —
resolving references and reading pages — so up to ``io_limit`` runs do it
concurrently; ``decide_all`` loads and drives the NLI model, so it gets a single
slot and runs queue for it in arrival order. A run therefore walks
``queued -> running -> waiting for verify -> verifying -> done``, and the states in
between are what the run block on screen prints.

Three constraints shape the threading, and none of them is negotiable:

* :class:`~proofpath.cache.Cache` opens SQLite with ``check_same_thread=True``, and
  both halves touch it. Each run therefore gets **one dedicated thread** that builds
  its engine, runs both halves and closes it again — never ``asyncio.to_thread``,
  whose pool hands out an arbitrary thread per call and would give the second half a
  connection it is not allowed to use.
* The event loop owns the scheduling. The verify slot is acquired on the loop and
  handed to the waiting thread through a ``threading.Event``, so the ordering rule
  lives in one place instead of in every worker.
* A run never raises out of its task. Cancellation ends in ``cancelled``, anything
  else in ``failed`` with the exception on the run, because a TUI that loses a
  traceback into a dead task is a TUI that silently stops verifying.

Events cross back the other way: the pipeline's listener is called on the worker
thread, and this module forwards it with ``loop.call_soon_threadsafe`` so the app
may ``post_message`` from the callback. ``on_state`` is only ever called on the
loop thread. Nothing here imports ``textual``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from proofpath.events import Cancelled, Event, Listener
from proofpath.report import Report
from proofpath.verify import Engine, Prepared
from proofpath.verify import decide_all as _decide_all
from proofpath.verify import prepare as _prepare

State = Literal[
    "queued",
    "running",
    "waiting for verify",
    "verifying",
    "done",
    "cancelled",
    "failed",
]

#: States a run never leaves. ``cancel()`` on one of these answers ``False``.
TERMINAL: frozenset[str] = frozenset({"done", "cancelled", "failed"})

DEFAULT_IO_LIMIT = 3

#: How long :meth:`Scheduler.close` waits for one worker thread to finish. A stage
#: stops between units of work, so this only ever expires on a unit that hangs.
JOIN_TIMEOUT = 5.0

#: What a worker thread hands back to the loop: the answer of one half of a run.
_T = TypeVar("_T")


@dataclass
class Run:
    """One verification, from the line that asked for it to the report it produced.

    ``report`` outlives the run's state on purpose: a cancelled ``decide_all`` hands
    over the partial report it had built (product rule 6), so a stopped run can still
    show what it decided instead of looking like a run that found nothing.
    """

    id: int  # 1-based, in submission order; what ``/cancel #n`` names
    command: str  # the echoed command line, e.g. "/check ~/paper.pdf"
    target: str
    state: State = "queued"
    cancel: threading.Event = field(default_factory=threading.Event)
    report: Report | None = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None

    @property
    def elapsed(self) -> float | None:
        """Seconds the run has been going, or took. ``None`` before it started."""
        if self.started is None:
            return None
        return (self.finished if self.finished is not None else time.monotonic()) - self.started


#: Built once per run, on that run's own thread. The run is handed over because the
#: engine is not generic: its consent gate must be able to ask *this* run's block the
#: section 7.1 question and get the answer back (spec section 13.1).
EngineFactory = Callable[[Run], Engine]


class Prepare(Protocol):
    """``verify.prepare``, as the scheduler calls it."""

    def __call__(
        self,
        target: str,
        engine: Engine,
        *,
        on_event: Listener | None = ...,
        cancel: threading.Event | None = ...,
    ) -> Prepared: ...


class DecideAll(Protocol):
    """``verify.decide_all``, as the scheduler calls it."""

    def __call__(
        self,
        prepared: Prepared,
        engine: Engine,
        *,
        on_event: Listener | None = ...,
        cancel: threading.Event | None = ...,
    ) -> Report: ...


class _Gate:
    """A counted gate that admits waiters in arrival order.

    ``asyncio.Semaphore`` bounds the count but has never promised the order, and the
    order is the part of spec section 13.1 that a user can see: the run that reached
    the verify queue first is the run that must verify first. Twenty lines here beat
    a scheduling property that holds on one Python version and not another.
    """

    def __init__(self, limit: int) -> None:
        self._free = limit
        self._waiters: deque[asyncio.Future[None]] = deque()

    async def acquire(self) -> None:
        # A free slot is taken only when nobody is already queued, so a late arrival
        # can never overtake a waiter that a release is about to wake.
        if self._free > 0 and not self._waiters:
            self._free -= 1
            return
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                # The slot was handed over before the cancellation arrived; pass it on
                # rather than leaking it.
                self.release()
            else:
                with suppress(ValueError):
                    self._waiters.remove(waiter)
            raise

    def release(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)  # the slot is handed over, not returned
                return
        self._free += 1

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()


@dataclass
class _Handoff:
    """The two-way channel between one run's task and its dedicated thread."""

    prepared: asyncio.Future[bool]  # thread -> loop: the I/O half is over
    finished: asyncio.Future[None]  # thread -> loop: the engine is closed, the run is over
    go: threading.Event = field(default_factory=threading.Event)  # loop -> thread
    verify: bool = False  # what the grant said: run the model half, or drain
    report: Report | None = None
    error: str | None = None
    cancelled: bool = False


class Scheduler:
    """Runs per session: how many start, how many verify, and in which order.

    The scheduler is created on the event loop and every public method except
    :meth:`runs` is called from it. ``on_state`` fires on the loop thread for every
    state change; ``on_event`` fires there too, forwarded from the worker thread.
    """

    def __init__(
        self,
        engine_factory: EngineFactory,
        *,
        io_limit: int = DEFAULT_IO_LIMIT,
        on_event: Callable[[Run, Event], None],
        on_state: Callable[[Run], None],
        prepare: Prepare = _prepare,
        decide_all: DecideAll = _decide_all,
    ) -> None:
        self._engine_factory = engine_factory
        self._on_event = on_event
        self._on_state = on_state
        self._prepare = prepare
        self._decide_all = decide_all
        self._io = _Gate(io_limit)
        self._verify = _Gate(1)  # the NLI model, one run at a time
        self._runs: list[Run] = []
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._threads: list[threading.Thread] = []
        self._closed = False
        #: Worker threads still alive when :meth:`close` gave up waiting for them.
        #: Empty on a clean shutdown; named here rather than silently abandoned.
        self.unstopped: tuple[str, ...] = ()

    # --- the surface the app drives -----------------------------------------------

    @property
    def runs(self) -> tuple[Run, ...]:
        """Every run of this session, in submission order."""
        return tuple(self._runs)

    def get(self, run_id: int) -> Run | None:
        for run in self._runs:
            if run.id == run_id:
                return run
        return None

    def last_done(self) -> Run | None:
        """The most recently finished run that produced a report, or ``None``.

        What ``/summarize`` means by "the last run": newest first, and only a run that
        actually decided something -- a cancelled or failed one has no finished report
        to summarise, and summarising a partial one as though it were whole is the
        absence-of-evidence collapse product rule 2 forbids.
        """
        for run in reversed(self._runs):
            if run.state == "done" and run.report is not None:
                return run
        return None

    def submit(self, target: str, *, command: str) -> Run:
        """Queue a run and return it at once; the work happens in its own task."""
        if self._closed:
            raise RuntimeError("the scheduler is closed")
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        run = Run(id=len(self._runs) + 1, command=command, target=target)
        self._runs.append(run)
        self._tasks[run.id] = asyncio.get_running_loop().create_task(
            self._drive(run), name=f"proofpath-run-{run.id}"
        )
        self._on_state(run)
        return run

    def cancel(self, run_id: int) -> bool:
        """Ask a run to stop. ``False`` when it is unknown or already over.

        A run that is parked says ``cancelled`` here and now and leaves the queue it
        was parked in: one still waiting for an I/O slot never builds an engine, and
        one waiting for the model gives up its place rather than holding an engine
        open until a slot it will not use comes free. Its task is cancelled, which
        releases the handoff its worker thread is parked on, so the thread drains and
        closes the engine without ever calling the model. A run that is *working*
        stops at its pipeline's next checkpoint, which is the granularity ``prepare``
        and ``decide_all`` already offer.
        """
        run = self.get(run_id)
        if run is None or run.state in TERMINAL:
            return False
        run.cancel.set()
        if run.state in {"queued", "waiting for verify"}:
            self._settle(run, "cancelled")
            task = self._tasks.get(run_id)
            if task is not None and not task.done():
                task.cancel()
        return True

    async def wait(self, run_id: int) -> Run:
        """Await one run to its end. Raises ``KeyError`` for an id never submitted."""
        run = self.get(run_id)
        if run is None:
            raise KeyError(run_id)
        task = self._tasks.get(run_id)
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # The run's own cancellation is how it ended, not a failure to wait;
                # anything else is this caller being cancelled and must propagate.
                if not task.cancelled():
                    raise
        return run

    async def close(self) -> None:
        """Stop every run, wait for its thread to close its engine, and return.

        A working run is asked to stop rather than torn down, because the pipeline
        already stops between units of work and a thread killed mid-unit would leave
        an open ``Cache`` and an ONNX session behind. Safe to call twice; afterwards
        :meth:`submit` refuses. Threads that outlast the join are named in
        :attr:`unstopped` instead of being quietly forgotten.
        """
        self._closed = True
        for run in self._runs:
            if run.state not in TERMINAL:
                self.cancel(run.id)
        tasks = [task for task in self._tasks.values() if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        threads = tuple(self._threads)
        self._threads = []
        loop = asyncio.get_running_loop()
        self.unstopped += await loop.run_in_executor(None, _join, threads)

    # --- one run, on the loop ------------------------------------------------------

    async def _drive(self, run: Run) -> None:
        """Own one run from the queue to its final state. Never raises."""
        loop = asyncio.get_running_loop()
        hand = _Handoff(prepared=loop.create_future(), finished=loop.create_future())
        try:
            async with self._io:
                if run.cancel.is_set():
                    self._settle(run, "cancelled")  # cancelled while it was queued
                    return
                run.started = time.monotonic()
                self._transition(run, "running")
                thread = threading.Thread(
                    target=self._work,
                    args=(run, hand, loop),
                    name=f"proofpath-run-{run.id}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
                ready = await hand.prepared
            # The I/O slot is released here: the next queued run starts its own
            # fetching while this one waits for the model.
            if ready:
                self._transition(run, "waiting for verify")
                async with self._verify:
                    hand.verify = not run.cancel.is_set()
                    if hand.verify:
                        self._transition(run, "verifying")
                    hand.go.set()
                    await hand.finished
            else:
                hand.go.set()  # nothing to grant; let the thread finish its cleanup
                await hand.finished
        except asyncio.CancelledError:
            # The task was cancelled out from under the run — by ``cancel()`` leaving
            # a queue, or by whoever owns the loop. The run still has to end as a run:
            # its thread is told to stop and its final state is reported, because a
            # cancelled task that never says so is a run block frozen mid-stage.
            run.cancel.set()
            self._settle(run, "cancelled")
            raise
        finally:
            # Whatever happened up there — including this task being cancelled — the
            # thread must never stay parked on the handoff with an engine open.
            hand.go.set()
        self._settle(run, _state_of(hand), hand)

    def _transition(self, run: Run, state: State) -> None:
        if run.state == state:
            return
        run.state = state
        self._on_state(run)

    def _settle(self, run: Run, state: State, hand: _Handoff | None = None) -> None:
        if run.state in TERMINAL:
            return
        if hand is not None:
            run.report = hand.report
            run.error = hand.error
        run.finished = time.monotonic()
        self._transition(run, state)

    # --- one run, on its own thread ------------------------------------------------

    def _work(self, run: Run, hand: _Handoff, loop: asyncio.AbstractEventLoop) -> None:
        """Build the engine, run both halves, close the engine. Never raises.

        Everything the run owns is created and destroyed here, on this one thread,
        because a ``Cache`` connection may not cross threads. Between the halves the
        thread parks on ``hand.go`` until the loop has granted it the verify slot.
        """
        engine: Engine | None = None
        prepared: Prepared | None = None
        listener = self._listener(run, loop)
        try:
            try:
                engine = self._engine_factory(run)
                prepared = self._prepare(run.target, engine, on_event=listener, cancel=run.cancel)
            except Cancelled as exc:
                hand.cancelled, hand.report = True, exc.report
            except Exception as exc:  # reported on the run, never raised at the loop
                hand.error = _describe(exc)
            finally:
                _resolve(loop, hand.prepared, prepared is not None)

            if prepared is not None and engine is not None:
                hand.go.wait()
                if not hand.verify:
                    hand.cancelled = True  # stopped while it waited for the slot
                else:
                    try:
                        hand.report = self._decide_all(
                            prepared, engine, on_event=listener, cancel=run.cancel
                        )
                    except Cancelled as exc:
                        hand.cancelled, hand.report = True, exc.report
                    except Exception as exc:
                        hand.error = _describe(exc)
        finally:
            if engine is not None:
                with suppress(Exception):
                    engine.close()
            _resolve(loop, hand.finished, None)

    def _listener(self, run: Run, loop: asyncio.AbstractEventLoop) -> Listener:
        """The pipeline's listener, forwarded from the worker thread to the loop."""

        def listen(event: Event) -> None:
            with suppress(RuntimeError):  # the loop closed under a run that outlived it
                loop.call_soon_threadsafe(self._on_event, run, event)

        return listen


def _state_of(hand: _Handoff) -> State:
    if hand.error is not None:
        return "failed"
    return "cancelled" if hand.cancelled else "done"


def _describe(exc: BaseException) -> str:
    """``"TypeError: ..."`` — what the run block prints when a run fails."""
    return f"{type(exc).__name__}: {exc}"


def _resolve(loop: asyncio.AbstractEventLoop, future: asyncio.Future[_T], value: _T) -> None:
    """Hand a worker thread's answer to the loop, whether or not the loop still runs."""

    def done() -> None:
        if not future.done():
            future.set_result(value)

    with suppress(RuntimeError):
        loop.call_soon_threadsafe(done)


def _join(threads: Iterable[threading.Thread]) -> tuple[str, ...]:
    """Wait for each worker thread, and name the ones that did not stop in time."""
    stalled: list[str] = []
    for thread in threads:
        thread.join(timeout=JOIN_TIMEOUT)
        if thread.is_alive():
            stalled.append(thread.name)
    return tuple(stalled)
