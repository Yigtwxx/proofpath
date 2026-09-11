"""What a run tells whoever is watching it, while it is still running.

The orchestrator (``verify.py``) never prints and never knows which front end it
is under: it emits these events and a listener — the TUI's progress panel, the
CLI's spinner, or a test's list — decides what to do with them. Findings travel
twice on purpose: as :class:`Emitted` while the stage that found them is still
running, and again in the finished :class:`~proofpath.report.Report`, so a long
run shows what it already knows instead of a silent bar.

:class:`Cancelled` is raised by the pipeline itself, between units of work, when
the caller's ``threading.Event`` is set. It is a control-flow signal rather than
a failure, which is why the report it leads to says ``cancelled`` and not
``error``: a stopped run is incomplete, not wrong (product rule 6).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from proofpath.report import Finding, Report


@dataclass(frozen=True)
class StageStart:
    """A stage began. ``by`` is the attribution the report will print for it."""

    name: str
    by: str


@dataclass(frozen=True)
class Progress:
    """One unit of a stage is done: ``done`` of ``total``, with an optional label."""

    name: str
    done: int
    total: int
    detail: str = ""


@dataclass(frozen=True)
class StageEnd:
    """A stage finished, with the summary line and the seconds it took."""

    name: str
    by: str
    summary: str
    elapsed: float


@dataclass(frozen=True)
class Note:
    """Something the run needs to say that is not a finding: a permission, a log line."""

    text: str


@dataclass(frozen=True)
class Emitted:
    """A finding, the moment it is found."""

    finding: Finding


@dataclass(frozen=True)
class Prompted:
    """The section 7.1 consent prompt was shown. Informational: the gate already asked."""

    text: str


Event = StageStart | Progress | StageEnd | Note | Emitted | Prompted
Listener = Callable[[Event], None]


class Cancelled(RuntimeError):  # noqa: N818 -- a stopped run is not an error
    """The caller asked the run to stop; raised between units of work.

    ``report`` is the partial run, where there is one to hand over. ``decide_all``
    stops with claims already decided and a coverage summary already true of what it
    read, and dropping that on the floor would leave a stopped run looking like a run
    that found nothing (product rule 6); the report it carries says ``cancelled`` of
    itself, so nobody can mistake it for a complete one. ``prepare`` raises without a
    report -- there is none until the models have been asked -- so a caller that
    wants to show what survived has to handle ``None``.
    """

    def __init__(self, *args: object, report: Report | None = None) -> None:
        super().__init__(*args)
        self.report = report
