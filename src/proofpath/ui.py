"""Shared output language for the CLI (spec section 13.3).

One thin layer over ``rich``. Every human line a command prints goes through
here, so the layout (10-column key, one space, value) and the colour tables are
defined once. Nothing else in the package imports ``rich`` or names a colour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import orjson
from rich.console import Console
from rich.text import Text

from proofpath.report import Diagnostic, Footer, Level, no_bibliography

KEY_WIDTH = 10

# Spec section 13.2: the stage table's three columns and the width the elapsed
# column is right-aligned to. Measured off the spec's own example, not guessed.
STAGE_WIDTH = 76
STAGE_NAME_WIDTH = 13
STAGE_BY_WIDTH = 34

# Colour tables are fixed by spec section 13.3 "Colour system": ANSI-16 names only,
# no hex, so light and dark themes both work and NO_COLOR removes everything.
# ACCENTS are for the TUI (Phase 8) but live here so one module owns every colour.
ACCENTS = ("cyan", "magenta", "blue", "bright_cyan", "bright_magenta")
GREEN = ("ok", "RESOLVED", "SUPPORTED", "fulltext", "not retracted")
YELLOW_PREFIXES = (
    "UNVERIFIED",
    "LOW CONFIDENCE",
    "AMBIGUOUS",
    "RESOLVED (low confidence)",
    "abstract",
    "PARAGRAPH-SCOPED",
    "UNSUPPORTED CITATION STYLE",
    "UNRESOLVED MARKER",
)
RED = ("GHOST REFERENCE", "REFUTED", "RETRACTED", "FAILED", "NOT SUPPORTED", "PARSE ERROR")
DIM = ("NEI", "none", "—", "cancelled")
_EXACT = {**dict.fromkeys(GREEN, "green"), **dict.fromkeys(RED, "red"), **dict.fromkeys(DIM, "dim")}
# The one place a diagnostic's severity becomes a colour (spec section 13.3). Level
# words are not state words: they say how loud a finding is, not what was found.
LEVEL_STYLES: dict[Level, str] = {"error": "red", "warning": "yellow", "note": "dim"}
# The coverage footer is the one place a number, not a word, carries meaning.
COVERAGE_KEYS = ("fulltext", "abstract", "unverified")
COVERAGE_STYLES = ("green", "yellow", "red")
WEAK_COVERAGE = "coverage is weak: unread sources may hold more, so this is a lower bound"


@dataclass
class Ui:
    """One instance per process, built by the CLI callback from the global flags."""

    color: bool  # False when --no-color, NO_COLOR is set, or stdout is not a TTY
    quiet: bool  # -q: stage and note lines suppressed
    out: Console  # stdout
    err: Console  # stderr


def build(
    *,
    no_color: bool = False,
    quiet: bool = False,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    force_terminal: bool | None = None,
) -> Ui:
    """Build the process-wide ``Ui``. ``force_terminal`` exists for tests only."""
    plain = no_color or "NO_COLOR" in os.environ
    kwargs: dict[str, Any] = {
        "force_terminal": force_terminal,
        "no_color": plain,
        "highlight": False,
        "markup": False,  # titles and ids may contain "[...]"; never interpret them
        "soft_wrap": True,  # a long value stays on one greppable line
        "emoji": False,
    }
    out = Console(file=stdout, **kwargs)
    err = Console(file=stderr, stderr=stderr is None, **kwargs)
    return Ui(color=not plain and out.is_terminal, quiet=quiet, out=out, err=err)


def style_state(ui: Ui, word: str) -> Text:
    """Colour a state word by meaning; unknown words and colourless output stay plain."""
    text = Text(word)
    style = _EXACT.get(word, "yellow" if word.startswith(YELLOW_PREFIXES) else "")
    if ui.color and style:
        text.stylize(style)
    return text


def kv(ui: Ui, key: str, value: str | Text, *, state: bool = False, err: bool = False) -> None:
    """``key`` padded to ``KEY_WIDTH``, one space, ``value``. Never suppressed.
    ``err`` sends the line to stderr, for when stdout must stay one JSON document."""
    if isinstance(value, str):
        value = style_state(ui, value) if state else Text(value)
    # assemble() keeps the key's bold as a span, so it never bleeds into the value.
    console = ui.err if err else ui.out
    console.print(Text.assemble((key.ljust(KEY_WIDTH) + " ", "bold" if ui.color else ""), value))


def state_line(ui: Ui, key: str, word: str, rest: str = "", *, prefix: str = "") -> None:
    """``key``: colour ``word`` by meaning, with optional plain ``prefix``/``rest``
    around it, untouched.

    Lets callers print a coloured state word — with uncoloured detail before it
    (a label like ``"abstract — "``), after it (latency, an HTTP status, a date),
    or both — without touching ``rich.Text`` themselves.
    """
    text = style_state(ui, word)
    if prefix:
        text = Text.assemble(prefix, text)
    if rest:
        text = Text.assemble(text, f" {rest}")
    kv(ui, key, text)


def note(ui: Ui, text: str, *, key: str = "note") -> None:
    """A fact about the run; dropped under ``-q``. ``key`` names a fact of another
    kind that follows the same rule (``attempt`` lines under ``fetch``)."""
    if not ui.quiet:
        kv(ui, key, text)


def hint(ui: Ui, text: str, *, err: bool = False) -> None:
    """What the user can do next; never dropped. ``err``: see ``kv``."""
    kv(ui, "hint", text, err=err)


def stage(ui: Ui, text: str) -> None:
    """Progress line; dropped under ``-q``."""
    if not ui.quiet:
        ui.out.print(text)


def blank(ui: Ui) -> None:
    if not ui.quiet:
        ui.out.print("")


def rule(ui: Ui) -> None:
    if ui.quiet:
        return
    if ui.color:
        ui.out.rule(style="dim")  # rich's default rule colour is green, reserved for meaning
    else:
        ui.out.print("─" * 60)


def diagnostic(ui: Ui, item: Diagnostic) -> None:
    """One compiler-style diagnostic (spec section 13.2).

    Colour lands on the level word and, when the header happens to repeat it, the
    state word; the body is printed exactly as ``report.render_diagnostics`` laid it
    out. Never suppressed: ``-q`` drops progress, never findings (rule 6).
    """
    pad = " " * item.indent
    rest = f"[{item.code}]: {item.title}"
    header = Text.assemble((item.level, LEVEL_STYLES[item.level] if ui.color else ""))
    at = rest.find(item.state) if item.state else -1
    if ui.color and at >= 0:
        header = Text.assemble(
            header, rest[:at], style_state(ui, item.state), rest[at + len(item.state) :]
        )
    else:
        header = Text.assemble(header, rest)
    ui.out.print(Text.assemble(pad, header))
    ui.out.print(f"{pad}  --> {item.location}")
    for line in item.lines:
        ui.out.print(pad + line)


def coverage(
    ui: Ui,
    full: int,
    abstract: int,
    unverified: int,
    *,
    weak: bool,
    unchecked_markers: int = 0,
) -> None:
    """The section 15 coverage block. Never suppressed, never abbreviated (rule 6).

    ``unchecked_markers`` wins over ``weak``: a run whose bibliography was never
    found has 0/0/0 and no weak share to report, and the three zeroes on their own
    read like a document with nothing to check. The markers say otherwise.
    """
    for key, style, share in zip(
        COVERAGE_KEYS, COVERAGE_STYLES, (full, abstract, unverified), strict=True
    ):
        number = Text(f"{share}%")
        if ui.color:
            number.stylize(style)
        kv(ui, key, number)
    if unchecked_markers:
        hint(ui, no_bibliography(unchecked_markers))
    elif weak:
        hint(ui, WEAK_COVERAGE)


def stage_row(ui: Ui, name: str, by: str, summary: str, elapsed: float) -> None:
    """One row of the stage table; dropped under ``-q``.

    A field wider than its column pushes the next one right rather than being cut —
    an attribution is never truncated — but the elapsed column stays at the right
    edge, so the table still reads as a table.
    """
    if ui.quiet:
        return
    row = f"  {name:<{STAGE_NAME_WIDTH}}{by:<{STAGE_BY_WIDTH}}{summary}".rstrip()
    tail = f"{elapsed:.1f}s"
    ui.out.print(f"{row}{' ' * max(1, STAGE_WIDTH - len(row) - len(tail))}{tail}")


def footer(ui: Ui, item: Footer) -> None:
    """The closing lines of a run: counts, coverage, and what the run cost."""
    if item.cancelled:
        state_line(ui, "run", "cancelled")
    ui.out.print(item.counts)
    coverage(ui, *item.coverage, weak=item.weak, unchecked_markers=item.unchecked_markers)
    if item.note:
        hint(ui, item.note)
    written = f"{item.written} written" if item.written else "no report written"
    ui.out.print(f"{written}  ·  {item.api_calls} API calls  ·  {item.elapsed:.1f}s")


def error(ui: Ui, text: str) -> None:
    """To stderr, so ``--format json`` on stdout stays a single document."""
    ui.err.print(Text.assemble(("error: ", "red" if ui.color else ""), text))


def emit_json(ui: Ui, payload: Any) -> None:
    """The result as one indented JSON document on stdout, nothing else."""
    # OPT_NON_STR_KEYS: ``FetchStats.counts`` is keyed by ``Outcome`` (an enum, by value).
    raw = orjson.dumps(
        payload,
        default=_json_default,
        option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
    )
    ui.out.file.write(raw.decode("utf-8") + "\n")
    ui.out.file.flush()


def _json_default(obj: Any) -> Any:
    # orjson handles dataclasses, enums and datetimes natively; the rest lives here.
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        return None  # embeddings and raw payloads are not part of a report
    if isinstance(obj, Path):
        return obj.as_posix()  # one spelling on every platform
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"cannot serialise {type(obj).__name__}")
