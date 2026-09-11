"""The output side: everything a report says, before anyone decides how to print it.

Spec sections 13.2, 13.3 and 15. ``verify()`` fills these types and the renderers
(text, JSON, SARIF, TUI) read them; the types themselves hold no formatting and no
colour. Three product rules are enforced here rather than left to each renderer:

- A finding that *asserts* something about a source — ``not-supported``,
  ``numeric-mismatch`` — cannot exist without the passage it rests on (rule 1). The
  passage travels inside :class:`~proofpath.models.Verdict`, which already refuses
  to exist without one.
- The honesty states of section 15 are carried verbatim as strings. ``UNVERIFIED
  (blocked, robots.txt)`` never degrades to "unverified" on its way to the page
  (rule 2), which is why ``state`` is a ``str`` and not another enum: the exact
  wording comes from ``fetch.Outcome``, ``oa.ABSTRACT_ONLY`` and ``resolve.State``.
- Every report carries its :class:`Coverage`, and a run that read almost nothing
  says so (rule 6).

The renderers at the bottom turn one :class:`Report` into the three shapes a run is
read in: compiler-style diagnostics (section 13.2), the closing footer, and the
markdown file. They lay text out and colour nothing; ``ui.py`` reads
:class:`Diagnostic` and :class:`Footer` and owns every colour in the package.

This module sits above the whole pipeline: it imports the stage types, and only the
output layer imports it back.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from proofpath.document import Claim, Document, Locator, Reference
from proofpath.fetch import Outcome
from proofpath.models import Label, Tier, Verdict
from proofpath.oa import ABSTRACT_ONLY
from proofpath.resolve import ResolveResult, Retraction, State

# Unverified share at or above which a run's conclusions are called weak (rule 6).
# Public: the renderers colour the coverage footer against it.
WEAK_UNVERIFIED = 0.25

# Prefix shared by every state in the UNVERIFIED family (spec section 15).
UNVERIFIED_PREFIX = "UNVERIFIED"


class Kind(str, Enum):
    """What a finding is. The value is the diagnostic code: ``error[ghost-reference]``."""

    GHOST = "ghost-reference"
    AMBIGUOUS = "ambiguous-reference"
    RETRACTED = "retracted"
    NUMERIC_MISMATCH = "numeric-mismatch"
    NOT_SUPPORTED = "not-supported"
    NEI = "nei"
    UNVERIFIED = "unverified"  # any UNVERIFIED (...) fetch or resolve state
    ABSTRACT_ONLY = "abstract-only"
    PARAGRAPH_SCOPED = "paragraph-scoped"
    UNSUPPORTED_STYLE = "unsupported-citation-style"
    UNRESOLVED_MARKER = "unresolved-marker"
    PARSE_ERROR = "parse-error"
    PROVIDER_UNAVAILABLE = "provider-unavailable"


Level = Literal["error", "warning", "note"]
# Public: the order every renderer prints levels in, and the order `by_level` emits.
LEVEL_ORDER: tuple[Level, ...] = ("error", "warning", "note")

# Severity is a property of the kind, not a judgement call at the call site. Only the
# three kinds that assert something false about the document are errors; a state that
# merely says "we could not tell" is a warning or a note, never an error (rule 2).
LEVELS: dict[Kind, Level] = {
    Kind.GHOST: "error",
    Kind.NUMERIC_MISMATCH: "error",
    Kind.NOT_SUPPORTED: "error",
    Kind.RETRACTED: "warning",
    Kind.AMBIGUOUS: "warning",
    Kind.UNVERIFIED: "warning",
    Kind.UNRESOLVED_MARKER: "warning",
    Kind.PARSE_ERROR: "warning",
    Kind.PROVIDER_UNAVAILABLE: "warning",
    Kind.NEI: "note",
    Kind.ABSTRACT_ONLY: "note",
    Kind.PARAGRAPH_SCOPED: "note",
    Kind.UNSUPPORTED_STYLE: "note",
}

# The state word a renderer colours (spec section 13.3), quoted from section 15 and
# never reworded. Where a stage already owns the wording it is taken from that stage
# rather than copied, so the two can never drift apart (rule 2). A numeric mismatch
# shows NOT SUPPORTED and puts "numeric mismatch" in the verdict's reason, because it
# is a rule's decision and not an entailment call. Kind.UNVERIFIED is the open-ended
# one: which UNVERIFIED (...) applies depends on *why* the fetch or the resolve
# failed, so the exact string arrives in ``Finding.state`` and this entry is the bare
# prefix every member of that family starts with.
STATE_WORDS: dict[Kind, str] = {
    Kind.GHOST: State.GHOST.value,
    Kind.AMBIGUOUS: State.AMBIGUOUS.value,
    Kind.RETRACTED: "RETRACTED",
    Kind.NUMERIC_MISMATCH: "NOT SUPPORTED",
    Kind.NOT_SUPPORTED: "NOT SUPPORTED",
    Kind.NEI: "NEI",
    Kind.UNVERIFIED: UNVERIFIED_PREFIX,
    Kind.ABSTRACT_ONLY: ABSTRACT_ONLY,
    Kind.PARAGRAPH_SCOPED: "PARAGRAPH-SCOPED",
    Kind.UNSUPPORTED_STYLE: "UNSUPPORTED CITATION STYLE",
    Kind.UNRESOLVED_MARKER: "UNRESOLVED MARKER",
    Kind.PARSE_ERROR: "PARSE ERROR",
    Kind.PROVIDER_UNAVAILABLE: Outcome.UNAVAILABLE.value,
}

# The kinds that claim the document says something the source does not. They are the
# only ones that assert, so they are the only ones that owe a passage (rule 1).
_ASSERTING = frozenset({Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH})

TextKind = Literal["fulltext", "abstract", "none"]


@dataclass(frozen=True)
class SourceStatus:
    """One cited reference, from bibliography entry to whatever text we ended up with."""

    reference: Reference
    resolve: ResolveResult | None
    retraction: Retraction | None
    source_id: str | None  # "doi:..." | "arxiv:..." | "url:..."
    text_kind: TextKind
    # "" for full text; else oa.ABSTRACT_ONLY, a fetch.Outcome value, a resolve.State
    # value, "UNVERIFIED (no identifier to fetch)" or "UNVERIFIED (reached, no text
    # extracted)" (both spec section 15). Verbatim, always (rule 2).
    state: str
    fetch_step: int | None
    url: str
    from_cache: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimResult:
    """One judged claim. Supported ones are kept too: ``--format json`` reports them all."""

    claim: Claim
    reference: int  # bibliography number the claim was judged against
    source_id: str
    verdict: Verdict
    from_cache: bool


@dataclass(frozen=True)
class Finding:
    """One line of the report: what is wrong, where, and the evidence for saying so."""

    kind: Kind
    level: Level
    locator: Locator
    title: str  # one line: "cited source does not exist"
    state: str  # STATE_WORDS[kind], or the exact UNVERIFIED (...) string
    reference: Reference | None
    claim: Claim | None
    verdict: Verdict | None  # the passage lives in here (rule 1)
    source_id: str | None
    fetch_step: int | None
    tier: Tier | None
    detail: tuple[str, ...] = ()  # the "= note:" lines under the snippet
    group: str | None = None  # shared by the claims of one paragraph-scoped citation

    def __post_init__(self) -> None:
        if self.level != LEVELS[self.kind]:
            expected = LEVELS[self.kind]
            raise ValueError(f"level for {self.kind.value} is {expected!r}, not {self.level!r}")
        # Rule 2 is only as good as the string that reaches the page, so the state is
        # checked here too: the UNVERIFIED family may say which flavour it is, every
        # other kind must carry its own word and not a paraphrase of it.
        if self.kind is Kind.UNVERIFIED:
            if not self.state.startswith(UNVERIFIED_PREFIX):
                raise ValueError(
                    f"state for {self.kind.value} must start with {UNVERIFIED_PREFIX!r}, "
                    f"got {self.state!r}"
                )
        elif self.state != STATE_WORDS[self.kind]:
            raise ValueError(
                f"state for {self.kind.value} is {STATE_WORDS[self.kind]!r}, not {self.state!r}"
            )
        if self.kind in _ASSERTING and (self.verdict is None or self.verdict.passage is None):
            raise ValueError(f"{self.kind.value} finding requires a verdict with a passage")


@dataclass(frozen=True)
class Coverage:
    """How much of the bibliography was actually read. Never optional (rule 6)."""

    references: int  # cited references; the denominator
    fulltext: int
    abstract: int
    unverified: int  # everything else: ghost, unreachable, blocked, no identifier...
    reasons: dict[str, int]  # state string -> count, so the "why" survives too
    browser_skipped: int  # sources that stopped at the section 7.1 consent gate
    network_denied: bool

    def pct(self) -> tuple[int, int, int]:
        """Full text, abstract and unverified as percentages of ``references``.

        They add up to 100 when the three counts add up to ``references``, which is
        the producer's invariant to keep — ``verify()`` builds this object from one
        pass over the sources, so every reference lands in exactly one bucket. When
        they do not, the percentages stay faithful to the counts they were given
        rather than being scaled into looking complete.
        """
        if self.references <= 0:
            return (0, 0, 0)
        counts = (self.fulltext, self.abstract, self.unverified)
        exact = [100 * count / self.references for count in counts]
        rounded = [int(value) for value in exact]
        # Largest remainder, so the three numbers still total what they should. Ties
        # go to the later bucket: a leftover point lands on unverified before
        # abstract and on abstract before full text, never the flattering way round.
        leftover = round(100 * sum(counts) / self.references) - sum(rounded)
        order = sorted(range(3), key=lambda i: (exact[i] - rounded[i], i), reverse=True)
        for i in order[: max(0, min(leftover, 3))]:
            rounded[i] += 1
        return (rounded[0], rounded[1], rounded[2])

    def weak(self) -> bool:
        """Whether the run read too little for its conclusions to mean much.

        The share is taken from the denominator — everything that is not full text
        and not an abstract — rather than from ``unverified``. A producer that
        forgets to tally a failure then cannot make a run look stronger than it was
        (rule 6); the worst an under-count can do is leave the *reason* unnamed.

        A document with no references at all is not weakly covered — there was
        nothing to cover. Warning that "0 of 0 sources could not be read" describes
        a gap that does not exist; what that run has to say for itself is the
        reference count, which every report prints.
        """
        if self.references <= 0:
            return False
        unread = self.references - self.fulltext - self.abstract
        return unread / self.references >= WEAK_UNVERIFIED


@dataclass(frozen=True)
class Stage:
    """One pipeline stage as the report shows it, with the provider that produced it."""

    name: str  # "Parsing" | "Resolving" | "Retractions" | "Fetching" | "Verifying"
    by: str  # attribution: "pymupdf", "Crossref, OpenAlex", "Retraction Watch", "cpu"
    summary: str  # "24 pages, 42 refs"
    elapsed: float


@dataclass(frozen=True)
class Report:
    """One run, complete. Every renderer takes this and adds nothing to it."""

    document: Document
    claims: int
    markers: int
    sources: tuple[SourceStatus, ...]
    results: tuple[ClaimResult, ...]
    findings: tuple[Finding, ...]
    coverage: Coverage
    stages: tuple[Stage, ...]
    # {"nli": ..., "embedder": ..., "device": ..., "thresholds": "decide=...;high=..."}
    models: dict[str, str]
    api_calls: int
    elapsed: float
    cancelled: bool = False
    summary: str | None = None  # Phase 9, model-written

    def exit_code(self) -> int:
        """0 when the run is clean, 1 when it has findings. 2 is never returned here.

        Spec section 13.3: every reported state counts, including the UNVERIFIED and
        LOW CONFIDENCE ones. A run that could not read its sources is not a clean run.

        ``cancelled`` is folded in so that a stopped run is never reported as clean by
        a caller that only looks at this number. The CLI does not use that branch: it
        exits 2 for a cancelled run, because stopping early is the tool not finishing
        rather than a verdict on the document, and ``Report.cancelled`` is what says so.
        """
        return 1 if self.findings or self.cancelled else 0

    def counts(self) -> dict[Kind, int]:
        """How many findings of each kind, in ``Kind`` declaration order.

        Kinds with no findings are left out, so the summary line lists only what
        actually happened.
        """
        tally = Counter(item.kind for item in self.findings)
        return {kind: tally[kind] for kind in Kind if tally[kind]}

    def by_level(self) -> dict[Level, tuple[Finding, ...]]:
        """Findings grouped error, warning, note; each group in document order.

        Empty levels are left out. Findings without a page sort before page 1, so a
        parse error for the whole file leads the list it belongs to.
        """
        ordered = sorted(self.findings, key=_position)
        groups = {level: tuple(f for f in ordered if f.level == level) for level in LEVEL_ORDER}
        return {level: group for level, group in groups.items() if group}


def _position(item: Finding) -> tuple[int, int, int]:
    return (item.locator.page or 0, item.locator.line, item.locator.column)


# --- rendering --------------------------------------------------------------------
#
# Three renderers, one report. They add no judgement: every string below either comes
# out of the report or is layout. ``ui.py`` colours what they produce and owns every
# colour; nothing here names one.

# The `   | ` / `   = ` gutter of a diagnostic body (spec section 13.2), and the width
# a snippet is cut to so a diagnostic stays one terminal line.
GUTTER = "   "
SNIPPET_LIMIT = 100
# A group member's diagnostic is shifted this far under its group finding.
MEMBER_INDENT = 2

# The section 15 coverage block, label column and all.
COVERAGE_LABELS = ("verified against full text", "abstract only", "unverified")
COVERAGE_WIDTH = 29

# ``pipeline.py`` writes this reason when a rule, not the model, refuted a claim. It
# is parsed back out rather than re-derived, so the caret line and the verdict can
# never disagree about which figure was wrong.
_NUMERIC_REASON = re.compile(
    r"^numeric mismatch: claim says (?P<claim>.+?), source says (?P<source>.+)$"
)

# What the footer's first line counts, in the order it lists them. Everything else a
# run can find is left out: the line is a headline, not a second copy of the findings.
_COUNTED: tuple[tuple[str, tuple[Kind, ...]], ...] = (
    ("ghost", (Kind.GHOST,)),
    ("retracted", (Kind.RETRACTED,)),
    ("unsupported", (Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH)),
)

_LEVEL_HEADINGS: dict[Level, str] = {
    "error": "Errors",
    "warning": "Warnings",
    "note": "Notes",
}


@dataclass(frozen=True)
class Diagnostic:
    """One compiler-style diagnostic, ready to print (spec section 13.2).

    ``lines`` is the body under the ``--> location`` line: the ``|`` block and the
    ``= `` lines, already laid out and pure ASCII. The header is left to the renderer
    because it is the one line that carries colour.
    """

    level: Level
    code: str  # the Kind value: error[ghost-reference]
    title: str
    location: str  # "paper.pdf:4:112" with a page, "draft.md:112" without
    lines: tuple[str, ...]
    state: str  # the word to colour, when the header happens to carry it
    indent: int = 0  # a paragraph-scoped group's members sit under their group


@dataclass(frozen=True)
class Footer:
    """The three closing lines of a run. ``coverage`` is never optional (rule 6)."""

    counts: str  # "42 refs: 3 ghost, 1 retracted, 6 unsupported, 32 ok"
    coverage: tuple[int, int, int]  # Coverage.pct()
    api_calls: int
    elapsed: float
    written: str | None  # the report path, or None when nothing was written
    weak: bool
    cancelled: bool


def render_diagnostics(report: Report) -> list[Diagnostic]:
    """Every finding as a diagnostic, in document order.

    A paragraph-scoped group comes out as its group finding followed by its member
    findings, indented: the group explains why the sentences are judged separately,
    the members carry the passages (spec section 9). A member whose group finding is
    missing is still rendered in place — a finding is never dropped for a layout
    reason.
    """
    # Layout is ASCII throughout; the text a finding is about travels verbatim.
    ordered = sorted(report.findings, key=_position)
    members: dict[str, list[Finding]] = {}
    for item in ordered:
        if item.group is not None and item.kind is not Kind.PARAGRAPH_SCOPED:
            members.setdefault(item.group, []).append(item)
    grouped = {item.group for item in ordered if item.kind is Kind.PARAGRAPH_SCOPED and item.group}

    rendered: list[Diagnostic] = []
    for item in ordered:
        if item.kind is not Kind.PARAGRAPH_SCOPED and item.group in grouped:
            continue  # printed under its group, below
        rendered.append(_diagnostic(report.document, item))
        # ``pop``: two group findings sharing one group would otherwise print the same
        # members twice, and a duplicated finding reads as a second problem.
        if item.kind is Kind.PARAGRAPH_SCOPED and item.group is not None:
            rendered.extend(
                _diagnostic(report.document, member, indent=MEMBER_INDENT)
                for member in members.pop(item.group, ())
            )
    return rendered


def render_footer(report: Report, *, written: str | None = None) -> Footer:
    """The closing lines. ``written`` is the path the caller wrote, if it wrote one."""
    return Footer(
        counts=_counts_line(report),
        coverage=report.coverage.pct(),
        api_calls=report.api_calls,
        elapsed=report.elapsed,
        written=written,
        weak=report.coverage.weak(),
        cancelled=report.cancelled,
    )


def render_markdown(report: Report, *, written_at: datetime | None = None) -> str:
    """The whole run as one markdown document. The caller writes it, UTF-8 encoded."""
    when = written_at if written_at is not None else datetime.now()
    lines: list[str] = [f"# proofpath report — {report.document.name}", ""]
    lines.append(f"- date: {when:%Y-%m-%d %H:%M:%S}")
    lines.extend(f"- {key}: {value}" for key, value in report.models.items())
    lines.append(f"- elapsed: {report.elapsed:.1f}s")
    lines.append(f"- api calls: {report.api_calls}")
    if report.cancelled:
        lines.append("- cancelled: the run stopped early, so this report is partial")
    lines.append("")
    lines.extend(_markdown_findings(report))
    lines.extend(_markdown_checked(report))
    lines.extend(_markdown_sources(report))
    lines.extend(_markdown_coverage(report))
    if report.summary:
        lines.extend(["## Summary (model-written)", "", report.summary, ""])
    return "\n".join(lines) + "\n"


# --- diagnostics ------------------------------------------------------------------


def _diagnostic(document: Document, item: Finding, *, indent: int = 0) -> Diagnostic:
    title = item.title
    if item.tier is not None:
        title = f"{title}  (confidence: {item.tier})"
    return Diagnostic(
        level=item.level,
        code=item.kind.value,
        title=title,
        location=_location(document, item.locator),
        lines=tuple(_body(item)),
        state=item.state,
        indent=indent,
    )


def _location(document: Document, locator: Locator) -> str:
    if locator.page is None:
        return f"{document.name}:{locator.line}"
    return f"{document.name}:{locator.page}:{locator.line}"


def _body(item: Finding) -> list[str]:
    tail = _evidence(item)
    lines: list[str] = []
    snippet = _snippet(item)
    if snippet is not None:
        lines.append(f"{GUTTER}|")
        lines.append(f"{GUTTER}| {snippet}")
        caret = _caret(item, snippet)
        if caret is not None:
            lines.append(caret)
        if tail:
            lines.append(f"{GUTTER}|")  # the separator only separates something
    lines.extend(tail)
    return lines


def _snippet(item: Finding) -> str | None:
    """The line the finding is about: the claim, or the bibliography entry itself."""
    if item.claim is not None:
        text = item.claim.text
    elif item.reference is not None:
        text = item.reference.raw
    else:
        return None
    text = " ".join(text.split())
    if len(text) > SNIPPET_LIMIT:
        text = text[: SNIPPET_LIMIT - 3] + "..."
    return text


def _caret(item: Finding, snippet: str) -> str | None:
    """The ``^^^`` line, for a numeric mismatch and nothing else.

    Only a rule-decided refusal knows which characters are wrong; an entailment call
    is about the sentence, so underlining part of it would claim a precision the
    model does not have. The figure is located in the snippet rather than assumed to
    be there: a truncated, reworded or doubly-numbered claim loses the carets, not the
    diagnostic.
    """
    if item.kind is not Kind.NUMERIC_MISMATCH or item.verdict is None:
        return None
    match = _NUMERIC_REASON.match(item.verdict.reason)
    if match is None:
        return None
    figure = match["claim"]
    # Whole-figure matches only, with the same left guard ``numerics`` extracts by, so
    # "8%" is never found inside "48%". Two matches mean the sentence names the figure
    # twice and nothing here knows which one the rule read: carets are dropped rather
    # than aimed at the wrong number.
    places = [
        found.start()
        for found in re.finditer(rf"(?<![\w.+-]){re.escape(figure)}(?!\w|\.\d)", snippet)
    ]
    if len(places) != 1:
        return None
    return f"{GUTTER}| {' ' * places[0]}{'^' * len(figure)} source reports {match['source']}"


def _evidence(item: Finding) -> list[str]:
    """The ``= `` lines: the quoted passage, the notes, and the honesty state."""
    lines: list[str] = []
    # Rule 1: quoted only when the finding actually carries a passage. There is no
    # branch here that invents one, and none that quotes a verdict without one.
    if item.verdict is not None and item.verdict.passage is not None:
        where = f"passage {item.verdict.passage.index}"
        if item.reference is not None:
            where = f"[{item.reference.number}] {where}"
        lines.append(f'{GUTTER}= source: "{item.verdict.passage.text}"  ({where})')
    lines.extend(f"{GUTTER}= note: {detail}" for detail in item.detail)
    # Rule 2: the UNVERIFIED family says which flavour it is, in its own words.
    if item.kind is Kind.UNVERIFIED:
        lines.append(f"{GUTTER}= state: {item.state}")
    return lines


# --- footer -----------------------------------------------------------------------


def _counts_line(report: Report) -> str:
    tally = report.counts()
    parts = [
        f"{total} {word}"
        for word, kinds in _COUNTED
        if (total := sum(tally.get(kind, 0) for kind in kinds))
    ]
    # "ok" is a property of a reference, not of a finding: one reference with three
    # problems is one reference that is not ok, and a note never makes one not ok.
    troubled = {
        item.reference.number
        for item in report.findings
        if item.reference is not None and item.level in ("error", "warning")
    }
    references = report.coverage.references
    parts.append(f"{max(0, references - len(troubled))} ok")
    return f"{references} refs: " + ", ".join(parts)


# --- markdown ---------------------------------------------------------------------


def _markdown_findings(report: Report) -> list[str]:
    lines = ["## Findings", ""]
    groups = report.by_level()
    if not groups:
        return [*lines, "No findings.", ""]
    for level, items in groups.items():
        lines.extend([f"### {_LEVEL_HEADINGS[level]}", ""])
        for item in items:
            lines.append(_markdown_finding(item))
            lines.extend(_markdown_evidence(item))
        lines.append("")
    return lines


def _markdown_finding(item: Finding) -> str:
    head = f"- **{item.locator.label()}**"
    if item.reference is not None:
        head += f" `[{item.reference.number}]`"
    head += f" {item.state} — {item.title}"
    if item.tier is not None:
        head += f" (confidence: {item.tier})"
    return head


def _markdown_evidence(item: Finding) -> list[str]:
    lines: list[str] = []
    if item.claim is not None:
        lines.append(f'  - you: "{_one_line(item.claim.text)}"')
    if item.verdict is not None and item.verdict.passage is not None:
        lines.append(f'  - source: "{_one_line(item.verdict.passage.text)}"')  # rule 1
    # No "- state:" line: unlike a diagnostic header, the bullet above already
    # carries the state string verbatim, and rule 2 asks for it once, not twice.
    lines.extend(f"  - note: {_one_line(detail)}" for detail in item.detail)
    return lines


def _markdown_checked(report: Report) -> list[str]:
    lines = ["## Checked", ""]
    rows = [item for item in report.results if item.verdict.label is Label.SUPPORTED]
    if not rows:
        return [*lines, "No claim came back supported.", ""]
    lines.extend(["| location | ref | tier | passage |", "| --- | --- | --- | --- |"])
    for row in rows:
        # A SUPPORTED verdict cannot exist without a passage (rule 1, enforced in
        # ``Verdict``), so the column is always filled.
        passage = row.verdict.passage.text if row.verdict.passage is not None else ""
        lines.append(
            f"| {row.claim.locator.label()} | [{row.reference}] "
            f"| {row.verdict.tier} | {_cell(passage)} |"
        )
    lines.append("")
    return lines


def _markdown_sources(report: Report) -> list[str]:
    lines = ["## Sources", ""]
    if not report.sources:
        return [*lines, "No reference was looked up.", ""]
    lines.extend(["| ref | state | text | step | url |", "| --- | --- | --- | --- | --- |"])
    for status in report.sources:
        step = str(status.fetch_step) if status.fetch_step is not None else "—"
        lines.append(
            f"| [{status.reference.number}] | {_cell(status.state or 'ok')} "
            f"| {status.text_kind} | {step} | {_cell(status.url or '—')} |"
        )
    lines.append("")
    return lines


def _markdown_coverage(report: Report) -> list[str]:
    coverage = report.coverage
    lines = ["## Coverage", "", "```"]
    lines.extend(
        f"{label:<{COVERAGE_WIDTH}}{share}%"
        for label, share in zip(COVERAGE_LABELS, coverage.pct(), strict=True)
    )
    lines.extend(["```", ""])
    if coverage.weak():
        # The count comes from the denominator, like ``Coverage.weak()`` itself, so an
        # under-counted failure cannot shrink the number the reader is warned with.
        unread = max(0, coverage.references - coverage.fulltext - coverage.abstract)
        lines.extend(
            [
                f"Coverage is weak: {unread} of {coverage.references} sources could not be "
                "read, so the findings above are a lower bound.",
                "",
            ]
        )
    return lines


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _cell(text: str) -> str:
    return _one_line(text).replace("|", "\\|")
