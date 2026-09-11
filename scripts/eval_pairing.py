"""Phase 5 harness: how often a citation marker is paired with the right sentence.

Two halves, like ``scripts/eval_coverage.py``:

* the **hand set** (``tests/data/pairing_set.jsonl``): ~60 hand-written passages whose
  expected ``(marker, sentence, refs, paragraph-scoped)`` pairs were written by reading
  the prose, not by running the code. Scoring it needs no network and is the gate the
  unit tests enforce (``tests/test_eval_pairing.py``, rate >= 0.95);
* the **real PDFs** (``--pdf``): each fetched through ``oa.OpenAccess`` and cached under
  ``cache_dir()/datasets/pairing/``, ingested with ``ingest.from_pdf`` and extracted, so
  the report can state what the same code does on printed two-column text. Those rows
  are counts plus sampled pairs for eyeballing: there is no hand truth for them, and a
  number without truth behind it is never presented as an accuracy.

A row expects three things, and every one of them is scored. ``expected`` holds the
(marker, sentence, refs, paragraph-scoped) pairs. ``expected_unsupported`` and
``expected_unresolved`` hold the markers that must be **reported** instead of paired --
an author-year marker (v0.1 never guesses which entry it names) and a number no
bibliography answers. A marker the extractor reports that the row did not declare is an
``extra`` miss: a report nobody asked for is as much a defect as a missing one, and
letting it pass would hide the day the patterns start seeing citations that are not
there.

Usage: uv run python scripts/eval_pairing.py [--set PATH] [--pdf DOI_OR_ARXIV ...]
                                             [--out PATH] [--samples N] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO

from proofpath import claims, fetch, ingest, oa
from proofpath.config import Config, Contact, load_config
from proofpath.document import Document
from proofpath.paths import cache_dir
from proofpath.polite import PoliteClient

DEFAULT_SET = Path(__file__).resolve().parents[1] / "tests" / "data" / "pairing_set.jsonl"
PDF_DIR = "pairing"  # under cache_dir()/datasets/
STYLES: tuple[str, ...] = ("numeric", "ranges", "paragraph", "mixed")
SAMPLES = 20
SAMPLE_WIDTH = 100  # characters of claim text printed per sampled pair

# Miss reasons, in the order the report explains them.
MISSING = "missing"
WRONG_SENTENCE = "wrong sentence"
WRONG_REFS = "wrong refs"
SCOPED_MISMATCH = "scoped mismatch"
EXTRA = "extra"


# --- pure, importable, offline-testable ---------------------------------------


@dataclass(frozen=True)
class Expected:
    """One expected claim, as a human reader of the passage sees it."""

    marker: str  # verbatim, e.g. "[12]"
    sentence: str  # the claim text after every marker of the paragraph is stripped
    refs: tuple[int, ...]  # the bibliography numbers the claim is checked against
    paragraph_scoped: bool = False


@dataclass(frozen=True)
class Row:
    """One passage of the hand set and everything expected of it."""

    id: str
    style: str
    text: str
    expected: tuple[Expected, ...]
    # Markers that must be reported rather than paired. Order does not matter and a
    # marker printed twice is declared once: the check is on the reported set.
    expected_unsupported: tuple[str, ...] = ()
    expected_unresolved: tuple[str, ...] = ()

    @property
    def checks(self) -> int:
        return len(self.expected) + len(self.expected_unsupported) + len(self.expected_unresolved)


@dataclass(frozen=True)
class Miss:
    """One expectation the extractor did not meet, or one claim nobody expected."""

    row_id: str
    marker: str
    reason: str


@dataclass(frozen=True)
class PairingResult:
    rows: int
    expected: int  # every expectation: pairs plus declared unsupported/unresolved markers
    correct: int
    misses: tuple[Miss, ...]
    pairs: int = 0  # of ``expected``, how many are marker-to-sentence pairs

    @property
    def rate(self) -> float:
        """Met expectations over expectations; ``0.0`` for an empty set, never a crash."""
        return self.correct / self.expected if self.expected else 0.0

    @property
    def reports(self) -> int:
        return self.expected - self.pairs


@dataclass(frozen=True)
class LiveRow:
    """What one real document came to. Counts only: no hand truth exists for it."""

    id: str
    source: str  # the URL the bytes came from, or how the text was obtained
    kind: str  # ingest kind: "pdf", or "text" when only extracted text was reachable
    pages: int
    paragraphs: int
    references: int
    markers: int
    claims: int
    unresolved: int
    unsupported: int
    scoped: int
    samples: tuple[tuple[str, str], ...]  # (marker, claim text) pairs for eyeballing


@dataclass(frozen=True)
class _Produced:
    """One claim, flattened to the four things an ``Expected`` speaks about."""

    marker: str
    sentence: str
    refs: tuple[int, ...]
    paragraph_scoped: bool


def load_rows(path: Path) -> list[Row]:
    """Read the JSONL hand set. Malformed rows raise rather than being skipped."""
    rows: list[Row] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: not valid JSON: {error}") from error
        row = _row(raw, where=f"{path}:{number}")
        if row.id in seen:
            raise ValueError(f"{path}:{number}: duplicate id {row.id!r}")
        seen.add(row.id)
        rows.append(row)
    return rows


def _row(raw: dict[str, Any], *, where: str) -> Row:
    for field in ("id", "style", "text", "expected"):
        if field not in raw:
            raise ValueError(f"{where}: missing {field!r}")
    if raw["style"] not in STYLES:
        raise ValueError(f"{where}: unknown style {raw['style']!r}, expected one of {STYLES}")
    row = Row(
        id=str(raw["id"]),
        style=str(raw["style"]),
        text=str(raw["text"]),
        expected=tuple(_expected(entry, where=where) for entry in raw["expected"]),
        expected_unsupported=tuple(str(m) for m in raw.get("expected_unsupported", ())),
        expected_unresolved=tuple(str(m) for m in raw.get("expected_unresolved", ())),
    )
    if not row.checks:
        raise ValueError(f"{where}: the row expects nothing at all")
    return row


def _expected(raw: dict[str, Any], *, where: str) -> Expected:
    expected = Expected(
        marker=str(raw["marker"]),
        sentence=str(raw.get("sentence", "")),
        refs=tuple(int(ref) for ref in raw.get("refs", ())),
        paragraph_scoped=bool(raw.get("paragraph_scoped", False)),
    )
    if not expected.refs:
        raise ValueError(
            f"{where}: {expected.marker!r} names no refs; a marker that resolves to "
            "nothing belongs in 'expected_unresolved'"
        )
    if not expected.sentence:
        raise ValueError(f"{where}: {expected.marker!r} names no sentence")
    return expected


def _produced_pairs(doc: Document) -> tuple[list[_Produced], set[str], set[str]]:
    """Run the extractor: the claims it made, and the markers it reported instead."""
    result = claims.extract(doc)
    pairs = [
        _Produced(
            marker=claim.marker.text if claim.marker is not None else "",
            sentence=claim.text,
            refs=claim.cited_refs,
            paragraph_scoped=claim.paragraph_scoped,
        )
        for claim in result.claims
    ]
    return (
        pairs,
        {marker.text for marker in result.unsupported},
        {marker.text for marker in result.unresolved},
    )


def score(rows: Sequence[Row]) -> PairingResult:
    """Ingest and extract every row, and compare the pairs with what was expected."""
    correct = 0
    misses: list[Miss] = []
    for row in rows:
        row_correct, row_misses = score_row(row)
        correct += row_correct
        misses.extend(row_misses)
    return PairingResult(
        rows=len(rows),
        expected=sum(row.checks for row in rows),
        correct=correct,
        misses=tuple(misses),
        pairs=sum(len(row.expected) for row in rows),
    )


def score_row(row: Row) -> tuple[int, list[Miss]]:
    """``(correct pairs, misses)`` for one row. Every claim is accounted for: one that
    no expectation claims is an ``extra`` miss, never quietly ignored."""
    doc = ingest.from_text(row.text, name=f"{row.id}.txt", kind="text")
    pairs, unsupported, unresolved = _produced_pairs(doc)
    unclaimed = list(range(len(pairs)))
    correct = 0
    misses: list[Miss] = []

    def take(index: int) -> _Produced:
        unclaimed.remove(index)
        return pairs[index]

    for expected in row.expected:
        candidates = [index for index in unclaimed if pairs[index].marker == expected.marker]
        exact = next(
            (index for index in candidates if _matches(pairs[index], expected)),
            None,
        )
        if exact is not None:
            take(exact)
            correct += 1
            continue
        same_sentence = next(
            (index for index in candidates if pairs[index].sentence == expected.sentence),
            None,
        )
        if same_sentence is not None:
            pair = take(same_sentence)
            reason = WRONG_REFS if pair.refs != expected.refs else SCOPED_MISMATCH
            misses.append(Miss(row.id, expected.marker, reason))
        elif candidates:
            take(candidates[0])
            misses.append(Miss(row.id, expected.marker, WRONG_SENTENCE))
        else:
            misses.append(Miss(row.id, expected.marker, MISSING))
    misses.extend(Miss(row.id, pairs[index].marker, EXTRA) for index in unclaimed)

    # The reported markers. A declared one that was not reported is "missing"; one the
    # row never declared is "extra" -- a marker reported out of nowhere is a defect too,
    # and the set has to be closed for the report to state its own coverage.
    for declared, reported in (
        (row.expected_unsupported, unsupported),
        (row.expected_unresolved, unresolved),
    ):
        for marker in declared:
            if marker in reported:
                correct += 1
            else:
                misses.append(Miss(row.id, marker, MISSING))
    undeclared = (
        (unsupported | unresolved) - set(row.expected_unsupported) - set(row.expected_unresolved)
    )
    misses.extend(Miss(row.id, marker, EXTRA) for marker in sorted(undeclared))
    return correct, misses


def _matches(pair: _Produced, expected: Expected) -> bool:
    return (
        pair.sentence == expected.sentence
        and pair.refs == expected.refs
        and pair.paragraph_scoped == expected.paragraph_scoped
    )


def style_breakdown(rows: Sequence[Row]) -> dict[str, tuple[int, int]]:
    """``style -> (expectations, correct)``, so a weak style cannot hide in the total."""
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.style, []).append(row)
    breakdown: dict[str, tuple[int, int]] = {}
    for style in sorted(grouped, key=lambda name: STYLES.index(name) if name in STYLES else 99):
        result = score(grouped[style])
        breakdown[style] = (result.expected, result.correct)
    return breakdown


def sample_pairs(pairs: Sequence[tuple[str, str]], count: int = SAMPLES) -> list[tuple[str, str]]:
    """``count`` evenly spaced pairs, or all of them when there are fewer."""
    if count <= 0:
        return []
    if len(pairs) <= count:
        return list(pairs)
    step = len(pairs) / count
    return [pairs[int(index * step)] for index in range(count)]


def measure_document(
    identifier: str, doc: Document, *, source: str, count: int = SAMPLES
) -> LiveRow:
    """Extract one real document and reduce it to the counts the report prints."""
    result = claims.extract(doc)
    pairs = [
        (
            claim.marker.text if claim.marker is not None else "",
            " ".join(claim.text.split())[:SAMPLE_WIDTH],
        )
        for claim in result.claims
    ]
    return LiveRow(
        id=identifier,
        source=source,
        kind=doc.kind,
        pages=doc.pages,
        paragraphs=len(doc.paragraphs),
        references=len(doc.references),
        markers=len(result.markers),
        claims=len(result.claims),
        unresolved=len(result.unresolved),
        unsupported=len(result.unsupported),
        scoped=sum(1 for claim in result.claims if claim.paragraph_scoped),
        samples=tuple(sample_pairs(pairs, count)),
    )


def render_report(
    result: PairingResult,
    *,
    by_style: dict[str, tuple[int, int]],
    live: Sequence[LiveRow] = (),
    date: str,
) -> str:
    """The markdown pairing report. ``## Notes`` is not written here: it is the one
    section a human has to write, and a generated placeholder would read like one."""
    lines = [f"# Citation pairing — {date}", ""]
    lines.append(f"- rows: {result.rows}")
    lines.append(
        f"- expectations: {result.expected} ({result.pairs} pairs, "
        f"{result.reports} reported markers)"
    )
    styles = ", ".join(f"{style} {checks}" for style, (checks, _) in by_style.items())
    lines.append(f"- styles (expectations): {styles or 'none'}")
    lines.append(f"- pairing rate: **{result.rate:.2f}** ({result.correct}/{result.expected})")
    lines.append("")

    lines.append("## Hand set")
    lines.append("")
    lines.append("| style | checks | correct | rate |")
    lines.append("|---|---|---|---|")
    for style, (checks, correct) in by_style.items():
        rate = f"{correct / checks:.2f}" if checks else "—"
        lines.append(f"| {style} | {checks} | {correct} | {rate} |")
    lines.append(f"| **all** | {result.expected} | {result.correct} | {result.rate:.2f} |")
    lines.append("")

    lines.append("## Misses")
    lines.append("")
    if result.misses:
        lines.append("| row | marker | reason |")
        lines.append("|---|---|---|")
        lines.extend(
            f"| {miss.row_id} | `{miss.marker}` | {miss.reason} |" for miss in result.misses
        )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Real PDFs")
    lines.append("")
    if not live:
        lines.append("- not run (no `--pdf` given)")
        lines.append("")
        return "\n".join(lines) + "\n"
    lines.append(
        "| id | kind | pages | paragraphs | refs | markers | claims | "
        "unresolved | unsupported | scoped |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in live:
        lines.append(
            f"| {row.id} | {row.kind} | {row.pages} | {row.paragraphs} | {row.references} | "
            f"{row.markers} | {row.claims} | {row.unresolved} | {row.unsupported} | "
            f"{row.scoped} |"
        )
    lines.append("")
    for row in live:
        lines.append(f"### Sampled pairs — {row.id}")
        lines.append("")
        lines.append(f"Source: {row.source or '—'}")
        lines.append("")
        if row.samples:
            lines.append("```")
            lines.extend(f"{marker}  {text}" for marker, text in row.samples)
            lines.append("```")
        else:
            # An empty fenced block would read like a rendering failure; the reason a
            # document produced no claim belongs in the hand-written notes.
            lines.append("- no claims to sample")
        lines.append("")
    return "\n".join(lines) + "\n"


def identifier(target: str) -> tuple[str | None, str | None]:
    """``(doi, arxiv_id)`` for a CLI target: ``arXiv:1907.11692`` or a bare DOI."""
    text = target.strip()
    if text.lower().startswith("arxiv:"):
        return None, text.split(":", 1)[1].strip()
    return text, None


def slug(target: str) -> str:
    """A file name for a DOI or arXiv id that survives every filesystem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", target.strip()).strip("_")


# --- live wiring (network; not exercised by unit tests) -----------------------


def build_chain(contact_email: str) -> tuple[oa.OpenAccess, fetch.Fetcher]:
    """The open-access chain with the browser step denied (spec 7.1, product rule 5):
    this harness never installs a ~280 MB engine to read one PDF."""
    config = Config(contact=Contact(email=contact_email))
    fetcher = fetch.Fetcher(config=config, gate=fetch.DenyingGate(), cache=None, interactive=False)
    client = PoliteClient(contact_email=contact_email)
    chain = oa.OpenAccess(fetcher, client, contact_email=contact_email, cache=None)
    return chain, fetcher


def ensure_pdf(
    target: str,
    chain: oa.OpenAccess,
    fetcher: fetch.Fetcher,
    directory: Path,
    *,
    refresh: bool = False,
    log: TextIO | None = None,
) -> tuple[Path | None, str]:
    """The PDF bytes for ``target``, cached on disk so a re-run costs no requests.

    ``oa.fetch`` returns extracted text, not bytes, so the locations are walked here
    and the first one that really answers with a PDF wins. ``(None, "")`` means no
    location did -- the caller falls back to the extracted text and says so.
    """
    log = log or sys.stderr
    path = directory / f"{slug(target)}.pdf"
    # The URL is kept beside the bytes: a report that names a cache path under someone's
    # home directory says nothing about where the document actually came from.
    origin = path.parent / f"{path.name}.url"
    if path.exists() and not refresh:
        url = origin.read_text(encoding="utf-8").strip() if origin.exists() else ""
        return path, f"{url} (cached)" if url else "cached PDF, origin not recorded"
    doi, arxiv_id = identifier(target)
    located = chain.locate(doi, arxiv_id)
    for note in located.notes:
        print(f"  note   {target}: {note}", file=log)
    for location in located.locations:
        fetched = fetcher.fetch(location.url, counts_as_source=False)
        print(
            f"  try    {location.label} {location.url} -> {fetched.outcome.value}/{fetched.kind}",
            file=log,
        )
        if fetched.ok and fetched.kind == "pdf" and fetched.body:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetched.body)
            url = fetched.final_url or location.url
            origin.write_text(url, encoding="utf-8")
            return path, url
    return None, ""


def measure_target(
    target: str,
    chain: oa.OpenAccess,
    fetcher: fetch.Fetcher,
    directory: Path,
    *,
    refresh: bool = False,
    count: int = SAMPLES,
    log: TextIO | None = None,
) -> LiveRow:
    """One real document, end to end: locate, fetch, ingest, extract, count."""
    log = log or sys.stderr
    path, source = ensure_pdf(target, chain, fetcher, directory, refresh=refresh, log=log)
    if path is not None:
        return measure_document(target, ingest.from_pdf(path), source=source, count=count)
    doi, arxiv_id = identifier(target)
    evidence = chain.fetch(doi, arxiv_id)
    print(
        f"  no pdf {target}: falling back to extracted {evidence.kind} "
        f"({evidence.words} words, source {evidence.source or '—'})",
        file=log,
    )
    doc = ingest.from_text(evidence.text, name=target, kind="text")
    return measure_document(
        target,
        doc,
        source=f"{evidence.url or '—'} (extracted text, not PDF)",
        count=count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default=str(DEFAULT_SET), help="the hand-built JSONL set")
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=[],
        metavar="DOI_OR_ARXIV",
        help="real documents to measure, e.g. arXiv:1907.11692 10.1371/journal.pone.0308142",
    )
    parser.add_argument(
        "--out", default="", help="markdown output path ('docs' for the default dated path)"
    )
    parser.add_argument("--samples", type=int, default=SAMPLES, help="sampled pairs per PDF")
    parser.add_argument("--refresh", action="store_true", help="re-download cached PDFs")
    args = parser.parse_args(argv)

    set_path = Path(args.set)
    if not set_path.exists():
        print(f"error: {set_path} not found", file=sys.stderr)
        return 2
    rows = load_rows(set_path)
    result = score(rows)
    breakdown = style_breakdown(rows)
    print(
        f"hand set  {result.rows} rows, {result.expected} expectations, "
        f"rate {result.rate:.3f} ({result.correct}/{result.expected})",
        file=sys.stderr,
    )
    for miss in result.misses:
        print(f"  miss   {miss.row_id} {miss.marker}: {miss.reason}", file=sys.stderr)

    live: list[LiveRow] = []
    if args.pdf:
        chain, fetcher = build_chain(load_config().contact.email)
        directory = cache_dir() / "datasets" / PDF_DIR
        try:
            for target in args.pdf:
                print(f"pdf       {target}", file=sys.stderr)
                row = measure_target(
                    target,
                    chain,
                    fetcher,
                    directory,
                    refresh=args.refresh,
                    count=args.samples,
                )
                live.append(row)
                print(
                    f"  {row.kind}  pages {row.pages}  paragraphs {row.paragraphs}  "
                    f"refs {row.references}  markers {row.markers}  claims {row.claims}  "
                    f"unresolved {row.unresolved}  unsupported {row.unsupported}  "
                    f"scoped {row.scoped}",
                    file=sys.stderr,
                )
        finally:
            fetcher.close()

    date_str = date.today().isoformat()
    report = render_report(result, by_style=breakdown, live=live, date=date_str)
    print(report)
    if args.out:
        out = Path("docs/eval") / f"{date_str}-pairing.md" if args.out == "docs" else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"written   {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
