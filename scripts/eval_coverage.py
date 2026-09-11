"""Phase 4 harness: the real open-access coverage number (OPEN-ITEMS 5.3, spec 6.1).

Runs **live** against the network — Semantic Scholar, Crossref, Unpaywall (with a
contact address), Europe PMC, arXiv and the fetch ladder itself — and reports which
locator and which ladder step delivered text, or which honesty state was reached
instead (spec section 15: unreachable, blocked and provider-unavailable stay
distinct facts, never collapsed into "no full text"). Results are cached under
``cache_dir()/datasets/coverage_results.json`` so a re-run costs no requests.

Reads its DOI set from ``cache_dir()/datasets/ghost_results.json``, Phase 3's
output (``scripts/eval_ghosts.py``): the resolved references that carry a DOI.

Usage: uv run python scripts/eval_coverage.py [--limit N] [--refresh] [--out PATH]
                                               [--allow-browser] [--locate-only]
                                               [--sleep SECONDS]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, TextIO

from proofpath import browser, fetch, oa
from proofpath.config import Config, Contact, Decision, load_config
from proofpath.paths import cache_dir
from proofpath.polite import PoliteClient

GHOSTS_FILE = "ghost_results.json"
RESULTS_FILE = "coverage_results.json"

# oa.Label's members, in chain order: a location the fetch ladder actually climbed,
# as opposed to a provider-supplied abstract ("abstract:<provider>") or the cache.
LOCATOR_LABELS: tuple[str, ...] = (
    "s2_pdf",
    "crossref_link",
    "unpaywall",
    "europepmc",
    "arxiv",
    "landing",
)

KIND_LABELS: dict[str, str] = {
    "fulltext": "full text",
    "abstract": "abstract only",
    "none": "none",
}

# The ``state`` of a record whose chain raised: the DOI was never measured, so it is
# neither a "none" result nor an honesty state — it is reported on its own line.
ERROR_STATE = "ERROR"


# --- pure, importable, offline-testable ---------------------------------------


def select_rows(cached: dict[str, dict[str, Any]], limit: int) -> list[dict[str, str]]:
    """The DOIs worth measuring: ``ghost_results.json`` entries whose ``state`` is
    ``RESOLVED`` or ``RESOLVED_LOW`` and whose ``best.doi`` is non-empty, in file
    order, capped at ``limit`` (``0`` means none — a Python slice, not "unlimited")."""
    rows: list[dict[str, str]] = []
    for raw, entry in cached.items():
        if entry.get("state") not in ("RESOLVED", "RESOLVED_LOW"):
            continue
        best = entry.get("best") or {}
        doi = str(best.get("doi") or "")
        if not doi:
            continue
        rows.append({"raw": raw, "doi": doi})
    return rows[:limit]


def _winning_step(record: dict[str, Any]) -> int | None:
    """The ladder step of the attempt that produced ``record``'s text — only when
    its source is a location the ladder actually fetched, not a provider abstract
    or the cache."""
    source = record.get("source", "")
    if source not in LOCATOR_LABELS:
        return None
    for attempt in record.get("attempts", []):
        if attempt["label"] == source and attempt["outcome"] == fetch.Outcome.OK.value:
            return int(attempt["step"])
    return None


def _worst_attempt_outcome(attempts: list[dict[str, Any]]) -> str | None:
    """The most informative outcome across a record's attempts, in the same
    priority order ``oa.py`` uses for a ``none`` result's ``Evidence.state``."""
    seen = {attempt["outcome"] for attempt in attempts}
    for outcome in oa.OUTCOME_PRIORITY:
        if outcome.value in seen:
            return str(outcome.value)
    return None


def render_report(records: list[dict[str, Any]], *, date: str, browser_allowed: bool) -> str:
    """The markdown coverage report (spec 6.1 / OPEN-ITEMS 5.3): headline numbers,
    the full-text locator and winning-ladder-step breakdowns, honesty states,
    word counts, the per-DOI table and deduplicated provider notes.

    Pure: no network, no ``Fetcher`` — every input is a plain record as persisted
    to ``coverage_results.json`` (``kind``, ``state``, ``source``, ``words``,
    ``url``, ``attempts``, ``notes``).
    """
    n = len(records)
    lines = [f"# Open-access coverage — {date}", ""]
    lines.append(f"- DOIs measured: {n}")
    lines.append(
        "- sources tried: Semantic Scholar, Crossref, Unpaywall, Europe PMC, arXiv, "
        "doi.org landing, OpenAlex (abstract only, last resort)"
    )
    lines.append(
        "- browser step (ladder step 3): "
        + ("allowed" if browser_allowed else "not allowed (--allow-browser not given)")
    )
    lines.append("")

    # 2. headline: full text / abstract only / none, and the DOIs the chain raised
    # on — those are measurement failures, not results, so they never pad "none".
    errors = [r for r in records if r["state"] == ERROR_STATE]
    kind_counts = Counter(str(r["kind"]) for r in records if r["state"] != ERROR_STATE)
    lines.append("## Headline")
    lines.append("")
    lines.append("| result | count | % |")
    lines.append("|---|---|---|")
    for kind in ("fulltext", "abstract", "none"):
        count = kind_counts.get(kind, 0)
        pct = f"{count / n:.0%}" if n else "—"
        lines.append(f"| {KIND_LABELS[kind]} | {count} | {pct} |")
    error_pct = f"{len(errors) / n:.0%}" if n else "—"
    lines.append(f"| errors | {len(errors)} | {error_pct} |")
    lines.append("")

    # 3. full text by locator.
    lines.append("## Full text by locator")
    lines.append("")
    lines.append("| locator | count |")
    lines.append("|---|---|")
    fulltext_by_locator = Counter(str(r["source"]) for r in records if r["kind"] == "fulltext")
    for label in LOCATOR_LABELS:
        lines.append(f"| {label} | {fulltext_by_locator.get(label, 0)} |")
    lines.append("")

    # 4. by ladder step, for the winning fetch (fulltext or a page-derived abstract).
    lines.append("## By ladder step (winning fetch)")
    lines.append("")
    step_counts: Counter[int] = Counter()
    for r in records:
        step = _winning_step(r)
        if step is not None:
            step_counts[step] += 1
    lines.append("| step | count |")
    lines.append("|---|---|")
    for step in (1, 2, 3, 4):
        lines.append(f"| {step} ({fetch.STEP_NAMES[step]}) | {step_counts.get(step, 0)} |")
    lines.append("")

    # 5. honesty states among abstract/none (product rule 2: never collapsed).
    lines.append("## Honesty states (abstract / none)")
    lines.append("")
    honesty_counts: Counter[str] = Counter()
    for r in records:
        if r["state"] == ERROR_STATE:
            continue
        if r["kind"] == "none":
            honesty_counts[str(r["state"])] += 1
        elif r["kind"] == "abstract":
            worst = _worst_attempt_outcome(r.get("attempts", []))
            if worst is not None:
                honesty_counts[worst] += 1
    lines.append("| state | count |")
    lines.append("|---|---|")
    for outcome in oa.OUTCOME_PRIORITY:
        lines.append(f"| {outcome.value} | {honesty_counts.get(outcome.value, 0)} |")
    lines.append("")

    # 6. words: median for full text, and abstracts that were a short-but-reachable
    # page rather than a provider abstract (feeds OPEN-ITEMS 8.16).
    lines.append("## Words")
    lines.append("")
    fulltext_words = [int(r["words"]) for r in records if r["kind"] == "fulltext"]
    if fulltext_words:
        lines.append(f"- median words, full text: {statistics.median(fulltext_words):.0f}")
    else:
        lines.append("- median words, full text: —")
    short_reachable = sum(
        1 for r in records if r["kind"] == "abstract" and r["source"] in LOCATOR_LABELS
    )
    lines.append(
        f"- abstract results whose best page was reachable but under "
        f"{oa.FULLTEXT_MIN_WORDS} words: {short_reachable}"
    )
    lines.append("")

    # 7. per-DOI table.
    lines.append("## Per-DOI")
    lines.append("")
    lines.append("| doi | kind | source | step | words | state |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        step = _winning_step(r)
        step_display = str(step) if step is not None else "—"
        source_display = str(r["source"]) or "—"
        state_display = str(r["state"]) or "—"
        lines.append(
            f"| {r['doi']} | {r['kind']} | {source_display} | {step_display} | "
            f"{r['words']} | {state_display} |"
        )
    lines.append("")

    # 8. provider failure notes, deduplicated with counts.
    lines.append("## Provider notes")
    lines.append("")
    note_counts: Counter[str] = Counter()
    for r in records:
        note_counts.update(str(note) for note in r.get("notes", []))
    if note_counts:
        for note, count in sorted(note_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {note} ({count}x)")
    else:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines) + "\n"


def build_gate(*, allow_browser: bool) -> browser.ConsentGate:
    """Spec 7.1 / product rule 5: never install the browser silently. Default deny,
    regardless of any stored "allow", so the baseline measurement stays honest.
    The script has no ``--no-browser`` flag, so the recorded reason names the flag
    it does have rather than the CLI's."""
    gate = browser.ConsentGate("deny", interactive=False, override=allow_browser)
    if not allow_browser:
        gate.decision = Decision("deny", "--allow-browser not given")
    return gate


# --- live wiring (network; not exercised by unit tests) -----------------------


def _record_from_evidence(doi: str, evidence: oa.Evidence, elapsed: float) -> dict[str, Any]:
    return {
        "doi": doi,
        "kind": evidence.kind,
        "state": evidence.state,
        "source": evidence.source,
        "words": len(evidence.text.split()),
        "url": evidence.url,
        "attempts": [
            {
                "label": attempt.location.label,
                "outcome": attempt.outcome.value,
                "step": attempt.step,
                "words": attempt.words,
            }
            for attempt in evidence.attempts
        ],
        "notes": list(evidence.notes),
        "elapsed": round(elapsed, 3),
    }


def _error_record(doi: str, exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "doi": doi,
        "kind": "none",
        "state": ERROR_STATE,
        "source": "",
        "words": 0,
        "url": "",
        "attempts": [],
        "notes": [f"{type(exc).__name__}: {exc}"],
        "elapsed": round(elapsed, 3),
    }


def measure(
    rows: list[dict[str, str]],
    fetch: Callable[[str], oa.Evidence],
    results: dict[str, dict[str, Any]],
    *,
    persist: Callable[[dict[str, dict[str, Any]]], None],
    sleep: float = 0.0,
    log: TextIO | None = None,
) -> None:
    """Run ``fetch`` for every row not already in ``results``, recording each DOI
    and persisting after every one — a run killed at DOI 37 keeps 36. A DOI whose
    chain raises is recorded as an ``ERROR`` row (one line to ``log``) and the run
    goes on: one provider bug must not cost the other 49 measurements.
    """
    log = log or sys.stderr  # resolved now, not at import: stderr may be redirected
    started = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        doi = row["doi"]
        if doi in results:
            continue
        t0 = time.perf_counter()
        try:
            evidence = fetch(doi)
        except Exception as exc:  # broad on purpose: isolation is the point
            results[doi] = _error_record(doi, exc, time.perf_counter() - t0)
            print(f"  error  {doi}: {type(exc).__name__}: {exc}", file=log)
        else:
            results[doi] = _record_from_evidence(doi, evidence, time.perf_counter() - t0)
        persist(results)
        if i % 5 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  {time.perf_counter() - started:.0f}s", file=log)
        if sleep:
            time.sleep(sleep)


def _render_locate_summary(records: list[dict[str, Any]], *, date_str: str) -> str:
    """The cheap, metadata-only answer to 5.3: which labels produced a location and
    whether the abstract-bearing providers had one, without climbing the ladder."""
    lines = [f"# Open-access coverage — locate-only — {date_str}", ""]
    lines.append(f"- DOIs checked: {len(records)}")
    lines.append("")
    label_counts: Counter[str] = Counter()
    s2_abstract = 0
    crossref_abstract = 0
    for r in records:
        label_counts.update(r["labels"])
        s2_abstract += 1 if r["s2_abstract"] else 0
        crossref_abstract += 1 if r["crossref_abstract"] else 0
    lines.append("## Locations found, by label")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---|")
    for label in LOCATOR_LABELS:
        lines.append(f"| {label} | {label_counts.get(label, 0)} |")
    lines.append("")
    n = len(records)
    lines.append(f"- S2 abstract present: {s2_abstract}/{n}")
    lines.append(f"- Crossref abstract present: {crossref_abstract}/{n}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=50, help="how many resolved DOIs to measure (0 = none)"
    )
    parser.add_argument("--refresh", action="store_true", help="ignore cached results")
    parser.add_argument(
        "--out", default="", help="markdown output path ('docs' for the default dated path)"
    )
    parser.add_argument(
        "--allow-browser",
        action="store_true",
        help="allow the fetch ladder's browser step (a ~280 MB one-time download)",
    )
    parser.add_argument(
        "--locate-only",
        action="store_true",
        help="call oa.locate() only; no fetch ladder, cheap metadata check",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between DOIs")
    args = parser.parse_args(argv)

    ghosts_path = cache_dir() / "datasets" / GHOSTS_FILE
    if not ghosts_path.exists():
        print(
            f"error: {ghosts_path} not found — run `uv run python scripts/eval_ghosts.py` "
            "first to produce the resolved DOI set.",
            file=sys.stderr,
        )
        return 2

    cached_ghosts: dict[str, dict[str, Any]] = json.loads(ghosts_path.read_text(encoding="utf-8"))
    rows = select_rows(cached_ghosts, args.limit)
    print(f"{len(rows)} DOIs selected (limit {args.limit})", file=sys.stderr)
    if not rows:
        print(f"0 DOIs to measure (limit {args.limit}).")
        return 0

    contact_email = load_config().contact.email
    config = Config(contact=Contact(email=contact_email))
    gate = build_gate(allow_browser=bool(args.allow_browser))
    fetcher = fetch.Fetcher(config=config, gate=gate, cache=None, interactive=False)
    client = PoliteClient(contact_email=contact_email)
    chain = oa.OpenAccess(fetcher, client, contact_email=contact_email, cache=None)

    results_path = cache_dir() / "datasets" / RESULTS_FILE
    results: dict[str, dict[str, Any]] = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists() and not args.refresh
        else {}
    )

    def persist(current: dict[str, dict[str, Any]]) -> None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(current, indent=1), encoding="utf-8")

    date_str = date.today().isoformat()
    if args.locate_only:
        locate_records: list[dict[str, Any]] = []
        for i, row in enumerate(rows, start=1):
            doi = row["doi"]
            try:
                located = chain.locate(doi)
            except Exception as exc:  # same isolation as ``measure``
                print(f"  error  {doi}: {type(exc).__name__}: {exc}", file=sys.stderr)
                record = {"labels": [], "s2_abstract": False, "crossref_abstract": False}
                record["notes"] = [f"{type(exc).__name__}: {exc}"]
            else:
                record = {
                    "labels": [loc.label for loc in located.locations],
                    "s2_abstract": bool(located.abstracts.get("s2")),
                    "crossref_abstract": bool(located.abstracts.get("crossref")),
                    "notes": list(located.notes),
                }
            locate_records.append({"doi": doi, **record})
            if i % 5 == 0:
                print(f"  {i}/{len(rows)}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)
        report = _render_locate_summary(locate_records, date_str=date_str)
    else:
        measure(rows, chain.fetch, results, persist=persist, sleep=args.sleep)
        measured = [results[row["doi"]] for row in rows if row["doi"] in results]
        report = render_report(measured, date=date_str, browser_allowed=args.allow_browser)

    print(report)
    if args.out:
        out = (
            Path("docs/eval") / f"{date_str}-coverage.md" if args.out == "docs" else Path(args.out)
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"written   {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
