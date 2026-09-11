"""Offline tests for scripts/eval_coverage.py's importable functions.

No network, no Fetcher: ``select_rows`` is a pure filter over a fake
``ghost_results.json`` payload, ``render_report`` is a pure markdown renderer
over hand-made per-DOI records (the shape ``eval_coverage.py`` persists to
``coverage_results.json``), and ``measure`` is the per-DOI loop driven here by an
injected ``fetch`` callable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from proofpath import oa

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_coverage.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_coverage", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_coverage = _load_script()
select_rows = eval_coverage.select_rows
render_report = eval_coverage.render_report
measure = eval_coverage.measure
build_gate = eval_coverage.build_gate


# --- select_rows ---------------------------------------------------------


FAKE_GHOSTS: dict[str, dict[str, Any]] = {
    "ref A": {"state": "RESOLVED", "best": {"doi": "10.1/a", "title": "A"}},
    "ref B (ghost)": {"state": "GHOST", "best": None},
    "ref C (low)": {"state": "RESOLVED_LOW", "best": {"doi": "10.1/c", "title": "C"}},
    "ref D (no doi)": {"state": "RESOLVED", "best": {"doi": "", "title": "D"}},
    "ref E (ambiguous)": {
        "state": "AMBIGUOUS",
        "best": {"doi": "10.1/e", "title": "E"},
    },
    "ref F": {"state": "RESOLVED", "best": {"doi": "10.1/f", "title": "F"}},
}


def test_select_rows_keeps_resolved_with_doi_in_file_order() -> None:
    rows = select_rows(FAKE_GHOSTS, 50)
    assert [row["doi"] for row in rows] == ["10.1/a", "10.1/c", "10.1/f"]
    assert [row["raw"] for row in rows] == ["ref A", "ref C (low)", "ref F"]


def test_select_rows_excludes_unresolved_ghost_and_ambiguous_and_missing_doi() -> None:
    rows = select_rows(FAKE_GHOSTS, 50)
    dois = {row["doi"] for row in rows}
    assert "10.1/e" not in dois  # AMBIGUOUS
    assert not any(row["raw"] == "ref B (ghost)" for row in rows)  # GHOST, best is None
    assert not any(row["raw"] == "ref D (no doi)" for row in rows)  # empty doi


def test_select_rows_respects_limit() -> None:
    rows = select_rows(FAKE_GHOSTS, 2)
    assert [row["doi"] for row in rows] == ["10.1/a", "10.1/c"]


def test_select_rows_limit_zero_returns_nothing() -> None:
    assert select_rows(FAKE_GHOSTS, 0) == []


def test_select_rows_on_empty_input() -> None:
    assert select_rows({}, 50) == []


# --- render_report ---------------------------------------------------------


RECORDS: list[dict[str, Any]] = [
    {
        "doi": "10.1/one",
        "kind": "fulltext",
        "source": "crossref_link",
        "words": 2200,
        "url": "https://example.org/one.pdf",
        "state": "",
        "attempts": [
            {"label": "s2_pdf", "outcome": "UNVERIFIED (blocked)", "step": 1, "words": 0},
            {"label": "crossref_link", "outcome": "ok", "step": 2, "words": 2200},
        ],
        "notes": ["s2 unavailable (HTTP 429)"],
        "elapsed": 1.2,
    },
    {
        "doi": "10.1/two",
        "kind": "abstract",
        "source": "unpaywall",
        "words": 900,
        "url": "https://example.org/two-landing",
        "state": "LOW CONFIDENCE (abstract only)",
        "attempts": [
            {"label": "s2_pdf", "outcome": "UNVERIFIED (blocked)", "step": 1, "words": 0},
            {"label": "unpaywall", "outcome": "ok", "step": 2, "words": 900},
        ],
        "notes": ["s2 unavailable (HTTP 429)"],
        "elapsed": 0.8,
    },
    {
        "doi": "10.1/three",
        "kind": "none",
        "source": "",
        "words": 0,
        "url": "",
        "state": "UNVERIFIED (unreachable)",
        "attempts": [
            {"label": "landing", "outcome": "UNVERIFIED (unreachable)", "step": 1, "words": 0},
        ],
        "notes": ["no open-access location found"],
        "elapsed": 0.3,
    },
]

# What ``measure`` records when the chain itself raised for a DOI (review Important 2):
# not a "none" result — the DOI was never measured.
ERROR_RECORD: dict[str, Any] = {
    "doi": "10.1/four",
    "kind": "none",
    "state": "ERROR",
    "source": "",
    "words": 0,
    "url": "",
    "attempts": [],
    "notes": ["RuntimeError: boom"],
    "elapsed": 0.1,
}


def test_render_report_headline_counts_and_percentages() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    assert "DOIs measured: 3" in report
    assert "| full text | 1 | 33% |" in report
    assert "| abstract only | 1 | 33% |" in report
    assert "| none | 1 | 33% |" in report


def test_render_report_full_text_by_locator() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    assert "| crossref_link | 1 |" in report
    assert "| s2_pdf | 0 |" in report


def test_render_report_ladder_step_of_winning_fetch() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    # both the fulltext win (crossref_link) and the abstract win (unpaywall)
    # landed at step 2; step 1 never actually delivered usable text.
    assert "| 2 (curl_cffi) | 2 |" in report
    assert "| 1 (httpx) | 0 |" in report


def test_render_report_honesty_states_worst_outcome() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    # record two: worst of {blocked, ok} is blocked. record three: state is
    # already the worst outcome (oa.Evidence for a "none" result).
    assert "| UNVERIFIED (blocked) | 1 |" in report
    assert "| UNVERIFIED (unreachable) | 1 |" in report


def test_render_report_words_median_and_short_reachable_count() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    assert "median words, full text: 2200" in report
    # record two is abstract-grade via a fetched page (unpaywall), reachable
    # but under FULLTEXT_MIN_WORDS.
    assert "reachable but under 1500 words: 1" in report


def test_render_report_per_doi_table_has_one_row_per_record() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    assert "10.1/one" in report
    assert "10.1/two" in report
    assert "10.1/three" in report
    assert "| 10.1/one | fulltext | crossref_link | 2 | 2200 |" in report


def test_render_report_dedupes_and_counts_provider_notes() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    assert "s2 unavailable (HTTP 429) (2x)" in report
    assert "no open-access location found (1x)" in report


def test_render_report_header_states_browser_permission() -> None:
    denied = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    # The script has no --no-browser flag; the reason names the flag it does have.
    assert "browser step (ladder step 3): not allowed (--allow-browser not given)" in denied
    allowed = render_report(RECORDS, date="2026-09-11", browser_allowed=True)
    assert "browser step (ladder step 3): allowed" in allowed


def test_build_gate_reason_names_the_scripts_own_flag() -> None:
    gate = build_gate(allow_browser=False)
    assert gate.decision.outcome == "deny"
    assert gate.decision.reason == "--allow-browser not given"
    assert gate.allow("nature.com", 403) is False
    assert gate.skipped_urls == 1
    allowed = build_gate(allow_browser=True)
    assert allowed.decision.outcome == "allow"
    assert allowed.decision.reason == "--allow-browser"


def test_render_report_empty_records_does_not_crash() -> None:
    report = render_report([], date="2026-09-11", browser_allowed=False)
    assert "DOIs measured: 0" in report


def test_render_report_counts_error_rows_separately_from_none() -> None:
    report = render_report([*RECORDS, ERROR_RECORD], date="2026-09-11", browser_allowed=False)
    assert "DOIs measured: 4" in report
    assert "| none | 1 | 25% |" in report  # the ERROR row is not a "none" result
    assert "| errors | 1 | 25% |" in report
    assert "| 10.1/four | none | — | — | 0 | ERROR |" in report
    assert "RuntimeError: boom (1x)" in report


def test_render_report_without_errors_shows_zero_errors() -> None:
    report = render_report(RECORDS, date="2026-09-11", browser_allowed=False)
    assert "| errors | 0 | 0% |" in report


# --- measure -------------------------------------------------------------------


ROWS: list[dict[str, str]] = [
    {"raw": "ref A", "doi": "10.1/a"},
    {"raw": "ref B", "doi": "10.1/b"},
    {"raw": "ref C", "doi": "10.1/c"},
]


def fulltext(doi: str) -> oa.Evidence:
    text = " ".join(["word"] * 1600)
    return oa.Evidence("fulltext", "", text, f"https://x.test/{doi}.pdf", "s2_pdf", [], [])


def test_measure_persists_after_every_doi() -> None:
    results: dict[str, dict[str, Any]] = {}
    snapshots: list[int] = []
    measure(ROWS, fulltext, results, persist=lambda r: snapshots.append(len(r)))
    assert snapshots == [1, 2, 3]
    assert list(results) == ["10.1/a", "10.1/b", "10.1/c"]
    assert results["10.1/a"]["kind"] == "fulltext"
    assert results["10.1/a"]["words"] == 1600


def test_measure_isolates_a_doi_whose_fetch_raises(capsys: pytest.CaptureFixture[str]) -> None:
    def fetch(doi: str) -> oa.Evidence:
        if doi == "10.1/b":
            raise RuntimeError("boom")
        return fulltext(doi)

    results: dict[str, dict[str, Any]] = {}
    snapshots: list[int] = []
    measure(ROWS, fetch, results, persist=lambda r: snapshots.append(len(r)))
    # The failing DOI is recorded as an ERROR row, persisted, and the loop went on.
    assert snapshots == [1, 2, 3]
    error = results["10.1/b"]
    assert error["kind"] == "none"
    assert error["state"] == "ERROR"
    assert error["source"] == ""
    assert error["words"] == 0
    assert error["attempts"] == []
    assert error["notes"] == ["RuntimeError: boom"]
    assert results["10.1/c"]["kind"] == "fulltext"
    err = capsys.readouterr().err
    assert "10.1/b" in err and "RuntimeError: boom" in err


def test_measure_skips_dois_already_in_results() -> None:
    calls: list[str] = []

    def fetch(doi: str) -> oa.Evidence:
        calls.append(doi)
        return fulltext(doi)

    results: dict[str, dict[str, Any]] = {"10.1/a": {"kind": "fulltext"}}
    measure(ROWS, fetch, results, persist=lambda r: None)
    assert calls == ["10.1/b", "10.1/c"]
