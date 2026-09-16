"""Offline tests for scripts/eval_averitec.py's importable functions.

Only the pure half is covered: ``score`` turns already-decided rows into the
numbers the report prints, and ``render_report`` turns those into markdown.
Neither touches the network, a model or the dataset — the rows are hand-made, the
way ``tests/test_eval_scifact.py`` makes them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_averitec.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_averitec", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the script's dataclasses resolve their annotations
    # through ``sys.modules[__module__]``, which is not there for an unregistered spec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_averitec = _load_script()
Row = eval_averitec.Row
score = eval_averitec.score
render_report = eval_averitec.render_report
state_for = eval_averitec.state_for
NOT_A_URL = eval_averitec.NOT_A_URL
NO_SOURCE = eval_averitec.NO_SOURCE

SUPPORTED = "Supported"
REFUTED = "Refuted"
NEI = "Not Enough Evidence"
CONFLICTING = "Conflicting Evidence/Cherrypicking"

OK = "ok"
BLOCKED = "UNVERIFIED (blocked)"
UNREACHABLE = "UNVERIFIED (unreachable)"
NO_TEXT = "UNVERIFIED (reached, no text extracted)"

LIVE = "https://a.example/one"
ARCHIVED = "https://web.archive.org/web/20200101/https://a.example/one"

# Five rows the three-way accuracy can use plus one Conflicting row it cannot:
# four of the five are right, and the golds split 2 Supported / 1 Refuted / 2 NEI.
# Two of the seven source URLs are wayback snapshots.
ROWS = [
    Row(claim_id=0, gold=SUPPORTED, predicted="SUPPORTED", states=(OK,), urls=(LIVE,)),
    Row(claim_id=1, gold=SUPPORTED, predicted="REFUTED", states=(OK,), urls=(ARCHIVED,)),
    Row(
        claim_id=2,
        gold=REFUTED,
        predicted="REFUTED",
        states=(OK, BLOCKED),
        urls=(LIVE, ARCHIVED),
    ),
    Row(claim_id=3, gold=NEI, predicted="NEI", states=(UNREACHABLE,), urls=(LIVE,)),
    # Nothing was fetched at all: no verdict, which counts as NEI.
    Row(claim_id=4, gold=NEI, predicted=None, states=(UNREACHABLE,), urls=(LIVE,)),
    Row(claim_id=5, gold=CONFLICTING, predicted="SUPPORTED", states=(OK,), urls=(LIVE,)),
]


class _StubFetched:
    """The three attributes ``state_for`` reads off a real ``fetch.Fetched``."""

    def __init__(self, outcome: str, *, ok: bool, text: str) -> None:
        self.outcome = SimpleNamespace(value=outcome)
        self.ok = ok
        self.text = text


# --- score ---------------------------------------------------------------


def test_n_counts_every_row_including_the_conflicting_one() -> None:
    assert score(ROWS).n == 6


def test_three_way_accuracy_leaves_conflicting_rows_out_of_the_denominator() -> None:
    # 4 correct of the 5 rows whose gold label is one of our three verdicts.
    assert score(ROWS).accuracy_3way == pytest.approx(4 / 5)


def test_four_way_accuracy_counts_conflicting_rows_as_wrong() -> None:
    assert score(ROWS).accuracy_4way == pytest.approx(4 / 6)


def test_a_row_with_no_prediction_is_scored_as_nei() -> None:
    rows = [Row(claim_id=0, gold=NEI, predicted=None, states=(UNREACHABLE,), urls=(LIVE,))]
    assert score(rows).accuracy_3way == pytest.approx(1.0)
    rows = [Row(claim_id=0, gold=SUPPORTED, predicted=None, states=(UNREACHABLE,), urls=(LIVE,))]
    assert score(rows).accuracy_3way == pytest.approx(0.0)


def test_majority_baseline_is_the_commonest_gold_among_the_three_way_rows() -> None:
    # Supported and NEI tie at 2 of the 5 counted rows; Conflicting does not count.
    assert score(ROWS).majority_baseline == pytest.approx(2 / 5)


def test_per_label_reports_n_and_correct_for_every_gold_label_seen() -> None:
    assert score(ROWS).per_label == {
        SUPPORTED: (2, 1),
        REFUTED: (1, 1),
        NEI: (2, 2),
        CONFLICTING: (1, 0),
    }


def test_coverage_counts_every_state_of_every_source() -> None:
    assert score(ROWS).coverage == {OK: 4, BLOCKED: 1, UNREACHABLE: 2}


def test_an_empty_run_scores_zero_rather_than_dividing_by_zero() -> None:
    result = score([])
    assert result.n == 0
    assert result.accuracy_3way == 0.0
    assert result.accuracy_4way == 0.0
    assert result.majority_baseline == 0.0
    assert result.per_label == {}
    assert result.coverage == {}
    assert result.counted == 0
    assert (result.archive_urls, result.total_urls) == (0, 0)


def test_rows_with_only_conflicting_gold_have_no_three_way_accuracy() -> None:
    rows = [Row(claim_id=0, gold=CONFLICTING, predicted="SUPPORTED", states=(OK,), urls=(LIVE,))]
    result = score(rows)
    assert result.accuracy_3way == 0.0
    assert result.accuracy_4way == 0.0
    assert result.majority_baseline == 0.0


def test_counted_is_the_three_way_denominator_the_result_carries() -> None:
    # Carried, not recomputed by the renderer: one definition of the denominator.
    assert score(ROWS).counted == 5
    assert score([]).counted == 0


def test_a_value_that_was_never_a_url_gets_its_own_coverage_state() -> None:
    rows = [Row(claim_id=0, gold=NEI, predicted=None, states=(NOT_A_URL,), urls=())]
    assert score(rows).coverage == {NOT_A_URL: 1}


def test_archive_snapshots_are_counted_against_every_source_url() -> None:
    result = score(ROWS)
    assert (result.archive_urls, result.total_urls) == (2, 7)


def test_a_run_with_no_urls_counts_no_archive_snapshots() -> None:
    result = score([Row(claim_id=0, gold=NEI, predicted=None, states=(NO_SOURCE,), urls=())])
    assert (result.archive_urls, result.total_urls) == (0, 0)


# --- state_for -----------------------------------------------------------


def test_state_for_reports_the_fetch_outcome_when_there_is_text() -> None:
    fetched = _StubFetched(OK, ok=True, text="a real sentence")
    assert state_for(fetched) == OK


def test_a_reached_page_with_no_extractable_text_is_not_a_clean_ok() -> None:
    # Reaching a page and getting nothing out of it is not the same as reading it;
    # recording it as `ok` would make an empty run look like a covered one (rule 6).
    fetched = _StubFetched(OK, ok=True, text="   \n  ")
    assert state_for(fetched) == NO_TEXT


def test_a_failed_fetch_keeps_its_own_outcome_even_though_it_has_no_text() -> None:
    fetched = _StubFetched(BLOCKED, ok=False, text="")
    assert state_for(fetched) == BLOCKED


# --- render_report -------------------------------------------------------


def test_report_has_every_section() -> None:
    text = render_report(score(ROWS), date="2026-09-15", limit=100)
    for heading in ("## Headline", "## Per label", "## Source coverage", "## Notes"):
        assert heading in text
    assert "2026-09-15" in text


def test_headline_states_the_three_way_accuracy_its_baseline_and_n() -> None:
    text = render_report(score(ROWS), date="2026-09-15", limit=100)
    headline = text.split("## Headline", 1)[1].split("## Per label", 1)[0]
    assert "0.800" in headline
    assert "0.400" in headline
    # Both counts: the five rows the accuracy is read over, of six claims scored.
    assert "5 of 6 claims" in headline


def test_headline_names_the_four_way_accuracy_too() -> None:
    headline = render_report(score(ROWS), date="2026-09-15", limit=100)
    assert "0.667" in headline
    assert "Conflicting Evidence/Cherrypicking" in headline


def test_per_label_table_lists_every_gold_label_with_its_accuracy() -> None:
    text = render_report(score(ROWS), date="2026-09-15", limit=100)
    assert "| label | n | correct | accuracy |" in text
    assert f"| {SUPPORTED} | 2 | 1 | 0.500 |" in text
    assert f"| {REFUTED} | 1 | 1 | 1.000 |" in text
    assert f"| {NEI} | 2 | 2 | 1.000 |" in text
    assert f"| {CONFLICTING} | 1 | 0 | 0.000 |" in text


def test_every_source_state_is_rendered_and_none_is_collapsed() -> None:
    text = render_report(score(ROWS), date="2026-09-15", limit=100)
    coverage = text.split("## Source coverage", 1)[1]
    assert "| state | count |" in coverage
    assert f"| {OK} | 4 |" in coverage
    assert f"| {BLOCKED} | 1 |" in coverage
    assert f"| {UNREACHABLE} | 2 |" in coverage


def test_the_not_a_url_state_gets_its_own_row_in_the_coverage_table() -> None:
    rows = [Row(claim_id=0, gold=NEI, predicted=None, states=(NOT_A_URL, OK), urls=(LIVE,))]
    coverage = render_report(score(rows), date="2026-09-15", limit=1).split(
        "## Source coverage", 1
    )[1]
    assert f"| {NOT_A_URL} | 1 |" in coverage


def test_coverage_says_how_much_of_what_was_measured_was_an_archive_snapshot() -> None:
    coverage = render_report(score(ROWS), date="2026-09-15", limit=100).split(
        "## Source coverage", 1
    )[1]
    assert "2 of 7 source URLs are web.archive.org snapshots (28.6 %)" in coverage


def test_a_run_with_no_sources_still_renders_a_coverage_section() -> None:
    text = render_report(score([]), date="2026-09-15", limit=0)
    assert "## Source coverage" in text
    assert "## Notes" in text


def test_the_report_ends_with_a_newline() -> None:
    assert render_report(score(ROWS), date="2026-09-15", limit=100).endswith("\n")


# --- main ----------------------------------------------------------------


def test_a_fresh_run_refuses_to_overwrite_an_existing_results_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A results file is hours of network; a run started without --resume must say so
    # and stop, rather than replace it. It stops before the dataset is even fetched.
    results = tmp_path / "datasets" / "averitec_results.json"
    results.parent.mkdir(parents=True)
    results.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(eval_averitec, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        eval_averitec.averitec,
        "ensure_downloaded",
        lambda _cache: pytest.fail("refused too late: the dataset was fetched anyway"),
    )

    assert eval_averitec.main([]) == 2

    message = capsys.readouterr().err
    assert "--resume" in message
    assert str(results) in message
    assert results.read_text(encoding="utf-8") == "[]"
