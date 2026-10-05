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
from typing import Any

import pytest

from proofpath.eval import averitec
from proofpath.search import SearchHit

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
search_urls = eval_averitec.search_urls
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


# --- search_urls: the product's search, gate and all ------------------------


class _Stub:
    name = "stub"

    def __init__(self, urls: list[str], *, error: Exception | None = None) -> None:
        self.urls = urls
        self.error = error
        self.asked: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.asked.append((query, max_results))
        if self.error is not None:
            raise self.error
        return [SearchHit(url, "", rank) for rank, url in enumerate(self.urls, start=1)]


def _claim(
    text: str = "The senator voted against the bill in 2019.", **fields: object
) -> averitec.Claim:
    return averitec.Claim(id=0, text=text, label="Refuted", source_urls=(), **fields)  # type: ignore[arg-type]


def test_search_urls_never_hand_back_the_fact_check_itself() -> None:
    stub = _Stub(
        ["https://factcheck.test/claim-1", "https://factcheck.test/other", "https://news.test/a"]
    )
    claim = _claim(fact_check_url="https://factcheck.test/claim-1")
    assert search_urls(stub, claim, 3).urls == ("https://news.test/a",)


def test_search_urls_exclude_the_checker_across_subdomains() -> None:
    stub = _Stub(
        [
            "https://www.afp.com/en/story",
            "https://factcheck.afp.com/other",
            "https://news.afp.com.evil.test/a",
            "https://notafp.com/b",
        ]
    )
    claim = _claim(fact_check_url="https://factcheck.afp.com/claim-1")
    assert search_urls(stub, claim, 3).urls == (
        "https://news.afp.com.evil.test/a",
        "https://notafp.com/b",
    )


def test_search_urls_ask_for_exactly_the_products_share_of_hits() -> None:
    """Final review, Important 3: the product asks for ``results_per_claim``, and so
    does the eval -- no over-fetch the product never makes."""
    stub = _Stub([])
    searched = search_urls(stub, _claim(), 3)
    assert [asked for _, asked in stub.asked] == [3]
    assert searched.state is None


def test_search_urls_query_with_the_products_sentence_query() -> None:
    stub = _Stub([])
    search_urls(stub, _claim("The senator voted against the #bill in 2019."), 3)
    assert [query for query, _ in stub.asked] == ["The senator voted against the bill in 2019."]


RUSSIAN = "Путин подписал закон о выборах."  # noqa: RUF001 - real Cyrillic letters


def test_a_claim_the_product_would_not_search_is_scored_not_dropped() -> None:
    stub = _Stub(["https://news.test/a"])
    searched = search_urls(stub, _claim(RUSSIAN), 3)
    assert searched.state == eval_averitec.LANGUAGE_UNSUPPORTED
    assert stub.asked == []  # never searched in a language the models cannot read
    row = eval_averitec.unsearched_row(_claim(RUSSIAN), searched.state)
    assert row.predicted is None and row.states == (eval_averitec.LANGUAGE_UNSUPPORTED,)
    assert score([row]).n == 1  # counted, as NEI


def test_a_claim_with_no_sentence_to_search_is_reported_as_not_searched() -> None:
    searched = search_urls(_Stub([]), _claim("???"), 3)
    assert searched.state == eval_averitec.NOT_SEARCHED


def test_a_provider_error_is_recorded_as_unverified_for_that_claim_only() -> None:
    from proofpath.polite import ProviderError

    searched = search_urls(_Stub([], error=ProviderError("HTTP 432", 432, None)), _claim(), 3)
    assert searched.state == eval_averitec.SEARCH_UNAVAILABLE
    assert search_urls(_Stub(["https://news.test/a"]), _claim(), 3).urls == ("https://news.test/a",)


def test_the_report_states_the_measured_path() -> None:
    assert eval_averitec.measured_path(None) == "sentence queries, no judge"
    text = render_report(
        score(ROWS), date="2026-09-29", limit=100, path=eval_averitec.measured_path(None)
    )
    assert "measured path: sentence queries, no judge" in text


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


def test_a_rejected_search_key_stops_the_run_at_the_first_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Round 2: a key the provider refuses would fail every claim the same way; the
    run says so once and exits 2 instead of scoring a hundred claims unverified."""
    from proofpath.config import Config
    from proofpath.search import SearchKeyError

    stub = _Stub(
        [], error=SearchKeyError("tavily rejected the key (HTTP 401); check TAVILY_API_KEY", 401)
    )
    closed: list[bool] = []
    engine = SimpleNamespace(
        searcher=stub,
        search_problem="",
        judge=None,
        escalate=True,
        close=lambda: closed.append(True),
    )
    claims = [_claim(f"The senator voted against the bill in 201{n}.") for n in range(3)]
    monkeypatch.setattr(eval_averitec, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(eval_averitec.averitec, "ensure_downloaded", lambda _cache: tmp_path)
    monkeypatch.setattr(eval_averitec.averitec, "load", lambda _path, limit=None: claims)
    monkeypatch.setattr(eval_averitec, "load_config", Config)
    monkeypatch.setattr(eval_averitec.Engine, "default", staticmethod(lambda *a, **k: engine))
    out = tmp_path / "report.md"

    assert eval_averitec.main(["--search", "--out", str(out)]) == 2

    assert len(stub.asked) == 1  # the first claim, and no retry on the others
    assert closed == [True]
    assert not out.exists()
    message = capsys.readouterr().err
    assert "TAVILY_API_KEY" in message and "rejected" in message


# --- per-source rows: the browser ablation and the judge's hypothetical column ------

Source = eval_averitec.Source
product_label = eval_averitec.product_label
judged_label = eval_averitec.judged_label
judge_decider = eval_averitec.judge_decider
without_browser = eval_averitec.without_browser

GROQ = "groq openai/gpt-oss-120b"
LOCAL = "ollama qwen3.5:9b"


def _read(
    label: str,
    score: float = 0.5,
    *,
    step: int | None = 1,
    tier: str = "medium",
    escalated: bool = False,
    judge_label: str | None = None,
    judge_model: str | None = None,
    url: str = LIVE,
) -> Any:
    return Source(
        url=url,
        state=OK,
        step=step,
        label=label,
        score=score,
        tier=tier,
        escalated=escalated,
        judge_label=judge_label,
        judge_model=judge_model,
    )


def _unread(state: str = UNREACHABLE, *, step: int = 1) -> Any:
    return Source(url=LIVE, state=state, step=step, label=None)


def test_the_product_label_is_the_strongest_assertion_among_the_sources() -> None:
    sources = (_read("SUPPORTED", 0.6), _read("REFUTED", 0.9), _read("NEI", 0.99))
    assert product_label(sources) == "REFUTED"


def test_on_a_tied_score_the_first_source_wins_as_it_always_did() -> None:
    first = _read("SUPPORTED", 0.7, url="https://a.example/1")
    second = _read("REFUTED", 0.7, url="https://b.example/2")
    assert product_label((first, second)) == "SUPPORTED"
    assert product_label((second, first)) == "REFUTED"


def test_the_product_label_is_nei_when_something_was_read_but_nothing_asserted() -> None:
    assert product_label((_read("NEI"), _unread())) == "NEI"


def test_the_product_label_is_none_when_nothing_was_read() -> None:
    assert product_label((_unread(), _unread(BLOCKED))) is None
    assert product_label(()) is None


def test_a_confident_model_verdict_wins_over_the_judge() -> None:
    sources = (
        _read("SUPPORTED", 0.8, tier="high"),
        _read("NEI", tier="low", escalated=True, judge_label="REFUTED", judge_model=GROQ),
    )
    assert judged_label(sources) == "SUPPORTED"
    assert judge_decider(sources) == eval_averitec.DECIDED_BY_MODELS


def test_without_a_confident_verdict_the_judge_opinions_decide() -> None:
    sources = (
        _read("NEI", tier="low", escalated=True, judge_label="SUPPORTED", judge_model=GROQ),
        _read(
            "REFUTED", 0.3, tier="low", escalated=True, judge_label="SUPPORTED", judge_model=GROQ
        ),
    )
    assert product_label(sources) == "REFUTED"  # the product keeps the model's verdict
    assert judged_label(sources) == "SUPPORTED"
    assert judge_decider(sources) == GROQ


def test_a_tied_judge_vote_is_nei() -> None:
    sources = (
        _read("NEI", tier="low", escalated=True, judge_label="SUPPORTED", judge_model=GROQ),
        _read("NEI", tier="low", escalated=True, judge_label="REFUTED", judge_model=LOCAL),
    )
    assert judged_label(sources) == "NEI"
    # No vote won, so every answerer is named: together they left it at NEI.
    assert judge_decider(sources) == f"{GROQ} + {LOCAL}"


def test_the_decider_is_the_model_whose_vote_won_not_every_model_that_answered() -> None:
    sources = (
        _read("NEI", tier="low", escalated=True, judge_label="SUPPORTED", judge_model=GROQ),
        _read("NEI", tier="low", escalated=True, judge_label="SUPPORTED", judge_model=GROQ),
        _read("NEI", tier="low", escalated=True, judge_label="REFUTED", judge_model=LOCAL),
    )
    assert judged_label(sources) == "SUPPORTED"
    assert judge_decider(sources) == GROQ


def test_an_escalation_the_judge_never_answered_keeps_the_model_label_as_its_vote() -> None:
    sources = (_read("REFUTED", 0.3, tier="low", escalated=True),)
    assert judged_label(sources) == "REFUTED"
    assert judge_decider(sources) == eval_averitec.NO_OPINION


def test_a_judge_that_says_nei_everywhere_leaves_a_read_claim_at_nei() -> None:
    sources = (
        _read("REFUTED", 0.3, tier="low", escalated=True, judge_label="NEI", judge_model=GROQ),
    )
    assert judged_label(sources) == "NEI"


def test_a_claim_with_nothing_escalated_or_read_has_no_judged_label() -> None:
    assert judged_label((_unread(),)) is None
    assert judge_decider((_unread(),)) == eval_averitec.NOTHING_ESCALATED


def test_without_browser_drops_only_what_the_browser_read() -> None:
    browser = _read("SUPPORTED", 0.9, step=3)
    cached = _read("REFUTED", 0.4, step=None)
    plain = _read("NEI", step=2)
    dropped = without_browser((browser, cached, plain))
    assert dropped[0].label is None
    assert dropped[1:] == (cached, plain)
    assert product_label(dropped) == "REFUTED"


# --- results files: one per run setup, the legacy name untouched ---------------------


@pytest.mark.parametrize(
    ("search", "browser", "judge", "fresh", "name"),
    [
        (False, False, False, False, "averitec_results.json"),
        (True, False, False, False, "averitec_search_results.json"),
        (False, True, True, True, "averitec_results-browser-judge-fresh.json"),
        (True, True, False, True, "averitec_search_results-browser-fresh.json"),
        (True, True, True, True, "averitec_search_results-browser-judge-fresh.json"),
        (False, False, False, True, "averitec_results-fresh.json"),
    ],
)
def test_every_run_setup_gets_its_own_results_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search: bool,
    browser: bool,
    judge: bool,
    fresh: bool,
    name: str,
) -> None:
    monkeypatch.setattr(eval_averitec, "cache_dir", lambda: tmp_path)
    path = eval_averitec._results_path(search=search, browser=browser, judge=judge, fresh=fresh)
    assert path == tmp_path / "datasets" / name


def test_two_search_setups_on_one_day_write_two_reports() -> None:
    day = "2026-10-05"
    sentence = eval_averitec._report_path(
        day=day, search=True, browser=True, judge=False, fresh=True
    )
    judged = eval_averitec._report_path(day=day, search=True, browser=True, judge=True, fresh=True)
    assert sentence == Path("docs/eval/2026-10-05-averitec-search-browser-fresh.md")
    assert judged == Path("docs/eval/2026-10-05-averitec-search-browser-judge-fresh.md")
    legacy = eval_averitec._report_path(
        day=day, search=False, browser=False, judge=False, fresh=False
    )
    assert legacy == Path("docs/eval/2026-10-05-averitec.md")


def test_rows_round_trip_with_their_sources(tmp_path: Path) -> None:
    row = Row(
        claim_id=7,
        gold=SUPPORTED,
        predicted="REFUTED",
        states=(OK, UNREACHABLE),
        urls=(LIVE, ARCHIVED),
        sources=(
            _read(
                "REFUTED",
                0.3,
                step=3,
                tier="low",
                escalated=True,
                judge_label="SUPPORTED",
                judge_model=GROQ,
            ),
            _unread(),
        ),
    )
    path = tmp_path / "results.json"
    eval_averitec._save_rows(path, [row])
    assert eval_averitec._load_rows(path) == [row]


def test_a_legacy_results_file_without_sources_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        '[{"claim_id": 1, "gold": "Refuted", "predicted": null, "states": ["ok"], "urls": []}]',
        encoding="utf-8",
    )
    (row,) = eval_averitec._load_rows(path)
    assert row.sources == ()


# --- score and report: the ablation and the hypothetical judge ----------------------

JUDGED_ROWS = [
    # The browser read the only assertion; the judge agrees with the model.
    Row(
        claim_id=0,
        gold=SUPPORTED,
        predicted="SUPPORTED",
        states=(OK,),
        urls=(LIVE,),
        sources=(_read("SUPPORTED", 0.9, step=3, tier="high"),),
    ),
    # The model is unsure and wrong; Groq's opinion would have been right.
    Row(
        claim_id=1,
        gold=SUPPORTED,
        predicted="REFUTED",
        states=(OK,),
        urls=(LIVE,),
        sources=(
            _read(
                "REFUTED",
                0.3,
                tier="low",
                escalated=True,
                judge_label="SUPPORTED",
                judge_model=GROQ,
            ),
        ),
    ),
    # Read from the cache: the step is unknown, and the local judge got it wrong.
    Row(
        claim_id=2,
        gold=REFUTED,
        predicted="NEI",
        states=(OK,),
        urls=(LIVE,),
        sources=(
            _read(
                "NEI",
                step=None,
                tier="low",
                escalated=True,
                judge_label="SUPPORTED",
                judge_model=LOCAL,
            ),
        ),
    ),
    # Nothing read; nothing escalated.
    Row(
        claim_id=3,
        gold=NEI,
        predicted=None,
        states=(UNREACHABLE,),
        urls=(LIVE,),
        sources=(_unread(),),
    ),
]


def test_score_leaves_the_product_accuracy_alone() -> None:
    result = score(JUDGED_ROWS, judge=True)
    assert result.accuracy_3way == pytest.approx(2 / 4)


def test_score_drops_browser_reads_for_the_ablation() -> None:
    ablation = score(JUDGED_ROWS).ablation
    assert ablation is not None
    assert ablation.read_sources == 3
    assert ablation.browser_sources == 1
    assert ablation.unknown_step == 1
    # Claim 0 loses its only source and falls to "nothing read" (NEI): 1 of 4 right.
    assert ablation.accuracy_3way == pytest.approx(1 / 4)


def test_rows_without_sources_have_no_ablation_and_no_judged_column() -> None:
    result = score(ROWS)
    assert result.ablation is None
    assert result.judged is None


def test_score_reports_the_hypothetical_judge_accuracy_and_who_decided() -> None:
    judged = score(JUDGED_ROWS, judge=True).judged
    assert judged is not None
    # Claims 0 (models) and 1 (Groq) right, 2 (local) wrong, 3 NEI right.
    assert judged.accuracy_3way == pytest.approx(3 / 4)
    assert judged.escalated == 2
    assert judged.unanswered == 0
    assert judged.opinions == {GROQ: 1, LOCAL: 1}
    assert judged.deciders[eval_averitec.DECIDED_BY_MODELS] == (1, 1)
    assert judged.deciders[GROQ] == (1, 1)
    assert judged.deciders[LOCAL] == (1, 0)


def test_a_judge_run_with_nothing_escalated_still_has_a_judged_column() -> None:
    judged = score(ROWS[:1], judge=True).judged
    assert judged is not None
    assert judged.escalated == 0


def test_the_report_states_the_run_setup() -> None:
    text = render_report(
        score(JUDGED_ROWS, judge=True),
        date="2026-10-05",
        limit=4,
        browser=True,
        fresh=True,
        judge=GROQ,
    )
    assert "browser step: allowed" in text
    assert "fetch: live, cache bypassed" in text
    assert f"judge: {GROQ}" in text


def test_the_ablation_bounds_the_pages_read_and_not_the_accuracy() -> None:
    text = render_report(score(JUDGED_ROWS), date="2026-10-05", limit=4, browser=True)
    assert "## Without the browser" in text
    assert "lower bound on the pages such a run reads" in text
    assert "not a bound on its accuracy" in text
    assert "1 of 3 read sources" in text
    assert "step that read them is unknown" in text  # a cache read is said, not hidden


def test_the_judge_column_says_the_product_does_not_do_this() -> None:
    text = render_report(score(JUDGED_ROWS, judge=True), date="2026-10-05", limit=4, judge=GROQ)
    assert "## Judge (hypothetical)" in text
    assert "never lets the judge change a verdict" in text
    assert "**0.750**" in text
    assert _md_line(GROQ, 1, 1, "1.000") in text  # per deciding model
    assert _md_line(LOCAL, 1, 0, "0.000") in text


def test_a_report_without_sources_or_judge_has_neither_section() -> None:
    text = render_report(score(ROWS), date="2026-10-05", limit=6)
    assert "## Without the browser" not in text
    assert "## Judge (hypothetical)" not in text


def _md_line(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


# --- _decide_claim: live fetch, step kept, the escalated verdicts asked --------------


def _fetched(
    url: str, *, step: int, text: str = "Some text. More text.", outcome: str = "ok"
) -> object:
    from proofpath.fetch import Fetched, Outcome

    return Fetched(
        url=url,
        final_url=url,
        step=step,
        outcome=Outcome(outcome),
        status=200,
        content_type="text/html",
        kind="html",
        body=b"",
        text=text,
        notes=[],
        from_cache=step == 0,
    )


class _Fetcher:
    def __init__(self, pages: dict[str, object]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, bool, bool]] = []

    def fetch(self, url: str, *, anonymous: bool = False, use_cache: bool = True) -> object:
        self.calls.append((url, anonymous, use_cache))
        return self.pages[url]


class _Judge:
    def __init__(self, answers: dict[str, str], model: str = GROQ) -> None:
        self.answers = answers  # passage text -> label value
        self.model = model
        self.asked: list[object] = []

    def review(self, items: list[object], **_: object) -> dict[str, object]:
        from proofpath.judge import JudgeOpinion
        from proofpath.models import Label

        self.asked.extend(items)
        return {
            item.id: JudgeOpinion(Label(self.answers[item.passage]), "why", self.model)  # type: ignore[attr-defined]
            for item in items
            if item.passage in self.answers  # type: ignore[attr-defined]
        }


def _engine(fetcher: _Fetcher) -> object:
    return SimpleNamespace(
        fetcher=fetcher,
        get_embedder=lambda: None,
        get_scorer=lambda: None,
        k=1,
        thresholds=None,
    )


def _verdicts(monkeypatch: pytest.MonkeyPatch, by_url: dict[str, object]) -> None:
    def decide(_claim: str, passages: list[object], *_a: object, **_k: object) -> object:
        return by_url[passages[0].source_id]  # type: ignore[attr-defined]

    monkeypatch.setattr(eval_averitec.pipeline, "decide", decide)


def _verdict(label: str, score: float, tier: str, url: str, reason: str = "") -> object:
    from proofpath.models import Label, Passage, Verdict

    return Verdict(Label(label), score, tier, Passage(f"passage of {url}", url, 0), reason)  # type: ignore[arg-type]


def test_a_fresh_run_climbs_past_the_cache_and_keeps_the_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one, two = "https://a.example/1", "https://b.example/2"
    fetcher = _Fetcher(
        {
            one: _fetched(one, step=3),
            two: _fetched(two, step=1, outcome="UNVERIFIED (unreachable)", text=""),
        }
    )
    _verdicts(monkeypatch, {one: _verdict("SUPPORTED", 0.9, "high", one)})
    claim = averitec.Claim(id=1, text="A claim.", label="Supported", source_urls=(one, two))

    row = eval_averitec._decide_claim(_engine(fetcher), claim, sleep=0, fresh=True)

    assert [call[2] for call in fetcher.calls] == [False, False]
    assert row.predicted == "SUPPORTED"
    assert [(s.step, s.label) for s in row.sources] == [(3, "SUPPORTED"), (1, None)]


def test_only_the_low_band_is_sent_to_the_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    low, high, numeric = "https://a.example/low", "https://b.example/high", "https://c.example/num"
    fetcher = _Fetcher({url: _fetched(url, step=1) for url in (low, high, numeric)})
    _verdicts(
        monkeypatch,
        {
            low: _verdict("REFUTED", 0.3, "low", low),
            high: _verdict("SUPPORTED", 0.9, "high", high),
            numeric: _verdict("REFUTED", 1.0, "low", numeric, reason="numeric mismatch: 3 vs 4"),
        },
    )
    judge = _Judge({f"passage of {low}": "SUPPORTED"})
    claim = averitec.Claim(
        id=2, text="A claim.", label="Supported", source_urls=(low, high, numeric)
    )

    row = eval_averitec._decide_claim(_engine(fetcher), claim, sleep=0, judge=judge)  # type: ignore[arg-type]

    assert [item.passage for item in judge.asked] == [f"passage of {low}"]  # type: ignore[attr-defined]
    by_url = {s.url: s for s in row.sources}
    assert by_url[low].escalated and by_url[low].judge_label == "SUPPORTED"
    assert by_url[low].judge_model == GROQ
    assert not by_url[high].escalated and not by_url[numeric].escalated
    # The product's rule is unchanged: the strongest assertion, here the numeric 1.0.
    assert row.predicted == "REFUTED"
    assert row.predicted == product_label(row.sources)


def test_without_a_judge_nothing_is_asked_and_escalation_is_still_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low = "https://a.example/low"
    fetcher = _Fetcher({low: _fetched(low, step=1)})
    _verdicts(monkeypatch, {low: _verdict("REFUTED", 0.3, "low", low)})
    claim = averitec.Claim(id=3, text="A claim.", label="Refuted", source_urls=(low,))

    row = eval_averitec._decide_claim(_engine(fetcher), claim, sleep=0)

    assert fetcher.calls[0][2] is True  # not fresh: the cache may answer
    (source,) = row.sources
    assert source.escalated and source.judge_label is None


# --- main: the new flags ---------------------------------------------------------------


def test_browser_and_no_browser_cannot_both_be_asked_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        eval_averitec.main(["--browser", "--no-browser"])
    assert exc.value.code == 2


def test_a_judge_run_without_its_key_stops_before_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from proofpath.config import Config

    monkeypatch.setattr(eval_averitec, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(eval_averitec, "load_config", Config)
    monkeypatch.setattr(eval_averitec.judge_mod, "resolve_api_key", lambda _name: None)
    monkeypatch.setattr(
        eval_averitec.averitec,
        "ensure_downloaded",
        lambda _cache: pytest.fail("the dataset was fetched before the key was checked"),
    )

    assert eval_averitec.main(["--judge"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


# --- review round 1: a resume over an older file, the judge's role in search mode -----

# A row from a results file written before sources were kept, right as the product
# scored it, beside one new row whose only source the browser read.
MIXED_ROWS = [
    Row(claim_id=0, gold=SUPPORTED, predicted="SUPPORTED", states=(OK,), urls=(LIVE,)),
    Row(
        claim_id=1,
        gold=REFUTED,
        predicted="REFUTED",
        states=(OK,),
        urls=(LIVE,),
        sources=(_read("REFUTED", 0.9, step=1, tier="high"),),
    ),
]


def test_an_older_row_keeps_the_products_label_in_the_ablation() -> None:
    ablation = score(MIXED_ROWS).ablation
    assert ablation is not None
    assert ablation.browser_sources == 0
    assert ablation.unrecorded == 1
    # No browser page was dropped, so nothing may move: 2 of 2, as the product scored.
    assert ablation.accuracy_3way == pytest.approx(1.0)


def test_an_older_row_keeps_the_products_label_in_the_judged_column() -> None:
    judged = score(MIXED_ROWS, judge=True).judged
    assert judged is not None
    assert judged.accuracy_3way == pytest.approx(1.0)
    assert judged.unrecorded == 1
    assert judged.deciders[eval_averitec.UNRECORDED] == (1, 1)


def test_the_report_counts_the_rows_with_no_per_source_record() -> None:
    text = render_report(score(MIXED_ROWS, judge=True), date="2026-10-05", limit=2, judge=GROQ)
    assert "1 rows have no per-source record" in text


def test_in_search_mode_the_report_says_the_judge_wrote_the_queries() -> None:
    result = score(JUDGED_ROWS, judge=True)
    gold = render_report(result, date="2026-10-05", limit=4, judge=GROQ)
    searched = render_report(
        result,
        date="2026-10-05",
        limit=4,
        judge=GROQ,
        path=eval_averitec.measured_path(SimpleNamespace(name=GROQ)),
    )
    assert "the verdicts are the models'" in gold
    assert "writes the search queries" in searched
    assert "writes the search queries" not in gold
