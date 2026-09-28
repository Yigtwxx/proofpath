"""The Searching stage end to end, offline (OPEN-ITEMS 17.1a, spec 2026-09-28).

A stub searcher answers by substring, pages come from ``StubFetcher``, and both models
are the fakes ``tests/test_verify.py`` uses. Nothing here opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proofpath import oa
from proofpath.cache import Cache, claim_hash
from proofpath.config import Config, Permissions, SearchConfig
from proofpath.fetch import Fetched, Outcome
from proofpath.judge import Judge, JudgeUnavailable
from proofpath.polite import ProviderError
from proofpath.report import (
    FOUND_BY_PROOFPATH,
    LANGUAGE_UNSUPPORTED,
    SEARCH_UNAVAILABLE,
    Kind,
)
from proofpath.search import SearchHit
from proofpath.search.queries import LLM_LIMIT, sentence_query
from proofpath.secrets import CREDENTIALS_MISSING
from proofpath.verify import (
    NO_SOURCE_HINT,
    NO_SOURCE_IN_TEXT,
    SEARCH_CANNOT_RUN,
    SEARCHING,
    Engine,
    decide_all,
    prepare,
    verify,
)
from tests.fakes import TableScorer, WordEmbedder
from tests.test_verify import (
    LOW_REFUTED_ROW,
    NEI_ROW,
    REFUTED_ROW,
    SUPPORTED_ROW,
    FakeJudgeClient,
    StubFetcher,
    _opinion,
    engine,
    fetched,
    words,
)

CLAIM = "OpenAI shut down ChatGPT in March 2025."
PAGE_A = "https://news.test/openai"
PAGE_B = "https://wiki.test/ChatGPT"
BACKING = "OpenAI shut down ChatGPT in March 2025 after a vote."
# A sentence of its own, so the page splits into the backing line and the rest, and
# long enough for the page to count as full text.
FILLER = f"The rest is padding: {words(oa.FULLTEXT_MIN_WORDS + 10)}."


class StubSearcher:
    name = "stub"

    def __init__(
        self, answers: dict[str, list[str]] | None = None, *, error: ProviderError | None = None
    ) -> None:
        self.answers = answers or {}
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        for key, urls in self.answers.items():
            if key in query:
                return [SearchHit(url, "", rank) for rank, url in enumerate(urls, start=1)]
        return []


def searching(
    searcher: StubSearcher | None,
    *,
    pages: dict[str, str] | None = None,
    table: dict[str, tuple[float, float, float]] | None = None,
    config: Config | None = None,
    network_allowed: bool = True,
) -> Engine:
    fetcher = StubFetcher(
        {url: fetched(url, text) for url, text in (pages or {}).items()},
        network_allowed=network_allowed,
        network_note="" if network_allowed else "network not permitted",
    )
    built = engine(
        fetcher=fetcher,
        config=config,
        embedder=WordEmbedder(),
        scorer=TableScorer(table or {BACKING: SUPPORTED_ROW}),
    )
    built.searcher = searcher
    return built


def test_a_bare_claim_is_searched_and_the_backing_page_is_reported_as_found() -> None:
    searcher = StubSearcher({"ChatGPT": [PAGE_A]})
    report = verify(CLAIM, searching(searcher, pages={PAGE_A: f"{BACKING} {FILLER}"}))

    assert searcher.queries == [sentence_query(CLAIM)]
    assert report.counts() == {Kind.EVIDENCE_FOUND: 1}
    found = report.findings[0]
    assert found.verdict is not None and found.verdict.passage is not None
    assert found.verdict.passage.text == BACKING
    assert found.detail[-1] == FOUND_BY_PROOFPATH
    assert report.sources[0].reference.origin == "search"
    assert report.sources[0].reference.raw == PAGE_A
    assert SEARCHING in [stage.name for stage in report.stages]
    assert report.search is not None
    assert (report.search.searched, report.search.pages_found, report.search.pages_read) == (
        1,
        1,
        1,
    )


def test_a_search_that_finds_nothing_is_no_evidence_never_refuted() -> None:
    report = verify(CLAIM, searching(StubSearcher()))

    assert report.counts() == {Kind.NO_EVIDENCE: 1}
    assert report.findings[0].claim is not None
    assert report.findings[0].claim.text == CLAIM


def test_an_unreadable_found_page_keeps_its_state_and_says_who_found_it() -> None:
    report = verify(CLAIM, searching(StubSearcher({"ChatGPT": [PAGE_A]})))

    assert report.counts() == {Kind.UNVERIFIED: 1, Kind.NO_EVIDENCE: 1}
    unread = next(f for f in report.findings if f.kind is Kind.UNVERIFIED)
    assert unread.detail[-1] == FOUND_BY_PROOFPATH


def test_a_page_that_says_nothing_either_way_is_nei_and_no_evidence() -> None:
    report = verify(
        CLAIM,
        searching(
            StubSearcher({"ChatGPT": [PAGE_A]}),
            pages={PAGE_A: f"{BACKING} {FILLER}"},
            table={BACKING: NEI_ROW},
        ),
    )
    assert Kind.NOT_SUPPORTED not in report.counts()
    assert report.counts()[Kind.NO_EVIDENCE] == 1
    assert report.counts()[Kind.NEI] == 1
    nei = next(f for f in report.findings if f.kind is Kind.NEI)
    assert nei.detail[-1] == FOUND_BY_PROOFPATH


def test_a_provider_that_fails_is_reported_and_the_claim_is_not_called_empty() -> None:
    searcher = StubSearcher(error=ProviderError("HTTP 503", 503, None))
    report = verify(CLAIM, searching(searcher))

    assert report.counts() == {Kind.UNVERIFIED: 1}
    assert report.findings[0].state == SEARCH_UNAVAILABLE
    assert report.search is not None and report.search.searched == 0


def test_without_a_searcher_the_old_parse_error_stands_with_the_setup_hint() -> None:
    ready = prepare(CLAIM, searching(None))
    errors = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert [f.title for f in errors] == [NO_SOURCE_IN_TEXT]
    assert errors[0].detail == (NO_SOURCE_HINT,)
    assert "search.provider tavily" in NO_SOURCE_HINT
    assert "search.provider searxng" in NO_SOURCE_HINT
    assert "TAVILY_API_KEY" in NO_SOURCE_HINT
    assert "search.base_url" in NO_SOURCE_HINT


def test_no_search_and_a_denied_network_search_nothing() -> None:
    off = StubSearcher({"ChatGPT": [PAGE_A]})
    built = searching(off)
    built.search = False
    prepare(CLAIM, built)
    denied = StubSearcher({"ChatGPT": [PAGE_A]})
    prepare(CLAIM, searching(denied, network_allowed=False))
    assert off.queries == [] and denied.queries == []


def test_a_configured_engine_with_search_off_gives_the_parse_error_with_no_hint() -> None:
    """A searcher is already set up; telling the user to configure it is wrong."""
    built = searching(StubSearcher({"ChatGPT": [PAGE_A]}))
    built.search = False
    ready = prepare(CLAIM, built)
    errors = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert [f.title for f in errors] == [NO_SOURCE_IN_TEXT]
    assert errors[0].detail == ()


def test_a_configured_provider_that_cannot_run_says_why() -> None:
    built = searching(None)
    built.search_problem = "set TAVILY_API_KEY in .env"
    ready = prepare(CLAIM, built)
    [item] = ready.findings
    assert item.kind is Kind.UNVERIFIED
    assert item.state == CREDENTIALS_MISSING
    assert item.detail == ("set TAVILY_API_KEY in .env",)


def test_a_page_with_no_text_and_a_broken_provider_keeps_the_parse_error() -> None:
    """Controller decision, fix round 1b (2026-09-28): a page with nothing to search
    keeps its plain parse error even when a provider is configured and broken -- there
    is no sentence to search, so pointing at the search credentials would mislead."""
    empty_url = "https://empty.test/story"
    fetcher = StubFetcher({empty_url: own_page(empty_url, "")})
    built = engine(fetcher=fetcher, embedder=WordEmbedder(), scorer=TableScorer({}))
    built.searcher = None
    built.search_problem = "set TAVILY_API_KEY in .env"
    ready = prepare(empty_url, built)
    assert ready.findings
    assert all(f.kind is Kind.PARSE_ERROR for f in ready.findings)
    assert CREDENTIALS_MISSING not in {f.state for f in ready.findings}
    assert SEARCH_CANNOT_RUN not in {f.title for f in ready.findings}


def test_search_off_suppresses_the_credential_finding_too() -> None:
    """``--no-search`` means no search, not even the "cannot run" finding (rule 3)."""
    built = searching(None)
    built.search_problem = "set TAVILY_API_KEY in .env"
    built.search = False
    ready = prepare(CLAIM, built)
    assert [f.title for f in ready.findings] == [NO_SOURCE_IN_TEXT]
    assert SEARCH_CANNOT_RUN not in [f.title for f in ready.findings]


def test_a_refuted_result_from_a_found_page_says_so() -> None:
    report = verify(
        CLAIM,
        searching(
            StubSearcher({"ChatGPT": [PAGE_A]}),
            pages={PAGE_A: f"{BACKING} {FILLER}"},
            table={BACKING: REFUTED_ROW},
        ),
    )
    assert report.counts() == {Kind.NOT_SUPPORTED: 1}
    finding = report.findings[0]
    assert finding.title == "claim is not supported by a page proofpath found"
    assert finding.detail[-1] == FOUND_BY_PROOFPATH


def own_page(url: str, text: str) -> Fetched:
    """A page fetched as the *target* itself, read straight from its text (no HTML
    parsing needed): its own host must never be counted as a source for it."""
    return Fetched(
        url=url,
        final_url=url,
        step=1,
        outcome=Outcome.OK,
        status=200,
        content_type="text/plain",
        kind="text",
        body=b"",
        text=text,
        notes=[],
    )


def test_a_hit_on_the_input_pages_own_host_is_dropped() -> None:
    own_url = "https://own.test/story"
    own_host_hit = "https://own.test/other-page"
    searcher = StubSearcher({"ChatGPT": [own_host_hit, PAGE_A]})
    fetcher = StubFetcher(
        {own_url: own_page(own_url, CLAIM), PAGE_A: fetched(PAGE_A, f"{BACKING} {FILLER}")}
    )
    built = engine(
        fetcher=fetcher, embedder=WordEmbedder(), scorer=TableScorer({BACKING: SUPPORTED_ROW})
    )
    built.searcher = searcher
    report = verify(own_url, built)
    assert report.search is not None and report.search.pages_found == 1
    assert [source.reference.raw for source in report.sources] == [PAGE_A]


class SecondCallFails:
    """A searcher that answers the first query and then goes down."""

    name = "stub"

    def __init__(self, first: list[str], error: ProviderError) -> None:
        self.first = first
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        if len(self.queries) == 1:
            return [SearchHit(url, "", rank) for rank, url in enumerate(self.first, start=1)]
        raise self.error


def test_a_provider_error_on_the_second_claim_keeps_the_first_claims_results() -> None:
    second_claim = "Google shut down Bard in June 2025."
    body = f"{CLAIM} {second_claim}"
    searcher = SecondCallFails([PAGE_A], ProviderError("HTTP 503", 503, None))
    report = verify(body, searching(searcher, pages={PAGE_A: f"{BACKING} {FILLER}"}))

    assert len(searcher.queries) == 2
    assert report.search is not None and report.search.searched == 1
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.claim is not None and found.claim.text == CLAIM
    assert report.counts()[Kind.UNVERIFIED] == 1
    unavailable = next(f for f in report.findings if f.state == SEARCH_UNAVAILABLE)
    assert unavailable.detail == ("stub: HTTP 503",)


def test_a_text_that_cites_something_is_never_searched() -> None:
    searcher = StubSearcher({"ChatGPT": [PAGE_A]})
    prepare(f"{CLAIM} See {PAGE_B}", searching(searcher))
    assert searcher.queries == []


def test_a_text_with_no_sentence_to_search_keeps_its_parse_error() -> None:
    """A search over nothing would end on no finding at all and read as clean (rule 6)."""
    searcher = StubSearcher()
    report = verify("???", searching(searcher))
    assert searcher.queries == []
    assert report.search is None
    assert [f.title for f in report.findings] == [NO_SOURCE_IN_TEXT]


def test_the_cap_limits_the_search_and_the_rest_is_counted() -> None:
    body = " ".join(f"Company{i} sold {i} million units in 2020." for i in range(7))
    searcher = StubSearcher()
    report = verify(body, searching(searcher, config=Config(search=SearchConfig(max_claims=5))))
    assert len(searcher.queries) == 5
    assert report.search is not None
    assert (report.search.eligible, report.search.searched) == (7, 5)


def test_a_rate_limited_judge_falls_back_to_the_sentence_and_the_report_says_so() -> None:
    searcher = StubSearcher()
    built = searching(searcher)
    built.judge = Judge(FakeJudgeClient([JudgeUnavailable("HTTP 429 from https://api.test")]))  # type: ignore[arg-type]
    report = verify(CLAIM, built)
    assert searcher.queries == [sentence_query(CLAIM)]
    assert report.search is not None
    assert report.search.notices[0].startswith(LLM_LIMIT)


@pytest.mark.parametrize("web_search", ["ask", "deny"])
def test_ask_or_deny_turns_the_search_off_in_the_default_engine(web_search: str) -> None:
    config = Config(
        permissions=Permissions(web_search=web_search),  # type: ignore[arg-type]
        search=SearchConfig(provider="searxng", base_url="http://localhost:8888"),
    )
    with Engine.default(config, interactive=True, no_cache=True) as built:
        assert built.searcher is not None
        assert built.search is False
    allowed = Config(search=SearchConfig(provider="searxng", base_url="http://localhost:8888"))
    with Engine.default(allowed, interactive=False, no_cache=True) as built:
        assert built.search is True


# --- Amendment A: short text, and claims that are not in English --------------------

TURKISH = "OpenAI battı."  # noqa: RUF001 - a real Turkish letter, not a typo
ENGLISH = "OpenAI went bankrupt."
BANKRUPT = "OpenAI went bankrupt in 2026, the court said."


def test_a_one_line_post_is_searched_even_when_nothing_in_it_looks_checkworthy() -> None:
    searcher = StubSearcher()
    report = verify("OpenAI shut down", searching(searcher))
    assert searcher.queries == ["OpenAI shut down"]
    assert report.counts() == {Kind.NO_EVIDENCE: 1}


def test_without_a_judge_a_turkish_claim_is_reported_not_searched() -> None:
    searcher = StubSearcher({"OpenAI": [PAGE_A]})
    report = verify(TURKISH, searching(searcher))
    assert searcher.queries == []
    [item] = report.findings
    assert item.kind is Kind.UNVERIFIED
    assert item.state == LANGUAGE_UNSUPPORTED
    assert report.search is not None and report.search.searched == 0


def test_the_judge_translates_a_turkish_claim_and_the_report_shows_both() -> None:
    answer = json.dumps(
        {"items": [{"id": 0, "english": ENGLISH, "queries": ["OpenAI bankruptcy"]}]}
    )
    searcher = StubSearcher({"bankruptcy": [PAGE_A]})
    built = searching(
        searcher,
        pages={PAGE_A: f"{BANKRUPT} {FILLER}"},
        table={BANKRUPT: SUPPORTED_ROW},
    )
    built.judge = Judge(FakeJudgeClient([answer]))  # type: ignore[arg-type]
    report = verify(TURKISH, built)
    assert searcher.queries == ["OpenAI bankruptcy"]
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.claim is not None and found.claim.text == TURKISH
    assert f"checked as: {ENGLISH}" in found.detail
    assert found.detail[-1] == FOUND_BY_PROOFPATH


def test_a_judge_that_echoes_a_turkish_claim_leaves_it_language_unsupported() -> None:
    """Fix round 1 A: Turkish never reaches the English-only models as if translated."""
    answer = json.dumps(
        {"items": [{"id": 0, "english": TURKISH, "queries": ["OpenAI bankruptcy"]}]}
    )
    searcher = StubSearcher({"bankruptcy": [PAGE_A]})
    built = searching(searcher, pages={PAGE_A: f"{BANKRUPT} {FILLER}"})
    built.judge = Judge(FakeJudgeClient([answer]))  # type: ignore[arg-type]
    report = verify(TURKISH, built)
    [item] = report.findings
    assert item.kind is Kind.UNVERIFIED
    assert item.state == LANGUAGE_UNSUPPORTED
    assert searcher.queries == []


def test_a_translated_claim_the_search_finds_nothing_for_says_what_was_checked() -> None:
    """Fix round 1, item 1: ``_no_evidence`` names the English text too, not just the
    reader's own sentence, when the claim was translated."""
    answer = json.dumps(
        {"items": [{"id": 0, "english": ENGLISH, "queries": ["OpenAI bankruptcy"]}]}
    )
    built = searching(StubSearcher())
    built.judge = Judge(FakeJudgeClient([answer]))  # type: ignore[arg-type]
    report = verify(TURKISH, built)
    assert report.counts() == {Kind.NO_EVIDENCE: 1}
    finding = report.findings[0]
    assert finding.claim is not None and finding.claim.text == TURKISH
    assert f"checked as: {ENGLISH}" in finding.detail


def test_a_translated_claims_judgement_is_cached_under_the_english_hash(tmp_path: Path) -> None:
    """The judgement cache is keyed the same way the verdict cache is: a translated
    claim's opinion belongs to the English text the judge actually read, not to the
    reader's own sentence (item 5, controller decision 2026-09-28)."""
    translate = json.dumps(
        {"items": [{"id": 0, "english": ENGLISH, "queries": ["OpenAI bankruptcy"]}]}
    )
    # A low-tier REFUTED (spec section 9 step 8): a real passage, so it escalates.
    opinion = _opinion("c0", "REFUTED")
    searcher = StubSearcher({"bankruptcy": [PAGE_A]})
    with Cache(tmp_path / "c.sqlite3") as db:
        built = engine(
            fetcher=StubFetcher({PAGE_A: fetched(PAGE_A, f"{BANKRUPT} {FILLER}")}),
            embedder=WordEmbedder(),
            scorer=TableScorer({BANKRUPT: LOW_REFUTED_ROW}),
            cache=db,
        )
        built.searcher = searcher
        judge = Judge(FakeJudgeClient([translate, opinion]))  # type: ignore[arg-type]
        built.judge = judge
        ready = prepare(TURKISH, built)
        # The ``sources`` row a real fetch would have written (``tests.test_verify``'s
        # ``source_row``, inlined here: the searching stage's page is found by URL,
        # not by DOI, so ``StubFetcher`` never registers it itself).
        db.add_source(
            f"url:{PAGE_A}",
            scheme="url",
            title="",
            url=PAGE_A,
            text_kind="fulltext",
            raw_text=None,
        )
        report = decide_all(ready, built)

        not_supported = next(f for f in report.findings if f.kind is Kind.NOT_SUPPORTED)
        assert not_supported.source_id is not None
        assert not_supported.judge is not None

        stored = db.get_judgement(claim_hash(ENGLISH), not_supported.source_id, judge.name)
        assert stored is not None
        # Never under the reader's own sentence: that would reuse this opinion for a
        # differently translated run of the same Turkish claim later on.
        assert db.get_judgement(claim_hash(TURKISH), not_supported.source_id, judge.name) is None
