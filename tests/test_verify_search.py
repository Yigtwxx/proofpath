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
    ABSTRACT_BASIS,
    FOUND_BY_PROOFPATH,
    LANGUAGE_UNSUPPORTED,
    SEARCH_UNAVAILABLE,
    UNDERSPECIFIED,
    Kind,
)
from proofpath.resolve import ResolveResult, State
from proofpath.sarif import to_sarif
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
    StubResolver,
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


def test_a_verdict_from_a_short_found_page_says_it_rests_on_the_abstract() -> None:
    """Rule 6: a page under ``oa.FULLTEXT_MIN_WORDS`` is graded abstract, and the
    verdict row must say so itself, not only the source's abstract-only row."""
    report = verify(CLAIM, searching(StubSearcher({"ChatGPT": [PAGE_A]}), pages={PAGE_A: BACKING}))

    assert report.counts() == {Kind.ABSTRACT_ONLY: 1, Kind.EVIDENCE_FOUND: 1}
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.detail == (ABSTRACT_BASIS, FOUND_BY_PROOFPATH)
    assert found.verdict is not None and found.verdict.passage is not None


def test_a_verdict_from_a_full_text_found_page_has_no_abstract_line() -> None:
    report = verify(CLAIM, _supported_engine())

    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.detail == (FOUND_BY_PROOFPATH,)


# --- OPEN-ITEMS 19.1: a claim that states too little -----------------------------

WON = "openai won"
AWARD = "OpenAI's recognition with a 2026 Global Recognition Award was announced."


def test_a_found_verdict_on_an_underspecified_claim_says_so() -> None:
    """The live case: the passage entails "openai won", but so would any win. The
    verdict and its tier stand; the row says how little the claim pinned down."""
    report = verify(
        WON,
        searching(
            StubSearcher({"openai": [PAGE_A]}),
            pages={PAGE_A: f"{AWARD} {FILLER}"},
            table={AWARD: SUPPORTED_ROW},
        ),
    )
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.detail == (UNDERSPECIFIED, FOUND_BY_PROOFPATH)
    assert found.verdict is not None and found.tier == found.verdict.tier


def test_a_specific_claim_carries_no_underspecified_line() -> None:
    report = verify(CLAIM, _supported_engine())
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert UNDERSPECIFIED not in found.detail


def test_the_underspecified_line_sits_after_the_abstract_line_and_before_provenance() -> None:
    report = verify(
        WON,
        searching(
            StubSearcher({"openai": [PAGE_A]}),
            pages={PAGE_A: AWARD},
            table={AWARD: REFUTED_ROW},
        ),
    )
    refuted = next(f for f in report.findings if f.kind is Kind.NOT_SUPPORTED)
    assert refuted.detail == (ABSTRACT_BASIS, UNDERSPECIFIED, FOUND_BY_PROOFPATH)


def test_a_translated_claim_is_judged_underspecified_on_its_english_form() -> None:
    kazandi = "OpenAI kazandı."  # noqa: RUF001 - a real Turkish letter, not a typo
    answer = json.dumps({"items": [{"id": 0, "english": "OpenAI won.", "queries": ["OpenAI won"]}]})
    built = searching(
        StubSearcher({"OpenAI won": [PAGE_A]}),
        pages={PAGE_A: f"{AWARD} {FILLER}"},
        table={AWARD: SUPPORTED_ROW},
    )
    built.judge = Judge(FakeJudgeClient([answer]))  # type: ignore[arg-type]
    report = verify(kazandi, built)
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.detail == ("checked as: OpenAI won.", UNDERSPECIFIED, FOUND_BY_PROOFPATH)


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
    limited = JudgeUnavailable("HTTP 429 from https://api.test", status=429)
    built.judge = Judge(FakeJudgeClient([limited]))  # type: ignore[arg-type]
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


# --- found pages are read as pages (final review, Critical 1) ------------------------

# The three addresses the final review reproduced a false ghost with: each carries a
# DOI or an arXiv id, which used to send the found page to the bibliographic indexes.
IDENTIFIER_PAGES = (
    "https://pubs.acs.org/doi/10.1021/acs.est.0c01234",
    "https://doi.org/10.1021/acs.est.0c01234",
    "https://arxiv.org/abs/2106.09685",
)


def _found_identifier_pages(cache: Cache | None = None) -> tuple[Engine, StubResolver]:
    resolver = StubResolver()  # answers GHOST to anything it is asked
    fetcher = StubFetcher({url: fetched(url, f"{BACKING} {FILLER}") for url in IDENTIFIER_PAGES})
    built = engine(
        resolver=resolver,
        fetcher=fetcher,
        cache=cache,
        config=Config(search=SearchConfig(results_per_claim=3)),
        embedder=WordEmbedder(),
        scorer=TableScorer({BACKING: SUPPORTED_ROW}),
    )
    built.searcher = StubSearcher({"ChatGPT": list(IDENTIFIER_PAGES)})
    return built, resolver


def test_a_found_page_with_a_doi_or_arxiv_id_is_read_as_a_page_never_a_ghost() -> None:
    built, resolver = _found_identifier_pages()
    report = verify(CLAIM, built)

    assert resolver.resolved == []  # the indexes are never asked about a found page
    assert Kind.GHOST not in report.counts()
    assert [source.source_id for source in report.sources] == [
        f"url:{url}" for url in IDENTIFIER_PAGES
    ]
    assert [source.reference.origin for source in report.sources] == ["search"] * 3
    assert report.counts() == {Kind.EVIDENCE_FOUND: 3}


def test_a_found_page_never_reads_a_cached_resolution_of_the_same_address(
    tmp_path: Path,
) -> None:
    # A bibliography that printed the bare address resolved it through the indexes;
    # that record is about an entry, not about the page the search found.
    with Cache(tmp_path / "c.sqlite3") as db:
        for url in IDENTIFIER_PAGES:
            db.put_resolution(url, ResolveResult(State.GHOST, None, []))
        built, resolver = _found_identifier_pages(cache=db)
        # ``prepare`` only: the resolving stage is the one under test, and the model
        # half would want the ``sources`` rows a real fetch writes.
        ready = prepare(CLAIM, built)
    assert resolver.resolved == []
    assert [status.state for status in ready.sources.values()].count("GHOST REFERENCE") == 0
    assert [status.source_id for status in ready.sources.values()] == [
        f"url:{url}" for url in IDENTIFIER_PAGES
    ]


def test_a_found_page_leaves_no_resolution_behind_for_a_bibliography_entry(
    tmp_path: Path,
) -> None:
    with Cache(tmp_path / "c.sqlite3") as db:
        built, _ = _found_identifier_pages(cache=db)
        prepare(CLAIM, built)
        assert all(db.get_resolution(url) is None for url in IDENTIFIER_PAGES)


# --- findings name their claim, and the hint is true in every case (final review) -----


def test_a_search_unavailable_finding_carries_the_claim_it_was_about() -> None:
    report = verify(CLAIM, searching(StubSearcher(error=ProviderError("HTTP 503", 503, None))))
    [item] = report.findings
    assert item.state == SEARCH_UNAVAILABLE
    assert item.claim is not None and item.claim.text == CLAIM
    [result] = to_sarif(report, artifact="claim.txt")["runs"][0]["results"]
    assert result["properties"]["claim"] == CLAIM


def test_a_language_unsupported_finding_carries_the_claim_it_was_about() -> None:
    report = verify(TURKISH, searching(StubSearcher()))
    [item] = report.findings
    assert item.state == LANGUAGE_UNSUPPORTED
    assert item.claim is not None and item.claim.text == TURKISH


def test_the_language_hint_is_true_when_a_judge_was_on_and_did_not_translate() -> None:
    """``--judge`` was on here, so "--judge translates a claim" would be false: the
    hint says that no judge translated *this* claim, and how to have one do it."""
    answer = json.dumps(
        {"items": [{"id": 0, "english": TURKISH, "queries": ["OpenAI bankruptcy"]}]}
    )
    built = searching(StubSearcher())
    built.judge = Judge(FakeJudgeClient([answer]))  # type: ignore[arg-type]
    [item] = verify(TURKISH, built).findings
    [hint] = item.detail
    assert "no judge translated this claim" in hint
    assert "--judge" in hint
    assert "/config" not in hint  # a TUI setting that would not help a TUI run


# --- the banner at the top, and the exit code (final review, minors) -----------------


def _cli_run(
    built: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str
) -> tuple[int, str, str]:
    from typing import Any

    from typer.testing import CliRunner

    from proofpath import verify as verify_mod
    from proofpath.cli import app

    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))

    def fake_default(config: Config, **kwargs: Any) -> Engine:
        return built

    monkeypatch.setattr(verify_mod.Engine, "default", staticmethod(fake_default))
    target = tmp_path / "claim.txt"
    target.write_text(CLAIM, encoding="utf-8")
    result = CliRunner().invoke(app, ["check", str(target), "--out", str(tmp_path / "r.md"), *args])
    return result.exit_code, result.stdout, (tmp_path / "r.md").read_text(encoding="utf-8")


def _supported_engine() -> Engine:
    return searching(StubSearcher({"ChatGPT": [PAGE_A]}), pages={PAGE_A: f"{BACKING} {FILLER}"})


def test_the_experimental_banner_opens_the_terminal_and_markdown_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from proofpath.report import search_experimental

    _, stdout, markdown = _cli_run(_supported_engine(), tmp_path, monkeypatch)
    banner = search_experimental()
    lines = stdout.splitlines()
    at = next(i for i, line in enumerate(lines) if banner in line)
    finding = next(i for i, line in enumerate(lines) if "[evidence-found]" in line)
    assert at < finding  # before the first finding, not in the footer after it
    assert stdout.count(banner) == 1
    body = markdown.splitlines()
    assert banner in body[2]  # straight under the title
    assert markdown.count(banner) == 1


def test_a_run_whose_only_findings_are_supported_found_pages_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec section 13.3 says 0 is clean and is silent on search; a claim a found page
    supports is the searched run's clean answer, like a supported cited claim."""
    code, _, _ = _cli_run(_supported_engine(), tmp_path, monkeypatch)
    assert code == 0


@pytest.mark.parametrize(
    ("searcher", "pages", "table"),
    [
        (StubSearcher({"ChatGPT": [PAGE_A]}), {PAGE_A: f"{BACKING} {FILLER}"}, REFUTED_ROW),
        (StubSearcher(), {}, SUPPORTED_ROW),  # NO EVIDENCE FOUND
        (StubSearcher({"ChatGPT": [PAGE_A]}), {}, SUPPORTED_ROW),  # unread page
        (StubSearcher(error=ProviderError("HTTP 503", 503, None)), {}, SUPPORTED_ROW),
    ],
)
def test_every_other_searched_outcome_still_exits_one(
    searcher: StubSearcher,
    pages: dict[str, str],
    table: tuple[float, float, float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = searching(searcher, pages=pages, table={BACKING: table})
    code, _, _ = _cli_run(built, tmp_path, monkeypatch)
    assert code == 1


def test_a_rejected_search_key_is_reported_as_a_credential_not_an_outage() -> None:
    from proofpath.search import SearchKeyError

    rejected = SearchKeyError(
        "tavily rejected the key (HTTP 401); check TAVILY_API_KEY in .env", 401
    )
    report = verify(CLAIM, searching(StubSearcher(error=rejected)))
    [item] = report.findings
    assert item.kind is Kind.UNVERIFIED
    assert item.state == CREDENTIALS_MISSING
    assert item.state != SEARCH_UNAVAILABLE
    assert any("TAVILY_API_KEY" in line for line in item.detail)
    assert item.claim is not None and item.claim.text == CLAIM
    assert report.search is not None and report.search.searched == 0


# --- round 2: a capped search is not a clean run (rule 6) ------------------------------

SEVEN_CLAIMS = " ".join(f"OpenAI shut down ChatGPT in March 202{n} after a vote." for n in range(7))


def test_a_capped_search_whose_searched_claims_are_all_supported_still_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seven check-worthy claims, five searched (the cap), every one of them backed:
    the two never searched are unchecked, and exit 0 would call the text clean."""
    backings = {
        f"OpenAI shut down ChatGPT in March 202{n} after a vote.": SUPPORTED_ROW for n in range(7)
    }
    pages = {f"https://news.test/{n}": f"{text} {FILLER}" for n, text in enumerate(backings)}
    searcher = StubSearcher({f"202{n}": [f"https://news.test/{n}"] for n in range(7)})
    built = searching(searcher, pages=pages, table=backings)
    report = verify(SEVEN_CLAIMS, built)
    assert report.search is not None
    assert (report.search.eligible, report.search.searched) == (7, 5)
    assert set(report.counts()) == {Kind.EVIDENCE_FOUND}
    assert report.exit_code() == 1


def test_a_found_page_is_fetched_without_the_contact_address_and_a_cited_one_with_it() -> None:
    """Round 2, privacy: the user chose the pages their text cites, not the ones the
    search found, so only the latter are fetched anonymously."""
    built = searching(StubSearcher({"ChatGPT": [PAGE_A]}), pages={PAGE_A: f"{BACKING} {FILLER}"})
    verify(CLAIM, built)
    assert built.fetcher.anonymous == [PAGE_A]  # type: ignore[attr-defined]

    cited = "https://cited.test/report"
    text = f"{CLAIM} [1]\n\nReferences\n\n[1] {cited}\n"
    built = engine(
        # The indexes do not cover it, so it is read from the address it prints.
        resolver=StubResolver({"cited.test": ResolveResult(State.NOT_INDEXED, None, [])}),
        fetcher=StubFetcher({cited: fetched(cited, f"{BACKING} {FILLER}")}),
        embedder=WordEmbedder(),
        scorer=TableScorer({BACKING: SUPPORTED_ROW}),
    )
    verify(text, built)
    assert cited in built.fetcher.calls  # type: ignore[attr-defined]
    assert built.fetcher.anonymous == []  # type: ignore[attr-defined]
