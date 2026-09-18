"""The orchestrator end to end, offline: ``prepare()``, ``decide_all()``, ``verify()``.

Every provider is a stub with scripted answers and both models are fakes, so the
whole pipeline is exercised without a socket and without an ONNX session. The tests
are written against the honesty states the stages must carry verbatim (product
rule 2), against the events a front end needs to draw progress, and against the
cache contract that makes a second run of the same document free (spec section 11).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from proofpath import ingest as ingest_mod
from proofpath import oa, pipeline, retrieval
from proofpath.browser import ConsentGate
from proofpath.cache import Cache
from proofpath.config import Config, Permissions
from proofpath.document import Claim, Document, Locator, PageError
from proofpath.entailment import Scorer
from proofpath.events import (
    Cancelled,
    Emitted,
    Event,
    Note,
    Progress,
    StageEnd,
    StageStart,
)
from proofpath.fetch import Fetched, FetchStats, Outcome
from proofpath.judge import Completion, Judge, JudgeCost, JudgeUnavailable
from proofpath.models import Label, Passage, Verdict
from proofpath.oa import ABSTRACT_ONLY, Attempt, Evidence, Location
from proofpath.pipeline import Thresholds
from proofpath.polite import ProviderError
from proofpath.report import (
    JUDGE_DETAIL_PREFIX,
    ClaimResult,
    Kind,
    Report,
    judge_detail,
    judge_unavailable,
)
from proofpath.resolve import Candidate, ResolveResult, Retraction, State
from proofpath.retrieval import Embedder
from proofpath.settings_hints import BROWSER_SETTING
from proofpath.ui import json_text
from proofpath.verify import (
    CACHE_BY,
    CLAIMS,
    FETCHING,
    JUDGING,
    LOADING_MODELS,
    MARKERS_SET_ASIDE,
    NO_IDENTIFIER,
    NO_MODEL,
    NO_SOURCE_HINT,
    NO_SOURCE_IN_TEXT,
    NO_TEXT,
    NOT_ATTEMPTED,
    NOTHING_TO_VERIFY,
    PARSING,
    POST_READER,
    RESOLVERS_BY,
    RESOLVING,
    RETRACTION_UNAVAILABLE,
    RETRACTIONS,
    SUMMARISING,
    VERIFYING,
    Engine,
    Prepared,
    _escalates,
    _parser_for,
    decide_all,
    prepare,
    target_document,
    verify,
)
from tests.fakes import TableScorer, WordEmbedder

DOI = "10.1038/s41586-021-03819-2"
ARXIV = "2103.00020"
PAGE = "https://example.test/report.html"


def words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


FULLTEXT = words(oa.FULLTEXT_MIN_WORDS + 10)
SHORT = words(40)


# --- stubs -------------------------------------------------------------------


class StubResolver:
    """``resolve.Resolver``'s two methods, keyed by a substring of the raw entry."""

    def __init__(
        self,
        results: dict[str, ResolveResult] | None = None,
        retractions: dict[str, Retraction] | None = None,
    ) -> None:
        self.results = results or {}
        self.retractions = retractions or {}
        self.resolved: list[str] = []
        self.asked_retraction: list[str] = []

    def resolve(self, raw: str) -> ResolveResult:
        self.resolved.append(raw)
        for key, result in self.results.items():
            if key in raw:
                return result
        return ResolveResult(State.GHOST, None, [], notes=["no record anywhere"])

    def retraction(self, doi: str) -> Retraction | None:
        self.asked_retraction.append(doi)
        return self.retractions.get(doi)


class StubOpenAccess:
    """``oa.OpenAccess.fetch``, keyed by the identifier it is called with."""

    def __init__(self, evidence: dict[str, Evidence] | None = None) -> None:
        self.evidence = evidence or {}
        self.calls: list[tuple[str | None, str | None]] = []
        self.on_call: Callable[[], None] | None = None

    def fetch(self, doi: str | None, arxiv_id: str | None = None) -> Evidence:
        self.calls.append((doi, arxiv_id))
        if self.on_call is not None:
            self.on_call()
        key = doi or arxiv_id or ""
        return self.evidence.get(key, none_evidence(Outcome.UNREACHABLE))


class StubFetcher:
    """``fetch.Fetcher``'s URL half, plus the two network attributes prepare reads."""

    def __init__(
        self,
        pages: dict[str, Fetched] | None = None,
        *,
        network_allowed: bool = True,
        network_note: str = "",
    ) -> None:
        self.pages = pages or {}
        self.network_allowed = network_allowed
        self.network_note = network_note
        self.calls: list[str] = []
        self.asked: list[tuple[str, str]] = []  # (url, text_kind) as the ladder saw it
        # (url, counts_as_source, use_cache): how the target page was asked for.
        self.options: list[tuple[str, bool, bool]] = []
        self.on_call: Callable[[], None] | None = None

    def fetch(
        self,
        url: str,
        *,
        text_kind: str = "fulltext",
        counts_as_source: bool = True,
        use_cache: bool = True,
    ) -> Fetched:
        self.calls.append(url)
        self.asked.append((url, text_kind))
        self.options.append((url, counts_as_source, use_cache))
        if self.on_call is not None:
            self.on_call()
        return self.pages.get(url, unreachable(url))

    def summary(self) -> FetchStats:
        return FetchStats(counts={}, browser_skipped=0)


def fetched(url: str, text: str, *, step: int = 1, from_cache: bool = False) -> Fetched:
    return Fetched(
        url=url,
        final_url=url,
        step=step,
        outcome=Outcome.OK,
        status=200,
        content_type="text/html",
        kind="html",
        body=b"",
        text=text,
        notes=[],
        from_cache=from_cache,
    )


def unreachable(url: str) -> Fetched:
    return Fetched(
        url=url,
        final_url=url,
        step=1,
        outcome=Outcome.UNREACHABLE,
        status=None,
        content_type="",
        kind="other",
        body=b"",
        text="",
        notes=["step 1 httpx: no response"],
        from_cache=False,
    )


def fulltext_evidence(url: str = "https://arxiv.test/paper.pdf") -> Evidence:
    return Evidence(
        kind="fulltext",
        state="",
        text=FULLTEXT,
        url=url,
        source="arxiv",
        attempts=[Attempt(Location("arxiv", url, "pdf"), Outcome.OK, 1, len(FULLTEXT.split()))],
        notes=[],
    )


def abstract_evidence(url: str = "https://api.test/abstract") -> Evidence:
    return Evidence(
        kind="abstract",
        state=ABSTRACT_ONLY,
        text=SHORT,
        url=url,
        source="abstract:s2",
        attempts=[],
        notes=["no open-access full text"],
    )


def none_evidence(outcome: Outcome) -> Evidence:
    return Evidence(
        kind="none",
        state=outcome.value,
        text="",
        url="",
        source="",
        attempts=[],
        notes=[f"every location {outcome.value}"],
    )


def no_model() -> NoReturn:
    raise AssertionError("no model may be built on this path")


def engine(
    *,
    resolver: StubResolver | None = None,
    chain: StubOpenAccess | None = None,
    fetcher: StubFetcher | None = None,
    config: Config | None = None,
    embedder: Embedder | None = None,
    scorer: Scorer | None = None,
    cache: Cache | None = None,
) -> Engine:
    """An engine whose every part is a stub. The two factories raise unless given."""
    return Engine(
        config=config or Config(),
        cache=cache,
        resolver=resolver or StubResolver(),
        fetcher=fetcher or StubFetcher(),
        oa=chain or StubOpenAccess(),
        gate=ConsentGate("deny", interactive=False),
        embedder=no_model if embedder is None else (lambda: embedder),
        scorer=no_model if scorer is None else (lambda: scorer),
    )


def draft(body: str, entries: Sequence[str] = ()) -> str:
    """The document as pasted text, with a bibliography when ``entries`` is given."""
    if not entries:
        return body
    printed = "\n".join(f"[{number}] {raw}" for number, raw in enumerate(entries, start=1))
    return f"{body}\n\nReferences\n\n{printed}\n"


def drafted(body: str, entries: Sequence[str] = ()) -> tuple[str, Document]:
    """The pasted text ``prepare`` is given, and the document it will build from it."""
    text = draft(body, entries)
    return text, target_document(text)


def collected() -> tuple[list[Event], Callable[[Event], None]]:
    events: list[Event] = []
    return events, events.append


REAL = "Vaswani, A. Attention is all you need. NeurIPS, 2017."
GHOSTLY = "Nobody, N. A study that was never written. Journal of Nothing, 2019."


def resolved(doi: str = DOI) -> ResolveResult:
    best = Candidate(
        doi=doi, title="Attention is all you need", first_author="Vaswani", year=2017,
        venue="NeurIPS", provider="crossref",
    )  # fmt: skip
    return ResolveResult(State.RESOLVED, best, [best])


# --- target_document ---------------------------------------------------------


def test_target_document_reads_an_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "draft.md"
    path.write_text("A claim [1].\n", encoding="utf-8")
    assert target_document(path).name == "draft.md"
    assert target_document(str(path)).name == "draft.md"


def test_target_document_treats_other_strings_as_pasted_text() -> None:
    doc = target_document("The effect was large [1].")
    assert doc.name == "pasted text"
    assert doc.kind == "text"
    assert target_document("hello", name="a post").name == "a post"


# --- a page as the target (a bare address on a host that is not a platform) ----

PAGE_URL = "https://example.test/news/story"
STORY_LINK = "https://other.test/report"
PAGE_HTML = (
    b"<html><body><nav><a href='/'>Home</a></nav><article>"
    b"<p>The count rose 17% last year, see <a href='/tally'>the tally</a>.</p>"
    b"<p>A separate body <a href='https://other.test/report'>disagreed</a>.</p>"
    b"<p>Nothing is linked from here.</p>"
    b"</article></body></html>"
)


def page(url: str = PAGE_URL, body: bytes = PAGE_HTML, *, outcome: Outcome = Outcome.OK) -> Fetched:
    ok = outcome is Outcome.OK
    return Fetched(
        url=url,
        final_url=url,
        step=1 if ok else 4,
        outcome=outcome,
        status=200 if ok else 403,
        content_type="text/html" if ok else "",
        kind="html" if ok else "other",
        body=body if ok else b"",
        text="",
        notes=["step 1 httpx: HTTP 200"]
        if ok
        else ["step 1 httpx: HTTP 403 (blocked)", "step 4 wayback: no snapshot"],
    )


def test_a_bare_page_address_is_read_as_the_document_up_the_ladder() -> None:
    fetcher = StubFetcher({PAGE_URL: page()})

    ready = prepare(PAGE_URL, engine(fetcher=fetcher))

    # The page itself is not a source, and the cache is climbed past: a cached copy
    # is text alone, and the page's links are its bibliography.
    assert fetcher.options[0] == (PAGE_URL, False, False)
    assert ready.document.name == PAGE_URL
    assert ready.document.kind == "linked"
    assert [r.raw for r in ready.document.references] == [
        "https://example.test/tally",
        STORY_LINK,
    ]
    assert ready.stages[0].name == PARSING
    assert ready.stages[0].by == "fetch ladder"
    assert ready.stages[0].summary == "1 pages, 2 refs"
    # Each paragraph's sentences cite that paragraph's links and nothing else.
    by_paragraph = {claim.paragraph: claim.cited_refs for claim in ready.claims.claims}
    assert by_paragraph == {0: (1,), 1: (2,)}
    # And the links were then fetched as sources, the ordinary way.
    assert fetcher.options[1:] == [
        ("https://example.test/tally", True, True),
        (STORY_LINK, True, True),
    ]


def test_a_page_the_ladder_could_not_read_is_refused_in_the_ladders_words() -> None:
    for outcome in (
        Outcome.BLOCKED,
        Outcome.BLOCKED_NO_BROWSER,
        Outcome.BLOCKED_ROBOTS,
        Outcome.UNREACHABLE,
        Outcome.UNAVAILABLE,
    ):
        fetcher = StubFetcher({PAGE_URL: page(outcome=outcome)})
        with pytest.raises(ingest_mod.IngestError) as refused:
            prepare(PAGE_URL, engine(fetcher=fetcher))
        assert PAGE_URL in str(refused.value), outcome
        assert outcome.value in str(refused.value), outcome
        # The step that decided it, not the archive miss every wall ends on.
        assert "step 1 httpx: HTTP 403 (blocked)" in str(refused.value), outcome
        assert "wayback" not in str(refused.value), outcome
        # Only the browser refusal has a setting to point at.
        hinted = f"config set {BROWSER_SETTING}" in str(refused.value)
        assert hinted == (outcome is Outcome.BLOCKED_NO_BROWSER), outcome


def test_a_denied_run_reads_no_page_at_all_and_says_so() -> None:
    fetcher = StubFetcher({PAGE_URL: page()}, network_allowed=False, network_note="network: deny")

    with pytest.raises(ingest_mod.IngestError) as refused:
        prepare(PAGE_URL, engine(fetcher=fetcher))

    assert fetcher.calls == []
    assert Outcome.NETWORK_DENIED.value in str(refused.value)


def test_a_page_with_no_links_is_reported_rather_than_read_as_clean() -> None:
    body = b"<article><p>Nothing is linked from here.</p></article>"
    ready = prepare(PAGE_URL, engine(fetcher=StubFetcher({PAGE_URL: page(body=body)})))

    assert ready.document.kind == "page"
    assert ready.document.references == ()
    assert ready.claims.claims == ()
    titles = [f.title for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert titles == ["page carries no links; nothing to verify against"]


def test_a_bare_claim_with_no_source_is_reported_rather_than_read_as_clean() -> None:
    """A claim pasted without any address or citation has nothing behind it to check.
    The run says so and says what to paste instead, rather than ending on an empty
    stage list that looks exactly like a clean one (product rule 6)."""
    ready = prepare("ChatGPT was shut down yesterday.", engine())

    assert ready.document.kind == "text"
    assert ready.document.references == ()
    assert ready.claims.claims == ()
    parse_errors = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert [f.title for f in parse_errors] == [NO_SOURCE_IN_TEXT]
    assert parse_errors[0].detail == (NO_SOURCE_HINT,)


def test_a_pasted_claim_ending_in_a_file_suffix_still_gets_the_paste_hint() -> None:
    """``_parser_for`` names a parser by suffix before anything is read; the hint
    must not depend on that guess, only on the target not being a file."""
    ready = prepare("ChatGPT was shut down, see notes.pdf", engine())

    parse_errors = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert [f.title for f in parse_errors] == [NO_SOURCE_IN_TEXT]
    assert parse_errors[0].detail == (NO_SOURCE_HINT,)


def test_a_file_that_cites_nothing_is_reported_without_the_paste_hint(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("A note that cites nothing at all.\n", encoding="utf-8")
    ready = prepare(path, engine())

    parse_errors = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert [f.title for f in parse_errors] == ["text cites nothing; nothing to verify against"]
    assert parse_errors[0].detail == ()


def test_a_page_that_prints_a_bibliography_is_paired_by_number() -> None:
    body = (
        b"<article><p>The effect was large [1].</p><h2>References</h2>"
        b"<ol><li>Vaswani, A. Attention is all you need. NeurIPS, 2017.</li></ol></article>"
    )
    resolver = StubResolver({"Vaswani": resolved()})
    ready = prepare(
        PAGE_URL, engine(resolver=resolver, fetcher=StubFetcher({PAGE_URL: page(body=body)}))
    )

    assert ready.document.kind == "page"
    assert [r.number for r in ready.document.references] == [1]
    assert [claim.cited_refs for claim in ready.claims.claims] == [(1,)]


def test_a_bracketed_number_in_a_page_does_not_cost_it_its_links() -> None:
    """Only a printed reference list makes a page cite by number. A "[1]" in the prose
    of a page with no list is prose, and the page still cites by linking."""
    body = b"<article><p>The effect was large [1], see <a href='/x'>x</a>.</p></article>"
    ready = prepare(PAGE_URL, engine(fetcher=StubFetcher({PAGE_URL: page(body=body)})))

    assert ready.document.kind == "linked"
    assert [r.raw for r in ready.document.references] == ["https://example.test/x"]
    assert ready.claims.unresolved == ()
    assert [claim.cited_refs for claim in ready.claims.claims] == [(1,)]
    # ... and says so, because a "Sources" list the finder missed looks the same.
    titles = [f.title for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert titles == [MARKERS_SET_ASIDE.format(count=1)]


def test_a_mastodon_status_on_an_unlisted_instance_is_still_a_post_not_a_page() -> None:
    """``is_social`` knows the listed platforms; a Mastodon status is known by its
    shape on any instance, and must keep being read as a post (task 10.3)."""
    url = "https://hachyderm.io/users/someone/statuses/109384756"
    fetcher = StubFetcher({url: page(url)}, network_allowed=False, network_note="network: deny")

    with pytest.raises(ingest_mod.IngestError) as refused:
        prepare(url, engine(fetcher=fetcher))

    # Refused by the post reader's first rule, before the ladder was ever asked.
    assert fetcher.calls == []
    assert Outcome.NETWORK_DENIED.value in str(refused.value)
    assert _parser_for(url) == POST_READER
    assert _parser_for("https://mastodon.social/explore") == POST_READER
    assert _parser_for(PAGE_URL) == "fetch ladder"


def test_a_lemmy_post_on_a_listed_instance_is_a_post_where_a_blog_post_is_a_page() -> None:
    """Lemmy is known by its host -- the ``lemmy.`` prefix or one of the big instances
    -- never by ``/post/<n>`` alone, which is how half the web addresses an article."""
    assert _parser_for("https://lemmy.world/post/4242") == POST_READER
    assert _parser_for("https://sh.itjust.works/comment/100") == POST_READER
    assert _parser_for("https://lemmy.world/c/science") == POST_READER
    assert _parser_for("https://lobste.rs/s/abc123") == POST_READER
    assert _parser_for("https://example.test/post/4242") == "fetch ladder"


def test_pasted_text_with_links_is_still_read_by_the_text_parser() -> None:
    """The ``linked`` kind is shared with a page; the stage line is not."""
    ready = prepare(f"A claim, see {STORY_LINK} for the tally.", engine())
    assert ready.document.kind == "linked"
    assert (ready.stages[0].by, ready.stages[0].name) == ("text", PARSING)


def test_a_page_address_with_a_space_around_it_is_still_a_page() -> None:
    fetcher = StubFetcher({PAGE_URL: page()})
    ready = prepare(f"  {PAGE_URL}\n", engine(fetcher=fetcher))
    assert ready.document.name == PAGE_URL


# --- stage 1: parsing --------------------------------------------------------


def test_page_error_becomes_a_parse_error_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    _, doc = drafted("The effect was large [1].", [REAL])
    broken = Document(
        name=doc.name,
        kind="pdf",
        paragraphs=doc.paragraphs,
        references=doc.references,
        pages=3,
        errors=(PageError(page=2, detail="damaged xref"),),
    )
    monkeypatch.setattr("proofpath.verify.target_document", lambda *a, **k: broken)
    ready = prepare("x.pdf", engine(resolver=StubResolver({"Vaswani": resolved()})))

    parse = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert len(parse) == 1
    assert parse[0].level == "warning"
    assert parse[0].locator.page == 2
    assert parse[0].locator.line == 1
    assert parse[0].title == "page could not be parsed"
    assert parse[0].detail == ("damaged xref",)
    assert ready.stages[0].name == PARSING
    assert ready.stages[0].by == "pymupdf"
    assert ready.stages[0].summary == "3 pages, 1 refs"


def test_a_document_with_no_text_at_all_cannot_look_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanned = Document(
        name="scan.pdf",
        kind="pdf",
        paragraphs=(),
        references=(),
        pages=4,
        errors=(PageError(page=1, detail="image only"),),
    )
    monkeypatch.setattr("proofpath.verify.target_document", lambda *a, **k: scanned)
    ready = prepare("scan.pdf", engine())

    titles = [f.title for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert "no text could be extracted from the document" in titles
    assert len(titles) == 2  # the per-page error and the whole-document one


# --- stage 2: claims ---------------------------------------------------------


def test_an_author_year_marker_no_entry_answers_is_reported_at_its_locator() -> None:
    # The only entry is Vaswani 2017, so "(Smith et al., 2020)" names nothing this
    # document lists. v0.2 pairs the style, so the marker is reported as unresolved --
    # a statement about this bibliography, not about what the tool can read.
    text, doc = drafted("The effect was large (Smith et al., 2020) and real.", [REAL])
    ready = prepare(text, engine())

    assert [f for f in ready.findings if f.kind is Kind.UNSUPPORTED_STYLE] == []
    unresolved = [f for f in ready.findings if f.kind is Kind.UNRESOLVED_MARKER]
    assert len(unresolved) == 1
    marker = ready.claims.unresolved[0]
    assert unresolved[0].locator == doc.locate(marker.paragraph, marker.start)
    assert unresolved[0].detail == ("(Smith et al., 2020)",)
    assert unresolved[0].claim is None


def test_an_author_year_marker_is_paired_with_the_entry_it_names() -> None:
    text = draft("The attention result was large (Vaswani, 2017) and real.", [REAL])
    ready = prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()})))

    assert ready.claims.unresolved == ()
    assert [claim.cited_refs for claim in ready.claims.claims] == [(1,)]
    assert ready.claims.claims[0].text == "The attention result was large and real."


def test_unresolved_marker_is_reported_with_its_text() -> None:
    text = draft("A claim about nothing [9] and one about something [1].", [REAL])
    ready = prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()})))

    unresolved = [f for f in ready.findings if f.kind is Kind.UNRESOLVED_MARKER]
    assert [f.detail for f in unresolved] == [("[9]",)]
    assert unresolved[0].title == "citation marker names no bibliography entry"
    claims_stage = ready.stages[1]
    assert (claims_stage.name, claims_stage.by) == (CLAIMS, "rules")
    assert claims_stage.summary == "2 citations, 1 unresolved"


# --- stage 3: resolving ------------------------------------------------------


def test_ghost_and_resolved_doi() -> None:
    text, doc = drafted(
        "A real claim [1]. A fabricated one [2].",
        [REAL, GHOSTLY],
    )
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    stub = StubResolver({"Vaswani": resolved()})
    ready = prepare(text, engine(resolver=stub, chain=chain))

    ghost = [f for f in ready.findings if f.kind is Kind.GHOST]
    assert len(ghost) == 1
    assert ghost[0].level == "error"
    assert ghost[0].state == State.GHOST.value
    assert ghost[0].title == "cited source does not exist"
    assert ghost[0].locator == doc.references[1].locator
    assert ghost[0].detail == ("no record anywhere",)

    assert set(ready.sources) == {1, 2}
    assert ready.sources[2].state == State.GHOST.value
    assert ready.sources[2].source_id is None
    assert ready.sources[1].source_id == f"doi:{DOI}"
    assert ready.sources[1].text_kind == "fulltext"
    assert ready.sources[1].state == ""
    assert ready.texts == {f"doi:{DOI}": FULLTEXT}
    assert chain.calls == [(DOI, None)]

    resolving = ready.stages[2]
    assert (resolving.name, resolving.by) == (RESOLVING, "Crossref, Semantic Scholar")
    assert resolving.summary == "1 ok, 0 amb, 1 ghost"


def test_ambiguous_lists_the_candidates_it_could_not_choose_between() -> None:
    text = draft("A claim [1].", [REAL])
    one = Candidate("10.1/a", "A study of things", "Smith", 2019, "J", "crossref")
    two = Candidate("10.1/b", "A study of other things", "Smith", 2019, "J", "openalex")
    stub = StubResolver({"Vaswani": ResolveResult(State.AMBIGUOUS, None, [one, two])})
    ready = prepare(text, engine(resolver=stub))

    ambiguous = [f for f in ready.findings if f.kind is Kind.AMBIGUOUS]
    assert len(ambiguous) == 1
    assert ambiguous[0].level == "warning"
    assert ambiguous[0].state == State.AMBIGUOUS.value
    assert ambiguous[0].detail == (one.title, two.title)
    assert ready.sources[1].state == State.AMBIGUOUS.value


def test_provider_unavailable_is_its_own_state() -> None:
    text = draft("A claim [1].", [REAL])
    stub = StubResolver({"Vaswani": ResolveResult(State.UNAVAILABLE, None, [])})
    ready = prepare(text, engine(resolver=stub))

    findings = [f for f in ready.findings if f.kind is Kind.PROVIDER_UNAVAILABLE]
    assert [f.state for f in findings] == [State.UNAVAILABLE.value]
    assert findings[0].title == "resolver unavailable"
    assert ready.stages[2].summary == "0 ok, 0 amb, 0 ghost, 1 unavailable"


def test_not_indexed_with_a_url_is_fetched_by_url() -> None:
    entry = f"WHO. Air quality guidelines. {PAGE} Accessed 2024."
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"WHO": ResolveResult(State.NOT_INDEXED, None, [])})
    fetcher = StubFetcher({PAGE: fetched(PAGE, FULLTEXT)})
    ready = prepare(text, engine(resolver=stub, fetcher=fetcher))

    assert fetcher.calls == [PAGE]
    assert ready.sources[1].source_id == f"url:{PAGE}"
    assert ready.sources[1].text_kind == "fulltext"
    assert ready.sources[1].url == PAGE
    assert ready.texts == {f"url:{PAGE}": FULLTEXT}
    assert not [f for f in ready.findings if f.kind is Kind.UNVERIFIED]


def test_not_indexed_without_a_url_keeps_the_exact_state() -> None:
    text = draft("A claim [1].", ["A committee report with no link at all."])
    stub = StubResolver({"committee": ResolveResult(State.NOT_INDEXED, None, [])})
    ready = prepare(text, engine(resolver=stub))

    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [State.NOT_INDEXED.value]
    assert unverified[0].title == "source is not in bibliographic indexes"
    assert ready.sources[1].state == State.NOT_INDEXED.value


def test_a_resolved_reference_with_no_identifier_says_so_exactly() -> None:
    text = draft("A claim [1].", ["Smith, J. A book with no DOI. Press, 1998."])
    best = Candidate("", "A book with no DOI", "Smith", 1998, "Press", "openlibrary")
    stub = StubResolver({"Smith": ResolveResult(State.RESOLVED_LOW, best, [best])})
    ready = prepare(text, engine(resolver=stub))

    assert ready.sources[1].source_id is None
    assert ready.sources[1].state == NO_IDENTIFIER
    assert NO_IDENTIFIER.startswith("UNVERIFIED")
    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [NO_IDENTIFIER]


def test_an_arxiv_id_in_the_raw_entry_becomes_the_source_id() -> None:
    entry = f"Radford, A. Learning transferable visual models. arXiv:{ARXIV}, 2021."
    text = draft("A claim [1].", [entry])
    best = Candidate("", "Learning transferable visual models", "Radford", 2021, "", "arxiv")
    stub = StubResolver({"Radford": ResolveResult(State.RESOLVED, best, [best])})
    chain = StubOpenAccess({ARXIV: abstract_evidence()})
    ready = prepare(text, engine(resolver=stub, chain=chain))

    assert chain.calls == [(None, ARXIV)]
    assert ready.sources[1].source_id == f"arxiv:{ARXIV}"
    assert ready.sources[1].text_kind == "abstract"
    assert ready.sources[1].state == ABSTRACT_ONLY
    only = [f for f in ready.findings if f.kind is Kind.ABSTRACT_ONLY]
    assert [f.level for f in only] == ["note"]
    assert only[0].state == ABSTRACT_ONLY


# --- stage 4: retractions ----------------------------------------------------


def test_a_retracted_source_is_reported_with_its_date() -> None:
    text = draft("A claim [1].", [REAL])
    notice = Retraction(source="retraction-watch", date="2021-05-04", notice_doi="10.1/r",
                        label="Retraction notice")  # fmt: skip
    stub = StubResolver({"Vaswani": resolved()}, retractions={DOI: notice})
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    ready = prepare(text, engine(resolver=stub, chain=chain))

    assert stub.asked_retraction == [DOI]
    retracted = [f for f in ready.findings if f.kind is Kind.RETRACTED]
    assert len(retracted) == 1
    assert retracted[0].level == "warning"
    assert retracted[0].title == "cited source was retracted 2021-05-04"
    assert retracted[0].detail == ("Retraction notice",)
    assert ready.sources[1].retraction == notice
    assert ready.stages[3].name == RETRACTIONS
    assert ready.stages[3].by == "Retraction Watch"
    assert ready.stages[3].summary == "1 retracted"


def test_no_retractions_says_none() -> None:
    text = draft("A claim [1].", [REAL])
    stub = StubResolver({"Vaswani": resolved()})
    ready = prepare(text, engine(resolver=stub, chain=StubOpenAccess({DOI: fulltext_evidence()})))
    assert ready.stages[3].summary == "none"


# --- stage 5: fetching -------------------------------------------------------


def test_a_source_with_no_text_carries_the_fetch_outcome_verbatim() -> None:
    text = draft("A claim [1].", [REAL])
    stub = StubResolver({"Vaswani": resolved()})
    chain = StubOpenAccess({DOI: none_evidence(Outcome.BLOCKED_ROBOTS)})
    ready = prepare(text, engine(resolver=stub, chain=chain))

    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [Outcome.BLOCKED_ROBOTS.value]
    assert unverified[0].level == "warning"
    assert ready.sources[1].state == Outcome.BLOCKED_ROBOTS.value
    assert ready.sources[1].text_kind == "none"
    assert ready.texts == {}
    assert ready.stages[4].name == FETCHING
    assert ready.stages[4].summary == "0 full text, 0 abstract, 1 unverified"


def test_a_short_page_is_an_abstract_not_full_text() -> None:
    entry = f"A blog post. {PAGE}"
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"blog": ResolveResult(State.NOT_INDEXED, None, [])})
    fetcher = StubFetcher({PAGE: fetched(PAGE, SHORT)})
    ready = prepare(text, engine(resolver=stub, fetcher=fetcher))

    assert ready.sources[1].text_kind == "abstract"
    assert ready.sources[1].state == ABSTRACT_ONLY
    assert ready.texts == {f"url:{PAGE}": SHORT}


def test_network_denied_calls_nothing_and_says_so_on_every_source() -> None:
    text, doc = drafted("A claim [1]. Another [2].", [REAL, GHOSTLY])
    note = (
        "network: not permitted (permissions.network = deny)"
        " — permissions.network allow turns it on"
    )
    fetcher = StubFetcher(network_allowed=False, network_note=note)
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    stub = StubResolver({"Vaswani": resolved()})
    events, listener = collected()
    config = Config(permissions=Permissions(network="deny"))
    ready = prepare(
        text,
        engine(resolver=stub, chain=chain, fetcher=fetcher, config=config),
        on_event=listener,
    )

    # Nothing was asked of anyone: not the ladder, not the chain, not the resolver.
    assert chain.calls == []
    assert fetcher.calls == []
    assert stub.resolved == []
    assert stub.asked_retraction == []
    # Every cited reference is unverified, in the one wording section 15 gives it.
    assert [status.state for status in ready.sources.values()] == [
        Outcome.NETWORK_DENIED.value,
        Outcome.NETWORK_DENIED.value,
    ]
    assert all(status.resolve is None for status in ready.sources.values())
    assert all(status.source_id is None for status in ready.sources.values())
    assert all(status.text_kind == "none" for status in ready.sources.values())
    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [Outcome.NETWORK_DENIED.value] * 2
    assert [f.locator for f in unverified] == [ref.locator for ref in doc.references]
    # A stage that was not attempted says so; "0 ghost" would be a claim about the
    # references, and nothing was looked at (product rule 2).
    assert ready.stages[2].summary == NOT_ATTEMPTED
    assert ready.stages[3].summary == NOT_ATTEMPTED
    # Short, because it sits in the elided summary column; the setting that turns the
    # stage on rides on the once-per-run denied note instead (permission hints).
    assert NOT_ATTEMPTED == "not attempted (network not permitted)"
    assert Note(note) in events
    assert note.endswith(" — permissions.network allow turns it on")
    assert [stage.name for stage in ready.stages] == [
        PARSING,
        CLAIMS,
        RESOLVING,
        RETRACTIONS,
        FETCHING,
    ]
    assert Note(note) in events
    assert ready.texts == {}


# --- events, cancellation, engine wiring -------------------------------------


def test_the_event_stream_names_every_stage_in_order() -> None:
    text = draft("A claim [1].", [REAL])
    stub = StubResolver({"Vaswani": resolved()})
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    events, listener = collected()
    prepare(text, engine(resolver=stub, chain=chain), on_event=listener)

    order = [
        (type(event).__name__, event.name)
        for event in events
        if isinstance(event, (StageStart, StageEnd))
    ]
    assert order == [
        ("StageStart", PARSING),
        ("StageEnd", PARSING),
        ("StageStart", CLAIMS),
        ("StageEnd", CLAIMS),
        ("StageStart", RESOLVING),
        ("StageEnd", RESOLVING),
        ("StageStart", RETRACTIONS),
        ("StageEnd", RETRACTIONS),
        ("StageStart", FETCHING),
        ("StageEnd", FETCHING),
    ]
    progress = [event for event in events if isinstance(event, Progress)]
    assert (RESOLVING, 1, 1) in [(p.name, p.done, p.total) for p in progress]
    assert all(stage.elapsed >= 0.0 for stage in [*events] if isinstance(stage, StageEnd))


def test_every_finding_is_emitted_as_it_is_found() -> None:
    text = draft("A claim [1].", [GHOSTLY])
    events, listener = collected()
    ready = prepare(text, engine(), on_event=listener)
    assert [event.finding for event in events if isinstance(event, Emitted)] == ready.findings


def test_cancel_set_inside_a_fetch_stops_after_that_unit() -> None:
    text = draft("A claim [1]. Another [2].", [REAL, REAL.replace("Vaswani", "Ba")])
    cancel = threading.Event()
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    chain.on_call = cancel.set
    stub = StubResolver({"Vaswani": resolved(), "Ba,": resolved("10.1/second")})
    with pytest.raises(Cancelled):
        prepare(text, engine(resolver=stub, chain=chain), cancel=cancel)
    assert len(chain.calls) == 1


def test_cancel_before_the_first_stage_raises_immediately() -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        prepare("A claim [1].", engine(), cancel=cancel)


def test_engine_default_wires_the_real_parts_and_closes_them() -> None:
    built = Engine.default(Config(), interactive=False, no_cache=True)
    assert built.cache is None
    assert built.k == 1
    assert isinstance(built.device, str) and built.device
    with built:
        pass  # the context manager closes every client it opened
    assert built.close() is None  # closing twice must not raise


def test_engine_default_never_builds_a_model() -> None:
    with Engine.default(Config(), interactive=False, no_cache=True) as built:
        assert callable(built.embedder)
        assert callable(built.scorer)


def test_prepare_is_typed_as_it_is_declared() -> None:
    text, doc = drafted("A claim [1].", [REAL])
    ready = prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()})))
    assert isinstance(ready, Prepared)
    assert ready.document == doc
    assert isinstance(ready.started, float)
    assert len(ready.stages) == 5


def test_a_citation_with_no_bibliography_at_all_is_still_reported() -> None:
    # A pasted paragraph carries no reference list, so nothing can answer "[1]".
    ready = prepare("The effect was large [1].", engine())

    unresolved = [f for f in ready.findings if f.kind is Kind.UNRESOLVED_MARKER]
    assert len(unresolved) == 1
    assert unresolved[0].detail == ("the document prints no bibliography",)
    assert unresolved[0].locator == ready.claims.claims[0].locator
    assert ready.sources == {}
    assert ready.stages[2].summary == "0 ok, 0 amb, 0 ghost"


def test_an_unreachable_page_keeps_the_ladder_s_own_outcome() -> None:
    entry = f"A blog post. {PAGE}"
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"blog": ResolveResult(State.NOT_INDEXED, None, [])})
    ready = prepare(text, engine(resolver=stub, fetcher=StubFetcher()))

    assert ready.sources[1].state == Outcome.UNREACHABLE.value
    assert ready.sources[1].text_kind == "none"
    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [Outcome.UNREACHABLE.value]


def test_a_page_reached_with_nothing_on_it_is_not_a_page_that_was_blocked() -> None:
    entry = f"A blog post. {PAGE}"
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"blog": ResolveResult(State.NOT_INDEXED, None, [])})
    fetcher = StubFetcher({PAGE: fetched(PAGE, "")})
    ready = prepare(text, engine(resolver=stub, fetcher=fetcher))

    assert ready.sources[1].state == NO_TEXT
    assert NO_TEXT.startswith("UNVERIFIED")
    assert ready.sources[1].state != Outcome.UNREACHABLE.value
    assert ready.texts == {}


def test_a_state_outside_the_unverified_family_is_carried_as_a_note() -> None:
    text = draft("A claim [1].", [REAL])
    odd = Evidence(kind="none", state="ok", text="", url="", source="", attempts=[], notes=[])
    stub = StubResolver({"Vaswani": resolved()})
    ready = prepare(text, engine(resolver=stub, chain=StubOpenAccess({DOI: odd})))

    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [NO_TEXT]
    assert unverified[0].detail == ("ok",)  # the producer's own word is never lost


def test_name_labels_a_target_that_is_not_a_file(tmp_path: Path) -> None:
    # A caller reading from a pipe knows what to call the document; without this
    # every piped run would be reported as "pasted text".
    assert prepare("A claim [1].", engine(), name="stdin").document.name == "stdin"
    assert verify("A claim [1].", engine(), name="stdin").document.name == "stdin"
    # A real file keeps its own name: the caller's label is a fallback, not an override.
    path = tmp_path / "paper.md"
    path.write_text(draft("A claim [1].", [GHOSTLY]), encoding="utf-8")
    assert prepare(path, engine(), name="stdin").document.name == "paper.md"


def test_prepare_reads_a_file_path(tmp_path: Path) -> None:
    path = tmp_path / "paper.md"
    path.write_text(draft("A claim [1].", [GHOSTLY]), encoding="utf-8")
    ready = prepare(path, engine())

    assert ready.document.name == "paper.md"
    assert [f.kind for f in ready.findings] == [Kind.GHOST]


def test_engine_default_opens_and_closes_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))
    with Engine.default(Config(), interactive=False) as built:
        assert built.cache is not None
        assert built.cache.path.is_file()
    # Closed: the connection is gone, so any use of it now raises.
    assert built.cache is not None
    with pytest.raises(sqlite3.ProgrammingError):
        built.cache.text_kind("doi:10.1/x")


def test_the_consent_gate_s_install_log_reaches_the_report_as_notes() -> None:
    text = draft("A claim [1].", [REAL])
    stub = StubResolver({"Vaswani": resolved()})
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    built = engine(resolver=stub, chain=chain)
    # What the ladder's step 3 would have written while this source was fetched.
    chain.on_call = lambda: built.gate.install_log.append("browser fetch failed: TimeoutError")
    events, listener = collected()
    prepare(text, built, on_event=listener)

    assert Note("browser fetch failed: TimeoutError") in events


def test_a_url_page_that_grades_as_an_abstract_is_refiled_in_the_cache(
    tmp_path: Path,
) -> None:
    # The ladder files a page as the "fulltext" it was asked for; only the word
    # count afterwards says what the url: row really holds (mirrors oa._climb_all).
    entry = f"A blog post. {PAGE}"
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"blog": ResolveResult(State.NOT_INDEXED, None, [])})
    fetcher = StubFetcher({PAGE: fetched(PAGE, SHORT, step=2)})
    with Cache(tmp_path / "c.sqlite3") as cache:
        cache.add_source(
            f"url:{PAGE}",
            scheme="url",
            title="",
            url=PAGE,
            text_kind="fulltext",
            raw_text=None,
        )
        built = engine(resolver=stub, fetcher=fetcher)
        built.cache = cache
        ready = prepare(text, built)

        assert fetcher.asked == [(PAGE, "fulltext")]
        assert ready.sources[1].text_kind == "abstract"
        assert cache.text_kind(f"url:{PAGE}") == "abstract"


def test_a_full_text_url_page_is_left_as_it_was_filed(tmp_path: Path) -> None:
    entry = f"A long report. {PAGE}"
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"report": ResolveResult(State.NOT_INDEXED, None, [])})
    fetcher = StubFetcher({PAGE: fetched(PAGE, FULLTEXT)})
    with Cache(tmp_path / "c.sqlite3") as cache:
        cache.add_source(
            f"url:{PAGE}",
            scheme="url",
            title="",
            url=PAGE,
            text_kind="fulltext",
            raw_text=None,
        )
        built = engine(resolver=stub, fetcher=fetcher)
        built.cache = cache
        prepare(text, built)

        assert cache.text_kind(f"url:{PAGE}") == "fulltext"


def test_a_cache_hit_keeps_its_step_and_says_it_came_from_the_cache() -> None:
    entry = f"A blog post. {PAGE}"
    text = draft("A claim [1].", [entry])
    stub = StubResolver({"blog": ResolveResult(State.NOT_INDEXED, None, [])})
    fetcher = StubFetcher({PAGE: fetched(PAGE, FULLTEXT, step=0, from_cache=True)})
    ready = prepare(text, engine(resolver=stub, fetcher=fetcher))

    assert ready.sources[1].fetch_step == 0
    assert ready.sources[1].from_cache is True
    assert ready.stages[4].by == "cache"


def test_an_open_access_hit_reports_the_step_its_winning_location_took() -> None:
    text = draft("A claim [1].", [REAL])
    stub = StubResolver({"Vaswani": resolved()})
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    ready = prepare(text, engine(resolver=stub, chain=chain))

    assert ready.sources[1].fetch_step == 1
    assert ready.sources[1].from_cache is False
    assert ready.stages[4].by == "arXiv"


# --- stage 6: verifying ------------------------------------------------------
#
# Both models are fakes: ``WordEmbedder`` ranks by word overlap so a claim retrieves
# the source sentence it shares its words with, and ``TableScorer`` raises on a
# premise it does not know, so a test whose retrieval went astray fails loudly
# instead of quietly scoring the wrong sentence.

SUPPORTING = "Transformers improved translation quality on every benchmark."
FIGURE = "The method yields a 4-8% speedup in throughput."
FILLER = "The rest of the paper is about tokenisers."
PAPER = f"{SUPPORTING} {FIGURE} {FILLER}"

SUPPORTED_ROW = (0.95, 0.02, 0.03)
REFUTED_ROW = (0.05, 0.92, 0.03)
NEI_ROW = (0.30, 0.20, 0.50)


def text_evidence(text: str, url: str = "https://arxiv.test/paper.pdf") -> Evidence:
    """Full text the stub chain hands back, worded by the test that needs it."""
    return Evidence(
        kind="fulltext",
        state="",
        text=text,
        url=url,
        source="arxiv",
        attempts=[Attempt(Location("arxiv", url, "pdf"), Outcome.OK, 1, len(text.split()))],
        notes=[],
    )


def source_row(cache: Cache, source_id: str, text: str) -> None:
    """The ``sources`` row the fetch ladder would have written for a cached source."""
    cache.add_source(source_id, scheme="doi", title="", url="", text_kind="fulltext", raw_text=text)


def paper_engine(
    *,
    table: dict[str, tuple[float, float, float]] | None = None,
    embedder: WordEmbedder | None = None,
    scorer: TableScorer | None = None,
    text: str = PAPER,
    cache: Cache | None = None,
) -> Engine:
    """One resolved DOI whose full text is ``text``, plus the two fakes."""
    return engine(
        resolver=StubResolver({"Vaswani": resolved()}),
        chain=StubOpenAccess({DOI: text_evidence(text)}),
        embedder=embedder or WordEmbedder(),
        scorer=scorer or TableScorer(table or {SUPPORTING: SUPPORTED_ROW}),
        cache=cache,
    )


BODY = (
    "Transformers improved translation quality [1]. The method yields a 40% speedup [1]. "
    "A fabricated finding [2]. Nothing further was measured."
)
# The same paragraph citing one reference. The closing sentence is not decoration: a
# marker that ends its paragraph cites the whole paragraph (spec section 9), and these
# two claims are meant to be one marker each.
ONE_SOURCE_BODY = (
    "Transformers improved translation quality [1]. The method yields a 40% speedup [1]. "
    "Nothing further was measured."
)


def test_verify_reports_a_supported_claim_a_numeric_mismatch_and_a_blocked_source() -> None:
    text = draft(BODY, [REAL, GHOSTLY])
    built = engine(
        resolver=StubResolver({"Vaswani": resolved(), "Nobody": resolved("10.5/blocked")}),
        chain=StubOpenAccess(
            {DOI: text_evidence(PAPER), "10.5/blocked": none_evidence(Outcome.BLOCKED_ROBOTS)}
        ),
        embedder=WordEmbedder(),
        scorer=TableScorer({SUPPORTING: SUPPORTED_ROW}),
    )
    report = verify(text, built)

    assert report.counts() == {Kind.NUMERIC_MISMATCH: 1, Kind.UNVERIFIED: 1}
    numeric = next(f for f in report.findings if f.kind is Kind.NUMERIC_MISMATCH)
    assert numeric.level == "error"
    assert numeric.title == "claim contradicts the cited source"
    # Product rule 1: the assertion travels with the passage it rests on.
    assert numeric.verdict is not None and numeric.verdict.passage is not None
    assert numeric.verdict.passage.text == FIGURE
    assert numeric.detail == ("numeric mismatch: claim says 40%, source says 4-8%",)
    assert numeric.claim is not None and numeric.claim.text == "The method yields a 40% speedup."
    assert numeric.tier == "high"

    blocked = next(f for f in report.findings if f.kind is Kind.UNVERIFIED)
    assert blocked.state == Outcome.BLOCKED_ROBOTS.value

    # A SUPPORTED claim leaves no finding but is still a result: --format json lists it.
    labels = [(item.reference, item.verdict.label) for item in report.results]
    assert labels == [(1, Label.SUPPORTED), (1, Label.REFUTED)]
    assert all(item.source_id == f"doi:{DOI}" for item in report.results)
    assert all(item.from_cache is False for item in report.results)

    assert (report.coverage.references, report.coverage.fulltext) == (2, 1)
    assert (report.coverage.abstract, report.coverage.unverified) == (0, 1)
    assert report.coverage.reasons == {Outcome.BLOCKED_ROBOTS.value: 1}
    assert report.coverage.network_denied is False
    assert report.claims == 3 and report.markers == 3
    assert report.api_calls == 0 and report.elapsed >= 0.0
    assert report.cancelled is False
    assert report.exit_code() == 1


def test_the_claim_citing_an_unreadable_source_gets_no_result() -> None:
    text = draft(BODY, [REAL, GHOSTLY])
    built = engine(
        resolver=StubResolver({"Vaswani": resolved(), "Nobody": resolved("10.5/blocked")}),
        chain=StubOpenAccess(
            {DOI: text_evidence(PAPER), "10.5/blocked": none_evidence(Outcome.BLOCKED_ROBOTS)}
        ),
        embedder=WordEmbedder(),
        scorer=TableScorer({SUPPORTING: SUPPORTED_ROW}),
    )
    report = verify(text, built)

    # Its source-level finding already says why; a second, model-shaped "we could not
    # tell" would report the same gap twice (product rule 2).
    assert [item.claim.text for item in report.results] == [
        "Transformers improved translation quality.",
        "The method yields a 40% speedup.",
    ]


def test_the_verifying_stage_names_the_device_and_counts_the_verdicts() -> None:
    text = draft(ONE_SOURCE_BODY, [REAL])
    report = verify(text, paper_engine())

    stage = report.stages[-1]
    assert (stage.name, stage.by) == (VERIFYING, "cpu")
    assert stage.summary == "2 claims: 1 supported, 1 not supported, 0 NEI"
    assert stage.elapsed >= 0.0
    assert report.models == {
        "nli": "table",
        "embedder": "words",
        "device": "cpu",
        # The calibrated defaults, spelled out rather than derived: a recalibration
        # has to come past this line and past docs/eval/2026-09-12-tiers.md together.
        "thresholds": "decide=0.45;high=0.99933;medium=0.457948",
    }


def test_a_run_carries_the_tier_note_of_the_thresholds_it_used() -> None:
    text = draft(ONE_SOURCE_BODY, [REAL])
    # An unreachable high cut is a property of this run's calibration, so the report
    # says so itself rather than the renderer asking a module-level default.
    no_high_tier = paper_engine()
    no_high_tier.thresholds = Thresholds(decide=0.45, high=1.0, medium=0.5)
    assert verify(text, no_high_tier).tier_note == pipeline.NO_HIGH_TIER
    assert verify(text, paper_engine()).tier_note == ""


def test_the_verifying_stage_emits_its_own_events() -> None:
    text = draft(ONE_SOURCE_BODY, [REAL])
    events, listener = collected()
    verify(text, paper_engine(), on_event=listener)

    names = [e.name for e in events if isinstance(e, (StageStart, StageEnd))]
    assert names[-2:] == [VERIFYING, VERIFYING]
    # Transient: the TUI takes it back once the models are in, so a finished run
    # never still reads "loading models …" under its Verifying stage.
    assert Note(LOADING_MODELS, transient=True) in events
    progress = [
        (e.done, e.total) for e in events if isinstance(e, Progress) and e.name == VERIFYING
    ]
    assert progress == [(1, 2), (2, 2)]


def test_no_source_text_loads_no_model_at_all() -> None:
    # ``engine()``'s factories raise, so a single call to either fails this test.
    report = verify(draft("A claim [1].", [GHOSTLY]), engine())

    assert report.results == ()
    assert report.stages[-1].name == VERIFYING
    assert report.stages[-1].summary == NOTHING_TO_VERIFY
    assert report.models["nli"] == NO_MODEL
    assert report.models["embedder"] == NO_MODEL
    assert report.models["device"] == "cpu"


# --- the cache ---------------------------------------------------------------


def test_a_second_run_over_the_same_document_calls_neither_model(tmp_path: Path) -> None:
    text = draft(ONE_SOURCE_BODY, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        first_embedder, first_scorer = WordEmbedder(), TableScorer({SUPPORTING: SUPPORTED_ROW})
        built = paper_engine(embedder=first_embedder, scorer=first_scorer, cache=cache)
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", PAPER)
        first = decide_all(ready, built)
        assert first_embedder.calls and first_scorer.seen

        again_embedder = WordEmbedder()
        again_scorer = TableScorer({})  # any score() call would raise KeyError
        # The second run is a second process in real life: it releases the models it
        # built and the next ``decide_all`` asks the factories again.
        built.close()
        built.embedder = lambda: again_embedder
        built.scorer = lambda: again_scorer
        second = decide_all(ready, built)

        assert again_embedder.calls == []
        assert again_scorer.seen == []
        assert [item.from_cache for item in second.results] == [True, True]
        assert [item.verdict for item in second.results] == [item.verdict for item in first.results]
        assert second.counts() == first.counts()


def test_changed_source_text_is_chunked_again(tmp_path: Path) -> None:
    text = draft(ONE_SOURCE_BODY, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        built = paper_engine(cache=cache)
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", PAPER)
        decide_all(ready, built)
        stored = cache.get_chunks(f"doi:{DOI}", "words")
        assert stored is not None and len(stored[0]) == 3

        # The source was re-fetched and now says something else: chunks cut from the
        # old text must not be quoted back as if they came from this one.
        grown = f"{PAPER} A new final sentence was added."
        ready.texts[f"doi:{DOI}"] = grown
        after_embedder = WordEmbedder()
        built.close()  # as above: a fresh run builds a fresh embedder
        built.embedder = lambda: after_embedder
        decide_all(ready, built)

        assert after_embedder.calls, "changed text must be embedded again"
        regrown = cache.get_chunks(f"doi:{DOI}", "words")
        assert regrown is not None and len(regrown[0]) == 4


REVISED_FIGURE = "The method yields a 7-9% speedup in throughput."


def test_a_source_whose_text_changed_is_judged_again_not_quoted_from_the_old_one(
    tmp_path: Path,
) -> None:
    # The verdict key is (claim_hash, source_id, model_id) and holds no digest, so the
    # cache itself has to notice: chunks cut from other text drop that source's
    # verdicts, and the claim is decided again against what the source says now.
    text = draft(ONE_SOURCE_BODY, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        built = paper_engine(cache=cache)
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", PAPER)
        first = decide_all(ready, built)
        was = next(item for item in first.results if item.verdict.label is Label.REFUTED)
        assert was.verdict.passage is not None and was.verdict.passage.text == FIGURE

        ready.texts[f"doi:{DOI}"] = f"{SUPPORTING} {REVISED_FIGURE} {FILLER}"
        built.close()  # as above: a fresh run builds fresh models
        built.embedder = lambda: WordEmbedder()
        built.scorer = lambda: TableScorer({SUPPORTING: SUPPORTED_ROW})
        second = decide_all(ready, built)

        now = next(item for item in second.results if item.verdict.label is Label.REFUTED)
        assert now.verdict.passage is not None and now.verdict.passage.text == REVISED_FIGURE
        assert now.verdict.reason == "numeric mismatch: claim says 40%, source says 7-9%"
        assert all(item.from_cache is False for item in second.results)


FOUR = (
    "Alpha rose sharply [1]. Beta fell slowly [1]. Gamma stayed flat [1]. "
    "Delta wobbled often [1]. Nothing further was measured."
)
FOUR_SENTENCES = (
    "Alpha rose sharply in the trial.",
    "Beta fell slowly in the trial.",
    "Gamma stayed flat in the trial.",
    "Delta wobbled often in the trial.",
)
FOUR_SOURCE = " ".join(FOUR_SENTENCES)


def test_a_cancelled_run_leaves_only_the_verdicts_it_finished(tmp_path: Path) -> None:
    cancel = threading.Event()
    scored = 0

    def stop_after_three() -> None:
        nonlocal scored
        scored += 1
        if scored == 3:
            cancel.set()

    scorer = TableScorer(dict.fromkeys(FOUR_SENTENCES, NEI_ROW), on_score=stop_after_three)
    text = draft(FOUR, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        built = paper_engine(scorer=scorer, text=FOUR_SOURCE, cache=cache)
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", FOUR_SOURCE)
        assert len(ready.claims.claims) == 4

        with pytest.raises(Cancelled) as stopped:
            decide_all(ready, built, cancel=cancel)

        # Every write is its own transaction, so the rows that are there are whole.
        detail = cache.detail(f"doi:{DOI}")
        assert detail is not None and len(detail.verdicts) == 3

        # A stopped run is incomplete, not lost: what it did decide comes with it,
        # and says of itself that it stopped (product rule 6).
        partial = stopped.value.report
        assert partial is not None
        assert partial.cancelled is True
        assert len(partial.results) == 3
        assert partial.exit_code() == 1
        assert partial.coverage.references == 1
        assert partial.stages[-1].name == VERIFYING


def test_prepare_raises_without_a_report_because_there_is_none_yet() -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled) as stopped:
        prepare("A claim [1].", engine(), cancel=cancel)
    assert stopped.value.report is None


# --- paragraph-scoped citations ----------------------------------------------

SCOPED_BODY = (
    "Transformers improved translation quality. The model was trained on a large cluster. "
    "Everything else stayed the same [1]."
)
SCOPED_SOURCE = (
    "Transformers improved translation quality on every benchmark. "
    "The model was trained on a large cluster of machines. "
    "Everything else stayed the same except the optimiser."
)
SCOPED_TABLE = {
    "Transformers improved translation quality on every benchmark.": SUPPORTED_ROW,
    "The model was trained on a large cluster of machines.": NEI_ROW,
    "Everything else stayed the same except the optimiser.": REFUTED_ROW,
}


def test_a_paragraph_scoped_citation_groups_its_sentences_under_one_finding() -> None:
    text = draft(SCOPED_BODY, [REAL])
    built = paper_engine(table=SCOPED_TABLE, text=SCOPED_SOURCE)
    report = verify(text, built)

    # Each sentence is judged on its own text, so none of them speaks for the paragraph.
    assert len(report.results) == 3
    assert all(item.claim.paragraph_scoped for item in report.results)
    assert [item.verdict.label for item in report.results] == [
        Label.SUPPORTED,
        Label.NEI,
        Label.REFUTED,
    ]

    scoped = [f for f in report.findings if f.kind is Kind.PARAGRAPH_SCOPED]
    assert len(scoped) == 1
    assert scoped[0].level == "note"
    assert scoped[0].title == "citation supports a paragraph, not one sentence"
    assert scoped[0].detail == ("3 sentences: 1 supported, 1 NEI, 1 not supported",)
    # The group finding explains; the members carry the evidence (controller note).
    assert scoped[0].verdict is None and scoped[0].claim is None
    assert scoped[0].group is not None

    members = [f for f in report.findings if f.kind is not Kind.PARAGRAPH_SCOPED]
    assert {f.kind for f in members} == {Kind.NEI, Kind.NOT_SUPPORTED}
    assert all(f.group == scoped[0].group for f in members)
    assert scoped[0].locator == report.results[0].claim.locator

    unsupported = next(f for f in members if f.kind is Kind.NOT_SUPPORTED)
    assert unsupported.title == "claim is not supported by the cited source"
    assert unsupported.verdict is not None and unsupported.verdict.passage is not None
    nei = next(f for f in members if f.kind is Kind.NEI)
    assert nei.title == "source neither supports nor contradicts the claim"
    assert nei.verdict is not None and nei.verdict.passage is None


TWO_SOURCE_BODY = "Alpha rose sharply. Beta fell slowly [1, 2]."
FIRST_TRIAL = "Alpha rose sharply in the first trial. Beta fell slowly in the first trial."
SECOND_TRIAL = "Alpha rose sharply in the second trial. Beta fell slowly in the second trial."


def test_a_group_citing_two_references_counts_sentences_and_names_both_sources() -> None:
    # Four verdicts, but only two sentences: the group note counts what the author
    # wrote, not how many times the pipeline had to ask.
    text = draft(TWO_SOURCE_BODY, [REAL, GHOSTLY])
    table = dict.fromkeys(
        [*retrieval.split_sentences(FIRST_TRIAL), *retrieval.split_sentences(SECOND_TRIAL)],
        NEI_ROW,
    )
    built = engine(
        resolver=StubResolver({"Vaswani": resolved(), "Nobody": resolved("10.5/second")}),
        chain=StubOpenAccess(
            {DOI: text_evidence(FIRST_TRIAL), "10.5/second": text_evidence(SECOND_TRIAL)}
        ),
        embedder=WordEmbedder(),
        scorer=TableScorer(table),
    )
    report = verify(text, built)

    assert len(report.results) == 4
    assert {item.reference for item in report.results} == {1, 2}
    scoped = [f for f in report.findings if f.kind is Kind.PARAGRAPH_SCOPED]
    assert len(scoped) == 1
    assert scoped[0].detail == ("2 sentences against 2 sources: 4 NEI",)


def test_findings_come_back_in_document_order() -> None:
    text = draft(BODY, [REAL, GHOSTLY])
    built = engine(
        resolver=StubResolver({"Vaswani": resolved(), "Nobody": resolved("10.5/blocked")}),
        chain=StubOpenAccess(
            {DOI: text_evidence(PAPER), "10.5/blocked": none_evidence(Outcome.BLOCKED_ROBOTS)}
        ),
        embedder=WordEmbedder(),
        scorer=TableScorer({SUPPORTING: SUPPORTED_ROW}),
    )
    report = verify(text, built)

    positions = [
        (f.locator.page or 0, f.locator.line, f.locator.column, f.kind.value)
        for f in report.findings
    ]
    assert positions == sorted(positions)


# --- model lifetime: the engine owns the ONNX sessions (spec section 13.3) -----


class ClosingModel:
    """A model that records its own release, like the ONNX sessions do at runtime."""

    name = "closing"
    dim = 2

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def embed(self, texts):
        return WordEmbedder().embed(texts)

    def score(self, pairs):
        return TableScorer({SUPPORTING: SUPPORTED_ROW}).score(pairs)


def test_engine_builds_each_model_once_and_keeps_it() -> None:
    built = 0

    def factory() -> Embedder:
        nonlocal built
        built += 1
        return ClosingModel()

    eng = engine(embedder=None, scorer=None)
    eng.embedder = factory
    assert eng.get_embedder() is eng.get_embedder()
    assert built == 1


def test_engine_close_releases_the_models_and_forgets_them() -> None:
    embedder_model = ClosingModel()
    scorer_model = ClosingModel()
    eng = engine(embedder=embedder_model, scorer=scorer_model)
    assert eng.get_embedder() is embedder_model
    assert eng.get_scorer() is scorer_model

    eng.close()
    assert embedder_model.closed == 1
    assert scorer_model.closed == 1
    # Closing twice must neither raise nor release a session a second time.
    eng.close()
    assert embedder_model.closed == 1
    assert scorer_model.closed == 1


def test_engine_close_survives_a_model_without_close() -> None:
    # The Embedder/Scorer protocols do not require close(); a stub stays a stub.
    eng = engine(embedder=WordEmbedder(), scorer=TableScorer({SUPPORTING: SUPPORTED_ROW}))
    eng.get_embedder()
    eng.get_scorer()
    assert eng.close() is None


def test_decide_all_leaves_the_models_on_the_engine() -> None:
    """Nothing may hold an ONNX session after the run: the engine is the only owner."""
    embedder_model = ClosingModel()
    scorer_model = ClosingModel()
    text = draft(ONE_SOURCE_BODY, [REAL])
    eng = paper_engine(embedder=embedder_model, scorer=scorer_model)
    report = decide_all(prepare(text, eng), eng)
    assert report.models["embedder"] == "closing"
    eng.close()
    assert embedder_model.closed == 1
    assert scorer_model.closed == 1


# --- the resolution and retraction cache (OPEN-ITEMS 10.8, 11.3) --------------


def test_a_second_prepare_of_the_same_document_makes_no_resolver_calls(tmp_path: Path) -> None:
    """The point of the cache: a warm run is an offline run, resolving included."""
    text, _ = drafted("The effect was large [1].", [REAL])
    retractions = {DOI: Retraction("retraction-watch", "2021-03-01", "10.1/n", "Retraction")}
    with Cache(tmp_path / "c.sqlite3") as db:
        first = StubResolver({"Vaswani": resolved()}, retractions)
        prepare(text, engine(resolver=first, cache=db))
        assert first.resolved and first.asked_retraction == [DOI]

        second = StubResolver({"Vaswani": resolved()}, retractions)
        ready = prepare(text, engine(resolver=second, cache=db))

    assert second.resolved == []
    assert second.asked_retraction == []
    status = ready.sources[1]
    assert status.resolve is not None and status.resolve.state is State.RESOLVED
    assert status.resolve_from_cache is True
    # The retraction survived the round trip, so the finding is still raised.
    assert status.retraction is not None and status.retraction.date == "2021-03-01"
    assert [f.kind for f in ready.findings if f.kind is Kind.RETRACTED] == [Kind.RETRACTED]


def test_a_cached_run_attributes_the_two_stages_to_the_cache(tmp_path: Path) -> None:
    text, _ = drafted("The effect was large [1].", [REAL])
    with Cache(tmp_path / "c.sqlite3") as db:
        prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()}), cache=db))
        ready = prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()}), cache=db))
    by = {stage.name: stage.by for stage in ready.stages}
    assert by[RESOLVING] == CACHE_BY
    assert by[RETRACTIONS] == CACHE_BY


def test_a_partly_cached_run_still_names_the_providers(tmp_path: Path) -> None:
    """One reference out of two came from the wire, so the stage did consult them."""
    first_text, _ = drafted("A [1].", [REAL])
    both_text, _ = drafted("A [1]. B [2].", [REAL, GHOSTLY])
    with Cache(tmp_path / "c.sqlite3") as db:
        prepare(first_text, engine(resolver=StubResolver({"Vaswani": resolved()}), cache=db))
        resolver = StubResolver({"Vaswani": resolved()})
        ready = prepare(both_text, engine(resolver=resolver, cache=db))
    assert resolver.resolved == [f"[2] {GHOSTLY}"]
    assert {stage.name: stage.by for stage in ready.stages}[RESOLVING] == RESOLVERS_BY


def test_an_unavailable_resolution_is_never_served_from_the_cache(tmp_path: Path) -> None:
    """Product rule 2: an outage is not knowledge, so the next run asks again."""
    text, _ = drafted("The effect was large [1].", [REAL])
    down = ResolveResult(State.UNAVAILABLE, None, [], notes=["crossref unavailable (HTTP 503)"])
    with Cache(tmp_path / "c.sqlite3") as db:
        prepare(text, engine(resolver=StubResolver({"Vaswani": down}), cache=db))
        second = StubResolver({"Vaswani": resolved()})
        ready = prepare(text, engine(resolver=second, cache=db))
    assert second.resolved == [f"[1] {REAL}"]
    status = ready.sources[1]
    assert status.resolve is not None and status.resolve.state is State.RESOLVED
    assert status.resolve_from_cache is False


def test_no_cache_bypasses_the_resolution_cache(tmp_path: Path) -> None:
    text, _ = drafted("The effect was large [1].", [REAL])
    with Cache(tmp_path / "c.sqlite3") as db:
        prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()}), cache=db))
    # ``--no-cache`` builds an engine with no cache at all; nothing is consulted.
    resolver = StubResolver({"Vaswani": resolved()})
    prepare(text, engine(resolver=resolver, cache=None))
    assert resolver.resolved == [f"[1] {REAL}"]


def test_a_denied_network_never_writes_a_resolution(tmp_path: Path) -> None:
    """Nothing was asked of anyone, so there is nothing to remember (rule 2)."""
    text, _ = drafted("The effect was large [1].", [REAL])
    fetcher = StubFetcher(network_allowed=False, network_note="network = deny")
    with Cache(tmp_path / "c.sqlite3") as db:
        prepare(text, engine(resolver=StubResolver(), fetcher=fetcher, cache=db))
        assert db.lookup_counts() == (0, 0)


# --- fix round 1: a retraction outage is never cached as "not retracted" ----------


class DownRetractions(StubResolver):
    """A resolver whose retraction check has lost every provider."""

    def retraction(self, doi: str) -> Retraction | None:
        self.asked_retraction.append(doi)
        raise ProviderError("retraction check unavailable (ConnectError)")


def test_a_retraction_outage_is_reported_and_never_cached(tmp_path: Path) -> None:
    """Product rule 2: nobody answered, so nothing is known -- and a 7-day cached
    "not retracted" would freeze that outage into a clean record."""
    text, _ = drafted("The effect was large [1].", [REAL])
    with Cache(tmp_path / "c.sqlite3") as db:
        down = DownRetractions({"Vaswani": resolved()})
        ready = prepare(text, engine(resolver=down, cache=db))
        assert down.asked_retraction == [DOI]
        assert db.get_retraction(DOI) is None  # nothing was stored

        # The next run asks again rather than serving the outage.
        again = DownRetractions({"Vaswani": resolved()})
        prepare(text, engine(resolver=again, cache=db))
        assert again.asked_retraction == [DOI]

    status = ready.sources[1]
    assert status.retraction is None
    assert any(note.startswith(RETRACTION_UNAVAILABLE) for note in status.notes)
    summary = {stage.name: stage.summary for stage in ready.stages}
    assert summary[RETRACTIONS] == "1 unavailable"  # never the green "none"
    assert not [f for f in ready.findings if f.kind is Kind.RETRACTED]


def test_a_clean_retraction_check_is_cached_and_says_none(tmp_path: Path) -> None:
    text, _ = drafted("The effect was large [1].", [REAL])
    with Cache(tmp_path / "c.sqlite3") as db:
        ready = prepare(text, engine(resolver=StubResolver({"Vaswani": resolved()}), cache=db))
        hit = db.get_retraction(DOI)
        assert hit is not None and hit.notice is None
    assert {stage.name: stage.summary for stage in ready.stages}[RETRACTIONS] == "none"


def test_a_retraction_hit_is_cached_and_counted(tmp_path: Path) -> None:
    text, _ = drafted("The effect was large [1].", [REAL])
    notice = Retraction("retraction-watch", "2021-03-01", "10.1/n", "Retraction")
    with Cache(tmp_path / "c.sqlite3") as db:
        ready = prepare(
            text,
            engine(resolver=StubResolver({"Vaswani": resolved()}, {DOI: notice}), cache=db),
        )
        hit = db.get_retraction(DOI)
        assert hit is not None and hit.notice == notice
    assert {stage.name: stage.summary for stage in ready.stages}[RETRACTIONS] == "1 retracted"


def test_an_outage_alongside_a_notice_is_counted_apart(tmp_path: Path) -> None:
    """Two sources, two different facts; the summary may not merge them."""
    other = "Jumper, J. et al. Highly accurate protein structure prediction. Nature, 2021."
    text, _ = drafted("A [1]. B [2].", [REAL, other])
    notice = Retraction("retraction-watch", "2021-03-01", "10.1/n", "Retraction")

    class OneDown(StubResolver):
        def retraction(self, doi: str) -> Retraction | None:
            self.asked_retraction.append(doi)
            if doi == DOI:
                return notice
            raise ProviderError("retraction check unavailable (HTTP 503)")

    resolver = OneDown({"Vaswani": resolved(), "Jumper": resolved("10.1038/other")})
    with Cache(tmp_path / "c.sqlite3") as db:
        ready = prepare(text, engine(resolver=resolver, cache=db))
        assert db.get_retraction("10.1038/other") is None
    summary = {stage.name: stage.summary for stage in ready.stages}
    assert summary[RETRACTIONS] == "1 retracted, 1 unavailable"


def test_a_retraction_note_survives_the_fetching_stage(tmp_path: Path) -> None:
    """Stage 5 replaces the source's notes with the fetch's own; a fact stage 4
    recorded must not be dropped on the way."""
    text, _ = drafted("The effect was large [1].", [REAL])
    chain = StubOpenAccess({DOI: abstract_evidence()})
    with Cache(tmp_path / "c.sqlite3") as db:
        ready = prepare(
            text, engine(resolver=DownRetractions({"Vaswani": resolved()}), chain=chain, cache=db)
        )
    notes = ready.sources[1].notes
    assert any(note.startswith(RETRACTION_UNAVAILABLE) for note in notes)
    assert "no open-access full text" in notes  # the fetch's own note is still there


# --- stage 7: judging (spec section 9 step 8, section 11.1) ------------------


class FakeJudgeClient:
    """``JudgeClient`` without a socket: scripted answers, counted calls.

    An answer the script does not cover is an agreeing ``SUPPORTED`` for every id the
    prompt carries, so a test only writes down the answers it is actually about.
    """

    provider = "fake"
    model = "judge-1"

    def __init__(self, answers: Sequence[str | Exception] = ()) -> None:
        self.cost = JudgeCost(model=self.model)
        self.prompts: list[str] = []
        self.answers = list(answers)
        self.closed = False

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        user = messages[-1]["content"]
        self.prompts.append(user)
        answer = self.answers.pop(0) if self.answers else _agreeing(user)
        if isinstance(answer, Exception):
            raise answer
        self.cost.calls += 1
        self.cost.prompt_tokens += 100
        self.cost.completion_tokens += 10
        return Completion(text=answer, prompt_tokens=100, completion_tokens=10, model=self.model)

    def close(self) -> None:
        self.closed = True


def _agreeing(user: str) -> str:
    ids = re.findall(r"id: (\S+)", user)
    agreed = [
        {"id": ident, "label": "SUPPORTED", "rationale": "the passage says so"} for ident in ids
    ]
    return json.dumps({"opinions": agreed})


def _opinion(ident: str, label: str, rationale: str = "the passage says so") -> str:
    return json.dumps({"opinions": [{"id": ident, "label": label, "rationale": rationale}]})


# 0.455 sits between ``decide`` (0.45) and ``medium`` (0.457948): decided, but only
# just -- which is exactly the band the judge exists for (spec section 9 step 8).
LOW_REFUTED_ROW = (0.02, 0.455, 0.525)

JUDGE_BODY = (
    "Transformers improved translation quality [1]. The method yields a 40% speedup [1]. "
    "The rest concerns tokenisers entirely [1]. Nothing further was measured."
)
# The three claims above, in order: a low-tier REFUTED, a rule-decided numeric
# mismatch, and an NEI whose score never cleared ``decide`` (so it has no passage).
JUDGE_TABLE = {SUPPORTING: LOW_REFUTED_ROW, FILLER: NEI_ROW}


def judged(
    client: FakeJudgeClient,
    *,
    body: str = JUDGE_BODY,
    table: dict[str, tuple[float, float, float]] | None = None,
    text: str = PAPER,
    cache: Cache | None = None,
    batch_size: int = 20,
    summarize: bool = False,
    on_event: Callable[[Event], None] | None = None,
) -> Report:
    built = paper_engine(table=table or JUDGE_TABLE, text=text, cache=cache)
    built.judge = Judge(client, batch_size=batch_size)  # type: ignore[arg-type]
    return verify(draft(body, [REAL]), built, summarize=summarize, on_event=on_event)


def test_only_a_low_tier_verdict_with_a_passage_is_escalated() -> None:
    """Rule-decided and passage-less verdicts are not the judge's business: one has
    nothing to add to, the other has nothing to judge against (product rule 1)."""
    client = FakeJudgeClient()
    report = judged(client)

    assert len(client.prompts) == 1
    prompt = client.prompts[0]
    assert "Transformers improved translation quality." in prompt
    assert "40% speedup" not in prompt  # rule-decided, score 1.0
    assert "tokenisers entirely" not in prompt  # NEI, no passage
    tiers = {item.claim.text: item.verdict.tier for item in report.results}
    assert tiers["Transformers improved translation quality."] == "low"


def test_the_judges_opinion_is_attached_and_alters_nothing() -> None:
    """Spec section 11.1: the judge gives an opinion, never a verdict."""
    client = FakeJudgeClient(
        [_opinion("c0", "SUPPORTED", 'it says "improved translation quality"')]
    )
    report = judged(client)

    judged_result = next(item for item in report.results if item.verdict.tier == "low")
    assert judged_result.judge is not None
    assert judged_result.judge.label is Label.SUPPORTED
    assert judged_result.judge.model == "fake judge-1"
    # The local verdict is untouched, and so is the finding built from it.
    assert judged_result.verdict.label is Label.REFUTED
    finding = next(f for f in report.findings if f.kind is Kind.NOT_SUPPORTED)
    assert finding.state == "NOT SUPPORTED"
    assert finding.verdict is not None and finding.verdict.label is Label.REFUTED
    assert finding.judge is not None and finding.judge.label is Label.SUPPORTED
    assert judge_detail(finding.judge) in finding.detail


def test_an_agreeing_judge_leaves_the_finding_without_a_judge_line() -> None:
    """A second opinion worth printing is one that differs; agreement is recorded
    on the finding but does not add a line saying the same thing twice."""
    client = FakeJudgeClient([_opinion("c0", "REFUTED", "it contradicts the claim")])
    report = judged(client)

    finding = next(f for f in report.findings if f.kind is Kind.NOT_SUPPORTED)
    assert finding.judge is not None and finding.judge.label is Label.REFUTED
    assert not any(detail.startswith(JUDGE_DETAIL_PREFIX) for detail in finding.detail)


def test_the_judging_stage_reports_what_it_reviewed_and_what_it_cost() -> None:
    client = FakeJudgeClient()
    events, listen = collected()
    report = judged(client, on_event=listen)

    stage = report.stages[-1]
    assert stage.name == JUDGING
    assert stage.by == "fake judge-1"
    assert stage.summary == "1 of 3 verdicts reviewed, 1 call, 100 prompt · 10 completion tokens"
    assert report.api_calls == 1
    assert report.judge_cost is not None and report.judge_cost.calls == 1
    assert report.models["judge"] == "fake judge-1"
    assert any(isinstance(e, StageStart) and e.name == JUDGING for e in events)
    assert [e for e in events if isinstance(e, Progress) and e.name == JUDGING] == [
        Progress(name=JUDGING, done=1, total=1, detail="")
    ]


def test_a_run_without_a_judge_says_nothing_about_one() -> None:
    """The default path makes zero LLM calls and the report shows no judge at all."""
    built = paper_engine(table=JUDGE_TABLE)
    report = verify(draft(JUDGE_BODY, [REAL]), built)

    assert report.api_calls == 0
    assert report.judge_cost is None
    assert "judge" not in report.models
    assert all(item.judge is None for item in report.results)
    assert all(item.judge is None for item in report.findings)
    assert [stage.name for stage in report.stages][-1] == VERIFYING


def test_a_cached_judgement_is_read_back_instead_of_asked_again(tmp_path: Path) -> None:
    text = draft(JUDGE_BODY, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        first_client = FakeJudgeClient()
        built = paper_engine(table=JUDGE_TABLE, cache=cache)
        built.judge = Judge(first_client)  # type: ignore[arg-type]
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", PAPER)
        first = decide_all(ready, built)
        assert first_client.cost.calls == 1

        again = FakeJudgeClient()
        built.close()
        built.embedder = lambda: WordEmbedder()
        built.scorer = lambda: TableScorer(JUDGE_TABLE)
        built.judge = Judge(again)  # type: ignore[arg-type]
        second = decide_all(ready, built)

        assert again.cost.calls == 0
        assert second.api_calls == 0
        opinions = [item.judge for item in second.results if item.judge is not None]
        assert len(opinions) == 1 and opinions[0] == first.results[0].judge


def test_a_run_that_asked_nobody_never_reports_the_provider_as_down(tmp_path: Path) -> None:
    """``unavailable`` describes the calls *this* stage made, and it made none.

    A judge reused for a second document remembers the first one's trouble, and a
    stage that read every opinion out of the cache would otherwise inherit it --
    reporting a provider as down on a run that never asked it anything.
    """
    text = draft(JUDGE_BODY, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        built = paper_engine(table=JUDGE_TABLE, cache=cache)
        reviewer = Judge(FakeJudgeClient())  # type: ignore[arg-type]
        built.judge = reviewer
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", PAPER)
        decide_all(ready, built)

        reviewer.unavailable = True  # as the document before this one left it
        reviewer.detail = "HTTP 401 from https://api.test/chat/completions"
        built.close()
        built.embedder = lambda: WordEmbedder()
        built.scorer = lambda: TableScorer(JUDGE_TABLE)
        warm = decide_all(ready, built)

    stage = next(item for item in warm.stages if item.name == JUDGING)
    assert "unavailable" not in stage.summary
    assert "1 of 3 verdicts reviewed" in stage.summary


def test_toggling_the_judge_never_invalidates_a_cached_verdict(tmp_path: Path) -> None:
    """``model_id`` says nothing about the judge, so a warm run stays warm."""
    text = draft(JUDGE_BODY, [REAL])
    with Cache(tmp_path / "c.sqlite3") as cache:
        built = paper_engine(table=JUDGE_TABLE, cache=cache)
        ready = prepare(text, built)
        source_row(cache, f"doi:{DOI}", PAPER)
        decide_all(ready, built)

        built.close()
        built.embedder = lambda: WordEmbedder()
        built.scorer = lambda: TableScorer({})  # any score() call would raise KeyError
        built.judge = Judge(FakeJudgeClient())  # type: ignore[arg-type]
        warm = decide_all(ready, built)

        assert all(item.from_cache for item in warm.results)


JUDGE_DOWN = (
    "judge unavailable after 0 calls "
    "(HTTP 401 from https://api.test/chat/completions); local verdicts stand"
)


def test_an_unavailable_judge_is_a_note_and_the_local_verdicts_stand() -> None:
    client = FakeJudgeClient([JudgeUnavailable("HTTP 401 from https://api.test/chat/completions")])
    events, listen = collected()
    report = judged(client, on_event=listen)

    notes = [e.text for e in events if isinstance(e, Note)]
    assert JUDGE_DOWN in notes
    # The detail says *why*, so a wrong key is visible rather than just "unavailable".
    assert "HTTP 401" in JUDGE_DOWN
    # Nothing about the run changed: the findings are the ones the models decided.
    assert report.counts() == {Kind.NUMERIC_MISMATCH: 1, Kind.NOT_SUPPORTED: 1, Kind.NEI: 1}
    assert all(item.judge is None for item in report.results)
    assert report.api_calls == 0


def test_an_unavailable_judge_is_carried_by_the_stage_summary_and_the_json() -> None:
    """A note alone is quiet-suppressed and never stored, so "the provider was down"
    and "nothing was escalated" would read identically -- the absence-of-evidence
    collapse product rules 2 and 6 forbid. The stage summary says which it was."""
    client = FakeJudgeClient([JudgeUnavailable("HTTP 401 from https://api.test/chat/completions")])
    report = judged(client)

    stage = report.stages[-1]
    assert stage.name == JUDGING
    assert stage.summary == JUDGE_DOWN
    # ``--format json`` goes through the same serialiser, so the durable surface has it.
    payload = json.loads(json_text(report))
    assert payload["stages"][-1]["summary"] == JUDGE_DOWN


def test_a_judge_that_answered_says_what_it_reviewed_not_that_it_was_unavailable() -> None:
    report = judged(FakeJudgeClient())

    assert "unavailable" not in report.stages[-1].summary


class BrokenJudgementCache(Cache):
    """A cache whose judgement write always fails -- what a ``sources`` row that is
    no longer there does under ``PRAGMA foreign_keys=ON``."""

    def put_judgement(self, *args: object, **kwargs: object) -> None:
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")


def test_a_judgement_that_cannot_be_cached_is_a_note_and_the_run_finishes(
    tmp_path: Path,
) -> None:
    """The optional judge layer is the last thing in the tool allowed to end a run:
    a cache that refuses the write costs the *next* run a call, nothing more."""
    with BrokenJudgementCache(tmp_path / "c.sqlite3") as cache:
        source_row(cache, f"doi:{DOI}", PAPER)
        events, listen = collected()
        report = judged(FakeJudgeClient(), cache=cache, on_event=listen)

    notes = [e.text for e in events if isinstance(e, Note)]
    assert "judgement not cached: IntegrityError" in notes
    # The opinion the run already paid for is still attached to this report.
    assert [item.judge for item in report.results if item.judge is not None]
    assert report.stages[-1].summary.startswith("1 of 3 verdicts reviewed")


def test_a_judge_that_answers_nonsense_is_noted_and_costs_no_verdict() -> None:
    client = FakeJudgeClient(["I would rather not say."])
    events, listen = collected()
    report = judged(client, on_event=listen)

    assert any("not JSON" in e.text for e in events if isinstance(e, Note))
    assert all(item.judge is None for item in report.results)
    assert report.api_calls == 1


FOUR_LOW = (
    "Alpha rose sharply [1]. Beta fell slowly [1]. Gamma stayed flat [1]. "
    "Delta wobbled often [1]. Nothing further was measured."
)


def test_the_judge_is_stopped_between_batches_when_the_run_is_cancelled() -> None:
    cancel = threading.Event()
    client = FakeJudgeClient()
    built = paper_engine(table=dict.fromkeys(FOUR_SENTENCES, LOW_REFUTED_ROW), text=FOUR_SOURCE)
    built.judge = Judge(client, batch_size=1)  # type: ignore[arg-type]
    ready = prepare(draft(FOUR_LOW, [REAL]), built)

    seen: list[Event] = []

    def watch(event: Event) -> None:
        seen.append(event)
        if isinstance(event, Progress) and event.name == JUDGING:
            cancel.set()

    with pytest.raises(Cancelled) as stopped:
        decide_all(ready, built, on_event=watch, cancel=cancel)

    assert client.cost.calls == 1  # one batch, then the cancel was seen
    partial = stopped.value.report
    assert partial is not None and partial.cancelled
    # A stopped run keeps what it learned, judge included (product rule 6).
    assert sum(1 for item in partial.results if item.judge is not None) == 1


# --- --summarize (task 9.3) ---------------------------------------------------------


class SummarisingJudgeClient(FakeJudgeClient):
    """Reviews as ``FakeJudgeClient`` does, then writes a paragraph when asked for one.

    The summary call is the one that carries no JSON schema: prose, not opinions.
    """

    PARAGRAPH = "42 references were checked and three do not say what the draft says."

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        if json_schema is None:
            self.answers.insert(0, self.PARAGRAPH)
        return super().complete(
            messages,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )


class MuteSummaryClient(FakeJudgeClient):
    """Reviews, then goes away before the summary: the paragraph is lost, not the run."""

    DOWN = "HTTP 401 from https://api.test/chat/completions"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        if json_schema is None:
            raise JudgeUnavailable(self.DOWN)
        return super().complete(
            messages,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )


def test_a_run_that_was_not_asked_for_a_summary_has_none() -> None:
    """Spec section 11.1: off by default, and off means no call and no paragraph."""
    client = FakeJudgeClient()
    report = judged(client)
    assert report.summary is None
    assert len(client.prompts) == 1  # the review, and nothing after it


def test_the_summary_costs_one_call_and_changes_nothing_it_read() -> None:
    """Spec section 11.1: it runs over a finished report and cannot move a verdict."""
    plain = judged(FakeJudgeClient())
    client = SummarisingJudgeClient()
    summarised = judged(client, summarize=True)

    assert summarised.summary == SummarisingJudgeClient.PARAGRAPH
    assert summarised.api_calls == plain.api_calls + 1
    assert len(client.prompts) == 2  # the review, then exactly one summary call
    assert summarised.results == plain.results
    assert summarised.findings == plain.findings
    assert summarised.judge_cost is not None and summarised.judge_cost.calls == 2


def test_the_summary_is_shown_the_finished_report_and_nothing_else() -> None:
    client = SummarisingJudgeClient()
    judged(client, summarize=True)
    asked = client.prompts[-1]

    assert "# proofpath report" in asked
    # It sees the report, never a source: the judge cannot check the draft itself.
    assert PAPER not in asked


def test_a_summary_the_judge_never_wrote_is_reported_not_silently_missing() -> None:
    """Product rule 2: an absent paragraph must not read as a report with none to give."""
    events, listen = collected()
    report = judged(MuteSummaryClient(), summarize=True, on_event=listen)

    assert report.summary == ""  # asked and got nothing, which is not "never asked"
    stage = report.stages[-1]
    assert stage.name == SUMMARISING
    assert stage.summary == (
        f"summary {judge_unavailable(0, MuteSummaryClient.DOWN, what='summary')}"
    )
    # No call count: one request was made, and ``cost.calls`` counts answers, so
    # "after 0 calls" would read as if the summary had never been attempted.
    assert "0 calls" not in stage.summary
    # The stage carries it, and the front ends print their own line from that; a note
    # beside it would say the same thing twice, as the live run did.
    assert not [e for e in events if isinstance(e, Note) and "summary" in e.text]
    # The escalation's own calls are not charged to the summary that never happened.
    assert report.api_calls == 1


def test_the_summarising_stage_says_what_the_paragraph_cost() -> None:
    seen: list[Event] = []
    report = judged(SummarisingJudgeClient(), summarize=True, on_event=seen.append)

    stage = report.stages[-1]
    assert stage.name == SUMMARISING
    assert stage.by == "fake judge-1"
    assert "1 call" in stage.summary
    assert "100 prompt" in stage.summary and "10 completion" in stage.summary
    ends = [e for e in seen if isinstance(e, StageEnd) and e.name == SUMMARISING]
    assert [e.summary for e in ends] == [stage.summary]


def test_the_json_payload_names_the_model_that_wrote_the_summary() -> None:
    """Spec section 11.1: model prose is labelled model-written wherever it is shown.

    A consumer rendering ``payload["summary"]`` on its own has nothing else to label
    it with -- the markdown heading and the terminal's label are not in the JSON --
    so the paragraph travels with a sibling naming who wrote it.
    """
    report = judged(SummarisingJudgeClient(), summarize=True)

    assert report.summary_model == "fake judge-1"
    payload = json.loads(json_text(report))
    assert payload["summary"] == SummarisingJudgeClient.PARAGRAPH
    assert payload["summary_model"] == "fake judge-1"


def test_a_report_with_no_summary_names_nobody_as_its_author() -> None:
    """Never asked and asked-but-unanswered are both "no model wrote this"."""
    assert judged(FakeJudgeClient()).summary_model is None
    unanswered = judged(MuteSummaryClient(), summarize=True)
    assert unanswered.summary == "" and unanswered.summary_model is None


def test_the_judging_stages_cost_is_a_snapshot_not_the_live_running_total() -> None:
    """A finished report is frozen, and ``JudgeCost`` is not: the client keeps adding
    to the same object for the next document, so aliasing it would let a report's
    stated cost keep changing after the run that earned it was over."""
    client = FakeJudgeClient()
    report = judged(client)

    assert report.judge_cost is not None
    frozen = replace(report.judge_cost)
    client.cost.calls += 7
    client.cost.prompt_tokens += 1_000
    assert report.judge_cost == frozen


def test_the_summarising_stages_cost_is_a_snapshot_too() -> None:
    client = SummarisingJudgeClient()
    report = judged(client, summarize=True)

    assert report.judge_cost is not None
    frozen = replace(report.judge_cost)
    client.cost.completion_tokens += 500
    assert report.judge_cost == frozen


def test_asking_for_a_summary_without_a_judge_is_a_programming_error() -> None:
    built = paper_engine(table=JUDGE_TABLE, text=PAPER)
    with pytest.raises(ValueError, match="summar"):
        verify(draft(JUDGE_BODY, [REAL]), built, summarize=True)


def _result(verdict: Verdict) -> ClaimResult:
    claim = Claim(
        text="the method is faster",
        locator=Locator(line=1),
        cited_refs=(1,),
        paragraph=0,
        sentence=0,
    )
    return ClaimResult(claim, 1, "doi:10.1/x", verdict, from_cache=False)


PASSAGE = Passage("the method is faster", "doi:10.1/x", 0)


@pytest.mark.parametrize(
    ("verdict", "escalated"),
    [
        (Verdict(Label.REFUTED, 0.455, "low", PASSAGE), True),
        (Verdict(Label.NEI, 0.46, "medium", PASSAGE), True),
        (Verdict(Label.SUPPORTED, 0.9999, "high", PASSAGE), False),
        (Verdict(Label.SUPPORTED, 0.46, "medium", PASSAGE), False),
        (Verdict(Label.NEI, 0.2, "low", None), False),
        (
            Verdict(Label.REFUTED, 1.0, "high", PASSAGE, reason="numeric mismatch: claim says 40%"),
            False,
        ),
    ],
)
def test_the_escalation_set_is_the_low_band_with_something_to_judge_against(
    verdict: Verdict, escalated: bool
) -> None:
    assert _escalates(_result(verdict)) is escalated
