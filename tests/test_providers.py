"""The two source families behind one interface, and the routing that picks between them.

Spec section 5.2. Nothing here touches the network: the resolver, the open-access
chain and the fetch ladder are stubs, so what is tested is the provider's own
reading of their answers — which identifier it fetches by, which state it carries
through verbatim (product rule 2), and who the fetching stage is attributed to.
"""

from __future__ import annotations

from proofpath.document import Locator, Reference
from proofpath.fetch import Fetched, FetchStats, Outcome
from proofpath.oa import ABSTRACT_ONLY, FULLTEXT_MIN_WORDS, Attempt, Evidence, Location
from proofpath.providers import (
    NO_IDENTIFIER,
    NO_TEXT,
    EvidenceDoc,
    Providers,
    checker_for,
    honest_state,
    provider_for,
    reader_for,
)
from proofpath.providers.academic import AcademicProvider
from proofpath.providers.web import WebProvider
from proofpath.resolve import Candidate, ResolveResult, Retraction, State

DOI = "10.1038/s41586-021-03819-2"
ARXIV = "2103.00020"
PAGE = "https://example.test/report.html"


def words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


FULLTEXT = words(FULLTEXT_MIN_WORDS + 10)
SHORT = words(40)


def ref(raw: str, number: int = 1) -> Reference:
    return Reference(number=number, raw=raw, locator=Locator(line=1))


def resolved(doi: str = DOI) -> ResolveResult:
    best = Candidate(
        doi=doi, title="Attention is all you need", first_author="Vaswani", year=2017,
        venue="NeurIPS", provider="crossref",
    )  # fmt: skip
    return ResolveResult(State.RESOLVED, best, [best])


NOT_INDEXED = ResolveResult(State.NOT_INDEXED, None, [])


# --- stubs -------------------------------------------------------------------


class StubResolver:
    def __init__(
        self,
        result: ResolveResult | None = None,
        retractions: dict[str, Retraction] | None = None,
    ) -> None:
        self.result = result or resolved()
        self.retractions = retractions or {}
        self.resolved: list[str] = []
        self.asked: list[str] = []

    def resolve(self, raw: str) -> ResolveResult:
        self.resolved.append(raw)
        return self.result

    def retraction(self, doi: str) -> Retraction | None:
        self.asked.append(doi)
        return self.retractions.get(doi)


class StubOpenAccess:
    def __init__(self, evidence: Evidence | None = None) -> None:
        self.evidence = evidence or none_evidence(Outcome.UNREACHABLE)
        self.calls: list[tuple[str | None, str | None]] = []

    def fetch(self, doi: str | None, arxiv_id: str | None = None) -> Evidence:
        self.calls.append((doi, arxiv_id))
        return self.evidence


class StubFetcher:
    network_allowed = True
    network_note = ""

    def __init__(self, page: Fetched | None = None) -> None:
        self.page = page
        self.calls: list[str] = []

    def fetch(
        self, url: str, *, text_kind: str = "fulltext", counts_as_source: bool = True
    ) -> Fetched:
        self.calls.append(url)
        return self.page if self.page is not None else unreachable(url)

    def summary(self) -> FetchStats:
        return FetchStats(counts={}, browser_skipped=0)


def fetched(url: str, text: str, *, step: int = 1, from_cache: bool = False) -> Fetched:
    return Fetched(
        url=url, final_url=url, step=step, outcome=Outcome.OK, status=200,
        content_type="text/html", kind="html", body=b"", text=text, notes=[],
        from_cache=from_cache,
    )  # fmt: skip


def unreachable(url: str) -> Fetched:
    return Fetched(
        url=url, final_url=url, step=1, outcome=Outcome.UNREACHABLE, status=None,
        content_type="", kind="other", body=b"", text="", notes=["step 1 httpx: no response"],
        from_cache=False,
    )  # fmt: skip


def fulltext_evidence(url: str = "https://arxiv.test/paper.pdf") -> Evidence:
    return Evidence(
        kind="fulltext", state="", text=FULLTEXT, url=url, source="arxiv",
        attempts=[Attempt(Location("arxiv", url, "pdf"), Outcome.OK, 1, len(FULLTEXT.split()))],
        notes=[],
    )  # fmt: skip


def abstract_evidence(url: str = "https://api.test/abstract") -> Evidence:
    return Evidence(
        kind="abstract", state=ABSTRACT_ONLY, text=SHORT, url=url, source="abstract:s2",
        attempts=[], notes=["no open-access full text"],
    )  # fmt: skip


def none_evidence(outcome: Outcome) -> Evidence:
    return Evidence(
        kind="none", state=outcome.value, text="", url="", source="", attempts=[],
        notes=[f"every location {outcome.value}"],
    )  # fmt: skip


def built(social: object | None = None) -> Providers:
    academic = AcademicProvider(StubResolver(), StubOpenAccess())
    web = WebProvider(StubFetcher())
    return Providers(academic=academic, web=web, social=social)  # type: ignore[arg-type]


class StubSocial:
    """A stand-in for the provider task 10.2 adds; routing is all it has to be."""

    scheme = "social"

    def resolve(self, reference: Reference) -> ResolveResult:
        return NOT_INDEXED

    def retraction(self, result: ResolveResult) -> Retraction | None:
        return None

    def fetch(self, reference: Reference, result: ResolveResult) -> list[EvidenceDoc]:
        return []


# --- provider_for: the routing table -----------------------------------------


def test_a_reference_that_is_nothing_but_an_address_is_a_web_source() -> None:
    providers = built()
    assert provider_for(ref(PAGE), providers) is providers.web
    assert provider_for(ref(f"  {PAGE}  "), providers) is providers.web
    # ``find_url`` drops the punctuation a style wraps an address in; what is left
    # over is punctuation too, so the entry is still only an address.
    assert provider_for(ref(f"[{PAGE}]"), providers) is providers.web
    assert provider_for(ref(f"{PAGE}."), providers) is providers.web


def test_a_reference_that_prints_an_address_beside_a_title_is_academic() -> None:
    """Routing on "contains a URL" would take every paper that prints its own DOI
    link away from Crossref. The indexes are asked first; the address is the
    fallback ``verify`` reaches for only when they say they do not cover it."""
    providers = built()
    entry = f"WHO. Air quality guidelines. {PAGE} Accessed 2024."
    assert provider_for(ref(entry), providers) is providers.academic
    assert provider_for(
        ref("Vaswani, A. Attention is all you need. NeurIPS, 2017."), providers
    ) is (providers.academic)
    assert provider_for(ref("https://doi.org/10.1/x and a title"), providers) is providers.academic


def test_a_reference_with_no_address_at_all_is_academic() -> None:
    providers = built()
    assert provider_for(ref("Smith, J. A book. Publisher, 2001."), providers) is providers.academic
    assert provider_for(ref(""), providers) is providers.academic


def test_a_social_address_goes_to_the_social_provider_when_there_is_one() -> None:
    social = StubSocial()
    providers = built(social=social)
    for raw in (
        "https://bsky.app/profile/a.test/post/1",
        "https://news.ycombinator.com/item?id=1",
        "https://old.reddit.com/r/x/comments/1/y/",
        "https://mastodon.social/@a/1",
        "https://x.com/a/status/1",
        "https://twitter.com/a/status/1",
    ):
        assert provider_for(ref(raw), providers) is social, raw


def test_a_social_address_is_read_as_a_web_page_until_there_is_a_social_provider() -> None:
    """``Providers.social`` is ``None`` in v0.4.0, which is exactly what v0.3.0 did
    with one of these: the ladder fetched the page. Task 10.2 flips it."""
    providers = built()
    assert providers.social is None
    post = "https://bsky.app/profile/a.test/post/1"
    assert provider_for(ref(post), providers) is providers.web


def test_an_entry_that_is_only_an_identifier_url_is_still_an_academic_reference() -> None:
    """``resolve.looks_unindexed`` asks this question first, and for the same reason.

    An address that *is* a DOI or an arXiv id names a record, and a record is
    resolved, checked for a retraction notice (spec section 8) and read through the
    open-access chain. Routing it by its address instead would file it under
    ``url:``, skip the notice check entirely and read the landing page — for an
    arXiv entry, the abstract page rather than the PDF, which costs the run a
    coverage grade it did not have to lose (product rule 6).
    """
    providers = built()
    for raw in (
        f"https://doi.org/{DOI}",
        f"https://dx.doi.org/{DOI}",
        f"https://arxiv.org/abs/{ARXIV}",
        f"https://arxiv.org/pdf/{ARXIV}",
    ):
        assert provider_for(ref(raw), providers) is providers.academic, raw
    # An address with no identifier in it is still read as the page it is.
    assert provider_for(ref(PAGE), providers) is providers.web


def test_a_numbered_entry_that_prints_only_an_address_is_still_academic() -> None:
    """``Reference.raw`` keeps the printed marker, so a numbered bibliography entry
    that is nothing but an address is not "nothing but an address" — it routes to
    the indexes exactly as v0.3.0 routed it. The unmarked shape an author-year list
    produces (one entry per paragraph, no marker) is the one this rule decides."""
    providers = built()
    assert provider_for(ref("[3] https://example.test/x"), providers) is providers.academic
    assert provider_for(ref("3. https://example.test/x"), providers) is providers.academic
    assert provider_for(ref("https://example.test/x"), providers) is providers.web


# --- reader_for: who reads a source the resolving stage has already placed ----


def test_a_url_source_is_read_by_the_web_provider_whoever_resolved_it() -> None:
    providers = built()
    assert reader_for("url:https://x.test/a", providers.academic, providers) is providers.web


def test_a_record_source_is_read_by_the_academic_provider_whoever_resolved_it() -> None:
    """The mirror of the rule above, and ``checker_for``'s rule for the same ids. A
    ``doi:`` or ``arxiv:`` source is a record whichever entry it came from: reading it
    up the web ladder would skip the open-access chain the identifier exists for and
    file the landing page's text under an id it did not come from (product rule 6)."""
    providers = built()
    assert reader_for(f"doi:{DOI}", providers.academic, providers) is providers.academic
    assert reader_for(f"doi:{DOI}", providers.web, providers) is providers.academic
    assert reader_for(f"arxiv:{ARXIV}", providers.academic, providers) is providers.academic
    assert reader_for(f"arxiv:{ARXIV}", providers.web, providers) is providers.academic


def test_a_source_with_no_id_at_all_is_read_by_the_academic_provider() -> None:
    """A resolved record with neither an identifier nor a ``url:`` id has nothing
    anyone can fetch it by, and v0.3.0 said exactly that (``NO_IDENTIFIER``) however
    the entry was routed. The academic provider is the one that says it."""
    providers = built()
    assert reader_for(None, providers.academic, providers) is providers.academic
    assert reader_for(None, providers.web, providers) is providers.academic


def test_a_source_id_no_family_claims_keeps_the_provider_that_resolved_it() -> None:
    """The fallback for an id no row above names: it belongs to whoever placed it."""
    providers = built(social=StubSocial())
    assert providers.social is not None
    assert reader_for("social:bsky:1", providers.social, providers) is providers.social


def test_a_url_source_that_is_a_post_is_read_by_the_social_family() -> None:
    """Every address is filed under ``url:``, posts included, so the id alone does
    not say who should read one. A post read up the web ladder is a rendering of a
    post rather than the post; the social family reads the post itself, and reads a
    platform it does not support up that same ladder (spec section 6.2)."""
    providers = built(social=StubSocial())
    post = "url:https://bsky.app/profile/a.test/post/1"
    assert reader_for(post, providers.academic, providers) is providers.social
    assert reader_for(post, providers.web, providers) is providers.social
    # With no social family assembled at all, the ladder reads it, as in v0.3.0.
    without = built()
    assert reader_for(post, without.academic, without) is without.web


# --- checker_for: who is asked whether a placed source was retracted ----------


def test_a_record_source_is_checked_for_retractions_by_the_academic_provider() -> None:
    """``WebProvider.retraction`` answers ``None`` unconditionally, and ``None`` is
    counted as "checked, and there was no notice". Asking it about a record would
    report an unchecked one as clean — absence of evidence as evidence of absence,
    which ``RETRACTION_UNAVAILABLE`` exists to prevent (product rule 2).

    ``arxiv:`` is asserted with the *web* provider as ``chosen``: with the academic
    one the row passes through the ``return chosen`` fallback and claims nothing at
    all. The retraction stage filters ``doi:`` before it calls here, so the row is
    unreachable today — which is exactly why it is pinned rather than left to the
    fallback, where widening that filter would silently make it wrong."""
    providers = built()
    assert checker_for(f"doi:{DOI}", providers.web, providers) is providers.academic
    assert checker_for(f"doi:{DOI}", providers.academic, providers) is providers.academic
    assert checker_for(f"arxiv:{ARXIV}", providers.web, providers) is providers.academic
    assert checker_for(f"arxiv:{ARXIV}", providers.academic, providers) is providers.academic


def test_every_other_source_is_checked_by_the_provider_that_resolved_it() -> None:
    """A page and a record with no identifier have no notice to look for, and both
    families say so, so neither is re-routed away from whoever resolved it."""
    providers = built()
    assert checker_for(f"url:{PAGE}", providers.web, providers) is providers.web
    assert checker_for(f"url:{PAGE}", providers.academic, providers) is providers.academic
    assert checker_for(None, providers.web, providers) is providers.web


# --- AcademicProvider --------------------------------------------------------


def test_the_academic_provider_resolves_the_entry_verbatim() -> None:
    resolver = StubResolver()
    provider = AcademicProvider(resolver, StubOpenAccess())
    entry = "Vaswani, A. Attention is all you need. NeurIPS, 2017."

    assert provider.resolve(ref(entry)) is resolver.result
    assert resolver.resolved == [entry]
    assert provider.scheme == "academic"


def test_the_academic_provider_asks_about_the_doi_it_resolved() -> None:
    notice = Retraction(source="retraction-watch", date="2021-05-01", notice_doi=None, label="x")
    resolver = StubResolver(retractions={DOI: notice})
    provider = AcademicProvider(resolver, StubOpenAccess())

    assert provider.retraction(resolved()) is notice
    assert resolver.asked == [DOI]


def test_a_record_with_no_doi_is_not_asked_about_at_all() -> None:
    """A retraction check is a question about a DOI. Nothing is invented to ask it
    with, and "nobody was asked" is reported as ``None``, not as a clean sheet."""
    resolver = StubResolver()
    provider = AcademicProvider(resolver, StubOpenAccess())

    assert provider.retraction(NOT_INDEXED) is None
    assert resolver.asked == []


def test_a_resolved_doi_is_fetched_by_its_doi() -> None:
    chain = StubOpenAccess(fulltext_evidence())
    provider = AcademicProvider(StubResolver(), chain)

    docs = provider.fetch(ref("Vaswani, A. Attention is all you need."), resolved())

    assert chain.calls == [(DOI, None)]
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_id == f"doi:{DOI}"
    assert (doc.text_kind, doc.state) == ("fulltext", "")
    assert doc.text == FULLTEXT
    assert doc.url == "https://arxiv.test/paper.pdf"
    assert (doc.step, doc.from_cache, doc.winner) == (1, False, "arXiv")


def test_a_record_with_no_doi_is_fetched_by_the_arxiv_id_the_entry_prints() -> None:
    chain = StubOpenAccess(fulltext_evidence())
    provider = AcademicProvider(StubResolver(), chain)

    docs = provider.fetch(ref(f"A preprint. arXiv:{ARXIV}"), NOT_INDEXED)

    assert chain.calls == [(None, ARXIV)]
    assert docs[0].source_id == f"arxiv:{ARXIV}"


def test_a_record_with_nothing_to_fetch_it_with_says_so_and_calls_nobody() -> None:
    chain = StubOpenAccess()
    provider = AcademicProvider(StubResolver(), chain)

    docs = provider.fetch(ref("A committee report, 2019."), NOT_INDEXED)

    assert chain.calls == []
    assert len(docs) == 1
    assert docs[0].state == NO_IDENTIFIER
    assert docs[0].text_kind == "none"
    assert docs[0].title == "the record carries no identifier to fetch it with"


def test_an_abstract_keeps_the_chains_own_words() -> None:
    provider = AcademicProvider(StubResolver(), StubOpenAccess(abstract_evidence()))

    doc = provider.fetch(ref("Smith, J."), resolved())[0]

    assert (doc.text_kind, doc.state) == ("abstract", ABSTRACT_ONLY)
    assert doc.notes == ("no open-access full text",)
    assert doc.winner == "Semantic Scholar"


def test_an_unreadable_source_carries_the_outcome_the_chain_reported() -> None:
    blocked = none_evidence(Outcome.BLOCKED_ROBOTS)
    provider = AcademicProvider(StubResolver(), StubOpenAccess(blocked))

    doc = provider.fetch(ref("Smith, J."), resolved())[0]

    assert doc.text_kind == "none"
    assert doc.state == Outcome.BLOCKED_ROBOTS.value
    assert doc.title == "the source text could not be read"


def test_text_out_of_the_cache_is_attributed_to_the_cache() -> None:
    evidence = Evidence(
        kind="fulltext", state="", text=FULLTEXT, url="https://arxiv.test/paper.pdf",
        source="arxiv", attempts=[], notes=[], from_cache=True,
    )  # fmt: skip
    provider = AcademicProvider(StubResolver(), StubOpenAccess(evidence))

    doc = provider.fetch(ref("Smith, J."), resolved())[0]

    assert (doc.winner, doc.step, doc.from_cache) == ("cache", 0, True)


# --- WebProvider -------------------------------------------------------------


def test_the_web_provider_answers_not_indexed_and_carries_the_address() -> None:
    """A page is not in a bibliographic index and never was. That is a fact about
    the indexes, not about the page (product rule 2), so it is never a ghost."""
    provider = WebProvider(StubFetcher())

    result = provider.resolve(ref(PAGE))

    assert result.state is State.NOT_INDEXED
    assert result.best is not None and result.best.url == PAGE
    assert provider.scheme == "web"


def test_a_web_reference_with_no_address_still_resolves_to_not_indexed() -> None:
    result = WebProvider(StubFetcher()).resolve(ref("A committee report, 2019."))

    assert result.state is State.NOT_INDEXED
    assert result.best is None


def test_a_page_is_never_retracted_because_nobody_publishes_notices_about_pages() -> None:
    assert WebProvider(StubFetcher()).retraction(NOT_INDEXED) is None


def test_a_long_page_is_full_text_and_is_filed_under_the_address_that_was_asked_for() -> None:
    fetcher = StubFetcher(fetched(PAGE, FULLTEXT))
    provider = WebProvider(fetcher)

    doc = provider.fetch(ref(f"A long report. {PAGE}"), NOT_INDEXED)[0]

    assert fetcher.calls == [PAGE]
    assert doc.source_id == f"url:{PAGE}"
    assert (doc.text_kind, doc.state) == ("fulltext", "")
    assert (doc.url, doc.step, doc.winner) == (PAGE, 1, "httpx")


def test_a_short_page_is_an_abstract_however_the_link_was_labelled() -> None:
    provider = WebProvider(StubFetcher(fetched(PAGE, SHORT)))

    doc = provider.fetch(ref(f"A blog post. {PAGE}"), NOT_INDEXED)[0]

    assert (doc.text_kind, doc.state) == ("abstract", ABSTRACT_ONLY)
    assert doc.text == SHORT


def test_a_page_that_was_reached_and_held_no_text_is_not_a_page_that_was_not_reached() -> None:
    provider = WebProvider(StubFetcher(fetched(PAGE, "")))

    doc = provider.fetch(ref(f"A blog post. {PAGE}"), NOT_INDEXED)[0]

    assert (doc.text_kind, doc.state) == ("none", NO_TEXT)
    assert doc.title == "the page was reached but held no text"


def test_a_page_that_could_not_be_reached_keeps_the_ladders_own_outcome() -> None:
    provider = WebProvider(StubFetcher(unreachable(PAGE)))

    doc = provider.fetch(ref(f"A blog post. {PAGE}"), NOT_INDEXED)[0]

    assert doc.state == Outcome.UNREACHABLE.value
    assert doc.notes == ("step 1 httpx: no response",)
    assert doc.title == "the source text could not be read"


def test_the_web_provider_reads_the_address_its_own_resolve_found() -> None:
    fetcher = StubFetcher(fetched(PAGE, FULLTEXT))
    provider = WebProvider(fetcher)

    provider.fetch(ref(PAGE), provider.resolve(ref(PAGE)))

    assert fetcher.calls == [PAGE]


def test_a_reference_with_no_address_is_not_fetched_at_all() -> None:
    fetcher = StubFetcher()
    provider = WebProvider(fetcher)

    assert provider.fetch(ref("A committee report, 2019."), NOT_INDEXED) == []
    assert fetcher.calls == []


# --- honest_state ------------------------------------------------------------


def test_a_state_from_outside_the_unverified_family_becomes_a_note_under_one() -> None:
    """Every producer words its own state and it is passed through untouched. A
    string the report cannot carry becomes a note rather than crashing the run."""
    assert honest_state("something else", "none") == (NO_TEXT, ("something else",))
    assert honest_state("", "none") == (NO_TEXT, ())
    assert honest_state(Outcome.BLOCKED_ROBOTS.value, "none") == (Outcome.BLOCKED_ROBOTS.value, ())
    assert honest_state(ABSTRACT_ONLY, "abstract") == (ABSTRACT_ONLY, ())


def test_the_address_the_entry_printed_wins_over_the_records_own() -> None:
    """The entry's address is the one ``verify`` files the source under. A document
    read from one address and filed under another would be quoted back under a URL
    it never came from (product rule 1)."""
    fetcher = StubFetcher(fetched(PAGE, FULLTEXT))
    elsewhere = Candidate(
        doi="", title="", first_author="", year=None, venue="", provider="url",
        url="https://elsewhere.test/other.html",
    )  # fmt: skip

    doc = WebProvider(fetcher).fetch(
        ref(f"A long report. {PAGE}"),
        ResolveResult(State.NOT_INDEXED, elsewhere, [elsewhere]),
    )[0]

    assert fetcher.calls == [PAGE]
    assert doc.source_id == f"url:{PAGE}"
