"""Which provider each stage asks, when the entry and the source disagree.

``tests/test_providers.py`` pins the routing functions in isolation. These tests pin
the two places in ``verify.prepare`` where the routing has to be re-decided from the
source rather than carried over from the entry, because the resolving stage can put
a reference somewhere its raw string does not predict:

* a warm cache hands a reference the DOI a *previous* run resolved, after the family
  has already been chosen (the retraction stage);
* a provider can answer the fetching stage with no document at all (task 10.2's
  social provider will, for a post that was deleted between resolve and fetch).

Both are silent failures if the core trusts the entry: an unchecked record reported
as clean, and a planned source that vanishes out of the counts (product rules 2, 6).
The stubs and the engine builder come from ``tests/test_verify.py`` so that both
files agree on what a fake run looks like; that file is not modified by importing it.
"""

from __future__ import annotations

from pathlib import Path

from proofpath.cache import Cache
from proofpath.document import Reference
from proofpath.providers import EvidenceDoc, Providers, Scheme
from proofpath.report import Kind
from proofpath.resolve import Candidate, ResolveResult, Retraction, State
from proofpath.verify import (
    CACHE_BY,
    FETCHING,
    NO_DOCUMENT,
    NO_IDENTIFIER,
    NOBODY,
    NOT_ATTEMPTED,
    RESOLVERS_BY,
    RESOLVING,
    RETRACTIONS,
    RETRACTIONS_BY,
    prepare,
)
from tests.test_verify import (
    DOI,
    PAGE,
    REAL,
    StubFetcher,
    StubOpenAccess,
    StubResolver,
    draft,
    engine,
    fulltext_evidence,
    resolved,
)

NOTICE = Retraction("retraction-watch", "2021-03-01", "10.1/n", "Retraction")

# An author-year bibliography carries no entry markers, so ``ingest`` cuts one entry
# per paragraph and the raw string is the address and nothing else -- the one shape
# ``provider_for`` routes to the web provider.
UNMARKED = f"A claim [1].\n\nReferences\n\n{PAGE}\n"
# The same bibliography with a paper in front of the address: one entry the indexes
# answer for and one they are never asked about, which is the shape the resolving
# stage's attribution has to tell apart.
MIXED = f"A claim [1]. Another [2].\n\nReferences\n\n{REAL}\n\n{PAGE}\n"
ACADEMIC_ONLY = f"A claim [1].\n\nReferences\n\n{REAL}\n"

# A record that resolved and carries no identifier at all -- a book, or a record
# whose provider has no DOI for it. ``_placed`` files it with ``source_id=None``.
NO_ID_RECORD = ResolveResult(
    State.RESOLVED,
    Candidate(
        doi="", title="Air quality guidelines", first_author="WHO", year=2021,
        venue="", provider="crossref",
    ),
    [],
)  # fmt: skip


def test_a_doi_source_is_checked_for_retractions_however_the_entry_was_routed(
    tmp_path: Path,
) -> None:
    """The resolving stage reads the cache *after* it has picked a family, so a warm
    row from a run that routed the same raw string differently can hand a web-routed
    entry a resolution with a DOI. The retraction stage iterates ``doi:`` sources, so
    that source reaches it -- and the web provider would answer ``None``, which the
    stage counts as "checked, no notice"."""
    stub = StubResolver(retractions={DOI: NOTICE})
    with Cache(tmp_path / "c.sqlite3") as cache:
        cache.put_resolution(PAGE, resolved())
        built = engine(resolver=stub, fetcher=StubFetcher())
        built.cache = cache
        ready = prepare(UNMARKED, built)

    assert ready.sources[1].source_id == f"doi:{DOI}"
    assert stub.asked_retraction == [DOI]
    assert ready.sources[1].retraction == NOTICE
    assert [f.kind for f in ready.findings if f.kind is Kind.RETRACTED] == [Kind.RETRACTED]


class SilentProvider:
    """A provider that resolves a source and then reads nothing at all.

    ``StubSocial`` in ``tests/test_providers.py`` already answers this way, and task
    10.2's real social provider will whenever a post is gone by the time it is read.
    """

    scheme: Scheme = "academic"

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def resolve(self, ref: Reference) -> ResolveResult:
        return resolved()

    def retraction(self, result: ResolveResult) -> Retraction | None:
        return None

    def fetch(self, ref: Reference, result: ResolveResult) -> list[EvidenceDoc]:
        self.fetched.append(ref.raw)
        return []


def test_a_provider_that_reads_nothing_still_leaves_an_honest_source() -> None:
    """No document must never mean no record of the attempt: the source keeps the
    ``state=""`` / ``text_kind="none"`` of a planned-but-unread row, which reads as a
    source that was fetched and simply held no text, and the fetching stage's counts
    lose it altogether (product rule 6)."""
    built = engine()
    silent = SilentProvider()
    built.providers = Providers(academic=silent, web=silent, social=None)

    ready = prepare(draft("A claim [1].", [REAL]), built)

    assert silent.fetched == [f"[1] {REAL}"]
    status = ready.sources[1]
    assert status.text_kind == "none"
    assert status.state == NO_DOCUMENT
    unverified = [f for f in ready.findings if f.kind is Kind.UNVERIFIED]
    assert [f.state for f in unverified] == [NO_DOCUMENT]
    assert unverified[0].source_id == f"doi:{DOI}"
    fetching = next(stage for stage in ready.stages if stage.name == FETCHING)
    assert fetching.summary == "0 full text, 0 abstract, 1 unverified"


def test_a_doi_source_is_read_by_the_academic_provider_however_the_entry_was_routed(
    tmp_path: Path,
) -> None:
    """The mirror of the retraction test above, for the same warm cache and the same
    reason: the fetching stage works from source ids too. Reading a ``doi:`` source
    up the web ladder skips the open-access chain the id exists for (an arXiv PDF, an
    Unpaywall copy) and degrades the run's coverage for nothing (product rule 6), and
    it files the page's text under a DOI that text never came from."""
    chain = StubOpenAccess({DOI: fulltext_evidence()})
    fetcher = StubFetcher()
    with Cache(tmp_path / "c.sqlite3") as cache:
        cache.put_resolution(PAGE, resolved())
        ready = prepare(
            UNMARKED,
            engine(resolver=StubResolver(), chain=chain, fetcher=fetcher, cache=cache),
        )

    assert ready.sources[1].source_id == f"doi:{DOI}"
    assert chain.calls == [(DOI, None)]
    assert fetcher.calls == []
    assert ready.sources[1].text_kind == "fulltext"


def test_a_record_with_no_identifier_is_not_read_from_the_entrys_own_address(
    tmp_path: Path,
) -> None:
    """A resolved record with nothing to fetch it by is filed under no source id at
    all, and v0.3.0 answered it with ``NO_IDENTIFIER`` whatever the entry printed:
    the address was only ever the fallback for a record the indexes do not cover
    (spec section 8), not for one they cover and have no identifier for."""
    chain = StubOpenAccess()
    fetcher = StubFetcher()
    with Cache(tmp_path / "c.sqlite3") as cache:
        cache.put_resolution(PAGE, NO_ID_RECORD)
        ready = prepare(
            UNMARKED,
            engine(resolver=StubResolver(), chain=chain, fetcher=fetcher, cache=cache),
        )

    assert ready.sources[1].source_id is None
    assert fetcher.calls == []
    assert chain.calls == []
    assert ready.sources[1].state == NO_IDENTIFIER


# --- what the resolving stage may claim ---------------------------------------


def test_the_resolving_stage_names_nobody_when_nobody_was_asked() -> None:
    """``WebProvider.resolve`` answers out of the entry itself, so a bibliography of
    bare addresses reaches Crossref and Semantic Scholar not once. Naming them on the
    stage line is the untruth ``CACHE_BY`` exists to prevent, by a different route
    (product rule 2)."""
    stub = StubResolver()
    ready = prepare(UNMARKED, engine(resolver=stub, fetcher=StubFetcher()))

    assert stub.resolved == []
    assert {stage.name: stage.by for stage in ready.stages}[RESOLVING] == NOBODY


def test_one_entry_asked_of_the_indexes_still_names_them() -> None:
    """The other half of the rule: a family that was consulted is claimed, whatever
    else the run contained. Unchanged from v0.3.0 for every academic bibliography."""
    stub = StubResolver({"Vaswani": resolved()})
    ready = prepare(MIXED, engine(resolver=stub, fetcher=StubFetcher()))

    assert stub.resolved == [REAL]
    assert {stage.name: stage.by for stage in ready.stages}[RESOLVING] == RESOLVERS_BY


def test_a_mixed_run_whose_only_lookup_came_from_the_cache_says_cache(
    tmp_path: Path,
) -> None:
    """One entry out of two came from the cache and the other asked nobody anything,
    so the run consulted exactly one thing and it was the cache. Counting the locally
    answered entry as "not cached" used to tip the line back to the two indexes."""
    with Cache(tmp_path / "c.sqlite3") as cache:
        prepare(ACADEMIC_ONLY, engine(resolver=StubResolver({"Vaswani": resolved()}), cache=cache))
        stub = StubResolver({"Vaswani": resolved()})
        ready = prepare(MIXED, engine(resolver=stub, fetcher=StubFetcher(), cache=cache))

    assert stub.resolved == []
    assert {stage.name: stage.by for stage in ready.stages}[RESOLVING] == CACHE_BY


def test_a_denied_run_still_names_the_resolvers_it_announced() -> None:
    """``_resolving_attribution``'s ``total == 0`` row, which nothing pinned. A run
    the network permission stopped placed no reference at all, so it has no units to
    speak for either way and keeps the provider the stage opened with; what says the
    stage did nothing is its summary, and that is the line product rule 2 needs."""
    built = engine(fetcher=StubFetcher(network_allowed=False, network_note="network: deny"))
    ready = prepare(MIXED, built)

    stages = {stage.name: stage for stage in ready.stages}
    assert stages[RESOLVING].by == RESOLVERS_BY
    assert stages[RESOLVING].summary == NOT_ATTEMPTED
    assert ready.sources and all(status.resolve is None for status in ready.sources.values())


def test_the_retraction_stage_names_nobody_when_there_was_no_doi_to_ask_about() -> None:
    """A bibliography of bare addresses gives the stage nothing to check, and
    "Retraction Watch | none" beside it reads as a clean sheet somebody checked --
    the same untruth ``CACHE_BY`` exists to prevent (product rule 2). It became
    visible next to ``Resolving ... none`` on an all-bare-URL report."""
    ready = prepare(UNMARKED, engine(resolver=StubResolver(), fetcher=StubFetcher()))

    stages = {stage.name: stage for stage in ready.stages}
    assert stages[RETRACTIONS].by == NOBODY
    assert stages[RETRACTIONS].summary == "none"
    # The row that does have DOIs is unchanged: the providers are still claimed.
    with_doi = prepare(ACADEMIC_ONLY, engine(resolver=StubResolver({"Vaswani": resolved()})))
    assert {s.name: s.by for s in with_doi.stages}[RETRACTIONS] == RETRACTIONS_BY
