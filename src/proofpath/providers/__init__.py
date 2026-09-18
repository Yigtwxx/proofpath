"""Evidence providers: one source family each, behind one interface.

Spec section 5.2. ``verify`` asks a provider three questions about a bibliography
entry — is the source real, was it retracted, and what does it say — and never asks
how the answer was got. That is the whole point of the package: a new source family
(section 5's ``social://``) is a module in here and a row in :func:`provider_for`,
and the orchestrator does not change.

What stays outside a provider is as deliberate as what moves in. The network
permission is resolved once, by ``verify``, before any provider is consulted (spec
section 7.1); the stage summaries, the findings and the cache are ``verify``'s too.
A provider decides one thing only: which identifier a reference is fetched by, and
what the answer it got amounts to — in the words its source gave it, never reworded
(product rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit

from proofpath.document import Reference
from proofpath.fetch import Fetched, FetchStats
from proofpath.oa import Evidence
from proofpath.report import UNVERIFIED_PREFIX, TextKind
from proofpath.resolve import ResolveResult, Retraction, find_arxiv_id, find_doi, find_url

Scheme = Literal["academic", "web", "social"]

# How a source id spells which family read it. ``verify`` writes them when the
# resolving stage places a reference and reads them back when the fetching stage
# asks who should read it, so they live here rather than in either caller.
DOI_PREFIX = "doi:"
ARXIV_PREFIX = "arxiv:"
URL_PREFIX = "url:"

# Two of spec section 15's states that no provider ever answers with: each is a
# reading of what happened rather than something a source said, and both keep the
# ``UNVERIFIED (...)`` shape of the family so no renderer has to special-case them
# (product rule 2). ``verify`` re-exports both -- they were defined there until
# v0.4.0 and the rest of the tool still imports them from there.
NO_IDENTIFIER = "UNVERIFIED (no identifier to fetch)"
NO_TEXT = "UNVERIFIED (reached, no text extracted)"

# Hosts whose content is a post rather than a document, all of spec section 6.2's
# list. ``providers.social`` answers for every one of them: Bluesky, Hacker News,
# Lobste.rs, Mastodon and Lemmy through public APIs, Reddit through the user's own
# free app, and X by saying it cannot be read and the text should be pasted instead.
SOCIAL_HOSTS = frozenset(
    {
        "bsky.app",
        "news.ycombinator.com",
        "reddit.com",
        "x.com",
        "twitter.com",
        "lobste.rs",
    }
)
# Mastodon is thousands of instances, not a host: there is no list to check against,
# only the shape the big ones share. Matched conservatively -- a miss costs a post
# the social provider and it is fetched as a page, which is what v0.3.0 did anyway.
# ``social._mastodon_status`` is deliberately broader: routing a cited address is a
# guess about a host, where reading one the user handed to ``check --url`` is not.
_MASTODON_PREFIXES = ("mastodon.", "mstdn.")
_MASTODON_SUFFIX = ".social"
# Lemmy is instances too, but its post address -- ``/post/<n>`` -- is how half the
# web's blogs address an article, so the shape alone can never say "Lemmy" the way
# ``/@user/<n>`` says "Mastodon". The host has to: the ``lemmy.`` prefix most
# instances carry, or one of the big ones that do not. The list is short on purpose
# and a miss costs what a Mastodon miss costs, a post read as a page. Public because
# ``social`` reads by the same rule: asking an arbitrary host for ``/api/v3/post``
# would send a request it never invited, and reading is held to the routing test.
LEMMY_PREFIXES = ("lemmy.",)
LEMMY_HOSTS = frozenset(
    {
        "lemm.ee",
        "sh.itjust.works",
        "beehaw.org",
        "programming.dev",
        "feddit.org",
    }
)

# What is left of an entry once its address is taken out, if the entry was nothing
# but an address. ``find_url`` already drops trailing ``.,;)``; a style that wraps
# the address in brackets or quotes leaves the opening half behind.
_PUNCTUATION = " \t\r\n.,;:()[]<>\"'"


# --- what a provider needs from the outside world ---------------------------------
#
# Protocols rather than the concrete classes: everything a provider wraps is
# network-bound and a unit test may not touch the network. ``resolve.Resolver``,
# ``oa.OpenAccess`` and ``fetch.Fetcher`` satisfy these structurally, so
# ``Engine.default`` hands over the real thing and a test hands over a stub without
# either side knowing.


class Resolves(Protocol):
    """``resolve.Resolver``, reduced to what the resolving and retraction stages use."""

    def resolve(self, raw: str) -> ResolveResult: ...

    def retraction(self, doi: str) -> Retraction | None: ...


class FetchesOpenAccess(Protocol):
    """``oa.OpenAccess``: a DOI or an arXiv id in, text or a labelled state out."""

    def fetch(self, doi: str | None, arxiv_id: str | None = None) -> Evidence: ...


class FetchesUrl(Protocol):
    """``fetch.Fetcher``: one URL up the ladder, plus the network permission it resolved."""

    network_allowed: bool
    network_note: str

    def fetch(
        self,
        url: str,
        *,
        text_kind: str = "fulltext",
        counts_as_source: bool = True,
        use_cache: bool = True,
    ) -> Fetched: ...

    def summary(self) -> FetchStats: ...


@dataclass(frozen=True)
class EvidenceDoc:
    """One source's text as the provider that read it labelled it.

    ``state`` is empty only for full text. Everything else carries the exact string
    its producer used — a ``fetch.Outcome``, ``oa.ABSTRACT_ONLY``, :data:`NO_TEXT`,
    :data:`NO_IDENTIFIER` — because a report that reworded it would be reporting the
    tool's opinion of what happened instead of what happened (product rule 2).

    ``winner`` and ``title`` are not part of the document; they are what the report
    owes the reader about it. The fetching stage names who answered, and a source
    with no text owes a finding a line to print. Both are per-source facts only the
    provider knows, so they travel with the document rather than being guessed at by
    the caller.
    """

    source_id: str
    text: str
    text_kind: TextKind
    state: str
    url: str
    step: int | None
    notes: tuple[str, ...] = ()
    from_cache: bool = False
    winner: str = ""  # attribution for the fetching stage line, "" when nobody was asked
    title: str = ""  # the finding's one line, for the states that owe one


class EvidenceProvider(Protocol):
    """One source family. Spec section 5.2, plus the retraction check of section 8.

    ``fetch`` returns a list because a post can be a thread and a page can be a set
    of them; v0.4.0's two providers each return exactly one document, or none when
    there was nothing to ask anyone for.
    """

    scheme: Scheme

    def resolve(self, ref: Reference) -> ResolveResult: ...

    def retraction(self, resolved: ResolveResult) -> Retraction | None: ...

    def fetch(self, ref: Reference, resolved: ResolveResult) -> list[EvidenceDoc]: ...


@dataclass(frozen=True)
class Providers:
    """The families a run can read from.

    ``social`` stays optional because a caller can assemble a run without it -- the
    routing below then reads a post as the page it renders to, which is what v0.3.0
    did. ``Engine`` always builds one.
    """

    academic: EvidenceProvider
    web: EvidenceProvider
    social: EvidenceProvider | None = None


def provider_for(ref: Reference, providers: Providers) -> EvidenceProvider:
    """Which family a reference belongs to, decided on the raw entry alone.

    Only an entry that is *nothing but* an address is routed by its address. A
    bibliography entry that prints a URL beside an author and a title is still an
    academic reference: routing on "contains a URL" would take every paper that
    prints its own DOI link away from Crossref. Such an entry is resolved as usual,
    and ``verify`` falls back to the address only when the indexes answer
    ``NOT_INDEXED`` — the rule spec section 8 already had.

    An identifier beats an address, which is why it is looked for first — the same
    order ``resolve.looks_unindexed`` uses, and for the same reason. An entry that is
    nothing but ``https://doi.org/10.…`` or ``https://arxiv.org/abs/…`` *is* an
    address, but the address names a record: routing it by its host would file it
    under ``url:``, skip the retraction check of spec section 8 entirely, and read
    the landing or abstract page instead of the open-access full text.

    A social address gets the social provider when there is one. A *numbered* entry
    that prints one does not: the printed ``[n] `` marker is part of the entry, so it
    is not "nothing but an address" and it goes to the indexes exactly as v0.3.0 sent
    it. Stripping the marker here would re-route every numbered bare-URL entry in
    every academic bibliography to decide one case that :func:`reader_for` already
    decides better — by then the resolving stage has filed the post under its own
    address, and the post is read as a post whatever its entry printed.
    """
    if find_doi(ref.raw) or find_arxiv_id(ref.raw):
        return providers.academic
    url = find_url(ref.raw)
    if url is None or not _is_bare(ref.raw, url):
        return providers.academic
    if providers.social is not None and is_social(url):
        return providers.social
    return providers.web


def reader_for(
    source_id: str | None,
    chosen: EvidenceProvider,
    providers: Providers,
) -> EvidenceProvider:
    """Which provider reads a source the resolving stage has already placed.

    Decided on the source id, not on the entry, for the reason :func:`checker_for`
    is: the resolving stage can place a reference somewhere its raw string never
    predicted, and by the fetching stage the id is the truth about what the source
    turned out to be. Each prefix names the family that can read it:

    * ``url:`` -- the fallback spec section 8 builds in. A reference no bibliographic
      index covers, whose entry prints its own address, is read from that address
      whoever resolved it.
    * ``doi:`` / ``arxiv:`` -- a record, read through the open-access chain the
      identifier exists for. Reading it up the web ladder instead would skip the
      arXiv PDF and the Unpaywall copy and settle for the landing page, costing the
      run a coverage grade for nothing (product rule 6), and would file that page's
      text under an identifier it did not come from.
    * no id at all -- a record that resolved and carries nothing to fetch it by. The
      academic provider is the one that says so (``NO_IDENTIFIER``), which is the
      answer v0.3.0 gave whatever the entry printed: the address is the fallback for
      a record the indexes do not cover, not for one they cover and have no
      identifier for.

    A ``url:`` source whose address is a post is the one place where the id alone
    does not settle it: the resolving stage files every address under ``url:``, and a
    post read up the web ladder is a rendering of a post rather than the post. The
    social family reads it instead when there is one, and reads a platform it does
    not support up that same ladder, so nothing loses a reading it used to have.

    Anything else keeps the provider that resolved it.
    """
    if source_id is None:
        return providers.academic
    if source_id.startswith(URL_PREFIX):
        url = source_id.removeprefix(URL_PREFIX)
        if providers.social is not None and is_social(url):
            return providers.social
        return providers.web
    if source_id.startswith((DOI_PREFIX, ARXIV_PREFIX)):
        return providers.academic
    return chosen


def checker_for(
    source_id: str | None,
    chosen: EvidenceProvider,
    providers: Providers,
) -> EvidenceProvider:
    """Which provider is asked whether a placed source carries a retraction notice.

    The mirror image of :func:`reader_for`, and for the same reason: the retraction
    stage iterates source *ids*, not entries, and a warm cache can hand an entry a
    DOI its raw string never predicted. A record is a record in the bibliographic
    indexes whoever resolved it, so the academic provider answers for it — asking a
    provider that has no retraction source would return ``None``, and ``None`` means
    "checked, and there was no notice" (product rule 2).

    ``arxiv:`` is listed beside ``doi:`` although the retraction stage never hands it
    one: that stage filters ``doi:`` before it calls here, so today the row is
    unreachable. It is written down anyway, because the day the stage widens the
    alternative is an ``arxiv:`` record answered by a family with nothing to answer
    with — the exact silent "checked, clean" this function exists to prevent.
    ``url:`` and ``None`` keep the provider that resolved them: a page and a record
    with no identifier have no notice to look for, and both families say so.
    """
    if source_id is not None and source_id.startswith((DOI_PREFIX, ARXIV_PREFIX)):
        return providers.academic
    return chosen


def is_social(url: str) -> bool:
    """Whether an address is a post on a platform rather than a document on a site."""
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    if any(host == known or host.endswith(f".{known}") for known in SOCIAL_HOSTS):
        return True
    if host.startswith(_MASTODON_PREFIXES) or host.endswith(_MASTODON_SUFFIX):
        return True
    return is_lemmy_host(host)


def is_lemmy_host(host: str) -> bool:
    """Whether a lowercased host, ``www.`` already gone, is one this tool reads as Lemmy."""
    return host.startswith(LEMMY_PREFIXES) or any(
        host == known or host.endswith(f".{known}") for known in LEMMY_HOSTS
    )


def doi_of(resolved: ResolveResult) -> str | None:
    """The DOI a resolve result settled on, or ``None`` when it settled on none."""
    best = resolved.best
    return best.doi if best is not None and best.doi else None


def identifiers(ref: Reference, resolved: ResolveResult) -> tuple[str | None, str | None]:
    """The DOI and the arXiv id a reference can be fetched by, in that order.

    A DOI wins: it is the identifier a record was resolved *to*, where an arXiv id is
    only what the entry happened to print. One helper rather than two call sites, so
    the id a source is filed under and the id it is fetched by can never disagree.
    """
    doi = doi_of(resolved)
    return doi, (None if doi else find_arxiv_id(ref.raw))


def honest_state(state: str, kind: str) -> tuple[str, tuple[str, ...]]:
    """A state for a source with no text, guaranteed to be one the report can carry.

    Every producer words its own state and it is passed through untouched. A string
    from outside the ``UNVERIFIED (...)`` family would make the finding
    unconstructible, so it becomes a note under a state that fits rather than
    crashing the run or disappearing from it.
    """
    if kind != "none" or state.startswith(UNVERIFIED_PREFIX):
        return state, ()
    return NO_TEXT, (state,) if state else ()


def _is_bare(raw: str, url: str) -> bool:
    """Whether the entry is the address and nothing else worth reading."""
    return not raw.replace(url, "", 1).strip(_PUNCTUATION)
