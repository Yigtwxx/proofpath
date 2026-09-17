"""The orchestrator: one document in, one :class:`~proofpath.report.Report` out.

Spec section 9. A run is two halves on purpose. ``prepare()`` parses the document,
pairs its citations, resolves the references, checks them for retractions and reads
whatever text is reachable; ``decide_all()`` then runs the models over what
``prepare()`` found. They are separately callable because they scale differently —
the first is I/O bound and the second is CPU bound — and because a front end wants
to show the first half's findings while the second is still running.

Two rules shape almost every line here:

* Nothing is skipped silently. A reference that could not be resolved, a page that
  could not be read, a marker in a style v0.1 does not pair — each one leaves a
  :class:`~proofpath.report.Finding` behind, so a run that learned little cannot
  look like a run that found nothing wrong (product rule 6).
* An honesty state is never reworded. The exact string from ``fetch.Outcome``,
  ``resolve.State`` or ``oa.ABSTRACT_ONLY`` is carried through to the report
  (product rule 2); the only strings this module writes itself are the two the
  stages below own, and both keep the ``UNVERIFIED (...)`` shape of section 15.

The embedder and the scorer arrive as factories and are not called until
:func:`decide_all` has found a claim with a source to check it against, so a document
whose references are all ghosts never pays for an ONNX session. :func:`verify` is
the two halves composed, for a caller that wants neither on its own.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import httpx

from proofpath import cache as cache_mod
from proofpath import claims as claims_mod
from proofpath import device as device_mod
from proofpath import ingest, pipeline, retrieval
from proofpath import resolve as resolve_mod
from proofpath.browser import Answer, ConsentGate
from proofpath.cache import Cache
from proofpath.claims import Claims
from proofpath.config import Config
from proofpath.document import CitationMarker, Claim, Document, Locator, Reference
from proofpath.entailment import OnnxNli, Scorer
from proofpath.events import (
    Cancelled,
    Emitted,
    Event,
    Listener,
    Note,
    Progress,
    StageEnd,
    StageStart,
)
from proofpath.fetch import Fetched, Fetcher, Outcome
from proofpath.judge import Judge, JudgeCost, JudgeItem, JudgeOpinion
from proofpath.models import Label, Passage, Verdict
from proofpath.oa import OpenAccess
from proofpath.paths import models_dir
from proofpath.pipeline import DEFAULT_THRESHOLDS, Thresholds
from proofpath.polite import PoliteClient, ProviderError, user_agent
from proofpath.providers import (  # noqa: F401 - two names re-exported, see below
    ARXIV_PREFIX,
    DOI_PREFIX,
    NO_IDENTIFIER,
    NO_TEXT,
    URL_PREFIX,
    EvidenceDoc,
    EvidenceProvider,
    FetchesOpenAccess,
    FetchesUrl,
    Providers,
    Resolves,
    Scheme,
    checker_for,
    identifiers,
    is_social,
    provider_for,
    reader_for,
    social,
)
from proofpath.providers.academic import AcademicProvider
from proofpath.providers.social import SocialProvider
from proofpath.providers.web import WebProvider
from proofpath.report import (
    LEVELS,
    STATE_WORDS,
    SUMMARY,
    ClaimResult,
    Coverage,
    Finding,
    Kind,
    Report,
    SourceStatus,
    Stage,
    TextKind,
    judge_detail,
    judge_unavailable,
    render_markdown,
)
from proofpath.resolve import Resolver, ResolveResult, Retraction, State
from proofpath.retrieval import Embedder, FastEmbedder, PassageIndex
from proofpath.settings_hints import BROWSER_SETTING

# Stage names, as the report and the TUI print them (spec section 13.2).
PARSING = "Parsing"
CLAIMS = "Claims"
RESOLVING = "Resolving"
RETRACTIONS = "Retractions"
FETCHING = "Fetching"
VERIFYING = "Verifying"
JUDGING = "Judging"
SUMMARISING = "Summarising"

# Attribution per stage. Parsing's and fetching's depend on the run and are
# computed; these three are fixed by which providers the stage consults.
CLAIMS_BY = "rules"
RESOLVERS_BY = "Crossref, Semantic Scholar"
# Whose ``resolve`` actually opens those two indexes. A provider declares its family
# and ``verify`` learns nothing else about it (spec 5.2), and this is the only family
# whose resolutions may be attributed to them: the web family answers out of the
# entry itself and the social family will answer its own platform.
INDEXED_FAMILY: Scheme = "academic"
RETRACTIONS_BY = "Retraction Watch"
FETCH_BY = "fetch ladder"
# What the resolving and retraction stages are attributed to when every reference
# came out of the cache: naming the providers would say they were consulted, and on
# a warm run nobody was (OPEN-ITEMS 10.8). The same word the fetching stage uses.
CACHE_BY = "cache"

# What a stage says when the network permission stopped it before it began. A
# count would be a claim about the references ("0 ghost" says they were checked
# and none was a ghost); a stage that ran on nothing has nothing to count.
# Short on purpose: it sits in a stage row's fixed, elided summary column. The
# setting that turns the network on rides on the once-per-run denied note instead
# (``Fetcher.network_note``, emitted below before the resolving stage).
NOT_ATTEMPTED = "not attempted (network not permitted)"

# What a stage is attributed to when nobody answered for it -- the fetching stage's
# own word for a run in which no provider won a document, reused rather than
# reworded. Not the same statement as ``NOT_ATTEMPTED``: that one says the run was
# not allowed to ask, this one says it asked nobody because there was nobody to ask.
NOBODY = "none"

# What a source says when every retraction provider failed. ``Resolver.retraction``
# returns ``None`` for "checked, and there is no notice" and raises for "nobody
# answered"; the two must never read alike, and the second is never cached -- a
# 7-day "not retracted" earned by an outage is exactly the absence-as-evidence
# product rule 2 forbids.
RETRACTION_UNAVAILABLE = "retraction check unavailable"

# ``NO_IDENTIFIER`` and ``NO_TEXT`` are imported above and re-exported from here --
# the only two names in that import this module does not use itself. Both are
# section 15 states that no provider answers with, because neither is a source's own
# word: each is a reading of what happened. They moved to ``providers`` in v0.4.0
# with the code that produces them, and the rest of the tool still reads them off
# this module.

# What a source says when its provider answered the fetching stage with no document
# at all. Unlike the two above, no provider produces this one: it is the core's
# reading of a provider's silence about a source it had already planned to read.
# Saying nothing about it would leave a status that looks fetched and textless and a
# stage count that has lost it (product rule 6).
NO_DOCUMENT = "UNVERIFIED (the provider returned no document)"

# What the verifying stage says when the models had nothing to run on. It is not the
# same statement as "0 claims": a document can cite plenty and still have no source
# text to check any of it against, and the report has to tell the two apart (rule 6).
NOTHING_TO_VERIFY = "nothing to verify"

# Said *before* the factories run rather than after: the first use of either model
# downloads it, which takes minutes, and a front end that says nothing here looks hung.
LOADING_MODELS = "loading models …"

# What ``Report.models`` names a model by when no model was loaded. A model name
# would claim the run used one; an empty string would read as a missing field.
NO_MODEL = "—"

# The prefix ``pipeline`` puts on a verdict a rule decided rather than the NLI model
# (spec section 10). Matched, never re-derived, so the two can never disagree.
NUMERIC_REASON = "numeric mismatch"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

POST_READER = "post"
# What the parsing stage says read a page: the fetch ladder, step by step.
PAGE_PARSER = "fetch ladder"
# What ``check --url`` says when the address is on a platform that is read but names
# no one post there -- a profile, a subreddit, a front page. There is no post to
# check and no set of links to check it against; the claim is the user's, and spec
# section 13 already has a way to give it. An address on any other host is a page,
# and a page is read as the document itself (``_page_document``).
NOT_A_POST = (
    "{url} is a page, not a post: give the claim as text with the address inside it "
    "(paste it in the TUI, or proofpath check - on the command line)"
)
# What a page is told when it prints citation markers but no reference list the
# bibliography finder recognises, and its links are used as its sources instead.
MARKERS_SET_ASIDE = (
    "{count} citation marker(s) found but no reference list; the page's links are its sources"
)
# A target that is one http(s) address and nothing else. Anchored, because an
# address *inside* a pasted paragraph is a source that paragraph cites, not the
# document itself.
_BARE_URL = re.compile(r"https?://\S+")

_PARSER_BY_KIND = {"pdf": "pymupdf", "docx": "python-docx", "post": POST_READER}
_PARSER_BY_SUFFIX = {".pdf": "pymupdf", ".docx": "python-docx"}
_TEXT_PARSER = "text"

# ``Resolves``, ``FetchesOpenAccess`` and ``FetchesUrl`` are imported above and
# re-exported from here. They describe what a provider needs from the outside world
# -- protocols rather than the concrete classes, because everything behind them is
# network-bound and a unit test may not touch the network -- so they live beside the
# providers that consume them.


@dataclass
class Engine:
    """Everything a run needs, assembled once and reused for every stage.

    The two model factories are deliberately not models: ``prepare()`` never calls
    them, so a run that is all I/O never loads an ONNX session, and a front end can
    build an engine long before it knows whether a document has any claim at all.
    """

    config: Config
    cache: Cache | None
    resolver: Resolves
    fetcher: FetchesUrl
    oa: FetchesOpenAccess
    gate: ConsentGate
    embedder: Callable[[], Embedder]
    scorer: Callable[[], Scorer]
    # k=1 and the thresholds beside it are the SciFact dev calibration of 2026-09-12
    # (spec section 14, docs/eval/2026-09-12-tiers.md): k=2 ties on accuracy and k=3
    # is worse, so the cheapest of the tied settings is the one that ships.
    k: int = 1
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    device: str = "cpu"
    # The opt-in second opinion. ``None`` is the default path, and the default path
    # makes zero LLM calls (spec section 11). Nothing downstream branches on it
    # except the judging stage, which does not run when it is absent.
    judge: Judge | None = None
    # Whether the judging stage may spend that judge. ``--summarize`` builds a judge
    # for the final summary alone, and spec section 11.1 caps the summary at one
    # call: escalating as well would quietly turn "one call" into several.
    escalate: bool = True
    # The polite HTTP client the social family reads posts through, and the one
    # ``target_document`` uses when the target *is* a post (spec section 6.2). Built
    # here by default so that every engine has one source family per spec section
    # 5.2 -- a run assembled by hand in a test reads a post exactly as a real one.
    client: PoliteClient = field(default_factory=PoliteClient, repr=False)
    _closers: list[Callable[[], None]] = field(default_factory=list, repr=False)
    # The models, once built. They live on the engine and nowhere else, so that
    # ``close()`` can release their native sessions while the interpreter is still
    # up: an onnxruntime session torn down at shutdown aborted the process (SIGABRT,
    # exit 134) once in three live runs on 2026-09-12, which the exit-code contract
    # of spec section 13.3 does not allow.
    _embedder: Embedder | None = field(default=None, repr=False)
    _scorer: Scorer | None = field(default=None, repr=False)
    # The source families this run can read from (spec section 5.2). Derived rather
    # than passed: the three collaborators above are what every caller already builds
    # and what every test already stubs, and a second way to say the same thing could
    # only ever disagree with the first.
    providers: Providers = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.providers = Providers(
            academic=AcademicProvider(self.resolver, self.oa),
            web=WebProvider(self.fetcher),
            social=SocialProvider(self.client, self.fetcher),
        )

    @classmethod
    def default(
        cls,
        config: Config,
        *,
        interactive: bool,
        browser: bool | None = None,
        no_cache: bool = False,
        prompt: Callable[[str, int | None], Answer] | None = None,
        k: int = 1,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        judge: Judge | None = None,
        escalate: bool = True,
    ) -> Engine:
        """The real wiring: the same one ``proofpath fetch`` builds, one layer up.

        ``interactive`` is passed down rather than sniffed here, so a caller that
        knows it is in CI (or under a pipe) gets ``ask`` treated as ``deny`` all the
        way through — the gate never prompts without a terminal (product rule 4).
        """
        closers: list[Callable[[], None]] = []
        if judge is not None:
            # Wired here, so closed here: the caller builds the judge but this is the
            # one place that knows when the run is over.
            closers.append(judge.close)
        try:
            gate = ConsentGate(
                config.permissions.install_browser,
                interactive=interactive,
                override=browser,
                prompt=prompt,
            )
            cache = None
            if not no_cache:
                cache = Cache()
                closers.append(cache.close)
            fetcher = Fetcher(config=config, gate=gate, cache=cache, interactive=interactive)
            closers.append(fetcher.close)
            email = config.contact.email
            client = PoliteClient(contact_email=email)
            closers.append(client.client.close)
            chain = OpenAccess(fetcher, client, contact_email=email, cache=cache)
            # The resolver builds its own polite client, which would then have no
            # owner to close it; handing it one keeps every socket on ``_closers``.
            resolver_client = httpx.Client(headers={"User-Agent": user_agent(email)}, timeout=20.0)
            closers.append(resolver_client.close)
            resolver = Resolver(contact_email=email, client=resolver_client)
            device_name = device_mod.onnx_device_name()
        except Exception:
            _close_all(closers)
            raise

        def embedder() -> Embedder:
            return FastEmbedder(
                EMBEDDING_MODEL,
                cache_dir=models_dir(),
                providers=device_mod.onnx_providers(),
            )

        def scorer() -> Scorer:
            # No ``providers`` and no ``onnx_file``: ``entailment.providers_for``
            # then picks them from the export it actually chose, which is what
            # keeps CoreML out of the int8 path (measured 3x slower there).
            return OnnxNli(cache_dir=models_dir())

        return cls(
            config=config,
            cache=cache,
            resolver=resolver,
            fetcher=fetcher,
            oa=chain,
            gate=gate,
            client=client,
            embedder=embedder,
            scorer=scorer,
            k=k,
            thresholds=thresholds,
            device=device_name,
            judge=judge,
            escalate=escalate,
            _closers=closers,
        )

    def get_embedder(self) -> Embedder:
        """The run's embedder, built at most once and owned by this engine."""
        if self._embedder is None:
            self._embedder = self.embedder()
            self._closers.append(partial(_close_model, self._embedder))
        return self._embedder

    def get_scorer(self) -> Scorer:
        """The run's NLI model, built at most once and owned by this engine."""
        if self._scorer is None:
            self._scorer = self.scorer()
            self._closers.append(partial(_close_model, self._scorer))
        return self._scorer

    def close(self) -> None:
        """Close everything this engine opened, newest first. Safe to call twice.

        The models go with it: their sessions are released here rather than left to
        whatever order the interpreter tears objects down in at exit.
        """
        _close_all(self._closers)
        self._closers.clear()
        self._embedder = None
        self._scorer = None

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _close_model(model: object) -> None:
    """Release one model's native session, if it has one to release.

    ``Embedder`` and ``Scorer`` are protocols about scoring, not about lifetime: a
    test double is a plain object and stays one. Only the real ONNX-backed models
    define ``close()``, and only they need it.
    """
    closer = getattr(model, "close", None)
    if callable(closer):
        closer()


def _close_all(closers: list[Callable[[], None]]) -> None:
    for closer in reversed(closers):
        # A client that fails to close must not hide the others, nor the error
        # that brought us here when this runs from an exception handler.
        with suppress(Exception):
            closer()


@dataclass
class Prepared:
    """Everything the I/O half found, and nothing the models have touched yet."""

    document: Document
    claims: Claims
    sources: dict[int, SourceStatus]  # by Reference.number; only references a claim cites
    texts: dict[str, str]  # source_id -> the text a claim will be checked against
    findings: list[Finding]
    stages: list[Stage]
    started: float  # time.monotonic() when the run began


def target_document(
    target: Path | str,
    *,
    name: str | None = None,
    client: PoliteClient | None = None,
    fetcher: FetchesUrl | None = None,
    network_allowed: bool = True,
) -> Document:
    """Parse a file, or take a string that is not a path as the document itself.

    A ``Path`` is always a file. A ``str`` is a file when one exists at that path; a
    string that is nothing but an address is the post at that address (spec section
    13's ``check --url``) when the host is a platform that is read, and otherwise
    the page at that address, read up the fetch ladder; anything else is pasted
    text, which is what makes ``proofpath verify "..."`` work without a flag to say
    which it was given.

    ``client`` is the engine's polite client, so a run reads a post on the same
    throttle as everything else it talks to; ``fetcher`` is the engine's ladder, so
    a page is read with the run's own permissions, consent and cache. A caller with
    no engine gets one of each of its own for the length of the read.

    ``network_allowed`` is the run's network permission, which the caller has already
    resolved. Reading a post or a page is a network call like any other, so a denied
    run must reach this with ``False`` and get the refusal rather than two HTTPS
    requests (spec section 7.1, product rule 4). Only a file and pasted text are
    readable without it, and neither consults this.
    """
    address = _bare_address(target)
    if address is not None:
        return _address_document(address, client, fetcher, network_allowed=network_allowed)[0]
    if isinstance(target, Path):
        return ingest.load(target)
    with suppress(OSError, ValueError):
        path = Path(target)
        if path.is_file():
            return ingest.load(path)
    return _pasted(target, name or "pasted text")


def _target(
    target: Path | str,
    *,
    name: str | None,
    client: PoliteClient | None,
    fetcher: FetchesUrl | None,
    network_allowed: bool,
) -> tuple[Document, tuple[str, ...]]:
    """:func:`target_document`, plus what the reader could not put into the document.

    The notes belong to the *reading*, not to the document: a quoted post nobody may
    read leaves no trace in the paragraphs it is missing from, so the reader is the
    only thing that ever knows it was there. :func:`prepare` reports them, which is
    what keeps a post that lost a paragraph from reading as a post that quoted
    nothing (product rule 2).

    Only a post or a page is read here. Everything else goes through
    :func:`target_document` itself, because that is the seam a caller replaces to
    hand the pipeline a document of its own, and a second way in would quietly stop
    honouring it.
    """
    address = _bare_address(target)
    if address is not None:
        return _address_document(address, client, fetcher, network_allowed=network_allowed)
    return (
        target_document(
            target, name=name, client=client, fetcher=fetcher, network_allowed=network_allowed
        ),
        (),
    )


def _bare_address(target: Path | str) -> str | None:
    """The target when it is one http(s) address and nothing else, else ``None``.

    The filesystem is asked first and a ``Path`` never asked at all, so a file that
    happens to be called ``https...`` is still read from disk.
    """
    if isinstance(target, Path):
        return None
    with suppress(OSError, ValueError):
        # A pasted paragraph is not a path, and asking the filesystem about one
        # can fail on its own (too long, embedded NUL) rather than answering.
        if Path(target).is_file():
            return None
    stripped = target.strip()
    return stripped if _BARE_URL.fullmatch(stripped) else None


def _address_document(
    url: str,
    client: PoliteClient | None,
    fetcher: FetchesUrl | None,
    *,
    network_allowed: bool,
) -> tuple[Document, tuple[str, ...]]:
    """What one address names: a post on a platform that is read, else a page.

    Both tests, because they answer different questions: ``is_social`` knows the
    listed platforms, and ``is_read_here`` knows a Mastodon status by its shape on
    an instance no list names. Either makes the address a platform's, and a
    platform's profile or front page is refused as one rather than read as a page.
    """
    if is_social(url) or social.is_read_here(url):
        return _post_document(url, client, network_allowed=network_allowed)
    return _page_document(url, fetcher, network_allowed=network_allowed)


def _page_document(
    url: str, fetcher: FetchesUrl | None, *, network_allowed: bool = True
) -> tuple[Document, tuple[str, ...]]:
    """The page at ``url`` as the document, or the reason there is none to check.

    The page is read up the same ladder its sources will be, so it meets the same
    permissions, the same consent gate and the same politeness -- and a page the
    ladder could not read is refused in the ladder's own words, one per honesty
    state (product rule 2): blocked, blocked by robots.txt, blocked with the browser
    not permitted, unreachable, provider unavailable. It is not a *source*
    (``counts_as_source=False``): a refusal here ends the run rather than skipping
    one of its references. The cache is climbed past because a cached copy is text
    alone, and a page's links are its bibliography (``Fetcher.fetch``).
    """
    if not network_allowed:
        raise ingest.IngestError(f"{url} could not be read: {Outcome.NETWORK_DENIED.value}")
    if fetcher is None:
        own_fetcher = Fetcher(config=Config(), interactive=False)
        try:
            fetched = own_fetcher.fetch(url, counts_as_source=False, use_cache=False)
        finally:
            own_fetcher.close()
    else:
        fetched = fetcher.fetch(url, counts_as_source=False, use_cache=False)
    if not fetched.ok:
        raise ingest.IngestError(_page_refusal(url, fetched))
    page = ingest.from_page(fetched)
    document = page.document
    # ``_pasted``'s rule with one difference: a page that prints its own reference
    # list cites by number, and any other page cites by linking. A marker alone does
    # not decide it -- "[1]" in one paragraph of a long page would otherwise cost the
    # page every link it carries, and a ``linked`` document reads a bracketed number
    # as prose, not as a citation (``claims.extract``).
    notes = page.notes
    if not document.references:
        document = ingest.with_carried_links(document, page.links)
        if markers := claims_mod.find_markers(document):
            # The override is said, not silent: a page whose list is under a heading
            # the bibliography finder does not know ("Sources", "Notes") has just had
            # its numbered citations set aside for its links (product rule 6).
            notes = (*notes, MARKERS_SET_ASIDE.format(count=len(markers)))
    return document, notes


def _post_document(
    url: str, client: PoliteClient | None, *, network_allowed: bool = True
) -> tuple[Document, tuple[str, ...]]:
    """The post at ``url``, or the reason there is no document to check at all.

    The three refusals are in the order the facts rank, not the order they are
    cheapest to test:

    * the network permission first, because reading a post *is* a network call and a
      denied run must make none at all (spec section 7.1). It is refused in the
      fetching stage's own words for the same permission -- one wording per fact.
    * then the platform. A post on a platform with no reader here is not "a page,
      not a post": saying that would tell the user something untrue about their own
      link and hide the hint that says what to do instead. Every platform spec
      section 6.2 lists has a reader since task 10.3 -- X's says the post cannot be
      read and the text should be pasted -- so this branch waits for the sixth.
    * then the shape. A profile or a front page on a platform that *is* read has no
      one post to check, and no set of links to check it against.

    A post that could not be read is refused too: there is nothing to report on, and
    a report about nothing would state a coverage it never had (product rule 6).
    """
    if not network_allowed:
        raise ingest.IngestError(f"{url} could not be read: {Outcome.NETWORK_DENIED.value}")
    if not social.is_read_here(url) and is_social(url):
        raise ingest.IngestError(f"{url} could not be read: {social.UNSUPPORTED_HINT}")
    if not social.is_post_url(url):
        raise ingest.IngestError(NOT_A_POST.format(url=url))
    if client is not None:
        read = social.read_post(url, client)
    else:
        own = PoliteClient()
        try:
            read = social.read_post(url, own)
        finally:
            own.client.close()
    if isinstance(read, social.Unreadable):
        raise ingest.IngestError(f"{url} could not be read: {read.hint}")
    document, dropped = ingest.from_post(read)
    # The platform's own notes first -- a quote nobody may read costs the document a
    # paragraph -- then what the ingest left out of the bibliography. Both are things
    # the post carried and the document does not, so both are reported (rule 2).
    return document, (*read.notes, *dropped)


def _page_refusal(url: str, fetched: Fetched) -> str:
    """The ladder's outcome, the step that decided it, and -- when the browser was
    the missing step -- what to type (permission hints, 2026-09-17).

    The step named is the last one that got an answer from the *page*, not the
    archive: "step 4 wayback: no snapshot" is true of every wall and says nothing
    about this one.
    """
    decisive = [note for note in fetched.notes if not note.startswith("step 4 wayback")]
    detail = f": {decisive[-1]}" if decisive else ""
    hint = ""
    if fetched.outcome is Outcome.BLOCKED_NO_BROWSER:
        hint = f" (to permit it: config set {BROWSER_SETTING})"
    return f"{url} could not be read: {fetched.outcome.value}{detail}{hint}"


def _pasted(text: str, name: str) -> Document:
    """Pasted text, with the addresses inside it as its bibliography when it prints
    no other one. A draft that numbers its citations keeps every one of them: the
    links are only read as the sources when there is nothing else claiming to be."""
    document = ingest.from_text(text, name=name, kind="text")
    if document.references or claims_mod.find_markers(document):
        return document
    return ingest.with_link_references(document)


def prepare(
    target: Path | str,
    engine: Engine,
    *,
    name: str | None = None,
    on_event: Listener | None = None,
    cancel: threading.Event | None = None,
) -> Prepared:
    """Parse, pair, resolve, check for retractions and read: everything but the models.

    ``name`` is what the report calls the document when the target is not a file: a
    caller reading from a pipe says ``"stdin"`` rather than letting every piped run
    be labelled "pasted text". A target that *is* a file keeps the file's own name.

    Stages run in order and each one is bracketed by a ``StageStart`` / ``StageEnd``
    pair with ``Progress`` per unit in between, so a front end can draw the run
    without knowing what a stage does. ``cancel`` is checked between units and
    raises :class:`~proofpath.events.Cancelled`; nothing partial is left behind,
    because the only writes this function makes go through ``Fetcher`` and
    ``OpenAccess``, whose cache writes are already atomic per source.
    """
    started = time.monotonic()
    emit: Listener = on_event if on_event is not None else _ignore
    findings: list[Finding] = []
    stages: list[Stage] = []

    def add(finding: Finding) -> None:
        findings.append(finding)
        emit(Emitted(finding))

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled("run cancelled")

    def opened(stage: str, by: str) -> float:
        check()
        emit(StageStart(name=stage, by=by))
        return time.monotonic()

    def closed(stage: str, by: str, summary: str, began: float) -> None:
        elapsed = time.monotonic() - began
        stages.append(Stage(name=stage, by=by, summary=summary, elapsed=elapsed))
        emit(StageEnd(name=stage, by=by, summary=summary, elapsed=elapsed))

    # The permission is resolved once, before anything at all is read -- the
    # document included. ``check --url`` reads its post over the network, so a
    # permission consulted after the parsing stage would already have been overtaken
    # by two HTTPS calls on a run that was told not to make any (spec 7.1, rule 4).
    # The note it prints stays below, where the run announces the stages it is about
    # to skip; what moves up here is only the reading of the permission.
    allowed = engine.fetcher.network_allowed
    denied_note = engine.fetcher.network_note

    # --- 1. parsing ---------------------------------------------------------
    reader = _parser_for(target)
    began = opened(PARSING, reader)
    document, parse_notes = _target(
        target, name=name, client=engine.client, fetcher=engine.fetcher, network_allowed=allowed
    )
    # What read the document, as the stage closes: the kind says for a file, and a
    # page is read by the ladder whatever it turned out to hold -- a fetched PDF
    # opens and closes as ``fetch ladder``, and pasted text with links in it,
    # which shares the ``linked`` kind, still closes as text.
    parser = reader if reader == PAGE_PARSER else _PARSER_BY_KIND.get(document.kind, _TEXT_PARSER)
    for error in document.errors:
        add(
            _finding(
                Kind.PARSE_ERROR,
                Locator(line=1, page=error.page),
                "page could not be parsed",
                detail=(error.detail,),
            )
        )
    for note in parse_notes:
        # Something the parser could not put into the text: a quoted post nobody may
        # read, whose paragraph and links are missing from the document. Reporting it
        # is what keeps the loss visible instead of it reading as a post that quoted
        # nothing (product rule 2).
        add(_finding(Kind.PARSE_ERROR, Locator(line=1), note))
    if document.kind in ("post", "page") and not document.references:
        # A post or a page that links to nothing has nothing behind it to check.
        # Reporting it as a document with no findings would make "nothing was
        # verified" look exactly like "everything checked out" (product rule 6).
        add(
            _finding(
                Kind.PARSE_ERROR,
                Locator(line=1),
                f"{document.kind} carries no links; nothing to verify against",
            )
        )
    if not document.paragraphs and not document.references:
        # Every page failed, or the file is a scan with no text layer. Without this
        # the run would report nothing at all and read as clean (product rule 6).
        add(
            _finding(
                Kind.PARSE_ERROR,
                Locator(line=1, page=1),
                "no text could be extracted from the document",
            )
        )
    emit(Progress(name=PARSING, done=1, total=1, detail=document.name))
    closed(
        PARSING,
        parser,
        f"{document.pages} pages, {len(document.references)} refs",
        began,
    )

    # --- 2. claims ----------------------------------------------------------
    began = opened(CLAIMS, CLAIMS_BY)
    claims = claims_mod.extract(document)
    # Empty in v0.2: `claims` pairs both the styles it detects (numeric and author-year),
    # so what is left here is a style a later version will detect but still not pair --
    # footnote-only citations, superscript letters. The loop stays because the day one of
    # those is detected it has to be *reported*, not counted as a document with no
    # citations (product rule 6).
    for marker in claims.unsupported:
        add(
            _finding(
                Kind.UNSUPPORTED_STYLE,
                _marker_locator(document, marker),
                "citation style is not paired in this version",
            )
        )
    for marker in claims.unresolved:
        add(
            _finding(
                Kind.UNRESOLVED_MARKER,
                _marker_locator(document, marker),
                "citation marker names no bibliography entry",
                detail=(marker.text,),
            )
        )
    emit(Progress(name=CLAIMS, done=1, total=1, detail=f"{len(claims.claims)} claims"))
    closed(
        CLAIMS,
        CLAIMS_BY,
        f"{len(claims.markers)} citations, {len(claims.unresolved)} unresolved",
        began,
    )

    # Every stage below is network-bound — the resolver and Retraction Watch as much
    # as the ladder — so a denied run says so once, here, and then calls nobody. The
    # permission itself was read before the parsing stage; see the comment there.
    if not allowed and denied_note:
        emit(Note(denied_note))

    # --- 3. resolving -------------------------------------------------------
    began = opened(RESOLVING, RESOLVERS_BY)
    # By printed number only. ``claims`` already guarantees that every cited number
    # is one an entry prints (and reports the rest as unresolved markers), so a
    # positional fallback could only ever point a claim at the wrong source.
    by_number = {reference.number: reference for reference in document.references}
    cited = sorted({number for claim in claims.claims for number in claim.cited_refs})
    sources: dict[int, SourceStatus] = {}
    # number -> the result it resolved to, for every reference stage 5 still has
    # something to ask a provider for. A number with an entry but no identifier at
    # all is in here too, so it is reported rather than dropped between the stages.
    pending: dict[int, ResolveResult] = {}
    # number -> the provider that resolved it, so stages 4 and 5 ask the same one.
    chosen: dict[int, EvidenceProvider] = {}
    tally: Counter[State] = Counter()
    resolved_total = 0
    # Split three ways for the stage line, because the three are different claims:
    # what the cache answered, what the bibliographic indexes were actually asked,
    # and (whatever is left over) what a family answered without asking anyone.
    resolved_from_cache = 0
    resolved_from_indexes = 0
    for index, number in enumerate(cited, start=1):
        reference = by_number.get(number)
        if reference is None:
            # Only reachable without a bibliography at all (a pasted paragraph):
            # ``claims`` cannot check a number against a list that is not there.
            add(
                _finding(
                    Kind.UNRESOLVED_MARKER,
                    _cited_locator(claims, number),
                    "citation marker names no bibliography entry",
                    detail=("the document prints no bibliography",),
                )
            )
            emit(Progress(name=RESOLVING, done=index, total=len(cited), detail=f"[{number}]"))
            check()
            continue
        if not allowed:
            # Nothing is asked of anyone. The reference is not a ghost, not
            # ambiguous and not missing: it is unlooked-at, and that is what it says.
            sources[number] = _not_attempted(reference, denied_note)
            add(
                _finding(
                    Kind.UNVERIFIED,
                    reference.locator,
                    "the source was not looked up",
                    state=Outcome.NETWORK_DENIED.value,
                    reference=reference,
                    detail=(denied_note,) if denied_note else (),
                )
            )
            emit(Progress(name=RESOLVING, done=index, total=len(cited), detail=f"[{number}]"))
            check()
            continue
        # Which family this entry belongs to, decided once and used by all three
        # stages below. The core never learns what the answer means (spec 5.2).
        provider = provider_for(reference, engine.providers)
        chosen[number] = provider
        # The cache answers for a reference, not for a source: a reference has no
        # source id until it resolves. A miss costs exactly what it used to.
        stored = None if engine.cache is None else engine.cache.get_resolution(reference.raw)
        if stored is None:
            result = provider.resolve(reference)
            if provider.scheme == INDEXED_FAMILY:
                resolved_from_indexes += 1
            if engine.cache is not None:
                # ``put_resolution`` drops UNVERIFIED (provider unavailable) itself:
                # an outage is not knowledge about the reference (product rule 2).
                engine.cache.put_resolution(reference.raw, result)
        else:
            result = stored
            resolved_from_cache += 1
        resolved_total += 1
        tally[result.state] += 1
        placed = _placed(result, reference, add)
        if placed.fetchable:
            pending[number] = result
        sources[number] = SourceStatus(
            reference=reference,
            resolve=result,
            retraction=None,
            source_id=placed.source_id,
            text_kind="none",
            state=placed.state,
            fetch_step=None,
            url=placed.url,
            from_cache=False,
            resolve_from_cache=stored is not None,
        )
        emit(Progress(name=RESOLVING, done=index, total=len(cited), detail=f"[{number}]"))
        check()
    # "0 ghost" would be a claim about the references; nothing was looked at.
    closed(
        RESOLVING,
        _resolving_attribution(
            total=resolved_total, indexes=resolved_from_indexes, cached=resolved_from_cache
        ),
        _resolve_summary(tally) if allowed else NOT_ATTEMPTED,
        began,
    )

    # --- 4. retractions -----------------------------------------------------
    began = opened(RETRACTIONS, RETRACTIONS_BY)
    dois = [
        (number, status.source_id.removeprefix(DOI_PREFIX))
        for number, status in sources.items()
        if status.source_id is not None and status.source_id.startswith(DOI_PREFIX)
    ]
    retracted = 0
    unavailable = 0
    checked_from_cache = 0
    for index, (number, doi) in enumerate(dois, start=1):
        hit = None if engine.cache is None else engine.cache.get_retraction(doi)
        if hit is None:
            try:
                # The DOI is the cache's key, not the provider's argument: the
                # provider asks about the record it resolved, which is where that
                # DOI came from in the first place.
                notice = _resolved_retraction(engine, chosen[number], sources[number])
            except ProviderError as exc:
                # Nothing was learned, so nothing is stored and nothing is claimed.
                unavailable += 1
                status = sources[number]
                sources[number] = replace(
                    status, notes=(*status.notes, f"{RETRACTION_UNAVAILABLE}: {exc}")
                )
                emit(Progress(name=RETRACTIONS, done=index, total=len(dois), detail=doi))
                check()
                continue
            if engine.cache is not None:
                engine.cache.put_retraction(doi, notice)
        else:
            notice = hit.notice
            checked_from_cache += 1
        if notice is not None:
            retracted += 1
            sources[number] = replace(sources[number], retraction=notice)
            when = f" {notice.date}" if notice.date else ""
            add(
                _finding(
                    Kind.RETRACTED,
                    sources[number].reference.locator,
                    f"cited source was retracted{when}",
                    reference=sources[number].reference,
                    source_id=sources[number].source_id,
                    detail=(notice.label,),
                )
            )
        emit(Progress(name=RETRACTIONS, done=index, total=len(dois), detail=doi))
        check()
    # Same distinction as above: "none" means the notices were checked and there
    # were none, which is not what a denied run knows.
    closed(
        RETRACTIONS,
        _retraction_attribution(total=len(dois), cached=checked_from_cache),
        _retraction_summary(retracted, unavailable) if allowed else NOT_ATTEMPTED,
        began,
    )

    # --- 5. fetching --------------------------------------------------------
    began = opened(FETCHING, FETCH_BY)
    texts: dict[str, str] = {}
    winners: list[str] = []
    counts: Counter[TextKind] = Counter()
    # Only the ladder's step 3 writes to the gate's log, so it can only fill here.
    # Whatever it says — a download, a failed install, a permission that could not
    # be saved — belongs in the report rather than on someone's terminal.
    logged = len(engine.gate.install_log)

    def record(number: int, read: EvidenceDoc) -> None:
        """One document a provider read, into the status, the texts and a finding."""
        status = sources[number]
        counts[read.text_kind] += 1
        if read.winner and read.winner not in winners:
            winners.append(read.winner)
        sources[number] = replace(
            status,
            text_kind=read.text_kind,
            state=read.state,
            fetch_step=read.step,
            url=read.url or status.url,
            from_cache=read.from_cache,
            # Appended, not replaced: an earlier stage may have recorded something
            # about this source (a retraction check nobody answered) that the fetch
            # knows nothing about and must not erase.
            notes=(*status.notes, *read.notes),
        )
        if read.text and status.source_id is not None:
            texts[status.source_id] = read.text
        if read.text_kind == "abstract":
            add(
                _finding(
                    Kind.ABSTRACT_ONLY,
                    status.reference.locator,
                    "only the abstract could be read",
                    reference=status.reference,
                    source_id=status.source_id,
                    fetch_step=read.step,
                    detail=read.notes,
                )
            )
        elif read.text_kind == "none":
            add(
                _finding(
                    Kind.UNVERIFIED,
                    status.reference.locator,
                    read.title,
                    state=read.state,
                    reference=status.reference,
                    source_id=status.source_id,
                    fetch_step=read.step,
                    detail=read.notes,
                )
            )
        if (
            engine.cache is not None
            and read.text_kind == "abstract"
            and not read.from_cache
            and read.source_id.startswith(URL_PREFIX)
        ):
            # The ladder filed the page as the "fulltext" it was asked for; the word
            # count now says what the url: row really holds, and the next run must
            # not read an abstract back as full text (mirrors ``oa._climb_all``).
            engine.cache.set_text_kind(read.source_id, "abstract")

    for index, (number, result) in enumerate(pending.items(), start=1):
        # A list because a post can be a thread; v0.4.0's two providers answer one
        # for one, so the loop runs exactly once per pending reference. An empty
        # list is not one of the answers: a source that was planned and then
        # silently dropped would keep the ``state=""`` of a row nobody touched,
        # which reads as fetched and simply textless (product rule 6).
        docs = _read_source(engine, chosen[number], sources[number], result, allowed=allowed)
        for read in docs or [_nothing_read(sources[number])]:
            record(number, read)
        logged = _forward_log(engine.gate, logged, emit)
        emit(Progress(name=FETCHING, done=index, total=len(pending), detail=f"[{number}]"))
        check()
    _forward_log(engine.gate, logged, emit)
    closed(
        FETCHING,
        ", ".join(winners) or NOBODY,
        f"{counts['fulltext']} full text, {counts['abstract']} abstract, "
        f"{counts['none']} unverified"
        if allowed
        else NOT_ATTEMPTED,
        began,
    )

    return Prepared(
        document=document,
        claims=claims,
        sources=sources,
        texts=texts,
        findings=findings,
        stages=stages,
        started=started,
    )


def decide_all(
    prepared: Prepared,
    engine: Engine,
    *,
    on_event: Listener | None = None,
    cancel: threading.Event | None = None,
) -> Report:
    """Run the models over what ``prepare()`` found, and assemble the report.

    The second half of section 9: chunk and embed every source that has text, decide
    every claim that cites it, and turn the verdicts into findings. Three properties
    are worth more than the code that produces them:

    * The models are built lazily and exactly once. A document whose sources are all
      ghosts never opens an ONNX session, and the download that may sit behind the
      first call is announced before it starts rather than after it finishes.
    * A source is chunked once per run and then asked about every claim that cites
      it, and both its chunks and its verdicts are cached (spec sections 11, 12), so
      a second run of the same document costs nothing.
    * Every cache write is its own transaction and ``cancel`` is checked before each
      source and each claim, so a run that stops leaves whole rows and no partial
      ones: three finished claims are three verdicts in the file, not two and a half.
      A cancelled run also *keeps* what it decided -- the ``Cancelled`` it raises
      carries the partial report, marked as partial (product rule 6).
    """
    emit: Listener = on_event if on_event is not None else _ignore
    findings = list(prepared.findings)
    stages = list(prepared.stages)
    results: list[ClaimResult] = []
    # Paragraph-scoped claims, collected as they are judged with the reference each
    # verdict was about: the group's own note counts sentences and sources, so it can
    # only be written once every member has an answer.
    grouped: dict[str, list[tuple[Claim, int, Verdict]]] = {}

    def add(finding: Finding) -> None:
        findings.append(finding)
        emit(Emitted(finding))

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled("run cancelled")

    jobs = _jobs(prepared)
    total = sum(len(job.claims) for job in jobs)

    began = time.monotonic()
    emit(StageStart(name=VERIFYING, by=engine.device))
    embedder: Embedder | None = None
    scorer: Scorer | None = None
    cached = 0
    cancelled = False
    try:
        if jobs:
            # A run told to stop before it started must not pay for two ONNX sessions.
            check()
            emit(Note(LOADING_MODELS, transient=True))
            # Built through the engine, not called as factories: the engine keeps
            # them, so nothing here outlives ``Engine.close()`` holding a session.
            embedder = engine.get_embedder()
            scorer = engine.get_scorer()
            # Everything that can change a verdict goes into the key, so a run with
            # other thresholds or another model never reads back an answer it did
            # not give.
            model_key = cache_mod.model_id(
                nli=scorer.name,
                embedder=embedder.name,
                k=engine.k,
                thresholds=engine.thresholds,
            )
            done = 0
            for job in jobs:
                check()
                # Once per source, before any of its claims: the passages are
                # embedded once a run and then asked about every claim that cites
                # them, and the digest inside keeps the stored chunks -- and the
                # verdicts quoting them -- honest about the text they came from.
                index = _index_for(engine.cache, job, embedder)
                status = prepared.sources[job.number]
                for claim in job.claims:
                    check()
                    result = _decide_claim(engine, job, claim, index, embedder, scorer, model_key)
                    results.append(result)
                    if result.from_cache:
                        cached += 1
                    finding = _verdict_finding(claim, status, result.verdict)
                    if finding is not None:
                        add(finding)
                    if claim.group is not None:
                        grouped.setdefault(claim.group, []).append(
                            (claim, job.number, result.verdict)
                        )
                    done += 1
                    emit(
                        Progress(
                            name=VERIFYING, done=done, total=total, detail=claim.locator.label()
                        )
                    )
    except Cancelled:
        # Not re-raised here: the stage is closed and the report assembled first, so
        # what the run did learn arrives with the exception instead of being lost.
        cancelled = True

    for group, members in grouped.items():
        add(_group_finding(group, members))
    summary = _verify_summary(results, cached) if jobs else NOTHING_TO_VERIFY
    elapsed = time.monotonic() - began
    stages.append(Stage(name=VERIFYING, by=engine.device, summary=summary, elapsed=elapsed))
    emit(StageEnd(name=VERIFYING, by=engine.device, summary=summary, elapsed=elapsed))

    # Stage 7, and only when asked for: the judge re-reads the verdicts the models
    # were least sure of and says what it thinks, beside them (spec section 9 step 8).
    # A run already told to stop is not started on a second round of network calls.
    judge_cost: JudgeCost | None = None
    if engine.judge is not None and engine.escalate and not cancelled:
        judged = _judging(engine, results, findings, emit=emit, check=check)
        results, findings = judged.results, judged.findings
        stages.append(judged.stage)
        # A copy, not the client's own: ``JudgeCost`` is mutable and the judge goes on
        # spending it -- on the summary, or on the next document -- so an alias here
        # would let a finished report's stated cost keep changing underneath it.
        judge_cost = replace(judged.cost)
        cancelled = judged.cancelled

    report = Report(
        document=prepared.document,
        claims=len(prepared.claims.claims),
        markers=len(prepared.claims.markers),
        sources=tuple(prepared.sources[number] for number in sorted(prepared.sources)),
        results=tuple(results),
        findings=tuple(sorted(findings, key=_ordered)),
        coverage=_coverage(prepared, engine),
        stages=tuple(stages),
        models=_models(embedder, scorer, engine),
        # Zero unless a judge ran: the default path calls nobody (spec section 11).
        api_calls=0 if judge_cost is None else judge_cost.calls,
        judge_cost=judge_cost,
        elapsed=time.monotonic() - prepared.started,
        tier_note=pipeline.tier_note(engine.thresholds),
        cancelled=cancelled,
    )
    if cancelled:
        raise Cancelled("run cancelled", report=report)
    return report


def verify(
    target: Path | str,
    engine: Engine,
    *,
    name: str | None = None,
    summarize: bool = False,
    on_event: Listener | None = None,
    cancel: threading.Event | None = None,
) -> Report:
    """One whole run: the I/O half and then the model half, over the same engine.

    The two are still separately callable -- a front end shows what ``prepare()``
    found while ``decide_all()`` is still running -- and this is the composition a
    caller that wants neither half on its own asks for.

    ``summarize`` adds the one optional call of spec section 11.1, and it is here
    rather than in ``decide_all`` for the reason the spec gives: the summary is
    written *over a finished report*, so there has to be a finished report first. It
    needs ``engine.judge``; a caller that asks for it without one is a bug, not a
    document the tool could not check.
    """
    prepared = prepare(target, engine, name=name, on_event=on_event, cancel=cancel)
    report = decide_all(prepared, engine, on_event=on_event, cancel=cancel)
    if not summarize:
        return report
    return _summarise(report, engine, started=prepared.started, on_event=on_event)


def _summarise(
    report: Report, engine: Engine, *, started: float, on_event: Listener | None
) -> Report:
    """The model-written paragraph, and what it cost, on a report that is already done.

    Three things make this safe to add to a finished run (spec section 11.1). The
    report's own markdown is the only input, so the judge never sees a source and has
    nothing to re-decide. ``results`` and ``findings`` are handed through untouched,
    so no wording here can reach a verdict. And it is one call: the stage says how
    many were made, so a second one could not hide.

    A provider that will not answer costs the paragraph and says so in the stage
    summary -- the same surface, and the same sentence, the judging stage uses -- so
    ``-q``, ``--format json`` and the markdown report all still tell a silent
    provider from a report with nothing to add (product rules 2 and 6). The sentence
    counts no calls: ``cost.calls`` counts answers, and one request was made.
    """
    judge = engine.judge
    if judge is None:
        raise ValueError("a summary needs a judge on the engine")
    emit: Listener = on_event if on_event is not None else _ignore
    began = time.monotonic()
    emit(StageStart(name=SUMMARISING, by=judge.name))
    before = replace(judge.cost)  # the running total, as it stood before this call
    text = judge.summarize(render_markdown(report))
    calls = judge.cost.calls - before.calls
    if text:
        summary = (
            f"{len(text.split())} words, {_calls(calls)}, "
            f"{judge.cost.prompt_tokens - before.prompt_tokens:,} prompt · "
            f"{judge.cost.completion_tokens - before.completion_tokens:,} completion tokens"
        )
    else:
        # No note beside it: the stage carries the sentence, and each front end prints
        # its own ``summary`` line from the finished report. A note as well would say
        # the same thing twice in the same terminal.
        summary = f"summary {judge_unavailable(calls, judge.detail, what=SUMMARY)}"
    elapsed = time.monotonic() - began
    emit(StageEnd(name=SUMMARISING, by=judge.name, summary=summary, elapsed=elapsed))
    return replace(
        report,
        # ``""`` and ``None`` are different answers: nobody asked, and nobody
        # answered. Only the second one is a state the run has to report.
        summary=text,
        # Named only when there is prose to attribute: a paragraph nobody wrote has
        # no author, and a model named beside an empty summary would read as one.
        summary_model=judge.name if text else None,
        stages=(
            *report.stages,
            Stage(name=SUMMARISING, by=judge.name, summary=summary, elapsed=elapsed),
        ),
        api_calls=judge.cost.calls,
        # A copy, not the client's own: ``JudgeCost`` is mutable and the client keeps
        # adding to it for the next document, so an alias here would let a finished
        # report's stated cost go on changing after the run that earned it was over.
        judge_cost=replace(judge.cost),
        # The extra call is part of what the run took; a footer that left it out
        # would price the summary at nothing.
        elapsed=time.monotonic() - started,
    )


# --- the verifying stage ----------------------------------------------------------

# The line the group note carries. Quoted from section 15's PARAGRAPH-SCOPED row.
PARAGRAPH_SCOPED_TITLE = "citation supports a paragraph, not one sentence"


@dataclass(frozen=True)
class _Job:
    """One source that has text, and the claims that cite it."""

    number: int
    source_id: str
    text: str
    claims: tuple[Claim, ...]


def _jobs(prepared: Prepared) -> list[_Job]:
    """What the models have to do, in bibliography order.

    A claim whose reference has no text is not in here and gets no result at all. Its
    source-level finding already says why it could not be checked, and a second,
    model-shaped "we could not tell" would report one gap as two (product rule 2).
    """
    jobs: list[_Job] = []
    for number in sorted(prepared.sources):
        source_id = prepared.sources[number].source_id
        if source_id is None:
            continue
        text = prepared.texts.get(source_id)
        if not text:
            continue
        claims = tuple(claim for claim in prepared.claims.claims if number in claim.cited_refs)
        if claims:
            jobs.append(_Job(number, source_id, text, claims))
    return jobs


def _index_for(cache: Cache | None, job: _Job, embedder: Embedder) -> PassageIndex:
    """A source's passages and their vectors, from the cache when they still fit it.

    The digest is the whole check: chunks cut from text the source no longer serves
    would be quoted back as evidence it never carried (product rule 1), so text that
    has changed is chunked again rather than read out of the cache.
    """
    digest = cache_mod.sha256_text(job.text)
    if cache is not None:
        stored = cache.get_chunks(job.source_id, embedder.name, text_sha256=digest)
        if stored is not None:
            passages, vectors = stored
            return PassageIndex.from_vectors(passages, vectors)
    passages = [
        Passage(sentence, job.source_id, ordinal)
        for ordinal, sentence in enumerate(retrieval.split_sentences(job.text))
    ]
    if not passages:
        # Nothing to embed and nothing worth storing. An empty index answers every
        # claim with NEI, which is all a source with no sentences honestly supports.
        return PassageIndex(dim=embedder.dim)
    vectors = embedder.embed([passage.text for passage in passages])
    if cache is not None:
        cache.put_chunks(job.source_id, embedder.name, passages, vectors, text_sha256=digest)
    return PassageIndex.from_vectors(passages, vectors)


def _decide_claim(
    engine: Engine,
    job: _Job,
    claim: Claim,
    index: PassageIndex,
    embedder: Embedder,
    scorer: Scorer,
    model_key: str,
) -> ClaimResult:
    """One claim against one source: the cache first, the models only on a miss.

    The write is a transaction of its own, so the claim is either wholly decided and
    stored or not stored at all -- which is what lets a cancelled run leave exactly
    as many verdicts behind as it finished claims.
    """
    key = cache_mod.claim_hash(claim.text)
    stored = (
        None if engine.cache is None else engine.cache.get_verdict(key, job.source_id, model_key)
    )
    if stored is not None:
        return ClaimResult(claim, job.number, job.source_id, stored, from_cache=True)
    verdict = pipeline.decide_indexed(
        claim.text, index, embedder, scorer, k=engine.k, thresholds=engine.thresholds
    )
    if engine.cache is not None:
        engine.cache.put_verdict(key, job.source_id, model_key, verdict)
    return ClaimResult(claim, job.number, job.source_id, verdict, from_cache=False)


def _verdict_finding(claim: Claim, status: SourceStatus, verdict: Verdict) -> Finding | None:
    """One judged claim as a finding, or ``None`` when the source backs the claim up.

    Only the two refused shapes assert something about the document, and both carry
    the passage they rest on -- ``Verdict`` refuses to exist without one, so rule 1
    is kept by construction rather than by this function remembering to. ``NEI`` is a
    note and may carry no passage at all: "the source does not say" is a different
    statement from "the source says otherwise" (rule 2).
    """
    if verdict.label is Label.SUPPORTED:
        return None
    kind: Kind
    detail: tuple[str, ...] = ()
    if verdict.label is Label.REFUTED and verdict.reason.startswith(NUMERIC_REASON):
        # A rule decided this one and named both figures; the reason is the finding's
        # only note, and ``report`` reads the carets back out of it.
        kind = Kind.NUMERIC_MISMATCH
        title = "claim contradicts the cited source"
        detail = (verdict.reason,)
    elif verdict.label is Label.REFUTED:
        kind = Kind.NOT_SUPPORTED
        title = "claim is not supported by the cited source"
    else:
        kind = Kind.NEI
        title = "source neither supports nor contradicts the claim"
    return Finding(
        kind=kind,
        level=LEVELS[kind],
        locator=claim.locator,
        title=title,
        state=STATE_WORDS[kind],
        reference=status.reference,
        claim=claim,
        verdict=verdict,
        source_id=status.source_id,
        fetch_step=status.fetch_step,
        tier=verdict.tier,
        detail=detail,
        group=claim.group,
    )


def _group_finding(group: str, members: Sequence[tuple[Claim, int, Verdict]]) -> Finding:
    """The one note saying why a paragraph's sentences were judged one by one.

    It carries no passage and no claim of its own: the group explains the shape of
    the citation, and the evidence lives on the member findings, each decided on its
    own sentence. That is what keeps a single sentence from being given a confident
    verdict on behalf of the paragraph it sits in (spec section 9).
    """
    return Finding(
        kind=Kind.PARAGRAPH_SCOPED,
        level=LEVELS[Kind.PARAGRAPH_SCOPED],
        locator=members[0][0].locator,
        title=PARAGRAPH_SCOPED_TITLE,
        state=STATE_WORDS[Kind.PARAGRAPH_SCOPED],
        reference=None,
        claim=None,
        verdict=None,
        source_id=None,
        fetch_step=None,
        tier=None,
        detail=(_group_detail(members),),
        group=group,
    )


def _group_detail(members: Sequence[tuple[Claim, int, Verdict]]) -> str:
    """``"3 sentences: 1 supported, 1 NEI, 1 not supported"``; zero counts left out.

    Sentences are counted, not verdicts. A marker naming two references gives every
    sentence of its paragraph two verdicts, and calling that four sentences would
    misstate what the author actually wrote; the second source is named instead:
    ``"2 sentences against 2 sources: 4 NEI"``.
    """
    sentences = len({(claim.paragraph, claim.sentence) for claim, _, _ in members})
    sources = len({number for _, number, _ in members})
    tally = Counter(verdict.label for _, _, verdict in members)
    parts = [
        f"{tally[label]} {word}"
        for label, word in (
            (Label.SUPPORTED, "supported"),
            (Label.NEI, "NEI"),
            (Label.REFUTED, "not supported"),
        )
        if tally[label]
    ]
    head = f"{sentences} {'sentence' if sentences == 1 else 'sentences'}"
    if sources > 1:
        head += f" against {sources} sources"
    return f"{head}: " + ", ".join(parts)


def _verify_summary(results: Sequence[ClaimResult], cached: int) -> str:
    """``"12 claims: 8 supported, 1 not supported, 3 NEI"``, plus what came for free."""
    tally = Counter(item.verdict.label for item in results)
    noun = "claim" if len(results) == 1 else "claims"
    summary = (
        f"{len(results)} {noun}: {tally[Label.SUPPORTED]} supported, "
        f"{tally[Label.REFUTED]} not supported, {tally[Label.NEI]} NEI"
    )
    return f"{summary}, {cached} cached" if cached else summary


def _coverage(prepared: Prepared, engine: Engine) -> Coverage:
    """How much of the bibliography was read, counted once over every cited source.

    Every reference lands in exactly one bucket, which is the invariant
    ``Coverage.pct`` rests on. ``unverified`` is the remainder rather than a tally of
    its own, so a failure nobody thought to count still shows up as unread instead of
    quietly improving the percentage (product rule 6).
    """
    statuses = list(prepared.sources.values())
    fulltext = sum(1 for status in statuses if status.text_kind == "fulltext")
    abstract = sum(1 for status in statuses if status.text_kind == "abstract")
    return Coverage(
        references=len(statuses),
        fulltext=fulltext,
        abstract=abstract,
        unverified=len(statuses) - fulltext - abstract,
        # Verbatim, and counted: the summary says how much was read, the reasons say
        # why the rest was not, in the words section 15 gives them.
        reasons=dict(Counter(status.state for status in statuses if status.state)),
        browser_skipped=engine.gate.skipped,
        network_denied=not engine.fetcher.network_allowed,
    )


def _models(embedder: Embedder | None, scorer: Scorer | None, engine: Engine) -> dict[str, str]:
    """What decided this run. ``NO_MODEL`` where none was loaded, never a name."""
    thresholds = engine.thresholds
    models = {
        "nli": NO_MODEL if scorer is None else scorer.name,
        "embedder": NO_MODEL if embedder is None else embedder.name,
        "device": engine.device,
        "thresholds": (
            f"decide={thresholds.decide:g};high={thresholds.high:g};medium={thresholds.medium:g}"
        ),
    }
    if engine.judge is not None:
        # Only when one was asked. A ``NO_MODEL`` dash here would put a judge row in
        # every report, and a run that consulted nobody has no judge to name.
        models["judge"] = engine.judge.name
    return models


# --- the judging stage (spec section 9 step 8, section 11.1) ----------------------


@dataclass(frozen=True)
class _Judged:
    """What the judging stage produced. The lists are new, never edited in place."""

    results: list[ClaimResult]
    findings: list[Finding]
    stage: Stage
    cost: JudgeCost
    cancelled: bool


def _escalates(result: ClaimResult) -> bool:
    """Whether this verdict is the judge's business (spec section 9 step 8).

    Three exclusions, each with a reason the product cannot do without:

    * No passage, no judging. There is nothing to hold the claim against, and asking
      a model to decide without one is exactly the assertion-without-evidence product
      rule 1 forbids -- of the judge as much as of the pipeline.
    * A numeric mismatch was decided by a rule that named both figures (spec section
      10). It scores 1.0 and there is no uncertainty for a second opinion to resolve.
    * Anything the models were confident about. The judge is an escape hatch for the
      low band, not a second pass over the whole document (spec section 11: 2-4 calls
      per paper, not one per claim).
    """
    verdict = result.verdict
    if verdict.passage is None or verdict.reason.startswith(NUMERIC_REASON):
        return False
    return verdict.tier == "low" or verdict.label is Label.NEI


def _judging(
    engine: Engine,
    results: list[ClaimResult],
    findings: list[Finding],
    *,
    emit: Listener,
    check: Callable[[], None],
) -> _Judged:
    """Ask the judge about the verdicts the models were least sure of.

    Nothing here can change a verdict, a finding's kind, level or state: the opinion
    is attached beside what the pipeline decided, and a disagreement becomes one more
    line under the finding (spec section 11.1). That is the whole contract, and it is
    why this stage runs after the verifying stage has already closed.
    """
    judge = engine.judge
    if judge is None:  # pragma: no cover - the caller checks, this keeps mypy honest
        raise ValueError("no judge on this engine")
    began = time.monotonic()
    emit(StageStart(name=JUDGING, by=judge.name))

    escalated = [(index, result) for index, result in enumerate(results) if _escalates(result)]
    opinions: dict[str, JudgeOpinion] = {}
    fresh: dict[str, JudgeOpinion] = {}
    keys: dict[str, tuple[str, str]] = {}
    ask: list[JudgeItem] = []
    for index, result in escalated:
        ident = f"c{index}"
        keys[ident] = (cache_mod.claim_hash(result.claim.text), result.source_id)
        stored = (
            None if engine.cache is None else engine.cache.get_judgement(*keys[ident], judge.name)
        )
        if stored is not None:
            opinions[ident] = stored
            continue
        passage = result.verdict.passage
        if passage is None:  # pragma: no cover - _escalates already refused these
            continue
        ask.append(
            JudgeItem(
                id=ident,
                claim=result.claim.text,
                passage=passage.text,
                verdict=result.verdict.label,
                tier=result.verdict.tier,
            )
        )

    cancelled = False
    if ask:

        def on_batch(done: int, total: int) -> None:
            emit(Progress(name=JUDGING, done=done, total=total))
            check()

        try:
            judge.review(ask, on_batch=on_batch, into=fresh)
        except Cancelled:
            # The batches already paid for are kept: they are in ``fresh``, and the
            # report they land in says of itself that it stopped (product rule 6).
            cancelled = True
    opinions.update(fresh)
    if engine.cache is not None:
        try:
            for ident, opinion in fresh.items():
                engine.cache.put_judgement(*keys[ident], judge.name, opinion)
        except sqlite3.Error as exc:
            # The cache is a speed-up, never a gate. A judgement that cannot be
            # stored -- a source row the cascade already took, a file gone read-only
            # -- costs the *next* run a call and nothing else: the opinion is
            # attached to this report either way. The optional judge layer is the
            # last thing in the tool allowed to end a run. The type is named and the
            # message is not, for the same reason it is not elsewhere.
            emit(Note(f"judgement not cached: {type(exc).__name__}"))

    for note in judge.skipped:
        emit(Note(note))
    # ``ask`` guards the read: a run whose opinions all came out of the cache made
    # no call at all, and a stale ``unavailable`` from an earlier document would
    # report a provider as down that this run never even asked.
    if ask and judge.unavailable:
        # One sentence, every surface: the note (for a run watching events), the
        # stage summary below (report, JSON and the terminal's stage row) and the
        # markdown header, so no surface can make a provider that was down look like
        # a document with nothing to escalate (product rule 6).
        summary = f"judge {judge_unavailable(judge.cost.calls, judge.detail)}"
        emit(Note(summary))
    else:
        summary = _judge_summary(len(opinions), len(results), judge.cost)

    by_claim = {keys[ident]: opinion for ident, opinion in opinions.items()}
    elapsed = time.monotonic() - began
    emit(StageEnd(name=JUDGING, by=judge.name, summary=summary, elapsed=elapsed))
    return _Judged(
        results=[
            replace(result, judge=opinions[f"c{index}"]) if f"c{index}" in opinions else result
            for index, result in enumerate(results)
        ],
        findings=[_with_opinion(item, by_claim) for item in findings],
        stage=Stage(name=JUDGING, by=judge.name, summary=summary, elapsed=elapsed),
        cost=judge.cost,
        cancelled=cancelled,
    )


def _with_opinion(item: Finding, by_claim: dict[tuple[str, str], JudgeOpinion]) -> Finding:
    """The finding with the judge's opinion beside it, and nothing else changed.

    ``kind``, ``level``, ``state`` and ``verdict`` are copied through untouched. A
    disagreement earns one more detail line -- agreement does not, because a line
    repeating the verdict above it says nothing the reader did not already read.
    """
    if item.claim is None or item.source_id is None:
        return item
    opinion = by_claim.get((cache_mod.claim_hash(item.claim.text), item.source_id))
    if opinion is None:
        return item
    detail = item.detail
    if item.verdict is not None and opinion.label is not item.verdict.label:
        detail = (*detail, judge_detail(opinion))
    return replace(item, judge=opinion, detail=detail)


def _judge_summary(reviewed: int, total: int, cost: JudgeCost) -> str:
    """``"12 of 118 verdicts reviewed, 3 calls, 18,402 prompt - 1,210 completion tokens"``.

    Both numbers matter: how much of the document got a second opinion, and what that
    opinion cost on a tier metered in tokens per minute (spec section 11).
    """
    return (
        f"{reviewed} of {total} verdicts reviewed, {_calls(cost.calls)}, "
        f"{cost.prompt_tokens:,} prompt \u00b7 {cost.completion_tokens:,} completion tokens"
    )


def _calls(count: int) -> str:
    """``"1 call"`` / ``"3 calls"``: one spelling, wherever a call count is printed."""
    return f"{count} call" + ("" if count == 1 else "s")


def _ordered(item: Finding) -> tuple[int, int, int, str]:
    """Document order, then the kind, so one line's findings never shuffle per run."""
    return (item.locator.page or 0, item.locator.line, item.locator.column, item.kind.value)


# --- stage helpers ----------------------------------------------------------------


def _retraction_summary(retracted: int, unavailable: int) -> str:
    """ "none" only when every DOI was checked and every check came back clean. A run
    that could not reach a provider says so instead of reporting a clean sheet."""
    parts = []
    if retracted:
        parts.append(f"{retracted} retracted")
    if unavailable:
        parts.append(f"{unavailable} unavailable")
    return ", ".join(parts) or "none"


def _attribution(providers: str, *, total: int, cached: int) -> str:
    """Who a network stage is attributed to: the providers, or the cache when every
    one of its units came from there and none of them was asked of anyone."""
    return CACHE_BY if total > 0 and cached == total else providers


def _retraction_attribution(*, total: int, cached: int) -> str:
    """Who the retraction stage is attributed to. ``_attribution``'s rule, plus the
    row it was missing: a run with no DOI in it asks Retraction Watch nothing, and a
    line naming them beside "none" reads as a clean sheet somebody checked. It is the
    same untruth :data:`CACHE_BY` exists to prevent (product rule 2), and it became
    visible on an all-bare-URL report, next to ``Resolving ... none``."""
    if total == 0:
        return NOBODY
    return _attribution(RETRACTIONS_BY, total=total, cached=cached)


def _resolving_attribution(*, total: int, indexes: int, cached: int) -> str:
    """Who the resolving stage is attributed to, counted per family.

    ``_attribution``'s rule applied to a stage whose references no longer all go to
    the same place. ``RESOLVERS_BY`` names Crossref and Semantic Scholar, and only
    the :data:`INDEXED_FAMILY` opens them: a bibliography of bare addresses is
    resolved out of the entries themselves, so a line naming two indexes nobody
    opened is the untruth :data:`CACHE_BY` exists to prevent, by a different route
    (product rule 2).

    ``total`` is every reference the stage placed, ``indexes`` the ones that really
    went to those two, ``cached`` the ones the cache answered; whatever is left over
    asked nobody anything and so can claim nobody. A stage that placed nothing at all
    -- a denied run, a document that cites nothing -- keeps the provider it announced
    at the start, because it has no units to speak for either way.
    """
    if total == 0 or indexes:
        return RESOLVERS_BY
    return CACHE_BY if cached else NOBODY


def _ignore(event: Event) -> None:
    """The listener a caller that wants no events gets, so ``emit`` is never optional."""


def _forward_log(gate: ConsentGate, seen: int, emit: Listener) -> int:
    """Pass on whatever the consent gate has written since ``seen``, and say how much."""
    for line in gate.install_log[seen:]:
        emit(Note(line))
    return len(gate.install_log)


def _finding(
    kind: Kind,
    locator: Locator,
    title: str,
    *,
    state: str | None = None,
    reference: Reference | None = None,
    source_id: str | None = None,
    fetch_step: int | None = None,
    detail: tuple[str, ...] = (),
) -> Finding:
    """A finding from a stage: never about a claim, so never carrying a verdict.

    ``state`` defaults to the kind's own word; only the ``UNVERIFIED`` family passes
    one, because which flavour applies depends on what failed (product rule 2).
    """
    return Finding(
        kind=kind,
        level=LEVELS[kind],
        locator=locator,
        title=title,
        state=STATE_WORDS[kind] if state is None else state,
        reference=reference,
        claim=None,
        verdict=None,
        source_id=source_id,
        fetch_step=fetch_step,
        tier=None,
        detail=detail,
    )


def _parser_for(target: Path | str) -> str:
    """The parser a target will need, before it has been read.

    An address that names a post is read by the platform's API, not by a parser;
    everything else has only its suffix to go on.
    """
    address = _bare_address(target)
    if address is not None:
        if is_social(address) or social.is_read_here(address):
            return POST_READER
        return PAGE_PARSER
    with suppress(ValueError):
        return _PARSER_BY_SUFFIX.get(Path(target).suffix.lower(), _TEXT_PARSER)
    return _TEXT_PARSER


def _marker_locator(document: Document, marker: CitationMarker) -> Locator:
    """Where a marker sits. ``locate`` raises on a marker from another document."""
    return document.locate(marker.paragraph, marker.start)


def _cited_locator(claims: Claims, number: int) -> Locator:
    """The first claim citing ``number``; used only when no entry answers it."""
    for claim in claims.claims:
        if number in claim.cited_refs:
            return claim.locator
    return Locator(line=1)


@dataclass(frozen=True)
class _Placed:
    """One resolve result as the resolving stage records it."""

    state: str
    source_id: str | None
    # Whether the fetching stage still has something to ask a provider for. ``False``
    # for a ghost, an ambiguous match, a provider that was down and a source no index
    # covers and no address points at: each is a different fact, keeps its own
    # wording (product rule 2), and none of them is ever softened into "unverified".
    fetchable: bool
    url: str = ""  # the address the entry printed, when that is what will be read


def _placed(
    result: ResolveResult,
    reference: Reference,
    add: Callable[[Finding], None],
) -> _Placed:
    """Where a resolve result leaves a reference; emits the state's own finding."""
    if result.state is State.GHOST:
        add(
            _finding(
                Kind.GHOST,
                reference.locator,
                "cited source does not exist",
                reference=reference,
                detail=tuple(result.notes),
            )
        )
        return _Placed(State.GHOST.value, None, fetchable=False)
    if result.state is State.AMBIGUOUS:
        add(
            _finding(
                Kind.AMBIGUOUS,
                reference.locator,
                "reference matches several records",
                reference=reference,
                detail=tuple(candidate.title for candidate in result.candidates),
            )
        )
        return _Placed(State.AMBIGUOUS.value, None, fetchable=False)
    if result.state is State.UNAVAILABLE:
        add(
            _finding(
                Kind.PROVIDER_UNAVAILABLE,
                reference.locator,
                "resolver unavailable",
                reference=reference,
                detail=tuple(result.notes),
            )
        )
        return _Placed(State.UNAVAILABLE.value, None, fetchable=False)
    if result.state is State.NOT_INDEXED:
        url = resolve_mod.find_url(reference.raw)
        if url:
            # Absence from a bibliographic index says nothing about a web page that
            # prints its own address: it is fetched like any other source.
            return _Placed("", f"{URL_PREFIX}{url}", fetchable=True, url=url)
        add(
            _finding(
                Kind.UNVERIFIED,
                reference.locator,
                "source is not in bibliographic indexes",
                state=State.NOT_INDEXED.value,
                reference=reference,
                detail=tuple(result.notes),
            )
        )
        return _Placed(State.NOT_INDEXED.value, None, fetchable=False)

    doi, arxiv_id = identifiers(reference, result)
    if doi:
        return _Placed("", f"{DOI_PREFIX}{doi}", fetchable=True)
    if arxiv_id:
        return _Placed("", f"{ARXIV_PREFIX}{arxiv_id}", fetchable=True)
    # Resolved, and still nothing to fetch it with: a book, or a record whose
    # provider has no identifier for it. Reported by stage 5 like any other miss.
    return _Placed("", None, fetchable=True)


def _not_attempted(reference: Reference, note: str) -> SourceStatus:
    """A cited reference nobody was asked about, because the network was denied.

    ``resolve`` stays ``None`` on purpose: there is no result, and an empty one
    would read as a lookup that found nothing (product rule 2).
    """
    return SourceStatus(
        reference=reference,
        resolve=None,
        retraction=None,
        source_id=None,
        text_kind="none",
        state=Outcome.NETWORK_DENIED.value,
        fetch_step=None,
        url="",
        from_cache=False,
        notes=(note,) if note else (),
    )


def _resolve_summary(tally: Counter[State]) -> str:
    """``"38 ok, 3 amb, 1 ghost"``, plus whatever else actually happened."""
    parts = [
        f"{tally[State.RESOLVED] + tally[State.RESOLVED_LOW]} ok",
        f"{tally[State.AMBIGUOUS]} amb",
        f"{tally[State.GHOST]} ghost",
    ]
    if tally[State.UNAVAILABLE]:
        parts.append(f"{tally[State.UNAVAILABLE]} unavailable")
    if tally[State.NOT_INDEXED]:
        parts.append(f"{tally[State.NOT_INDEXED]} not indexed")
    return ", ".join(parts)


def _resolved_retraction(
    engine: Engine, provider: EvidenceProvider, status: SourceStatus
) -> Retraction | None:
    """The provider's retraction check for a source that resolved to an identifier.

    The provider that is asked is the source id's, not the entry's --
    ``providers.checker_for`` explains why they can differ. ``SourceStatus.resolve``
    is ``None`` only for a reference nobody was asked about, and such a reference has
    no source id to have reached this stage with.
    """
    if status.resolve is None:
        return None
    checker = checker_for(status.source_id, provider, engine.providers)
    return checker.retraction(status.resolve)


def _read_source(
    engine: Engine,
    provider: EvidenceProvider,
    status: SourceStatus,
    resolved: ResolveResult,
    *,
    allowed: bool,
) -> list[EvidenceDoc]:
    """Ask a provider to read one planned source, unless the permission says no.

    The gate is here rather than in a provider because it is a property of the run,
    not of a source family: a denied run calls nobody at all, and says so once (spec
    section 7.1). The provider that reads is not always the one that resolved --
    ``providers.reader_for`` explains the one case where it is not.
    """
    if not allowed:
        # Nothing is called at all; the permission is the answer.
        return [
            EvidenceDoc(
                source_id=status.source_id or "",
                text="",
                text_kind="none",
                state=Outcome.NETWORK_DENIED.value,
                url=status.url,
                step=None,
                notes=(engine.fetcher.network_note,) if engine.fetcher.network_note else (),
                title="the source was not fetched",
            )
        ]
    reader = reader_for(status.source_id, provider, engine.providers)
    return reader.fetch(status.reference, resolved)


def _nothing_read(status: SourceStatus) -> EvidenceDoc:
    """The document the fetching stage files when its provider returned none.

    It is one source, unverified, in the only words the core can honestly use: it
    cannot say the page was unreachable or the text was empty, because nobody told
    it either. What it can say is that the source it planned to read was not read.
    """
    return EvidenceDoc(
        source_id=status.source_id or "",
        text="",
        text_kind="none",
        state=NO_DOCUMENT,
        url=status.url,
        step=None,
        title="the source was not read",
    )
