"""The open-access chain (spec sections 6.1 and 9, decision 8.3): from a DOI or an
arXiv id to full text, else a labelled abstract, else a labelled honesty state.

Direct full text exists for well under half of citations, so the abstract is a
primary path, not an edge case — and it is never allowed to look like full text:
an abstract-grade result carries ``ABSTRACT_ONLY`` as its state, and a result with
nothing at all carries the most informative fetch outcome (product rule 2).

Metadata comes from providers in a fixed order — Semantic Scholar, Crossref,
Unpaywall (only with a contact address), Europe PMC, arXiv, the doi.org landing
page — and every candidate URL then climbs the fetch ladder (``fetch.py``), so
blocked, unreachable and rate limited stay distinct facts. OpenAlex is consulted
last and only for an abstract: its free tier is a small daily budget.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from proofpath import fetch
from proofpath.cache import Cache
from proofpath.fetch import Fetched, Outcome  # by name: in the class, ``fetch`` is the method
from proofpath.polite import PoliteClient, ProviderError
from proofpath.resolve import CROSSREF, OPENALEX

# The arXiv abstract page measured 827 words (spec section 6.1); below this a page
# is abstract-grade, whatever its URL promised.
FULLTEXT_MIN_WORDS = 1500
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2"
ABSTRACT_ONLY = "LOW CONFIDENCE (abstract only)"

S2_FIELDS = "openAccessPdf,abstract,externalIds"
# Crossref ``link`` entries meant for machines; the rest are syndication and the like.
TDM_APPLICATIONS = ("text-mining", "similarity-checking")
# Publisher TDM endpoints that need an API key: a certain 401, not an open location.
KEY_GATED_HOSTS = ("api.elsevier.com", "api.wiley.com")

# Which honesty state a "none" result reports when the attempts disagree: the
# outcome that says most about why (a wall beats a missing page beats a wait).
OUTCOME_PRIORITY = (
    fetch.Outcome.BLOCKED_NO_BROWSER,
    fetch.Outcome.BLOCKED,
    fetch.Outcome.BLOCKED_ROBOTS,
    fetch.Outcome.UNREACHABLE,
    fetch.Outcome.UNAVAILABLE,
    fetch.Outcome.NETWORK_DENIED,
)

_TAG = re.compile(r"<[^>]+>")
# DataCite's arXiv DOIs (``10.48550/arXiv.<id>``, also what ``resolve.py`` synthesises
# from an S2 ArXiv id) name the arXiv id themselves. S2 does not reliably echo it
# back as an externalId and Crossref has no links for them, so without reading it
# off the DOI the PDF at arxiv.org was never tried (9 of 50 DOIs, 2026-09-11).
_DATACITE_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)
_JATS_BODY = re.compile(r"<body[\s>].*?</body>", re.S)
# Closing tags after which JATS text starts a new line: paragraphs, headings,
# sections, captions, list items and table cells.
_JATS_BREAK = re.compile(r"</(?:p|title|sec|label|caption|list-item|td|th|tr)>")
_JATS_ABSTRACT_HEADING = re.compile(r"<jats:title>\s*abstract\s*</jats:title>", re.I)
# Numbered inline citation markers, e.g. <xref ref-type="bibr" rid="CR1">1</xref>:
# dropped whole, so a reference number never glues onto the word before it.
_JATS_BIBR_XREF = re.compile(r'<xref\s+ref-type="bibr"[^>]*>.*?</xref>', re.S)

Label = Literal["s2_pdf", "crossref_link", "unpaywall", "europepmc", "arxiv", "landing"]


@dataclass(frozen=True)
class Location:
    label: Label
    url: str
    expects: fetch.Kind  # what the provider promised; the ladder trusts headers, not this


@dataclass(frozen=True)
class Attempt:
    location: Location
    outcome: fetch.Outcome
    step: int
    words: int


@dataclass(frozen=True)
class Located:
    locations: list[Location]  # in chain order, de-duplicated by URL
    abstracts: dict[str, str]  # provider -> abstract text; "openalex" is filled lazily
    arxiv_id: str | None
    pmcid: str | None
    notes: list[str]  # provider failures, e.g. "s2 unavailable (HTTP 429)"


@dataclass(frozen=True)
class Evidence:
    kind: Literal["fulltext", "abstract", "none"]
    state: str  # "" for full text; ABSTRACT_ONLY; else the most informative Outcome value
    text: str
    url: str  # where the text came from ("" for none)
    source: str  # a Label, "abstract:<provider>", "cache" or ""
    attempts: list[Attempt]
    notes: list[str]
    from_cache: bool = False

    @property
    def words(self) -> int:
        return len(self.text.split())


# --- pure helpers on provider payloads ---------------------------------------


def strip_tags(text: str) -> str:
    """Tags replaced with a space (so a word and a number either side of a tag never
    glue together), entities decoded, whitespace collapsed to single spaces."""
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def jats_body_text(xml: bytes) -> str:
    """Europe PMC ``fullTextXML`` → the text of ``<body>`` (the whole article when there
    is none), one line per paragraph, heading, caption or cell. Inline bibliographic
    citation markers (``<xref ref-type="bibr">…</xref>``) are dropped whole, not turned
    into stray digits stuck onto the preceding word."""
    document = xml.decode("utf-8", errors="replace")
    match = _JATS_BODY.search(document)
    body = match.group(0) if match else document
    body = _JATS_BIBR_XREF.sub("", body)
    lines = (strip_tags(part) for part in _JATS_BREAK.split(body))
    return "\n".join(line for line in lines if line)


def arxiv_id_from_doi(doi: str | None) -> str | None:
    """The arXiv id a DataCite arXiv DOI carries, version suffix and all; else ``None``."""
    match = _DATACITE_ARXIV_DOI.match(doi or "")
    return match.group(1) if match else None


def _pmcid(value: Any) -> str | None:
    """Europe PMC wants ``PMC8371605``; Semantic Scholar stores the bare number."""
    if not value:
        return None
    digits = str(value).strip()
    return digits if digits.upper().startswith("PMC") else f"PMC{digits}"


def locations_from_s2(
    payload: dict[str, Any],
) -> tuple[Location | None, str | None, str | None, str]:
    """``(pdf location, arxiv id, pmcid, abstract)`` from a ``graph/v1/paper`` record."""
    pdf = payload.get("openAccessPdf") or {}
    url = pdf.get("url")
    location = Location("s2_pdf", str(url), "pdf") if url else None
    ids = payload.get("externalIds") or {}
    arxiv_id = str(ids["ArXiv"]) if ids.get("ArXiv") else None
    return location, arxiv_id, _pmcid(ids.get("PubMedCentral")), str(payload.get("abstract") or "")


def locations_from_crossref(payload: dict[str, Any]) -> tuple[list[Location], str]:
    """``(text-mining links, abstract)`` from a ``works/<doi>`` record.

    Links are kept when meant for text mining or similarity checking and not behind
    a publisher API key; the content type decides what to expect, with an
    ``unspecified`` type read off the URL suffix.
    """
    message = payload.get("message") or {}
    links: list[Location] = []
    for link in message.get("link") or []:
        url = str(link.get("URL") or "")
        if not url or link.get("intended-application") not in TDM_APPLICATIONS:
            continue
        if httpx.URL(url).host in KEY_GATED_HOSTS:
            continue
        media = str(link.get("content-type") or "unspecified").lower()
        expects: fetch.Kind
        if media == "application/pdf":
            expects = "pdf"
        elif media == "text/html":
            expects = "html"
        elif media == "unspecified":
            expects = "pdf" if url.lower().endswith(".pdf") else "html"
        else:
            continue  # XML and friends: the ladder extracts nothing from them
        links.append(Location("crossref_link", url, expects))
    raw_abstract = str(message.get("abstract") or "")
    abstract = strip_tags(_JATS_ABSTRACT_HEADING.sub("", raw_abstract))
    return _dedupe(links), abstract


def location_from_unpaywall(payload: dict[str, Any]) -> Location | None:
    """The best OA location, PDF first, from a ``v2/<doi>`` record."""
    best = payload.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        return Location("unpaywall", str(best["url_for_pdf"]), "pdf")
    if best.get("url"):
        return Location("unpaywall", str(best["url"]), "html")
    return None


def pmcid_from_europepmc(payload: dict[str, Any]) -> str | None:
    """The first hit's PMCID, only when Europe PMC itself holds the article."""
    results = (payload.get("resultList") or {}).get("result") or []
    if not results:
        return None
    first = results[0]
    if first.get("inEPMC") != "Y":
        return None
    return _pmcid(first.get("pmcid"))


def abstract_from_openalex(payload: dict[str, Any]) -> str:
    """The abstract rebuilt from OpenAlex's ``abstract_inverted_index``."""
    index = payload.get("abstract_inverted_index") or {}
    positions: list[tuple[int, str]] = []
    for word, places in index.items():
        positions.extend((int(place), str(word)) for place in places or [])
    return " ".join(word for _, word in sorted(positions))


def _dedupe(locations: list[Location]) -> list[Location]:
    """First label wins for a URL the chain lists twice."""
    seen: set[str] = set()
    unique: list[Location] = []
    for location in locations:
        if location.url not in seen:
            seen.add(location.url)
            unique.append(location)
    return unique


def _worst_outcome(attempts: list[Attempt]) -> tuple[fetch.Outcome, str | None]:
    """The most informative outcome, plus a note when every attempt reached the page
    but none yielded text (product rule 2: that is not the same as unreachable)."""
    seen = {attempt.outcome for attempt in attempts}
    outcome = next((o for o in OUTCOME_PRIORITY if o in seen), fetch.Outcome.UNREACHABLE)
    note = "reached but no text extracted" if seen == {fetch.Outcome.OK} else None
    return outcome, note


# --- the chain ---------------------------------------------------------------


class OpenAccess:
    """Providers in decision 8.3's order, then the fetch ladder, then the abstract."""

    def __init__(
        self,
        fetcher: fetch.Fetcher,
        client: PoliteClient,
        *,
        contact_email: str = "",
        cache: Cache | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._client = client
        self._email = contact_email
        self._cache = cache

    # --- providers ------------------------------------------------------------

    def _json(
        self, url: str, params: dict[str, Any] | None = None, *, mailto: bool
    ) -> dict[str, Any] | None:
        """One provider record, or ``None`` on 404. ``ProviderError`` propagates."""
        response = self._client.get(url, params, mailto=mailto)
        if response.status_code == 404:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("invalid JSON") from exc
        return payload if isinstance(payload, dict) else None

    def _s2(self, doi: str | None, arxiv_id: str | None) -> dict[str, Any] | None:
        ident = f"DOI:{doi}" if doi else f"arXiv:{arxiv_id}"
        return self._json(f"{S2_PAPER}/{ident}", {"fields": S2_FIELDS}, mailto=False)

    def _crossref(self, doi: str) -> dict[str, Any] | None:
        return self._json(f"{CROSSREF}/{doi}", mailto=True)

    def _unpaywall(self, doi: str) -> dict[str, Any] | None:
        return self._json(f"{UNPAYWALL}/{doi}", {"email": self._email}, mailto=False)

    def _europepmc_search(self, doi: str) -> dict[str, Any] | None:
        params = {"query": f"DOI:{doi}", "format": "json", "resultType": "lite"}
        return self._json(f"{EUROPEPMC}/search", params, mailto=False)

    def _openalex(self, doi: str) -> dict[str, Any] | None:
        return self._json(f"{OPENALEX}/https://doi.org/{doi}", mailto=True)

    # --- locate ---------------------------------------------------------------

    def locate(self, doi: str | None, arxiv_id: str | None = None) -> Located:
        """Every open-access URL the providers know, in chain order, plus their abstracts."""
        _require(doi, arxiv_id)
        arxiv_id = arxiv_id or arxiv_id_from_doi(doi)
        if not self._fetcher.network_allowed:
            # The providers are network too: nothing is asked (spec section 15).
            return Located([], {}, arxiv_id, None, [self._fetcher.network_note])
        locations: list[Location] = []
        abstracts: dict[str, str] = {}
        notes: list[str] = []
        pmcid: str | None = None

        def consult(name: str, call: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
            try:
                return call()
            except ProviderError as exc:
                notes.append(f"{name} unavailable ({exc})")
                return None

        # 1. Semantic Scholar: the PDF, the other ids and the abstract in one call.
        payload = consult("s2", lambda: self._s2(doi, arxiv_id))
        if payload is not None:
            pdf, s2_arxiv, pmcid, abstract = locations_from_s2(payload)
            if pdf is not None:
                locations.append(pdf)
            arxiv_id = arxiv_id or s2_arxiv
            if abstract:
                abstracts["s2"] = abstract
        if doi:
            # 2. Crossref: publisher text-mining links.
            payload = consult("crossref", lambda: self._crossref(doi))
            if payload is not None:
                links, abstract = locations_from_crossref(payload)
                locations.extend(links)
                if abstract:
                    abstracts["crossref"] = abstract
            # 3. Unpaywall wants an address; without one it is not asked.
            if self._email:
                payload = consult("unpaywall", lambda: self._unpaywall(doi))
                if payload is not None:
                    best = location_from_unpaywall(payload)
                    if best is not None:
                        locations.append(best)
            # 4. Europe PMC: only searched when S2 did not already name the PMCID.
            if pmcid is None:
                payload = consult("europepmc", lambda: self._europepmc_search(doi))
                if payload is not None:
                    pmcid = pmcid_from_europepmc(payload)
        if pmcid:
            locations.append(Location("europepmc", f"{EUROPEPMC}/{pmcid}/fullTextXML", "text"))
        # 5. arXiv, 6. the landing page: always worth a climb.
        if arxiv_id:
            locations.append(Location("arxiv", f"https://arxiv.org/pdf/{arxiv_id}", "pdf"))
        if doi:
            locations.append(Location("landing", f"https://doi.org/{doi}", "html"))
        return Located(_dedupe(locations), abstracts, arxiv_id, pmcid, notes)

    # --- fetch ----------------------------------------------------------------

    def fetch(self, doi: str | None, arxiv_id: str | None = None) -> Evidence:
        """Full text if any location yields it, else a labelled abstract, else ``none``."""
        _require(doi, arxiv_id)
        source_id = f"doi:{doi}" if doi else f"arxiv:{arxiv_id}"
        if not self._fetcher.network_allowed:
            denied = self._fetcher.network_note
            return Evidence("none", Outcome.NETWORK_DENIED.value, "", "", "", [], [denied])
        cached = self._cached(source_id)
        if cached is not None:
            return cached
        evidence = self._climb_all(doi, arxiv_id, source_id)
        # One source, however many of its locations were walls (fetch.BrowserGate).
        if any(a.outcome is Outcome.BLOCKED_NO_BROWSER for a in evidence.attempts):
            self._fetcher.gate.skipped += 1
        return evidence

    def _climb_all(self, doi: str | None, arxiv_id: str | None, source_id: str) -> Evidence:
        located = self.locate(doi, arxiv_id)
        notes = list(located.notes)
        attempts: list[Attempt] = []
        best_page: tuple[int, str, str, str] | None = None  # words, text, url, label
        for location in located.locations:
            fetched = self._fetcher.fetch(
                location.url, text_kind="fulltext", counts_as_source=False
            )
            text = self._page_text(location, fetched)
            words = len(text.split())
            attempts.append(Attempt(location, fetched.outcome, fetched.step, words))
            if not fetched.ok or not words:
                continue
            if self._cache is not None and not fetched.from_cache:
                # The ladder filed the page as the "fulltext" it was asked for; the
                # word count now says what the url: row really holds.
                grade = "fulltext" if words >= FULLTEXT_MIN_WORDS else "abstract"
                self._cache.set_text_kind(f"url:{location.url}", grade)
            if words >= FULLTEXT_MIN_WORDS:
                self._store(source_id, "fulltext", text, fetched.final_url)
                return Evidence(
                    "fulltext", "", text, fetched.final_url, location.label, attempts, notes
                )
            # Reachable but abstract-grade: the longest such page is a fallback.
            if best_page is None or words > best_page[0]:
                best_page = (words, text, fetched.final_url, location.label)

        # The abstract, in provider order; the page text only after real abstracts.
        for provider in ("s2", "crossref"):
            if located.abstracts.get(provider):
                return self._abstract(
                    source_id,
                    located.abstracts[provider],
                    self._abstract_url(provider, doi, located.arxiv_id),
                    f"abstract:{provider}",
                    attempts,
                    notes,
                )
        if best_page is not None:
            _, text, url, label = best_page
            return self._abstract(source_id, text, url, label, attempts, notes)
        # OpenAlex only now: its daily budget is small (decision 8.3).
        if doi:
            try:
                payload = self._openalex(doi)
            except ProviderError as exc:
                notes.append(f"openalex unavailable ({exc})")
            else:
                abstract = abstract_from_openalex(payload) if payload is not None else ""
                if abstract:
                    url = self._abstract_url("openalex", doi, located.arxiv_id)
                    return self._abstract(
                        source_id, abstract, url, "abstract:openalex", attempts, notes
                    )

        if not located.locations and not attempts:
            notes.append("no open-access location found")
            state = Outcome.UNREACHABLE
        else:
            state, note = _worst_outcome(attempts)
            if note:
                notes.append(note)
        return Evidence("none", state.value, "", "", "", attempts, notes)

    def _page_text(self, location: Location, fetched: Fetched) -> str:
        """The ladder's text, except that Europe PMC's JATS XML needs its own extraction."""
        if (
            location.label == "europepmc"
            and fetched.ok
            and fetched.kind not in ("html", "pdf")
            and fetched.body.lstrip().startswith(b"<")
        ):
            return jats_body_text(fetched.body)
        return fetched.text

    def _abstract(
        self,
        source_id: str,
        text: str,
        url: str,
        source: str,
        attempts: list[Attempt],
        notes: list[str],
    ) -> Evidence:
        self._store(source_id, "abstract", text, url)
        return Evidence("abstract", ABSTRACT_ONLY, text, url, source, attempts, notes)

    @staticmethod
    def _abstract_url(provider: str, doi: str | None, arxiv_id: str | None) -> str:
        """The provider record the abstract was read from."""
        if provider == "s2":
            return f"{S2_PAPER}/{f'DOI:{doi}' if doi else f'arXiv:{arxiv_id}'}"
        if provider == "crossref":
            return f"{CROSSREF}/{doi}"
        return f"{OPENALEX}/https://doi.org/{doi}"

    # --- cache ----------------------------------------------------------------

    def _cached(self, source_id: str) -> Evidence | None:
        if self._cache is None:
            return None
        text = self._cache.get_raw_text(source_id)
        if not text:  # only non-empty text is ever stored, so empty means miss
            return None
        stored = self._cache.text_kind_and_url(source_id)
        kind, url = stored if stored is not None else ("abstract", "")
        if kind == "fulltext":
            return Evidence("fulltext", "", text, url, "cache", [], ["cache: hit"], True)
        return Evidence("abstract", ABSTRACT_ONLY, text, url, "cache", [], ["cache: hit"], True)

    def _store(self, source_id: str, text_kind: str, text: str, url: str) -> None:
        if self._cache is not None and text:
            self._cache.add_source(
                source_id, scheme="academic", title="", url=url, text_kind=text_kind, raw_text=text
            )


def _require(doi: str | None, arxiv_id: str | None) -> None:
    if not doi and not arxiv_id:
        raise ValueError("a DOI or an arXiv id is required")
