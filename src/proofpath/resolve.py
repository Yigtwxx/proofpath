"""Reference resolution: is the cited source real, and which record is it?

Spec section 8. Matcher scores from Crossref or OpenAlex are never evidence;
they only produce candidates. Identity is decided by field agreement between a
candidate record and the raw reference string: title token coverage, first
author surname, year (±1). The bias is deliberately conservative — uncertainty
resolves to ``AMBIGUOUS``, never to ``GHOST`` — and a ghost verdict needs both
providers to have been consulted, because Crossref alone does not index arXiv.
"""

from __future__ import annotations

import re
import time  # noqa: F401 - re-exported so tests can monkeypatch rs.time.sleep/monotonic
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from proofpath.polite import (  # noqa: F401 - re-exported for existing callers and tests
    MAX_RETRY_AFTER,
    MIN_INTERVAL,
    REPO_URL,
    PoliteClient,
    ProviderError,
)

CROSSREF = "https://api.crossref.org/works"
OPENALEX = "https://api.openalex.org/works"
ARXIV = "https://export.arxiv.org/api/query"
S2 = "https://api.semanticscholar.org/graph/v1/paper/search/match"
OPENLIBRARY = "https://openlibrary.org/search.json"

STRONG_TITLE = 0.8
WEAK_TITLE = 0.5
MIN_REFERENCE_WORDS = 6
YEAR_TOLERANCE = 1

_STOPWORDS = {
    "the", "of", "and", "in", "for", "on", "with", "a", "an", "to", "by", "at", "from",
    "et", "al", "is", "as", "its", "vs", "via", "into", "using", "toward", "towards",
}  # fmt: skip


class State(str, Enum):
    RESOLVED = "RESOLVED"
    RESOLVED_LOW = "RESOLVED (low confidence)"
    AMBIGUOUS = "AMBIGUOUS"
    GHOST = "GHOST REFERENCE"
    UNAVAILABLE = "UNVERIFIED (provider unavailable)"
    # Web pages, reports and organisation-authored documents are not covered by
    # bibliographic indexes; their absence there says nothing about existence.
    NOT_INDEXED = "UNVERIFIED (not in bibliographic indexes)"


@dataclass(frozen=True)
class Candidate:
    doi: str
    title: str
    first_author: str
    year: int | None
    venue: str
    provider: str  # crossref | s2 | openalex | arxiv | openlibrary | doi
    url: str = ""  # record URL when there is no DOI (OpenAlex works without one)

    @property
    def key(self) -> str:
        return (self.doi or self.url).lower()


@dataclass(frozen=True)
class FieldMatch:
    title: float  # coverage x length ratio, in [0, 1]
    author: bool
    year: bool


@dataclass(frozen=True)
class ResolveResult:
    state: State
    best: Candidate | None
    candidates: list[Candidate]
    notes: list[str] = field(default_factory=list)
    match: FieldMatch | None = None


@dataclass(frozen=True)
class Retraction:
    source: str  # retraction-watch | openalex
    date: str | None
    notice_doi: str | None
    label: str


# --- text normalisation ---------------------------------------------------------


def tokens(text: str) -> list[str]:
    """Lower-case ASCII word tokens, stopwords and single characters removed."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]+", ascii_text.lower())
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def title_coverage(title: str, raw: str) -> float:
    """Share of the candidate title's tokens that appear anywhere in the raw string."""
    title_tokens = set(tokens(title))
    if not title_tokens:
        return 0.0
    return len(title_tokens & set(tokens(raw))) / len(title_tokens)


def title_score(title: str, raw: str) -> float:
    """Coverage, discounted when the candidate title is much shorter than the cited one.

    A three-word candidate title fully contained in a ten-word cited title is
    not a match; the ratio keeps short titles from scoring as strong.
    """
    coverage = title_coverage(title, raw)
    segments = title_segments(raw)
    title_tokens = set(tokens(title))
    if not segments or not title_tokens:
        return coverage
    # Compare lengths against the segment the title actually overlaps with, not
    # the longest one (which is often the venue).
    cited = max(segments, key=lambda seg: len(title_tokens & set(tokens(seg))))
    if not title_tokens & set(tokens(segments[0])):
        # The candidate title shares nothing with the cited title segment: it may
        # overlap the venue (a proceedings-volume record) or nothing at all.
        # Either way it is not the cited work.
        return 0.0
    cited_len = len(set(tokens(cited)))
    ratio = min(1.0, len(title_tokens) / cited_len) if cited_len else 1.0
    return coverage * ratio


def author_matches(family: str, raw: str) -> bool:
    family_tokens = tokens(family)
    if not family_tokens:
        return False
    raw_tokens = set(tokens(raw))
    return all(t in raw_tokens for t in family_tokens)


def _years(raw: str) -> set[int]:
    return {int(t) for t in re.findall(r"\b(1[89]\d{2}|20\d{2}|2100)\b", raw)}


def year_matches(year: int | None, raw: str, *, tolerance: int = YEAR_TOLERANCE) -> bool:
    if year is None:
        return False
    return any(abs(year - y) <= tolerance for y in _years(raw))


_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)")


def find_doi(raw: str) -> str | None:
    match = _DOI.search(raw)
    if not match:
        return None
    doi = match.group(1).rstrip(".,;:")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;:")
    return doi


_ARXIV_ID = re.compile(
    r"(?:arxiv[:\s]*|arxiv\.org/(?:abs|pdf)/)"
    r"((?:\d{4}\.?\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?",
    re.I,
)

# A target that is *entirely* an arXiv id, with no "arxiv" marker at all — e.g. a
# CLI argument typed as ``2103.00020`` rather than ``arXiv:2103.00020``. Anchored
# to the whole (stripped) string so a bare id-shaped number inside a longer
# reference is never mistaken for one (that still requires the marker above).
_ARXIV_ID_BARE = re.compile(
    r"^(?:(?P<new>\d{4}\.\d{4,5})|(?P<old>[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?$",
    re.I,
)


def find_arxiv_id(raw: str) -> str | None:
    bare = _ARXIV_ID_BARE.fullmatch(raw.strip())
    if bare:
        return bare.group("new") or bare.group("old")
    match = _ARXIV_ID.search(raw)
    if not match:
        return None
    ident = match.group(1)
    if ident[:4].isdigit() and "." not in ident:
        ident = f"{ident[:4]}.{ident[4:]}"  # "160706450" as printed by some styles
    return ident


_QUOTED = re.compile(r"[\"“]([^\"”]{10,}?)[,.]?[\"”]")
_SPLIT = re.compile(r"[.?!]\s+")
# Every marker form ingest._ENTRY_START accepts: "[12] ", "12. ", "12) " and the
# wide-space "12  " (already collapsed to one space by ingest; 1-3 digits so a year
# opening an author-year entry survives).
_MARKER = re.compile(r"^\s*(\[\d+\]|\d+[.)]|\d{1,3}(?=\s))\s*")
_NAME = r"[A-Z][\w'\u2019-]+(?:\s[A-Z][\w'\u2019-]+)*"
_INITIALS = r"(?:[A-Z]{1,3}\.?(?![a-z])\s*(?:-\s*)?){1,3}"
_YEAR = r"(?:19|20)\d{2}[a-z]?"
# "Robert C. Moore and William Lewis. 2010. Title" (ACM, full first names)
_AUTHORS_THEN_YEAR = re.compile(rf"^\s*(.*?)(?<![A-Z])\.\s+{_YEAR}\.\s+")
# "Devlin, J., Chang, M.-W., & Toutanova, K. (2019). Title" (APA)
_AUTHORS_THEN_PAREN_YEAR = re.compile(rf"^\s*(.*?)\(\s*{_YEAR}\s*\)\.?\s+")
_AUTHOR_UNIT = rf"(?:{_NAME},?\s*{_INITIALS}|{_INITIALS}{_NAME})"
_AUTHOR_BLOCK = re.compile(
    rf"^\s*(?:{_AUTHOR_UNIT}[,;]?\s*(?:(?:and|&)\s*)?)+(?:et al\.?,?\s*)?"
    rf"(?:\(\d{{4}}[a-z]?\)\.?\s*)?"
)


# A period inside an author list only ever follows an initial or "et al".
_SENTENCE_PERIOD = re.compile(r"(?<![A-Z])(?<!\bal)(?<!\bJr)(?<!\bSt)\.")


def _strip_authors(text: str) -> str:
    """Drop a leading author list so names and initials never pollute the title."""
    for pattern in (_AUTHORS_THEN_YEAR, _AUTHORS_THEN_PAREN_YEAR):
        match = pattern.match(text)
        if match and len(match.group(1)) < 1500 and not _SENTENCE_PERIOD.search(match.group(1)):
            # The year delimits the authors; do not touch what follows.
            return text[match.end() :]
    return _AUTHOR_BLOCK.sub("", text, count=1)


def author_hint(raw: str) -> str:
    """Surname of the first author, from the stripped author block. Hint only."""
    cleaned = _QUOTED.sub(" ", _MARKER.sub("", raw))
    block = cleaned[: len(cleaned) - len(_strip_authors(cleaned))]
    unit = re.split(r",|\band\b|&", block)[0]
    words = [w for w in re.findall(r"[A-Za-z][\w'\u2019-]+", unit) if len(w) > 1]
    words = [w for w in words if w.lower() not in {"et", "al"}]
    return words[-1] if words else ""


_URL = re.compile(r"https?://|\bwww\.", re.I)
# Documents that bibliographic indexes do not cover, or preprint servers none
# of the consulted providers search. Their absence proves nothing.
_UNINDEXED_WORDS = re.compile(
    r"\b(blog|dashboard|press release|user manual|reference manual|technical report|"
    r"white ?paper|accessed|retrieved|psyarxiv|socarxiv|osf\.io|zenodo|github\.com)\b",
    re.I,
)


def looks_unindexed(raw: str) -> bool:
    """Web pages, reports, blogs and organisation-authored documents."""
    if find_doi(raw) or find_arxiv_id(raw):
        return False
    if _URL.search(raw) or _UNINDEXED_WORDS.search(raw):
        return True
    return _strip_authors(raw) == raw  # no person-style author list at all


def _looks_like_authors(segment: str) -> bool:
    initials = len(re.findall(r"\b[A-Z]\b", segment))
    all_words = re.findall(r"[A-Za-z][\w'\u2019-]+", segment)
    words = len(all_words)
    if initials >= 2 and initials * 3 >= words:
        return True
    # Full-name lists: nearly every word capitalised, several commas, no title-like
    # lower-case run ("Jeffrey Wu, Clemens Winter, and Dario Amodei").
    capitalised = sum(w[0].isupper() for w in all_words)
    has_digits = bool(re.search(r"\d", segment))
    return words >= 4 and segment.count(",") >= 2 and capitalised >= 0.8 * words and not has_digits


def title_segments(raw: str) -> list[str]:
    """Title-like spans, quoted spans first, for candidate generation only."""
    found = [m.group(1).strip() for m in _QUOTED.finditer(raw)]
    rest = _strip_authors(_QUOTED.sub(" ", _MARKER.sub("", raw)))
    for part in _SPLIT.split(rest):
        # "(CreateSpace, 2009)" style trailing parentheticals are not title words.
        segment = re.sub(r"\s*\([^()]*\)\.?\s*$", "", part).strip().strip(",;:").strip()
        words = len(re.findall(r"[A-Za-z]{2,}", segment))
        # Two words is enough ("Layer normalization", "Using Language"); the
        # title comes before the venue in every style, so order is kept.
        if words < 2 or _looks_like_authors(segment):
            continue
        found.append(segment)
    return found[:3]


# --- classification ---------------------------------------------------------------


def match_fields(candidate: Candidate, raw: str) -> FieldMatch:
    return FieldMatch(
        title=title_score(candidate.title, raw),
        author=author_matches(candidate.first_author, raw),
        year=year_matches(candidate.year, raw),
    )


def _candidate_state(m: FieldMatch, candidate: Candidate | None = None) -> State | None:
    if candidate is not None and candidate.provider == "openlibrary" and not m.author:
        # Book titles are short and generic; without the author they prove nothing.
        return None
    if m.title >= STRONG_TITLE:
        if m.author and m.year:
            return State.RESOLVED
        if m.author or m.year:
            return State.RESOLVED_LOW
        return State.AMBIGUOUS
    if m.title >= WEAK_TITLE and m.author and m.year:
        return State.AMBIGUOUS
    return None


def classify(candidates: Sequence[Candidate], raw: str) -> ResolveResult:
    """Decide the state from field agreement alone (spec section 8 table)."""
    if len(tokens(raw)) < MIN_REFERENCE_WORDS:
        return ResolveResult(
            State.AMBIGUOUS,
            None,
            list(candidates),
            notes=["reference too short to verify against a record"],
        )
    scored = [(c, match_fields(c, raw)) for c in candidates]
    for state in (State.RESOLVED, State.RESOLVED_LOW):
        hits = [(c, m) for c, m in scored if _candidate_state(m, c) is state]
        if hits:
            best, match = max(hits, key=lambda cm: cm[1].title)
            return ResolveResult(state, best, list(candidates), match=match)
    ambiguous = [c for c, m in scored if _candidate_state(m, c) is State.AMBIGUOUS]
    if ambiguous:
        return ResolveResult(
            State.AMBIGUOUS,
            None,
            ambiguous,
            notes=[f"{len(ambiguous)} plausible record(s), none agrees on every field"],
        )
    if looks_unindexed(raw):
        return ResolveResult(
            State.NOT_INDEXED,
            None,
            list(candidates),
            notes=["web page, report or organisation-authored document; not covered by indexes"],
        )
    return ResolveResult(State.GHOST, None, list(candidates))


# --- provider records -> candidates -----------------------------------------------


def candidates_from_crossref(payload: dict[str, Any]) -> list[Candidate]:
    items = payload.get("message", {}).get("items", [])
    return [c for c in (_crossref_candidate(item) for item in items) if c is not None]


def _crossref_candidate(item: dict[str, Any], provider: str = "crossref") -> Candidate | None:
    titles = item.get("title") or []
    if not titles or not item.get("DOI"):
        return None
    authors = item.get("author") or []
    family = str(authors[0].get("family", "")) if authors else ""
    parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    year = parts[0][0] if parts and parts[0] else None
    venues = item.get("container-title") or []
    return Candidate(
        doi=str(item["DOI"]),
        title=str(titles[0]),
        first_author=family,
        year=int(year) if year else None,
        venue=str(venues[0]) if venues else "",
        provider=provider,
    )


def candidates_from_openalex(payload: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for item in payload.get("results", []):
        title = item.get("display_name") or item.get("title")
        doi = item.get("doi") or ""
        if not title or not (doi or item.get("id")):
            continue
        authorships = item.get("authorships") or []
        name = (
            str((authorships[0].get("author") or {}).get("display_name", "")) if authorships else ""
        )
        family = name.split()[-1] if name else ""
        location = item.get("primary_location") or {}
        venue = str(((location.get("source") or {}).get("display_name")) or "")
        out.append(
            Candidate(
                doi=str(doi).removeprefix("https://doi.org/"),
                title=str(title),
                first_author=family,
                year=item.get("publication_year"),
                venue=venue,
                provider="openalex",
                url=str(item.get("id") or ""),
            )
        )
    return out


def candidates_from_s2(payload: dict[str, Any]) -> list[Candidate]:
    """Semantic Scholar ``paper/search/match``: the single best title match, or none."""
    out: list[Candidate] = []
    for item in payload.get("data") or []:
        title = item.get("title")
        if not title:
            continue
        ids = item.get("externalIds") or {}
        doi = str(ids.get("DOI") or "")
        if not doi and ids.get("ArXiv"):
            doi = f"10.48550/arXiv.{ids['ArXiv']}"
        authors = item.get("authors") or []
        name = str(authors[0].get("name", "")) if authors else ""
        out.append(
            Candidate(
                doi=doi,
                title=str(title),
                first_author=name.split()[-1] if name else "",
                year=item.get("year"),
                venue=str(item.get("venue") or ""),
                provider="s2",
                url=f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
            )
        )
    return out


def candidates_from_openlibrary(payload: dict[str, Any]) -> list[Candidate]:
    out: list[Candidate] = []
    for doc in payload.get("docs") or []:
        title = doc.get("title")
        if not title:
            continue
        authors = doc.get("author_name") or []
        name = str(authors[0]) if authors else ""
        publishers = doc.get("publisher") or []
        out.append(
            Candidate(
                doi="",
                title=str(title),
                first_author=name.split()[-1] if name else "",
                year=doc.get("first_publish_year"),
                venue=str(publishers[0]) if publishers else "",
                provider="openlibrary",
                url=f"https://openlibrary.org{doc.get('key', '')}",
            )
        )
    return out


def candidates_from_arxiv(xml: str) -> list[Candidate]:
    import xml.etree.ElementTree as ET

    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out: list[Candidate] = []
    for entry in root.findall("a:entry", ns):
        ident = entry.findtext("a:id", default="", namespaces=ns)
        match = re.search(r"abs/(.+?)(?:v\d+)?$", ident)
        title = " ".join((entry.findtext("a:title", default="", namespaces=ns)).split())
        if not match or not title:
            continue
        authors = [
            a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)
        ]
        first = authors[0].split()[-1] if authors and authors[0] else ""
        published = entry.findtext("a:published", default="", namespaces=ns)
        out.append(
            Candidate(
                doi=f"10.48550/arXiv.{match.group(1)}",
                title=title,
                first_author=first,
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue="arXiv",
                provider="arxiv",
            )
        )
    return out


def retraction_from_crossref(payload: dict[str, Any]) -> Retraction | None:
    for update in payload.get("message", {}).get("updated-by") or []:
        if str(update.get("type", "")).lower() != "retraction":
            continue
        parts = (update.get("updated") or {}).get("date-parts") or [[]]
        date = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts[0])) or None
        return Retraction(
            source=str(update.get("source") or "crossref"),
            date=date,
            notice_doi=update.get("DOI"),
            label=str(update.get("label") or "Retraction"),
        )
    return None


# --- the resolver -------------------------------------------------------------------


def dedupe(candidates: Sequence[Candidate], raw: str) -> list[Candidate]:
    """One candidate per identifier, keeping the record that agrees best with ``raw``.

    Providers disagree on metadata quality: Crossref titles an ACM paper "Numba",
    Semantic Scholar has the full title. Keeping the first seen would lose it.
    """
    best: dict[str, tuple[float, Candidate]] = {}
    order: list[str] = []
    for candidate in candidates:
        score = match_fields(candidate, raw).title
        if candidate.key not in best:
            order.append(candidate.key)
            best[candidate.key] = (score, candidate)
        elif score > best[candidate.key][0]:
            best[candidate.key] = (score, candidate)
    return [best[key][1] for key in order]


class Resolver:
    """Crossref + OpenAlex over HTTP, polite and with backoff. No key needed."""

    def __init__(
        self,
        *,
        contact_email: str = "",
        client: httpx.Client | None = None,
        retries: int = 2,
        timeout: float = 20.0,
    ) -> None:
        self._polite = PoliteClient(
            contact_email=contact_email, client=client, retries=retries, timeout=timeout
        )

    def _get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        return self._polite.get(url, params)

    def crossref_doi(self, doi: str) -> Candidate | None:
        response = self._get(f"{CROSSREF}/{doi}", {})
        if response.status_code == 404:
            return None
        return _crossref_candidate(response.json().get("message", {}), provider="doi")

    def crossref_bibliographic(self, raw: str, *, rows: int = 5) -> list[Candidate]:
        response = self._get(
            CROSSREF,
            {
                "query.bibliographic": raw,
                "rows": rows,
                "select": "DOI,title,author,issued,container-title,score,type",
            },
        )
        return candidates_from_crossref(response.json())

    def openalex_search(self, segment: str, *, per_page: int = 3) -> list[Candidate]:
        response = self._get(
            OPENALEX,
            {
                "search": segment,
                "per-page": per_page,
                "select": "id,doi,title,display_name,authorships,publication_year,"
                "primary_location,is_retracted",
            },
        )
        return candidates_from_openalex(response.json())

    def s2_match(self, segment: str) -> list[Candidate]:
        response = self._get(
            S2, {"query": segment, "fields": "title,authors,year,externalIds,venue"}
        )
        if response.status_code == 404:
            return []  # "Title match not found"
        return candidates_from_s2(response.json())

    def openlibrary_search(
        self, segment: str, *, author: str = "", limit: int = 3
    ) -> list[Candidate]:
        params: dict[str, Any] = {
            "title": segment,
            "limit": limit,
            "fields": "title,author_name,first_publish_year,publisher,key",
        }
        if author:
            params["author"] = author
        response = self._get(OPENLIBRARY, params)
        return candidates_from_openlibrary(response.json())

    def arxiv_id(self, arxiv_id: str) -> Candidate | None:
        response = self._get(ARXIV, {"id_list": arxiv_id, "max_results": 1})
        found = candidates_from_arxiv(response.text)
        return found[0] if found else None

    def arxiv_title(self, segment: str, *, max_results: int = 3) -> list[Candidate]:
        words = " ".join(re.findall(r"[A-Za-z0-9]+", segment))
        response = self._get(ARXIV, {"search_query": f'ti:"{words}"', "max_results": max_results})
        return candidates_from_arxiv(response.text)

    def resolve(self, raw: str) -> ResolveResult:
        notes: list[str] = []
        pool: list[Candidate] = []
        arxiv_id = find_arxiv_id(raw)
        if arxiv_id:
            try:
                direct = self.arxiv_id(arxiv_id)
            except ProviderError as exc:
                notes.append(f"arxiv unavailable for id lookup ({exc})")
            else:
                if direct is None:
                    notes.append(f"arXiv:{arxiv_id} does not resolve")
                else:
                    result = classify([direct], raw)
                    if result.state in (State.RESOLVED, State.RESOLVED_LOW):
                        return ResolveResult(
                            result.state,
                            direct,
                            [direct],
                            notes=[f"arXiv:{arxiv_id} resolved"],
                            match=result.match,
                        )
                    notes.append(f"arXiv:{arxiv_id} resolves to a different work")
                    pool.append(direct)

        doi = find_doi(raw)
        if doi:
            try:
                direct = self.crossref_doi(doi)
            except ProviderError as exc:
                notes.append(f"crossref unavailable for DOI lookup ({exc})")
            else:
                if direct is None:
                    notes.append(f"DOI {doi} does not resolve")
                else:
                    result = classify([direct], raw)
                    if result.state in (State.RESOLVED, State.RESOLVED_LOW):
                        return ResolveResult(
                            result.state,
                            direct,
                            [direct],
                            notes=[f"DOI {doi} resolved"],
                            match=result.match,
                        )
                    notes.append(f"DOI {doi} resolves to a different work")
                    pool.append(direct)

        segments = title_segments(raw)[:2]
        required_down: list[str] = []

        def consult(name: str, call: Callable[[], list[Candidate]]) -> None:
            try:
                pool.extend(call())
            except ProviderError as exc:
                required_down.append(name)
                notes.append(f"{name} unavailable ({exc})")

        # Primary providers, both free without a daily budget.
        consult("crossref", lambda: self.crossref_bibliographic(raw))
        if segments:
            consult("s2", lambda: self.s2_match(segments[0]))

        unique = dedupe(pool, raw)
        if len(required_down) == 2 and not unique:
            return ResolveResult(State.UNAVAILABLE, None, [], notes=notes)
        result = classify(unique, raw)

        if result.state in (State.GHOST, State.AMBIGUOUS) and segments:
            # Second opinions before any ghost call, and on a shaky match. arXiv and
            # Open Library are required (RoBERTa was missing from Crossref and
            # OpenAlex on 2026-09-11; books live nowhere else); OpenAlex is best
            # effort because its free tier is a small daily budget.
            consult("arxiv", lambda: self.arxiv_title(segments[0]))
            consult(
                "openlibrary",
                lambda: self.openlibrary_search(segments[0], author=author_hint(raw)),
            )
            notes.append("arxiv and openlibrary consulted before the ghost call")
            try:
                pool.extend(self.openalex_search(segments[0]))
            except ProviderError as exc:
                notes.append(f"openalex unavailable ({exc}); best-effort, not required")
            unique = dedupe(pool, raw)
            result = classify(unique, raw)

        if result.state is State.GHOST and required_down:
            # Part of the required evidence is missing; a ghost call would be unsafe.
            return ResolveResult(State.UNAVAILABLE, None, unique, notes=notes)
        return ResolveResult(
            result.state,
            result.best,
            result.candidates,
            notes=[*notes, *result.notes],
            match=result.match,
        )

    def retraction(self, doi: str) -> Retraction | None:
        """Retraction Watch data via Crossref, then OpenAlex's flag as a fallback."""
        try:
            response = self._get(f"{CROSSREF}/{doi}", {})
            if response.status_code != 404:
                found = retraction_from_crossref(response.json())
                if found is not None:
                    return found
        except ProviderError:
            pass
        try:
            response = self._get(f"{OPENALEX}/https://doi.org/{doi}", {"select": "is_retracted"})
        except ProviderError:
            return None
        if response.status_code != 404 and response.json().get("is_retracted"):
            return Retraction(source="openalex", date=None, notice_doi=None, label="Retraction")
        return None
