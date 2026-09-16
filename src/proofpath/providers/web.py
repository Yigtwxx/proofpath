"""The ``web://`` family: a page, read up the fetch ladder.

Spec section 5. A page has no bibliographic identity to resolve — it has an address
— so the interesting half of this provider is the grading: length decides whether
what came back is a document or a summary of one, and a page that was *reached* and
held no text is a different fact from a page that could not be reached at all
(product rule 2).
"""

from __future__ import annotations

from proofpath.document import Reference
from proofpath.fetch import STEP_NAMES, Fetched
from proofpath.oa import ABSTRACT_ONLY, FULLTEXT_MIN_WORDS
from proofpath.providers import (
    NO_TEXT,
    URL_PREFIX,
    EvidenceDoc,
    FetchesUrl,
    Scheme,
    honest_state,
)
from proofpath.report import TextKind
from proofpath.resolve import Candidate, ResolveResult, Retraction, State, find_url


class WebProvider:
    """Pages, reports and organisation-authored documents: anything with an address."""

    scheme: Scheme = "web"

    def __init__(self, fetcher: FetchesUrl) -> None:
        self._fetcher = fetcher

    def resolve(self, ref: Reference) -> ResolveResult:
        """Not in the indexes, and never was.

        That is a fact about the indexes, not about the page (``resolve.State``
        spells it out), so it is never a ghost: absence from a bibliographic index
        says nothing about a document that prints its own address (product rule 3).
        The address travels on the candidate so nothing has to look for it twice.
        """
        url = find_url(ref.raw)
        if url is None:
            return ResolveResult(State.NOT_INDEXED, None, [])
        best = Candidate(
            doi="", title="", first_author="", year=None, venue="", provider="url", url=url
        )
        return ResolveResult(State.NOT_INDEXED, best, [best])

    def retraction(self, resolved: ResolveResult) -> Retraction | None:
        """Nobody publishes retraction notices about web pages, so nobody is asked."""
        return None

    def fetch(self, ref: Reference, resolved: ResolveResult) -> list[EvidenceDoc]:
        """One address, up the ladder. No address, nothing asked of anyone.

        The entry's own address wins over the resolved record's, because that is the
        one ``verify`` filed the source under: a document read from one address and
        filed under another would be cached, quoted and reported under a URL it never
        came from. The record's address is the fallback for a reference that reached
        this provider without printing one.
        """
        url = find_url(ref.raw) or _address(resolved)
        if not url:
            return []
        return [_from_page(url, self._fetcher.fetch(url))]


def _address(resolved: ResolveResult) -> str | None:
    """The address a resolve result recorded, when the record carries one."""
    best = resolved.best
    return best.url if best is not None and best.url else None


def _from_page(url: str, fetched: Fetched) -> EvidenceDoc:
    """A bare URL. Length decides the grade: a page under the full-text bar is an
    abstract however the link was labelled, and a page that was reached but held no
    text is not the same thing as a page that could not be reached.

    The document is filed under the address that was *asked for*, not the one the
    ladder ended up at: that is the key the cache already holds the page under, and
    the two must not drift apart.
    """
    winner = "cache" if fetched.from_cache else STEP_NAMES.get(fetched.step, "web")
    source_id = f"{URL_PREFIX}{url}"
    final_url = fetched.final_url or fetched.url
    if not fetched.ok:
        state, extra = honest_state(fetched.outcome.value, "none")
        return EvidenceDoc(
            source_id=source_id,
            text="",
            text_kind="none",
            state=state,
            url=final_url,
            step=fetched.step,
            notes=(*extra, *fetched.notes),
            from_cache=fetched.from_cache,
            winner=winner,
            title="the source text could not be read",
        )
    kind: TextKind
    if fetched.words >= FULLTEXT_MIN_WORDS:
        kind, state = "fulltext", ""
    elif fetched.words:
        kind, state = "abstract", ABSTRACT_ONLY
    else:
        kind, state = "none", NO_TEXT
    return EvidenceDoc(
        source_id=source_id,
        text=fetched.text if kind != "none" else "",
        text_kind=kind,
        state=state,
        url=final_url,
        step=fetched.step,
        notes=tuple(fetched.notes),
        from_cache=fetched.from_cache,
        winner=winner,
        title="the page was reached but held no text",
    )
