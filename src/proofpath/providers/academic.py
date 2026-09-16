"""The ``academic://`` family: a cited record, resolved and then read open access.

Spec section 5. The provider is two collaborators and no state of its own: the
resolver decides what the entry names (section 8), the open-access chain decides how
much of it can be read (section 7). Everything either of them says about a source
travels out of here in its own words.
"""

from __future__ import annotations

from proofpath.document import Reference
from proofpath.oa import Evidence
from proofpath.providers import (
    ARXIV_PREFIX,
    DOI_PREFIX,
    NO_IDENTIFIER,
    EvidenceDoc,
    FetchesOpenAccess,
    Resolves,
    Scheme,
    doi_of,
    honest_state,
    identifiers,
)
from proofpath.resolve import ResolveResult, Retraction

# What a fetched source is attributed to in the stage line. ``oa.Evidence.source``
# is a provider label or ``"abstract:<provider>"``.
PROVIDER_NAMES = {
    "s2": "Semantic Scholar",
    "s2_pdf": "Semantic Scholar",
    "crossref": "Crossref",
    "crossref_link": "Crossref",
    "unpaywall": "Unpaywall",
    "europepmc": "Europe PMC",
    "arxiv": "arXiv",
    "openalex": "OpenAlex",
    "landing": "publisher page",
    "cache": "cache",
}


class AcademicProvider:
    """Papers, preprints, books and reports: anything a bibliographic index covers."""

    scheme: Scheme = "academic"

    def __init__(self, resolver: Resolves, oa: FetchesOpenAccess) -> None:
        self._resolver = resolver
        self._oa = oa

    def resolve(self, ref: Reference) -> ResolveResult:
        """The entry verbatim, to the matcher. Parsing it is ``resolve``'s job."""
        return self._resolver.resolve(ref.raw)

    def retraction(self, resolved: ResolveResult) -> Retraction | None:
        """A retraction notice is a fact about a DOI, so a record without one is not
        asked about at all. ``None`` means nobody was asked or nobody had a notice;
        an outage raises, and the two must never read alike (product rule 2)."""
        doi = doi_of(resolved)
        return None if doi is None else self._resolver.retraction(doi)

    def fetch(self, ref: Reference, resolved: ResolveResult) -> list[EvidenceDoc]:
        """Read the record through the open-access chain, by whichever id it has.

        A record with neither a DOI nor an arXiv id — a book, or a record whose
        provider has no identifier for it — is not silently dropped: it comes back
        as a document that says there was nothing to fetch it with (product rule 6).
        """
        doi, arxiv_id = identifiers(ref, resolved)
        if doi is None and arxiv_id is None:
            return [
                EvidenceDoc(
                    source_id="",
                    text="",
                    text_kind="none",
                    state=NO_IDENTIFIER,
                    url="",
                    step=None,
                    title="the record carries no identifier to fetch it with",
                )
            ]
        source_id = f"{DOI_PREFIX}{doi}" if doi else f"{ARXIV_PREFIX}{arxiv_id}"
        return [_from_evidence(source_id, self._oa.fetch(doi, arxiv_id))]


def _from_evidence(source_id: str, evidence: Evidence) -> EvidenceDoc:
    """The open-access chain already labelled its own result; it is carried as it is."""
    step = 0 if evidence.from_cache else _winning_step(evidence)
    winner = "cache" if evidence.from_cache else provider_name(evidence.source)
    state, extra = honest_state(evidence.state, evidence.kind)
    return EvidenceDoc(
        source_id=source_id,
        text=evidence.text,
        text_kind=evidence.kind,
        state=state,
        url=evidence.url,
        step=step,
        notes=(*extra, *evidence.notes),
        from_cache=evidence.from_cache,
        winner=winner,
        title="the source text could not be read",
    )


def _winning_step(evidence: Evidence) -> int | None:
    """Which ladder step produced the text the chain kept."""
    for attempt in evidence.attempts:
        if attempt.location.url == evidence.url:
            return attempt.step
    return evidence.attempts[-1].step if evidence.attempts else None


def provider_name(source: str) -> str:
    """``"s2_pdf"`` and ``"abstract:s2"`` both read as "Semantic Scholar"."""
    label = source.removeprefix("abstract:")
    return PROVIDER_NAMES.get(label, label)
