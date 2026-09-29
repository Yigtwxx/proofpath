"""Evidence search for a text that cites nothing (OPEN-ITEMS 17.1a).

A searcher turns a query into addresses and nothing else. What a page says is read by
the fetch ladder, exactly like a cited source, so a provider's snippet can never
become the passage a verdict rests on (product rule 1).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from proofpath.polite import ProviderError
from proofpath.providers import is_social


@dataclass(frozen=True)
class SearchHit:
    """One address a provider answered with, in its own order."""

    url: str
    title: str
    rank: int  # 1-based position in the provider's answer


class SearchKeyError(ProviderError):
    """The provider refused the key (HTTP 401/403): a credential to fix, not an outage.

    A ``ProviderError`` still, so a caller that knows only outages stops the search
    as before. The message names the variable to check and never the key.
    """


class Searcher(Protocol):
    """A web search provider. ``search`` raises ``polite.ProviderError`` when the
    provider does not answer, ``SearchKeyError`` when it refuses the key; an empty
    list means it answered with nothing."""

    name: str  # "tavily" | "searxng": what the Searching stage is attributed to

    def search(self, query: str, max_results: int) -> list[SearchHit]: ...


# Parameters that name the campaign a link came from, not the page it points at.
_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid)$")


def canonical(url: str) -> str:
    """One spelling per page: lower-cased scheme and host, no fragment, no tracking."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not _TRACKING.match(k)
        ]
    )
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", query, ""))


def _bare_host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def same_site(host: str, site: str) -> bool:
    """One host is the other, or a subdomain of it, in either direction.

    Both directions, so that excluding ``factcheck.afp.com`` also rules out
    ``www.afp.com`` and excluding ``afp.com`` rules out ``factcheck.afp.com``. It errs
    towards reading fewer pages, never towards a page vouching for its own site.
    """
    host = host.lower().removeprefix("www.")
    site = site.lower().removeprefix("www.")
    if not host or not site:
        return False
    return host == site or host.endswith(f".{site}") or site.endswith(f".{host}")


def keep_hits(
    hits: Sequence[SearchHit], *, exclude: Collection[str] = (), limit: int
) -> list[SearchHit]:
    """The hits worth reading, in order: one per page, no posts, no excluded site.

    A post is dropped because a platform post is a claim, not a source. X, the case
    that brought this feature, cannot be read at all (spec section 3). ``exclude``
    holds the document's own host, so a page is never found to support itself, and
    whatever else a caller must not read (the eval's fact-checking site).
    """
    kept: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        key = canonical(hit.url)
        if key in seen or is_social(hit.url):
            continue
        if any(same_site(_bare_host(hit.url), site) for site in exclude):
            continue
        seen.add(key)
        kept.append(hit)
        if len(kept) >= limit:
            break
    return kept


def search_claim(
    queries: Sequence[str],
    searcher: Searcher,
    results: int,
    *,
    exclude: Collection[str] = (),
) -> list[SearchHit]:
    """The product's search for one claim: each query asked for ``results`` hits, and
    at most ``results`` of them kept (final review, Important 3).

    One helper for ``verify`` and for the AVeriTeC eval, so the eval measures the
    search the product runs rather than a copy of it that drifts. Raises
    ``polite.ProviderError`` when the provider does not answer; the caller decides
    what that costs.
    """
    hits: list[SearchHit] = []
    for query in queries:
        hits.extend(searcher.search(query, results))
    return keep_hits(hits, exclude=exclude, limit=results)
