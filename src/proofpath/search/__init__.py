"""Evidence search for a text that cites nothing (OPEN-ITEMS 17.1a).

A searcher turns a query into addresses and nothing else. What a page says is read by
the fetch ladder, exactly like a cited source, so a provider's snippet can never
become the passage a verdict rests on (product rule 1).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from proofpath.providers import is_social


@dataclass(frozen=True)
class SearchHit:
    """One address a provider answered with, in its own order."""

    url: str
    title: str
    rank: int  # 1-based position in the provider's answer


class Searcher(Protocol):
    """A web search provider. ``search`` raises ``polite.ProviderError`` when the
    provider does not answer; an empty list means it answered with nothing."""

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


def keep_hits(hits: Sequence[SearchHit], *, own_host: str | None, limit: int) -> list[SearchHit]:
    """The hits worth reading, in order: one per page, no posts, not the document itself.

    A post is dropped because a platform post is a claim, not a source. X, the case
    that brought this feature, cannot be read at all (spec section 3). The document's
    own host is dropped so a page is never found to support itself.
    """
    own = (own_host or "").lower().removeprefix("www.")
    kept: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        key = canonical(hit.url)
        if key in seen or is_social(hit.url) or (own and _bare_host(hit.url) == own):
            continue
        seen.add(key)
        kept.append(hit)
        if len(kept) >= limit:
            break
    return kept
