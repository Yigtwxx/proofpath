"""The two search providers of OPEN-ITEMS 17.1a: Tavily (a key) and SearXNG (an address)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from proofpath.config import SearchConfig
from proofpath.polite import PoliteClient, ProviderError
from proofpath.search import Searcher, SearchHit
from proofpath.secrets import ApiKey, missing_hint, resolve_api_key

TAVILY_URL = "https://api.tavily.com/search"
# Tavily refuses a longer query (API reference, checked 2026-09-28).
TAVILY_QUERY_LIMIT = 400
_WEB = ("http://", "https://")
SEARXNG_NEEDS_URL = "set search.base_url to your SearXNG instance"


class TavilySearcher:
    name = "tavily"

    def __init__(self, client: PoliteClient, key: ApiKey) -> None:
        self._client = client
        self._key = key

    def __repr__(self) -> str:
        return f"TavilySearcher(key={self._key!r})"  # ApiKey's repr never shows the value

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        response = self._client.post(
            TAVILY_URL,
            json_body={
                "query": query[:TAVILY_QUERY_LIMIT],
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            headers={"Authorization": f"Bearer {self._key.value}"},
        )
        return _hits(_payload(response, "api.tavily.com").get("results"), max_results)


class SearxngSearcher:
    name = "searxng"

    def __init__(self, client: PoliteClient, base_url: str) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/search"

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        # ``format=json`` must be enabled in the instance's settings.yml; an instance
        # that has not enabled it answers 403, which is reported, not guessed around.
        response = self._client.get(self._url, {"q": query, "format": "json"}, mailto=False)
        host = httpx.URL(self._url).host
        return _hits(_payload(response, host).get("results"), max_results)


def _payload(response: httpx.Response, host: str) -> dict[str, Any]:
    """The answer's JSON object. The body is never quoted: a provider echoes keys."""
    if response.status_code != 200:
        raise ProviderError(f"HTTP {response.status_code}", response.status_code, None)
    try:
        body = response.json()
    except ValueError:
        raise ProviderError(
            f"{host} answered with a non-JSON body", response.status_code, None
        ) from None
    if not isinstance(body, dict):
        raise ProviderError(f"{host} answered with a non-JSON body", response.status_code, None)
    return body


def _hits(results: Any, limit: int) -> list[SearchHit]:
    """Addresses and titles, in the provider's order. ``content`` is never read."""
    hits: list[SearchHit] = []
    if not isinstance(results, list):
        return hits
    for entry in results:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.lower().startswith(_WEB):
            continue
        title = entry.get("title")
        hits.append(
            SearchHit(url=url, title=title if isinstance(title, str) else "", rank=len(hits) + 1)
        )
        if len(hits) >= limit:
            break
    return hits


@dataclass(frozen=True)
class SearchSetup:
    """The run's searcher, or why a configured one could not be built.

    ``problem`` is empty when search is simply off. A named provider that cannot run
    (no key, no address) is reported rather than skipped (rule 2).
    """

    searcher: Searcher | None
    problem: str


def build_searcher(
    config: SearchConfig,
    client: PoliteClient,
    *,
    resolve_key: Callable[[str], ApiKey | None] = resolve_api_key,
) -> SearchSetup:
    if config.provider == "tavily":
        key = resolve_key(config.api_key_env)
        if key is None:
            return SearchSetup(None, missing_hint(config.api_key_env))
        return SearchSetup(TavilySearcher(client, key), "")
    if config.provider == "searxng":
        if not config.base_url:
            return SearchSetup(None, SEARXNG_NEEDS_URL)
        return SearchSetup(SearxngSearcher(client, config.base_url), "")
    return SearchSetup(None, "")
