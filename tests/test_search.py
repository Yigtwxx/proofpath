"""Evidence search providers, offline: respx over fixtures, no socket (OPEN-ITEMS 17.1a)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from proofpath import polite as pl
from proofpath.config import SearchConfig
from proofpath.polite import ProviderError
from proofpath.search import SearchHit, canonical, keep_hits
from proofpath.search.providers import (
    TAVILY_QUERY_LIMIT,
    TAVILY_URL,
    SearxngSearcher,
    TavilySearcher,
    build_searcher,
)
from proofpath.secrets import ApiKey

FIX = Path(__file__).parent / "fixtures" / "search"
SEARXNG = "http://localhost:8888"


def fixture(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def client() -> pl.PoliteClient:
    return pl.PoliteClient(throttle=pl.HostThrottle(), retries=0)


@respx.mock
def test_tavily_sends_the_key_as_a_bearer_header_and_reads_addresses_only() -> None:
    route = respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(200, json=fixture("tavily.json"))
    )
    searcher = TavilySearcher(client(), ApiKey("tvly-secret", source="test"))

    hits = searcher.search("ChatGPT shut down 2025", 5)

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer tvly-secret"
    body = json.loads(sent.content)
    assert body["query"] == "ChatGPT shut down 2025"
    assert body["max_results"] == 5
    assert body["include_raw_content"] is False
    # The entry with no address is dropped; nothing of ``content`` survives (rule 1).
    assert [hit.url for hit in hits] == [
        "https://news.test/openai-statement?utm_source=x",
        "https://news.test/openai-statement",
        "https://x.com/someone/status/1",
        "https://wiki.test/ChatGPT",
    ]
    assert [hit.rank for hit in hits] == [1, 2, 3, 4]
    assert "SNIPPET" not in repr(hits)


@respx.mock
def test_tavily_cuts_a_long_query_to_its_limit() -> None:
    route = respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    TavilySearcher(client(), ApiKey("k", source="test")).search("word " * 200, 3)
    assert len(json.loads(route.calls.last.request.content)["query"]) <= TAVILY_QUERY_LIMIT


@respx.mock
def test_a_provider_that_refuses_is_a_provider_error_without_the_body() -> None:
    respx.post(TAVILY_URL).mock(return_value=httpx.Response(401, json={"detail": "tvly-secret"}))
    with pytest.raises(ProviderError) as caught:
        TavilySearcher(client(), ApiKey("tvly-secret", source="test")).search("q", 3)
    assert "tvly-secret" not in str(caught.value)


@respx.mock
def test_a_non_json_answer_is_a_provider_error() -> None:
    respx.get(f"{SEARXNG}/search").mock(return_value=httpx.Response(200, text="<html>"))
    with pytest.raises(ProviderError, match="non-JSON"):
        SearxngSearcher(client(), SEARXNG).search("q", 3)


@respx.mock
def test_searxng_asks_for_json_and_keeps_only_web_addresses() -> None:
    route = respx.get(f"{SEARXNG}/search").mock(
        return_value=httpx.Response(200, json=fixture("searxng.json"))
    )
    hits = SearxngSearcher(client(), SEARXNG + "/").search("ChatGPT shut down 2025", 3)
    params = route.calls.last.request.url.params
    assert params["format"] == "json"
    assert params["q"] == "ChatGPT shut down 2025"
    assert "mailto" not in params
    assert hits == [SearchHit("https://wiki.test/ChatGPT", "ChatGPT", 1)]


def test_canonical_drops_the_fragment_and_tracking_parameters() -> None:
    assert canonical("HTTPS://News.Test/a?utm_source=x&id=2#top") == "https://news.test/a?id=2"
    assert canonical("https://news.test") == "https://news.test/"


def test_keep_hits_dedupes_and_drops_posts_and_the_documents_own_host() -> None:
    hits = [
        SearchHit("https://news.test/a?utm_source=x", "", 1),
        SearchHit("https://news.test/a", "", 2),
        SearchHit("https://x.com/u/status/1", "", 3),
        SearchHit("https://www.blog.test/post", "", 4),
        SearchHit("https://wiki.test/A", "", 5),
        SearchHit("https://other.test/B", "", 6),
    ]
    kept = keep_hits(hits, own_host="blog.test", limit=2)
    assert [hit.url for hit in kept] == ["https://news.test/a?utm_source=x", "https://wiki.test/A"]


def test_build_searcher_is_off_by_default() -> None:
    setup = build_searcher(SearchConfig(), client())
    assert setup.searcher is None
    assert setup.problem == ""


def test_build_searcher_names_the_missing_key_and_never_a_value() -> None:
    setup = build_searcher(SearchConfig(provider="tavily"), client(), resolve_key=lambda _: None)
    assert setup.searcher is None
    assert setup.problem == "set TAVILY_API_KEY in .env"


def test_build_searcher_builds_tavily_with_a_key() -> None:
    setup = build_searcher(
        SearchConfig(provider="tavily"),
        client(),
        resolve_key=lambda _: ApiKey("tvly-SECRET", source="t"),
    )
    assert setup.searcher is not None and setup.searcher.name == "tavily"
    assert "tvly-SECRET" not in repr(setup)


def test_build_searcher_needs_a_searxng_address() -> None:
    assert build_searcher(SearchConfig(provider="searxng"), client()).problem == (
        "set search.base_url to your SearXNG instance"
    )
    built = build_searcher(SearchConfig(provider="searxng", base_url=SEARXNG), client())
    assert built.searcher is not None and built.searcher.name == "searxng"
