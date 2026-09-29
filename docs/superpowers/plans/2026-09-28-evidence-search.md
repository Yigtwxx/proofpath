# Evidence Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A text that cites no source is searched on the web (Tavily or SearXNG), the pages found are read and checked like cited sources, and the report says what was found and that proofpath (not the author) found it.

**Architecture:** A new `Searching` stage in `verify.prepare()`, between Claims and Resolving, runs only when the document cites nothing and a searcher is configured. It picks check-worthy sentences (`claims.checkworthy`), builds queries (the sentence text, or judge-written with a noticed fallback), asks a `Searcher` for addresses, and turns the kept hits into synthetic `Reference(origin="search")` entries. From there, Resolving, Fetching and `decide_all` run unchanged. The report gains three things:
- two kinds, `EVIDENCE_FOUND` and `NO_EVIDENCE`;
- a provenance note on every finding about a found page;
- a `SearchSummary` block.

**Tech Stack:** Python 3.10+, httpx via `polite.PoliteClient`, typer, textual, pytest + respx.

**Spec:** `docs/superpowers/specs/2026-09-28-evidence-search-design.md` (read it first).

## Global Constraints

- Type annotations are mandatory on every function signature. `ruff check` and `ruff format --check` must be clean.
- `pathlib` everywhere; every file read and write passes `encoding="utf-8"`.
- **No network in unit tests.** Use `respx` plus `tests/fixtures/search/*.json`, or stub searchers.
- Secrets: the Tavily key is read only through `secrets.resolve_api_key`. It is never logged, never put in an exception, never in a `repr`, never in config.
- Rule 1: a provider's snippet or `content` is never evidence. Only pages fetched by the ladder are.
- Rule 2: `NO EVIDENCE FOUND (searched)` is never REFUTED; an unreadable found page keeps its own §15 state.
- Rule 4: search never prompts. `permissions.web_search` of `ask` or `deny` turns it off.
- `cli` and `tui` hold no logic: they pass `search=` through to `Engine.default`.
- Code, identifiers and comments in English. Match the surrounding comment style: comments say *why*.
- **Git:** do not commit. Each task ends by staging its files (`git add …`). The user commits.
- Run tests with `uv run pytest …`.

---

### Task 1: `[search]` config section and `permissions.web_search`

**Files:**
- Modify: `src/proofpath/config.py`
- Modify: `src/proofpath/settings_hints.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `config.SearchProvider = Literal["off", "tavily", "searxng"]` and `config.SEARCH_PROVIDERS: tuple[str, ...]`.
  - `config.SearchConfig(provider: SearchProvider = "off", api_key_env: str = "TAVILY_API_KEY", base_url: str = "", max_claims: int = 5, results_per_claim: int = 3)`.
  - `Config.search: SearchConfig`.
  - `Permissions.web_search: Permission = "allow"`.
  - `settings_hints.SEARCH_SETTING = "search.provider tavily"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_config.py`; add any missing imports: `SearchConfig`, `set_value`, `load_config`, `ConfigError`, `pytest`, `Path`)

```python
def test_search_is_off_by_default_and_web_search_is_allowed() -> None:
    config = Config()
    assert config.search == SearchConfig()
    assert config.search.provider == "off"
    assert config.search.api_key_env == "TAVILY_API_KEY"
    assert (config.search.max_claims, config.search.results_per_claim) == (5, 3)
    assert config.permissions.web_search == "allow"


def test_the_search_section_loads_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[search]\nprovider = "searxng"\nbase_url = "http://localhost:8888"\nmax_claims = 2\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.search.provider == "searxng"
    assert config.search.base_url == "http://localhost:8888"
    assert config.search.max_claims == 2


@pytest.mark.parametrize(
    "body",
    ['provider = "bing"', "max_claims = 0", "max_claims = true", 'max_claims = "5"'],
)
def test_bad_search_values_are_refused_by_name(tmp_path: Path, body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[search]\n{body}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"search\."):
        load_config(path)


def test_set_value_parses_an_integer_and_round_trips_it(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    updated = set_value("search.max_claims", "3", path)
    assert updated.search.max_claims == 3
    assert "max_claims = 3" in path.read_text(encoding="utf-8")
    assert load_config(path).search.max_claims == 3
    with pytest.raises(ConfigError, match="positive integer"):
        set_value("search.max_claims", "many", path)


def test_set_value_takes_a_search_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    assert set_value("search.provider", "tavily", path).search.provider == "tavily"
    with pytest.raises(ConfigError, match="search.provider"):
        set_value("search.provider", "bing", path)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_config.py -k "search" -v`
Expected: FAIL with `ImportError: cannot import name 'SearchConfig'`.

- [ ] **Step 3: Implement in `src/proofpath/config.py`**

Change the typing import to `from typing import Any, Literal, get_args, get_origin`. Add `web_search` to `Permissions`:

```python
@dataclass(frozen=True)
class Permissions:
    # Step 3 of the fetch ladder: ~280 MB browser engine. Never installed silently.
    install_browser: Permission = "ask"
    network: Permission = "allow"
    # Evidence search for a text that cites nothing (OPEN-ITEMS 17.1a). Only matters
    # once ``search.provider`` is set: configuring a provider is the consent. The
    # search never prompts, so ``ask`` is read as ``deny`` (rule 4).
    web_search: Permission = "allow"
```

After `JudgeConfig`, add:

```python
SearchProvider = Literal["off", "tavily", "searxng"]
SEARCH_PROVIDERS: tuple[str, ...] = get_args(SearchProvider)


@dataclass(frozen=True)
class SearchConfig:
    """Evidence search for a text that cites nothing (OPEN-ITEMS 17.1a). Off until a
    provider is named. The key is never stored here: ``api_key_env`` names the
    variable that holds it, exactly as ``JudgeConfig`` does."""

    provider: SearchProvider = "off"
    api_key_env: str = "TAVILY_API_KEY"
    base_url: str = ""  # SearXNG only: the user's own instance
    max_claims: int = 5
    results_per_claim: int = 3
```

Add `search: SearchConfig = field(default_factory=SearchConfig)` to `Config` after `judge`, and `"search": SearchConfig,` to `_SECTIONS` after `"judge"`.

In `_coerce`, after the `Permission` branch, add:

```python
    if get_origin(expected) is Literal:
        choices = get_args(expected)
        if isinstance(value, str) and value in choices:
            return value
        raise ConfigError(f"{path}: {where} must be one of {', '.join(choices)}, got {value!r}")
    if expected is int:
        # ``bool`` is an ``int`` to Python and a different answer to a reader.
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        raise ConfigError(f"{path}: {where} must be a positive integer, got {value!r}")
```

In `_resolve_type`, extend the map:

```python
        return {
            "Permission": Permission,
            "SearchProvider": SearchProvider,
            "bool": bool,
            "int": int,
            "str": str,
        }[annotation]
```

In `_toml_value`, after the `bool` check (the order matters, because a bool is an int):

```python
    if isinstance(value, int):
        return str(value)
```

In `set_value`, after the `bool` block:

```python
if expected is int:
    try:
        value = int(raw_value.strip())
    except ValueError:
        raise ConfigError(f"{dotted_key} must be a positive integer, got {raw_value!r}") from None
```

In `src/proofpath/settings_hints.py`, append:

```python
#: ``search.provider`` set to ``tavily``: a text that cites nothing is searched with
#: the user's own Tavily key (OPEN-ITEMS 17.1a).
SEARCH_SETTING = "search.provider tavily"
```

- [ ] **Step 4: Run the config tests**

Run: `uv run pytest tests/test_config.py tests/test_config_cli.py -v`
Expected: PASS. If a test pins `render_config`'s exact output or the list of sections, extend its expectation with the new `[search]` block and `web_search = "allow"`. Do not change the renderer.

- [ ] **Step 5: Stage**

```bash
git add src/proofpath/config.py src/proofpath/settings_hints.py tests/test_config.py tests/test_config_cli.py
```

---

### Task 2: JSON POST on `PoliteClient`, and Tavily's interval

**Files:**
- Modify: `src/proofpath/polite.py` (`MIN_INTERVAL`, `post`, `_send`)
- Test: `tests/test_polite.py`

**Interfaces:**
- Produces: `PoliteClient.post(url, *, data=None, json_body: dict[str, Any] | None = None, headers=None, auth=None) -> httpx.Response`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_polite.py`)

```python
@respx.mock
def test_post_sends_a_json_body_and_keeps_headers_out_of_the_url() -> None:
    route = respx.post("https://api.test/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    client = pl.PoliteClient(throttle=pl.HostThrottle())
    response = client.post(
        "https://api.test/search",
        json_body={"query": "a claim", "max_results": 3},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    sent = route.calls.last.request
    assert json.loads(sent.content) == {"query": "a claim", "max_results": 3}
    assert sent.headers["Authorization"] == "Bearer secret"
    assert "secret" not in str(sent.url)


def test_tavily_has_a_polite_interval() -> None:
    assert pl.MIN_INTERVAL["api.tavily.com"] == 0.5
```

(Add `import json` to the imports if it is missing.)

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_polite.py -k "json_body or tavily" -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'json_body'`.

- [ ] **Step 3: Implement**

In `MIN_INTERVAL`, after `"oauth.reddit.com": 1.0,`:

```python
    # Evidence search (OPEN-ITEMS 17.1a): a run asks at most a handful of queries, and
    # the user's own key is what they spend. SearXNG is the user's own instance, so it
    # is not listed and gets the default.
    "api.tavily.com": 0.5,
```

Change `post` to accept a JSON body, and pass it through `_send`:

```python
    def post(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> httpx.Response:
        """A POST, throttled and retried like :meth:`get`.

        Two callers: the OAuth token exchange a user's own Reddit app makes (spec
        section 6.2), which sends a form in ``data``, and the evidence search, which
        sends JSON in ``json_body``. A credential rides in ``auth`` or ``headers`` and
        never in the URL: a query string is the part of a request that gets logged.
        """
        return self._send("POST", url, data=data, json_body=json_body, headers=headers, auth=auth)
```

In `_send`, add the parameter `json_body: dict[str, Any] | None = None` after `data`, and pass `json=json_body,` to `self._client.request(...)` after `data=data,`.

- [ ] **Step 4: Run the polite tests**

Run: `uv run pytest tests/test_polite.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/proofpath/polite.py tests/test_polite.py
```

---

### Task 3: The `search` package (the protocol, Tavily, SearXNG, hit filtering)

**Files:**
- Create: `src/proofpath/search/__init__.py`
- Create: `src/proofpath/search/providers.py`
- Create: `tests/fixtures/search/tavily.json`
- Create: `tests/fixtures/search/searxng.json`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `PoliteClient.post(json_body=…)` (Task 2), `SearchConfig` (Task 1), `secrets.resolve_api_key`, `secrets.missing_hint`, `polite.ProviderError`, `providers.is_social`.
- Produces:
  - `search.SearchHit(url: str, title: str, rank: int)`, frozen.
  - `search.Searcher` Protocol: `name: str`; `search(query: str, max_results: int) -> list[SearchHit]`, which raises `ProviderError` when the provider does not answer.
  - `search.canonical(url: str) -> str` and `search.keep_hits(hits: Sequence[SearchHit], *, own_host: str | None, limit: int) -> list[SearchHit]`.
  - `search.providers.TavilySearcher(client: PoliteClient, key: ApiKey)` and `search.providers.SearxngSearcher(client: PoliteClient, base_url: str)`.
  - `search.providers.SearchSetup(searcher: Searcher | None, problem: str)` and `search.providers.build_searcher(config: SearchConfig, client: PoliteClient, *, resolve_key: Callable[[str], ApiKey | None] = resolve_api_key) -> SearchSetup`.
  - `search.providers.TAVILY_URL = "https://api.tavily.com/search"` and `TAVILY_QUERY_LIMIT = 400`.

- [ ] **Step 1: Write the fixtures**

`tests/fixtures/search/tavily.json`:

```json
{
  "query": "ChatGPT shut down 2025",
  "results": [
    {"title": "OpenAI statement", "url": "https://news.test/openai-statement?utm_source=x", "content": "SNIPPET-MUST-NOT-BE-EVIDENCE", "score": 0.91},
    {"title": "Duplicate", "url": "https://news.test/openai-statement", "content": "dup", "score": 0.8},
    {"title": "A post", "url": "https://x.com/someone/status/1", "content": "post", "score": 0.7},
    {"title": "Encyclopedia", "url": "https://wiki.test/ChatGPT", "content": "wiki", "score": 0.6},
    {"title": "No address", "content": "broken entry", "score": 0.5}
  ],
  "response_time": 0.8
}
```

`tests/fixtures/search/searxng.json`:

```json
{
  "query": "ChatGPT shut down 2025",
  "number_of_results": 2,
  "results": [
    {"url": "https://wiki.test/ChatGPT", "title": "ChatGPT", "content": "wiki", "engine": "wikipedia"},
    {"url": "ftp://files.test/x", "title": "Not the web", "content": "", "engine": "x"}
  ]
}
```

- [ ] **Step 2: Write the failing tests** in `tests/test_search.py`:

```python
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
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proofpath.search'`.

- [ ] **Step 4: Implement `src/proofpath/search/__init__.py`**

```python
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
```

- [ ] **Step 5: Implement `src/proofpath/search/providers.py`**

```python
"""The two search providers of OPEN-ITEMS 17.1a: Tavily (a key) and SearXNG (an address)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from proofpath.config import SearchConfig
from proofpath.polite import PoliteClient, ProviderError
from proofpath.search import SearchHit, Searcher
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
```

Check `ProviderError.__init__` in `polite.py` before this step. It is raised as `ProviderError(last, status, code)`. If its parameters are keyword-only or named differently, match them.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS.

- [ ] **Step 7: Stage**

```bash
git add src/proofpath/search tests/fixtures/search tests/test_search.py
```

---

### Task 4: Check-worthy sentences (`claims.checkworthy`)

**Files:**
- Modify: `src/proofpath/claims.py` (next to `pair_links`)
- Test: `tests/test_claims.py`

**Interfaces:**
- Produces:
  - `claims.is_checkworthy(text: str) -> bool`.
  - `claims.Checkworthy(claims: tuple[Claim, ...], eligible: int, sentences: int)`, frozen.
  - `claims.checkworthy(doc: Document, *, limit: int) -> Checkworthy`. The claims it returns have `cited_refs=()` and `marker=None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_claims.py`; import `from proofpath import ingest` if it is missing)

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ChatGPT was shut down yesterday.", True),  # a name with an inner capital
        ("The unemployment rate fell to 3.9% in 2023.", True),  # a figure
        ("Istanbul is the largest city in Turkey.", True),  # a name past the first word
        ('He said "we will never raise taxes" on stage.', True),  # a quotation
        ("I think this is really bad for everyone.", False),  # opinion, nothing to find
        ("Did Tesla really sell 2 million cars?", False),  # a question
        ("It rose 5%.", False),  # too short to state anything searchable
    ],
)
def test_is_checkworthy(text: str, expected: bool) -> None:
    assert claims.is_checkworthy(text) is expected


def test_checkworthy_keeps_document_order_up_to_the_cap_and_counts_the_rest() -> None:
    doc = ingest.from_text(
        "NASA landed on the Moon in 1969. I love this so much honestly. "
        "Paris is the capital of France. The rate was 4% in 2020 overall.",
        name="pasted text",
        kind="text",
    )
    found = claims.checkworthy(doc, limit=2)

    assert [claim.text for claim in found.claims] == [
        "NASA landed on the Moon in 1969.",
        "Paris is the capital of France.",
    ]
    assert found.eligible == 3
    assert found.sentences == 4
    assert all(claim.cited_refs == () and claim.marker is None for claim in found.claims)
```

(Adjust the `claims` alias to how the test module already imports the module; it may be `from proofpath import claims` or `claims_mod`.)

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_claims.py -k checkworthy -v`
Expected: FAIL with `AttributeError: module 'proofpath.claims' has no attribute 'is_checkworthy'`.

- [ ] **Step 3: Implement** (in `claims.py`, after `pair_links`; `dataclass` is already imported there because `Claims` is one, so check)

```python
# What makes a sentence worth searching for (OPEN-ITEMS 17.1a): long enough to state
# something, not a question, and carrying something a page could confirm -- a figure,
# a name, or a quotation. Opinion ("I think this is bad") has nothing to find.
CHECKWORTHY_MIN_WORDS = 5
_WORDS = re.compile(r"\w+")
_FIGURE = re.compile(r"\d")
_QUOTED = re.compile(r"[\"“”«»]")


@dataclass(frozen=True)
class Checkworthy:
    """The sentences a search will be run for, and what the cap left out."""

    claims: tuple[Claim, ...]  # in document order, at most the cap
    eligible: int  # check-worthy sentences before the cap
    sentences: int  # every sentence with a word in it: the coverage denominator


def is_checkworthy(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return False
    words = _WORDS.findall(stripped)
    if len(words) < CHECKWORTHY_MIN_WORDS:
        return False
    if _FIGURE.search(stripped) or _QUOTED.search(stripped):
        return True
    # A capital past the first word is a name ("Turkey"); a capital inside a word is
    # one too ("ChatGPT", "NASA"), even as the sentence's first word.
    return any(word[:1].isupper() for word in words[1:]) or any(
        any(char.isupper() for char in word[1:]) for word in words
    )


def checkworthy(doc: Document, *, limit: int) -> Checkworthy:
    """Claims for a document that cites nothing, to be searched for.

    They cite no reference yet. The Searching stage points each at the pages it finds,
    and a claim it finds nothing for keeps an empty ``cited_refs`` and is reported
    as NO EVIDENCE FOUND, never dropped (product rule 6).
    """
    kept: list[Claim] = []
    eligible = 0
    sentences = 0
    for paragraph in doc.paragraphs:
        for position, sentence in enumerate(paragraph.sentences):
            text = strip_links(sentence.text)
            if not _WORD.search(text):
                continue
            sentences += 1
            if not is_checkworthy(text):
                continue
            eligible += 1
            if len(kept) < limit:
                kept.append(
                    Claim(
                        text=text,
                        locator=sentence.locator,
                        cited_refs=(),
                        paragraph=paragraph.index,
                        sentence=position,
                    )
                )
    return Checkworthy(claims=tuple(kept), eligible=eligible, sentences=sentences)
```

If `ingest.from_text` splits the test paragraph into sentences differently than expected, check `ingest`'s sentence splitter. Adjust the **test input** (for example, put each sentence on its own line), not the splitter.

- [ ] **Step 4: Run the claims tests**

Run: `uv run pytest tests/test_claims.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/proofpath/claims.py tests/test_claims.py
```

#### Amendment A (2026-09-28): review fixes, decided with the user

The heuristic above was revised in a fix round; the authoritative code is the one in the
tree. Summary of the changes:
- `#tags` and `@handles` are stripped before counting.
- A word keeps its apostrophe suffix ("Türkiye'nin").
- The minimum drops to 3 words.
- A capitalised first word counts as a name unless it is a common English or Turkish opener.
- When **no** sentence qualifies, every sentence is taken, up to the cap, and `eligible` then equals `sentences`. This is why "OpenAI battı" is searched.

---

### Task 5: Queries (the sentence text, `Judge.queries`, the LLM fallback notice)

**Files:**
- Create: `src/proofpath/prompts/queries.md`
- Create: `src/proofpath/search/queries.py`
- Modify: `src/proofpath/judge.py` (`Judge.__init__`, a new `Judge.queries`, `_QUERIES_SCHEMA`, `_queries_from`)
- Test: `tests/test_search_queries.py`

**Interfaces:**
- Consumes: `Judge`, `JudgeUnavailable`, `Completion`, and `JudgeCost` from `judge.py`.
- Produces:
  - `Judge.queries(claims: Sequence[str]) -> dict[int, tuple[str, ...]]`. It never raises; on trouble it sets `unavailable` and `detail`.
  - `search.queries.QUERY_LIMIT = 400`, `SENTENCE_BY = "sentence"`, `LLM_LIMIT = "LLM limit reached — searched with the sentence text"`, `LLM_UNAVAILABLE = "LLM did not answer — searched with the sentence text"`.
  - `search.queries.sentence_query(text: str) -> str`.
  - `search.queries.QueryPlan(queries: tuple[tuple[str, ...], ...], by: str, notice: str | None)`.
  - `search.queries.plan_queries(texts: Sequence[str], judge: Judge | None) -> QueryPlan`.

- [ ] **Step 1: Write the prompt** `src/proofpath/prompts/queries.md`

```markdown
## System

You write web search queries for a fact-checking tool. You never say whether a claim
is true: another stage reads the pages your queries find and decides that from their
own words. Your only job is queries that would find a page stating the facts the
claim is about.

## User

For each claim below, write one or two search queries.

Rules:

- Make each query stand alone: replace pronouns and vague references ("he", "the
  company", "yesterday") with what the claim itself names, when it names it.
- Keep the claim's names, numbers and dates; drop opinion words and emotion.
- Plain keywords or a short sentence, at most 25 words, in the claim's language.
- Add no fact, answer or source that is not in the claim.

Answer with JSON only: {"items": [{"id": <id>, "queries": ["...", "..."]}]}

Claims:

$items
```

- [ ] **Step 2: Write the failing tests** in `tests/test_search_queries.py`:

```python
"""Query building for the evidence search: the sentence, the judge, and the fallback."""

from __future__ import annotations

import json
from typing import Any

from proofpath.judge import Completion, Judge, JudgeCost, JudgeUnavailable
from proofpath.search.queries import (
    LLM_LIMIT,
    LLM_UNAVAILABLE,
    QUERY_LIMIT,
    SENTENCE_BY,
    plan_queries,
    sentence_query,
)


class ScriptedClient:
    """``JudgeClient`` without a socket: one scripted answer or exception."""

    provider = "fake"
    model = "q-1"

    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.cost = JudgeCost(model=self.model)
        self.prompts: list[str] = []

    def complete(self, messages: list[dict[str, str]], **_: Any) -> Completion:
        self.prompts.append(messages[-1]["content"])
        if isinstance(self.answer, Exception):
            raise self.answer
        return Completion(text=self.answer, prompt_tokens=1, completion_tokens=1, model=self.model)

    def close(self) -> None:
        pass


def judge(answer: str | Exception) -> Judge:
    return Judge(ScriptedClient(answer))  # type: ignore[arg-type]


def test_sentence_query_strips_links_handles_hashtag_signs_and_emoji() -> None:
    text = "🚨 @newsbot says #ChatGPT was shut down https://t.co/xyz today!"
    assert sentence_query(text) == "says ChatGPT was shut down today!"


def test_sentence_query_cuts_at_a_word_boundary() -> None:
    query = sentence_query("word " * 200)
    assert len(query) <= QUERY_LIMIT
    assert not query.endswith(" ")


def test_without_a_judge_the_sentence_is_the_query() -> None:
    plan = plan_queries(["ChatGPT was shut down in 2025."], None)
    assert plan.queries == (("ChatGPT was shut down in 2025.",),)
    assert plan.by == SENTENCE_BY
    assert plan.notice is None


def test_the_judge_writes_decontextualised_queries() -> None:
    answer = json.dumps({"items": [{"id": 0, "queries": ["OpenAI ChatGPT shutdown 2025"]}]})
    plan = plan_queries(["He shut it down in 2025."], judge(answer))
    assert plan.queries == (("OpenAI ChatGPT shutdown 2025",),)
    assert plan.by == "fake q-1"
    assert plan.notice is None


def test_a_rate_limited_judge_falls_back_to_the_sentence_and_says_so() -> None:
    plan = plan_queries(
        ["ChatGPT was shut down in 2025."],
        judge(JudgeUnavailable("HTTP 429 from https://api.test/v1/chat/completions")),
    )
    assert plan.queries == (("ChatGPT was shut down in 2025.",),)
    assert plan.by == SENTENCE_BY
    assert plan.notice is not None and plan.notice.startswith(LLM_LIMIT)


def test_a_judge_that_is_down_falls_back_with_its_own_words() -> None:
    plan = plan_queries(["ChatGPT was shut down in 2025."], judge(JudgeUnavailable("HTTP 503")))
    assert plan.notice is not None and plan.notice.startswith(LLM_UNAVAILABLE)


def test_a_claim_the_judge_skipped_keeps_its_sentence() -> None:
    answer = json.dumps({"items": [{"id": 1, "queries": ["Paris capital France"]}]})
    plan = plan_queries(["NASA landed in 1969.", "Paris is the capital."], judge(answer))
    assert plan.queries == (("NASA landed in 1969.",), ("Paris capital France",))


def test_judge_queries_never_raises_on_an_unreadable_answer() -> None:
    built = judge("not json at all")
    assert built.queries(["a claim"]) == {}
    assert built.unavailable is False
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/test_search_queries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proofpath.search.queries'`.

- [ ] **Step 4: Implement `Judge.queries`** in `judge.py`

Add near `_REVIEW_SCHEMA`:

```python
# Evidence search (OPEN-ITEMS 17.1a): the judge writes *queries*, never an answer.
# Deciding a claim without a passage is the truth oracle spec section 3 forbids.
_QUERIES_TOKENS = 1500
_QUERIES_PER_CLAIM = 2
_QUERY_CHARS = 400
_QUERIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "queries"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
```

In `Judge.__init__`, after `self._summary_template = …`:

```python
        self._queries_template = load_prompt("queries")
```

Add this method after `summarize`:

```python
    def queries(self, claims: Sequence[str]) -> dict[int, tuple[str, ...]]:
        """One or two search queries per claim, keyed by its position; ``{}`` on trouble.

        It never raises, like ``review``: a provider that is down or out of quota costs
        the written queries, and ``unavailable``/``detail`` say why, so the run can fall
        back to the sentence itself and tell the reader (OPEN-ITEMS 17.1a).
        """
        self.unavailable = False
        self.detail = ""
        if not claims:
            return {}
        items = "\n".join(
            f"- id: {index}\n  claim: {_one_line(text)}" for index, text in enumerate(claims)
        )
        system, user = _split_prompt(self._queries_template.substitute(items=items))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            completion = self._client.complete(
                messages,
                json_schema=_QUERIES_SCHEMA,
                max_tokens=_QUERIES_TOKENS,
                reasoning_effort=_REASONING_EFFORT,
            )
        except JudgeUnavailable as exc:
            self._give_up(str(exc))
            return {}
        except Exception as exc:
            # The type is named, the message is not: it can carry the key.
            self._give_up(f"{type(exc).__name__} from the judge client")
            return {}
        return _queries_from(completion.text, len(claims))
```

After `_loads`, add:

```python
def _queries_from(text: str, count: int) -> dict[int, tuple[str, ...]]:
    """The ``items`` of a queries answer, read leniently; an id out of range is dropped."""
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            continue
        found: dict[int, tuple[str, ...]] = {}
        for entry in payload["items"]:
            if not isinstance(entry, dict):
                continue
            try:
                ident = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            raw = entry.get("queries")
            if not 0 <= ident < count or not isinstance(raw, list):
                continue
            written = tuple(
                _one_line(query)[:_QUERY_CHARS]
                for query in raw
                if isinstance(query, str) and query.strip()
            )[:_QUERIES_PER_CLAIM]
            if written:
                found[ident] = written
        return found
    return {}
```

- [ ] **Step 5: Implement `src/proofpath/search/queries.py`**

```python
"""What the evidence search asks: the sentence itself, or what the judge wrote.

The default makes zero LLM calls (spec section 11). With ``--judge``, the judge
rewrites each claim as a query that stands on its own. When it cannot -- a rate limit,
an exhausted quota, an outage -- the sentence is used instead and the run says so,
because a weaker query is a weaker search and the reader has to know which one they
got (product rule 6).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proofpath.judge import Judge

QUERY_LIMIT = 400
SENTENCE_BY = "sentence"
LLM_LIMIT = "LLM limit reached — searched with the sentence text"
LLM_UNAVAILABLE = "LLM did not answer — searched with the sentence text"

_URL = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"(?<!\w)@\w+")
# A hashtag is often the claim's subject ("#ChatGPT"): the sign goes, the word stays.
_HASH = re.compile(r"(?<!\w)#(\w+)")
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️‍]")


def sentence_query(text: str) -> str:
    """The sentence as a query: no addresses, handles or emoji, at most 400 characters."""
    cleaned = _EMOJI.sub(" ", _HANDLE.sub(" ", _URL.sub(" ", text)))
    cleaned = " ".join(_HASH.sub(r"\1", cleaned).split())
    if len(cleaned) <= QUERY_LIMIT:
        return cleaned
    cut = cleaned[:QUERY_LIMIT]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


@dataclass(frozen=True)
class QueryPlan:
    queries: tuple[tuple[str, ...], ...]  # per claim, in claim order; never empty
    by: str  # SENTENCE_BY, or the judge's name
    notice: str | None  # set when the judge was asked and the sentence was used instead


def fallback_notice(detail: str) -> str:
    head = LLM_LIMIT if "HTTP 429" in detail else LLM_UNAVAILABLE
    return f"{head} ({detail})"


def plan_queries(texts: Sequence[str], judge: Judge | None) -> QueryPlan:
    fallback = tuple((sentence_query(text),) for text in texts)
    if judge is None or not texts:
        return QueryPlan(fallback, SENTENCE_BY, None)
    written = judge.queries(texts)
    if judge.unavailable:
        return QueryPlan(fallback, SENTENCE_BY, fallback_notice(judge.detail))
    queries = tuple(written.get(index) or fallback[index] for index in range(len(texts)))
    return QueryPlan(queries, judge.name, None)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_search_queries.py tests/test_judge.py -v`
Expected: PASS. Check that `prompts/queries.md` ships in the wheel: `grep -n "prompts" pyproject.toml`. If the prompts are listed file by file, add `queries.md` beside `review.md`.

- [ ] **Step 7: Stage**

```bash
git add src/proofpath/prompts/queries.md src/proofpath/search/queries.py src/proofpath/judge.py tests/test_search_queries.py pyproject.toml
```

#### Amendment A (2026-09-28): claims that are not in English

The models are English-only (`cross-encoder/nli-deberta-v3-base`, `bge-small-en-v1.5`). The user decided: a claim that does not look English is **translated by the judge** when there is one. The English text drives both the queries and the check. With no judge, or a judge that did not answer, the claim has no English text and Task 7 reports it as `UNVERIFIED (language not supported)`. This amendment changes the interfaces above:

- **Produces (new):**
  - `search.language.looks_english(text: str) -> bool`.
  - `search.queries.QueryPlan` gains `hypotheses: tuple[str | None, ...]`: per claim, the English text the models check, or `None` when the claim is not English and nothing translated it.
- **Changes:** `Judge.queries(claims) -> dict[int, JudgedClaim]`, where `judge.JudgedClaim(queries: tuple[str, ...], english: str)`, frozen. `english` is `""` when the model gave none.

Create `src/proofpath/search/language.py`:

```python
"""Whether a claim reads as English, the only language the models were trained on.

Deliberately small and one-sided: English is the default, and a claim is called
something else only on positive evidence -- a letter outside ASCII (ç, ğ, ı, ö, ş,
ü, Cyrillic, CJK...) or a common function word of another language. A wrong
"English" costs a weaker check; a wrong "not English" would cost the claim its
search without a judge, so the doubt goes to English.
"""

from __future__ import annotations

import re

# Function words of languages a reader is likely to paste, none of them English words.
_FOREIGN_WORDS = frozenset(
    {
        # Turkish
        "ve", "bir", "bu", "için", "ile", "değil", "çok", "daha", "gibi", "olan",
        "olarak", "ama", "şu", "mi", "mı", "mu", "mü", "da", "de", "ki", "yok", "var",
        # Spanish / Portuguese / Italian
        "el", "los", "las", "que", "y", "es", "por", "con", "una", "para", "del",
        "não", "uma", "il", "della", "che", "sono",
        # German / Dutch
        "der", "das", "und", "ist", "nicht", "ein", "eine", "mit", "den", "het", "een",
        # French
        "le", "les", "et", "est", "une", "des", "du", "pas", "pour", "dans",
    }
)  # fmt: skip
_WORDS = re.compile(r"[^\W\d_]+")


def looks_english(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if any(ord(char) > 127 for char in letters):
        return False
    return not any(word.casefold() in _FOREIGN_WORDS for word in _WORDS.findall(text))
```

In `judge.py`:
- Add `JudgedClaim`, next to `JudgeItem`:

  ```python
  @dataclass(frozen=True)
  class JudgedClaim:
      """What the judge wrote for one claim: search queries, and the claim in English."""

      queries: tuple[str, ...]
      english: str  # "" when the model gave none
  ```
- Extend `_QUERIES_SCHEMA`'s item with `"english": {"type": "string"}` and add `"english"` to `required`.
- `_queries_from` returns `dict[int, JudgedClaim]`. Keep an entry when it has queries **or** a non-empty `english`. `english = _one_line(str(entry.get("english", "")))[:_QUERY_CHARS]`.
- `queries()`'s return annotation and docstring change to match.

In `prompts/queries.md`:
- Change the user instruction to: "For each claim below, write one or two search queries **in English**, and the claim itself in English."
- Add this rule: "- `english`: the claim in English, as close to word for word as English allows; if it is already English, repeat it unchanged. Add nothing, soften nothing."
- The JSON line becomes: `Answer with JSON only: {"items": [{"id": <id>, "english": "...", "queries": ["...", "..."]}]}`

In `search/queries.py`:
- `QueryPlan` gains `hypotheses: tuple[str | None, ...]` after `queries`.
- `plan_queries` becomes:

```python
def plan_queries(texts: Sequence[str], judge: Judge | None) -> QueryPlan:
    english = tuple(looks_english(text) for text in texts)
    fallback = tuple((sentence_query(text),) for text in texts)
    # An English claim is checked as written; another language only once translated.
    as_written = tuple(text if eng else None for text, eng in zip(texts, english, strict=True))
    if judge is None or not texts:
        return QueryPlan(fallback, as_written, SENTENCE_BY, None)
    written = judge.queries(texts)
    if judge.unavailable:
        return QueryPlan(fallback, as_written, SENTENCE_BY, fallback_notice(judge.detail))
    queries = tuple(
        (written[i].queries if i in written and written[i].queries else fallback[i])
        for i in range(len(texts))
    )
    hypotheses = tuple(
        texts[i] if english[i] else ((written[i].english or None) if i in written else None)
        for i in range(len(texts))
    )
    return QueryPlan(queries, hypotheses, judge.name, None)
```

Tests to add in `tests/test_search_queries.py`. Answers now carry `english`. Update the existing judge answers to include `"english": "<the claim>"`.

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("OpenAI shut down ChatGPT in 2025.", True),
        ("NASA confirms water on Mars", True),
        ("OpenAI battı.", False),  # ı
        ("OpenAI batti ve kapandi.", False),  # ASCII Turkish, "ve"
        ("El gobierno cerró la empresa.", False),
        ("Die Firma ist pleite.", False),  # "ist"
    ],
)
def test_looks_english(text: str, expected: bool) -> None:
    assert looks_english(text) is expected


def test_an_english_claim_is_checked_as_written() -> None:
    plan = plan_queries(["OpenAI shut down ChatGPT in 2025."], None)
    assert plan.hypotheses == ("OpenAI shut down ChatGPT in 2025.",)


def test_without_a_judge_a_turkish_claim_has_nothing_to_check() -> None:
    plan = plan_queries(["OpenAI battı."], None)
    assert plan.hypotheses == (None,)


def test_the_judge_translates_a_turkish_claim_for_the_check_and_the_queries() -> None:
    answer = json.dumps(
        {"items": [{"id": 0, "english": "OpenAI went bankrupt.", "queries": ["OpenAI bankruptcy"]}]}
    )
    plan = plan_queries(["OpenAI battı."], judge(answer))
    assert plan.hypotheses == ("OpenAI went bankrupt.",)
    assert plan.queries == (("OpenAI bankruptcy",),)


def test_a_rate_limited_judge_leaves_a_turkish_claim_untranslated() -> None:
    plan = plan_queries(
        ["OpenAI battı."], judge(JudgeUnavailable("HTTP 429 from https://api.test"))
    )
    assert plan.hypotheses == (None,)
    assert plan.notice is not None and plan.notice.startswith(LLM_LIMIT)
```

(Import `pytest` and `looks_english`. `src/proofpath/search/language.py` is a new file: `git add -N` it.)

---

### Task 6: The report model (origin, the two kinds, `SearchSummary`, renderers)

**Files:**
- Modify: `src/proofpath/document.py` (`Reference.origin`)
- Modify: `src/proofpath/report.py`
- Modify: `src/proofpath/sarif.py` (`RULE_DESCRIPTIONS`, `_run_properties`)
- Modify: `tests/data/verify-report-golden.json` (regenerated)
- Test: `tests/test_report.py`, `tests/test_sarif.py`

**Interfaces:**
- Produces:
  - `document.Origin = Literal["author", "search"]` and `Reference.origin: Origin = "author"`.
  - `Kind.EVIDENCE_FOUND = "evidence-found"`: level `note`, state `"SUPPORTED (found by proofpath)"`, asserting (needs a passage).
  - `Kind.NO_EVIDENCE = "no-evidence-found"`: level `warning`, state `"NO EVIDENCE FOUND (searched)"`.
  - `report.FOUND_BY_PROOFPATH = "FOUND BY PROOFPATH (not cited by the author)"`, `report.SEARCH_UNAVAILABLE = "UNVERIFIED (search unavailable)"`, `report.AVERITEC_SEARCH_SCORE = "not measured yet"`, and `report.search_experimental() -> str`.
  - `report.SearchSummary(by, queries_by, sentences, eligible, searched, pages_found, pages_read, notices=())`.
  - `Report.search: SearchSummary | None = None`.
  - `report.search_lines(report: Report) -> tuple[str, ...]` and `Footer.search: tuple[str, ...] = ()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_report.py`, which already has `report_with(*findings)` to build a `Report`)

```python
def test_the_two_search_kinds_carry_their_section_15_words() -> None:
    assert STATE_WORDS[Kind.EVIDENCE_FOUND] == "SUPPORTED (found by proofpath)"
    assert LEVELS[Kind.EVIDENCE_FOUND] == "note"
    assert STATE_WORDS[Kind.NO_EVIDENCE] == "NO EVIDENCE FOUND (searched)"
    assert LEVELS[Kind.NO_EVIDENCE] == "warning"


def test_evidence_found_needs_a_passage() -> None:
    with pytest.raises(ValueError, match="passage"):
        Finding(
            kind=Kind.EVIDENCE_FOUND, level="note", locator=Locator(line=1),
            title="t", state=STATE_WORDS[Kind.EVIDENCE_FOUND], reference=None, claim=None,
            verdict=None, source_id=None, fetch_step=None, tier=None,
        )  # fmt: skip


def test_a_reference_is_the_authors_unless_the_search_found_it() -> None:
    assert Reference(1, "x", Locator(line=1)).origin == "author"
    assert Reference(1, "x", Locator(line=1), origin="search").origin == "search"


SUMMARY = SearchSummary(
    by="tavily", queries_by="sentence", sentences=7, eligible=4, searched=3,
    pages_found=9, pages_read=6, notices=("LLM limit reached — searched with the sentence text",),
)  # fmt: skip


def test_search_lines_state_what_was_searched_and_what_was_not() -> None:
    lines = search_lines(replace(report_with(), search=SUMMARY))
    assert lines[0] == search_experimental()
    assert "evidence search by tavily, queries by sentence" in lines
    assert "claims searched 3 of 4 check-worthy (7 sentences)" in lines
    assert "claims not searched 4" in lines
    assert "pages found / read 9 / 6" in lines
    assert lines[-1].startswith("LLM limit reached")


def test_a_run_that_did_not_search_has_no_search_lines() -> None:
    assert search_lines(report_with()) == ()
    assert render_footer(report_with()).search == ()


def test_the_footer_and_the_markdown_carry_the_search_block() -> None:
    report = replace(report_with(), search=SUMMARY)
    assert render_footer(report).search == search_lines(report)
    markdown = render_markdown(report)
    assert "## Evidence search" in markdown
    assert "- claims not searched 4" in markdown
```

(Import `replace` from `dataclasses`, and `SearchSummary`, `search_lines`, `search_experimental`, `render_footer`, `render_markdown`, `LEVELS`, `STATE_WORDS`, `Finding`, `Kind` from `proofpath.report`, and `Reference`, `Locator` from `proofpath.document`. Add any that the module does not import yet.)

Append to `tests/test_sarif.py`:

```python
def test_every_kind_has_a_rule_description() -> None:
    assert set(RULE_DESCRIPTIONS) == set(Kind)


def test_a_search_run_states_its_search_in_the_run_properties() -> None:
    report = replace(report_with(), search=SearchSummary("tavily", "sentence", 2, 2, 2, 3, 1))
    props = to_sarif(report, artifact="draft.md")["runs"][0]["properties"]
    assert props["evidenceSearch"]["pagesRead"] == 1
    assert props["evidenceSearch"]["experimental"] is True
```

(`report_with` is already imported from `tests.test_report`; add `replace`, `RULE_DESCRIPTIONS`, `to_sarif` and `SearchSummary` to the imports if they are missing.)

Also add `Kind.EVIDENCE_FOUND` to the two `@pytest.mark.parametrize("kind", [Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH])` lists in `tests/test_report.py`. They pin which kinds must carry a passage, and `test_an_asserting_finding_with_a_passage_is_valid` must then pass for it too.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_report.py tests/test_sarif.py -v`
Expected: FAIL with `AttributeError: EVIDENCE_FOUND` and import errors.

- [ ] **Step 3: Implement `document.py`**

```python
# Who put a reference in the document: its author, or proofpath's evidence search
# (OPEN-ITEMS 17.1a). Carried on the reference so every finding about a found page
# can say so, whichever stage wrote it.
Origin = Literal["author", "search"]


@dataclass(frozen=True)
class Reference:
    """One bibliography entry, kept verbatim; resolve.py does the parsing."""

    number: int  # 1-based bibliography position, what "[12]" refers to
    raw: str  # the entry verbatim, whitespace collapsed
    locator: Locator
    origin: Origin = "author"
```

- [ ] **Step 4: Implement `report.py`**

- Append to `Kind`, after `PROVIDER_UNAVAILABLE`:

```python
    EVIDENCE_FOUND = "evidence-found"  # a page the search found supports the claim
    NO_EVIDENCE = "no-evidence-found"  # the search found nothing that decides the claim
```

- Add `Kind.NO_EVIDENCE: "warning",` to `LEVELS` beside the other warnings, and `Kind.EVIDENCE_FOUND: "note",` beside the notes.
- Add to `STATE_WORDS`:

```python
    Kind.EVIDENCE_FOUND: "SUPPORTED (found by proofpath)",
    Kind.NO_EVIDENCE: "NO EVIDENCE FOUND (searched)",
```

- Change `_ASSERTING` to `frozenset({Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH, Kind.EVIDENCE_FOUND})`, and extend its comment: "EVIDENCE_FOUND asserts support, and support needs its passage just as much (rule 1)."
- Below `UNVERIFIED_PREFIX`, add:

```python
# Evidence search (OPEN-ITEMS 17.1a). The provenance note every finding about a found
# page carries, so no renderer can show one as a source the author cited.
FOUND_BY_PROOFPATH = "FOUND BY PROOFPATH (not cited by the author)"
SEARCH_UNAVAILABLE = f"{UNVERIFIED_PREFIX} (search unavailable)"
# The §17.1 gate: printed on every run that searched until the eval beats the
# majority baseline. Updated by hand from docs/eval/*-averitec-search.md.
AVERITEC_SEARCH_SCORE = "not measured yet"
AVERITEC_BASELINE = "0.708"


def search_experimental() -> str:
    return (
        f"evidence search is experimental — AVeriTeC search-mode {AVERITEC_SEARCH_SCORE} "
        f"vs {AVERITEC_BASELINE} majority baseline"
    )
```

- After `Coverage`, add:

```python
@dataclass(frozen=True)
class SearchSummary:
    """What the evidence search did: the block beneath the coverage block (rule 6)."""

    by: str  # the searcher: "tavily" | "searxng"
    queries_by: str  # "sentence", or the judge that wrote them
    sentences: int  # sentences with a word in them
    eligible: int  # of those, check-worthy
    searched: int  # of those, actually searched
    pages_found: int
    pages_read: int  # full text or abstract
    notices: tuple[str, ...] = ()  # the LLM fallback, said once
```

- Add `search: SearchSummary | None = None` as the last field of `Report`, with the comment `# Set only on a run that searched (OPEN-ITEMS 17.1a).`
- Add `search: tuple[str, ...] = ()` as the last field of `Footer`, with the comment `# ``search_lines``: empty unless the run searched.`, and set `search=search_lines(report),` in `render_footer`.
- Add the functions:

```python
def search_lines(report: Report) -> tuple[str, ...]:
    """The evidence-search block, one line each, for every surface to print as is."""
    item = report.search
    if item is None:
        return ()
    return (
        search_experimental(),
        f"evidence search by {item.by}, queries by {item.queries_by}",
        f"claims searched {item.searched} of {item.eligible} check-worthy "
        f"({item.sentences} sentences)",
        f"claims not searched {item.sentences - item.searched}",
        f"pages found / read {item.pages_found} / {item.pages_read}",
        *item.notices,
    )


def _markdown_search(report: Report) -> list[str]:
    lines = search_lines(report)
    if not lines:
        return []
    return ["## Evidence search", "", *(f"- {line}" for line in lines), ""]
```

- In `render_markdown`, call `lines.extend(_markdown_search(report))` right after `lines.extend(_markdown_coverage(report))`.

- [ ] **Step 5: Implement `sarif.py`**

Add to `RULE_DESCRIPTIONS`:

```python
    Kind.EVIDENCE_FOUND: "a page proofpath found, not one the author cited, supports the claim",
    Kind.NO_EVIDENCE: "the evidence search found no page that supports or contradicts the claim",
```

In `_run_properties`, add after `"cancelled": report.cancelled,`:

```python
        "evidenceSearch": None
        if report.search is None
        else {
            "experimental": True,
            "by": report.search.by,
            "queriesBy": report.search.queries_by,
            "sentences": report.search.sentences,
            "eligible": report.search.eligible,
            "searched": report.search.searched,
            "pagesFound": report.search.pages_found,
            "pagesRead": report.search.pages_read,
            "notices": list(report.search.notices),
        },
```

- [ ] **Step 6: Regenerate the golden report, and check that only additions changed**

Look at how `tests/data/verify-report-golden.json` is indented (`head -5`), then match it in `indent=` below:

```bash
uv run python - <<'EOF'
import json
from proofpath.verify import verify
from tests.test_verify_golden import BODY, CLOSED, GHOSTLY, GOLDEN_PATH, REAL, WEB, built_engine, draft, stable

report = verify(draft(BODY, [REAL, GHOSTLY, WEB, CLOSED]), built_engine())
GOLDEN_PATH.write_text(json.dumps(stable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
EOF
git diff tests/data/verify-report-golden.json
```

Expected diff: only `+ "origin": "author"` inside each reference, and `+ "search": null` on the report. Anything else means the regeneration changed formatting. Revert and match the file's original formatting.

- [ ] **Step 7: Run the report, SARIF, golden and UI tests**

Run: `uv run pytest tests/test_report.py tests/test_sarif.py tests/test_verify_golden.py tests/test_ui.py tests/test_tui_rich.py -v`
Expected: PASS. A test that iterates `for kind in Kind` and expects a colour, a style or a description per kind must be satisfied by adding the entry for the two new kinds where that test says, never by skipping them.

- [ ] **Step 8: Stage**

```bash
git add src/proofpath/document.py src/proofpath/report.py src/proofpath/sarif.py tests/data/verify-report-golden.json tests/test_report.py tests/test_sarif.py
```

#### Amendment A (2026-09-28): the language state

Add a constant next to `SEARCH_UNAVAILABLE` in `report.py`:

```python
# A claim in another language that no judge translated: the models read English only,
# so it is not searched, and it is not called NEI either (OPEN-ITEMS 17.1a).
LANGUAGE_UNSUPPORTED = f"{UNVERIFIED_PREFIX} (language not supported)"
```

Test it in `tests/test_report.py`:
`assert LANGUAGE_UNSUPPORTED == "UNVERIFIED (language not supported)"`.

---

### Task 7: The Searching stage in `verify`

**Files:**
- Modify: `src/proofpath/verify.py`
- Test: `tests/test_verify_search.py` (new); `tests/test_verify.py` (only if the hint wording is pinned)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces:
  - `verify.SEARCHING = "Searching"`.
  - `Engine.searcher: Searcher | None = None`, `Engine.search_problem: str = ""`, `Engine.search: bool = True`.
  - `Engine.default(…, search: bool = True)`.
  - `Prepared.search: SearchSummary | None = None`.
  - `verify.NO_EVIDENCE_TITLE`, `verify.NO_EVIDENCE_NOTE`, `verify.EVIDENCE_FOUND_TITLE`, `verify.SEARCH_CANNOT_RUN`.

- [ ] **Step 1: Write the failing tests** in `tests/test_verify_search.py`:

```python
"""The Searching stage end to end, offline (OPEN-ITEMS 17.1a, spec 2026-09-28).

A stub searcher answers by substring, pages come from ``StubFetcher``, and both models
are the fakes ``tests/test_verify.py`` uses. Nothing here opens a socket.
"""

from __future__ import annotations

from proofpath import oa
from proofpath.config import Config, Permissions, SearchConfig
from proofpath.judge import Judge, JudgeUnavailable
from proofpath.polite import ProviderError
from proofpath.report import (
    FOUND_BY_PROOFPATH,
    SEARCH_UNAVAILABLE,
    Kind,
)
from proofpath.search import SearchHit
from proofpath.search.queries import LLM_LIMIT, sentence_query
from proofpath.secrets import CREDENTIALS_MISSING
from proofpath.verify import (
    NO_SOURCE_HINT,
    NO_SOURCE_IN_TEXT,
    SEARCHING,
    Engine,
    prepare,
    verify,
)
from tests.fakes import TableScorer, WordEmbedder
from tests.test_verify import (
    NEI_ROW,
    SUPPORTED_ROW,
    FakeJudgeClient,
    StubFetcher,
    engine,
    fetched,
    words,
)

CLAIM = "OpenAI shut down ChatGPT in March 2025."
PAGE_A = "https://news.test/openai"
PAGE_B = "https://wiki.test/ChatGPT"
BACKING = "OpenAI shut down ChatGPT in March 2025 after a vote."
FILLER = words(oa.FULLTEXT_MIN_WORDS + 10)


class StubSearcher:
    name = "stub"

    def __init__(
        self, answers: dict[str, list[str]] | None = None, *, error: ProviderError | None = None
    ) -> None:
        self.answers = answers or {}
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        for key, urls in self.answers.items():
            if key in query:
                return [SearchHit(url, "", rank) for rank, url in enumerate(urls, start=1)]
        return []


def searching(
    searcher: StubSearcher | None,
    *,
    pages: dict[str, str] | None = None,
    table: dict[str, tuple[float, float, float]] | None = None,
    config: Config | None = None,
    network_allowed: bool = True,
) -> Engine:
    fetcher = StubFetcher(
        {url: fetched(url, text) for url, text in (pages or {}).items()},
        network_allowed=network_allowed,
        network_note="" if network_allowed else "network not permitted",
    )
    built = engine(
        fetcher=fetcher,
        config=config,
        embedder=WordEmbedder(),
        scorer=TableScorer(table or {BACKING: SUPPORTED_ROW}),
    )
    built.searcher = searcher
    return built


def test_a_bare_claim_is_searched_and_the_backing_page_is_reported_as_found() -> None:
    searcher = StubSearcher({"ChatGPT": [PAGE_A]})
    report = verify(CLAIM, searching(searcher, pages={PAGE_A: f"{BACKING} {FILLER}"}))

    assert searcher.queries == [sentence_query(CLAIM)]
    assert report.counts() == {Kind.EVIDENCE_FOUND: 1}
    found = report.findings[0]
    assert found.verdict is not None and found.verdict.passage is not None
    assert found.verdict.passage.text == BACKING
    assert found.detail[-1] == FOUND_BY_PROOFPATH
    assert report.sources[0].reference.origin == "search"
    assert report.sources[0].reference.raw == PAGE_A
    assert SEARCHING in [stage.name for stage in report.stages]
    assert report.search is not None
    assert (report.search.searched, report.search.pages_found, report.search.pages_read) == (
        1,
        1,
        1,
    )


def test_a_search_that_finds_nothing_is_no_evidence_never_refuted() -> None:
    report = verify(CLAIM, searching(StubSearcher()))

    assert report.counts() == {Kind.NO_EVIDENCE: 1}
    assert report.findings[0].claim is not None
    assert report.findings[0].claim.text == CLAIM


def test_an_unreadable_found_page_keeps_its_state_and_says_who_found_it() -> None:
    report = verify(CLAIM, searching(StubSearcher({"ChatGPT": [PAGE_A]})))

    assert report.counts() == {Kind.UNVERIFIED: 1, Kind.NO_EVIDENCE: 1}
    unread = next(f for f in report.findings if f.kind is Kind.UNVERIFIED)
    assert unread.detail[-1] == FOUND_BY_PROOFPATH


def test_a_page_that_says_nothing_either_way_is_nei_and_no_evidence() -> None:
    report = verify(
        CLAIM,
        searching(
            StubSearcher({"ChatGPT": [PAGE_A]}),
            pages={PAGE_A: f"{BACKING} {FILLER}"},
            table={BACKING: NEI_ROW},
        ),
    )
    assert Kind.NOT_SUPPORTED not in report.counts()
    assert report.counts()[Kind.NO_EVIDENCE] == 1


def test_a_provider_that_fails_is_reported_and_the_claim_is_not_called_empty() -> None:
    searcher = StubSearcher(error=ProviderError("HTTP 503", 503, None))
    report = verify(CLAIM, searching(searcher))

    assert report.counts() == {Kind.UNVERIFIED: 1}
    assert report.findings[0].state == SEARCH_UNAVAILABLE
    assert report.search is not None and report.search.searched == 0


def test_without_a_searcher_the_old_parse_error_stands_with_the_setup_hint() -> None:
    ready = prepare(CLAIM, searching(None))
    errors = [f for f in ready.findings if f.kind is Kind.PARSE_ERROR]
    assert [f.title for f in errors] == [NO_SOURCE_IN_TEXT]
    assert errors[0].detail == (NO_SOURCE_HINT,)
    assert "search.provider" in NO_SOURCE_HINT


def test_no_search_and_a_denied_network_search_nothing() -> None:
    off = StubSearcher({"ChatGPT": [PAGE_A]})
    built = searching(off)
    built.search = False
    prepare(CLAIM, built)
    denied = StubSearcher({"ChatGPT": [PAGE_A]})
    prepare(CLAIM, searching(denied, network_allowed=False))
    assert off.queries == [] and denied.queries == []


def test_a_configured_provider_that_cannot_run_says_why() -> None:
    built = searching(None)
    built.search_problem = "set TAVILY_API_KEY in .env"
    ready = prepare(CLAIM, built)
    [item] = ready.findings
    assert item.kind is Kind.UNVERIFIED
    assert item.state == CREDENTIALS_MISSING
    assert item.detail == ("set TAVILY_API_KEY in .env",)


def test_a_text_that_cites_something_is_never_searched() -> None:
    searcher = StubSearcher({"ChatGPT": [PAGE_A]})
    prepare(f"{CLAIM} See {PAGE_B}", searching(searcher))
    assert searcher.queries == []


def test_the_cap_limits_the_search_and_the_rest_is_counted() -> None:
    body = " ".join(f"Company{i} sold {i} million units in 2020." for i in range(7))
    searcher = StubSearcher()
    report = verify(body, searching(searcher, config=Config(search=SearchConfig(max_claims=5))))
    assert len(searcher.queries) == 5
    assert report.search is not None
    assert (report.search.eligible, report.search.searched) == (7, 5)


def test_a_rate_limited_judge_falls_back_to_the_sentence_and_the_report_says_so() -> None:
    searcher = StubSearcher()
    built = searching(searcher)
    built.judge = Judge(FakeJudgeClient([JudgeUnavailable("HTTP 429 from https://api.test")]))  # type: ignore[arg-type]
    report = verify(CLAIM, built)
    assert searcher.queries == [sentence_query(CLAIM)]
    assert report.search is not None
    assert report.search.notices[0].startswith(LLM_LIMIT)


def test_ask_or_deny_turns_the_search_off_in_the_default_engine() -> None:
    config = Config(
        permissions=Permissions(web_search="ask"),
        search=SearchConfig(provider="searxng", base_url="http://localhost:8888"),
    )
    with Engine.default(config, interactive=True, no_cache=True) as built:
        assert built.searcher is not None
        assert built.search is False
    allowed = Config(search=SearchConfig(provider="searxng", base_url="http://localhost:8888"))
    with Engine.default(allowed, interactive=False, no_cache=True) as built:
        assert built.search is True
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_verify_search.py -v`
Expected: FAIL with `ImportError: cannot import name 'SEARCHING'`.

- [ ] **Step 3: Implement `Engine` fields and `Engine.default`**

Imports to add in `verify.py`:

```python
from proofpath.report import FOUND_BY_PROOFPATH, SEARCH_UNAVAILABLE, SearchSummary
from proofpath.search import SearchHit, Searcher, canonical, keep_hits
from proofpath.search.providers import build_searcher
from proofpath.search.queries import plan_queries
from proofpath.secrets import CREDENTIALS_MISSING
from proofpath.settings_hints import SEARCH_SETTING
```

(Merge them into the existing `from proofpath.report import (…)` and `settings_hints` import lines.)

Stage constant, beside the others: `SEARCHING = "Searching"`.

`Engine` fields, after `escalate`:

```python
    # Evidence search for a text that cites nothing (OPEN-ITEMS 17.1a). ``None`` when
    # no provider is configured; ``search_problem`` then says why a configured one
    # could not be built (a missing key), and ``search`` is this run's own switch
    # (``--no-search``, or ``permissions.web_search`` other than ``allow``).
    searcher: Searcher | None = None
    search_problem: str = ""
    search: bool = True
```

`Engine.default`: add the parameter `search: bool = True`. After `client = PoliteClient(contact_email=email)` and its closer, add:

```python
            setup = build_searcher(config.search, client)
```

Then pass to `cls(...)`:

```python
searcher = (setup.searcher,)
search_problem = (setup.problem,)
# The search never prompts, so ``ask`` is ``deny`` here (rule 4).
search = (search and config.permissions.web_search == "allow",)
```

- [ ] **Step 4: Rewrite the no-source message**

```python
# What the parsing stage says of a text that neither links nor cites anything: a bare
# claim. Without a search provider there is nothing to check it against, and ending
# with every stage at zero would read as a clean run (product rule 6). The hint names
# the one setting that makes proofpath look for evidence itself (OPEN-ITEMS 17.1a).
NO_SOURCE_IN_TEXT = "the text links or cites no source; nothing to verify against"
NO_SOURCE_HINT = (
    "proofpath checks a claim against the sources it cites; to have it search the web "
    f"for evidence instead: proofpath config set {SEARCH_SETTING}, with TAVILY_API_KEY in .env"
)
# A provider is configured but cannot run (no key, no address). Reported as the
# credential state, naming what to set -- never as a text that simply cites nothing.
SEARCH_CANNOT_RUN = "the text cites no source, and the evidence search could not run"
EVIDENCE_FOUND_TITLE = "a page proofpath found supports the claim"
NO_EVIDENCE_TITLE = "no page proofpath found supports or contradicts the claim"
NO_EVIDENCE_NOTE = "absence of evidence is not evidence against the claim"
```

- [ ] **Step 5: Decide in the parsing stage whether the run will search**

Replace the `if document.kind in ("post", "page") and not document.references:` … `elif …:` block in `prepare` with:

```python
    sourceless = not document.references and (
        document.kind in ("post", "page")
        or (bool(document.paragraphs) and not claims_mod.find_markers(document))
    )
    will_search = (
        sourceless
        and bool(document.paragraphs)
        and allowed
        and engine.search
        and engine.searcher is not None
    )
    if sourceless and not will_search:
        if engine.search and engine.search_problem and allowed:
            add(
                _finding(
                    Kind.UNVERIFIED,
                    Locator(line=1),
                    SEARCH_CANNOT_RUN,
                    state=CREDENTIALS_MISSING,
                    detail=(engine.search_problem,),
                )
            )
        elif document.kind in ("post", "page"):
            # (keep the existing comment)
            add(
                _finding(
                    Kind.PARSE_ERROR,
                    Locator(line=1),
                    f"{document.kind} carries no links; nothing to verify against",
                )
            )
        else:
            # (keep the existing comment)
            pasted = not _is_file(target)
            add(
                _finding(
                    Kind.PARSE_ERROR,
                    Locator(line=1),
                    NO_SOURCE_IN_TEXT
                    if pasted
                    else f"{document.kind} cites nothing; nothing to verify against",
                    detail=(NO_SOURCE_HINT,) if pasted else (),
                )
            )
```

A missing SearXNG address (`SEARXNG_NEEDS_URL`) is not a credential, but it is fixed the same way, with one setting. It shares `CREDENTIALS_MISSING`, so the report needs no new state (spec §4).

- [ ] **Step 6: Add the Searching stage and `_search`**

In `prepare`, right after `if not allowed and denied_note: emit(Note(denied_note))` and before `# --- 3. resolving`:

```python
    # --- 2b. searching (OPEN-ITEMS 17.1a) ------------------------------------
    # Only for a document that cites nothing. Its claims are the check-worthy
    # sentences, and its references are the pages found for them, each marked as
    # found (``origin="search"``). From here on every stage runs exactly as it does
    # for a cited source.
    search: SearchSummary | None = None
    if will_search:
        found = _search(document, engine, add=add, emit=emit, check=check, opened=opened, closed=closed)
        document, claims, search = found.document, found.claims, found.summary
```

Before `return Prepared(...)`, count what was read, and pass it:

```python
    if search is not None:
        search = replace(
            search,
            pages_read=sum(1 for status in sources.values() if status.text_kind != "none"),
        )
```

Add `search=search,` to `Prepared(...)`, and the field `search: SearchSummary | None = None` to `Prepared`, after `started`.

Module-level helpers, placed after `_nothing_read` at the end of the file:

```python
@dataclass(frozen=True)
class _Searched:
    document: Document
    claims: Claims
    summary: SearchSummary


def _search(
    document: Document,
    engine: Engine,
    *,
    add: Callable[[Finding], None],
    emit: Listener,
    check: Callable[[], None],
    opened: Callable[[str, str], float],
    closed: Callable[[str, str, str, float], None],
) -> _Searched:
    """Search for each check-worthy sentence and turn the kept hits into references.

    A provider that fails is reported once and the search stops there: the claims it
    did not reach count as not searched rather than as searched and empty, which
    would read as "nothing out there" (product rule 2).
    """
    searcher = engine.searcher
    assert searcher is not None  # guarded by ``will_search``
    settings = engine.config.search
    began = opened(SEARCHING, searcher.name)
    chosen = claims_mod.checkworthy(document, limit=settings.max_claims)
    plan = plan_queries(
        [claim.text for claim in chosen.claims], engine.judge if engine.escalate else None
    )
    if plan.notice:
        emit(Note(plan.notice))
    own = _own_host(document)
    numbers: dict[str, int] = {}
    references: list[Reference] = []
    searched: list[Claim] = []
    for index, (claim, queries) in enumerate(
        zip(chosen.claims, plan.queries, strict=True), start=1
    ):
        hits: list[SearchHit] = []
        try:
            for query in queries:
                hits.extend(searcher.search(query, settings.results_per_claim))
        except ProviderError as exc:
            add(
                _finding(
                    Kind.UNVERIFIED,
                    claim.locator,
                    "the evidence search did not answer",
                    state=SEARCH_UNAVAILABLE,
                    detail=(f"{searcher.name}: {exc}",),
                )
            )
            break
        cited: list[int] = []
        for hit in keep_hits(hits, own_host=own, limit=settings.results_per_claim):
            key = canonical(hit.url)
            if key not in numbers:
                numbers[key] = len(references) + 1
                references.append(Reference(numbers[key], hit.url, claim.locator, origin="search"))
            cited.append(numbers[key])
        searched.append(replace(claim, cited_refs=tuple(cited)))
        emit(
            Progress(
                name=SEARCHING, done=index, total=len(chosen.claims), detail=claim.locator.label()
            )
        )
        check()
    closed(
        SEARCHING,
        searcher.name,
        f"{len(searched)} of {chosen.eligible} claims, {len(references)} pages",
        began,
    )
    summary = SearchSummary(
        by=searcher.name,
        queries_by=plan.by,
        sentences=chosen.sentences,
        eligible=chosen.eligible,
        searched=len(searched),
        pages_found=len(references),
        pages_read=0,  # filled in once the fetching stage has run
        notices=(plan.notice,) if plan.notice else (),
    )
    return _Searched(
        document=replace(document, references=tuple(references)),
        claims=Claims(claims=tuple(searched), markers=(), unsupported=(), unresolved=()),
        summary=summary,
    )


def _own_host(document: Document) -> str | None:
    """The host of a page or post read as the document: never found to back itself."""
    match = _BARE_URL.fullmatch(document.name.strip())
    return urlsplit(match.group(0)).hostname if match else None
```

Add `from urllib.parse import urlsplit` to the imports. `markers=()` is deliberate: the author made no citation, and the report's marker count must not say they did.

- [ ] **Step 7: Put provenance on every finding about a found page, and give a supported found page its own finding**

```python
def _provenance(reference: Reference | None, detail: tuple[str, ...]) -> tuple[str, ...]:
    """A found page says so on every finding about it (OPEN-ITEMS 17.1a). Last, not
    first: a numeric mismatch's reason is read back out of ``detail[0]``."""
    if reference is not None and reference.origin == "search":
        return (*detail, FOUND_BY_PROOFPATH)
    return detail
```

In `_finding`, set `detail=_provenance(reference, detail),`.

In `_verdict_finding`, replace everything from the opening `if verdict.label is Label.SUPPORTED:` down to just before `return Finding(` with:

```python
    found = status.reference.origin == "search"
    source = "a page proofpath found" if found else "the cited source"
    kind: Kind
    detail: tuple[str, ...] = ()
    if verdict.label is Label.SUPPORTED:
        if not found:
            return None
        # The answer the user asked for: shown, with its passage (rule 1), where a
        # supported *cited* claim stays silent.
        kind = Kind.EVIDENCE_FOUND
        title = EVIDENCE_FOUND_TITLE
    elif verdict.label is Label.REFUTED and verdict.reason.startswith(NUMERIC_REASON):
        # (keep the existing comment)
        kind = Kind.NUMERIC_MISMATCH
        title = f"claim contradicts {source}"
        detail = (verdict.reason,)
    elif verdict.label is Label.REFUTED:
        kind = Kind.NOT_SUPPORTED
        title = f"claim is not supported by {source}"
    else:
        kind = Kind.NEI
        title = (
            "a page proofpath found neither supports nor contradicts the claim"
            if found
            else "source neither supports nor contradicts the claim"
        )
```

In the `return Finding(...)` that follows, change `detail=detail,` to `detail=_provenance(status.reference, detail),`. The author-side titles stay byte for byte as they are today ("claim contradicts the cited source", "claim is not supported by the cited source", "source neither supports nor contradicts the claim").

- [ ] **Step 8: Report NO EVIDENCE FOUND per searched claim in `decide_all`**

After the `for group, members in grouped.items(): add(...)` loop:

```python
    if prepared.search is not None and not cancelled:
        # One line per searched claim that no found page decided. A cancelled run is
        # left alone: an undecided claim there is unfinished, not unbacked.
        backed = {
            (result.claim.paragraph, result.claim.sentence)
            for result in results
            if result.verdict.label is not Label.NEI
        }
        for claim in prepared.claims.claims:
            if (claim.paragraph, claim.sentence) not in backed:
                add(_no_evidence(claim))
```

Also pass `search=prepared.search,` to `Report(...)`, and add:

```python
def _no_evidence(claim: Claim) -> Finding:
    return Finding(
        kind=Kind.NO_EVIDENCE,
        level=LEVELS[Kind.NO_EVIDENCE],
        locator=claim.locator,
        title=NO_EVIDENCE_TITLE,
        state=STATE_WORDS[Kind.NO_EVIDENCE],
        reference=None,
        claim=claim,
        verdict=None,
        source_id=None,
        fetch_step=None,
        tier=None,
        detail=(NO_EVIDENCE_NOTE,),
    )
```

- [ ] **Step 9: Run the search tests, then the whole verify suite**

Run: `uv run pytest tests/test_verify_search.py -v`
Expected: PASS.

Run: `uv run pytest tests/test_verify.py tests/test_verify_golden.py tests/test_verify_routing.py -v`
Expected: PASS. The old bare-claim tests still pass, because `engine()` builds no searcher. If one pins the old `NO_SOURCE_HINT` wording by value rather than by constant, update that expectation to the constant.

- [ ] **Step 10: Stage**

```bash
git add src/proofpath/verify.py tests/test_verify_search.py tests/test_verify.py
```

#### Amendment A (2026-09-28): short text and claims that are not in English

Task 4 now returns every sentence when none looks check-worthy, so "OpenAI battı" becomes one claim. Task 5 now gives `QueryPlan.hypotheses`: the English text to check for each claim, or `None`. Task 6 adds `report.LANGUAGE_UNSUPPORTED`. This amendment wires them in.

**1. `document.Claim`** gets a field after `marker`:

```python
    # The English text the models check, when it is not ``text`` itself: a claim in
    # another language that the judge translated (OPEN-ITEMS 17.1a). ``text`` stays
    # the reader's own sentence, and that is what the report shows.
    hypothesis: str | None = None
```

**2. In `_search`**, iterate over the hypotheses too, and skip an untranslated claim:

```python
for index, (claim, queries, hypothesis) in enumerate(
    zip(chosen.claims, plan.queries, plan.hypotheses, strict=True), start=1
):
    if hypothesis is None:
        add(
            _finding(
                Kind.UNVERIFIED,
                claim.locator,
                LANGUAGE_TITLE,
                state=LANGUAGE_UNSUPPORTED,
                detail=(LANGUAGE_HINT,),
            )
        )
        emit(
            Progress(
                name=SEARCHING, done=index, total=len(chosen.claims), detail=claim.locator.label()
            )
        )
        check()
        continue
    # … search as before …
    searched.append(
        replace(
            claim,
            cited_refs=tuple(cited),
            hypothesis=None if hypothesis == claim.text else hypothesis,
        )
    )
```

Constants, next to the other titles:

```python
LANGUAGE_TITLE = "the claim is not in English and was not translated"
LANGUAGE_HINT = "the models read English only; --judge translates a claim before it is searched"
```

Import `LANGUAGE_UNSUPPORTED` from `proofpath.report`. The untranslated claim is not appended to `searched`, so it gets no NO EVIDENCE FINDING, and the summary counts it as not searched.

**3. The models check the hypothesis.**
- In `_decide_claim`, use `checked = claim.hypothesis or claim.text` for both `cache_mod.claim_hash(...)` and `pipeline.decide_indexed(...)`.
- In `_judging`, the `JudgeItem(claim=...)` becomes `result.claim.hypothesis or result.claim.text`.
- Leave the other `claim_hash(result.claim.text)` / `claim_hash(item.claim.text)` pair in `_judging` / `_with_opinion` as it is: it only matches an opinion back to its result, and both sides use the same text.

**4. A translated claim says what was checked.** In `_verdict_finding`, right before the `return Finding(...)`:

```python
    if claim.hypothesis:
        detail = (*detail, f"{CHECKED_AS}{claim.hypothesis}")
```

with `CHECKED_AS = "checked as: "`. It is appended after the numeric reason, which must stay `detail[0]`, and before the provenance note, which `_provenance` appends last.

**5. Tests**, appended to `tests/test_verify_search.py`:

```python
TURKISH = "OpenAI battı."
ENGLISH = "OpenAI went bankrupt."
BANKRUPT = "OpenAI went bankrupt in 2026, the court said."


def test_a_one_line_post_is_searched_even_when_nothing_in_it_looks_checkworthy() -> None:
    searcher = StubSearcher()
    report = verify("OpenAI shut down", searching(searcher))
    assert searcher.queries == ["OpenAI shut down"]
    assert report.counts() == {Kind.NO_EVIDENCE: 1}


def test_without_a_judge_a_turkish_claim_is_reported_not_searched() -> None:
    searcher = StubSearcher({"OpenAI": [PAGE_A]})
    report = verify(TURKISH, searching(searcher))
    assert searcher.queries == []
    [item] = report.findings
    assert item.kind is Kind.UNVERIFIED
    assert item.state == LANGUAGE_UNSUPPORTED
    assert report.search is not None and report.search.searched == 0


def test_the_judge_translates_a_turkish_claim_and_the_report_shows_both() -> None:
    answer = json.dumps(
        {"items": [{"id": 0, "english": ENGLISH, "queries": ["OpenAI bankruptcy"]}]}
    )
    searcher = StubSearcher({"bankruptcy": [PAGE_A]})
    built = searching(
        searcher,
        pages={PAGE_A: f"{BANKRUPT} {FILLER}"},
        table={BANKRUPT: SUPPORTED_ROW},
    )
    built.judge = Judge(FakeJudgeClient([answer]))  # type: ignore[arg-type]
    report = verify(TURKISH, built)
    assert searcher.queries == ["OpenAI bankruptcy"]
    found = next(f for f in report.findings if f.kind is Kind.EVIDENCE_FOUND)
    assert found.claim is not None and found.claim.text == TURKISH
    assert f"checked as: {ENGLISH}" in found.detail
    assert found.detail[-1] == FOUND_BY_PROOFPATH
```

(Import `json`, `LANGUAGE_UNSUPPORTED`. The judging stage runs after verifying with the same `FakeJudgeClient`. Its unscripted answers agree, so they change nothing, but check that `report.findings` holds no judge-unavailable line that breaks the `[item] =` destructuring in the no-judge test, where there is no judge at all.)

---

### Task 8: The surfaces (`--no-search`, footer lines, TUI footer and `/config`)

**Files:**
- Modify: `src/proofpath/cli.py` (`check`)
- Modify: `src/proofpath/ui.py` (`footer`)
- Modify: `src/proofpath/tui/widgets/footer.py` (`_hints`)
- Modify: `src/proofpath/tui/widgets/config_panel.py` (`sections`, `_draw_summary`)
- Test: `tests/test_check_cli.py`, `tests/test_ui.py`, and the TUI config panel test that pins `sections()`

**Interfaces:**
- Consumes: `Engine.default(search=…)` (Task 7), `Footer.search` (Task 6), `SearchConfig` and `SEARCH_PROVIDERS` (Task 1).

- [ ] **Step 1: Write the failing tests**

In `tests/test_check_cli.py` (it has `runner = CliRunner()` and imports `app`):

```python
def test_no_search_is_passed_to_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install(monkeypatch)
    path = write(tmp_path, "Transformers improved translation quality [1].")
    runner.invoke(app, ["check", "--no-search", str(path)])
    assert seen["search"] is False


def test_search_is_on_unless_told_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = install(monkeypatch)
    runner.invoke(
        app, ["check", str(write(tmp_path, "Transformers improved translation quality [1]."))]
    )
    assert seen["search"] is True
```


In `tests/test_ui.py`, render a `Footer` with `search=("evidence search is experimental — …", "claims not searched 2")` through `ui.footer` into the module's captured console, and assert both lines appear, each after the key `search`. Follow the file's existing footer test for the console setup.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_check_cli.py -k search tests/test_ui.py -k footer -v`
Expected: FAIL with `KeyError: 'search'` and the missing lines.

- [ ] **Step 3: Implement**

`cli.py` `check`: add the option after `no_cache`:

```python
no_search: Annotated[
    bool,
    typer.Option(
        "--no-search",
        help="Do not search the web for evidence when the text cites no source.",
    ),
] = (False,)
```

and pass `search=not no_search,` to `verify_mod.Engine.default(...)`.

`ui.py` `footer`: after `skipped(ui, item.browser_skipped)`:

```python
    # The evidence-search block (OPEN-ITEMS 17.1a): never suppressed, because a run
    # that searched must not read like a run that checked cited sources (rule 6).
    for line in item.search:
        kv(ui, "search", line)
```

`tui/widgets/footer.py` `_hints`: before `if item.note:`, add:

```python
    # The docked footer has one hints row: the experimental banner and the searched
    # line go there, and the full block is in the run's report.
    hints.extend(item.search[:3])
```

`tui/widgets/config_panel.py`:

- In `sections()`, build `search = SearchConfig()` beside the others.
- Add a `permissions.web_search` choice row after `permissions.network`, with the meaning `"search the web when a text cites nothing; ask = deny (never prompts)"`.
- Append a new section after `judge`, with rows following the `judge` section's text-row shape:
  - `Row("search.provider", "choice", SEARCH_PROVIDERS, "off until set; tavily needs TAVILY_API_KEY in .env", _default(search.provider))`
  - `Row("search.base_url", "text", (), "your SearXNG instance", _default(search.base_url))`
  - `Row("search.api_key_env", "text", (), "the variable holding the key", _default(search.api_key_env))`
- In `_draw_summary`, add `f"search={config.search.provider}",` after `judge=`.

Import `SearchConfig` and `SEARCH_PROVIDERS` from `proofpath.config`.

- [ ] **Step 4: Run the surface tests**

Run: `uv run pytest tests/test_check_cli.py tests/test_ui.py tests/test_tui_app.py tests/test_tui_commands.py -v`
Expected: PASS. A config-panel test that pins the row list or the summary line gets the new rows and `search=off` added to its expectation.

- [ ] **Step 5: Stage**

```bash
git add src/proofpath/cli.py src/proofpath/ui.py src/proofpath/tui/widgets/footer.py src/proofpath/tui/widgets/config_panel.py tests/test_check_cli.py tests/test_ui.py tests/test_tui_*.py
```

---

### Task 9: The AVeriTeC search-mode eval (the §17.1 gate)

**Files:**
- Modify: `src/proofpath/eval/averitec.py` (`Claim.fact_check_url`)
- Modify: `scripts/eval_averitec.py` (`--search`)
- Test: `tests/test_averitec.py`, `tests/test_eval_averitec.py`

**Interfaces:**
- Consumes: `build_searcher`, `keep_hits`, `sentence_query` (Tasks 3 and 5).
- Produces:
  - `averitec.Claim.fact_check_url: str = ""`.
  - `eval_averitec.search_urls(searcher: Searcher, claim: averitec.Claim, limit: int) -> tuple[str, ...]`.

- [ ] **Step 1: Confirm the dataset key**

Run: `uv run python -c "import json; from proofpath.paths import cache_dir; p = cache_dir() / 'datasets'; f = next(p.glob('*dev*.json')); print(sorted(json.loads(f.read_text(encoding='utf-8'))[0]))"`
Expected: a key list that includes `fact_checking_article`. If the key is named differently, use that name in Step 3. If the dataset is not downloaded, read the key list from the AVeriTeC README instead. Do not download it in a test.

- [ ] **Step 2: Write the failing tests**

`tests/test_averitec.py`: extend the existing inline dev-row fixture with `"fact_checking_article": "https://factcheck.test/claim-1"`, and assert `load(...)[0].fact_check_url == "https://factcheck.test/claim-1"`. Also assert that a row without the key loads as `""`.

`tests/test_eval_averitec.py`:

```python
def test_search_urls_never_hand_back_the_fact_check_itself() -> None:
    class Stub:
        name = "stub"

        def search(self, query: str, max_results: int) -> list[SearchHit]:
            return [
                SearchHit("https://factcheck.test/claim-1", "", 1),
                SearchHit("https://factcheck.test/other", "", 2),
                SearchHit("https://news.test/a", "", 3),
            ]

    claim = averitec.Claim(
        id=0, text="A claim.", label="Refuted", source_urls=(),
        fact_check_url="https://factcheck.test/claim-1",
    )  # fmt: skip
    assert eval_averitec.search_urls(Stub(), claim, 3) == ("https://news.test/a",)
```

(Import the script the way `test_eval_averitec.py` already does.)

- [ ] **Step 3: Implement**

`averitec.Claim`: add `fact_check_url: str = ""` after `non_urls`, with the comment `# The article that settled the claim: the answer key, never a search result.` In `load`, pass `fact_check_url=str(row.get("fact_checking_article") or ""),`.

`scripts/eval_averitec.py`:

```python
from urllib.parse import urlsplit  # already imported

from proofpath.search import Searcher, keep_hits
from proofpath.search.providers import build_searcher
from proofpath.search.queries import sentence_query

SEARCH_RESULTS_NAME = "averitec_search_results.json"


def search_urls(searcher: Searcher, claim: averitec.Claim, limit: int) -> tuple[str, ...]:
    """What the evidence search would read for this claim, minus the answer key.

    The whole fact-checking site is excluded, not just the one article: a site that
    rated the claim quotes its own rating on every related page (the leakage the
    §17.1 gate must not measure).
    """
    checker = (urlsplit(claim.fact_check_url).hostname or "").removeprefix("www.")
    hits = [
        hit
        for hit in searcher.search(sentence_query(claim.text), limit + 3)
        if not checker or (urlsplit(hit.url).hostname or "").removeprefix("www.") != checker
    ]
    return tuple(hit.url for hit in keep_hits(hits, own_host=None, limit=limit))
```

In `main`:
- Add `parser.add_argument("--search", action="store_true", help="read what the evidence search finds instead of the gold URLs")`.
- Make `_results_path()` take `search: bool` and return `SEARCH_RESULTS_NAME` when it is set.
- After the engine is built:

```python
searcher = None
if args.search:
    setup = build_searcher(config.search, engine.client)
    if setup.searcher is None:
        print(
            f"error     evidence search is not set up: {setup.problem or 'search.provider is off'}",
            file=sys.stderr,
        )
        engine.close()
        return 2
    searcher = setup.searcher
```

- In the loop, when `searcher is not None`, decide `replace(claim, source_urls=search_urls(searcher, claim, config.search.results_per_claim), non_urls=0)`. Import `replace` from `dataclasses`.
- The default `--out` becomes `…-averitec-search.md` under `--search`.
- Update the module docstring's usage line.

- [ ] **Step 4: Run the eval tests**

Run: `uv run pytest tests/test_averitec.py tests/test_eval_averitec.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/proofpath/eval/averitec.py scripts/eval_averitec.py tests/test_averitec.py tests/test_eval_averitec.py
```

- [ ] **Step 6 (manual, needs the user's key; not a subagent step):** run `uv run python scripts/eval_averitec.py --search --limit 100 --no-browser --resume`, and copy the 3-way score into `report.AVERITEC_SEARCH_SCORE` (for example `"0.312"`).

---

### Task 10: Docs and the whole-suite check

**Files:**
- Modify: `README.md` (a new section, and the X row)
- Modify: `src/proofpath/providers/social.py` (`X_HINT`)
- Modify: `docs/superpowers/OPEN-ITEMS.md` (§17.1a status, §17.2 closed)

- [ ] **Step 1: README.** After `## Posts and the links inside them (v0.4)`, add `## Claims with no source (experimental)`. Content:
  - When it runs: a text with no link, marker or bibliography, and a provider set.
  - The setup lines:
    ```bash
    proofpath config set search.provider tavily
    echo 'TAVILY_API_KEY=tvly-…' >> .env
    ```
    or `search.provider searxng` plus `search.base_url`.
  - The two states, `SUPPORTED (found by proofpath)` and `NO EVIDENCE FOUND (searched)`, and the `FOUND BY PROOFPATH (not cited by the author)` note.
  - The cap (5 × 3), `--no-search`, the judge-written queries and their fallback notice.
  - The experimental banner, with the AVeriTeC number.

  Change the X row to: "cannot be read at all. Paste the text instead: a link inside it is verified normally, and with evidence search set up a text with no link is searched (experimental)."

- [ ] **Step 1b: "What it reads" (added 2026-09-28).** Main gained a plain "What it reads" overview after this branch was cut: README `## What it reads` and the landing page's `site/src/data/reads.ts`. First rebase this branch on `main` (`git rebase main` in the worktree; the controller does this, not a subagent). Then add one row to each.
  - README table, after the Sites-that-block-robots row: `| 🔑 | Claims with no source, searched on the web (experimental) | a free Tavily key, or your own SearXNG |`
  - `reads.ts`, in the "Needs one thing from you" column, after the LLM item: `{ icon: 'search', name: 'Claims with no source', note: 'searched on the web: a free Tavily key, or your own SearXNG' }`. The `search` icon already exists in `SketchIcon.astro`.

- [ ] **Step 2: `social.X_HINT`** → `"X cannot be read; paste the post text — its links are verified, and with search.provider set a text with no link is searched"`. Update the test that pins the old string by value, if one does.

- [ ] **Step 3: OPEN-ITEMS.** Mark 17.1a with `**Implemented <date>**` and the plan path. Close 17.2 with a strike-through and "fixed by the README X row and `X_HINT`, 2026-09-28".

- [ ] **Step 4: The whole suite and the linters**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
Expected: ruff clean, all tests pass. Paste the final summary line in the task report.

- [ ] **Step 5: Stage**

```bash
git add README.md src/proofpath/providers/social.py docs/superpowers/OPEN-ITEMS.md tests/
```

- [ ] **Step 6 (manual):** try it by hand:
  - `proofpath check -` with a real Tavily key, fed a real bare claim (for example "ChatGPT was released by OpenAI in November 2022.").
  - The same text in the TUI.
  - `--no-search`, which should give the old parse error.
  - The key removed, which should give `UNVERIFIED (credentials missing)` naming `TAVILY_API_KEY`.

---

### Task 11: Local fallback judge (spec §11)

**Files:**
- Modify: `src/proofpath/config.py` (`JudgeConfig.fallback: str = "auto"`, `/config` coercion if needed)
- Modify: `src/proofpath/judge.py` (model picker, fallback client, notice)
- Modify: `src/proofpath/verify.py` (emit the switch as a `Note`, carry it into the report footer)
- Modify: `src/proofpath/report.py` (footer line for the notice)
- Modify: `src/proofpath/cli.py`, `src/proofpath/tui/app.py` (build the fallback-capable judge; pass through only)
- Create: `src/proofpath/tui/widgets/judge_notice.py` (right-aligned one-line notice above the prompt)
- Modify: `src/proofpath/settings_hints.py` (the `judge.fallback` hint), the TUI `/config` panel entry
- Test: `tests/test_judge_fallback.py`, plus the TUI test module that covers `#bottom`

**Interfaces:**
- Produces:
  - `judge.pick_local_model(names: Sequence[str]) -> str | None`. This is the pure selection rule of spec §11.
  - `judge.installed_models(base_url: str, *, timeout: float) -> list[str]`: `GET {root}/api/tags`, where `root` is the Ollama base URL without `/v1`. Returns `[]` on any error; the error never raises.
  - `judge.FallbackClient`. It has the same public surface `Judge` uses from `JudgeClient`: `complete`, `provider`, `model`, `cost`, `close`, and the context manager.
    - It wraps a primary `JudgeClient` plus a factory for the local client, which is built lazily on the first failure.
    - After a switch, `provider` and `model` report the local model.
    - `switched: str | None` holds the notice text once a switch has happened.
    - `on_switch: Callable[[str], None] | None` is called once, at the switch.
  - `JUDGE_FALLBACK_LIMIT = "{provider} limit reached — judging with local ollama {model}"` and `JUDGE_FALLBACK_DOWN = "{provider} did not answer — judging with local ollama {model}"`. The wording is verbatim; the provider name is capitalised as displayed (`Groq`).

- [ ] **Step 1: Tests first** (respx, no network):
  - Model choice:
    - `pick_local_model` over the user's real list returns `"qwen3.6:35b-a3b"`. The list is `qwen3-embedding:8b`, `nomic-embed-text:latest`, `qwen3-embedding:0.6b`, `gemma4:12b`, `qwen3.5:9b`, `qwen3.6:35b-a3b`, `qwen3:8b`, `qwen2.5-coder:7b-instruct-q4_K_M`, `llama3.1:8b`.
    - Only `qwen2.5-coder:7b` → it is chosen (a coder beats nothing).
    - Only embeddings → `None`.
  - Switching:
    - A primary 429 (after the client's own retries; set a zero backoff) followed by a local 200 → the opinion's model is `ollama qwen3.6:35b-a3b`, `switched` equals the LIMIT wording, and `on_switch` was called once.
    - A second `complete` after the switch never calls Groq again. Assert this with respx call counts.
    - A primary 503 gives the DOWN wording.
  - When there is no fallback, the original `JudgeUnavailable` is raised, with its detail plus `; no local fallback: <why>`. This covers three cases: `/api/tags` connection refused, no qwen model installed, and `fallback = "off"`.
  - `fallback = "<explicit name>"` uses that name without calling `/api/tags`.
  - A primary that is already `ollama` gets no fallback wrapper.
  - `verify`: a run whose judge switches emits exactly one `Note` with the notice, and the report footer carries it.
  - CLI: the notice is on stderr.
  - TUI (Textual pilot): after a run whose events include the switch `Note`, the notice widget is visible, right-aligned, directly above `PromptFrame` inside `#bottom`, and hidden again when the next run starts. It must not break the fixed-height footer rule. Hide it with `display: none`, never with an empty line.
- [ ] **Step 2: Implement.**
  - The local client is a normal `JudgeClient` built from `provider_defaults("ollama")` with `model=<picked>` and a 180 s timeout. The first call loads the model into memory.
  - Keys are never logged or echoed. The local client has no key.
  - `Judge.queries` and `Judge.review` need no change beyond receiving the wrapper.
  - `search/queries.fallback_notice` stays for the case where the fallback fails as well.
- [ ] **Step 3:** `uv run pytest -q`, then `uv run ruff check . && uv run ruff format --check .`. Do not commit; leave the work unstaged for review.

#### Task 11 — Amendment B: switch at once, and prove it live (user, 2026-09-28)

A similar fallback in earlier projects of the user's "did not quite work". This one
must switch **immediately and without a visible stall**. These are requirements, not
polish:

1. **Fail fast on the primary.** Today `JudgeClient.complete` makes up to 3 attempts
   and honours `Retry-After` on a 429 (up to `max_wait`). When a fallback exists, that
   waiting is the bug.
   - Add `JudgeClient(..., fail_fast: bool = False)`. With it on, the first 429, 413,
     5xx, 4xx other than the two 400 downgrades, or transport error raises
     `JudgeUnavailable` at once: no retry, no sleep.
   - `FallbackClient` builds its primary with `fail_fast=True`.
   - Test: the respx route for Groq is called exactly once, and the fake sleep is never
     called.
   - Without a fallback (`fallback = off`, or no local model found), keep today's retry
     behaviour. Either decide the fallback before the first call (`installed_models` is
     cheap), or hand the error back to a normal retrying client. Pick one and test it.
2. **The failed batch is not lost.** The request that failed on Groq is sent again to
   the local model inside the same `complete` call. The caller sees one successful
   `Completion`.
3. **Thread-safe switch.** The TUI runs judges on worker threads. The switch goes
   through a lock, `on_switch` fires exactly once, and every later call goes straight
   to the local client. Test: two threads calling `complete` against a 429 primary
   give one switch, one notice, and one Groq call per thread at most.
4. **The notice comes before the wait.** `on_switch` fires *before* the first local
   request, so the user sees "judging with local ollama …" while the model loads.
5. **Local model output must parse.**
   - qwen3.x are thinking models. The request must not let thinking eat `max_tokens`
     or leak `<think>…</think>` into the JSON. Check the current Ollama
     OpenAI-compatible API with context7 (`reasoning_effort`, the `think` option,
     `response_format` json_schema support) and send whatever Ollama really honours.
   - Strip a leading `<think>…</think>` block before JSON parsing, defensively.
   - Test with a fixture reply containing a think block.
6. **Warm load, generous timeout.** The local client uses a 180 s timeout (the first
   call loads the model into RAM) and `keep_alive` so later batches do not reload it.
7. **Cost stays honest.** Local calls are counted apart from API calls: `api_calls`
   counts the provider requests only. The report's cost line says how many
   completions went to the local model.
8. **Live verification (controller runs this; the implementer runs it too if Ollama
   is up).** A script under `scripts/` (not a test):
   - Build a `Judge` whose primary points at a local respx-free stub server, or at a
     `base_url` that returns 429. A tiny `http.server` thread that always answers 429
     is enough.
   - Run a real `Judge.queries(["OpenAI battı."])` and a real `review` on one item
     against the real local Ollama.
   - Print the notice, the chosen model, the wall time from the first call to the
     switch (it must be under 1 s), the time to the first local answer, and the parsed
     result.
   - It must exit 0 on this machine, and its output goes in the report.
9. **Every run starts on the primary again (user, 2026-09-28).** The switch is sticky
   within one run only. The next run tries the configured API first, whether it is a
   new link in the TUI or a new `proofpath check` call. Today the CLI builds its judge
   per invocation and the TUI builds engines and judges per run or per summary, so this
   holds by construction. Pin it:
   - `FallbackClient.reset()` returns to the primary and clears `switched`.
   - Whatever builds a judge for a run either builds a fresh `FallbackClient` or calls
     `reset()` at run start.
   - Test: two runs in one TUI session. Run 1 sees a 429 and switches. Run 2's primary
     route answers 200, and run 2 uses Groq with no notice. The notice from run 1 is
     cleared when run 2 starts.
