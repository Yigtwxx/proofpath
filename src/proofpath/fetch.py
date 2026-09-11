"""The fetch ladder (spec section 7): get the bytes behind a URL, honestly.

Every URL climbs the same fixed steps and stops at the first success: httpx with a
descriptive User-Agent (1), ``curl_cffi`` TLS impersonation (2), a real browser
engine behind an explicit consent gate (3), the Wayback Machine (4). What happens
along the way is written down step by step, and every failure ends in a distinct
``Outcome`` (spec section 15) — blocked, unreachable, rate limited and "not
permitted" are different facts about the world and are never folded into one.

Content type comes from the response headers, never from the URL: a link that ends
in ``.html`` may serve a PDF. ``robots.txt`` gates steps 1 and 2 (spec section 16).

The browser step is only an interface here (``BrowserGate``); the consent prompt
and the install live elsewhere. ``DenyingGate`` is the default and counts every
refusal, so a report can say how much coverage was lost to it (product rule 6).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

import httpx
import pymupdf
from protego import Protego
from scrapling.parser import Selector

from proofpath.cache import Cache
from proofpath.config import Config, is_interactive, resolve_permission
from proofpath.polite import RETRYABLE, PoliteClient, backoff_delay, user_agent

ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"
WAYBACK_AVAILABLE = "https://archive.org/wayback/available"
PRODUCT_TOKEN = "proofpath"  # what robots.txt rules are matched against
RETRIES = 2  # same budget as PoliteClient: three attempts on 429/5xx

# A 200 that is really a bot wall. Matched case-insensitively against HTML bodies.
CHALLENGE_MARKERS = (
    "cf-chl",
    "just a moment",
    "access denied",
    "captcha",
    "enable javascript and cookies",
)
CHALLENGE_SCAN_BYTES = 200_000  # challenge pages are small; do not lowercase a whole PDF
# A 2xx HTML page with no extractable text at all is a bot wall too: the
# JavaScript/cookie shell publishers serve instead of a 403 (spec section 6.1
# measured Science at "403, 3 words"). A live 50-DOI run showed every observed
# bot-wall shell landed at exactly 0 extracted words -- never 1+ -- so the
# threshold stays at 1, not higher: raising it would misclassify a genuinely
# short real page (a redirect stub, a one-line erratum, a terse abstract page)
# as blocked, and `_failed()` sets text="", discarding real content for a false
# UNVERIFIED(blocked) -- the opposite collapse of product rule 2. Not applied to
# PDF or plain text: a short PDF is still content.
MIN_HTML_WORDS = 1

# Dropped from HTML before text extraction, subtree and all.
BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "noscript", "iframe")

# Wayback serves the raw archived page (no toolbar) when the timestamp ends in ``id_``.
_WAYBACK_TIMESTAMP = re.compile(r"(/web/\d+)/")


class TransportFailure(RuntimeError):  # noqa: N818 -- name fixed by the review finding
    """A transport-layer error from a fetch step, translated from the underlying
    library's own exception type (e.g. ``curl_cffi``'s), so the ladder never has to
    import — or blindly catch everything from — a transport it only calls lazily."""


class Outcome(str, Enum):
    OK = "ok"
    BLOCKED = "UNVERIFIED (blocked)"
    BLOCKED_ROBOTS = "UNVERIFIED (blocked, robots.txt)"
    BLOCKED_NO_BROWSER = "UNVERIFIED (blocked, browser not permitted)"
    UNREACHABLE = "UNVERIFIED (unreachable)"
    UNAVAILABLE = "UNVERIFIED (provider unavailable)"  # 429/5xx after backoff
    NETWORK_DENIED = "UNVERIFIED (network not permitted)"  # permissions.network == "deny"


# The ladder's steps by number, for reports: ``step 2 (curl_cffi)``.
STEP_NAMES = {0: "cache", 1: "httpx", 2: "curl_cffi", 3: "browser", 4: "wayback"}

Kind = Literal["pdf", "html", "text", "other"]
Verdict = Literal["ok", "blocked", "unreachable", "retryable"]


@dataclass(frozen=True)
class Fetched:
    url: str
    final_url: str
    step: int  # 0 = cache, 1 httpx, 2 curl_cffi, 3 browser, 4 wayback; the last step tried
    outcome: Outcome
    status: int | None
    content_type: str
    kind: Kind
    body: bytes
    text: str
    notes: list[str]  # one line per step tried, e.g. "step 1 httpx: HTTP 403 (blocked)"
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class FetchStats:
    """What the report needs to state its own coverage (product rule 6)."""

    counts: dict[Outcome, int]
    browser_skipped: int


class BrowserGate(Protocol):
    """Step 3. ``allow`` decides (and counts refusals); ``fetch`` returns status 0 on failure.

    Two counters, because the report speaks of sources and the gate sees URLs:
    ``skipped_urls`` is bumped by ``allow`` on every refusal, while ``skipped`` is
    bumped once per *source* by whoever knows what a source is — ``Fetcher.fetch``
    for a bare URL, the open-access chain for a DOI with several locations.
    """

    skipped: int  # sources the browser could have rescued (what the report prints)
    skipped_urls: int  # every refusal, one per location the ladder asked about

    def allow(self, host: str, status: int | None) -> bool: ...

    def fetch(self, url: str) -> tuple[int, bytes, str]: ...  # (status, body, content_type)


class DenyingGate:
    """The default gate: the browser is never used, and every refusal is counted."""

    def __init__(self) -> None:
        self.skipped = 0
        self.skipped_urls = 0

    def allow(self, host: str, status: int | None) -> bool:
        self.skipped_urls += 1
        return False

    def fetch(self, url: str) -> tuple[int, bytes, str]:
        raise RuntimeError("DenyingGate never allows a browser fetch")


CurlGet = Callable[[str], tuple[int, bytes, str, str]]  # (status, body, content_type, final_url)
WaybackLookup = Callable[[str], str | None]  # url -> snapshot url or None
# The ladder's own lookup also says *why* there is no snapshot: (snapshot, error).
# An injected ``WaybackLookup`` is wrapped into this shape with error ``None``.
_WaybackLookup = Callable[[str], tuple[str | None, str | None]]


@dataclass(frozen=True)
class _Response:
    """One step's answer, whichever transport produced it."""

    status: int | None  # None or 0: no HTTP response at all
    body: bytes
    content_type: str
    final_url: str
    error: str = ""  # exception type name when there is no status
    retry_after: float | None = None  # numeric Retry-After header, in seconds

    @property
    def kind(self) -> Kind:
        return content_kind(self.content_type, self.body)

    @property
    def verdict(self) -> Verdict:
        return classify(self.status, self.kind, self.body)

    def describe(self, detail: str) -> str:
        what = f"HTTP {self.status}" if self.status else (self.error or "no response")
        return f"{what} ({detail})"

    def explain(self, verdict: Verdict) -> str:
        """``describe`` for the verdict, naming the reason when a 2xx HTML page was
        still ``blocked`` without a challenge marker: it had no text to speak of."""
        if (
            verdict == "blocked"
            and self.status
            and 200 <= self.status < 300
            and self.kind == "html"
            and not _has_challenge(self.body)
        ):
            return self.describe("blocked: empty page")
        return self.describe(verdict)


# --- pure helpers ------------------------------------------------------------


def classify(status: int | None, kind: Kind, body: bytes) -> Verdict:
    """Sort a response into what the ladder does next (spec section 15)."""
    if not status:
        return "unreachable"
    if status in (404, 410):
        return "unreachable"
    if status in (401, 403, 406):
        return "blocked"
    if kind == "html" and (200 <= status < 300 or status == 503) and _has_challenge(body):
        return "blocked"
    if kind == "html" and 200 <= status < 300 and _html_words(body) < MIN_HTML_WORDS:
        return "blocked"
    if 200 <= status < 300:
        return "ok"
    if status in RETRYABLE:
        return "retryable"
    if 400 <= status < 500:
        return "blocked"
    if status >= 500:
        return "retryable"
    # 1xx/3xx that survived redirect following: nothing usable came back.
    return "unreachable"


def _has_challenge(body: bytes) -> bool:
    head = body[:CHALLENGE_SCAN_BYTES].decode("utf-8", errors="replace").lower()
    return any(marker in head for marker in CHALLENGE_MARKERS)


def _html_words(body: bytes) -> int:
    """The same extraction ``_finish`` runs, so the count is the one a report would
    show. The whole body is parsed rather than a slice: a real article page whose
    text starts late must never read as a wall, and a parse is cheap next to the
    request that produced it."""
    return len(extract_text(body, "html").split())


def content_kind(content_type: str, body: bytes) -> Kind:
    """From the header, not the URL. Only an absent or opaque type is sniffed."""
    media = content_type.split(";", 1)[0].strip().lower()
    if media == "application/pdf":
        return "pdf"
    if media in ("text/html", "application/xhtml+xml"):
        return "html"
    if media == "text/plain":
        return "text"
    if media in ("", "application/octet-stream"):
        return "pdf" if body.startswith(b"%PDF-") else "other"
    return "other"


def extract_text(body: bytes, kind: Kind) -> str:
    """Plain text for the pipeline. Raises ``pymupdf.FileDataError`` on a corrupt PDF."""
    if kind == "pdf":
        # pymupdf ships py.typed but leaves ``open`` itself unannotated.
        with pymupdf.open(stream=body, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            return "\f".join(page.get_text() for page in doc)
    if kind == "html":
        if not body.strip():
            return ""
        page = Selector(body)
        # The first article or main element is the content; the body is the fallback.
        for selector in ("article", "main", "body"):
            found = page.css(selector)
            if found:
                text: str = found[0].get_all_text(
                    separator="\n", strip=True, ignore_tags=BOILERPLATE_TAGS
                )
                return text
        return ""
    if kind == "text":
        return body.decode("utf-8", errors="replace")
    return ""


def html_title(body: bytes) -> str:
    if not body.strip():
        return ""
    title = Selector(body).css("title::text").get()
    return str(title).strip() if title else ""


def wayback_snapshot_url(payload: dict[str, Any]) -> str | None:
    """The closest snapshot from the availability API, rewritten to its raw ``id_`` form."""
    closest = payload.get("archived_snapshots", {}).get("closest") or {}
    url = closest.get("url")
    if not closest.get("available") or not isinstance(url, str) or not url:
        return None
    return _WAYBACK_TIMESTAMP.sub(r"\1id_/", url, count=1)


def default_curl_get(
    url: str, *, timeout: float = 20.0, contact_email: str = ""
) -> tuple[int, bytes, str, str]:
    """Step 2: a Chrome-impersonating GET. Imported lazily so the module stays light."""
    from curl_cffi import requests as curl_requests
    from curl_cffi.requests.exceptions import RequestException

    try:
        response = curl_requests.get(
            url,
            impersonate="chrome",
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": user_agent(contact_email), "Accept": ACCEPT},
        )
    except (RequestException, OSError) as exc:
        # Translated so callers can catch one type instead of importing curl_cffi's
        # exception tree (or catching every exception, which would hide real bugs).
        raise TransportFailure(type(exc).__name__) from exc
    return (
        response.status_code,
        response.content,
        response.headers.get("content-type", ""),
        str(response.url),
    )


def network_permission(config: Config, *, interactive: bool) -> tuple[bool, str]:
    """``(allowed, note)`` for ``permissions.network`` on this run. The note is the
    line a report carries: why nothing was fetched, or that ``ask`` was let through
    because a terminal could have been asked (the ladder has no prompt of its own
    for plain HTTP). Empty for a plain ``allow``."""
    decision = resolve_permission(config.permissions.network, interactive=interactive)
    if decision.outcome == "deny":
        reason = (
            "permissions.network = deny"
            if config.permissions.network == "deny"
            else decision.reason
        )
        return False, f"network: not permitted ({reason})"
    if decision.outcome == "prompt":
        return True, "network permission is 'ask'; allowed for this run"
    return True, ""


# --- the ladder --------------------------------------------------------------


class Fetcher:
    def __init__(
        self,
        *,
        config: Config,
        gate: BrowserGate | None = None,
        cache: Cache | None = None,
        client: httpx.Client | None = None,
        curl_get: CurlGet | None = None,
        wayback_lookup: WaybackLookup | None = None,
        timeout: float = 20.0,
        interactive: bool | None = None,
    ) -> None:
        """``client``, if given, must follow redirects: the ladder relies on
        ``response.url`` already being the post-redirect ``final_url``.

        ``interactive`` decides what ``permissions.network = "ask"`` means for this
        run (spec section 7.1: no TTY, no prompt — ``ask`` is ``deny``). ``None``
        looks at the real terminal; tests inject a value.
        """
        self._config = config
        # Resolved once: every step, and the open-access chain's providers, read
        # ``network_allowed`` instead of the config, so nothing can slip past it.
        if interactive is None:
            interactive = is_interactive()
        self.network_allowed, self.network_note = network_permission(
            config, interactive=interactive
        )
        self._gate: BrowserGate = gate or DenyingGate()
        self._cache = cache
        self._timeout = timeout
        self._polite = PoliteClient(
            contact_email=config.contact.email,
            client=client,
            timeout=timeout,
            follow_redirects=True,
        )
        self._curl_get = curl_get
        self._wayback: _WaybackLookup = (
            self._lookup_wayback if wayback_lookup is None else _plain_lookup(wayback_lookup)
        )
        self._robots: dict[str, Protego | None] = {}  # host -> rules; None = no usable file
        self._counts: Counter[Outcome] = Counter()

    def close(self) -> None:
        self._polite.client.close()

    @property
    def gate(self) -> BrowserGate:
        return self._gate

    def summary(self) -> FetchStats:
        return FetchStats(counts=dict(self._counts), browser_skipped=self._gate.skipped)

    def fetch(
        self, url: str, *, text_kind: str = "fulltext", counts_as_source: bool = True
    ) -> Fetched:
        """One climb. A bare URL is one source, so a browser refusal here is one
        skipped source; the open-access chain fetches several locations for one
        DOI and counts that source itself (``counts_as_source=False``)."""
        result = self._climb(url, text_kind)
        self._counts[result.outcome] += 1
        if counts_as_source and result.outcome is Outcome.BLOCKED_NO_BROWSER:
            self._gate.skipped += 1
        return result

    def _climb(self, url: str, text_kind: str) -> Fetched:
        notes: list[str] = []
        if self.network_note:
            notes.append(self.network_note)
        if not self.network_allowed:
            return _failed(url, 0, Outcome.NETWORK_DENIED, _no_response(url), notes)
        if self._cache is not None:
            cached = self._cache.get_raw_text(f"url:{url}")
            if cached:  # only non-empty text is ever stored, so empty means miss
                notes.append("cache: hit")
                return Fetched(
                    url=url,
                    final_url=url,
                    step=0,
                    outcome=Outcome.OK,
                    status=None,
                    content_type="",
                    kind="text",
                    body=b"",
                    text=cached,
                    notes=notes,
                    from_cache=True,
                )
        if self._config.fetch.respect_robots and not self._robots_allow(url):
            notes.append(f"step 1 httpx: robots.txt disallows {PRODUCT_TOKEN}")
            return _failed(url, 1, Outcome.BLOCKED_ROBOTS, _no_response(url), notes)

        # Step 1: httpx.
        step = 1
        response, verdict, note = self._with_backoff(url)
        notes.append(f"step 1 httpx: {note}")
        if verdict == "ok":
            return self._finish(url, step, response, text_kind, notes)
        # The final outcome comes from the last step that got an HTTP answer. A step
        # that failed at transport level (curl_cffi's TLS, a browser crash) only
        # leaves its note: it says nothing about the page, so it never turns a
        # wall into "unreachable" or skips the consent gate (product rule 2).
        failure, failure_verdict = response, verdict
        browser_refused = False
        if failure_verdict == "blocked":
            # Step 2: TLS impersonation.
            step = 2
            response, verdict, note = self._curl(url)
            notes.append(f"step 2 curl_cffi: {note}")
            if verdict == "ok":
                return self._finish(url, step, response, text_kind, notes)
            if response.status:
                failure, failure_verdict = response, verdict
        if failure_verdict == "blocked":
            # Step 3: browser, only with consent. The gate counts its own refusals.
            step = 3
            if self._gate.allow(httpx.URL(url).host, failure.status):
                response, verdict, note = self._browser(url)
                notes.append(f"step 3 browser: {note}")
                if verdict == "ok":
                    return self._finish(url, step, response, text_kind, notes)
                if response.status:
                    failure, failure_verdict = response, verdict
            else:
                browser_refused = True
                notes.append("step 3 browser: not permitted")
        if failure_verdict == "retryable":
            # Rate limited or down: a wait, not a wall. An archive copy would hide that.
            return _failed(url, step, Outcome.UNAVAILABLE, failure, notes)

        # Step 4: Wayback, for blocked and unreachable alike. A miss reports the
        # last real failure, not the archive's.
        step = 4
        snapshot, error = self._wayback(url)
        if snapshot is None:
            notes.append(
                f"step 4 wayback: lookup failed ({error})"
                if error
                else "step 4 wayback: no snapshot"
            )
        else:
            response, verdict, note = self._with_backoff(snapshot)
            notes.append(f"step 4 wayback: {note}")
            if verdict == "ok":
                return self._finish(url, step, response, text_kind, notes)
        if failure_verdict == "unreachable":
            outcome = Outcome.UNREACHABLE
        elif browser_refused:
            outcome = Outcome.BLOCKED_NO_BROWSER
        else:
            outcome = Outcome.BLOCKED
        return _failed(url, step, outcome, failure, notes)

    def _finish(
        self, url: str, step: int, response: _Response, text_kind: str, notes: list[str]
    ) -> Fetched:
        kind = content_kind(response.content_type, response.body)
        try:
            text = extract_text(response.body, kind)
        except pymupdf.FileDataError as exc:
            # Unparseable input fails loudly and the run continues (spec section 15).
            notes.append(f"text extraction failed: {type(exc).__name__}")
            text = ""
        if not text:
            # Reached is not read: an image, a corrupt PDF or an opaque type gives the
            # pipeline nothing to verify against, and the report must say so.
            notes.append("reached but no text extracted")
        if self._cache is not None and text:
            self._cache.add_source(
                f"url:{url}",
                scheme="web",
                title=html_title(response.body) if kind == "html" else "",
                url=response.final_url,
                text_kind=text_kind,
                raw_text=text,
            )
        return Fetched(
            url=url,
            final_url=response.final_url,
            step=step,
            outcome=Outcome.OK,
            status=response.status,
            content_type=response.content_type,
            kind=kind,
            body=response.body,
            text=text,
            notes=notes,
        )

    # --- transports ---------------------------------------------------------

    def _httpx_get(self, url: str) -> _Response:
        # PoliteClient.get raises on 429/5xx; the ladder needs the status to classify.
        try:
            response = self._polite.client.get(url, headers={"Accept": ACCEPT})
        except httpx.HTTPError as exc:
            return _Response(None, b"", "", url, error=type(exc).__name__)
        retry_after = response.headers.get("Retry-After", "")
        return _Response(
            response.status_code,
            response.content,
            response.headers.get("content-type", ""),
            str(response.url),
            retry_after=float(retry_after) if retry_after.isdigit() else None,
        )

    def _with_backoff(self, url: str) -> tuple[_Response, Verdict, str]:
        """An httpx GET with the PoliteClient backoff; the last response is returned, not raised."""
        for attempt in range(RETRIES + 1):
            self._polite.throttle(url)
            response = self._httpx_get(url)
            verdict = response.verdict
            if verdict != "retryable":
                return response, verdict, response.explain(verdict)
            retry_after = (
                str(int(response.retry_after)) if response.retry_after is not None else None
            )
            delay = backoff_delay(attempt, response.status or 0, retry_after)
            if delay is None:
                # A daily budget is gone (OpenAlex answers with hours); do not wait.
                detail = f"retryable, Retry-After {response.retry_after:.0f}s exceeds cap"
                return response, verdict, response.describe(detail)
            if attempt < RETRIES:
                time.sleep(delay)
        detail = f"retryable, gave up after {RETRIES + 1} attempts"
        return response, "retryable", response.describe(detail)

    def _curl(self, url: str) -> tuple[_Response, Verdict, str]:
        self._polite.throttle(url)
        get = self._curl_get or self._default_curl
        try:
            status, body, content_type, final_url = get(url)
        except TransportFailure as exc:
            # default_curl_get already translated the underlying curl_cffi exception;
            # its message is that exception's type name.
            return _transport_failed(url, str(exc))
        except OSError as exc:
            # An injected transport (test double or future implementation) may raise a
            # plain OSError directly: still a transport failure, not a bug in the ladder.
            return _transport_failed(url, type(exc).__name__)
        response = _Response(status, body, content_type, final_url)
        return response, response.verdict, response.explain(response.verdict)

    def _default_curl(self, url: str) -> tuple[int, bytes, str, str]:
        # Looked up at call time so tests can monkeypatch ``proofpath.fetch.default_curl_get``.
        return default_curl_get(
            url, timeout=self._timeout, contact_email=self._config.contact.email
        )

    def _browser(self, url: str) -> tuple[_Response, Verdict, str]:
        self._polite.throttle(url)
        status, body, content_type = self._gate.fetch(url)
        if not status:  # the gate's "the browser itself failed" answer
            return _transport_failed(url, "")
        response = _Response(status, body, content_type, url)
        return response, response.verdict, response.explain(response.verdict)

    def _lookup_wayback(self, url: str) -> tuple[str | None, str | None]:
        """``(snapshot url, error)``. The error is set only when the call itself failed,
        never for a well-formed response that simply has no snapshot (a plain miss)."""
        self._polite.throttle(WAYBACK_AVAILABLE)
        try:
            response = self._polite.client.get(WAYBACK_AVAILABLE, params={"url": url})
        except httpx.HTTPError as exc:
            return None, type(exc).__name__
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        try:
            payload = response.json()
        except ValueError as exc:
            return None, type(exc).__name__
        if not isinstance(payload, dict):
            return None, "malformed response"
        return wayback_snapshot_url(payload), None

    # --- robots -------------------------------------------------------------

    def _robots_allow(self, url: str) -> bool:
        parsed = httpx.URL(url)
        host = parsed.host
        if host not in self._robots:
            robots_url = str(parsed.copy_with(path="/robots.txt", query=None, fragment=None))
            self._robots[host] = self._load_robots(robots_url)
        rules = self._robots[host]
        return rules is None or bool(rules.can_fetch(url, PRODUCT_TOKEN))

    def _load_robots(self, robots_url: str) -> Protego | None:
        """A missing, erroring or unreachable robots.txt allows everything."""
        self._polite.throttle(robots_url)
        try:
            response = self._polite.client.get(robots_url)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        return Protego.parse(response.text)


def _no_response(url: str) -> _Response:
    return _Response(None, b"", "", url)


def _transport_failed(url: str, error: str) -> tuple[_Response, Verdict, str]:
    """A step that never got an HTTP answer. Its verdict is ``unreachable`` for the
    step itself; ``_climb`` decides whether that says anything about the page."""
    response = _Response(None, b"", "", url, error=error)
    return response, "unreachable", response.describe("transport failed")


def _plain_lookup(lookup: WaybackLookup) -> _WaybackLookup:
    """An injected lookup has no failure channel: a ``None`` is a plain miss."""
    return lambda url: (lookup(url), None)


def _failed(
    url: str, step: int, outcome: Outcome, response: _Response, notes: list[str]
) -> Fetched:
    return Fetched(
        url=url,
        final_url=response.final_url,
        step=step,
        outcome=outcome,
        status=response.status,
        content_type=response.content_type,
        kind=content_kind(response.content_type, response.body),
        body=response.body,
        text="",
        notes=notes,
    )
