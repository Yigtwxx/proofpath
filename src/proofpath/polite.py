"""The polite HTTP core shared by ``resolve.py``, and by the fetch ladder and the
open-access chain in later phases (spec section 7).

A single client concern lives here: a descriptive User-Agent, a minimum interval
between requests to the same host, and exponential backoff on 429/5xx that honours
``Retry-After`` up to a cap. None of this is provider-specific — Crossref, OpenAlex,
a publisher landing page and a repository API all get the same treatment.

The interval is a property of the *host*, not of the object talking to it, so the
throttle lives in one :class:`HostThrottle` that every client shares by default.
A TUI session runs several verifications at once (spec section 13.1) and each one
builds its own engine, hence its own clients; per-instance pacing would let three
runs hit Crossref three times in the interval meant for one.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import httpx

from proofpath import __version__

REPO_URL = "https://github.com/Yigtwxx/proofpath"

DEFAULT_MIN_INTERVAL = 1.0  # seconds between requests to a host not listed below

# Minimum seconds between requests to one host. Crossref and OpenAlex tolerate
# a few requests per second in the polite pool; arXiv asks for one every 3 s.
MIN_INTERVAL: dict[str, float] = {
    "api.crossref.org": 0.25,
    "api.openalex.org": 0.25,
    "api.semanticscholar.org": 1.1,
    "api.unpaywall.org": 0.5,
    "export.arxiv.org": 3.0,
    "openlibrary.org": 1.0,
    "www.ebi.ac.uk": 0.5,
    # The read-only social APIs of spec section 6.2. Bluesky and Hacker News publish
    # no rate limit for unauthenticated reads, so they get the interval a polite
    # client would keep anyway; a run reads one post, not a feed. Reddit does publish
    # one -- 60 requests a minute for an OAuth client -- and both of its hosts are
    # written down at that rate rather than left to the default that happens to match.
    # Lobste.rs publishes none either and gets the same courtesy. Mastodon and Lemmy
    # are not here: the host is the instance, and there is no list of those.
    "public.api.bsky.app": 0.5,
    "hacker-news.firebaseio.com": 0.5,
    "lobste.rs": 0.5,
    "www.reddit.com": 1.0,
    "oauth.reddit.com": 1.0,
}
MAX_RETRY_AFTER = 60.0
RETRYABLE = (429, 500, 502, 503, 504)


class ProviderError(RuntimeError):
    """A request that never produced an answer, and the status that ended it.

    ``status`` is ``None`` when nothing came back at all (a transport error). It is
    carried because 4xx and 5xx are different facts about a source -- gone, versus a
    provider that was down and said nothing about it (product rule 2) -- and the
    message alone cannot be read back reliably by the caller that has to tell them
    apart.

    ``code`` is the one word a JSON error body named, when it named one -- Lemmy's
    ``{"error": "not_logged_in"}`` against its ``{"error": "couldnt_find_post"}``,
    the same 400 for two facts spec section 15 keeps apart. Only a bare identifier
    is kept (:data:`ERROR_CODE`): a message or an echoed request is not a code, and
    must not travel in an exception that ends up in a report (security rules).
    """

    def __init__(self, message: str, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


#: The shape of an error token worth carrying: a short snake_case identifier.
ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def error_code(response: httpx.Response) -> str | None:
    """The bare token an error body's ``error`` field named, or ``None``."""
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("error")
    return code if isinstance(code, str) and ERROR_CODE.match(code) else None


class HostThrottle:
    """The minimum interval between two requests to one host, shared and thread-safe.

    One lock per host, held across the wait, so concurrent callers queue instead of
    all reading the same last-call time and firing together; the dict of hosts has a
    lock of its own, so two runs talking to two hosts never wait on each other.

    ``MIN_INTERVAL`` is read at call time rather than copied in, so a caller that
    adjusts a host's interval is obeyed by clients that already exist.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}

    @staticmethod
    def interval(host: str) -> float:
        return MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)

    def wait(self, host: str) -> float:
        """Block until this host may be called again; return the seconds waited."""
        with self._guard:
            lock = self._locks.setdefault(host, threading.Lock())
        interval = self.interval(host)
        with lock:
            with self._guard:
                last = self._last.get(host)
            waited = 0.0
            if last is not None:
                waited = last + interval - time.monotonic()
                if waited > 0:
                    time.sleep(waited)
            with self._guard:
                self._last[host] = time.monotonic()
            return max(waited, 0.0)

    def reset(self) -> None:
        """Forget every recorded call. For tests and for a session that starts over."""
        with self._guard:
            self._last.clear()


#: The throttle every :class:`PoliteClient` uses unless it is handed another one.
SHARED_THROTTLE = HostThrottle()


def user_agent(contact_email: str = "") -> str:
    """``proofpath/<version> (<REPO_URL>)``, with ``; mailto:<email>`` when given."""
    agent = f"proofpath/{__version__} ({REPO_URL}"
    agent += f"; mailto:{contact_email})" if contact_email else ")"
    return agent


def backoff_delay(attempt: int, status: int, retry_after: str | None) -> float | None:
    """The wait before the next attempt on a retryable response.

    ``0.5 * 2**attempt`` by default; a numeric ``Retry-After`` overrides it, unless it
    exceeds ``MAX_RETRY_AFTER`` (a daily budget is gone), in which case ``None`` tells
    the caller to stop instead of waiting. A bare 429 with no ``Retry-After`` waits at
    least ``2.0 * 2**attempt`` (APIs that 429 without a header still mean "slow down").
    """
    delay = 0.5 * 2.0**attempt
    if retry_after and retry_after.isdigit():
        seconds = float(retry_after)
        return None if seconds > MAX_RETRY_AFTER else seconds
    if status == 429:
        return max(delay, 2.0 * 2.0**attempt)
    return delay


class PoliteClient:
    """An httpx client that is polite: descriptive UA, throttled, retried with backoff."""

    def __init__(
        self,
        *,
        contact_email: str = "",
        client: httpx.Client | None = None,
        retries: int = 2,
        timeout: float = 20.0,
        follow_redirects: bool = False,
        throttle: HostThrottle | None = None,
    ) -> None:
        self._email = contact_email
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent(contact_email)},
            timeout=timeout,
            follow_redirects=follow_redirects,
        )
        self._retries = retries
        # Shared by default: politeness is owed to the host, not kept per client.
        self._throttle = throttle if throttle is not None else SHARED_THROTTLE

    @property
    def email(self) -> str:
        return self._email

    @property
    def client(self) -> httpx.Client:
        return self._client

    def throttle(self, url: str) -> None:
        """Wait out this host's minimum interval, counting every client's calls."""
        self._throttle.wait(httpx.URL(url).host)

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        mailto: bool = True,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        params = dict(params or {})
        if mailto and self._email:
            params["mailto"] = self._email
        return self._send("GET", url, params=params, headers=headers)

    def post(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> httpx.Response:
        """A form POST, throttled and retried like :meth:`get`.

        Only one caller needs it: the OAuth token exchange a user's own Reddit app
        makes (spec section 6.2). ``auth`` is HTTP basic and ``data`` is the form
        body, so neither ever reaches the URL — a credential does not belong in a
        query string, which is the part of a request that gets logged.
        """
        return self._send("POST", url, data=data, headers=headers, auth=auth)

    def _send(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> httpx.Response:
        """The one request loop: throttle, then retry 429/5xx with backoff.

        ``ProviderError`` carries the status and nothing else. It is raised into
        callers that put it in a report, so the body -- which is where a provider
        echoes an account, a key or a token back at you -- is never read into it.
        """
        last = ""
        status: int | None = None
        code: str | None = None
        for attempt in range(self._retries + 1):
            self.throttle(url)
            delay: float | None = 0.5 * 2.0**attempt
            try:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=headers,
                    auth=auth or httpx.USE_CLIENT_DEFAULT,
                )
            except httpx.HTTPError as exc:
                last = type(exc).__name__
                status = None
                code = None
            else:
                if response.status_code == 404 or response.status_code < 400:
                    return response
                last = f"HTTP {response.status_code}"
                status = response.status_code
                code = error_code(response)
                if response.status_code not in RETRYABLE:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = backoff_delay(attempt, response.status_code, retry_after)
                if delay is None:
                    # A daily budget is gone (OpenAlex answers with hours).
                    raise ProviderError(f"{last}, retry after {retry_after}s", status, code)
            if attempt < self._retries:
                # None only when backoff_delay's cap check already raised, above.
                assert delay is not None
                time.sleep(delay)
        raise ProviderError(last, status, code)
