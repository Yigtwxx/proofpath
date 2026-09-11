"""The polite HTTP core shared by ``resolve.py``, and by the fetch ladder and the
open-access chain in later phases (spec section 7).

A single client concern lives here: a descriptive User-Agent, a minimum interval
between requests to the same host, and exponential backoff on 429/5xx that honours
``Retry-After`` up to a cap. None of this is provider-specific — Crossref, OpenAlex,
a publisher landing page and a repository API all get the same treatment.
"""

from __future__ import annotations

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
}
MAX_RETRY_AFTER = 60.0
RETRYABLE = (429, 500, 502, 503, 504)


class ProviderError(RuntimeError):
    pass


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
    ) -> None:
        self._email = contact_email
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent(contact_email)},
            timeout=timeout,
            follow_redirects=follow_redirects,
        )
        self._retries = retries
        self._last_call: dict[str, float] = {}

    @property
    def email(self) -> str:
        return self._email

    @property
    def client(self) -> httpx.Client:
        return self._client

    def throttle(self, url: str) -> None:
        host = httpx.URL(url).host
        wait = (
            self._last_call.get(host, -1e9)
            + MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
            - time.monotonic()
        )
        if wait > 0:
            time.sleep(wait)
        self._last_call[host] = time.monotonic()

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
        last = ""
        for attempt in range(self._retries + 1):
            self.throttle(url)
            delay: float | None = 0.5 * 2.0**attempt
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last = type(exc).__name__
            else:
                if response.status_code == 404 or response.status_code < 400:
                    return response
                last = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE:
                    break
                retry_after = response.headers.get("Retry-After")
                delay = backoff_delay(attempt, response.status_code, retry_after)
                if delay is None:
                    # A daily budget is gone (OpenAlex answers with hours).
                    raise ProviderError(f"{last}, retry after {retry_after}s")
            if attempt < self._retries:
                # None only when backoff_delay's cap check already raised, above.
                assert delay is not None
                time.sleep(delay)
        raise ProviderError(last)
