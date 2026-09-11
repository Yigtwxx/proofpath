"""The fetch ladder (spec section 7): steps 1, 2 and 4, robots, content type, honesty states."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
import respx

from proofpath import fetch as fx
from proofpath.cache import Cache
from proofpath.config import Config, FetchConfig, Permissions
from proofpath.polite import DEFAULT_MIN_INTERVAL, user_agent

PAGE = "https://x.test/paper.html"
SNAPSHOT = "https://web.archive.org/web/20240101000000id_/https://x.test/paper.html"

HTML = b"""<html><head><title>Evidence &amp; co</title>
<script>var tracked = 1;</script><style>p { color: red }</style></head>
<body>
<nav><a href="/">Home</a><a href="/about">About the journal</a></nav>
<header><h2>Site header banner</h2></header>
<article><h1>Findings</h1><p>The trial enrolled 240 patients.</p>
<p>They were followed for two years across three sites, and the primary endpoint
was met in 61% of the treatment arm versus 42% of controls.</p>
<noscript>Enable scripts</noscript></article>
<footer>Copyright boilerplate</footer>
</body></html>"""

CHALLENGE = b"<html><head><title>Just a moment...</title></head><body>cf-chl</body></html>"
# A 200 that is really a bot wall: a JavaScript/cookie shell with zero extracted
# words -- all the content lives in <script>, which extract_text drops (it is a
# boilerplate tag) -- matching every bot-wall shell observed in the 2026-09-11
# coverage run (Elsevier, JSTOR, IEEE and T&F landing pages all measured 0 words).
EMPTY_SHELL = (
    b"<html><head><script>boot()</script></head>"
    b"<body><script>document.write('Loading the article page now')</script></body></html>"
)
# A genuinely short but real page (a redirect stub / one-line erratum / terse
# abstract): non-zero extracted words, so it must stay `ok`, not `blocked`.
SHORT_REAL_PAGE = b"<html><body><p>Moved. See the new page.</p></body></html>"

WAYBACK_PAYLOAD: dict[str, Any] = {
    "url": "http://example.com/",
    "archived_snapshots": {
        "closest": {
            "status": "200",
            "available": True,
            "url": "http://web.archive.org/web/20130919044612/http://example.com/",
            "timestamp": "20130919044612",
        }
    },
}


def pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    body: bytes = doc.tobytes()
    doc.close()
    return body


class StubGate:
    """Records what the ladder asked, answers with a fixed decision."""

    def __init__(self, allow: bool = False, result: tuple[int, bytes, str] | None = None) -> None:
        self._allow = allow
        self._result = result or (200, HTML, "text/html; charset=utf-8")
        self.skipped = 0  # per source; the ladder and the OA chain count these
        self.skipped_urls = 0  # per refusal, counted here like a real gate
        self.calls: list[tuple[str, int | None]] = []
        self.fetched: list[str] = []

    def allow(self, host: str, status: int | None) -> bool:
        self.calls.append((host, status))
        if not self._allow:
            self.skipped_urls += 1
        return self._allow

    def fetch(self, url: str) -> tuple[int, bytes, str]:
        self.fetched.append(url)
        return self._result


def curl_returning(status: int, body: bytes = b"", ctype: str = "text/html") -> fx.CurlGet:
    return lambda url: (status, body, ctype, url)


def curl_forbidden(url: str) -> tuple[int, bytes, str, str]:
    raise AssertionError(f"step 2 must not run for {url}")


def wayback_none(url: str) -> str | None:
    return None


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fake time: sleep advances the clock, so throttling never really waits."""
    now = {"t": 1000.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(fx.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(fx.time, "sleep", fake_sleep)
    return slept


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(follow_redirects=True) as c:
        yield c


def make(
    client: httpx.Client,
    *,
    respect_robots: bool = False,
    network: str = "allow",
    interactive: bool = False,
    gate: StubGate | None = None,
    curl: fx.CurlGet | None = curl_forbidden,
    wayback: fx.WaybackLookup | None = wayback_none,
    cache: Cache | None = None,
) -> fx.Fetcher:
    config = Config(
        permissions=Permissions(network=network),  # type: ignore[arg-type]
        fetch=FetchConfig(respect_robots=respect_robots),
    )
    return fx.Fetcher(
        config=config,
        gate=gate,
        cache=cache,
        client=client,
        curl_get=curl,
        wayback_lookup=wayback,
        interactive=interactive,
    )


def serve(url: str = PAGE, status: int = 200, body: bytes = HTML, ctype: str = "text/html") -> Any:
    return respx.get(url).mock(
        return_value=httpx.Response(status, content=body, headers={"content-type": ctype})
    )


# --- content type and text extraction ---------------------------------------


@respx.mock
def test_pdf_detected_from_header_not_suffix(client: httpx.Client) -> None:
    serve(body=pdf_bytes("Evidence lives here"), ctype="application/pdf")
    result = make(client).fetch(PAGE)
    assert result.ok and result.step == 1
    assert result.kind == "pdf"
    assert "Evidence lives here" in result.text
    assert result.final_url == PAGE


@respx.mock
def test_html_text_prefers_article_and_drops_boilerplate(client: httpx.Client) -> None:
    serve()
    result = make(client).fetch(PAGE)
    assert result.kind == "html"
    assert "The trial enrolled 240 patients." in result.text
    assert "Findings" in result.text
    for boilerplate in (
        "Home",
        "About the journal",
        "Site header banner",
        "tracked",
        "Copyright",
        "Enable scripts",
    ):
        assert boilerplate not in result.text
    assert fx.html_title(HTML) == "Evidence & co"
    assert result.words == 31


def test_html_falls_back_to_main_then_body() -> None:
    assert (
        fx.extract_text(b"<body><nav>x</nav><main>Main text</main></body>", "html") == "Main text"
    )
    assert (
        fx.extract_text(b"<body><p>Plain body</p><footer>f</footer></body>", "html") == "Plain body"
    )
    assert fx.extract_text(b"", "html") == ""
    assert fx.html_title(b"<body>no title</body>") == ""


@respx.mock
def test_octet_stream_pdf_is_sniffed_by_magic(client: httpx.Client) -> None:
    serve(body=pdf_bytes("Sniffed"), ctype="application/octet-stream")
    result = make(client).fetch(PAGE)
    assert result.kind == "pdf"
    assert "Sniffed" in result.text


@respx.mock
def test_unknown_content_type_is_other_with_empty_text(client: httpx.Client) -> None:
    serve(body=b"\x00\x01binary", ctype="image/png")
    result = make(client).fetch(PAGE)
    assert result.ok
    assert result.kind == "other"
    assert result.text == ""
    assert result.words == 0
    assert result.notes[-1] == "reached but no text extracted"


@pytest.mark.parametrize(
    ("ctype", "body", "kind"),
    [
        ("application/pdf", b"", "pdf"),
        ("APPLICATION/PDF; charset=binary", b"", "pdf"),
        ("text/html; charset=utf-8", b"", "html"),
        ("application/xhtml+xml", b"", "html"),
        ("text/plain", b"", "text"),
        ("", b"%PDF-1.7", "pdf"),
        ("", b"<html>", "other"),
        ("application/octet-stream", b"%PDF-1.7", "pdf"),
        ("application/octet-stream", b"zip", "other"),
        ("application/json", b"%PDF-1.7", "other"),
    ],
)
def test_content_kind_comes_from_headers(ctype: str, body: bytes, kind: str) -> None:
    assert fx.content_kind(ctype, body) == kind


def test_text_kind_decodes_utf8_with_replacement() -> None:
    assert fx.extract_text(b"caf\xc3\xa9 \xff", "text") == "café �"
    assert fx.extract_text(b"anything", "other") == ""


@respx.mock
def test_corrupt_pdf_is_noted_not_crashed(client: httpx.Client) -> None:
    serve(body=b"%PDF-1.4 not really a pdf", ctype="application/pdf")
    result = make(client).fetch(PAGE)
    assert result.ok and result.kind == "pdf"
    assert result.text == ""
    assert any("extraction failed" in note for note in result.notes)
    assert result.notes[-1] == "reached but no text extracted"


# --- classification ----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind", "body", "expected"),
    [
        (200, "html", HTML, "ok"),
        (200, "pdf", b"%PDF-", "ok"),
        (200, "text", b"three short words", "ok"),
        (200, "html", CHALLENGE, "blocked"),
        (200, "html", EMPTY_SHELL, "blocked"),  # 0 extracted words: a bot wall
        (200, "html", SHORT_REAL_PAGE, "ok"),  # 5 words: short but real, not a wall
        (200, "html", b"", "blocked"),
        (204, "html", b"", "blocked"),
        (200, "html", b"<p>Please enable JavaScript and cookies</p>", "blocked"),
        (200, "html", b"<p>Access Denied</p>", "blocked"),
        (200, "html", b"<p>Solve the CAPTCHA</p>", "blocked"),
        (503, "html", CHALLENGE, "blocked"),
        (503, "html", b"<p>maintenance</p>", "retryable"),
        (401, "html", b"", "blocked"),
        (403, "html", b"", "blocked"),
        (406, "html", b"", "blocked"),
        (418, "html", b"", "blocked"),
        (404, "html", b"", "unreachable"),
        (410, "html", b"", "unreachable"),
        (429, "html", b"", "retryable"),
        (500, "html", b"", "retryable"),
        (502, "html", b"", "retryable"),
        (504, "html", b"", "retryable"),
        (599, "html", b"", "retryable"),
        (None, "other", b"", "unreachable"),
        (0, "other", b"", "unreachable"),
    ],
)
def test_classify(status: int | None, kind: fx.Kind, body: bytes, expected: str) -> None:
    assert fx.classify(status, kind, body) == expected


# --- the ladder --------------------------------------------------------------


@respx.mock
def test_403_escalates_to_curl_then_reports_blocked(client: httpx.Client) -> None:
    page = serve(status=403)
    gate = StubGate(allow=False)
    lookups: list[str] = []

    def wayback(url: str) -> str | None:
        lookups.append(url)
        return None

    result = make(client, gate=gate, curl=curl_returning(403), wayback=wayback).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED_NO_BROWSER
    assert not result.ok
    assert result.step == 4
    assert result.status == 403
    assert result.text == ""
    assert page.call_count == 1
    assert gate.calls == [("x.test", 403)]
    assert gate.skipped == 1 and gate.skipped_urls == 1
    assert lookups == [PAGE]
    steps = [note.split(":")[0] for note in result.notes]
    assert steps == ["step 1 httpx", "step 2 curl_cffi", "step 3 browser", "step 4 wayback"]
    assert result.notes[0] == "step 1 httpx: HTTP 403 (blocked)"
    assert result.notes[2] == "step 3 browser: not permitted"


@respx.mock
def test_blocked_without_a_gate_decision_is_plain_blocked(client: httpx.Client) -> None:
    """A gate that allowed the browser but still got blocked is BLOCKED, not NO_BROWSER."""
    serve(status=403)
    gate = StubGate(allow=True, result=(403, b"", "text/html"))
    result = make(client, gate=gate, curl=curl_returning(403)).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED
    assert gate.skipped == 0


@respx.mock
def test_curl_step_2_succeeds_after_403(client: httpx.Client) -> None:
    page = serve(status=403)
    result = make(client, curl=curl_returning(200, HTML, "text/html; charset=utf-8")).fetch(PAGE)
    assert result.ok and result.step == 2
    assert "240 patients" in result.text
    assert page.call_count == 1
    assert result.notes == ["step 1 httpx: HTTP 403 (blocked)", "step 2 curl_cffi: HTTP 200 (ok)"]


@respx.mock
def test_curl_exception_after_403_keeps_blocked_and_asks_gate(client: httpx.Client) -> None:
    """A plain OSError from an injected step 2 is a transport failure, not a verdict:
    the 403 from step 1 still decides, so the gate is asked and the outcome is a wall."""
    serve(status=403)

    def curl(url: str) -> tuple[int, bytes, str, str]:
        raise OSError("tls handshake failed")

    gate = StubGate(allow=False)
    lookups: list[str] = []

    def wayback(url: str) -> str | None:
        lookups.append(url)
        return None

    result = make(client, gate=gate, curl=curl, wayback=wayback).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED_NO_BROWSER
    assert result.status == 403
    assert result.notes[1] == "step 2 curl_cffi: OSError (transport failed)"
    assert gate.calls == [("x.test", 403)]
    assert lookups == [PAGE]


@respx.mock
def test_403_then_curl_transport_failure_still_asks_gate(client: httpx.Client) -> None:
    """The typical bot wall: HTTP 403 at step 1, then curl_cffi's TLS fails outright.
    That is still `blocked` (product rule 2: never collapsed into `unreachable`),
    so step 3 is consulted, the refusal is counted and the state names the wall."""
    serve(status=403)

    def curl(url: str) -> tuple[int, bytes, str, str]:
        raise fx.TransportFailure("RequestsError")

    gate = StubGate(allow=False)
    result = make(client, gate=gate, curl=curl).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED_NO_BROWSER
    assert not result.ok
    assert result.status == 403
    assert result.step == 4
    assert gate.calls == [("x.test", 403)]
    assert gate.skipped == 1
    assert result.notes == [
        "step 1 httpx: HTTP 403 (blocked)",
        "step 2 curl_cffi: RequestsError (transport failed)",
        "step 3 browser: not permitted",
        "step 4 wayback: no snapshot",
    ]


@respx.mock
def test_403_403_browser_crash_reports_blocked_not_unreachable(client: httpx.Client) -> None:
    """Steps 1 and 2 answered 403; the browser crashed (status 0). The last HTTP
    answer decides: BLOCKED, and the gate was allowed so it is not NO_BROWSER."""
    serve(status=403)
    gate = StubGate(allow=True, result=(0, b"", ""))
    result = make(client, gate=gate, curl=curl_returning(403)).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED
    assert result.status == 403
    assert gate.fetched == [PAGE]
    assert gate.skipped == 0
    assert result.notes[2] == "step 3 browser: no response (transport failed)"


@respx.mock
def test_transport_failure_at_step_1_is_unreachable(client: httpx.Client) -> None:
    """With no HTTP answer at all, `unreachable` is the honest word."""
    respx.get(PAGE).mock(side_effect=httpx.ConnectError("dns"))
    gate = StubGate(allow=False)
    result = make(client, gate=gate).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert gate.calls == []  # never blocked, so the browser is never in question


@respx.mock
def test_curl_unexpected_exception_propagates(client: httpx.Client) -> None:
    """A bug in a curl_get implementation must fail loudly, not read as ``unreachable``."""
    serve(status=403)

    def curl(url: str) -> tuple[int, bytes, str, str]:
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        make(client, curl=curl).fetch(PAGE)


@respx.mock
def test_challenge_page_with_200_counts_as_blocked(client: httpx.Client) -> None:
    serve(status=200, body=CHALLENGE)
    result = make(client, curl=curl_returning(200, CHALLENGE)).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED_NO_BROWSER
    assert result.notes[0] == "step 1 httpx: HTTP 200 (blocked)"


@respx.mock
def test_empty_html_200_is_blocked_and_escalates(client: httpx.Client) -> None:
    """A 2xx HTML shell with almost no text is bot protection (spec section 6.1:
    Science answered "403, 3 words"), so the ladder climbs on instead of stopping."""
    page = serve(status=200, body=EMPTY_SHELL)
    result = make(client, curl=curl_returning(200, HTML, "text/html; charset=utf-8")).fetch(PAGE)
    assert result.ok and result.step == 2
    assert "240 patients" in result.text
    assert page.call_count == 1
    assert result.notes == [
        "step 1 httpx: HTTP 200 (blocked: empty page)",
        "step 2 curl_cffi: HTTP 200 (ok)",
    ]


@respx.mock
def test_short_real_html_page_stays_ok(client: httpx.Client) -> None:
    """A genuinely short real page (redirect stub, one-line erratum, terse abstract
    page) must not be misclassified as a bot wall and lose its text (product rule
    2): 5 extracted words is `ok` at step 1, no escalation to curl_cffi."""
    serve(status=200, body=SHORT_REAL_PAGE)
    result = make(client, curl=curl_forbidden).fetch(PAGE)
    assert result.ok and result.step == 1
    assert result.words == 5
    assert "Moved" in result.text
    assert result.notes == ["step 1 httpx: HTTP 200 (ok)"]


@respx.mock
def test_empty_html_everywhere_ends_blocked_not_ok(client: httpx.Client) -> None:
    serve(status=200, body=EMPTY_SHELL)
    result = make(client, curl=curl_returning(200, EMPTY_SHELL)).fetch(PAGE)
    assert result.outcome is fx.Outcome.BLOCKED_NO_BROWSER
    assert not result.ok
    assert result.text == ""
    assert result.notes[1] == "step 2 curl_cffi: HTTP 200 (blocked: empty page)"


@respx.mock
def test_short_but_real_pdf_is_not_blocked(client: httpx.Client) -> None:
    """The empty-page rule is for HTML shells only: a three-word PDF is content."""
    serve(body=pdf_bytes("three short words"), ctype="application/pdf")
    result = make(client).fetch(PAGE)
    assert result.ok and result.step == 1
    assert result.words == 3


@respx.mock
def test_404_rescued_by_wayback_step_4(client: httpx.Client) -> None:
    serve(status=404)
    snapshot = serve(url=SNAPSHOT)
    result = make(client, wayback=lambda url: SNAPSHOT).fetch(PAGE)
    assert result.ok and result.step == 4
    assert result.url == PAGE
    assert result.final_url == SNAPSHOT
    assert "240 patients" in result.text
    assert snapshot.call_count == 1
    assert result.notes == ["step 1 httpx: HTTP 404 (unreachable)", "step 4 wayback: HTTP 200 (ok)"]


@respx.mock
def test_404_without_snapshot_is_unreachable(client: httpx.Client) -> None:
    serve(status=404)
    result = make(client).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert result.step == 4
    assert result.status == 404
    assert result.notes[-1] == "step 4 wayback: no snapshot"


@respx.mock
def test_wayback_snapshot_that_fails_keeps_the_original_failure(client: httpx.Client) -> None:
    serve(status=404)
    serve(url=SNAPSHOT, status=404)
    result = make(client, wayback=lambda url: SNAPSHOT).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert result.notes[-1] == "step 4 wayback: HTTP 404 (unreachable)"


@respx.mock
def test_connect_error_is_unreachable(client: httpx.Client) -> None:
    respx.get(PAGE).mock(side_effect=httpx.ConnectError("dns"))
    result = make(client).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert result.status is None
    assert result.notes[0] == "step 1 httpx: ConnectError (unreachable)"


@respx.mock
def test_429_backoff_then_unavailable(client: httpx.Client, clock: list[float]) -> None:
    page = serve(status=429)
    lookups: list[str] = []

    def wayback(url: str) -> str | None:
        lookups.append(url)
        return None

    result = make(client, wayback=wayback).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNAVAILABLE
    assert result.step == 1
    assert page.call_count == 3
    assert clock == [2.0, 4.0]
    assert lookups == []  # a rate limit is not a dead link; wayback is not consulted
    assert result.notes == ["step 1 httpx: HTTP 429 (retryable, gave up after 3 attempts)"]


@respx.mock
def test_retry_after_is_honoured(client: httpx.Client, clock: list[float]) -> None:
    page = respx.get(PAGE)
    page.side_effect = [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(200, content=HTML, headers={"content-type": "text/html"}),
    ]
    result = make(client).fetch(PAGE)
    assert result.ok and result.step == 1
    assert page.call_count == 2
    assert clock == [7.0]


@respx.mock
def test_retry_after_above_the_cap_stops_immediately(
    client: httpx.Client, clock: list[float]
) -> None:
    page = respx.get(PAGE).mock(return_value=httpx.Response(429, headers={"Retry-After": "3600"}))
    result = make(client).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNAVAILABLE
    assert page.call_count == 1
    assert clock == []


@respx.mock
def test_step_1_sends_accept_header_without_mailto(client: httpx.Client) -> None:
    page = serve()
    make(client).fetch(PAGE)
    request = page.calls.last.request
    assert request.headers["accept"] == fx.ACCEPT
    assert "mailto" not in str(request.url)


# --- robots ------------------------------------------------------------------


@respx.mock
def test_robots_disallow_is_reported(client: httpx.Client) -> None:
    robots = respx.get("https://x.test/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
    )
    private = respx.get(url__regex=r"https://x\.test/private/.*").mock(
        return_value=httpx.Response(200)
    )
    public = serve(url="https://x.test/public")
    fetcher = make(client, respect_robots=True)
    for path in ("a", "b"):
        result = fetcher.fetch(f"https://x.test/private/{path}")
        assert result.outcome is fx.Outcome.BLOCKED_ROBOTS
        assert not result.ok
        assert "robots.txt" in result.notes[0]
    assert private.call_count == 0
    assert robots.call_count == 1
    assert fetcher.fetch("https://x.test/public").ok
    assert public.call_count == 1
    assert robots.call_count == 1


@respx.mock
def test_robots_missing_allows(client: httpx.Client) -> None:
    respx.get("https://x.test/robots.txt").mock(return_value=httpx.Response(404))
    serve()
    assert make(client, respect_robots=True).fetch(PAGE).ok


@respx.mock
def test_robots_fetch_failure_allows(client: httpx.Client) -> None:
    robots = respx.get("https://x.test/robots.txt").mock(side_effect=httpx.ReadTimeout("slow"))
    serve()
    serve(url="https://x.test/other")
    fetcher = make(client, respect_robots=True)
    assert fetcher.fetch(PAGE).ok
    assert fetcher.fetch("https://x.test/other").ok
    assert robots.call_count == 1  # the failure is memoised, not retried per URL


@respx.mock
def test_robots_ignored_when_config_disables_it(client: httpx.Client) -> None:
    robots = respx.get("https://x.test/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n")
    )
    serve()
    assert make(client, respect_robots=False).fetch(PAGE).ok
    assert robots.call_count == 0


# --- permissions and cache ---------------------------------------------------


@respx.mock
def test_network_deny_makes_no_request(client: httpx.Client) -> None:
    serve()
    result = make(client, network="deny", curl=curl_forbidden).fetch(PAGE)
    assert result.outcome is fx.Outcome.NETWORK_DENIED
    assert result.step == 0
    assert respx.calls.call_count == 0
    assert result.notes == ["network: not permitted (permissions.network = deny)"]


@respx.mock
def test_network_ask_without_a_tty_is_denied_and_reported(client: httpx.Client) -> None:
    """Product rule 4: ``ask`` with no terminal is ``deny``, and the report says why."""
    serve()
    fetcher = make(client, network="ask", interactive=False, curl=curl_forbidden)
    assert fetcher.network_allowed is False
    result = fetcher.fetch(PAGE)
    assert result.outcome is fx.Outcome.NETWORK_DENIED
    assert result.step == 0
    assert respx.calls.call_count == 0
    assert result.notes == [
        "network: not permitted (permission is set to ask but there is no interactive terminal)"
    ]


@respx.mock
def test_network_ask_with_a_tty_is_allowed_and_noted(client: httpx.Client) -> None:
    serve()
    fetcher = make(client, network="ask", interactive=True)
    assert fetcher.network_allowed is True
    result = fetcher.fetch(PAGE)
    assert result.ok and result.step == 1
    assert result.notes == [
        "network permission is 'ask'; allowed for this run",
        "step 1 httpx: HTTP 200 (ok)",
    ]


def test_network_allowed_is_derived_from_config_once(client: httpx.Client) -> None:
    assert make(client, network="allow").network_allowed is True
    assert make(client, network="deny").network_allowed is False


@respx.mock
def test_cache_hit_makes_no_request(client: httpx.Client, tmp_path: Path) -> None:
    serve()
    with Cache(tmp_path / "c.sqlite3") as cache:
        cache.add_source(
            f"url:{PAGE}",
            scheme="web",
            title="t",
            url=PAGE,
            text_kind="fulltext",
            raw_text="cached words",
        )
        result = make(client, cache=cache).fetch(PAGE)
    assert result.ok and result.from_cache
    assert result.step == 0
    assert result.text == "cached words"
    assert result.body == b""
    assert result.kind == "text"
    assert respx.calls.call_count == 0


@respx.mock
def test_success_is_written_to_cache(client: httpx.Client, tmp_path: Path) -> None:
    serve()
    with Cache(tmp_path / "c.sqlite3") as cache:
        result = make(client, cache=cache).fetch(PAGE, text_kind="fulltext")
        assert result.ok and not result.from_cache
        stored = cache.get_raw_text(f"url:{PAGE}")
        assert stored == result.text
        detail = cache.detail(f"url:{PAGE}")
    assert detail is not None
    assert detail.summary.title == "Evidence & co"
    assert detail.summary.scheme == "web"
    assert detail.summary.text_kind == "fulltext"


@respx.mock
def test_failures_and_empty_text_are_not_cached(client: httpx.Client, tmp_path: Path) -> None:
    serve(status=404)
    with Cache(tmp_path / "c.sqlite3") as cache:
        make(client, cache=cache).fetch(PAGE)
        assert cache.get_raw_text(f"url:{PAGE}") is None


# --- browser gate ------------------------------------------------------------


@respx.mock
def test_gate_allow_uses_browser_step_3(client: httpx.Client) -> None:
    serve(status=403)
    gate = StubGate(allow=True)
    result = make(client, gate=gate, curl=curl_returning(403)).fetch(PAGE)
    assert result.ok and result.step == 3
    assert "240 patients" in result.text
    assert gate.calls == [("x.test", 403)]
    assert gate.fetched == [PAGE]
    assert result.notes[2] == "step 3 browser: HTTP 200 (ok)"


@respx.mock
def test_browser_transport_failure_keeps_the_last_http_answer(client: httpx.Client) -> None:
    serve(status=404)
    # Step 1 answered 404, so the ladder never reaches the browser; a browser
    # crash after 403/403 is covered by test_403_403_browser_crash_reports_blocked_not_unreachable.
    gate = StubGate(allow=True, result=(0, b"", ""))
    result = make(client, gate=gate).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert gate.calls == []


def test_denying_gate_counts_every_refusal() -> None:
    gate = fx.DenyingGate()
    assert gate.allow("x.test", 403) is False
    assert gate.allow("y.test", 403) is False
    assert gate.skipped_urls == 2
    assert gate.skipped == 0  # sources are counted by whoever knows what a source is
    with pytest.raises(RuntimeError):
        gate.fetch(PAGE)


@respx.mock
def test_skipped_counts_sources_not_urls(client: httpx.Client) -> None:
    """``skipped`` is what the report prints: one per source the browser could have
    rescued. A bare URL is one source; a location fetched on behalf of a DOI
    (``counts_as_source=False``) is counted by the open-access chain instead."""
    serve(url="https://x.test/a", status=403)
    serve(url="https://x.test/b", status=403)
    serve(url="https://x.test/c", status=403)
    gate = fx.DenyingGate()
    fetcher = make(client, gate=gate, curl=curl_returning(403))
    fetcher.fetch("https://x.test/a")
    fetcher.fetch("https://x.test/b")
    assert (gate.skipped, gate.skipped_urls) == (2, 2)
    fetcher.fetch("https://x.test/c", counts_as_source=False)
    assert (gate.skipped, gate.skipped_urls) == (2, 3)
    assert fetcher.summary().browser_skipped == 2


@respx.mock
def test_summary_counts_outcomes_and_browser_skips(client: httpx.Client) -> None:
    serve(url="https://x.test/ok")
    serve(url="https://x.test/blocked", status=403)
    serve(url="https://x.test/gone", status=404)
    gate = StubGate(allow=False)
    fetcher = make(client, gate=gate, curl=curl_returning(403))
    for path in ("ok", "blocked", "gone"):
        fetcher.fetch(f"https://x.test/{path}")
    stats = fetcher.summary()
    assert stats.counts == {
        fx.Outcome.OK: 1,
        fx.Outcome.BLOCKED_NO_BROWSER: 1,
        fx.Outcome.UNREACHABLE: 1,
    }
    assert stats.browser_skipped == 1


# --- wayback and curl defaults -----------------------------------------------


def test_wayback_snapshot_url_rewrites_to_raw_form() -> None:
    assert fx.wayback_snapshot_url(WAYBACK_PAYLOAD) == (
        "http://web.archive.org/web/20130919044612id_/http://example.com/"
    )
    assert fx.wayback_snapshot_url({"archived_snapshots": {}}) is None
    unavailable = {"archived_snapshots": {"closest": {"available": False, "url": "x"}}}
    assert fx.wayback_snapshot_url(unavailable) is None
    assert fx.wayback_snapshot_url({}) is None


@respx.mock
def test_default_wayback_lookup_queries_the_availability_api(client: httpx.Client) -> None:
    serve(status=404)
    available = respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(
            200,
            json={
                "archived_snapshots": {
                    "closest": {
                        "available": True,
                        "url": "https://web.archive.org/web/20240101000000/https://x.test/paper.html",
                    }
                }
            },
        )
    )
    snapshot = serve(url=SNAPSHOT)
    result = make(client, wayback=None).fetch(PAGE)
    assert result.ok and result.step == 4
    assert result.final_url == SNAPSHOT
    assert available.calls.last.request.url.params["url"] == PAGE
    assert snapshot.call_count == 1


@respx.mock
def test_default_wayback_lookup_failure_is_reported_not_hidden_as_no_snapshot(
    client: httpx.Client,
) -> None:
    serve(status=404)
    respx.get("https://archive.org/wayback/available").mock(return_value=httpx.Response(503))
    result = make(client, wayback=None).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert result.notes[-1] == "step 4 wayback: lookup failed (HTTP 503)"


@respx.mock
def test_default_wayback_lookup_with_no_snapshot_available_is_a_plain_miss(
    client: httpx.Client,
) -> None:
    """A well-formed response with no snapshot is a miss, not a failure."""
    serve(status=404)
    respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json={"archived_snapshots": {}})
    )
    result = make(client, wayback=None).fetch(PAGE)
    assert result.outcome is fx.Outcome.UNREACHABLE
    assert result.notes[-1] == "step 4 wayback: no snapshot"


def test_default_curl_get_impersonates_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    import curl_cffi.requests

    seen: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.content = b"<p>hi</p>"
            self.headers = {"content-type": "text/html"}
            self.url = "https://x.test/final"

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        seen["url"] = url
        seen.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(curl_cffi.requests, "get", fake_get)
    status, body, ctype, final_url = fx.default_curl_get(PAGE, timeout=5.0, contact_email="a@b.c")
    assert (status, body, ctype, final_url) == (
        200,
        b"<p>hi</p>",
        "text/html",
        "https://x.test/final",
    )
    assert seen["url"] == PAGE
    assert seen["impersonate"] == "chrome"
    assert seen["allow_redirects"] is True
    assert seen["timeout"] == 5.0
    assert seen["headers"] == {"User-Agent": user_agent("a@b.c"), "Accept": fx.ACCEPT}


def test_default_curl_get_translates_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import curl_cffi.requests
    from curl_cffi.requests.exceptions import RequestException

    def fake_get(url: str, **kwargs: Any) -> Any:
        raise RequestException("tls handshake failed")

    monkeypatch.setattr(curl_cffi.requests, "get", fake_get)
    with pytest.raises(fx.TransportFailure, match="RequestException"):
        fx.default_curl_get(PAGE)


@respx.mock
def test_default_curl_step_is_monkeypatchable(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(status=403)
    calls: list[str] = []

    def fake(url: str, **kwargs: Any) -> tuple[int, bytes, str, str]:
        calls.append(url)
        return 200, HTML, "text/html", url

    monkeypatch.setattr(fx, "default_curl_get", fake)
    result = make(client, curl=None).fetch(PAGE)
    assert result.ok and result.step == 2
    assert calls == [PAGE]


def test_fetcher_owns_and_closes_its_default_client() -> None:
    fetcher = fx.Fetcher(config=Config())
    fetcher.close()
    assert fetcher._polite.client.is_closed


@respx.mock
def test_throttle_spaces_requests_to_one_host(client: httpx.Client, clock: list[float]) -> None:
    serve(url="https://x.test/a")
    serve(url="https://x.test/b")
    fetcher = make(client)
    fetcher.fetch("https://x.test/a")
    fetcher.fetch("https://x.test/b")
    assert clock == [pytest.approx(DEFAULT_MIN_INTERVAL)]
