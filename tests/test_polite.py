"""Polite HTTP core (spec section 7): descriptive UA, per-host throttling, backoff."""

from __future__ import annotations

import httpx
import pytest
import respx

from proofpath import polite as pl


def test_user_agent_without_email() -> None:
    agent = pl.user_agent("")
    assert agent == f"proofpath/{pl.__version__} ({pl.REPO_URL})"


def test_user_agent_with_email() -> None:
    agent = pl.user_agent("a@b.c")
    assert agent == f"proofpath/{pl.__version__} ({pl.REPO_URL}; mailto:a@b.c)"


def test_default_client_sends_the_descriptive_user_agent() -> None:
    client = pl.PoliteClient(contact_email="a@b.c")
    assert client.client.headers["user-agent"] == pl.user_agent("a@b.c")


def test_email_property_exposes_the_contact_address() -> None:
    assert pl.PoliteClient(contact_email="a@b.c").email == "a@b.c"


def test_throttle_does_not_sleep_on_the_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(pl.time, "sleep", lambda s: slept.append(s))
    pl.PoliteClient().throttle("https://api.crossref.org/works")
    assert slept == []


def test_throttle_sleeps_the_remaining_interval_for_a_known_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    slept: list[float] = []
    monkeypatch.setattr(pl.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(pl.time, "sleep", lambda s: slept.append(s))
    client = pl.PoliteClient()
    client.throttle("https://export.arxiv.org/api/query")
    client.throttle("https://export.arxiv.org/api/query")  # same instant
    assert slept == [pytest.approx(pl.MIN_INTERVAL["export.arxiv.org"])]


def test_throttle_uses_the_default_interval_for_an_unknown_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    slept: list[float] = []
    monkeypatch.setattr(pl.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(pl.time, "sleep", lambda s: slept.append(s))
    client = pl.PoliteClient()
    client.throttle("https://example.org/x")
    client.throttle("https://example.org/x")  # same instant
    assert slept == [pytest.approx(pl.DEFAULT_MIN_INTERVAL)]


def test_backoff_delay_defaults_to_exponential_backoff() -> None:
    assert pl.backoff_delay(0, 503, None) == 0.5
    assert pl.backoff_delay(2, 503, None) == 2.0


def test_backoff_delay_honours_a_numeric_retry_after_header() -> None:
    assert pl.backoff_delay(0, 503, "7") == 7.0


def test_backoff_delay_returns_none_above_the_cap() -> None:
    assert pl.backoff_delay(0, 429, "3600") is None


def test_backoff_delay_enforces_a_minimum_for_429_without_retry_after() -> None:
    assert pl.backoff_delay(0, 429, None) == 2.0
    assert pl.backoff_delay(1, 429, None) == 4.0


@respx.mock
def test_get_adds_mailto_only_when_an_email_is_set_and_requested() -> None:
    route = respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(200))
    pl.PoliteClient(contact_email="a@b.c").get("https://api.crossref.org/works")
    assert "mailto=a%40b.c" in str(route.calls.last.request.url)


@respx.mock
def test_get_omits_mailto_when_requested_off() -> None:
    route = respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(200))
    pl.PoliteClient(contact_email="a@b.c").get("https://api.crossref.org/works", mailto=False)
    assert "mailto" not in str(route.calls.last.request.url)


@respx.mock
def test_get_omits_mailto_when_no_email_is_set() -> None:
    route = respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(200))
    pl.PoliteClient().get("https://api.crossref.org/works")
    assert "mailto" not in str(route.calls.last.request.url)


@respx.mock
def test_404_is_returned_not_raised() -> None:
    respx.get("https://api.crossref.org/works/x").mock(return_value=httpx.Response(404))
    response = pl.PoliteClient().get("https://api.crossref.org/works/x")
    assert response.status_code == 404


@respx.mock
def test_403_raises_without_retry() -> None:
    route = respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(403))
    with pytest.raises(pl.ProviderError, match="HTTP 403"):
        pl.PoliteClient().get("https://api.crossref.org/works")
    assert route.call_count == 1


@respx.mock
def test_503_then_200_succeeds_on_the_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(pl.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(pl.time, "sleep", fake_sleep)
    route = respx.get("https://api.crossref.org/works")
    route.side_effect = [httpx.Response(503), httpx.Response(200)]
    response = pl.PoliteClient(retries=1).get("https://api.crossref.org/works")
    assert response.status_code == 200
    assert route.call_count == 2
    assert slept == [0.5]


@respx.mock
def test_429_with_large_retry_after_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(pl.time, "sleep", lambda s: slept.append(s))
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "3600"})
    )
    with pytest.raises(pl.ProviderError, match="retry after"):
        pl.PoliteClient().get("https://api.crossref.org/works")
    assert slept == []


@respx.mock
def test_three_consecutive_500s_raise_after_retries_plus_one_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)
    route = respx.get("https://api.crossref.org/works").mock(return_value=httpx.Response(500))
    with pytest.raises(pl.ProviderError, match="HTTP 500"):
        pl.PoliteClient(retries=2).get("https://api.crossref.org/works")
    assert route.call_count == 3


@respx.mock
def test_connect_error_is_retried_and_surfaces_as_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)
    respx.get("https://api.crossref.org/works").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(pl.ProviderError, match="ConnectError"):
        pl.PoliteClient(retries=1).get("https://api.crossref.org/works")
