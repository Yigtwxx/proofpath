"""The local fallback judge (spec section 11, Task 11 and its Amendment B).

Groq's free tier runs out; the run then switches, at once, to qwen3.5:9b in the
user's own Ollama (or the model ``judge.fallback`` names), and says so. Every failure mode the user
has seen in a fallback like this before has a test here: a primary that still retries or waits, a
lost batch, a switch that races across threads, a notice that only comes after the
model has loaded, thinking output that breaks the JSON, and a new run that stays on
the local model. Nothing here opens a socket: every endpoint is a respx route.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from proofpath import judge
from proofpath import verify as verify_mod
from proofpath.cli import app
from proofpath.config import Config, JudgeConfig
from proofpath.events import Event, Note
from proofpath.models import Label
from proofpath.report import render_footer, render_markdown
from proofpath.secrets import ApiKey
from tests.test_verify_search import CLAIM, StubSearcher, searching

GROQ = JudgeConfig()
KEY = ApiKey("gsk_fallback_test_secret", source="test")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAGS_URL = "http://localhost:11434/api/tags"
WARM_URL = "http://localhost:11434/api/generate"
LOCAL_URL = "http://localhost:11434/v1/chat/completions"
LOCAL = "qwen3.5:9b"  # the default ``judge.fallback``
# The user's own `ollama list` on 2026-09-28, after the clean-up, in Ollama's order.
INSTALLED = [
    "qwen3-embedding:8b",
    "nomic-embed-text:latest",
    "qwen3-embedding:0.6b",
    "gemma4:12b",
    "qwen3.5:9b",
    "qwen3.6:35b-a3b",
    "qwen2.5-coder:7b-instruct-q4_K_M",
]
MESSAGES = [
    {"role": "system", "content": "Answer only from the passage."},
    {"role": "user", "content": "- id: c1\n  claim: x\n  passage: y\n  verdict: NEI (low)"},
]
SCHEMA = {"type": "object", "properties": {"opinions": {"type": "array"}}}
OPINION = json.dumps({"opinions": [{"id": "c1", "label": "NEI", "rationale": "silent"}]})
LIMIT = f"Groq limit reached — judging with local ollama {LOCAL}"


@pytest.fixture
def mock() -> Iterator[respx.MockRouter]:
    """Every endpoint as a route; a route a test does not reach is not an error."""
    with respx.mock(assert_all_called=False) as router:
        yield router


def _tags(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": name} for name in names]})


def _ok(content: str, model: str = LOCAL) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        },
    )


class Sleeps:
    """The client's ``sleep``, recorded: a fallback must never be reached by waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def fallback(
    *,
    setting: str = LOCAL,
    sleep: Sleeps | None = None,
    on_switch: Callable[[str], None] | None = None,
) -> judge.FallbackClient:
    """A Groq primary behind the fallback, exactly as ``build_client`` wires it."""
    primary = judge.JudgeClient(GROQ, KEY, sleep=sleep or Sleeps())
    return judge.FallbackClient(primary, fallback=setting, on_switch=on_switch)


def routes(
    mock: respx.MockRouter,
    *,
    groq: httpx.Response | Exception | list[httpx.Response] | None = None,
    tags: httpx.Response | Exception | None = None,
    local: httpx.Response | None = None,
) -> tuple[respx.Route, respx.Route, respx.Route, respx.Route]:
    """Groq, ``/api/tags``, the warm-up and the local chat route, in that order."""
    groq_route = mock.post(GROQ_URL)
    if isinstance(groq, (list, Exception)):
        groq_route.mock(side_effect=groq)
    else:
        groq_route.mock(return_value=groq or httpx.Response(429))
    tags_route = mock.get(TAGS_URL)
    if isinstance(tags, Exception):
        tags_route.mock(side_effect=tags)
    else:
        tags_route.mock(return_value=tags or _tags(*INSTALLED))
    warm = mock.post(WARM_URL).mock(return_value=httpx.Response(200, json={"done": True}))
    local_route = mock.post(LOCAL_URL).mock(return_value=local or _ok(OPINION))
    return groq_route, tags_route, warm, local_route


# --- which model ------------------------------------------------------------------


def test_the_default_fallback_is_qwen3_5() -> None:
    """The user's choice (2026-09-28): Groq, then qwen3.5:9b, and nothing else."""
    assert JudgeConfig().fallback == "qwen3.5:9b" == judge.LOCAL_JUDGE_MODEL


def test_the_default_config_falls_back_to_qwen3_5(mock: respx.MockRouter) -> None:
    _, tags, _, local = routes(mock)
    client = judge.build_client(GROQ, KEY)
    client.complete(MESSAGES)
    assert tags.call_count == 1  # only to confirm it is installed
    assert json.loads(local.calls.last.request.content)["model"] == "qwen3.5:9b"
    assert client.switched == "Groq limit reached — judging with local ollama qwen3.5:9b"
    client.close()


def test_the_local_model_is_always_asked_not_to_think(mock: respx.MockRouter) -> None:
    _, _, _, local = routes(mock)
    fallback().complete(MESSAGES, reasoning_effort="high")
    assert json.loads(local.calls.last.request.content)["reasoning_effort"] == "none"


def test_a_local_model_that_refuses_the_effort_is_still_downgraded(
    mock: respx.MockRouter,
) -> None:
    """The 400 safety stays: a model that names ``reasoning_effort`` in a 400 is asked
    again without it, and that retry is not a failure."""
    _, _, _, local = routes(mock)
    local.mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "invalid reasoning_effort"}}),
            _ok(OPINION),
        ]
    )
    assert fallback().complete(MESSAGES).text == OPINION
    assert "reasoning_effort" not in json.loads(local.calls.last.request.content)


def test_installed_models_reads_the_tags_from_the_root_not_from_v1(mock: respx.MockRouter) -> None:
    route = mock.get(TAGS_URL).mock(return_value=_tags("qwen3.5:9b", "gemma4:12b"))
    names = judge.installed_models("http://localhost:11434/v1", timeout=2.0)
    assert names == ["qwen3.5:9b", "gemma4:12b"]
    assert route.call_count == 1


@pytest.mark.parametrize(
    "answer",
    [httpx.ConnectError("refused"), httpx.Response(500), httpx.Response(200, text="not json")],
)
def test_installed_models_is_empty_on_any_error_and_never_raises(
    mock: respx.MockRouter,
    answer: httpx.Response | Exception,
) -> None:
    route = mock.get(TAGS_URL)
    if isinstance(answer, Exception):
        route.mock(side_effect=answer)
    else:
        route.mock(return_value=answer)
    assert judge.installed_models("http://localhost:11434/v1", timeout=2.0) == []


# --- the switch -------------------------------------------------------------------


def test_a_429_switches_at_once_and_the_opinion_names_the_local_model(
    mock: respx.MockRouter,
) -> None:
    """Amendment B 1, 2 and 4: one Groq call, no sleep, the batch answered locally."""
    sleep = Sleeps()
    seen: list[str] = []
    groq, _, warm, local = routes(mock, groq=httpx.Response(429, headers={"Retry-After": "60"}))
    client = fallback(sleep=sleep, on_switch=seen.append)
    opinions = judge.Judge(client).review([judge.JudgeItem("c1", "x", "y", Label.NEI, "low")])

    assert opinions["c1"].model == f"ollama {LOCAL}"
    assert client.switched == LIMIT
    assert seen == [LIMIT]
    assert groq.call_count == 1  # no retry
    assert sleep.calls == []  # and no Retry-After wait
    assert warm.call_count == 1
    assert local.call_count == 1
    assert (client.provider, client.model) == ("ollama", LOCAL)


def test_the_failed_request_is_sent_again_to_the_local_model_whole(mock: respx.MockRouter) -> None:
    """Amendment B 2: the caller sees one completion, for the very request that failed."""
    groq, _, _, local = routes(mock)
    client = fallback()
    done = client.complete(MESSAGES, json_schema=SCHEMA, max_tokens=4096, reasoning_effort="low")

    assert done.text == OPINION
    first = json.loads(groq.calls.last.request.content)
    again = json.loads(local.calls.last.request.content)
    assert again["messages"] == first["messages"] == MESSAGES
    assert again["max_tokens"] == 4096
    assert again["model"] == LOCAL
    assert again["response_format"]["json_schema"]["schema"] == SCHEMA


def test_after_the_switch_groq_is_never_asked_again(mock: respx.MockRouter) -> None:
    groq, _, warm, local = routes(mock)
    client = fallback()
    client.complete(MESSAGES)
    client.complete(MESSAGES)
    client.complete(MESSAGES)

    assert groq.call_count == 1
    assert local.call_count == 3
    assert warm.call_count == 1  # loaded once, kept warm after


@pytest.mark.parametrize(
    ("failure", "notice"),
    [
        (httpx.Response(413), LIMIT),
        (
            httpx.Response(401),
            f"Groq rejected the key (HTTP 401) — judging with local ollama {LOCAL}",
        ),
        (
            httpx.Response(403),
            f"Groq rejected the key (HTTP 403) — judging with local ollama {LOCAL}",
        ),
        (
            httpx.Response(500),
            f"Groq did not answer (HTTP 500) — judging with local ollama {LOCAL}",
        ),
        (
            httpx.Response(503),
            f"Groq did not answer (HTTP 503) — judging with local ollama {LOCAL}",
        ),
        (
            httpx.ReadTimeout("slow"),
            f"Groq did not answer (timeout) — judging with local ollama {LOCAL}",
        ),
        (
            httpx.ConnectError("down"),
            f"Groq did not answer (connection failed) — judging with local ollama {LOCAL}",
        ),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
            ),
            f"Groq did not answer (empty answer) — judging with local ollama {LOCAL}",
        ),
    ],
)
def test_the_notice_names_the_real_cause_of_the_switch(
    mock: respx.MockRouter,
    failure: httpx.Response | Exception,
    notice: str,
) -> None:
    """Fix round 1 B: every failure switches, and the notice says which one it was --
    never a body, never the key."""
    sleep = Sleeps()
    groq, _, _, local = routes(mock, groq=failure)
    client = fallback(sleep=sleep)
    client.complete(MESSAGES)

    assert client.switched == notice
    assert KEY.value not in notice
    assert groq.call_count == 1
    assert local.call_count == 1
    assert sleep.calls == []


def test_a_hanging_groq_switches_on_a_short_connect_timeout(mock: respx.MockRouter) -> None:
    """Fix round 1 D3: with a fallback, the primary gets five seconds to connect."""
    groq, _, _, local = routes(mock, groq=httpx.ConnectTimeout("no route"))
    client = fallback()
    client.complete(MESSAGES)

    timeout = groq.calls.last.request.extensions["timeout"]
    assert timeout["connect"] == judge.PRIMARY_CONNECT_TIMEOUT == 5.0
    assert timeout["read"] == 60.0
    assert client.switched == f"Groq did not answer (timeout) — judging with local ollama {LOCAL}"
    assert local.call_count == 1


def test_without_a_fallback_the_primary_keeps_its_own_timeout(mock: respx.MockRouter) -> None:
    groq, _, _, _ = routes(mock, groq=httpx.Response(401))
    client = fallback(setting="off")
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    assert groq.calls.last.request.extensions["timeout"]["connect"] == 60.0


def test_a_listener_that_raises_does_not_lose_the_batch(mock: respx.MockRouter) -> None:
    """Fix round 1 D1: the notice is best effort; the answer is not."""

    def broken(text: str) -> None:
        raise RuntimeError("listener bug")

    _, _, _, local = routes(mock)
    client = fallback(on_switch=broken)
    assert client.complete(MESSAGES).text == OPINION
    assert client.switched == LIMIT
    assert local.call_count == 1


def test_a_local_model_that_fails_is_not_asked_again_this_run(mock: respx.MockRouter) -> None:
    """Fix round 1 D4: a dead Ollama costs one wait, not one per batch."""
    groq, _, _, local = routes(mock, local=httpx.Response(500))
    client = fallback(sleep=Sleeps())
    with pytest.raises(judge.JudgeUnavailable) as first:
        client.complete(MESSAGES)
    asked = local.call_count
    with pytest.raises(judge.JudgeUnavailable) as second:
        client.complete(MESSAGES)

    assert str(second.value) == str(first.value)
    assert str(first.value).startswith(f"HTTP 429 from {GROQ_URL}; local ollama {LOCAL}: ")
    assert local.call_count == asked  # the second call asked nobody
    assert groq.call_count == 1


def test_a_local_timeout_is_not_retried(mock: respx.MockRouter) -> None:
    """Fix round 1 D4: one 180 s wait at most, not three."""
    _, _, _, local = routes(mock)
    local.mock(side_effect=httpx.ReadTimeout("loading"))
    client = fallback(sleep=Sleeps())
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)
    assert local.call_count == 1
    assert "request failed: ReadTimeout" in str(caught.value)


def test_reset_forgets_a_dead_local_model(mock: respx.MockRouter) -> None:
    groq, _, _, local = routes(mock, groq=[httpx.Response(429), httpx.Response(429)])
    local.mock(side_effect=[httpx.Response(404), _ok(OPINION)])
    client = fallback(sleep=Sleeps())
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    client.reset()
    assert client.complete(MESSAGES).text == OPINION
    assert groq.call_count == 2


def test_one_url_for_tags_warm_up_and_chat(mock: respx.MockRouter) -> None:
    """Fix round 1 D5: ``ollama_url`` reaches the local client too."""
    other = "http://ollama.test:1234"
    tags = mock.get(f"{other}/api/tags").mock(return_value=_tags(LOCAL))
    warm = mock.post(f"{other}/api/generate").mock(return_value=httpx.Response(200, json={}))
    chat = mock.post(f"{other}/v1/chat/completions").mock(return_value=_ok(OPINION))
    mock.post(GROQ_URL).mock(return_value=httpx.Response(429))
    primary = judge.JudgeClient(GROQ, KEY, sleep=Sleeps())
    client = judge.FallbackClient(primary, ollama_url=f"{other}/v1")
    client.complete(MESSAGES)
    assert (tags.call_count, warm.call_count, chat.call_count) == (1, 1, 1)


@pytest.mark.parametrize("setting", ["off", "Off", "OFF", " off "])
def test_off_turns_the_fallback_off_in_any_case(mock: respx.MockRouter, setting: str) -> None:
    """Fix round 1 D6: the keyword in any case; Ollama is never asked."""
    _, tags, _, local = routes(mock)
    client = fallback(setting=setting, sleep=Sleeps())
    with pytest.raises(judge.JudgeUnavailable, match=r"judge\.fallback is off"):
        client.complete(MESSAGES)
    assert tags.call_count == local.call_count == 0


def test_the_notice_fires_before_the_model_is_loaded_or_asked(mock: respx.MockRouter) -> None:
    """Amendment B 4: the user reads the line while the model loads, not after."""
    _, _, warm, local = routes(mock)
    counts: list[tuple[int, int]] = []
    client = fallback(on_switch=lambda text: counts.append((warm.call_count, local.call_count)))
    client.complete(MESSAGES)

    assert counts == [(0, 0)]


def test_the_local_request_turns_thinking_off_and_the_warm_up_keeps_the_model_loaded(
    mock: respx.MockRouter,
) -> None:
    """Amendment B 5 and 6, as Ollama 0.34 honours them: ``reasoning_effort: none`` on
    the OpenAI route, and ``keep_alive`` on the native one (the OpenAI route ignores
    it, measured 2026-09-28)."""
    _, _, warm, local = routes(mock)
    fallback().complete(MESSAGES, reasoning_effort="low")

    sent = json.loads(local.calls.last.request.content)
    assert sent["reasoning_effort"] == "none"
    loaded = json.loads(warm.calls.last.request.content)
    assert loaded == {"model": LOCAL, "keep_alive": judge.LOCAL_KEEP_ALIVE}
    assert "authorization" not in local.calls.last.request.headers
    assert "authorization" not in warm.calls.last.request.headers


def test_the_local_client_waits_long_enough_for_a_cold_load() -> None:
    client = judge.local_client(LOCAL)
    assert client.timeout == judge.LOCAL_TIMEOUT == 180.0
    assert (client.provider, client.model) == ("ollama", LOCAL)
    client.close()


def test_a_warm_up_that_fails_does_not_cost_the_answer(mock: respx.MockRouter) -> None:
    _, _, warm, local = routes(mock)
    warm.mock(side_effect=httpx.ConnectError("slow"))
    done = fallback().complete(MESSAGES)
    assert done.text == OPINION
    assert local.call_count == 1


def test_a_leading_think_block_never_reaches_the_json_reader(mock: respx.MockRouter) -> None:
    """Amendment B 5: a qwen that thinks out loud, braces and all, still parses."""
    answer = json.dumps(
        {"items": [{"id": 0, "english": "OpenAI went bankrupt.", "queries": ["OpenAI bankrupt"]}]}
    )
    routes(
        mock, local=_ok(f"<think>\nThe user wants {{json}}. Let me think.\n</think>\n\n{answer}")
    )
    written = judge.Judge(fallback()).queries(["OpenAI battı."])  # noqa: RUF001 - Turkish, verbatim

    assert written == {
        0: judge.JudgedClaim(queries=("OpenAI bankrupt",), english="OpenAI went bankrupt.")
    }


def test_an_orphan_closing_think_tag_is_stripped_too(mock: respx.MockRouter) -> None:
    """Fix round 1 D2: some templates open the block in the prompt, so the answer
    carries only its end -- with a brace in the reasoning to trip the JSON reader."""
    answer = json.dumps({"items": [{"id": 0, "english": "Paris is big.", "queries": ["Paris"]}]})
    routes(mock, local=_ok(f"The user wants {{items}} in JSON, fine.\n</think>\n{answer}"))
    written = judge.Judge(fallback()).queries(["Paris is big."])
    assert written == {0: judge.JudgedClaim(queries=("Paris",), english="Paris is big.")}


def test_two_threads_against_a_rate_limited_primary_switch_once(mock: respx.MockRouter) -> None:
    """Amendment B 3: one switch, one notice, at most one Groq call per thread."""
    barrier = threading.Barrier(2, timeout=5)

    def limited(request: httpx.Request) -> httpx.Response:
        barrier.wait()  # both threads are inside the primary before either switches
        return httpx.Response(429)

    groq, _, warm, local = routes(mock)
    groq.mock(side_effect=limited)
    seen: list[str] = []
    client = fallback(on_switch=seen.append)
    done: list[str] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            done.append(client.complete(MESSAGES).text)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert done == [OPINION, OPINION]
    assert seen == [LIMIT]
    assert groq.call_count == 2
    assert warm.call_count == 1
    assert local.call_count == 2


def test_calls_after_a_threaded_switch_go_straight_to_the_local_model(
    mock: respx.MockRouter,
) -> None:
    groq, _, _, local = routes(mock)
    client = fallback()
    client.complete(MESSAGES)
    threads = [threading.Thread(target=client.complete, args=(MESSAGES,)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert groq.call_count == 1
    assert local.call_count == 5


# --- no fallback --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "why"),
    [
        (httpx.ConnectError("refused"), "ollama is not reachable at http://localhost:11434"),
        (_tags("gemma4:12b", "qwen3-embedding:8b"), "qwen3.5:9b is not installed"),
        (_tags(), "qwen3.5:9b is not installed"),
    ],
)
def test_without_a_local_model_the_original_error_says_why(
    mock: respx.MockRouter, tags: httpx.Response | Exception, why: str
) -> None:
    """No fallback means today's behaviour: the retries, then ``JudgeUnavailable``."""
    sleep = Sleeps()
    groq, _, warm, local = routes(mock, tags=tags)
    client = fallback(sleep=sleep)
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)

    assert str(caught.value) == f"HTTP 429 from {GROQ_URL}; no local fallback: {why}"
    assert groq.call_count == 3  # the normal retrying client
    assert len(sleep.calls) == 2
    assert client.switched is None
    assert warm.call_count == local.call_count == 0


def test_fallback_off_never_asks_ollama_and_keeps_the_retries(mock: respx.MockRouter) -> None:
    sleep = Sleeps()
    groq, tags, _, local = routes(mock)
    client = fallback(setting="off", sleep=sleep)
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)

    assert str(caught.value).endswith("; no local fallback: judge.fallback is off")
    assert groq.call_count == 3
    assert tags.call_count == 0
    assert local.call_count == 0


def test_a_users_own_override_is_honoured_as_written(mock: respx.MockRouter) -> None:
    """``judge.fallback`` names a model: it is checked against the tags and used as
    written. The name here is a stand-in, not a model proofpath knows about."""
    _, tags, _, local = routes(mock, tags=_tags("my-judge:latest"))
    client = fallback(setting="my-judge:latest")
    client.complete(MESSAGES)

    assert tags.call_count == 1
    assert json.loads(local.calls.last.request.content)["model"] == "my-judge:latest"
    assert client.switched == "Groq limit reached — judging with local ollama my-judge:latest"


def test_an_untagged_override_matches_its_latest_tag(mock: respx.MockRouter) -> None:
    _, _, _, local = routes(mock, tags=_tags("my-judge:latest"))
    fallback(setting="my-judge").complete(MESSAGES)
    assert json.loads(local.calls.last.request.content)["model"] == "my-judge"


def test_the_model_is_never_pulled(mock: respx.MockRouter) -> None:
    """Rule 5: a model that is not installed is reported, never fetched."""
    pull = mock.post("http://localhost:11434/api/pull").mock(return_value=httpx.Response(200))
    routes(mock, tags=_tags("gemma4:12b"))
    with pytest.raises(judge.JudgeUnavailable, match=r"qwen3\.5:9b is not installed"):
        fallback(sleep=Sleeps()).complete(MESSAGES)
    assert pull.call_count == 0


def test_a_local_model_that_fails_too_keeps_the_primarys_reason(mock: respx.MockRouter) -> None:
    """The search's ``fallback_notice`` reads ``HTTP 429`` out of this detail."""
    routes(mock, local=httpx.Response(404))
    client = fallback()
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)
    text = str(caught.value)
    assert text.startswith(f"HTTP 429 from {GROQ_URL}; local ollama {LOCAL}: HTTP 404")


def test_a_primary_that_is_already_ollama_gets_no_wrapper() -> None:
    built = judge.build_client(judge.provider_defaults("ollama"), None)
    assert isinstance(built, judge.JudgeClient)
    built.close()


def test_a_groq_primary_is_wrapped_with_the_configured_setting() -> None:
    built = judge.build_client(GROQ, KEY)
    assert isinstance(built, judge.FallbackClient)
    assert (built.provider, built.model) == ("groq", GROQ.model)
    built.close()


def test_no_key_reaches_a_repr() -> None:
    client = fallback()
    assert KEY.value not in repr(client)
    assert KEY.value not in str(client)
    client.close()


def test_no_key_reaches_an_error(mock: respx.MockRouter) -> None:
    routes(mock, tags=httpx.ConnectError("refused"))
    client = fallback(sleep=Sleeps())
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)
    assert KEY.value not in str(caught.value)


# --- cost, and a new run -------------------------------------------------------------


def test_local_answers_are_counted_apart_from_api_calls(mock: respx.MockRouter) -> None:
    """Amendment B 7: ``calls`` counts every answer, ``local_calls`` the local ones."""
    groq, _, _, _ = routes(mock, groq=[_ok(OPINION, model=GROQ.model), httpx.Response(429)])
    client = fallback()
    client.complete(MESSAGES)
    client.complete(MESSAGES)
    client.complete(MESSAGES)

    assert groq.call_count == 2
    assert client.cost.calls == 3
    assert client.cost.local_calls == 2
    assert client.cost.model == LOCAL


def test_reset_goes_back_to_the_primary_and_forgets_the_notice(mock: respx.MockRouter) -> None:
    """Amendment B 9: the switch lasts one run; the next one tries the API first."""
    groq, _, _, local = routes(mock, groq=[httpx.Response(429), _ok(OPINION, model=GROQ.model)])
    seen: list[str] = []
    client = fallback(on_switch=seen.append)
    client.complete(MESSAGES)
    assert client.switched == LIMIT

    client.reset()
    assert client.switched is None
    assert (client.provider, client.model) == ("groq", GROQ.model)
    client.complete(MESSAGES)

    assert groq.call_count == 2
    assert local.call_count == 1
    assert seen == [LIMIT]
    # What the local model already answered is not forgotten by the reset.
    assert (client.cost.calls, client.cost.local_calls) == (2, 1)


# --- the run: one note, the footer, the CLI ------------------------------------------

QUERIES = json.dumps(
    {"items": [{"id": 0, "english": CLAIM, "queries": ["ChatGPT shut down March 2025"]}]}
)


def test_a_run_that_switches_emits_one_notice_and_the_footer_carries_it(
    mock: respx.MockRouter,
) -> None:
    routes(mock, local=_ok(QUERIES))
    built = searching(StubSearcher())
    built.judge = judge.Judge(judge.build_client(GROQ, KEY))
    events: list[Event] = []
    report = verify_mod.verify(CLAIM, built, on_event=events.append)

    notices = [event for event in events if isinstance(event, Note) and event.notice]
    assert [note.text for note in notices] == [LIMIT]
    assert report.judge_notice == LIMIT
    assert report.api_calls == 0  # Groq answered nothing; the one answer was local
    assert report.judge_cost is not None and report.judge_cost.local_calls == 1
    footer = render_footer(report)
    assert footer.judge_notice == LIMIT
    assert footer.local_calls == 1
    markdown = render_markdown(report)
    assert f"- judge fallback: {LIMIT}" in markdown
    assert "- local calls: 1" in markdown


def test_a_second_run_on_the_same_engine_starts_on_the_primary_again(
    mock: respx.MockRouter,
) -> None:
    groq, _, _, local = routes(mock, groq=[httpx.Response(429), _ok(QUERIES, model=GROQ.model)])
    local.mock(return_value=_ok(QUERIES))
    built = searching(StubSearcher())
    built.judge = judge.Judge(judge.build_client(GROQ, KEY))
    first = verify_mod.verify(CLAIM, built)
    events: list[Event] = []
    second = verify_mod.verify(CLAIM, built, on_event=events.append)

    assert first.judge_notice == LIMIT
    assert second.judge_notice is None
    assert not [event for event in events if isinstance(event, Note) and event.notice]
    assert groq.call_count == 2


runner = CliRunner()


def test_the_cli_prints_the_notice_on_stderr(
    mock: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes(mock, local=_ok(QUERIES))
    monkeypatch.setenv("GROQ_API_KEY", KEY.value)
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    built = searching(StubSearcher())

    def fake_default(config: Config, **kwargs: Any) -> verify_mod.Engine:
        built.judge = kwargs["judge"]
        built.escalate = kwargs.get("escalate", True)
        return built

    monkeypatch.setattr(verify_mod.Engine, "default", staticmethod(fake_default))
    target = tmp_path / "claim.txt"
    target.write_text(CLAIM, encoding="utf-8")
    result = runner.invoke(app, ["check", str(target), "--judge", "--format", "json"])

    assert LIMIT in result.stderr
    # stdout is the one JSON document and nothing else; the notice rides inside it.
    assert json.loads(result.stdout)["judge_notice"] == LIMIT
    assert KEY.value not in result.output


# --- the setting and the cost line ---------------------------------------------------


def test_judge_fallback_is_a_setting_that_survives_a_provider_switch(tmp_path: Path) -> None:
    from proofpath import commands
    from proofpath.config import load_config, set_value

    path = tmp_path / "config.toml"
    assert load_config(path).judge.fallback == "qwen3.5:9b"
    set_value("judge.fallback", "off", path)
    commands.config_set("judge.provider", "gemini", path)
    # A provider preset carries its model, URL and key variable, not this choice.
    assert load_config(path).judge.fallback == "off"


def test_the_cost_line_names_the_local_answers_only_when_there_are_some() -> None:
    from proofpath.report import calls_text

    assert calls_text(3) == "3 API calls"
    assert calls_text(0, 2) == "0 API calls + 2 local"


# --- fix round 2 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(
            200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        ),
        httpx.Response(200, text="<html>not a completion</html>"),
        httpx.Response(400, json={"error": {"message": "prompt is too long"}}),
    ],
)
def test_one_bad_local_answer_does_not_disable_the_model(
    mock: respx.MockRouter, first: httpx.Response
) -> None:
    """Fix round 2, 1: only a dead model (no answer, 5xx, 404) is latched. A cut-off,
    unreadable or refused answer costs that call; the next call is still asked."""
    item = judge.JudgeItem("c1", "x", "y", Label.NEI, "low")
    _, _, _, local = routes(mock)
    local.mock(side_effect=[first, _ok(OPINION)])
    j = judge.Judge(fallback(sleep=Sleeps()))

    assert j.queries(["OpenAI battı."]) == {}  # noqa: RUF001 - Turkish, verbatim
    assert j.unavailable
    opinions = j.review([item])

    assert not j.unavailable
    assert opinions["c1"].model == f"ollama {LOCAL}"
    assert local.call_count == 2


@pytest.mark.parametrize(
    "dead", [httpx.Response(404), httpx.Response(500), httpx.ConnectError("gone")]
)
def test_a_dead_local_model_is_latched(mock: respx.MockRouter, dead: Any) -> None:
    _, _, _, local = routes(mock)
    # Every attempt dead: the local client's own retries are not what is tested here.
    if isinstance(dead, Exception):
        local.mock(side_effect=dead)
    else:
        local.mock(return_value=dead)
    client = fallback(sleep=Sleeps())
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    asked = local.call_count
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    assert local.call_count == asked


def test_a_second_thread_waits_for_the_warm_up_before_its_request(
    mock: respx.MockRouter,
) -> None:
    """Fix round 2, 2: a thread that did not switch must not start its 180 s read
    timer while the model is still loading; it waits for the warm-up, however long."""
    warming, release = threading.Event(), threading.Event()
    warm_done: list[float] = []
    asked: list[float] = []

    def slow_warm(request: httpx.Request) -> httpx.Response:
        warming.set()
        assert release.wait(timeout=10)
        warm_done.append(time.monotonic())
        return httpx.Response(200, json={})

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(time.monotonic())
        return _ok(OPINION)

    _, _, warm, local = routes(mock)
    warm.mock(side_effect=slow_warm)
    local.mock(side_effect=answer)
    client = fallback()
    results: list[str] = []
    first = threading.Thread(target=lambda: results.append(client.complete(MESSAGES).text))
    first.start()
    assert warming.wait(timeout=5)
    second = threading.Thread(target=lambda: results.append(client.complete(MESSAGES).text))
    second.start()
    time.sleep(0.3)
    assert local.call_count == 0  # neither thread asked while the model was loading
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert results == [OPINION, OPINION]
    assert warm.call_count == 1
    assert all(when >= warm_done[0] for when in asked)


def test_the_warm_up_wait_is_bounded_by_the_warm_ups_own_timeout() -> None:
    assert judge.WARM_WAIT_S > judge.LOCAL_TIMEOUT


@pytest.mark.parametrize(
    "text",
    [
        '{"items": [{"id": 0, "english": "He wrote </think> {x}", "queries": ["q"]}]}',
        '[{"note": "</think> [not a prefix]"}]',
    ],
)
def test_json_that_mentions_the_closing_tag_is_left_intact(text: str) -> None:
    """Fix round 2, 3: an orphan ``</think>`` is a prefix only when the answer does not
    already start as JSON and JSON follows it."""
    assert judge._strip_think(text) == text


def test_an_orphan_closing_tag_before_prose_is_not_stripped() -> None:
    assert judge._strip_think("I think </think> so.") == "I think </think> so."


def test_an_orphan_closing_tag_before_json_is_stripped() -> None:
    assert judge._strip_think('reasoning {a}\n</think>\n{"items": []}') == '{"items": []}'
    assert judge._strip_think('<think>r</think> {"items": []}') == '{"items": []}'
