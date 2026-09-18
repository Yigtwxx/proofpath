"""Judge provider settings and API key resolution. The key is never stored in config."""

from __future__ import annotations

import importlib.resources
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from proofpath import judge
from proofpath.config import Config, JudgeConfig
from proofpath.models import Label, Tier


def test_default_judge_is_groq_gpt_oss() -> None:
    cfg = Config()
    assert cfg.judge.provider == "groq"
    assert cfg.judge.model == "openai/gpt-oss-120b"
    assert cfg.judge.api_key_env == "GROQ_API_KEY"
    assert cfg.judge.base_url == "https://api.groq.com/openai/v1"


def test_known_providers_fill_in_defaults() -> None:
    assert judge.provider_defaults("gemini").api_key_env == "GEMINI_API_KEY"
    assert judge.provider_defaults("ollama").api_key_env == ""
    with pytest.raises(judge.JudgeError, match="unknown provider"):
        judge.provider_defaults("nvidia")


def test_read_dotenv_handles_quotes_comments_and_export(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport GROQ_API_KEY='gsk_abc'\nOTHER=\"x y\" # trailing\nBARE=plain\n\n",
        encoding="utf-8",
    )
    assert judge.read_dotenv(env_file) == {
        "GROQ_API_KEY": "gsk_abc",
        "OTHER": "x y",
        "BARE": "plain",
    }


def test_read_dotenv_missing_file_is_empty(tmp_path: Path) -> None:
    assert judge.read_dotenv(tmp_path / ".env") == {}


def test_resolve_api_key_prefers_environment_then_dotenv_files(tmp_path: Path) -> None:
    first = tmp_path / "a" / ".env"
    second = tmp_path / "b" / ".env"
    first.parent.mkdir()
    second.parent.mkdir()
    second.write_text("GROQ_API_KEY=from-second\n", encoding="utf-8")
    found = judge.resolve_api_key("GROQ_API_KEY", environ={}, dotenv_paths=[first, second])
    assert found == judge.ApiKey("from-second", source=str(second))
    first.write_text("GROQ_API_KEY=from-first\n", encoding="utf-8")
    found = judge.resolve_api_key("GROQ_API_KEY", environ={}, dotenv_paths=[first, second])
    assert found is not None and found.source == str(first)
    found = judge.resolve_api_key(
        "GROQ_API_KEY", environ={"GROQ_API_KEY": "from-env"}, dotenv_paths=[first]
    )
    assert found == judge.ApiKey("from-env", source="environment variable GROQ_API_KEY")


def test_resolve_api_key_returns_none_when_absent(tmp_path: Path) -> None:
    assert (
        judge.resolve_api_key("GROQ_API_KEY", environ={}, dotenv_paths=[tmp_path / ".env"]) is None
    )


def test_api_key_never_leaks_through_repr() -> None:
    key = judge.ApiKey("gsk_secret_value", source="x")
    assert "gsk_secret_value" not in repr(key)
    assert "gsk_secret_value" not in str(key)


@respx.mock
def test_check_sends_bearer_and_reports_latency() -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"model": "openai/gpt-oss-120b", "choices": [{"message": {"content": "OK"}}]},
        )
    )
    cfg = JudgeConfig()
    result = judge.check(cfg, judge.ApiKey("gsk_x", source="test"))
    assert route.called
    assert route.calls.last.request.headers["authorization"] == "Bearer gsk_x"
    assert result.ok and result.model == "openai/gpt-oss-120b" and result.latency_ms >= 0
    assert "gsk_x" not in result.detail


@respx.mock
def test_check_reports_http_errors_without_the_raw_body() -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API Key gsk_x"}})
    )
    result = judge.check(JudgeConfig(), judge.ApiKey("gsk_x", source="test"))
    assert not result.ok
    assert "401" in result.detail
    assert "gsk_x" not in result.detail


def test_check_without_a_key_on_a_keyed_provider_fails_cleanly() -> None:
    result = judge.check(JudgeConfig(), None)
    assert not result.ok
    assert "GROQ_API_KEY" in result.detail


# --- the HTTP client (task 9.1) ---------------------------------------------
# Ollama is the preset with no API key, so these tests need no secret at all.

OLLAMA = judge.provider_defaults("ollama")
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MESSAGES = [
    {"role": "system", "content": "Answer only from the passage."},
    {"role": "user", "content": "1. id=c1 claim=... passage=..."},
]
SCHEMA = {"type": "object", "properties": {"opinions": {"type": "array"}}}


def _ok(content: str = '{"opinions": []}', **extra: object) -> httpx.Response:
    body: dict[str, object] = {
        "model": "llama3.1",
        "choices": [{"message": {"content": content}}],
    }
    body.update(extra)
    return httpx.Response(200, json=body)


def test_estimate_tokens_is_four_characters_plus_overhead() -> None:
    assert judge.estimate_tokens("") == 8
    assert judge.estimate_tokens("a" * 400) == 108


@respx.mock
def test_complete_sends_the_json_schema_and_fills_usage_from_the_response() -> None:
    route = respx.post(OLLAMA_URL).mock(
        return_value=_ok(usage={"prompt_tokens": 120, "completion_tokens": 7})
    )
    client = judge.JudgeClient(OLLAMA, None)
    done = client.complete(MESSAGES, json_schema=SCHEMA)
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "llama3.1"
    assert sent["messages"] == MESSAGES
    assert sent["max_tokens"] == 1024
    assert sent["temperature"] == 0.0
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert sent["response_format"]["json_schema"]["strict"] is True
    # Ollama takes no key, so no Authorization header may be sent.
    assert "authorization" not in route.calls.last.request.headers
    assert done == judge.Completion(
        text='{"opinions": []}', prompt_tokens=120, completion_tokens=7, model="llama3.1"
    )
    assert client.cost.calls == 1
    assert client.cost.prompt_tokens == 120
    assert client.cost.completion_tokens == 7
    assert client.cost.waited_s == 0.0
    assert client.cost.model == "llama3.1"


@respx.mock
def test_complete_without_response_format_omits_it_and_estimates_missing_usage() -> None:
    route = respx.post(OLLAMA_URL).mock(return_value=_ok("no opinion"))
    client = judge.JudgeClient(OLLAMA, None)
    done = client.complete(MESSAGES)
    assert "response_format" not in json.loads(route.calls.last.request.content)
    expected_prompt = sum(judge.estimate_tokens(m["content"]) for m in MESSAGES)
    assert done.prompt_tokens == expected_prompt
    assert done.completion_tokens == judge.estimate_tokens("no opinion")


@respx.mock
def test_a_provider_that_rejects_the_schema_is_retried_once_with_json_object() -> None:
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(
                400, json={"error": {"message": "response_format json_schema is not supported"}}
            ),
            _ok(),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    done = client.complete(MESSAGES, json_schema=SCHEMA)
    assert done.text == '{"opinions": []}'
    assert route.call_count == 2
    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert first["response_format"]["type"] == "json_schema"
    assert second["response_format"] == {"type": "json_object"}


@respx.mock
def test_complete_sends_the_reasoning_effort_it_was_given_and_omits_it_otherwise() -> None:
    """A reasoning model spends its budget thinking unless it is told how much to."""
    route = respx.post(OLLAMA_URL).mock(return_value=_ok("a paragraph"))
    client = judge.JudgeClient(OLLAMA, None)

    client.complete(MESSAGES, reasoning_effort="low")
    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == "low"
    client.complete(MESSAGES)
    assert "reasoning_effort" not in json.loads(route.calls.last.request.content)
    client.close()


@respx.mock
def test_a_provider_that_rejects_reasoning_effort_is_retried_once_without_it() -> None:
    """Ollama and Gemini models refuse the field; refusing it must not cost the call."""
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "unknown parameter reasoning_effort"}}),
            _ok("a paragraph"),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    done = client.complete(MESSAGES, reasoning_effort="low")

    assert done.text == "a paragraph"
    assert route.call_count == 2
    assert "reasoning_effort" in json.loads(route.calls[0].request.content)
    assert "reasoning_effort" not in json.loads(route.calls[1].request.content)
    client.close()


@respx.mock
def test_each_downgrade_costs_one_extra_request_and_no_more() -> None:
    """Both fields can be refused by the same model; neither retry spends an attempt."""
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(
                400, json={"error": {"message": "response_format json_schema is not supported"}}
            ),
            httpx.Response(400, json={"error": {"message": "unknown parameter: reasoning_effort"}}),
            _ok(),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    done = client.complete(MESSAGES, json_schema=SCHEMA, reasoning_effort="low")

    assert done.text == '{"opinions": []}'
    assert route.call_count == 3
    sent = [json.loads(call.request.content) for call in route.calls]
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[0]["reasoning_effort"] == "low"
    assert sent[1]["response_format"] == {"type": "json_object"}
    assert sent[1]["reasoning_effort"] == "low"
    assert sent[2]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in sent[2]
    client.close()


@respx.mock
def test_a_provider_that_refuses_reasoning_effort_is_only_taught_once() -> None:
    """The downgrade is a fact about the provider, not about one call.

    A judge client outlives a batch -- 9.2 shares one across every batch and the
    summary -- so a latch kept per call would pay the same wasted 400 again on each
    of them, against a per-minute tier that has no room for it.
    """
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "unknown parameter reasoning_effort"}}),
            _ok("first"),
            _ok("second"),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    client.complete(MESSAGES, reasoning_effort="low")
    client.complete(MESSAGES, reasoning_effort="low")

    assert route.call_count == 3  # one downgrade in total, not one per call
    sent = [json.loads(call.request.content) for call in route.calls]
    assert sent[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in sent[1]
    assert "reasoning_effort" not in sent[2]
    client.close()


@respx.mock
def test_a_provider_that_refuses_the_schema_is_only_taught_once() -> None:
    """Same latch, same reason: the second batch must not re-learn the first's 400."""
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(
                400, json={"error": {"message": "response_format json_schema is not supported"}}
            ),
            _ok(),
            _ok(),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    client.complete(MESSAGES, json_schema=SCHEMA)
    client.complete(MESSAGES, json_schema=SCHEMA)

    assert route.call_count == 3
    sent = [json.loads(call.request.content) for call in route.calls]
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[1]["response_format"] == {"type": "json_object"}
    assert sent[2]["response_format"] == {"type": "json_object"}
    client.close()


@respx.mock
def test_a_client_that_never_sees_a_400_keeps_asking_for_both_fields() -> None:
    """The latch only ever closes on a refusal: a provider that takes both keeps both."""
    route = respx.post(OLLAMA_URL).mock(return_value=_ok())
    client = judge.JudgeClient(OLLAMA, None)
    client.complete(MESSAGES, json_schema=SCHEMA, reasoning_effort="low")
    client.complete(MESSAGES, json_schema=SCHEMA, reasoning_effort="low")

    assert route.call_count == 2
    for call in route.calls:
        sent = json.loads(call.request.content)
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["reasoning_effort"] == "low"
    client.close()


@respx.mock
def test_a_400_that_only_uses_the_word_reasoning_is_not_a_downgrade() -> None:
    """``reasoning_effort`` is the field; a 400 that merely says "reasoning" is a 400.

    Dropping the field on that evidence throws away the one thing that keeps a
    reasoning model from spending its whole budget thinking, and buys a second
    request for a refusal that was never about the field.
    """
    route = respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"message": "this model has no reasoning mode"}}
        )
    )
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES, reasoning_effort="low")
    assert route.call_count == 1
    client.close()


@respx.mock
def test_a_400_naming_the_reasoning_effort_field_is_still_a_downgrade() -> None:
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "Unsupported: 'reasoning_effort'"}}),
            _ok("a paragraph"),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    assert client.complete(MESSAGES, reasoning_effort="low").text == "a paragraph"
    assert route.call_count == 2
    client.close()


@respx.mock
def test_a_400_about_nothing_in_particular_is_still_not_retried() -> None:
    """The two downgrades are targeted; a plain 400 must not become three requests."""
    route = respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "model not found"}})
    )
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES, json_schema=SCHEMA, reasoning_effort="low")
    assert route.call_count == 1
    client.close()


@respx.mock
def test_a_plain_400_is_not_retried_and_never_carries_the_body() -> None:
    route = respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad model gsk_secret_value"}})
    )
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES, json_schema=SCHEMA)
    assert route.call_count == 1
    assert "400" in str(caught.value)
    assert "gsk_secret_value" not in str(caught.value)
    assert "bad model" not in str(caught.value)


@respx.mock
def test_a_429_waits_out_retry_after_then_succeeds() -> None:
    respx.post(OLLAMA_URL).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "2"}, json={}), _ok()]
    )
    slept: list[float] = []
    client = judge.JudgeClient(OLLAMA, None, sleep=slept.append)
    done = client.complete(MESSAGES)
    assert done.text == '{"opinions": []}'
    assert slept == [2.0]
    assert client.cost.waited_s == 2.0
    assert client.cost.calls == 1


@respx.mock
def test_a_retry_after_date_is_understood_and_capped_at_max_wait() -> None:
    respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}, json={}),
            _ok(),
        ]
    )
    slept: list[float] = []
    client = judge.JudgeClient(OLLAMA, None, sleep=slept.append, max_wait=5.0)
    client.complete(MESSAGES)
    assert slept == [5.0]
    assert client.cost.waited_s == 5.0


@respx.mock
def test_three_429s_give_up_with_judge_unavailable() -> None:
    route = respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "1"}, json={})
    )
    slept: list[float] = []
    client = judge.JudgeClient(OLLAMA, None, sleep=slept.append)
    with pytest.raises(judge.JudgeUnavailable, match="429"):
        client.complete(MESSAGES)
    assert route.call_count == 3
    assert slept == [1.0, 1.0]  # no wait after the last attempt
    assert client.cost.calls == 0
    # The waits happened, so the report must still show them after giving up.
    assert client.cost.waited_s == 2.0


@respx.mock
def test_two_503s_back_off_exponentially_then_succeed() -> None:
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[httpx.Response(503, json={}), httpx.Response(503, json={}), _ok()]
    )
    slept: list[float] = []
    client = judge.JudgeClient(OLLAMA, None, sleep=slept.append)
    done = client.complete(MESSAGES)
    assert done.text == '{"opinions": []}'
    assert route.call_count == 3
    assert slept == [0.5, 1.0]
    assert client.cost.waited_s == 1.5


@respx.mock
def test_transport_errors_are_retried_and_reported_by_type_only() -> None:
    route = respx.post(OLLAMA_URL).mock(side_effect=httpx.ConnectError("connection to gsk_x"))
    slept: list[float] = []
    client = judge.JudgeClient(OLLAMA, None, sleep=slept.append)
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)
    assert route.call_count == 3
    assert "ConnectError" in str(caught.value)
    assert "gsk_x" not in str(caught.value)


@respx.mock
def test_a_200_that_is_not_json_is_unavailable_rather_than_a_crash() -> None:
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, text="<html>rate limited</html>"))
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    assert client.cost.calls == 0


@respx.mock
def test_a_200_with_the_wrong_json_shape_is_unavailable_rather_than_a_crash() -> None:
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json=["not", "a", "completion"]))
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    assert client.cost.calls == 0


@respx.mock
def test_a_200_with_no_choices_is_unavailable_rather_than_an_empty_answer() -> None:
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(200, json={"choices": []}))
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    assert client.cost.calls == 0


@respx.mock
def test_a_filtered_answer_is_unavailable_and_names_the_finish_reason_only() -> None:
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "system_fingerprint": "gsk_secret_value",
                "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}],
            },
        )
    )
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable) as caught:
        client.complete(MESSAGES)
    assert "content_filter" in str(caught.value)
    assert "gsk_secret_value" not in str(caught.value)
    assert client.cost.calls == 0


@respx.mock
def test_an_empty_string_answer_is_unavailable_rather_than_an_empty_completion() -> None:
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200, json={"model": "llama3.1", "choices": [{"message": {"content": ""}}]}
        )
    )
    client = judge.JudgeClient(OLLAMA, None)
    with pytest.raises(judge.JudgeUnavailable):
        client.complete(MESSAGES)
    assert client.cost.calls == 0


@respx.mock
def test_an_honest_zero_in_usage_is_kept_and_never_replaced_by_an_estimate() -> None:
    respx.post(OLLAMA_URL).mock(
        return_value=_ok("hi", usage={"prompt_tokens": 0, "completion_tokens": 0})
    )
    client = judge.JudgeClient(OLLAMA, None)
    done = client.complete(MESSAGES)
    assert done.prompt_tokens == 0
    assert done.completion_tokens == 0
    assert client.cost.prompt_tokens == 0
    assert client.cost.completion_tokens == 0


@respx.mock
def test_cost_accumulates_across_calls_on_one_client() -> None:
    respx.post(OLLAMA_URL).mock(
        side_effect=[
            _ok(usage={"prompt_tokens": 120, "completion_tokens": 7}),
            _ok(usage={"prompt_tokens": 30, "completion_tokens": 2}),
        ]
    )
    client = judge.JudgeClient(OLLAMA, None)
    client.complete(MESSAGES)
    client.complete(MESSAGES)
    assert client.cost.calls == 2
    assert client.cost.prompt_tokens == 150
    assert client.cost.completion_tokens == 9


def test_close_leaves_an_injected_client_open() -> None:
    injected = httpx.Client()
    client = judge.JudgeClient(OLLAMA, None, client=injected)
    client.close()
    assert not injected.is_closed
    injected.close()


def test_close_closes_a_client_it_built_itself() -> None:
    with judge.JudgeClient(OLLAMA, None) as client:
        owned = client._client
        assert not owned.is_closed
    assert owned.is_closed


@respx.mock
def test_a_keyed_provider_sends_the_bearer_but_never_shows_the_key() -> None:
    route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-120b",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )
    )
    client = judge.JudgeClient(JudgeConfig(), judge.ApiKey("gsk_secret_value", source="test"))
    done = client.complete(MESSAGES)
    assert route.calls.last.request.headers["authorization"] == "Bearer gsk_secret_value"
    assert done.model == "openai/gpt-oss-120b"
    assert "gsk_secret_value" not in repr(client)
    assert "gsk_secret_value" not in str(client)


def test_a_keyed_provider_without_a_key_refuses_to_build() -> None:
    with pytest.raises(judge.JudgeError) as caught:
        judge.JudgeClient(JudgeConfig(), None)
    # The same wording `config check` prints for a missing key.
    assert str(caught.value) == judge.check(JudgeConfig(), None).detail


def test_review_prompt_renders_and_demands_strict_json() -> None:
    text = judge.load_prompt("review").substitute(items="1. id=c1 claim=... passage=...")
    assert "1. id=c1 claim=... passage=..." in text
    assert "SUPPORTED" in text and "REFUTED" in text and "NEI" in text
    assert '"opinions"' in text
    assert "rationale" in text
    # The per-item field the batch builder fills is `verdict` (task 9.1 brief).
    assert "`verdict`" in text
    assert "local_verdict" not in text


def test_summarize_prompt_renders_the_finished_report() -> None:
    text = judge.load_prompt("summarize").substitute(report="# proofpath report\n42 references")
    assert "# proofpath report\n42 references" in text
    assert "coverage" in text.lower()
    # ``render_markdown`` puts the paragraph last; the prompt may not promise otherwise.
    assert "at the end of a finished proofpath" in text
    assert "top of a finished" not in text


def test_prompt_files_are_packaged_with_the_module() -> None:
    root = importlib.resources.files("proofpath.prompts")
    for name in ("review.md", "summarize.md"):
        assert (root / name).is_file()


# --- the batching judge (task 9.2) -------------------------------------------
# Still Ollama: packing, parsing and escalation need no secret and no network.


def _item(ident: str, *, size: int = 238, tier: Tier = "low") -> judge.JudgeItem:
    """One escalated verdict, padded so a batch's token budget is predictable."""
    return judge.JudgeItem(
        id=ident,
        claim="The method is faster. " + "c" * size,
        passage="The method is faster. " + "p" * size,
        verdict=Label.NEI,
        tier=tier,
    )


def _answers(label: str = "NEI", rationale: str = 'the passage says "faster"') -> Any:
    """Answer every id the request actually carries, so packing decides the shape."""

    def respond(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        ids = re.findall(r"id: (\S+)", sent["messages"][-1]["content"])
        payload = {"opinions": [{"id": i, "label": label, "rationale": rationale} for i in ids]}
        return _ok(json.dumps(payload))

    return respond


def _prompts(route: Any) -> list[str]:
    return [json.loads(call.request.content)["messages"][-1]["content"] for call in route.calls]


def _prompt_tokens(route: Any) -> list[int]:
    """What each batch actually asked for: system *and* user, as the cap counts them."""
    return [
        sum(
            judge.estimate_tokens(m["content"])
            for m in json.loads(call.request.content)["messages"]
        )
        for call in route.calls
    ]


def test_split_prompt_returns_the_system_and_user_halves() -> None:
    system, user = judge._split_prompt(judge.load_prompt("review").substitute(items="ITEMS"))
    assert system.startswith("You are a careful evidence reviewer")
    assert "## System" not in system and "## User" not in system
    assert user.startswith("Review every item below")
    assert "ITEMS" in user
    assert "ITEMS" not in system


@respx.mock
def test_review_sends_two_messages_and_a_strict_schema() -> None:
    route = respx.post(OLLAMA_URL).mock(side_effect=_answers())
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    opinions = reviewer.review([_item("c1")])

    sent = json.loads(route.calls.last.request.content)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"]["properties"]["opinions"]
    # The per-item field the template names is `verdict`: the local label and tier.
    assert "verdict: NEI (low)" in sent["messages"][-1]["content"]
    assert opinions["c1"] == judge.JudgeOpinion(
        label=Label.NEI, rationale='the passage says "faster"', model="ollama llama3.1"
    )
    assert reviewer.cost.calls == 1
    assert not reviewer.unavailable and reviewer.skipped == []


@respx.mock
def test_review_asks_for_low_reasoning_and_a_budget_it_can_answer_within() -> None:
    """A reasoning model charges its thinking to ``max_tokens``: too small a budget
    comes back ``finish_reason=length`` with no opinions in it at all."""
    route = respx.post(OLLAMA_URL).mock(side_effect=_answers())
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    reviewer.review([_item("c1")])

    sent = json.loads(route.calls.last.request.content)
    assert sent["reasoning_effort"] == "low"
    assert sent["max_tokens"] == 4096
    reviewer.close()


@respx.mock
def test_packing_stops_at_the_item_cap() -> None:
    """118 escalated verdicts of ~150 tokens: 20 to a batch, so 6 calls, not 118."""
    route = respx.post(OLLAMA_URL).mock(side_effect=_answers())
    reviewer = judge.Judge(
        judge.JudgeClient(OLLAMA, None), batch_size=20, token_cap=judge.TOKEN_CAP
    )
    items = [_item(f"c{n}") for n in range(118)]
    opinions = reviewer.review(items)

    assert route.call_count == 6
    assert len(opinions) == 118
    # Both caps hold at once: the item cap closed every batch, and no batch's whole
    # rendered prompt -- template included -- went over the token budget.
    assert all(len(ids) <= 20 for ids in (re.findall(r"id: (\S+)", p) for p in _prompts(route)))
    assert max(_prompt_tokens(route)) <= judge.TOKEN_CAP


def test_the_token_cap_leaves_room_for_the_answer_the_batch_asks_for() -> None:
    """The cap counts the *prompt*; the answer budget is charged on top of it.

    Groq's free tier is 8k tokens a minute, so a full batch plus the review's
    ``max_tokens`` has to fit under that or every batch buys a 429 (spec section 11).
    """
    assert judge.TOKEN_CAP == 3500
    assert judge.TOKEN_CAP + judge._REVIEW_TOKENS <= 8000


@respx.mock
def test_packing_closes_a_batch_when_the_token_cap_is_reached() -> None:
    """The item cap is not the only cap: a batch closes on tokens too (spec 11)."""
    route = respx.post(OLLAMA_URL).mock(side_effect=_answers())
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None), batch_size=20, token_cap=2000)
    opinions = reviewer.review([_item(f"c{n}") for n in range(20)])

    assert route.call_count > 1  # the 20 would have fitted the item cap
    assert len(opinions) == 20
    assert max(_prompt_tokens(route)) <= 2000


@respx.mock
def test_one_oversized_item_is_still_sent_rather_than_dropped() -> None:
    """A batch of one over the cap is the smallest prompt there is; dropping the
    item would lose a verdict the run was told to escalate."""
    route = respx.post(OLLAMA_URL).mock(side_effect=_answers())
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None), batch_size=20, token_cap=10)
    opinions = reviewer.review([_item("c1"), _item("c2")])

    assert route.call_count == 2
    assert set(opinions) == {"c1", "c2"}


@respx.mock
def test_lenient_parsing_keeps_the_good_opinions_and_notes_the_rest() -> None:
    """One bad entry must not cost the batch: the good ids are kept, the rest noted."""
    respx.post(OLLAMA_URL).mock(
        return_value=_ok(
            '```json\n{"opinions": ['
            '{"id": "c1", "label": "SUPPORTED", "rationale": "it says so"},'
            '{"id": "nobody", "label": "NEI", "rationale": "x"},'
            '{"id": "c2", "label": "MAYBE", "rationale": "x"},'
            '{"id": "c3", "label": "NEI", "rationale": "   "},'
            '{"id": "c1", "label": "REFUTED", "rationale": "second helping"}'
            "]}\n```"
        )
    )
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    opinions = reviewer.review([_item("c1"), _item("c2"), _item("c3")])

    assert set(opinions) == {"c1"}
    assert opinions["c1"].label is Label.SUPPORTED
    assert len(reviewer.skipped) == 4
    joined = " | ".join(reviewer.skipped)
    assert "nobody" in joined and "MAYBE" in joined and "c3" in joined
    assert not reviewer.unavailable


@respx.mock
def test_an_unparseable_answer_skips_the_batch_without_raising() -> None:
    respx.post(OLLAMA_URL).mock(return_value=_ok("I would rather not."))
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    assert reviewer.review([_item("c1")]) == {}
    assert len(reviewer.skipped) == 1
    assert not reviewer.unavailable


@respx.mock
def test_becoming_unavailable_mid_way_keeps_what_was_already_gathered() -> None:
    """The judge is an extra opinion: losing it mid-run must not lose the run."""
    respx.post(OLLAMA_URL).mock(
        side_effect=[
            _ok('{"opinions": [{"id": "c1", "label": "NEI", "rationale": "silent"}]}'),
            httpx.Response(401, json={"error": {"message": "Invalid API Key gsk_secret"}}),
        ]
    )
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None), batch_size=1)
    opinions = reviewer.review([_item("c1"), _item("c2")])

    assert set(opinions) == {"c1"}
    assert reviewer.unavailable
    # The detail says *why*, so a wrong key is visible rather than just "unavailable".
    assert "401" in reviewer.detail
    assert "gsk_secret" not in reviewer.detail


@respx.mock
def test_a_second_review_starts_from_a_clean_state() -> None:
    """``skipped``, ``unavailable`` and ``detail`` are about *this* call. A ``Judge``
    kept across two documents must not report the first one's trouble on the second,
    and a provider that has come back up must not stay down."""
    respx.post(OLLAMA_URL).mock(
        side_effect=[
            httpx.Response(401, json={"error": {"message": "Invalid API Key"}}),
            _ok('{"opinions": [{"id": "c1", "label": "NEI", "rationale": "silent"}]}'),
        ]
    )
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None), batch_size=1)
    reviewer.review([_item("c1")])
    assert reviewer.unavailable and reviewer.detail

    opinions = reviewer.review([_item("c1")])

    assert set(opinions) == {"c1"}
    assert not reviewer.unavailable
    assert reviewer.detail == ""
    assert reviewer.skipped == []


@respx.mock
def test_progress_is_reported_per_batch_and_a_raising_hook_is_the_callers_own() -> None:
    """``on_batch`` is where a run checks its cancel flag, between batches."""
    route = respx.post(OLLAMA_URL).mock(side_effect=_answers())
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None), batch_size=1)
    seen: list[tuple[int, int]] = []

    def hook(done: int, total: int) -> None:
        seen.append((done, total))
        if done == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        reviewer.review([_item(f"c{n}") for n in range(4)], on_batch=hook)
    assert seen == [(1, 4), (2, 4)]
    assert route.call_count == 2


def test_review_never_raises_when_the_client_misbehaves() -> None:
    """Nothing the optional layer does may reach the run (spec section 11.1)."""

    class Broken:
        cost = judge.JudgeCost(model="llama3.1")
        provider = "ollama"
        model = "llama3.1"

        def complete(self, *args: object, **kwargs: object) -> judge.Completion:
            raise ZeroDivisionError("a bug in the adapter")

    reviewer = judge.Judge(Broken())  # type: ignore[arg-type]
    assert reviewer.review([_item("c1")]) == {}
    assert reviewer.unavailable and "ZeroDivisionError" in reviewer.detail


def test_the_judge_names_itself_by_provider_and_model() -> None:
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    assert reviewer.name == "ollama llama3.1"
    reviewer.close()


def test_closing_the_judge_closes_the_client_it_was_given() -> None:
    client = judge.JudgeClient(OLLAMA, None)
    judge.Judge(client).close()
    assert client._client.is_closed


# --- the summary (task 9.3) --------------------------------------------------
# One call over a report that is already final: plain text, no schema, and a
# provider that will not answer costs the paragraph rather than the run.

REPORT = "# proofpath report — draft.md\n\n- api calls: 0\n\n## Findings\n\nnone\n"
PARAGRAPH = "42 references were checked and three do not say what the draft says."


@respx.mock
def test_summarize_sends_the_finished_report_as_plain_text() -> None:
    route = respx.post(OLLAMA_URL).mock(return_value=_ok(f"  {PARAGRAPH}\n"))
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    text = reviewer.summarize(REPORT)

    sent = json.loads(route.calls.last.request.content)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    # The report is the whole input, and it goes in the user turn, not the system one.
    assert REPORT.strip() in sent["messages"][-1]["content"]
    assert REPORT.strip() not in sent["messages"][0]["content"]
    # Prose, so no JSON schema is asked for at all.
    assert "response_format" not in sent
    assert sent["max_tokens"] == 1500
    assert sent["reasoning_effort"] == "low"
    assert text == PARAGRAPH
    assert reviewer.cost.calls == 1
    assert not reviewer.unavailable and reviewer.detail == ""
    reviewer.close()


@respx.mock
def test_summarize_takes_its_token_budget_from_the_caller() -> None:
    route = respx.post(OLLAMA_URL).mock(return_value=_ok(PARAGRAPH))
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    reviewer.summarize(REPORT, max_tokens=120)
    assert json.loads(route.calls.last.request.content)["max_tokens"] == 120
    reviewer.close()


@respx.mock
def test_a_blank_paragraph_is_no_answer_at_all_rather_than_nothing_to_say() -> None:
    """A 200 whose content is whitespace would otherwise render "after 1 call ()"."""
    respx.post(OLLAMA_URL).mock(return_value=_ok("   \n  "))
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))

    assert reviewer.summarize(REPORT) == ""
    assert reviewer.unavailable and reviewer.detail == "empty answer"
    reviewer.close()


@respx.mock
def test_a_provider_that_will_not_write_a_summary_says_so_instead() -> None:
    """Product rule 2: an empty paragraph would read as "nothing to say"."""
    respx.post(OLLAMA_URL).mock(return_value=httpx.Response(401))
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))

    assert reviewer.summarize(REPORT) == ""
    assert reviewer.unavailable
    assert "401" in reviewer.detail
    reviewer.close()


@respx.mock
def test_a_summary_never_raises_when_the_client_misbehaves() -> None:
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))

    def boom(*args: Any, **kwargs: Any) -> None:
        raise ZeroDivisionError("a library changed under us")

    reviewer._client.complete = boom  # type: ignore[method-assign]
    assert reviewer.summarize(REPORT) == ""
    assert reviewer.unavailable and "ZeroDivisionError" in reviewer.detail
    reviewer.close()


@respx.mock
def test_a_summary_starts_from_a_clean_state_after_an_unavailable_review() -> None:
    """The paragraph reports its *own* call, not the review's earlier trouble."""
    route = respx.post(OLLAMA_URL).mock(
        side_effect=[httpx.Response(401), _ok(PARAGRAPH)],
    )
    reviewer = judge.Judge(judge.JudgeClient(OLLAMA, None))
    reviewer.review([_item("c1")])
    assert reviewer.unavailable

    assert reviewer.summarize(REPORT) == PARAGRAPH
    assert not reviewer.unavailable and reviewer.detail == ""
    assert route.call_count == 2
    reviewer.close()


def test_known_providers_are_the_provider_table_sorted() -> None:
    """The settings panel's provider row cycles over exactly what ``provider_defaults``
    accepts, in the order its own error message lists them."""
    assert judge.known_providers() == ("gemini", "groq", "ollama")
    assert judge.known_providers() == tuple(sorted(judge._PROVIDERS))
