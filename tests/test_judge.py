"""Judge provider settings and API key resolution. The key is never stored in config."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from proofpath import judge
from proofpath.config import Config, JudgeConfig


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
