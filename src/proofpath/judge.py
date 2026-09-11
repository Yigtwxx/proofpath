"""Optional LLM judge: provider settings, API key resolution, connectivity check.

Every supported provider speaks the OpenAI ``chat/completions`` shape, so one
adapter covers all of them (see ``docs/research/2026-09-11-free-llm-api-tiers.md``).
The batching and escalation logic is Phase 9; this module only makes sure a key
can be found and works, without ever printing it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from proofpath import __version__
from proofpath.config import JudgeConfig
from proofpath.paths import config_dir


class JudgeError(ValueError):
    """Bad judge settings."""


# Free, no card, OpenAI-compatible (verified 2026-09-11). Ollama is the offline default.
_PROVIDERS: dict[str, JudgeConfig] = {
    "groq": JudgeConfig(),
    "gemini": JudgeConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    "ollama": JudgeConfig(
        provider="ollama",
        model="llama3.1",
        base_url="http://localhost:11434/v1",
        api_key_env="",
    ),
}


def provider_defaults(name: str) -> JudgeConfig:
    try:
        return _PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS))
        raise JudgeError(f"unknown provider {name!r}; known: {known}") from None


@dataclass(frozen=True, repr=False)
class ApiKey:
    """A secret plus where it came from. ``repr``/``str`` never show the value."""

    value: str = field(repr=False)
    source: str

    def __repr__(self) -> str:
        return f"ApiKey(source={self.source!r})"

    __str__ = __repr__


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``.env`` reader: ``KEY=value``, optional ``export``, quotes, comments."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in {"'", '"'} and value.count(value[0]) >= 2:
            quote = value[0]
            value = value[1 : value.index(quote, 1)]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = value
    return values


def default_dotenv_paths() -> list[Path]:
    """Project ``.env`` first, then the user config dir."""
    return [Path.cwd() / ".env", config_dir() / ".env"]


def resolve_api_key(
    env_name: str,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Sequence[Path] | None = None,
) -> ApiKey | None:
    """Environment variable first, then each ``.env`` file in order."""
    if not env_name:
        return None
    environ = os.environ if environ is None else environ
    value = environ.get(env_name)
    if value:
        return ApiKey(value, source=f"environment variable {env_name}")
    for path in default_dotenv_paths() if dotenv_paths is None else dotenv_paths:
        value = read_dotenv(path).get(env_name)
        if value:
            return ApiKey(value, source=str(path))
    return None


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    model: str
    latency_ms: int
    detail: str


def check(config: JudgeConfig, key: ApiKey | None, *, timeout: float = 30.0) -> CheckResult:
    """One tiny completion to prove the provider, model and key work together."""
    if config.api_key_env and key is None:
        return CheckResult(
            False, config.model, 0, f"no API key: set {config.api_key_env} or add it to .env"
        )
    headers = {"User-Agent": f"proofpath/{__version__}"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key.value}"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    url = config.base_url.rstrip("/") + "/chat/completions"
    started = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return CheckResult(False, config.model, 0, f"request failed: {type(exc).__name__}")
    latency = int((time.perf_counter() - started) * 1000)
    if response.status_code != 200:
        # Never echo the body: providers put the key or account details in it.
        return CheckResult(False, config.model, latency, f"HTTP {response.status_code} from {url}")
    body = response.json()
    model = str(body.get("model") or config.model)
    return CheckResult(True, model, latency, "ok")
