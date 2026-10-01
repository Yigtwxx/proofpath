"""Optional LLM judge: provider settings, API key resolution, prompts, HTTP client.

Every supported provider speaks the OpenAI ``chat/completions`` shape, so one
adapter covers all of them (see ``docs/research/2026-09-11-free-llm-api-tiers.md``).
This module renders the packaged prompt templates and makes one retried,
cost-accounted call at a time. Batching and escalation sit on top of it, in the
pipeline.

Finding the key itself moved to ``proofpath.secrets`` in v0.4.0, when Reddit became
the second thing in the tool that takes a credential (spec section 6.2): one lookup,
one rule about never printing a value. :class:`ApiKey`, :func:`resolve_api_key`,
:func:`read_dotenv` and :func:`default_dotenv_paths` are re-exported here unchanged,
because the CLI, the TUI and ``commands`` all ask *this* module for a key.
"""

from __future__ import annotations

import importlib.resources
import json
import re
import string
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any

import httpx

from proofpath import __version__
from proofpath.config import LOCAL_JUDGE_MODEL, JudgeConfig
from proofpath.models import Label, Tier
from proofpath.secrets import ApiKey, default_dotenv_paths, read_dotenv, resolve_api_key

# Named so the re-exports above are exports rather than unused imports. Nothing
# does ``from proofpath.judge import *``; this list is documentation for the reader
# and for the linter, not the module's public surface.
__all__ = [
    "ApiKey",
    "default_dotenv_paths",
    "read_dotenv",
    "resolve_api_key",
]


class JudgeError(ValueError):
    """Bad judge settings."""


class JudgeUnavailable(JudgeError):  # noqa: N818 - reads as a state, not as a failed run
    """The provider did not answer after retries. A run reports this, never fails on it.

    ``status`` is the HTTP status of the last answer, or ``None`` when there was none
    (a transport error, an unreadable 200). ``cause`` is the same fact in a few words
    a notice can print (``HTTP 500``, ``timeout``, ``empty answer``): never a body,
    never a URL, never a key. The fallback reads both to say why it switched without
    parsing its own message back.
    """

    def __init__(self, detail: str, *, status: int | None = None, cause: str = "") -> None:
        super().__init__(detail)
        self.status = status
        self.cause = cause or (f"HTTP {status}" if status is not None else "no answer")


#: ``think: false`` in Ollama's OpenAI-compatible terms, sent to every Ollama model
#: whatever the caller asked for: qwen3.5:9b on ``low`` spent a 1500-token budget
#: thinking and answered nothing, where ``none`` answered in 1.6 s (Ollama 0.34.4,
#: measured 2026-09-28). A 400 naming the field still drops it.
LOCAL_REASONING = "none"

# Free, no card, OpenAI-compatible (verified 2026-09-11). Ollama is the offline default.
_PROVIDERS: dict[str, JudgeConfig] = {
    "groq": JudgeConfig(),
    "gemini": JudgeConfig(
        provider="gemini",
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    # One local model for everything: the same one the fallback uses (spec section 11).
    "ollama": JudgeConfig(
        provider="ollama",
        model=LOCAL_JUDGE_MODEL,
        base_url="http://localhost:11434/v1",
        api_key_env="",
    ),
}


def provider_defaults(name: str) -> JudgeConfig:
    try:
        return _PROVIDERS[name]
    except KeyError:
        known = ", ".join(known_providers())
        raise JudgeError(f"unknown provider {name!r}; known: {known}") from None


def known_providers() -> tuple[str, ...]:
    """Every provider ``provider_defaults`` accepts, in the order its error lists them."""
    return tuple(sorted(_PROVIDERS))


def _no_key_detail(env_name: str) -> str:
    """The one wording for a missing key, shared by ``check`` and ``JudgeClient``."""
    return f"no API key: set {env_name} or add it to .env"


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    model: str
    latency_ms: int
    detail: str


def check(config: JudgeConfig, key: ApiKey | None, *, timeout: float = 30.0) -> CheckResult:
    """One tiny completion to prove the provider, model and key work together."""
    if config.api_key_env and key is None:
        return CheckResult(False, config.model, 0, _no_key_detail(config.api_key_env))
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


def load_prompt(name: str) -> string.Template:
    """The packaged prompt ``<name>.md`` as a ``string.Template``.

    Prompts live in files, not in code, so they can be read and diffed on their
    own. Each file is markdown with a ``## System`` and a ``## User`` section and
    uses ``$placeholder`` substitution, which leaves ``{`` and ``}`` free for the
    JSON example the model is asked to copy.
    """
    source = importlib.resources.files("proofpath.prompts") / f"{name}.md"
    return string.Template(source.read_text(encoding="utf-8"))


def estimate_tokens(text: str) -> int:
    """A deliberately rough token count: ~4 characters per token, plus overhead.

    Only used for budgeting batches and for filling in a ``usage`` block a
    provider did not send. Never presented as an exact figure.
    """
    return len(text) // 4 + 8


@dataclass(frozen=True)
class Completion:
    """One answer from the provider, with what it cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


@dataclass
class JudgeCost:
    """What a run spent on the judge. ``calls`` counts answers, not attempts."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    waited_s: float = 0.0
    model: str = ""
    #: Of ``calls``, the answers the local fallback gave (spec section 11). Counted
    #: apart because ``api_calls`` promises provider requests, and a report that
    #: folded local answers into it would overstate what the free tier was spent on.
    local_calls: int = 0


# Three attempts per request: Groq's free tier answers roughly once a minute, so a
# fourth try costs more waiting than it is worth (spec section 11).
_MAX_ATTEMPTS = 3
_SCHEMA_NAME = "proofpath_response"


def _backoff(attempt: int) -> float:
    return 0.5 * 2.0**attempt


def _retry_after_seconds(value: str) -> float | None:
    """``Retry-After`` in seconds: a bare number, or an HTTP date relative to now."""
    text = value.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


# A thinking model that ignores the "no thinking" request can still open its answer
# with its reasoning. Only a *leading*, *closed* block is taken off: anything else is
# the answer itself, and guessing at it would be worse than reporting it unreadable.
_THINK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
_CLOSE = "</think>"


def _strip_think(text: str) -> str:
    """The answer without a leading reasoning block.

    Some chat templates open the block in the prompt, so the answer carries only its
    end (``...reasoning...</think>{json}``). That orphan end is taken off only when
    the answer does not already start as JSON and JSON follows it: a valid answer
    that merely mentions ``</think>`` inside a string must reach the reader intact.
    """
    if _THINK.match(text):
        return _THINK.sub("", text, count=1)
    head = text.lstrip()[:1]
    end = text.find(_CLOSE)
    if end < 0 or head in ("{", "["):
        return text
    rest = text[end + len(_CLOSE) :].lstrip()
    return rest if rest[:1] in ("{", "[") else text


def _blames_response_format(response: httpx.Response) -> bool:
    """True when a 400 is about ``response_format``. The body is read, never printed."""
    body = response.text.lower()
    return "response_format" in body or "json_schema" in body


def _blames_reasoning(response: httpx.Response) -> bool:
    """True when a 400 names ``reasoning_effort``. The body is read, never printed.

    The *field*, not the word: a model that says it "has no reasoning mode" is
    refusing something else, and dropping the field on that evidence throws away the
    only thing keeping a reasoning model from spending its whole budget thinking --
    and buys a second request for a refusal that was never about the field.
    """
    return "reasoning_effort" in response.text.lower()


class JudgeClient:
    """One OpenAI-shaped ``chat/completions`` endpoint, retried and accounted for.

    Retries are bounded and quiet: a rate limit is waited out (honouring
    ``Retry-After`` up to ``max_wait``), a server error is backed off twice, and
    anything still failing raises ``JudgeUnavailable`` so the caller can report a
    missing second opinion instead of losing the run. Response bodies never reach
    a message or a log: providers echo the key and account details in them.
    """

    def __init__(
        self,
        config: JudgeConfig,
        key: ApiKey | None,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_wait: float = 120.0,
        timeout: float | httpx.Timeout = 60.0,
        fail_fast: bool = False,
        force_reasoning: str | None = None,
        retry_timeouts: bool = True,
    ) -> None:
        if config.api_key_env and key is None:
            raise JudgeError(_no_key_detail(config.api_key_env))
        self._config = config
        self._url = config.base_url.rstrip("/") + "/chat/completions"
        self._headers = {"User-Agent": f"proofpath/{__version__}"}
        if key is not None:
            self._headers["Authorization"] = f"Bearer {key.value}"
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._sleep = sleep
        self._max_wait = max_wait
        self._timeout = timeout
        # What this provider has already refused, latched for the client's whole life
        # rather than per call. A client outlives a batch -- 9.2 shares one across
        # every batch and the summary -- so a per-call flag would pay the same wasted
        # 400 again on each of them, against a per-minute tier with no room for it.
        self._schema_refused = False
        self._effort_refused = False
        #: With a fallback behind this client, waiting is the bug (Amendment B 1): the
        #: first failure raises at once, no retry and no ``Retry-After``. Public and
        #: mutable because ``FallbackClient`` only knows whether a fallback exists once
        #: it has asked Ollama, which it does on the first call, not at construction.
        self.fail_fast = fail_fast
        # Replaces whatever reasoning effort a caller asks for. The local fallback
        # sends ``none``: Ollama maps it to ``think: false``, the one value every
        # model accepts, where ``low`` is a 400 on a model that cannot think and a
        # budget spent thinking on one that can (measured on Ollama 0.34.4). Every
        # Ollama client gets it, the direct ``judge.provider ollama`` one included.
        if force_reasoning is None and config.provider == "ollama":
            force_reasoning = LOCAL_REASONING
        self._force_reasoning = force_reasoning
        # A local model that timed out after 180 s is loading or stuck; asking twice
        # more would hold the run for nine minutes (fix round 1, D4).
        self._retry_timeouts = retry_timeouts
        self.cost = JudgeCost(model=config.model)

    def __repr__(self) -> str:
        return f"JudgeClient(provider={self._config.provider!r}, model={self._config.model!r})"

    __str__ = __repr__

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def model(self) -> str:
        """The model as *configured*, not as the provider echoed it back.

        Cache keys are built from this, so a provider that answers with a dated
        alias one day and a bare name the next cannot split one model's judgements
        across two rows.
        """
        return self._config.model

    @property
    def timeout(self) -> float | httpx.Timeout:
        """How long one request may take before it counts as unanswered.

        Settable per client, because ``FallbackClient`` only learns that a fallback
        exists on the first call, and only then may it shorten the primary's wait.
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value: float | httpx.Timeout) -> None:
        self._timeout = value

    def close(self) -> None:
        """Release the HTTP client, but only one this client built for itself.

        An injected client belongs to the caller — 9.2 shares one across batches —
        so closing it here would break the next batch.
        """
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JudgeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        """One completion. Raises ``JudgeUnavailable`` when the provider will not answer.

        ``reasoning_effort`` caps what a reasoning model spends thinking. It has to be
        asked for, not assumed: such a model charges its reasoning to ``max_tokens``
        and, left to itself, hits the ceiling before it writes a word -- a 200 with
        ``finish_reason=length`` and no content, which costs the answer. Providers that
        do not know the field say so with a 400, and that is a downgrade, not a failure.

        Either downgrade, once learned, is remembered by the client: a provider does
        not change its mind between batches, so the first 400 is the only one paid.
        """
        response_format: dict[str, Any] | None = None
        if json_schema is not None:
            response_format = (
                {"type": "json_object"}
                if self._schema_refused
                else {
                    "type": "json_schema",
                    "json_schema": {"name": _SCHEMA_NAME, "schema": json_schema, "strict": True},
                }
            )
        if self._force_reasoning is not None:
            reasoning_effort = self._force_reasoning
        if self._effort_refused:
            reasoning_effort = None
        attempt = 0
        last = "no request was made"
        status: int | None = None
        cause = ""
        while attempt < _MAX_ATTEMPTS:
            payload: dict[str, Any] = {
                "model": self._config.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
            try:
                response = self._client.post(
                    self._url, json=payload, headers=self._headers, timeout=self._timeout
                )
            except httpx.HTTPError as exc:
                last = f"request failed: {type(exc).__name__}"
                status = None
                timed_out = isinstance(exc, httpx.TimeoutException)
                cause = "timeout" if timed_out else "connection failed"
                delay = _backoff(attempt)
                if timed_out and not self._retry_timeouts:
                    break
            else:
                status = response.status_code
                if status == 200:
                    return self._record(response, messages)
                # Never echo the body: providers put the key or account details in it.
                last = f"HTTP {status} from {self._url}"
                cause = f"HTTP {status}"
                if (
                    status == 400
                    and response_format is not None
                    and not self._schema_refused
                    and _blames_response_format(response)
                ):
                    # Some models take `json_object` but not a strict schema. One
                    # downgrade, and it deliberately does not spend a retry attempt:
                    # the brief counts the downgrade separately from the 3 attempts.
                    self._schema_refused = True
                    response_format = {"type": "json_object"}
                    continue
                if (
                    status == 400
                    and reasoning_effort is not None
                    and not self._effort_refused
                    and _blames_reasoning(response)
                ):
                    # Same shape as the ``response_format`` downgrade above, and for the
                    # same reason: a model that does not know the field is not a model
                    # that is down. It does not spend a retry attempt either.
                    self._effort_refused = True
                    reasoning_effort = None
                    continue
                if status == 429:
                    header = response.headers.get("Retry-After", "")
                    seconds = _retry_after_seconds(header) if header else None
                    # A bare 429 still means "slow down", so wait longer than a 5xx.
                    delay = 2.0 * 2.0**attempt if seconds is None else seconds
                elif status >= 500:
                    delay = _backoff(attempt)
                else:
                    break
            if self.fail_fast:
                # Checked after the two downgrades above, which ``continue`` before
                # reaching here: a model refusing a field is not a model that is down.
                break
            attempt += 1
            if attempt < _MAX_ATTEMPTS:
                self._wait(delay)
        raise JudgeUnavailable(last, status=status, cause=cause)

    def _wait(self, delay: float) -> None:
        delay = min(delay, self._max_wait)
        self.cost.waited_s += delay
        self._sleep(delay)

    def _record(self, response: httpx.Response, messages: list[dict[str, str]]) -> Completion:
        """Read the answer and add it to the running cost. Usage is estimated when absent."""
        try:
            body = response.json()
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            content = choice.get("message", {}).get("content")
            finish_reason = choice.get("finish_reason")
            text = _strip_think(content) if isinstance(content, str) else ""
            usage = body.get("usage") or {}
            # A missing count is estimated; an honest zero is kept exactly as sent.
            reported_prompt = usage.get("prompt_tokens")
            reported_completion = usage.get("completion_tokens")
            prompt_tokens = (
                sum(estimate_tokens(message.get("content", "")) for message in messages)
                if reported_prompt is None
                else int(reported_prompt)
            )
            completion_tokens = (
                estimate_tokens(text) if reported_completion is None else int(reported_completion)
            )
            model = str(body.get("model") or self._config.model)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # A 200 that is not a completion is a bot wall or a proxy page. Its shape
            # is not ours to guess at, and its body is never repeated back.
            raise JudgeUnavailable(
                f"unreadable 200 response from {self._url}", status=200, cause="unreadable answer"
            ) from None
        if not text:
            # No choices, or a choice with no text: a refusal, a cut-off answer or a
            # wall, never an empty opinion. It costs the caller a `JudgeUnavailable`,
            # not a silent `Completion("")`, and it does not count as an answer.
            # `finish_reason` is a fixed provider token, so it may be named; the body
            # may not.
            why = f" (finish_reason={finish_reason})" if isinstance(finish_reason, str) else ""
            raise JudgeUnavailable(
                f"no completion in the 200 response from {self._url}{why}",
                status=200,
                cause="empty answer",
            )
        self.cost.calls += 1
        self.cost.prompt_tokens += prompt_tokens
        self.cost.completion_tokens += completion_tokens
        self.cost.model = model
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )


# --- the local fallback (spec section 11) -----------------------------------------

#: The switch notices (spec section 11; the cause added in fix round 1). ``provider``
#: is the name as a reader knows it (``Groq``), ``model`` the Ollama tag the run
#: switched to. Every failure switches; the notice says which one it was, so a wrong
#: key is never mistaken for a busy free tier.
_LIMIT_HEAD = "{provider} limit reached"
_KEY_HEAD = "{provider} rejected the key (HTTP {status})"
_DOWN_HEAD = "{provider} did not answer ({cause})"
_JUDGING = " — judging with local ollama {model}"
JUDGE_FALLBACK_LIMIT = _LIMIT_HEAD + _JUDGING
JUDGE_FALLBACK_KEY = _KEY_HEAD + _JUDGING
JUDGE_FALLBACK_DOWN = _DOWN_HEAD + _JUDGING
#: What ``switched`` says once the local model has failed too: the primary's reason,
#: then the local one. "Judging with local ollama" would then be false (final review).
JUDGE_FALLBACK_FAILED = "{head}, and local ollama {model} failed too ({cause})"
#: The install/pull hint: offered once, beside the original error, when there is no
#: fallback for a reason the user can fix by adding the local model -- never when
#: they chose ``judge.fallback = off`` themselves, and never a notice of a switch
#: that did not happen (``JudgeClient.hint`` is its own attribute, apart from
#: ``switched``, so the two states are never conflated). ``{model}`` is the fallback
#: as configured (``judge.fallback``, ``qwen3.5:9b`` unless the user named another);
#: ``{provider}`` the primary's display name, same as the switch notices above.
JUDGE_FALLBACK_HINT_UNREACHABLE = (
    "install Ollama (https://ollama.com) and run: ollama pull {model} — "
    "the judge then keeps going locally when {provider} runs out"
)
JUDGE_FALLBACK_HINT_MISSING = (
    "run: ollama pull {model} — the judge then keeps going locally when {provider} runs out"
)
#: The statuses that mean "over the provider's limit", read wherever a limit is named
#: (the switch notice, the search's fallback notice), so both classify it one way.
LIMIT_STATUSES = frozenset({429, 413})  # Groq answers 413 when a request busts TPM
_KEY_STATUSES = frozenset({401, 403})
#: ``judge.fallback``: the Ollama model to switch to (``LOCAL_JUDGE_MODEL`` unless the
#: user named another), or ``off``, in any case, for no fallback.
FALLBACK_OFF = "off"
#: The first local call loads the model into memory, which can take tens of seconds
#: on first use, so the local client waits far longer than an API client does.
LOCAL_TIMEOUT = 180.0
#: How long Ollama keeps the model loaded after the warm-up. Sent on the native
#: ``/api/generate`` route because the OpenAI-compatible one ignores ``keep_alive``
#: (Ollama 0.34.4, measured 2026-09-28); later requests keep the session's value.
LOCAL_KEEP_ALIVE = "30m"
#: ``/api/tags`` is a local listing; a server that takes longer than this is not
#: one to hand a run to.
TAGS_TIMEOUT = 2.0
#: The primary's wait once a fallback exists: a host that does not even accept the
#: connection within five seconds is not going to answer, and the local model is.
PRIMARY_CONNECT_TIMEOUT = 5.0
PRIMARY_READ_TIMEOUT = 60.0
#: How long a thread that did not switch waits for the switching thread's notice and
#: warm-up: the warm-up's own timeout plus a margin, so its request never starts its
#: read timer while the model is still loading, and a hung warm-up cannot hold it
#: for ever.
WARM_WAIT_S = LOCAL_TIMEOUT + 5.0
# The local failures that mean the model is gone for this run: no answer at all
# (``status`` None: refused, reset, timed out), a server error, or no such model.
# A cut-off, unreadable or refused *answer* is about that one request, not the model.
_DEAD_STATUSES = frozenset({404})

_DISPLAY_NAMES = {"groq": "Groq", "gemini": "Gemini", "ollama": "Ollama"}
# Why there is no fallback, in the words the error carries after "no local fallback:".
_WHY_OFF = "judge.fallback is off"


def _ollama_url() -> str:
    return provider_defaults("ollama").base_url


def _ollama_root(base_url: str) -> str:
    """The server root: the native API lives beside ``/v1``, not under it."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _display(provider: str) -> str:
    return _DISPLAY_NAMES.get(provider, provider.capitalize())


def _tags(base_url: str, *, timeout: float) -> list[str] | None:
    """The installed model names, or ``None`` when Ollama could not be asked."""
    try:
        response = httpx.get(f"{_ollama_root(base_url)}/api/tags", timeout=timeout)
        if response.status_code != 200:
            return None
        models = response.json().get("models") or []
        return [str(item["name"]) for item in models if isinstance(item, dict) and "name" in item]
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        return None


def _installed(model: str, names: Sequence[str]) -> bool:
    """Whether ``model`` is among ``names``; an untagged name means its ``:latest``."""
    wanted = model.strip().lower()
    if ":" not in wanted:
        wanted += ":latest"
    return any(name.lower() == wanted for name in names)


def installed_models(base_url: str, *, timeout: float) -> list[str]:
    """``GET {root}/api/tags``: what Ollama has installed. ``[]`` on any error."""
    return _tags(base_url, timeout=timeout) or []


def local_client(model: str, base_url: str | None = None) -> JudgeClient:
    """The fallback's client: Ollama's OpenAI route, no key, thinking off, a timeout
    long enough for the first call to load the model, and no second try after it."""
    settings = replace(provider_defaults("ollama"), model=model)
    if base_url is not None:
        settings = replace(settings, base_url=base_url)
    return JudgeClient(
        settings,
        None,
        timeout=LOCAL_TIMEOUT,
        force_reasoning=LOCAL_REASONING,
        retry_timeouts=False,
    )


def warm_local(base_url: str, model: str) -> None:
    """Load ``model`` and keep it loaded for ``LOCAL_KEEP_ALIVE``. Never raises.

    An empty ``/api/generate`` is Ollama's documented "load this model" request. A
    warm-up that fails costs nothing: the chat request after it loads the model
    itself, and says so if it cannot.
    """
    url = f"{_ollama_root(base_url)}/api/generate"
    try:
        httpx.post(
            url, json={"model": model, "keep_alive": LOCAL_KEEP_ALIVE}, timeout=LOCAL_TIMEOUT
        )
    except httpx.HTTPError:
        return


@dataclass(frozen=True)
class _Plan:
    """Whether this run has a fallback: the model, or why there is none."""

    model: str | None
    why: str = ""
    #: The hint template for ``why``, or ``None`` when there is nothing to suggest
    #: (a fallback, or the user's own ``off``). Formatted with the model and provider
    #: only once the primary has actually failed (``_switch``): asking Ollama is
    #: already paid for here, but the hint is only worth showing beside a real error.
    hint: str | None = None


class FallbackClient:
    """A primary ``JudgeClient`` that switches, once and at once, to a local model.

    The same surface ``Judge`` uses from ``JudgeClient``. The first call decides
    whether a fallback exists (``judge.fallback``, and one cheap ``/api/tags`` to
    confirm that model is installed). With one, the primary fails fast and connects within
    ``PRIMARY_CONNECT_TIMEOUT``: its first failure of any kind raises, the notice
    goes out naming the cause, the model is warmed up and the very request that
    failed is sent to it, so the caller sees one completion. Every later call goes
    straight to the local model. Without a fallback, the primary keeps its retries
    and its error says why no fallback was used. A local model that fails too is
    latched dead for the run: every later call raises at once with the same detail.

    The switch is sticky for the life of the client, which is one run: whatever
    builds a judge for a run builds a new client or calls :meth:`reset`.
    """

    def __init__(
        self,
        primary: JudgeClient,
        *,
        fallback: str = LOCAL_JUDGE_MODEL,
        on_switch: Callable[[str], None] | None = None,
        ollama_url: str | None = None,
        local_factory: Callable[[str, str], JudgeClient] = local_client,
        warm: Callable[[str, str], None] | None = warm_local,
    ) -> None:
        self._primary = primary
        self._primary_timeout = primary.timeout
        self._fallback = fallback.strip()
        self._ollama_url = ollama_url or _ollama_url()
        self._local_factory = local_factory
        self._warm = warm
        self.on_switch = on_switch
        #: The notice, once the switch has happened; ``None`` before.
        self.switched: str | None = None
        #: The install/pull hint, once raised for this run; ``None`` before, and
        #: ``None`` for good when there was nothing to suggest. Apart from
        #: ``switched``: a hint is offered instead of a switch, never alongside one,
        #: and the two must never be read as the same thing.
        self.hint: str | None = None
        # Worker threads share one client (the TUI's summary, a run's stages): the
        # plan and the switch go through this lock so that exactly one thread
        # switches and every other one sees the result.
        self._lock = threading.Lock()
        # Set once the switching thread has handed the notice over and the warm-up
        # has finished (or failed). A thread that finds the switch already made waits
        # for it, so no local request starts before the notice is out, and none starts
        # its read timer while the model is still loading.
        self._ready = threading.Event()
        self._plan: _Plan | None = None
        self._local: JudgeClient | None = None
        self._primary_detail = ""
        # The primary's failure and the notice it earned, kept for the moment the
        # local model fails too and ``switched`` has to say so.
        self._primary_failure: JudgeUnavailable | None = None
        self._notice: str | None = None
        #: The combined detail once the local model has failed too; ``None`` before.
        self._dead: str | None = None
        # Local answers from before a ``reset``: the client is gone, the spend is not.
        self._spent = JudgeCost()

    def __repr__(self) -> str:
        return (
            f"FallbackClient(primary={self._primary!r}, fallback={self._fallback!r}, "
            f"switched={self.switched is not None})"
        )

    __str__ = __repr__

    @property
    def provider(self) -> str:
        return "ollama" if self._local is not None else self._primary.provider

    @property
    def model(self) -> str:
        return self._local.model if self._local is not None else self._primary.model

    @property
    def cost(self) -> JudgeCost:
        """Primary and local spend together; ``local_calls`` says which were local."""
        primary = self._primary.cost
        parts = [primary, self._spent]
        if self._local is not None:
            parts.append(self._local.cost)
        local = self._spent.calls + (self._local.cost.calls if self._local is not None else 0)
        return JudgeCost(
            calls=sum(part.calls for part in parts),
            prompt_tokens=sum(part.prompt_tokens for part in parts),
            completion_tokens=sum(part.completion_tokens for part in parts),
            waited_s=sum(part.waited_s for part in parts),
            model=self._local.cost.model if self._local is not None else primary.model,
            local_calls=local,
        )

    def reset(self) -> None:
        """Back to the primary, for a new run: the notice, the plan and a dead local
        model are all forgotten."""
        with self._lock:
            if self._local is not None:
                spent = self._local.cost
                self._spent.calls += spent.calls
                self._spent.prompt_tokens += spent.prompt_tokens
                self._spent.completion_tokens += spent.completion_tokens
                self._spent.waited_s += spent.waited_s
                self._local.close()
            self._local = None
            self._plan = None
            self._dead = None
            self.switched = None
            self.hint = None
            self._primary_detail = ""
            self._primary_failure = None
            self._notice = None
            self._ready = threading.Event()
            self._primary.fail_fast = False
            self._primary.timeout = self._primary_timeout

    def close(self) -> None:
        self._primary.close()
        if self._local is not None:
            self._local.close()

    def __enter__(self) -> FallbackClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        """One completion, from the primary or, once it has failed, from Ollama."""

        def ask(client: JudgeClient) -> Completion:
            return client.complete(
                messages,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

        local = self._route()
        if local is None:
            try:
                return ask(self._primary)
            except JudgeUnavailable as exc:
                local, notice = self._switch(exc)
            if notice is not None:
                try:
                    self._announce(notice)
                    if self._warm is not None:
                        # After the notice, before the request: the user reads the
                        # line while the model loads (Amendment B 4). Outside the lock.
                        self._warm(self._ollama_url, local.model)
                finally:
                    self._ready.set()
        # A thread that did not switch waits for the notice and the warm-up, bounded
        # by the warm-up's own timeout (fix round 2).
        self._ready.wait(timeout=WARM_WAIT_S)
        if self._dead is not None:
            raise JudgeUnavailable(self._dead)
        try:
            answer = ask(local)
        except JudgeUnavailable as exc:
            # The primary's reason first: it is what the run switched away from
            # (spec section 11).
            detail = f"{self._primary_detail}; local ollama {local.model}: {exc}"
            # "Judging with local ollama" is no longer true, so the notice the report
            # carries says both failed instead (final review, minor).
            self._both_failed(local.model, exc)
            if exc.status is None or exc.status >= 500 or exc.status in _DEAD_STATUSES:
                # Latched: a dead or stuck Ollama costs this run one wait, not one per
                # batch. Anything else failed this request only (fix round 2).
                self._dead = detail
            raise JudgeUnavailable(detail, status=exc.status, cause=exc.cause) from None
        if self._notice is not None:
            # Answering again after a failure that was about one request only.
            self.switched = self._notice
        return answer

    def _both_failed(self, model: str, failure: JudgeUnavailable) -> None:
        """Say in ``switched`` that the local model failed too, after the primary."""
        primary = self._primary_failure
        if primary is None:  # pragma: no cover - a local client exists only after a switch
            return
        head = _head(self._primary.provider, primary)
        self.switched = JUDGE_FALLBACK_FAILED.format(head=head, model=model, cause=failure.cause)

    def _announce(self, notice: str) -> None:
        """Hand the notice to the listener, once, and never at the answer's expense."""
        if self.on_switch is not None:
            # A listener that raises costs the line, not the batch (fix round 1).
            with suppress(Exception):
                self.on_switch(notice)

    def _route(self) -> JudgeClient | None:
        """The local client once switched; otherwise ``None``, with the plan made."""
        with self._lock:
            if self._local is not None:
                return self._local
            if self._plan is None:
                self._plan = self._choose()
                if self._plan.model is not None:
                    # Decided before the first request (Amendment B 1): with a
                    # fallback the primary must not wait, neither on a retry nor on a
                    # host that does not answer the connect. Without one it keeps
                    # today's retries and timeout.
                    self._primary.fail_fast = True
                    self._primary.timeout = httpx.Timeout(
                        PRIMARY_READ_TIMEOUT, connect=PRIMARY_CONNECT_TIMEOUT
                    )
            return None

    def _choose(self) -> _Plan:
        """The configured model, once ``/api/tags`` confirms it is installed.

        The tags are asked only to confirm, never to choose: the order is fixed
        (the configured API, then this model) and nothing is ever pulled (rule 5).
        """
        setting = self._fallback
        if not setting or setting.lower() == FALLBACK_OFF:
            return _Plan(None, _WHY_OFF)
        names = _tags(self._ollama_url, timeout=TAGS_TIMEOUT)
        if names is None:
            return _Plan(
                None,
                f"ollama is not reachable at {_ollama_root(self._ollama_url)}",
                hint=JUDGE_FALLBACK_HINT_UNREACHABLE,
            )
        if not _installed(setting, names):
            return _Plan(None, f"{setting} is not installed", hint=JUDGE_FALLBACK_HINT_MISSING)
        return _Plan(setting)  # as written: the user's own spelling of the name

    def _switch(self, failure: JudgeUnavailable) -> tuple[JudgeClient, str | None]:
        """Switch once: the local client, and the notice when *this* call switched.

        Raises the primary's error, with the reason, when there is no fallback. When
        that reason is one the user can fix -- Ollama unreachable, the model not
        installed, never ``judge.fallback = off`` -- the hint goes out through the
        same listener the switch notice uses, once per run (the ``self.hint is None``
        guard: this branch runs again on every later call, since without a local
        client nothing ever leaves ``_route`` early). The detail the error carries is
        unchanged; the hint travels beside it, never inside it.

        The notice (or hint) is announced after the lock is released, same as a
        switch: the listener runs after the lock is released, so a slow or blocking
        listener can never hold another thread's switch (fix round 1, C).
        """
        hint: str | None = None
        with self._lock:
            if self._local is not None:
                return self._local, None  # another thread switched while we waited
            plan = self._plan
            if plan is None or plan.model is None:
                why = _WHY_OFF if plan is None else plan.why
                if plan is not None and plan.hint is not None and self.hint is None:
                    hint = plan.hint.format(
                        model=self._fallback, provider=_display(self._primary.provider)
                    )
                    self.hint = hint
                error = JudgeUnavailable(
                    f"{failure}; no local fallback: {why}",
                    status=failure.status,
                    cause=failure.cause,
                )
            else:
                notice = _notice(self._primary.provider, plan.model, failure)
                self._primary_detail = str(failure)
                self._primary_failure = failure
                self._notice = notice
                self._local = self._local_factory(plan.model, self._ollama_url)
                self.switched = notice
                return self._local, notice
        if hint is not None:
            self._announce(hint)
        raise error from None


def _head(provider: str, failure: JudgeUnavailable) -> str:
    """Why the primary was left, in the words every notice opens with (fix round 1, B)."""
    name = _display(provider)
    if failure.status in LIMIT_STATUSES:
        return _LIMIT_HEAD.format(provider=name)
    if failure.status in _KEY_STATUSES:
        return _KEY_HEAD.format(provider=name, status=failure.status)
    return _DOWN_HEAD.format(provider=name, cause=failure.cause)


def _notice(provider: str, model: str, failure: JudgeUnavailable) -> str:
    """The switch notice: why the primary was left, and who judges now."""
    return _head(provider, failure) + _JUDGING.format(model=model)


def build_client(
    config: JudgeConfig,
    key: ApiKey | None,
    *,
    on_switch: Callable[[str], None] | None = None,
) -> JudgeClient | FallbackClient:
    """The judge's client for one run: the configured provider, with the local
    fallback behind it unless the provider already is Ollama (spec section 11)."""
    primary = JudgeClient(config, key)
    if config.provider == "ollama":
        return primary
    return FallbackClient(primary, fallback=config.fallback, on_switch=on_switch)


# --- the batching judge -----------------------------------------------------------


@dataclass(frozen=True)
class JudgeOpinion:
    """What the judge thought of one claim. Never a verdict: a verdict is local.

    ``model`` is the judge's full name (``"groq openai/gpt-oss-120b"``), so a report
    that carries the opinion also carries who gave it.
    """

    label: Label
    rationale: str
    model: str


@dataclass(frozen=True)
class JudgeItem:
    """One escalated verdict, as the prompt sees it.

    ``verdict`` and ``tier`` are the local answer, handed over as a second opinion
    the model may disagree with -- the template says so in as many words.
    """

    id: str
    claim: str
    passage: str
    verdict: Label
    tier: Tier


@dataclass(frozen=True)
class JudgedClaim:
    """What the judge wrote for one claim: search queries, and the claim in English."""

    queries: tuple[str, ...]
    english: str  # "" when the model gave none


# 20 claims per prompt, and a prompt under 3.5k tokens: Groq's free tier caps at 8k
# tokens per minute and the cap below counts the *prompt* only, so it has to leave
# room for ``_REVIEW_TOKENS`` on top of it -- 3500 + 4096 still fits under 8k, where
# a 7k prompt plus the same answer budget would buy nothing but a 429 (spec 11).
BATCH_SIZE = 20
TOKEN_CAP = 3500

# Reasoning models (``openai/gpt-oss-120b``, the default) charge their thinking to
# ``max_tokens`` before writing anything, so an answer budget sized for the answer
# alone comes back empty with ``finish_reason=length``. Both calls therefore ask for
# the cheapest thinking the provider offers and leave room for it: 20 opinions of a
# sentence each for a review, a few sentences for a summary.
_REASONING_EFFORT = "low"
_REVIEW_TOKENS = 4096
_SUMMARY_TOKENS = 1500

# Groq is the default because it does not train on submitted prompts. Gemini does,
# for free-tier traffic outside the EEA, the UK and Switzerland, so choosing it says
# so once, out loud (spec section 11).
GEMINI_DATA_USE = (
    "gemini trains on free-tier prompts outside the EEA, UK and Switzerland; "
    "use another provider for confidential drafts"
)

# The shape the answer has to come back in. ``strict`` is asked for; a provider that
# refuses it is downgraded by ``JudgeClient`` and the parser stays lenient anyway.
_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "opinions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string", "enum": [label.value for label in Label]},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "label", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["opinions"],
    "additionalProperties": False,
}

# Evidence search (OPEN-ITEMS 17.1a): the judge writes *queries*, never an answer.
# Deciding a claim without a passage is the truth oracle spec section 3 forbids.
#
# The answer budget grows with the batch (final review). ``gpt-oss-120b`` charges its
# reasoning to ``max_tokens`` even at low effort; an item -- the claim in English and
# two queries -- is ~170 tokens (``estimate_tokens``, a 300-character claim), so a
# flat 1500 left five claims ~650 tokens to think in, and a pass that ran over came
# back ``finish_reason=length`` with every query and translation lost. 1500 for the
# thinking plus 250 per claim gives five claims 2750, and ~840 of prompt beside it
# still fits Groq's 8k tokens a minute. The ceiling keeps a large ``search.max_claims``
# under that budget too.
_QUERIES_REASONING_TOKENS = 1500
_QUERIES_TOKENS_PER_CLAIM = 250
_QUERIES_MAX_TOKENS = 5000
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
                    "english": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "english", "queries"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_SYSTEM_HEADING = "## System"
_USER_HEADING = "## User"


def _queries_budget(count: int) -> int:
    """``max_tokens`` for one queries call over ``count`` claims."""
    return min(_QUERIES_MAX_TOKENS, _QUERIES_REASONING_TOKENS + _QUERIES_TOKENS_PER_CLAIM * count)


def _split_prompt(text: str) -> tuple[str, str]:
    """A packaged prompt's ``## System`` and ``## User`` halves, as two messages.

    The templates are written as one readable markdown file, but a chat endpoint
    wants the standing instruction and the request separately: a system message is
    what providers weight as policy, and folding it into the user turn quietly
    demotes the rules the judge is held to.
    """
    _, _, rest = text.partition(_SYSTEM_HEADING)
    system, _, user = rest.partition(_USER_HEADING)
    return system.strip(), user.strip()


def _one_line(text: str) -> str:
    """Whitespace folded, so one item stays one block the model can count."""
    return " ".join(text.split())


class Judge:
    """The escalation layer: batches of low-confidence verdicts, one opinion each.

    Three properties are the whole point of this class, and none of them is about
    accuracy:

    * It never changes anything. ``review`` returns opinions keyed by item id and
      the caller attaches them beside the local verdicts; nothing here can reach a
      ``Verdict`` (spec section 11.1).
    * It never raises. A provider that goes away, an answer that is not JSON, an id
      nobody asked about -- each one costs an opinion, never the run. What was
      gathered before the trouble is returned, and ``unavailable``/``skipped`` say
      what was lost so the report can too (product rule 6).
    * It never spends more than it was told to. Batches are capped by item count and
      by an estimated token budget, because the free tier that makes this feature
      usable is a tokens-per-minute one.
    """

    def __init__(
        self,
        client: JudgeClient | FallbackClient,
        *,
        batch_size: int = BATCH_SIZE,
        token_cap: int = TOKEN_CAP,
    ) -> None:
        self._client = client
        self._batch_size = max(1, batch_size)
        self._token_cap = token_cap
        self._template = load_prompt("review")
        self._summary_template = load_prompt("summarize")
        self._queries_template = load_prompt("queries")
        #: One line per opinion that could not be used, in the order they arrived.
        self.skipped: list[str] = []
        #: True once the provider stopped answering; ``detail`` says why, and
        #: ``status`` is the HTTP status of that failure (``None`` when there was none).
        self.unavailable = False
        self.detail = ""
        self.status: int | None = None

    @property
    def name(self) -> str:
        """``"groq openai/gpt-oss-120b"``: who the second opinion is from."""
        return f"{self._client.provider} {self._client.model}"

    @property
    def cost(self) -> JudgeCost:
        """What the judge has spent so far. The client keeps the running total."""
        return self._client.cost

    @property
    def switched(self) -> str | None:
        """The local-fallback notice once this run has switched, else ``None``."""
        return self._client.switched if isinstance(self._client, FallbackClient) else None

    @property
    def hint(self) -> str | None:
        """The install/pull hint this run has raised, once there is one, else
        ``None``. Distinct from ``switched``: a hint means nothing switched."""
        return self._client.hint if isinstance(self._client, FallbackClient) else None

    def listen(self, on_switch: Callable[[str], None] | None) -> None:
        """Point the switch notice at whoever is watching this run.

        Set per run rather than when the judge is built: the judge is built before
        the run's listener exists, and a stage that runs later (the summary) may
        report to another one.
        """
        if isinstance(self._client, FallbackClient):
            self._client.on_switch = on_switch

    def reset(self) -> None:
        """A new run starts on the configured provider again (Amendment B 9)."""
        if isinstance(self._client, FallbackClient):
            self._client.reset()

    def close(self) -> None:
        self._client.close()

    def review(
        self,
        items: Sequence[JudgeItem],
        *,
        on_batch: Callable[[int, int], None] | None = None,
        into: dict[str, JudgeOpinion] | None = None,
    ) -> dict[str, JudgeOpinion]:
        """One opinion per item, keyed by item id; missing ids simply have none.

        ``on_batch(done, total)`` is called after each batch, with items counted
        rather than calls: it is where a run draws progress and checks whether it has
        been told to stop. An exception raised *there* belongs to the caller and is
        let through -- it is the only way out of this method that is not a return.

        ``into`` is the dictionary to fill, for a caller that has to survive its own
        ``on_batch`` raising: a cancelled run still holds every opinion the batches
        before the cancel paid for, which is what keeps a stopped run from looking
        like a run that learned nothing (product rule 6).
        """
        # Per call, not per instance: a ``Judge`` reused for a second document must
        # not report the first one's dropped opinions, and a provider that has come
        # back up must not stay "unavailable" because it once was not.
        self.skipped = []
        self.unavailable = False
        self.detail = ""
        self.status = None
        opinions: dict[str, JudgeOpinion] = {} if into is None else into
        total = len(items)
        done = 0
        for batch in self._batches(items):
            try:
                completion = self._client.complete(
                    self._messages(batch),
                    json_schema=_REVIEW_SCHEMA,
                    max_tokens=_REVIEW_TOKENS,
                    reasoning_effort=_REASONING_EFFORT,
                )
            except JudgeUnavailable as exc:
                self._give_up(str(exc), status=exc.status)
                break
            except Exception as exc:
                # A bug in the adapter, a library that changed under us: whatever it
                # is, it is not worth a run. The type is named, the message is not:
                # a provider's exception can carry the key.
                self._give_up(f"{type(exc).__name__} from the judge client")
                break
            self._collect(batch, completion, opinions)
            done += len(batch)
            if on_batch is not None:
                on_batch(done, total)
        return opinions

    def summarize(self, report_markdown: str, *, max_tokens: int = _SUMMARY_TOKENS) -> str:
        """The finished report in a few plain sentences, or ``""`` when nobody answered.

        Exactly one call, and the report is its only input (spec section 11.1): it
        runs after every verdict is settled, so there is nothing here that could
        reach one. Plain prose, so no JSON schema is asked for.

        Trouble costs the paragraph, never the run, for the same reason ``review``
        never raises -- and it is reported rather than swallowed: ``unavailable`` and
        ``detail`` say what happened, so the caller prints the silence instead of an
        empty summary that would read as "nothing worth saying" (product rule 2).
        """
        # Per call, exactly as ``review`` resets: a provider that was down for the
        # escalation and came back for the summary must not be reported as down, and
        # one that goes down now must not be hidden by an earlier success.
        self.unavailable = False
        self.detail = ""
        self.status = None
        rendered = self._summary_template.substitute(report=report_markdown)
        system, user = _split_prompt(rendered)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            completion = self._client.complete(
                messages, max_tokens=max_tokens, reasoning_effort=_REASONING_EFFORT
            )
        except JudgeUnavailable as exc:
            self._give_up(str(exc), status=exc.status)
            return ""
        except Exception as exc:
            # As in ``review``: the type is named, the message is not, because a
            # provider's own exception can carry the key.
            self._give_up(f"{type(exc).__name__} from the judge client")
            return ""
        text = completion.text.strip()
        if not text:
            # A 200 whose content is whitespace is not a summary with nothing to say.
            # ``JudgeClient`` already refuses an empty one; this catches the blank that
            # is technically a string, so every silence takes the one reported path.
            self._give_up("empty answer")
            return ""
        return text

    def queries(self, claims: Sequence[str]) -> dict[int, JudgedClaim]:
        """One or two search queries per claim, and the claim in English, keyed by
        its position; ``{}`` on trouble.

        It never raises, like ``review``: a provider that is down or out of quota
        costs the written queries and the translation, and ``unavailable``/``detail``
        say why, so the run can fall back to the sentence itself and tell the reader
        (OPEN-ITEMS 17.1a). The English text is what Amendment A hands the check
        models, which were trained on English alone.
        """
        self.unavailable = False
        self.detail = ""
        self.status = None
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
                max_tokens=_queries_budget(len(claims)),
                reasoning_effort=_REASONING_EFFORT,
            )
        except JudgeUnavailable as exc:
            # The status travels with the detail: the search names a limit from the
            # status alone, never from a substring of the detail (final review).
            self._give_up(str(exc), status=exc.status)
            return {}
        except Exception as exc:
            # The type is named, the message is not: it can carry the key.
            self._give_up(f"{type(exc).__name__} from the judge client")
            return {}
        return _queries_from(completion.text, len(claims))

    def _give_up(self, detail: str, *, status: int | None = None) -> None:
        self.unavailable = True
        self.detail = detail
        self.status = status

    def _batches(self, items: Sequence[JudgeItem]) -> list[list[JudgeItem]]:
        """Items packed to the smaller of the item cap and the token budget.

        A single item that busts the budget on its own is still sent: a batch of one
        is the smallest prompt there is, and dropping it would silently lose a
        verdict the run was told to escalate.
        """
        batches: list[list[JudgeItem]] = []
        batch: list[JudgeItem] = []
        for item in items:
            candidate = [*batch, item]
            if batch and (len(candidate) > self._batch_size or self._over_budget(candidate)):
                batches.append(batch)
                batch = [item]
            else:
                batch = candidate
        if batch:
            batches.append(batch)
        return batches

    def _over_budget(self, batch: Sequence[JudgeItem]) -> bool:
        """The whole prompt, template and all, against the budget -- not the items
        alone. The template is most of a small batch, and a budget that ignored it
        would send prompts the provider counts as over its per-minute cap."""
        rendered = sum(estimate_tokens(message["content"]) for message in self._messages(batch))
        return rendered > self._token_cap

    def _messages(self, batch: Sequence[JudgeItem]) -> list[dict[str, str]]:
        rendered = self._template.substitute(items="\n\n".join(_render(i) for i in batch))
        system, user = _split_prompt(rendered)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _collect(
        self,
        batch: Sequence[JudgeItem],
        completion: Completion,
        into: dict[str, JudgeOpinion],
    ) -> None:
        """Read one answer leniently: every usable opinion is kept, the rest noted."""
        payload = _loads(completion.text)
        if payload is None:
            ids = ", ".join(item.id for item in batch)
            self.skipped.append(f"the judge's answer was not JSON; {len(batch)} items ({ids})")
            return
        known = {item.id for item in batch}
        for entry in payload:
            if not isinstance(entry, dict):
                self.skipped.append(f"an opinion was {type(entry).__name__}, not an object")
                continue
            ident = str(entry.get("id", ""))
            if ident not in known:
                self.skipped.append(f"an opinion named {ident!r}, which was not in the batch")
                continue
            if ident in into:
                self.skipped.append(f"{ident} was answered twice; the second was dropped")
                continue
            try:
                label = Label(str(entry.get("label", "")).strip().upper())
            except ValueError:
                self.skipped.append(f"{ident} came back labelled {entry.get('label')!r}")
                continue
            rationale = _one_line(str(entry.get("rationale", "")))
            if not rationale:
                # Rule 1 applies to the judge too: no quoted reason, no opinion.
                self.skipped.append(f"{ident} came back with no rationale")
                continue
            into[ident] = JudgeOpinion(label=label, rationale=rationale, model=self.name)


def _render(item: JudgeItem) -> str:
    """One item as the template's ``$items`` block expects it."""
    return (
        f"- id: {item.id}\n"
        f"  claim: {_one_line(item.claim)}\n"
        f"  passage: {_one_line(item.passage)}\n"
        f"  verdict: {item.verdict.value} ({item.tier})"
    )


def _loads(text: str) -> list[Any] | None:
    """The ``opinions`` list out of an answer, or ``None`` when there is none.

    Models fence their JSON, prefix it with "Sure!", or answer prose. The outermost
    braces are tried once before giving up, which recovers a fenced answer without
    guessing at anything a stricter reader would refuse.
    """
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("opinions"), list):
            return list(payload["opinions"])
    return None


def _queries_from(text: str, count: int) -> dict[int, JudgedClaim]:
    """The ``items`` of a queries answer, read leniently; an id out of range is dropped."""
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            continue
        found: dict[int, JudgedClaim] = {}
        for entry in payload["items"]:
            if not isinstance(entry, dict):
                continue
            try:
                ident = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= ident < count:
                continue
            raw = entry.get("queries")
            written = (
                tuple(
                    _one_line(query)[:_QUERY_CHARS]
                    for query in raw
                    if isinstance(query, str) and query.strip()
                )[:_QUERIES_PER_CLAIM]
                if isinstance(raw, list)
                else ()
            )
            english = _one_line(str(entry.get("english", "")))[:_QUERY_CHARS]
            if written or english:
                found[ident] = JudgedClaim(queries=written, english=english)
        return found
    return {}


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start and (start, end + 1) != (0, len(stripped)):
        return [stripped, stripped[start : end + 1]]
    return [stripped]
