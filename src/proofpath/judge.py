"""Optional LLM judge: provider settings, API key resolution, prompts, HTTP client.

Every supported provider speaks the OpenAI ``chat/completions`` shape, so one
adapter covers all of them (see ``docs/research/2026-09-11-free-llm-api-tiers.md``).
This module finds a key without ever printing it, renders the packaged prompt
templates, and makes one retried, cost-accounted call at a time. Batching and
escalation sit on top of it, in the pipeline.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import string
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from proofpath import __version__
from proofpath.config import JudgeConfig
from proofpath.models import Label, Tier
from proofpath.paths import config_dir


class JudgeError(ValueError):
    """Bad judge settings."""


class JudgeUnavailable(JudgeError):  # noqa: N818 - reads as a state, not as a failed run
    """The provider did not answer after retries. A run reports this, never fails on it."""


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
        timeout: float = 60.0,
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
        if self._effort_refused:
            reasoning_effort = None
        attempt = 0
        last = "no request was made"
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
                delay = _backoff(attempt)
            else:
                status = response.status_code
                if status == 200:
                    return self._record(response, messages)
                # Never echo the body: providers put the key or account details in it.
                last = f"HTTP {status} from {self._url}"
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
            attempt += 1
            if attempt < _MAX_ATTEMPTS:
                self._wait(delay)
        raise JudgeUnavailable(last)

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
            text = content if isinstance(content, str) else ""
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
            raise JudgeUnavailable(f"unreadable 200 response from {self._url}") from None
        if not text:
            # No choices, or a choice with no text: a refusal, a cut-off answer or a
            # wall, never an empty opinion. It costs the caller a `JudgeUnavailable`,
            # not a silent `Completion("")`, and it does not count as an answer.
            # `finish_reason` is a fixed provider token, so it may be named; the body
            # may not.
            why = f" (finish_reason={finish_reason})" if isinstance(finish_reason, str) else ""
            raise JudgeUnavailable(f"no completion in the 200 response from {self._url}{why}")
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

_SYSTEM_HEADING = "## System"
_USER_HEADING = "## User"


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
        client: JudgeClient,
        *,
        batch_size: int = BATCH_SIZE,
        token_cap: int = TOKEN_CAP,
    ) -> None:
        self._client = client
        self._batch_size = max(1, batch_size)
        self._token_cap = token_cap
        self._template = load_prompt("review")
        self._summary_template = load_prompt("summarize")
        #: One line per opinion that could not be used, in the order they arrived.
        self.skipped: list[str] = []
        #: True once the provider stopped answering; ``detail`` says why.
        self.unavailable = False
        self.detail = ""

    @property
    def name(self) -> str:
        """``"groq openai/gpt-oss-120b"``: who the second opinion is from."""
        return f"{self._client.provider} {self._client.model}"

    @property
    def cost(self) -> JudgeCost:
        """What the judge has spent so far. The client keeps the running total."""
        return self._client.cost

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
                self._give_up(str(exc))
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
        rendered = self._summary_template.substitute(report=report_markdown)
        system, user = _split_prompt(rendered)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            completion = self._client.complete(
                messages, max_tokens=max_tokens, reasoning_effort=_REASONING_EFFORT
            )
        except JudgeUnavailable as exc:
            self._give_up(str(exc))
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

    def _give_up(self, detail: str) -> None:
        self.unavailable = True
        self.detail = detail

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


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start and (start, end + 1) != (0, len(stripped)):
        return [stripped, stripped[start : end + 1]]
    return [stripped]
