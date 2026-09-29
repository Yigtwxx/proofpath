"""Live check of the local fallback judge (spec section 11, Task 11 Amendment B 8).

Usage:
    uv run python scripts/live_judge_fallback.py [--keep-loaded]

A stand-in for Groq runs on a local port and answers every request with a 429 and a
``Retry-After: 60``. The judge is built exactly as ``proofpath check --judge`` builds
it (``judge.build_client``), pointed at that stand-in, and asked for real:
``Judge.queries`` on the Turkish claim for "OpenAI went bust" and one ``Judge.review``
item. Both are answered by the configured fallback, qwen3.5:9b, which must already
be installed in the local Ollama; nothing is pulled.

What it proves, and exits 1 if it does not hold:
  * the switch happens on the first 429, with no retry and no ``Retry-After`` wait
    (the stand-in sees exactly one request, and the switch comes in under a second);
  * the notice is the spec's wording and fires before the local model is loaded;
  * the failed request is answered by the local model, and both answers parse.

By default the chosen model is unloaded first, so the time to the first local answer
includes a cold load, which is the case a user meets. ``--keep-loaded`` skips that.

This is not a test: it needs a running Ollama with qwen3.5:9b installed. No
API key is read, sent or printed; the stand-in needs none.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import httpx

from proofpath import judge
from proofpath.config import JudgeConfig
from proofpath.models import Label
from proofpath.search.queries import plan_queries

SWITCH_BUDGET_S = 1.0
CLAIM = "OpenAI battı."  # noqa: RUF001 - Turkish, verbatim: the claim the user tried
REVIEW_ITEM = judge.JudgeItem(
    id="c1",
    claim="OpenAI went bankrupt in 2025.",
    passage="OpenAI reported record revenue in 2025 and closed a new funding round.",
    verdict=Label.REFUTED,
    tier="low",
)


class RateLimited(BaseHTTPRequestHandler):
    """Groq on a bad day: every request is a 429 asking for a minute's patience."""

    hits: ClassVar[list[float]] = []

    def do_POST(self) -> None:
        RateLimited.hits.append(time.perf_counter())
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = b'{"error": {"message": "Rate limit reached", "type": "tokens"}}'
        self.send_response(429)
        self.send_header("Retry-After", "60")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
        return  # quiet: the script prints its own lines


def _unload(model: str) -> None:
    """Take ``model`` out of memory, so the first local answer includes the load."""
    root = judge.provider_defaults("ollama").base_url.removesuffix("/v1")
    try:
        httpx.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except httpx.HTTPError as exc:
        print(f"warning: could not unload {model}: {type(exc).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep-loaded", action="store_true", help="do not unload first")
    args = parser.parse_args()

    ollama = judge.provider_defaults("ollama").base_url
    installed = judge.installed_models(ollama, timeout=judge.TAGS_TIMEOUT)
    expected = JudgeConfig().fallback
    print(f"ollama models     {', '.join(installed) or '(none)'}")
    print(f"judge.fallback    {expected} (the default)")
    if expected not in installed:
        print(f"FAIL: {expected} is not installed in the local Ollama; it is never pulled")
        return 1
    print(f"reasoning sent    {judge.LOCAL_REASONING}")
    if not args.keep_loaded:
        _unload(expected)
        print(f"unloaded          {expected} (the first local answer includes a cold load)")

    server = ThreadingHTTPServer(("127.0.0.1", 0), RateLimited)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    # Groq as configured by default, only pointed at the stand-in. No key variable:
    # the stand-in needs none, and this script never reads a real one.
    primary = JudgeConfig(base_url=f"http://127.0.0.1:{port}/openai/v1", api_key_env="")

    notices: list[tuple[float, str]] = []
    client = judge.build_client(
        primary, None, on_switch=lambda text: notices.append((time.perf_counter(), text))
    )
    assert isinstance(client, judge.FallbackClient)
    j = judge.Judge(client)

    started = time.perf_counter()
    written = j.queries([CLAIM])
    first_answer = time.perf_counter()
    review_started = time.perf_counter()
    opinions = j.review([REVIEW_ITEM])
    review_done = time.perf_counter()
    server.shutdown()

    failures: list[str] = []
    print()
    if notices:
        switched_at, notice = notices[0]
        switch_s = switched_at - started
        print(f"notice            {notice}")
        print(f"chosen model      {client.model} (provider {client.provider})")
        print(f"time to switch    {switch_s:.3f}s (first call to the notice; budget 1s)")
        if switch_s >= SWITCH_BUDGET_S:
            failures.append(f"the switch took {switch_s:.3f}s")
        wording = judge.JUDGE_FALLBACK_LIMIT.format(provider="Groq", model=expected)
        if notice != wording:
            failures.append(f"the notice was {notice!r}, not {wording!r}")
        if len(notices) != 1:
            failures.append(f"the notice fired {len(notices)} times")
    else:
        failures.append("no switch notice")
    print(f"first local answer {first_answer - started:.2f}s after the first call")
    print(f"review answer     {review_done - review_started:.2f}s (model already loaded)")
    print(f"groq stand-in hit {len(RateLimited.hits)} time(s) (no retry, no Retry-After wait)")
    if len(RateLimited.hits) != 1:
        failures.append(f"the rate-limited primary was asked {len(RateLimited.hits)} times")
    if client.model != expected:
        failures.append(f"switched to {client.model}, expected {expected}")

    print()
    print(f"queries({CLAIM!r})")
    if j.unavailable:
        failures.append(f"queries: judge unavailable ({j.detail})")
    for index, claim in sorted(written.items()):
        print(f"  [{index}] english  {claim.english}")
        for query in claim.queries:
            print(f"      query    {query}")
    if 0 not in written or not written[0].queries:
        failures.append("queries: no parsed query for claim 0")
    # Reported, not failed on: ``plan_queries`` refuses an echo either way, so an
    # untranslated claim ends up "language not supported" instead of being checked.
    plan = plan_queries([CLAIM], j)
    translated = plan.hypotheses[0] is not None
    print(f"  translated       {'yes' if translated else 'no'} (hypothesis {plan.hypotheses[0]!r})")

    print()
    print(f"review({REVIEW_ITEM.claim!r} / verdict {REVIEW_ITEM.verdict.value})")
    for ident, opinion in opinions.items():
        print(f"  {ident}  {opinion.label.value}  by {opinion.model}")
        print(f"      rationale {opinion.rationale}")
    for line in j.skipped:
        print(f"  skipped: {line}")
    if j.unavailable:
        failures.append(f"review: judge unavailable ({j.detail})")
    if "c1" not in opinions:
        failures.append("review: no parsed opinion for c1")
    elif opinions["c1"].model != f"ollama {expected}":
        failures.append(f"review: the opinion names {opinions['c1'].model}")

    cost = client.cost
    print()
    print(
        f"cost              {cost.calls} answers, {cost.local_calls} local, "
        f"{cost.calls - cost.local_calls} API; {cost.prompt_tokens:,} prompt · "
        f"{cost.completion_tokens:,} completion tokens; waited {cost.waited_s:.1f}s"
    )
    if cost.waited_s:
        failures.append(f"the client waited {cost.waited_s:.1f}s")
    client.close()

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("OK: switched at once, answered locally, both answers parsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
