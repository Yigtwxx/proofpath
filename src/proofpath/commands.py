"""The wiring behind the verbs both front ends offer (spec section 13.3).

The mirror rule says every CLI verb exists in the TUI as the same-named slash
command and that *both surfaces call the same library functions*. This module is
those functions. Each one opens what its verb needs, does the one thing the verb
names, closes it again and returns a result object; none of them prints, formats,
exits or reads a terminal. ``cli.py`` turns the result into key/value lines and
``tui/app.py`` turns it into widgets, and neither can drift from the other about
what the verb actually *did*.

Two consequences are deliberate. Every client is closed on the way out, including
on the error path, because the TUI calls these repeatedly in one process where the
CLI called them once per process and let exit clean up. And nothing here raises
``typer.Exit``: a caller decides what an outcome means for its own exit code, which
is why the result objects answer ``clean`` rather than carrying a code.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from proofpath import config as cfg
from proofpath import fetch as fetch_mod
from proofpath import judge as judge_mod
from proofpath import oa as oa_mod
from proofpath import resolve as resolve_mod
from proofpath.browser import Answer, ConsentGate
from proofpath.cache import Cache, Cleared, SourceDetail, SourceSummary
from proofpath.config import Config, JudgeConfig
from proofpath.paths import config_path
from proofpath.polite import PoliteClient, ProviderError, user_agent

#: ``judge.provider`` is not one setting: switching provider without its model, base
#: URL and key variable would leave three fields pointed at the old one.
JUDGE_PRESET_FIELDS = ("provider", "model", "base_url", "api_key_env")

#: How a front end asks the section 7.1 question: one host and its HTTP status in,
#: one of the four answers out. The terminal asks with ``browser.ask_terminal``; the
#: TUI asks with an inline widget, and neither reads the other's stream (rule 4).
PromptFn = Callable[[str, int | None], Answer]


class TargetError(ValueError):
    """``fetch`` was handed something that is neither a URL, a DOI nor an arXiv id."""


# --- resolve ------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """What ``resolve`` found: the reference's state, and whether it was retracted.

    ``retraction`` is ``None`` for "checked, no notice" and for "never asked" (no
    DOI); ``retraction_error`` is set when every retraction provider failed. That is
    a third state, not a clean one: nobody said the source is unretracted (rule 2).
    """

    result: resolve_mod.ResolveResult
    retraction: resolve_mod.Retraction | None
    retraction_error: str | None = None

    @property
    def clean(self) -> bool:
        """A resolved, unretracted reference. Every other state is a finding (13.3)."""
        return (
            self.result.state is resolve_mod.State.RESOLVED
            and self.retraction is None
            and self.retraction_error is None
        )


def resolve_reference(reference: str, *, config: Config) -> Resolved:
    """Look one bibliography entry up, and check the record it found for a retraction.

    The retraction query is skipped when there is no DOI to ask about, which is also
    the only case where a ``None`` here does not mean "not retracted". A check every
    provider failed is carried as ``retraction_error`` rather than raised: the record
    was found, and the front ends report both facts side by side.
    """
    email = config.contact.email
    client = httpx.Client(headers={"User-Agent": user_agent(email)}, timeout=20.0)
    retraction = retraction_error = None
    try:
        resolver = resolve_mod.Resolver(contact_email=email, client=client)
        result = resolver.resolve(reference)
        best = result.best
        if best is not None and best.doi:
            try:
                retraction = resolver.retraction(best.doi)
            except ProviderError as exc:
                retraction_error = str(exc)
    finally:
        client.close()
    return Resolved(result, retraction, retraction_error)


# --- fetch --------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchOutcome:
    """One source fetched: the result, the ladder's counters, and the section 7.1 gate."""

    target: str
    result: fetch_mod.Fetched | oa_mod.Evidence
    stats: fetch_mod.FetchStats
    gate: ConsentGate

    @property
    def clean(self) -> bool:
        """Full text and something to read. Abstract-only is a finding (spec 13.3)."""
        if isinstance(self.result, fetch_mod.Fetched):
            # Reached with nothing to read is not clean: a 200 with no words is a
            # source that was not actually read (rule 2).
            return self.result.ok and self.result.words > 0
        return self.result.kind == "fulltext"


def fetch_target(
    target: str,
    *,
    config: Config,
    interactive: bool,
    override: bool | None = None,
    no_cache: bool = False,
    prompt: PromptFn | None = None,
) -> FetchOutcome:
    """Fetch one target: a URL up the ladder, a DOI or arXiv id down the OA chain.

    ``prompt`` is how the front end asks the section 7.1 question; ``None`` leaves the
    gate with its terminal prompt, and ``interactive=False`` means it never asks at
    all (product rule 4). Raises :class:`TargetError` for anything that names no
    source; every other failure is reported inside the result.
    """
    is_url = target.startswith(("http://", "https://"))
    doi = arxiv_id = None
    if not is_url:
        doi = resolve_mod.find_doi(target)
        arxiv_id = None if doi else resolve_mod.find_arxiv_id(target)
        if doi is None and arxiv_id is None:
            raise TargetError(f"not a URL, DOI or arXiv id: {target!r}")

    gate = ConsentGate(
        config.permissions.install_browser,
        interactive=interactive,
        override=override,
        prompt=prompt,
    )
    result: fetch_mod.Fetched | oa_mod.Evidence
    with ExitStack() as stack:
        cache = None if no_cache else stack.enter_context(Cache())
        fetcher = fetch_mod.Fetcher(config=config, gate=gate, cache=cache, interactive=interactive)
        stack.callback(fetcher.close)
        if is_url:
            result = fetcher.fetch(target)
        else:
            client = PoliteClient(contact_email=config.contact.email)
            stack.callback(client.client.close)
            chain = oa_mod.OpenAccess(
                fetcher, client, contact_email=config.contact.email, cache=cache
            )
            result = chain.fetch(doi, arxiv_id)
        stats = fetcher.summary()
    return FetchOutcome(target, result, stats, gate)


# --- config -------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigView:
    """The config file as both front ends show it: where it is, and what it holds."""

    path: Path
    exists: bool
    config: Config
    toml: str


def config_view() -> ConfigView:
    """Load the config and render it. Raises ``ConfigError`` on an unreadable file."""
    current = cfg.load_config()
    path = config_path()
    return ConfigView(path, path.exists(), current, cfg.render_config(current))


def config_set(key: str, value: str, path: Path | None = None) -> tuple[tuple[str, str], ...]:
    """Change one setting and return every ``(key, value)`` pair actually written.

    ``judge.provider`` writes four: a provider without its own model, base URL and
    key variable would leave the previous provider's settings in place under a new
    name. Raises ``JudgeError`` for an unknown provider, ``ConfigError`` for a key or
    value the config does not accept.
    """
    if key != "judge.provider":
        cfg.set_value(key, value, path)
        return ((key, value),)
    preset = judge_mod.provider_defaults(value)
    written: list[tuple[str, str]] = []
    for name in JUDGE_PRESET_FIELDS:
        field_value = str(getattr(preset, name))
        cfg.set_value(f"judge.{name}", field_value, path)
        written.append((f"judge.{name}", field_value))
    return tuple(written)


@dataclass(frozen=True)
class JudgeCheck:
    """``config check``: the provider that would be used, and whether it answered.

    ``result`` is ``None`` exactly when ``missing_key`` is true — the request was
    never sent, which is a different thing from a request that failed, and the two
    are never collapsed into one "not ok" (rule 2).
    """

    judge: JudgeConfig
    key: judge_mod.ApiKey | None
    missing_key: bool
    dotenv_paths: tuple[Path, ...]
    result: judge_mod.CheckResult | None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok


def config_check(config: Config) -> JudgeCheck:
    """Send one tiny request to prove the judge provider, model and key work."""
    judge = config.judge
    key = judge_mod.resolve_api_key(judge.api_key_env)
    if judge.api_key_env and key is None:
        return JudgeCheck(judge, None, True, tuple(judge_mod.default_dotenv_paths()), None)
    return JudgeCheck(judge, key, False, (), judge_mod.check(judge, key))


# --- cache --------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheOverview:
    """Where the cache lives and how much it holds.

    The two lookup counts are kept apart from the source list because they are keyed
    by reference and by DOI rather than by source: adding them to ``sources`` would
    overstate how many documents the cache can actually replay.
    """

    path: Path
    sources: int
    chunks: int
    verdicts: int
    resolutions: int
    retractions: int


def cache_overview() -> CacheOverview:
    with Cache() as db:
        entries = db.summary()
        resolutions, retractions = db.lookup_counts()
        return CacheOverview(
            path=db.path,
            sources=len(entries),
            chunks=sum(entry.chunks for entry in entries),
            verdicts=sum(entry.verdicts for entry in entries),
            resolutions=resolutions,
            retractions=retractions,
        )


@dataclass(frozen=True)
class CacheListing:
    """Every cached source, with the moment the expiries were measured against."""

    entries: tuple[SourceSummary, ...]
    resolutions: int
    retractions: int
    now: str  # ISO-8601 UTC, so a caller compares expiries against one instant

    @property
    def empty(self) -> bool:
        return not self.entries and not self.resolutions and not self.retractions


def cache_list() -> CacheListing:
    now = datetime.now(timezone.utc).isoformat()
    with Cache() as db:
        entries = db.summary()
        resolutions, retractions = db.lookup_counts()
    return CacheListing(tuple(entries), resolutions, retractions, now)


def cache_detail(source_id: str) -> SourceDetail | None:
    """One cached source with its chunks and verdicts; ``None`` when it is not there."""
    with Cache() as db:
        return db.detail(source_id)


def cache_clear(*, expired: bool = False) -> Cleared:
    with Cache() as db:
        return db.clear(expired_only=expired)


def cache_file() -> Path:
    """The SQLite file's path. Opening the cache is what creates it, so this does."""
    with Cache() as db:
        return db.path


__all__ = [
    "JUDGE_PRESET_FIELDS",
    "Answer",
    "CacheListing",
    "CacheOverview",
    "ConfigView",
    "FetchOutcome",
    "JudgeCheck",
    "PromptFn",
    "Resolved",
    "TargetError",
    "cache_clear",
    "cache_detail",
    "cache_file",
    "cache_list",
    "cache_overview",
    "config_check",
    "config_set",
    "config_view",
    "fetch_target",
    "resolve_reference",
]
