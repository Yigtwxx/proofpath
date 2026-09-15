"""The shared wiring behind the mirrored verbs (spec section 13.3's mirror rule).

These are the functions ``cli.py`` and the TUI both call, so the tests here are
about what each one *does* and hands back — never about how either front end
prints it. Nothing touches the network: the providers are stubbed and the cache and
config live under ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from proofpath import commands
from proofpath import config as cfg
from proofpath import fetch as fetch_mod
from proofpath import judge as judge_mod
from proofpath import resolve as rs
from proofpath.cache import Cache, sha256_text
from proofpath.config import Config, Contact, Permissions
from proofpath.models import Label, Passage, Verdict

NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
ALPHAFOLD = "Jumper J, et al. Highly accurate protein structure prediction. Nature. 2021."
BEST = rs.Candidate(
    doi="10.1038/s41586-021-03819-2",
    title="Highly accurate protein structure prediction with AlphaFold",
    first_author="Jumper",
    year=2021,
    venue="Nature",
    provider="crossref",
)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))


# --- resolve ------------------------------------------------------------------------


def _resolver(
    monkeypatch: pytest.MonkeyPatch,
    result: rs.ResolveResult,
    retraction: rs.Retraction | None = None,
) -> list[str]:
    """Stub both provider calls and record the DOIs the retraction check was given."""
    asked: list[str] = []

    def retract(self: rs.Resolver, doi: str) -> rs.Retraction | None:
        asked.append(doi)
        return retraction

    monkeypatch.setattr(rs.Resolver, "resolve", lambda self, raw: result)
    monkeypatch.setattr(rs.Resolver, "retraction", retract)
    return asked


def test_resolve_reference_returns_the_record_and_its_retraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch = rs.Retraction("retraction-watch", "2024-01-02", "10.1/n", "Retraction")
    asked = _resolver(monkeypatch, rs.ResolveResult(rs.State.RESOLVED, BEST, [BEST]), watch)
    resolved = commands.resolve_reference(ALPHAFOLD, config=Config())
    assert resolved.result.best is BEST
    assert resolved.retraction is watch
    assert asked == [BEST.doi]
    # Resolved but retracted is a finding, not a clean answer (spec section 13.2).
    assert not resolved.clean


def test_resolve_reference_is_clean_only_when_resolved_and_not_retracted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolver(monkeypatch, rs.ResolveResult(rs.State.RESOLVED, BEST, [BEST]))
    assert commands.resolve_reference(ALPHAFOLD, config=Config()).clean
    _resolver(monkeypatch, rs.ResolveResult(rs.State.GHOST, None, []))
    assert not commands.resolve_reference(ALPHAFOLD, config=Config()).clean


def test_resolve_reference_never_asks_about_a_record_without_a_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``None`` retraction on a DOI-less record would mean "not retracted" (rule 2)."""
    no_doi = rs.Candidate(None, "A book", "Nagel", 1974, "OUP", "openlibrary")
    asked = _resolver(monkeypatch, rs.ResolveResult(rs.State.RESOLVED, no_doi, [no_doi]))
    assert commands.resolve_reference("Nagel 1974", config=Config()).retraction is None
    assert asked == []


def _recording_client(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the HTTP client the resolver is handed, and record its closing."""
    closed: list[str] = []

    class Recorder(httpx.Client):
        def close(self) -> None:
            closed.append("closed")
            super().close()

    monkeypatch.setattr(commands.httpx, "Client", Recorder)
    return closed


def test_resolve_reference_closes_its_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The TUI calls this many times in one process; a leaked socket per call is a leak."""
    closed = _recording_client(monkeypatch)
    _resolver(monkeypatch, rs.ResolveResult(rs.State.RESOLVED, BEST, [BEST]))
    commands.resolve_reference(ALPHAFOLD, config=Config())
    assert closed == ["closed"]


def test_resolve_reference_closes_its_client_when_a_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _recording_client(monkeypatch)

    def boom(self: rs.Resolver, raw: str) -> rs.ResolveResult:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(rs.Resolver, "resolve", boom)
    with pytest.raises(httpx.ConnectError):
        commands.resolve_reference(ALPHAFOLD, config=Config())
    assert closed == ["closed"]


# --- fetch --------------------------------------------------------------------------


def test_fetch_target_rejects_something_that_names_no_source() -> None:
    with pytest.raises(commands.TargetError) as caught:
        commands.fetch_target("just some words", config=Config(), interactive=False)
    assert "not a URL, DOI or arXiv id" in str(caught.value)


def test_fetch_target_walks_the_ladder_for_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = fetch_mod.Fetched(
        url="https://x.test/p.html",
        final_url="https://x.test/p.html",
        step=1,
        outcome=fetch_mod.Outcome.OK,
        status=200,
        content_type="text/html",
        kind="fulltext",
        body=b"",
        text="cats purr " * 20,
        notes=[],
    )
    monkeypatch.setattr(fetch_mod.Fetcher, "fetch", lambda self, url, **kw: fetched)
    outcome = commands.fetch_target(
        "https://x.test/p.html", config=Config(), interactive=False, no_cache=True
    )
    assert outcome.result is fetched
    assert outcome.clean
    assert outcome.stats is not None
    # Rule 4: no terminal, so ``ask`` was resolved to deny before anything was fetched.
    assert outcome.gate.decision.outcome == "deny"


def test_fetch_target_calls_the_prompt_the_caller_handed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 5: the front end owns the section 7.1 question, and the gate asks nobody else."""
    asked: list[tuple[str, int | None]] = []

    def prompt(host: str, status: int | None) -> commands.Answer:
        asked.append((host, status))
        return "no"

    blocked = _blocked_fetch()

    def walk(self: fetch_mod.Fetcher, url: str, **kw: Any) -> fetch_mod.Fetched:
        # What step 2 does when a host answers 403: it asks the gate for step 3.
        self._gate.allow("x.test", 403)
        return blocked

    monkeypatch.setattr(fetch_mod.Fetcher, "fetch", walk)
    outcome = commands.fetch_target(
        "https://x.test/p.html",
        config=Config(),
        interactive=True,
        no_cache=True,
        prompt=prompt,
    )
    assert asked == [("x.test", 403)]
    assert outcome.gate.decision.reason == "user answered no"
    assert not outcome.clean


def _blocked_fetch() -> fetch_mod.Fetched:
    return fetch_mod.Fetched(
        url="https://x.test/p.html",
        final_url="https://x.test/p.html",
        step=2,
        outcome=fetch_mod.Outcome.BLOCKED,
        status=403,
        content_type="text/html",
        kind="none",
        body=b"",
        text="",
        notes=[],
    )


def test_fetch_target_reports_abstract_only_as_not_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _abstract_evidence()
    monkeypatch.setattr("proofpath.oa.OpenAccess.fetch", lambda self, doi, arxiv=None: evidence)
    outcome = commands.fetch_target(
        "10.1038/s41586-021-03819-2", config=Config(), interactive=False, no_cache=True
    )
    assert outcome.result is evidence
    assert not outcome.clean  # abstract-only is a finding, never a clean fetch


def _abstract_evidence() -> Any:
    from proofpath import oa

    return oa.Evidence(
        kind="abstract",
        state="UNVERIFIED (abstract only)",
        text="cats purr.",
        url="https://doi.org/10.1/x",
        source="abstract:crossref",
        attempts=[],
        notes=[],
    )


# --- config -------------------------------------------------------------------------


def test_config_view_says_whether_the_file_exists_yet() -> None:
    view = commands.config_view()
    assert not view.exists  # nothing written: the defaults are what is shown
    assert "[permissions]" in view.toml
    commands.config_set("permissions.install_browser", "deny")
    assert commands.config_view().exists
    assert commands.config_view().config.permissions.install_browser == "deny"


def test_config_set_writes_one_pair() -> None:
    assert commands.config_set("contact.email", "a@b.test") == (("contact.email", "a@b.test"),)
    assert cfg.load_config().contact.email == "a@b.test"


def test_config_set_provider_writes_the_whole_preset() -> None:
    """A provider without its own model and key variable would point at the old one."""
    written = commands.config_set("judge.provider", "gemini")
    assert [key for key, _ in written] == [f"judge.{name}" for name in commands.JUDGE_PRESET_FIELDS]
    saved = cfg.load_config().judge
    preset = judge_mod.provider_defaults("gemini")
    assert (saved.provider, saved.model, saved.api_key_env) == (
        preset.provider,
        preset.model,
        preset.api_key_env,
    )


def test_config_set_rejects_an_unknown_provider() -> None:
    with pytest.raises(judge_mod.JudgeError):
        commands.config_set("judge.provider", "nope")


def test_config_check_reports_a_missing_key_without_sending_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that was never sent is not a request that failed (rule 2)."""
    monkeypatch.setattr(judge_mod, "resolve_api_key", lambda name: None)
    monkeypatch.setattr(
        judge_mod,
        "check",
        lambda *a, **k: pytest.fail("no request may be sent"),
    )
    checked = commands.config_check(Config())
    assert checked.missing_key
    assert checked.result is None
    assert not checked.ok
    assert checked.dotenv_paths


def test_config_check_returns_the_providers_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    key = judge_mod.ApiKey(value="k", source="env")
    answer = judge_mod.CheckResult(ok=True, model="m", latency_ms=12, detail="")
    monkeypatch.setattr(judge_mod, "resolve_api_key", lambda name: key)
    monkeypatch.setattr(judge_mod, "check", lambda config, api_key, **kw: answer)
    checked = commands.config_check(Config())
    assert checked.result is answer
    assert checked.key is key
    assert checked.ok


# --- cache --------------------------------------------------------------------------


@pytest.fixture
def filled_cache() -> Path:
    with Cache() as db:
        db.add_source(
            "doi:10.1/x",
            scheme="academic",
            title="Paper X",
            url="",
            text_kind="abstract",
            raw_text="cats purr.",
            now=NOW,
        )
        db.put_chunks(
            "doi:10.1/x",
            "bge@rev",
            [Passage("cats purr.", "doi:10.1/x", 0)],
            np.ones((1, 3), dtype=np.float32),
            text_sha256=sha256_text("cats purr."),
        )
        db.put_verdict(
            "h",
            "doi:10.1/x",
            "m",
            Verdict(Label.SUPPORTED, 0.9, "high", Passage("cats purr.", "doi:10.1/x", 0)),
            now=NOW,
        )
        return db.path


def test_cache_overview_counts_sources_chunks_and_verdicts(filled_cache: Path) -> None:
    held = commands.cache_overview()
    assert held.path == filled_cache
    assert (held.sources, held.chunks, held.verdicts) == (1, 1, 1)


def test_cache_list_carries_the_instant_its_expiries_were_read_at(filled_cache: Path) -> None:
    """One instant for the whole listing, so two rows cannot disagree about "expired"."""
    listing = commands.cache_list()
    assert not listing.empty
    assert [entry.source_id for entry in listing.entries] == ["doi:10.1/x"]
    assert datetime.fromisoformat(listing.now).tzinfo is not None


def test_cache_list_is_empty_on_a_fresh_cache() -> None:
    assert commands.cache_list().empty


def test_cache_detail_is_none_for_a_source_that_is_not_there(filled_cache: Path) -> None:
    assert commands.cache_detail("doi:10.1/x") is not None
    assert commands.cache_detail("doi:nope") is None


def test_cache_clear_removes_what_it_says_it_removed(filled_cache: Path) -> None:
    removed = commands.cache_clear()
    assert removed.sources == 1
    assert commands.cache_list().empty


def test_cache_file_names_the_sqlite_file(filled_cache: Path) -> None:
    assert commands.cache_file() == filled_cache


# --- the contract both front ends rely on --------------------------------------------


def test_nothing_here_prints_or_exits() -> None:
    """The mirror rule's other half: the wiring returns, the front ends print."""
    source = Path(commands.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # past the module docstring, which names both
    assert "typer" not in body
    assert "print(" not in body


def test_a_config_with_a_contact_address_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The polite contact address comes from config, never from a literal (no secrets)."""
    seen: list[str] = []
    monkeypatch.setattr(commands, "user_agent", lambda email: seen.append(email) or "ua")
    _resolver(monkeypatch, rs.ResolveResult(rs.State.RESOLVED, BEST, [BEST]))
    config = Config(contact=Contact(email="a@b.test"), permissions=Permissions())
    commands.resolve_reference(ALPHAFOLD, config=config)
    assert seen == ["a@b.test"]


def test_resolve_reference_reports_a_retraction_check_nobody_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every provider down is a distinct state, not ``None`` ("not retracted") and
    not a crash: the record is still returned, and the outcome is not clean (rule 2)."""
    from proofpath.polite import ProviderError

    def boom(self: rs.Resolver, doi: str) -> rs.Retraction | None:
        raise ProviderError("crossref and openalex unavailable (HTTP 503)")

    monkeypatch.setattr(
        rs.Resolver, "resolve", lambda self, raw: rs.ResolveResult(rs.State.RESOLVED, BEST, [BEST])
    )
    monkeypatch.setattr(rs.Resolver, "retraction", boom)
    resolved = commands.resolve_reference(ALPHAFOLD, config=Config())
    assert resolved.result.best is BEST
    assert resolved.retraction is None
    assert resolved.retraction_error == "crossref and openalex unavailable (HTTP 503)"
    assert not resolved.clean


def test_resolve_reference_has_no_retraction_error_when_the_check_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolver(monkeypatch, rs.ResolveResult(rs.State.RESOLVED, BEST, [BEST]))
    assert commands.resolve_reference(ALPHAFOLD, config=Config()).retraction_error is None
