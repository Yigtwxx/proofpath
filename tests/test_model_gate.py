"""ModelGate: the consent gate in front of the accurate NLI profile's download.

No network and no real model: ``prompt``, ``download`` and ``installed`` are always
injected, ``hf_hub_download`` is monkeypatched, and the config lives under tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proofpath import model_gate as mg
from proofpath.browser import TERMINAL_ANSWERS, TUI_ANSWERS, Answer
from proofpath.config import load_config
from proofpath.profiles import ACCURATE_PROFILE, DEFAULT_PROFILE, NliProfile


@pytest.fixture(autouse=True)
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path))
    return tmp_path


def raising_prompt(subject: str, status: int | None) -> Answer:
    raise AssertionError("prompt must not be called")


def raising_download(profile: NliProfile) -> None:
    raise AssertionError("download must not be called")


class Recorder:
    """A ``download`` that remembers what it was asked to fetch."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, profile: NliProfile) -> None:
        self.calls.append(profile.name)


def missing(profile: NliProfile) -> bool:
    return False


# --- installed --------------------------------------------------------------------


def test_installed_allows_without_prompt_or_download() -> None:
    gate = mg.ModelGate(
        "deny",
        interactive=True,
        prompt=raising_prompt,
        download=raising_download,
        installed=lambda p: True,
    )
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == "allow"
    assert decision.reason == "installed"


# --- stored permission ------------------------------------------------------------


def test_allow_permission_downloads_without_prompt() -> None:
    download = Recorder()
    gate = mg.ModelGate(
        "allow", interactive=False, prompt=raising_prompt, download=download, installed=missing
    )
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == "allow"
    assert decision.reason == "permission is set to allow"
    assert download.calls == ["accurate"]
    assert any("643 MB" in line for line in gate.log)


def test_deny_permission_never_downloads() -> None:
    gate = mg.ModelGate(
        "deny",
        interactive=True,
        prompt=raising_prompt,
        download=raising_download,
        installed=missing,
    )
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == "deny"
    assert decision.reason == "permission is set to deny"
    assert any("permission is set to deny" in line for line in gate.log)


def test_ask_without_tty_denies_and_reports() -> None:
    gate = mg.ModelGate(
        "ask",
        interactive=False,
        prompt=raising_prompt,
        download=raising_download,
        installed=missing,
    )
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == "deny"
    assert decision.reason == "permission is set to ask but there is no interactive terminal"
    assert any("no interactive terminal" in line for line in gate.log)


def test_override_does_not_grant_consent() -> None:
    """``--accurate`` picks the profile; the 643 MB still needs a yes (rule 5)."""
    gate = mg.ModelGate(
        "ask",
        interactive=False,
        override=True,
        prompt=raising_prompt,
        download=raising_download,
        installed=missing,
    )
    assert gate.ensure(ACCURATE_PROFILE).outcome == "deny"


# --- ask with a TTY: each answer ----------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "outcome", "reason", "downloads"),
    [
        ("once", "allow", "user answered once", ["accurate"]),
        ("always", "allow", "user answered always", ["accurate"]),
        ("no", "deny", "user answered no", []),
        ("never", "deny", "user answered never", []),
    ],
)
def test_ask_with_tty_each_answer(
    answer: Answer, outcome: str, reason: str, downloads: list[str]
) -> None:
    asked: list[tuple[str, int | None]] = []

    def prompt(subject: str, status: int | None) -> Answer:
        asked.append((subject, status))
        return answer

    download = Recorder()
    gate = mg.ModelGate(
        "ask", interactive=True, prompt=prompt, download=download, installed=missing
    )
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == outcome
    assert decision.reason == reason
    assert download.calls == downloads
    assert asked == [(mg.prompt_subject(ACCURATE_PROFILE), None)]
    assert any(reason in line for line in gate.log)


def test_ask_is_asked_once_per_run() -> None:
    asked: list[str] = []

    def prompt(subject: str, status: int | None) -> Answer:
        asked.append(subject)
        return "no"

    gate = mg.ModelGate(
        "ask", interactive=True, prompt=prompt, download=raising_download, installed=missing
    )
    assert gate.ensure(ACCURATE_PROFILE).outcome == "deny"
    assert gate.ensure(ACCURATE_PROFILE).outcome == "deny"
    assert len(asked) == 1


def test_download_runs_once_per_run() -> None:
    download = Recorder()
    gate = mg.ModelGate(
        "allow", interactive=False, prompt=raising_prompt, download=download, installed=missing
    )
    gate.ensure(ACCURATE_PROFILE)
    gate.ensure(ACCURATE_PROFILE)
    assert download.calls == ["accurate"]


# --- persistence --------------------------------------------------------------------


def test_always_persists_allow(tmp_path: Path) -> None:
    path = tmp_path / "custom" / "config.toml"
    gate = mg.ModelGate(
        "ask",
        interactive=True,
        prompt=lambda s, st: "always",
        download=Recorder(),
        installed=missing,
        config_path=path,
    )
    gate.ensure(ACCURATE_PROFILE)
    assert load_config(path).permissions.install_model == "allow"
    # The browser permission is a different key and stays as it was.
    assert load_config(path).permissions.install_browser == "ask"


def test_never_persists_deny(tmp_path: Path) -> None:
    path = tmp_path / "custom" / "config.toml"
    gate = mg.ModelGate(
        "ask",
        interactive=True,
        prompt=lambda s, st: "never",
        download=raising_download,
        installed=missing,
        config_path=path,
    )
    gate.ensure(ACCURATE_PROFILE)
    assert load_config(path).permissions.install_model == "deny"


@pytest.mark.parametrize("answer", ["once", "no"])
def test_once_and_no_are_not_persisted(tmp_path: Path, answer: Answer) -> None:
    path = tmp_path / "config.toml"
    gate = mg.ModelGate(
        "ask",
        interactive=True,
        prompt=lambda s, st: answer,
        download=Recorder(),
        installed=missing,
        config_path=path,
    )
    gate.ensure(ACCURATE_PROFILE)
    assert not path.exists()


def test_persist_failure_is_logged_not_raised(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("nope", encoding="utf-8")
    gate = mg.ModelGate(
        "ask",
        interactive=True,
        prompt=lambda s, st: "always",
        download=Recorder(),
        installed=missing,
        config_path=blocked / "config.toml",
    )
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == "allow"  # the in-run answer is kept
    assert any("could not save permission" in line for line in gate.log)


# --- download failure ---------------------------------------------------------------


class FakeResponse:
    status_code = 503
    text = "<html>secret upstream body</html>"


class FakeHttpError(Exception):
    def __init__(self) -> None:
        super().__init__("503 Server Error: <html>secret upstream body</html>")
        self.response = FakeResponse()


def test_download_failure_denies_with_a_short_reason() -> None:
    def download(profile: NliProfile) -> None:
        raise FakeHttpError

    gate = mg.ModelGate("allow", interactive=False, download=download, installed=missing)
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.outcome == "deny"
    assert decision.reason == "model download failed: FakeHttpError (HTTP 503)"
    assert all("secret upstream body" not in line for line in gate.log)
    assert all("Traceback" not in line for line in gate.log)


def test_download_failure_without_a_response_names_the_type() -> None:
    def download(profile: NliProfile) -> None:
        raise ConnectionError("dns failure for huggingface.co: raw details")

    gate = mg.ModelGate("allow", interactive=False, download=download, installed=missing)
    decision = gate.ensure(ACCURATE_PROFILE)
    assert decision.reason == "model download failed: ConnectionError"


def test_download_failure_is_not_retried() -> None:
    calls: list[str] = []

    def download(profile: NliProfile) -> None:
        calls.append(profile.name)
        raise OSError("disk full")

    gate = mg.ModelGate("allow", interactive=False, download=download, installed=missing)
    gate.ensure(ACCURATE_PROFILE)
    assert gate.ensure(ACCURATE_PROFILE).outcome == "deny"
    assert calls == ["accurate"]


# --- the prompt text -----------------------------------------------------------------


def test_prompt_text_names_repo_and_size() -> None:
    text = mg.model_prompt_text(ACCURATE_PROFILE)
    assert ACCURATE_PROFILE.repo in text
    assert "643 MB" in text
    assert text.endswith(TERMINAL_ANSWERS)


def test_prompt_text_has_three_parts_like_the_browser_prompt() -> None:
    text = mg.model_prompt_text(ACCURATE_PROFILE, answers=TUI_ANSWERS)
    assert text.endswith(TUI_ANSWERS)
    # what, why + cost, answers: separated by blank lines.
    assert len([block for block in text.split("\n\n") if block.strip()]) >= 3


def test_prompt_subject_round_trips() -> None:
    subject = mg.prompt_subject(ACCURATE_PROFILE)
    assert mg.subject_profile(subject) is ACCURATE_PROFILE
    assert mg.subject_profile(mg.prompt_subject(DEFAULT_PROFILE)) is DEFAULT_PROFILE
    # A blocked host is never mistaken for a model question.
    assert mg.subject_profile("nature.com") is None
    assert mg.subject_profile("model:unknown") is None


def test_default_prompt_reads_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    echoed: list[str] = []
    replies = iter(["maybe", "a"])
    monkeypatch.setattr(mg.typer, "echo", lambda message, err=False: echoed.append(message))
    monkeypatch.setattr(mg.typer, "prompt", lambda *args, **kwargs: next(replies))
    assert mg.ask_model_terminal(ACCURATE_PROFILE) == "always"
    assert ACCURATE_PROFILE.repo in echoed[0]
    assert any("Please answer one of" in line for line in echoed)


# --- download_profile ----------------------------------------------------------------


def test_download_profile_fetches_the_three_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    calls: list[tuple[str, str, str | None, str | None]] = []

    def fake_download(
        repo_id: str, filename: str, *, revision: str | None = None, cache_dir: str | None = None
    ) -> str:
        calls.append((repo_id, filename, revision, cache_dir))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cache = tmp_path / "models"
    mg.download_profile(ACCURATE_PROFILE, cache, "arm64")
    assert [c[1] for c in calls] == ["config.json", "tokenizer.json", "onnx/model_quantized.onnx"]
    assert {c[0] for c in calls} == {ACCURATE_PROFILE.repo}
    assert {c[2] for c in calls} == {ACCURATE_PROFILE.revision}
    assert {c[3] for c in calls} == {str(cache)}
    assert cache.is_dir()
