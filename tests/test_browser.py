"""ConsentGate: the step-3 browser consent gate, installer and prompt (spec section 7.1).

No subprocess, no network, no real browser: ``run``, ``prompt``, ``installer``,
``installed`` and ``fetch`` are always injected.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import httpx
import pytest
import respx

from proofpath import browser as bw
from proofpath import fetch as fx
from proofpath.config import Config, FetchConfig, load_config


@pytest.fixture(autouse=True)
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path))
    return tmp_path


def raising_prompt(host: str, status: int | None) -> bw.Answer:
    raise AssertionError("prompt must not be called")


# --- ConsentGate.allow --------------------------------------------------------


def test_ask_without_tty_denies_and_never_prompts() -> None:
    gate = bw.ConsentGate("ask", interactive=False, prompt=raising_prompt)
    assert gate.allow("nature.com", 403) is False
    assert gate.skipped_urls == 1
    assert "no interactive terminal" in gate.decision.reason


def test_prompt_asked_once_then_cached() -> None:
    calls: list[tuple[str, int | None]] = []

    def prompt(host: str, status: int | None) -> bw.Answer:
        calls.append((host, status))
        return "once"

    gate = bw.ConsentGate("ask", interactive=True, prompt=prompt, installed=lambda: True)
    assert gate.allow("nature.com", 403) is True
    assert gate.allow("nature.com", 403) is True
    assert calls == [("nature.com", 403)]
    assert gate.skipped_urls == 0


def test_answer_no_denies_for_the_run_without_reasking() -> None:
    calls: list[tuple[str, int | None]] = []

    def prompt(host: str, status: int | None) -> bw.Answer:
        calls.append((host, status))
        return "no"

    gate = bw.ConsentGate("ask", interactive=True, prompt=prompt)
    assert gate.allow("nature.com", 403) is False
    assert gate.allow("nature.com", 403) is False
    assert len(calls) == 1
    assert gate.skipped_urls == 2


def test_answer_always_persists_allow() -> None:
    gate = bw.ConsentGate(
        "ask", interactive=True, prompt=lambda h, s: "always", installed=lambda: True
    )
    assert gate.allow("nature.com", 403) is True
    assert load_config().permissions.install_browser == "allow"


def test_answer_never_persists_deny() -> None:
    gate = bw.ConsentGate("ask", interactive=True, prompt=lambda h, s: "never")
    assert gate.allow("nature.com", 403) is False
    assert load_config().permissions.install_browser == "deny"


def test_override_allow_skips_prompt_and_config() -> None:
    gate = bw.ConsentGate(
        "deny", interactive=False, override=True, prompt=raising_prompt, installed=lambda: True
    )
    assert gate.allow("nature.com", 403) is True
    assert load_config().permissions.install_browser == "ask"  # config untouched


def test_override_deny_wins_over_allow_config() -> None:
    gate = bw.ConsentGate("allow", interactive=False, override=False, prompt=raising_prompt)
    assert gate.allow("nature.com", 403) is False
    assert gate.skipped_urls == 1


def test_allow_config_needs_no_prompt() -> None:
    gate = bw.ConsentGate("allow", interactive=False, prompt=raising_prompt, installed=lambda: True)
    assert gate.allow("nature.com", 403) is True


def test_install_runs_when_not_installed_and_failure_denies() -> None:
    installer_calls: list[list[str]] = []

    def installer(log: list[str]) -> bool:
        installer_calls.append(log)
        return False

    gate = bw.ConsentGate("allow", interactive=False, installed=lambda: False, installer=installer)
    assert gate.allow("nature.com", 403) is False
    assert gate.skipped_urls == 1
    assert gate.decision.reason == "browser install failed"
    assert gate.allow("nature.com", 403) is False
    assert gate.skipped_urls == 2
    assert len(installer_calls) == 1


def test_install_success_runs_installer_once_and_allows() -> None:
    installer_calls: list[list[str]] = []

    def installer(log: list[str]) -> bool:
        installer_calls.append(log)
        log.append("installed scrapling browser extras")
        return True

    gate = bw.ConsentGate("allow", interactive=False, installed=lambda: False, installer=installer)
    assert gate.allow("nature.com", 403) is True
    assert len(installer_calls) == 1
    assert gate.install_log != []
    assert gate.skipped_urls == 0

    # A second allow() call, even against a different host, must not install again.
    assert gate.allow("example.com", 403) is True
    assert len(installer_calls) == 1


def test_installed_checked_at_most_once_per_run() -> None:
    calls = {"n": 0}

    def installed() -> bool:
        calls["n"] += 1
        return True

    gate = bw.ConsentGate("allow", interactive=False, installed=installed)
    assert gate.allow("nature.com", 403) is True
    assert gate.allow("example.com", 403) is True
    assert calls["n"] == 1


def test_persist_failure_after_always_answer_does_not_kill_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Point PROOFPATH_CONFIG_DIR at a path that is a file, so config.toml's parent
    # directory cannot be created and set_value's save_config raises OSError.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("nope", encoding="utf-8")
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(blocked))

    gate = bw.ConsentGate(
        "ask", interactive=True, prompt=lambda h, s: "always", installed=lambda: True
    )
    assert gate.allow("nature.com", 403) is True  # in-run decision kept
    assert gate.decision.outcome == "allow"
    assert any("could not save permission" in line for line in gate.install_log)


# --- is_installed() ------------------------------------------------------


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        ({"patchright", "scrapling"}, True),
        ({"patchright", "scrapling", "playwright"}, True),
        ({"playwright", "scrapling"}, False),  # StealthyFetcher drives patchright
        ({"patchright"}, False),
        (set(), False),
    ],
)
def test_is_installed_needs_patchright_and_scrapling(
    monkeypatch: pytest.MonkeyPatch, present: set[str], expected: bool
) -> None:
    monkeypatch.setattr(
        bw.importlib.util, "find_spec", lambda name: object() if name in present else None
    )
    monkeypatch.setattr(bw, "browser_binary_present", lambda: True)
    assert bw.is_installed() is expected


def _chromium_in(path: Path) -> None:
    (path / "chromium-1140" / "chrome-mac").mkdir(parents=True)


def test_is_installed_needs_a_browser_binary_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A venv with the wheels but no downloaded chromium used to skip the installer
    and fail inside ``fetch_with_browser`` (OPEN-ITEMS 11.6)."""
    monkeypatch.setattr(bw.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv(bw.BROWSERS_PATH_ENV, str(tmp_path))
    assert bw.is_installed() is False
    _chromium_in(tmp_path)
    assert bw.is_installed() is True


def test_the_browsers_path_env_replaces_the_platform_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """playwright and patchright look **only** where the variable points, so neither
    may proofpath: a browser found in a default the driver will not read would let the
    gate skip an install the fetch then needs."""
    monkeypatch.setenv(bw.BROWSERS_PATH_ENV, str(tmp_path))
    assert bw.browser_cache_dirs() == [tmp_path]
    assert bw.browser_binary_present() is False
    _chromium_in(tmp_path)
    assert bw.browser_binary_present() is True


def test_a_browsers_path_of_zero_falls_back_to_the_platform_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """playwright reads ``0`` as "next to the package", not as a directory."""
    monkeypatch.setenv(bw.BROWSERS_PATH_ENV, "0")
    assert all(directory.name == "ms-playwright" for directory in bw.browser_cache_dirs())


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("darwin", "Library/Caches/ms-playwright"),
        ("linux", ".cache/ms-playwright"),
        ("win32", "ms-playwright"),
    ],
)
def test_the_platform_default_cache_is_where_playwright_puts_it(
    monkeypatch: pytest.MonkeyPatch, platform: str, expected: str
) -> None:
    monkeypatch.delenv(bw.BROWSERS_PATH_ENV, raising=False)
    monkeypatch.setattr(bw.sys, "platform", platform)
    directories = bw.browser_cache_dirs()
    # With no variable to obey, this platform's own default leads and the other two
    # follow: a venv carried between machines still finds what it downloaded.
    assert directories[0].as_posix().endswith(expected)
    assert len(directories) == 3
    assert len(set(directories)) == 3


def test_a_missing_cache_directory_is_simply_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bw.BROWSERS_PATH_ENV, str(tmp_path / "nowhere"))
    assert bw.browser_binary_present() is False


def test_a_file_named_chromium_is_not_an_installed_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bw.BROWSERS_PATH_ENV, str(tmp_path))
    (tmp_path / "chromium-1140").write_text("not a directory", encoding="utf-8")
    assert bw.browser_binary_present() is False


def test_the_consent_log_records_whether_the_binary_was_there() -> None:
    """The line is about the ~200 MB download, so it reads ``browser_binary_present``
    and not the combined ``is_installed`` -- otherwise a venv missing only the wheels
    would be logged as missing a browser it has."""
    gate = bw.ConsentGate(
        "allow",
        interactive=False,
        installed=lambda: False,
        binary=lambda: False,
        installer=lambda _: True,
    )
    assert gate.allow("example.test", 403) is True
    assert "browser binary: missing" in gate.install_log

    ready = bw.ConsentGate("allow", interactive=False, installed=lambda: True, binary=lambda: True)
    assert ready.allow("example.test", 403) is True
    assert "browser binary: found" in ready.install_log


def test_the_binary_line_does_not_repeat_the_wheels_verdict() -> None:
    """Wheels missing, browser already downloaded: the installer runs for the wheels
    and the log still says the binary was found."""
    gate = bw.ConsentGate(
        "allow",
        interactive=False,
        installed=lambda: False,
        binary=lambda: True,
        installer=lambda _: True,
    )
    assert gate.allow("example.test", 403) is True
    assert "browser binary: found" in gate.install_log


# --- install() -----------------------------------------------------------


def test_install_falls_back_to_uv_when_pip_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bw.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls: list[list[str]] = []
    call_kwargs: list[dict[str, object]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        call_kwargs.append(kwargs)
        if cmd[:2] == [bw.sys.executable, "-m"] and "pip" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No module named pip")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    log: list[str] = []
    result = bw.install(log, run=run)
    assert result is True
    assert calls == bw.install_commands()
    assert log[0] == f"$ {shlex.join(bw.install_commands()[0])}"
    assert log[1] == "exit 1"
    assert log[2] == "No module named pip"
    for kwargs in call_kwargs:
        assert kwargs == {"capture_output": True, "text": True, "check": False}


def test_install_fails_without_pip_or_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bw.shutil, "which", lambda name: None)

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No module named pip")

    log: list[str] = []
    result = bw.install(log, run=run)
    assert result is False
    # Only the pip command was attempted: no uv, no scrapling install step.
    assert log == [f"$ {shlex.join(bw.install_commands()[0])}", "exit 1", "No module named pip"]


def test_install_pip_success_skips_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bw.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    log: list[str] = []
    result = bw.install(log, run=run)
    assert result is True
    pip_cmd, uv_cmd, module_cmd = bw.install_commands()
    assert calls == [pip_cmd, module_cmd]
    assert uv_cmd not in calls


def test_install_run_raises_oserror_logs_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bw.shutil, "which", lambda name: None)

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no such file or directory: 'pip'")

    log: list[str] = []
    result = bw.install(log, run=run)
    assert result is False
    assert log[0] == f"$ {shlex.join(bw.install_commands()[0])}"
    assert log[1] == "exit ? (FileNotFoundError: no such file or directory: 'pip')"


# --- prompt_text / ask_terminal -----------------------------------------------


def test_prompt_text_matches_spec() -> None:
    """Spec section 7.1: the prompt *is* this block, blank lines included."""
    assert bw.prompt_text("nature.com", 403) == (
        "  ⚠ nature.com blocked this request (HTTP 403).\n"
        "\n"
        "    proofpath can retry with a real browser engine, but that needs a\n"
        "    one-time download:\n"
        "\n"
        "      playwright + patchright wheels    ~81 MB\n"
        "      chromium browser                 ~200 MB\n"
        "      installed into this tool's own environment only\n"
        "\n"
        "    Allow?  [y] yes, once   [a] always (save to config)   [n] no   "
        "[never] never ask again"
    )
    assert "no HTTP status" in bw.prompt_text("nature.com", None)


def test_prompt_text_does_not_call_a_200_a_block() -> None:
    """The empty-body bot wall answers 200 and hands over nothing readable.

    Calling that "blocked this request (HTTP 200)" reads like a contradiction to
    anyone looking at the status line, and it describes something the host did not
    say. The first line states what actually happened; the offer below it is the same.
    """
    first = bw.prompt_text("cell.com", 200).splitlines()[0]
    assert first == "  \u26a0 cell.com answered without readable text (HTTP 200)."
    assert "blocked" not in first


def test_ask_terminal_maps_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bw.typer, "prompt", lambda *a, **k: "A")
    assert bw.ask_terminal("nature.com", 403) == "always"

    answers = iter(["bogus", "y"])
    call_count = {"n": 0}

    def flaky_prompt(*a: object, **k: object) -> str:
        call_count["n"] += 1
        return next(answers)

    monkeypatch.setattr(bw.typer, "prompt", flaky_prompt)
    assert bw.ask_terminal("nature.com", 403) == "once"
    assert call_count["n"] == 2


# --- fetch ---------------------------------------------------------------


def test_fetch_failure_returns_empty_and_logs() -> None:
    def bad_fetch(url: str) -> tuple[int, bytes, str]:
        raise RuntimeError("boom")

    gate = bw.ConsentGate("allow", interactive=False, installed=lambda: True, fetch=bad_fetch)
    result = gate.fetch("https://nature.com/paper")
    assert result == (0, b"", "")
    assert gate.install_log == ["browser fetch failed: RuntimeError"]


# --- integration: ConsentGate satisfies fetch.BrowserGate --------------------


@respx.mock
def test_consent_gate_satisfies_browser_gate_protocol() -> None:
    respx.get("https://x.test/paper.html").mock(return_value=httpx.Response(403))

    def curl_forbidden(url: str) -> tuple[int, bytes, str, str]:
        return 403, b"", "text/html", url

    def browser_fetch(url: str) -> tuple[int, bytes, str]:
        # At least fetch.MIN_HTML_WORDS words: a shorter HTML page reads as a bot wall.
        paragraph = " ".join(["Evidence here"] * 12)
        return 200, f"<html><body><p>{paragraph}</p></body></html>".encode(), "text/html"

    gate: fx.BrowserGate = bw.ConsentGate(
        "allow", interactive=False, installed=lambda: True, fetch=browser_fetch
    )
    config = Config(fetch=FetchConfig(respect_robots=False))
    with httpx.Client(follow_redirects=True) as client:
        fetcher = fx.Fetcher(config=config, gate=gate, client=client, curl_get=curl_forbidden)
        result = fetcher.fetch("https://x.test/paper.html")
    assert result.ok
    assert result.step == 3
    assert "Evidence here" in result.text


# --- install_commands: the scrapling CLI is a console script, not a module -----


def test_install_commands_never_uses_dash_m_scrapling() -> None:
    """scrapling 0.4.15 has no ``__main__``; ``-m scrapling`` aborts (live run item 8)."""
    for cmd in bw.install_commands():
        assert cmd[:3] != [bw.sys.executable, "-m", "scrapling"]


def test_install_commands_runs_the_scrapling_cli_through_dash_c() -> None:
    pip_cmd, uv_cmd, module_cmd = bw.install_commands()
    assert pip_cmd[:4] == [bw.sys.executable, "-m", "pip", "install"]
    assert uv_cmd[:2] == ["uv", "pip"]
    # click reads sys.argv[1:], which under -c is exactly what follows the code string.
    assert module_cmd == [bw.sys.executable, "-c", bw.SCRAPLING_CLI, "install"]
    assert "from scrapling.cli import main" in bw.SCRAPLING_CLI


def test_a_failed_browser_install_is_not_attempted_twice() -> None:
    """There is no second way in: the console script is this same interpreter running
    this same import, so a retry could only repeat a ~200 MB download."""
    pip_cmd, _uv_cmd, module_cmd = bw.install_commands()
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0 if cmd == pip_cmd else 1, stdout="", stderr="x")

    log: list[str] = []
    assert bw.install(log, run=run) is False
    assert calls == [pip_cmd, module_cmd]


def test_the_install_log_can_be_pasted_back_into_a_shell() -> None:
    """The log is what a user is shown when an install fails; a line they cannot
    re-run tells them less than one they can."""
    log: list[str] = []
    bw.install(
        log,
        run=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    _pip_cmd, _uv_cmd, module_cmd = bw.install_commands()
    # The -c program holds spaces and a semicolon, and the pip spec holds brackets
    # and a `>`: both are one argument and both must come back quoted as one.
    assert log[-2] == f"$ {shlex.join(module_cmd)}"
    assert f"'{bw.SCRAPLING_CLI}'" in log[-2]
    assert "'scrapling[fetchers]>=0.4.15'" in log[0]
