"""Step 3 of the fetch ladder: the browser consent gate (spec section 7.1).

Downloading the ~280 MB browser engine (playwright/patchright wheels plus the
chromium binary) is a real action taken on the user's machine, so it is never
silent: product rules 4 and 5 bind here. ``ConsentGate`` resolves the permission
once per run, asks at most once when the terminal allows it, and never re-asks a
denied answer. Every refusal — by stored config, by an override flag, by a "no"
answer, or by a failed install — is counted in ``skipped_urls``; ``skipped``, the
number the report prints, counts sources and is kept by the ladder and the
open-access chain (see ``fetch.BrowserGate``), so a DOI with four walled locations
is one skipped source, not four (product rule 6).
"""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import typer

from proofpath.config import ConfigError, Decision, Permission, resolve_permission, set_value

Answer = Literal["once", "always", "no", "never"]
PROMPT_CHOICES = ("y", "a", "n", "never")
_CHOICE_ANSWERS: dict[str, Answer] = {"y": "once", "a": "always", "n": "no", "never": "never"}

WHEELS_SIZE = "~81 MB"
BROWSER_SIZE = "~200 MB"
INSTALL_SPEC = "scrapling[fetchers]>=0.4.15"
# scrapling 0.4.15 ships no ``scrapling.__main__``: its CLI is the console script
# ``scrapling`` = ``scrapling.cli:main``, a click group. ``-m scrapling`` therefore
# aborts before it starts, which is what left every live run of 2026-09-12 with
# "browser install failed" (docs/eval/2026-09-12-v0.1-live.md, item 8). Importing the
# group and calling it is the same entry point the console script uses, and click
# reads ``sys.argv[1:]`` — under ``-c`` that is exactly the arguments after this
# string, so ``install`` arrives as the subcommand.
SCRAPLING_CLI = "from scrapling.cli import main; main()"


def prompt_text(host: str, status: int | None) -> str:
    """The spec section 7.1 consent block, rendered for one blocked host: three
    parts (what happened, what it costs, the choices) separated by blank lines."""
    status_desc = f"HTTP {status}" if status is not None else "no HTTP status"
    lines = [
        f"  ⚠ {host} blocked this request ({status_desc}).",
        "",
        "    proofpath can retry with a real browser engine, but that needs a",
        "    one-time download:",
        "",
        f"      playwright + patchright wheels    {WHEELS_SIZE}",
        f"      chromium browser                 {BROWSER_SIZE}",
        "      installed into this tool's own environment only",
        "",
        "    Allow?  [y] yes, once   [a] always (save to config)   [n] no   "
        "[never] never ask again",
    ]
    return "\n".join(lines)


def ask_terminal(host: str, status: int | None) -> Answer:
    """The default prompt. ``ConsentGate`` never calls this without a TTY.

    Spec section 13.3: the prompt goes to stderr, so ``--format json`` on stdout
    stays a single document even when a run happens to hit this path.
    """
    typer.echo(prompt_text(host, status), err=True)
    while True:
        raw = typer.prompt("Allow?", default="n", err=True).strip().lower()
        if raw in _CHOICE_ANSWERS:
            return _CHOICE_ANSWERS[raw]
        typer.echo(f"Please answer one of: {', '.join(PROMPT_CHOICES)}", err=True)


def is_installed() -> bool:
    """True once the browser extras ``fetch_with_browser`` needs are importable.

    scrapling's ``StealthyFetcher`` drives ``patchright`` (scrapling 0.4.15,
    ``engines/_browsers/_stealth.py``); a bare ``playwright`` install is not enough,
    so it is not what is checked.
    """
    return (
        importlib.util.find_spec("patchright") is not None
        and importlib.util.find_spec("scrapling") is not None
    )


def install_commands() -> list[list[str]]:
    """The three commands ``install`` draws on, in order: wheels by pip, wheels by uv
    when pip is missing, then the browser binary through the scrapling CLI. Kept as
    data so it is testable on its own, without a real subprocess.

    There is no fourth, console-script step. ``.venv/bin/scrapling`` is this same
    interpreter running this same import, so it cannot rescue an import that just
    failed — all it could do is start the ~200 MB download again.
    """
    return [
        [sys.executable, "-m", "pip", "install", INSTALL_SPEC],
        ["uv", "pip", "install", "--python", sys.executable, INSTALL_SPEC],
        [sys.executable, "-c", SCRAPLING_CLI, "install"],
    ]


def _run_logged(
    cmd: list[str],
    log: list[str],
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    # Quoted, not merely spaced: the log is what the user is shown when an install
    # fails, and a line they can paste back into a shell says more than one they
    # cannot. The -c program and the pip spec are each one argument.
    log.append(f"$ {shlex.join(cmd)}")
    try:
        result = run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        # e.g. the interpreter/binary named in cmd does not exist on this machine.
        log.append(f"exit ? ({type(exc).__name__}: {exc})")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
    log.append(f"exit {result.returncode}")
    if result.returncode != 0:
        stderr_lines = (result.stderr or "").strip().splitlines()
        if stderr_lines:
            log.append(stderr_lines[-1])
    return result


def install(
    log: list[str],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    """Install the wheels, then the browser binary. Every command is logged, pass or
    fail. ``pip`` missing in a uv-managed venv falls back to ``uv pip install`` only
    when ``uv`` is on PATH; otherwise the install fails outright."""
    pip_cmd, uv_cmd, module_cmd = install_commands()
    result = _run_logged(pip_cmd, log, run)
    if result.returncode != 0:
        if not shutil.which("uv"):
            return False
        result = _run_logged(uv_cmd, log, run)
        if result.returncode != 0:
            return False
    # No retry behind this one: see ``install_commands``. A failed browser download
    # is reported with its log, not paid for twice.
    return _run_logged(module_cmd, log, run).returncode == 0


def fetch_with_browser(url: str, *, timeout: float = 60.0) -> tuple[int, bytes, str]:
    """Step 3's real transport. Imported lazily so the base install never needs
    scrapling's browser extras."""
    from scrapling.fetchers import StealthyFetcher

    page = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=int(timeout * 1000))
    body: bytes = page.body if isinstance(page.body, bytes) else page.body.encode("utf-8")
    content_type: str = page.headers.get("content-type", "text/html")
    return page.status, body, content_type


class ConsentGate:
    """Implements ``fetch.BrowserGate``. Asks at most once per run.

    The permission is resolved eagerly at construction time: an override flag or a
    stored "allow"/"deny" needs no host or status to decide. Only "ask" plus an
    interactive terminal stays open (``decision.outcome == "prompt"``) until the
    first blocked host is known, at which point ``allow`` asks once and the answer
    is cached in ``decision`` for the rest of the run.
    """

    def __init__(
        self,
        permission: Permission,
        *,
        interactive: bool,
        override: bool | None = None,
        prompt: Callable[[str, int | None], Answer] | None = None,
        installer: Callable[[list[str]], bool] | None = None,
        installed: Callable[[], bool] | None = None,
        fetch: Callable[[str], tuple[int, bytes, str]] | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._prompt = prompt or ask_terminal
        self._installer = installer or install
        self._installed = installed or is_installed
        self._fetch = fetch or fetch_with_browser
        self._config_path = config_path

        if override is True:
            self.decision = Decision("allow", "--allow-browser")
        elif override is False:
            self.decision = Decision("deny", "--no-browser")
        else:
            self.decision = resolve_permission(permission, interactive=interactive)

        self.skipped = 0  # per source; see fetch.BrowserGate
        self.skipped_urls = 0  # per refusal, counted by allow()
        self.install_log: list[str] = []
        self.asked = False
        # True once the ladder reached step 3 at all: a report mentions the
        # permission only when it mattered for this run.
        self.consulted = False
        # Once the browser is known ready (already installed, or just installed
        # successfully), never re-check `installed()` or run the installer again.
        self._ready = False

    def _persist(self, value: str) -> None:
        """Save the user's "always"/"never" answer. A failure here must not lose
        the in-run decision already made — it is only logged (product rule 6:
        the report states its own coverage, including config write failures)."""
        try:
            set_value("permissions.install_browser", value, self._config_path)
        except (ConfigError, OSError) as exc:
            self.install_log.append(f"could not save permission: {exc}")

    def allow(self, host: str, status: int | None) -> bool:
        self.consulted = True
        if self.decision.outcome == "prompt" and not self.asked:
            self.asked = True
            answer = self._prompt(host, status)
            if answer == "once":
                self.decision = Decision("allow", "user answered once")
            elif answer == "always":
                self.decision = Decision("allow", "user answered always")
                self._persist("allow")
            elif answer == "no":
                self.decision = Decision("deny", "user answered no")
            else:  # "never"
                self.decision = Decision("deny", "user answered never")
                self._persist("deny")

        if self.decision.outcome != "allow":
            self.skipped_urls += 1
            return False

        if not self._ready:
            if self._installed():
                self._ready = True
            else:
                installed_ok = self._installer(self.install_log)
                if not installed_ok:
                    self.decision = Decision("deny", "browser install failed")
                    self.skipped_urls += 1
                    return False
                # A fresh install may have added packages that were not importable
                # (and thus cached as absent) a moment ago.
                importlib.invalidate_caches()
                self._ready = True

        return True

    def fetch(self, url: str) -> tuple[int, bytes, str]:
        try:
            return self._fetch(url)
        except Exception as exc:  # a browser crash must not crash the run
            self.install_log.append(f"browser fetch failed: {type(exc).__name__}")
            return (0, b"", "")
