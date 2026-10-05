"""``proofpath update``: replace this install with PyPI's latest release.

docs/superpowers/specs/2026-10-05-update-command-design.md is the source of truth;
the section numbers below are that spec's.

Three things shape this module. Running the command is the consent (decision 4), so
nothing here asks: ``--check`` only reports, and the plain verb updates straight away.
The installed extras are written into the install spec (decision 3), because a bare
``uv tool upgrade`` was measured to drop packages added to the tool's environment
after it was installed -- the browser wheels the user consented to among them. And the
result is read back from a fresh process (section 2.8): the code imported here is the
old version and cannot say what the command installed.

Like :mod:`proofpath.commands`, nothing here prints, exits or reads a terminal. The
runner is injected, as in ``browser.install``, so a test never installs anything.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from proofpath import __version__, shell
from proofpath.config import Config
from proofpath.fetch import network_permission
from proofpath.polite import user_agent

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib  # type: ignore[import-not-found]

DIST = "proofpath"
PYPI_URL = f"https://pypi.org/pypi/{DIST}/json"
PYPI_TIMEOUT = 10.0
#: Run by the venv's own interpreter after the update (section 2.8), under ``-I``: an
#: isolated interpreter ignores ``PYTHON*`` variables and the working directory, so a
#: proofpath checkout the user happens to stand in cannot answer for the install.
VERSION_PROBE = f"import importlib.metadata as m; print(m.version({DIST!r}))"
#: Each extra the update carries over, and the modules that import only when it is
#: installed. All of them must import: ``scrapling`` alone is a base dependency.
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "browser": ("scrapling", "patchright"),
    "gpu": ("torch", "sentence_transformers"),
}
#: The development extra is a checkout's business, never a user install's.
NEVER_CARRIED = frozenset({"dev"})
#: Section 2.9: a running ``proofpath.exe`` can be locked while it is being replaced.
WINDOWS_NOTE = "close proofpath and run the command above"
EDITABLE_HINT = "update it with git pull and uv sync"
#: Section 2.4: what fetches the newest release again for each temporary environment.
#: ``uvx name@latest`` refreshes uv's cached copy; ``pipx run`` reuses its cached venv
#: for days unless told not to, hence ``--no-cache``.
UVX_RERUN = f"uvx {DIST}@latest"
PIPX_RUN_RERUN = f"pipx run --no-cache {DIST}"

InstallKind = Literal["editable", "ephemeral", "uv", "pipx", "pip"]
UpdateState = Literal[
    "up to date", "available", "ahead", "editable", "ephemeral", "updated", "failed"
]
Runner = Callable[..., subprocess.CompletedProcess[str]]

_KIND_NAMES: dict[InstallKind, str] = {
    "editable": "editable",
    "ephemeral": "temporary environment",
    "uv": "uv tool",
    "pipx": "pipx",
    "pip": "pip",
}


class UpdateError(RuntimeError):
    """PyPI could not say which version is the latest. The message is the reason."""


@dataclass(frozen=True)
class Install:
    """How this copy of proofpath was installed, and so how it is updated.

    ``python`` is the interpreter that reads the installed version back afterwards:
    the venv's own for a uv tool or pipx install, ``sys.executable`` otherwise.
    ``source`` is the checkout an editable install points at, when it can be read;
    ``rerun`` is the command that fetches the newest release for an ephemeral one.
    """

    kind: InstallKind
    prefix: Path
    python: Path
    source: Path | None = None
    rerun: str = ""

    @property
    def updatable(self) -> bool:
        """False for the two kinds this command never runs anything for."""
        return self.kind not in ("editable", "ephemeral")

    @property
    def name(self) -> str:
        return _KIND_NAMES[self.kind]


@dataclass(frozen=True)
class UpdateResult:
    """What ``update`` found and did. ``latest`` is ``None`` when PyPI was not asked
    or did not answer, which is never the same thing as "up to date" (rule 2).

    ``command`` is what ran, or what would run; on a failure it is the command to run
    by hand. ``log`` holds each command as ``$ cmd`` with its exit code.
    """

    current: str
    latest: str | None
    state: UpdateState
    install: Install | None = None
    extras: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    log: tuple[str, ...] = ()
    installed: str | None = None
    detail: str = ""
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Every state but ``failed``: a caller's exit code is 0 or 1 by this alone."""
        return self.state != "failed"

    @property
    def summary(self) -> str:
        """The one sentence both front ends print for the state."""
        if self.state == "up to date":
            return "up to date"
        if self.state == "available":
            return f"update available: {self.current} → {self.latest}"
        if self.state == "ahead":
            return (
                f"this build ({self.current}) is newer than PyPI's {self.latest}:"
                " a development build, nothing to do"
            )
        if self.state == "editable":
            source = self.install.source if self.install is not None else None
            where = f" from {source}" if source is not None else ""
            return (
                f"{self.latest} is out, but this is an editable install{where}: "
                f"not touched, {EDITABLE_HINT}"
            )
        if self.state == "ephemeral":
            rerun = self.install.rerun if self.install is not None else ""
            return (
                f"{self.latest} is out, but this is a temporary environment;"
                f" `{rerun}` already fetches the newest"
            )
        if self.state == "updated":
            return f"updated to {self.installed}"
        return self.detail


def update(
    config: Config,
    *,
    interactive: bool = False,
    check_only: bool = False,
    run: Runner | None = None,
) -> UpdateResult:
    """Compare this build with PyPI's latest and, unless ``check_only``, update to it.

    ``interactive`` is the caller's answer to "is there a terminal": it decides what
    ``permissions.network = ask`` means, exactly as for the fetch ladder (rule 4), and
    defaults to the safe "no". Only an available update goes on to look at the
    install: an up-to-date or ahead build has nothing to run, and an editable or
    ephemeral one is never touched. Every failure is reported in the result, never
    raised. ``run`` defaults to ``subprocess.run``, looked up at call time.
    """
    run = subprocess.run if run is None else run
    current = __version__
    # The fetcher's own rule, not a copy of it: ``deny`` and a no-TTY ``ask`` send
    # nothing and say why; an ``ask`` with a terminal goes ahead with a note, since
    # typing the command is the answer that terminal would have been asked for.
    allowed, note = network_permission(config, interactive=interactive)
    if not allowed:
        return UpdateResult(current, None, "failed", detail=f"PyPI was not asked; {note}")
    notes = (note,) if note else ()
    try:
        latest = latest_version()
    except UpdateError as exc:
        return UpdateResult(current, None, "failed", detail=str(exc), notes=notes)

    if Version(latest) == Version(current):
        return UpdateResult(current, latest, "up to date", notes=notes)
    if Version(latest) < Version(current):
        return UpdateResult(current, latest, "ahead", notes=notes)

    install = detect_install()
    if not install.updatable:
        held: UpdateState = "editable" if install.kind == "editable" else "ephemeral"
        return UpdateResult(current, latest, held, install=install, notes=notes)
    extras = detect_extras(install)
    commands = update_commands(install, install_spec(extras))
    ran = commands[0]
    log: list[str] = []

    def outcome(
        state: UpdateState,
        *,
        detail: str = "",
        installed: str | None = None,
        windows: bool = False,
    ) -> UpdateResult:
        # Every state from here on has an install, its extras and a command: the one
        # that ran, would run, or is the one to run by hand.
        return UpdateResult(
            current,
            latest,
            state,
            install=install,
            extras=extras,
            command=tuple(ran),
            log=tuple(log),
            installed=installed,
            detail=detail,
            notes=(*notes, WINDOWS_NOTE) if windows else notes,
        )

    if check_only:
        return outcome("available")
    if install.kind in ("uv", "pipx") and shutil.which(ran[0]) is None:
        return outcome("failed", detail=f"{ran[0]} is not on PATH: run the command below by hand")

    result = shell.run_logged(ran, log, run)
    if result.returncode != 0 and len(commands) > 1 and shutil.which("uv"):
        # The pip install of a uv-made venv has no pip; ``browser.install`` falls back
        # the same way, and only when uv is there to fall back on.
        ran = commands[1]
        result = shell.run_logged(ran, log, run)
    if result.returncode != 0:
        return outcome(
            "failed",
            detail=f"the update command failed (exit {result.returncode}):"
            " run the command below by hand",
            windows=shell.on_windows(),
        )

    installed = read_installed(install, log, run)
    if installed is None:
        return outcome(
            "failed", detail="the update ran, but the installed version could not be read"
        )
    if Version(installed) < Version(latest):
        # The resolver held it back (a pin elsewhere, an incompatible Python): the
        # command passed and the version did not move to the latest.
        return outcome(
            "failed",
            detail=f"the update installed {installed}, not PyPI's latest {latest}",
            installed=installed,
        )
    return outcome("updated", installed=installed)


def latest_version() -> str:
    """PyPI's ``info.version`` for proofpath. Raises :class:`UpdateError` with the reason.

    The request carries no contact address and no key (section 2.2): the user agent
    is the bare one ``polite`` builds without an address.
    """
    client = httpx.Client(headers={"User-Agent": user_agent()}, timeout=PYPI_TIMEOUT)
    try:
        response = client.get(PYPI_URL)
    except httpx.HTTPError as exc:
        raise UpdateError(f"PyPI did not answer: {type(exc).__name__}: {exc}") from exc
    finally:
        client.close()
    if response.status_code != 200:
        raise UpdateError(f"PyPI did not answer: HTTP {response.status_code}")
    try:
        version = response.json()["info"]["version"]
    except (ValueError, KeyError, TypeError) as exc:
        raise UpdateError("PyPI's answer had no version") from exc
    if not isinstance(version, str) or not version:
        raise UpdateError("PyPI's answer had no version")
    try:
        Version(version)
    except InvalidVersion as exc:
        raise UpdateError(f"PyPI's latest version {version!r} is not a version") from exc
    return version


def direct_url() -> dict[str, Any] | None:
    """PEP 610's ``direct_url.json`` for the installed distribution, or ``None``.

    Only an install from a URL or a local path has one; a plain index install does not.
    """
    try:
        text = metadata.distribution(DIST).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def detect_install() -> Install:
    """Section 2.4, in its order: editable, ephemeral, uv tool, pipx, and pip for
    anything else.

    The two kinds the command never touches come first, whatever else their prefix
    happens to hold: a ``pipx run`` venv carries pipx's metadata like an installed one.
    """
    prefix = Path(sys.prefix)
    executable = Path(sys.executable)
    record = direct_url()
    if record is not None and _is_editable(record):
        return Install("editable", prefix, executable, _source_dir(record))
    rerun = ephemeral_rerun(prefix)
    if rerun:
        return Install("ephemeral", prefix, executable, rerun=rerun)
    if (prefix / "uv-receipt.toml").exists():
        return Install("uv", prefix, venv_python(prefix))
    if (prefix / "pipx_metadata.json").exists():
        return Install("pipx", prefix, venv_python(prefix))
    return Install("pip", prefix, executable)


def ephemeral_rerun(prefix: Path) -> str:
    """The command that fetches the newest release, when ``prefix`` is a temporary
    environment; ``""`` when it is not. Judged from the prefix alone (section 2.4):

    - ``uvx`` / ``uv tool run`` builds its environment in uv's cache, under an
      ``archive-v0`` directory, so that component in the prefix means uv's cache.
    - ``pipx install`` keeps every venv in ``<pipx home>/venvs/<package>``; ``pipx run``
      keeps its venvs in pipx's cache instead, wherever the platform puts that
      (``~/Library/Caches/pipx`` on macOS, ``...\\pipx\\Cache`` on Windows,
      ``$PIPX_HOME/.cache`` when that is set). Both write ``pipx_metadata.json``, so
      pipx's metadata outside a ``venvs`` directory means ``pipx run``.

    Updating either in place would be undone the next time the cache is pruned, and a
    ``pipx install`` into pipx's cache would not be the environment that runs next.
    """
    if "archive-v0" in prefix.parts:
        return UVX_RERUN
    if (prefix / "pipx_metadata.json").exists() and prefix.parent.name.casefold() != "venvs":
        return PIPX_RUN_RERUN
    return ""


def detect_extras(install: Install) -> tuple[str, ...]:
    """The extras to write into the spec, sorted (section 2.5).

    An extra counts when all its modules import. For a uv tool install the extras its
    receipt already records are added: they are what ``uv tool install`` was asked
    for, whether or not they import right now. ``dev`` is never carried over.
    """
    found = {
        extra
        for extra, modules in EXTRA_MODULES.items()
        if all(_importable(module) for module in modules)
    }
    if install.kind == "uv":
        found |= receipt_extras(install.prefix / "uv-receipt.toml")
    return tuple(sorted(found - NEVER_CARRIED))


def receipt_extras(path: Path) -> set[str]:
    """The extras ``uv-receipt.toml`` lists for proofpath; empty when it cannot be read.

    An unreadable receipt costs nothing worse than the importable extras alone, which
    already cover every extra an installed module can prove.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    tool = data.get("tool")
    requirements = tool.get("requirements") if isinstance(tool, dict) else None
    found: set[str] = set()
    for requirement in requirements if isinstance(requirements, list) else ():
        if not isinstance(requirement, dict):
            continue
        if canonicalize_name(str(requirement.get("name", ""))) != DIST:
            continue
        extras = requirement.get("extras")
        if isinstance(extras, list):
            found.update(str(extra) for extra in extras)
    return found


def install_spec(extras: tuple[str, ...]) -> str:
    """``proofpath[a,b]``, or ``proofpath``. Never version-pinned (section 2.6): a pin
    would freeze the uv receipt or the pipx metadata, and a later upgrade would not move."""
    return f"{DIST}[{','.join(extras)}]" if extras else DIST


def update_commands(install: Install, spec: str) -> list[list[str]]:
    """The commands that update ``install``, in the order they are tried. Kept as data
    so each install kind's command is testable without a subprocess.

    Only pip has a second one: ``uv pip install`` into the same interpreter, the
    fallback ``browser.install`` uses. An editable install has none.
    """
    if install.kind == "uv":
        return [["uv", "tool", "install", "--upgrade", spec]]
    if install.kind == "pipx":
        return [["pipx", "install", "--force", spec]]
    if install.kind == "pip":
        python = str(install.python)
        return [
            [python, "-m", "pip", "install", "--upgrade", spec],
            ["uv", "pip", "install", "--python", python, "--upgrade", spec],
        ]
    return []


def read_installed(install: Install, log: list[str], run: Runner) -> str | None:
    """The version a fresh interpreter sees after the update, or ``None``."""
    probe = [str(install.python), "-I", "-c", VERSION_PROBE]
    result = shell.run_logged(probe, log, run)
    lines = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not lines:
        return None
    text = lines[-1].strip()
    try:
        Version(text)
    except InvalidVersion:
        return None
    return text


def venv_python(prefix: Path) -> Path:
    """The interpreter inside the venv at ``prefix``, by this platform's layout."""
    return prefix / "Scripts" / "python.exe" if shell.on_windows() else prefix / "bin" / "python"


def _importable(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        # A module already imported with no spec raises ValueError; a broken parent
        # package raises ImportError. Neither proves the extra is installed.
        return False


def _is_editable(record: dict[str, Any]) -> bool:
    dir_info = record.get("dir_info")
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


def _source_dir(record: dict[str, Any]) -> Path | None:
    url = record.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(parsed.path))


__all__ = [
    "EDITABLE_HINT",
    "PYPI_URL",
    "VERSION_PROBE",
    "WINDOWS_NOTE",
    "Install",
    "InstallKind",
    "UpdateError",
    "UpdateResult",
    "UpdateState",
    "detect_extras",
    "detect_install",
    "direct_url",
    "ephemeral_rerun",
    "install_spec",
    "latest_version",
    "update",
    "update_commands",
]
