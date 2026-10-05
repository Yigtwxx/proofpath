"""``proofpath update``: the version check, the install kinds, the command, the
verification (docs/superpowers/specs/2026-10-05-update-command-design.md).

Nothing here touches the network or installs anything. PyPI answers through
``respx``, the install kind is a tmp directory standing in for ``sys.prefix``, the
extras are a faked ``find_spec`` and every command goes to a runner that records it.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from proofpath import shell, update
from proofpath.config import Config, Contact, Permissions

CURRENT = "1.0.0"
NEWER = "1.1.0"
PYPI = update.PYPI_URL


class FakeRun:
    """Stands in for ``subprocess.run``: records each command, answers from a script.

    Each scripted answer is ``(returncode, stdout, stderr)``; once the script runs out
    every further command succeeds and prints nothing.
    """

    def __init__(self, *answers: tuple[int, str, str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        self.kwargs.append(kwargs)
        code, stdout, stderr = self.answers.pop(0) if self.answers else (0, "", "")
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr=stderr)


def ok(version: str = NEWER) -> tuple[tuple[int, str, str], tuple[int, str, str]]:
    """The install command succeeds, then the fresh process reports ``version``."""
    return (0, "", ""), (0, f"{version}\n", "")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A plain pip install of version ``CURRENT``, no extras, every tool on PATH.

    Each test turns it into the install it needs by writing a marker file into the
    prefix it returns, or by patching one of the hooks below.
    """
    # Inside a ``venvs`` directory, as a ``pipx install`` venv is: the ``pipx`` marker
    # below then means an installed pipx venv, not a ``pipx run`` one.
    prefix = tmp_path / "venvs" / "proofpath"
    (prefix / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "executable", str(prefix / "bin" / "python3"))
    monkeypatch.setattr(update, "__version__", CURRENT)
    monkeypatch.setattr(update, "direct_url", lambda: None)
    monkeypatch.setattr(update, "find_spec", lambda name: None)
    monkeypatch.setattr(update.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(shell, "on_windows", lambda: False)
    return prefix


@pytest.fixture
def pypi() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def latest(router: respx.MockRouter, version: str = NEWER) -> respx.Route:
    return router.get(PYPI).mock(
        return_value=httpx.Response(200, json={"info": {"version": version}})
    )


def venv_python(prefix: Path) -> str:
    # ``env`` pins ``shell.on_windows`` to False, and the venv layout follows that hook, so
    # this is the POSIX layout on every runner, Windows included.
    return str(prefix / "bin" / "python")


def uv_tool(prefix: Path, *extras: str) -> None:
    requirement = '{ name = "proofpath"'
    if extras:
        requirement += ", extras = [" + ", ".join(f'"{e}"' for e in extras) + "]"
    requirement += " }"
    (prefix / "uv-receipt.toml").write_text(
        f"[tool]\nrequirements = [{requirement}]\n", encoding="utf-8"
    )


def pipx(prefix: Path) -> None:
    (prefix / "pipx_metadata.json").write_text("{}", encoding="utf-8")


def with_extras(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    present = set(modules)
    monkeypatch.setattr(update, "find_spec", lambda name: object() if name in present else None)


# --- the version check ---------------------------------------------------------------


def test_up_to_date_runs_nothing(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi, CURRENT)
    run = FakeRun()
    result = update.update(Config(), run=run)
    assert result.state == "up to date"
    assert result.ok
    assert (result.current, result.latest) == (CURRENT, CURRENT)
    assert run.calls == []


def test_a_build_ahead_of_pypi_runs_nothing(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi, "0.9.0")
    run = FakeRun()
    result = update.update(Config(), run=run)
    assert result.state == "ahead"
    assert result.ok
    assert "development build" in result.summary
    assert run.calls == []


def test_versions_compare_as_versions_not_strings(env: Path, pypi: respx.MockRouter) -> None:
    """``1.10.0`` is newer than ``1.9.0``, although it sorts before it as text."""
    latest(pypi, "1.10.0")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(update, "__version__", "1.9.0")
        result = update.update(Config(), check_only=True, run=FakeRun())
    assert result.state == "available"


def test_check_reports_an_update_and_runs_nothing(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi)
    run = FakeRun()
    result = update.update(Config(), check_only=True, run=run)
    assert result.state == "available"
    assert result.ok
    assert result.latest == NEWER
    assert run.calls == []
    # What would run is still worked out, so ``--check`` can show it.
    assert result.command == (sys.executable, "-m", "pip", "install", "--upgrade", "proofpath")


def test_the_request_carries_no_contact_address(env: Path, pypi: respx.MockRouter) -> None:
    route = latest(pypi, CURRENT)
    config = Config(contact=Contact(email="someone@example.org"))
    update.update(config, run=FakeRun())
    assert route.called
    request = route.calls.last.request
    assert "someone@example.org" not in str(request.url)
    assert all("someone@example.org" not in value for value in request.headers.values())


def test_network_ask_without_a_terminal_is_deny_and_says_so(
    env: Path, pypi: respx.MockRouter
) -> None:
    """Rule 4: no TTY, no question, and ``ask`` is ``deny`` -- reported, not silent."""
    route = latest(pypi)
    run = FakeRun()
    config = Config(permissions=Permissions(network="ask"))
    result = update.update(config, interactive=False, run=run)
    assert not route.called
    assert result.state == "failed"
    assert result.latest is None
    assert "no interactive terminal" in result.summary
    assert "permissions.network allow" in result.summary
    assert run.calls == []


def test_network_ask_with_a_terminal_goes_ahead_with_a_note(
    env: Path, pypi: respx.MockRouter
) -> None:
    route = latest(pypi, CURRENT)
    config = Config(permissions=Permissions(network="ask"))
    result = update.update(config, interactive=True, run=FakeRun())
    assert route.called
    assert result.state == "up to date"
    assert result.notes == ("network permission is 'ask'; allowed for this run",)


def test_network_deny_sends_nothing(env: Path, pypi: respx.MockRouter) -> None:
    route = latest(pypi)
    run = FakeRun()
    config = Config(permissions=Permissions(network="deny"))
    result = update.update(config, interactive=True, run=run)
    assert not route.called
    assert result.state == "failed"
    assert not result.ok
    # Never "up to date": nobody was asked (rule 2).
    assert result.latest is None
    assert "permissions.network" in result.summary
    assert run.calls == []


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(503), "HTTP 503"),
        (httpx.Response(200, json={"info": {}}), "no version"),
        (httpx.Response(200, text="not json"), "no version"),
        (httpx.Response(200, json={"info": {"version": "not a version"}}), "not a version"),
    ],
)
def test_a_pypi_failure_is_a_failure_with_its_reason(
    env: Path, pypi: respx.MockRouter, response: httpx.Response, reason: str
) -> None:
    pypi.get(PYPI).mock(return_value=response)
    run = FakeRun()
    result = update.update(Config(), run=run)
    assert result.state == "failed"
    assert reason in result.summary
    assert run.calls == []


def test_an_unreachable_pypi_is_a_failure(env: Path, pypi: respx.MockRouter) -> None:
    pypi.get(PYPI).mock(side_effect=httpx.ConnectError("no route to host"))
    result = update.update(Config(), run=FakeRun())
    assert result.state == "failed"
    assert "ConnectError" in result.summary


# --- install detection and the command ----------------------------------------------


def test_a_uv_tool_install_is_upgraded_through_uv(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi)
    uv_tool(env)
    run = FakeRun(*ok())
    result = update.update(Config(), run=run)
    assert result.install is not None and result.install.kind == "uv"
    assert run.calls == [
        ["uv", "tool", "install", "--upgrade", "proofpath"],
        [venv_python(env), "-I", "-c", update.VERSION_PROBE],
    ]
    assert result.state == "updated"
    assert result.installed == NEWER
    assert result.ok


def test_extras_are_the_union_of_the_importable_ones_and_the_receipt(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    uv_tool(env, "gpu", "dev")
    with_extras(monkeypatch, "scrapling", "patchright")
    run = FakeRun(*ok())
    result = update.update(Config(), run=run)
    assert result.extras == ("browser", "gpu")
    # ``dev`` is never carried over, even when the receipt names it.
    assert run.calls[0] == ["uv", "tool", "install", "--upgrade", "proofpath[browser,gpu]"]


@pytest.mark.parametrize(
    ("modules", "extras"),
    [
        (("scrapling",), ()),  # the browser extra needs patchright too
        (("scrapling", "patchright"), ("browser",)),
        (("torch",), ()),  # the gpu extra needs sentence_transformers too
        (("torch", "sentence_transformers"), ("gpu",)),
    ],
)
def test_an_extra_counts_only_when_all_its_modules_import(
    env: Path,
    pypi: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    modules: tuple[str, ...],
    extras: tuple[str, ...],
) -> None:
    latest(pypi)
    with_extras(monkeypatch, *modules)
    result = update.update(Config(), check_only=True, run=FakeRun())
    assert result.extras == extras


def test_a_pipx_install_is_reinstalled_through_pipx(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    pipx(env)
    with_extras(monkeypatch, "scrapling", "patchright")
    run = FakeRun(*ok())
    result = update.update(Config(), run=run)
    assert result.install is not None and result.install.kind == "pipx"
    assert run.calls == [
        ["pipx", "install", "--force", "proofpath[browser]"],
        [venv_python(env), "-I", "-c", update.VERSION_PROBE],
    ]
    assert result.state == "updated"


def test_a_pip_install_is_upgraded_with_its_own_interpreter(
    env: Path, pypi: respx.MockRouter
) -> None:
    latest(pypi)
    run = FakeRun(*ok())
    result = update.update(Config(), run=run)
    assert result.install is not None and result.install.kind == "pip"
    assert run.calls == [
        [sys.executable, "-m", "pip", "install", "--upgrade", "proofpath"],
        [sys.executable, "-I", "-c", update.VERSION_PROBE],
    ]
    assert result.state == "updated"


def test_pip_falls_back_to_uv_pip_when_pip_fails(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi)
    run = FakeRun((1, "", "No module named pip"), *ok())
    result = update.update(Config(), run=run)
    assert run.calls[1] == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--upgrade",
        "proofpath",
    ]
    assert run.calls[2] == [sys.executable, "-I", "-c", update.VERSION_PROBE]
    assert result.state == "updated"


def test_pip_without_uv_to_fall_back_on_fails(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    monkeypatch.setattr(update.shutil, "which", lambda name: None)
    run = FakeRun((1, "", "No module named pip"))
    result = update.update(Config(), run=run)
    assert len(run.calls) == 1
    assert result.state == "failed"
    assert result.command == (sys.executable, "-m", "pip", "install", "--upgrade", "proofpath")


def test_the_spec_is_never_pinned(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin would freeze the uv receipt or pipx metadata at this version."""
    latest(pypi)
    uv_tool(env, "browser")
    with_extras(monkeypatch, "torch", "sentence_transformers")
    run = FakeRun(*ok())
    update.update(Config(), run=run)
    spec = run.calls[0][-1]
    assert spec == "proofpath[browser,gpu]"
    assert not any(marker in spec for marker in ("=", "<", ">", "~", NEWER))


def test_an_editable_install_is_never_touched(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    latest(pypi)
    uv_tool(env)  # editable wins over every other marker
    source = tmp_path / "proofpath-src"
    monkeypatch.setattr(
        update,
        "direct_url",
        lambda: {"url": source.as_uri(), "dir_info": {"editable": True}},
    )
    run = FakeRun()
    result = update.update(Config(), run=run)
    assert run.calls == []
    assert result.state == "editable"
    assert result.install is not None and result.install.kind == "editable"
    assert result.install.source == source
    assert str(source) in result.summary
    assert "git pull" in result.summary and "uv sync" in result.summary


@pytest.mark.parametrize(
    ("parts", "rerun"),
    [
        # ``uvx`` / ``uv tool run``: uv's cache, ``<cache>/archive-v0/<hash>``.
        ((".cache", "uv", "archive-v0", "Xa1b2"), "uvx proofpath@latest"),
        # ``pipx run``: pipx's cache, ``$PIPX_HOME/.cache/<hash>``.
        (("pipx", ".cache", "0f9e8d"), "pipx run --no-cache proofpath"),
    ],
)
def test_an_ephemeral_environment_runs_nothing(
    env: Path,
    pypi: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parts: tuple[str, ...],
    rerun: str,
) -> None:
    latest(pypi)
    prefix = tmp_path.joinpath(*parts)
    prefix.mkdir(parents=True)
    pipx(prefix)  # a ``pipx run`` venv has pipx's metadata too; ephemeral wins
    monkeypatch.setattr(sys, "prefix", str(prefix))
    run = FakeRun()
    for check_only in (False, True):
        result = update.update(Config(), check_only=check_only, run=run)
        assert result.state == "ephemeral"
        assert result.ok
        assert result.install is not None and result.install.kind == "ephemeral"
        assert "temporary environment" in result.summary
        assert f"`{rerun}`" in result.summary
        assert result.command == ()
    assert run.calls == []


@pytest.mark.parametrize(
    ("parts", "kind"),
    [
        # ``pipx run`` venvs live in pipx's cache: platformdirs' user cache when
        # PIPX_HOME is unset, or ``$PIPX_HOME/.cache``. Never in a ``venvs`` directory.
        (("Library", "Caches", "pipx", "0f9e8d"), "ephemeral"),  # macOS
        (("AppData", "Local", "pipx", "pipx", "Cache", "0f9e8d"), "ephemeral"),  # Windows
        ((".local", "pipx", ".cache", "0f9e8d"), "ephemeral"),  # Linux, PIPX_HOME
        # ``pipx install`` venvs live in ``<pipx home>/venvs/<package>``.
        ((".local", "pipx", "venvs", "proofpath"), "pipx"),
        (("Library", "Application Support", "pipx", "venvs", "proofpath"), "pipx"),
        (("AppData", "Local", "pipx", "pipx", "venvs", "proofpath"), "pipx"),
    ],
)
def test_a_pipx_run_venv_is_told_apart_from_a_pipx_install(
    env: Path,
    pypi: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parts: tuple[str, ...],
    kind: str,
) -> None:
    latest(pypi)
    prefix = tmp_path.joinpath(*parts)
    prefix.mkdir(parents=True)
    pipx(prefix)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    run = FakeRun()
    result = update.update(Config(), run=run)
    assert result.install is not None and result.install.kind == kind
    if kind == "ephemeral":
        assert run.calls == []  # never installed for good into pipx's cache
        assert "pipx run --no-cache proofpath" in result.summary


def test_a_non_editable_direct_url_is_not_editable(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel installed from a local path has a ``direct_url.json`` too."""
    latest(pypi)
    monkeypatch.setattr(update, "direct_url", lambda: {"url": "file:///w.whl", "archive_info": {}})
    result = update.update(Config(), check_only=True, run=FakeRun())
    assert result.install is not None and result.install.kind == "pip"


def test_direct_url_reads_pep_610_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dist:
        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return json.dumps({"url": "file:///src", "dir_info": {"editable": True}})

    monkeypatch.setattr(update.metadata, "distribution", lambda name: Dist())
    assert update.direct_url() == {"url": "file:///src", "dir_info": {"editable": True}}

    def missing(name: str) -> Any:
        raise update.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(update.metadata, "distribution", missing)
    assert update.direct_url() is None


@pytest.mark.parametrize(("marker", "tool"), [(uv_tool, "uv"), (pipx, "pipx")])
def test_a_missing_tool_is_a_failure_that_prints_the_command(
    env: Path,
    pypi: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    marker: Any,
    tool: str,
) -> None:
    latest(pypi)
    marker(env)
    monkeypatch.setattr(update.shutil, "which", lambda name: None)
    run = FakeRun()
    result = update.update(Config(), run=run)
    assert run.calls == []
    assert result.state == "failed"
    assert f"{tool} is not on PATH" in result.summary
    assert result.command[0] == tool


# --- running it ---------------------------------------------------------------------


def test_output_is_captured_never_inherited(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi)
    uv_tool(env)
    run = FakeRun(*ok())
    update.update(Config(), run=run)
    assert all(kwargs.get("capture_output") is True for kwargs in run.kwargs)


def test_every_command_is_logged_so_it_can_be_pasted_back(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    uv_tool(env)
    with_extras(monkeypatch, "scrapling", "patchright")
    result = update.update(Config(), run=FakeRun(*ok()))
    assert result.log[0] == "$ uv tool install --upgrade 'proofpath[browser]'"
    assert result.log[1] == "exit 0"


def test_a_failed_command_says_so_and_prints_it(env: Path, pypi: respx.MockRouter) -> None:
    latest(pypi)
    uv_tool(env)
    run = FakeRun((2, "", "error: resolution failed"))
    result = update.update(Config(), run=run)
    assert len(run.calls) == 1  # nothing to verify
    assert result.state == "failed"
    assert "exit 2" in result.summary
    assert "error: resolution failed" in result.log
    assert result.command == ("uv", "tool", "install", "--upgrade", "proofpath")
    assert update.WINDOWS_NOTE not in result.notes


def test_a_failed_command_on_windows_adds_the_lock_line(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    uv_tool(env)
    monkeypatch.setattr(shell, "on_windows", lambda: True)
    result = update.update(Config(), run=FakeRun((1, "", "Access is denied.")))
    assert result.state == "failed"
    assert update.WINDOWS_NOTE in result.notes
    assert update.WINDOWS_NOTE == "close proofpath and run the command above"


def test_a_success_on_windows_adds_no_lock_line(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    uv_tool(env)
    monkeypatch.setattr(shell, "on_windows", lambda: True)
    result = update.update(Config(), run=FakeRun(*ok()))
    assert result.state == "updated"
    assert result.notes == ()


def test_commands_are_shown_the_way_this_platform_pastes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cmd = ["C:\\Program Files\\Python\\python.exe", "-m", "pip", "install", "proofpath[browser]"]
    monkeypatch.setattr(shell, "on_windows", lambda: False)
    assert shell.display_command(cmd) == shlex.join(cmd)
    monkeypatch.setattr(shell, "on_windows", lambda: True)
    assert shell.display_command(cmd) == subprocess.list2cmdline(cmd)
    assert shell.display_command(cmd).startswith('"C:\\Program Files\\Python\\python.exe" -m')


def test_the_log_on_windows_is_pasteable_into_cmd(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    uv_tool(env)
    with_extras(monkeypatch, "scrapling", "patchright")
    monkeypatch.setattr(shell, "on_windows", lambda: True)
    result = update.update(Config(), run=FakeRun(*ok()))
    # ``shlex.join`` would wrap the spec in single quotes, which cmd.exe keeps.
    assert result.log[0] == "$ uv tool install --upgrade proofpath[browser]"


def test_a_windows_venv_is_verified_with_its_scripts_python(
    env: Path, pypi: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest(pypi)
    uv_tool(env)
    monkeypatch.setattr(shell, "on_windows", lambda: True)
    run = FakeRun(*ok())
    result = update.update(Config(), run=run)
    assert run.calls[1] == [str(env / "Scripts" / "python.exe"), "-I", "-c", update.VERSION_PROBE]
    assert result.state == "updated"


def test_verification_that_sees_an_old_version_is_a_failure(
    env: Path, pypi: respx.MockRouter
) -> None:
    """The resolver held the update back: the command passed, the version did not move."""
    latest(pypi)
    uv_tool(env)
    result = update.update(Config(), run=FakeRun(*ok(CURRENT)))
    assert result.state == "failed"
    assert result.installed == CURRENT
    assert CURRENT in result.summary and NEWER in result.summary


def test_verification_that_cannot_read_a_version_is_a_failure(
    env: Path, pypi: respx.MockRouter
) -> None:
    latest(pypi)
    uv_tool(env)
    result = update.update(Config(), run=FakeRun((0, "", ""), (1, "", "ModuleNotFoundError")))
    assert result.state == "failed"
    assert result.installed is None
    assert "installed version" in result.summary
