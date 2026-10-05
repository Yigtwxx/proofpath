"""``proofpath update [--check]``: wiring, output, exit codes (update spec section 2.10).

The ``--check`` paths run end to end with PyPI answering through ``respx``: nothing
is installed by a check, so nothing has to be stubbed below the library. The paths
that would install stub ``commands.update_install`` instead; what the command does
is ``test_update.py``'s business, and this file only checks how the CLI says it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from proofpath import commands as lib
from proofpath import update
from proofpath.cli import app
from proofpath.config import Config, set_value

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A plain pip install of 1.0.0 under a fresh config dir."""
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    prefix = tmp_path / "venv"
    prefix.mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(update, "__version__", "1.0.0")
    monkeypatch.setattr(update, "direct_url", lambda: None)
    monkeypatch.setattr(update, "find_spec", lambda name: None)
    return prefix


def pypi(version: str) -> respx.Route:
    return respx.get(update.PYPI_URL).mock(
        return_value=httpx.Response(200, json={"info": {"version": version}})
    )


@respx.mock
def test_check_when_up_to_date_exits_zero() -> None:
    pypi("1.0.0")
    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 0, result.output
    assert "current    1.0.0" in result.stdout
    assert "latest     1.0.0" in result.stdout
    assert "state      up to date" in result.stdout


@respx.mock
def test_check_with_an_update_available_exits_zero_and_shows_the_command() -> None:
    pypi("1.1.0")
    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 0, result.output
    assert "latest     1.1.0" in result.stdout
    assert "update available: 1.0.0 → 1.1.0" in result.stdout
    assert "install    pip" in result.stdout
    assert "extras     none" in result.stdout
    assert "-m pip install --upgrade proofpath" in result.stdout
    assert "proofpath update installs it" in result.stdout


@respx.mock
def test_a_pypi_failure_exits_one_with_the_reason() -> None:
    respx.get(update.PYPI_URL).mock(return_value=httpx.Response(503))
    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 1
    assert "HTTP 503" in result.stderr
    assert "up to date" not in result.output


@respx.mock(assert_all_called=False)
def test_network_deny_sends_nothing_and_exits_one() -> None:
    route = pypi("1.1.0")
    set_value("permissions.network", "deny")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    assert not route.called
    assert "permissions.network" in result.stderr


@respx.mock(assert_all_called=False)
def test_network_ask_without_a_terminal_sends_nothing_and_exits_one() -> None:
    """Rule 4: CliRunner is no TTY, so ``ask`` is ``deny``, and the CLI says why."""
    route = pypi("1.1.0")
    set_value("permissions.network", "ask")
    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 1
    assert not route.called
    assert "no interactive terminal" in result.stderr


@respx.mock
def test_an_editable_install_exits_zero_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Update spec section 2.10: editable is not a failure, it is updated by hand."""
    pypi("1.1.0")
    source = tmp_path / "checkout"
    monkeypatch.setattr(
        update, "direct_url", lambda: {"url": source.as_uri(), "dir_info": {"editable": True}}
    )

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("an editable install must never run anything")

    monkeypatch.setattr(update.subprocess, "run", never)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    assert "install    editable" in result.stdout
    assert "git pull and uv sync" in result.stdout
    assert "command" not in result.stdout


@respx.mock
def test_an_ephemeral_environment_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pypi("1.1.0")
    prefix = tmp_path / "uv-cache" / "archive-v0" / "abc123"
    prefix.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    assert "temporary environment" in result.stdout
    assert "uvx proofpath@latest" in result.stdout


def test_a_successful_update_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[bool] = []

    def stub(*, config: Config, interactive: bool, check_only: bool = False) -> update.UpdateResult:
        asked.append(check_only)
        return update.UpdateResult(
            current="1.0.0",
            latest="1.1.0",
            state="updated",
            install=update.Install("uv", Path("/venv"), Path("/venv/bin/python")),
            extras=("browser",),
            command=("uv", "tool", "install", "--upgrade", "proofpath[browser]"),
            log=("$ uv tool install --upgrade 'proofpath[browser]'", "exit 0"),
            installed="1.1.0",
        )

    monkeypatch.setattr(lib, "update_install", stub)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output
    assert asked == [False]
    assert "install    uv tool" in result.stdout
    assert "extras     browser" in result.stdout
    assert "run        $ uv tool install --upgrade 'proofpath[browser]'" in result.stdout
    assert "state      updated to 1.1.0" in result.stdout


def test_a_failed_update_exits_one_and_prints_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = update.UpdateResult(
        current="1.0.0",
        latest="1.1.0",
        state="failed",
        install=update.Install("uv", Path("/venv"), Path("/venv/bin/python")),
        command=("uv", "tool", "install", "--upgrade", "proofpath"),
        log=("$ uv tool install --upgrade proofpath", "exit 1", "Access is denied."),
        detail="the update command failed (exit 1)",
        notes=(update.WINDOWS_NOTE,),
    )
    monkeypatch.setattr(lib, "update_install", lambda **kwargs: failed)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    assert "the update command failed (exit 1)" in result.stderr
    assert "command    uv tool install --upgrade proofpath" in result.stdout
    assert "Access is denied." in result.stdout
    assert f"note       {update.WINDOWS_NOTE}" in result.stdout


def test_an_unknown_option_is_a_usage_error() -> None:
    result = runner.invoke(app, ["update", "--nope"])
    assert result.exit_code == 2
