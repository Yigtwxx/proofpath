"""The bare invocation and the global flags (spec section 13.3).

``proofpath`` with no subcommand is the TUI, so ``tui.app.run`` is replaced here
rather than started: a real ``App.run()`` wants a terminal, and what this file is
about is the wiring — which config and which ``Ui`` the front-end is handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proofpath import __version__, ui
from proofpath.cli import app
from proofpath.config import Config
from proofpath.tui import app as tui_app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config directory of this test's own, so the run reads no real file."""
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))


def record(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the TUI's entry point and return what the CLI passed it."""
    seen: dict[str, Any] = {}

    def fake_run(config: Config, out: ui.Ui) -> None:
        seen["config"] = config
        seen["out"] = out

    monkeypatch.setattr(tui_app, "run", fake_run)
    return seen


def test_version_flag_reports_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_flag_is_one_line_and_says_nothing_else() -> None:
    """It is read by scripts as often as by people: one line, the version in it."""
    result = runner.invoke(app, ["--version"])
    assert result.stdout.strip() == f"proofpath {__version__}"


def test_bare_invocation_opens_the_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not a help screen and no longer a "coming in v0.2" line: the front-end itself."""
    seen = record(monkeypatch)
    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert isinstance(seen["config"], Config)
    assert isinstance(seen["out"], ui.Ui)


def test_the_tui_is_handed_the_same_ui_the_global_flags_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-color`` and ``-q`` are the app callback's, so they bind both front-ends."""
    seen = record(monkeypatch)
    result = runner.invoke(app, ["--no-color", "-q"])
    assert result.exit_code == 0, result.output
    assert seen["out"].color is False
    assert seen["out"].quiet is True


def test_a_broken_config_file_is_an_error_not_a_half_opened_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The TUI is opened with a config, so a config that will not load stops here —
    on one line, with exit 2, rather than inside a screen the user cannot read."""
    seen = record(monkeypatch)
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "config.toml").write_text("permissions = [oops\n", encoding="utf-8")
    result = runner.invoke(app, [])
    assert result.exit_code == 2, result.output
    assert "error:" in result.output
    assert "config" not in seen
