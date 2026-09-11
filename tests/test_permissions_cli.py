from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from proofpath.cli import app

runner = CliRunner()


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_permissions_prints_path_and_current_values(config_dir: Path) -> None:
    result = runner.invoke(app, ["permissions"])
    assert result.exit_code == 0
    assert str(config_dir / "config.toml") in result.stdout
    assert "install_browser" in result.stdout
    assert "ask" in result.stdout


def test_permissions_set_persists_and_reports(config_dir: Path) -> None:
    result = runner.invoke(app, ["permissions", "set", "install_browser", "deny"])
    assert result.exit_code == 0
    assert (config_dir / "config.toml").exists()
    shown = runner.invoke(app, ["permissions"])
    assert "deny" in shown.stdout


def test_permissions_set_rejects_bad_value(config_dir: Path) -> None:
    result = runner.invoke(app, ["permissions", "set", "install_browser", "maybe"])
    assert result.exit_code == 2
    assert "maybe" in result.output
