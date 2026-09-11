from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from proofpath import judge
from proofpath.cli import app

runner = CliRunner()


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    conf = tmp_path / "conf"
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(conf))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return conf


# --- config / show / path ---------------------------------------------------------


def test_config_prints_path_and_every_section_as_toml(config_dir: Path) -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert str(config_dir / "config.toml") in result.stdout
    assert "not written yet" in result.stdout
    for section in ("[permissions]", "[fetch]", "[contact]", "[judge]"):
        assert section in result.stdout
    assert 'install_browser = "ask"' in result.stdout
    assert "GROQ_API_KEY" in result.stdout


def test_config_show_is_the_same_as_bare_config(config_dir: Path) -> None:
    assert runner.invoke(app, ["config", "show"]).stdout == runner.invoke(app, ["config"]).stdout


def test_config_path_prints_only_the_path(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert result.stdout.strip() == str(config_dir / "config.toml")


# --- config set -------------------------------------------------------------------


def test_config_set_permission_persists_and_reports(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "set", "permissions.install_browser", "allow"])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("permissions.install_browser = allow  (")
    assert str(config_dir / "config.toml") in result.stdout
    assert (config_dir / "config.toml").exists()
    assert 'install_browser = "allow"' in runner.invoke(app, ["config"]).stdout
    assert "not written yet" not in runner.invoke(app, ["config"]).stdout


def test_config_set_judge_model(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "set", "judge.model", "x"])
    assert result.exit_code == 0, result.output
    assert 'model = "x"' in runner.invoke(app, ["config"]).stdout


def test_config_set_judge_provider_applies_the_provider_preset(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "set", "judge.provider", "gemini"])
    assert result.exit_code == 0, result.output
    assert "judge.provider = gemini" in result.stdout
    assert "judge.model = gemini-3.8-flash" in result.stdout
    assert "judge.api_key_env = GEMINI_API_KEY" in result.stdout
    show = runner.invoke(app, ["config"]).stdout
    assert 'model = "gemini-3.8-flash"' in show
    assert 'api_key_env = "GEMINI_API_KEY"' in show
    assert 'provider = "gemini"' in show


@pytest.mark.parametrize("name", ["nvidia", "bogus"])
def test_config_set_judge_provider_rejects_unknown_provider(config_dir: Path, name: str) -> None:
    result = runner.invoke(app, ["config", "set", "judge.provider", name])
    assert result.exit_code == 2
    assert "unknown provider" in result.output
    assert "groq" in result.output and "gemini" in result.output and "ollama" in result.output


def test_config_set_rejects_bad_value(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "set", "permissions.install_browser", "maybe"])
    assert result.exit_code == 2
    assert "maybe" in result.output


def test_config_set_rejects_unknown_key(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "set", "bogus.key", "v"])
    assert result.exit_code == 2
    assert "bogus.key" in result.output


# --- config check -----------------------------------------------------------------


def test_config_check_reports_missing_key_and_where_to_put_it(config_dir: Path) -> None:
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 2
    assert "GROQ_API_KEY" in result.output
    assert str(config_dir.parent / ".env") in result.output


def test_config_check_finds_key_in_dotenv_and_never_prints_it(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (config_dir.parent / ".env").write_text("GROQ_API_KEY=gsk_secret\n", encoding="utf-8")
    monkeypatch.setattr(
        judge, "check", lambda cfg, key, **_: judge.CheckResult(True, cfg.model, 123, "ok")
    )
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 0, result.output
    assert "gsk_secret" not in result.output
    assert str(config_dir.parent / ".env") in result.output
    assert "openai/gpt-oss-120b" in result.output
    assert "123" in result.output


def test_config_check_reports_a_failed_probe(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    monkeypatch.setattr(
        judge, "check", lambda cfg, key, **_: judge.CheckResult(False, cfg.model, 0, "401")
    )
    result = runner.invoke(app, ["config", "check"])
    assert result.exit_code == 2
    assert "FAILED" in result.output and "401" in result.output
    assert "gsk_secret" not in result.output


# --- removed groups ---------------------------------------------------------------


@pytest.mark.parametrize("group", ["permissions", "judge"])
def test_old_top_level_groups_are_gone(config_dir: Path, group: str) -> None:
    result = runner.invoke(app, [group])
    assert result.exit_code == 2
    assert "No such command" in result.output
