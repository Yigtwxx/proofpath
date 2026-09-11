from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from proofpath import judge
from proofpath.cli import app

runner = CliRunner()


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return tmp_path


def test_judge_check_reports_missing_key_and_where_to_put_it(isolated: Path) -> None:
    result = runner.invoke(app, ["judge", "check"])
    assert result.exit_code == 2
    assert "GROQ_API_KEY" in result.output
    assert str(isolated / ".env") in result.output


def test_judge_check_finds_key_in_dotenv_and_never_prints_it(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (isolated / ".env").write_text("GROQ_API_KEY=gsk_secret\n", encoding="utf-8")
    monkeypatch.setattr(
        judge, "check", lambda cfg, key, **_: judge.CheckResult(True, cfg.model, 123, "ok")
    )
    result = runner.invoke(app, ["judge", "check"])
    assert result.exit_code == 0
    assert "gsk_secret" not in result.output
    assert str(isolated / ".env") in result.output
    assert "openai/gpt-oss-120b" in result.output
    assert "123" in result.output


def test_judge_set_provider_applies_that_providers_defaults(isolated: Path) -> None:
    result = runner.invoke(app, ["judge", "set", "provider", "gemini"])
    assert result.exit_code == 0
    shown = runner.invoke(app, ["judge"])
    assert "gemini-3.8-flash" in shown.output
    assert "GEMINI_API_KEY" in shown.output
