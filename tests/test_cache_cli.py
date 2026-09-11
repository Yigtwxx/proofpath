from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from proofpath.cache import Cache
from proofpath.cli import app
from proofpath.models import Label, Passage, Verdict

runner = CliRunner()
NOW = datetime(2026, 9, 11, tzinfo=timezone.utc)


@pytest.fixture
def cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path))
    with Cache() as db:
        db.add_source(
            "doi:10.1/x",
            scheme="academic",
            title="Paper X",
            url="",
            text_kind="abstract",
            raw_text="cats purr.",
            now=NOW,
        )
        db.put_chunks(
            "doi:10.1/x",
            "bge@rev",
            [Passage("cats purr.", "doi:10.1/x", 0)],
            np.ones((1, 3), dtype=np.float32),
        )
        db.put_verdict(
            "h",
            "doi:10.1/x",
            "m",
            Verdict(Label.SUPPORTED, 0.9, "high", Passage("cats purr.", "doi:10.1/x", 0)),
            now=NOW,
        )
        db.add_source(
            "old",
            scheme="academic",
            title="Old",
            url="",
            text_kind="abstract",
            raw_text="t",
            now=NOW - timedelta(days=30),
        )
        return db.path


def test_cache_path_prints_the_file(cache_path: Path) -> None:
    result = runner.invoke(app, ["cache", "path"])
    assert result.exit_code == 0
    assert result.stdout.strip() == str(cache_path)


def test_cache_ls_lists_sources_with_counts(cache_path: Path) -> None:
    result = runner.invoke(app, ["cache", "ls"])
    assert result.exit_code == 0
    assert "doi:10.1/x" in result.stdout and "Paper X" in result.stdout
    assert "1 chunk" in result.stdout and "1 verdict" in result.stdout
    assert "expired" in result.stdout  # the 30-day-old one


def test_cache_show_prints_chunks_and_verdicts(cache_path: Path) -> None:
    result = runner.invoke(app, ["cache", "show", "doi:10.1/x"])
    assert result.exit_code == 0
    assert "cats purr." in result.stdout
    assert "SUPPORTED" in result.stdout and "dim=3" in result.stdout


def test_cache_show_unknown_source_fails(cache_path: Path) -> None:
    result = runner.invoke(app, ["cache", "show", "nope"])
    assert result.exit_code == 2


def test_cache_clear_expired_then_all(cache_path: Path) -> None:
    result = runner.invoke(app, ["cache", "clear", "--expired"])
    assert result.exit_code == 0 and "1 source" in result.stdout
    result = runner.invoke(app, ["cache", "clear"])
    assert result.exit_code == 0 and "1 source" in result.stdout
    assert "0 source" in runner.invoke(app, ["cache"]).stdout
