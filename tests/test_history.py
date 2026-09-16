"""The prompt's command history: a file of lines and a shell-style walk over them."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from proofpath import paths
from proofpath.tui.history import History


def test_state_dir_reads_the_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    assert paths.state_dir() == tmp_path
    assert paths.history_path() == tmp_path / "history"


def test_add_appends_and_saves(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("/check a.pdf")
    history.add("/help")
    assert history.entries == ("/check a.pdf", "/help")
    assert (tmp_path / "history").read_text(encoding="utf-8") == "/check a.pdf\n/help\n"


def test_add_skips_blank_and_consecutive_duplicate(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("   ")
    history.add("/help")
    history.add("/help")
    history.add("/quit")
    history.add("/help")
    assert history.entries == ("/help", "/quit", "/help")


def test_load_reads_an_existing_file(tmp_path: Path) -> None:
    (tmp_path / "history").write_text("one\ntwo\n", encoding="utf-8")
    history = History(tmp_path / "history")
    history.load()
    assert history.entries == ("one", "two")


def test_load_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    history = History(tmp_path / "missing" / "history")
    history.load()
    assert history.entries == ()


def test_load_of_an_unreadable_file_is_empty(tmp_path: Path) -> None:
    (tmp_path / "history").write_bytes(b"/help\n\xff\xfe broken")
    history = History(tmp_path / "history")
    history.load()
    assert history.entries == ()


def test_default_path_is_the_state_dir_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    history = History()
    history.add("/help")
    assert (tmp_path / "history").read_text(encoding="utf-8") == "/help\n"


def test_add_creates_the_directory(tmp_path: Path) -> None:
    history = History(tmp_path / "deep" / "er" / "history")
    history.add("/help")
    assert (tmp_path / "deep" / "er" / "history").read_text(encoding="utf-8") == "/help\n"


def test_a_write_failure_keeps_the_session_history(tmp_path: Path) -> None:
    # A *file* where the directory should be: mkdir and the write both fail.
    (tmp_path / "state").write_text("", encoding="utf-8")
    history = History(tmp_path / "state" / "history")
    history.add("/help")
    assert history.entries == ("/help",)


def test_the_file_keeps_only_the_newest_lines(tmp_path: Path) -> None:
    history = History(tmp_path / "history", limit=3)
    for line in ("a", "b", "c", "d"):
        history.add(line)
    assert history.entries == ("b", "c", "d")
    assert (tmp_path / "history").read_text(encoding="utf-8") == "b\nc\nd\n"


def test_previous_walks_back_and_next_restores_the_draft(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("one")
    history.add("two")
    assert history.previous("draft") == "two"
    assert history.previous("draft") == "one"
    assert history.previous("draft") is None  # at the oldest: stay put
    assert history.next() == "two"
    assert history.next() == "draft"
    assert history.next() is None  # past the newest: nothing more to show


def test_previous_on_an_empty_history_is_none(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    assert history.previous("draft") is None
    assert history.next() is None


def test_reset_ends_the_walk(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("one")
    assert history.previous("draft") == "one"
    history.reset()
    assert history.next() is None
    assert history.previous("new draft") == "one"
    assert history.next() == "new draft"


def test_add_ends_the_walk(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("one")
    history.previous("")
    history.add("two")
    assert history.next() is None
    assert history.previous("") == "two"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits; chmod is a no-op there")
def test_the_file_is_private_to_the_user(tmp_path: Path) -> None:
    """Shell histories are 0600: a pasted URL can carry a token, and this is a URL log."""
    history = History(tmp_path / "history")
    history.add("/check https://example.org/paper?token=abc")
    assert (tmp_path / "history").stat().st_mode & 0o777 == 0o600
