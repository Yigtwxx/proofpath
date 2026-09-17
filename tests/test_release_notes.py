"""Offline tests for scripts/release_notes.py: the CHANGELOG section a GitHub Release
takes as its notes, and the exit codes the release workflow gates on."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "release_notes.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_notes = _load_script()
section = release_notes.section
main = release_notes.main

CHANGELOG = """# Changelog

## [Unreleased]

### Added
- something coming

## [0.4.2] - 2026-09-17

### Changed
- **The raven.** A two-tone pixel raven replaces the ferret.

### Fixed
- a bug

## [0.4.1] - 2026-09-15

### Fixed
- the last section, at the end of the file
"""


# --- section() ------------------------------------------------------------


def test_section_is_the_body_under_the_heading_up_to_the_next_one() -> None:
    body = section("0.4.2", CHANGELOG)
    assert body == (
        "### Changed\n- **The raven.** A two-tone pixel raven replaces the ferret.\n\n"
        "### Fixed\n- a bug\n"
    )
    assert "## [" not in body


def test_section_of_the_last_version_runs_to_the_end_of_the_file() -> None:
    assert section("0.4.1", CHANGELOG) == (
        "### Fixed\n- the last section, at the end of the file\n"
    )


def test_section_is_none_for_a_version_without_a_heading() -> None:
    assert section("0.4.0", CHANGELOG) is None
    # A prefix of a real version is not that version.
    assert section("0.4", CHANGELOG) is None


# --- main() ---------------------------------------------------------------


def _use_changelog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(release_notes, "CHANGELOG", path)


def test_main_prints_the_section_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_changelog(tmp_path, monkeypatch, CHANGELOG)
    assert main(["release_notes.py", "0.4.2"]) == 0
    out, err = capsys.readouterr()
    assert out == section("0.4.2", CHANGELOG)
    assert err == ""


def test_main_takes_the_tag_form_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_changelog(tmp_path, monkeypatch, CHANGELOG)
    assert main(["release_notes.py", "v0.4.1"]) == 0
    assert capsys.readouterr().out == section("0.4.1", CHANGELOG)


def test_main_exits_1_when_the_version_has_no_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_changelog(tmp_path, monkeypatch, CHANGELOG)
    assert main(["release_notes.py", "9.9.9"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "no section for 9.9.9" in err


def test_main_exits_1_when_the_heading_has_an_empty_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A heading with nothing under it is as missing as no heading: a release never
    goes out with empty notes."""
    _use_changelog(tmp_path, monkeypatch, "# Changelog\n\n## [0.5.0] - 2026-10-01\n\n\n")
    assert main(["release_notes.py", "0.5.0"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "no section for 0.5.0" in err


def test_main_exits_2_without_exactly_one_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["release_notes.py"]) == 2
    assert "usage" in capsys.readouterr().err
    assert main(["release_notes.py", "0.4.2", "extra"]) == 2
