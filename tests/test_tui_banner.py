"""Exact-string tests for the ``plain`` raven (raven design section 5).

The six ``SPEC_BLOCK`` lines are copied verbatim out of the design's own 80-column
block; they are the golden output the renderer has to reproduce.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from proofpath.tui import banner

# 2026-09-16-raven-pet-design.md section 5, and the amended block of design.md 13.1.
SPEC_BLOCK = (
    "       __",
    "      (o >",
    "    _/ /",
    "   /  /       proofpath v0.1.0                        academic . offline . mps",
    "  /__/        paste a file path, a URL, or a claim.      /help  /config  /quit",
    " ____||_______________________________________________________________________",
)
CONTEXT = "academic . offline . mps"
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"


def render(width: int, **overrides: str) -> banner.Banner:
    kwargs = {"version": "0.1.0", "context": CONTEXT, "hint": HINT, **overrides}
    return banner.render(width, **kwargs)


def test_width_80_reproduces_the_spec_block() -> None:
    assert render(80).lines == SPEC_BLOCK


@pytest.mark.parametrize("width", [30, 39, 40, 60, 80, 200])
def test_every_line_is_pure_ascii(width: int) -> None:
    for line in render(width).lines:
        assert all(ord(character) < 128 for character in line), line


def test_the_art_comes_from_the_art_constant() -> None:
    drawn = render(80)
    assert drawn.lines[:3] == banner.ART[:3]
    assert drawn.lines[3].startswith(banner.ART[3])
    assert drawn.lines[4].startswith(banner.ART[4])
    assert drawn.lines[5].startswith(banner.ART[5])


def test_text_lines_start_at_the_indent() -> None:
    drawn = render(80)
    assert drawn.lines[3][len(banner.TEXT_INDENT) :].startswith("proofpath v0.1.0")
    assert drawn.lines[4][len(banner.TEXT_INDENT) :].startswith("paste a file path")


# 75 is the narrowest width at which the hint line (14 + 37 + 21 columns) still fits.
@pytest.mark.parametrize("width", [75, 80, 100, 200])
def test_context_and_commands_end_two_columns_short_of_the_edge(width: int) -> None:
    drawn = render(width)
    assert len(drawn.lines[3]) == width - banner.RIGHT_MARGIN
    assert drawn.lines[3].endswith(CONTEXT)
    assert len(drawn.lines[4]) == width - banner.RIGHT_MARGIN
    assert drawn.lines[4].endswith("/help  /config  /quit")


def test_commands_that_do_not_fit_overflow_by_one_space() -> None:
    drawn = render(60)
    assert len(drawn.lines[3]) == 60 - banner.RIGHT_MARGIN
    assert drawn.lines[4].endswith(" /help  /config  /quit")
    assert len(drawn.lines[4]) == len(banner.TEXT_INDENT) + 37 + 1 + 21


@pytest.mark.parametrize("width", [40, 80, 200])
def test_ground_runs_to_two_columns_short_of_the_edge(width: int) -> None:
    line = render(width).lines[5]
    assert len(line) == width - banner.RIGHT_MARGIN
    assert line.startswith(banner.ART[5])
    assert set(line[len(banner.ART[5]) :]) == {banner.GROUND}


def test_width_39_does_not_extend_the_ground() -> None:
    assert render(39).lines[5] == banner.ART[5]


def test_hint_without_a_gap_is_written_whole() -> None:
    drawn = render(80, hint="type something")
    assert drawn.lines[4] == banner.ART[4].ljust(len(banner.TEXT_INDENT)) + "type something"


def test_long_version_is_never_truncated_and_context_overflows_by_one_space() -> None:
    drawn = render(40, version="0.1.0-rc1+build.12345")
    assert "proofpath v0.1.0-rc1+build.12345" in drawn.lines[3]
    assert drawn.lines[3].endswith(" " + CONTEXT)


def test_tones_cover_the_art_of_every_line_and_the_ground() -> None:
    drawn = render(80)
    assert len(drawn.tones) == len(drawn.lines)
    for line, runs in zip(drawn.lines, drawn.tones, strict=True):
        assert len(runs) == 1
        start, _end, tone = runs[0]
        assert tone == "light"
        assert start == len(line) - len(line.lstrip(" "))
    # The runs stop where the text begins, and the ground run is the whole ground.
    assert drawn.tones[3] == ((3, len(banner.ART[3]), "light"),)
    assert drawn.tones[4] == ((2, len(banner.ART[4]), "light"),)
    assert drawn.tones[5] == ((1, len(drawn.lines[5]), "light"),)


def test_banner_is_frozen() -> None:
    drawn = render(80)
    with pytest.raises(AttributeError):
        drawn.lines = ()  # type: ignore[misc]


def test_split_hint() -> None:
    assert banner.split_hint(HINT) == (
        "paste a file path, a URL, or a claim.",
        "/help  /config  /quit",
    )
    assert banner.split_hint("no gap here") == ("no gap here", None)


def test_module_imports_no_textual() -> None:
    tree = ast.parse(Path(banner.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("textual") for name in imported), imported
