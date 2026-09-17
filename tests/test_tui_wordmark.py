"""The wordmark in ``rich``: the six lines, the tone runs and the layout (wordmark
design sections 3 and 4). Pure: no Textual, no clock."""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace

import pytest

from proofpath.tui import banner, wordmark

CONTEXT = "academic · online · coreml"
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"
VERSION = "0.4.4"

RICH = SimpleNamespace(name="rich")
PLAIN = SimpleNamespace(name="plain")

ART = (
    "██████╗                        █████╗██████╗               ██╗",
    "██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║",
    "██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗",
    "██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗",
    "██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║",
    "╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝",
)
RICH_80 = (
    *ART,
    "proofpath v0.4.4                                    academic · online · coreml",
    "paste a file path, a URL, or a claim.                    /help  /config  /quit",
)
RICH_100 = (
    *ART,
    "proofpath v0.4.4                                                        academic · online · coreml",  # noqa: E501
    "paste a file path, a URL, or a claim.                                        /help  /config  /quit",  # noqa: E501
)


def render(width: int, theme: SimpleNamespace = RICH, **overrides: str) -> banner.Banner:
    kwargs = {"version": VERSION, "context": CONTEXT, "hint": HINT, **overrides}
    return wordmark.render(width, theme, **kwargs)


# --- the art -----------------------------------------------------------------------------


def test_wordmark_is_six_lines_at_most_66_wide_with_no_trailing_spaces() -> None:
    assert len(wordmark.WORDMARK) == 6
    assert max(len(line) for line in wordmark.WORDMARK) == 66
    for line in wordmark.WORDMARK:
        assert line == line.rstrip(), line


def test_wordmark_uses_only_the_block_the_shadow_glyphs_and_the_space() -> None:
    allowed = set(wordmark.BLOCK + wordmark.SHADOW + " ")
    for line in wordmark.WORDMARK:
        assert set(line) <= allowed, line


def test_wordmark_glyphs_are_never_wide() -> None:
    for line in wordmark.WORDMARK:
        for character in line:
            assert unicodedata.east_asian_width(character) not in {"W", "F"}, character


def test_wordmark_is_the_spec_art() -> None:
    assert wordmark.WORDMARK == ART


# --- tones -------------------------------------------------------------------------------


def test_tones_run_light_over_blocks_and_dark_over_shadow() -> None:
    assert wordmark.tones("██╔══██╗ █") == (
        (0, 2, "light"),
        (2, 5, "dark"),
        (5, 7, "light"),
        (7, 8, "dark"),
        (9, 10, "light"),
    )
    assert wordmark.tones("") == ()
    assert wordmark.tones("   ") == ()


def test_tones_cover_every_inked_cell_and_nothing_else() -> None:
    drawn = render(80)
    for row, (line, runs) in enumerate(zip(drawn.lines, drawn.tones, strict=True)):
        covered = {i for start, end, _ in runs for i in range(start, end)}
        if row in (wordmark.VERSION_ROW, wordmark.HINT_ROW):
            assert runs == (), row
            continue
        inked = {i for i, c in enumerate(line) if c != " "}
        assert covered == inked, row
        for start, end, tone in runs:
            glyphs = set(line[start:end])
            assert glyphs == {wordmark.BLOCK} if tone == "light" else glyphs <= set(wordmark.SHADOW)


# --- layout ------------------------------------------------------------------------------


def test_rich_80_is_the_golden_block() -> None:
    assert render(80).lines == RICH_80


def test_rich_100_is_the_golden_block() -> None:
    assert render(100).lines == RICH_100


@pytest.mark.parametrize("width", range(69, 201))
def test_rich_is_eight_lines_and_the_wordmark_never_moves(width: int) -> None:
    drawn = render(width)
    assert len(drawn.lines) == 8
    assert drawn.lines[:6] == wordmark.WORDMARK
    assert drawn.lines[wordmark.HINT_ROW].endswith("/help  /config  /quit")


def test_rich_text_starts_at_column_zero_and_ends_at_the_margin() -> None:
    drawn = render(80)
    assert drawn.lines[wordmark.VERSION_ROW].startswith("proofpath v0.4.4")
    assert drawn.lines[wordmark.HINT_ROW].startswith("paste a file path")
    assert len(drawn.lines[wordmark.VERSION_ROW]) == 80 - banner.RIGHT_MARGIN
    assert len(drawn.lines[wordmark.HINT_ROW]) == 80 - banner.RIGHT_MARGIN
    assert wordmark.TEXT_COLUMN == 0


def test_rich_hint_without_a_gap_is_written_whole() -> None:
    drawn = render(80, hint="paste a file path")
    assert drawn.lines[wordmark.HINT_ROW] == "paste a file path"


def test_rich_long_version_is_never_truncated() -> None:
    drawn = render(69, version="0.4.4-rc1+build.12345.deadbeefcafe")
    line = drawn.lines[wordmark.VERSION_ROW]
    assert line.startswith("proofpath v0.4.4-rc1+build.12345.deadbeefcafe")
    assert line.endswith(" " + CONTEXT)


def test_rich_68_returns_the_plain_banner() -> None:
    assert render(68) == banner.render(68, version=VERSION, context=CONTEXT, hint=HINT)


def test_rich_69_is_still_the_wordmark() -> None:
    assert render(69).lines[:6] == wordmark.WORDMARK
    assert wordmark.RICH_WIDTH_FLOOR == 69


@pytest.mark.parametrize("width", [40, 68, 80, 100, 200])
def test_plain_delegates_to_banner_at_every_width(width: int) -> None:
    assert render(width, PLAIN) == banner.render(width, version=VERSION, context=CONTEXT, hint=HINT)


def test_real_theme_objects_are_accepted() -> None:
    from proofpath.tui.theme import PLAIN as PLAIN_THEME
    from proofpath.tui.theme import RICH as RICH_THEME

    assert render(80, RICH_THEME).lines == RICH_80
    assert render(80, PLAIN_THEME) == banner.render(80, version=VERSION, context=CONTEXT, hint=HINT)


# --- hint_row ------------------------------------------------------------------------------


def test_hint_row_matches_the_floor_render_gives_way_at() -> None:
    assert wordmark.hint_row(68) == banner.HINT_LINE
    assert wordmark.hint_row(69) == wordmark.HINT_ROW


@pytest.mark.parametrize("width", [40, 68, 69, 80, 100, 200])
def test_hint_row_points_at_the_row_render_actually_put_the_hint_on(width: int) -> None:
    assert render(width).lines[wordmark.hint_row(width)].endswith("/help  /config  /quit")


def test_module_is_pure_and_imports_no_textual() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path(wordmark.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("textual") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("textual")
    source = Path(wordmark.__file__).read_text(encoding="utf-8")
    assert "#" + "8f0f2b" not in source and "#" + "c4173a" not in source  # names no colour
