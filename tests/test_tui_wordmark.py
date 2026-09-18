"""The wordmark in both themes: the six lines, the gradient runs, the rule and the
beside/under/narrow layouts (wordmark design sections 3, 4, 8 and 8.1). Pure: no
Textual, no clock."""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace

import pytest

from proofpath.tui import banner, wordmark

CONTEXT = "academic · online · coreml"
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"
VERSION = "0.4.5"

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


def render(width: int, theme: SimpleNamespace = RICH, **overrides: str) -> banner.Banner:
    kwargs = {"version": VERSION, "context": CONTEXT, "hint": HINT, **overrides}
    return wordmark.render(width, theme, **kwargs)


UNDER_80 = (
    *ART,
    "proofpath v0.4.5                                    academic · online · coreml",
    "paste a file path, a URL, or a claim.                    /help  /config  /quit",
    "─" * 78,
)


def beside_row(art: str, left: str, right: str, width: int) -> str:
    return banner.right_align(art.ljust(wordmark.BESIDE_COLUMN) + left, right, width)


BESIDE_140 = (
    ART[0],
    ART[1],
    beside_row(ART[2], "proofpath v0.4.5", CONTEXT, 140),
    beside_row(ART[3], "paste a file path, a URL, or a claim.", "/help  /config  /quit", 140),
    ART[4],
    ART[5],
    "─" * 138,
)
NARROW_60 = (
    banner.right_align("proofpath v0.4.5", CONTEXT, 60),
    banner.right_align("paste a file path, a URL, or a claim.", "/help  /config  /quit", 60),
    "─" * 58,
)


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


# --- layout ------------------------------------------------------------------------------


def test_rich_80_is_the_under_layout_with_the_rule() -> None:
    assert render(80).lines == UNDER_80


def test_rich_140_is_the_beside_layout_with_the_rule() -> None:
    assert render(140).lines == BESIDE_140


def test_below_the_floor_there_is_no_mark_in_either_theme() -> None:
    assert render(60).lines == NARROW_60
    assert render(60, PLAIN).lines == tuple(
        line.translate(wordmark.PLAIN_GLYPHS) for line in NARROW_60
    )
    assert render(68).lines[0].startswith("proofpath v")
    assert render(69).lines[:6] == ART


def test_plain_is_the_same_mark_transliterated_to_ascii() -> None:
    rich, plain = render(80), render(80, PLAIN)
    assert plain.lines == tuple(line.translate(wordmark.PLAIN_GLYPHS) for line in rich.lines)
    assert plain.tones == rich.tones
    # The mark and the rule are seven-bit; the texts are the caller's. The app's
    # ``run_context`` is ASCII whatever the theme, so the whole plain banner is.
    for line in render(80, PLAIN, context="academic, online, coreml").lines:
        assert line.isascii(), line
    assert render(140, PLAIN).lines == tuple(
        line.translate(wordmark.PLAIN_GLYPHS) for line in BESIDE_140
    )


def test_plain_glyph_table_is_the_spec_mapping() -> None:
    assert "█╗╔╝╚═║─".translate(wordmark.PLAIN_GLYPHS) == "#++++-|-"


def test_rows_switch_to_beside_exactly_where_the_texts_fit() -> None:
    assert wordmark.rows(128, version=VERSION, context=CONTEXT, hint=HINT) == wordmark.Rows(
        6, 7, 8, "under"
    )
    assert wordmark.rows(129, version=VERSION, context=CONTEXT, hint=HINT) == wordmark.Rows(
        2, 3, 6, "beside"
    )
    # With the command list dropped the version row (43 columns) is the longer one,
    # so the switch is at 68 + 43 + 2.
    prompt_only = HINT.split("  ")[0]
    assert wordmark.rows(112, version=VERSION, context=CONTEXT, hint=prompt_only).beside is False
    assert wordmark.rows(113, version=VERSION, context=CONTEXT, hint=prompt_only).beside is True
    assert wordmark.rows(68, version=VERSION, context=CONTEXT, hint=HINT) == wordmark.Rows(
        0, 1, 2, "narrow"
    )


@pytest.mark.parametrize("width", [40, 68, 69, 80, 128, 129, 140, 200])
@pytest.mark.parametrize("theme", [RICH, PLAIN])
def test_the_rule_is_the_last_row_and_reaches_the_margin(
    width: int, theme: SimpleNamespace
) -> None:
    drawn = render(width, theme)
    where = wordmark.rows(width, version=VERSION, context=CONTEXT, hint=HINT)
    rule = wordmark.RULE if theme.name == "rich" else wordmark.RULE.translate(wordmark.PLAIN_GLYPHS)
    assert where.rule == len(drawn.lines) - 1
    assert drawn.lines[where.rule] == rule * (width - banner.RIGHT_MARGIN)
    assert drawn.tones[where.rule] == ((0, width - banner.RIGHT_MARGIN, wordmark.SHADOW_TONE),)
    assert drawn.lines[where.hint].endswith("/help  /config  /quit")
    assert drawn.lines[where.version].endswith(CONTEXT)
    assert len(drawn.lines) == (
        3 if width < wordmark.RICH_WIDTH_FLOOR else 7 if where.beside else 9
    )


def test_text_rows_end_at_the_margin_and_are_never_truncated() -> None:
    for width in (69, 80, 140):
        where = wordmark.rows(width, version=VERSION, context=CONTEXT, hint=HINT)
        drawn = render(width)
        assert len(drawn.lines[where.version]) == width - banner.RIGHT_MARGIN
        assert len(drawn.lines[where.hint]) == width - banner.RIGHT_MARGIN
    long = render(69, version="0.4.5-rc1+build.12345.deadbeefcafe")
    assert long.lines[6].startswith("proofpath v0.4.5-rc1+build.12345.deadbeefcafe")
    assert long.lines[6].endswith(" " + CONTEXT)


def test_a_hint_without_a_gap_is_written_whole() -> None:
    assert render(80, hint="paste a file path").lines[7] == "paste a file path"
    assert (
        render(140, hint="paste a file path").lines[3]
        == ART[3].ljust(wordmark.BESIDE_COLUMN) + "paste a file path"
    )


# --- tones -------------------------------------------------------------------------------


def test_tones_follow_the_column_band_for_blocks_and_shadow_for_the_rest() -> None:
    for row, line in enumerate(wordmark.WORDMARK):
        painted: dict[int, str] = {}
        for start, end, tone in wordmark.tones(line):
            painted.update(dict.fromkeys(range(start, end), tone))
        for column, character in enumerate(line):
            if character == wordmark.BLOCK:
                expected = wordmark.GRADIENT_TONES[min(4, column * 5 // wordmark.MARK_WIDTH)]
                assert painted[column] == expected, (row, column)
            elif character in wordmark.SHADOW:
                assert painted[column] == wordmark.SHADOW_TONE, (row, column)
            else:
                assert column not in painted, (row, column)


def test_band_splits_any_width_into_five_left_light_right_dark() -> None:
    """The same arithmetic serves the mark (over its 66 columns) and the bar's frame
    (over the terminal's): the left end is always ``g0``, the right always ``g4``."""
    assert wordmark.band(0, 10) == "g0" and wordmark.band(9, 10) == "g4"
    assert [wordmark.band(c, 5) for c in range(5)] == list(wordmark.GRADIENT_TONES)
    assert wordmark.band(65, 66) == "g4" and wordmark.band(0, 66) == "g0"
    assert wordmark.band(0, 1) == "g0"  # a one-column frame is not a division by zero
    assert wordmark.band(0, 0) == "g0"  # nor is a frame that has no width yet


def test_a_block_run_splits_where_the_band_changes() -> None:
    assert wordmark.tones(wordmark.BLOCK * 66) == (
        (0, 14, "g0"),
        (14, 27, "g1"),
        (27, 40, "g2"),
        (40, 53, "g3"),
        (53, 66, "g4"),
    )
    assert wordmark.tones("╔" * 30) == ((0, 30, "shadow"),)
    assert wordmark.tones("") == ()


def test_the_text_beside_or_under_the_mark_carries_no_run() -> None:
    for width in (80, 140):
        drawn = render(width)
        where = wordmark.rows(width, version=VERSION, context=CONTEXT, hint=HINT)
        for row in (where.version, where.hint):
            assert all(end <= wordmark.MARK_WIDTH for _, end, _ in drawn.tones[row]), (width, row)


def test_real_theme_objects_are_accepted() -> None:
    from proofpath.tui.theme import PLAIN as PLAIN_THEME
    from proofpath.tui.theme import RICH as RICH_THEME

    assert render(80, RICH_THEME).lines == UNDER_80
    assert render(80, PLAIN_THEME).lines == tuple(
        line.translate(wordmark.PLAIN_GLYPHS) for line in UNDER_80
    )


# --- purity ------------------------------------------------------------------------------


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
