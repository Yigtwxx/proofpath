"""The raven in ``rich``: the bitmap, the Braille encoder and the layout (raven
design sections 3 and 4). Pure: no Textual, no clock."""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace

import pytest

from proofpath.tui import banner, pet

CONTEXT = "academic · online · coreml"
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"
VERSION = "0.3.0"

RICH = SimpleNamespace(name="rich")
PLAIN = SimpleNamespace(name="plain")

RICH_80 = (
    "       ⣴⣿⠿⣷⣀",
    "      ⢀⣿⣿⣿⡿⠿⣿⠂",
    "     ⢠⣿⣿⣿⣿⡇",
    "    ⢠⣿⣿⣿⣿⣿⡇       proofpath v0.3.0                  academic · online · coreml",
    "   ⢠⣿⣿⣿⣿⣿⡟        paste a file path, a URL, or a claim.  /help  /config  /quit",
    "  ⣠⣿⣿⣿⣿⡿⠋",
    "⣠⣾⣿⣿⠿⣿⠉⡏⡇",
    "⠛⠙⠃  ⠛⠒⠓⠓⠒⠒⠒⠒⠒⠂" + "⠒" * 63,
)
RICH_100 = (
    "       ⣴⣿⠿⣷⣀",
    "      ⢀⣿⣿⣿⡿⠿⣿⠂",
    "     ⢠⣿⣿⣿⣿⡇",
    "    ⢠⣿⣿⣿⣿⣿⡇       proofpath v0.3.0                                      academic · online · coreml",  # noqa: E501
    "   ⢠⣿⣿⣿⣿⣿⡟        paste a file path, a URL, or a claim.                      /help  /config  /quit",  # noqa: E501
    "  ⣠⣿⣿⣿⣿⡿⠋",
    "⣠⣾⣿⣿⠿⣿⠉⡏⡇",
    "⠛⠙⠃  ⠛⠒⠓⠓⠒⠒⠒⠒⠒⠂" + "⠒" * 83,
)
TONES_80 = (
    ((7, 12, "dark"),),
    ((6, 7, "light"), (7, 11, "dark"), (11, 14, "light")),
    ((5, 9, "light"), (9, 11, "dark")),
    ((4, 9, "light"), (9, 11, "dark")),
    ((3, 8, "light"), (8, 10, "dark")),
    ((2, 6, "light"), (6, 9, "dark")),
    ((0, 9, "dark"),),
    ((0, 3, "dark"), (5, 6, "light"), (6, 9, "dark"), (9, 15, "light"), (15, 78, "light")),
)


def render(width: int, theme: SimpleNamespace = RICH, **overrides: str) -> banner.Banner:
    kwargs = {"version": VERSION, "context": CONTEXT, "hint": HINT, **overrides}
    return pet.render(width, theme, **kwargs)


# --- the bitmap ------------------------------------------------------------------------


def test_bitmap_is_29_by_30_and_uses_three_symbols() -> None:
    assert len(pet.BITMAP) == 30
    assert all(len(row) == 29 for row in pet.BITMAP)
    assert set("".join(pet.BITMAP)) == {pet.DARK, pet.LIGHT, pet.EMPTY}


def test_the_eye_is_a_hole_at_row_3_columns_18_and_19() -> None:
    assert pet.BITMAP[3][18:20] == ".."
    assert pet.BITMAP[3][14:18] == "####" and pet.BITMAP[3][20:23] == "###"


# --- the encoder -----------------------------------------------------------------------


def test_encoder_maps_the_eight_dots_to_the_standard_bits() -> None:
    lines, tones = pet.encode(["#.", ".#", "#.", ".#"])
    # dots 1,5,3,8 -> bits 0x01 | 0x10 | 0x04 | 0x80
    assert lines == (chr(0x2800 + 0x95),)
    assert tones == (((0, 1, "dark"),),)
    lines, tones = pet.encode([".#", "#.", ".#", "#."])
    # dots 4,2,6,7 -> bits 0x08 | 0x02 | 0x20 | 0x40
    assert lines == (chr(0x2800 + 0x6A),)
    assert tones == (((0, 1, "dark"),),)


def test_encoder_gives_a_cell_the_majority_tone_dark_on_a_tie() -> None:
    _lines, tones = pet.encode(["#+", "++", "..", ".."])
    assert tones == (((0, 1, "light"),),)
    _lines, tones = pet.encode(["#+", "+#", "..", ".."])
    assert tones == (((0, 1, "dark"),),)


def test_encoder_pads_odd_widths_and_heights_and_strips_trailing_blanks() -> None:
    lines, tones = pet.encode(["#..", "...", "..."])
    assert lines == ("⠁",)
    assert tones == (((0, 1, "dark"),),)


def test_encoder_merges_neighbouring_cells_of_one_tone_into_one_run() -> None:
    lines, tones = pet.encode(["####..++", "....", "....", "...."])
    assert lines == ("⠉⠉ ⠉",)
    assert tones == (((0, 2, "dark"), (3, 4, "light")),)


# --- the RICH banner -------------------------------------------------------------------


def test_rich_80_is_the_golden_block() -> None:
    assert render(80).lines == RICH_80


def test_rich_100_is_the_golden_block() -> None:
    assert render(100).lines == RICH_100


def test_rich_80_tones_are_the_golden_runs() -> None:
    assert render(80).tones == TONES_80


@pytest.mark.parametrize("width", range(48, 201))
def test_rich_is_eight_lines_and_the_bird_never_moves(width: int) -> None:
    drawn = render(width)
    assert len(drawn.lines) == 8
    for row, line in enumerate(drawn.lines):
        if row not in (pet.VERSION_ROW, pet.HINT_ROW):
            assert line.startswith(RICH_80[row][:10]), (row, line)
    assert len(drawn.lines[pet.GROUND_ROW]) == width - banner.RIGHT_MARGIN


@pytest.mark.parametrize("width", [48, 60, 80, 100, 200])
def test_rich_art_is_braille_or_space_and_single_width(width: int) -> None:
    drawn = render(width)
    for row, line in enumerate(drawn.lines):
        art = line if row not in (pet.VERSION_ROW, pet.HINT_ROW) else line[: pet.TEXT_COLUMN]
        for character in art:
            assert character == " " or 0x2800 <= ord(character) <= 0x28FF, (row, character)
            assert unicodedata.east_asian_width(character) not in {"W", "F"}


def test_rich_text_starts_at_the_text_column_and_ends_at_the_margin() -> None:
    drawn = render(80)
    assert drawn.lines[pet.VERSION_ROW][pet.TEXT_COLUMN :].startswith("proofpath v0.3.0")
    assert drawn.lines[pet.HINT_ROW][pet.TEXT_COLUMN :].startswith("paste a file path")
    assert len(drawn.lines[pet.VERSION_ROW]) == 80 - banner.RIGHT_MARGIN
    assert len(drawn.lines[pet.HINT_ROW]) == 80 - banner.RIGHT_MARGIN


def test_rich_hint_without_a_gap_is_written_whole() -> None:
    drawn = render(80, hint="type something")
    assert drawn.lines[pet.HINT_ROW] == RICH_80[pet.HINT_ROW][: pet.TEXT_COLUMN] + "type something"


def test_rich_long_version_is_never_truncated() -> None:
    drawn = render(60, version="0.3.0-rc1+build.12345")
    assert "proofpath v0.3.0-rc1+build.12345" in drawn.lines[pet.VERSION_ROW]
    assert drawn.lines[pet.VERSION_ROW].endswith(" " + CONTEXT)


def test_rich_tones_cover_every_inked_cell_and_nothing_else() -> None:
    drawn = render(80)
    for row, (line, runs) in enumerate(zip(drawn.lines, drawn.tones, strict=True)):
        art_end = len(line) if row not in (pet.VERSION_ROW, pet.HINT_ROW) else pet.TEXT_COLUMN
        covered = {column for start, end, _ in runs for column in range(start, end)}
        inked = {column for column, char in enumerate(line[:art_end]) if char != " "}
        assert covered == inked, row
        assert all(end > start and tone in ("dark", "light") for start, end, tone in runs)


def test_rich_ground_run_is_light_and_reaches_the_margin() -> None:
    drawn = render(120)
    _start, end, tone = drawn.tones[pet.GROUND_ROW][-1]
    assert tone == "light" and end == 120 - banner.RIGHT_MARGIN
    assert set(drawn.lines[pet.GROUND_ROW][15:]) == {pet.GROUND}


# --- the floor and PLAIN -----------------------------------------------------------------


def test_the_hint_row_is_the_same_line_in_both_arts() -> None:
    """The widget's drop rule reads the drawn hint back by ``pet.HINT_ROW`` even when
    the plain art was drawn below the floor, so it relies on the two agreeing."""
    assert pet.HINT_ROW == banner.HINT_LINE


def test_rich_47_returns_the_plain_pet() -> None:
    assert render(47) == banner.render(47, version=VERSION, context=CONTEXT, hint=HINT)


def test_rich_48_is_still_the_rich_pet() -> None:
    assert render(48).lines[0] == RICH_80[0]


@pytest.mark.parametrize("width", [39, 47, 60, 80, 200])
def test_plain_delegates_to_banner_at_every_width(width: int) -> None:
    assert render(width, PLAIN) == banner.render(width, version=VERSION, context=CONTEXT, hint=HINT)


def test_real_theme_objects_are_accepted() -> None:
    from proofpath.tui import theme

    assert render(80, theme.RICH).lines == RICH_80
    assert (
        render(80, theme.PLAIN).lines
        == banner.render(80, version=VERSION, context=CONTEXT, hint=HINT).lines
    )


def test_module_is_pure_and_imports_no_textual() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path(pet.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("textual") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("textual")
    for name in ("STAMP", "EYES", "AnimState", "next_eyes", "next_tail", "TAIL_TIPS"):
        assert not hasattr(pet, name), name
