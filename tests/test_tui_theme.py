"""The two TUI themes and their detection (TUI v2 design, section 2).

Every detection trigger is exercised in both directions with an injected
environment, so no test depends on the terminal it runs in. ``PLAIN`` is what CI
and ``run_test`` get: nothing here may import Textual.
"""

from __future__ import annotations

import colorsys
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from proofpath import ui
from proofpath.tui import theme
from proofpath.tui.theme import PLAIN, RICH, Glyphs
from proofpath.tui.theme import detect as _detect


def detect(env, **kwargs):
    """``theme.detect`` pinned to a POSIX platform: the Windows conhost rule has its own
    tests, and every other expectation must hold on the Windows CI runner too."""
    kwargs.setdefault("platform", "linux")
    return _detect(env, **kwargs)


TRUECOLOR = {"COLORTERM": "truecolor"}


def test_module_imports_no_textual():
    source = Path(theme.__file__).read_text(encoding="utf-8")
    assert "textual" not in source


# --- detection ---------------------------------------------------------------------


def test_no_evidence_is_plain():
    assert detect({}) is PLAIN


@pytest.mark.parametrize("value", ["truecolor", "24bit"])
def test_colorterm_is_rich(value):
    assert detect({"COLORTERM": value}) is RICH


def test_unknown_colorterm_is_plain():
    assert detect({"COLORTERM": "yes"}) is PLAIN


@pytest.mark.parametrize("program", ["iTerm.app", "WezTerm", "vscode", "ghostty", "Apple_Terminal"])
def test_known_term_program_is_rich(program):
    assert detect({"TERM_PROGRAM": program}) is RICH


def test_unknown_term_program_is_plain():
    assert detect({"TERM_PROGRAM": "Hyper"}) is PLAIN


@pytest.mark.parametrize("marker", ["KITTY_WINDOW_ID", "ALACRITTY_LOG", "WT_SESSION"])
def test_terminal_marker_is_rich(marker):
    assert detect({marker: "1"}) is RICH


def test_no_color_env_wins_over_truecolor():
    assert detect({**TRUECOLOR, "NO_COLOR": "1"}) is PLAIN


def test_no_color_env_is_honoured_even_when_empty():
    # The NO_COLOR convention: presence, not value.
    assert detect({**TRUECOLOR, "NO_COLOR": ""}) is PLAIN


def test_no_color_flag_wins_over_truecolor():
    assert detect(TRUECOLOR, no_color=True) is PLAIN


def test_quiet_wins_over_truecolor():
    assert detect(TRUECOLOR, quiet=True) is PLAIN


def test_term_dumb_wins_over_truecolor():
    assert detect({**TRUECOLOR, "TERM": "dumb"}) is PLAIN


def test_term_other_than_dumb_does_not_block():
    assert detect({**TRUECOLOR, "TERM": "xterm-256color"}) is RICH


def test_windows_conhost_is_plain_even_with_truecolor():
    assert detect(TRUECOLOR, platform="win32") is PLAIN


def test_windows_terminal_is_rich():
    assert detect({"WT_SESSION": "abc"}, platform="win32") is RICH


def test_windows_terminal_still_honours_no_color():
    assert detect({"WT_SESSION": "abc", "NO_COLOR": "1"}, platform="win32") is PLAIN


def test_darwin_without_evidence_is_plain():
    assert detect({}, platform="darwin") is PLAIN


# T4: CJK ambiguous-width terminals draw box drawing two cells wide.


@pytest.mark.parametrize("locale", ["zh_CN.UTF-8", "ja_JP.UTF-8", "ko_KR.UTF-8", "JA_JP"])
def test_cjk_lang_is_plain_even_with_truecolor(locale):
    assert detect({**TRUECOLOR, "LANG": locale}) is PLAIN


def test_lc_all_overrides_lang_for_the_cjk_trigger():
    assert detect({**TRUECOLOR, "LANG": "en_US.UTF-8", "LC_ALL": "zh_TW.UTF-8"}) is PLAIN
    assert detect({**TRUECOLOR, "LANG": "ja_JP.UTF-8", "LC_ALL": "en_US.UTF-8"}) is RICH


def test_lc_ctype_sits_between_lc_all_and_lang():
    assert detect({**TRUECOLOR, "LC_CTYPE": "ko_KR.UTF-8"}) is PLAIN
    assert detect({**TRUECOLOR, "LANG": "ja_JP.UTF-8", "LC_CTYPE": "en_US.UTF-8"}) is RICH
    assert detect({**TRUECOLOR, "LC_CTYPE": "ja_JP.UTF-8", "LC_ALL": "C.UTF-8"}) is RICH


@pytest.mark.parametrize("locale", ["en_US.UTF-8", "C.UTF-8", "tr_TR.UTF-8", ""])
def test_other_locales_do_not_block_rich(locale):
    assert detect({**TRUECOLOR, "LANG": locale}) is RICH


def test_the_override_beats_the_cjk_trigger():
    assert detect({**TRUECOLOR, "LANG": "ja_JP.UTF-8", "PROOFPATH_THEME": "rich"}) is RICH


@pytest.mark.parametrize("value", ["rich", "RICH", " Rich "])
def test_override_rich_beats_every_plain_trigger(value):
    env = {"PROOFPATH_THEME": value, "NO_COLOR": "1", "TERM": "dumb"}
    assert detect(env, no_color=True, quiet=True, platform="win32") is RICH


@pytest.mark.parametrize("value", ["plain", "PLAIN"])
def test_override_plain_beats_truecolor(value):
    assert detect({**TRUECOLOR, "PROOFPATH_THEME": value}) is PLAIN


@pytest.mark.parametrize("value", ["", "fancy", "0"])
def test_unknown_override_is_ignored(value):
    assert detect({**TRUECOLOR, "PROOFPATH_THEME": value}) is RICH
    assert detect({"PROOFPATH_THEME": value}) is PLAIN


def test_detect_defaults_to_the_running_platform():
    # Only the platform default is under test; the environment is injected.
    expected = PLAIN if sys.platform.startswith("win") else RICH
    assert _detect(TRUECOLOR) is expected


# --- style_for -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "name"),
    [
        (ui.GREEN[0], "green"),
        (ui.YELLOW_PREFIXES[0], "yellow"),
        ("UNVERIFIED (paywalled)", "yellow"),
        (ui.RED[0], "red"),
        (ui.DIM[0], "dim"),
    ],
)
def test_style_for_maps_each_table_to_its_tone(word, name):
    assert PLAIN.style_for(word) == name == PLAIN.tones[name]
    assert RICH.style_for(word) == RICH.tones[name]


def test_tone_is_the_meanings_style_in_both_themes():
    assert RICH.tone("ok") == RICH.tones["green"]
    assert RICH.tone("muted") == RICH.tones["dim"]
    assert PLAIN.tone("finding") == "red"
    assert PLAIN.tone("caution") == "yellow"


def test_rich_badge_is_ink_on_the_bare_colour():
    assert RICH.badge_style(ui.RED[0]) == f"white on {RICH.colours['red']}"
    assert RICH.badge_style(ui.GREEN[0]) == f"black on {RICH.colours['green']}"
    assert RICH.badge_style("UNVERIFIED (paywalled)") == f"black on {RICH.colours['yellow']}"
    assert RICH.badge_style(ui.DIM[0]) == f"white on {RICH.colours['dim']}"
    assert RICH.badge_style("nonsense") == ""


def test_plain_draws_no_badge():
    assert PLAIN.badge_style(ui.RED[0]) == ""
    assert PLAIN.ink == {}


def test_rich_colours_are_the_tones_without_their_weight():
    for name, colour in RICH.colours.items():
        assert colour.startswith("#")
        assert RICH.tones[name].endswith(colour)


def test_style_for_unknown_word_is_empty():
    assert PLAIN.style_for("banana") == ""
    assert RICH.style_for("banana") == ""
    assert RICH.style_for("") == ""


def test_style_for_agrees_with_ui_style_state():
    out = ui.build(force_terminal=True)
    for word in (*ui.GREEN, *ui.YELLOW_PREFIXES, *ui.RED, *ui.DIM, "nope"):
        styled = ui.style_state(out, word)
        expected = str(styled.spans[0].style) if styled.spans else ""
        assert PLAIN.style_for(word) == expected


def test_plain_tones_are_the_ansi_names():
    assert PLAIN.tones == {"green": "green", "yellow": "yellow", "red": "red", "dim": "dim"}


def test_rich_tones_are_truecolor():
    assert set(RICH.tones) == set(PLAIN.tones)
    for name, style in RICH.tones.items():
        assert "#" in style, (name, style)


# --- glyphs ---------------------------------------------------------------------------


def test_every_plain_glyph_is_ascii():
    for field in fields(Glyphs):
        value = getattr(PLAIN.glyphs, field.name)
        assert value.isascii(), (field.name, value)


def test_plain_has_no_box_and_rich_has_a_round_one():
    assert PLAIN.glyphs.box == "none"
    assert RICH.glyphs.box == "round"
    assert PLAIN.unicode is False
    assert RICH.unicode is True


def test_rich_glyphs_are_the_design_set():
    g = RICH.glyphs
    assert (g.stage_pending, g.stage_active, g.stage_done, g.stage_flag) == ("·", "⏺", "✓", "⏺")
    assert (g.error, g.warning, g.note) == ("✗", "⚠", "·")
    assert (g.bar_full, g.bar_empty) == ("▰", "▱")
    assert (g.cov_full, g.cov_abstract, g.cov_unverified) == ("█", "▓", "░")
    assert (g.rule, g.copy, g.prompt, g.sep) == ("─", "⧉", "›", "·")  # noqa: RUF001


def test_plain_glyphs_are_the_design_set():
    g = PLAIN.glyphs
    assert (g.stage_pending, g.stage_active, g.stage_done, g.stage_flag) == ("-", "*", "+", "*")
    assert (g.error, g.warning, g.note) == ("x", "!", "-")
    assert (g.bar_full, g.bar_empty) == ("#", "-")
    assert (g.cov_full, g.cov_abstract, g.cov_unverified) == ("#", "=", ".")
    assert (g.rule, g.copy, g.prompt, g.sep) == ("-", "[copy]", ">", ",")


def test_single_column_glyphs_are_one_character():
    # ``copy`` is the one glyph allowed to be a word in PLAIN.
    for t in (RICH, PLAIN):
        for field in fields(Glyphs):
            if field.name in {"box", "copy"}:
                continue
            assert len(getattr(t.glyphs, field.name)) == 1, (t.name, field.name)


# --- accents and the banner ------------------------------------------------------------


def test_plain_accents_are_ui_accents() -> None:
    assert PLAIN.accents == ui.ACCENTS
    assert PLAIN.banner == {"dark": ui.PET_COLOUR, "light": ui.PET_COLOUR}
    assert PLAIN.badge is False


FORBIDDEN_NAMES = {"red", "yellow", "green", "bright_red", "bright_yellow", "bright_green"}


def _hue(style: str) -> float:
    hex_part = next(part for part in style.split() if part.startswith("#"))
    r, g, b = (int(hex_part[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def test_rich_accents_are_five_distinct_hex_hues_never_red_yellow_green():
    assert len(RICH.accents) == 5
    assert len(set(RICH.accents)) == 5
    assert RICH.badge is True
    for accent in RICH.accents:
        assert accent not in FORBIDDEN_NAMES
        assert accent.startswith("#") and len(accent) == 7
        hue = _hue(accent)
        # Red wraps around 0/360; yellow sits near 60; green spans ~75-165.
        assert not (hue < 20 or hue > 335), (accent, hue)
        assert not (35 <= hue <= 75), (accent, hue)
        assert not (75 < hue <= 165), (accent, hue)


def test_rich_banner_is_two_crimsons_and_neither_is_an_accent() -> None:
    assert set(RICH.banner) == {"dark", "light"}
    for tone in RICH.banner.values():
        hue = _hue(tone)
        assert hue < 20 or hue > 335, tone
        assert tone not in RICH.accents
    assert RICH.banner["dark"] != RICH.banner["light"]


def test_no_theme_has_a_stamp_any_more() -> None:
    assert not hasattr(RICH, "stamp")
    assert not hasattr(ui, "STAMP_COLOUR")


def test_accent_rotates_by_index():
    for t in (RICH, PLAIN):
        assert t.accent(0) == t.accents[0]
        assert t.accent(4) == t.accents[4]
        assert t.accent(5) == t.accents[0]
        assert t.accent(12) == t.accents[2]


def test_theme_names():
    assert RICH.name == "rich"
    assert PLAIN.name == "plain"
