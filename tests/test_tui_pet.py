"""Exact-string tests for both ferrets and the animation (TUI v2 design section 3).

``RICH_80`` and ``RICH_100`` are the golden renders of the RICH ferret; the PLAIN
ferret is ``banner.render``'s block, which ``test_tui_banner.py`` already pins to
the design spec. The state machine is driven with a fake clock, never a timer.
"""

from __future__ import annotations

import random
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from proofpath.tui import banner, pet

CONTEXT = "academic · online · coreml"
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"
VERSION = "0.3.0"

RICH = SimpleNamespace(name="rich")
PLAIN = SimpleNamespace(name="plain")

RICH_80 = (
    "    ╭╮ ╭╮",
    "   ╭╯╰─╯╰───────────────────────────────────────────────────╮",
    "  ╸┤ o o                                                    ╰~~~~~~~~~~[PROOF]",
    "   ╰─┬─┬───────────────────────────────────────────────┬─┬──╯",
    "     ˘ ˘                                               ˘ ˘",
    "      proofpath v0.3.0                              academic · online · coreml",
    "      paste a file path, a URL, or a claim.              /help  /config  /quit",
)
RICH_100 = (
    "    ╭╮ ╭╮",
    "   ╭╯╰─╯╰────────────────────────────────────────────────────────────────────╮",
    "  ╸┤ o o                                                                     ╰~~~~~~~~~~~~~[PROOF]",  # noqa: E501
    "   ╰─┬─┬────────────────────────────────────────────────────────────────┬─┬──╯",
    "     ˘ ˘                                                                ˘ ˘",
    "      proofpath v0.3.0                                                  academic · online · coreml",  # noqa: E501
    "      paste a file path, a URL, or a claim.                                  /help  /config  /quit",  # noqa: E501
)
# No stamp below 60 columns, and the tail shortens to four.
RICH_59 = (
    "    ╭╮ ╭╮",
    "   ╭╯╰─╯╰───────────────────────────────────────────╮",
    "  ╸┤ o o                                            ╰~~~~",
    "   ╰─┬─┬───────────────────────────────────────┬─┬──╯",
    "     ˘ ˘                                       ˘ ˘",
    "      proofpath v0.3.0         academic · online · coreml",
    "      paste a file path, a URL, or a claim. /help  /config  /quit",
)


def render(width, theme=RICH, **overrides):
    kwargs = {"version": VERSION, "context": CONTEXT, "hint": HINT, **overrides}
    return pet.render(width, theme, **kwargs)


def art_lines(drawn):
    """The animal itself: everything above the two text lines."""
    return drawn.lines[: len(drawn.lines) - 2]


# --- the RICH ferret --------------------------------------------------------------


def test_rich_80_is_the_golden_block():
    assert render(80).lines == RICH_80


def test_rich_100_is_the_golden_block():
    assert render(100).lines == RICH_100


def test_rich_59_is_the_golden_block():
    assert render(59).lines == RICH_59


@pytest.mark.parametrize("width", range(48, 201))
def test_rich_is_seven_lines_each_within_the_width(width):
    """The animal never overflows; only the two text lines may, and only below 65.

    Below 65 columns the hint (37 + 21 characters, one space between, six in)
    no longer fits: it overflows rather than truncate, exactly as PLAIN does.
    """
    drawn = render(width)
    assert len(drawn.lines) == 7
    for line in art_lines(drawn):
        assert len(line) <= width, line
    for line in drawn.lines[5:]:
        if width >= 65:
            assert len(line) <= width, line
    assert len(render(48).lines[6]) > 48


@pytest.mark.parametrize("width", [48, 59, 60, 80, 100, 200])
def test_rich_uses_only_single_width_characters(width):
    for line in render(width).lines:
        for character in line:
            assert unicodedata.east_asian_width(character) not in {"W", "F"}, line


def test_rich_spans_point_at_the_stamp_the_eyes_and_the_tail():
    drawn = render(80)
    line, start, end = drawn.stamp_span
    assert drawn.lines[line][start:end] == pet.STAMP
    line, start, end = drawn.eyes_span
    assert drawn.lines[line][start:end] == pet.EYES["idle"]
    line, start, end = drawn.tail_span
    assert drawn.lines[line][start:end] == "~" * (end - start)
    assert line == 2


def test_rich_stamp_ends_two_columns_short_of_the_edge_like_plain():
    for width in (60, 80, 100, 200):
        drawn = render(width)
        line, _start, end = drawn.stamp_span
        assert line == 2
        assert end == width - banner.RIGHT_MARGIN
        assert len(drawn.lines[2]) == width - banner.RIGHT_MARGIN
        if width >= 80:
            assert len(drawn.lines[5]) == len(drawn.lines[6]) == width - banner.RIGHT_MARGIN


def test_rich_rump_corners_share_one_column_and_the_tail_leaves_its_middle():
    for width in (48, 59, 60, 80, 100, 200):
        drawn = render(width)
        back, face, belly = drawn.lines[1:4]
        edge = back.index("╮")
        assert len(back) == edge + 1
        assert face[edge] == "╰"
        assert belly[edge] == "╯" and len(belly) == edge + 1
        _line, start, _end = drawn.tail_span
        assert start == edge + 1


@pytest.mark.parametrize(
    ("width", "tail"), [(48, 4), (59, 4), (60, 8), (80, 10), (100, 13), (200, 30)]
)
def test_rich_tail_is_at_least_eight_and_grows_with_the_width(width, tail):
    drawn = render(width)
    line, start, end = drawn.tail_span
    assert line == 2
    assert end - start == tail
    assert drawn.lines[line][start:end] == "~" * tail
    if drawn.stamp_span is not None:
        assert drawn.lines[line][end:] == pet.STAMP
    else:
        assert len(drawn.lines[line]) == end


def test_rich_tail_never_passes_a_third_of_the_body():
    for width in range(60, 301):
        drawn = render(width)
        _line, start, end = drawn.tail_span
        body = drawn.lines[1].count("─")
        assert end - start <= body / 3 + 1, width


def test_rich_has_four_feet_under_four_leg_joints():
    drawn = render(80)
    belly, feet = drawn.lines[3], drawn.lines[4]
    joints = [column for column, glyph in enumerate(belly) if glyph == "┬"]
    assert len(joints) == 4
    assert feet.count("˘") == 4
    assert all(feet[column] == "˘" for column in joints)


@pytest.mark.parametrize("eyes", sorted(pet.EYES))
def test_rich_every_eyes_variant_keeps_the_face_line_intact(eyes):
    drawn = render(80, eyes=eyes)
    assert len(drawn.lines[2]) == len(RICH_80[2])
    line, start, end = drawn.eyes_span
    assert drawn.lines[line][start:end] == pet.EYES[eyes]
    assert drawn.lines[1] == RICH_80[1]
    assert drawn.lines[2][8:] == RICH_80[2][8:]


def test_rich_59_drops_the_stamp_and_keeps_the_body():
    drawn = render(59)
    assert drawn.stamp_span is None
    assert pet.STAMP not in "".join(drawn.lines)
    assert len(drawn.lines) == 7
    assert drawn.lines[2].endswith("╰~~~~")
    assert len(drawn.lines[2]) == 59 - banner.RIGHT_MARGIN


def test_rich_47_returns_the_plain_pet():
    drawn = render(47)
    plain = banner.render(47, version=VERSION, context=CONTEXT, hint=HINT)
    assert drawn.lines == plain.lines
    assert len(drawn.lines) == 4
    assert drawn.tail_span is None
    assert drawn.stamp_span is None


def test_rich_48_is_still_the_rich_pet():
    assert len(render(48).lines) == 7


def test_rich_text_lines_are_right_aligned_with_a_two_column_margin():
    for width in (80, 100, 200):
        drawn = render(width)
        assert drawn.lines[5].endswith(CONTEXT)
        assert drawn.lines[6].endswith("/help  /config  /quit")
        assert len(drawn.lines[5]) == width - banner.RIGHT_MARGIN
        assert len(drawn.lines[6]) == width - banner.RIGHT_MARGIN
        assert drawn.lines[5].startswith(pet.TEXT_INDENT + f"proofpath v{VERSION}")
        assert drawn.lines[6].startswith(pet.TEXT_INDENT + "paste a file path")


def test_rich_hint_without_a_gap_is_indented_whole():
    drawn = render(80, hint="type a claim")
    assert drawn.lines[6] == pet.TEXT_INDENT + "type a claim"


def test_rich_long_version_is_never_truncated():
    version = "0.3.0-rc1+build.2026091501234567890"
    drawn = render(48, version=version)
    assert f"proofpath v{version}" in drawn.lines[5]
    assert drawn.lines[5].endswith(" " + CONTEXT)
    assert "  " + CONTEXT not in drawn.lines[5]


# --- the tail --------------------------------------------------------------------


def test_tail_offset_selects_one_of_three_wag_frames():
    tip = render(80).tail_span[2] - 1
    tips = [render(80, tail_offset=offset).lines[2][tip] for offset in range(3)]
    assert tips == list(pet.TAIL_TIPS) == ["~", "‾", "_"]


def test_tail_offset_wraps_and_moves_nothing_else():
    for offset in (1, 2, 3, 4, 31, -1):
        drawn = render(80, tail_offset=offset)
        assert drawn.lines[:2] == RICH_80[:2]
        assert drawn.lines[3:] == RICH_80[3:]
        line, start, end = drawn.tail_span
        tail = drawn.lines[line][start:end]
        assert len(tail) == 10
        assert tail[:-1] == "~" * 9
        assert tail[-1] == pet.TAIL_TIPS[offset % 3]
        assert drawn.lines[2].endswith(pet.STAMP)
        assert drawn.stamp_span == render(80).stamp_span
        assert drawn.tail_span == render(80).tail_span
    assert render(80, tail_offset=3).lines == RICH_80


def test_tail_wags_without_the_stamp_too():
    drawn = render(59, tail_offset=1)
    line, start, end = drawn.tail_span
    assert drawn.lines[line][start:end] == "~~~‾"
    assert len(drawn.lines[line]) == end


# --- the PLAIN ferret ------------------------------------------------------------


@pytest.mark.parametrize("width", [39, 47, 59, 80, 200])
def test_plain_delegates_to_banner_at_every_width(width):
    drawn = render(width, PLAIN)
    plain = banner.render(width, version=VERSION, context=CONTEXT, hint=HINT)
    assert drawn.lines == plain.lines
    assert drawn.tail_span is None
    if plain.stamp_span is None:
        assert drawn.stamp_span is None
    else:
        assert drawn.stamp_span == (1, *plain.stamp_span)


def test_plain_80_is_the_spec_block_with_the_spec_context():
    context, hint = "academic . offline . mps", HINT
    drawn = pet.render(80, PLAIN, version="0.1.0", context=context, hint=hint)
    plain = banner.render(80, version="0.1.0", context=context, hint=hint)
    assert drawn.lines == plain.lines
    assert drawn.lines[1].startswith("  (o.o)~~~")


@pytest.mark.parametrize("eyes", sorted(pet.EYES))
def test_plain_maps_the_eyes_key_to_banner_eyes(eyes):
    drawn = render(80, PLAIN, eyes=eyes)
    line, start, end = drawn.eyes_span
    assert drawn.lines[line][start:end] == banner.EYES[eyes]
    assert drawn.stamp_span is not None
    _line, start, end = drawn.stamp_span
    assert drawn.lines[1][start:end] == banner.STAMP


def test_both_themes_share_one_set_of_eyes_keys():
    assert set(pet.EYES) == set(banner.EYES)


def test_real_theme_objects_are_accepted_when_present():
    theme = pytest.importorskip("proofpath.tui.theme")
    assert len(render(80, theme.RICH).lines) == 7
    assert len(render(80, theme.PLAIN).lines) == 4


def test_pet_is_frozen():
    drawn = render(80)
    with pytest.raises(AttributeError):
        drawn.lines = ()  # type: ignore[misc]


def test_module_is_pure_and_imports_no_textual():
    source = Path(pet.__file__).read_text(encoding="utf-8")
    assert "import textual" not in source
    assert "from textual" not in source


# --- the animation ---------------------------------------------------------------


def start(now=0.0, seed=7):
    return pet.AnimState.start(now, rng=random.Random(seed))


def step(state, now, *, running=False, ended_with=None):
    return pet.next_eyes(state, now, running=running, ended_with=ended_with)


def test_first_blink_is_scheduled_six_to_ten_seconds_out():
    state = start(100.0)
    assert 100.0 + pet.BLINK_WINDOW[0] <= state.blink_at <= 100.0 + pet.BLINK_WINDOW[1]


def test_idle_before_the_blink_is_idle():
    state = start(0.0)
    state, eyes = step(state, state.blink_at - 0.01)
    assert eyes == "idle"


def test_blink_lasts_150ms_then_reschedules_within_the_window():
    state = start(0.0)
    due = state.blink_at
    state, eyes = step(state, due)
    assert eyes == "blink"
    assert state.blink_until == pytest.approx(due + pet.BLINK_SECONDS)
    assert due + pet.BLINK_WINDOW[0] <= state.blink_at <= due + pet.BLINK_WINDOW[1]
    state, eyes = step(state, due + pet.BLINK_SECONDS - 0.01)
    assert eyes == "blink"
    state, eyes = step(state, due + pet.BLINK_SECONDS)
    assert eyes == "idle"


def test_busy_while_running_and_an_owed_blink_is_dropped():
    state = start(0.0)
    due = state.blink_at
    state, eyes = step(state, due - 1.0, running=True)
    assert eyes == "busy"
    # The run outlives the scheduled blink; ending it must not fire the blink.
    state, eyes = step(state, due + 1.0, running=True)
    assert eyes == "busy"
    assert state.blink_at > due + 1.0
    state, eyes = step(state, due + 1.0)
    assert eyes == "idle"


@pytest.mark.parametrize("mood", ["findings", "clean"])
def test_end_of_run_flashes_the_mood_for_two_seconds(mood):
    state = start(0.0)
    state, eyes = step(state, 1.0, ended_with=mood)
    assert eyes == mood
    state, eyes = step(state, 1.0 + pet.FLASH_SECONDS - 0.01)
    assert eyes == mood
    state, eyes = step(state, 1.0 + pet.FLASH_SECONDS)
    assert eyes == "idle"
    assert state.mood is None


def test_mood_outranks_a_run_still_going():
    state = start(0.0)
    state, eyes = step(state, 1.0, running=True, ended_with="findings")
    assert eyes == "findings"
    state, eyes = step(state, 2.0, running=True)
    assert eyes == "findings"
    state, eyes = step(state, 3.5, running=True)
    assert eyes == "busy"


def test_mood_outranks_a_blink():
    state = start(0.0)
    due = state.blink_at
    state, eyes = step(state, due - 0.5, ended_with="clean")
    state, eyes = step(state, due)
    assert eyes == "clean"


def test_a_new_end_restarts_the_flash():
    state = start(0.0)
    state, _ = step(state, 1.0, ended_with="clean")
    state, eyes = step(state, 2.5, ended_with="findings")
    assert eyes == "findings"
    state, eyes = step(state, 4.4)
    assert eyes == "findings"
    state, eyes = step(state, 4.5)
    assert eyes == "idle"


def test_state_machine_is_deterministic_for_a_seed():
    a, b = start(0.0, seed=3), start(0.0, seed=3)
    for now in (0.5, 7.0, 7.1, 7.2, 20.0, 40.0):
        a, eyes_a = step(a, now)
        b, eyes_b = step(b, now)
        assert eyes_a == eyes_b
    assert a.blink_at == b.blink_at


def test_every_eyes_key_the_machine_emits_renders_in_both_themes():
    state = start(0.0)
    seen = set()
    for now, running, ended in [
        (1.0, False, None),
        (state.blink_at, False, None),
        (state.blink_at + 0.5, True, None),
        (state.blink_at + 1.0, False, "findings"),
        (state.blink_at + 4.0, False, "clean"),
    ]:
        state, eyes = step(state, now, running=running, ended_with=ended)
        seen.add(eyes)
        render(80, RICH, eyes=eyes)
        render(80, PLAIN, eyes=eyes)
    assert seen == {"idle", "blink", "busy", "findings", "clean"}


def test_anim_state_is_frozen():
    state = start()
    with pytest.raises(AttributeError):
        state.blink_at = 1.0  # type: ignore[misc]


def wag(state, now, *, running):
    return pet.next_tail(state, now, running=running)


def test_tail_is_still_while_idle():
    state = start(0.0)
    for now in (0.0, 1.0, 5.0, 60.0):
        state, frame = wag(state, now, running=False)
        assert frame == 0


def test_tail_wags_at_two_hertz_while_busy():
    state = start(0.0)
    frames = []
    now = 10.0
    while now < 12.5:
        state, frame = wag(state, now, running=True)
        frames.append(frame)
        now += 0.05
    # Frame steps every half second, cycling through the three tips.
    # lift, level, drop, level: the tip never jumps from lifted straight to dropped.
    assert frames[0] == 1
    assert frames[9] == 1 and frames[10] == 0
    assert frames[19] == 0 and frames[20] == 2
    assert frames[29] == 2 and frames[30] == 0
    assert frames[39] == 0 and frames[40] == 1
    assert set(frames) == {0, 1, 2}


def test_tail_drops_to_rest_the_moment_the_run_ends():
    state = start(0.0)
    state, frame = wag(state, 10.0, running=True)
    state, frame = wag(state, 10.6, running=True)
    assert frame == 0  # level, on the way down
    state, frame = wag(state, 11.1, running=True)
    assert frame == 2  # dropped
    state, frame = wag(state, 11.2, running=False)
    assert frame == 0
    # The next run starts its wag afresh, not mid-cycle.
    state, frame = wag(state, 20.0, running=True)
    assert frame == 1


def test_every_wag_frame_renders():
    state = start(0.0)
    for now in (1.0, 1.5, 2.0):
        state, frame = wag(state, now, running=True)
        drawn = render(80, tail_offset=frame)
        assert len(drawn.lines[2]) == 78
