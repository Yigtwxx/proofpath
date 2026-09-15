"""Exact-string tests for the ferret banner (spec section 13.1).

The four ``SPEC_BLOCK`` lines are copied verbatim out of the design spec's own
80-column block; they are the golden output the renderer has to reproduce.
"""

from __future__ import annotations

import pytest

from proofpath.tui import banner

# design.md section 13.1, the block under "The ferret".
SPEC_BLOCK = (
    "   ,_,",
    "  (o.o)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[PROOF]",
    '   " "    proofpath v0.1.0                            academic . offline . mps',
    "          paste a file path, a URL, or a claim.            /help  /config  /quit",
)
CONTEXT = "academic . offline . mps"
HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"


def render80():
    return banner.render(80, version="0.1.0", context=CONTEXT, hint=HINT)


def test_width_80_reproduces_the_spec_block():
    assert render80().lines == SPEC_BLOCK


def test_every_line_is_pure_ascii():
    for width in (39, 40, 59, 60, 80, 200):
        result = banner.render(width, version="0.1.0", context=CONTEXT, hint=HINT)
        for line in result.lines:
            assert all(ord(character) < 128 for character in line), line


def test_stamp_span_points_at_the_stamp_on_line_two():
    result = render80()
    start, end = result.stamp_span
    assert result.lines[1][start:end] == banner.STAMP


def test_head_top_and_the_eyes_come_from_the_head_art():
    result = render80()
    assert result.lines[0] == banner.HEAD[0]
    assert result.lines[1].startswith("  " + banner.EYES["idle"])
    assert result.lines[2].startswith(banner.HEAD[2])


def test_width_80_body_length_and_right_margin():
    result = render80()
    assert result.lines[1].count("~") == 64
    assert len(result.lines[1]) == 78
    assert len(result.lines[2]) == 78


@pytest.mark.parametrize("eyes", sorted(banner.EYES.values()))
def test_every_eyes_variant_keeps_line_twos_length(eyes):
    result = banner.render(80, version="0.1.0", context=CONTEXT, hint=HINT, eyes=eyes)
    assert len(result.lines[1]) == len(SPEC_BLOCK[1])
    assert result.lines[1][2:7] == eyes
    assert result.stamp_span == render80().stamp_span


def test_width_59_drops_the_stamp():
    result = banner.render(59, version="0.1.0", context=CONTEXT, hint=HINT)
    assert result.stamp_span is None
    assert banner.STAMP not in result.lines[1]
    assert result.lines[1] == "  (o.o)" + "~" * 50
    assert len(result.lines[1]) == 57


def test_width_39_drops_the_body_as_well():
    result = banner.render(39, version="0.1.0", context=CONTEXT, hint=HINT)
    assert result.stamp_span is None
    assert result.lines[1] == "  (o.o)"


def test_width_200_stretches_the_body_and_keeps_one_stamp():
    result = banner.render(200, version="0.1.0", context=CONTEXT, hint=HINT)
    assert len(result.lines[1]) == 198
    assert result.lines[1] == "  (o.o)" + "~" * 184 + "[PROOF]"
    assert len(result.lines[2]) == 198
    assert result.lines[2].endswith(CONTEXT)


def test_minimum_body_is_never_undercut():
    result = banner.render(40, version="0.1.0", context=CONTEXT, hint=HINT)
    assert result.lines[1].count("~") >= banner.MIN_BODY


def test_long_version_is_never_truncated_and_context_overflows_by_one_space():
    version = "0.1.0-rc1+build.2026091201234567890"
    result = banner.render(40, version=version, context=CONTEXT, hint=HINT)
    assert f"proofpath v{version}" in result.lines[2]
    assert result.lines[2].endswith(" " + CONTEXT)
    assert "  " + CONTEXT not in result.lines[2]


def test_context_is_right_aligned_at_every_width():
    for width in (60, 80, 120, 200):
        result = banner.render(width, version="0.1.0", context=CONTEXT, hint=HINT)
        assert len(result.lines[2]) == width - 2
        assert result.lines[2].endswith(CONTEXT)


def test_hint_line_is_the_hint_behind_ten_spaces():
    result = render80()
    assert result.lines[3] == " " * 10 + HINT


def test_banner_is_frozen():
    result = render80()
    with pytest.raises(AttributeError):
        result.lines = ()  # type: ignore[misc]
