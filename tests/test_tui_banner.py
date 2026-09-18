"""What both banner themes share (wordmark design section 8.1): the result type, the
right margin and the two text-row helpers. The art itself is ``test_tui_wordmark.py``'s.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from proofpath.tui import banner

HINT = "paste a file path, a URL, or a claim.            /help  /config  /quit"


def test_the_right_margin_is_two_columns() -> None:
    assert banner.RIGHT_MARGIN == 2


@pytest.mark.parametrize("width", [60, 80, 100, 200])
def test_right_align_pads_so_the_right_part_ends_at_the_margin(width: int) -> None:
    line = banner.right_align("proofpath v0.1.0", "academic . offline . mps", width)
    assert len(line) == width - banner.RIGHT_MARGIN
    assert line.startswith("proofpath v0.1.0 ")
    assert line.endswith(" academic . offline . mps")


def test_right_align_never_truncates_and_keeps_at_least_one_space() -> None:
    left, right = "proofpath v0.1.0-rc1+build.12345", "academic . offline . mps"
    assert banner.right_align(left, right, 40) == left + " " + right
    assert banner.right_align(left, right, 10) == left + " " + right
    # Exactly one space at the width where the two parts just meet.
    exact = len(left) + 1 + len(right) + banner.RIGHT_MARGIN
    assert banner.right_align(left, right, exact) == left + " " + right
    assert banner.right_align(left, right, exact + 1) == left + "  " + right


def test_split_hint_splits_at_the_first_run_of_two_or_more_spaces() -> None:
    assert banner.split_hint(HINT) == (
        "paste a file path, a URL, or a claim.",
        "/help  /config  /quit",
    )
    assert banner.split_hint("a  b  c") == ("a", "b  c")
    assert banner.split_hint("a   b") == ("a", "b")


def test_split_hint_returns_no_commands_when_there_is_no_gap() -> None:
    assert banner.split_hint("no gap here") == ("no gap here", None)
    assert banner.split_hint("") == ("", None)


def test_banner_is_frozen() -> None:
    drawn = banner.Banner(("a",), (((0, 1, "shadow"),),))
    with pytest.raises(AttributeError):
        drawn.lines = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "name",
    ["ART", "render", "GROUND", "GROUND_LINE", "HINT_LINE", "TEXT_INDENT", "BODY_WIDTH_FLOOR"],
)
def test_the_raven_and_its_renderer_are_gone(name: str) -> None:
    assert not hasattr(banner, name), name


def test_module_imports_no_textual() -> None:
    tree = ast.parse(Path(banner.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("textual") for name in imported), imported
