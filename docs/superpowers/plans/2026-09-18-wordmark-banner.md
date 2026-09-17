# Wordmark banner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The `rich` TUI banner draws `ProofPath` as a six-row block-letter wordmark in the two crimsons, with the version and hint lines under it; the Braille raven module goes.

**Architecture:** A new pure module `tui/wordmark.py` carries the six lines verbatim and renders the eight-row banner (`banner.Banner`: lines + tone runs) the way `tui/pet.py` did, delegating to `banner.render` for `plain` and narrow widths. `pet.py` is deleted; the widget, the theme field and the tests are re-pointed. `banner.py` (the `plain` raven) does not change.

**Tech Stack:** Python 3.10+, `textual` 8.2 (widget only), `pytest`.

Spec: `docs/superpowers/specs/2026-09-18-wordmark-banner-design.md` (§3 art, §4 layout, §5 modules, §6 tests, §7 docs).

## Global Constraints

- Type annotations on every function signature; `ruff check` and `ruff format --check` clean from the repo root (CI: `uv run ruff format --check`, which also formats fenced Python in Markdown).
- Code, identifiers and comments in English, in the voice of the surrounding modules (`pet.py`/`banner.py` explain *why* in full sentences).
- `pathlib` everywhere; every file read/write passes `encoding="utf-8"`.
- The wordmark is exactly the six lines of spec §3 (≤ 66 columns, only `█ ╗ ╔ ═ ╝ ║ ╚` and spaces, no trailing spaces). Tones: `light` over `█` runs, `dark` over shadow-glyph runs; rows 6–7 uncoloured.
- `TEXT_COLUMN = 0`, `VERSION_ROW = 6`, `HINT_ROW = 7`, `RICH_WIDTH_FLOOR = 69`; eight rows in `rich`; `plain` and `banner.py` untouched.
- `wordmark.py` imports no Textual and names no colour.
- Product rules are untouched.
- **Subagents never commit.** After a clean review the controller stages; the user commits.

---

### Task 1: `tui/wordmark.py` replaces `tui/pet.py`; widget, theme and tests re-pointed

**Files:**
- Create: `src/proofpath/tui/wordmark.py`
- Delete: `src/proofpath/tui/pet.py`
- Create: `tests/test_tui_wordmark.py`
- Delete: `tests/test_tui_pet.py`
- Modify: `src/proofpath/tui/widgets/banner.py` (imports, docstrings, `_draw`, `render`'s `theme.pet` read)
- Modify: `src/proofpath/tui/theme.py` (`Theme.pet` → `Theme.banner`, lines ~86–88 and ~133; both theme instances)
- Modify: `src/proofpath/tui/banner.py:37-38` (comment only)
- Modify: `src/proofpath/tui/app.py:128,531-533` (comments/docstrings; `raven` local → `mark`)
- Modify: `tests/test_tui_rich.py:21,558-615` and `tests/test_tui_theme.py:271-309`

**Interfaces:**
- Consumes: `banner.Banner(lines, tones)`, `banner.Run = (start, end, tone)`, `banner.right_align`, `banner.split_hint`, `banner.render`, `banner.RIGHT_MARGIN`; the widget's `_draw`/`_render_pet`; `Theme.pet` readers (widget `render`, `test_tui_theme.py`, `test_tui_rich.py`).
- Produces: `wordmark.WORDMARK`, `BLOCK`, `SHADOW`, `TEXT_COLUMN`, `VERSION_ROW`, `HINT_ROW`, `RICH_WIDTH_FLOOR`, `tones(line: str) -> tuple[Run, ...]`, `render(width: int, theme: ThemeLike, *, version: str, context: str, hint: str) -> banner.Banner`; `Theme.banner: Mapping[str, str]`.

- [ ] **Step 1: Read what is being replaced**

Run: `sed -n 1,30p src/proofpath/tui/pet.py; sed -n 60,80p src/proofpath/tui/pet.py; sed -n 136,188p src/proofpath/tui/pet.py; sed -n 1,60p tests/test_tui_pet.py; grep -n "pet" src/proofpath/tui/widgets/banner.py src/proofpath/tui/theme.py src/proofpath/tui/app.py src/proofpath/tui/banner.py tests/test_tui_rich.py tests/test_tui_theme.py`

Note `ThemeLike` (a `Protocol` with `name: str`) in `pet.py` — keep it in `wordmark.py` so the tests can pass a `SimpleNamespace(name="rich")`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_tui_wordmark.py`:

```python
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
```

Then edit `tests/test_tui_rich.py`:
- line 21: `from proofpath.tui import pet` → `from proofpath.tui import wordmark`.
- The `# --- the pet` section (558–615) becomes:

```python
# --- the wordmark ----------------------------------------------------------------------


async def test_the_rich_banner_is_the_wordmark_with_the_text_under_it() -> None:
    app, _ = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        mark = app.query_one(Banner)
        drawn = mark.drawn
        assert drawn is not None
        assert mark.region.height == 8
    assert len(drawn.lines) == 8
    assert drawn.lines[:6] == wordmark.WORDMARK
    assert drawn.lines[wordmark.VERSION_ROW].startswith(f"proofpath v{__version__}")
    assert drawn.lines[wordmark.HINT_ROW].endswith("/help  /config  /quit")
    for line in drawn.lines:
        assert cell_len(line) == len(line) <= 100, line
    allowed = set(wordmark.BLOCK + wordmark.SHADOW + " ")
    assert all(set(line) <= allowed for line in drawn.lines[:6])


async def test_the_rich_wordmark_is_painted_in_the_theme_tones() -> None:
    app, _ = rich_app()
    async with app.run_test(size=SIZE):
        text = app.query_one(Banner).render()
    styles = {str(span.style) for span in text.spans}
    assert RICH.banner["dark"] in styles and RICH.banner["light"] in styles
    # The text lines are not coloured.
    plain_rows = text.plain.split("\n")
    for row in (wordmark.VERSION_ROW, wordmark.HINT_ROW):
        text_start = sum(len(line) + 1 for line in plain_rows[:row]) + wordmark.TEXT_COLUMN
        assert not any(span.start <= text_start < span.end for span in text.spans), row


async def test_the_wordmark_carries_no_styles_without_colour() -> None:
    app, _ = rich_app(out=ui.build(force_terminal=True, no_color=True))
    async with app.run_test(size=SIZE):
        text = app.query_one(Banner).render()
    assert text.spans == []


async def test_the_wordmark_gives_way_to_the_plain_banner_below_the_floor() -> None:
    app, _ = rich_app()
    async with app.run_test(size=(69, 24)) as pilot:
        drawn = app.query_one(Banner).drawn
        assert drawn is not None
        assert drawn.lines[:6] == wordmark.WORDMARK
        assert drawn.lines[wordmark.HINT_ROW].endswith("/help  /config  /quit")
        await pilot.resize_terminal(68, 24)
        await pilot.pause()
        drawn = app.query_one(Banner).drawn
        assert drawn is not None
        assert drawn.lines[0] == banner.ART[0]
```

Check `__version__` and `banner` are importable in that file (`grep -n "^from proofpath\|^import" tests/test_tui_rich.py`); add `from proofpath import __version__` / `from proofpath.tui import banner` if missing. If any other test in `test_tui_rich.py` or `test_tui_app.py` reads `pet.` or asserts the Braille art or `GROUND`, re-point or drop it the same way (grep `pet\.\|⠒\|GROUND` across `tests/`).

Then edit `tests/test_tui_theme.py`: every `PLAIN.pet` / `RICH.pet` → `PLAIN.banner` / `RICH.banner`; rename `test_rich_pet_is_two_crimsons_and_neither_is_an_accent` → `test_rich_banner_is_two_crimsons_and_neither_is_an_accent`; the section comment `# --- accents and the pet` → `# --- accents and the banner`.

Delete `tests/test_tui_pet.py` (`git rm` is a git write — use `rm tests/test_tui_pet.py`; the controller stages the deletion).

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tui_wordmark.py -q 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'wordmark' from 'proofpath.tui'`.

- [ ] **Step 4: Write `src/proofpath/tui/wordmark.py`**

```python
"""The ``rich`` banner: ``ProofPath`` in block letters (wordmark design §3-§4).

:data:`WORDMARK` is the source of truth -- six lines in the block-and-shadow style
of the figlet font *ANSI Shadow*, hand-drawn because that font has no lowercase and
the two ``P`` are the only capitals -- and is kept here verbatim so the mark can be
read without running anything. :func:`render` puts the version line and the hint
line under it; below :data:`RICH_WIDTH_FLOOR` it draws the ``plain`` banner whatever
the theme, as the raven did below its own floor.

Pure: no Textual, and no colour. A block cell is ``light`` and a shadow cell is
``dark``; the result carries colour *runs*, and the widget asks the theme what a tone
is painted with.
"""

from __future__ import annotations

from typing import Protocol

from proofpath.tui import banner
from proofpath.tui.banner import Banner, Run, Tone

#: The mark. ``P`` is six rows; ``r o a`` sit on the baseline four rows tall; ``f h``
#: rise the full six; ``t`` five. Trailing spaces are stripped; only :data:`BLOCK`,
#: :data:`SHADOW` and the space appear (``test_tui_wordmark.py`` reads it back).
WORDMARK: tuple[str, ...] = (
    "██████╗                        █████╗██████╗               ██╗",
    "██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║",
    "██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗",
    "██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗",
    "██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║",
    "╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝",
)
#: The letter body, painted in the light tone.
BLOCK = "█"
#: The shadow to the right of and under each letter, painted in the dark tone.
SHADOW = "╗╔═╝║╚"
#: The two text lines start at the left edge, under the mark, not beside it.
TEXT_COLUMN = 0
VERSION_ROW = len(WORDMARK)
HINT_ROW = VERSION_ROW + 1
#: The mark's width plus the right margin, plus one: narrower than this the mark
#: would touch or cross the margin, so the ``plain`` banner is drawn instead.
RICH_WIDTH_FLOOR = max(len(line) for line in WORDMARK) + banner.RIGHT_MARGIN + 1


class ThemeLike(Protocol):
    """What :func:`render` needs of a theme: its name."""

    name: str


def tones(line: str) -> tuple[Run, ...]:
    """The colour runs of one art line: ``light`` over blocks, ``dark`` over shadow.

    Spaces belong to no run, so a run ends at a space and the next begins after it;
    neighbouring cells of one tone are one run.
    """
    runs: list[Run] = []
    start: int | None = None
    tone: Tone | None = None
    for column, character in enumerate(line):
        here: Tone | None = (
            "light" if character == BLOCK else "dark" if character in SHADOW else None
        )
        if here != tone:
            if tone is not None and start is not None:
                runs.append((start, column, tone))
            start, tone = (column, here) if here is not None else (None, None)
    if tone is not None and start is not None:
        runs.append((start, len(line), tone))
    return tuple(runs)


def render(width: int, theme: ThemeLike, *, version: str, context: str, hint: str) -> Banner:
    """Draw the banner ``theme`` calls for at ``width`` columns.

    In ``rich`` the mark is fixed at its six rows; the version line and the hint
    line follow, each with its right part aligned to the margin. Below
    :data:`RICH_WIDTH_FLOOR` the ``plain`` banner is drawn instead.
    """
    if theme.name != "rich" or width < RICH_WIDTH_FLOOR:
        return banner.render(width, version=version, context=context, hint=hint)
    lines = list(WORDMARK)
    lines.append(banner.right_align(f"proofpath v{version}", context, width))
    prompt, commands = banner.split_hint(hint)
    lines.append(prompt if commands is None else banner.right_align(prompt, commands, width))
    art_tones = tuple(tones(line) for line in WORDMARK)
    return Banner(tuple(lines), (*art_tones, (), ()))


__all__ = [
    "BLOCK",
    "HINT_ROW",
    "RICH_WIDTH_FLOOR",
    "SHADOW",
    "TEXT_COLUMN",
    "VERSION_ROW",
    "WORDMARK",
    "ThemeLike",
    "render",
    "tones",
]
```

Delete `src/proofpath/tui/pet.py` (`rm`). Check nothing else imports it: `grep -rn "tui.pet\|import pet\|pet\." src tests` must return only the widget line you are about to change.

- [ ] **Step 5: Re-point the widget, the theme, the comments**

`src/proofpath/tui/widgets/banner.py`:
- `from proofpath.tui import banner, pet` → `from proofpath.tui import banner, wordmark`.
- Module docstring: `"""The wordmark, pinned above the log and redrawn on every resize (wordmark design §4; the plain raven, banner.py, is unchanged). The art and the layout are :mod:`proofpath.tui.wordmark`'s; …` — keep the rest of the paragraph, replacing "raven" with "wordmark" and "pet tones" with "banner tones".
- Class docstring: `"""The wordmark, pinned above the log and redrawn on every resize."""`.
- The `animated` comment and the `set_busy`/`flash` docstrings: keep "Reserved" and the reference to the raven design §2 (that decision still stands), e.g. `"""Reserved: the banner does not watch the runs yet (raven design §2)."""`.
- `render`: `self._theme.pet[tone]` → `self._theme.banner[tone]`.
- `_draw`: `pet.HINT_ROW` → `wordmark.HINT_ROW`; `_render_pet` → rename `_render_banner` and call `wordmark.render(...)`.

`src/proofpath/tui/theme.py`:
- The field: `pet: Mapping[str, str]` → `banner: Mapping[str, str]`; its comment: `#: The banner's two tones, keyed ``dark`` and ``light`` (``banner.Tone``), …` (keep the rest of the sentence).
- Both theme instances: `pet=` → `banner=`. The comment at ~133 (`#: The raven: the landing page's --crimson-deep and --crimson.`) → `#: The wordmark and the plain raven: the landing page's --crimson-deep and --crimson.` Check for a `PLAIN` reference to `ui.PET_COLOUR` — the name stays (the plain raven is still the pet).

`src/proofpath/tui/banner.py:37-38`: the comment `pet.HINT_ROW` → `wordmark.HINT_ROW`, `test_tui_pet.py` → `test_tui_wordmark.py`. Nothing else in the file changes.

`src/proofpath/tui/app.py`: line ~128 comment and the docstring at ~531 — "the raven" → "the banner" (the decision reference `(raven design section 2)` stays); the local `raven = self.query_one(Banner)` at ~533 → `mark`. Grep `raven` in `app.py` afterwards; anything left must be about the `plain` theme or the design reference.

- [ ] **Step 6: Run the new and re-pointed tests**

Run: `.venv/bin/pytest tests/test_tui_wordmark.py tests/test_tui_rich.py tests/test_tui_theme.py tests/test_tui_banner.py -q 2>&1 | tail -3`
Expected: all PASS. If `test_rich_100_is_the_golden_block` fails on spacing, print `render(100).lines[6:]` and correct the golden (the padding is `100 − 2 − len(left) − len(right)`; do not change `render`).

- [ ] **Step 7: Whole suite, lint, format**

Run: `.venv/bin/pytest -q 2>&1 | tail -3 && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: all PASS, `All checks passed!`, no reformat. Then `grep -rn "\bpet\b" src tests --include=*.py` — the only hits allowed are `ui.PET_COLOUR` and the `plain` raven's own comments.

- [ ] **Step 8: Look at it**

Run: `.venv/bin/python -c "from proofpath.tui import wordmark; from proofpath.tui.theme import RICH; print('\n'.join(wordmark.render(80, RICH, version='0.4.4', context='academic · online · coreml', hint='paste a file path, a URL, or a claim.            /help  /config  /quit').lines))"`
Expected: the §4 block. Put this output in your report.

- [ ] **Step 9: Hand over for review (no commit)**

---

### Task 2: Docs — README, main spec §13.1, the raven spec's status, CHANGELOG

**Files:**
- Modify: `README.md:24-37` (the Braille block and the sentence after it) and the sentence at ~39-41 that links the SVG screenshots
- Modify: `docs/superpowers/specs/2026-09-10-proofpath-design.md` §13.1 (one sentence after the `plain` golden block, ~line 575)
- Modify: `docs/superpowers/specs/2026-09-16-raven-pet-design.md:3` (Status)
- Modify: `CHANGELOG.md` `## [Unreleased]`

**Interfaces:**
- Consumes: the §4 layout of the wordmark spec (verbatim).

- [ ] **Step 1: README**

Replace the fenced block that starts with `       ⣴⣿⠿⣷⣀` (8 lines, ending with the `⠒` ground) with:

````
```
██████╗                        █████╗██████╗               ██╗
██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║
██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗
██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗
██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║
╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝
proofpath v0.4.4                                                        academic · online · coreml
paste a file path, a URL, or a claim.                                        /help  /config  /quit
```
````

(the 100-column layout, to match the width the old block used). Replace the paragraph

> The bird is a raven — the same one as on the website — drawn in Braille cells in the `rich` theme and in ASCII in `plain`.

with:

> The `rich` theme opens with the wordmark in the two crimsons; the `plain` theme opens with the raven — the same bird as on the website — in ASCII.

In the sentence that links the SVG screenshots, change `with SVG screenshots of a real run in` to `with SVG screenshots of a real run on v0.4.1, before the wordmark, in`. Leave the `plain` block and its paragraph alone.

- [ ] **Step 2: Main spec §13.1**

After the fenced `plain` golden block (the one starting `       __` / `      (o >`), before `› ~/Desktop/paper.pdf` — check the block's extent first; the run block under the banner is part of the same fence — add, as the first sentence of the paragraph that follows the fence (or a new paragraph if the fence is followed by a bullet):

> In `rich` the banner is the `ProofPath` wordmark of `2026-09-18-wordmark-banner-design.md`; the block above is the `plain` banner.

- [ ] **Step 3: Raven spec status**

Line 3 of `docs/superpowers/specs/2026-09-16-raven-pet-design.md`, after `implementation follows this document.`, add: ` **`RICH` superseded 2026-09-18 by `2026-09-18-wordmark-banner-design.md`; §5 (`PLAIN`) stands.**`

- [ ] **Step 4: CHANGELOG**

Under `## [Unreleased]`, add a `### Changed` section (after the existing `### Fixed` if one is there, keeping Keep-a-Changelog order: Added, Changed, Deprecated, Removed, Fixed, Security — so `### Changed` goes *before* `### Fixed`):

```markdown
### Changed
- **TUI: the `rich` banner is a wordmark.** `ProofPath` in block letters, crimson on
  deep crimson, with the version and hint lines under it; the Braille raven is gone.
  The ASCII raven of the `plain` theme and the website's raven stay
  (`docs/superpowers/specs/2026-09-18-wordmark-banner-design.md`).
```

- [ ] **Step 5: Verify**

Run: `uv run ruff format --check . 2>&1 | tail -2; grep -c "wordmark" README.md CHANGELOG.md docs/superpowers/specs/2026-09-10-proofpath-design.md docs/superpowers/specs/2026-09-16-raven-pet-design.md; grep -c "⣿" README.md`
Expected: no `Would reformat`; counts `1 1 1 1` (at least); the Braille count is `0`.

- [ ] **Step 6: Hand over for review (no commit)**
