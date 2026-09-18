# Wordmark everywhere: rule, gradient, beside layout, ASCII plain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Both TUI themes open with the `ProofPath` wordmark — `rich` in block glyphs with a five-band crimson gradient, `plain` in ASCII — the banner ends in a full-width rule, the version/hint rows sit beside the mark on a wide terminal and under it otherwise, and the ASCII raven is gone from the TUI, the README and the main spec.

**Architecture:** `tui/wordmark.py` becomes the only banner renderer: `render(width, theme, …)` picks a glyph table by `theme.name`, decides beside/under/narrow from the actual text lengths, and reports the decision through `rows(...)`; `tones()` paints column bands. `tui/banner.py` keeps only the shared helpers (`Banner`, `Run`, `Tone`, `RIGHT_MARGIN`, `right_align`, `split_hint`). The widget asks `rows(...)` for the hint row. Spec: `docs/superpowers/specs/2026-09-18-wordmark-banner-design.md` §8 and §8.1.

**Tech Stack:** Python 3.10+, `textual` 8.2 (widget only), `pytest`.

## Global Constraints

- Type annotations on every function signature; `uv run ruff check` / `uv run ruff format --check` clean from the repo root.
- Comments in English, in the voice of `wordmark.py` (why, in full sentences). `wordmark.py` and `banner.py` import no Textual and name no colour.
- Spec §8/§8.1 values verbatim: rule `─` (`-` in plain) from column 0 to `width − 2`, last row; `BESIDE_COLUMN = 68`; beside = texts on the mark's rows 2 (version) and 3 (hint); under = six mark rows, version, hint, rule; below `RICH_WIDTH_FLOOR = 69` no mark: version, hint, rule (3 rows); gradient `min(4, column * 5 // 66)` with tones `g0`…`g4`; shadow glyphs and the rule are tone `shadow`; `RICH` colours `#e0455f #d02f4c #c4173a #a91330 #8f0f2b`, shadow `#6b0a20`; `PLAIN` maps every tone to `ui.PET_COLOUR`; plain transliteration `█→# ╗╔╝╚→+ ═→- ║→| ─→-`.
- Nothing is ever truncated. Product rules untouched. **Subagents never commit or stage.**

---

### Task 1: `wordmark.py` renders both themes; `banner.py` shrinks; widget, theme and tests follow

**Files:**
- Modify: `src/proofpath/tui/wordmark.py` (rewrite most of it)
- Modify: `src/proofpath/tui/banner.py` (remove the raven; keep the shared helpers)
- Modify: `src/proofpath/tui/theme.py` (`Theme.banner` tones; `BANNER_TONES`)
- Modify: `src/proofpath/tui/widgets/banner.py` (`_draw`, docstrings)
- Modify: `tests/test_tui_wordmark.py`, `tests/test_tui_banner.py`, `tests/test_tui_rich.py` (banner section), `tests/test_tui_theme.py` (banner keys), `tests/test_tui_app.py` (~lines 305–330: the three `banner.render`/pure-ASCII tests)

**Interfaces:**
- Consumes: `banner.Banner(lines, tones)`, `banner.Run`, `banner.right_align`, `banner.split_hint`, `banner.RIGHT_MARGIN`; today's `wordmark.WORDMARK/BLOCK/SHADOW/RICH_WIDTH_FLOOR/_fits`.
- Produces: `wordmark.RULE = "─"`, `PLAIN_GLYPHS` (a `str.maketrans` table), `BESIDE_COLUMN = 68`, `BESIDE_VERSION_ROW = 2`, `BESIDE_HINT_ROW = 3`, `MARK_WIDTH = 66`, `BANDS = 5`, `GRADIENT_TONES = ("g0", "g1", "g2", "g3", "g4")`, `SHADOW_TONE = "shadow"`, `Rows(version: int, hint: int, rule: int, beside: bool)` (frozen dataclass), `rows(width, *, version, context, hint) -> Rows`, `tones(line) -> tuple[Run, ...]`, `render(width, theme, *, version, context, hint) -> Banner` (both themes). Removed: `VERSION_ROW`, `HINT_ROW`, `hint_row`, `banner.render`, `banner.ART/GROUND/GROUND_LINE/HINT_LINE/TEXT_INDENT/BODY_WIDTH_FLOOR/_beside/_first_ink/_run`. `banner.Tone = str`. `theme.BANNER_TONES: tuple[str, ...]`; `Theme.banner` keyed by it.

- [ ] **Step 1: Read what changes**

Run: `cat src/proofpath/tui/wordmark.py src/proofpath/tui/banner.py; sed -n 80,140p src/proofpath/tui/theme.py; sed -n 60,110p src/proofpath/tui/widgets/banner.py; grep -n "banner\.\|wordmark\." tests/test_tui_app.py tests/test_tui_rich.py | head -40; cat tests/test_tui_banner.py`

- [ ] **Step 2: Write the tests first**

`tests/test_tui_wordmark.py` — keep the four art tests and the purity test; replace everything else with:

```python
VERSION = "0.4.5"
RICH = SimpleNamespace(name="rich")
PLAIN = SimpleNamespace(name="plain")


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
    for line in plain.lines:
        assert line.isascii(), line
    assert render(140, PLAIN).lines == tuple(
        line.translate(wordmark.PLAIN_GLYPHS) for line in BESIDE_140
    )


def test_plain_glyph_table_is_the_spec_mapping() -> None:
    assert "█╗╔╝╚═║─".translate(wordmark.PLAIN_GLYPHS) == "#++++-|-"


def test_rows_switch_to_beside_exactly_where_the_texts_fit() -> None:
    assert wordmark.rows(128, version=VERSION, context=CONTEXT, hint=HINT) == wordmark.Rows(
        6, 7, 8, False
    )
    assert wordmark.rows(129, version=VERSION, context=CONTEXT, hint=HINT) == wordmark.Rows(
        2, 3, 6, True
    )
    prompt_only = HINT.split("  ")[0]
    assert wordmark.rows(106, version=VERSION, context=CONTEXT, hint=prompt_only).beside is False
    assert wordmark.rows(107, version=VERSION, context=CONTEXT, hint=prompt_only).beside is True
    assert wordmark.rows(68, version=VERSION, context=CONTEXT, hint=HINT) == wordmark.Rows(
        0, 1, 2, False
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


def test_tones_follow_the_column_band_for_blocks_and_shadow_for_the_rest() -> None:
    for row, line in enumerate(wordmark.WORDMARK):
        painted: dict[int, str] = {}
        for start, end, tone in wordmark.tones(line):
            painted.update({column: tone for column in range(start, end)})
        for column, character in enumerate(line):
            if character == wordmark.BLOCK:
                expected = wordmark.GRADIENT_TONES[min(4, column * 5 // wordmark.MARK_WIDTH)]
                assert painted[column] == expected, (row, column)
            elif character in wordmark.SHADOW:
                assert painted[column] == wordmark.SHADOW_TONE, (row, column)
            else:
                assert column not in painted, (row, column)


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
```

`tests/test_tui_banner.py` — replace the raven tests with the helpers' contracts: `right_align` pads to `width − RIGHT_MARGIN` and never below one space; `split_hint` splits at the first run of two or more spaces and returns `None` for the commands when there is none; `RIGHT_MARGIN == 2`; `banner.py` exposes no `ART`, `render`, `GROUND`, `HINT_LINE` (`assert not hasattr(banner, name)` for each); the module imports no Textual (same AST check as `test_tui_wordmark.py`).

`tests/test_tui_app.py` (~305–330): `test_banner_reproduces_the_renderer_at_eighty_columns` compares against `wordmark.render(80, PLAIN, …)`; `test_banner_is_pure_ascii` stays (the plain mark is ASCII); the docstring "the pet renders identically" → "the banner renders identically". Import `wordmark`.

`tests/test_tui_rich.py` banner section: at `SIZE` (80) the region height is 9 and `lines[:6] == WORDMARK`; new test at `size=(140, 24)`: height 7, `lines[3].startswith(WORDMARK[3])`, `lines[3].endswith("/help  /config  /quit")`; tones test: `RICH.banner["g0"]`, `["g4"]`, `["shadow"]` among the styles; the below-the-floor test: at 68 columns `lines[0].startswith(f"proofpath v{__version__}")` and height 3; the no-colour test unchanged.

`tests/test_tui_theme.py`: `set(RICH.banner) == set(theme.BANNER_TONES) == {"g0","g1","g2","g3","g4","shadow","light","dark"}`; each `RICH.banner` value has hue `< 20 or > 335` and is not an accent; `RICH.banner["g0"] != RICH.banner["g4"]`; `PLAIN.banner == {tone: ui.PET_COLOUR for tone in theme.BANNER_TONES}`.

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/pytest tests/test_tui_wordmark.py tests/test_tui_banner.py -q 2>&1 | tail -2` — expected `AttributeError` on `rows`/`RULE`/`PLAIN_GLYPHS`.

- [ ] **Step 4: `banner.py` — the shared helpers only**

Keep the module docstring's intent but rewrite it: "What both banner themes share: the rendered-lines-plus-runs result type, the right margin and the two text-row helpers. The art lives in `wordmark`." Keep `Banner`, `Run`, `RIGHT_MARGIN = 2`, `HINT_GAP`, `right_align`, `split_hint`. Change `Tone = Literal["dark", "light"]` to `Tone = str` with the comment `#: A tone name the theme maps to a colour — the wordmark's gradient bands and "shadow".` Delete `ART`, `GROUND`, `GROUND_LINE`, `HINT_LINE`, `TEXT_INDENT`, `BODY_WIDTH_FLOOR`, `render`, `_beside`, `_first_ink`, `_run`, and the unused imports (`re` stays for `HINT_GAP`; drop `Literal` if unused). Update `__all__`.

- [ ] **Step 5: `wordmark.py`**

Replace the module (keep `WORDMARK`, `BLOCK`, `SHADOW`, `ThemeLike`, `_fits`, `RICH_WIDTH_FLOOR` as they are). New constants after `SHADOW`:

```python
#: The rule that closes the banner in both layouts, from column 0 to the margin.
RULE = "─"
#: The ``plain`` theme draws the same mark in seven-bit ASCII, cell for cell, so the
#: rows and the runs are identical in both themes and only the glyphs differ.
PLAIN_GLYPHS = str.maketrans("█╗╔╝╚═║─", "#++++-|-")
MARK_WIDTH = max(len(line) for line in WORDMARK)
#: When the texts sit beside the mark they start two columns past its widest line.
BESIDE_COLUMN = MARK_WIDTH + 2
#: The mark's middle two rows, which the two texts share when they sit beside it.
BESIDE_VERSION_ROW, BESIDE_HINT_ROW = 2, 3
#: Left to right, the blocks run through these tones in equal column bands of the
#: mark; the bands are fixed by the mark's width, never by the terminal's.
GRADIENT_TONES: tuple[str, ...] = ("g0", "g1", "g2", "g3", "g4")
BANDS = len(GRADIENT_TONES)
#: The shadow glyphs and the rule share one tone, darker than the gradient's end.
SHADOW_TONE = "shadow"
```

```python
@dataclass(frozen=True)
class Rows:
    """Where :func:`render` put the version, the hint and the rule for one width."""

    version: int
    hint: int
    rule: int
    beside: bool


def _tone(column: int, character: str) -> str | None:
    """The tone one cell paints: its band for a block, the shadow tone for a shadow
    glyph, nothing for a space."""
    if character == BLOCK:
        return GRADIENT_TONES[min(BANDS - 1, column * BANDS // MARK_WIDTH)]
    return SHADOW_TONE if character in SHADOW else None


def tones(line: str) -> tuple[Run, ...]:
    """The colour runs of one art line. A block run is split where its band changes;
    a shadow run is not, because the shadow has one tone. Spaces belong to no run."""
    runs: list[Run] = []
    start = 0
    tone: str | None = None
    for column, character in enumerate(line):
        here = _tone(column, character)
        if here != tone:
            if tone is not None:
                runs.append((start, column, tone))
            start, tone = column, here
    if tone is not None:
        runs.append((start, len(line), tone))
    return tuple(runs)


def _beside_fits(width: int, *, version: str, context: str, hint: str) -> bool:
    """Whether both text rows fit to the right of the mark, each with at least one
    space between its halves, ending at the margin."""
    prompt, commands = banner.split_hint(hint)
    longest = max(
        len(f"proofpath v{version}") + 1 + len(context),
        len(prompt) + (1 + len(commands) if commands is not None else 0),
    )
    return width - banner.RIGHT_MARGIN - BESIDE_COLUMN >= longest


def rows(width: int, *, version: str, context: str, hint: str) -> Rows:
    """Where :func:`render` puts things at ``width``: beside the mark when both texts
    fit to its right, under it otherwise, and with no mark at all below the floor.
    The same for both themes; the widget asks this rather than guessing."""
    if not _fits(width):
        return Rows(version=0, hint=1, rule=2, beside=False)
    if _beside_fits(width, version=version, context=context, hint=hint):
        return Rows(
            version=BESIDE_VERSION_ROW, hint=BESIDE_HINT_ROW, rule=len(WORDMARK), beside=True
        )
    return Rows(version=len(WORDMARK), hint=len(WORDMARK) + 1, rule=len(WORDMARK) + 2, beside=False)


def render(width: int, theme: ThemeLike, *, version: str, context: str, hint: str) -> Banner:
    """Draw the banner for ``theme`` at ``width`` columns: the mark (when it fits),
    the two text rows beside or under it, and the rule. ``plain`` is the same
    drawing through :data:`PLAIN_GLYPHS`.
    """
    where = rows(width, version=version, context=context, hint=hint)
    prompt, commands = banner.split_hint(hint)
    span = width - banner.RIGHT_MARGIN

    def text_rows(prefix_version: str, prefix_hint: str) -> tuple[str, str]:
        version_line = banner.right_align(prefix_version + f"proofpath v{version}", context, width)
        hint_line = (
            prefix_hint + prompt
            if commands is None
            else banner.right_align(prefix_hint + prompt, commands, width)
        )
        return version_line, hint_line

    if not _fits(width):
        lines = [*text_rows("", ""), RULE * span]
        runs: list[tuple[Run, ...]] = [(), (), ((0, span, SHADOW_TONE),)]
    elif where.beside:
        lines = list(WORDMARK)
        runs = [tones(line) for line in WORDMARK]
        lines[where.version], lines[where.hint] = text_rows(
            WORDMARK[where.version].ljust(BESIDE_COLUMN), WORDMARK[where.hint].ljust(BESIDE_COLUMN)
        )
        lines.append(RULE * span)
        runs.append(((0, span, SHADOW_TONE),))
    else:
        lines = [*WORDMARK, *text_rows("", ""), RULE * span]
        runs = [*(tones(line) for line in WORDMARK), (), (), ((0, span, SHADOW_TONE),)]
    if theme.name != "rich":
        lines = [line.translate(PLAIN_GLYPHS) for line in lines]
    return Banner(tuple(lines), tuple(runs))
```

Rewrite the module docstring for the new scope (both themes, the rule, beside/under/narrow, the gradient), add `from dataclasses import dataclass`, and set `__all__` to the produced names (drop `VERSION_ROW`, `HINT_ROW`, `hint_row`, `TEXT_COLUMN`).

- [ ] **Step 6: theme and widget**

`src/proofpath/tui/theme.py`: add `BANNER_TONES: tuple[str, ...] = ("g0", "g1", "g2", "g3", "g4", "shadow", "light", "dark")` with a comment (the gradient bands left to right, the shadow/rule tone, and the two the site palette names); `_RICH_BANNER = {"g0": "#e0455f", "g1": "#d02f4c", "g2": "#c4173a", "g3": "#a91330", "g4": "#8f0f2b", "shadow": "#6b0a20", "light": "#c4173a", "dark": "#8f0f2b"}` (comment: `g2` is `--crimson`, `g4` `--crimson-deep`); `PLAIN`'s `banner={tone: ui.PET_COLOUR for tone in BANNER_TONES}`. Keep the `Theme.banner` field.

`src/proofpath/tui/widgets/banner.py`: `_draw`'s overflow check becomes
```python
where = wordmark.rows(width, version=self._version, context=self._banner_context, hint=self._hint)
if len(self._drawn.lines[where.hint]) > width:
```
for **both** themes (the plain banner now overflows the same way), then re-render with the prompt only. `get_content_height` docstring: "the banner is 3, 7 or 9 rows (no mark, beside, under); a text row longer than the width wraps, so the container is sized to what was drawn." Module/class docstrings: "The wordmark, in both themes".

- [ ] **Step 7: Run the touched test files, then everything**

Run: `.venv/bin/pytest tests/test_tui_wordmark.py tests/test_tui_banner.py tests/test_tui_rich.py tests/test_tui_theme.py tests/test_tui_app.py -q 2>&1 | tail -2` then `.venv/bin/pytest -q 2>&1 | tail -1 && uv run ruff check src tests && uv run ruff format --check src tests`. Fix goldens only where `right_align`'s padding differs from your arithmetic (print the line; never bend `render` to a golden). `grep -rn "ART\b\|GROUND\|HINT_LINE\|TEXT_INDENT\|raven" src` must show nothing about the raven except the `set_busy`/`flash` "(raven design §2)" references.

- [ ] **Step 8: Look at it**

Print `render(80, RICH)`, `render(140, RICH)`, `render(80, PLAIN)` and `render(60, RICH)` with version `0.4.5`, context `academic · online · coreml`, the app's `HINT`; paste all four into the report.

- [ ] **Step 9: Hand over (no commit)**

---

### Task 2: README, main spec §13.1, raven spec status, CHANGELOG

**Files:** `README.md`, `docs/superpowers/specs/2026-09-10-proofpath-design.md` (§13.1 golden), `docs/superpowers/specs/2026-09-16-raven-pet-design.md` (status line), `CHANGELOG.md`.

- [ ] **Step 1: README.** Compute the two blocks with the implemented renderer:
  `.venv/bin/python -c "from proofpath.tui import wordmark; from proofpath.tui.theme import RICH, PLAIN; H='paste a file path, a URL, or a claim.            /help  /config  /quit'; [print('\n'.join(wordmark.render(100, t, version='0.4.5', context=c, hint=H).lines)+'\n') for t, c in ((RICH, 'academic · online · coreml'), (PLAIN, 'academic . online . coreml'))]"`
  Replace the `rich` fenced block (~lines 24–32) with the first output and the `plain` fenced block (~lines 67–74, the ASCII raven) with the second. Replace the sentence after the rich block ("The `rich` theme opens with the wordmark…; the `plain` theme opens with the raven…") with: "Both themes open with the wordmark: `rich` in block glyphs through five crimson bands, `plain` in ASCII. Under it, or beside it on a wide terminal, the version and the hint; a rule closes the banner." Fix the plain-theme paragraph (~line 62–65) so it no longer says the plain theme draws a bird ("draws the same runs as flat ASCII rows" stays). Grep the README for `raven`/`bird` afterwards; the only allowed mentions are the website's raven and the SVG screenshot sentence ("on v0.4.1, before the wordmark").
- [ ] **Step 2: Main spec §13.1.** Replace the `plain` golden banner (the six raven lines at the top of the fence, ending `____||____…`) with the plain wordmark at 80 columns for `version="0.1.0"`, `context="academic . offline . mps"`, the same hint — nine lines, computed with the renderer; the run block under it in the same fence is unchanged. Replace the sentence added earlier ("In `rich` the banner is the `ProofPath` wordmark…; the block above is the `plain` banner.") with: "Both themes open with the `ProofPath` wordmark of `2026-09-18-wordmark-banner-design.md`; the block above is the `plain` form."
- [ ] **Step 3: Raven spec.** Line 3–4 status → "**Superseded in full 2026-09-18 by `2026-09-18-wordmark-banner-design.md` (§8.1); the raven remains on the website only.**" (replace the earlier "§5 (`PLAIN`) stands" clause).
- [ ] **Step 4: CHANGELOG** `[Unreleased]` → `### Changed`:
  `- **TUI: the wordmark everywhere.** The \`plain\` theme draws the same \`ProofPath\` mark in ASCII and the raven is gone from the terminal; \`rich\` runs it through five crimson bands left to right with the shadow one tone darker; a full-width rule closes the banner in both themes; at 129 columns and up the version and hint rows sit beside the mark, narrower terminals keep them under it, and below 69 columns only the two rows and the rule are drawn (\`docs/superpowers/specs/2026-09-18-wordmark-banner-design.md\` §8).`
- [ ] **Step 5:** `uv run ruff format --check .` clean; `grep -n "raven\|bird" README.md docs/superpowers/specs/2026-09-10-proofpath-design.md | grep -v "website\|site\|v0.4.1"` prints nothing new.

---

### Task 3: The bar's frame, drawn in the wordmark's gradient (spec §9)

**Files:**
- Modify: `src/proofpath/tui/wordmark.py` (`band(column, width)`; `_tone` uses it)
- Modify: `src/proofpath/tui/widgets/prompt.py` (`PromptFrame`, `FrameEdge`, edge statics)
- Modify: `src/proofpath/tui/app.py` (`CSS`, `compose` ~line 436, `_relayout`/`_accent_prompt` ~lines 460–490)
- Modify: `tests/test_tui_rich.py` (~481–486, ~547–550, the narrow test ~657), `tests/test_tui_wordmark.py` (`band`), `tests/test_tui_app.py:703` if it reads the row's border

**Interfaces:**
- Consumes: `Theme.banner` (`g0`…`g4`), `theme.glyphs.box` (`"round"`/`"none"`), `panelled(theme, width)`, the existing `#prompt-row`/`#caret`/`Prompt` composition and `#bottom.rich[.narrow]` heights.
- Produces: `wordmark.band(column: int, width: int) -> str`; `PromptFrame(Vertical)` with id `prompt-frame`; `FrameEdge(Static)` with ids `frame-top`/`frame-bottom`; `Static`s `#edge-left` (`"│ "`) and `#edge-right` (`" │"`).

- [ ] **Step 1: Tests first**

`tests/test_tui_wordmark.py`:
```python
def test_band_splits_any_width_into_five_left_light_right_dark() -> None:
    assert wordmark.band(0, 10) == "g0" and wordmark.band(9, 10) == "g4"
    assert [wordmark.band(c, 5) for c in range(5)] == list(wordmark.GRADIENT_TONES)
    assert wordmark.band(65, 66) == "g4" and wordmark.band(0, 66) == "g0"
```
`tests/test_tui_rich.py`: replace the two `row.styles.border` assertions in `test_two_panels_wear_two_accents_and_so_does_the_prompt` with: the frame's top row plain text is `"╭" + "─" * (w - 2) + "╮"` where `w` is `#frame-top`'s `region.width`; its first span's style is `RICH.banner["g0"]` and its last span's `RICH.banner["g4"]`; `#edge-left` renders `"│ "` styled `g0`, `#edge-right` `" │"` styled `g4`; the caret still turns to `RICH.accents[1]` after the second run. In `test_a_forced_rich_theme_without_colour_still_draws_its_frames`: `#frame-top`'s text has no spans and still reads `╭…╮`. Add `test_the_prompt_keeps_its_width_inside_the_frame`: at `SIZE` the `Prompt.region.width` equals the value you measure on `main` **before** your change (measure it first: `.venv/bin/python - <<'EOF'` with `rich_app()` + `run_test(size=SIZE)` printing `app.query_one(Prompt).region.width`; write that number into the test as the expected value with a comment saying it is today's width). Narrow test (~657): the frame rows and edges have `display is False` when `#bottom` is `narrow`, and come back on widening.

- [ ] **Step 2: `wordmark.band`**
```python
def band(column: int, width: int) -> str:
    """The gradient tone of ``column`` in something ``width`` columns wide: five equal
    bands, light at the left, dark at the right. The mark uses it over its own width,
    the bar's frame over the terminal's, so the two run the same way."""
    return GRADIENT_TONES[min(BANDS - 1, column * BANDS // max(width, 1))]
```
`_tone` calls `band(column, MARK_WIDTH)`.

- [ ] **Step 3: The frame widgets** (`widgets/prompt.py`)
```python
class FrameEdge(Static):
    """One horizontal edge of the bar's frame, coloured through the wordmark's bands."""

    def __init__(self, *, top: bool, theme: Theme, coloured: bool, id: str) -> None:  # noqa: A002
        super().__init__(id=id)
        self._corners = ("╭", "╮") if top else ("╰", "╯")
        self._theme = theme
        self._coloured = coloured

    def render(self) -> Text:
        width = self.size.width or DEFAULT_WIDTH
        left, right = self._corners
        text = Text(left + "─" * max(width - 2, 0) + right)
        if self._coloured:
            for column in range(width):
                text.stylize(self._theme.banner[wordmark.band(column, width)], column, column + 1)
        return text

    def on_resize(self) -> None:
        self.refresh()
```
(Merge neighbouring columns of one band into one `stylize` call — five calls, not `width` — the test reads the first and last span.) `PromptFrame(Vertical)` composes `FrameEdge(top=True, id="frame-top")`, the existing `Horizontal(id="prompt-row")` (now with `Static("│ ", id="edge-left")` before the caret and `Static(" │", id="edge-right")` after the `Prompt`, each styled once with `theme.banner["g0"]` / `["g4"]` when coloured), `FrameEdge(top=False, id="frame-bottom")`. Move the `compose` block from `app.py` into `PromptFrame.compose` (it takes `theme`, `coloured`, `placeholder`, `history`, `complete`, `sep`), and yield `PromptFrame(...)` from the app.

- [ ] **Step 4: CSS and relayout** (`app.py`)
Replace the `#prompt-row` rules: `#prompt-frame { height: 1; margin: 0; }`, `#prompt-row { height: 1; padding: 0 1; }`, `#frame-top, #frame-bottom, #edge-left, #edge-right { display: none; }`, `#edge-left, #edge-right { width: 2; }`; under `.rich`: `#bottom.rich #prompt-frame { height: 3; margin: 0 1; }`, `#bottom.rich #prompt-row { padding: 0; }`, `#bottom.rich #frame-top, #bottom.rich #frame-bottom, #bottom.rich #edge-left, #bottom.rich #edge-right { display: block; }`; under `.rich.narrow`: `#bottom.rich.narrow #prompt-frame { height: 1; margin: 0; }`, `#bottom.rich.narrow #prompt-row { padding: 0 1; }`, and the four hidden again. Keep `#bottom.rich { height: 7 }` / `.narrow { height: 5 }`. In `_relayout`, delete the `row.styles.border = …` branches (the docstring's "the border is set here" paragraph goes with them; the class still moves the heights). `_accent_prompt` keeps colouring the caret only. Check the `Prompt` width at `SIZE` against the number from Step 1 and adjust `#edge-*` widths/padding until it matches — the input must not shrink.

- [ ] **Step 5: Run** `.venv/bin/pytest tests/test_tui_rich.py tests/test_tui_app.py tests/test_tui_wordmark.py -q 2>&1 | tail -2`, then the whole suite + ruff. Print a 100-column `rich` screen (`app.run_test(size=(100, 24))` → `app.save_screenshot`? — simpler: `pilot.app.screen._compositor.render_update`… skip; instead print `#frame-top`, `#prompt-row` children and `#frame-bottom` `.render().plain` lines in the report).

- [ ] **Step 6: Hand over (no commit)**

---

### Task 4: Slash suggestions above the bar (spec §10)

**Files:**
- Modify: `src/proofpath/tui/commands.py` (`DESCRIPTIONS`)
- Create: `src/proofpath/tui/widgets/suggestions.py`
- Modify: `src/proofpath/tui/widgets/prompt.py` (`Prompt` exposes its cycle; `↑`/`↓` defer to the list when it shows)
- Modify: `src/proofpath/tui/app.py` (compose: the list inside `#bottom` between the footer and the prompt frame; CSS: `#bottom { height: auto }` with the children's fixed heights; `on_input_changed` → feed the list)
- Test: `tests/test_tui_commands.py` (`DESCRIPTIONS` covers `VERBS` exactly), `tests/test_tui_app.py` (new section), `tests/test_tui_rich.py` (one look test)

**Interfaces:**
- Consumes: `commands.complete(text, run_ids=…)`, `commands.VERBS`, `commands.NEEDS_ARGUMENT`, `commands.ALLOW_ANSWERS`, `Prompt._candidates`/`_candidate`/`action_complete`/`_show`, `Theme.tone("muted")`, `TINT_ALPHA`, the accent the app already gives the prompt (`_accent_prompt`), `PromptFrame` from Task 3.
- Produces: `commands.DESCRIPTIONS: Mapping[str, str]`; `Suggestions(Static, id="suggestions")` with `show(candidates: Sequence[str], selected: int) -> None` and `hide() -> None`, `visible: bool` property, `rows: tuple[str, ...]` (what it last drew, for tests); `Prompt.candidates -> tuple[str, ...]` and `Prompt.selected -> int` properties; `Prompt.select(delta: int) -> None` (moves the selection without touching the bar; rebuilds the cycle from the bar when stale, like `action_complete`); the app's `Prompt.Suggest` message? — no: the app reads the prompt's properties on `Input.Changed` and after each `Tab`/`↑`/`↓` (bindings on the app forward to the prompt and then refresh the list).

- [ ] **Step 1: Tests first**

`tests/test_tui_commands.py`:
```python
def test_every_verb_has_a_short_description_and_nothing_else_does() -> None:
    assert set(commands.DESCRIPTIONS) == set(commands.VERBS)
    for verb, text in commands.DESCRIPTIONS.items():
        assert 0 < len(text) <= 40 and not text.endswith("."), verb
```
`tests/test_tui_app.py` (new section `# --- slash suggestions`), using `build_app()` and a `type_into(pilot, text)` helper that sets `prompt.value` char by char via `pilot.press` (so `Changed` fires as it would for a user):
- typing `/` shows one row per verb, in `VERBS` order, each starting with `/{verb}`; row 0 is selected; the bar still reads `/`;
- typing `/h` narrows to `/help`; `/he` still `/help`; `/x` hides the list (`visible is False`);
- `/check ` (with the trailing space the completion adds) hides the list (no candidates); `/allow ` shows the four answers; `/cancel ` with a queued run shows `/cancel #1`;
- `↓` moves the selection to row 1 without changing the bar; `↑` back; `Tab` puts the selected candidate into the bar and keeps the list (now narrowed to that candidate or its siblings — assert the bar value equals the previously selected row's completion);
- Enter with `/help` in the bar runs `/help` (the note lines appear) and the list hides (the bar is empty);
- `↑` on an empty bar still walks history (existing test must keep passing); `↓` with the list showing does not touch history (`prompt.value` unchanged);
- the list never covers the log: `RunLog.region.bottom <= Suggestions.region.y` and `Suggestions.region.bottom <= PromptFrame.region.y` while showing; `#bottom.region.height` grows by `len(rows)` and shrinks back when hidden.
`tests/test_tui_rich.py`: with `rich_app()` type `/` and assert the selected row's background alpha ≈ `TINT_ALPHA` and the other rows' background is `None`/transparent; in `plain` (`build_app()`), the selected row starts with `> `.

- [ ] **Step 2: `commands.DESCRIPTIONS`** — the spec §10 table, verbatim, with a comment saying it feeds the list above the bar and `/help`.

- [ ] **Step 3: `Suggestions` widget** — a `Static` whose `render()` builds one `Text` line per row: completion padded to the widest completion + 2, placeholder padded to the widest placeholder + 2 (omit the column entirely when every candidate's placeholder is empty), description; `plain` prefixes the selected row with `"> "` and the others with `"  "`; `rich` stylizes the selected row's background with `Color(accent).with_alpha(TINT_ALPHA)` (reuse `textual_colour` + `TINT_ALPHA` from `prompt.py`) and the completion column with the text tone, the rest muted. Height: `height: auto`; `hide()` sets `display = False`, `show()` sets it `True` and refreshes. Takes `theme` and `coloured`, and `accent: str` via `set_accent(accent)` (the app calls it from `_accent_prompt`).

- [ ] **Step 4: `Prompt` and the app** — `Prompt.candidates`/`selected` read the cycle (`_candidates`/`_candidate`, rebuilding from `complete(self.value)` when the bar no longer holds the current candidate, exactly as `action_complete` does before it steps); `Prompt.select(delta)` moves `_candidate` modulo the count. Bindings: keep `up`/`down` on the prompt but make `action_history_previous`/`_next` call `self.select(∓1)` and post a `Prompt.SuggestionsChanged` message instead of walking history when `self.value.startswith("/")` and `self.candidates` is non-empty. The app handles `on_input_changed` (already exists) and `on_prompt_suggestions_changed` by calling `_refresh_suggestions()`: `candidates = prompt.candidates` (with `run_ids` from the scheduler for `/cancel`); `show(candidates, prompt.selected)` when the bar starts with `/` and candidates exist, else `hide()`. `action_complete` posts the same message after it steps. CSS: `#bottom { height: auto; }`, `#suggestions { height: auto; padding: 0 1; display: none; }`; remove the fixed `#bottom` heights (`4`/`7`/`5`) — the footer and the frame keep their own fixed heights, so the sum is unchanged when the list is hidden (assert this in the layout test: with the list hidden `#bottom.region.height` is 4 in plain and 7 in rich at `SIZE`, 5 when narrow).

- [ ] **Step 5: Run** `tests/test_tui_commands.py tests/test_tui_app.py tests/test_tui_rich.py`, then the whole suite + ruff. Check the existing history and completion tests still pass unchanged (`-k "history or complet"`).

- [ ] **Step 6: Hand over (no commit)**

---

### Task 5: Typing always reaches the bar (spec §11)

**Files:**
- Modify: `src/proofpath/tui/app.py` (`RunLog.can_focus = False`; new `on_key` on `ProofpathApp`)
- Modify: `src/proofpath/tui/widgets/finding.py` (remove `Line.on_key`; keep its docstring's reasoning in the app handler)
- Test: `tests/test_tui_app.py` (new section `# --- typing reaches the bar`)

**Interfaces:**
- Consumes: `Prompt` (`#prompt`), `Line`, `PermissionPrompt`'s `Button`s, `tevents.Key`, `Screen.set_focus`.
- Produces: `ProofpathApp.on_key(event: tevents.Key) -> None` forwarding printable keys to the bar; `RunLog.can_focus = False`.

- [ ] **Step 1: Tests first** (`tests/test_tui_app.py`, `build_app()`):
  - `test_a_click_on_the_log_background_keeps_the_bar_focused`: at `SIZE`, `await pilot.click(RunLog, offset=(10, 5))` on an empty log → `app.focused is app.query_one(Prompt)`; then `await pilot.press("x")` → `prompt.value == "x"`.
  - `test_a_click_on_the_banner_keeps_the_bar_focused`: `await pilot.click(Banner)` → same.
  - `test_typing_after_clicking_a_finding_goes_to_the_bar`: start a run, land a finding (use the existing fixtures/helpers in this file that mount a `FindingLine`), `await pilot.click(FindingLine)` → `app.focused` is the line; `await pilot.press("h")` → `prompt.value == "h"` and `app.focused is prompt`; `await pilot.click(FindingLine)`; `await pilot.press("c")` → the bar is unchanged (still `"h"`) and the line's copy ran (assert via the existing clipboard/copy test pattern in this file, or that `app.focused` is still the line).
  - `test_typing_on_a_permission_button_goes_to_the_bar`: mount a permission prompt the way the existing permission tests do, `app.query_one(Button).focus()`, `await pilot.press("y")` → `prompt.value == "y"`, focused is the prompt; `enter`/`space` on the button are not forwarded (the existing button-answer test keeps passing).
  - `test_the_log_is_not_in_the_focus_chain`: `RunLog` not in `app.screen.focus_chain`; `Tab` from the bar with one finding on screen focuses the line, `shift+tab` returns to the bar (existing behaviour; assert it still holds).
- [ ] **Step 2: `RunLog.can_focus = False`** with a comment: keyboard scrolling is the app's priority bindings; a focusable log would swallow typing after a click on its background.
- [ ] **Step 3: App `on_key`** (place it near the other key/binding handlers):
```python
    def on_key(self, event: tevents.Key) -> None:
        """A printable key anywhere is typing, and typing belongs to the bar (spec §11).

        ``c`` on a focused line is that line's copy (spec §13.1) and stays with it. The
        key is stopped and posted to the bar afresh rather than inserted: focus is set
        on the screen directly -- ``Widget.focus`` defers, and the key would overtake
        the ``Focus`` -- and the key is posted after it, so the bar types it the way it
        types every other one.
        """
        focused = self.focused
        prompt = self.query_one(Prompt)
        if focused is prompt or not event.is_printable or event.character is None:
            return
        if isinstance(focused, Line) and event.key == "c":
            return
        event.stop()
        event.prevent_default()
        self.screen.set_focus(prompt)
        prompt.post_message(tevents.Key(event.key, event.character))
```
  Delete `Line.on_key` in `finding.py` (and its now-unused `Input` import if any); keep a one-line comment where it was pointing to the app handler. Confirm `Line`'s `c` binding still fires (the existing copy test).
- [ ] **Step 4: Run** `tests/test_tui_app.py tests/test_tui_rich.py -q`, then the whole suite + ruff.
- [ ] **Step 5: Hand over (no commit)**

---

### Task 6: `/config` opens a settings panel (spec §12)

**Files:**
- Create: `src/proofpath/tui/widgets/config_panel.py`
- Modify: `src/proofpath/tui/app.py` (`_dispatch`: `/config` with an empty argument mounts the panel; handlers for the panel's messages)
- Modify: `src/proofpath/tui/commands.py` (`DESCRIPTIONS["config"]` → `"settings panel, or show / set / check"`)
- Modify: `src/proofpath/judge.py` (export `known_providers() -> tuple[str, ...]` = `tuple(sorted(_PROVIDERS))`; one line)
- Test: `tests/test_tui_app.py` (new section `# --- the config panel`), `tests/test_tui_rich.py` (one look test), `tests/test_judge.py` (`known_providers`)

**Interfaces:**
- Consumes: `proofpath.commands.config_view() -> ConfigView(path, exists, config, toml)`, `proofpath.commands.config_set(key, value) -> tuple[tuple[str, str], ...]` (four pairs for `judge.provider`), `config.PERMISSION_VALUES`, `ConfigError`, `JudgeError`, `judge.known_providers()`, `secrets.resolve_api_key(env_name)` / `judge.default_dotenv_paths()` for the `key` line, `Theme` (`badge`, `tone("muted")`, `glyphs.prompt`, `glyphs.box`), the app's `_reload_config()`, `_note()`, the error-line helper the mirrored verbs use, and `Banner.set_context(run_context(config))`; Task 5's forwarding (a printable key while the panel has focus goes to the bar; the panel sees `Blur`).
- Produces: `ConfigPanel(Vertical)` with `can_focus = True`; `BINDINGS`: `up`/`down` → `move(∓1)`, `left`/`right` → `cycle(∓1)`, `enter` → `activate` (cycle forward on a choice row, open the edit on a text row), `backspace` → `reset`, `escape` → `close` (or cancel an open edit); messages `ConfigPanel.Written(pairs: tuple[tuple[str, str], ...], path: Path)`, `ConfigPanel.Failed(message: str)`, `ConfigPanel.Closed()`; `SECTIONS: tuple[Section, ...]` with `Section(name, rows)`, `Row(key, kind: Literal["choice", "text"], values: tuple[str, ...], meaning: str, default: str)`; `panel.selected: int`, `panel.value(key) -> str`, `panel.collapsed: bool`, `panel.editing: bool`, `panel.rows: tuple[Row, ...]` (flattened, selectable order).

- [ ] **Step 1: Tests first** (`build_app()`; the autouse `isolated` fixture points `PROOFPATH_CONFIG_DIR` at `tmp_path`; `monkeypatch.delenv("GROQ_API_KEY", raising=False)` unless a test sets it):
  - **opens and reads**: `/config` mounts one `ConfigPanel` in the log, focused, not collapsed; the head has `config` + the path + `(not written yet, showing defaults)` and a `key` line containing `GROQ_API_KEY` and `not set`; with `monkeypatch.setenv("GROQ_API_KEY", "secret-for-test")` the `key` line says `found in the environment` and `"secret-for-test"` is nowhere in the panel's rendered text.
  - **row order**: `panel.rows` keys are `permissions.install_browser, permissions.network, fetch.respect_robots, judge.provider, judge.model, judge.base_url, judge.api_key_env, contact.email`; kinds `choice ×4, text ×4`; row 0 selected; `↓` ×8 stops at the last row; `↑` ×9 stops at 0.
  - **choice rows write at once**: `→` on row 0 → `load_config()` (tmp path) has `install_browser == "allow"`, `app._config.permissions.install_browser == "allow"`, a note `permissions.install_browser = allow  (<path>)` is in the log; `→ →` wraps `deny → ask`; `←` from `ask` gives `deny`; `Enter` on a choice row equals `→`.
  - **network drives the banner**: select row 1, `→ →` to `deny` → `Banner` context text contains `offline`; `→` to `ask` → `online` again (whatever `run_context` prints for `ask`; assert against `run_context(app._config)`).
  - **bool round-trips**: row 2 `→` → `respect_robots is False` on disk.
  - **provider switch writes four**: row 3 `→` → provider `gemini`; four notes (`judge.provider`, `judge.model`, `judge.base_url`, `judge.api_key_env`) and the text rows show gemini's presets; the `key` line now names gemini's `api_key_env`.
  - **text edit**: select `judge.model`, `Enter` → `panel.editing` and an `Input` inside the panel holds the current model; type `-x`, `Enter` → disk has the new model, note printed, `panel.editing is False`; `Enter`, type `zzz`, `Esc` → value unchanged on disk and in the row.
  - **reset**: on `permissions.install_browser` after setting `deny`, `Backspace` → `ask` on disk and a note; on `judge.provider` after `gemini`, `Backspace` → `groq` and four notes.
  - **errors**: monkeypatch `proofpath.tui.widgets.config_panel.config_set` to raise `ConfigError("boom")`; `→` leaves the row and the disk unchanged and an error line `boom` appears under the panel.
  - **closing**: `Esc` (not editing) → `panel.collapsed`, the summary line reads `config  install_browser=… network=… respect_robots=… judge=…`, the bar is focused; a fresh panel + `x` typed → `x` in the bar and the panel collapsed (§11 + `on_blur`); `/config` twice → two panels; `/config show` still prints the TOML (existing test).
  - **legend**: the panel's last line contains `esc close`; in `plain` it contains `up/down`.
  `tests/test_tui_rich.py`: after `/config`, the selected row starts with `›`, the current value of row 0 is drawn as a badge (follow the findings' badge assertion pattern), and the panel border is `round` in the run accent.
- [ ] **Step 2: the model of the panel** (`config_panel.py`):
```python
Kind = Literal["choice", "text"]


@dataclass(frozen=True)
class Row:
    key: str
    kind: Kind
    values: tuple[str, ...]  # the choices; empty for a text row
    meaning: str  # one dim line under a choice row; "" for none
    default: str


@dataclass(frozen=True)
class Section:
    name: str
    rows: tuple[Row, ...]


def sections() -> tuple[Section, ...]:
    """The panel's rows in the config's own order. Built, not constant, because the
    provider list is the judge module's."""
    return (
        Section(
            "permissions",
            (
                Row(
                    "permissions.install_browser",
                    "choice",
                    PERMISSION_VALUES,
                    "step 3 of the fetch ladder: a ~280 MB browser engine, spec §7.1",
                    "ask",
                ),
                Row(
                    "permissions.network",
                    "choice",
                    PERMISSION_VALUES,
                    "every fetch and every provider call; deny = offline",
                    "allow",
                ),
            ),
        ),
        Section(
            "fetch",
            (
                Row(
                    "fetch.respect_robots",
                    "choice",
                    ("true", "false"),
                    "honour robots.txt when fetching",
                    "true",
                ),
            ),
        ),
        Section(
            "judge",
            (
                Row(
                    "judge.provider",
                    "choice",
                    known_providers(),
                    "switching a provider also sets its model, base_url and api_key_env",
                    "groq",
                ),
                Row("judge.model", "text", (), "", JudgeConfig().model),
                Row("judge.base_url", "text", (), "", JudgeConfig().base_url),
                Row("judge.api_key_env", "text", (), "", JudgeConfig().api_key_env),
            ),
        ),
        Section(
            "contact",
            (
                Row(
                    "contact.email",
                    "text",
                    (),
                    "optional, for the Crossref / OpenAlex polite pools",
                    "",
                ),
            ),
        ),
    )
```
  Defaults come from the dataclasses (`Permissions()`, `FetchConfig()`, `JudgeConfig()`, `Contact()`), not literals — derive them in `sections()` so a changed default cannot drift. Current values: `str(getattr(getattr(config, section), key))` with bools lowered.
- [ ] **Step 3: the widget.** `ConfigPanel(Vertical)`, `can_focus = True`, built from `Static` lines: head (2 lines), then per section a heading line and per row a row line (+ meaning line for choices), then the legend; a hidden `Static` for the collapsed summary; an `Input` mounted in place of the row line while editing (`editing` true; the panel's own bindings are inert while it is open except `escape` → cancel). `move`, `cycle`, `activate`, `reset`, `close` per the interfaces. Writes go through a module-level `config_set` name (imported from `proofpath.commands`) so a test can monkeypatch it; after a successful write, re-read `config_view()` to redraw every row (a provider switch changes four) and the `key` line (`resolve_api_key(env_name, dotenv_paths=default_dotenv_paths())` → `found in <source>` / `not set — put <ENV>=… in <first dotenv path>`). Draw with the theme: section names and meanings muted; the current value as a badge in the panel's accent when `theme.badge`, `[value]` otherwise; the selected row's marker `theme.glyphs.prompt` at column 1; the legend uses arrows in `rich`, words in `plain`. Border: `(theme.glyphs.box, accent)` with `border_title = "/config"` exactly as `CommandBlock` does it (read `run_block.py`). `on_blur` → `close()` unless editing or collapsed.
- [ ] **Step 4: the app.** In `_dispatch`, before the `MIRRORED` branch: `if command.verb == "config" and not command.arg: self._open_config_panel(); return`. `_open_config_panel` builds the panel with the prompt's current accent, mounts it in the `RunLog` like a `CommandBlock`, scrolls, focuses it. Handlers: `on_config_panel_written` → `_reload_config()`, `self.query_one(Banner).set_context(run_context(self._config))`, one `_note(f"{key} = {value}  ({path})", dim=False)` per pair; `on_config_panel_failed` → the mirrored verbs' error line; `on_config_panel_closed` → `self.query_one(Prompt).focus()`. `commands.DESCRIPTIONS["config"]` → `"settings panel, or show / set / check"`. `judge.known_providers()` → `tuple(sorted(_PROVIDERS))`.
- [ ] **Step 5: Run** `tests/test_tui_app.py tests/test_tui_rich.py tests/test_tui_commands.py tests/test_judge.py -q`, then the whole suite + ruff.
- [ ] **Step 6: Hand over (no commit)**
