"""The RICH theme on screen (TUI v2 design, sections 3, 4 and 7).

Every test injects ``theme=RICH`` through ``ProofpathApp`` and drives the app the way
``test_tui_app.py`` does, with the same fake scheduler, so nothing here depends on
the terminal the suite runs in. The PLAIN goldens stay where they are: this file is
the proof that the second look draws the same truth.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from rich.cells import cell_len
from textual.content import Content

from proofpath import __version__, ui, verify
from proofpath.document import Claim, Locator, Reference
from proofpath.events import Emitted, Note, Progress, StageEnd, StageStart
from proofpath.tui import banner, wordmark
from proofpath.tui.app import (
    Banner,
    CoverageFooter,
    FindingLine,
    ProofpathApp,
    RunBlock,
    RunHeader,
    StageLine,
)
from proofpath.tui.theme import PLAIN, RICH
from proofpath.tui.widgets import PANEL_FLOOR, CoverageLine
from proofpath.tui.widgets.run_block import NoteLine
from tests.test_tui_app import (
    FakeScheduler,
    a_finding,
    a_report,
    a_verdict_finding,
    build_app,
    submit,
)

SIZE = (100, 34)
#: The run's accent at ``SIZE``: run #1 takes the first of the five.
ACCENT = RICH.accents[0]


def rich_app(**kwargs: Any) -> tuple[ProofpathApp, list[FakeScheduler]]:
    return build_app(theme=RICH, **kwargs)


async def a_run(pilot: Any, schedulers: list[FakeScheduler], target: str = "draft.md") -> Any:
    """Submit one ``/check`` and hand back the run the fake scheduler made for it."""
    await submit(pilot, f"/check {target}")
    return schedulers[0].runs[0]


def rows_of(text: str) -> list[str]:
    return text.split("\n")


# --- the default theme -------------------------------------------------------------


def test_an_app_built_by_hand_is_plain_whatever_the_terminal_says() -> None:
    """What a test sees never depends on the terminal it runs in (design section 6)."""
    app, _ = build_app()
    assert app._theme is PLAIN
    assert rich_app()[0]._theme is RICH


# --- the stage table (design section 4) --------------------------------------------


async def test_the_stage_table_has_fixed_columns_and_dot_separators() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.move(run, "running")
        scheduler.push(run, StageStart("Parsing", "pymupdf"))
        scheduler.push(run, StageEnd("Parsing", "pymupdf", "24 pages, 42 refs", 1.2))
        scheduler.push(run, StageStart("Resolving", "Crossref, S2"))
        scheduler.push(run, StageEnd("Resolving", "Crossref, S2", "38 ok, 3 amb, 1 ghost", 3.4))
        scheduler.push(run, StageStart("Verifying", "coreml"))
        await pilot.pause()
        lines = [line.render().plain for line in app.query(StageLine)]
        width = app.query_one(StageLine).size.width
    # 100 columns, less the log's padding, the border and the panel's padding.
    assert width == 94
    assert lines == [
        "✓ Parsing       24 pages · 42 refs                                             pymupdf    1.2s",  # noqa: E501 - a 94-column golden, verbatim
        "⏺ Resolving     38 ok · 3 amb · 1 ghost                                   Crossref, S2    3.4s",  # noqa: E501 - a 94-column golden, verbatim
        "⏺ Verifying                                                                             coreml",  # noqa: E501 - a 94-column golden, verbatim
    ]


async def test_a_stage_that_left_something_unverified_keeps_the_active_mark() -> None:
    """Rule 2 on screen: a tick would say the stage settled everything it touched."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        for name, summary in [
            ("Parsing", "24 pages, 42 refs"),
            ("Resolving", "38 ok, 0 amb, 0 ghost"),
            ("Retractions", "1 retracted"),
            ("Fetching", "22 full text, 11 abstract, 9 unverified"),
            ("Verifying", "118 claims: 40 supported, 6 not supported, 72 NEI"),
        ]:
            scheduler.push(run, StageStart(name, "x"))
            scheduler.push(run, StageEnd(name, "x", summary, 1.0))
        await pilot.pause()
        marks = [line.render().plain[0] for line in app.query(StageLine)]
        styles = [str(line.render().spans[0].style) for line in app.query(StageLine)]
    # 72 NEI claims are claims the stage could not settle: caution, never a tick.
    assert marks == ["✓", "✓", "⏺", "⏺", "⏺"]
    assert styles == [
        RICH.tone("ok"),
        RICH.tone("ok"),
        RICH.tone("caution"),
        RICH.tone("caution"),
        RICH.tone("caution"),
    ]


@pytest.mark.parametrize(
    "summary",
    [
        verify.NOT_ATTEMPTED,
        "118 citations, 3 unresolved",
        "38 ok, 0 amb, 0 ghost, 2 unavailable",
        "38 ok, 0 amb, 0 ghost, 1 not indexed",
        "0 retracted, 2 unavailable",
    ],
)
async def test_a_stage_that_was_not_run_or_left_a_marker_never_gets_a_tick(summary: str) -> None:
    """``not attempted`` and ``N unresolved`` are the engine's own words for "not checked"."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, StageStart("Resolving", "x"))
        schedulers[0].push(run, StageEnd("Resolving", "x", summary, 0.1))
        await pilot.pause()
        rendered = app.query_one(StageLine).render()
    assert rendered.plain[0] == RICH.glyphs.stage_flag
    assert str(rendered.spans[0].style) == RICH.tone("caution")


async def test_the_active_stage_draws_a_real_progress_bar() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.move(run, "verifying")
        scheduler.push(run, StageStart("Verifying", "coreml"))
        seen: list[str] = []
        for done in (0, 51, 118):
            scheduler.push(run, Progress("Verifying", done, 118, "p.4 L112"))
            await pilot.pause()
            seen.append(app.query_one(StageLine).render().plain)
        bar = app.query_one(StageLine).render()
        # The bar is the run's accent, as the border is; the count is plain.
        assert (str(bar.spans[1].style), bar.plain[bar.spans[1].start : bar.spans[1].end]) == (
            ACCENT,
            "▰" * 18,
        )
        # The PLAIN header is not on screen in RICH, and the panel carries no bar of
        # its own: the stage row is the one place the progress is drawn.
        assert not app.query_one(RunHeader).display
    assert seen == [
        "⏺ Verifying     ▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱  0/118                                               coreml",  # noqa: E501 - a 94-column golden, verbatim
        "⏺ Verifying     ▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱  51/118                                              coreml",  # noqa: E501 - a 94-column golden, verbatim
        "⏺ Verifying     ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰  118/118                                             coreml",  # noqa: E501 - a 94-column golden, verbatim
    ]


async def test_the_bar_goes_when_the_stage_ends_and_the_summary_takes_its_cell() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Progress("Verifying", 51, 118, ""))
        scheduler.push(run, StageEnd("Verifying", "coreml", "118 claims: 40 supported", 21.4))
        await pilot.pause()
        text = app.query_one(StageLine).render().plain
    assert "▰" not in text and "▱" not in text
    assert text.startswith("✓ Verifying     118 claims: 40 supported")
    assert text.endswith("coreml   21.4s")


def _note_texts(app: ProofpathApp) -> list[str]:
    return [line.render().plain for line in app.query(NoteLine)]


async def test_a_transient_note_leaves_when_its_stage_ends() -> None:
    """``loading models …`` is over when the stage is: a finished run must not say it."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Note("kept"))
        scheduler.push(run, Note(verify.LOADING_MODELS, transient=True))
        await pilot.pause()
        texts = _note_texts(app)
        assert any("loading models" in t for t in texts)
        assert any("kept" in t for t in texts)
        scheduler.push(run, StageEnd("Verifying", "coreml", "1 claim", 0.1))
        await pilot.pause()
        texts = _note_texts(app)
    assert not any("loading models" in t for t in texts)
    assert any("kept" in t for t in texts)


async def test_a_lasting_note_takes_a_transient_one_back() -> None:
    """A note that is not about progress means the progress it followed is over."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Note(verify.LOADING_MODELS, transient=True))
        await pilot.pause()
        assert any("loading models" in t for t in _note_texts(app))
        scheduler.push(run, Note("kept"))
        await pilot.pause()
        texts = _note_texts(app)
    assert not any("loading models" in t for t in texts)
    assert any("kept" in t for t in texts)


async def test_a_transient_note_survives_progress_but_not_a_finding() -> None:
    """Progress can tick while a model is still downloading; a finding means it is in."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Note(verify.LOADING_MODELS, transient=True))
        scheduler.push(run, Progress("Verifying", 1, 118, ""))
        await pilot.pause()
        assert any("loading models" in t for t in _note_texts(app))
        scheduler.push(run, Emitted(a_verdict_finding()))
        await pilot.pause()
        texts = _note_texts(app)
    assert not any("loading models" in t for t in texts)


async def test_a_transient_note_is_taken_back_when_the_run_settles() -> None:
    """A run that fails while loading must not keep saying it is loading."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Note(verify.LOADING_MODELS, transient=True))
        await pilot.pause()
        assert any("loading models" in t for t in _note_texts(app))
        run.error = "RuntimeError: no model"
        scheduler.move(run, "failed")
        await pilot.pause()
        texts = _note_texts(app)
    assert not any("loading models" in t for t in texts)
    assert any("no model" in t for t in texts)


# --- findings and badges -----------------------------------------------------------


async def test_a_finding_row_carries_a_badge_and_its_notes() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, Emitted(a_finding()))
        await pilot.pause()
        rendered = app.query_one(FindingLine).render()
    assert rows_of(rendered.plain) == [
        "✗ p.4 L112      [12] Zhang et al. 2021                                GHOST REFERENCE         ",  # noqa: E501 - a 94-column golden, verbatim
        "       DOI 10.1016/j.xxxx.2021.99999 resolves to nothing",
    ]
    spans = {rendered.plain[span.start : span.end]: str(span.style) for span in rendered.spans}
    assert spans[" GHOST REFERENCE "] == RICH.badge_style("GHOST REFERENCE")
    assert spans["✗"] == RICH.tones[ui.LEVEL_STYLES["error"]]


async def test_a_verdict_shows_what_you_said_and_what_the_source_said() -> None:
    """Rule 1: the passage is on screen, and in RICH the claim it was checked against."""
    item = replace(
        a_verdict_finding(),
        reference=Reference(number=31, raw="Kumar 2022", locator=Locator(line=800)),
    )
    item = replace(
        item,
        claim=Claim(
            text="the method yields a 40% speedup",
            locator=Locator(line=260, page=9),
            cited_refs=(31,),
            paragraph=3,
            sentence=1,
        ),
    )
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, Emitted(item))
        await pilot.pause()
        line = app.query_one(FindingLine)
        rendered = line.render()
    assert rows_of(rendered.plain) == [
        "✗ p.9 L260      [31] Kumar 2022                                         NOT SUPPORTED   high  ",  # noqa: E501 - a 94-column golden, verbatim
        '       you     "the method yields a 40% speedup"',
        '       source  "we observed a 4-8% improvement in throughput" ⧉',
    ]
    assert line._copy_cell == (2, 15 + len('"we observed a 4-8% improvement in throughput"') + 1)


@pytest.mark.parametrize("tier", ["high", "medium", None])
async def test_every_badge_ends_on_the_same_column_whatever_the_tier(tier: str | None) -> None:
    # A ghost carries no tier at all; a verdict carries the one it was given.
    item = (
        replace(a_finding(), tier=None) if tier is None else replace(a_verdict_finding(), tier=tier)
    )
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, Emitted(item))
        await pilot.pause()
        line = app.query_one(FindingLine)
        row = rows_of(line.render().plain)[0]
        width = line.size.width
    badge = f" {item.state} "
    # Two spaces and a six-wide tier cell after the badge, on every row.
    assert row.index(badge) + len(badge) == width - 8
    assert row.endswith(f"  {(tier or '').ljust(6)}")


async def test_notes_elide_until_opened_and_wrap_after() -> None:
    long_note = "the DOI resolves to a landing page whose title, authors and year all differ " * 2
    item = replace(a_finding(), detail=(long_note.strip(),))
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, Emitted(item))
        await pilot.pause()
        line = app.query_one(FindingLine)
        folded = rows_of(line.render().plain)
        line.focus()
        await pilot.press("enter")
        await pilot.pause()
        opened = rows_of(line.render().plain)
        width = line.size.width
    assert len(folded) == 2 and folded[1].endswith("…")
    assert len(opened) > 2 and not any(row.endswith("…") for row in opened)
    assert " ".join(row.strip() for row in opened[1:]) == long_note.strip()
    assert all(len(row) <= width for row in opened)


async def test_the_copy_glyph_click_copies_in_rich_too() -> None:
    copied: list[str] = []
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, Emitted(a_verdict_finding()))
        await pilot.pause()
        line = app.query_one(FindingLine)
        row, column = line._copy_cell
        await pilot.click(FindingLine, offset=(column, row))
        await pilot.pause()
    assert copied == ["we observed a 4-8% improvement in throughput"]
    assert not line.expanded


# --- the panel and its border -------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "style"),
    [
        ("queued", RICH.tone("muted")),
        ("running", ACCENT),
        ("verifying", ACCENT),
        ("done", RICH.tone("ok")),
        ("cancelled", RICH.tone("muted")),
        ("failed", RICH.tone("finding")),
    ],
)
async def test_every_run_state_is_drawn_on_the_border(state: str, style: str) -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers, "~/Desktop/paper.pdf")
        if state != "queued":
            schedulers[0].move(run, state, report=a_report() if state == "done" else None)  # type: ignore[arg-type]
        await pilot.pause()
        block = app.query_one(RunBlock)
        title = block.border_title or ""
        subtitle = block.border_subtitle or ""
        capacity = block.content_region.width - 2
        assert block.styles.border.top[0] == "round"
        assert block.styles.border.top[1].hex.lower() == ACCENT
    # The command, a fill in the border's colour, the state word in its meaning: padded
    # to the frame, so the word ends where the border's own dash begins (design 4).
    plain = Content.from_markup(title).plain
    assert plain.startswith("#1  /check ~/Desktop/paper.pdf ─")
    assert plain.endswith(f" {state}")
    assert len(plain) == capacity == 92
    assert f"[{style}]{state}[/" in title
    # The bottom border carries the coverage once there is one, and nothing before.
    if state == "done":
        assert Content.from_markup(subtitle).plain == "coverage 62/21/17%"
        assert f"[{RICH.tone('ok')}]62[/" in subtitle
    else:
        assert subtitle == ""


async def test_a_title_with_brackets_is_not_read_as_markup() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        await a_run(pilot, schedulers, "notes [v2].md")
        await pilot.pause()
        title = app.query_one(RunBlock).border_title or ""
    assert title.startswith(r"#1  /check notes \[v2].md ─")
    assert Content.from_markup(title).plain.startswith("#1  /check notes [v2].md ─")


async def test_a_long_command_elides_before_the_state_word_does() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=(70, 24)) as pilot:
        await a_run(pilot, schedulers, "~/a/very/long/path/to/some/deeply/nested/paper-final.pdf")
        await pilot.pause()
        block = app.query_one(RunBlock)
        plain = Content.from_markup(block.border_title or "").plain
        assert len(plain) == block.content_region.width - 2
    assert plain.endswith(" queued")
    assert "…" in plain


async def test_a_failed_run_prints_its_error_inside_the_panel() -> None:
    """The PLAIN header carried the error; the panel has no header, so a line does."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        run.error = "OSError: no such file"
        schedulers[0].move(run, "failed")
        await pilot.pause()
        block = app.query_one(RunBlock)
        texts = [line.render().plain for line in block.query(".stages > *")]
        styles = [str(line.render().style) for line in block.query(".stages > *")]
    assert "  OSError: no such file" in texts
    assert RICH.tone("finding") in styles


async def test_a_click_on_the_top_border_folds_the_run_and_keeps_its_coverage() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, StageStart("Parsing", "pymupdf"))
        schedulers[0].move(run, "done", report=a_report())
        await pilot.pause()
        block = app.query_one(RunBlock)
        await pilot.click(RunBlock, offset=(4, 0))
        await pilot.pause()
        assert block.collapsed
        assert not block.query_one(".stages").display
        # Rule 6: the title, the state and the coverage never fold away.
        assert "done" in (block.border_title or "")
        assert "coverage" in (block.border_subtitle or "")
        # The frame carries it, not a row: the row is kept for a narrowing, hidden.
        assert not block.query_one(CoverageLine).display
        await pilot.click(RunBlock, offset=(4, 0))
        await pilot.pause()
        assert not block.collapsed


async def test_two_panels_wear_two_accents_and_so_does_the_prompt() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.md")
        row = app.query_one("#prompt-row")
        assert row.styles.border.top[1].hex.lower() == RICH.accents[0]
        await submit(pilot, "/check two.md")
        blocks = list(app.query(RunBlock))
        assert [block.accent for block in blocks] == [RICH.accents[0], RICH.accents[1]]
        assert row.styles.border.top[1].hex.lower() == RICH.accents[1]
        assert app.query_one("#caret").render().plain == RICH.glyphs.prompt  # type: ignore[attr-defined]
    assert len(schedulers[0].submitted) == 2


# --- the footer ----------------------------------------------------------------------


async def test_the_coverage_bar_is_proportional_and_coloured_by_meaning() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].move(run, "done", report=a_report())
        await pilot.pause()
        footer = app.query_one(CoverageFooter)
        rendered = footer.render()
        assert footer.region.height == 4
    assert rows_of(rendered.plain) == [
        "  ███████████████████▓▓▓▓▓▓░░░░░  █ 62% full text · ▓ 21% abstract · ░ 17% unverified",
        "  42 refs: 42 ok",
        "  no report written · 0 API calls · 38.0s",
        "  ",
    ]
    coloured = [(rendered.plain[s.start : s.end], str(s.style)) for s in rendered.spans]
    assert coloured[:6] == [
        ("█" * 19, RICH.tone("ok")),
        ("▓" * 6, RICH.tone("caution")),
        ("░" * 5, RICH.tone("finding")),
        ("█ 62%", RICH.tone("ok")),
        ("▓ 21%", RICH.tone("caution")),
        ("░ 17%", RICH.tone("finding")),
    ]


async def test_a_run_that_counted_nothing_draws_an_empty_track() -> None:
    """0/0/0 is not a share; painting the whole bar one colour would claim one."""
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].move(run, "done", report=a_report(fulltext=0, abstract=0, unverified=0))
        await pilot.pause()
        rows = rows_of(app.query_one(CoverageFooter).render().plain)
    assert rows[0] == "  " + "─" * 30 + "  █ 0% full text · ▓ 0% abstract · ░ 0% unverified"
    assert len(rows) == 4


async def test_the_footer_before_any_run_still_fills_its_four_rows() -> None:
    app, _ = rich_app()
    async with app.run_test(size=SIZE):
        text = app.query_one(CoverageFooter).render().plain
    assert text.count("\n") == 3
    assert "no run yet" in text


async def test_a_forced_rich_theme_without_colour_still_draws_its_frames() -> None:
    """A border in the background colour would be three empty rows, not a prompt box."""
    app, schedulers = rich_app(out=ui.build(force_terminal=True, no_color=True))
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].move(run, "done", report=a_report())
        await pilot.pause()
        row = app.query_one("#prompt-row")
        block = app.query_one(RunBlock)
        assert row.styles.border.top[0] == "round"
        assert row.styles.border.top[1].ansi == -1  # the terminal's default colour
        assert block.styles.border.top[1].ansi == -1
        assert Content.from_markup(block.border_title or "").plain.endswith(" done")
        assert "[" not in (block.border_subtitle or "")  # no markup without colour
        assert block.border_subtitle == "coverage 62/21/17%"


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


# --- widths ----------------------------------------------------------------------------


@pytest.mark.parametrize("width", [100, 80, 60, 46])
async def test_no_row_is_wider_than_the_terminal_and_nothing_is_clipped(width: int) -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=(width, 40)) as pilot:
        run = await a_run(pilot, schedulers, "~/Desktop/paper.pdf")
        scheduler = schedulers[0]
        scheduler.move(run, "running")
        scheduler.push(run, StageStart("Resolve references", "Crossref, OpenAlex"))
        scheduler.push(
            run,
            StageEnd(
                "Resolve references",
                "Crossref, OpenAlex",
                "38 resolved, 3 ambiguous, 1 ghost, 0 retracted",
                4.4,
            ),
        )
        scheduler.push(run, Emitted(a_verdict_finding()))
        scheduler.push(run, Emitted(a_finding()))
        scheduler.move(run, "done", report=a_report(fulltext=8, abstract=4, unverified=30))
        await pilot.pause()
        block = app.query_one(RunBlock)
        assert block.panelled == (width >= PANEL_FLOOR)
        assert app.query_one(RunHeader).display == (width < PANEL_FLOOR)
        widgets = [*app.query(StageLine), *app.query(FindingLine), app.query_one(CoverageFooter)]
        for widget in widgets:
            assert widget.region.right <= width
            for row in rows_of(widget.render().plain):
                assert cell_len(row) <= widget.size.width, (width, row)
        stage = app.query_one(StageLine).render().plain
        assert "Crossref, OpenAlex" in stage and "4.4s" in stage
        findings = [line.render().plain for line in app.query(FindingLine)]
        assert "NOT SUPPORTED" in findings[0] and "high" in findings[0]
        assert "GHOST REFERENCE" in findings[1]
        footer = app.query_one(CoverageFooter).render().plain
        # The bar, the legend alone, or the compact cell: the numbers are always there.
        assert "19%" in footer or "19/10/71%" in footer
        assert ui.WEAK_COVERAGE.split(":")[0] in footer
        for line in app.query_one(Banner).drawn.lines:  # type: ignore[union-attr]
            assert cell_len(line) == len(line)


async def test_narrow_terminals_get_the_flat_rows_and_the_panel_comes_back_on_resize() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=(50, 30)) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, StageStart("Parsing", "pymupdf"))
        schedulers[0].push(run, StageEnd("Parsing", "pymupdf", "24 pages, 42 refs", 1.2))
        await pilot.pause()
        block = app.query_one(RunBlock)
        assert not block.panelled
        assert app.query_one(RunHeader).display
        assert app.query_one("#bottom").has_class("narrow")
        # Today's two-line stage row, with the theme's own glyphs.
        assert rows_of(app.query_one(StageLine).render().plain)[0].startswith("✓ Parsing")
        assert "24 pages, 42 refs" in rows_of(app.query_one(StageLine).render().plain)[1]
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert block.panelled
        assert not app.query_one(RunHeader).display
        assert not app.query_one("#bottom").has_class("narrow")
        assert "24 pages · 42 refs" in app.query_one(StageLine).render().plain


async def test_a_run_that_finished_narrow_shows_its_coverage_once_after_widening() -> None:
    """The coverage row and the border subtitle never both show (rule 6, said once)."""
    app, schedulers = rich_app()
    async with app.run_test(size=(50, 30)) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].move(run, "done", report=a_report())
        await pilot.pause()
        block = app.query_one(RunBlock)
        assert not block.panelled
        coverage = block.query_one(CoverageLine)
        assert coverage.display
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert block.panelled
        assert "coverage" in (block.border_subtitle or "")
        assert not coverage.display
        assert len(block.query(CoverageLine)) == 1
        # And back: the frame goes, the row returns, still the one row.
        await pilot.resize_terminal(50, 30)
        await pilot.pause()
        assert not block.panelled
        assert coverage.display
        assert len(block.query(CoverageLine)) == 1


async def test_a_run_that_finished_wide_keeps_its_coverage_when_narrowed() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=(100, 30)) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].move(run, "done", report=a_report())
        await pilot.pause()
        block = app.query_one(RunBlock)
        assert not block.query_one(CoverageLine).display
        await pilot.resize_terminal(50, 30)
        await pilot.pause()
        assert not block.panelled
        assert block.query_one(CoverageLine).display
        assert "coverage 62/21/17%" in block.query_one(CoverageLine).render().plain


@pytest.mark.parametrize("width", [60, 66, 72, 79])
async def test_a_tier_less_badge_stays_on_one_row_between_the_floor_and_eighty(width: int) -> None:
    """A ghost carries no tier; its empty tier cell must not be what stacks the row."""
    app, schedulers = rich_app()
    async with app.run_test(size=(width, 30)) as pilot:
        run = await a_run(pilot, schedulers)
        schedulers[0].push(run, Emitted(replace(a_finding(), tier=None)))
        await pilot.pause()
        line = app.query_one(FindingLine)
        rows = rows_of(line.render().plain)
        assert line.size.width <= width
    # The badge, the label and the location all on the first row; the note under it.
    assert "GHOST REFERENCE" in rows[0] and "[12] Zhang" in rows[0] and "p.4" in rows[0]
    assert rows[0].rstrip().endswith("GHOST REFERENCE")
    assert rows[1].strip().startswith("DOI 10.1016")


async def test_a_rich_render_uses_only_single_cell_glyphs() -> None:
    """Windows CI runs this too: the risk there is a glyph two cells wide, not colour."""
    app, schedulers = rich_app()
    async with app.run_test(size=(80, 30)) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.move(run, "verifying")
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Progress("Verifying", 51, 118, ""))
        scheduler.push(run, Emitted(a_verdict_finding()))
        await pilot.pause()
        texts = [w.render().plain for w in (*app.query(StageLine), *app.query(FindingLine))]
        texts.extend(app.query_one(Banner).drawn.lines)  # type: ignore[union-attr]
        scheduler.move(run, "done", report=a_report())
        await pilot.pause()
        texts.append(app.query_one(CoverageFooter).render().plain)
    for text in texts:
        for row in rows_of(text):
            assert cell_len(row) == len(row) <= 80, row
