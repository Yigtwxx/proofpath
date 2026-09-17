# Multi-line paste Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A post of several lines pasted into the TUI bar is checked whole; today Textual keeps its first line and drops the rest.

**Architecture:** `Prompt` (the `Input` subclass in `src/proofpath/tui/widgets/prompt.py`) overrides `_on_paste`: a paste of two or more lines is held verbatim in `Prompt.held` while the bar shows a one-line summary; `take()` hands the text over on Enter, `drop()` forgets it on any edit or `escape`. `ProofpathApp.on_input_submitted` asks the bar for the held text before it reads `event.value`, keeps it out of the newline-delimited history file, and labels the run `/check pasted text`. Nothing below the app changes: `verify.target_document` already reads a non-address, non-file string as pasted text.

**Tech Stack:** Python 3.10+, `textual` 8.2 (`Input`, `events.Paste`, `events.Key`), `pytest` with Textual's `run_test` pilot.

Spec: `docs/superpowers/specs/2026-09-17-multiline-paste-design.md`.

## Global Constraints

- Type annotations on every function signature; `ruff check` and `ruff format --check` clean (run from the repo root; CI also formats fenced Python in Markdown).
- Code, identifiers and comments in English. Match the surrounding comment density and voice (`prompt.py` explains *why*, in full sentences).
- `pathlib` everywhere; every file read/write passes `encoding="utf-8"`.
- No network in tests; the TUI tests use `build_app()`'s fake scheduler.
- **Subagents never commit.** After a clean review the controller stages with `git add -A`; the user commits.
- Product rules are untouched by this plan: no state word, verdict or honesty sentence changes.

---

### Task 1: The bar holds a multi-line paste and Enter checks it whole

**Files:**
- Modify: `src/proofpath/tui/widgets/prompt.py` (imports at top; `Prompt.__init__` ~line 128; new methods after `_show` ~line 194; `on_input_changed` ~line 200)
- Modify: `src/proofpath/tui/app.py` (`compose` ~line 438; `on_input_submitted` ~line 586; `_mirror` ~line 641; `_check` ~line 773; `action_leave_awaiting` ~line 849)
- Test: `tests/test_tui_app.py` (new section after the history tests, ~line 975)

**Interfaces:**
- Consumes: `textual.events.Paste(text: str)`; `Input._on_paste(self, event)`; `Input._on_key(self, event)` (async); `Prompt._show(line)` (already routes a bar-owned replacement past `on_input_changed`); `tests/test_tui_app.py::build_app(theme=…)`, `::submit(pilot, line)`, `FakeScheduler.submitted: list[tuple[str, str]]`.
- Produces:
  - `prompt.HELD_SUMMARY: str` — format string with `{sep}`, `{lines}`, `{chars}`.
  - `Prompt.__init__(…, sep: str = "·")` — the theme's separator glyph.
  - `Prompt.held: str | None` — the held paste, or `None`.
  - `Prompt.take() -> str | None` — returns the held text and clears the hold.
  - `Prompt.drop() -> None` — forgets the held text and empties the bar; no-op when nothing is held.

- [ ] **Step 1: Read the two Textual methods being overridden**

Run: `sed -n 740,770p .venv/lib/python*/site-packages/textual/widgets/_input.py`

Confirm: `_on_key` is `async` and inserts `event.character` when `event.is_printable`; `_on_paste` is sync, takes `event.text.splitlines()[0]`, and calls `event.stop()`. The overrides below call these `super()` methods; if the signatures differ from what is shown here, follow the installed source.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_tui_app.py`, after `test_history_survives_a_restart` (the last history test). `PLAIN`, `CommandBlock` and `tevents` are already imported; change the theme import to `from proofpath.tui.theme import PLAIN, RICH` and add `from proofpath.tui.widgets.prompt import HELD_SUMMARY` in the `proofpath` import block (alphabetical: after `from proofpath.tui.runs import Run, State`).

```python
# --- a multi-line paste (spec 2026-09-17-multiline-paste) ---------------------------

POST = (
    "The vaccine trial enrolled 40,000 people.\n\n"
    "Source: https://example.org/trial\nSee also doi:10.1000/xyz123"
)


async def paste(pilot: Any, text: str) -> None:
    """Paste ``text`` into the bar the way a bracketed paste arrives: one event."""
    prompt = pilot.app.query_one(Prompt)
    prompt.focus()
    prompt.post_message(tevents.Paste(text))
    await pilot.pause()


async def test_a_multiline_paste_is_held_and_summarised() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        prompt = app.query_one(Prompt)
        assert prompt.held == POST
        # Three non-blank lines of four; the blank one is not counted.
        assert prompt.value == HELD_SUMMARY.format(sep=",", lines=3, chars=len(POST))
    assert schedulers[0].submitted == []


async def test_the_summary_uses_the_rich_separator() -> None:
    app, _ = build_app(theme=RICH)
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        assert app.query_one(Prompt).value == HELD_SUMMARY.format(sep="·", lines=3, chars=len(POST))


async def test_enter_checks_the_held_paste_whole() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("enter")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""
    # The target keeps its line breaks; the label is one line.
    assert schedulers[0].submitted == [(POST, "/check pasted text")]


async def test_a_held_paste_is_the_awaited_argument() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check")
        await paste(pilot, POST)
        await pilot.press("enter")
        await pilot.pause()
        assert app.awaiting is None
    assert schedulers[0].submitted == [(POST, "/check pasted text")]


async def test_a_typed_character_drops_the_held_paste() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("x")
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == "x"


async def test_backspace_drops_the_held_paste_and_empties_the_bar() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("backspace")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""


async def test_escape_drops_the_held_paste() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, POST)
        await pilot.press("escape")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""


async def test_a_history_step_drops_the_held_paste() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await paste(pilot, POST)
        await pilot.press("up")
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == "/help"


async def test_a_one_line_paste_lands_in_the_bar_as_text() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        await paste(pilot, "~/Desktop/paper.pdf")
        assert prompt.held is None
        assert prompt.value == "~/Desktop/paper.pdf"
        prompt.value = ""
        await paste(pilot, "~/Desktop/paper.pdf\n")  # a trailing newline is still one line
        assert prompt.held is None
        assert prompt.value == "~/Desktop/paper.pdf"


async def test_a_blank_paste_holds_nothing() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await paste(pilot, "\n\n  \n")
        prompt = app.query_one(Prompt)
        assert prompt.held is None
        assert prompt.value == ""


async def test_a_held_paste_stays_out_of_the_history_file(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await paste(pilot, POST)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("up")
        assert app.query_one(Prompt).value == "/help"
    assert (tmp_path / "state" / "history").read_text(encoding="utf-8") == "/help\n"


async def test_a_mirrored_verb_labels_a_held_paste_on_one_line() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/resolve")
        await paste(pilot, "Smith J.\nA title.\nJournal 2020")
        await pilot.press("enter")
        await pilot.pause()
        block = app.query_one(CommandBlock)
        assert block.command == "/resolve Smith J. A title. Journal 2020"
```

`CommandBlock.command` is the echoed line (`run_block.py:343`). The history file lands under `tmp_path / "state"` through the module's autouse `isolated` fixture (`PROOFPATH_STATE_DIR`), so `tmp_path` is the right argument and no history object is passed to `build_app`. `HELD_SUMMARY` is not re-exported by `proofpath.tui.app`; import it with `from proofpath.tui.widgets.prompt import HELD_SUMMARY`.

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tui_app.py -k "paste or held" -q`
Expected: FAIL — `ImportError: cannot import name 'HELD_SUMMARY'` (or, once the import is fixed, `AttributeError: 'Prompt' object has no attribute 'held'`).

- [ ] **Step 4: Teach `Prompt` to hold, take and drop**

In `src/proofpath/tui/widgets/prompt.py`:

Add the import (keep the alphabetical order of the `textual` block):

```python
from textual import events
```

Add the module constant after `ANSWER_ID`:

```python
#: What the bar shows while it holds a paste of several lines: the paste itself would
#: not fit, and Textual's ``Input`` would keep its first line and drop the rest.
HELD_SUMMARY = "pasted {sep} {lines} lines {sep} {chars} chars {sep} enter to check, esc to drop"
```

Extend the `Prompt` docstring with one paragraph after the ``tab`` paragraph:

```python
    A paste of two or more lines is *held*: the bar shows :data:`HELD_SUMMARY` and
    keeps the text whole in :attr:`held` until ``enter`` takes it or an edit drops
    it. A one-line paste is text in the bar like any other. The app hands in ``sep``
    because the bar has no theme of its own.
```

Change `__init__`'s signature and body:

```python
    def __init__(
        self,
        placeholder: str = "",
        *,
        id: str | None = None,  # noqa: A002 - Textual's own keyword, kept for callers
        history: History | None = None,
        complete: Callable[[str], list[str]] | None = None,
        sep: str = "·",
    ) -> None:
```

and, after `self._pending: deque[str] = deque()`:

```python
        self._sep = sep
        #: A multi-line paste, whole, while the bar shows its summary. ``take`` is the
        #: only way it leaves; ``_show`` forgets it, because whatever replaces the
        #: summary — a history step, a completion, an empty bar — is not the paste.
        self.held: str | None = None
```

Make the history walk drop a held paste before it starts, so the summary is never the draft the walk keeps and brings back on ``down``:

```python
def action_history_previous(self) -> None:
    self.drop()  # a held paste is not a draft; the walk must not bring it back
    line = self.history.previous(self.value)
    if line is not None:
        self._show(line)


def action_history_next(self) -> None:
    self.drop()
    line = self.history.next()
    if line is not None:
        self._show(line)
```

Add the three methods after `_show` (and update `_show` to forget the hold):

```python
def _show(self, line: str) -> None:
    # A reactive posts no ``Changed`` for an equal assignment, so only a line that
    # actually replaces the bar's value has a ``Changed`` to wait for.
    self.held = None
    if line != self.value:
        self._pending.append(line)
        self.value = line
    self.cursor_position = len(line)


def _on_paste(self, event: events.Paste) -> None:
    # ``Input`` keeps the first line of a paste and drops the rest, silently. A
    # post is several lines, so one of two or more is held whole behind a summary
    # instead. Trailing whitespace is cut first: a path pasted with its newline is
    # still one line and still lands in the bar as text.
    text = event.text.rstrip()
    lines = text.splitlines()
    if len(lines) < 2:
        super()._on_paste(event)
        return
    counted = sum(1 for line in lines if line.strip())
    self._show(HELD_SUMMARY.format(sep=self._sep, lines=counted, chars=len(text)))
    self.held = text  # after ``_show``, which forgets any hold
    event.stop()


async def _on_key(self, event: events.Key) -> None:
    # A printable key into the summary would edit the summary, which is nothing:
    # the paste is dropped first, so the key lands in an empty bar.
    if self.held is not None and event.is_printable:
        self.drop()
    await super()._on_key(event)


def take(self) -> str | None:
    """The held paste, whole, and the bar no longer holds it; ``None`` when none."""
    held, self.held = self.held, None
    return held


def drop(self) -> None:
    """Forget the held paste and empty the bar. Nothing when nothing is held."""
    if self.held is not None:
        self._show("")
```

Extend `on_input_changed` so any other edit of the summary (backspace, delete, a cut) drops:

```python
    def on_input_changed(self, event: Input.Changed) -> None:
        # Typing, deleting or pasting ends the walk and the cycle; the bar's own
        # replacements (a history step, a completion) do not.
        if self._pending and event.value == self._pending[0]:
            self._pending.popleft()
            return
        self._pending.clear()
        self.history.reset()
        self._candidates = []
        # An edit of the summary is an edit of nothing: the paste goes with it.
        self.drop()
```

- [ ] **Step 5: Wire the app**

In `src/proofpath/tui/app.py`:

`compose` — pass the separator:

```python
                yield Prompt(
                    placeholder=DEFAULT_PLACEHOLDER,
                    id="prompt",
                    history=self._history,
                    complete=self._completions,
                    sep=self._theme.glyphs.sep,
                )
```

`on_input_submitted` — take the held text first and keep it out of history:

```python
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = self.query_one(Prompt)
        held = prompt.take()
        # A held paste is the line; the summary in the bar is not. It stays out of
        # the history, whose file is one line per entry.
        line = event.value if held is None else held
        event.input.value = ""
        if held is None:
            self._history.add(line)
        await self._run_line(line)
```

`_check` — a one-line label for a multi-line target:

```python
    def _check(self, target: str) -> None:
        if self._scheduler is None:  # pragma: no cover - mount always runs first
            return
        # The run is about to be decided by the config, so the config is re-read first:
        # this is what makes a "never ask again" answered two runs ago stick.
        self._reload_config()
        # The label is one line: a pasted post is named the way the report names it.
        label = "pasted text" if "\n" in target else target
        self._scheduler.submit(target, command=f"/check {label}")
```

`_mirror` — collapse whitespace in the echoed line only:

```python
        line = " ".join(f"/{command.verb} {command.arg}".split())
```

`action_leave_awaiting` — `escape` drops a held paste too:

```python
    def action_leave_awaiting(self) -> None:
        self._leave_awaiting()
        self.query_one(Prompt).drop()
        # The app's priority ``escape`` shadows the screen's own ``_key_escape``,
        # which is where Textual drops a mouse selection; without this a selection
        # could only be cleared with the mouse.
        self.screen.clear_selection()
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tui_app.py -k "paste or held" -q`
Expected: all PASS.

If `test_a_typed_character_drops_the_held_paste` ends with `value == ""` instead of `"x"`, `Input._on_key` is not the method that inserts printable keys in the installed Textual; read the file from Step 1 again and move the `drop()` into the method that is.

- [ ] **Step 7: Run the whole TUI suite, lint and format**

Run: `.venv/bin/pytest tests/test_tui_app.py tests/test_tui_rich.py tests/test_tui_commands.py -q && ruff check src tests && ruff format --check src tests`
Expected: all PASS, `All checks passed!`, no files would be reformatted. `test_the_awaited_argument_starts_the_run` and the history tests must still pass unchanged: `_show` now clears `held`, and `on_input_submitted` still adds a typed line to history.

- [ ] **Step 8: Hand over for review (no commit)**

Report the diff of the three files. The controller stages after a clean review.

---

### Task 2: Say so in the spec, the README and the changelog

**Files:**
- Modify: `docs/superpowers/specs/2026-09-10-proofpath-design.md` (§13.1, after the awaiting-mode bullet ~line 680)
- Modify: `README.md` (the "Paste a path and it runs" paragraph ~line 48)
- Modify: `CHANGELOG.md` (`## [Unreleased]`, line 8)

**Interfaces:**
- Consumes: the behaviour of Task 1 — `HELD_SUMMARY`'s wording, `escape` drops, Enter checks.
- Produces: nothing code-level.

- [ ] **Step 1: Main spec §13.1**

Insert after the awaiting-mode bullet (the one ending `nothing else in the log keeps that colour.` and its fenced example), before `- **Several runs per session`:

```markdown
- **A multi-line paste is held whole (decided 2026-09-17).** Textual's `Input` keeps
  the first line of a bracketed paste; the bar overrides that. A paste of two or more
  lines is kept verbatim behind a one-line summary — `pasted · 3 lines · 118 chars ·
  enter to check, esc to drop` — and `Enter` checks the whole text, line breaks
  included, as an implicit `/check` (or as the argument of a waiting `/check`). Any
  edit, `Esc`, or a history step drops it and empties the bar; it is never added to
  the history file, whose entries are one line each. The run is labelled
  `/check pasted text`. A one-line paste is text in the bar, as before.
  Spec: `2026-09-17-multiline-paste-design.md`.
```

- [ ] **Step 2: README**

Change the opening of the paragraph at ~line 48 from

```markdown
Paste a path and it runs; every one-shot verb is a slash command (`/check`, `/resolve`,
```

to

```markdown
Paste a path, a URL or a claim and it runs — a post of several lines pastes whole: the
bar holds it behind a one-line summary, `Enter` checks it, `Esc` drops it. Every
one-shot verb is a slash command (`/check`, `/resolve`,
```

Re-wrap the following lines of that paragraph so no line exceeds the file's existing width (the README wraps at ~90 columns).

- [ ] **Step 3: CHANGELOG**

Under `## [Unreleased]`:

```markdown
## [Unreleased]

### Fixed
- **TUI: a multi-line paste is no longer cut to its first line.** Textual's input
  keeps the first line of a bracketed paste; the bar now holds the whole text behind
  a one-line summary (`pasted · 3 lines · 118 chars · enter to check, esc to drop`),
  `Enter` checks it whole and `Esc` drops it. The run is labelled `/check pasted
  text` and the paste stays out of the command history
  (`docs/superpowers/specs/2026-09-17-multiline-paste-design.md`).
```

- [ ] **Step 4: Verify the docs format**

Run: `ruff format --check . 2>&1 | tail -3; grep -c "multiline-paste" README.md CHANGELOG.md docs/superpowers/specs/2026-09-10-proofpath-design.md`
CI runs `uv run ruff format --check` from the repo root (`.github/workflows/ci.yml:23`); run exactly that. Expected: it passes (a `Markdown formatting is experimental` line for a `.md` path is ruff declining to format Markdown, not a failure — but `Would reformat` for any file is); the grep prints `0`, `1`, `1` (the README names the behaviour, not the spec file).

- [ ] **Step 5: Hand over for review (no commit)**

Report the diff of the three files.
