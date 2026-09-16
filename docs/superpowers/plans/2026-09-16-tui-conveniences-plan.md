# TUI terminal conveniences — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the TUI the reflexes of a shell prompt — command history, `Tab` completion, scrolling and walking the log from the keyboard, `Ctrl+L`, and copying a mouse selection — without breaking spec §13.1's rule that nothing is mouse-only or keyboard-only.

**Architecture:** A pure `History` class (no Textual import) owns the file and the walk; `Prompt` wires `↑`/`↓`/`Tab` to it and to a completion callback the app supplies; `Line` gains neighbour navigation; the app adds scroll, clear and copy-or-quit bindings and the `/help` lines. Spec: `docs/superpowers/specs/2026-09-16-tui-conveniences-design.md`.

**Tech Stack:** Python 3.10+, Textual 8.2, `platformdirs`, pytest with Textual's `Pilot` (`app.run_test`).

## Global Constraints

- Type annotations on every function signature; `ruff check` and `ruff format --check` clean.
- `pathlib` everywhere; every file read/write passes `encoding="utf-8"`.
- Code, identifiers and comments in English. Match the surrounding comment density and voice (the codebase explains *why* in prose comments).
- No network in tests; the TUI tests use the existing `FakeScheduler` in `tests/test_tui_app.py` and run at `SIZE = (80, 24)`.
- Nothing mouse-only, nothing keyboard-only (spec §13.1). Rule 6: the footer never goes away.
- History file: newest **500** lines; consecutive duplicates collapsed; blank lines never stored.
- `Tab` completes only when the bar's text starts with `/`; otherwise it keeps Textual's focus-next.
- Run the full suite before each commit: `.venv/bin/python -m pytest tests -q`.

## File map

| File | Change |
|---|---|
| `src/proofpath/paths.py` | add `STATE_DIR_ENV`, `state_dir()`, `history_path()` |
| `src/proofpath/tui/history.py` | **new** — `History` |
| `src/proofpath/tui/widgets/prompt.py` | `Prompt`: history walk, completion |
| `src/proofpath/tui/widgets/finding.py` | `Line`: `↑`/`↓` neighbours, printable key → bar |
| `src/proofpath/tui/app.py` | history load/save, completion callback, scroll/clear/copy bindings, `/help` |
| `tests/test_history.py` | **new** |
| `tests/test_tui_app.py` | Pilot tests for every key |
| `docs/superpowers/specs/2026-09-10-proofpath-design.md` | §13.1 amendment |
| `CHANGELOG.md` | `[Unreleased]` → `### Added` |

---

### Task 1: `state_dir()` and the `History` class

**Files:**
- Modify: `src/proofpath/paths.py`
- Create: `src/proofpath/tui/history.py`
- Test: `tests/test_history.py`

**Interfaces:**
- Produces: `paths.STATE_DIR_ENV = "PROOFPATH_STATE_DIR"`, `paths.state_dir() -> Path`, `paths.history_path() -> Path`.
- Produces: `History(path: Path | None = None, *, limit: int = 500)`, `History.load() -> None`, `History.add(line: str) -> None` (adds, caps, saves), `History.previous(draft: str) -> str | None`, `History.next() -> str | None`, `History.reset() -> None`, `History.entries -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_history.py
"""The prompt's command history: a file of lines and a shell-style walk over them."""

from __future__ import annotations

from pathlib import Path

import pytest

from proofpath import paths
from proofpath.tui.history import History


def test_state_dir_reads_the_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    assert paths.state_dir() == tmp_path
    assert paths.history_path() == tmp_path / "history"


def test_add_appends_and_saves(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("/check a.pdf")
    history.add("/help")
    assert history.entries == ("/check a.pdf", "/help")
    assert (tmp_path / "history").read_text(encoding="utf-8") == "/check a.pdf\n/help\n"


def test_add_skips_blank_and_consecutive_duplicate(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("   ")
    history.add("/help")
    history.add("/help")
    history.add("/quit")
    history.add("/help")
    assert history.entries == ("/help", "/quit", "/help")


def test_load_reads_an_existing_file(tmp_path: Path) -> None:
    (tmp_path / "history").write_text("one\ntwo\n", encoding="utf-8")
    history = History(tmp_path / "history")
    history.load()
    assert history.entries == ("one", "two")


def test_load_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    history = History(tmp_path / "missing" / "history")
    history.load()
    assert history.entries == ()


def test_add_creates_the_directory(tmp_path: Path) -> None:
    history = History(tmp_path / "deep" / "er" / "history")
    history.add("/help")
    assert (tmp_path / "deep" / "er" / "history").read_text(encoding="utf-8") == "/help\n"


def test_a_write_failure_keeps_the_session_history(tmp_path: Path) -> None:
    # A *file* where the directory should be: mkdir and the write both fail.
    (tmp_path / "state").write_text("", encoding="utf-8")
    history = History(tmp_path / "state" / "history")
    history.add("/help")
    assert history.entries == ("/help",)


def test_the_file_keeps_only_the_newest_lines(tmp_path: Path) -> None:
    history = History(tmp_path / "history", limit=3)
    for line in ("a", "b", "c", "d"):
        history.add(line)
    assert history.entries == ("b", "c", "d")
    assert (tmp_path / "history").read_text(encoding="utf-8") == "b\nc\nd\n"


def test_previous_walks_back_and_next_restores_the_draft(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("one")
    history.add("two")
    assert history.previous("draft") == "two"
    assert history.previous("draft") == "one"
    assert history.previous("draft") is None  # at the oldest: stay put
    assert history.next() == "two"
    assert history.next() == "draft"
    assert history.next() is None  # past the newest: nothing more to show


def test_previous_on_an_empty_history_is_none(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    assert history.previous("draft") is None
    assert history.next() is None


def test_reset_ends_the_walk(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("one")
    assert history.previous("draft") == "one"
    history.reset()
    assert history.next() is None
    assert history.previous("new draft") == "one"
    assert history.next() == "new draft"


def test_add_ends_the_walk(tmp_path: Path) -> None:
    history = History(tmp_path / "history")
    history.add("one")
    history.previous("")
    history.add("two")
    assert history.next() is None
    assert history.previous("") == "two"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_history.py -q`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'proofpath.tui.history'`.

- [ ] **Step 3: Add `state_dir()` to `paths.py`**

Change the import and add, after `CACHE_DIR_ENV`, then after `cache_dir()`:

```python
from platformdirs import user_cache_dir, user_config_dir, user_state_dir
```

```python
STATE_DIR_ENV = "PROOFPATH_STATE_DIR"
```

```python
def state_dir() -> Path:
    """Where session state lives: things that are neither config nor re-downloadable.

    Today that is the prompt's command history. It is not under the cache root on
    purpose: ``cache clear`` must never forget what the user typed.
    """
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else Path(user_state_dir(APP_NAME))


def history_path() -> Path:
    """The TUI's command history, one submitted line per row. It may not exist yet."""
    return state_dir() / "history"
```

- [ ] **Step 4: Write `history.py`**

```python
# src/proofpath/tui/history.py
"""The input bar's command history: a file of lines and a shell-style walk over them.

Pure: no Textual, no app. The bar asks :meth:`History.previous` and
:meth:`History.next` and shows what comes back; the file is the user's own typing
and nothing else (spec ``2026-09-16-tui-conveniences-design.md`` section 2.1).
"""

from __future__ import annotations

import logging
from pathlib import Path

from proofpath.paths import history_path

logger = logging.getLogger(__name__)

#: How many lines the file keeps. Older ones are dropped on save.
LIMIT = 500


class History:
    """Submitted lines, newest last, and a cursor for walking back through them.

    The walk has shell semantics: ``previous`` moves towards the oldest entry and
    stops there; ``next`` moves towards the newest and, one step past it, hands back
    the *draft* -- whatever the bar held when the walk began. ``add`` and ``reset``
    end a walk; the bar calls ``reset`` on any edit.
    """

    def __init__(self, path: Path | None = None, *, limit: int = LIMIT) -> None:
        self._path = path if path is not None else history_path()
        self._limit = limit
        self._entries: list[str] = []
        #: Index into ``_entries`` while walking; ``None`` when not walking.
        self._cursor: int | None = None
        self._draft = ""

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def load(self) -> None:
        """Read the file. A missing or unreadable file is an empty history, not an error."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            self._entries = []
            return
        lines = [line for line in text.splitlines() if line.strip()]
        self._entries = lines[-self._limit :]

    def add(self, line: str) -> None:
        """Record a submitted line, cap the list, write the file, end any walk."""
        self.reset()
        if not line.strip():
            return
        if self._entries and self._entries[-1] == line:
            return
        self._entries.append(line)
        del self._entries[: -self._limit]
        self._save()

    def previous(self, draft: str) -> str | None:
        """One step towards the oldest entry, or ``None`` at the end (or when empty).

        ``draft`` is remembered on the first step so ``next`` can bring it back.
        """
        if not self._entries:
            return None
        if self._cursor is None:
            self._draft = draft
            self._cursor = len(self._entries) - 1
        elif self._cursor == 0:
            return None
        else:
            self._cursor -= 1
        return self._entries[self._cursor]

    def next(self) -> str | None:
        """One step towards the newest entry; one past it is the draft; then ``None``."""
        if self._cursor is None:
            return None
        self._cursor += 1
        if self._cursor >= len(self._entries):
            self._cursor = None
            return self._draft
        return self._entries[self._cursor]

    def reset(self) -> None:
        """End a walk. The next ``previous`` starts again from the newest entry."""
        self._cursor = None
        self._draft = ""

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                "".join(f"{line}\n" for line in self._entries), encoding="utf-8"
            )
        except OSError as exc:
            # The session keeps its in-memory history; only persistence is lost.
            logger.debug("could not write %s: %s", self._path, exc)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_history.py -q`
Expected: all pass.

- [ ] **Step 6: Lint and the full suite**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest tests -q`
Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add src/proofpath/paths.py src/proofpath/tui/history.py tests/test_history.py
git commit -m "feat(tui): a command history file and a shell-style walk over it"
```

---

### Task 2: `↑`/`↓` in the bar

**Files:**
- Modify: `src/proofpath/tui/widgets/prompt.py` (class `Prompt`)
- Modify: `src/proofpath/tui/app.py` (`__init__`, `on_mount`, `on_input_submitted`)
- Test: `tests/test_tui_app.py`

**Interfaces:**
- Consumes: `History` from Task 1.
- Produces: `Prompt.history: History` (set by the app before mount), `Prompt.action_history_previous()`, `Prompt.action_history_next()`. `ProofpathApp.__init__` gains keyword `history: History | None = None` (a test passes one on a temp path; `None` means `History()` on the real state path).

- [ ] **Step 1: Point the test fixture at a temp state dir and write the failing tests**

In `tests/test_tui_app.py`, the autouse `isolated` fixture gains one line:

```python
    monkeypatch.setenv("PROOFPATH_STATE_DIR", str(tmp_path / "state"))
```

Add these tests after `test_a_fresh_command_leaves_awaiting_mode`:

```python
# --- history ------------------------------------------------------------------------


async def test_up_brings_back_the_last_line() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/config")
        prompt = app.query_one(Prompt)
        await pilot.press("up")
        assert prompt.value == "/config"
        assert prompt.cursor_position == len("/config")
        await pilot.press("up")
        assert prompt.value == "/help"
        await pilot.press("up")  # at the oldest: stays
        assert prompt.value == "/help"


async def test_down_past_the_newest_restores_the_draft() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        prompt = app.query_one(Prompt)
        prompt.value = "half a th"
        await pilot.press("up")
        assert prompt.value == "/help"
        await pilot.press("down")
        assert prompt.value == "half a th"
        await pilot.press("down")  # nothing newer than the draft
        assert prompt.value == "half a th"


async def test_an_edit_ends_the_walk() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/config")
        prompt = app.query_one(Prompt)
        await pilot.press("up")
        await pilot.press("x")
        assert prompt.value == "/configx"
        await pilot.press("up")  # a fresh walk: starts at the newest again
        assert prompt.value == "/config"


async def test_history_survives_a_restart(tmp_path: Path) -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
    assert (tmp_path / "state" / "history").read_text(encoding="utf-8") == "/help\n"
    again, _ = build_app()
    async with again.run_test(size=SIZE) as pilot:
        await pilot.press("up")
        assert again.query_one(Prompt).value == "/help"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k "history or draft or ends_the_walk or last_line"`
Expected: 4 failures (`up` does nothing; `prompt.value` unchanged).

- [ ] **Step 3: Give `Prompt` the walk**

In `prompt.py`, add imports and rewrite `Prompt`:

```python
from typing import ClassVar

from textual.binding import Binding, BindingType

from proofpath.tui.history import History
```

```python
class Prompt(Input):
    """The input bar. Awaiting mode tints it with the next run's accent.

    ``up`` and ``down`` walk the command history the way a shell does: the draft in
    the bar when the walk starts comes back one step past the newest entry, and any
    edit ends the walk. The app hands in the :class:`History` before the bar mounts.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "history_previous", "previous command", show=False),
        Binding("down", "history_next", "next command", show=False),
    ]

    def __init__(self, *args: object, history: History | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.history = history if history is not None else History()
        #: Set while the bar's value is being replaced by the walk itself, so the
        #: change it causes is not read as an edit that ends the walk.
        self._walking = False

    def action_history_previous(self) -> None:
        line = self.history.previous(self.value)
        if line is not None:
            self._show(line)

    def action_history_next(self) -> None:
        line = self.history.next()
        if line is not None:
            self._show(line)

    def _show(self, line: str) -> None:
        self._walking = True
        try:
            self.value = line
            self.cursor_position = len(line)
        finally:
            self._walking = False

    def on_input_changed(self, event: Input.Changed) -> None:
        # Typing, deleting or pasting ends the walk; the bar's own replacement does not.
        if not self._walking:
            self.history.reset()

    def tint(self, accent: str) -> None:
        ...  # unchanged

    def untint(self) -> None:
        ...  # unchanged
```

(Keep the existing bodies of `tint` and `untint`; the `...` above marks them unchanged.) If `Input.__init__`'s signature makes the `*args: object` typing awkward under ruff/mypy, spell the parameters the app actually uses instead: `def __init__(self, placeholder: str = "", *, id: str | None = None, history: History | None = None) -> None: super().__init__(placeholder=placeholder, id=id)`.

- [ ] **Step 4: Wire the app**

In `app.py`:

```python
from proofpath.tui.history import History
```

`__init__` gains the keyword `history: History | None = None` and stores `self._history = history if history is not None else History()` next to `self._theme`.

In `compose`, the bar is built with the history: `yield Prompt(placeholder=DEFAULT_PLACEHOLDER, id="prompt", history=self._history)`.

In `on_mount`, before `self._accent_prompt(0)`: `self._history.load()`.

`on_input_submitted` records the line before running it:

```python
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value
        event.input.value = ""
        self._history.add(line)
        await self._run_line(line)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k "history or draft or ends_the_walk or last_line"`
Expected: 4 pass.

- [ ] **Step 6: Lint and the full suite**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest tests -q`
Expected: clean, all pass. (If `Input.Changed` fires on the initial empty value and a test sees a stale draft, `reset()` on an empty walk is harmless — it is not a failure.)

- [ ] **Step 7: Commit**

```bash
git add src/proofpath/tui/widgets/prompt.py src/proofpath/tui/app.py tests/test_tui_app.py
git commit -m "feat(tui): up and down walk the command history in the bar"
```

---

### Task 3: `Tab` completion

**Files:**
- Modify: `src/proofpath/tui/commands.py` (new pure function `complete`)
- Modify: `src/proofpath/tui/widgets/prompt.py` (`Prompt`)
- Modify: `src/proofpath/tui/app.py` (the callback)
- Test: `tests/test_commands.py`, `tests/test_tui_app.py`

**Interfaces:**
- Produces: `commands.complete(text: str, *, run_ids: Iterable[int] = ()) -> list[str]` — full replacement lines, in order, empty when there is nothing to complete.
- Produces: `Prompt.__init__` keyword `complete: Callable[[str], list[str]] | None = None`; `Prompt.action_complete()`; `Prompt.check_action` disables `complete` unless the text starts with `/`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_commands.py` (check its imports; it already imports `commands` from `proofpath.tui`):

```python
@pytest.mark.parametrize(
    ("text", "run_ids", "expected"),
    [
        ("/ch", (), ["/check "]),
        ("/c", (), ["/check ", "/config ", "/cache ", "/cancel "]),
        ("/", (), [f"/{verb} " if verb in commands.NEEDS_ARGUMENT else f"/{verb}" for verb in commands.VERBS]),
        ("/help", (), ["/help"]),
        ("/allow ", (), ["/allow once", "/allow always", "/allow no", "/allow never"]),
        ("/allow n", (), ["/allow no", "/allow never"]),
        ("/cancel ", (2, 3), ["/cancel #2", "/cancel #3"]),
        ("/cancel #3", (2, 3), ["/cancel #3"]),
        ("/cancel ", (), []),
        ("/check ", (), []),  # a target is not completed
        ("paper.pdf", (), []),  # not a command
        ("", (), []),
        ("/zz", (), []),
    ],
)
def test_complete(text: str, run_ids: tuple[int, ...], expected: list[str]) -> None:
    assert commands.complete(text, run_ids=run_ids) == expected
```

Note on `/c`: the expected order is `VERBS` order filtered by prefix — `check`, `config`, `cache`, `cancel` — and `/config` and `/cache` take an optional argument, so check `NEEDS_ARGUMENT` for which verbs get the trailing space; adjust the expected list to whatever `NEEDS_ARGUMENT` says (`config`/`cache` are *not* in it today, so they complete without a space: `["/check ", "/config", "/cache", "/cancel "]`). Use the rule, not this paragraph, as the source of truth.

In `tests/test_tui_app.py`:

```python
# --- completion ---------------------------------------------------------------------


async def test_tab_completes_a_verb() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        prompt.focus()
        prompt.value = "/ch"
        prompt.cursor_position = 3
        await pilot.press("tab")
        assert prompt.value == "/check "
        assert app.focused is prompt


async def test_tab_cycles_through_the_allow_answers() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        prompt = app.query_one(Prompt)
        prompt.focus()
        prompt.value = "/allow "
        prompt.cursor_position = 7
        await pilot.press("tab")
        assert prompt.value == "/allow once"
        await pilot.press("tab")
        assert prompt.value == "/allow always"
        await pilot.press("tab")
        await pilot.press("tab")
        await pilot.press("tab")  # wraps
        assert prompt.value == "/allow once"


async def test_tab_offers_the_runs_that_can_be_cancelled() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await submit(pilot, "/check two.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done")
        await pilot.pause()
        prompt = app.query_one(Prompt)
        prompt.value = "/cancel "
        prompt.cursor_position = 8
        await pilot.press("tab")
        assert prompt.value == "/cancel #2"


async def test_tab_on_a_target_moves_focus_instead() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        prompt = app.query_one(Prompt)
        prompt.value = "paper.pdf"
        await pilot.press("tab")
        assert prompt.value == "paper.pdf"
        assert app.focused is not prompt
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_commands.py tests/test_tui_app.py -q -k "complete or tab_"`
Expected: `AttributeError: module 'proofpath.tui.commands' has no attribute 'complete'` and the Pilot tests fail on `prompt.value`.

- [ ] **Step 3: Write `commands.complete`**

In `commands.py` (it already has `VERBS`, `NEEDS_ARGUMENT`, `ALLOW_ANSWERS`):

```python
from collections.abc import Iterable
```

```python
def complete(text: str, *, run_ids: Iterable[int] = ()) -> list[str]:
    """Every line ``text`` could be finished into, in order; empty when there is none.

    Only slash commands complete. A verb is finished with a space when it takes an
    argument, so the next keystroke is already the argument; ``/allow`` offers its
    four answers and ``/cancel`` the runs that can still be cancelled. A target is
    never completed: the OS has better file pickers than a bar could.
    """
    if not text.startswith("/"):
        return []
    body = text[1:]
    if " " not in body:
        return [
            f"/{verb} " if verb in NEEDS_ARGUMENT else f"/{verb}"
            for verb in VERBS
            if verb.startswith(body)
        ]
    verb, _, partial = body.partition(" ")
    if verb == "allow":
        return [f"/allow {answer}" for answer in ALLOW_ANSWERS if answer.startswith(partial)]
    if verb == "cancel":
        return [f"/cancel #{run_id}" for run_id in run_ids if f"#{run_id}".startswith(partial)]
    return []
```

- [ ] **Step 4: Give `Prompt` the `Tab`**

In `prompt.py`, add to the imports `from collections.abc import Callable`, and extend `Prompt`:

```python
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "history_previous", "previous command", show=False),
        Binding("down", "history_next", "next command", show=False),
        Binding("tab", "complete", "complete", show=False),
    ]
```

`__init__` gains `complete: Callable[[str], list[str]] | None = None`, stored as `self._complete = complete or (lambda text: [])`, plus two fields:

```python
        #: The candidates of the cycle in progress and where the cycle is in them;
        #: cleared by any edit that is not the cycle's own.
        self._candidates: list[str] = []
        self._candidate = 0
```

Then:

```python
    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        # ``tab`` is completion only inside a slash command; anywhere else it stays
        # Textual's focus-next, which is how the keyboard reaches the log.
        if action == "complete":
            return self.value.startswith("/")
        return True

    def action_complete(self) -> None:
        if not self._candidates:
            self._candidates = self._complete(self.value)
            self._candidate = 0
            if not self._candidates:
                return
        else:
            self._candidate = (self._candidate + 1) % len(self._candidates)
        candidates = self._candidates  # ``_show`` triggers ``Changed``, which clears them
        self._show(candidates[self._candidate])
        self._candidates = candidates
```

and `on_input_changed` also clears the cycle:

```python
    def on_input_changed(self, event: Input.Changed) -> None:
        # Typing, deleting or pasting ends the walk and the cycle; the bar's own
        # replacements (a history step, a completion) do not.
        if not self._walking:
            self.history.reset()
            self._candidates = []
```

- [ ] **Step 5: Supply the callback from the app**

In `compose`: `yield Prompt(placeholder=DEFAULT_PLACEHOLDER, id="prompt", history=self._history, complete=self._completions)`. Add to the app, near `_help`:

```python
    def _completions(self, text: str) -> list[str]:
        """What ``Tab`` may finish ``text`` into. Only the app knows which runs are live."""
        live = () if self._scheduler is None else tuple(
            run.id for run in self._scheduler.runs if run.state not in ("done", "cancelled", "failed")
        )
        return commands.complete(text, run_ids=live)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_commands.py tests/test_tui_app.py -q -k "complete or tab_"`
Expected: all pass. If `test_tab_on_a_target_moves_focus_instead` fails because `check_action` is consulted only for the widget's own bindings and the key is still swallowed, return `None` instead of `False` for the inactive case — Textual treats `None` as "binding not shown and not active, let it bubble".

- [ ] **Step 7: Lint and the full suite**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest tests -q`
Expected: clean, all pass.

- [ ] **Step 8: Commit**

```bash
git add src/proofpath/tui/commands.py src/proofpath/tui/widgets/prompt.py src/proofpath/tui/app.py tests/test_commands.py tests/test_tui_app.py
git commit -m "feat(tui): tab completes slash commands, /allow answers and #n"
```

---

### Task 4: scrolling from the bar and walking the log's lines

**Files:**
- Modify: `src/proofpath/tui/app.py` (BINDINGS + actions)
- Modify: `src/proofpath/tui/widgets/finding.py` (`Line`)
- Test: `tests/test_tui_app.py`

**Interfaces:**
- Produces: app actions `scroll_log(direction: str)` bound to `pageup`, `pagedown`, `shift+up`, `shift+down`, `ctrl+home`, `ctrl+end`.
- Produces: `Line.action_neighbour(step: int)` bound to `up`/`down`; `Line.on_key` forwards printable keys (except `c`) to the bar.

- [ ] **Step 1: Write the failing tests**

There is a helper pattern in the file that pushes events into a run (`scheduler.push(run, StageStart(...))` etc.); reuse it to make a log tall enough to scroll. Add:

```python
# --- the log from the keyboard ------------------------------------------------------


async def _tall_log(pilot: Any, app: ProofpathApp, scheduler: FakeScheduler) -> None:
    """Enough help lines to overflow 24 rows, so there is something to scroll."""
    for _ in range(6):
        await submit(pilot, "/help")
    await pilot.pause()
    assert app.query_one(RunLog).max_scroll_y > 0


async def test_pageup_scrolls_the_log_without_leaving_the_bar() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await _tall_log(pilot, app, schedulers[0])
        log = app.query_one(RunLog)
        at_end = log.scroll_y
        await pilot.press("pageup")
        await pilot.pause()
        assert log.scroll_y < at_end
        assert app.focused is app.query_one(Prompt)
        await pilot.press("ctrl+end")
        await pilot.pause()
        assert log.scroll_y == at_end
        await pilot.press("ctrl+home")
        await pilot.pause()
        assert log.scroll_y == 0
        await pilot.press("shift+down")
        await pilot.pause()
        assert log.scroll_y == 1


async def test_up_and_down_walk_the_lines() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        run = scheduler.runs[0]
        scheduler.move(run, "running")
        scheduler.push(run, StageStart(stage="parse"))
        scheduler.push(run, StageEnd(stage="parse", elapsed=0.1, summary="ok"))
        scheduler.push(run, StageStart(stage="resolve"))
        await pilot.pause()
        header = app.query_one(RunHeader)
        header.focus()
        await pilot.press("down")
        assert isinstance(app.focused, StageLine)
        assert app.focused.stage == "parse"
        await pilot.press("down")
        assert app.focused.stage == "resolve"
        await pilot.press("down")  # last line: stays
        assert app.focused.stage == "resolve"
        await pilot.press("up")
        await pilot.press("up")
        assert app.focused is header
        await pilot.press("up")  # first line: stays
        assert app.focused is header


async def test_a_collapsed_block_is_skipped_by_the_walk() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await submit(pilot, "/check two.pdf")
        scheduler = schedulers[0]
        for run in scheduler.runs:
            scheduler.move(run, "running")
            scheduler.push(run, StageStart(stage="parse"))
        await pilot.pause()
        first, second = app.query(RunHeader)
        first.focus()
        await pilot.press("enter")  # collapse the first block
        await pilot.press("down")
        assert app.focused is second


async def test_typing_on_a_line_goes_to_the_bar() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        await pilot.pause()
        app.query_one(RunHeader).focus()
        await pilot.press("slash")
        prompt = app.query_one(Prompt)
        assert app.focused is prompt
        assert prompt.value == "/"
        await pilot.press("h")
        assert prompt.value == "/h"
```

Check the `StageStart`/`StageEnd` constructor fields against `src/proofpath/events.py` and the existing tests in this file; use the same spelling they do. If `StageEnd` needs more fields, copy a call from an existing test.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k "pageup or walk_the_lines or collapsed_block or typing_on_a_line"`
Expected: 4 failures.

- [ ] **Step 3: The app's scroll bindings**

In `app.py`, extend `BINDINGS`:

```python
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "leave_awaiting", "cancel awaiting", priority=True),
        Binding("ctrl+c", "quit_app", "quit", priority=True),
        Binding("ctrl+d", "quit_app", "quit", priority=True),
        # The log from the bar (design section 2.3): the bar keeps ``up``, ``down``,
        # ``home`` and ``end``; these reach past it whatever has focus.
        Binding("pageup", "scroll_log('page_up')", "log page up", show=False, priority=True),
        Binding("pagedown", "scroll_log('page_down')", "log page down", show=False, priority=True),
        Binding("shift+up", "scroll_log('up')", "log up", show=False, priority=True),
        Binding("shift+down", "scroll_log('down')", "log down", show=False, priority=True),
        Binding("ctrl+home", "scroll_log('home')", "log top", show=False, priority=True),
        Binding("ctrl+end", "scroll_log('end')", "log bottom", show=False, priority=True),
    ]
```

and the action, next to `_scroll_log`:

```python
    def action_scroll_log(self, direction: str) -> None:
        """Move the log without moving focus. ``direction`` names a ``Widget.scroll_*``."""
        log = self.query_one(RunLog)
        move = {
            "page_up": log.scroll_page_up,
            "page_down": log.scroll_page_down,
            "up": log.scroll_up,
            "down": log.scroll_down,
            "home": log.scroll_home,
            "end": log.scroll_end,
        }[direction]
        move(animate=False)
```

- [ ] **Step 4: `Line`'s neighbours and the hand-off to the bar**

In `finding.py`, extend `Line`:

```python
    BINDINGS: ClassVar[list[BindingType]] = [
        # Not ``toggle``: ``DOMNode.action_toggle`` already means "flip a reactive"
        # and shadowing it would break every binding that uses the real one.
        Binding("enter", "expand", "expand or collapse"),
        Binding("c", "copy", "copy"),
        Binding("up", "neighbour(-1)", "previous line", show=False),
        Binding("down", "neighbour(1)", "next line", show=False),
    ]
```

```python
    def action_neighbour(self, step: int) -> None:
        """Focus the previous (``-1``) or next (``1``) displayed line; the ends stay put.

        The screen's focus chain already skips everything inside a collapsed block
        (``display: none``), so a folded run is one line to step over, not many.
        """
        lines = [w for w in self.screen.focus_chain if isinstance(w, Line)]
        try:
            index = lines.index(self)
        except ValueError:  # pragma: no cover - a line that is not displayed has no focus
            return
        target = index + step
        if 0 <= target < len(lines):
            lines[target].focus()

    def on_key(self, event: tevents.Key) -> None:
        """A printable key on a line is typing, and typing belongs to the bar.

        ``c`` is the one exception: it is this line's copy (spec section 13.1).
        """
        if not event.is_printable or event.character is None or event.key == "c":
            return
        event.stop()
        event.prevent_default()
        prompt = self.screen.query_one("#prompt")
        prompt.focus()
        prompt.insert_text_at_cursor(event.character)  # type: ignore[attr-defined]
```

`#prompt` rather than importing `Prompt`: `prompt.py` already imports from `_shared`, and `finding.py` must not grow a cycle. If the `type: ignore` is unwelcome, `from textual.widgets import Input` and `query_one("#prompt", Input)` gives the typed handle.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k "pageup or walk_the_lines or collapsed_block or typing_on_a_line"`
Expected: 4 pass. If `focus_chain` includes the permission buttons or the bar between lines, that is fine — the filter keeps only `Line`s. If `shift+down` scrolls by more than one row, check `scroll_down`'s default `y=1` in Textual 8 and pass `y=1` explicitly.

- [ ] **Step 6: Lint and the full suite**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest tests -q`
Expected: clean, all pass.

- [ ] **Step 7: Commit**

```bash
git add src/proofpath/tui/app.py src/proofpath/tui/widgets/finding.py tests/test_tui_app.py
git commit -m "feat(tui): scroll the log from the bar and walk its lines with the arrows"
```

---

### Task 5: `Ctrl+L` and copy-or-quit

**Files:**
- Modify: `src/proofpath/tui/app.py`
- Test: `tests/test_tui_app.py`

**Interfaces:**
- Produces: `ProofpathApp.action_clear_log()` on `ctrl+l`; `action_quit_app` copies instead of quitting when `self.screen.get_selected_text()` is not `None`; `super+c` joins `ctrl+c`.

- [ ] **Step 1: Write the failing tests**

```python
# --- clearing and copying -----------------------------------------------------------


async def test_ctrl_l_drops_finished_blocks_and_keeps_live_ones() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/check one.pdf")
        await submit(pilot, "/check two.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done")
        scheduler.move(scheduler.runs[1], "running")
        await pilot.pause()
        assert len(app.query(RunBlock)) == 2
        assert app.query(NoteLine)
        await pilot.press("ctrl+l")
        await pilot.pause()
        blocks = list(app.query(RunBlock))
        assert [block.run.id for block in blocks] == [2]
        assert not app.query(NoteLine)
        # The scheduler still knows the cleared run: /cancel and /summarize see it.
        assert scheduler.get(1) is not None


async def test_ctrl_l_keeps_the_footer() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/check one.pdf")
        scheduler = schedulers[0]
        scheduler.move(scheduler.runs[0], "done", report=make_report())
        await pilot.pause()
        before = str(app.query_one(CoverageFooter).render())
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert str(app.query_one(CoverageFooter).render()) == before


async def test_ctrl_c_with_a_selection_copies_instead_of_quitting() -> None:
    app, _ = build_app()
    copied: list[str] = []
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        line = app.query(NoteLine).first()
        app.screen.select_all_in_widget(line) if hasattr(app.screen, "select_all_in_widget") else None
        await pilot.pause()
        if app.screen.get_selected_text() is None:
            pytest.skip("this Textual has no programmatic selection to test with")
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert copied and "/check" in copied[0]
        assert app.is_running


async def test_ctrl_c_without_a_selection_quits() -> None:
    app, schedulers = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert schedulers[0].closed
```

`make_report()` — look for how the existing tests build a finished `Report` (search the file for `Report(` or a helper) and use that; there is one used by the footer tests. If the selection API has a different name in Textual 8.2 (`grep -n "def select" .venv/lib/python3.12/site-packages/textual/screen.py`), use it; the `skip` keeps the test honest if there is none.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k "ctrl_l or ctrl_c"`
Expected: `ctrl_l` tests fail (blocks still there); `without_a_selection_quits` already passes; `with_a_selection` fails or skips.

- [ ] **Step 3: The bindings and actions**

In `app.py` `BINDINGS`, replace the `ctrl+c` line and add `ctrl+l`:

```python
        Binding("ctrl+c,super+c", "quit_app", "copy or quit", priority=True),
        Binding("ctrl+d", "quit_app", "quit", priority=True),
        Binding("ctrl+l", "clear_log", "clear finished runs", show=False, priority=True),
```

Then:

```python
    async def action_quit_app(self) -> None:
        # With a selection on screen the key means "copy", as it does in a terminal;
        # without one it means what it always meant. ``ctrl+d`` never copies.
        selected = self.screen.get_selected_text()
        if selected is not None:
            self.copy_to_clipboard(selected)
            return
        await self.action_quit()

    def action_clear_log(self) -> None:
        """Drop every finished block; keep the ones still streaming and the footer.

        Nothing is forgotten: the scheduler still holds every run, so ``/cancel #n``
        and ``/summarize`` are unaffected. The footer is the latest finished run's
        coverage and rule 6 says it does not go away.
        """
        finished = frozenset({"done", "cancelled", "failed"})
        for block in list(self.query(RunBlock)):
            if block.run.state in finished:
                self._blocks.pop(block.run.id, None)
                block.remove()
        for block in list(self.query(CommandBlock)):
            if block not in self._command_blocks.values():
                block.remove()
        for note in list(self.query(NoteLine)):
            if isinstance(note.parent, RunLog):
                note.remove()
```

Check `self._blocks` is keyed by `run.id` (search `_blocks[` in `app.py`) and that `NoteLine`s inside a `RunBlock` have a parent other than `RunLog` — the `isinstance(note.parent, RunLog)` guard keeps a block's own notes with the block. `ctrl+d` keeps its own `quit_app` binding but must never copy; if the shared action makes that untrue, give `ctrl+d` its own `action_quit_only` that calls `self.action_quit()` directly.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k "ctrl_l or ctrl_c"`
Expected: pass (or one skip).

- [ ] **Step 5: Lint and the full suite**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest tests -q`
Expected: clean, all pass.

- [ ] **Step 6: Commit**

```bash
git add src/proofpath/tui/app.py tests/test_tui_app.py
git commit -m "feat(tui): ctrl+l clears finished runs, ctrl+c copies a selection"
```

---

### Task 6: `/help`, the spec amendment and the changelog

**Files:**
- Modify: `src/proofpath/tui/app.py` (`_help`)
- Modify: `docs/superpowers/specs/2026-09-10-proofpath-design.md` (§13.1)
- Modify: `CHANGELOG.md`
- Test: `tests/test_tui_app.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_help_lists_the_keys() -> None:
    app, _ = build_app()
    async with app.run_test(size=SIZE) as pilot:
        await submit(pilot, "/help")
        await pilot.pause()
        notes = [str(note.render()) for note in app.query(NoteLine)]
        assert any("history" in note and "Tab" in note for note in notes)
        assert any("Ctrl+L" in note for note in notes)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k help_lists_the_keys`
Expected: FAIL.

- [ ] **Step 3: The help lines**

```python
#: The key reference ``/help`` prints after the verbs (design section 2.7).
KEY_HELP = (
    "↑ ↓ history · Tab completes /verbs · PageUp/PageDown Shift+↑↓ scroll the log",
    "Tab into the log: ↑ ↓ walk lines, Enter opens, c copies · Ctrl+L clears finished runs",
)
```

Place it after `SUMMARY_IN_PROGRESS` and print it in `_help`:

```python
    def _help(self) -> None:
        """Every verb with what it wants, so the list is also the syntax; then the keys."""
        for verb in commands.VERBS:
            placeholder = commands.NEEDS_ARGUMENT.get(verb, "")
            self._note(f"/{verb} {placeholder}".rstrip(), dim=False)
        self._note("a line that is not a command is checked as a target")
        for line in KEY_HELP:
            self._note(line)
```

In PLAIN (ASCII-only) the arrows must not appear. Check how the theme exposes glyph choice (`theme.glyphs`, `theme.name == "plain"`) and use `"up/down"` and `"Shift+up/down"` there:

```python
        arrows = "↑ ↓" if self._theme.name == "rich" else "up/down"
```

and build the two lines with an f-string from that (keep `KEY_HELP` as a function of `arrows` if the constant would otherwise duplicate text: `def key_help(arrows: str) -> tuple[str, str]`).

- [ ] **Step 4: Run it to verify it passes; full suite**

Run: `.venv/bin/python -m pytest tests/test_tui_app.py -q -k help_lists_the_keys && .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest tests -q`
Expected: pass; clean.

- [ ] **Step 5: Amend spec §13.1**

In `docs/superpowers/specs/2026-09-10-proofpath-design.md`, in the "Mouse works everywhere the keyboard does" bullet, replace `Right-click and drag do nothing special.` with `Drag selects text and Ctrl+C / ⌘C copies it (without a selection they quit); right-click does nothing special.` and add, after that bullet:

```markdown
- **Shell reflexes in the bar (decided 2026-09-16, `2026-09-16-tui-conveniences-design.md`).**
  `↑`/`↓` walk a 500-line command history kept in `state_dir()/history`; `Tab`
  completes `/verbs`, `/allow` answers and `/cancel #n`, and elsewhere still moves
  focus; `PageUp`/`PageDown`, `Shift+↑`/`↓` and `Ctrl+Home`/`End` scroll the log from
  the bar; on a focused log line `↑`/`↓` step between lines and any other printable
  key returns to the bar with the character; `Ctrl+L` drops finished blocks and keeps
  running ones and the footer (rule 6).
```

- [ ] **Step 6: CHANGELOG**

Under `## [Unreleased]`, add an `### Added` section **above** the existing `### Fixed`:

```markdown
### Added
- **The TUI bar behaves like a shell prompt.** `↑`/`↓` walk a command history kept
  across sessions (`state_dir()/history`, 500 lines); `Tab` completes `/verbs`,
  `/allow` answers and `/cancel #n`; `PageUp`/`PageDown`, `Shift+↑`/`↓` and
  `Ctrl+Home`/`End` scroll the log without leaving the bar; `↑`/`↓` on a focused log
  line step between lines and typing returns to the bar; `Ctrl+L` clears finished
  runs and keeps the coverage footer; a mouse selection is copied by `Ctrl+C`/`⌘C`
  instead of quitting. `/help` lists the keys.
```

- [ ] **Step 7: Commit**

```bash
git add src/proofpath/tui/app.py tests/test_tui_app.py docs/superpowers/specs/2026-09-10-proofpath-design.md CHANGELOG.md docs/superpowers/specs/2026-09-16-tui-conveniences-design.md docs/superpowers/plans/2026-09-16-tui-conveniences-plan.md
git commit -m "docs(tui): /help lists the keys; spec and changelog for the bar's shell reflexes"
```

---

## Self-review

- **Spec coverage:** 2.1 → Task 1–2; 2.2 → Task 3; 2.3 → Task 4 (bindings); 2.4 → Task 4 (`Line`); 2.5–2.6 → Task 5; 2.7 → Task 6; §4 tests are spread over the tasks' Step 1s; spec amendment and changelog → Task 6.
- **Names used across tasks:** `History.load/add/previous/next/reset/entries` (Task 1) match the calls in Task 2; `commands.complete(text, *, run_ids)` (Task 3) matches `_completions`; `Prompt._show` (Task 2) is reused by `action_complete` (Task 3); `RunLog`, `RunBlock.run`, `CommandBlock`, `NoteLine`, `CoverageFooter` are existing exports.
- **Known checks left to the implementer, called out in place:** `Input.__init__` typing (Task 2), `NEEDS_ARGUMENT` membership for the `/c` expectation and `check_action` return value (Task 3), event constructor fields and `scroll_down` step (Task 4), `_blocks` key, `make_report` helper and the selection API name (Task 5), PLAIN arrows (Task 6).
