# Permission Hints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wherever a permission or an input rule stops a run, tell the user what to type — `/allow …`, `/config set …`, `proofpath config set …`, or "paste the claim as text" — per `docs/superpowers/specs/2026-09-17-permission-hints-design.md`.

**Architecture:** Strings only, appended after the sentence that states the fact. `browser.prompt_text` grows an `answers` keyword so the TUI prompt names `/allow` instead of the terminal keys; `ui.py` names the two settings once; the TUI footer, the CLI gate line and the report line each spell their own `config set` form; `verify.NOT_A_POST` / `NOT_ATTEMPTED` and the app's `nothing to allow` note are reworded.

**Tech Stack:** Python 3.10+, pytest; Textual only in the widget/app tests.

## Global Constraints

- No state word, verdict or honesty sentence is reworded; hints are appended after ` — ` (TUI/CLI/report lines) or inside the existing parentheses (`NOT_ATTEMPTED`, `NOT_A_POST`).
- Exact strings (copy verbatim):
  - `TUI_ANSWERS = "    Allow?  click a button, or type  /allow once   /allow always   /allow no   /allow never"`
  - `ui.BROWSER_SETTING = "permissions.install_browser ask"`, `ui.NETWORK_SETTING = "permissions.network allow"`
  - footer: `f"{n} source(s) {BROWSER_SKIPPED_REASON} — /config set {ui.BROWSER_SETTING}"`
  - CLI `skipped`: `f"{gate.skipped} source(s) because the browser was not permitted — proofpath config set {ui.BROWSER_SETTING}"`
  - report: `f"Skipped {n} source(s) {BROWSER_SKIPPED_REASON} — proofpath config set {ui.BROWSER_SETTING}."`
  - `verify.NOT_ATTEMPTED = "not attempted (network not permitted; permissions.network allow turns it on)"` — build it with `ui.NETWORK_SETTING`.
  - `verify.NOT_A_POST = "{url} is a page, not a post: give the claim as text with the address inside it (paste it in the TUI, or proofpath check - on the command line)"`
  - app note: `"nothing to allow — the question appears under the stage a site blocks; answer it there, or with /allow once|always|no|never"` — build the `once|always|no|never` part from `commands.ALLOW_ANSWERS`.
- Type annotations everywhere; `uv run ruff check`, `uv run ruff format --check`, `uv run mypy` clean; `uv run pytest` green; no network in tests. English.
- **Do not commit.** Stage with `git add` when done.

---

### Task 1: The hints (code + tests)

**Files:**
- Modify: `src/proofpath/browser.py` (`prompt_text`, `TERMINAL_ANSWERS`, `TUI_ANSWERS`), `src/proofpath/ui.py` (two settings), `src/proofpath/verify.py` (`NOT_A_POST`, `NOT_ATTEMPTED`), `src/proofpath/cli.py` (`_print_gate`), `src/proofpath/report.py` (the `Skipped …` line), `src/proofpath/tui/widgets/footer.py` (`_hints`), `src/proofpath/tui/widgets/prompt.py` (`answers=TUI_ANSWERS`), `src/proofpath/tui/app.py` (`_allow`'s note)
- Tests: `tests/test_browser.py`, and every test that asserts one of the changed strings (`grep -rn "browser was not permitted\|not attempted (network\|is a page, not a post\|nothing to allow\|\[y\] yes, once" tests` finds them; goldens under `tests/data` included)

**Interfaces:**
- Produces: `browser.prompt_text(host: str, status: int | None, *, answers: str = TERMINAL_ANSWERS) -> str`; `browser.TERMINAL_ANSWERS`, `browser.TUI_ANSWERS`; `ui.BROWSER_SETTING`, `ui.NETWORK_SETTING`.

- [ ] **Step 1: Tests first.** In `tests/test_browser.py` keep `test_prompt_text_matches_spec` as is (terminal golden) and add:

```python
def test_prompt_text_tui_form_names_the_four_allow_answers() -> None:
    from proofpath.tui import commands

    terminal = bw.prompt_text("nature.com", 403).splitlines()
    tui = bw.prompt_text("nature.com", 403, answers=bw.TUI_ANSWERS).splitlines()
    assert tui[:-1] == terminal[:-1]
    assert tui[-1] == bw.TUI_ANSWERS
    for answer in commands.ALLOW_ANSWERS:
        assert f"/allow {answer}" in tui[-1]
    assert "[y]" not in tui[-1]
```

In `tests/test_tui_app.py` (or `test_tui_rich.py`, wherever a `PermissionPrompt` is mounted in an existing test) add an assertion that the rendered prompt text contains `/allow once` and not `[y] yes`; and a test that `/allow` with nothing pending notes a line starting `nothing to allow — ` that contains `/allow once|always|no|never`. Update every other test the grep finds to the new strings. Run the touched test files: they fail on the old strings.

- [ ] **Step 2: Implement** each string per Global Constraints. In `browser.py`:

```python
#: The consent prompt's last line on a terminal: the keys ``ask_terminal`` reads.
TERMINAL_ANSWERS = (
    "    Allow?  [y] yes, once   [a] always (save to config)   [n] no   "
    "[never] never ask again"
)
#: The same line in the TUI, where the keys do nothing: the buttons, or the
#: ``/allow`` answers they are worth (spec section 13.1).
TUI_ANSWERS = (
    "    Allow?  click a button, or type  /allow once   /allow always   /allow no   /allow never"
)


def prompt_text(host: str, status: int | None, *, answers: str = TERMINAL_ANSWERS) -> str:
```

and the last list element becomes `answers`. `widgets/prompt.py`: `self.question = prompt_text(host, status, answers=TUI_ANSWERS)`. `app.py::_allow`: the note built as `"nothing to allow — the question appears under the stage a site blocks; answer it there, or with /allow " + "|".join(commands.ALLOW_ANSWERS)`. Footer, CLI, report, verify per the constraints; `verify.py` imports `ui` already or gains `from proofpath import ui`.

- [ ] **Step 3: Verify.** `uv run pytest -o addopts="" -q 2>&1 | tail -1`; `uv run ruff check`; `uv run ruff format --check`; `uv run mypy`. If a golden under `tests/data` or `docs/eval` contains a changed sentence, regenerate only the golden the tests read (say which) — never the eval records.

- [ ] **Step 4: Stage.** `git add` every touched file. No commit.

### Task 2: Docs

**Files:** `README.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-09-10-proofpath-design.md` (§7.1 and §13.1 one-line notes).

- [ ] README "Posts and the links inside them": after the code block, the sentence that explains pages vs posts (add if absent): "A page is not a post: paste the claim as text with the address inside it, and the page is fetched as that claim's source." In the paragraph on the inline permission question (around "asked **inline, under the stage that hit the wall**"), append: " — or, from the bar, `/allow once`, `/allow always`, `/allow no`, `/allow never`."
- [ ] CHANGELOG `[Unreleased]` → `### Changed`: "**Permission hints.** The TUI's consent question names the `/allow` answers instead of the terminal's keys; a run that skipped sources because the browser or the network was not permitted says which setting turns it on (`/config set …` in the TUI, `proofpath config set …` elsewhere); a page given where a post was expected says to paste the claim as text; `/allow` with nothing waiting says where the question appears."
- [ ] Main spec: under §7.1's prompt block add "> *2026-09-17:* on the TUI the last line reads `Allow?  click a button, or type  /allow once   /allow always   /allow no   /allow never`; the keys are the terminal's." Under §13.1 where `/allow` is described (or at its end): "> *2026-09-17:* `/allow` with no question waiting answers `nothing to allow — the question appears under the stage a site blocks; answer it there, or with /allow once|always|no|never`."
- [ ] `uv run ruff format --check` from the repo root; `git add` the three files. No commit.

---

### Task 3: The `loading models …` note does not outlive the loading

Added 2026-09-17 after a live run: the note is emitted once before the models load and
is appended to the run block as a permanent line, so a finished run still reads
"loading models …" under its `Verifying` stage. On the CLI the line scrolls by in a
log and is right; in the TUI it must go when the loading is over.

**Files:**
- Modify: `src/proofpath/events.py` (`Note` gains `transient: bool = False`), `src/proofpath/verify.py` (`emit(Note(LOADING_MODELS, transient=True))`), `src/proofpath/tui/widgets/run_block.py` (`NoteLine` for a transient note is removed on the next `StageEnd`, `Emitted` or `Note` of the same run)
- Tests: `tests/test_verify.py` (`Note(LOADING_MODELS, transient=True) in events`), `tests/test_tui_app.py` or `tests/test_tui_rich.py` (a run that pushes `Note(LOADING_MODELS, transient=True)` then `StageEnd("Verifying", …)` has no `NoteLine` with that text left; a non-transient `Note` stays)

**Interfaces:**
- Produces: `events.Note(text: str, transient: bool = False)` — frozen dataclass, positional `text` unchanged so every existing `Note("…")` still works and `Note(x) == Note(x)` still holds.

- [ ] **Step 1: Tests first.** `tests/test_verify.py:1037`: `assert Note(LOADING_MODELS, transient=True) in events`. In the TUI test file that already drives a run with `scheduler.push(run, …)` events (see `test_tui_rich.py` around the `StageStart("Verifying", "coreml")` pushes), add:

```python
async def test_a_transient_note_leaves_when_its_stage_ends() -> None:
    app, schedulers = rich_app()
    async with app.run_test(size=SIZE) as pilot:
        run = await a_run(pilot, schedulers)
        scheduler = schedulers[0]
        scheduler.push(run, StageStart("Verifying", "coreml"))
        scheduler.push(run, Note("loading models …", transient=True))
        scheduler.push(run, Note("kept"))
        await pilot.pause()
        texts = [w.render().plain for w in app.query(NoteLine)]
        assert any("loading models" in t for t in texts) and any("kept" in t for t in texts)
        scheduler.push(run, StageEnd("Verifying", "coreml", "1 claim", 0.1))
        await pilot.pause()
        texts = [w.render().plain for w in app.query(NoteLine)]
        assert not any("loading models" in t for t in texts)
        assert any("kept" in t for t in texts)
```

(Adapt the helper names — `rich_app`, `a_run`, `SIZE`, the `StageEnd` signature — to what the file actually defines; import `NoteLine` from `proofpath.tui.widgets.run_block` and `Note`/`StageStart`/`StageEnd` from `proofpath.events`.) Run: RED (`TypeError: unexpected keyword 'transient'`).

- [ ] **Step 2: Implement.** `events.Note`: add `#: A line that describes something in progress and is taken back when it is over (the TUI removes it; the CLI logs it).` `transient: bool = False`. `verify.py:1015`: `emit(Note(LOADING_MODELS, transient=True))`. `run_block.py`: keep `self._transient: list[NoteLine] = []`; on a `Note` with `event.transient`, append the created `NoteLine` to it; at the top of the `StageEnd`, `Emitted` and non-transient `Note` branches call `self._settle_transient()` which `remove()`s each widget in the list and clears it. (A `Progress` event does not settle it: progress can tick while a model is still being downloaded.) Also settle on the run's terminal states if the block has a hook for "done/failed/cancelled" (`grep -n "def finish\|def done\|state ==" run_block.py`); if not, `StageEnd` covers the real case.

- [ ] **Step 3: Verify.** `uv run pytest -o addopts="" -q 2>&1 | tail -1`; `uv run ruff check`; `uv run ruff format --check`; `uv run mypy`.

- [ ] **Step 4: Stage.** `git add` the touched files. No commit.

- [ ] **Step 5 (docs, same task):** CHANGELOG `### Fixed` under `[Unreleased]`: "**`loading models …` no longer outlives the loading in the TUI.** The note is transient and is taken back when the `Verifying` stage ends; a finished run no longer looks like it is still loading."
