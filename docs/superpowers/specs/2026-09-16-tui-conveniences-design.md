# proofpath TUI — terminal conveniences — design

**Date:** 2026-09-16 · **Status:** approved by the author, implementation follows.
Amends spec §13.1 of `2026-09-10-proofpath-design.md`; everything not mentioned
here is unchanged.

## 1. Problem

The TUI has the honesty rules and the mouse/keyboard parity of §13.1, but not the
reflexes a shell user brings to any prompt: `↑` does not bring back the last
command, `Tab` does not finish `/che`, the log cannot be scrolled without leaving
the bar, `Ctrl+L` does nothing, and a selection made with the mouse is lost to
`Ctrl+C` quitting the app. Each absence is a small paper cut; together they make the
TUI feel like a form rather than a terminal.

## 2. Decisions

One rule governs all of them, the §13.1 rule: **nothing is mouse-only and nothing is
keyboard-only.** Every convenience below is a keyboard path to something the mouse
already reaches, or a shell habit with no mouse equivalent to begin with.

| Convenience | Keys | Where it lives |
|---|---|---|
| Command history | `↑` / `↓` in the bar | `tui/history.py` + `Prompt` |
| Completion | `Tab` in the bar, when the text starts with `/` | `Prompt` + a callback from the app |
| Scroll the log from the bar | `PageUp` / `PageDown`, `Shift+↑` / `Shift+↓`, `Ctrl+Home` / `Ctrl+End` | app bindings |
| Walk the log's lines | `↑` / `↓` while a line has focus | `Line` |
| Clear the log | `Ctrl+L` | app binding |
| Copy a selection | `Ctrl+C` / `⌘C` with a selection; without one they quit as before | app |

### 2.1 Command history

- **What is kept:** every submitted line, verbatim, including `/allow` answers and
  bare targets. Blank lines and a line equal to the previous entry are not added.
- **Where:** `paths.state_dir() / "history"`, one line per entry, UTF-8. `state_dir()`
  is `platformdirs.user_state_dir("proofpath")` with the override `PROOFPATH_STATE_DIR`,
  in the pattern of `config_dir()` and `cache_dir()`. The file holds the newest
  **500** lines; older ones are dropped on save. It is loaded once when the app
  mounts and rewritten after every submission (a few kilobytes; not worth batching).
  A missing or unreadable file is an empty history, never an error; a write failure
  is logged at debug level and ignored — the session continues in memory.
- **Navigation (shell semantics):** `↑` moves to the previous entry, `↓` to the next.
  Moving past the newest entry restores the **draft** — whatever was in the bar when
  the walk began. Any edit to the bar ends the walk and makes the current text the
  new draft. Submitting resets the cursor to the end.
- **Privacy:** the history is what the user typed; it can contain file paths and
  URLs, never a credential (secrets come from the environment, never from the bar).
  It is state, not cache, and is not deleted by `/cache clear`.

### 2.2 Completion

Only when the bar's text starts with `/`; otherwise `Tab` keeps its Textual meaning
(focus the next widget), which is how the keyboard reaches the log today.

| Text | Candidates |
|---|---|
| `/` + partial verb, no space | verbs from `commands.VERBS` with that prefix |
| `/allow ` + partial | `commands.ALLOW_ANSWERS` |
| `/cancel ` + partial | `#n` for every run that is not `done` or `cancelled` |
| anything else | none — `Tab` does nothing (and does not move focus) |

One candidate completes at once, with a trailing space when the verb takes an
argument. Several candidates: the first `Tab` inserts the first one, each further
`Tab` cycles to the next, wrapping; any edit ends the cycle. The run list comes from
the app through a callback, so `Prompt` stays free of scheduler knowledge. File
path completion is out of scope (the OS's own file picker and drag-and-drop paste a
path in one gesture).

### 2.3 Scrolling the log from the bar

App-level bindings that act on `RunLog` whatever has focus: `PageUp` / `PageDown`
one page, `Shift+↑` / `Shift+↓` one line, `Ctrl+Home` / `Ctrl+End` to the ends.
`Home`, `End`, `↑` and `↓` on their own stay the bar's (cursor and history). The log
still follows new output: the auto-scroll on every event is unchanged, so scrolling
up to read is a temporary excursion, the way it is in a terminal.

### 2.4 Walking the log's lines

`Line` (the base of `RunHeader`, `StageLine` and `FindingLine`) gains `↑` / `↓`:
focus the previous / next `Line` in document order among those currently displayed.
The first line's `↑` and the last line's `↓` do nothing. `Enter` and `c` keep their
meanings (§13.1). Any other **printable** key pressed while a line has focus moves
focus to the bar and types the character there, so `/` or the first letter of a
path picks up where the mouse left off without a `Tab` hunt. `c` is the one
exception and stays copy, as the spec already says.

### 2.5 Clearing the log

`Ctrl+L` removes every finished block — runs in `done` or `cancelled`, mirrored
command blocks that have ended, notes and help lines — and keeps running, queued
and waiting ones, which are still streaming and would otherwise reappear headless.
The footer is not touched: it is the latest finished run's coverage and **rule 6**
says it never goes away. Clearing does not forget anything the scheduler knows: a
cleared run keeps its number and its report. What it loses is its block, and a
summary is a line *in* that block (§11.1), so `/summarize` after `Ctrl+L` says
`nothing to summarize: the last finished run was cleared (ctrl+l)` instead of the
misleading "finish a /check first". `/cancel` is unaffected: only finished runs
are cleared and there is nothing left to cancel in them.

### 2.6 Text selection

Textual draws a selection on drag and offers `screen.copy_text` on `Ctrl+C`; the
app's priority `Ctrl+C → quit` shadows it. Resolution: `Ctrl+C` and `⌘C` copy when
`screen.get_selected_text()` is not `None`, and quit otherwise. `Ctrl+D` quits
unconditionally, as today. §13.1's "right-click and drag do nothing special" becomes
"drag selects text; right-click does nothing special".

### 2.7 `/help`

After the verbs, one line per key group, in the same dim note style, so the list is
also the reference:

```
↑ ↓ history · Tab completes /verbs · PageUp/PageDown Shift+↑↓ scroll the log
Tab into the log: ↑ ↓ walk lines, Enter opens, c copies · Ctrl+L clears finished runs
```

## 3. Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `paths.state_dir()` | where state lives, env-overridable | `platformdirs` |
| `tui/history.py` — `History` | load, append, cap, save; `previous(draft)`, `next()`, `reset()` | `pathlib` only |
| `Prompt` | `↑`/`↓` → history; `Tab` → completion via `complete: Callable[[str], list[str]]` | `History`, `commands` |
| `Line` | `↑`/`↓` neighbours; printable key → bar | `RunLog` query |
| `ProofpathApp` | scroll bindings, `Ctrl+L`, `Ctrl+C` decision, `/help` lines, the completion callback | scheduler state |

`History` is pure and has no Textual import, so it is unit-tested without a Pilot.

## 4. Testing

- `tests/test_history.py`: cap at 500, consecutive-duplicate rule, draft restore,
  edit-ends-walk, missing file, unwritable directory, round trip through a temp dir
  (`PROOFPATH_STATE_DIR` set by the existing `isolated` fixture pattern).
- `tests/test_tui_app.py`, Pilot-driven at 80×24 like the rest: `↑` restores the
  last submission; `/ch` + `Tab` → `/check `; `/allow ` + `Tab` cycles; `Tab` with a
  bare target moves focus; `PageUp` from the bar changes `RunLog.scroll_y`; `↓` from
  a `RunHeader` lands on its first `StageLine`; typing `/` on a focused line puts
  `/` in the bar; `Ctrl+L` keeps a running block and drops a done one; `Ctrl+C`
  with no selection quits.
- No network, no model: the fake scheduler as today.

## 5. Out of scope

Reverse search (`Ctrl+R`), file path completion, a key-help overlay, persistent
scroll position, multi-line input.
