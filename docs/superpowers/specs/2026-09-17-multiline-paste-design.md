# Multi-line paste — the bar takes a whole post

**Status:** approved 2026-09-17 (option A of two); implementation follows this
document. Amends spec §13.1 (the input bar) of `2026-09-10-proofpath-design.md`.
No product rule, state word or verdict changes.

## 1. Why

The promise on the bar is *paste a file path, a URL, or a claim*. A claim pasted
as a post is usually several lines. Textual's `Input._on_paste` keeps the first
line of a bracketed paste and drops the rest (`textual/widgets/_input.py`,
`splitlines()[0]`), so a three-paragraph post is checked as its first sentence and
the user is never told. The pipeline itself is fine: `verify.target_document` already
reads any string that is neither an address nor a file as pasted text, line breaks
included; `proofpath check -` on the command line proves it. Only the bar loses the
text.

## 2. Decisions (2026-09-17, with the user)

| Topic | Decision |
|---|---|
| Where | The TUI only. `proofpath check` keeps taking a file or `-`; the bar is the place a person pastes. |
| Confirmation | A multi-line paste is *held*, not run: the bar shows a one-line summary and Enter starts the run (option A). Running on paste alone (option B) was rejected — no way to drop a wrong paste. |
| Single line | Unchanged. A one-line paste lands in the bar as text and can be edited, as today. |
| Editing | A held paste is not editable in the bar. Any typed character, `escape`, or a history step drops it and the bar is empty again. Editing a post before checking it is a text-editor's job. |

## 3. Behaviour

**Held paste.** `Prompt` overrides `_on_paste`. Let `lines` be the pasted text's
`splitlines()` after stripping trailing whitespace from the whole text. If `lines`
has fewer than two entries, the base behaviour runs (first line inserted at the
cursor; a path pasted with its trailing newline still works). Otherwise the bar:

- keeps the full text, with only that trailing whitespace cut, in `Prompt.held: str | None`;
- replaces its value with a summary line
  `pasted {sep} {n} lines {sep} {m} chars {sep} enter to check, esc to drop`, where
  `sep` is the theme's separator glyph (`·` rich, `,` plain), handed to `Prompt` at
  construction the way `complete` is; `n` counts non-blank lines, `m` counts
  characters of the held text. No new glyph is added to `Glyphs`;
- moves the cursor to the end and stops the event.

The summary is the bar's own replacement and goes through `_show`, so
`on_input_changed` treats it like a history step rather than an edit.

**Enter.** `on_input_submitted` asks the bar for `prompt.take()`, which returns the
held text and clears the hold, or `None`. When there is held text, the submitted
line is the held text, not the summary. It reaches `_run_line` unchanged:

- in awaiting mode (`/check` or `/resolve` typed first) it is that verb's argument;
- otherwise it is an implicit `/check`, exactly as a one-line claim is today.

`commands.parse` needs no change: a held text that starts with `/` is a post that
starts with a slash, which `_split_verb` already handles for absolute paths and is
the one ambiguity we accept (a post whose first token is a bare verb name like
`/check` is not a real case).

**Dropping.** `Prompt.drop()` forgets the held text and empties the bar. A
printable key drops first and then lands in the empty bar (`_on_key`); any other edit
of the summary (backspace, delete, a cut) drops through `on_input_changed`, leaving
the bar empty; `action_leave_awaiting` (`escape`) drops; a history step or a
completion replaces the bar through `_show`, which drops. `take()` is the only way
the text leaves the bar whole.

**History.** The history file is newline-delimited, so a multi-line line cannot be
stored verbatim and a held paste is not added to history at all (`on_input_submitted`
skips `history.add` for it). The next `up` shows the previous real command.

**Run label.** `Scheduler.submit(target, command=…)` keeps a one-line label. The app
labels a held paste `/check pasted text` (the name `_pasted` already gives the
document), not `/check {target}`, so the run header, the footer notes and the
`#n /check …` log lines stay one line. A one-line claim keeps today's label. A
mirrored verb (`/resolve`, `/fetch`) given a held paste as its argument keeps the raw
argument and collapses whitespace only in its `CommandBlock` label.

**Empty and whitespace.** A paste of blank lines only has `lines == []` and takes the
base path, which inserts nothing.

## 4. Non-goals

- Editing a held paste in place. Drop it and paste again.
- A `TextArea` bar. History, completion and the awaiting tint are built on `Input`.
- Any change to `proofpath check -`, `--url`, or the CLI's file argument.

## 5. Tests

- `test_tui_rich.py` (or `test_tui_app.py`, wherever the bar is driven with a pilot):
  - a `Paste` event with three lines leaves the bar showing the summary, `held`
    equal to the text, and no run started;
  - Enter after it submits one run whose `target` is the full text with its line
    breaks and whose `command` is `/check pasted text`; the bar is empty after;
  - `/check` then a three-line paste, then Enter, runs `/check` with the full text
    and leaves awaiting mode;
  - a typed character after a held paste clears `held` and the bar holds only the
    character; a backspace clears `held` and leaves the bar empty; `escape` clears
    both;
  - a one-line paste, with and without a trailing newline, still inserts the line
    into the bar and holds nothing;
  - a held paste does not appear in the history file; the previous command does.
- `test_tui_commands.py`: unchanged — `parse` is not touched.
- `test_tui_rich.py`/plain: the summary uses the theme's `sep` (one assertion per
  theme, mirroring how other glyph tests are written).

## 6. Docs

- Main spec §13.1: one paragraph — the bar holds a multi-line paste and Enter runs
  it; what drops it.
- README, the TUI section that lists what the bar takes: "a post of several lines
  pastes whole; Enter checks it, Esc drops it".
- CHANGELOG `[Unreleased]` → `### Fixed`: "TUI: a multi-line paste is no longer cut
  to its first line; the bar holds it and Enter checks the whole text."
