# Permission hints — say what to type, where it is needed

**Status:** approved 2026-09-17; implementation follows this document. Amends spec
§7.1 (the consent prompt's last line on the TUI) and §13.1 (the "nothing to allow"
note) of `2026-09-10-proofpath-design.md`. No state word, verdict or honesty sentence
changes; every change is an appended *how-to* hint.

## 1. Why

A user pasted a Nature URL, got `IngestError: … (proofpath check - with the claim …)`,
typed `/allow`, and got `nothing to allow`. Three things were true at once: the hint
named a command-line form inside the TUI; the consent prompt, when it does appear,
shows the terminal's `[y] [a] [n] [never]` keys, which do nothing in the TUI; and a run
that ends with "browser not permitted" never says how to permit it. The mechanism
(`/allow`, `/config set`) exists; the user is not told about it where they need it.

## 2. Decisions (2026-09-17, with the user)

| Topic | Decision |
|---|---|
| Where | Every place a permission or an input rule stops a run: the consent prompt, the "not permitted" summaries, the page-not-a-post error, and `/allow` with nothing waiting. |
| Surfaces | Both. The TUI names its slash commands; the CLI and the Markdown report name `proofpath config set …`. A string shared by both surfaces names the *setting*, not a command. |
| Product rules | Unchanged. The hint is appended after the sentence that states the fact; the fact is never reworded. |

## 3. The four changes

1. **Consent prompt, TUI form.** `browser.prompt_text(host, status, *, answers=TERMINAL_ANSWERS)`
   takes its last line as a parameter. `TERMINAL_ANSWERS` is today's `[y] yes, once …` line
   (the terminal test's golden is unchanged). `TUI_ANSWERS` is
   `    Allow?  click a button, or type  /allow once   /allow always   /allow no   /allow never`.
   `widgets/prompt.py` passes `answers=TUI_ANSWERS`. `ask_terminal` is unchanged.
2. **"Not permitted" summaries.** `ui.py` gains two setting names, spelled the way
   `config set` takes them: `BROWSER_SETTING = "permissions.install_browser ask"` and
   `NETWORK_SETTING = "permissions.network allow"`.
   - TUI footer hint: `"{n} source(s) because the browser was not permitted — /config set permissions.install_browser ask"`.
   - CLI `_print_gate` `skipped` line and the report's `Skipped N source(s) …` line get
     ` — proofpath config set permissions.install_browser ask` appended.
   - `verify.NOT_ATTEMPTED` becomes `"not attempted (network not permitted; permissions.network allow turns it on)"` — one wording, both surfaces; the parenthesised fact stays first.
     > *2026-09-17, whole-branch review:* reverted. The summary sits in a fixed, elided column (cut in the TUI, 130-column rows on the CLI), so `NOT_ATTEMPTED` stays `"not attempted (network not permitted)"` and the setting rides on the once-per-run denied note instead: `fetch.network_permission` returns `"network: not permitted (<reason>) — permissions.network allow turns it on"`. The two settings live in `settings_hints.py` (a leaf; `fetch` cannot import `report`/`ui`), re-exported by both.
3. **Page, not a post.** `verify.NOT_A_POST` becomes
   `"{url} is a page, not a post: give the claim as text with the address inside it (paste it in the TUI, or proofpath check - on the command line)"`.
4. **`/allow` with nothing waiting.** The note becomes
   `"nothing to allow — the question appears under the stage a site blocks; answer it there, or with /allow once|always|no|never"`.

## 4. Tests

- `test_browser.py`: the terminal golden unchanged; a new test that `prompt_text(…, answers=TUI_ANSWERS)` ends with the TUI line and shares the first lines with the terminal form; the TUI line names all four `/allow` answers (derive from `commands.ALLOW_ANSWERS` in the test, so a fifth answer would fail it).
- `test_tui_rich.py`/`test_tui_app.py`: the mounted `PermissionPrompt` text contains `/allow once` and not `[y]`; `/allow` with nothing waiting notes the new sentence.
- `test_tui_footer`/report/cli tests that assert the skipped line: extend the expected strings; a report golden (`tests/data/verify-report-golden.json` or the Markdown golden) is regenerated only if it contains the sentence — check first and say so.
- `test_verify` (or wherever `NOT_A_POST`/`NOT_ATTEMPTED` are asserted): update the expected strings.

## 5. Docs

- Main spec §7.1: one line — on the TUI the last line of the prompt names the four `/allow` answers instead of the keys. §13.1: the `nothing to allow` wording.
- README, "Posts and the links inside them": the page-not-a-post sentence; the paragraph that describes the inline permission question adds "or type `/allow once` … from the bar".
- CHANGELOG `[Unreleased]` → `### Changed`: "Permission hints: …".
