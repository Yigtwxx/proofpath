# The wordmark — `ProofPath` in block letters, the raven leaves the RICH banner

**Status:** approved 2026-09-18; implementation follows this document. Supersedes
§3–§4 and the `RICH` half of §6–§7 of `2026-09-16-raven-pet-design.md` (the `PLAIN`
raven of its §5 stays exactly as it is). Amends spec §13.1 of
`2026-09-10-proofpath-design.md` (the `RICH` golden banner only).

## 1. Why

The Braille raven never read as a bird: eight rows of 2×4-dot cells are too coarse
for a silhouette, and the dots render as a speckle in most terminal fonts. The user
looked at it for two days and did not like it. A block-letter wordmark — the way
Gemini CLI, Claude Code and Codex open — says what the program is at a glance, in
the same crimson, in the same eight rows. The raven keeps its place on the website
and in the `PLAIN` theme, where six lines of ASCII still draw a recognisable bird.

## 2. Decisions (2026-09-18, with the user)

| Topic | Decision |
|---|---|
| Content | The word `ProofPath`, in the block-and-shadow style of the figlet font *ANSI Shadow* (`█` blocks, box-drawing `╗╔═╝║╚` shadow). Not all capitals: the two `P` are full height, the other letters are shorter, drawn as lowercase. No such font exists with lowercase, so the six lines are hand-drawn and kept verbatim as the source of truth. |
| Size | 66 columns × 6 rows. Two text rows under it: version + context, hint + commands. Eight rows in all — the raven banner's height, unchanged. |
| Colour | Two tones, the raven's own: blocks `█` in `light` (`#c4173a`, the site's `--crimson`), the shadow glyphs in `dark` (`#8f0f2b`, `--crimson-deep`). No gradient. No colour under `-q`/`NO_COLOR`, as before. |
| Spelling | The wordmark says `ProofPath`; the command, the package and every sentence keep saying `proofpath`. A logo's spelling is not the program's name. |
| `PLAIN` | Unchanged: the ASCII raven of the raven spec §5, in ANSI `red`, text beside it. Nothing in `banner.py` changes. |
| Narrow terminals | Below `RICH_WIDTH_FLOOR = 69` columns (66 + the 2-column right margin + 1) the `PLAIN` banner is drawn whatever the theme, as the raven did below 48. |
| Animation, stamp | Still none. `Banner.set_busy` / `flash` stay as no-ops. |

## 3. The wordmark

Six lines, verbatim. Only the characters `█ ╗ ╔ ═ ╝ ║ ╚` and the space appear;
trailing spaces are stripped. This is the source of truth; `tui/wordmark.py` carries
it as `WORDMARK` and the tests read it back.

```
██████╗                        █████╗██████╗               ██╗
██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║
██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗
██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗
██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║
╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝
```

Letter heights: `P` 6 rows; `r o a` 4 rows on the baseline; `f h` 6 rows (ascender);
`t` 5 rows. The implementer may nudge individual cells (the `r` hook, the `t` bar) if
a glyph reads badly once it is on screen in colour, but the width stays ≤ 66, the
row count stays 6, and the golden tests are updated with the final art in the same
change — the constant and the tests never disagree.

## 4. `RICH` layout

```
██████╗                        █████╗██████╗               ██╗
██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║
██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗
██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗
██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║
╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝
proofpath v0.4.4                                    academic · online · coreml
paste a file path, a URL, or a claim.                    /help  /config  /quit
```

- Rows 0–5 are the wordmark. Row 6 (`VERSION_ROW`) is `proofpath v{version}` at
  column 0 with the context right-aligned to end at column `width − 2`
  (`banner.RIGHT_MARGIN`, as today, never truncated). Row 7 (`HINT_ROW`) is the hint's
  prompt part at column 0 with its command list right-aligned the same way
  (`banner.split_hint`, as today). `TEXT_COLUMN = 0`.
- The widget keeps today's rule — when the hint plus its command list does not fit,
  it redraws with the command list dropped (it reads `HINT_ROW` back) — but with the
  text at column 0 the hint row is 38 + 21 + 3 = 62 columns at its shortest, below
  the 69-column floor, so in `RICH` the rule never fires; it stays as the guard it is.
- No ground row. Nothing grows with the width except the gap inside rows 6 and 7.
- Colour runs: `render` returns, per row, `(start, end, tone)` runs — `light` over
  every maximal run of `█`, `dark` over every maximal run of shadow glyphs; spaces
  belong to no run; rows 6 and 7 have none. The widget maps tones through the theme
  exactly as it does today; `wordmark.py` names no colour.

## 5. Modules

| Module | Change |
|---|---|
| `tui/wordmark.py` (new) | `WORDMARK: tuple[str, ...]` (6 lines), `BLOCK = "█"`, `SHADOW = "╗╔═╝║╚"`, `TEXT_COLUMN = 0`, `VERSION_ROW = 6`, `HINT_ROW = 7`, `RICH_WIDTH_FLOOR = 69`, `tones(line) -> tuple[Run, ...]`, `render(width, theme, *, version, context, hint) -> banner.Banner`. Pure: no Textual, no colour names. Delegates to `banner.render` for `PLAIN` and below the floor. |
| `tui/pet.py` | Deleted. `BITMAP`, `encode`, the Braille tables and its `render` go with it; nothing else imports them (`test_tui_rich.py` and the widget are re-pointed). |
| `tui/widgets/banner.py` | Imports `wordmark` instead of `pet`; `_draw`'s hint-drop rule reads `wordmark.HINT_ROW`; docstrings say wordmark, not raven. `set_busy`/`flash` stay. |
| `tui/theme.py` | `Theme.pet` is renamed `Theme.banner` (same keys `dark`/`light`, same values in both themes). The one reader is the widget. |
| `tui/banner.py` | Unchanged, except the comment on `HINT_LINE` that names `pet.HINT_ROW` / `test_tui_pet.py` now names `wordmark.HINT_ROW` / `test_tui_wordmark.py`. |
| `tui/app.py` | Unchanged in behaviour. Docstrings/comments that say "raven" for the `RICH` banner say "wordmark". |

## 6. Tests

- `tests/test_tui_wordmark.py` (replaces `test_tui_pet.py`): `WORDMARK` is 6 rows,
  max width 66, uses only `█`, the six shadow glyphs and the space, no trailing
  spaces; `tones(line)` on a known line gives the known runs and covers exactly the
  non-space cells; golden `RICH` renders at 100 and 80 columns (full lines, from
  the layout in §4 with `version="0.4.4"`, `context="academic · online · coreml"`,
  the app's `HINT`); at 68 columns and in `PLAIN` at 100 the result equals
  `banner.render(...)`; `VERSION_ROW`/`HINT_ROW` carry the text at `TEXT_COLUMN` with
  the right parts ending at `width − 2`; rows 6–7 have no tone runs; at every
  `RICH` width from 69 up the hint row ends with the command list (the drop rule
  never fires); the module imports no Textual.
- `tests/test_tui_rich.py`: the banner assertions that read `pet.HINT_ROW`,
  `pet.TEXT_COLUMN`, `pet.VERSION_ROW` read the `wordmark` names; the
  `GROUND_ROW`/`GROUND` assertion is removed (no ground); the "art cells are Braille"
  assertion becomes "art cells are block or shadow glyphs"; the 76/77-column
  hint-drop test becomes: at 69 columns the hint row still ends with the command
  list, and at 68 the drawn banner is the `PLAIN` one.
- `tests/test_tui_theme.py` (or wherever `Theme.pet` is asserted): `banner`.
- `tests/test_tui_banner.py`: unchanged (`PLAIN` is untouched).

## 7. Docs

- README: the `RICH` banner block near the top (the one with Braille) becomes the §4
  layout; the sentence after it that says "The bird is a raven … drawn in Braille
  cells" says the wordmark is drawn in block letters and the raven stays on the site
  and in the `plain` theme. The `plain` block and its paragraph are unchanged.
- Main spec §13.1: its golden block is the `PLAIN` banner — unchanged. One sentence
  after it: in `rich` the banner is the `ProofPath` wordmark of this spec.
- `docs/eval/tui-raven-rich.svg` is a record of a real run on 2026-09-15 and stays
  as it is; the README sentence that links it says "as of v0.4.1".
- `2026-09-16-raven-pet-design.md`: one line under **Status** — "`RICH` superseded
  2026-09-18 by `2026-09-18-wordmark-banner-design.md`; §5 (`PLAIN`) stands."
- CHANGELOG `[Unreleased]` → `### Changed`: "TUI: the `rich` banner is a `ProofPath`
  wordmark in block letters, crimson on deep crimson; the Braille raven is gone
  (the ASCII raven of the `plain` theme and the website's stay)."
- Site: untouched.

## 8. Amendment (2026-09-18, after v0.4.5, with the user)

Seen live, the user asked for three changes. They replace §4's layout and §5's
`Theme.banner`; everything else stands.

| Topic | Decision |
|---|---|
| Rule | A full-width rule (`─`, from column 0 to `width − 2`) is the banner's last row, under everything. It separates the banner from the log the way the raven's ground did. |
| Text beside the mark | The version/context row and the hint/commands row sit **to the right of the wordmark**, on the mark's rows 2 and 3, starting at column 68 (`BESIDE_COLUMN` = 66 + 2), with their right parts aligned to `width − 2` — when they fit. When they do not (any terminal narrower than `68 + longest text row + 2`, which is 129 columns with the default hint, 113 once the widget drops the command list — the version row, 43 columns, is then the longer one), the two rows go **under** the mark as in §4. The rule is the last row in both cases: 7 rows beside, 9 rows under. |
| Gradient | The blocks run light→dark from left to right in five bands of the mark's 66 columns (`band = min(4, column * 5 // 66)`): `#e0455f`, `#d02f4c`, `#c4173a` (`--crimson`), `#a91330`, `#8f0f2b` (`--crimson-deep`). The shadow glyphs and the rule are one tone darker than the end of the gradient: `#6b0a20`. |
| Floor | `RICH_WIDTH_FLOOR = 69` is unchanged (below it the `plain` banner). The beside/under choice is made by `render` from the actual text lengths, never by the widget. |

**Layout, beside (width 140):**

```
██████╗                        █████╗██████╗               ██╗
██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║
██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗  proofpath v0.4.5                        academic · online · coreml
██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗  paste a file path, a URL, or a claim.            /help  /config  /quit
██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║
╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

**Layout, under (width 80):** §4's eight rows, then the rule as row 8.

**Tones.** `tones(line)` now needs the column, so the run tone names are
`g0`…`g4` (gradient bands) and `shadow`; a block run is split where the band
changes, a shadow run is never split by bands. The rule row is one `shadow` run
over its whole length. Text rows and the text part of a beside row carry no run.
`banner.Tone` widens from `Literal["dark", "light"]` to `str` (type only; `banner.py`
otherwise unchanged; its own runs still say `"light"`, which `PLAIN.banner` maps).

**`Theme.banner`** is a mapping with keys `g0`…`g4`, `shadow`, `light`, `dark`:
`RICH` as above (`light` = `#c4173a`, `dark` = `#8f0f2b` kept for anything that
still asks); `PLAIN` maps every key to `ui.PET_COLOUR`.

**Module surface.** `VERSION_ROW`/`HINT_ROW` stop being constants: `rows(width, *,
version, hint) -> Rows` (`Rows(version: int, hint: int, rule: int, beside: bool)`)
says where `render` put things for that width and those texts, and the widget's
hint-drop rule reads `rows(...).hint` (replacing `hint_row`). `BESIDE_COLUMN = 68`,
`RULE = "─"`, `BANDS = 5`, `GRADIENT_TONES = ("g0", "g1", "g2", "g3", "g4")`.

**Tests** (replace the §6 goldens): beside golden at 140, under golden at 80 with the
rule; `rows()` at 128 (under), 129 (beside), 112/113 with the commands dropped;
the rule row is `width − 2` long and one `shadow` run; every block cell's tone is
the band of its column; shadow cells are `shadow`; the widget test at 140 columns
reads the hint beside the mark and at 80 under it.

**Docs.** README's block becomes the under layout at 100 with the rule; CHANGELOG
`[Unreleased]` → `### Changed`: "TUI: the wordmark runs light→dark, the banner ends
in a rule, and on a wide terminal the version and hint sit beside the mark."

### 8.1 The raven leaves `PLAIN` too (2026-09-18, same session)

Shown the README, the user asked for the ASCII raven to go as well: "yazı olsun hep"
— text everywhere. So:

- **`PLAIN` draws the same wordmark in ASCII.** The six lines are transliterated cell
  for cell — `█`→`#`, `╗ ╔ ╝ ╚`→`+`, `═`→`-`, `║`→`|`, and the rule `─`→`-` — so the
  layout, the rows and the runs are identical in both themes; only the glyphs and
  the colours differ (`PLAIN.banner` maps every tone to ANSI `red`). The mark is
  seven-bit ASCII in `plain`, which is what that theme is for.
- **Below `RICH_WIDTH_FLOOR` (69) both themes draw no mark**: the version row, the
  hint row and the rule, three rows, text at column 0. Nothing is ever truncated.
- **`banner.py` shrinks** to what both themes share: `Banner`, `Run`, `Tone`,
  `RIGHT_MARGIN`, `right_align`, `split_hint`. `ART`, `GROUND`, `GROUND_LINE`,
  `HINT_LINE`, `TEXT_INDENT`, `BODY_WIDTH_FLOOR`, `HINT_GAP`'s raven-specific users
  and `banner.render` go; `wordmark.render(width, theme, …)` is the only renderer
  and picks the glyph table by `theme.name`. `test_tui_banner.py` tests the shared
  helpers and the `plain` transliteration.
- **README** shows the `rich` block (under layout at 100 columns, with the rule) and
  the `plain` block (the same, in ASCII); the sentence between them says the plain
  theme draws the same mark in ASCII. The main spec §13.1 golden block becomes the
  `plain` ASCII wordmark banner (same `v0.1.0` / `academic . offline . mps` values).
  `2026-09-16-raven-pet-design.md` is superseded in full; the website's raven is
  not touched.

## 9. The bar's frame in the same gradient (2026-09-18, same session)

The input bar's rounded border was the *next run's accent* (§13.1's "the prompt in the
current run's accent"). Seen next to the wordmark it reads as an unrelated blue box.
The user asked for it to be crimson too, "sağdan sola koyudan açık" — dark at the
right, light at the left: the wordmark's direction.

| Topic | Decision |
|---|---|
| What | In `rich` at panel width, the bar is framed by a drawn frame, not a CSS border: a top row `╭───…───╮`, a bottom row `╰───…───╯`, and a `│` at each end of the input row. Same three rows as the border took; the input keeps exactly the width it has today. |
| Colour | The same five bands as the wordmark, over the frame's own width: column `c` of a `w`-wide frame is `GRADIENT_TONES[min(4, c * 5 // w)]`, so the left corner is `g0` (light) and the right corner `g4` (dark). The left `│` is `g0`, the right `│` is `g4`. Without colour (`-q`, `NO_COLOR`) the frame is drawn in the terminal's default colour, as the border was. |
| Accent | The caret keeps the next run's accent and the awaiting tint stays; only the frame stops following the accent. The run panels' accent borders are unchanged. |
| Narrow / `plain` | Below the panel floor the frame's rows and edges are hidden and the bar is one flat row, as the border was; `plain` never draws a frame, as today. |
| Where | `widgets/prompt.py`: `PromptFrame` (the container) and `FrameEdge` (the top/bottom rows). `wordmark.band(column, width) -> str` is the one place the band arithmetic lives; `tones()` uses it with the mark's width, the frame with its own. |

## 10. Slash suggestions above the bar (2026-09-18, same session)

Typing `/` should show what can be typed, the way Claude Code does: a list above
the bar that narrows with every character (`/h` → `/help`) and disappears when the
line stops being a command. Today the same knowledge exists only as `Tab`
completion, which shows one candidate at a time and only on request.

| Topic | Decision |
|---|---|
| When | Shown whenever the bar's text starts with `/` and `commands.complete(text)` has at least one candidate; hidden otherwise (empty bar, a claim, a path, a `/verb` with an argument that completes to nothing). It never covers the log: `#bottom` grows by the list's rows and the log shrinks. |
| What | One row per candidate, in `VERBS` order, three columns: the completion as `Tab` would type it (`/check`), the verb's placeholder (`paste a file path or URL`, empty for verbs without one), and a one-line description (`commands.DESCRIPTIONS`, new). `/allow` and `/cancel` candidates (`/allow once …`, `/cancel #1 …`) list the answers/run ids with no placeholder; their description is the verb's. |
| Selection | One row is the *selected* candidate, drawn with the bar's accent tint as its background. `Tab` cycles it (today's cycle, now visible), `↑`/`↓` move it while the list is showing (they are the history walk only when the list is hidden). Moving the selection does **not** change the bar's text; `Tab` does (today's behaviour). `Enter` runs whatever is in the bar — the list never runs anything by itself. |
| Look | `rich`: the rows sit inside the bar's frame gap, i.e. directly above the frame's top edge, in the muted tone with the completion in the text tone; the selected row is tinted (`TINT_ALPHA`, the accent). `plain`: the same rows, the selected one marked with `>` in column 0. Nothing is truncated; a row longer than the width wraps as any line does. |
| Where | `widgets/suggestions.py`: `Suggestions(Static)` fed `(candidates, selected)` by the app on every `Input.Changed` and every `Tab`/`↑`/`↓`; `commands.DESCRIPTIONS: Mapping[str, str]` (one sentence per verb, ≤ 40 characters, no trailing period): `check` "verify a file, URL or pasted text", `resolve` "does a cited reference exist", `fetch` "read one URL, DOI or arXiv id", `config` "settings panel, or show/set/check", `cache` "list or drop cached sources", `allow` "answer the permission question", `summarize` "a paragraph over the last run", `cancel` "stop a run", `help` "commands and keys", `quit` "leave". |
| Not in scope | Fuzzy matching (prefix only, as `complete` does today); descriptions for file paths; a popup for anything that is not a slash command. |

## 11. Typing always reaches the bar (2026-09-18, same session)

The user does not want to aim at the bar: a click anywhere on the screen, then
typing, should type into the bar. Today a click on a finding line moves focus to
the line and printable keys are already handed back to the bar (`Line.on_key`), but
a click on the log's empty background focuses the log itself (`VerticalScroll` is
focusable by default) and the keys go nowhere.

| Topic | Decision |
|---|---|
| The log is not focusable | `RunLog.can_focus = False`. Keyboard scrolling is the app's own priority bindings (`shift+↑↓`, `PageUp/Down`, `ctrl+Home/End`) and never needed the log's focus. A click on the log's background, the banner, the footer, the rule, a run's panel border or the frame leaves focus where it is — on the bar. |
| Every printable key is typing | An app-level `on_key`: when the focused widget is not the bar and the key is printable, the key is stopped, focus is set to the bar (on the screen, synchronously, as `Line.on_key` does) and the key is posted to the bar afresh. Exceptions: `c` on a focused `Line` (its copy binding, spec §13.1) and any key the focused widget binds itself (a `Button` binds only `enter`, which is not printable anyway; `space` on a button is typing). `Line.on_key` is folded into this one handler — one rule, one place. |
| Clicks on lines and buttons | Unchanged: a click on a finding line still focuses it (so `Enter` expands it and `↑`/`↓` walk the lines), a click on a permission button still answers. The first printable key after that goes to the bar and brings focus with it. |
| Not in scope | Mouse text selection (Textual's own, untouched); a click that *places the caret* at a column inside the bar (the bar's own behaviour, untouched). |

## 12. `/config` is a settings panel (2026-09-18, same session; detailed the same day)

`/config` prints the TOML file. The user wants what Claude Code's `/config` is: the
settings as rows you move through with the arrow keys and change in place, with
what cannot be changed that way shown above them.

### 12.1 What opens

`/config` with no argument mounts a **`ConfigPanel`** block in the log — inline,
never a modal (spec §13.1), in the next run's accent like a `CommandBlock` — and
gives it focus. `/config show`, `path`, `set K V` and `check` are unchanged and stay
the scriptable path; the panel writes through the very same `library.config_set`,
so the two can never disagree about what a value means.

The panel wears the accent of the run that is current when it opens (the same as
`CommandBlock`), not the next one — it is a record, not a run.

### 12.2 The panel, `rich`, 100 columns

```
 ╭ /config ─────────────────────────────────────────────────────────────────────────╮
 │  config   ~/Library/Application Support/proofpath/config.toml                     │
 │  key      GROQ_API_KEY · found in ~/Projects/proofpath/.env                       │
 │                                                                                   │
 │  permissions                                                                      │
 │ › install_browser   ask   allow   deny                                            │
 │     step 3 of the fetch ladder: a ~280 MB browser engine, spec §7.1               │
 │   network           ask   allow   deny                                            │
 │     every fetch and every provider call; deny = offline                           │
 │                                                                                   │
 │  fetch                                                                            │
 │   respect_robots    true   false                                                  │
 │     honour robots.txt when fetching                                               │
 │                                                                                   │
 │  judge                                                                            │
 │   provider          groq   gemini   ollama                                        │
 │     switching a provider also sets its model, base_url and api_key_env            │
 │   model             openai/gpt-oss-120b                                           │
 │   base_url          https://api.groq.com/openai/v1                                │
 │   api_key_env       GROQ_API_KEY                                                  │
 │                                                                                   │
 │  contact                                                                          │
 │   email             (unset) · optional, for the Crossref / OpenAlex polite pools  │
 │                                                                                   │
 │  ↑↓ row   ←→ change   enter edit text   backspace default   esc close             │
 ╰───────────────────────────────────────────────────────────────────────────────────╯
   permissions.install_browser = allow  (~/Library/Application Support/proofpath/config.toml)
```

- **Head (read-only).** `config` + the file's path, with `(not written yet, showing
  defaults)` when it does not exist; `key` + the name in `judge.api_key_env` and where
  a value for it was found — `found in <path>` / `found in the environment` /
  `not set — put GROQ_API_KEY=… in <first dotenv path>`. The value itself is never
  shown (`secrets.ApiKey` never prints it). The `key` line re-resolves after any
  change to `judge.*`.
- **Sections** in the config's own order: `permissions`, `fetch`, `judge`, `contact`.
  A section name is a muted heading; its rows are indented under it.
- **Choice rows** (`←`/`→` cycle, wrapping): `install_browser` and `network` over
  `ask allow deny`; `respect_robots` over `true false`; `judge.provider` over the
  known providers (`judge._PROVIDERS`, sorted). The current value is a **badge** in
  the panel's accent (`theme.badge`, the same convention the findings use), the
  others muted. One dim *meaning* line under each choice row (text above).
- **Text rows** (`Enter` edits in place): `judge.model`, `judge.base_url`,
  `judge.api_key_env`, `contact.email`. The value is shown as text; empty shows
  `(unset)` plus, for `email`, the dim note in the mock. `Enter` replaces the value
  with an `Input` holding the current text; `Enter` again writes it (`config_set`),
  `Esc` cancels the edit and restores the row. While an edit is open, keys go to
  that `Input` (the §11 forwarding does not apply: the focused widget *is* an
  input).
- **Selection.** One row is selected; `›` (`>` in `plain`) at column 1. `↑`/`↓` move
  over rows only (headings and meaning lines are skipped); the ends stay put.
- **Reset.** `Backspace` on a row writes the row's default (the dataclass default;
  for `judge.provider` that is `groq` with its four presets) and says so in the
  note.
- **Every write is immediate.** `config_set` runs on the change; on success the row
  redraws, the app reloads its config (`_reload_config`), the banner's context
  redraws (`online`/`offline` follows `permissions.network` at once), and one note
  line under the panel records what was written — every pair `config_set` returns,
  so a provider switch lists four lines. On `ConfigError`/`JudgeError` the row keeps
  its value and an error line appears under the panel instead.
- **Legend.** The last line inside the panel names the keys (mock above); `plain`
  spells the arrows (`up/down`, `left/right`).
- **Closing.** `Esc` (outside a text edit) collapses the panel to one line —
  `config  install_browser=ask  network=allow  respect_robots=true  judge=groq` —
  and focuses the bar. Losing focus any other way (a printable key going to the
  bar under §11, a click on a finding) collapses it the same way. A collapsed
  panel is a record, like a finished block; `/config` again opens a fresh one.

### 12.3 `plain`

The same rows in ASCII: `>` marks the selected row, the current value is written
`[ask]` and the others bare, no border (the block's `box` is `none`), the legend
in words. Nothing is truncated.

### 12.4 Not in scope

Adding settings the config does not have; editing the `.env` file; validating a
model name against the provider (that is `/config check`, unchanged).
