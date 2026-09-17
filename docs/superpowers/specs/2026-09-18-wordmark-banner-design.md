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
