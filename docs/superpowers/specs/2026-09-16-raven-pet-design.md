# The raven — the TUI pet, redrawn

**Status:** approved 2026-09-16; implementation follows this document.
Amends spec §13.1 of `2026-09-10-proofpath-design.md` and §3 of
`2026-09-15-tui-v2-design.md` (the pet only; nothing else in either changes).

## 1. Why

The landing page (`2026-09-16-landing-page-design.md`) built the product's picture
around Odin's ravens: Huginn flies to the source, Muninn brings the passage back.
The terminal still drew a ferret. One product, one animal: the ferret goes, a raven
takes its place in both themes.

## 2. Decisions (2026-09-16, with the user)

| Topic | Decision |
|---|---|
| Animal | A perched raven in profile, head top-right, tail bottom-left, standing on a ground line that runs to the right edge of the terminal (what is left of the ferret's "path"). |
| Style | Pixel art. `RICH` draws it in **Braille cells** (U+2800–U+28FF, 2×4 dots per cell), which is how the sibling project's spider is drawn and gives an 8-row banner the detail of a 30-pixel-tall bitmap. `PLAIN` draws a small pure-ASCII raven in the same pose. |
| Size | `RICH`: 15 columns × 8 rows, fixed. The two text lines sit to the right of the bird, not under it. `PLAIN`: 6 lines. |
| Colour | Two tones per bitmap: dark `#` = `#8f0f2b` (the site's `--crimson-deep`), light `+` = `#c4173a` (`--crimson`). A Braille cell takes one colour, the majority tone of its set dots. `PLAIN` draws the bird in ANSI `red`. No colour under `-q`/`NO_COLOR` (the `coloured` flag, as before). |
| Stamp | Dropped. There is no `[PROOF]` stamp and no trail; the ground line is the only thing that grows with the width. May return later. |
| Animation | Dropped. No blink, no busy eyes, no tail. May return later; the eye is a hole in the bitmap so a future frame can fill it. |
| Originality | The pose follows a stock pixel-art raven the user chose as a reference; the pixels are our own (different body, beak, tail, feet, ground). Nothing is traced. |
| "8-bit" | No caption, no pixel font. The text lines are ordinary text. |

## 3. The bitmap

29 × 30 pixels, three symbols: `#` dark, `+` light, `.` empty. This is the source of
truth; `tui/pet.py` carries it verbatim as a constant and the tests read it back.

```
................#####........
...............#######.......
..............########.......
..............####..###+.....
..............#########+++...
..............########+++++..
..............+#######++++...
.............++++####...++...
............+++++####........
............++++++###........
...........+++++++###........
...........+++++++###........
..........++++++++###........
..........++++++++###........
.........+++++++++###........
.........++++++++####........
........+++++++++###.........
........++++++++####.........
.......+++++++++###..........
.......++++++++####..........
......++++++++####...........
......+++++++####............
.....++++++#####.............
....++++++#####..............
...+++++#####+###............
..++++######..+.#............
.########+++..#.#............
########..++..#.#............
#####.....++..#.#............
##.##.....+++##+##+++++++++++
```

The eye is the `..` hole at row 3, columns 18–19. The last row is the bird's feet on
its ground; the ground continues to the right as `⠒` (dots 2 and 5, the middle pair)
in the light tone.

## 4. `RICH` layout

Braille cell (col `c`, row `r`) covers pixels x ∈ [2c, 2c+1], y ∈ [4r, 4r+3]; dot bits
in the standard order (1,2,3,7 down the left, 4,5,6,8 down the right). 29 columns pad
to 30; 30 rows are 8 cell rows (the last cell row holds pixel rows 28–29 plus two
empty). A cell with no dots is a space; trailing spaces are stripped. Every glyph is
East Asian width `N`, so the art is one cell wide in every locale — safer than box
drawing (the CJK rule in `theme.detect` stays, for the panels).

```
       ⣴⣿⠿⣷⣀
      ⢀⣿⣿⣿⡿⠿⣿⠂
     ⢠⣿⣿⣿⣿⡇
    ⢠⣿⣿⣿⣿⣿⡇       proofpath v0.4.1                  academic · online · coreml
   ⢠⣿⣿⣿⣿⣿⡟        paste a file path, a URL, or a claim.  /help  /config  /quit
  ⣠⣿⣿⣿⣿⡿⠋
⣠⣾⣿⣿⠿⣿⠉⡏⡇
⠛⠙⠃  ⠛⠒⠓⠓⠒⠒⠒⠒⠒⠂⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒
```

- Rows are numbered 0–7. The version line is row 3, the hint line row 4, both starting
  at column 18 (`TEXT_COLUMN`). The context and the command list are right-aligned to
  end at column `width − 2`, exactly as today (`banner.RIGHT_MARGIN`, never truncated).
- The hint keeps today's `RICH` rule: when the hint plus its command list does not fit
  the line, the widget redraws with the command list dropped.
- Row 7 (the feet) is extended with `⠒` from the end of the bird's row to
  column `width − 2`.
- Below `RICH_WIDTH_FLOOR = 48` the `PLAIN` pet is drawn whatever the theme, as today.
- Colour spans: `render` returns, per row, a list of `(start, end, tone)` runs where
  `tone` is `"dark"` or `"light"`; the ground run is `"light"`. The widget maps them
  through `theme.pet` and stylizes; nothing in `pet.py` names a colour.

## 5. `PLAIN` layout

Pure ASCII, six lines; the same pose. Lines 3 and 4 carry the text at column 14
(`TEXT_INDENT`): the version with the context right-aligned, then the hint with its
command list right-aligned the same way (split at the first run of two or more
spaces, as the `RICH` hint has always been; a hint with no such gap is written whole).
Line 5 is the feet on a ground of `_` extended to column `width − 2`; below
`BODY_WIDTH_FLOOR = 40` the ground is not extended. Nothing is ever truncated; a line
that does not fit overflows by one space, as today.

```
       __
      (o >
    _/ /
   /  /       proofpath v0.1.0                        academic . offline . mps
  /__/        paste a file path, a URL, or a claim.      /help  /config  /quit
 ____||_______________________________________________________________________
```

The §13.1 golden block of the main spec is amended to this banner (the same
`v0.1.0` / `academic . offline . mps` values as the block it replaces); the run block
under it is unchanged.

## 6. Modules

| Module | Change |
|---|---|
| `tui/pet.py` | Rewritten. `BITMAP`, `TONES`, the Braille encoder, `render(width, theme, *, version, context, hint) -> Pet` where `Pet` is `lines` + `tones` (per-row colour runs) + `text_column`. The animation state machine, `EYES`, `STAMP`, the tail and the spans go. Delegates to `banner.render` for `PLAIN`. |
| `tui/banner.py` | `PLAIN` raven (`ART` lines), ground extension, the two text lines; `EYES`, `STAMP`, `stamp_span` go. `Banner` dataclass keeps `lines` and gains `tones` in the same shape as `pet.Pet` (one run, `"light"`, over the art of each art line) so the widget has one path. |
| `tui/widgets/banner.py` | No timer, no clock, no state. `set_busy`, `flash`, `start_animation` stay as no-ops so `app.py` does not change; `Drawing` keeps `lines` and drops the stamp fields; `eyes` and `timer` go. `render` stylizes each tone run with `theme.pet`. |
| `tui/theme.py` | `Theme.stamp: str` becomes `Theme.pet: Mapping[str, str]` with keys `dark`/`light`: `RICH` `{"dark": "#8f0f2b", "light": "#c4173a"}`, `PLAIN` `{"dark": "red", "light": "red"}`. `ui.STAMP_COLOUR` is left alone if anything else reads it; otherwise removed. |
| `tui/app.py` | Unchanged (the three calls become no-ops). Docstrings that say "ferret" say "raven". |

## 7. Tests

- `test_tui_pet.py`: the bitmap is 29×30 and uses only `#+.`; the Braille encoder on a
  known 2×4 block gives the known glyph; golden `RICH` renders at 100, 80, 60 and 47
  columns (47 is `PLAIN`); every art glyph is in U+2800–U+28FF or a space; the text
  column and the right margin; the hint drops its commands at a width where they do
  not fit; the tone runs cover exactly the non-space art cells and the ground.
- `test_tui_rich.py` / `test_tui_app.py`: remove the animation tests (blink, busy,
  flash, tail, timer); the banner-in-app tests assert the raven rows and that the
  three no-op methods exist and do nothing.
- PLAIN golden at 80 equals the §5 block; every character is ASCII.
- No network, no Textual in `test_tui_pet.py`.

## 8. Documentation and the site

- `README.md`: both banner blocks (the `RICH` one near the top, the `PLAIN` one under
  "Windows / `NO_COLOR`") become the raven; the prose says "raven" where it said "ferret".
- `CHANGELOG.md` `[Unreleased]` → `### Changed`: the raven, the stamp and animation gone.
- Main spec §13.1: amendment note pointing here; golden block replaced by §5.
- TUI v2 spec §3: amendment note pointing here.
- `OPEN-ITEMS.md` 8.19: the pet is a raven; ferret decision superseded.
- `site/`: `demo.ts` exports `raven: string[]` (the `PLAIN` art, static) instead of
  `ferret`/`Eyes`; the two `id: 'ferret'` frames go; `Terminal.astro` and
  `terminal.ts` drop `setEyes`. `npm run check`, `format:check` and `build` clean.
- `docs/eval/*.svg` and the recorded sessions are history and are not regenerated.

## 9. Verification

`ruff check`, `ruff format --check` (repo root, which formats fenced Python in Markdown
too), `pytest` green on the three platforms in CI; the site's three checks; a live
`PROOFPATH_THEME=rich proofpath` and `PROOFPATH_THEME=plain proofpath` screenshot at
80 and 100 columns, eyeballed.
