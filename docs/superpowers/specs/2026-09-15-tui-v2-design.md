# proofpath TUI v2 — design

**Date:** 2026-09-15 · **Status:** approved by the author, implementation follows.
Amends spec §13.1 and §13.3 of `2026-09-10-proofpath-design.md`; everything not
mentioned here is unchanged.

## 1. Problem

v0.2.0's TUI is functionally complete and structurally honest, but it reads as a
prototype: a three-line ASCII pet, no visual hierarchy between the banner, a run
and its findings, labels glued to state words, no progress bars, a cramped footer.
The author rated it 2/10 and asked for 10/10 without giving up the honesty rules
or the Windows/`NO_COLOR` guarantees.

## 2. Decision: two themes, one truth

The spec's "pure ASCII, ANSI-16 only" rule existed so the TUI renders identically
in Windows Terminal at 80 columns and under `NO_COLOR`. That goal is *readable
everywhere*, not *identical everywhere*. The TUI therefore has two themes:

| Theme | When | Glyphs | Colour |
|---|---|---|---|
| `RICH` | `COLORTERM` is `truecolor`/`24bit`, or the terminal is known to support it (iTerm2, WezTerm, kitty, Ghostty, Alacritty, Windows Terminal via `WT_SESSION`, VS Code via `TERM_PROGRAM=vscode`, and macOS Terminal.app — 256-colour, the tones downgrade faithfully), **and** none of the PLAIN triggers apply | Unicode: box drawing, `⏺ ✓ ✗ ⚠ ● ▰ ▱ █ ▓ ░ ⧉ ›` | truecolor tones of the same four meanings + one accent per run |
| `PLAIN` | `NO_COLOR`, `--no-color`, `-q`, `TERM=dumb`, legacy conhost (`WT_SESSION` unset on Windows), a CJK locale (`LC_ALL`/`LC_CTYPE`/`LANG` naming `zh`/`ja`/`ko` — box drawing renders double-width there), or no truecolor evidence | ASCII only: `* + x ! # = - ~ >` | ANSI-16 names exactly as today |

Detection lives in one place (`tui/theme.py`) and is testable by injecting the
environment. `PROOFPATH_THEME=rich|plain` overrides detection (documented, for
screenshots and bug reports).

**Meaning colours stay owned by `ui.py`.** Green/yellow/red/dim keep their tables
and the state-word rules of §13.3; `theme.py` maps each meaning to a truecolor tone
in `RICH` and to the ANSI name in `PLAIN`. Accents (one per run) stay the rotating
five; in `RICH` they are five distinct hues, never red/yellow/green. The banner's
only colour remains the red stamp.

## 3. The pet

`RICH` draws a seven-line ferret: head at the left with a `╸┤` nose and `╭╮ ╭╮` ears, a
low body that grows with the width and ends in a rounded rump, four feet under the
joints, and a **real tail** that leaves the rump at mid-height and runs `~~~~` to the
`[PROOF]` stamp, whose right edge aligns with the context and hint lines below. The
two text lines are lines six and seven. Tail length ≥ 8 and grows with the width (up
to about a third of the body); below 60 columns the stamp is dropped and the tail
shortens; below 48 the `PLAIN` pet is used regardless of theme. Exact art lives in
`tui/pet.py` as a commented constant; the reviewer judges it. `PLAIN` keeps today's three-line ASCII
ferret unchanged, so the golden block in §13.1 still holds for that theme.

Animation (both themes, off under `-q`/`NO_COLOR`): eyes blink `(-.-)` once every
6–10 s; while any run is `running`/`verifying` the eyes look down the path `(>.>)`
and in `RICH` the tail wags (three frames at ~2 Hz); a run that ends with
findings shows `(O.O)` for two seconds, a clean one `(^.^)`.

## 4. Layout

```
╭─ #1  /check ~/Desktop/paper.pdf ──────────────────────────── verifying ─╮
│  ✓ Parsing       24 pages · 42 refs                        pymupdf  1.2s │
│  ✓ Resolving     38 ok · 3 ambiguous · 1 ghost   Crossref, S2  3.4s │
│  ✓ Retractions   1 retracted                      Retraction Watch  0.8s │
│  ✓ Fetching      22 full text · 11 abstract · 9 unverified          14.7s │
│  ⏺ Verifying     ▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱  51/118                    coreml │
│  ──────────────────────────────────────────────────────────────────────── │
│  ✗ p.4  L112  [12] Zhang, K. et al. (2021)          GHOST REFERENCE      │
│       DOI 10.1016/j.xxxx.2021.99999 resolves to nothing                   │
│  ⚠ p.7  L203  [28] Lee, S. (2019)                   RETRACTED  2023-06    │
│  ✗ p.9  L260  [31] Kumar 2022                       NOT SUPPORTED  high   │
│       you     "the method yields a 40% speedup"                           │
│       source  "we observed a 4-8% improvement in throughput"          ⧉   │
╰───────────────────────────────────────────────────────────────────────────╯
```

Rules:
- One panel per run, border in the run's accent, title = `#n  /check <target>`,
  state word at the right of the top border (`queued` dim, `running`/`verifying`
  accent, `done` green, `cancelled` dim, `failed` red).
- Stages are a fixed-column table: symbol + name (14), summary (grows, `·`
  separators), attribution (right, dim), elapsed (6, right). The active stage
  shows a real progress bar (`▰▱`, `Progress.done/total`) with `done/total`;
  finished stages show `✓`, a stage that ended with an honesty count keeps `⏺`.
- A hairline rule separates stages from findings. Finding rows: symbol, location
  (`p.4  L112`, fixed 12), reference label (elided first), state word rendered as
  a **badge** (`RICH`: coloured background + black/white text; `PLAIN`: the word
  in its meaning colour, as today), tier right-aligned. Detail lines (`you:` /
  `source:` / notes) indented, dim, with `⧉` to copy the passage; toggled by
  click/`enter` as today.
- Blocks are separated by one blank line. The `PLAIN` theme keeps today's flat
  layout (no borders), so nothing regresses where borders cannot draw.
- Footer (docked, never scrolls): a coverage **bar** `█ 62 % full text ▓ 21 %
  abstract ░ 17 % unverified` (green/yellow/red), then the counts line, then
  `report.md written · 0 API calls · 38 s`; hints (weak coverage, no bibliography,
  skipped sources, tier note) on the third row exactly as today. `PLAIN`: the
  three `kv` lines as today.
- Prompt: `›` (RICH) / `>` (PLAIN) in the current run's accent, a one-line rounded
  border in `RICH`; awaiting mode and the tint are unchanged.
- Widths: ≥ 80 full layout; 60–79 borders kept, summaries elide first; < 60 no
  borders (today's rows); < 48 `PLAIN` pet.

## 5. Structure

`tui/app.py` (1813 lines) is split as part of this work — OPEN-ITEMS 12.4:

| Module | Owns |
|---|---|
| `tui/theme.py` | `Theme` dataclass (glyphs, tones, accents), `detect(env, *, no_color, quiet) -> Theme`, `RICH`, `PLAIN` |
| `tui/pet.py` | both ferrets, `render(width, theme, *, version, context, hint, eyes, tail_offset)`, the animation state machine (pure) |
| `tui/widgets/run_block.py` | `RunBlock`, `RunHeader`, `StageLine`, progress bar |
| `tui/widgets/finding.py` | `FindingsRule`, `FindingLine`, badge rendering |
| `tui/widgets/footer.py` | `CoverageFooter`, coverage bar |
| `tui/widgets/prompt.py` | `Prompt`, `PermissionPrompt` |
| `tui/app.py` | `ProofpathApp`: composition, scheduler wiring, slash commands — target ≤ 700 lines |

`banner.py` stays as the `PLAIN` ferret's implementation (imported by `pet.py`);
`commands.py`, `runs.py` are untouched. No logic moves into widgets: every widget
renders a `Report`/`Finding`/`Stage`/`Progress` it is handed.

## 6. What does not change

Every state word and every product-rule sentence (`UNVERIFIED (...)`, `NEI`,
`coverage is weak`, `no bibliography was found; N citation markers could not be
checked`, the §7.1 prompt text and its four answers); the exit codes; the one-shot
CLI (never draws the pet); the scheduler; the mirror rule; rule 4 (the gate calls
the TUI prompt); the 83 existing TUI tests (they run under `PLAIN`, which is the
theme `run_test` gets by default because no truecolor evidence exists there).

## 7. Testing

- `theme.detect` table: every trigger in §2, both directions, `PROOFPATH_THEME`.
- Pet: golden renders at 80/100/60/47 columns in both themes; `PLAIN` at 80 still
  equals the §13.1 block; `RICH` pet is seven lines and every line ≤ width.
- Widgets: golden strings per widget in both themes (`app.run_test` with the theme
  injected), including the badge, the progress bar at 0/51/118, the coverage bar at
  62/21/17 and 0/0/0, and every run state on the border.
- Windows CI: the `PLAIN` ASCII assertions stay; add one `RICH` render test that
  asserts no exception and correct widths (Unicode glyph *widths* are the risk).
- A by-hand session in a real terminal (both themes, `PROOFPATH_THEME` forced)
  recorded as text in `docs/eval/2026-09-15-tui-v2-live.md`.

## 8. Release

Ships as v0.2.1 (no behaviour change, presentation and structure only); CHANGELOG
`### Changed`; README replaces the banner block with the `RICH` one and shows the
`PLAIN` one under "Windows / NO_COLOR".
