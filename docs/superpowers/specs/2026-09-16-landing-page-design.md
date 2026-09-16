# proofpath landing page — design

**Status:** implemented on branch `landing-page`, under `site/`. Preview build published
as a Claude artifact on 2026-09-16; Vercel deploy pending (see §6).

## 1. Purpose

A single page that says what proofpath does, shows a real run, and gets someone to
`uv tool install proofpath`. It must carry the product's own rules: it shows passages,
it names its coverage, and it never dresses up an example as a real result.

## 2. Decisions (2026-09-16, with the user)

| Topic | Decision |
|---|---|
| Scope | proofpath only |
| Look | Warm paper ground, crimson field for the hero / rules / footer, rose only as a wash. "Strange" dithered public-domain engravings. Pencil-drawn UI: wobbly SVG boxes, rules, arrows, underlines, badges. |
| Mythology | Norse: Odin's ravens. Huginn (thought) flies to the source — levels 1 and 2. Muninn (memory) brings the passage back — level 3. A raven back with nothing says "nothing found", not "nothing exists" (rule 2). |
| Artwork | No hand-drawn ravens (rejected). Ravens are details lifted from the engravings themselves. |
| Type | Fraunces (display, `opsz`/`SOFT`/`WONK` axes) · Inter (body) · IBM Plex Mono (terminal, findings, code) · Caveat (margin notes and captions — the "pen"). Self-hosted via fontsource; the four first-paint files are preloaded. |
| Emphasis | Key words in body copy are set in the pen colour: `*word*` in the data files → `<em class="key">` via `src/lib/mark.ts`. |
| Stack | Astro 7, static output, no UI framework (React can be added as islands later without a rewrite). npm, Node ≥ 22.12, TypeScript strict, Prettier. |
| Motion | Medium: strokes draw themselves on scroll, a typewriter terminal, hover swaps a denser halftone. Everything is gated on `prefers-reduced-motion` and on `html.js`, so no-JS and reduced-motion readers get complete lines and the final terminal state. |
| Hosting | Vercel via Git integration, root directory `site`. `site` in `astro.config.mjs` is a placeholder until a domain exists. |

## 3. Page

1. **Hero** (crimson) — "Don't guess. Show the evidence.", one-paragraph lede, install
   box with `uv` / `pipx` / `pip` tabs and a copy button, Bracquemond's *Le Corbeau*.
2. **A run, as it happens** (paper) — the real `/check tests/data/draft-live.md` run from
   `docs/eval/2026-09-15-tui-v2-live.md`, replayed frame by frame with the ferret banner;
   margin notes point at what matters. Full transcript is in the markup for screen readers.
3. **Three questions, asked in order** (paper) — exists / still valid / supports, each with
   an engraving, the services asked, the states it can produce, and one finding. Level 2's
   finding is illustrative and says so on the page.
4. **Six things it will never do** (crimson) — the six rules from `CLAUDE.md`; rule 6 carries
   the coverage bar and the "coverage is weak … lower bound" line.
5. **The same engine behind a pipe** (paper) — install box again, the four CLI lines, exit
   codes, SARIF/JSON, "ask → deny in CI, reported".
6. **Footer** (crimson) — Heath Robinson band, links, and generated credits for every
   engraving with source, licence and year.

## 4. Images

`site/images.manifest.json` lists each source (Wikimedia Commons / Met Open Access, all
CC0 or public domain, checked on the file page), its crop and tuning. `npm run images`
(`scripts/dither.mjs`) fetches once into `.cache/`, then writes two-level Bayer 8×8
halftones to `public/images/` — a `paper` colourway (crimson darks, paper transparent) and a
`crimson` colourway (paper lights, crimson transparent), each in 720 and 1440 widths and a
denser hover variant. Output is deterministic and committed; the build does not fetch.
`src/data/images.generated.ts` is written by the same script and feeds `<Halftone>` and the
footer credits. Wikimedia rate-limits aggressively; the script backs off on 429.

## 5. Verification

- `npm run check` (astro check) — 0 errors; `npm run format:check` clean; `npm run build` clean.
- Lighthouse (mobile, preview build): accessibility 100, best practices 100, SEO 100.
- No horizontal scroll at 390 px; all grid children carry `min-width: 0`.
- Contrast: crimson on paper 5.28:1, slate on paper 5.06:1, rose-soft on crimson 4.69:1,
  ink-mute on paper 5.01:1.
- Main checkout on `main` untouched; this branch adds only `site/` and this file.

## 6. Open

- Domain, then `site` in `astro.config.mjs` and `public/robots.txt`.
- Connect the GitHub repo to a Vercel project with root directory `site`.
- README link to the site and a CHANGELOG line, after merge (the other session owns those files).
- `public/og.png` is a browser capture of `/og` (see `site/scripts/og.md`); regenerate when the hero copy changes.
