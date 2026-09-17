# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2026-09-17

A raven in the terminal, and a website to go with it.

### Added
- **A website**, under `site/`: a static Astro page with a replay of a real `/check`
  run, the three questions, the six rules, and the CLI surface. The engravings are
  public-domain plates (Met Open Access, Wikimedia Commons) halftoned at build time by
  `site/scripts/dither.mjs` from `site/images.manifest.json`; the outputs are committed,
  so the build never fetches. Design notes in
  `docs/superpowers/specs/2026-09-16-landing-page-design.md`.

### Changed
- **The pet is a raven.** The ferret is gone from both themes: `rich` draws a
  two-tone pixel raven in Braille cells (fifteen columns, eight rows, the text beside
  it, its ground running to the right edge), `plain` a small ASCII one. The
  `[PROOF]` stamp and the blink/busy/tail animation are gone with it; the widget's
  `set_busy`/`flash` remain as no-ops. Design notes in
  `docs/superpowers/specs/2026-09-16-raven-pet-design.md`.

## [0.4.1] - 2026-09-16

The TUI gets a shell's reflexes, and sits on the terminal's own background.

### Added
- **The TUI bar behaves like a shell prompt.** `↑`/`↓` walk a command history kept
  across sessions (`state_dir()/history`, 500 lines); `Tab` completes `/verbs`,
  `/allow` answers and `/cancel #n`; `PageUp`/`PageDown`, `Shift+↑`/`↓` and
  `Ctrl+Home`/`End` scroll the log without leaving the bar; `↑`/`↓` on a focused log
  line step between lines and typing returns to the bar; `Ctrl+L` clears finished
  runs and keeps the coverage footer; a mouse selection is copied by `Ctrl+C`/`⌘C`
  instead of quitting. `/help` lists the keys.

### Fixed
- The TUI **sits on the terminal's own background** instead of Textual's grey: every
  surface is `ansi_default` and palette names reach the terminal untranslated, so the app
  no longer shows up as a layer over the terminal it runs in.

## [0.4.0] - 2026-09-16

Sources that are not papers: a post's links, and the coverage block that finally says why
a source could not be read. Also the first end-to-end measurement on real web claims, which
says plainly that this is not yet the thing to point at a news story
(`docs/eval/2026-09-16-averitec.md`).

### Added
- **`check --url`** reads a post and verifies **the links inside it**, never the post's own
  words (spec §6.2). Bluesky (`public.api.bsky.app`) and Hacker News (the Firebase API) are
  first-class and need no account; Reddit reads with a free app you register yourself
  (`REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` in `.env`); Mastodon is best effort per
  instance; X cannot be read and says so, asking you to paste the text. Pasted text with
  links in it is verified the same way.
- **`UNVERIFIED (credentials missing)`** (spec §15): a platform that reads only with a
  credential this machine does not have is its own state, never "unreachable". No request is
  made, both variable names are printed, and the secret itself never reaches a log, an error
  or a `repr`.
- **The coverage block now prints a line per reason** on the terminal and in the markdown
  report, not only in SARIF. `blocked: 3`, `credentials missing: 1`, `unreachable: 2` — a
  source with no recorded reason is listed rather than dropped, so the lines account for
  every unverified source. Product rule 6 on its main surface. The TUI footer does not carry
  them yet.
- **A `providers/` package** (`academic`, `web`, `social`) behind the `EvidenceProvider`
  protocol of spec §5.2, so a new source family no longer means touching the core. The
  academic and web paths came through byte-identical, pinned by a golden report.
- **`docs/eval/2026-09-16-averitec.md`**: 100 AVeriTeC dev claims through the whole product.

### Fixed
- A bare DOI or arXiv **URL** is resolved as a record again instead of being fetched as a
  web page, so it keeps its retraction check and its open-access full text.
- A `doi:` source is read through the open-access chain and retraction-checked by the
  academic provider whoever resolved the entry — a warm cache could previously route both
  to the web ladder, where a missing retraction notice was reported as "no notice".
- The `Resolving` stage no longer names Crossref and Semantic Scholar on a run that asked
  neither.
- A document that cites by linking is never paired as if it printed a numbered bibliography,
  and vice versa: a PDF with a bibliography and no detected marker could previously pair a
  body sentence to a reference it never cited.
- `check --url` consults the network permission before reading anything; a denied run makes
  no request at all.

### Measured
- AVeriTeC dev, 100 claims: **0.270 3-way accuracy against a 0.708 majority baseline**
  (4-way 0.240). A third of the claims had no readable source; on the rest the score is
  0.361. Every `Supported` claim was missed. Nothing was tuned after the measurement.

## [0.3.0] - 2026-09-15

The judge layer: an opt-in LLM second opinion and an opt-in model-written summary.
The default run still makes zero LLM calls, and nothing the model says can change a
verdict. Live run on Groq in `docs/eval/2026-09-15-judge-live.md`.

### Added
- **`check --judge`** (`judge.py`, `verify.py`). After the local verdicts are final, the
  `low`-tier ones — never a numeric mismatch, never a claim without a quoted passage —
  go to the model in batches of up to 20 items (about 7k tokens), each with its claim and
  passage. The opinion (`SUPPORTED | REFUTED | NEI` plus a one-sentence rationale) is
  attached beside the verdict: a `= judge (groq openai/gpt-oss-120b): …` line under a
  finding it disagrees with, a `judge` column in the markdown `## Checked` table, and
  `judge` fields in the JSON. The local `Verdict`, the finding kind and every state are
  untouched (spec §11.1). Opinions are cached in the new `judgements` table (schema v4,
  additive; wiped with the verdicts when a source's text changes), so a re-run asks
  nothing, and `model_id` is untouched, so toggling `--judge` never invalidates a verdict.
- **`check --summarize`** and the TUI's **`/summarize`**: one extra call over the finished
  markdown report, run after the report is final, off by default in both front-ends,
  printed as `summary    (model-written, groq openai/gpt-oss-120b) …` and as
  `## Summary (model-written, …)` in the file. `--summarize` alone is exactly one call.
- **`JudgeClient`**: one adapter for Groq (default `openai/gpt-oss-120b`), Gemini and
  Ollama over the OpenAI `chat/completions` shape; strict JSON-schema output with a
  `json_object` fallback, `reasoning_effort=low` with a fallback for providers that
  reject it, `Retry-After` on 429 (capped, accounted), exponential backoff on 5xx, then
  `JudgeUnavailable`. The key comes from the environment or `.env`, never from config,
  never appears in `repr`, errors or logs; provider bodies are never echoed. Prompts are
  packaged template files (`proofpath/prompts/review.md`, `summarize.md`).
- **Cost on every surface**: the `Judging` and `Summarising` stage lines carry calls and
  prompt/completion tokens; the footer counts the calls; `Report.judge_cost` and
  `models["judge"]` land in the JSON.
- **An unanswered judge is reported, not hidden**: `judge unavailable after N calls
  (HTTP 401 from …); local verdicts stand` in the stage line, the report header
  (`judge status:` / `summary status:`), the JSON and an unsuppressed terminal line — a
  `-q` or piped run cannot look like a judged-clean one. The SARIF log does not carry it.
  The run never fails because of the judge.
- Gemini prints its data-use warning once per run (spec §11).

### Changed
- `check --judge` / `--summarize` no longer exit with `arrives in v0.3`.
- Cache schema **v4** (`judgements`); a v1 file still migrates through the whole chain.
- The markdown `## Checked` table always carries a `judge` column; without `--judge` every
  cell is `—`, so a v0.2 report and a v0.3 one differ by that column alone.
- `--format json` gained `summary`, `summary_model`, `judge_cost` and per-result `judge`
  fields; every one of them is `null` on a default run.

## [0.2.1] - 2026-09-15

The TUI's second look. No behaviour change: every state word, every honesty sentence,
the exit codes, the scheduler and the one-shot CLI are exactly v0.2.0's. Design in
`docs/superpowers/specs/2026-09-15-tui-v2-design.md`; the by-hand session in both
themes, with SVG screenshots, in `docs/eval/2026-09-15-tui-v2-live.md`.

### Changed
- **Two themes, one truth** (`tui/theme.py`). `rich` draws box borders, Unicode glyphs
  and truecolor tones when the terminal gives evidence of them (`COLORTERM`, Windows
  Terminal, iTerm2, kitty, WezTerm, Ghostty, VS Code, Terminal.app); `plain` is a pure-ASCII, ANSI-16 look
  (v0.2.0's `⏺ ✗ ⚠ › ⧉` become `* x ! > [copy]`, and findings now print their `= note:`
  lines as the CLI does), chosen under `NO_COLOR`, `--no-color`, `-q`,
  `TERM=dumb`, legacy conhost, a CJK locale, or any session without truecolor evidence (an SSH
  or tmux session that strips `COLORTERM` gets `plain`). `PROOFPATH_THEME=rich|plain` overrides
  detection. The meaning colours stay `ui.py`'s tables; a theme changes how a meaning
  looks, never what a word means. Spec §13.1's "pure ASCII" rule for the pet now binds
  `plain` only.
- **The pet** (`tui/pet.py`): `rich` draws a seven-line ferret with a real tail running
  to the `[PROOF]` stamp; `plain` keeps the three-line one unchanged. Same eyes, same
  blink, same reactions; the tail wags in `rich` while a run works.
- **Run panels** (`tui/widgets/run_block.py`): in `rich` each run is a rounded panel in
  its accent, the command and the state word on the top border, the coverage on the
  bottom one; the stages are a fixed-column table (symbol, name, summary with `·`
  separators, attribution, elapsed) and the active stage carries a real `▰▱` progress
  bar with `done/total`. A finished stage that left something unverified keeps the
  `⏺` mark instead of a tick. Below 60 columns the borders go and the flat rows
  return; a run that crossed the floor either way says its coverage exactly once.
- **Findings** (`tui/widgets/finding.py`): the state word is a badge, the location a
  fixed cell, the tier right-aligned; the finding's notes and the claim (`you`) and
  the passage (`source`) it was checked against are printed under it. A tier-less
  badge keeps its row at 60–79 columns. `plain` prints the notes as the CLI's own
  `= note:` rows and keeps `[copy]`.
- **Footer** (`tui/widgets/footer.py`): `rich` draws the coverage as a proportional
  `█▓░` bar in the three meaning colours over the counts line; `plain` keeps the
  `kv` lines. The prompt wears a one-line rounded border in the run's accent.
- **Structure** (OPEN-ITEMS 12.4): `tui/app.py` is split — `theme.py`, `pet.py`,
  `verbs.py` and `tui/widgets/` (`run_block`, `finding`, `footer`, `prompt`,
  `banner`, `_shared`) — and is now 681 lines of composition, scheduler wiring and
  slash commands, no rendering.

## [0.2.0] - 2026-09-15

The interactive front-end, author-year citations, SARIF output and a cache for the
two lookups that made a warm re-run slow. Bare `proofpath` now opens the TUI; the
one-shot verbs are unchanged for CI. Live checks in `docs/eval/2026-09-15-v0.2-live.md`.

### Added
- **TUI** (spec §13.1): bare `proofpath` opens a `textual` session — a four-line banner
  with the pet, one prompt bar, and a scrolling log of run blocks. Every CLI verb is a
  slash command (`/check`, `/resolve`, `/fetch`, `/config`, `/cache`; plus `/allow`,
  `/cancel #n`, `/help`, `/quit`), a bare verb waits for its argument (`Esc` leaves
  it), and a line that is not a command is something to check. Runs are scheduled
  concurrently through one `Scheduler` (network stages in parallel under a shared
  politeness limiter, the NLI model one run at a time), each run is **cancellable**
  mid-flight and keeps what it had decided, the coverage footer never scrolls away,
  and the §7.1 permission prompt is drawn **inline under the stage that hit the wall**
  with `[allow once] [always] [no] [never]` buttons (or `/allow …`). Every log line
  works by mouse and by keyboard: run headers fold, stage rows hide their summary,
  findings open the full quoted passage, `⧉`/`c` copies it, a finding's reference is
  an OSC 8 link to its source. The pet's eyes blink, watch a run and react to its
  result — only with colour, never under `--no-color` or `-q`.
- **Author-year citations** (spec §9, §17): `(Smith et al., 2020)`, `Smith (2020)`,
  `(Smith, 2020; Jones, 2019)`, `2020a`/`2020b` collisions, `ibid.` and `op. cit.`
  back-references, and mixed `(Smith, 2020; [12])` all pair with their bibliography
  entry. Hand-built set of 55 passages, 83 expectations, rate **0.940**
  (`docs/eval/2026-09-12-pairing-author-year.md`); the numeric set is unchanged at
  108/109. A marker no entry matches is reported as `UNRESOLVED MARKER`, never guessed.
- **`check --format sarif`** (spec §13.2): the run as a SARIF 2.1.0 log for VS Code or
  any SARIF 2.1.0 viewer — one result per finding on its line, the quoted passage in
  the message of every asserting result, the exact honesty state in `properties`, and
  the run's coverage in `runs[0].properties`. `--out FILE` writes the same document.
  Validated against the schemastore schema, vendored in
  `tests/data/sarif-schema-2.1.0.json`; not yet exercised against GitHub code scanning.
- **Resolution and retraction cache** (cache schema v3): `resolutions` (30-day TTL,
  keyed by the marker-free folded entry) and `retractions` (30 days for a notice, 7
  for its absence). Neither an `UNVERIFIED (provider unavailable)` nor a retraction
  check every provider failed is ever stored. A warm re-run of the seven-reference
  draft: **1.35 s** wall clock, 0 network calls (v0.1: 15.8 s). `proofpath cache`
  counts both tables.
- The §7.1 aggregate line `skipped N source(s) because the browser was not permitted`
  is printed by `check` too (terminal and markdown), never dropped by `-q`.
- `browser_binary_present()`: the consent gate checks for a chromium build under
  `PLAYWRIGHT_BROWSERS_PATH` or the platform default, so a half-installed environment
  runs the idempotent installer instead of failing inside the fetch; the consent log
  gains `browser binary: found|missing`.
- `commands.py`: the shared wiring behind every mirrored verb (`resolve_reference`,
  `fetch_target`, `config_*`, `cache_*`) returns result objects and never prints, so
  `cli.py` and the TUI can only differ in how a result is drawn.
- Bibliography fallback for PDFs without a `References` heading (paged formats):
  the last contiguous run of numbered paragraphs is read as the list. AlphaFold:
  0 → 17 references found, 97 markers now reported as unresolved instead of invisible.
- Resolver rescues: `First Last and First Last` author lists resolve instead of landing
  in `AMBIGUOUS`; an arXiv id whose record agrees on author and year is accepted as
  `RESOLVED (low confidence)` the way a DOI already was. Ghost set 274 rows: false-ghost
  **0.0 %**, ghost recall 99.1 % (`docs/eval/2026-09-12-ghosts.md`).
- `tests/data/draft-author-year.md`, a committed author-year draft with an offline
  smoke test; live: 6 citations paired, 0 unresolved.

### Changed
- Cache schema **v3** (migrated in place from v1/v2; versions compared as integers).
- `Claims.unsupported` now means "a style this version cannot pair" and is empty by
  construction; `UNSUPPORTED CITATION STYLE` is left for footnote-only and
  superscript-letter styles. Markers that pair with nothing are `unresolved`.
- `resolve.looks_unindexed` reads only the initials-path author patterns: a full-name
  list (`First Last and First Last`) no longer counts as "this entry printed an
  author list", so a title-first book keeps its title (see Known issues for the price).
- `--format sarif` on `resolve` or `fetch` now says `applies to check only` instead of
  naming a future version.
- Stage row `Claims` reads `N citations, M unresolved` (was `M unsupported`).

### Fixed
- A provider answering with a non-JSON body (a bot wall or maintenance page under a
  200) escaped `resolve.py` as a raw `JSONDecodeError` traceback. Every provider body
  is decoded in one place and that failure is a `ProviderError`, so `resolve`, `fetch`
  and `check` report `UNVERIFIED (provider unavailable)` and never read it as evidence
  the work does not exist.
- The §7.1 prompt said `blocked this request (HTTP 200)` for the empty-body bot wall;
  it now says `answered without readable text (HTTP 200)`.
- `resolve` on the CLI leaked one HTTP client per invocation; `commands.resolve_reference`
  owns and closes it.
- A retraction check every provider failed was recorded as "not retracted". It now
  raises, is reported as `retraction check unavailable` and counted in the stage
  summary (`1 unavailable`), and nothing is cached, so the next run asks again
  (product rule 2). Found alongside it: the fetching stage replaced a source's notes
  instead of appending, which would have dropped that very note.
- Versions in the cache file compared as strings (`"10" < "9"`); compared as integers now.

### Known issues
- `api_calls` in the footer counts LLM calls only (none yet), so a cold run that spent
  30 s on Crossref and Semantic Scholar still prints `0 API calls`; the provider lookups
  are shown on the stage lines instead.
- A cached resolution reprints the notes it was stored with (for example an `openalex
  unavailable (HTTP 429)` from the day it was resolved) as if they were current.
- `proofpath check -` names its SARIF artifact `-`; give the draft a file name when the
  log is meant for a viewer.
- Author-year pairing matches the first author and the exact year; `(Lindqvist, 2019)`
  against "Okafor, C. and Lindqvist, S." and a year off by one are reported unresolved,
  and two surnames sharing a last word (`Berg` / `van der Berg`) are reported ambiguous
  rather than guessed. `Smith 2020` with no comma is not a marker; `(WHO, 2020)` does
  not pair with "World Health Organization".
- An unnumbered two-column bibliography is cut at line breaks (RoBERTa: 103 entries
  for ~50), so author-year pairing over such a list resolves few items. The largest
  open item for author-year pairing.
- The headless bibliography fallback takes only the *last* run of numbered paragraphs
  (AlphaFold: 17 of 84 entries) and drops the prose printed before a block's first entry.
- A fabricated `First Last and First Last. Title. Venue.` with no year between the
  names and the title reaches `UNVERIFIED (not in bibliographic indexes)`, not `GHOST`
  (the price of the rescue above; rule 3 outranks recall). A proceedings-*volume*
  record can accept a fabricated paper cited into that volume as `RESOLVED (low
  confidence)` — the ghost set's one fabricated acceptance (0.9 %).
- A cold run of a long bibliography is still serial through the providers (129
  references: minutes); only the TUI's concurrent scheduler runs them in parallel.
- In the TUI the `loading models …` note is drawn after the `Verifying` row it precedes
  (it is emitted inside that stage).
- A `#n` run reference in the TUI is a click target that folds its block, not a
  hyperlink (spec §13.1); a run number has no address to open.
- A provider body that is valid JSON but not an object (a bare list or string) still
  escapes `resolve` as an error, exit 2, rather than being reported as `UNVERIFIED
  (provider unavailable)`.
- `/quit` in the TUI waits for an in-flight mirrored `/fetch` to finish before the
  app exits; a `/check` run is cancelled, a `/fetch` is not.

## [0.1.0] - 2026-09-12

First working release: `proofpath check` verifies a document's citations end to end
and writes a report that states its own coverage.

### Added
- `proofpath check TARGET` as the v0.1 surface: compiler-style diagnostics, a
  markdown `report.md`, `--format json`, `-q`, and exit codes `0` / `1` / `2`.
- Confidence tiers calibrated on SciFact dev rather than chosen by hand
  (`decide=0.45`, `medium=0.457948`, `high=0.99933`;
  `docs/eval/2026-09-12-tiers.md`), shipped as `pipeline.DEFAULT_THRESHOLDS`.
- Nine live user-like runs recorded in `docs/eval/2026-09-12-v0.1-live.md`, and
  `scripts/zero_network_check.py`, which guards both HTTP clients and re-verifies a
  document: with the network gone the fetch ladder makes no attempt at all, because
  resolution produces no identifiers to fetch with.
- Config and permissions module: `config.toml` under the platform config dir,
  `proofpath config` / `proofpath config set permissions.<key>`, and the rule that an
  `ask` permission without a TTY resolves to `deny` and is reported (spec §7.1).
- Core types (`Verdict` cannot be `SUPPORTED`/`REFUTED` without a passage), device
  selection (CUDA → CoreML → CPU), sentence retrieval over `fastembed` embeddings
  scanned with numpy, ONNX NLI entailment on `cross-encoder/nli-deberta-v3-base`, and the aggregation
  pipeline.
- Reference resolution (spec §8): Crossref + Semantic Scholar first, then arXiv,
  Open Library and OpenAlex before any ghost call; identity decided only by
  field agreement against the raw string; DOI / arXiv id resolved directly;
  retraction check via Crossref's Retraction Watch data and OpenAlex; new state
  `UNVERIFIED (not in bibliographic indexes)`; per-host throttling, `Retry-After`,
  OpenAlex daily-budget handling. `proofpath resolve "<reference>"` on the CLI.
  Hand-built ghost set (106 real, 100 fabricated, 20 mutated) and
  `scripts/eval_ghosts.py`: false-ghost rate 0 %.
- Persistent cache: one plain SQLite file (`sources`, `raw_text` with 7-day TTL,
  `chunks` with float32 embeddings, `verdicts`), a schema `CHECK` that refuses an
  asserted verdict without a passage, and `proofpath cache` / `cache path` / `ls` /
  `show` / `clear [--expired]`.
- Numeric claim layer (spec §10): percentages, factors and unit counts with
  direction are compared before NLI; an unambiguous contradiction is refuted by
  rule with both figures named (`Verdict.reason`). Conservative by design: one
  comparable figure on each side, change never against level.
- Judge settings (`[judge]` in config, Groq default) with `proofpath config check`
  and `proofpath config set judge.<key>`; API key resolved from the environment or
  `.env`, never stored or printed. `.env.example` added.
- SciFact loader pinned to the AI2 tarball by sha256, evaluation metrics, and
  `scripts/eval_scifact.py`. First measured result: dev accuracy 0.606 vs 0.406
  trivial baseline (`docs/eval/2026-09-11-scifact-dev.md`).

- Fetch ladder (spec §7): `httpx` → `curl_cffi` TLS impersonation → browser engine
  behind the §7.1 consent prompt → Wayback Machine; `robots.txt` via `protego`;
  content type from headers; per-host throttling and backoff shared in `polite.py`;
  fetched text cached with the 7-day TTL. Distinct honesty states for blocked,
  blocked-by-robots, browser-not-permitted, unreachable, provider-unavailable and
  network-denied — never collapsed.
- Consent gate for the ~280 MB browser engine: asks at most once per run, never
  without a TTY, `always`/`never` persist to config, installs with `pip` (or `uv`)
  and `scrapling install`, and reports how many sources were skipped.
- Open-access chain: Semantic Scholar → Crossref TDM links → Unpaywall (only with a
  contact address) → Europe PMC → arXiv → landing page → abstract (OpenAlex last);
  abstract-only results labelled `LOW CONFIDENCE (abstract only)`. DataCite arXiv
  DOIs resolve straight to the arXiv PDF. `proofpath fetch <url|doi|arXiv id>`.
- `scripts/eval_coverage.py`: measured 72 % full text / 18 % abstract / 10 % none
  on 50 DOIs (`docs/eval/2026-09-11-coverage.md`).

- Document ingest and claim extraction (spec §9): `document.py` value types with
  page/line locators; `ingest.py` for PDF (pymupdf blocks, per-page line numbers,
  running header/footer removal, superscript citations), docx (paragraphs and tables),
  markdown and plain text, with the bibliography kept as raw strings, every
  unparseable page reported as a `PageError` and a page holding an image and no text
  reported as a scan rather than passed on as an empty page; `claims.py` pairs numeric
  markers (`[12]`, `[12,15]`, `[12-15]`) with their sentence, applies the
  `PARAGRAPH-SCOPED` rule, reports author-year markers as `UNSUPPORTED CITATION STYLE`
  and any number the bibliography does not print as unresolved. Hand-built pairing set
  (61 passages, rate 0.98) and `scripts/eval_pairing.py`; four real documents (three
  PDFs and one extracted text) measured in `docs/eval/2026-09-11-pairing.md`.

- `proofpath check` (spec §9, §13.2, §15): one `verify()` entry point built as
  `prepare()` (parse, claims, resolve, retractions, fetch) and `decide_all()` (retrieval,
  numeric rule, NLI, cached per source and claim); compiler-style diagnostics with the
  quoted passage, the confidence tier and the exact honesty state; a coverage block in
  every run and a "coverage is weak" line when a quarter or more of the sources could
  not be read; `report.md` written by default, `--format json`, `-q`, `--out`, `check -`
  for stdin, a real Ctrl-C that keeps what was decided (exit 2). A second run of the
  same document re-decides nothing: chunks and verdicts come from the cache and neither
  model scores again, although the models are still loaded and reference resolution and
  the retraction check still go to the network. Cache schema v2: chunks carry the text
  digest they were cut from and a source's verdicts are dropped when its text changes.
  `permissions.network = deny` now also skips reference resolution and the retraction
  check, each reported as not attempted.

### Changed
- CLI surface (spec §13.3): `permissions` and `judge` groups replaced by `config`
  (`config` / `show` / `path` / `set SECTION.KEY VALUE` / `check`); global
  `--no-color` and `-q`; one `ui.py` layer over `rich` owns every colour and the
  10-column key/value layout; `resolve --format json`; "provider unavailable"
  exits `1` (a finding), no longer `2`. A resolved but retracted reference and a
  URL that was reached but yielded no text (`reached but no text extracted`) are
  findings too (`1`). `skipped N source(s)` counts sources, not the URLs tried for
  them; `--format json` adds `browser.skipped_urls`. `permissions.network` binds
  the open-access providers as well as the ladder, and `ask` without a TTY is
  `deny`, reported.
- Retrieval no longer depends on `sqlite-vec`: a numpy cosine scan is faster at
  every measured scale and the plain SQLite file opens in any GUI.
- Spec: `PARAGRAPH-SCOPED` and `UNSUPPORTED CITATION STYLE` states, three-tier
  confidence display, 7-day raw-text cache TTL, v0.1 limited to numeric citation
  markers.

### Fixed (found by the release's own live runs, `docs/eval/2026-09-12-v0.1-live.md`)
- A real reference whose author list carries a surname particle (`van der Walt`) was
  called a `GHOST REFERENCE` — product rule 3. Reference resolution now understands
  particles (including glued `al-`/`el-` forms), scores the title against every
  title-like segment, and accepts a DOI whose record agrees on first author and year
  as `RESOLVED (low confidence)` instead of a ghost. The ghost set grew to 258 rows
  (`docs/eval/2026-09-12-ghosts.md`): false-ghost rate 0.0 %, fabricated recall 100 %.
- A fabricated reference in a numbered bibliography was reported as
  `UNVERIFIED (not in bibliographic indexes)` instead of a ghost, because the printed
  marker (`[7] `) blinded the "is this even a paper" check. The marker is stripped once
  at the resolver's entry, guarded so it can never remove a year, an identifier or a
  title that begins with a number.
- A markdown draft piped through `proofpath -q check -` lost its `## References`.
- The consent-gated browser step (fetch-ladder step 3) could not install itself
  (`scrapling` has no `__main__`); it now runs the package's own CLI after `pip`/`uv`,
  and a paywalled Cell landing page was read through it live (3,632 words).
- A piped run occasionally aborted with exit `134` from ONNX runtime teardown after
  printing a complete report; the engine now releases both models explicitly and the
  CLI flushes its streams before exiting (20 of 20 consecutive runs exit `1`).
- A document with citation markers but no detected bibliography printed a clean-looking
  `0 %` coverage block; it now says `no bibliography was found; N citation markers could
  not be checked`.

### Known issues
- Reference resolution and the retraction check are not cached, so a warm re-run is
  still a network run: a 68-reference PDF takes about 13 minutes cold and 3.7 minutes
  cached, a 129-reference one 24 minutes cold. Concurrent resolution and a resolution
  cache are v0.2 work.
- Two-author lists written `First Last and First Last` resolve to `AMBIGUOUS`, not
  `RESOLVED`; an arXiv id whose record agrees on author and year is not yet rescued the
  way a DOI is.
- `browser.is_installed()` checks that the packages import, not that a browser binary
  exists; a half-installed environment skips the installer and fails inside the browser
  fetch (reported as an unverified source, never a crash).

## [0.0.1] - 2026-09-10

First release. The verification pipeline is not implemented; this reserves the name
and establishes the interface, packaging and CI that later phases build on.

### Added
- Design specification with measured source-access data (spec §6), the fetch ladder
  and its permission model (§7), and corrected reference resolution (§8).
- Phased implementation plan, ordered by risk retired rather than user-visible
  progress.
- `proofpath` command. A bare invocation is a first-class entry point rather than a
  help screen, which is where the TUI will attach. Exit codes are fixed: `0` clean,
  `1` findings, `2` the run itself failed.
- Cross-platform CI on Linux, macOS and Windows, and PyPI publishing through trusted
  publishing rather than a stored API token.
