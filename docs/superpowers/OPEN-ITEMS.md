# Open items

Everything left unresolved as of **2026-09-10**, written down so none of it has to be
reconstructed from memory. Each entry says enough to be picked up cold.

**2026-09-11 update:** every item in §2 and §5 was put to the author and decided; the
two assumptions that gate packaging (3.1, 3.4) were checked. Outcomes are recorded
inline and copied into the spec.

Spec: `specs/2026-09-10-proofpath-design.md` · Plan: `plans/2026-09-10-proofpath-implementation-plan.md`

---

## 1. Blocked — resolved 2026-09-10

| # | Item | Outcome |
|---|---|---|
| 1.1 | CI workflow could not be pushed; the token lacked the `workflow` scope. The Contents API refuses workflow files for the same reason, so the token refresh was the only route. | **Done.** CI runs on Linux, macOS and Windows × Python 3.10 and 3.13. All seven jobs green. |
| 1.2 | PyPI name `proofpath` was free but unreserved. | **Done.** Published 0.0.1 through trusted publishing (OIDC), so no API token is stored anywhere. A working CLI was added first — the declared entry point did not exist, and publishing it would have shipped a command that raises ImportError. |

---

## 2. Decisions — confirmed by the author 2026-09-11

| # | Decision | Outcome |
|---|---|---|
| 2.1 | `install_browser` default (spec §7.1) | **`ask`.** `deny` would hide that a blocked source was recoverable |
| 2.2 | Reddit in v0.4 | **Included, optional.** User supplies an OAuth app; missing credentials are reported, never silently skipped |
| 2.3 | `spiyweb` keeps its name | yes (not re-raised) |
| 2.4 | `reasonhound` keeps its name | yes (not re-raised) |
| 2.5 | Turkish sources | **After v0.4**, as a separate provider (TR Dizin / DergiPark class) |
| 2.6 | `--summarize` in the TUI | **Off by default in both front-ends**; TUI enables with `/summarize` |
| 2.7 | Database | **One plain SQLite file**, no extension (`sqlite-vec` removed 2026-09-11 after an architecture review: numpy is faster at every scale, GUIs cannot open `vec0` tables). Chroma / LanceDB / Qdrant rejected for this workload — see spec §12 |

---

## 3. Unvalidated assumptions — test before building on them

These were asserted during design without being checked. Each one, if wrong, changes
a phase.

| # | Assumption | How to check | If wrong |
|---|---|---|---|
| 3.1 | **A usable NLI cross-encoder exists as ONNX.** | **Checked 2026-09-11 — true.** `cross-encoder/nli-deberta-v3-base` @ `6c749ce` ships `onnx/` (739 MB fp32, 388 MB O4, 244 MB int8 ×4) plus `tokenizer.json`; `-small`/`-xsmall` likewise. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` has no ONNX. Quality on SciFact is still Phase 1's job | (fallback agreed if quality fails: export a better checkpoint with `optimum` ourselves; torch stays opt-in) |
| 3.2 | `sqlite-vec` ships working wheels for Windows and macOS arm64 | **Checked 2026-09-11 in CI — wheels fine, loading is not:** python.org-style macOS builds lack `enable_load_extension`, and older SQLite on Windows rejects `LIMIT ?` on knn queries. Both handled: numpy brute-force fallback + `k = ?` syntax. CI green on all six jobs | — |
| 3.3 | `textual` renders the §13.1 layout correctly in Windows Terminal at 80 columns | render a fixture screen in CI on Windows | simplify the box drawing to ASCII |
| 3.4 | SciFact, AVeriTeC and PubHealth are still downloadable and pinnable | **SciFact checked 2026-09-11:** HF `allenai/scifact` and `bigbio/scifact` are script-based loaders that `datasets ≥ 4` refuses; the AI2 tarball `scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` returns 200 (Last-Modified 2021-01-26) → download with `httpx`, pin sha256. AVeriTeC / PubHealth still unchecked | substitute a comparable set and note it in the spec |
| 3.5 | X Community Notes dumps are still published and parseable | download one day's file, inspect the columns | drop the X path entirely; Bluesky and HN already carry the social provider |
| 3.6 | `fastembed` model quality is sufficient for passage ranking | compare recall@k against a sentence-transformers baseline in Phase 1 | move embeddings to the `[gpu]` extra as well |

---

## 4. Design details deliberately left to implementation

Not oversights — they need real data to set, and guessing now would be false
precision. Each must end up written into the spec once measured.

| # | Detail | Set during |
|---|---|---|
| 4.1 | Title token-set Jaccard threshold for reference matching (spec §8) | **Set 2026-09-11:** coverage × length-ratio, strong ≥ 0.8, weak ≥ 0.5; measured on the ghost set (`docs/eval/2026-09-11-ghosts.md`) |
| 4.2 | Year tolerance beyond ±1 for online-first publications | **Kept at ±1** (2026-09-11); no real reference in the ghost set failed on year alone |
| 4.3 | Numeric comparison tolerance, and how to treat ranges vs point values (spec §10) | **Set 2026-09-11:** relative 10 % on the source side; point-in-stretched-range; overlapping ranges; single-figure attribution rule; change ≠ level. Spec §10 |
| 4.4 | Confidence threshold that routes a verdict to the judge | Phase 1 calibration |
| 4.5 | Chunk size and `k` for retrieval. Abstracts are trivial; a 12,000-word full text is not, and the two may need different settings | Phase 1, re-checked in Phase 4 |
| 4.6 | Which embedding model and which NLI model, by name and revision | Phase 1 |

---

## 5. Previously unsolved problems — decided 2026-09-11

**5.1 — A citation that supports a paragraph, not a sentence.**
**Decided:** distinct state `PARAGRAPH-SCOPED`. Triggers when the marker sits at the
end of a paragraph and its carrying sentence holds no other marker. Every sentence of
the paragraph is verified separately and grouped under one finding; no single sentence
gets a confident verdict on the paragraph's behalf. Spec §9, §15.

**5.2 — Author-year citation styles.**
**Decided:** v0.1 pairs numeric markers only; an author-year marker is reported as
`UNSUPPORTED CITATION STYLE`, never guessed. Author-year moves to v0.2 as its own
task with its own test set (`ibid.`, `op. cit.`, same author-year collisions). Spec
§9, §17.

**5.3 — Coverage is the real product ceiling.**
**Decided:** measure at the **start of Phase 4**, before the fetch ladder is written:
a 50-DOI sample, OpenAlex-only vs +Semantic Scholar +CORE +OpenAIRE, result written
into spec §6.1. Still open as a measurement, closed as a decision.

**5.4 — What "confidence" means to a user.**
**Decided:** three calibrated tiers `high` / `medium` / `low`, cut-points derived from
the Phase 1 threshold sweep on SciFact dev. Raw score only in `--format json`. Spec
§13.1, §14.

**5.5 — Caching fetched full text.**
**Decided:** raw publisher text cached with a 7-day TTL; verdicts and embeddings kept
until `proofpath cache clear`. One SQLite file under the user cache dir. Spec §5.1,
§16.

---

## 6. Deferred by choice

Not problems, just not now. Recorded so they are not rediscovered as new ideas.

| Item | Revisit when |
|---|---|
| Landing site: one Astro site, four pages (`evidencelab.dev/`, `/proofpath`, `/reasonhound`, `/spiyweb`) | after v0.1 ships something runnable |
| GitHub org `evidencelab` (free at time of checking) | when the landing site is built |
| `evidencelab.dev` domain (free at time of checking) | same |
| Pet family and names: one pixel animal per project in the same drawing style (proofpath dropped its red stamp on 2026-09-16; a shared mark is open again) (proofpath raven · reasonhound hound · spiyweb spider); names only if the landing page presents them as characters, chosen for all three at once | when the landing site is built |
| `spiyweb` as an alternative retrieval backend for `proofpath` | after Phase 6; the real tie between the three projects |
| GROBID parser as an opt-in `--parser` | if Phase 5 citation pairing accuracy proves inadequate |
| Turkish sources | after v0.4 |

---

## 7. Status 2026-09-11 evening — Phase 0 and Phase 1 done

- Config & permissions module shipped (`paths.py`, `config.py`, `proofpath
  permissions [set]`), `ask` + no-TTY → `deny` covered by tests.
- Phase 1 measured: **SciFact dev accuracy 0.606 vs trivial baseline 0.406
  (+0.200)**, recall@3 0.85, zero verdicts without a passage. Kill criterion
  passed. Full table and notes: `docs/eval/2026-09-11-scifact-dev.md`.
- Free-tier survey for the judge: `docs/research/2026-09-11-free-llm-api-tiers.md`.

### Decisions — confirmed by the author 2026-09-11 (evening)

| # | Decision | Outcome |
|---|---|---|
| 7.1 | Tier cut-points: precision targets 0.85/0.70 leave `high` unreachable on this model | **Decided 2026-09-12 (Phase 7).** `high` was never unreachable — the harness printed cut-points to two decimals and a real cut of 0.99933 showed as `1.00`. Calibrated on SciFact dev at k=1: `decide=0.45, high=0.99933, medium=0.457948` (85% / 70% precision targets); `Report.tier_note` stays empty because a high tier exists. `docs/eval/2026-09-12-tiers.md` |
| 7.2 | CPU is 3× faster than CoreML for the int8 NLI graph | **Done.** `entailment.providers_for()` skips CoreML for int8 exports; the CUDA → CoreML → CPU rule is unchanged for fp32 |
| 7.3 | Default judge provider (Phase 9) | **Groq `openai/gpt-oss-120b`** default; Gemini and Ollama selectable with `proofpath judge set provider`; NVIDIA build.nvidia.com is dev-only and not documented. Key lives in env or `.env`, never in config (`judge.py`, `proofpath judge check`) |
| 7.4 | Second embedding model for assumption 3.6 | compare `all-MiniLM-L6-v2` once, in Phase 4 alongside full text |
| 7.5 | Vector store: keep `sqlite-vec`, or Chroma? | **Neither.** Plain SQLite BLOB + numpy scan (`cache.py`, `retrieval.py`); `proofpath cache path/ls/show/clear` for inspection, DB Browser for SQLite for the GUI |
| 7.6 | Full-text chunk text after the 7-day TTL | **NULLed** with the raw text; embeddings kept; the verdict keeps its own quoted passage |
| 7.7 | `verdicts.model_id` contents | **NLI + embedder + k + thresholds**, so a threshold change never serves a stale verdict |

### Phase 2 and Phase 3 — done 2026-09-11 (evening)

- Numeric layer: `numerics.py`, SciFact unchanged-or-better, spec §10.
- Reference resolution: `resolve.py`, ghost set measured, spec §8.1. Findings that
  changed the design: OpenAlex free tier is now a daily budget (~100 searches);
  RoBERTa missing from Crossref and OpenAlex; books and web pages need their own
  state. `proofpath resolve` available on the CLI.
- Cache: plain SQLite (`cache.py`), sqlite-vec removed, `proofpath cache` commands.

### New open items

| # | Item | Note |
|---|---|---|
| 7.8 | `proofpath config set <section.key> <value>` | `permissions set` only covers permissions; `contact.email` had to be set from Python |
| 7.9 | Semantic Scholar API key | free key lifts the shared 1 req/s pool; optional, same pattern as the judge key |
| 7.10 | Ghost recall ceiling | fabricated references written like blogs/reports/org authors resolve to `NOT_INDEXED`, not `GHOST` — by design; Phase 4 fetches URLs and reports what it finds |

### Next session

1. **Phase 4:** fetch ladder and permissions (spec §7) — start with the 50-DOI
   coverage sample (5.3), then steps 1, 2 and 4; step 3 behind the consent prompt.
2. Wire `cache.py` into fetching (raw text + TTL) as it lands.

---

## 8. Phase 4 — decided 2026-09-11 (night), before the fetch ladder was written

Findings from inspecting the installed base environment, and the decisions they forced.

| # | Item | Outcome |
|---|---|---|
| 8.1 | `scrapling 0.4.15`'s static `Fetcher` imports `playwright` at module load, so it cannot run in the base install | **Step 2 uses `curl_cffi` directly** (`impersonate="chrome"`); `scrapling` stays in the base install only as the HTML→text parser (`scrapling.parser.Selector`, which imports cleanly). Spec §7 table unchanged in cost, changed in wording |
| 8.2 | `robots.txt` parser | **`protego` added** to the base dependencies (pure Python, wildcard-aware). Consulted on steps 1 and 2 only; a disallow is reported as `UNVERIFIED (blocked, robots.txt)` |
| 8.3 | Open-access chain metadata source, given OpenAlex's ~100 searches/day budget | **Semantic Scholar `openAccessPdf` → Crossref `link` → Europe PMC → arXiv → landing page (doi.org) → abstract.** Unpaywall only when `contact.email` is set (it requires an address). OpenAlex last, abstract fallback only |
| 8.4 | What happens when the step-3 prompt is answered "yes" | **Automatic install** into the tool's own environment: `python -m pip install "scrapling[fetchers]"`, falling back to `uv pip install --python <sys.executable>` when the venv has no pip, then `python -m scrapling install` for the browser. Every step and its outcome is written into the report; tests mock the subprocess |
| 8.5 | Coverage sample (5.3) | Taken from the 106 real references of the ghost set (those with a DOI), 50 of them, before writing the ladder; result in `docs/eval/` and spec §6.1 |

### CLI surface — decided 2026-09-11 (night), spec §13.3

| # | Decision | Outcome |
|---|---|---|
| 8.6 | Command tree | **Verbs flat (`check`, `resolve`, `fetch`); settings under `config` (`show`, `path`, `set section.key value`, `check`); `cache` stays.** `permissions` and `judge` top-level commands removed; closes 7.8 |
| 8.7 | Output language | **`rich` through one thin `ui.py`, restrained:** coloured state words on a TTY only, borderless key/value grids, progress bars on a TTY only, no panels; plain text when piped or `NO_COLOR` |
| 8.8 | Machine output | **`--format text|json` on every verb, `sarif` only on `check`;** JSON alone on stdout, humans on stderr |
| 8.9 | Exit codes | **0 clean · 1 any finding or honesty state · 2 tool failure only.** `resolve`'s unavailable → 2 becomes 1 |
| 8.10 | Global flags | **`--no-color`, `-q/--quiet` only.** `--config`, `--offline` deferred until asked for |
| 8.11 | TUI slash commands | **Mirror rule:** every verb/group is the same-named slash command; `/allow /summarize /help /quit` are TUI-only |
| 8.12 | Entry model | **TUI is the primary surface:** the user types `proofpath` and works inside it. One-shot subcommands stay only as the no-TTY path (CI, pipes, SARIF to file) and are documented as such |
| 8.13 | TUI slash awaiting mode | **`/verb` alone tints the input bar and waits for the argument (Esc cancels); `/verb <arg>` runs at once; the echoed command line keeps the verb's accent colour in the log, nothing else does.** Spec §13.1; built in Phase 8 |
| 8.14 | Several runs per TUI session | **Each `/check` is its own numbered, distinctly coloured block, listed at once. Scheduling by stage: parse/resolve/fetch concurrent (≤ 3 runs), verify a single FIFO slot; states queued / running / waiting for verify / verifying / done / cancelled; `/cancel #n`.** Spec §13.1; Phase 8 |
| 8.15 | Mouse in the TUI | **Enabled (Textual native, no new dependency); nothing mouse-only, nothing keyboard-only.** Click targets: run header, stage line, finding (toggle detail), permission prompt buttons, `#n`/URL hyperlinks (OSC 8), copy-passage. Spec §13.1; Phase 8 |
| 8.18 | Colour system | **ANSI-16 only; meaning layer (green ok / yellow caution / red finding / dim NEI, key column bold) + rotating run accents (cyan, magenta, blue, bright cyan, bright magenta). Red/yellow/green never used as accents.** Spec §13.3; `ui.py` owns the tables |
| 8.19 | Pet | **A raven** (2026-09-16; was a ferret 2026-09-11 → v0.4.1). Two-tone pixel art, Braille in `rich`, ASCII in `plain`, no stamp, no animation. Spec `2026-09-16-raven-pet-design.md`. |

### New open items (Phase 4, from review)

| # | Item | Note |
|---|---|---|
| 8.16 | `FULLTEXT_MIN_WORDS = 1500` starves short papers | **Kept at 1500 (2026-09-11).** In the 50-DOI sample only 3 abstract-only results had a reachable page under the threshold, and each was a landing page (abstract + boilerplate), not a short paper. Revisit if a real short-paper case appears |
| 8.17 | Crossref `text/xml` TDM links are dropped | PLOS/Frontiers-style publishers serve JATS XML through Crossref links; `jats_body_text` already exists, so keeping them as `Location(..., "text")` is cheap. Decide after the coverage numbers |

### Phase 4 — done 2026-09-11 (night)

- Fetch ladder, consent gate, OA chain, `proofpath fetch`, `ui.py` + `config` group
  shipped; 458 tests, all offline. Live: Science.org 403 → `curl_cffi` → 200;
  coverage 72 / 18 / 10 % on 50 DOIs (`docs/eval/2026-09-11-coverage.md`), every
  miss carrying an honesty state; 12 of 50 wait on the browser consent.
- Findings that changed the code: DataCite arXiv DOIs carry their own arXiv id;
  a 200 page with zero extractable text is a bot wall; `scrapling`'s fetchers need
  playwright, so step 2 calls `curl_cffi` directly.
- CLI/TUI design decided and written into the spec (§13.1 pet, awaiting mode,
  multi-run scheduling, mouse; §13.3 surface, colours, exit codes).

### New open items

| # | Item | Note |
|---|---|---|
| 8.20 | ~~Browser step never exercised live~~ **Closed 2026-09-12 (Task 7.4)** | Run live: `proofpath fetch 10.1016/j.tibs.2014.10.005 --allow-browser` installed the engine through scrapling's CLI and read the paywalled Cell landing page at step 3 (3,632 words); log in `docs/eval/2026-09-12-v0.1-live.md` §8b |
| 8.21 | Semantic Scholar is the single biggest full-text source (23/36) | Its shared 1 req/s pool and occasional 429s make it the fragile link; 7.9 (free S2 API key) moves up in priority |
| 8.22 | `install_log` is now a general gate log | Rename to `gate.log`/`events` when the report layer (Phase 6) consumes it |
| 8.23 | Deferred polish from the Phase 4 reviews | `resolve.py` blanket `noqa: F401` on the re-export block; `fetch.py` `retry_after` float↔str round trip and no 3xx→`final_url` test; `polite.py` assert-based narrowing; `oa.py` `_JATS_BIBR_XREF` needs `ref-type` as first attribute; six `ui` tests pin exact ANSI sequences; `_json_default` dead branches; `evidence` line does not dim `none`; `Cache()` OSError → traceback instead of exit 2; OA and Fetcher hold separate `PoliteClient` throttles (Europe PMC search + fullTextXML not spaced); `_browser` loses redirects in `final_url`; `install()` is silent for minutes; a cached `doi:` abstract short-circuits a later `--allow-browser` run (use `--no-cache`) |
| 8.24 | Re-fetch with changed content keeps old chunks | `add_source` now upserts (keeps chunks/verdicts, spec §16). If the re-fetched raw text differs (`raw_text.sha256` changes), Phase 5 ingest must re-chunk and re-embed instead of trusting the stored chunks |

### Next session

1. **Phase 5:** document ingest (`pymupdf`, `python-docx`, markdown) and claim
   extraction with numeric markers only; `PARAGRAPH-SCOPED` and
   `UNSUPPORTED CITATION STYLE` states (spec §9).
2. Wire `oa.OpenAccess` + `Cache` into the pipeline so `check` produces its first
   end-to-end verdicts.

---

## 9. Phase 5 — done 2026-09-11 (night)

- Document model, ingest (PDF / docx / markdown / text) and claim extraction shipped;
  680 tests, all offline. Pairing rate on the hand set 0.98; four real documents (three
  PDFs and one extracted text) measured in `docs/eval/2026-09-11-pairing.md`.
- Decisions taken while building: PDF line numbers count every extracted line
  (headers/footers included) so `p.4 L3` matches what a reader sees; digits-only
  furniture is dropped only in the top/bottom 10 % band; a bibliography heading is a
  paragraph boundary even without a blank line; `Reference.raw` keeps its printed
  marker (`resolve._MARKER` strips every form ingest accepts); a superscript after
  punctuation is a footnote, after a letter or `)` a citation; sentence boundaries are
  never placed after a single capital initial (`J. Smith`), at the accepted cost of
  merging `vitamin D. The …`.

### New open items

| # | Item | Note |
|---|---|---|
| 9.1 | Byline affiliation superscripts become claims | 34 of AlphaFold's 129 claims come from the author list; needs block-level filtering (a byline is not prose). Phase 6 or 8 |
| 9.2 | ~~Mixed `(Smith, 2020; [12])` loses its author-year half~~ **Closed 2026-09-12 (Task 8.5)** | Both halves are markers now: the author-year marker keeps the whole parenthesis as its text and the numeric one keeps its brackets, and `strip_markers` drops the inner span it has already cut. Scoping changed as predicted — two markers in the carrying sentence block a paragraph-scoped citation — so `num-15`'s claim text closed up over the parenthesis. The numeric hand set went 107/109 → **108/109** (`docs/eval/2026-09-12-pairing-author-year.md`) |
| 9.3 | ~~Nature PDF without a `References` heading~~ **Closed 2026-09-12 (Task 8.7, with 11.4)** | AlphaFold's list is not detected, so the paper has **no reference list at all** and its markers cannot be checked against anything: they are neither resolved nor reported. The 2026-09-11 re-run shows the row as 126 markers, 0 references, **0 unresolved** — the citations are visible and not yet checkable, which the coverage line has to say. (An earlier note here claimed every marker was reported unresolved; it never was.) Try the last numbered block as a fallback in Phase 8 |
| 9.4 | Cross-bracket ranges `[1]-[3]` | paired as 1 and 3, an IEEE reader means 1–3 |
| 9.5 | `km²` / `m³` read as citations `[2]` / `[3]` | accepted cost of the superscript rule; only when the PDF marks the digit as superscript |
| 9.6 | `retrieval.sentence_spans` needs whitespace after a stop | `…cycles.[15] Later` stays one sentence. Not shared with the SciFact numbers as previously written, but not `ingest`-only either (as a later note here claimed): `verify._index_for` chunks fetched source text with `split_sentences` before embedding it, so a boundary change moves retrieval as well as claims. Left alone because a boundary before `[15]` with no whitespace after the stop would also split decimals and version strings (`v1.2`) |
| 9.7 | An unnumbered first bibliography entry is dropped | Text standing before the first printed number is heading residue as often as it is an entry, so ingest drops it and the list starts at `[2]`. The `[1]` citing that entry is now reported as unresolved (fix round 3) rather than pointed at entry 2 — visible, but a real reference nobody can reach |
| 9.8 | A decimal followed by a superscript reads as a citation | `0.5³` → `[3]`. The cost of keeping `OpenMM v.7.3.1⁶⁹`: a digit after a full stop no longer takes an exponent. Same trade as 9.5 — a false marker a reader can see, against a real citation nobody would |
| 9.9 | .docx footnotes, endnotes, text boxes and nested tables are not read | `_docx_lines` walks body-level paragraphs and top-level table rows only; python-docx exposes neither the footnote/endnote parts nor drawing-anchored text, and `cell.text` stops at the cell's own paragraphs. A claim printed in any of them is invisible, not reported |
| 9.10 | An unnumbered two-column bibliography is cut at line breaks | RoBERTa's ACL list yields **103 entries for ~50**: `split_references` falls back to one entry per paragraph, and the PDF hands each entry over as two paragraphs split at a hyphenated line break, so 54 of the 103 carry no author at all. Author-year pairing then resolves 5 of 96 items; rejoining an entry with the one before it whenever the first prints no year gives 47 entries and **65 claims instead of 15** (measured as a diagnostic, not shipped). The largest open item for author-year pairing |
| 9.11 | Author-year pairing matches the first author and the exact year only | `(Lindqvist, 2019)` against "Okafor, C. and Lindqvist, S. …" and `(Smith, 2019)` against an entry printing 2020 are both reported unresolved. A ±1 year tolerance would not widen a score here, it would *pick* an entry; matching every author needs the entry's author list parsed, which §8 refuses to do locally. Rows `hd-01` and `hd-04` of `tests/data/pairing_author_year.jsonl` |
| 9.12 | Two surnames sharing a last word collide | `(Berg, 2018)` against a list holding both "Berg, T." and "van der Berg, P." resolves to neither. The fold keeps only the last word so the body's `van der Berg` can meet the `Berg` that `author_hint` returns; the collision is paid as an ambiguous *report*, never a guess (row `hd-05`) |
| 9.13 | ~~`ibid.` does not reach back to a numeric marker~~ **Closed 2026-09-12 (Task 8.5, fix round 1)** | `ibid.` now takes the refs of the marker **immediately before it in reading order**, whatever style it was written in, and never reaches back past it: if that marker resolved to nothing, or stands more than one paragraph back, the `ibid.` is reported instead of being pointed at the citation before that. `op. cit.` is unchanged — it names an author, so it still searches the surnames cited so far. Rows `bk-08`, `bk-09`, `hd-03` |
| 9.14 | `Smith 2020` with no comma is not a marker | Inside a parenthesis the year must follow a comma or `et al.`, or `(Figure 2020)` becomes a citation. One citation missed in the styles that print no comma (row `hd-02`) |

### Next session

1. **Phase 6:** `verify()` (prepare / decide_all), cache wiring (chunks keyed by text
   sha256, verdict rows), `report.py`, `proofpath check`. Briefs in
   `.superpowers/sdd/phase6/`.

---

## 10. Phase 6 — done 2026-09-12

- `verify()` (prepare / decide_all), `report.py`, `proofpath check`, cache schema v2.
  All offline tests; the CLI runs end to end on a real DOI (20 s first run, full text on
  the second).
- Decisions taken while building: `Finding.state` is validated against `STATE_WORDS`
  (an `UNVERIFIED (…)` string is the only free form); `Coverage.weak()` is derived from
  the denominator so an under-counted producer cannot flatter a run; verdict rows are
  deleted inside `Cache.put_chunks`' transaction when a source's text digest changes
  (spec §16 keeps verdicts across a TTL expiry, never across different text);
  `permissions.network = deny` skips resolve and retraction stages too, each summarised
  as `not attempted (network not permitted)`; a cancelled `decide_all` raises
  `Cancelled(report=partial)` and the CLI renders it and exits 2; the caret under a
  numeric claim is dropped when the figure occurs twice in the sentence (never point at
  the wrong number); layout characters of a diagnostic are ASCII, content is verbatim.
- New state: `UNVERIFIED (reached, no text extracted)` (spec §15).

### New open items

| # | Item | Note |
|---|---|---|
| 10.1 | `oa._worst_outcome` labels "reached but no text" `UNVERIFIED (unreachable)` | `verify` uses `UNVERIFIED (reached, no text extracted)` for the same fact on URL sources; align `oa.py` in Phase 8 |
| 10.2 | A fully cached re-run still loads both ONNX models | `model_id` needs the model names; expose names without loading (Phase 8, with the TUI's multi-run scheduler) |
| 10.3 | ~~`Cache` schema versions compare as strings~~ **Closed 2026-09-12 (Task 8.7)** | `_version()` compares them as integers, and an unreadable version counts as the oldest file so the chain repairs it. Judgements become **v4** (`.superpowers/sdd/phase9/task-9.2-brief.md` updated) |
| 10.4 | `cache ls` counts stale chunk rows after a digest change until the next write | cosmetic |
| 10.5 | Exit code 1 for a document whose only finding is a `PARAGRAPH-SCOPED` note | spec §13.3 says every state counts; a `--fail-on` flag is the escape hatch if users object |
| 10.6 | Unexpected non-`ProviderError` exceptions abort `prepare` | a per-unit guard turning them into `PROVIDER_UNAVAILABLE` would match rule 6 better |
| 10.7 | Terminal and markdown word the weak-coverage warning differently | one voice, Phase 7 README pass |
| 10.8 | ~~Reference resolution and the retraction check are not cached~~ **Closed 2026-09-12 (Task 8.7)** | cache schema v3 adds `resolutions` (30-day TTL, keyed by the marker-free folded entry) and `retractions` (30 days for a notice, 7 for its absence); neither `UNVERIFIED (provider unavailable)` nor a retraction check every provider failed is ever stored (`Resolver.retraction` raises `ProviderError` for that, `prepare` reports `retraction check unavailable` and the stage summary counts it). `scripts/zero_network_check.py` on the warm draft: **0 blocked calls, 1.3 s**, all three network stages attributed to `cache` (`docs/eval/2026-09-12-v0.1-live.md` §7b) |
| 10.9 | For one run after the v1→v2 migration a changed source keeps its old verdicts | the digest set `put_chunks` reads is empty right after the migration, so it has nothing to compare the new text against and drops nothing; the run after that is correct |
| 10.10 | ~~Three concurrent TUI runs get three independent politeness limiters~~ **Closed 2026-09-12 (Task 8.2, `polite.SHARED_THROTTLE`)** | `PoliteClient._last_call` was per instance; spec §13.1 wants one shared across the runs a single user has open, and `SHARED_THROTTLE` is that one |
| 10.11 | `Cache` connections are thread-bound (`sqlite3.connect` is left at `check_same_thread=True`) | `prepare` and `decide_all` on different pool threads would fail; Phase 8's scheduler must give each run one dedicated thread, or `Cache` must become lock-protected |

### Next session

1. **Phase 7:** calibrated tiers (7.1), live user-like runs, README, CHANGELOG `[0.1.0]`,
   tag `v0.1.0` (briefs in `.superpowers/sdd/phase7/`).

---

## 11. Phase 7 — v0.1.0, 2026-09-12

- Tiers calibrated on SciFact dev (7.1): `decide 0.45`, `medium 0.457948`, `high 0.99933`
  — `high` is reachable (the Phase 1 "unreachable" was a two-decimal display artefact);
  the 85 % figure is an in-sample estimate on 21 verdicts and `low` is practically
  empty, both stated in spec §14 and `docs/eval/2026-09-12-tiers.md`.
- Live user-like runs (7.2, `docs/eval/2026-09-12-v0.1-live.md`) found what the unit
  and eval suites could not: a **false ghost** on a real NumPy reference (particle
  surname `van der Walt`) and a fabricated `[7]` reported as `not in indexes` because
  `Reference.raw` keeps its printed marker. Both fixed in 7.3; the ghost set grew to
  258 rows (particles, glued `al-`, marked strings, year-first and numeral-first
  entries) at 0.0 % false-ghost and 100 % fabricated recall
  (`docs/eval/2026-09-12-ghosts.md`). A piped markdown draft (`check -`) lost its
  `## References`; fixed in `ingest.py`.
- 7.4 fixed three more live findings: the browser step could not install itself
  (`scrapling` has no `__main__`; the package CLI is now invoked and a paywalled Cell
  page was read live at step 3 — closes 8.20), an intermittent exit `134` at ONNX
  teardown after a complete report (models released explicitly, streams flushed; 20/20
  runs clean), and a clean-looking `0 %` coverage block on a document with markers but
  no detected bibliography (now a hint naming the unchecked markers).

### New open items

| # | Item | Note |
|---|---|---|
| 11.1 | ~~Two-author lists `First Last and First Last` reach `AMBIGUOUS`, not `RESOLVED`~~ **Closed 2026-09-12 (Task 8.7, fix round 1)** | `_AUTHORS_FULL_NAMES` consumes a full-name list joined by `,`/`and`/`&`, but only when a full stop closes it **and what remains still reads like a reference** — ≥ 2 title-like segments, the first ≥ 3 words — so a title-first book (`Pattern Recognition and Machine Learning. Springer Verlag, Berlin, 2006.`) keeps its title. `looks_unindexed` asks `_strip_initials_authors`, never this pattern. Ghost set 271 rows, false-ghost 0.0 % |
| 11.2 | ~~The arXiv-id branch has no author+year rescue~~ **Closed 2026-09-12 (Task 8.7)** | mirrored from the DOI branch, same note (`title could not be matched in the reference string`); three real arXiv-id ghost-set rows now reach `RESOLVED (low confidence)`, and a mutated row with the wrong author and year still does not |
| 11.3 | ~~Resolution is the slow path of a cached re-run~~ **Half closed 2026-09-12 (Task 8.7)** | the `resolutions` cache (10.8) removes it from a *re*-run entirely (the live draft: 28.5 s cold → 10.8 s warm → 1.3 s with the network gone). A **cold** run of 129 references is still serial and still minutes long; concurrent resolve in the TUI scheduler is what is left, tracked as 11.10 |
| 11.4 | ~~Nature PDFs expose no `References` heading~~ **Closed 2026-09-12 (Task 8.7)** | `ingest.find_last_numbered_run` reads the list by its shape when no heading exists (paged formats only): last contiguous run of paragraphs whose lines open with a printed number, ≥ 5 entries, numbers ascending, bounded at the last numbered paragraph. AlphaFold: 0 → **17** references (59–75) and 0 → **97** reported unresolved markers (`docs/eval/2026-09-11-pairing.md`, 2026-09-12 re-run). Also closes 9.3 |
| 11.5 | `stdin` decoding replaces undecodable bytes silently | a `\ufffd` in a claim should be noted (rule 2) |
| 11.7 | ~~The §7.1 aggregate line `skipped N source(s) because the browser was not permitted` is not printed by `check`~~ **Closed 2026-09-12 (Task 8.7)** | `Footer.browser_skipped` carries `Coverage.browser_skipped`; `ui.skipped` prints it with a yellow count (never dropped by `-q`) and `_markdown_coverage` prints the same sentence, whose wording both surfaces import from `report.BROWSER_SKIPPED_REASON` |
| 11.8 | Bibliographies with ≥ 1000 entries lose ghost detection | `_MARKER` bare forms are capped at three digits, so `1024. Smith…` keeps its marker and exits as `NOT_INDEXED` (never a false ghost) |
| 11.9 | `mypy` covers `src/` only | `scripts/` is ruff-linted and its pure halves are tested, but not type-checked; two pre-existing errors in `scripts/eval_ghosts.py` |
| 11.10 | Cold resolution is still serial | what is left of 11.3: 129 references cold is minutes of one-at-a-time provider calls. The TUI scheduler's concurrent resolve (spec §13.1, decision 8.14) is the fix, and `polite.SHARED_THROTTLE` is what keeps it polite |
| 11.11 | The headless bibliography fallback takes only the *last* contiguous run | AlphaFold's two-column list is four blocks (1–29, 30–58, 59–75, 76–84) separated by body paragraphs, so 17 of 84 entries are recovered. Taking every dense block of ascending entries would recover all of them; a numbered table is the case that makes it harder |
| 11.13 | A fabricated `First Last and First Last. Title. Venue.` — **with no year between the names and the title** — reaches `NOT_INDEXED`, not `GHOST` | the price of 11.1's guard, **measured** (task 8.7 fix round 2) rather than predicted: 3 fabricated rows in the dated form `… and … . 2019. Title.` came back **2 `GHOST`, 1 accepted for an unrelated reason, 0 `NOT_INDEXED`** — that form is consumed by `_AUTHORS_THEN_YEAR`, an initials-path pattern, so the ghost call survives. Only the undated form loses it, pinned by two offline tests in `tests/test_resolve.py`. OPEN-ITEMS 7.10's ceiling, one citation style wider; rule 3 outranks recall |
| 11.14 | A proceedings **volume** record can resolve a fabricated paper cited into that volume | found by 11.13's first row: `Marcus Halvorsen and Priya Raghunathan. 2019. Latent drift correction for streaming recommender systems. In Proceedings of the 13th ACM Conference on Recommender Systems.` → `RESOLVED (low confidence)` against `10.1145/3298689`, title 1.0 / author false / year true. `title_score`'s proceedings-volume guard only fires when the candidate shares no token with `segments[0]`, and the volume title shares "recommender" and "systems" with the fabricated title. Pre-dates task 8.7; the ghost set's only fabricated acceptance (0.9 %) |
| 11.12 | A mixed block loses the prose printed before its first entry | the AlphaFold block holding entries 59–75 opens with a Methods sentence, which `split_references` drops as heading residue (9.7's rule). Cheap only once the fallback knows where the list truly starts |
| 11.6 | ~~`browser.is_installed()` checks imports, not a browser binary~~ **Closed 2026-09-12 (Task 8.7)** | `browser_binary_present()` looks for a `chromium*` directory under `PLAYWRIGHT_BROWSERS_PATH` or the platform default (`~/Library/Caches/ms-playwright`, `~/.cache/ms-playwright`, `%LOCALAPPDATA%\ms-playwright`); missing → the idempotent installer runs. The consent log gains `browser binary: found|missing` |
| 11.15 | ~~A provider's non-JSON body escapes `resolve.py` as a raw `JSONDecodeError`~~ **Closed 2026-09-15 (Task 8.6b)** — `resolve._json()` is the one place a body is decoded; a `ValueError` there is a `ProviderError("<host> answered with a non-JSON body (…)")`, so the reference is `UNVERIFIED (provider unavailable)` and a retraction check stays unmade (three tests in `test_resolve.py`) | found by Task 8.4's manual session: `proofpath resolve "Scott JC. Against the Grain… Yale University Press; 2017. doi:10.2307/J.CTT1PWT9W5"` ends in an uncaught traceback (`Expecting value: line 1 column 1`) instead of the reported `UNVERIFIED (provider unavailable)` state and exit 1 that spec §13.3 requires; inside the TUI the run ends `failed` with the exception on its header, so the app survives it. Fix scheduled by the 8.6b brief |
| 11.16 | ~~`browser.prompt_text` says `blocked this request (HTTP 200)` when the wall was a 200~~ **Closed 2026-09-15 (Task 8.6b)** — `answered without readable text (HTTP 200)` when the status is 200, unchanged otherwise | the ladder rightly classes a bot-wall page with a 200 status as blocked, but the §7.1 prompt then reads as a contradiction; wording only, shared by the CLI and the TUI. Fix scheduled by the 8.6b brief |
| 11.17 | `tui/app.py` is ~1730 lines | Task 8.4 put the mirrored verbs' renderers (`verb_lines`, `_resolve_lines`, `_fetch_lines`, `_config_lines`, `_cache_lines` and helpers, ~250 lines of pure functions over result objects) in `app.py` because the brief named one new module. They belong in a `tui/lines.py` with their own tests; nothing else in `app.py` depends on where they live |

### Next session

1. **Phase 8:** TUI (briefs 8.1–8.6 in `.superpowers/sdd/phase8/`), author-year pairing,
   SARIF, `commands.py` shared wiring, resolution cache → v0.2.0.

---

## 12. Phase 8 — v0.2.0, 2026-09-15

- Shipped: the TUI (8.1 banner and slash commands, 8.2 `Scheduler`, 8.3 app, 8.4 inline
  §7.1 prompt, mirrored verbs over `commands.py`, mouse/keyboard targets, the eyes),
  author-year pairing (8.5; hand set 0.940, `docs/eval/2026-09-12-pairing-author-year.md`),
  `sarif.py` (8.6a) wired as `check --format sarif` (8.6b), and the v0.1 field findings
  (8.7: resolution + retraction cache, browser-binary check, resolver rescues,
  headless-bibliography fallback; ghost set 274 rows at 0.0 % false-ghost).
- 8.6b: bare `proofpath` opens the TUI (config loaded first, so a broken file exits 2 on
  one line); `--format sarif` on `resolve`/`fetch` says `applies to check only`; 11.15
  and 11.16 fixed; `ui.json_text` is the one JSON spelling stdout and `--out` share.
- Live checks (`docs/eval/2026-09-15-v0.2-live.md`): the SARIF log validates against the
  vendored schema (14 results); warm re-run **1.35 s** with every network stage `cache`;
  the TUI driven in a real pty (`/check` → done → `/check` alone → `Esc` → `/cancel 1`
  → `/quit`, exit 0); `tests/data/draft-author-year.md` paired 6 of 6 markers with an
  offline smoke test.
- Not done, by decision: the `code` CLI was absent, so the SARIF file was validated but
  not opened in VS Code's viewer; the plan's "opens in VS Code" gate is carried by the
  schema validation and 8.6a's location tests.

### New open items

| # | Item | Note |
|---|---|---|
| 12.1 | Two-column unnumbered bibliographies (9.10) | still the largest author-year item: RoBERTa yields 103 entries for ~50 and 15 claims where a rejoin would give 65. Rejoining an entry with the one before it whenever the first prints no year was measured, not shipped |
| 12.2 | The undated fabricated pair form (11.13) and the proceedings-volume acceptance (11.14) | carried into v0.2 as Known issues; both are rule-3-first trades and stay open until a guard is found that does not cost a real reference |
| 12.3 | Headless fallback limits (11.11, 11.12) | only the last numbered run is taken and the prose before a block's first entry is dropped; AlphaFold recovers 17 of 84 |
| 12.4 | ~~`tui/app.py` is 1798 lines (11.17)~~ **Closed 2026-09-15 (v0.2.1, TUI v2 §5)** | split into `tui/theme.py`, `tui/pet.py`, `tui/verbs.py` (the mirrored verbs' renderers; their tests stay in `tests/test_tui_app.py`) and `tui/widgets/`; `app.py` is 681 lines of composition, wiring and slash commands |
| 12.5 | `#n` in a run header is a click target, not a hyperlink | spec §13.1 says "a `#n` reference or a source URL" is an OSC 8 link; a run number has no address, so the header folds its block (`enter`) and the link went to the finding's reference instead (8.4 deviation 3). Either the spec wording or a `proofpath://run/n` scheme settles it |
| 12.6 | ~~TUI finding rows print the entry's marker twice~~ **Closed 2026-09-15 (final review fix: `strip_marker` on the label)** | `[7] [7] Marchetti, …` in the pty session: `Reference.raw` keeps its printed marker (Phase 5) and `FindingLine` prefixes the number again. The CLI diagnostic prints the entry once. Cosmetic; `strip_marker` on the label is the fix |
| 12.7 | `loading models …` is drawn under the `Verifying` row in the TUI, above it on the CLI | the true order: `verify` emits the note inside the stage, after `StageStart`, and the TUI draws the row on `StageStart` while the CLI prints only `StageEnd`. Moving the note before `StageStart` in `verify.py` would put it outside the stage it belongs to; a `StageLine` that shows notes as its own children is the tidier fix. Cosmetic, not a `report`/`ui` one-liner. 2026-09-17: the note is now transient in the TUI (removed when the stage ends); its *position* is still open. |
| 12.8 | The SARIF artifact for `check -` is `-` | `to_sarif(report, artifact=target)` takes the target as typed; a stdin run has no file, and `-` under `%SRCROOT%` is what a viewer gets. `report.document.name` (`stdin`) is no better an address; a `--artifact` override is the only honest option |
| 12.9 | `api_calls` in the footer counts LLM calls only — **restated 2026-09-15 (v0.3.0)**: it is now the judge's real counter (answers, not attempts); provider lookups are still uncounted | a cold resolution of five references (33.5 s against Crossref and Semantic Scholar) still prints `0 API calls`; the counter is the judge's (Phase 9) and the wording says nothing about provider lookups. Rename or count — one line either way |
| 12.10 | A cached resolution reprints its notes as if current | the live draft's ghost still carries `openalex unavailable (HTTP 429, retry after 65567s)` from the day the resolution was stored. Correct (nothing was re-checked) but it reads like today's outage; a `cached:` prefix on stored notes would say so |
| 12.11 | A provider body that is valid JSON but not an object escapes `resolve` as an error | `_json` guards against a non-JSON body (`ProviderError`), but a bare list or string parses and then fails the `.get(...)` that follows with an `AttributeError`: exit 2 instead of `UNVERIFIED (provider unavailable)`. A type check in `_json` is the fix |
| 12.12 | `/quit` waits for an in-flight mirrored `/fetch` | a `/check` run is cancelled through its scheduler; a mirrored verb runs in `asyncio.to_thread` with no cancel hook, and the loop's shutdown joins that executor, so the app exits only when the ladder returns. Bounded by the fetch timeouts, but a browser step can take a while |
| 12.13 | ~~Phase-9 briefs are untracked~~ **Closed 2026-09-15**: Phase 9 shipped from the tracked `docs/superpowers/plans/2026-09-12-phases-9-10-plan.md`; the drift amendments live in the same git-ignored briefs and are summarised in §14 | — |

### Next session (as of v0.2.0 — superseded by §14)

1. Phases 9–10 were deferred after v0.2.0 and resumed 2026-09-15; Phase 9 is §14.

## 13. TUI v2 — v0.2.1, 2026-09-15

- Shipped: the two themes (`tui/theme.py`, detection by injected environment,
  `PROOFPATH_THEME` override), the seven-line `RICH` ferret, run panels with the state
  on the border and the coverage on its foot, the fixed-column stage table with a real
  progress bar, badges, `you`/`source` rows, the coverage bar in the footer, the
  bordered prompt, and the `app.py` split (12.4). Design:
  `docs/superpowers/specs/2026-09-15-tui-v2-design.md`; spec §13.1 and §13.3 amended
  by reference. Live session in both themes with SVG screenshots:
  `docs/eval/2026-09-15-tui-v2-live.md`. 1723 tests pass, 1 skipped.
- The T4 follow-ups: a run that finished below the panel floor no longer shows its
  coverage twice after widening (and one that finished wide keeps it when narrowed —
  the `CoverageLine` is always mounted and shown only while the block is flat); a
  tier-less badge keeps its row at 60–79 columns.

### New open items

| # | Item | Note |
|---|---|---|
| 13.1 | Badges do not end on one column at 60–79 columns when a tier-less finding would otherwise stack | `FindingLine._draw` drops the empty six-column tier cell only when the label's floor would not fit beside it (the T4 follow-up). At those widths a `GHOST REFERENCE` row ends eight columns right of a `NOT SUPPORTED  high` row. Kept: a stacked row costs a line per finding, a ragged right edge costs nothing a reader misreads. `≥ 80` columns is unaffected |
| 13.2 | `⏺` (U+23FA) is an `Emoji=Yes` character | Every width table says one cell (`unicodedata` EAW `N`, `wcwidth` 1, Rich 1), and in Chromium's font fallback on this Mac — which is what VS Code's terminal draws with — it measures 1.04 cells and is drawn as a text glyph in the accent colour, so the switch to `●` was **not** made (T5 brief: only if it renders wide). Terminals that give `Emoji=Yes` characters emoji presentation (Apple Terminal with some fonts, Windows Terminal) may draw it as the colour "record" glyph, two cells wide and ignoring the accent. If a report says so, `Glyphs.stage_active` and `stage_flag` in `tui/theme.py` become `●` (U+25CF, EAW `A`, one cell outside CJK locales, which already select `PLAIN`) — two characters and three golden strings. `⧉` (U+29C9) measures 1.29 cells in the same fallback; it sits before a space and the border, so the overflow is invisible, but a narrower copy glyph would be safer |
| 13.3 | `PLAIN` keeps the flat v0.2.0 layout the author rated 2/10 (now pure ASCII, with `= note:` rows) | By design (v2 §2, "nothing regresses where borders cannot draw"): the flat rows, the three-line pet and the `kv` footer are v0.2.0's layout with ASCII glyphs, so a legacy-conhost, `NO_COLOR` or `-q` user sees the prototype. A `PLAIN` pass — the fixed-column stage table needs no border and would fit — is the one improvement that does not touch the ASCII rule |
| 13.4 | The exported SVGs reference a webfont | `App.save_screenshot` (Rich's exporter) emits an `@font-face` for Fira Code with `local()` first and a `cdnjs` URL second; the files are 96 KB and 66 KB, no network is needed to read them, but an offline viewer without Fira Code falls back to its own monospace and the box drawing may not join |
| 13.5 | 12.7 still shows in the live session | `loading models …` is drawn under the `Verifying` row in both themes (true event order); the tidier fix — a `StageLine` that owns its notes — was out of T1–T5's scope |

## 14. Phase 9 — v0.3.0, 2026-09-15

- Shipped: `JudgeClient` (9.1; Groq / Gemini / Ollama over `chat/completions`, packaged
  prompts, retry/backoff, cost), `Judge` escalation with the `judgements` cache (9.2;
  schema v4, `check --judge`), `--summarize` and the TUI `/summarize` (9.3). Briefs under
  `.superpowers/sdd/phase9/` with their 2026-09-15 drift amendments; plan
  `plans/2026-09-12-phases-9-10-plan.md`.
- Live (`docs/eval/2026-09-15-judge-live.md`): on `tests/data/draft-live.md` the judge
  escalated 1 of 10 verdicts (1 call, 613 prompt · 193 completion tokens) and answered
  `NEI` beside a low-tier `NOT SUPPORTED`; the summary cost 1 call (1,608 · 343); the first `--summarize` attempt came back
  empty with `finish_reason=length` — `openai/gpt-oss-120b` spends its completion budget
  on reasoning — which is why requests now carry `reasoning_effort=low` and larger
  budgets. The v0.1 lesson held again: the suite was green, the live run found the defect.
- Reviewer-found before the tag: an empty 200 was counted as an answer (9.1); an
  unavailable judge reached only the terminal `Note` (9.2); an unanswered summary was
  invisible in the markdown file (9.3). All three were rule-2/6 failures and were fixed.

### New open items

| # | Item | Note |
|---|---|---|
| 14.1 | The `NEI`-with-a-passage escalation clause is unreachable | `pipeline.aggregate` and `decide_indexed` return `Verdict(NEI, …, passage=None)`, so the escalation set is the `low` tier only (a ~0.008-wide band, 1 of 10 verdicts on the live draft). Widening it is a pipeline decision: attach the best passage to an `NEI` and let the judge see it, or narrow the clause in the spec |
| 14.2 | An agreeing opinion leaves no mark on a rendered finding | by design only a disagreement adds the `= judge:` line; agreement shows in the markdown `judge` column and in JSON. A markdown reader of a low-tier `NOT SUPPORTED` cannot tell "judge confirmed" from "judge never saw it" without the table |
| 14.3 | Judgements are matched by `(claim_hash, source_id)` | two results with identical claim text against the same source share one opinion and are asked twice in one batch. Rare; carrying the result index would settle it |
| 14.4 | `Progress` on the Judging stage counts asked items, the summary counts all results | a fully warm run emits no `Progress` and reports `1 of 10 verdicts reviewed` |
| 14.5 | `report.py` and `cache.py` import `judge` | no cycle, but `httpx` now loads behind the output and storage layers; `JudgeOpinion`/`JudgeCost` could move to `models.py` |
| 14.6 | `Cache.detail()` / `Cleared` do not count judgements | they are removed by cascade and on `clear`, just not reported |
| 14.7 | A wrong or revoked key (401/403) arrives as `JudgeUnavailable` | same type as an outage; the detail names the HTTP status, so it is reportable, but a config error and a provider outage are one exception type |
| 14.8 | ~~`--format json` carries `summary` with no inline model-written label~~ **Closed by the whole-phase review: `summary_model` sits beside it** | attribution is `models["judge"]` plus the `Summarising` stage; an inline label would change the schema |
| 14.9 | An empty 200 is not retried while a 5xx is | right for `content_filter`, pessimistic for a flaky truncation |
| 14.10 | Small test gaps | the bare-429 ladder (no `Retry-After`) is untested; `finish_reason` is echoed unbounded in the unavailable detail; the packaged-prompt test cannot fail on an editable install (the wheel listing is the evidence) |
| 14.13 | The CLI's unavailable-judge line stutters | it prints `judge      judge unavailable after 0 calls (…)` because the line returns the one shared sentence verbatim while the markdown header and the sibling `summary` line strip the leading subject. Kept for v0.3.0: byte-identical wording on every surface was the point, and the same string is also printed as a `Note` where the subject is needed |
| 14.14 | The 3,500-token batch cap rests on a rough estimator | `estimate_tokens` is `len(text) // 4 + 8` and says so; 3,500 + 4,096 leaves ~400 tokens under Groq's 8K tier, so a badly estimated batch can still meet a 429 (waited out, then reported). Calibrating the estimator against the `usage` block the provider already returns would settle it |
| 14.11 | The unanswered judge is absent from the SARIF log | `_run_properties` carries `models` and `apiCalls` but no `judge status` / `summary status`, so a `--format sarif` log of a run whose provider was down reads like one with nothing to escalate. SARIF is a findings document, and the markdown, JSON and terminal all say it; still, a viewer-only workflow does not see it |
| 14.12 | Groq's free tier is metered on prompt **plus** requested answer | the whole-phase review caught `TOKEN_CAP = 7000` left over from the 1,024-token answer budget: with the 4,096-token budget of 9.3 a full batch would ask for ~11.1k against an 8K/minute tier. The cap is now 3,500. Nothing measured this live — the live escalation set was one item |

### Carried into Phase 10

| # | Item | Note |
|---|---|---|
| 14.21 | `Document.kind` now mixes origin with citation style | the five original values say where a document came from (`pdf`, `docx`, `markdown`, `text`, `post`); `linked`, added so `claims.extract` could stay a pure function of the document, says how it cites — a paste is `text` in origin and the field no longer records that. The clean shape is a second field (`cites: "marker" | "link"`), blocked today only because `Report.document` is serialised field by field into the frozen golden payload, so any new field rewrites it. Visible to a user only as `document.kind` in `--format json` |
| 14.16 | A post is never cached | the fetch cache lives inside `Fetcher`, which the social provider bypasses, so a post is re-read on every run and `--no-cache` changes nothing for one. Nothing stale is served (the chunk cache is pinned to the text's digest), but the `Fetching` line can never attribute a post to `cache` |
| 14.17 | A post's text is graded `fulltext` at any length | `text_kind` answers "how much of the source was read", and an API-returned post is complete, so `fulltext` is the honest word — but a three-word post cited by a bibliography then counts as a full-text source in the coverage block. Distinguishing "short source, read whole" from "long document read whole" is a `Coverage` change, not a grading one |
| 14.18 | A directly constructed `Engine` builds a `PoliteClient` it never closes | `Engine.default` is clean (it passes the client already on `_closers`); a hand-built one allocates an `httpx` transport per engine. No socket opens until a post is read, so nothing leaks functionally |
| 14.19 | Pasted text with no links gets no "nothing to verify" note | the stage-1 note is gated on the post path, and a paste carrying no addresses never becomes `kind="linked"`, so it prints zero findings — which is what a clean run looks like. Spec §6.2 makes this the designated path for an unreadable post, so it is the one place the gap matters |
| 14.20 | `check --url` flattens two §15 states into one exit code | an unreachable post and an unavailable provider both exit 2; the hint distinguishes them, the exit code does not |
| 14.15 | The resolution cache is shared across routing changes | `cache.put_resolution` stores whatever the entry's provider answered under the entry's raw string, so a row written by one release is read back by another whose routing differs. Both of Task 10.1's warm-cache defects (a `doi:` source read by the web provider; the same source skipping the retraction check) reached the code through exactly this door, and it will open again whenever a provider's routing moves |

### Next session

1. **Phase 10** (`providers/` refactor, Bluesky/HN, Reddit/Mastodon/X, AVeriTeC) runs
   from `plans/2026-09-12-phases-9-10-plan.md`; Task 10.4a (AVeriTeC loader and scorer)
   is already implemented and reviewed, unstaged behind the v0.3.0 tag.

## 15. Phase 10 — v0.4.0, 2026-09-16

- Shipped: the `providers/` package (10.1; academic and web moved behind the §5.2 protocol
  with the report byte-identical, pinned by a golden payload written before the refactor),
  Bluesky and Hacker News with `check --url` and pasted-text links (10.2), Reddit/Mastodon/X
  and the per-reason coverage block (10.3), and the AVeriTeC harness (10.4).
- **The headline measurement is below baseline and is reported as such everywhere**:
  0.270 3-way against a 0.708 majority baseline on 100 AVeriTeC dev claims
  (`docs/eval/2026-09-16-averitec.md`). A third of the claims had no readable source; on the
  rest the score is 0.361; every `Supported` claim was missed. Nothing was tuned after the
  run and no blocked URL was dropped from it.
- Reviewers caught, before the tag: a bare DOI/arXiv URL losing its record and its retraction
  check; a warm-cache `doi:` source read up the web ladder with the open-access chain never
  asked, and retraction-checked by a provider that always answers "no notice"; the `Resolving`
  line naming providers nobody called; a PDF's body sentences paired to a bibliography entry
  they never cited; `check --url` fetching before the network permission was read. Five
  product-rule failures the 1,900-test suite did not catch on its own.

### New open items

| # | Item | Note |
|---|---|---|
| 15.1 | proofpath is below baseline on real-world web claims | the two causes are separable: a third of the claims had nothing readable (the browser step is off by default, `robots.txt` is honoured, news sites block), and the entailment models were calibrated on scientific abstracts. Neither is fixed by tuning a threshold. A browser-enabled run was not measured |
| 15.2 | The Reddit path has never run against Reddit | fixtures reproduce the documented shapes; nobody on this machine has an app to register. The first real-credential run should check the comment-listing shape first |
| 15.3 | `/r/<sub>/s/<hash>` share links are unrecognised | they degrade to the web ladder, so nothing is invented; the shape is simply not parsed |
| 15.4 | The TUI footer ignores `Footer.reasons` | the per-reason coverage lines reach the terminal and the markdown report but not the TUI, so a TUI run whose sources were half blocked still shows three numbers and no reason |
| 15.7 | A Mastodon status is read outside the fetch ladder | `check --url https://<any host>/@a/123` issues a request to that host's `/api/v1/statuses/123` directly: no `robots.txt` check and no entry in `polite.MIN_INTERVAL`. The user typed the address, so it is not a rule breach, but it is the one read path that skips the politeness machinery. *v0.4.7:* Lemmy shares this exactly — the host is the instance — with one narrowing: only a host that passes `providers.is_lemmy_host` is asked, so the request never goes to an arbitrary site |
| 15.8 | `target_document`'s `network_allowed` defaults to `True` | every in-tree caller passes the resolved permission, so the default is unreachable today; it is a footgun for the next caller and should be a required keyword |
| 15.9 | The Reddit token exchange follows redirects with basic auth attached | httpx re-applies the credential along a redirect chain. The URL is a fixed Reddit constant, so it needs Reddit itself to be the attacker, but `follow_redirects=False` on that one call costs nothing |
| 15.5 | `NO_REASON_RECORDED` is a user-visible state outside §15 | it is a guard that should never print; if it can print, it belongs in the table |
| 15.6 | `NOT_READ_HERE` is provably unreachable through `verify` | kept deliberately for a sixth platform, like `checker_for`'s `arxiv:` row. *v0.4.7:* the sixth and seventh (Lobste.rs, Lemmy) arrived and each got a reader, so it is still unreachable and still kept, now for the eighth |

## 16. Page targets — v0.4.4, 2026-09-17

A bare address on any host that is not a platform is read as the document itself
(`2026-09-17-page-target-design.md`).

### New open items

| # | Item | Note |
|---|---|---|
| 16.1 | No cap on a link-heavy page | a Wikipedia article carries hundreds of links and every paragraph's links are fetched as sources; coverage is reported and the run can be cancelled, but a cap with a reported remainder would be kinder |
| 16.2 | A page's words and `href`s inside `PAGE_SKIP_TAGS` are dropped with the element | `aside`, `form`, `button`, `svg`, `template` on top of the ladder's boilerplate; a site that puts its citations in an `aside` loses them silently |
| 16.5 | `find_bibliography` knows five headings | a page (or paper) whose list sits under "Sources", "Notes", "Footnotes" or "Citations" is not seen to print one; a page then cites by its links and says so (`MARKERS_SET_ASIDE`), a paper reports its markers unresolved. Widening `_BIBLIOGRAPHY_HEADING` needs the eval sets re-run, so it is not done here |
| 16.4 | A page's self-link rule ignores the query | `_page_key` drops `?utm_source=` and `?page=2` alike, so a page that keys its articles by query (`?id=123`) would drop a link to a sibling article as a link to itself; no such page has been seen |
| 16.3 | Scheme-less links in pasted text are not references | `resolve._HTTP_URL` wants `http(s)://`; `nature.com/articles/…` in a pasted paragraph is verified against nothing |

## 17. Source-less claims — 2026-09-18

A bare claim pasted with no address, bibliography or marker now ends with a `PARSE
ERROR` saying the text links or cites no source and what to paste instead (v0.4.6+,
`04341ea`). It does not end with anything checked, because nothing can be: the tool
verifies the sources *behind* a claim and goes looking for none.

The case that prompted it: a tweet. X cannot be read (§3 non-goal), so the user is
told to paste the text — and a tweet that links nothing then has nothing to verify
against. "Paste the text" is only an answer for a tweet that links its source.

### New open items

| # | Item | Note |
|---|---|---|
| 17.1 | **Evidence search for a source-less claim** (decided 2026-09-18: this is the fix, path B) | An opt-in stage for a document that cites nothing: query a web search provider with the user's own key (Tavily / Brave / SearXNG; consent per §7.1, never on by default), take the top N pages and run them through the existing fetch → retrieval → entailment path. The report must carry a **distinct state** for such a source — `found by proofpath, not cited by the author` — on every finding and in coverage; rules 1–2 stay: passage or no verdict, unreachable stays unreachable. **Gate:** AVeriTeC (§15.1) measures exactly this scenario at 0.270 vs a 0.708 majority baseline, so the stage ships only once a web-claim eval is above baseline, or it ships flagged experimental with that number printed in the report. Not the judge: a judge answering with no source is the truth oracle §3 forbids |
| 17.2 | README:280 and the `check --url` X message overstate "paste the text" | both should say a tweet with no link cannot be checked, until 17.1 lands |
| 17.3 | The "cites nothing" finding fires for a whole document, never per paragraph | a long pasted post where only some paragraphs link is fine today (it has references); a `.txt` mixing cited and uncited claims reports nothing about the uncited ones — that is the existing per-claim coverage's job, but worth checking it says so |

## 18. Lobste.rs and Lemmy — v0.4.7, 2026-09-18

Two readers added on the Hacker News and Mastodon patterns respectively
(`providers/social.py`). Both probed without auth and read live before release (spec §6.2).

### New open items

| # | Item | Note |
|---|---|---|
| 18.1 | The Lemmy instance list is a list | `providers.LEMMY_HOSTS` is five hosts plus the `lemmy.` prefix, chosen 2026-09-18 over shape-only matching so no `/api/v3` request is sent to a blog whose articles live at `/post/<n>`. A big instance not on it is read as a page. A probe path — try `/api/v3/site` once per unknown host and remember the answer — would widen coverage without the list, at the cost of one request to every `/post/<n>` address that is not Lemmy |
| 18.2 | Lemmy cross-posts and Lobste.rs "hats" are not read | a Lemmy post's `cross_posts` list names the same link in other communities; neither is a source of the post and neither is followed. Recorded so nobody reads their absence as an oversight |
| 18.3 | A Lobste.rs comment permalink's story is never read | `/c/<id>` reads the comment alone (its parent is other people's words, as on Hacker News). A user who pastes `/s/<id>/<slug>#c_<id>` meaning "this thread" gets the one comment; the address says so, but the report does not |
| 18.4 | `_is_own_story` sees only absolute anchors | a relative `href="/s/<id>"` in a Lobste.rs description has no host and passes the filter; downstream `find_url` reads no address out of it either, so nothing is fetched, but the link is not named among the dropped ones |
| 18.5 | `ProviderError.code` is read by one caller | the token is parsed for every 4xx/5xx a `PoliteClient` raises; Crossref, OpenAlex and the rest could use it the same way the Lemmy reader does, and today none does |
