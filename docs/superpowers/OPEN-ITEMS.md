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
| Pet family and names: one ASCII animal per project in the same drawing style with the same single red stamp (proofpath ferret · reasonhound hound · spiyweb spider); names only if the landing page presents them as characters, chosen for all three at once | when the landing site is built |
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
| 7.1 | Tier cut-points: precision targets 0.85/0.70 leave `high` unreachable on this model | **Deferred to after Phase 2**; the numeric layer changes the numbers. Pipeline default stays `decide=0.5, high=0.9, medium=0.7` |
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
| 8.19 | Pet | **A ferret** ("ferrets out the facts"; body `~` = the proof path ending in a red `[PROOF]` stamp — the banner's only colour). Pure ASCII, re-flows with width, eyes animate by run state in the TUI only. Owl rejected (Syft, Odoo Owl, Owl language), lynx rejected (`lynx` terminal browser). Spec §13.1; Phase 8 |

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
| 8.20 | Browser step never exercised live | `--allow-browser` path (pip/uv install + `StealthyFetcher`) is unit-tested with mocks only. Run once by hand before v0.1: `proofpath fetch <blocked url> --allow-browser` on a machine where the ~280 MB download is acceptable |
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
| 9.2 | Mixed `(Smith, 2020; [12])` loses its author-year half | `find_markers` lets the numeric marker win the overlap; fixing it changes paragraph scoping. Phase 8 with author-year pairing |
| 9.3 | Nature PDF without a `References` heading | AlphaFold's list is not detected, so the paper has **no reference list at all** and its markers cannot be checked against anything: they are neither resolved nor reported. The 2026-09-11 re-run shows the row as 126 markers, 0 references, **0 unresolved** — the citations are visible and not yet checkable, which the coverage line has to say. (An earlier note here claimed every marker was reported unresolved; it never was.) Try the last numbered block as a fallback in Phase 8 |
| 9.4 | Cross-bracket ranges `[1]-[3]` | paired as 1 and 3, an IEEE reader means 1–3 |
| 9.5 | `km²` / `m³` read as citations `[2]` / `[3]` | accepted cost of the superscript rule; only when the PDF marks the digit as superscript |
| 9.6 | `retrieval.sentence_spans` needs whitespace after a stop | `…cycles.[15] Later` stays one sentence. Not shared with the SciFact numbers as previously written — `sentence_spans` / `split_sentences` is used by `ingest` alone today. Left alone because a boundary before `[15]` with no whitespace after the stop would also split decimals and version strings (`v1.2`) |
| 9.7 | An unnumbered first bibliography entry is dropped | Text standing before the first printed number is heading residue as often as it is an entry, so ingest drops it and the list starts at `[2]`. The `[1]` citing that entry is now reported as unresolved (fix round 3) rather than pointed at entry 2 — visible, but a real reference nobody can reach |
| 9.8 | A decimal followed by a superscript reads as a citation | `0.5³` → `[3]`. The cost of keeping `OpenMM v.7.3.1⁶⁹`: a digit after a full stop no longer takes an exponent. Same trade as 9.5 — a false marker a reader can see, against a real citation nobody would |
| 9.9 | .docx footnotes, endnotes, text boxes and nested tables are not read | `_docx_lines` walks body-level paragraphs and top-level table rows only; python-docx exposes neither the footnote/endnote parts nor drawing-anchored text, and `cell.text` stops at the cell's own paragraphs. A claim printed in any of them is invisible, not reported |

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
| 10.3 | `Cache` schema versions compare as strings | fine below v10; Phase 9 bumps to "3" |
| 10.4 | `cache ls` counts stale chunk rows after a digest change until the next write | cosmetic |
| 10.5 | Exit code 1 for a document whose only finding is a `PARAGRAPH-SCOPED` note | spec §13.3 says every state counts; a `--fail-on` flag is the escape hatch if users object |
| 10.6 | Unexpected non-`ProviderError` exceptions abort `prepare` | a per-unit guard turning them into `PROVIDER_UNAVAILABLE` would match rule 6 better |
| 10.7 | Terminal and markdown word the weak-coverage warning differently | one voice, Phase 7 README pass |
| 10.8 | Reference resolution and the retraction check are not cached | every run re-queries Crossref/S2/Retraction Watch, so a re-run is not offline however warm the cache is; a `resolutions` table (Phase 8) would make it truly offline |
| 10.9 | For one run after the v1→v2 migration a changed source keeps its old verdicts | the digest set `put_chunks` reads is empty right after the migration, so it has nothing to compare the new text against and drops nothing; the run after that is correct |
| 10.10 | Three concurrent TUI runs get three independent politeness limiters | `PoliteClient._last_call` is per instance; spec §13.1 wants one shared across the runs a single user has open |
| 10.11 | `Cache` connections are thread-bound (`sqlite3.connect` is left at `check_same_thread=True`) | `prepare` and `decide_all` on different pool threads would fail; Phase 8's scheduler must give each run one dedicated thread, or `Cache` must become lock-protected |

### Next session

1. **Phase 7:** calibrated tiers (7.1), live user-like runs, README, CHANGELOG `[0.1.0]`,
   tag `v0.1.0` (briefs in `.superpowers/sdd/phase7/`).

