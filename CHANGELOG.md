# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
