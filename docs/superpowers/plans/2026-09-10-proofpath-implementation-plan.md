# proofpath — Implementation Plan

**Date:** 2026-09-10
**Spec:** `docs/superpowers/specs/2026-09-10-proofpath-design.md`

## Ordering principle

Phases are ordered by **risk retired per unit of work**, not by how a user would
experience the product.

The riskiest unknown is not PDF parsing or HTTP fetching — those are known-hard but
known-solvable. The riskiest unknown is whether local retrieval plus entailment can
judge a claim against a source accurately enough to be worth shipping. If that
answer is no, every other phase is wasted effort.

SciFact supplies claims already paired with their source abstracts. That means the
core quality question can be answered with **no PDF parsing, no network, and no
reference resolution** — the cheapest possible test of the most expensive risk. So it
comes first.

Each phase states what "done" means as a check that can be run, not as a feeling.

---

## Phase 0 — Foundations

**Goal:** a repository where every later phase can be verified automatically.

| Task | How |
|---|---|
| Package skeleton | `src/proofpath`, hatchling, `py.typed` |
| CLI entry point | Typer app with `invoke_without_command=True`; bare `proofpath` reaches the TUI stub |
| Config & permissions module | `platformdirs` config dir, TOML read/write, `ask/allow/deny` resolution, **TTY detection** |
| CI | ruff + mypy strict + pytest on Linux/macOS/Windows × py3.10/3.13 |

**Done when:** `proofpath --version` works on all three platforms in CI, and
`proofpath permissions` prints the config path and current values.

**Status (2026-09-11): done.** Repository, spec, tooling, packaging, CI (7 jobs
green), the 0.0.1 PyPI release, and the config & permissions module
(`proofpath permissions`, `ask` + no-TTY → `deny`).

---

## Phase 1 — Core quality spike (offline, SciFact only)

**Goal:** answer "is the entailment core good enough?" before building anything
around it.

No network. No PDF. Input is a SciFact claim plus its gold abstract.

| Task | How |
|---|---|
| Load SciFact | the AI2 tarball via `httpx`, sha256 pinned, cached under the platform cache dir — the HF loader is script-based and unusable with `datasets ≥ 4` |
| Chunk & embed | ONNX embeddings via `fastembed`; abstracts are short, so sentence-level chunks |
| Rank passages | cosine similarity in `sqlite-vec`; top-k with k tuned on dev split |
| Entailment | `cross-encoder/nli-deberta-v3-base` @ `6c749ce`, int8 ONNX chosen by CPU arch, run with `onnxruntime` + `tokenizers` → `SUPPORTED / REFUTED / NEI` + score |
| Calibration | threshold sweep on dev → the `NEI` cut-off, the judge escalation threshold, and the `high / medium / low` tier cut-points shown in reports |
| Harness | `scripts/eval_scifact.py` printing per-stage metrics against two baselines (majority label, "source exists → SUPPORTED"); results table committed under `docs/eval/` |

**Done when:**
- Label accuracy and rationale selection are measured and written into the repo.
- The result **beats a "cited source exists → SUPPORTED" baseline** by a clear margin.
- Every `SUPPORTED`/`REFUTED` produced carries the passage it came from.

**Kill criterion:** if accuracy lands near the trivial baseline even after threshold
tuning, stop and reconsider the approach before Phase 2. This is the phase that is
allowed to end the project cheaply.

**Status (2026-09-11): done, passed.** Accuracy 0.606 vs 0.406 baseline on 340 dev
pairs, recall@3 0.85, 0 verdicts without a passage. See
`docs/eval/2026-09-11-scifact-dev.md`. Tier cut-points still need a decision
(OPEN-ITEMS 7.1).

---

## Phase 2 — Numeric claim layer

**Goal:** catch the failure NLI is worst at, which is also the most common real one.

"The method gives a 40% speedup" against "we observed a 4-8% improvement in
throughput" is the canonical case. Generic entailment models often call that
supported.

| Task | How |
|---|---|
| Extract numeric spans | `(value, unit, direction, subject)` from claim and passage |
| Normalize units | shared unit table; percentages, factors, absolute values |
| Compare | magnitude comparison with an explicit tolerance |
| Route | numeric mismatch → `REFUTED (numeric mismatch)` showing **both** figures; no numeric span → fall through to NLI |

**Done when:** a hand-built numeric test set passes, and adding this layer does not
lower Phase 1's SciFact numbers.

**Status (2026-09-11): done.** 51 hand-built cases in `tests/test_numerics.py`;
SciFact dev accuracy with the layer 0.609 vs 0.606 without, one correct firing.
The rule set was tightened three times against real dev failures (spec §10).

---

## Phase 3 — Reference resolution and ghost detection

**Goal:** decide whether a cited source is real, without ever using Crossref's match
score as evidence (spec §8).

| Task | How |
|---|---|
| Candidate lookup | raw reference string → Crossref `query.bibliographic`; **no local bibliography parsing** |
| Field verification | author surnames, year ±1, title token-set Jaccard, venue fuzzy match |
| Verdict mapping | `RESOLVED` / `RESOLVED (low confidence)` / `AMBIGUOUS` / `GHOST REFERENCE` |
| Retraction check | Retraction Watch via Crossref, keyed on resolved DOI |
| Ghost test set | ~100 real references from real papers + ~100 fabricated ones, hand-built and committed |

**Done when:**
- **False-ghost rate is at or near zero** on the test set. This gate outranks ghost
  recall: calling a researcher's real citation fabricated destroys trust in one shot.
- The fabricated reference from spec §8 that scored 38.5 is correctly rejected.
- Unit tests use recorded fixtures, not live API calls.

---

## Phase 4 — Fetch ladder and permissions

**Goal:** get source text from the open web without silent failures or silent
installs.

| Task | How |
|---|---|
| Step 1 | `httpx`, descriptive User-Agent, `robots.txt` via `protego` |
| Step 2 | `scrapling` + `curl_cffi` TLS impersonation (bundled, 2.7 MB) |
| Step 3 | browser engine, **behind the consent prompt** (spec §7.1) |
| Step 4 | Wayback Machine fallback |
| OA chain | `pdf_url` → PMC → arXiv → landing page scrape → abstract |
| Honesty states | `UNVERIFIED (blocked / unreachable / provider unavailable)` |
| Politeness | backoff on 429/5xx, optional contact address from config or env |

**Done when:**
- Science.org (403 in testing) is either retrieved through step 2/3 or reported as
  `UNVERIFIED (blocked)` — never silently dropped.
- With no TTY, an `ask` permission resolves to `deny` **without prompting**, and the
  report says how many sources were skipped for that reason.
- Content type comes from headers, verified by a test serving a PDF from a
  `.html` URL.

---

## Phase 5 — Document ingest and claim extraction

**Goal:** turn a real draft into locatable claims.

| Task | How |
|---|---|
| Parse | `pymupdf` for PDF, `python-docx`, plain md/txt |
| Locators | keep page and line numbers per sentence, needed for SARIF later |
| Citation markers | **v0.1:** numeric `[12]`, `[12,15]`, `[12-15]` only. Author-year `(Smith et al., 2020)` is detected and reported as `UNSUPPORTED CITATION STYLE`; pairing it is a **v0.2** task (Phase 8) with its own test set |
| Pairing | marker → carrying sentence; when the marker ends a paragraph and the sentence has no other marker, every sentence of the paragraph is verified and grouped as `PARAGRAPH-SCOPED` |
| Bibliography | extracted as **raw strings only**, handed to Phase 3 unparsed |

**Done when:** on a corpus of real open-access PDFs, citation markers are paired with
the correct sentence at a measured rate that is written into the repo, and every
finding can be traced back to a page and line.

---

## Phase 6 — Report and coverage

**Goal:** make the tool's own limits visible in every output.

| Task | How |
|---|---|
| Markdown report | per-line findings, verdict, confidence, quoted passage, fetch step used |
| JSON report | same data, machine-readable |
| Coverage summary | full text % / abstract only % / unverified %, always present |
| Cache | sqlite, keyed `(claim_hash, source_id, model_id)` |

**Done when:** a report with low coverage is visibly different from a clean one at a
glance, and a second run of the same document makes zero network calls.

---

## Phase 7 — v0.1 release

Academic provider, CLI only, offline by default.

**Release gates:**
1. Phase 1 accuracy target met and published in the README.
2. Phase 3 false-ghost rate at or near zero.
3. No `SUPPORTED` or `REFUTED` anywhere without an attached passage.
4. Green CI on Linux, macOS and Windows.
5. README states honestly that many verdicts will be `NEI` and why.

Publish to PyPI, tag `v0.1.0`, write the CHANGELOG entry.

---

## Phase 8 — TUI and SARIF (v0.2)

| Task | How |
|---|---|
| TUI | `textual`, streaming-prompt layout per spec §13.1; paste a path, URL or raw text; **cancellable mid-run**; permission prompts inline |
| Author-year citations | `(Smith et al., 2020)`, `ibid.`, `op. cit.`, same author-year collisions → bibliography entry; own hand-built test set with a measured pairing rate |
| SARIF output | maps findings to page/line so VS Code shows them inline without an extension |
| Shared core | TUI and CLI both call one `verify()`; no logic in either front-end |

**Done when:** a SARIF file opens in VS Code with findings on the right lines,
cancelling a run leaves no partial cache entries, the coverage footer never scrolls
away, and the TUI renders correctly at 80 columns.

---

## Phase 9 — Judge layer (v0.3)

| Task | How |
|---|---|
| Adapter | Ollama default (offline), OpenRouter and Gemini as alternatives, BYOK |
| Escalation | only verdicts below the Phase 1 calibration threshold |
| Batching | 20 claims per prompt |
| Cost reporting | calls made, tokens used, printed at the end of the run |
| Report summary | `--summarize`: one final call over the finished report, cannot alter verdicts, labelled as model-written (spec §11.1) |

**Done when:** a 118-citation paper costs 2-4 calls, works under a 1 request/minute
limit, and disabling the judge changes cost to zero without changing the report
format. `--summarize` adds exactly one call and its absence changes nothing else.

---

## Phase 10 — Social provider (v0.4)

Built **Bluesky-first**, because measurement says so (spec §6.2).

| Platform | Approach |
|---|---|
| Bluesky | `public.api.bsky.app`, no auth — first-class |
| Hacker News | official Firebase API, no auth — first-class |
| Reddit | free OAuth app, user supplies their own credentials; absent credentials are reported in the coverage summary, never silently skipped |
| Mastodon | per-instance, best effort |
| X / Twitter | not readable; Community Notes bulk dumps only |

For any unreadable post, the user pastes the text and `proofpath` verifies the
**links inside it** rather than the post itself.

**Done when:** measured on AVeriTeC, and no platform integration requires a paid API.

---

## What is deliberately not planned yet

Turkish sources, `spiyweb` as an alternative retrieval backend, and the GROBID
parser. Each is a real want, none is on the path to a working v0.1.
