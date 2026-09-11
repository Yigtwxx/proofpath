# proofpath — Design

**Date:** 2026-09-10
**Status:** Draft, revised 2026-09-11 — open decisions confirmed, NLI/SciFact assumptions verified

> Don't guess. Show the evidence.

## 1. Problem

Claims travel faster than their sources. Two symptoms of the same failure:

- **Academic / professional writing.** LLM-assisted drafts cite papers that do not
  exist, cite retracted papers, or cite real papers that do not actually support the
  sentence they are attached to. Reviewers cannot check 100+ references by hand.
- **Social / news claims.** A post asserts something; the primary source is missing,
  misread, or contradicted by the very link it points to.

Existing tools check whether a DOI *resolves*. Almost nothing checks whether the
source *supports the claim*. That gap is the product.

## 2. Goals

1. For every citation or claim, decide whether the source (a) exists, (b) is still
   valid, and (c) entails the claim.
2. Always show the evidence passage behind a verdict. Never assert without a quote.
3. Run offline and free by default. No API key required for a useful result.
4. Identical behaviour on Windows, Linux and macOS from a single install command.
5. Report its own coverage honestly. An unverifiable claim is reported as
   unverifiable, never as verified.

## 3. Non-goals

- Not a truth oracle. Verdicts are *supported / contradicted / not enough
  information*, each with a passage. The tool never stamps content "fake".
- Not a plagiarism detector, bibliography manager, or reference formatter.
- v1 does not target Turkish-language sources.
- v1 does not attempt to read X/Twitter posts (see §6.2).

## 4. Users

| User | Trigger | Wants |
|---|---|---|
| Researcher / grad student | before submitting a draft | catch ghost and retracted citations |
| Peer reviewer / editor | while reviewing | per-line report to check against |
| Technical writer | before publishing | claims backed by primary sources |
| Curious reader | saw a post | what the linked source actually says |

## 5. Architecture

The core is source-agnostic. Everything platform-specific lives behind an
**evidence provider**.

```
   input                    ┌────────────────────────────────────┐
 ┌────────────┐             │             CORE                   │
 │ .pdf/.docx │             │  1. claim extraction               │
 │ .md/.txt   │────────────▶│  2. evidence retrieval (RAG)       │───▶ report
 │ URL        │             │  3. entailment (local NLI)         │     .md .json .sarif
 │ raw text   │             │  4. verdict + passage + confidence │
 └────────────┘             └──────────────┬─────────────────────┘
                                           │  EvidenceProvider
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
               academic://              web://               social://
           Crossref, OpenAlex,     search + primary      Bluesky, Reddit, HN,
           PubMed, arXiv,          source fetch          Mastodon, Community
           Unpaywall,                                    Notes dumps
           Retraction Watch
```

### 5.1 Components

| Component | Responsibility | Depends on |
|---|---|---|
| `ingest` | file/URL/text → normalized document with page+line offsets | pymupdf, python-docx |
| `claims` | document → `Claim{text, locator, cited_ref, numerics}` | ingest |
| `fetch` | URL → HTML/PDF/JSON, through the escalation ladder (§7) | httpx, scrapling, wayback |
| `resolve` | raw reference string → verified source identity or a honesty state (§8) | Crossref, OpenAlex |
| `providers` | `Claim` → `EvidenceDoc` candidates | fetch, resolve |
| `retrieval` | `EvidenceDoc` → ranked passages for a claim | ONNX embeddings, numpy cosine scan |
| `entailment` | (claim, passage) → `SUPPORTED / REFUTED / NEI` + score | local NLI model (ONNX) |
| `numerics` | numeric/unit claims, checked before NLI (§10) | — |
| `judge` | optional second opinion on low-confidence verdicts, and an optional plain-language summary of the finished report | LLM adapter (opt-in) |
| `report` | verdicts + coverage stats → markdown / json / sarif | — |
| `cache` | one **plain** SQLite file under the platform cache dir (`proofpath.sqlite3`): `sources`, `raw_text` (7-day TTL, §16), `chunks` with float32 embeddings as BLOBs, `verdicts` keyed `(claim_hash, source_id, model_id)` where `model_id` includes NLI, embedder, `k` and thresholds. A `CHECK` constraint refuses `SUPPORTED`/`REFUTED` rows without a passage. No extension, no server: DB Browser for SQLite opens it; `proofpath cache path/ls/show/clear` reads it | sqlite3 (stdlib), numpy, platformdirs |
| `cli` / `tui` | two front-ends over one `verify()` entry point | all of the above |

No logic lives in `cli` or `tui`.

### 5.2 EvidenceProvider interface

```python
class EvidenceProvider(Protocol):
    scheme: str  # "academic" | "web" | "social"

    def resolve(self, ref: Reference) -> ResolveResult:
        """Identity + validity of the cited source. See §8."""

    def fetch(self, ref: Reference) -> list[EvidenceDoc]:
        """Full text where openly available, abstract otherwise."""
```

Adding a provider must not require touching the core.

## 6. Source access reality (measured 2026-09-10)

These numbers set the ceiling on what the product can deliver. They are recorded
here so no later decision silently assumes better access than exists.

### 6.1 Academic full text

Sampled 5 well-known DOIs through OpenAlex:

| Outcome | Count |
|---|---|
| Open access **with** a direct `pdf_url` | 2 / 5 |
| Open access but **no** `pdf_url` (landing page must be scraped) | 2 / 5 |
| Fully closed | 1 / 5 |

Direct full text is available for well under half of citations. The abstract
fallback is therefore a primary path, not an edge case, and must be labelled.

**Measured again on 2026-09-11 with the real chain (Phase 4, 50 resolved DOIs from
the ghost set, browser step denied):** full text **72 %** (36/50: Semantic Scholar
`openAccessPdf` 23, arXiv 9, Crossref TDM links 3, Europe PMC 1), abstract only
18 %, nothing 10 %. Every non-full-text result carried an honesty state — 12 of them
`UNVERIFIED (blocked, browser not permitted)`, which is the ceiling the consent
prompt (§7.1) exists to lift. Step 2 (`curl_cffi`) won 7 fetches that step 1 lost,
Wayback 3. Two findings changed the code: DataCite arXiv DOIs (`10.48550/arXiv.*`)
carry their own arXiv id (9 of the 50), and a 200 page with no extractable text is
a bot wall, not content. Full table: `docs/eval/2026-09-11-coverage.md`.

Raw HTTP fetch results:

| Target | HTTP | Extracted text | Usable |
|---|---|---|---|
| Nature (OA article) | 200 | 12,019 words | yes |
| PMC full text | 200 | 12,719 words | yes |
| arXiv abstract page | 200 | 827 words | yes (abstract) |
| **Science (publisher)** | **403** | 3 words | no — bot protection |

### 6.2 Social platforms

Tested without authentication:

| Platform | Endpoint | Result | Plan |
|---|---|---|---|
| **Bluesky** | `public.api.bsky.app` (AT Protocol) | **200, clean JSON, no auth** | first-class support |
| **Hacker News** | official Firebase API | **200, no auth** | first-class support |
| **Reddit** | `.json` suffix, `www` and `old` | **403 / HTML login wall** | free OAuth app registration required; user supplies credentials |
| **Mastodon** | `/api/v1/timelines/public` | 422, auth required (instance-dependent) | per-instance, best effort |
| **X / Twitter** | status URL | **404, 92 words** | not readable; Community Notes dumps only |

Consequence: the social provider is built **Bluesky-first**, not X-first. X is the
exception, not the model. Where a post cannot be read, the user pastes its text and
`proofpath` verifies the *links inside it* rather than the post itself.

## 7. Fetch ladder

Every outbound fetch escalates through fixed steps and stops at the first success.
The step that succeeded is recorded in the report.

Measured install cost per step (2026-09-10, macOS arm64 wheels):

| Step | Method | Added install cost | Consent |
|---|---|---|---|
| 1 | `httpx` + descriptive User-Agent | — (base) | none |
| 2 | `curl_cffi` TLS impersonation (called directly; `scrapling` is the HTML parser only, its fetchers need playwright) | **2.7 MB** | none — bundled |
| 3 | `scrapling[fetchers]` browser engine (playwright/patchright) | **~81 MB wheels + ~200 MB browser download** | **explicit, required** |
| 4 | Wayback Machine API | — | none |
| 5 | give up → `UNVERIFIED (blocked)` / `(unreachable)` | — | — |

`scrapling` core is 0.2 MB with six pure-Python dependencies, so it ships as a normal
dependency. `curl_cffi` at 2.7 MB defeats a large share of TLS-fingerprint blocks
without any browser at all, so it also ships by default.

Only step 3 is expensive, and only step 3 asks.

The same principle governs the inference runtime: the default install uses ONNX
runtime, not `torch`. Asking consent for a 281 MB browser while silently installing
800 MB of `torch` would be incoherent. GPU acceleration is the opt-in `[gpu]` extra.

Content type is detected from response headers, not the URL suffix: a link may serve
a PDF.

### 7.1 Permissions

Downloading ~280 MB of browser engine is a real action taken on the user's machine.
It is never silent, never implicit, and always recorded.

Permissions live in a config file under the platform config dir
(`platformdirs.user_config_dir("proofpath")`), in the spirit of Claude Code's
settings:

```toml
# ~/.config/proofpath/config.toml   (%APPDATA%\proofpath\config.toml on Windows)

[permissions]
# "ask" | "allow" | "deny"
install_browser = "ask"      # step 3: playwright/patchright + chromium, ~280 MB
                             # default confirmed 2026-09-11: "deny" would hide that
                             # a blocked source was recoverable
network         = "allow"    # outbound HTTP at all

[fetch]
respect_robots  = true

[contact]
email = ""                   # optional, for Crossref/OpenAlex polite pool
```

When step 3 is first needed and the setting is `ask`, the run pauses and prompts:

```
  ⚠ nature.com blocked this request (HTTP 403).

    proofpath can retry with a real browser engine, but that needs a
    one-time download:

      playwright + patchright wheels    ~81 MB
      chromium browser                 ~200 MB
      installed into this tool's own environment only

    Allow?  [y] yes, once   [a] always (save to config)   [n] no   [never] never ask again
```

Rules:

| Situation | Behaviour |
|---|---|
| `allow` | proceed, log the install in the report |
| `deny` / `never` | skip step 3, emit `UNVERIFIED (blocked, browser not permitted)` |
| `ask` + interactive TTY | prompt as above |
| `ask` + **no TTY** (CI, piped, `--format sarif > file`) | **never prompt** — treat as `deny` and report it |
| `--allow-browser` / `--no-browser` flag | overrides config for that run |

`proofpath permissions` prints the current state and where it is stored.
`proofpath permissions set install_browser allow` edits it without opening a file.

The prompt appears at most once per run. A denied answer is not re-asked, and the
report states plainly how many sources were skipped because of it, so a low-coverage
run is never mistaken for a clean one.

## 8. Reference resolution — corrected

**The naive approach does not work.** Crossref's `query.bibliographic` matcher always
returns a result and its score does not separate real from fabricated:

| Input | Score | Returned | Correct |
|---|---|---|---|
| AlphaFold (real) | 51.0 | AlphaFold | yes |
| Stochastic Parrots (real) | 77.0 | Stochastic Parrots | yes |
| Lewis et al. RAG 2020 (real) | 38.9 | unrelated 2025 NLP paper | **no** |
| **Fabricated reference** | **38.5** | unrelated real paper | **no — ghost missed** |

A fabricated reference scored *inside the same band* as a real reference that failed
to match. Thresholding on score would both flag real papers as ghosts and let
fabricated ones through.

**Decision: the matcher score is never used as evidence.** It only produces
candidates. Identity is decided by field-level agreement against the input string:

| Field | Check |
|---|---|
| Author surnames | at least the first surname matches, normalized |
| Year | exact, or ±1 (online-first vs issue date) |
| Title | token-set Jaccard above threshold, stopwords removed |
| Venue | fuzzy match when present in the input |

| Fields agreeing | Verdict |
|---|---|
| Title **and** first author **and** year | `RESOLVED` |
| Title strong but one other field off | `RESOLVED (low confidence)` |
| Title weak, others agree | `AMBIGUOUS` — all candidates reported, no verdict |
| Nothing agrees | `GHOST REFERENCE` |

The bias is deliberately **conservative**: telling a researcher their real citation
is fabricated destroys trust in one shot, so uncertainty resolves to `AMBIGUOUS`,
never to `GHOST`.

Retraction Watch is queried independently of this, keyed on the resolved DOI.

### 8.1 Implementation notes (2026-09-11)

- **Providers, in two rounds.** Round one asks Crossref (`query.bibliographic`,
  the only true bibliographic matcher) and Semantic Scholar (`paper/search/match`
  on a title-like segment). Both are free without a daily budget. If that round
  ends in `GHOST` or `AMBIGUOUS`, round two asks arXiv (title search) and Open
  Library (title + first-author hint, for books), and OpenAlex as best effort.
  A ghost call needs Crossref, Semantic Scholar, arXiv and Open Library to have
  all answered; if any of them was unavailable the state is
  `UNVERIFIED (provider unavailable)` instead.
- **Why not OpenAlex first.** Its free tier became a daily budget (measured
  2026-09-11: 1000 credits, a search costs 10 → ~100 searches per day), and it
  answers `429` with a `Retry-After` of hours once spent. Such responses are
  treated as "unavailable now", never slept on. RoBERTa (arXiv 1907.11692) was
  also missing from it that day.
- **Identifiers first.** A DOI or arXiv id in the string (including the dotless
  `arXiv:160706450` form) is resolved directly; agreement on the fields means
  `RESOLVED` at once, a different work is noted and the search continues.
- **Candidate generation only.** Title-like segments come from the raw string
  after the leading author list is stripped (ACM `Authors. 2010. Title`, APA
  `(2019).`, or initials-based lists), quoted spans first, trailing
  `(Publisher, Year)` removed, two words minimum. Segments and the first-author
  hint are used to *find* candidates; identity is still decided by the field
  checks against the raw string.
- **Title agreement** is token coverage (share of the candidate title's tokens
  present in the raw string) times a length ratio against the segment it
  overlaps with, so a short candidate title inside a longer cited one cannot
  score as strong. Cut-points: strong ≥ 0.8, weak ≥ 0.5 (OPEN-ITEMS 4.1); year
  ±1 (4.2). When two providers return the same identifier the record that agrees
  best is kept — Crossref titles the Numba paper "Numba", Semantic Scholar has
  the full title.
- **Not everything is a paper.** Web pages, blogs, dashboards, manuals, reports,
  organisation-authored documents and preprint servers no provider searches
  (PsyArXiv, OSF, Zenodo) get `UNVERIFIED (not in bibliographic indexes)` rather
  than `GHOST`: their absence from indexes says nothing. Strings with fewer than
  six content tokens are `AMBIGUOUS`.
- **Politeness.** One client, descriptive User-Agent, optional `mailto` from config
  or `PROOFPATH_CONTACT_EMAIL`, per-host minimum intervals (0.25 s Crossref and
  OpenAlex, 1.1 s Semantic Scholar, 1 s Open Library, 3 s arXiv), `Retry-After`
  honoured up to 60 s, exponential backoff on 5xx.
- **Measured** on the hand-built ghost set (`tests/data/ghost_set.jsonl`: 106 real
  reference strings taken verbatim from five real papers' reference lists, 100
  fabricated ones in five citation styles, 20 real ones with author and year
  mutated). Results and the release gate are in `docs/eval/2026-09-11-ghosts.md`.

## 9. Data flow (academic path)

1. `ingest` parses the document, keeping page and line numbers per sentence.
2. `claims` pairs each in-text citation marker with the sentence carrying it, and
   extracts numeric spans (§10). v0.1 handles **numeric markers only** (`[12]`,
   `[12,15]`, `[12-15]`); an author-year marker such as `(Smith et al., 2020)` is
   reported as `UNSUPPORTED CITATION STYLE` and skipped, never guessed. Author-year
   pairing is a v0.2 task with its own test set (§17).

   **Paragraph-scoped citations.** When the marker sits at the end of a paragraph
   and its carrying sentence holds no other marker, the citation is treated as
   supporting the whole paragraph: every sentence of that paragraph is verified
   separately against the source, and the report groups them under one
   `PARAGRAPH-SCOPED` finding. A single sentence is never given a confident verdict
   on behalf of the paragraph it sits in.
3. `resolve` sends the **raw reference string** to Crossref — no local bibliography
   parsing — then applies field-level verification (§8).
4. Retraction Watch check on the resolved DOI → `RETRACTED` if hit.
5. `fetch` walks the OA chain: `pdf_url` → PMC → arXiv → landing page scrape →
   abstract, through the ladder in §7.
6. `retrieval` chunks the source and ranks passages against the claim.
7. `numerics` runs first on numeric claims; `entailment` classifies the top passages.
8. `judge` re-checks only low-confidence verdicts, in batches, if enabled.
9. `report` emits per-line findings with quotes, plus the coverage summary (§15).
10. `judge` optionally writes a plain-language summary of the finished report
    (§11.1). This runs last, reads only what the report already contains, and is
    off by default.

## 10. Numeric claims

NLI models are unreliable on numbers and units. A claim of "40% speedup" against a
source saying "4-8% improvement in throughput" is exactly the failure case the tool
exists to catch, and a generic entailment model will often mark it supported.

Numeric claims are therefore handled before NLI:

1. Extract quantities from claim and passage: percentages (`40%`, `4-8%`, `-31%`),
   factors (`2x`, `10-fold`, `three times`) and counts with a unit (`50 mg`,
   `1,000 genomes`, `2.3 million people`), each with an optional direction read
   from nearby words (`increased`, `reduction`, `faster`…).
2. Compare only like with like: same kind, same normalised unit.
3. Mismatch → `REFUTED (numeric mismatch)` with both figures shown, decided by rule
   (`score 1.0`, tier `high`, reason string on the verdict).
4. Otherwise the claim falls through to NLI. A matching number never asserts
   support on its own.

Rules fixed 2026-09-11 (OPEN-ITEMS 4.3), each one traced to a wrong firing on
SciFact dev:

| Rule | Why |
|---|---|
| Tolerance is relative **10 %**, applied to the source figure; a point agrees with a range if it lies in the stretched range, two ranges agree if they overlap | `40%` vs `38%` agrees, `40%` vs `35%` does not, `8.5%` vs `4-8%` agrees |
| Refute only when the claim holds **one** quantity of that kind and the passage holds **exactly one** comparable figure | with two candidate figures the layer cannot tell which the claim refers to; every ambiguous firing on dev was wrong |
| A change (explicit direction) is never refuted by a level (no direction), or vice versa | `decreased by 10%` vs `57% women` |
| Opposite explicit directions disagree even with equal values | `12% increase` vs `12% decrease` |
| `NN% CI` is a confidence level, not a quantity; numbers glued to letters or signs (`H3.3`, `+1`, `CD4`) are labels; `less/more than` are bounds, not directions | all three produced false refutations |
| No subject matching in v0.1 | too unreliable without parsing; the single-figure rule stands in for it |

Measured effect on SciFact dev: never lower than without the layer (+0.003
accuracy), one firing on the top-k passages, correct. SciFact holds almost no
numeric contradictions, so the hand-built set in `tests/test_numerics.py` is the
acceptance test for this stage.

## 11. LLM budget

The default configuration makes **zero** LLM API calls.

| Stage | Method | LLM calls |
|---|---|---|
| parse | pymupdf, rule-based | 0 |
| citation ↔ sentence pairing | regex + rules | 0 |
| reference resolution | Crossref / OpenAlex + field checks | 0 |
| retraction check | Retraction Watch | 0 |
| fetch | httpx / scrapling / wayback | 0 |
| evidence retrieval | local embeddings + sqlite-vec | 0 |
| numeric check | rule-based | 0 |
| entailment | local NLI (DeBERTa-MNLI class) | 0 |
| **judge (opt-in)** | 20 claims per prompt, batched | **2-4** per paper |
| **report summary (opt-in)** | one prompt over the finished report | **1** |

Verdicts are cached in sqlite keyed by `(claim_hash, source_id, model_id)`, so a
re-run of the same document costs 0 calls.

### 11.1 Report summary

`--summarize` adds one final call that turns the finished report into a few
sentences a human can act on:

> 42 references, 3 of them fabricated and 1 retracted. Six claims are not supported
> by the source they cite; the most serious is on page 9, where a 40% speedup is
> attributed to a source reporting 4-8%. Coverage is weak — 17% of sources could not
> be reached, so the real figure may be higher.

Constraints that make this safe rather than a second guessing layer:

- It runs **after** the report is final and **cannot change a single verdict**.
- Its only input is the report the deterministic pipeline produced. It never sees a
  source document, so it cannot introduce a claim of its own.
- It is off by default in **both** the CLI and the TUI. `proofpath` stays fully
  offline unless asked otherwise; the TUI enables it with `/summarize`.
- The summary is labelled as model-written in the output, so it is never mistaken
  for a computed result.

Even a 1 request/minute free tier finishes a 118-citation paper in ~4 minutes with
the judge enabled. With Ollama there is no wait at all.

**Provider (decided 2026-09-11).** Default is Groq `openai/gpt-oss-120b`: free
without a card, no training on submitted data, strict JSON schema output, but an
8K tokens-per-minute cap that forces batches under ~7k tokens and roughly one call
per minute. Gemini 3.8 Flash is selectable but Google trains on free-tier prompts
outside the EEA/UK/CH, so choosing it prints a data-use warning. Ollama is the
offline option. All three speak the OpenAI ``chat/completions`` shape, so one
adapter covers them. The API key is read from an environment variable or a
``.env`` file, never written to config and never printed. Survey:
``docs/research/2026-09-11-free-llm-api-tiers.md``.

## 12. Cross-platform constraints

| Component | Risk | Decision |
|---|---|---|
| PDF parsing | GROBID needs Java + Docker | pure-Python `pymupdf` default; GROBID opt-in via `--parser grobid` |
| Reference parsing | structured extraction is hard | avoided entirely — raw string to Crossref (§8) |
| Vector store | Qdrant server needs Docker; Chroma embedded adds 47 packages / 161 MB (kubernetes, grpc, otel) and a directory; LanceDB +280 MB; `sqlite-vec` needs extension loading that python.org macOS builds lack | **numpy brute-force cosine scan** over a per-source matrix: measured 2026-09-11 at 1.2 ms for 10⁵ vectors vs 7.4 ms for sqlite-vec. Queries are always scoped to one source (≤500 chunks), so no ANN index earns its weight. Vectors persist as float32 BLOBs in the plain SQLite cache. Revisit only for cross-source search above ~2×10⁵ vectors — then `usearch` (+4 MB), not a server |
| Scrapling | browser engines are heavy | core + `curl_cffi` bundled (2.7 MB); browser engine installed on demand with explicit consent (§7.1) |
| **Inference runtime** | **`torch` + `sentence-transformers` is ~800 MB — larger than the browser engine we ask consent for** | **ONNX runtime by default (base install stays small). `torch` moves to an opt-in `[gpu]` extra.** |
| Compute device | CUDA on Windows, MPS on Mac | preference order is always CUDA → MPS → CPU; under ONNX the equivalent is CUDA → CoreML → CPU |
| **NLI model** | a ready-made ONNX cross-encoder had to exist for the "no torch" decision to hold | **verified 2026-09-11:** `cross-encoder/nli-deberta-v3-base` @ `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` ships ONNX (`onnx/model.onnx` 739 MB, `model_O4` 388 MB, int8 variants 244 MB) and `tokenizer.json`. The int8 file is chosen by `platform.machine()` (`qint8_arm64` / `quint8_avx2`). `-small` / `-xsmall` ship the same layout as lighter fallbacks. No `optimum` export, no `transformers` |
| Embedding model | must run under ONNX too | `fastembed` with a pinned model; first candidate `BAAI/bge-small-en-v1.5`, compared against one alternative in the Phase 1 results table |
| Evaluation data | `allenai/scifact` on Hugging Face is a script-based loader that `datasets ≥ 4` refuses | the original AI2 tarball (`scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`, verified live 2026-09-11) is downloaded with `httpx` and its sha256 pinned in the loader. No `datasets` dependency |
| Model cache | different cache roots per OS | `platformdirs` |
| Install | global pip pollution | `uv tool install proofpath` / `pipx install proofpath` |
| Paths & encoding | `\` separators, cp1252 | `pathlib` everywhere, explicit `encoding="utf-8"` |

## 13. Interfaces

```bash
uv tool install proofpath             # everything except the browser engine

proofpath                                  # bare command → interactive TUI
proofpath check paper.pdf                  # academic provider (default)
proofpath check --url https://bsky.app/... # social provider
proofpath check draft.md --format sarif    # inline problems in VS Code
proofpath check paper.pdf --judge llm      # opt-in second opinion

proofpath resolve "Jumper J, ... Nature 2021"   # one reference, one stage
proofpath fetch 10.1038/s41586-021-03819-2 # one source through the OA chain and ladder

proofpath config                           # show every setting + config path
proofpath config set permissions.install_browser allow
proofpath config check                     # judge provider, key, reachability
proofpath cache clear
```

The full surface, output language, exit codes and global flags are fixed in §13.3.

`proofpath` with no arguments launches the TUI; `proofpath <subcommand>` runs
one-shot. Implemented as a Typer callback with `invoke_without_command=True`, so a
bare invocation is a first-class entry point rather than a help screen.

### 13.1 TUI — streaming prompt

The TUI is a streaming log with a prompt at the bottom, in the manner of Claude Code
and OpenClaw. Chosen over a split-pane browser because it handles several documents
in one session naturally, surfaces the permission prompt (§7.1) inline in the flow
where it happened, and keeps one code path with the one-shot output.

Built with `textual`. Accepts a file path, a URL, or raw pasted text. Streams
progress, is cancellable mid-run, and writes a report on completion.

```
   ,_,
  (o.o)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[PROOF]
   " "    proofpath v0.1.0                            academic . offline . mps
          paste a file path, a URL, or a claim.            /help  /config  /quit

› ~/Desktop/paper.pdf

  ⏺ Parse                                                                 1.2s
    24 pages · 42 references · 118 citations

  ⏺ Resolve references                                       Crossref·OpenAlex
    ✓ 38 resolved    ⚠ 3 ambiguous    ✗ 1 ghost

  ⏺ Retractions                                              Retraction Watch
    ⚠ 1 retracted

  ⏺ Fetch sources                              ███████████████░░░░░     34/42
    22 full text · 11 abstract · 9 blocked
    ⚠ sciencedirect.com blocked — allow browser engine?  /allow

  ⏺ Verify claims                              ████████░░░░░░░░░░░░    51/118
    running locally on mps

 ─────────────────────────────────────────────────────────────────────────────
  ✗  p.4  L112   [12] Zhang 2021                              GHOST REFERENCE
         DOI 10.1016/j.xxxx.2021.99999 resolves to nothing
         no author, title or year agreement with any candidate

  ⚠  p.7  L203   [28] Lee 2019                             RETRACTED  2023-06
         "Concerns about data integrity" — Retraction Watch

  ✗  p.9  L260   [31] Kumar 2022                        NOT SUPPORTED    high
         you     "the method yields a 40% speedup"
         source  "we observed a 4-8% improvement in throughput"
         → numeric mismatch, not an entailment call
 ─────────────────────────────────────────────────────────────────────────────
  42 refs · 3 ghost · 1 retracted · 6 unsupported      coverage 62/21/17%
  report.md written · 0 API calls · 38s

› _
```

**The ferret (decided 2026-09-11).** The banner at the top is the tool's pet, in the
manner of Claude Code's welcome header: shown once at launch, pinned above the log,
never repeated. A ferret because English *ferrets out* the facts, and because its
long body becomes the path the name promises — the `~` line runs from the head to a
red `[PROOF]` stamp at the far right. The stamp is the banner's **only** coloured
element (the single brand use of red allowed by §13.3); the animal itself is drawn
in the default foreground, bold.

```
   ,_,
  (o.o)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[PROOF]
   " "    proofpath v0.1.0                            academic . offline . mps
          paste a file path, a URL, or a claim.            /help  /config  /quit
```

Rules for drawing it:

- Pure ASCII — no box drawing, no emoji — so Windows Terminal at 80 columns
  (assumption 3.3) renders it identically. The rest of the TUI may use box drawing;
  the pet may not.
- The body stretches: `~` count = terminal width − head − stamp − margins, minimum 12.
  Below 60 columns the stamp is dropped; below 40 the body is dropped and only the
  head and the two text lines remain. It re-flows on resize.
- Line 2 carries the version and the run context (provider · online/offline ·
  device); line 3 the prompt hint and the slash commands. Both are plain text; the
  version is never coloured.
- The eyes are the only animated part, and only in the TUI: `(o.o)` idle, with one
  blink `(-.-)` of ~150 ms every 6–10 s at random; `(>.>)` while a run is fetching or
  verifying (looking down the path); `(O.O)` for two seconds when a run ends with
  findings; `(^.^)` for two seconds when a run ends clean. Nothing else moves, and
  the animation is disabled when the terminal reports no colour support, on
  `--no-color`, and in `-q`.
- The one-shot CLI never prints the pet; `proofpath --version` prints one line.
- Ownership: the banner lives in `tui/banner.py`; `ui.py` supplies the red for the
  stamp; no other module draws any part of it.

Design rules:

- Each pipeline stage is one collapsible block. The provider or model that produced
  a result is named on its right, so no number is unattributable.
- Findings appear below a rule, after the stages, newest run last.
- Every finding carries page and line, the verdict, the confidence tier, and the
  quoted passage. A finding without a passage is a bug, not a display choice.
- Confidence is shown as one of three calibrated tiers — `high`, `medium`, `low` —
  never as a raw model score. A `0.91` is a softmax output, not a probability of
  being right, and showing it invites exactly that misreading. The cut-points come
  from the Phase 1 calibration on SciFact dev (§14). The raw score is kept in
  `--format json` for anyone who wants it.
- The footer always shows coverage. It is not optional and does not scroll away.
- Permission prompts appear inline at the point of failure with a slash command to
  answer, never as a modal that blocks the log.
- **Slash commands have an awaiting mode (decided 2026-09-11).** A verb typed without
  its argument — `/check`, `/resolve`, `/fetch` — does not error: the input bar takes
  the verb's accent tint and a placeholder (`paste a file path or URL`), and waits.
  `Esc` cancels and restores the bar. Once the argument is submitted the bar returns
  to its normal colour and the run starts; typed with the argument on one line
  (`/check ~/paper.pdf`) the mode is skipped and the run starts at once. In both
  cases the echoed command line stays in the verb's accent colour in the log, so a
  session with several runs reads as a sequence of coloured headings — nothing else
  in the log keeps that colour.

  ```
  › /check                                   ← bar tinted, waiting
  ›  paste a file path or URL

  › /check ~/Desktop/paper.pdf               ← stays tinted in the log
    ⏺ Parse …
  ```
- **Several runs per session (decided 2026-09-11).** Every submitted `/check` becomes
  its own block in the log at once, numbered and coloured from a rotating palette
  (`#1`, `#2`, `#3` … each with a distinct accent; the block header, its command line
  and its progress bars carry that colour). Runs are scheduled **by stage, not by
  kind**: the I/O-bound stages — parse, resolve, fetch — run concurrently for up to
  three runs at a time (per-host politeness limits are shared across runs), while
  **verify** (the NLI model) is a single FIFO slot, because one model on one device
  is the bottleneck. A block therefore shows one of `queued`, `running`, `waiting for
  verify`, `verifying`, `done`, `cancelled`. `/cancel #n` stops one run; finished
  blocks keep their findings and coverage line in place, newest run last.

  ```
  #1  /check https://…/s41586-021-03819-2        done          coverage 62/21/17%
  #2  /check ~/Desktop/draft.md                  verifying     ████████░░░░  51/118
  #3  /check ~/Papers/review.pdf                 waiting for verify  ·  fetched 30/34
  ```
- **Mouse works everywhere the keyboard does (decided 2026-09-11).** Textual gives
  clicks, wheel scrolling and hover natively; the rule is that no action is
  mouse-only and none is keyboard-only. Clickable targets: a run header toggles the
  block open/closed; a stage line toggles its detail; a finding toggles its full
  quoted passage; the inline permission prompt renders `[allow once] [always] [no]
  [never]` as buttons, equal to typing `/allow …`; a `#n` reference or a source URL
  is a terminal hyperlink (OSC 8) that opens in the browser, and `⧉` next to a
  passage copies it. Right-click and drag do nothing special. In a terminal without
  mouse reporting every target keeps its keyboard path, so nothing is lost.

### 13.2 One-shot output

`proofpath check` prints compiler-style diagnostics: pipeable, greppable, and
readable in CI logs. It shares the verdict data with the TUI and adds no logic.

```
$ proofpath check paper.pdf

  Parsing      paper.pdf                         24 pages, 42 refs      1.2s
  Resolving    Crossref, OpenAlex                38 ok, 3 amb, 1 ghost  3.4s
  Retractions  Retraction Watch                  1 retracted            0.8s
  Fetching     22 full text, 11 abstract, 9 blocked                    14.7s
  Verifying    118 claims on mps                                       21.4s

error[ghost-reference]: cited source does not exist
  --> paper.pdf:4:112
   |
   | [12] Zhang, K. et al. (2021). Neural cascade alignment for zero-shot...
   |      ^^^^^^^^^^^^^^^^^^^^^^^ no record in Crossref or OpenAlex
   |
   = note: no author, title or year agreement with any candidate

error[numeric-mismatch]: claim contradicts the cited source
  --> paper.pdf:9:260
   |
   | The method yields a 40% speedup on long-context workloads [31]
   |                     ^^^^^^^^^^^^ source reports 4-8%
   |
   = source: "we observed a 4-8% improvement in throughput"  ([31] p.6 §4.2)

warning[retracted]: cited source was retracted 2023-06
  --> paper.pdf:7:203
   |
   | [28] Lee, S. (2019). Adaptive gating for efficient inference
   |
   = note: "Concerns about data integrity" — Retraction Watch

  42 refs: 3 ghost, 1 retracted, 6 unsupported, 32 ok
  coverage: 62% full text, 21% abstract, 17% unverified
  report.md written  ·  0 API calls  ·  38.4s
```

Exit codes: `0` clean, `1` findings present, `2` the run itself failed. CI can gate
on this without parsing the text.

### 13.3 CLI surface and output language (decided 2026-09-11)

**The TUI is the primary surface.** A user types `proofpath` and does everything inside
it; the README, `--help` and every example lead with that. The one-shot subcommands
below exist for the places a TUI cannot run — CI, pipes, scripts, `--format sarif >
file` — and are documented as that path, not as the main one. They share one
vocabulary. Verbs are flat; settings live under `config`; the cache is data, not a
setting, and stays its own group.

```
proofpath                                   bare → TUI
proofpath [--no-color] [-q|--quiet] <verb|group> ...

proofpath check TARGET      [--format text|json|sarif] [--judge] [--allow-browser|--no-browser] [--no-cache]
proofpath resolve REF       [--format text|json]
proofpath fetch TARGET      [--format text|json] [--allow-browser|--no-browser] [--no-cache] [--show N]

proofpath config            = config show: config path, then every section as TOML
proofpath config show | path
proofpath config set SECTION.KEY VALUE      permissions.install_browser allow · contact.email … · judge.provider …
proofpath config check                      judge provider, key presence, reachability (was `judge check`)

proofpath cache path | ls | show ID | clear [--expired]
```

`permissions` and `judge` as top-level commands are removed before any release
carries them; `config set permissions.<key>` and `config set judge.<key>` replace them.

**Output language.** All human output goes through one thin module, `ui.py`, built on
`rich.Console`; commands never format on their own.

| Rule | Decision |
|---|---|
| Layout | key/value lines as a borderless grid, key column 10 characters (`state      RESOLVED`); no panels, no boxed tables — output must stay greppable when piped |
| Colour | only on a TTY and only on state words: `ok`/`RESOLVED`/`SUPPORTED` green · `UNVERIFIED …`/`LOW CONFIDENCE …`/`AMBIGUOUS` yellow · `GHOST REFERENCE`/`REFUTED`/`RETRACTED` red · `NEI` dim. `NO_COLOR`, `--no-color` or a non-TTY stdout → plain text |
| Progress | bars only on a TTY, one per stage (§13.2); never in piped output |
| Vocabulary | every verb opens with one state line: `state` (resolve) · `outcome` (fetch URL) · `evidence` (fetch DOI) · `verdict` (check). Then `note` lines (facts), `hint` lines (what the user can do), and the permission lines `browser` / `skipped` (§7.1) |
| Streams | data on stdout; errors and the §7.1 prompt on stderr |
| `--format json` | the result dataclass dumped with `orjson` (enums by value, bytes omitted); stdout carries **only** the JSON, human messages go to stderr |
| `--format sarif` | `check` only |
| `-q` / `--quiet` | stage lines and `note` lines suppressed; findings, the coverage summary and the exit code remain — a quiet run is never a silent one (rule 6) |
| Exit codes | `0` clean · `1` findings, including every `UNVERIFIED` and `LOW CONFIDENCE` state · `2` only when the tool itself failed (bad arguments, config error, I/O, uncaught exception). "Provider unavailable" is a reported state, so it is `1`, not `2` |

**Colour system (decided 2026-09-11).** Two layers, both drawn from the terminal's own
16-colour ANSI palette — no hex values, so light and dark themes both work and
`NO_COLOR` removes everything cleanly.

| Layer | Colour | Used for |
|---|---|---|
| meaning | green | `ok` `RESOLVED` `SUPPORTED` `fulltext` `not retracted` |
| meaning | yellow | `UNVERIFIED …` `LOW CONFIDENCE …` `AMBIGUOUS` `RESOLVED (low confidence)` `abstract`; the §7.1 permission prompt |
| meaning | red | `GHOST REFERENCE` `REFUTED` `RETRACTED` `FAILED`; `error:` lines on stderr |
| meaning | dim | `NEI` `none` `—` |
| meaning | bold, no colour | the key column (`state`, `outcome`, `evidence`, `verdict`) |
| accent | cyan · magenta · blue · bright cyan · bright magenta, rotating | one per run in the TUI (§13.1): its header, command line, progress bars, and the input bar's background tint (~15–20 %) while awaiting an argument |
| — | underline, no colour | hyperlinks |

Red, yellow and green are reserved for meaning and never used as accents; accents
never use red, yellow or green. The coverage footer colours its three numbers the
same way (full text green, abstract yellow, unverified red). `ui.py` owns these
tables; nothing else in the package names a colour.

**TUI mirror rule.** Every CLI verb and group exists in the TUI as the same-named slash
command: `/check`, `/resolve`, `/fetch`, `/config`, `/cache`. Only `/allow`,
`/summarize`, `/help` and `/quit` are TUI-specific. Both surfaces call the same
library functions; neither holds logic.

## 14. Evaluation

The tool is not shippable without a frozen evaluation set. Metrics are reported per
stage so a regression can be located.

| Dataset | Stage measured | Metric |
|---|---|---|
| SciFact (AI2 tarball, sha256 pinned; not the HF loader) | retrieval + entailment on scientific claims | label accuracy, macro-F1, rationale F1 |
| FEVER | retrieval + entailment baseline | label accuracy |
| AVeriTeC | real-world web claim verification | AVeriTeC score |
| PubHealth | high-harm domain behaviour | label accuracy |
| X Community Notes | social provider sanity check | agreement with human notes |
| Hand-built ghost set | reference resolution (§8) | precision/recall on ghosts, **false-ghost rate** |

**Expectation setting.** Published SciFact results sit around 70-75 F1, not 95. A
large share of verdicts will legitimately be `NEI`. The README must state this
plainly; a tool that appears certain about everything is the failure mode being
fought here.

Primary release gate: beat a "reference exists → assume supported" baseline on
SciFact, and never emit `SUPPORTED` without an attached passage.

Secondary gate: **false-ghost rate near zero** on the hand-built set.

**Confidence tiers.** The `high` / `medium` / `low` cut-points shown in every report
(§13.1) are chosen on the SciFact dev split, not by hand: the threshold sweep in
Phase 1 records precision per score band, and the tiers are the bands. They are
written into the repo with the results table and re-derived whenever the NLI
model or its revision changes.

## 15. Error handling and honesty states

Every non-verdict is an explicit, reported state. Absence of evidence is never
presented as evidence of absence.

| State | Cause |
|---|---|
| `LOW CONFIDENCE (abstract only)` | full text unavailable, abstract used |
| `UNVERIFIED (blocked)` | 403/bot protection, Scrapling absent or defeated |
| `UNVERIFIED (unreachable)` | dead link, Wayback miss |
| `UNVERIFIED (blocked, browser not permitted)` | steps 1–2 blocked and the §7.1 consent was denied, absent, or impossible without a TTY |
| `UNVERIFIED (blocked, robots.txt)` | the site's `robots.txt` disallows the fetch; steps 3–4 are not attempted |
| `UNVERIFIED (network not permitted)` | `permissions.network = deny`; nothing was fetched |
| `UNVERIFIED (provider unavailable)` | API down or rate limited after backoff |
| `AMBIGUOUS` | multiple plausible reference candidates — all listed |
| `NEI` | source read, but it neither supports nor contradicts |
| `UNVERIFIED (not in bibliographic indexes)` | web page, blog, report, manual or organisation-authored document; indexes do not cover it, so absence proves nothing (§8.1) |
| `PARAGRAPH-SCOPED` | the citation supports a paragraph, not one sentence (§9); every sentence is verified separately and grouped |
| `UNSUPPORTED CITATION STYLE` | author-year marker found; v0.1 pairs numeric markers only, so the claim is listed but not judged |

Unparseable pages fail loudly with the page number and processing continues.

Every report ends with a coverage summary:

```
verified against full text   62%
abstract only                21%
unverified                   17%
```

A run where coverage is low is a run whose conclusions are weak, and the user is
told so directly.

## 16. Politeness and legal

- Crossref and OpenAlex polite pools require a contact address. It is read from
  config or `PROOFPATH_CONTACT_EMAIL`, **never hardcoded**, and is optional.
- Descriptive User-Agent identifying the tool and its repository.
- `robots.txt` respected on step 1 and step 2 of the fetch ladder.
- Exponential backoff on 429/5xx, then an explicit `UNVERIFIED` state.
- Fetched full text is cached locally and not redistributed. Raw publisher text
  expires after **7 days**, and with it the chunk texts derived from it are set to
  NULL (a full text split into sentences is still the full text); embeddings stay,
  and the short passage quoted on each verdict stays with the verdict. Everything
  lives in one SQLite file under the user cache dir; `proofpath cache clear`
  removes all of it, `--expired` only what has aged out.

## 17. Milestones

**v0.1 — academic provider.** Ingest PDF/MD, **numeric citation markers only**,
reference resolution with field-level verification (§8), retraction check, fetch
ladder steps 1 and 3, local retrieval and entailment, numeric checker, markdown
report with coverage summary, CLI only. Measured on SciFact plus the hand-built
ghost set.

**v0.2 — TUI, SARIF, fetch ladder, author-year.** `textual` REPL with cancellation,
SARIF output, verdict cache, fetch ladder steps 2-4 including the permission prompt
and config file (§7.1), and author-year citation pairing (`(Smith et al., 2020)`,
`ibid.`, same author-year collisions) with its own test set.

**v0.3 — judge layer.** Opt-in LLM second opinion, Ollama default, OpenRouter and
Gemini adapters, batching and cost reporting.

**v0.4 — social provider.** Bluesky and Hacker News first, Reddit via user-supplied
OAuth app (missing credentials are reported, never silently skipped), Mastodon
best-effort, Community Notes dumps for X. Measured on AVeriTeC.

**Later.** Turkish sources as a separate provider (TR Dizin / DergiPark class),
`spiyweb` graph retrieval as an alternative backend, GROBID parser.

## 18. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Full text unavailable for most sources (§6.1) | weakens core value | abstract fallback, explicit labelling, coverage % in every report |
| False ghost calls (§8) | destroys user trust instantly | conservative thresholds, `AMBIGUOUS` state, false-ghost rate as a release gate |
| Local NLI accuracy on domain text | wrong verdicts | measure on SciFact first, numeric checker in front, judge as escape hatch, never assert without passage |
| Publisher bot protection | coverage gaps | Scrapling, Wayback, explicit `UNVERIFIED (blocked)` |
| Scrapling breakage as sites change | steps 2-3 silently degrade | each step's success is reported per fetch; failures surface as explicit `UNVERIFIED` states |
| Browser download refused or unavailable | coverage drops | reported in the coverage summary with a count of skipped sources |
| Scope creep into social too early | nothing finished | v0.1-v0.3 are academic only |

## 19. Ecosystem

`proofpath` is the third tool in the same idea: *don't guess, show the evidence*.

- `reasonhound` — proves vulnerabilities instead of listing suspicions
- `spiyweb` — graph retrieval that reports what it does not know
- `proofpath` — ties a claim back to its source

`spiyweb` is a candidate alternative retrieval backend, kept optional so `proofpath`
installs standalone.
