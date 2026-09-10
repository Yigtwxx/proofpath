# proofpath — Design

**Date:** 2026-09-10
**Status:** Draft, revised 2026-09-10 after empirical source-access testing

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
| `retrieval` | `EvidenceDoc` → ranked passages for a claim | ONNX embeddings, sqlite-vec |
| `entailment` | (claim, passage) → `SUPPORTED / REFUTED / NEI` + score | local NLI model (ONNX) |
| `numerics` | numeric/unit claims, checked before NLI (§10) | — |
| `judge` | optional second opinion on low-confidence verdicts | LLM adapter (opt-in) |
| `report` | verdicts + coverage stats → markdown / json / sarif | — |
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
| 2 | `scrapling` parsing + `curl_cffi` TLS impersonation | **2.7 MB** | none — bundled |
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

## 9. Data flow (academic path)

1. `ingest` parses the document, keeping page and line numbers per sentence.
2. `claims` pairs each in-text citation marker with the sentence carrying it, and
   extracts numeric spans (§10).
3. `resolve` sends the **raw reference string** to Crossref — no local bibliography
   parsing — then applies field-level verification (§8).
4. Retraction Watch check on the resolved DOI → `RETRACTED` if hit.
5. `fetch` walks the OA chain: `pdf_url` → PMC → arXiv → landing page scrape →
   abstract, through the ladder in §7.
6. `retrieval` chunks the source and ranks passages against the claim.
7. `numerics` runs first on numeric claims; `entailment` classifies the top passages.
8. `judge` re-checks only low-confidence verdicts, in batches, if enabled.
9. `report` emits per-line findings with quotes, plus the coverage summary (§15).

## 10. Numeric claims

NLI models are unreliable on numbers and units. A claim of "40% speedup" against a
source saying "4-8% improvement in throughput" is exactly the failure case the tool
exists to catch, and a generic entailment model will often mark it supported.

Numeric claims are therefore handled before NLI:

1. Extract `(value, unit, direction, subject)` spans from claim and passage.
2. Compare magnitudes with unit normalization.
3. Mismatch beyond tolerance → `REFUTED (numeric mismatch)` with both figures shown.
4. Only if no numeric span is found does the claim fall through to NLI.

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

Verdicts are cached in sqlite keyed by `(claim_hash, source_id, model_id)`, so a
re-run of the same document costs 0 calls.

Even a 1 request/minute free tier finishes a 118-citation paper in ~4 minutes with
the judge enabled. With Ollama there is no wait at all.

## 12. Cross-platform constraints

| Component | Risk | Decision |
|---|---|---|
| PDF parsing | GROBID needs Java + Docker | pure-Python `pymupdf` default; GROBID opt-in via `--parser grobid` |
| Reference parsing | structured extraction is hard | avoided entirely — raw string to Crossref (§8) |
| Vector store | Qdrant server needs Docker | per-document corpus is ~10²-10³ chunks; `sqlite-vec`, no server |
| Scrapling | browser engines are heavy | core + `curl_cffi` bundled (2.7 MB); browser engine installed on demand with explicit consent (§7.1) |
| **Inference runtime** | **`torch` + `sentence-transformers` is ~800 MB — larger than the browser engine we ask consent for** | **ONNX runtime by default (base install stays small). `torch` moves to an opt-in `[gpu]` extra.** |
| Compute device | CUDA on Windows, MPS on Mac | preference order is always CUDA → MPS → CPU; under ONNX the equivalent is CUDA → CoreML → CPU |
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

proofpath permissions                      # show current permissions + config path
proofpath permissions set install_browser allow
proofpath cache clear
```

`proofpath` with no arguments launches the TUI; `proofpath <subcommand>` runs
one-shot. Implemented as a Typer callback with `invoke_without_command=True`, so a
bare invocation is a first-class entry point rather than a help screen.

TUI is built with `textual`. It accepts a file path, a URL, or raw pasted text,
streams progress, supports cancellation mid-run, and writes a report on completion.

## 14. Evaluation

The tool is not shippable without a frozen evaluation set. Metrics are reported per
stage so a regression can be located.

| Dataset | Stage measured | Metric |
|---|---|---|
| SciFact | retrieval + entailment on scientific claims | label accuracy, rationale F1 |
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

## 15. Error handling and honesty states

Every non-verdict is an explicit, reported state. Absence of evidence is never
presented as evidence of absence.

| State | Cause |
|---|---|
| `LOW CONFIDENCE (abstract only)` | full text unavailable, abstract used |
| `UNVERIFIED (blocked)` | 403/bot protection, Scrapling absent or defeated |
| `UNVERIFIED (unreachable)` | dead link, Wayback miss |
| `UNVERIFIED (provider unavailable)` | API down or rate limited after backoff |
| `AMBIGUOUS` | multiple plausible reference candidates — all listed |
| `NEI` | source read, but it neither supports nor contradicts |

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
- Fetched full text is cached locally for the run and not redistributed. Cache lives
  under the user cache dir and is clearable with `proofpath cache clear`.

## 17. Milestones

**v0.1 — academic provider.** Ingest PDF/MD, reference resolution with field-level
verification (§8), retraction check, fetch ladder steps 1 and 3, local retrieval and
entailment, numeric checker, markdown report with coverage summary, CLI only.
Measured on SciFact plus the hand-built ghost set.

**v0.2 — TUI, SARIF, fetch ladder.** `textual` REPL with cancellation, SARIF output,
verdict cache, fetch ladder steps 2-4 including the permission prompt and config file
(§7.1).

**v0.3 — judge layer.** Opt-in LLM second opinion, Ollama default, OpenRouter and
Gemini adapters, batching and cost reporting.

**v0.4 — social provider.** Bluesky and Hacker News first, Reddit via user-supplied
OAuth app, Mastodon best-effort, Community Notes dumps for X. Measured on AVeriTeC.

**Later.** Turkish sources, `spiyweb` graph retrieval as an alternative backend,
GROBID parser.

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
