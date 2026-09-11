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
