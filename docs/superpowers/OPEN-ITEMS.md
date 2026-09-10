# Open items

Everything left unresolved as of **2026-09-10**, written down so none of it has to be
reconstructed from memory. Each entry says enough to be picked up cold.

Spec: `specs/2026-09-10-proofpath-design.md` · Plan: `plans/2026-09-10-proofpath-implementation-plan.md`

---

## 1. Blocked — resolved 2026-09-10

| # | Item | Outcome |
|---|---|---|
| 1.1 | CI workflow could not be pushed; the token lacked the `workflow` scope. The Contents API refuses workflow files for the same reason, so the token refresh was the only route. | **Done.** CI runs on Linux, macOS and Windows × Python 3.10 and 3.13. All seven jobs green. |
| 1.2 | PyPI name `proofpath` was free but unreserved. | **Done.** Published 0.0.1 through trusted publishing (OIDC), so no API token is stored anywhere. A working CLI was added first — the declared entry point did not exist, and publishing it would have shipped a command that raises ImportError. |

---

## 2. Decisions taken but not confirmed by the author

Recorded with the recommendation that was made, so they can be accepted or reversed
deliberately rather than by default.

| # | Decision | Recommended | Why it might change |
|---|---|---|---|
| 2.1 | `install_browser` default (spec §7.1) | `ask` | `deny` is safer but hides from the user that a blocked source was recoverable |
| 2.2 | Reddit in v0.4 | include, optional | needs a user-registered OAuth app; Bluesky and HN work without one, so Reddit is never blocking |
| 2.3 | `spiyweb` keeps its name | yes | the spider-web-as-graph metaphor fits; every clean alternative on PyPI was worse |
| 2.4 | `reasonhound` keeps its name | yes | already a good name |
| 2.5 | Turkish sources | after v0.4 | OpenAlex/Crossref coverage is much weaker for Turkish; half-supporting it would damage trust |
| 2.6 | `--summarize` in the TUI | off by default there too | keeping "offline unless asked" true in both front-ends |

---

## 3. Unvalidated assumptions — test before building on them

These were asserted during design without being checked. Each one, if wrong, changes
a phase.

| # | Assumption | How to check | If wrong |
|---|---|---|---|
| 3.1 | **A usable NLI cross-encoder exists as ONNX.** The whole "no torch in the base install" decision rests on this and it was never verified. | Search the ONNX model zoo / HF for a DeBERTa-MNLI or similar exported model; measure it on SciFact dev | either export one with `optimum`, or make `[gpu]`/torch the default and accept the install size |
| 3.2 | `sqlite-vec` ships working wheels for Windows and macOS arm64 | install on all three in CI during Phase 0 | fall back to numpy brute force; per-document corpora are small enough |
| 3.3 | `textual` renders the §13.1 layout correctly in Windows Terminal at 80 columns | render a fixture screen in CI on Windows | simplify the box drawing to ASCII |
| 3.4 | SciFact, AVeriTeC and PubHealth are still downloadable and pinnable | fetch each once, record the revision | substitute a comparable set and note it in the spec |
| 3.5 | X Community Notes dumps are still published and parseable | download one day's file, inspect the columns | drop the X path entirely; Bluesky and HN already carry the social provider |
| 3.6 | `fastembed` model quality is sufficient for passage ranking | compare recall@k against a sentence-transformers baseline in Phase 1 | move embeddings to the `[gpu]` extra as well |

---

## 4. Design details deliberately left to implementation

Not oversights — they need real data to set, and guessing now would be false
precision. Each must end up written into the spec once measured.

| # | Detail | Set during |
|---|---|---|
| 4.1 | Title token-set Jaccard threshold for reference matching (spec §8) | Phase 3, tuned on the ghost test set |
| 4.2 | Year tolerance beyond ±1 for online-first publications | Phase 3 |
| 4.3 | Numeric comparison tolerance, and how to treat ranges vs point values (spec §10) | Phase 2 |
| 4.4 | Confidence threshold that routes a verdict to the judge | Phase 1 calibration |
| 4.5 | Chunk size and `k` for retrieval. Abstracts are trivial; a 12,000-word full text is not, and the two may need different settings | Phase 1, re-checked in Phase 4 |
| 4.6 | Which embedding model and which NLI model, by name and revision | Phase 1 |

---

## 5. Genuinely unsolved problems

Known gaps with no chosen answer yet. These are the ones worth thinking about away
from the keyboard.

**5.1 — A citation that supports a paragraph, not a sentence.**
Spec §5.1 mentions a paragraph-level fallback but does not define when it triggers.
Attaching a paragraph-wide claim to one sentence produces confident nonsense. Needs a
rule, and probably a distinct reported state.

**5.2 — Author-year citation styles.**
`(Smith et al., 2020)` is much harder than `[12]`: the marker does not index the
bibliography directly, several works share an author-year, and `ibid.`/`op. cit.`
exist. Phase 5 currently treats both styles as one task; they are not.

**5.3 — Coverage is the real product ceiling.**
Measured: direct full text for well under half of sampled citations (spec §6.1). Every
abstract-only verdict is a weak verdict. Worth investigating whether Semantic Scholar,
CORE, or OpenAIRE meaningfully raise this before accepting the number.

**5.4 — What "confidence" means to a user.**
A `0.91` is a model score, not a probability of being right. Showing it as-is invites
misreading, hiding it removes signal. No decision yet on how to present it.

**5.5 — Caching fetched full text.**
Spec §16 says cached locally, not redistributed, clearable. Retention period and
whether publisher content should be cached at all are unresolved.

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

## 7. Next session starts here

1. Check assumption **3.1** — if no usable ONNX NLI model exists, the packaging story
   changes and it is better to know before Phase 1 than during it.
2. Begin **Phase 1**: pin SciFact, build `scripts/eval.py`, get a first number.

Phase 1 carries a kill criterion on purpose. Getting to a real number quickly is the
point of the whole ordering.
