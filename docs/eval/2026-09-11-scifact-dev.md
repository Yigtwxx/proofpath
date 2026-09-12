# SciFact dev — 2026-09-11

- pairs: 340  (limit=none)
- embedder: `BAAI/bge-small-en-v1.5`
- nli: `cross-encoder/nli-deberta-v3-base@6c749ce:model_qint8_arm64`  providers: `CPUExecutionProvider`
- numeric layer: on
- machine: Darwin arm64  scoring 63.6s, 187 ms/pair

## Baselines

| baseline | accuracy | macro-F1 |
|---|---|---|
| majority label (SUPPORTED) | 0.406 | 0.192 |
| source exists → SUPPORTED | 0.406 | 0.192 |

## Retrieval (assumption 3.6)

| k | recall@k (any gold sentence in top-k) |
|---|---|
| 1 | 0.560 |
| 2 | 0.751 |
| 3 | 0.852 |
| 5 | 0.923 |

## Entailment — threshold sweep per k

| k | decide | accuracy | macro-F1 | rationale F1 | high cut | medium cut | asserted w/o passage |
|---|---|---|---|---|---|---|---|
| 1 | 0.45 | 0.609 | 0.597 | 0.299 | 1.00 | 0.46 | 0 |
| 2 | 0.75 | 0.609 | 0.601 | 0.324 | 1.00 | 0.96 | 0 |
| 3 | 0.50 | 0.603 | 0.598 | 0.342 | 1.00 | 1.00 | 0 |
| 5 | 0.75 | 0.544 | 0.542 | 0.329 | 1.00 | 1.00 | 0 |

Numeric layer firings on the top-k passages:

- k=1: numeric layer refuted 1, 1 correct
- k=2: numeric layer refuted 1, 1 correct
- k=3: numeric layer refuted 1, 1 correct
- k=5: numeric layer refuted 0, 0 correct

## Verdict on Phase 1

Best: k=1, decide=0.45, accuracy 0.609 vs best trivial baseline 0.406 → margin **+0.203**.

Kill criterion: margin near zero after tuning means stop before Phase 2.
## Notes

- **Kill criterion passed.** +0.200 accuracy over the trivial baseline with no
  tuning beyond the decide threshold. Published SciFact abstract-level label
  accuracy for purpose-built systems sits around 0.70-0.75; a generic MNLI
  cross-encoder with no domain training landing at 0.61 is the expected gap, and it
  is the gap the numeric layer (Phase 2) and the optional judge (Phase 9) exist to
  narrow.
- **k=1 or k=2 is enough on abstracts.** Larger k hurts: more passages mean more
  chances for a spurious strong score. Full text (Phase 4) will need re-checking.
- **Retrieval is not the bottleneck.** recall@3 = 0.85 and recall@5 = 0.92 with
  `bge-small-en-v1.5`, so assumption 3.6 holds for abstracts. Not yet compared with a
  second embedding model; recorded as still open.
- **Rationale F1 is low (0.30-0.34) by construction.** A verdict carries exactly one
  passage, while SciFact gold rationales often span two or three sentences. This
  measures "is the quoted sentence a gold sentence", not the full rationale set.
- **Tier cut-points are not usable yet.** With a precision target of 0.85, `high` is
  never reached; with 0.70, `medium` covers nearly every decided verdict. The
  calibrated tiers therefore collapse to `medium`/`low` on this model. Options:
  lower the targets, or accept that this model does not earn a `high` tier and say
  so in the report. Decision pending; the pipeline default stays
  `Thresholds(decide=0.5, high=0.9, medium=0.7)` until it is made.
  **Corrected 2026-09-12:** this note was wrong. `high` *is* reached, at 0.99933 —
  the two-decimal column above rounded it to the same `1.00` the harness prints when
  a target is never met. Cut-points now print to six decimals. The calibration that
  replaced the placeholder is `docs/eval/2026-09-12-tiers.md`.
- **CPU beats CoreML for this int8 graph.** 64 pairs: CPU 0.40s, CoreML 1.14s.
  CoreML takes only 880 of 2,524 nodes, so the partition overhead dominates. The
  CUDA → CoreML → CPU rule was kept; the eval ran with `--providers cpu`.
  Whole dev split on CPU: 65s for 340 pairs (191 ms/pair including embedding).
- Reproduce: `uv run python scripts/eval_scifact.py --providers cpu --k 1,2,3,5`.
- **Phase 2 (numeric layer) added later the same day.** Accuracy with the layer on
  vs off: k=1 0.609 vs 0.606, k=2 0.609 vs 0.606, k=3 0.603 vs 0.600 — never lower.
  On the top-k passages it refuted 1 pair, correctly. Reproduce the "off" row with
  `--no-numerics`.
- **The numeric layer was tightened three times against this split.** The first
  version fired 4 times over whole abstracts, all wrong: `5% of perinatal mortality`
  against a `95% CI` bound, `decreased by 10%` against `57% women`, `H3.3` parsed as
  `3.3`, "less than 10%" read as a decrease. Rules now: numbers glued to letters or
  signs are labels; "than" comparators carry no direction; `NN% CI` is a confidence
  level; a mismatch needs exactly one comparable figure in the passage and one in
  the claim; a change (with direction) is never refuted by a level (without). After
  that, whole-abstract firings on dev dropped to 0 and the layer only decides when
  attribution is unambiguous. SciFact has almost no numeric contradictions, so the
  hand-built set in `tests/test_numerics.py` is the real acceptance test.
