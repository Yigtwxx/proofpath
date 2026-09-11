# SciFact dev — 2026-09-11

- pairs: 340  (limit=none)
- embedder: `BAAI/bge-small-en-v1.5`
- nli: `cross-encoder/nli-deberta-v3-base@6c749ce:model_qint8_arm64`  providers: `CPUExecutionProvider`
- machine: Darwin arm64  scoring 65.1s, 191 ms/pair

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
| 1 | 0.45 | 0.606 | 0.593 | 0.300 | 1.00 | 0.46 | 0 |
| 2 | 0.75 | 0.606 | 0.597 | 0.324 | 1.00 | 0.96 | 0 |
| 3 | 0.50 | 0.600 | 0.595 | 0.343 | 1.00 | 1.00 | 0 |
| 5 | 0.75 | 0.544 | 0.542 | 0.329 | 1.00 | 1.00 | 0 |

## Verdict on Phase 1

Best: k=1, decide=0.45, accuracy 0.606 vs best trivial baseline 0.406 → margin **+0.200**.

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
- **CPU beats CoreML for this int8 graph.** 64 pairs: CPU 0.40s, CoreML 1.14s.
  CoreML takes only 880 of 2,524 nodes, so the partition overhead dominates. The
  CUDA → CoreML → CPU rule was kept; the eval ran with `--providers cpu`.
  Whole dev split on CPU: 65s for 340 pairs (191 ms/pair including embedding).
- Reproduce: `uv run python scripts/eval_scifact.py --providers cpu --k 1,2,3,5`.
