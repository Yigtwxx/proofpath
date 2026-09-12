# SciFact dev — 2026-09-12

- pairs: 340  (limit=none)
- embedder: `BAAI/bge-small-en-v1.5`
- nli: `cross-encoder/nli-deberta-v3-base@6c749ce:model_qint8_arm64`  providers: `CPUExecutionProvider`
- numeric layer: on
- machine: Darwin arm64  scoring 62.8s, 185 ms/pair

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

## Entailment — threshold sweep per k

| k | decide | accuracy | macro-F1 | rationale F1 | high cut | medium cut | asserted w/o passage |
|---|---|---|---|---|---|---|---|
| 1 | 0.45 | 0.609 | 0.597 | 0.299 | 0.999330 | 0.457948 | 0 |
| 2 | 0.75 | 0.609 | 0.601 | 0.324 | 0.999714 | 0.963921 | 0 |
| 3 | 0.50 | 0.603 | 0.598 | 0.342 | 0.999824 | 0.998824 | 0 |

Numeric layer firings on the top-k passages:

- k=1: numeric layer refuted 1, 1 correct
- k=2: numeric layer refuted 1, 1 correct
- k=3: numeric layer refuted 1, 1 correct

## Confidence tiers (k=1, decide=0.45)

| band | n | precision |
|---|---|---|
| [0.45, 0.60) | 4 | 1.000 |
| [0.60, 0.70) | 5 | 0.600 |
| [0.70, 0.80) | 4 | 1.000 |
| [0.80, 0.90) | 6 | 0.500 |
| [0.90, 1.00] | 116 | 0.716 |

| high target | medium target | high cut | n ≥ high | medium cut | n ≥ medium |
|---|---|---|---|---|---|
| 0.85 | 0.70 | 0.999330 | 21 | 0.457948 | 134 |
| 0.80 | 0.65 | 0.998824 | 29 | 0.457948 | 134 |

`low` is effectively empty: the `medium` cut (0.457948) sits on `decide` (0.45), so 134 of the decided verdicts are medium or better and the display is two tiers, not three.

## Verdict on Phase 1

Best: k=1, decide=0.45, accuracy 0.609 vs best trivial baseline 0.406 → margin **+0.203**.

Kill criterion: margin near zero after tuning means stop before Phase 2.
