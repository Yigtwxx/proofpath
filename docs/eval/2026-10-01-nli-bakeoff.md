# NLI bake-off — 2026-10-01

- spec: `docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md`
- calibrated on SciFact train (919 pairs); reported on SciFact dev (340 pairs) and AVeriTeC dev (100 claims, 89 answerable (3-way gold label), 80 of those with a readable source; frozen snapshot, browser not permitted)
- embedder `BAAI/bge-small-en-v1.5`, top 5 passages per item, numeric layer on
- machine: Darwin arm64
- coverage: all candidate models were scored

## Decision rule (fixed before any number was seen)

The steps run in this order. Each one filters the rows the previous step left.

1. **Eligible:** zero assertions without a passage, and AVeriTeC readable-subset accuracy
   not below the `base` × k=1 × `max` row.
2. **Beats the baseline:** a row stays only if its SciFact dev macro-F1 beats
   `base` × k=1 × `max` by **≥ 0.01**. If no row stays, the result is "no change". It is
   written up as such, and the next package (B, coverage) starts.
3. **Size gate:** a `large*` row stays only if it beats the highest macro-F1 of any
   eligible `base` row by **≥ 0.03**. That is the strongest eligible `base` row, not the
   one the tie rule would pick.
4. **Winner:** the highest SciFact dev macro-F1 among the rows left.
5. **Tie:** rows left within 0.01 of the winner go to the smaller k, then to the simpler
   aggregation, in the order `max`, `max_nei`, `margin`.

## Result

**Winner:** `large-fever` × k=2 × `max` is +0.117 macro-F1 over `base` × k=1 × `max`.

Over the best eligible `base` row, `base` × k=2 × `margin` (0.608), the winner is +0.089 macro-F1.

## Models

| model | repo | size (MB) | ms/pair |
|---|---|---|---|
| base | `cross-encoder/nli-deberta-v3-base` | 244 | 19 |
| large | `cross-encoder/nli-deberta-v3-large` | 643 | 55 |
| large-fever | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 643 | 54 |

## Every combination

AVeriTeC majority baseline: 0.708. The 2026-09-16 run reported 0.270 (all) and 0.361 (readable), but on a different snapshot with different cuts, over 89 answerable / 61 readable claims; this run counts 89 answerable / 80 readable. A gap between the two runs is partly a coverage change, not only a model gain.

| model | k | aggregation | decide | medium | high | dev acc | dev macro-F1 | F1 S / R / NEI | rationale F1 | AVeriTeC all | AVeriTeC readable | w/o passage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1 | max | 0.20 | 0.915356 | 1.000000 | 0.591 | 0.580 | 0.54 / 0.57 / 0.63 | 0.299 | 0.404 | 0.438 | 0 |
| base | 1 | max_nei | 0.40 | 0.915356 | 1.000000 | 0.594 | 0.578 | 0.52 / 0.57 / 0.65 | 0.291 | 0.360 | 0.388 | 0 |
| base | 1 | margin | 0.20 | 0.878350 | 1.000000 | 0.591 | 0.579 | 0.54 / 0.57 / 0.63 | 0.296 | 0.393 | 0.425 | 0 |
| base | 2 | max | 0.20 | 0.999791 | 1.000000 | 0.609 | 0.606 | 0.61 / 0.59 / 0.62 | 0.324 | 0.494 | 0.537 | 0 |
| base | 2 | max_nei | 0.40 | 0.999791 | 1.000000 | 0.594 | 0.589 | 0.57 / 0.58 / 0.62 | 0.316 | 0.472 | 0.512 | 0 |
| base | 2 | margin | 0.15 | 0.992588 | 1.000000 | 0.609 | 0.608 | 0.60 / 0.61 / 0.62 | 0.326 | 0.483 | 0.525 | 0 |
| base | 3 | max | 0.80 | 1.000000 | 1.000000 | 0.594 | 0.588 | 0.59 / 0.55 / 0.63 | 0.339 | 0.494 | 0.537 | 0 |
| base | 3 | max_nei | 0.80 | 1.000000 | 1.000000 | 0.594 | 0.588 | 0.59 / 0.55 / 0.63 | 0.339 | 0.494 | 0.537 | 0 |
| base | 3 | margin | 0.05 (grid floor) | 1.000000 | 1.000000 | 0.562 | 0.563 | 0.61 / 0.54 / 0.54 | 0.330 | 0.551 | 0.600 | 0 |
| large | 1 | max | 0.70 | 0.924491 | 1.000000 | 0.582 | 0.572 | 0.49 / 0.61 / 0.62 | 0.292 | 0.393 | 0.425 | 0 |
| large | 1 | max_nei | 0.70 | 0.924491 | 1.000000 | 0.582 | 0.572 | 0.49 / 0.61 / 0.62 | 0.292 | 0.393 | 0.425 | 0 |
| large | 1 | margin | 0.70 | 0.920455 | 1.000000 | 0.582 | 0.573 | 0.49 / 0.61 / 0.62 | 0.288 | 0.393 | 0.425 | 0 |
| large | 2 | max | 0.80 | 0.999586 | 1.000000 | 0.609 | 0.607 | 0.54 / 0.65 / 0.63 | 0.338 | 0.427 | 0.463 | 0 |
| large | 2 | max_nei | 0.80 | 0.999586 | 1.000000 | 0.609 | 0.607 | 0.54 / 0.65 / 0.63 | 0.338 | 0.427 | 0.463 | 0 |
| large | 2 | margin | 0.70 | 0.986530 | 1.000000 | 0.597 | 0.597 | 0.52 / 0.66 / 0.61 | 0.331 | 0.416 | 0.450 | 0 |
| large | 3 | max | 0.90 | 0.999698 | 1.000000 | 0.597 | 0.595 | 0.57 / 0.60 / 0.61 | 0.343 | 0.494 | 0.537 | 0 |
| large | 3 | max_nei | 0.90 | 0.999698 | 1.000000 | 0.597 | 0.595 | 0.57 / 0.60 / 0.61 | 0.343 | 0.494 | 0.537 | 0 |
| large | 3 | margin | 0.90 | 0.999396 | 1.000000 | 0.588 | 0.585 | 0.52 / 0.63 / 0.61 | 0.334 | 0.449 | 0.487 | 0 |
| large-fever | 1 | max | 0.05 (grid floor) | 0.074872 | 0.997081 | 0.647 | 0.642 | 0.61 / 0.64 / 0.67 | 0.339 | 0.472 | 0.512 | 0 |
| large-fever | 1 | max_nei | 0.40 | 0.440642 | 0.997081 | 0.632 | 0.624 | 0.53 / 0.67 / 0.67 | 0.313 | 0.360 | 0.388 | 0 |
| large-fever | 1 | margin | 0.05 (grid floor) | 0.064411 | 1.000000 | 0.638 | 0.633 | 0.59 / 0.64 / 0.66 | 0.338 | 0.472 | 0.512 | 0 |
| large-fever | 2 | max | 0.25 | 0.252721 | 1.000000 | 0.697 | 0.697 | 0.66 / 0.72 / 0.71 | 0.383 | 0.438 | 0.475 | 0 |
| large-fever | 2 | max_nei | 0.45 | 0.465651 | 1.000000 | 0.679 | 0.680 | 0.64 / 0.71 / 0.69 | 0.372 | 0.416 | 0.450 | 0 |
| large-fever | 2 | margin | 0.20 | 0.232304 | 1.000000 | 0.688 | 0.690 | 0.65 / 0.73 / 0.69 | 0.373 | 0.438 | 0.475 | 0 |
| large-fever | 3 | max | 0.45 | 0.552317 | 1.000000 | 0.674 | 0.672 | 0.65 / 0.68 / 0.69 | 0.356 | 0.472 | 0.512 | 0 |
| large-fever | 3 | max_nei | 0.45 | 0.552317 | 1.000000 | 0.671 | 0.669 | 0.64 / 0.68 / 0.68 | 0.356 | 0.472 | 0.512 | 0 |
| large-fever | 3 | margin | 0.20 | 0.438075 | 1.000000 | 0.674 | 0.675 | 0.65 / 0.70 / 0.67 | 0.361 | 0.483 | 0.525 | 0 |

## AVeriTeC per label

Cells are correct/n per gold label. `n (not scored)` marks a gold label with no 3-way verdict; it is not a model failure.

| model | k | aggregation | Conflicting Evidence/Cherrypicking | Not Enough Evidence | Refuted | Supported |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1 | max | 11 (not scored) | 5/7 | 30/63 | 1/19 |
| base | 1 | max_nei | 11 (not scored) | 5/7 | 27/63 | 0/19 |
| base | 1 | margin | 11 (not scored) | 5/7 | 29/63 | 1/19 |
| base | 2 | max | 11 (not scored) | 2/7 | 41/63 | 1/19 |
| base | 2 | max_nei | 11 (not scored) | 2/7 | 39/63 | 1/19 |
| base | 2 | margin | 11 (not scored) | 2/7 | 40/63 | 1/19 |
| base | 3 | max | 11 (not scored) | 3/7 | 40/63 | 1/19 |
| base | 3 | max_nei | 11 (not scored) | 3/7 | 40/63 | 1/19 |
| base | 3 | margin | 11 (not scored) | 2/7 | 46/63 | 1/19 |
| large | 1 | max | 11 (not scored) | 5/7 | 27/63 | 3/19 |
| large | 1 | max_nei | 11 (not scored) | 5/7 | 27/63 | 3/19 |
| large | 1 | margin | 11 (not scored) | 5/7 | 27/63 | 3/19 |
| large | 2 | max | 11 (not scored) | 3/7 | 31/63 | 4/19 |
| large | 2 | max_nei | 11 (not scored) | 3/7 | 31/63 | 4/19 |
| large | 2 | margin | 11 (not scored) | 3/7 | 31/63 | 3/19 |
| large | 3 | max | 11 (not scored) | 2/7 | 38/63 | 4/19 |
| large | 3 | max_nei | 11 (not scored) | 2/7 | 38/63 | 4/19 |
| large | 3 | margin | 11 (not scored) | 2/7 | 35/63 | 3/19 |
| large-fever | 1 | max | 11 (not scored) | 5/7 | 34/63 | 3/19 |
| large-fever | 1 | max_nei | 11 (not scored) | 6/7 | 24/63 | 2/19 |
| large-fever | 1 | margin | 11 (not scored) | 5/7 | 34/63 | 3/19 |
| large-fever | 2 | max | 11 (not scored) | 4/7 | 33/63 | 2/19 |
| large-fever | 2 | max_nei | 11 (not scored) | 4/7 | 31/63 | 2/19 |
| large-fever | 2 | margin | 11 (not scored) | 4/7 | 33/63 | 2/19 |
| large-fever | 3 | max | 11 (not scored) | 4/7 | 35/63 | 3/19 |
| large-fever | 3 | max_nei | 11 (not scored) | 4/7 | 35/63 | 3/19 |
| large-fever | 3 | margin | 11 (not scored) | 4/7 | 36/63 | 3/19 |

## Tiers on SciFact dev

Cells are n · precision (95 % Wilson interval); n below 20 is too few to judge.

| model | k | aggregation | high | medium | low |
|---|---|---|---|---|---|
| base | 1 | max | 1 · too few to judge | 109 · 0.725 (Wilson 0.63–0.80) | 33 · 0.515 (Wilson 0.35–0.67) |
| base | 1 | max_nei | 1 · too few to judge | 109 · 0.725 (Wilson 0.63–0.80) | 18 · too few to judge |
| base | 1 | margin | 1 · too few to judge | 112 · 0.732 (Wilson 0.64–0.81) | 27 · 0.481 (Wilson 0.31–0.66) |
| base | 2 | max | 1 · too few to judge | 13 · too few to judge | 175 · 0.611 (Wilson 0.54–0.68) |
| base | 2 | max_nei | 1 · too few to judge | 13 · too few to judge | 158 · 0.614 (Wilson 0.54–0.69) |
| base | 2 | margin | 1 · too few to judge | 75 · 0.800 (Wilson 0.70–0.87) | 110 · 0.527 (Wilson 0.43–0.62) |
| base | 3 | max | 1 · too few to judge | 0 · too few to judge | 181 · 0.608 (Wilson 0.54–0.68) |
| base | 3 | max_nei | 1 · too few to judge | 0 · too few to judge | 181 · 0.608 (Wilson 0.54–0.68) |
| base | 3 | margin | 1 · too few to judge | 0 · too few to judge | 220 · 0.559 (Wilson 0.49–0.62) |
| large | 1 | max | 1 · too few to judge | 112 · 0.759 (Wilson 0.67–0.83) | 8 · too few to judge |
| large | 1 | max_nei | 1 · too few to judge | 112 · 0.759 (Wilson 0.67–0.83) | 8 · too few to judge |
| large | 1 | margin | 1 · too few to judge | 112 · 0.750 (Wilson 0.66–0.82) | 7 · too few to judge |
| large | 2 | max | 1 · too few to judge | 31 · 0.935 (Wilson 0.79–0.98) | 117 · 0.650 (Wilson 0.56–0.73) |
| large | 2 | max_nei | 1 · too few to judge | 31 · 0.935 (Wilson 0.79–0.98) | 117 · 0.650 (Wilson 0.56–0.73) |
| large | 2 | margin | 1 · too few to judge | 93 · 0.828 (Wilson 0.74–0.89) | 48 · 0.500 (Wilson 0.36–0.64) |
| large | 3 | max | 1 · too few to judge | 34 · 0.882 (Wilson 0.73–0.95) | 129 · 0.605 (Wilson 0.52–0.68) |
| large | 3 | max_nei | 1 · too few to judge | 34 · 0.882 (Wilson 0.73–0.95) | 129 · 0.605 (Wilson 0.52–0.68) |
| large | 3 | margin | 1 · too few to judge | 28 · 0.929 (Wilson 0.77–0.98) | 114 · 0.640 (Wilson 0.55–0.72) |
| large-fever | 1 | max | 27 · 0.963 (Wilson 0.82–0.99) | 121 · 0.678 (Wilson 0.59–0.75) | 5 · too few to judge |
| large-fever | 1 | max_nei | 27 · 0.963 (Wilson 0.82–0.99) | 93 · 0.763 (Wilson 0.67–0.84) | 0 · too few to judge |
| large-fever | 1 | margin | 1 · too few to judge | 147 · 0.735 (Wilson 0.66–0.80) | 1 · too few to judge |
| large-fever | 2 | max | 1 · too few to judge | 165 · 0.776 (Wilson 0.71–0.83) | 0 · too few to judge |
| large-fever | 2 | max_nei | 1 · too few to judge | 155 · 0.781 (Wilson 0.71–0.84) | 0 · too few to judge |
| large-fever | 2 | margin | 1 · too few to judge | 161 · 0.783 (Wilson 0.71–0.84) | 3 · too few to judge |
| large-fever | 3 | max | 1 · too few to judge | 167 · 0.743 (Wilson 0.67–0.80) | 6 · too few to judge |
| large-fever | 3 | max_nei | 1 · too few to judge | 167 · 0.743 (Wilson 0.67–0.80) | 5 · too few to judge |
| large-fever | 3 | margin | 1 · too few to judge | 161 · 0.758 (Wilson 0.69–0.82) | 21 · 0.429 (Wilson 0.24–0.63) |

## Numeric layer

The numeric layer (spec §10) runs on the same top-k passages before any model. A mismatch decides REFUTED by rule at score 1.0, and no model is credited with it. It depends on k only, so one row per k covers every model and aggregation. A firing is correct when the gold label is REFUTED.

| k | SciFact train fired | train correct | SciFact dev fired | dev correct |
|---|---|---|---|---|
| 1 | 4 | 0 | 1 | 1 |
| 2 | 5 | 0 | 1 | 1 |
| 3 | 6 | 0 | 1 | 1 |

Rule-decided rows score exactly 1.0, the top of the scale, so they are among the first rows the `high` cut's walk down SciFact train meets: a wrong firing lowers the precision every lower cut is measured with, and a `high` cut of 1.000000 means no score below 1.0 held 0.85 precision, so only a score of 1.0 reaches the tier. The `high` cut is 1.000000 in 25 of 27 combinations. In 25 of 27, every SciFact dev `high` verdict is a rule firing (the high-tier n equals the dev firings at that k): there the high tier on dev comes from the rule, not the model.

## Limits

- **Dev chooses and reports.** SciFact dev both picks the winner and reports it. The winner's 0.697 macro-F1 comes from the split that chose it, out of 27 combinations whose highest dev macro-F1 is 0.697, so it is optimistic; a held-out split would likely read lower.
- **int8 scores depend on the batch.** The candidate files are dynamically quantised to int8, so a pair's probabilities depend on what else is in its batch. Re-scoring 48 SciFact dev top-1 pairs one at a time moved probabilities by up to 0.197 (mean 0.008) and flipped 1 argmax (measured 2026-10-01 by the final review). That likely explains why `base` × k=1 × `max` reaches dev accuracy 0.591 here (0.597 even at decide 0.45) against the 0.609 published on 2026-09-12 with dev-fitted cuts.
- **The AVeriTeC snapshot is new.** It is not the 2026-09-16 run's: coverage improved since (then 61 of 89 answerable claims were readable, now 80 of 89), so AVeriTeC numbers here are not comparable with that run's.
- **Grid floor.** 3 of 27 combinations put `decide` on the grid floor (0.05), marked `(grid floor)` above: SciFact train accuracy was best at the lowest cut tried, so the best cut may lie below the grid.
