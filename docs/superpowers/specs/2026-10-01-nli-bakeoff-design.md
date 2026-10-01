# NLI bake-off — design (2026-10-01)

## Why

The entailment step is where proofpath loses most of its accuracy, and the obvious fix
has already been measured and does not work:

- SciFact dev retrieval recall goes from 0.560 at k=1 to 0.852 at k=3, but accuracy does
  not follow: k=1 0.609, k=2 0.609, k=3 0.603 (`docs/eval/2026-09-12-scifact-dev.md`).
  Taking the strongest of more passages adds as many wrong assertions as right ones.
- The model is overconfident: 116 verdicts score in [0.90, 1.00] and only 71.6 % of them
  are right. The `medium` cut (0.457948) sits on `decide` (0.45), so the report really
  has two tiers, and every cut was fitted and measured on the same split.
- On AVeriTeC every one of the 19 `Supported` claims was missed, and readable-source
  accuracy is 0.361 against a 0.708 baseline (`docs/eval/2026-09-16-averitec.md`).

So the next step is not a code change to the product. It is a measured comparison of
NLI models and aggregation rules, with the winner chosen by a rule fixed in advance.
Product integration is a separate spec written after the result.

## Candidates

| id | repo | export | size | trained on |
|---|---|---|---|---|
| `base` (current) | `cross-encoder/nli-deberta-v3-base` @ `6c749ce` | int8 per CPU | 244 MB | SNLI + MNLI |
| `large` | `cross-encoder/nli-deberta-v3-large` @ `bab4bc7` | int8 per CPU | 643 MB | SNLI + MNLI |
| `large-fever` | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` @ `b3546ea` | `onnx/model_quantized.onnx` | 643 MB | MNLI, FEVER-NLI, ANLI, LingNLI, WANLI |

All three load through the existing `entailment.OnnxNli(repo_id=, revision=, onnx_file=)`;
`label_permutation` already maps each model's `id2label`. Revisions are pinned to full
SHAs in the script. The downloads land on the developer's machine for this eval only;
nothing in the product downloads them (product rule 5).

## Data

- **SciFact train** — calibration only (`eval.scifact.load(split="train")`).
- **SciFact dev** — the reported, held-out split. It is the same 340 pairs as before, so
  the numbers compare directly with 2026-09-12.
- **AVeriTeC dev, first 100 claims** — a frozen snapshot. The 2026-09-16 run's fetched
  text has expired from the cache (7-day TTL), and re-fetching for each candidate would
  compare models on different pages. One network pass with the current ladder, browser
  not permitted (the default install, as in the published number), writes
  `<cache>/datasets/averitec_snapshot.json`: per claim, the gold label, and per source
  URL the fetch state and the extracted text. The snapshot stays outside the repo
  (third-party page text). Every candidate is scored from that one file.

## Method

`scripts/eval_nli_bakeoff.py`, dev-only, three subcommands:

1. `snapshot` — builds the AVeriTeC snapshot (network, `--sleep`, `--resume`). It reuses
   `eval_averitec.py`'s engine construction, `state_for` and per-URL fetch loop.
2. `score --model <id>` — offline. Splits each source into sentences with
   `retrieval.split_sentences`, as the product does, and retrieves the top 5 with
   `bge-small-en-v1.5` once; the retrieval is shared by every model and stored. Runs the
   model's NLI once over every (passage, claim) pair in that top 5 and stores the raw
   probabilities in `<cache>/datasets/bakeoff/<id>.npz`, with ms/pair. The numeric layer
   runs unchanged on the same top-k (`numerics.check`), so a candidate is never credited
   with what the rule decided.
3. `report` — offline. From the stored probabilities it computes every combination of
   model × k ∈ {1, 2, 3} × aggregation, calibrates each on SciFact train, scores it on
   SciFact dev and on AVeriTeC, and writes `docs/eval/<date>-nli-bakeoff.md`.

### Aggregation variants

Pure functions over `(hits, probs)` that return a `Verdict`. They live in the script; a
variant moves into `pipeline.py` only if it wins.

- `max` — today's `pipeline.aggregate`: the strongest p(SUPPORTED) or p(REFUTED) over
  the hits decides.
- `max_nei` — as `max`, but a hit only counts when its asserted label beats its own
  p(NEI).
- `margin` — score = max p(S) − max p(R) over the hits. The sign gives the label and the
  absolute value is the score, so a supporting passage and a contradicting passage on
  the same source cancel out instead of the louder one winning.

### Calibration

For each combination, on SciFact **train**:

- `decide` is the best-accuracy cut from `eval.metrics.sweep_decide`.
- `high` and `medium` come from `eval.metrics.tier_cutpoints` at the existing 0.85 and
  0.70 precision targets.

Those cuts are then applied unchanged to dev and AVeriTeC. Each tier's dev precision is
reported with its n and a 95 % Wilson interval. A tier that holds fewer than 20 dev
verdicts is reported as "too few to judge" rather than given a precision.

### AVeriTeC scoring

Per claim, the same rule as `eval_averitec._decide_claim`:

- the strongest non-NEI verdict that carries a passage, across the claim's sources, wins;
- a claim with a readable source but no assertion is `NEI`;
- a claim with no readable source is unanswered.

Two numbers are reported: 3-way accuracy over all answerable claims (comparable to
0.270), and over claims with at least one readable source (comparable to 0.361). The
0.708 majority baseline is printed beside both.

## Metrics in the report

For every combination:

- SciFact dev: accuracy, macro-F1, per-label F1, rationale F1, and the tier table;
- AVeriTeC: both 3-way accuracies and the per-label counts;
- for each model: ms/pair (Apple Silicon CPU, int8) and download size;
- "asserted without a passage", which must be 0 in every row (product rule 1).

## Decision rule (fixed before any number is seen)

1. **Eligible:** zero assertions without a passage, and AVeriTeC readable-subset accuracy
   not below the `base` × k=1 × `max` row.
2. **Winner:** the highest SciFact dev macro-F1 among eligible rows.
3. **Size gate:** a `large*` row wins only if it beats the best eligible `base` row by
   **≥ 0.03** macro-F1. Otherwise the best `base` row wins.
4. **Tie:** two rows within 0.01 of each other go to the smaller k, then to the simpler
   aggregation, in the order `max`, `max_nei`, `margin`.
5. If nothing beats `base` × k=1 × `max` by ≥ 0.01 macro-F1, the result is "no change".
   It is written up as such, and the next package (B, coverage) starts.

The rule is copied into the report verbatim, next to the numbers it chose.

## What happens after

A follow-up spec, written after the report:

- **`base` wins with a new k or aggregation:** change `pipeline.aggregate` and
  `DEFAULT_THRESHOLDS`, and update `cache.model_id` keying. Old verdicts invalidate
  through the threshold key that already exists.
- **A `large*` row wins:** add an opt-in "accurate" NLI model. It is downloaded only with
  consent (product rule 5, spec §7.1) and is never the default, because the default
  install must stay small. It carries its own thresholds.

Either way, the README "Measured" table, spec §14 and the tier note are updated with the
new numbers.

## Tests (no network)

- The aggregation variants are tested on hand-built probability rows: the `margin`
  cancellation, the NEI veto in `max_nei`, and no passage → no assertion.
- The train/dev split guard: calibration must never see a dev pair id.
- Wilson interval values, and the "too few to judge" cut-off.
- The snapshot round trip: write, load, and `--resume` skipping finished ids, against a
  fixture `Fetched`.
- The decision rule as a function over synthetic result rows, covering the size gate,
  the tie order and "no change".

## Out of scope

- Changing the embedder or chunking (OPEN-ITEMS 3.6 stays open).
- Fetch coverage, including running with the browser allowed; package B covers that.
- Any change to product code or defaults; that is the follow-up spec.
- A FEVER-trained `base` model: it has no ONNX export, and exporting one needs torch.
