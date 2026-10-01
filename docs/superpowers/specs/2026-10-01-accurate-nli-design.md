# Opt-in accurate NLI model: design (2026-10-01)

## Why

The NLI bake-off (`docs/eval/2026-10-01-nli-bakeoff.md`) picked
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` at k=2 with `max` aggregation.
Its SciFact dev macro-F1 is 0.697. The current default scores 0.580, and the best
configuration of the default model 0.608. The winner costs 643 MB instead of 244 MB and
runs at 54 ms per pair instead of 19 ms. It does **not** improve the news and fact-check
case: AVeriTeC on readable claims is 0.475, against 0.537 for the default model at k=2,
and both are below the 0.708 majority baseline. The default install must stay small
(product rule 5, spec §7.1), so the large model ships as an **opt-in** profile, not a
new default.

The bake-off left three open points (OPEN-ITEMS 20.11). This design settles two of them:

- **(a)** Rule-decided rows (the numeric layer, score 1.0) no longer set the `high` cut.
  They fired 4–6 times on SciFact train and none was correct, which pinned `high` at
  1.000000. The accurate profile's cuts are calibrated with those rows left out of the
  tier walk (they still count toward `decide`).
- **(b)** A run whose `medium` cut sits on `decide` now says that it has no `low`
  tier, in the same derived way it already says when it has no `high` tier.

**(c)** stays open, and the documentation says so: AVeriTeC is not better with the
accurate model, so it is recommended for scientific sources, not for news.

## User-facing behaviour

- **Config:** a new section with one key, `[models] nli = "default" | "accurate"`. The
  default is `"default"`. It is set with
  `proofpath config set models.nli accurate`, or from the `/config` panel.
- **One run:** `proofpath check --accurate` uses the accurate profile for one run.
- **Consent:** a new permission, `permissions.install_model = "ask" | "allow" | "deny"`.
  Its default is `"ask"`.
  - When the accurate profile is chosen and its files are not yet in the model cache, the
    run asks once before downloading. The prompt states the model and its size
    (643 MB). The answers are the browser gate's: once, always, no, never.
    "Always" and "never" are saved to `permissions.install_model`.
  - With no interactive terminal, `ask` is treated as `deny` and the denial is reported
    (product rule 4).
  - When the download is denied or fails, the run **falls back to the default
    profile**. It says so as a Note, on stderr in the CLI, and on a `models` entry in the
    report. Silently using another model would make the report claim a model it did
    not run.
  - Once the files are cached, no prompt is shown.
- **Report:** the `models` block gains `k`. When the accurate profile was requested but
  not used, it also gains `nli_requested: accurate (<reason>)`. The verdict cache is keyed by
  model id, k and thresholds already, so the two profiles never share cached verdicts.

## Profiles

An `NliProfile` (in `profiles.py`, not `entailment.py`: that would be an import cycle with `pipeline`) has these fields:

- `name`: `"default"` or `"accurate"`.
- `repo` and `revision`: the revision is the full SHA.
- `onnx_file(machine) -> str`.
- `k`.
- `thresholds`.
- `size_mb`: shown in the consent prompt.

| profile | repo @ revision | file | k | decide | medium | high |
|---|---|---|---|---|---|---|
| default | `cross-encoder/nli-deberta-v3-base` @ `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` | `pick_onnx_file` | 1 | 0.45 | 0.457948 | 0.99933 |
| accurate | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` @ `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` | `onnx/model_quantized.onnx` | 2 | 0.25 | 0.252721 | 0.999142 |

The default profile's values are today's values, unchanged. The accurate profile's cuts
come from SciFact train with rule-decided rows left out of the tier walk.

On SciFact dev these cuts give:

- `high`: 15 of 15 right (fewer than 20 verdicts, so too few to judge);
- `medium`: 114 of 151 right (0.755);
- `low`: empty.

`providers_for` treats a `quantized` file stem as int8 too, so CoreML is skipped for the
accurate file, as it is for the default one. Without this, a Mac would run the large
model on CoreML, which was measured 3× slower for int8 graphs.

## Tier note

`pipeline.tier_note(thresholds)` already returns `NO_HIGH_TIER` when `high >= 1.0`. It now
also returns `NO_LOW_TIER` when `medium - decide < 0.01` (the margin `eval_scifact` uses
for the same warning). The two notes are joined with `"; "`.

This changes the default profile's output too: its `medium` (0.457948) sits 0.008 above
`decide` (0.45). The 2026-09-12 tier doc already says this ("`low` is effectively
empty"), so the report now admits what the measurement found.

## Out of scope

- Making the large model the default.
- Re-calibrating the default profile.
- A multilingual model.
- Fixing AVeriTeC.
- A TUI flag for `/check`. The TUI uses the config key.

## Tests (no network)

- **Profiles and providers:** profile values; `providers_for` on a `quantized` stem.
- **Tier note:** each note alone, and both together.
- **Config:** the new section and key parse; an unknown value is refused; `set_value`
  round trip.
- **Model gate:** allow, deny, ask with TTY (all four answers), and ask without TTY.
  Always and never are persisted. A failed download falls back.
- **Engine:** the accurate profile resolves to its k and thresholds when installed. It
  falls back with the reason when consent is denied. `Engine.default` still never builds
  a model.
- **CLI:** `--accurate` reaches `Engine.default`.
- **TUI:** the `/config` rows exist, and the model prompt renders.
- **Report:** `k` and `nli_requested` appear in `models`. The golden JSON is updated.
