# Opt-in Accurate NLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship the bake-off winner as an opt-in `accurate` NLI profile. It is downloaded only with consent, falls back to the default profile with a reported reason, and the report says when it has no `low` tier.

**Spec:** `docs/superpowers/specs/2026-10-01-accurate-nli-design.md`. The exact values to use are in its profile table and its "Tier note" section.

## Global Constraints

- **Python and checks:** Python 3.10+ with type annotations on every signature. `uv run ruff check`, `uv run ruff format --check` and `uv run mypy` (strict, whole `src/`; CI now runs it) must be clean. `pathlib` throughout, and every read or write passes `encoding="utf-8"`.
- **No network in unit tests.** Model downloads and `hf_hub_download` are injected or monkeypatched.
- **Product rule 4:** with no TTY, an `ask` permission becomes `deny` and the denial is reported.
- **Product rule 5:** the 643 MB model is never downloaded without consent.
- **Product rule 6:** a fallback is reported, never silent.
- **`cli` and `tui` contain no logic.** Both call `verify()` / `Engine.default`.
- **Device order** stays CUDA → CoreML → CPU, and the int8 and `quantized` files skip CoreML.
- **The default profile's behaviour and values are unchanged.** The only exception is the new `NO_LOW_TIER` note.
- **Accurate profile values:** repo `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`, file `onnx/model_quantized.onnx`, k=2, decide 0.25, medium 0.252721, high 0.999142, size 643 MB.
- **Low-tier margin:** 0.01.
- **Git:** subagents never commit. The controller commits after a clean review.

---

### Task 1: Profiles, tier note, config keys

**Files:**
- `src/proofpath/entailment.py`
- `src/proofpath/pipeline.py`
- `src/proofpath/config.py`
- `src/proofpath/settings_hints.py`
- Tests: `tests/test_entailment.py`, `tests/test_pipeline.py`, `tests/test_config.py`, `tests/test_config_cli.py`

**Produces:**
- `entailment.NliProfile`: a frozen dataclass with fields `name: str`, `repo: str`, `revision: str`, `onnx_file: Callable[[str], str]`, `k: int`, `thresholds: pipeline.Thresholds`, `size_mb: int`.
  - `entailment` must not import `pipeline` at module level if that creates an import cycle. If it does, put `NliProfile` and the two profiles in a new `src/proofpath/profiles.py` and say so in the report.
- `DEFAULT_PROFILE`, `ACCURATE_PROFILE`, and `PROFILES: Mapping[str, NliProfile]`.
- `profile_installed(profile: NliProfile, cache_dir: Path, machine: str) -> bool`. It uses `huggingface_hub.try_to_load_from_cache` for `config.json`, `tokenizer.json` and the onnx file. It makes no network calls.
- `providers_for`: CoreML is also skipped when the stem contains `quantized`.
- `pipeline.NO_LOW_TIER`: one sentence in the style of `NO_HIGH_TIER`.
- `pipeline.LOW_TIER_MARGIN = 0.01`.
- `pipeline.tier_note`: returns both notes joined with `"; "` when both apply.
- `config.ModelsConfig`: one field, `nli: Literal["default", "accurate"] = "default"`. Register it in `_SECTIONS` and `_resolve_type`, and add it to `Config` as `models`.
- `Permissions.install_model: Permission = "ask"`.
- `settings_hints.MODEL_SETTING`: the `config set` hint for `permissions.install_model`, worded like `BROWSER_SETTING`.

**Tests:**
- Profile values match the spec table.
- `providers_for` skips CoreML for `onnx/model_quantized.onnx`.
- `tier_note` returns no-high alone, no-low alone, both joined, and neither.
- `profile_installed` returns True/False against a monkeypatched `try_to_load_from_cache`.
- Config:
  - `[models] nli = "accurate"` parses;
  - `"fast"` is refused with a `ConfigError`;
  - `install_model` parses;
  - `config set models.nli accurate` round-trips through `set_value` / `render_config`.
- Existing tests that assert the default `tier_note` text must be updated to include the low-tier note. List them in the report.

### Task 2: Model consent gate

**Files:**
- Create `src/proofpath/model_gate.py`
- Test: `tests/test_model_gate.py`

Model it on `browser.ConsentGate` (`src/proofpath/browser.py:243-341`) and reuse `config.resolve_permission`, `browser.Answer` / `PROMPT_CHOICES` / `TERMINAL_ANSWERS` / `TUI_ANSWERS` and `ask_terminal`.

**Produces:**
- `model_prompt_text(profile: NliProfile, *, answers: Sequence[str]) -> str`. It follows the same three-part layout as `browser.prompt_text`: what, why, the size in MB, and the answers.
- `ModelGate`. Constructor:
  - `permission: Permission`
  - `interactive: bool`
  - `override: bool | None = None` (True means `--accurate` given on the command line. It does **not** grant consent; it only selects the profile.)
  - `prompt: Callable[[str, int | None], Answer] | None = None`
  - `config_path: Path | None = None`
  - `download: Callable[[NliProfile], None]`
  - `installed: Callable[[NliProfile], bool]`

  Methods and state:
  - `ensure(profile) -> Decision` (`config.Decision`):
    - If installed, return allow with reason "installed".
    - Otherwise resolve the permission. With no TTY, `ask` becomes `deny` (reason as `resolve_permission` words it).
    - On `ask` with a TTY, prompt once. "always" and "never" are persisted to `permissions.install_model` through `config.set_value`; a persist failure is logged, not raised.
    - On allow, call `download(profile)`. An exception means `Decision("deny", "model download failed: <short reason>")`, and no raw response text goes into the reason.
  - `log: list[str]`: install and decision lines for the run.
- `download_profile(profile: NliProfile, cache_dir: Path, machine: str) -> None`: `hf_hub_download` of the three files.

**Tests:**
- installed → no prompt;
- allow → download called;
- deny → no download;
- ask without TTY → deny, with the reason;
- ask with TTY, each of the 4 answers;
- always and never persisted (to a tmp config path);
- download raising → deny with the reason, no traceback;
- the prompt text names the repo and 643 MB.

### Task 3: Engine, verify stage, report, CLI flag

**Files:**
- `src/proofpath/verify.py`
- `src/proofpath/cli.py`
- `src/proofpath/report.py` (only if needed)
- Tests: `tests/test_verify.py`, `tests/test_check_cli.py`, `tests/test_report.py`, `tests/data/verify-report-golden.json`

**Engine changes:**
- `Engine.default(config, *, …, nli: str | None = None)`. `nli` overrides `config.models.nli`; the CLI passes `"accurate"` for `--accurate`. Explicit `k` and `thresholds` arguments still win, for tests and eval.
- The Engine records `nli_requested` and a `ModelGate`. It builds the gate with `permission=config.permissions.install_model`, the same `interactive` and `prompt` as the browser gate, `installed=profile_installed`, and `download=download_profile`.
- **`Engine.default` must stay model-free and network-free.** Keep `test_engine_default_never_builds_a_model` green.
- Resolve the profile lazily, once, at the start of the verify stage, before `LOADING_MODELS`, through a method such as `engine.resolve_nli() -> NliProfile`:
  - default requested → default profile;
  - accurate requested → `gate.ensure(ACCURATE_PROFILE)`. Allow uses the accurate profile. Deny or failure falls back to `DEFAULT_PROFILE` and records the reason.
- The scorer factory builds `OnnxNli` from the resolved profile, passing `repo_id`, `revision` and `onnx_file(platform.machine())`.
- `engine.k` and `engine.thresholds` must be the resolved profile's values at the point where `cache_mod.model_id` is computed (`verify.py:~1293`) and wherever `pipeline.decide_indexed` gets `k` and `thresholds`. Pick the least invasive shape (resolved fields set once, or properties) and explain it in the report.
- On fallback:
  - emit a Note event: "accurate NLI model not used: <reason> — using the default model. To allow the download: <MODEL_SETTING>";
  - forward the gate's log lines as Notes, the way `install_log` is forwarded for the browser.

**Report:**
- `_models()` gains `"k": str(k)`.
- On fallback it also gains `"nli_requested": "accurate (<reason>)"`.
- `tier_note` uses the resolved thresholds.
- Update the golden JSON for `k` and the new low-tier note.

**CLI:**
- `check --accurate` (help: "use the larger accurate NLI model for this run (643 MB, downloaded once with consent)").
- It is passed as `nli="accurate"`. Without the flag it passes `None`, so config decides.
- The CLI prints fallback notes on stderr through the existing Note path.

**Tests:**
- installed accurate → k=2 and accurate thresholds reach `model_id` and decide;
- deny → default profile, the reason in `models["nli_requested"]` and a Note;
- `--accurate` reaches `Engine.default` as `nli="accurate"` (the pattern in `test_check_cli.py:592+`);
- `Engine.default` builds no model;
- the cache key differs between the two profiles.

### Task 4: TUI

**Files:**
- `src/proofpath/tui/widgets/config_panel.py`
- `src/proofpath/tui/app.py`
- `src/proofpath/tui/widgets/prompt.py` (only if the model prompt needs it)
- Tests: `tests/test_tui_app.py`, `tests/test_tui_rich.py`

**Changes:**
- `/config` panel:
  - a `models` section with a choice row `models.nli` (values `default` / `accurate`), its meaning text naming the size and the SciFact gain;
  - a `permissions.install_model` row next to `install_browser`, built the same way.
- The TUI's engine already passes `prompt=self._prompt_for(run.id)` to `Engine.default`. Make sure the model gate's prompt reaches that same `PermissionPrompt` path, with `model_prompt_text(..., answers=TUI_ANSWERS)`.
  - Generalise `PermissionPrompt` to take ready-made text if it currently only builds browser text.
  - The `/allow` verb must answer a model prompt too.

**Tests:**
- the rows exist with the right values;
- a model prompt renders and an answer settles it;
- the quit path answers "no".

### Task 5: Docs and site

**Files:**
- `README.md`
- `docs/superpowers/specs/2026-09-10-proofpath-design.md` (§7.1 permission table, §12 NLI row, §14 calibration)
- `CHANGELOG.md` (`[Unreleased]` → Added / Changed)
- `docs/superpowers/OPEN-ITEMS.md` (§20.11: (a) and (b) settled, (c) open)
- `site/src/data/measured.ts` (mirror the README Measured rows)

**README:**
- A short "Accurate model (opt-in)" section with:
  - what it is;
  - its size and speed;
  - the SciFact numbers: 0.697 against 0.580 macro-F1;
  - the AVeriTeC caveat: not better on news; recommended for scientific sources;
  - how to turn it on;
  - the consent behaviour.
- The Measured table gains a row for the accurate profile.
- The Speed section mentions the 643 MB download.

**Site:** if `site/` data changed, run the site build per the repo's site README and report the result. Keep wording consistent with the README.
