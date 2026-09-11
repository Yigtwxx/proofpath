# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Config and permissions module: `config.toml` under the platform config dir,
  `proofpath permissions` / `proofpath permissions set`, and the rule that an `ask`
  permission without a TTY resolves to `deny` and is reported (spec §7.1).
- Core types (`Verdict` cannot be `SUPPORTED`/`REFUTED` without a passage), device
  selection (CUDA → CoreML → CPU), sentence retrieval over `fastembed` + `sqlite-vec`,
  ONNX NLI entailment on `cross-encoder/nli-deberta-v3-base`, and the aggregation
  pipeline.
- Persistent cache: one plain SQLite file (`sources`, `raw_text` with 7-day TTL,
  `chunks` with float32 embeddings, `verdicts`), a schema `CHECK` that refuses an
  asserted verdict without a passage, and `proofpath cache` / `cache path` / `ls` /
  `show` / `clear [--expired]`.
- Numeric claim layer (spec §10): percentages, factors and unit counts with
  direction are compared before NLI; an unambiguous contradiction is refuted by
  rule with both figures named (`Verdict.reason`). Conservative by design: one
  comparable figure on each side, change never against level.
- Judge settings (`[judge]` in config, Groq default) with `proofpath judge`,
  `judge check` and `judge set`; API key resolved from the environment or `.env`,
  never stored or printed. `.env.example` added.
- SciFact loader pinned to the AI2 tarball by sha256, evaluation metrics, and
  `scripts/eval_scifact.py`. First measured result: dev accuracy 0.606 vs 0.406
  trivial baseline (`docs/eval/2026-09-11-scifact-dev.md`).

### Changed
- Retrieval no longer depends on `sqlite-vec`: a numpy cosine scan is faster at
  every measured scale and the plain SQLite file opens in any GUI.
- Spec: `PARAGRAPH-SCOPED` and `UNSUPPORTED CITATION STYLE` states, three-tier
  confidence display, 7-day raw-text cache TTL, v0.1 limited to numeric citation
  markers.

## [0.0.1] - 2026-09-10

First release. The verification pipeline is not implemented; this reserves the name
and establishes the interface, packaging and CI that later phases build on.

### Added
- Design specification with measured source-access data (spec §6), the fetch ladder
  and its permission model (§7), and corrected reference resolution (§8).
- Phased implementation plan, ordered by risk retired rather than user-visible
  progress.
- `proofpath` command. A bare invocation is a first-class entry point rather than a
  help screen, which is where the TUI will attach. Exit codes are fixed: `0` clean,
  `1` findings, `2` the run itself failed.
- Cross-platform CI on Linux, macOS and Windows, and PyPI publishing through trusted
  publishing rather than a stored API token.
