# Contributing

Thanks for considering a contribution.

## Before you start

Read `docs/superpowers/specs/2026-09-10-proofpath-design.md`. It records the design
decisions and, more importantly, the measurements behind them. Several obvious-looking
approaches were tested and rejected — the spec says which and why.

For anything beyond a small fix, open an issue first so we can agree on the approach.

## Setup

```bash
uv sync --all-extras
uv run pytest
uv run ruff check
uv run ruff format --check
```

## Standards

- Type annotations on every function signature.
- `ruff check` and `ruff format --check` clean.
- Tests pass on Linux, macOS and Windows.
- No network access in unit tests — use fixtures.
- New reported states go in the table in spec §15, not just in code.

## The rules that matter most

A pull request that makes the tool sound more confident than the evidence supports
will be rejected, however good the code is. See the non-negotiable rules in
`CLAUDE.md`.
