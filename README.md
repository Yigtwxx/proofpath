# proofpath

> Don't guess. Show the evidence.

`proofpath` checks whether the sources behind a claim actually say what the claim
says. Point it at a paper, a draft, or a link — it verifies every citation on three
levels and shows you the passage behind each verdict.

1. **Does the source exist?** — resolved against Crossref, OpenAlex, PubMed, arXiv.
2. **Is it still valid?** — checked against Retraction Watch.
3. **Does it support the claim?** — retrieval + entailment against the source text.

Most tools stop at step 1. Step 3 is the point.

Runs offline and free by default: no API key, no Docker, no server. Windows, Linux
and macOS.

## What it will not do

It does not pretend to be certain. Full text is openly available for well under half
of published citations, so many verdicts will honestly be *not enough information*,
and every report ends with its own coverage figures. A tool that sounds sure about
everything is the problem this one exists to fight.

```bash
uv tool install proofpath

proofpath                        # interactive TUI
proofpath check paper.pdf        # one-shot report
proofpath check draft.md --format sarif
```

## Status

🚧 Design stage — no working code yet.

- [Design specification](docs/superpowers/specs/2026-09-10-proofpath-design.md) — what it does and the measurements behind each decision
- [Implementation plan](docs/superpowers/plans/2026-09-10-proofpath-implementation-plan.md) — phases, ordered by risk retired

## License

MIT
