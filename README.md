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

## Try it today: does a citation exist?

```bash
proofpath resolve "Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. Nature. 2021;596:583-589."
```

Prints `RESOLVED`, `RESOLVED (low confidence)`, `AMBIGUOUS` (candidates listed) or
`GHOST REFERENCE`, with the record it matched, how each field agreed, and whether
the paper was retracted. Crossref and Semantic Scholar are asked first; arXiv and
OpenAlex only before a ghost call. Exit code `1` when something is wrong with the
reference, `2` when a provider was unreachable — never a ghost call on missing
evidence.

## Looking inside the cache

Everything proofpath fetches, embeds and decides lands in one plain SQLite file:

```bash
proofpath cache            # where it is and what it holds
proofpath cache ls         # sources, chunk/verdict counts, text expiry
proofpath cache show <id>  # one source's chunks and verdicts
proofpath cache clear --expired
```

Open `proofpath cache path` in [DB Browser for SQLite](https://sqlitebrowser.org/),
TablePlus or DBeaver — plain tables, no extension needed. Raw publisher text
expires after 7 days; verdicts keep the passage they quote.

## Optional LLM judge

Everything above runs locally. An LLM is used only at the very end, as an opt-in
second opinion on low-confidence verdicts and for a plain-language summary — it
never sees a source document and cannot change a verdict. Default provider is Groq
(free, no card, no training on your data); Gemini and a local Ollama are the
alternatives.

```bash
cp .env.example .env             # then paste GROQ_API_KEY=gsk_... into .env
proofpath judge check            # one tiny request to prove the key works
proofpath judge set provider gemini
```

The key is read from the environment or a `.env` file, never stored in config, and
never printed.

## Status

🚧 Core measured, product not assembled yet. On SciFact dev the local retrieval +
entailment core scores 0.606 accuracy against a 0.406 trivial baseline, with no
verdict ever emitted without its passage
([results](docs/eval/2026-09-11-scifact-dev.md)). Document ingest, reference
resolution and fetching are the next phases.

- [Design specification](docs/superpowers/specs/2026-09-10-proofpath-design.md) — what it does and the measurements behind each decision
- [Implementation plan](docs/superpowers/plans/2026-09-10-proofpath-implementation-plan.md) — phases, ordered by risk retired
- [Open items](docs/superpowers/OPEN-ITEMS.md) — what is unresolved, and what has not been verified yet

## License

MIT
