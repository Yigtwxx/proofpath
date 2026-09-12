# proofpath

> Don't guess. Show the evidence.

`proofpath` checks whether the sources behind a claim actually say what the claim
says. Point it at a paper or a draft and it verifies every citation on three levels,
then shows you the passage behind each verdict.

1. **Does the source exist?** — Crossref, Semantic Scholar, arXiv, Open Library, OpenAlex.
2. **Is it still valid?** — Crossref's Retraction Watch data, and OpenAlex.
3. **Does it support the claim?** — retrieval and entailment against the source text,
   with a numeric rule that runs before the model.

Most tools stop at step 1. Step 3 is the point.

Runs locally and free: no API key, no Docker, no server. Windows, Linux, macOS.

```bash
uv tool install proofpath

proofpath check paper.pdf          # one-shot report → report.md, exit 0/1/2
proofpath check draft.md --format json | jq '.coverage'
proofpath -q check - < draft.md    # stdin; findings and coverage only
```

The install carries **no browser engine**. When a publisher blocks the plain fetch,
proofpath asks once before downloading one — about 280 MB, into its own environment
and the shared browser cache, never system-wide — and remembers the answer.
`proofpath config set permissions.install_browser never` stops it asking at all.

## Honesty

**Most verdicts are "not enough information", and that is the honest answer.** On
SciFact dev only **135 of 340** claim–source pairs score above the decision threshold
at all; the other 60 % are `NEI` rather than guessed. Full text is openly reachable
for well under half of published citations, so a real document loses more on top.

**Every report states its own coverage** (and adds a warning line when it is thin):

```
7 refs: 1 ghost, 2 unsupported, 4 ok
fulltext   72%
abstract   14%
unverified 14%
```

When a quarter or more of the sources could not be read, a further line says so:
`coverage is weak: unread sources may hold more, so this is a lower bound`.

**No verdict without its passage.** `SUPPORTED` and `REFUTED` cannot exist without
the quoted sentence they rest on — the cache schema itself refuses to store one.

**The confidence tiers are measured, not chosen** — read off a sweep on SciFact dev
([details](docs/eval/2026-09-12-tiers.md)): `decide = 0.45`, `medium = 0.457948`,
`high = 0.99933`. Two caveats belong next to those numbers. `high` was fitted on the
same split it is reported on: an in-sample point estimate over **21 verdicts** (at
least 18 correct — roughly 0.65–0.95 at 95 % confidence), so read it as "the model
was near-certain here", not as a guarantee of 85 % precision. And `medium` lands
almost exactly on `decide`, so `low` is practically empty among asserted verdicts —
the display is effectively **two tiers**: near-certain, and asserted at all.

## What the states mean

Absence of evidence is never reported as evidence of absence. Each of these is a
distinct, printed state (spec §15), never collapsed into a verdict:

| State | Cause |
|---|---|
| `LOW CONFIDENCE (abstract only)` | full text unavailable, abstract used |
| `UNVERIFIED (blocked)` | 403 or bot protection |
| `UNVERIFIED (blocked, robots.txt)` | the site's `robots.txt` disallows the fetch |
| `UNVERIFIED (blocked, browser not permitted)` | steps 1–2 blocked and the browser consent was denied or impossible |
| `UNVERIFIED (unreachable)` | dead link, Wayback miss |
| `UNVERIFIED (reached, no text extracted)` | 200 answered, nothing readable came back |
| `UNVERIFIED (network not permitted)` | `permissions.network = deny` |
| `UNVERIFIED (provider unavailable)` | API down or rate limited after backoff |
| `UNVERIFIED (not in bibliographic indexes)` | web page, blog, report — indexes do not cover it, so absence proves nothing |
| `AMBIGUOUS` | several plausible records, all listed |
| `NEI` | the source was read and neither supports nor contradicts |
| `PARAGRAPH-SCOPED` | the citation covers a paragraph; each sentence is judged separately |
| `UNSUPPORTED CITATION STYLE` | an author–year marker; v0.1 pairs numeric markers only |

Exit codes: `0` clean, `1` findings (every `UNVERIFIED` and `LOW CONFIDENCE` counts),
`2` the run itself failed — no text parsing needed to gate a CI job. An earlier build
sometimes aborted with `134` after printing a complete report (ONNX runtime
teardown); fixed in this release — 20 of 20 piped runs exit `1` ([live runs](docs/eval/2026-09-12-v0.1-live.md)).

## What v0.1 cannot do yet

- **Numeric citation markers only** — `[12]`, `[12,15]`, `[12-15]`; an author–year
  citation is listed as `UNSUPPORTED CITATION STYLE`, not judged.
- **Superscript citations only when the PDF marks them as superscript**; a PDF that
  draws them as ordinary digits loses them, and `km²` can be read as `[2]`.
- **Abstract fallback**: when only an abstract is reachable the verdict is labelled
  `LOW CONFIDENCE (abstract only)`. Three sentences is not a source.
- **A reference list without a `References` heading is not found at all**, so its
  markers are reported as uncheckable rather than checked.
- **Resolution and the retraction check are not cached**, so even a warm re-run goes
  to the network for them.
- **The 0 % false-ghost rate is a property of the hand set's citation style.** Live
  runs hit styles that set does not contain, and misjudged real references because of
  it ([the live runs](docs/eval/2026-09-12-v0.1-live.md)).
- **A refused or failed browser install is reported, not hidden**: the source is
  `UNVERIFIED (blocked, browser not permitted)` with the install log, never silently
  counted as unreachable.
- **Coverage is not perfectly reproducible**: two runs minutes apart can read a
  different number of sources, depending on which providers answered.
- The TUI and `--format sarif` arrive in **v0.2**; the LLM judge and `--summarize`
  in **v0.3**.

## Speed

Apple Silicon Mac, models already downloaded ([live runs](docs/eval/2026-09-12-v0.1-live.md)):

| document | first run | cached re-run |
|---|---|---|
| 1-page markdown draft, 7 references | 79.3 s | 15.8 s * |
| 19-page arXiv PDF, 68 references | 12 m 57 s | 3 m 42 s |

 * the draft's cached re-run followed the earlier of the two recorded cold runs; the live doc keeps both.

The **first ever** run also downloads about 250 MB of ONNX models. A cached re-run
reads its chunks and verdicts back from the cache instead of recomputing them, but it
is not a no-op: both models are still loaded, reference resolution and the retraction
check still query the network, and any source whose text has expired or was never
read is fetched again — the PDF re-run above still spent 22 s fetching (it went out to
Wayback and arXiv) and re-scored 3 of 102 claims.

## Looking inside the cache

Everything proofpath fetches, embeds and decides lands in one plain SQLite file:

```bash
proofpath cache            # where it is and what it holds
proofpath cache ls         # sources, chunk/verdict counts, text expiry
proofpath cache show <id>  # one source's chunks and verdicts
proofpath cache clear --expired
```

Open `proofpath cache path` in [DB Browser for SQLite](https://sqlitebrowser.org/),
TablePlus or DBeaver — plain tables, no extension. Raw publisher text expires after
7 days; verdicts keep the passage they quote. `proofpath resolve REF` and `proofpath
fetch URL|DOI` run either half on its own.

## Optional LLM judge — arrives in v0.3

Everything above runs locally. An LLM is used only at the end, as an opt-in second
opinion on low-confidence verdicts — it never sees a source document and cannot
change a verdict. The settings exist today (`proofpath config check` proves a key
works, default Groq); `check --judge` and `--summarize` are wired in v0.3. The key
comes from the environment or a `.env` file, never from config, and is never printed.

## Measured

| What | Set | Result |
|---|---|---|
| Retrieval + entailment | SciFact dev, 340 pairs | 0.609 accuracy, 0.597 macro-F1, against a 0.406 trivial baseline ([details](docs/eval/2026-09-12-scifact-dev.md)) |
| Reference resolution | hand-built ghost set, 258 references | 0 % false-ghost, 100 % ghost recall ([details](docs/eval/2026-09-12-ghosts.md)) — but see the live-run caveat above |
| Source access | 50 DOIs | 72 % full text, 18 % abstract only, 10 % nothing ([details](docs/eval/2026-09-11-coverage.md)) — a real biomedical paper in the live runs reached 33 % full text |
| Citation pairing | 61 hand-built passages | 0.98 ([details](docs/eval/2026-09-11-pairing.md)) |

Published SciFact results sit around 70–75 F1, not 95. Nothing is tuned on a test
split, and no number is quoted without the run that produced it.

- [Design specification](docs/superpowers/specs/2026-09-10-proofpath-design.md) — what it does and the measurements behind each decision
- [Open items](docs/superpowers/OPEN-ITEMS.md) — what is unresolved, and what has not been verified yet · [Changelog](CHANGELOG.md)

## License

MIT
