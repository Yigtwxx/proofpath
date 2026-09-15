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
proofpath
```

```
    ╭╮ ╭╮
   ╭╯╰─╯╰────────────────────────────────────────────────────────────────────╮
  ╸┤ o o                                                                     ╰~~~~~~~~~~~~~[PROOF]
   ╰─┬─┬────────────────────────────────────────────────────────────────┬─┬──╯
     ˘ ˘                                                                ˘ ˘
      proofpath v0.2.1                                                  academic . online . coreml
      paste a file path, a URL, or a claim.                                  /help  /config  /quit
```

Bare `proofpath` opens the terminal UI ([a recorded session in both themes](docs/eval/2026-09-15-tui-v2-live.md),
with [SVG screenshots](docs/eval/tui-v2-rich.svg)). Each run is one panel in its own
accent: the command on the top border and the run's state at its right, a fixed-column
stage table underneath (`✓` finished, `⏺` still running or finished with something
unverified, a real progress bar on the active stage, the provider that produced each
number at the right), a rule, then the findings — location, reference, the state word
as a badge, the tier — with the finding's notes and the claim (`you`) and the passage
(`source`) it was checked against under it. The bottom border carries the run's
coverage; the docked footer draws it as a bar and never scrolls away.

Paste a path and it runs; every one-shot verb is a slash command (`/check`, `/resolve`,
`/fetch`, `/config`, `/cache`), runs can be started while others are in flight and
stopped with `/cancel #n` — a stopped run keeps what it had decided. Click (or press
`enter` on) a finding to read the whole quoted passage; `⧉` copies it; a finding's
reference is a link to its source. When a publisher blocks the plain fetch, the
permission question is asked **inline, under the stage that hit the wall**, with
`[allow once] [always] [no] [never]`.

**Windows / `NO_COLOR`.** The look above is the `rich` theme, chosen when the terminal
gives evidence of truecolor (`COLORTERM`, Windows Terminal, iTerm2, kitty, WezTerm,
Ghostty, VS Code, Terminal.app). Under `NO_COLOR`, `--no-color`, `-q`, `TERM=dumb`,
legacy conhost or a CJK locale the `plain` theme draws the same runs as flat ASCII rows
in the terminal's own 16 colours — the state words, the coverage and every honesty
sentence are identical, only the drawing changes. `PROOFPATH_THEME=rich|plain` forces
either, for screenshots and bug reports.

```
   ,_,
  (o.o)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[PROOF]
   " "    proofpath v0.2.1                                              academic . online . coreml
          paste a file path, a URL, or a claim.            /help  /config  /quit
```

The same engine behind a pipe or in CI:

```bash
proofpath check paper.pdf                       # report → report.md, exit 0/1/2
proofpath check draft.md --format json | jq '.coverage'
proofpath check draft.md --format sarif --out draft.sarif   # any SARIF 2.1.0 viewer (VS Code's SARIF Viewer, …)
proofpath -q check - < draft.md                 # stdin; findings and coverage only
```

`--format sarif` writes a SARIF 2.1.0 log: one result per finding on its line, the
quoted passage in the message, the exact honesty state in `properties`, and the run's
coverage in the run's properties, so a log opened without the terminal still says how
much was read. Exit codes are the interface: `0` clean, `1` findings (every
`UNVERIFIED` and `LOW CONFIDENCE` counts), `2` the run itself failed — no text
parsing needed to gate a job. Piped or in CI there is **no prompt**: an `ask`
permission is treated as `deny` and reported.

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
`coverage is weak: unread sources may hold more, so this is a lower bound`. When the
browser step was not permitted, a line counts the sources it cost:
`skipped N source(s) because the browser was not permitted`.

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

## Citations it reads

Numeric markers — `[12]`, `[12,15]`, `[12-15]`, Nature-style superscripts — and, since
v0.2, **author-year**: `(Smith et al., 2020)`, `Smith (2020)`, `(Smith, 2020; Jones,
2019)`, `2020a`/`2020b` collisions, `ibid.` and `op. cit.`, and a mixed
`(Smith, 2020; [12])`. Pairing rate on the hand-built author-year set: **0.940**
(83 expectations over 55 passages, [details](docs/eval/2026-09-12-pairing-author-year.md));
the numeric set is at 108 of 109. Still unpaired, and reported rather than guessed:
a surname that is not the entry's *first* author, a year off by one, two surnames
sharing a last word (`Berg` / `van der Berg`), `Smith 2020` with no comma, and an
initialism such as `(WHO, 2020)` against "World Health Organization". A marker no
entry matches is listed as an unresolved marker.

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
| `UNVERIFIED (provider unavailable)` | API down, rate limited after backoff, or answering with a page instead of a record |
| `UNVERIFIED (not in bibliographic indexes)` | web page, blog, report — indexes do not cover it, so absence proves nothing |
| `AMBIGUOUS` | several plausible records, all listed |
| `NEI` | the source was read and neither supports nor contradicts |
| `PARAGRAPH-SCOPED` | the citation covers a paragraph; each sentence is judged separately |
| `UNSUPPORTED CITATION STYLE` | reserved; no detected style produces it in v0.2 |

An earlier build sometimes aborted with `134` after printing a complete report (ONNX
runtime teardown); fixed in v0.1 — 20 of 20 piped runs exit `1`
([live runs](docs/eval/2026-09-12-v0.1-live.md)).

## Known limitations

- **Superscript citations only when the PDF marks them as superscript**; a PDF that
  draws them as ordinary digits loses them, and `km²` can be read as `[2]`.
- **Abstract fallback**: when only an abstract is reachable the verdict is labelled
  `LOW CONFIDENCE (abstract only)`. Three sentences is not a source.
- **An unnumbered two-column bibliography is cut at line breaks** (RoBERTa's ACL
  list: 103 entries for ~50, half of them with no author), so author-year pairing over
  such a list resolves few items — 15 claims where a rejoined list would give 65. The
  largest open item for author-year pairing.
- **A reference list without a `References` heading** is read by its shape in paged
  formats — the last contiguous run of numbered paragraphs — which recovers only that
  last block (AlphaFold: 17 of 84 entries) and drops the prose printed before a
  block's first entry.
- **The 0 % false-ghost rate is measured on 274 hand-built references**, and rule 3
  is paid for in recall: a fabricated `First Last and First Last. Title. Venue.` with
  no year between the names and the title is reported as `not in bibliographic
  indexes`, not as a ghost, and a proceedings-*volume* record can accept a fabricated
  paper cited into that volume as `RESOLVED (low confidence)` — the set's one
  fabricated acceptance (0.9 %) ([details](docs/eval/2026-09-12-ghosts.md)).
- **A cold run of a long bibliography is still serial** through the providers (129
  references: minutes). Only the TUI runs several documents' network stages at once.
- **A refused or failed browser install is reported, not hidden**: the source is
  `UNVERIFIED (blocked, browser not permitted)` with the install log, never silently
  counted as unreachable.
- **Coverage is not perfectly reproducible**: two runs minutes apart can read a
  different number of sources, depending on which providers answered.
- The LLM judge and `--summarize` arrive in **v0.3**.

## Speed

Apple Silicon Mac, models already downloaded:

| document | first run | cached re-run |
|---|---|---|
| 1-page markdown draft, 7 references | 79.3 s (v0.1 cold) | **1.35 s** ([v0.2](docs/eval/2026-09-15-v0.2-live.md)) — 15.8 s in v0.1 |
| 19-page arXiv PDF, 68 references | 12 m 57 s | 3 m 42 s ([v0.1](docs/eval/2026-09-12-v0.1-live.md)) |

The **first ever** run also downloads about 250 MB of ONNX models. Since v0.2 a
cached re-run asks the network for nothing: reference resolution and the retraction
check are cached (resolutions 30 days; a retraction hit 30 days, a miss 7 days), the
fetched text for 7 days, and chunks and verdicts for as long as the text is unchanged. The models are still loaded, and any source whose text
has expired or was never read is fetched again — the PDF re-run above (v0.1) still
spent 22 s fetching and re-scored 3 of 102 claims.

## Looking inside the cache

Everything proofpath fetches, embeds and decides lands in one plain SQLite file:

```bash
proofpath cache            # where it is, what it holds, how many lookups it remembers
proofpath cache ls         # sources, chunk/verdict counts, text expiry
proofpath cache show <id>  # one source's chunks and verdicts
proofpath cache clear --expired
```

Open `proofpath cache path` in [DB Browser for SQLite](https://sqlitebrowser.org/),
TablePlus or DBeaver — plain tables, no extension. Raw publisher text expires after
7 days; verdicts keep the passage they quote; a provider outage is never stored.
`proofpath resolve REF` and `proofpath fetch URL|DOI` run either half on its own.

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
| Reference resolution | hand-built ghost set, 274 references | 0.0 % false-ghost, 99.1 % ghost recall ([details](docs/eval/2026-09-12-ghosts.md)) |
| Source access | 50 DOIs | 72 % full text, 18 % abstract only, 10 % nothing ([details](docs/eval/2026-09-11-coverage.md)) — a real biomedical paper in the live runs reached 33 % full text |
| Citation pairing, numeric | 61 hand-built passages | 0.99 ([details](docs/eval/2026-09-11-pairing.md)) |
| Citation pairing, author-year | 55 hand-built passages, 83 expectations | 0.940 ([details](docs/eval/2026-09-12-pairing-author-year.md)) |

Published SciFact results sit around 70–75 F1, not 95. Nothing is tuned on a test
split, and no number is quoted without the run that produced it.

- [Design specification](docs/superpowers/specs/2026-09-10-proofpath-design.md) — what it does and the measurements behind each decision
- [Open items](docs/superpowers/OPEN-ITEMS.md) — what is unresolved, and what has not been verified yet · [Changelog](CHANGELOG.md)

## License

MIT
