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
██████╗                        █████╗██████╗               ██╗
██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║
██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗
██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗
██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║
╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝
proofpath v0.4.6                                                        academic · online · coreml
paste a file path, a URL, or a claim.                                        /help  /config  /quit
──────────────────────────────────────────────────────────────────────────────────────────────────
```

Both themes open with the wordmark: `rich` in block glyphs through five crimson bands,
`plain` in ASCII. Under it, or beside it on a wide terminal, the version and the hint; a
rule closes the banner.

Bare `proofpath` opens the terminal UI ([a recorded session in both themes](docs/eval/2026-09-15-tui-v2-live.md),
with SVG screenshots of a real run on v0.4.1, before the wordmark, in [`rich`](docs/eval/tui-raven-rich.svg) and
[`plain`](docs/eval/tui-raven-plain.svg)). Each run is one panel in its own
accent: the command on the top border and the run's state at its right, a fixed-column
stage table underneath (`✓` finished, `⏺` still running or finished with something
unverified, a real progress bar on the active stage, the provider that produced each
number at the right), a rule, then the findings — location, reference, the state word
as a badge, the tier — with the finding's notes and the claim (`you`) and the passage
(`source`) it was checked against under it. The bottom border carries the run's
coverage; the docked footer draws it as a bar and never scrolls away.

Paste a path, a URL or a claim and it runs — a post of several lines pastes whole: the
bar holds it behind a one-line summary, `Enter` checks it, `Esc` drops it. Every
one-shot verb is a slash command (`/check`, `/resolve`, `/fetch`, `/config`, `/cache`),
runs can be started while others are in flight and stopped with `/cancel #n` — a
stopped run keeps what it had decided. Type `/` and every command is listed above the
bar with what it wants; `Tab` completes, `↑`/`↓` pick. `/config` opens the settings as
rows you change with the arrow keys. Click (or press `enter` on) a finding to read the
whole quoted passage; `⧉` copies it; a finding's reference is a link to its source.
When a publisher blocks the plain fetch, the
permission question is asked **inline, under the stage that hit the wall**, with
`[allow once] [always] [no] [never]` — or, from the bar, `/allow once`, `/allow always`,
`/allow no`, `/allow never`.

**Windows / `NO_COLOR`.** The look above is the `rich` theme, chosen when the terminal
gives evidence of truecolor (`COLORTERM`, Windows Terminal, iTerm2, kitty, WezTerm,
Ghostty, VS Code, Terminal.app). Under `NO_COLOR`, `--no-color`, `-q`, `TERM=dumb`,
legacy conhost or a CJK locale the `plain` theme draws the same runs as flat ASCII rows
in the terminal's own 16 colours — the state words, the coverage and every honesty
sentence are identical, only the drawing changes. `PROOFPATH_THEME=rich|plain` forces
either, for screenshots and bug reports.

```
######+                        #####+######+               ##+
##+--##+                      ##+---+##+--##+         ##+  ##|
######++##+##+ #####+  #####+ #####+ ######++ #####+ #####+######+
##+---+ ####++##+--##+##+--##+##+--+ ##+---+ ##+--##++##+-+##+-##+
##|     ##+-+ +#####+++#####++##|    ##|     #######| ##|  ##| ##|
+-+     +-+    +----+  +----+ +-+    +-+     +-+  +-+ +-+  +-+ +-+
proofpath v0.4.6                                                        academic . online . coreml
paste a file path, a URL, or a claim.                                        /help  /config  /quit
--------------------------------------------------------------------------------------------------
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
`skipped N source(s) because the browser was not permitted — proofpath config set
permissions.install_browser ask`.

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
| `UNVERIFIED (credentials missing)` | the platform reads only with a credential this machine has none of — Reddit's free app, `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`. Nobody was asked, so it is not "unreachable"; the coverage block names both variables |
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
- **The judge is a second opinion, not a second verdict.** `--judge` asks the model only
  about the low-tier verdicts (1 of 10 on the live draft), and its answer is printed
  beside the local verdict, never in place of it. `--summarize` is one extra call over
  the finished report, labelled model-written; if the provider does not answer, the
  markdown report, the JSON and the terminal say so (`judge status` / `summary status`
  in the header). The SARIF log does not carry it — it is a findings document.

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

## Posts and the links inside them (v0.4)

A social post is not a source. What proofpath checks is whether the **links inside it**
back what it says:

```bash
proofpath check --url https://bsky.app/profile/bsky.app/post/3movpwtbjgs2d
proofpath check --url https://news.ycombinator.com/item?id=8863
proofpath check -                 # paste the text of a post that cannot be read
```

Any other address is a **page**, and a page is read as the document itself:

```bash
proofpath check --url https://en.wikipedia.org/wiki/AlphaFold
```

The page is fetched up the same ladder its sources are — same permissions, same
consent prompt for the browser step — and cut into paragraphs. A page that prints a
reference list (a journal's article page) is paired by citation number like a paper; any
other page cites by linking, like a post: each paragraph's sentences are checked against
the pages that paragraph links to. A page the ladder could not read ends the run with the
ladder's own words (`UNVERIFIED (blocked)`, `UNVERIFIED (unreachable)`, …), never with a
verdict. A profile, a subreddit or a platform's front page is still "a page, not a post":
there is no one post there to check. To check one claim *about* a page rather than the
page's own words, paste the claim as text with the address inside it.

Every sentence of the post is checked against the pages its links point to, and the post's
own words are never allowed to stand as their own evidence.

| Platform | How it is read |
|---|---|
| Bluesky | `public.api.bsky.app`, no account, first-class. Link cards, rich-text links and one level of quoted post |
| Hacker News | the official Firebase API, no account, first-class. A story's URL and the links in a comment |
| Reddit | with a **free app you register yourself**: put `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in `.env`. Without them the run says `UNVERIFIED (credentials missing)` and names both variables — it never quietly skips the post |
| Mastodon | best effort, per instance. Many instances now require a login for the public API, and that answer is reported as `UNVERIFIED (blocked)`, not as a missing post |
| X / Twitter | cannot be read at all. `check --url` says so and asks you to paste the text; the links inside it are then verified normally |

The Reddit path is built against Reddit's documented shapes and covered by fixtures, but it
has never run against Reddit on this machine — nobody here has an app to register. Bluesky,
Hacker News and Mastodon were each read live before release.

## Optional LLM judge (v0.3)

Everything above runs locally, and the default run makes **zero** LLM calls. Two flags
add an LLM at the end, and only there:

```bash
proofpath check paper.pdf --judge        # a second opinion on the low-confidence verdicts
proofpath check paper.pdf --summarize    # one model-written paragraph over the finished report
proofpath config check                   # proves the key works before you spend a run on it
```

**What `--judge` does.** After the local verdicts are final, the verdicts the models
were least sure about — the `low` tier, never a numeric mismatch and never a claim
without a quoted passage — go to the model in batches of up to 20 (about 7k tokens),
each with its claim and the passage it was checked against. The model answers from the
passage alone, and its label and one-sentence rationale are printed **beside** the local
verdict: `= judge (groq openai/gpt-oss-120b): NEI — …`. The local verdict, the finding
kind and the report's states never change. Opinions are cached with the verdict, so a
re-run asks nothing.

**What it cannot do.** It never sees a source document, so it cannot introduce a claim
or an evidence passage of its own; it cannot turn `NEI` into `SUPPORTED`; it cannot
hide a source that could not be read. If the provider is down, rate-limited or the key
is wrong, the run finishes on the local verdicts and says so in the stage line, the
report header (`judge status:`) and the JSON — `-q` cannot hide it.

**What `--summarize` does.** One final call turns the finished markdown report into 3–5
plain sentences a reader can act on. It runs after the report is complete, its only
input is that report, it is off by default in the CLI and the TUI (`/summarize` there),
and the output is labelled `(model-written, <provider> <model>)`. `--summarize` alone
is exactly one call; with `--judge` the escalation runs first.

**Cost.** The footer counts the calls and the stage line the tokens:
`Judging … 1 of 10 verdicts reviewed, 1 call, 613 prompt · 193 completion tokens` and
`Summarising … 98 words, 1 call, 1,608 prompt · 343 completion tokens`
on the live draft ([details](docs/eval/2026-09-15-judge-live.md)). Groq's free tier
allows roughly one call a minute. Only the low-tier verdicts are sent — 1 of 10 on that
draft — and up to 20 go in one call, so a long bibliography costs a handful of calls, not
one per citation.

**Providers.** Default is Groq `openai/gpt-oss-120b` (free without a card, no training
on submitted data). `proofpath config set judge.provider gemini` switches to Gemini —
note that Google trains on free-tier prompts outside the EEA/UK/CH, and proofpath prints
that warning once per run. `judge.provider ollama` runs fully offline. Gemini and Ollama are
fixture-tested and were not exercised live in v0.3.0. All three speak
the OpenAI `chat/completions` shape. The key comes from `GROQ_API_KEY` / `GEMINI_API_KEY`
in the environment or a `.env` file, never from config, and is never printed.

## Measured

| What | Set | Result |
|---|---|---|
| Retrieval + entailment | SciFact dev, 340 pairs | 0.609 accuracy, 0.597 macro-F1, against a 0.406 trivial baseline ([details](docs/eval/2026-09-12-scifact-dev.md)) |
| Reference resolution | hand-built ghost set, 274 references | 0.0 % false-ghost, 99.1 % ghost recall ([details](docs/eval/2026-09-12-ghosts.md)) |
| Source access | 50 DOIs | 72 % full text, 18 % abstract only, 10 % nothing ([details](docs/eval/2026-09-11-coverage.md)) — a real biomedical paper in the live runs reached 33 % full text |
| Citation pairing, numeric | 61 hand-built passages | 0.99 ([details](docs/eval/2026-09-11-pairing.md)) |
| Citation pairing, author-year | 55 hand-built passages, 83 expectations | 0.940 ([details](docs/eval/2026-09-12-pairing-author-year.md)) |
| **End to end on real web claims** | AVeriTeC dev, 100 claims | **0.270 3-way accuracy against a 0.708 majority baseline — worse than always guessing "refuted"** ([details](docs/eval/2026-09-16-averitec.md)) |

**The AVeriTeC row is the one to read before trusting this tool on a news claim.** A third
of those claims had no readable source at all: 32 of the source URLs needed the browser
step, 29 were unreachable, 14 were refused by `robots.txt`. On the claims that *did* have a
readable source the score is 0.361 — still below the baseline, and every one of the 19
`Supported` claims was missed. The retrieval and entailment models were calibrated on
scientific abstracts, and a fact-check page is a different object: long, discursive, and
usually quoting the claim it debunks. Nothing was tuned after that measurement, and no
blocked URL was dropped from it.

What proofpath is good at is the academic path the other rows measure: finding out whether a
cited paper exists, whether it was retracted, and whether its text says what the sentence
citing it claims. Pointed at a news claim on the open web, it is currently a coverage
report with a weak verdict attached.

Published SciFact results sit around 70–75 F1, not 95. Nothing is tuned on a test
split, and no number is quoted without the run that produced it.

- [Design specification](docs/superpowers/specs/2026-09-10-proofpath-design.md) — what it does and the measurements behind each decision
- [Open items](docs/superpowers/OPEN-ITEMS.md) — what is unresolved, and what has not been verified yet · [Changelog](CHANGELOG.md)

## The website

The landing page lives in [`site/`](site/) — a static [Astro](https://astro.build) site
with no framework runtime. It replays a real `/check` run, walks the three questions and
the six rules, and credits every engraving it uses (Doré's plates for *The Raven* and
Bracquemond's *Le Corbeau*, all public domain).

```bash
cd site
npm install
npm run dev          # http://localhost:4321
npm run build        # dist/
npm run images       # regenerate the halftones from images.manifest.json (committed)
```

Design decisions and the image pipeline are described in
[`docs/superpowers/specs/2026-09-16-landing-page-design.md`](docs/superpowers/specs/2026-09-16-landing-page-design.md).

## License

MIT
