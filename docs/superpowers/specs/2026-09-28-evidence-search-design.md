# Evidence search for source-less text

**Status:** approved 2026-09-28; implementation follows this document. Implements
OPEN-ITEMS §17.1 and revises its "never on by default". Amends spec §9 (data flow),
§11 (LLM budget), §13 (CLI surface) and §15 (honesty states) of
`2026-09-10-proofpath-design.md`. The search is **experimental** until the §7 gate passes.

## 1. Why

A pasted text that has no link, no citation marker and no bibliography currently ends
in `PARSE ERROR: the text links or cites no source; nothing to verify against`
(`verify.NO_SOURCE_IN_TEXT`, v0.4.6, `04341ea`). The common case is a linkless tweet:
X cannot be read (§3 non-goal), and pasted tweet text that links nothing gives the tool
nothing to check. The user wants plain text recognised as plain text and searched, with
proofpath saying whether it found evidence for what the text claims.

## 2. Decisions (2026-09-28, with the user)

| Question | Decision |
|---|---|
| Search provider | **Tavily** (API key) and **SearXNG** (the user's own instance, `base_url`, no key), both behind one `Searcher` protocol. Brave and the Google Fact Check API are not in v1 |
| Query | **Hybrid.** By default the query is the sentence itself, cleaned, with zero LLM calls, so §11 holds. When the judge is configured, the LLM writes 1–2 decontextualised queries per claim. If the LLM fails because of a rate limit, quota or outage, the run **falls back to the sentence and tells the user**: "LLM limit reached — searched with the sentence text" |
| Which sentences | **A heuristic check-worthiness filter plus a cap**: at most 5 claims, each with up to 3 results. Sentences left unsearched are counted in coverage (rule 6) |
| Short text (added 2026-09-28) | When **no** sentence passes the filter, every sentence is searched, up to the cap. A one-line post ("OpenAI battı") is exactly the claim the user wants checked, so it is never filtered away |
| Language (added 2026-09-28) | The models are English-only (`nli-deberta-v3-base`, `bge-small-en-v1.5`). A claim that does not look English is **translated by the judge** when `--judge` is on: the English rendering is used for the queries and is the text the models check, while the report shows the user's sentence with a `checked as: …` note. With no judge, or a judge that did not answer, the claim is `UNVERIFIED (language not supported)` and is not searched. That is never a silent NEI |
| Underspecified claims (added 2026-09-30, OPEN-ITEMS 19.1) | **A rule-based note, never a verdict change.** A claim is underspecified when the text the models check (the English hypothesis when translated) has ≤ 2 content terms — hashtags, handles, punctuation and function words dropped, a number one term, a run of capitalised words one name — and no linking verb between two content terms ("The Earth is flat" is a complete predication; "openai won" is entailed by any page about any win). Each of its verdict findings, cited or found, carries `underspecified claim: it names no object, time or scope, so many different events match it; check that the passage is about the one meant`, after the abstract line and before the provenance tag, and the SARIF message appends it. Kind, level, tier and exit code are unchanged, and no judge or NLP dependency is used. Known limits: a passive ("OpenAI was acquired") counts as complete, and a cited Turkish sentence is counted untranslated |
| Trigger | **Automatic.** When the text has no source and a search provider is configured, it is searched without asking. Configuring the key or URL is the consent. `--no-search` turns it off for one run, and `permissions.web_search = deny` turns it off for good |

Rejected, and kept rejected: letting `--judge` answer when there is no source. That makes
it a truth oracle (§3 non-goal). The LLM only writes *queries*; verdicts still come from
passages that were fetched.

## 3. Flow

A new stage, `Searching`, runs in `verify.prepare()` between Claims and Resolving. It runs
only when all of these hold:

- the document has no `references`, and `find_markers` finds nothing;
- for a post or page kind, it carries no links (the conditions at `verify.py:755` and
  `:763`);
- network permission is `allow`, and `permissions.web_search` is not `deny`;
- `--no-search` is not set;
- a search provider is configured.

1. **Select claims.** `claims.checkworthy(doc) -> Claims` is modelled on `pair_links`: each
   claim gets a synthetic `CitationMarker` with an empty span. A sentence qualifies when it
   has at least 5 words, is not a question, and contains at least one of: a digit (a
   number, year or percentage), a capitalised token that is not the first word, or a
   quotation. The first `search.max_claims` (5) qualifying sentences are kept, in document
   order.
2. **Build queries** (`search/queries.py`).
   - *Default*: strip URLs, `#hashtags`, `@handles` and emoji from the sentence, collapse
     the whitespace, and truncate at a word boundary to 400 characters (Tavily's query
     limit).
   - *With the judge*: one batched `JudgeClient.complete` call using
     `prompts/queries.md` and a strict JSON schema
     `{items: [{claim: int, queries: [str, 1..2]}]}`. It shares the run's LLM budget (§11).
   - *Fallback*: on `JudgeUnavailable` or `JudgeError`, use the default queries and add a
     run-level notice (§5). A new `Judge.queries()` follows `review()`'s rule and
     never raises. A 429 ("HTTP 429 from …" in `Judge.detail`) is reported as
     "LLM limit reached"; any other failure is reported as "LLM did not answer".
3. **Search** (`search/providers.py`).
   - The protocol is `Searcher.search(query: str, max_results: int) -> list[SearchHit]`,
     with `SearchHit(url, title, rank)`.
   - `TavilySearcher` sends `POST https://api.tavily.com/search`. The key comes from
     `secrets.resolve_api_key(search.api_key_env)`, which defaults to `TAVILY_API_KEY`.
   - `SearxngSearcher` sends `GET {base_url}/search?q=…&format=json`.
   - Both go through `polite.PoliteClient`, with their hosts added to `MIN_INTERVAL`.
   - **The provider's own snippet or `content` is never used as evidence.** Each page is
     fetched by proofpath through the fetch ladder, so the quoted passage comes from the
     source itself (rule 1).
4. **Filter hits.**
   - Canonicalise URLs (drop the fragment and tracking parameters) and dedupe them.
   - Drop the input's own host (for a post or page), and drop hosts that cannot be read
     (`social.is_social` hosts that proofpath cannot read, e.g. x.com).
   - Keep `search.results_per_claim` (3) per claim. Claims may share a page.
5. **Synthetic references.**
   - Each distinct hit becomes `Reference(number, raw=url, locator, origin="search")`.
     `origin` is a new field on `document.Reference` and defaults to `"author"`.
   - Each claim's `cited_refs` point at its hits.
   - From here Resolving (a bare URL goes to the `WebProvider`), Fetching and
     `decide_all` run **unchanged**. `_jobs` makes one job per page, so a claim gets one
     `ClaimResult` per page.

## 4. Honesty states (amends §15)

| State | Cause |
|---|---|
| `FOUND BY PROOFPATH (not cited by the author)` | provenance tag carried by **every** finding and source whose reference has `origin="search"`, together with its usual verdict or state. It is never dropped in any renderer (terminal, markdown, JSON, SARIF, TUI) |
| `SUPPORTED (found by proofpath)` | a page the search found supports the claim. A note, with its passage (rule 1): this is the answer the user asked for, so it is shown and not left silent the way a supported *cited* claim is |
| `NO EVIDENCE FOUND (searched)` | the search ran for a claim, but no hit could be read, or every read hit was NEI. This is **not** REFUTED, and the report says so (rule 2) |
| `UNVERIFIED (language not supported)` | the claim is not in English and no judge translated it (§2). The finding says `--judge` would translate it |
| `UNVERIFIED (search unavailable)` | the search provider was down, rate limited, or answered with a non-JSON body after backoff (`ProviderError`) |
| `UNVERIFIED (credentials missing)` | existing state; reused when a configured provider cannot run: `tavily` with no key, or `searxng` with no `base_url`. The finding names what to set |

A page found by the search that turns out unreadable keeps its usual §15 state
(blocked, unreachable, robots.txt …). Rules 1–3 are unchanged: no passage, no verdict,
and unreachable stays unreachable.

## 5. Report and coverage

- **Experimental banner**, printed at the top of any run that searched:
  `evidence search is experimental — AVeriTeC search-mode <score> vs 0.708 majority baseline`.
  The number is a constant, updated whenever the eval is re-run (§7).
- **Run notices.** The LLM-fallback notice from §3 step 2 is shown in the terminal,
  markdown, JSON (`notices: [...]`) and the TUI run header.
- **Coverage.** Search only runs on a document that cites nothing (§3), so a run never
  mixes the author's sources with found ones. The usual coverage block therefore counts
  the pages that were found, and a separate block beneath it says where they came from
  and how many claims were left unsearched:
  ```
  evidence search (experimental)
    claims searched        3 of 7 check-worthy
    claims not searched    4 (cap 5 / filtered)
    pages found / read     9 / 6
  ```
- **No provider configured.** The PARSE ERROR stays as it is. `NO_SOURCE_HINT` now names
  the one-line setup: `proofpath config set search.provider tavily` plus
  `TAVILY_API_KEY`, or `search.provider searxng` plus `search.base_url`.

## 6. Configuration and CLI (amends §13)

```toml
[search]
provider = "off"              # "off", "tavily", "searxng"
api_key_env = "TAVILY_API_KEY"
base_url = ""                 # SearXNG only
max_claims = 5
results_per_claim = 3

[permissions]
web_search = "allow"          # only effective once a provider is set; "ask"/"deny" turn it off (search never prompts)
```

- `config._coerce` gains an `int` branch.
- `--no-search` goes on `check` and on the TUI (`/config` panel entries plus
  `settings_hints.py` strings).
- `cli` and `tui` only pass the flag through; all logic stays in `verify()`.
- The key is only ever read through `secrets.resolve_api_key`. It is never written to
  config or logged.

## 7. Evaluation (the §17.1 gate)

`scripts/eval_averitec.py --search` replaces each claim's gold `source_urls` with the
search hits and then scores the same way (`score`, `render_report`). The claim's own
fact-check article is excluded by URL and host, so the tool does not find the answer
key. Results go to `docs/eval/<date>-averitec-search.md`. The feature ships flagged
experimental, with the number printed, and the flag comes off only once the score is
above the 0.708 majority baseline.

## 8. Tests (no network)

- `respx` plus `tests/fixtures/search/{tavily,searxng}.json`, and the fakes in
  `tests/fakes.py`.
- A plain tweet text leads to searched claims, each finding tagged `FOUND BY PROOFPATH`.
- Empty hits or all-NEI results give `NO EVIDENCE FOUND (searched)` and never REFUTED.
- A judge 429 falls back to sentence queries, and the run carries the notice.
- With no provider, the old PARSE ERROR appears with the new hint. `--no-search` and
  `web_search = deny` mean no search call is made.
- Coverage counts the sentences past the cap as not searched. The provider snippet
  never appears as a passage.
- The golden report `tests/data/verify-report-golden.json` is updated.

## 9. Product rules, checked

1. No passage, no verdict: pages are fetched and the passage comes from the page, never
   from the search snippet.
2. Absence of evidence is not evidence of absence: `NO EVIDENCE FOUND (searched)` is its
   own state.
3. No false ghosts: not affected, because search results are web pages and never
   bibliographic references.
4. No prompt without a TTY: the search never prompts, and the consent is the configured
   provider.
5. No large install: none.
6. Coverage: a separate block, plus the experimental banner.

## 10. Out of scope (v1)

- Searching for the uncited sentences of a partly cited document (§17.3).
- Brave, Google Fact Check (ClaimReview), and LLM-based check-worthiness.
- Using the search snippet as a fallback when the page cannot be read.

## 11. Local fallback judge (decided 2026-09-28)

The judge runs on Groq's free tier, which hits its rate limit. When the configured
judge stops answering, the run switches to a model already installed in the user's
local Ollama instead of losing the second opinion (and, for the search, the
judge-written queries and translations).

- **Trigger.** Any `JudgeUnavailable` from the primary provider: a 429, a quota
  error, or the provider not answering. The switch lasts for the rest of the run, so
  a rate-limited Groq is not hit again on every batch.
  The switch is immediate: with a fallback available, the primary gets no retry and
  no `Retry-After` wait. The failed request is sent again to the local model, so no
  batch is lost, and the notice appears before the model starts loading.
  The next run (a new link, a new `check`) tries the configured API first again.
- **Which model.** The fallback is `qwen3.5:9b` (the user's choice, backed by a
  2026-09-28 translation benchmark: 15/15 correct TR→EN, where qwen3.6 and
  gpt-oss:20b changed the meaning). It must already be installed; proofpath never
  pulls it (rule 5). `GET {ollama}/api/tags` only confirms that it is there. Not
  installed, or no Ollama, means no fallback: the run reports the original
  `JudgeUnavailable` with `; no local fallback: qwen3.5:9b is not installed` (or
  why Ollama could not be asked). The order is fixed: the configured API, then this
  model, and nothing else.
- **Reasoning.** Every Ollama request (the fallback, and a judge configured as
  `ollama` directly) sends `reasoning_effort: none` (`think: false`): a thinking qwen
  spent its whole token budget and answered nothing. `judge.provider ollama` now
  defaults to the same `qwen3.5:9b`, so there is one local model. A 400 that names
  `reasoning_effort` still drops the field, as for any provider.
- **Config.** `judge.fallback`: `qwen3.5:9b` by default; `off` (in any case) turns
  the fallback off; another model name is the user's own override, used as written
  and checked against the tags the same way. It does nothing when the judge is
  already Ollama.
- **Telling the user.** The switch is never silent.
  - The run emits a `Note` that names the cause: `Groq limit reached — judging with
    local ollama <model>` (429, 413), `Groq rejected the key (HTTP 401) — …` (401,
    403), or `Groq did not answer (<cause>) — …` for anything else (`HTTP 500`,
    `timeout`, `connection failed`, `empty answer`). Every failure switches; the
    cause never carries a body or a key.
  - The CLI prints that line on stderr.
  - The TUI shows it in a one-line notice, right-aligned, directly above the prompt
    bar. It stays until the next run starts.
  - The report records it as well. Each opinion's `model` already names who gave it
    (`ollama qwen3.5:9b`), and the footer carries the notice.
- **Honesty.** A local model's opinion is still only a second opinion (spec §11). It
  never replaces a verdict's passage.
