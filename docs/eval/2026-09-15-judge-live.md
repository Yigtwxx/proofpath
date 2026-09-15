# Judge layer live runs — 2026-09-15 (v0.3.0)

`proofpath check tests/data/draft-live.md --judge --summarize` on Groq
(`openai/gpt-oss-120b`, key from `.env`), Apple Silicon, models already downloaded.
Three runs: the first against the pre-fix build (it found the defect), then a cold run
(`--no-cache`) and a warm run against the shipped build. Stage lines are copied from
the terminal; the key never appears in any output.

## Run 1 — pre-fix build: the summary came back empty

```
Judging      groq openai/gpt-oss-120b   1 of 10 verdicts reviewed, 1 call, 661 prompt · 211 completion tokens 0.9s
Summarising  groq openai/gpt-oss-120b   summary unavailable after 0 calls (no completion in the 200 response from https://api.groq.com/openai/v1/chat/completions (finish_reason=length)); local verdicts stand 1.0s
```

`openai/gpt-oss-120b` is a reasoning model: with the planned `max_tokens=400` the
completion budget was spent on reasoning and the visible content was empty, and the
written `report.md` said nothing about the summary at all (the 9.3 reviewer had found
the same gap from the code). Both are fixed in the shipped build: requests carry
`reasoning_effort=low` (dropped on a 400 from providers that reject it), the review
budget is 4096 tokens and the summary budget 1500, and an unanswered summary is
reported on the stage line, the `summary` CLI line and a `- summary status:` header
line in the file. The 1,888-test suite was green before this run; the live run found it.

## Run 2 — cold (`--no-cache`), shipped build

```
Parsing      text                              1 pages, 7 refs        0.0s
Claims       rules                             7 citations, 0 unresolved 0.0s
Resolving    Crossref, Semantic Scholar        6 ok, 0 amb, 1 ghost   9.3s
Retractions  Retraction Watch                  none                   2.5s
Fetching     arXiv, Semantic Scholar           5 full text, 1 abstract, 0 unverified 23.8s
Verifying    coreml                            10 claims: 1 supported, 2 not supported, 7 NEI 44.5s
Judging      groq openai/gpt-oss-120b          1 of 10 verdicts reviewed, 1 call, 613 prompt · 193 completion tokens 1.1s
Summarising  groq openai/gpt-oss-120b          98 words, 1 call, 1,608 prompt · 343 completion tokens 1.0s
… written  ·  2 API calls  ·  82.1s
judge      2,221 prompt · 536 completion tokens
```

The escalation set was one verdict — the `low`-tier `NOT SUPPORTED` on the SciPy
sentence — and the judge's opinion was printed beside it, the local verdict untouched:

```
error[not-supported]: claim is not supported by the cited source  (confidence: low)
   = judge (groq openai/gpt-oss-120b): NEI — Passage states "SciPy provides fundamental
     algorithms for scientific computing" but does not mention SciPy building on anything.
```

The numeric mismatch (`high`, rule-decided) and the ghost reference were not sent, as
specified. The summary, verbatim:

> The report checked seven cited references, and it found two claims not supported and
> one ghost reference; it does not give a total number of claims that were examined. The
> most serious problem is the high‑confidence "NOT SUPPORTED" error on line 6, where the
> claim about training time for the Transformer contradicts the source. The coverage
> analysis shows that only 72 % of the material was verified against full text, 14 % was
> limited to abstracts, and the remaining 14 % could not be read, so the counts above
> apply only to the portion that was actually examined.

It repeats the coverage caveat as instructed and invents no finding. Two things it
gets slightly wrong, which is why it is labelled model-written and sits under the
computed report: it calls the line-6 numeric mismatch a "NOT SUPPORTED" error (the
report says `numeric-mismatch: claim contradicts the cited source`), and "it does not
give a total number of claims" is true of the markdown file (the stage table is not
printed there) but not of the run (`10 claims`).

## Run 3 — warm, shipped build

```
Resolving    cache                             6 ok, 0 amb, 1 ghost   0.0s
Fetching     cache                             5 full text, 1 abstract, 0 unverified 0.0s
Verifying    coreml                            10 claims: 1 supported, 2 not supported, 7 NEI, 10 cached 0.8s
Judging      groq openai/gpt-oss-120b          1 of 10 verdicts reviewed, 0 calls, 0 prompt · 0 completion tokens 0.0s
Summarising  groq openai/gpt-oss-120b          106 words, 1 call, 1,615 prompt · 263 completion tokens 0.9s
… written  ·  1 API calls  ·  1.8s
judge      1,615 prompt · 263 completion tokens
```

The judgement came back from the schema-v4 `judgements` table (0 calls); the summary
is always one fresh call. A plain `check` of the same file (no flags) reports
`api_calls: 0` and `summary: null`.

## What this does and does not show

- The judge layer works end to end on Groq's free tier: batching, strict-JSON output,
  cost accounting, caching, attribution, and the report unchanged by the model.
- One document, one low-tier verdict, one provider. Nothing here measures the judge's
  accuracy; the escalation band is narrow by design (OPEN-ITEMS 14.1), so a paper with
  118 citations would send a handful of verdicts, not dozens.
- Gemini and Ollama were not exercised live; they share the adapter and are covered by
  the `respx` tests only.
- These runs were made with the batch cap at 7,000 prompt tokens. The whole-phase review
  lowered it to 3,500 so that a full batch plus its 4,096-token answer budget stays under
  Groq's 8K-per-minute tier; with one escalated item the runs above never approached
  either figure, so nothing here measures the cap.
