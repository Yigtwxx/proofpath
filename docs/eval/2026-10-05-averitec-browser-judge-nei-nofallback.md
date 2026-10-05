# AVeriTeC dev — 2026-10-05

- claims: 100  (limit=100)
- measured path: gold source URLs, no search
- browser step: allowed
- fetch: cache allowed
- judge: groq openai/gpt-oss-120b (its opinions are the hypothetical column below; the verdicts are the models')
- NEI probes: on (every read NEI source was also sent to the judge, with its closest passage)
- judge fallback: off (every opinion is from the configured model)
- dataset: `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json`  (sha256 499793726b4a…)
- one run of the whole product: real claims, real source pages, real fetch ladder.

## Headline

3-way accuracy **0.371** vs majority baseline 0.708, over 89 of 100 claims.

The 3-way number excludes the Conflicting Evidence/Cherrypicking rows, which proofpath has no verdict for; counting them as wrong gives a 4-way accuracy of 0.330.

## Per label

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 0 | 0.000 |
| Refuted | 63 | 28 | 0.444 |
| Not Enough Evidence | 7 | 5 | 0.714 |
| Conflicting Evidence/Cherrypicking | 11 | 0 | 0.000 |

## Source coverage

| state | count |
|---|---|
| ok | 165 |
| UNVERIFIED (blocked, robots.txt) | 17 |
| not a url | 9 |
| UNVERIFIED (unreachable) | 8 |
| UNVERIFIED (blocked) | 4 |
| UNVERIFIED (reached, no text extracted) | 4 |
| UNVERIFIED (provider unavailable) | 2 |

76 of 200 source URLs are web.archive.org snapshots (38.0 %).

## Without the browser

0 of 165 read sources were read by the browser step. Treating them as unread gives a 3-way accuracy of **0.371**.

Those pages never reached the Wayback step, which a run without the browser tries next, so this is a lower bound on the pages such a run reads. It is not a bound on its accuracy: dropping a page can lose a right verdict or a wrong one.

165 read sources came from the cache, so the step that read them is unknown; they are kept as read.

## Judge (hypothetical)

The product never lets the judge change a verdict (spec section 11.1). This column is what the accuracy would be if it did: a medium or high model verdict still wins; otherwise the judge's opinions on the escalated sources decide, by majority of the asserting votes (an NEI opinion abstains), a tie being NEI. An escalation the judge did not answer votes with the models' label.

3-way accuracy **0.371**, against 0.371 for the product. 1 sources were escalated; 0 got no opinion back.

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 0 | 0.000 |
| Refuted | 63 | 28 | 0.444 |
| Not Enough Evidence | 7 | 5 | 0.714 |

| judge model | escalation | opinions |
|---|---|---|
| groq openai/gpt-oss-120b | nei | 105 |
| groq openai/gpt-oss-120b | escalated | 1 |

| decided by | n | correct | accuracy |
|---|---|---|---|
| models (medium or high tier) | 41 | 28 | 0.683 |
| nothing escalated | 48 | 5 | 0.104 |

### Band plus NEI probes

`--judge-nei` also sent every read source whose verdict was NEI to the judge, as NEI at low tier, with the passage the models came closest to deciding on. Here those opinions vote too, by the same rule; a probe the judge did not answer abstains.

3-way accuracy **0.427**, against 0.371 for the product. 105 NEI sources were probed; 0 got no opinion back.

Against the product, claim by claim: 5 fixed, 0 broken (exact McNemar, two-sided p = 0.062).

Refuted called SUPPORTED plus Supported called REFUTED, the wrong way: 11 (5 + 6) with the probes, against 11 (5 + 6) for the product.

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 4 | 0.211 |
| Refuted | 63 | 29 | 0.460 |
| Not Enough Evidence | 7 | 5 | 0.714 |

### Dropped by the quote check

A SUPPORTED or REFUTED opinion whose quote is not in the passage it was shown is dropped (OPEN-ITEMS 20.9), and the source then counts as unanswered above.

| escalation | reason | dropped |
|---|---|---|
| none | — | 0 |

## Notes

- **Same read pages as the 2026-10-05 gold run.** The 165 read sources are the same pages
  as in `2026-10-05-averitec-browser-judge-fresh.md`. Every one of them carries no step,
  which marks a cache hit, so they were most likely served from the fetch cache that run
  filled. The per-claim predictions are identical, so the product column (0.371) is that
  run's; only the judge's part is new. The unread sources differ slightly: 8 that were
  "provider unavailable" there are "unreachable" here. No verdict changes because of it.
- **What the judge did with the NEIs.** 105 read sources came back NEI and were shown to
  the judge with their closest passage. It answered NEI on 97, SUPPORTED on 4 and
  REFUTED on 4. Counting those 8 as votes fixes 5 claims and breaks none: 0.427 against
  0.371, exact McNemar p = 0.062. The wrong-way count stays at 11 (5 + 6). `Supported`
  goes from 0/19 to 4/19.
- **Below the bar set before the run.** OPEN-ITEMS §23 asked for p < 0.05 and no more
  false SUPPORTED than the product (5 of 63). The second condition holds and the first
  misses narrowly; with 5 discordant claims the smallest possible p is 0.0625. So this
  is recorded, not shipped. A larger set (the 300 train claims planned for 20.14) could
  settle it.
- **Most NEIs look like something other than a reading failure.** For 97 of the 105,
  a strong LLM shown the closest passage also finds nothing that settles the claim. That
  points to the deciding passage not being the closest one, or not being on the page:
  retrieval and coverage rather than entailment. This is an inference, not a measurement;
  the judge may also be cautious.
- **The quote check dropped nothing.** All 106 opinions (105 probes plus 1 escalation)
  came from `groq openai/gpt-oss-120b` with no fallback, and every asserting opinion's
  quote was found in its passage. The check costs nothing when the judge complies.
- Measured with the quote check of OPEN-ITEMS 20.9 (`#q1` cache key), `--no-fallback`,
  and the default NLI profile.

