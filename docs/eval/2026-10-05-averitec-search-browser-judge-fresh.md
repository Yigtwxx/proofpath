# AVeriTeC dev — 2026-10-05

- claims: 100  (limit=100)
- measured path: judge queries (groq openai/gpt-oss-120b)
- browser step: allowed
- fetch: live, cache bypassed
- judge: groq openai/gpt-oss-120b (writes the search queries; its opinions are the hypothetical column below)
- dataset: `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json`  (sha256 499793726b4a…)
- one run of the whole product: real claims, real source pages, real fetch ladder.

## Headline

3-way accuracy **0.483** vs majority baseline 0.708, over 89 of 100 claims.

The 3-way number excludes the Conflicting Evidence/Cherrypicking rows, which proofpath has no verdict for; counting them as wrong gives a 4-way accuracy of 0.430.

## Per label

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 10 | 0.526 |
| Refuted | 63 | 31 | 0.492 |
| Not Enough Evidence | 7 | 2 | 0.286 |
| Conflicting Evidence/Cherrypicking | 11 | 0 | 0.000 |

## Source coverage

| state | count |
|---|---|
| ok | 319 |
| UNVERIFIED (blocked, robots.txt) | 9 |
| UNVERIFIED (blocked) | 4 |

0 of 332 source URLs are web.archive.org snapshots (0.0 %).

## Without the browser

17 of 319 read sources were read by the browser step. Treating them as unread gives a 3-way accuracy of **0.483**.

Those pages never reached the Wayback step, which a run without the browser tries next, so this is a lower bound on the pages such a run reads. It is not a bound on its accuracy: dropping a page can lose a right verdict or a wrong one.

## Judge (hypothetical)

The product never lets the judge change a verdict (spec section 11.1). This column is what the accuracy would be if it did: a medium or high model verdict still wins; otherwise the judge's opinions on the escalated sources decide, by majority of the asserting votes (an NEI opinion abstains), a tie being NEI. An escalation the judge did not answer votes with the models' label.

3-way accuracy **0.483**, against 0.483 for the product. 1 sources were escalated; 0 got no opinion back.

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 10 | 0.526 |
| Refuted | 63 | 31 | 0.492 |
| Not Enough Evidence | 7 | 2 | 0.286 |

| judge model | opinions |
|---|---|
| groq openai/gpt-oss-120b | 1 |

| decided by | n | correct | accuracy |
|---|---|---|---|
| models (medium or high tier) | 71 | 41 | 0.577 |
| nothing escalated | 18 | 2 | 0.111 |

## Notes

- **The one gain that cleared the noise.** The only difference from
  `2026-10-05-averitec-search-browser-fresh.md` (0.371) is who writes the search
  queries: here the judge, `groq openai/gpt-oss-120b`. Paired over the same 89 claims,
  the judge's queries were right where the sentence queries were wrong 14 times, and
  the reverse 4 times (exact McNemar p = 0.031). `Supported` went from 6/19 to 10/19.
- **The share of fact-check pages did not change.** By a host/URL heuristic, they were
  22.0 % of the pages here and 22.5 % with sentence queries. That says the mix did not
  shift, not that no better fact-check page was found. The checker's own site is
  excluded in both, as in every search run.
- **Many claims still come back the wrong way.** 25 of the 89 claims came back the wrong
  way: 20 false claims were called SUPPORTED, and 5 true ones REFUTED. Reading the gold
  URLs gives 11. The count leaves out verdicts given to gold-NEI claims. One likely
  cause, not yet measured: a page found by searching for a claim often repeats it.
- **One risk this run cannot rule out.** A model writing the queries may carry what it
  remembers about a claim into them. The verdicts still come from the pages read, each
  with its passage, but the choice of pages may not be blind.
- All queries were written by Groq. The judge never switched to its local fallback;
  the run printed no switch notice.
- The browser read 17 of the 319 pages read; dropping them leaves the score at 0.483.
  The hypothetical judge column equals the product: one source was escalated (see the
  gold-URL run's note on NEI verdicts).
- Still below the 0.708 majority baseline, so the search stays experimental.

