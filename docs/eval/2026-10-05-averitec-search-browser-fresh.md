# AVeriTeC dev — 2026-10-05

- claims: 100  (limit=100)
- measured path: sentence queries, no judge
- browser step: allowed
- fetch: live, cache bypassed
- dataset: `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json`  (sha256 499793726b4a…)
- one run of the whole product: real claims, real source pages, real fetch ladder.

## Headline

3-way accuracy **0.371** vs majority baseline 0.708, over 89 of 100 claims.

The 3-way number excludes the Conflicting Evidence/Cherrypicking rows, which proofpath has no verdict for; counting them as wrong gives a 4-way accuracy of 0.330.

## Per label

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 6 | 0.316 |
| Refuted | 63 | 26 | 0.413 |
| Not Enough Evidence | 7 | 1 | 0.143 |
| Conflicting Evidence/Cherrypicking | 11 | 0 | 0.000 |

## Source coverage

| state | count |
|---|---|
| ok | 313 |
| UNVERIFIED (blocked, robots.txt) | 6 |
| UNVERIFIED (blocked) | 5 |
| UNVERIFIED (provider unavailable) | 1 |

0 of 325 source URLs are web.archive.org snapshots (0.0 %).

## Without the browser

18 of 313 read sources were read by the browser step. Treating them as unread gives a 3-way accuracy of **0.348**.

Those pages never reached the Wayback step, which a run without the browser tries next, so this is a lower bound on the pages such a run reads. It is not a bound on its accuracy: dropping a page can lose a right verdict or a wrong one.

## Notes

- Evidence search with sentence queries (no judge), browser allowed, every page fetched
  live. The 2026-09-29 run of the same path, without the browser, scored 0.404. The
  three-claim gap sits inside the ±0.05 standard error at 89 claims, and the search
  results themselves differ between the days.
- The browser read 18 of the 313 pages read. Dropping them gives 0.348.
- Coverage is close to complete: 313 of 325 pages read, 11 blocked (6 by `robots.txt`).
- **Wrong-way verdicts:** 29 of 89. 22 false claims were called SUPPORTED and 7 true
  ones REFUTED, against 11 when the gold URLs are read. The count leaves out verdicts
  given to gold-NEI claims. One likely cause, not yet measured: a page found by
  searching for a claim often repeats it.
- This run is the baseline for `2026-10-05-averitec-search-browser-judge-fresh.md`,
  which differs only in who writes the queries.

