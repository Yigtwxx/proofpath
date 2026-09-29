# AVeriTeC dev — 2026-09-29

- claims: 100  (limit=100)
- measured path: sentence queries, no judge
- dataset: `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json`  (sha256 499793726b4a…)
- one run of the whole product: real claims, real source pages, real fetch ladder.

## Headline

3-way accuracy **0.404** vs majority baseline 0.708, over 89 of 100 claims.

The 3-way number excludes the Conflicting Evidence/Cherrypicking rows, which proofpath has no verdict for; counting them as wrong gives a 4-way accuracy of 0.360.

## Per label

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 9 | 0.474 |
| Refuted | 63 | 25 | 0.397 |
| Not Enough Evidence | 7 | 2 | 0.286 |
| Conflicting Evidence/Cherrypicking | 11 | 0 | 0.000 |

## Source coverage

| state | count |
|---|---|
| ok | 224 |
| UNVERIFIED (blocked, browser not permitted) | 74 |
| UNVERIFIED (blocked, robots.txt) | 20 |
| UNVERIFIED (unreachable) | 3 |
| UNVERIFIED (provider unavailable) | 2 |
| UNVERIFIED (reached, no text extracted) | 1 |

0 of 324 source URLs are web.archive.org snapshots (0.0 %).

## Notes

- This is the evidence-search gate of spec §17.1. At 0.404 it is below the 0.708
  majority baseline, so the feature stays marked experimental.
- For comparison: the same 100 claims read from their **gold** source URLs scored 0.270
  (`2026-09-16-averitec.md`). An earlier search run on 2026-09-28 scored 0.326. That run
  took the old path, which over-fetched results and skipped the language gate; the
  final review replaced that path with this one.
- 74 pages needed the browser step, which this run did not use (`--no-browser`). A run
  with the browser allowed would read more pages.
- Measured with sentence queries and no judge. A judge writes better queries; that
  path has not been measured yet.
