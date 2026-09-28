# AVeriTeC dev — 2026-09-28

- claims: 100  (limit=100)
- dataset: `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json`  (sha256 499793726b4a…)
- one run of the whole product: real claims, real source pages, real fetch ladder.

## Headline

3-way accuracy **0.326** vs majority baseline 0.708, over 89 of 100 claims.

The 3-way number excludes the Conflicting Evidence/Cherrypicking rows, which proofpath has no verdict for; counting them as wrong gives a 4-way accuracy of 0.290.

## Per label

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 9 | 0.474 |
| Refuted | 63 | 18 | 0.286 |
| Not Enough Evidence | 7 | 2 | 0.286 |
| Conflicting Evidence/Cherrypicking | 11 | 0 | 0.000 |

## Source coverage

| state | count |
|---|---|
| ok | 178 |
| UNVERIFIED (blocked, browser not permitted) | 80 |
| UNVERIFIED (blocked, robots.txt) | 36 |
| UNVERIFIED (unreachable) | 2 |
| UNVERIFIED (provider unavailable) | 1 |

0 of 297 source URLs are web.archive.org snapshots (0.0 %).

## Notes

<!-- filled in by hand after the run -->

