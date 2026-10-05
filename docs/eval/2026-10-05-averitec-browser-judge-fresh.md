# AVeriTeC dev — 2026-10-05

- claims: 100  (limit=100)
- measured path: gold source URLs, no search
- browser step: allowed
- fetch: live, cache bypassed
- judge: groq openai/gpt-oss-120b (its opinions are the hypothetical column below; the verdicts are the models')
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
| UNVERIFIED (provider unavailable) | 10 |
| not a url | 9 |
| UNVERIFIED (blocked) | 4 |
| UNVERIFIED (reached, no text extracted) | 4 |

76 of 200 source URLs are web.archive.org snapshots (38.0 %).

## Without the browser

10 of 165 read sources were read by the browser step. Treating them as unread gives a 3-way accuracy of **0.348**.

Those pages never reached the Wayback step, which a run without the browser tries next, so this is a lower bound on the pages such a run reads. It is not a bound on its accuracy: dropping a page can lose a right verdict or a wrong one.

## Judge (hypothetical)

The product never lets the judge change a verdict (spec section 11.1). This column is what the accuracy would be if it did: a medium or high model verdict still wins; otherwise the judge's opinions on the escalated sources decide, by majority of the asserting votes (an NEI opinion abstains), a tie being NEI. An escalation the judge did not answer votes with the models' label.

3-way accuracy **0.371**, against 0.371 for the product. 1 sources were escalated; 0 got no opinion back.

| label | n | correct | accuracy |
|---|---|---|---|
| Supported | 19 | 0 | 0.000 |
| Refuted | 63 | 28 | 0.444 |
| Not Enough Evidence | 7 | 5 | 0.714 |

| judge model | opinions |
|---|---|
| groq openai/gpt-oss-120b | 1 |

| decided by | n | correct | accuracy |
|---|---|---|---|
| models (medium or high tier) | 41 | 28 | 0.683 |
| nothing escalated | 48 | 5 | 0.104 |

## Notes

- Same 100 dev claims as `2026-09-16-averitec.md` (0.270), but not the same web. That
  run used `--no-browser` and read 120 sources; this one read 165 of 200, the browser
  step 10 of them. Pages that were unreachable then answer now, so the two numbers
  measure different snapshots of the same claims.
- **The browser is worth about two claims here.** Dropping the 10 pages it read gives
  0.348, against 0.371. At 89 claims the standard error is about ±0.05, so this is
  within the noise.
- **The judge saw one source.** `pipeline.decide` never gives an NEI verdict a
  passage, and `verify.escalates` sends nothing without one (rule 1). 105 of the 165
  read sources are NEI. The hypothetical column therefore equals the product. The judge
  never switched to its local fallback.
- **Where the 56 wrong claims are.** 36 had a readable source and came back NEI where
  the gold label was Supported or Refuted. 11 were called the wrong way: Refuted read as
  Supported 5 times, Supported as Refuted 6 times. 7 had nothing readable, and 2 gold
  NEI claims got a verdict. Coverage is no longer the main loss; the reading is.
- Of the 33 claims scored right, one is a gold-NEI claim with nothing readable, which
  the score counts as NEI. 32 are verdicts or NEIs on sources that were read.
- Measured with the default NLI profile (pinned), `--fresh`, `--sleep 1.0`.

