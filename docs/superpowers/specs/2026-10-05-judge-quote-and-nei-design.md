# Judge quote check (20.9) and the NEI → judge measurement (20.13) — design (2026-10-05)

The user asked for both items to be done end to end, with the best option chosen
without further questions. The choices below were made on that mandate and are written
here so they can be revisited. OPEN-ITEMS 20.9 and 20.13 point here.

## 1. 20.9: the judge's quote is checked (product change)

**Problem.** `review.md` asks for a rationale that "repeats the deciding words from the
passage verbatim", but nothing checks that it does. A judge that answers from memory,
or paraphrases, still gets its opinion attached beside the verdict. Rule 1 ("never
assert without a passage") binds the judge as much as the pipeline (spec §11).

**Decision.**

1. **A separate `quote` field.** The review schema and prompt add `"quote"`: the exact
   words from the passage that decide the item. The rationale stays the one-sentence
   reason. A structured field can be checked; a quote buried in prose cannot.
2. **Checked for asserting labels only.** A `SUPPORTED` or `REFUTED` opinion whose
   quote does not occur in the passage is dropped. The drop is noted through
   `Judge.skipped` as `"<id>: the quoted words are not in the passage; opinion
   dropped"`, which the run already reports. An asserting opinion that came back with
   no quote at all is noted separately, as `"<id>: no quoted words came back; opinion
   dropped"`, so the report can count the two reasons apart. An `NEI` opinion may have
   an empty quote: it asserts nothing.
3. **Matching** is exact after one normalisation, applied to both sides (revised
   after review, 2026-10-05):
   - Unicode NFC (not NFKC: NFKC turns `10⁶` into `106`, a false numeric match);
   - every Unicode space separator (`Zs`, the no-break space among them) becomes a
     space;
   - curly quotes and apostrophes become straight ones;
   - every `Pd` dash and the minus sign U+2212 become `-`;
   - the fraction slash U+2044 becomes `/`;
   - every format character (`Cf`: the soft hyphen U+00AD, the zero-width space
     U+200B and the like) is removed;
   - every whitespace run becomes one space;
   - then casefold. A documented limit: "MW" equals "mW". Ligatures and full-width
     letters are not folded, so a model that types "fi" for "ﬁ" fails.

   The quote is then split at every ellipsis (`...` or `…`). Each part loses the
   whitespace, quote marks, brackets and sentence punctuation (`.,;:!?`) at its ends,
   and never `-`, `%` or `+`: "−3.2%" must not become "3.2". The checks then run in
   this order, and the first failure is the reason:
   - **no quote**: there is not a single word (`\w+`) left;
   - **too short**: a part has fewer than two words. One word ("the", "e", a "5")
     is in nearly every passage and proves nothing;
   - **not in the passage**: some part does not occur in the passage at word
     boundaries (the characters before and after the match are not letters or digits,
     or are the passage's ends), in order and without overlapping. "5 million" is not
     in "15 million", and "ploy" is not in "Unemployment".

   Each reason has its own note in `Judge.skipped` (`QUOTE_MISSING`,
   `QUOTE_TOO_SHORT`, `QUOTE_NOT_FOUND`).
4. **No stale opinions, and the quote is shown.** Judgements cached before this change
   were never checked, so the cache must not serve them as checked. The judgement
   cache key gains `#q1` (the checks an opinion passed) and a 16-hex-digit sha256 of
   the whitespace-folded passage it was checked against:
   `<judge model>#q1#<passage hash>`. The passage is in the key because the check is
   about that passage: the same claim and source can be decided on another passage
   after an NLI profile or `k` change, and an opinion checked against one says nothing
   about the other. `JudgeOpinion` gains `quote: str = ""`: the verified quote less
   its wrapping, empty for an NEI that quoted nothing or quoted wrongly (an NEI is
   kept either way, but a quote the report shows is always a checked one). It is
   stored in a `quote` column, added to older files by a guarded migration
   (`PRAGMA table_info`, then `ALTER TABLE judgements ADD COLUMN quote TEXT NOT NULL
   DEFAULT ''`). The judge line shows it as a quoted span before the rationale,
   `judge (<model>): SUPPORTED — "<quote>" — <rationale>`, on every surface; JSON
   gains the field and renames nothing.
5. **The batch cap holds the quote.** Every opinion now carries a quote. A typical
   one is about 50 tokens; budgeted at 100 for a long quote, the existing per-batch
   item cap (`BATCH_SIZE` = 20) writes 2000 tokens, under half of the review's 4096
   (`_REVIEW_TOKENS`). The other half is left to a reasoning model's thinking, which
   it charges to the same budget. The arithmetic is a comment beside `_OPINION_TOKENS`
   and a test.
6. The judge's other jobs, query writing and the summary, are unchanged.

## 2. 20.13: what the judge would do with the NEIs (measurement only)

**Problem.** On AVeriTeC dev (2026-10-05), 36 of 89 claims had a readable source and
came back NEI where the gold label was Supported or Refuted. `aggregate` drops the
passage of an NEI verdict (`pipeline.py`), so `verify.escalates` never sends an NEI to
the judge. Only 1 of 165 read sources was escalated.

**Decision.**

1. **The closest passage.** For each read source whose verdict is NEI, the eval takes
   the passage the models came closest to deciding on: the one with the highest
   SUPPORTED or REFUTED probability, which `aggregate` already finds and then
   discards. Expose it through a small pure helper used by `aggregate`. **The
   product's verdicts must not change**, and existing pipeline tests must stay green
   unmodified.
2. **`--judge-nei`** (requires `--judge`) sends those NEI sources to the judge with
   `verdict=NEI` and `tier=low`, beside the normally escalated ones. Each source
   records which kind of escalation it was: `escalated` for the product's band and
   `nei` for this probe. The hypothetical column then reports two numbers:
   - the product's band only (as today);
   - the product's band plus the NEI probes.
3. **`--no-fallback`** (requires `--judge`) sets `judge.fallback = off` for the run. The provider then waits
   out a 429 as it normally does without a fallback, so every opinion is from the
   configured model (`groq openai/gpt-oss-120b`), not a mix.
4. **Quote check on.** The run uses 20.9, so an opinion whose quote is not in the
   passage is dropped and counted. The report says how many: per kind of escalation,
   and per reason (no quote, a part under two words, quote not in the passage).
5. The setup suffix in file names gains `-judge-nei` and `-nofallback`.
6. **The run.** Gold URLs, browser allowed, cache allowed: today's pages are within the
   7-day TTL, so this reads the same pages as the 2026-10-05 gold run. `--sleep 1.0`,
   100 dev claims.
7. **Reading the result.** The NEI-probe column is compared with the product,
   claim by claim (exact McNemar), and its wrong-way count is reported beside the
   product's. Whether the product should escalate NEI with its closest passage stays a
   separate decision. The bar is a gain that clears the noise (p < 0.05) with no more
   false claims called SUPPORTED than the product (5 of 63). Unless it clears that bar,
   the item is recorded and not shipped.

## 3. Tests (no network)

- **Quote check:**
  - exact and normalised matches pass;
  - a paraphrase fails;
  - an ellipsis with parts in order passes, and with parts out of order fails;
  - an empty quote fails for SUPPORTED and REFUTED but passes for NEI;
  - the dropped opinion's note reaches `Judge.skipped`.
- **Cache:** an opinion stored under the old key is not served after the change.
- **Pipeline:** the helper returns the passage `aggregate` would have chosen, and
  `aggregate`'s verdicts are unchanged (existing tests untouched).
- **Eval:**
  - `--judge-nei` without `--judge` is a usage error;
  - NEI sources are sent with their closest passage and recorded as `nei`;
  - the two hypothetical numbers and the drop counts are scored and rendered;
  - `--no-fallback` builds a client without a fallback;
  - the new file-name suffixes are used;
  - old results files still load.
