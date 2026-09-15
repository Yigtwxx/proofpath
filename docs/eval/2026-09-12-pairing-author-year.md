# Citation pairing (author-year) — 2026-09-12

- rows: 55
- expectations: 83 (74 pairs, 9 reported markers)
- styles (expectations): mixed 8, author-year 41, collision 6, back-reference 28
- pairing rate: **0.94** (78/83)

## Hand set (author-year)

| style | checks | correct | rate |
|---|---|---|---|
| mixed | 8 | 8 | 1.00 |
| author-year | 41 | 36 | 0.88 |
| collision | 6 | 6 | 1.00 |
| back-reference | 28 | 28 | 1.00 |
| **all** | 83 | 78 | 0.94 |

## Hand set (numeric) — unchanged by this run

| set | rows | checks | correct | rate |
|---|---|---|---|---|
| numeric | 61 | 109 | 108 | 0.99 |

## Misses

| row | marker | reason |
|---|---|---|
| ay-32 | `(WHO, 2020)` | missing |
| ay-32 | `(WHO, 2020)` | extra |
| hd-01 | `(Smith, 2019)` | missing |
| hd-01 | `(Smith, 2019)` | extra |
| hd-02 | `Smith 2020` | missing |
| hd-04 | `(Lindqvist, 2019)` | missing |
| hd-04 | `(Lindqvist, 2019)` | extra |
| hd-05 | `(Berg, 2018)` | missing |
| hd-05 | `(Berg, 2018)` | extra |

## Real PDFs

| id | kind | pages | paragraphs | refs | markers | claims | unresolved | unsupported | scoped |
|---|---|---|---|---|---|---|---|---|---|
| arXiv:1907.11692 | pdf | 13 | 189 | 103 | 89 | 15 | 80 | 0 | 2 |

### Sampled pairs — arXiv:1907.11692

Source: https://arxiv.org/pdf/1907.11692 (cached)

```
(Lample and Conneau, 2019)  Self-training methods such as ELMo, GPT, BERT, XLM, and XLNet have brought signiﬁcant performance ga
(Kingma and Ba, 2015)  BERT is optimized with Adam using the following parameters: β1 = 0.9, β2 = 0.999, ǫ = 1e-6 and L2 we
(Hendrycks and Gimpel, 2016)  BERT trains with a dropout of 0.1 on all layers and attention weights, and a GELU activation functio
Trinh and Le (2018)  • STORIES, a dataset introduced in containing a subset of CommonCrawl data ﬁltered to match the stor
(Nagel, 2016)  • CC-NEWS, which we collected from the English portion of the CommonCrawl News dataset.
(Gokaslan and Cohen, 2019)  • OPENWEBTEXT, an open-source recreation of the WebText cor-
Zellers et al. (2019)  4We use news-please to collect and extract CC-NEWS.
Zellers et al. (2019)  CC-NEWS is similar to the RE-ALNEWS dataset described in.
(Ott et al., 2018)  Past work in Neural Machine Translation has shown that training with very large mini-batches can bot
(Honnibal and Montani, 2017)  For a given input sentence, we use spaCy to extract additional candidate noun phrases from the sente
(Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018)  Pretraining methods have been designed with different training objectives, including language modeli
(Devlin et al., 2019; Lample and Conneau, 2019)  Pretraining methods have been designed with different training objectives, including language modeli
```

## Notes

Written by hand after the run (task 8.5 including fix round 1, spec §9 step 2, §17 v0.2).
The hand set was written by reading the passages, not by running the code: the
expectations are what a reader of each passage pairs, and the five the code does not meet
are recorded below rather than softened out of the set. Nothing in `claims.py` was tuned
against these rows; three patterns *were* changed because rows written from real prose
showed them wrong (the Oxford comma, the possessive, and `ibid.` after a numeric marker),
and each change is in the direction of reading more citations, never of inventing one.

### The hand set

55 passages, **83 expectations**: 74 marker-to-sentence pairs and 9 markers that must be
*reported* rather than paired. 78 met, rate **0.940**, against a gate of 0.90 for this
style. The scorer closes the set in the other direction too: any claim no expectation
accounts for, and any reported marker a row declares in neither list, counts as an `extra`
miss, so a pattern that starts seeing citations which are not there cannot pass unnoticed.

Coverage: one author, two with `and` and with `&`, three with and without the Oxford
comma, `et al.` with and without a comma before the year, `;` lists of two and three,
narrative `Jones (2019)` / `Novak et al. (2021)` / `Rahman and Ortiz (2021)`, a possessive
narrative (`Smith's (2020)`), `see` and `e.g.,` lead-ins, page / page-range / chapter
tails, a particle surname (`van der Berg`, `de Vries`), a hyphenated surname
(`Ortiz-Vega`), diacritics folded both ways (`Muller` → `Müller`, `Hernandez` →
`Hernández`), `2019a`/`2019b` collisions and an ambiguous `2019` with no letter, a letter
past the end of the collision, `ibid.` in the same sentence run, in the next paragraph,
two paragraphs back, chained, after a **numeric** marker, after an **unresolved** one, and
with nothing in front of it, `op. cit.` with and without a surname, a name no entry
carries, a year no entry prints, an organisation author, a paragraph-final marker that
scopes its paragraph, and four mixed numeric + author-year passages including `(see Smith,
2020; [2])`.

### The five misses

Each is a limitation of the rules as written, not a bug against them, and each was kept in
the set so the next change to `claims.py` has to argue with it.

**`hd-01` — the year has to match exactly.** The body cites `(Smith, 2019)` where the list
prints 2020: an arXiv year against a published one, which is how a good half of real
citations drift. `resolve.year_matches` has a ±1 tolerance for exactly this reason, and
`pair_author_year` deliberately does not use it. A tolerance here does not widen a score,
it *picks an entry*: with two entries by one author a year apart, ±1 makes both match and
the citation resolves to whichever the letter rule happens to reach. Reported unresolved
is the honest answer until the collision rule can tell them apart.

**`hd-02` — `Smith 2020` with no comma is not detected at all.** Inside a parenthesis the
year must follow a comma or `et al.`; a capitalised word next to a bare year is `(Figure
2020)` at least as often as it is a citation, and the false one would report a source the
sentence never named. The cost is one citation missed in the styles that print no comma.

**`hd-04` — only the first author of an entry is matched.** The body cites `(Lindqvist,
2019)`, the second author of "Okafor, C. and Lindqvist, S. …". `resolve.author_hint`
returns the first author's surname and nothing else, and matching against every author of
an entry would need the entry's author *list* parsed — which is exactly the local
bibliography parsing spec §8 refuses to do.

**`hd-05` — two families sharing a last word collide.** `(Berg, 2018)` against a list
holding both "Berg, T." and "van der Berg, P." folds to one key and resolves to neither.
The fold keeps only the last word so that the body's `van der Berg` can meet the `Berg`
that `author_hint` reads out of the entry; the collision is the price, and it is paid as
an *ambiguous* report, never as a guess (product rule 3).

**`ay-32` — an organisation cited by its acronym.** `(WHO, 2020)` against "World Health
Organization. Guidance on clinic staffing levels. WHO Press, 2020." A reader pairs it
without a thought. An initialism rule would too — and would equally pair `(WHO, 2020)` with
"Wolfson, H., Hart, D. and Owen, K. …". The surname comparison has no way to tell a name
from an acronym, so this is recorded as a miss rather than closed with a rule that guesses.

### `ibid.` after fix round 1

`ibid.` takes the refs of the marker **immediately before it in reading order**, whatever
style that one was written in. `(Smith, 2020) … [2] … (ibid.)` resolves the `ibid.` to
entry 2 (`bk-08`): a paper that numbers half its list and names the other half is the
normal case, and the earlier rule — a history built from author-year items only — lost
the back-reference there (the miss recorded as `hd-03` before this round).

It never reaches back *past* that marker. `(Smith, 2020) … (Okonkwo, 2022, which no entry
answers) … (ibid.)` reports the `ibid.` (`bk-09`) instead of resolving it to Smith: what
the reader is looking at is Okonkwo, and skipping over it would answer "which source?"
with the one they are not reading. The same holds when the previous marker stands more
than one paragraph back. `op. cit.` is unchanged and deliberately different — it prints an
author, so it searches every surname cited so far.

### Two patterns the set found, and the fixes

**The Oxford comma.** `(Novak, Silva, and Chen, 2021)` matched nothing: the name list
allowed `, Surname` and a trailing ` and Surname`, but not `, and Surname`, so the whole
marker was lost rather than its third author. The list now allows the comma before `and`.

**The possessive.** `Smith's (2020)` was detected and then matched no entry, because the
folded surname was `smith's`. A trailing `'s` is now dropped, and a typographic apostrophe
is normalised before the ASCII fold — without that, `Smith’s` folded to `smiths`, which no
amount of suffix matching would have rescued.

A third change is in `retrieval.py` rather than `claims.py`: `p.`, `pp.`, `ch.` and
`chap.` no longer end a sentence **when a digit follows**, so `(Smith, 2020, p. 12).` stays
one sentence and one marker, while `It rose by 5 pp. The trend…` still splits. That
splitter is not `ingest`'s alone — `verify._index_for` chunks fetched source text with it
before embedding — so the change moves retrieval as well as claims; the tests for both are
in `tests/test_retrieval.py`.

### The numeric hand set

Unchanged as a measurement of pairing, and **up by one**: 107/109 before this task,
**108/109** after. The single row whose *score* moved is `num-15`, the mixed `(see Smith,
2020; [12])` that the 2026-09-11 report records as "Miss 2 — the author-year half of a
mixed citation is never reported" and OPEN-ITEMS lists as 9.2.

Two things about that row changed and nothing else did. Its declaration moved from
"reported as an unsupported style" to "reported as an unresolved marker", because v0.2
pairs the style and so reports the *marker*; and its expected claim **text** now closes up
over the whole parenthesis ("…in some journals and confuses automated tools.", where v0.1
left "…in some journals (see Smith, 2020;)…"), because the author-year half is a marker
now and is stripped with the rest. The pair itself — marker `[12]`, the sentence it sits
in, refs `(12,)`, not paragraph-scoped — is what it always was. Five *other* rows
(`num-14`, `par-14`, `mix-02`, `mix-07`, `mix-14`) only moved their author-year
declarations from `expected_unsupported` to `expected_unresolved`; none of their pairs
changed in any field. **No pair in the set changed its marker, its sentence or its refs**,
and the one remaining miss is still `num-05` (`[15]`, wrong sentence, OPEN-ITEMS 9.6).

`tests/test_eval_pairing.py::test_the_numeric_hand_set_has_not_moved` pins 108/109 in both
directions, so a later change cannot drift it in silence.

### `arXiv:1907.11692` (RoBERTa) — 89 markers, 15 claims, 80 unresolved

The same cached PDF the 2026-09-11 report measured, re-ingested and re-extracted offline.
Before this task the row read **88 markers, 0 claims, 0 unresolved, 88 unsupported**: every
citation in the paper is author-year, and v0.1 reported the lot as a style it could not
pair. The 89th marker is the one the Oxford-comma fix recovered.

15 claims out of 89 markers is not a pairing failure. **It is the bibliography.** RoBERTa
is ACL-style: the reference list prints no numbers, so `ingest.split_references` falls back
to one entry per paragraph, and the two-column PDF hands it each entry as *two* paragraphs
split at a hyphenated line break:

```
[1] Eneko Agirre, Llu'is M'arquez, and Richard Wicen-
[2] towski, editors. 2007. Proceedings of the Fourth International Workshop …
```

103 "entries" for what the paper prints as roughly 50. **54 of the 103 have no author hint
at all** (`resolve.author_hint` returns `""`), because a head fragment carries no year and
no initials for the author-list patterns to end on. Of the 96 author-year items in the
body, 87 fail on "surname not in the list", 4 on the year, and 5 resolve.

Measured, as a diagnostic only, by rejoining an entry with the one before it whenever the
first prints no year: 103 → 47 entries, 9 still without a hint, and the same document then
yields **65 claims and 42 unresolved markers** from the same 89 markers. That is the size
of the gap and where it sits: upstream of `claims.py`, in how an unnumbered two-column
bibliography is cut into entries. It is recorded as a new open item; the crude rejoin above
is a measurement, not a patch, and nothing in this task ships it.

The two paragraph-scoped claims and the twelve sampled pairs above were read against the
printed pages: `(Kingma and Ba, 2015)` on the Adam sentence, `(Hendrycks and Gimpel, 2016)`
on the GELU sentence, `(Nagel, 2016)` on CC-NEWS and `Trinh and Le (2018)` on STORIES are
each on the sentence the paper cites them from, and the narrative form is on its own
sentence rather than the one before it. Two of the twelve sit on a footnote line
(`4We use news-please …`) that ingest reads as prose — the byline/non-prose problem of
OPEN-ITEMS 9.1, not a pairing error.

### What this run does and does not license

0.940 is the rate on hand-written prose with clean sentence boundaries and a bibliography
that splits correctly. It says the rules of spec §9 step 2 for author-year are implemented
as written, bar the five misses named above. On a real ACL PDF the ceiling is set by the
reference list, not by the marker rules: 5 of 96 items resolve as the list is cut today,
and roughly 55 of 96 with the entries rejoined.

1. **An unnumbered two-column bibliography is cut at line breaks** (RoBERTa: 103 entries
   for ~50). The largest open item for author-year pairing.
2. **The year must match exactly** (`hd-01`), **the first author only** (`hd-04`),
   **surnames sharing a last word collide** (`hd-05`), and **an organisation acronym never
   matches its entry** (`ay-32`).
3. **`Smith 2020` without a comma is not a marker** (`hd-02`), and neither is a footnote-only
   or superscript-letter citation — those reach neither `claims.unresolved` nor
   `claims.unsupported`, because nothing detects them at all.
