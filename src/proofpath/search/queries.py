"""What the evidence search asks: the sentence itself, or what the judge wrote.

The default makes zero LLM calls (spec section 11). With ``--judge``, the judge
rewrites each claim as a query that stands on its own, and translates it to English
when it is not (Amendment A): the check models are English-only, so a claim that is
not English has no query, no English text, and nothing for Task 7 to check unless a
judge translated it. When the judge cannot answer -- a rate limit, an exhausted
quota, an outage -- the sentence is used instead and the run says so, because a
weaker query is a weaker search and the reader has to know which one they got
(product rule 6).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from proofpath.judge import LIMIT_STATUSES
from proofpath.search.language import certainly_english, looks_english

if TYPE_CHECKING:
    from proofpath.judge import Judge

QUERY_LIMIT = 400
SENTENCE_BY = "sentence"
LLM_LIMIT = "LLM limit reached — searched with the sentence text"
LLM_UNAVAILABLE = "LLM did not answer — searched with the sentence text"

_URL = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"(?<!\w)@\w+")
# A hashtag is often the claim's subject ("#ChatGPT"): the sign goes, the word stays.
_HASH = re.compile(r"(?<!\w)#(\w+)")
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿️‍]")


def sentence_query(text: str) -> str:
    """The sentence as a query: no addresses, handles or emoji, at most 400 characters."""
    cleaned = _EMOJI.sub(" ", _HANDLE.sub(" ", _URL.sub(" ", text)))
    cleaned = " ".join(_HASH.sub(r"\1", cleaned).split())
    if len(cleaned) <= QUERY_LIMIT:
        return cleaned
    cut = cleaned[:QUERY_LIMIT]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


@dataclass(frozen=True)
class QueryPlan:
    queries: tuple[tuple[str, ...], ...]  # per claim, in claim order; never empty
    hypotheses: tuple[str | None, ...]  # per claim: the English text to check, or None
    by: str  # SENTENCE_BY, or the judge's name
    notice: str | None  # set when the judge was asked and the sentence was used instead


def fallback_notice(detail: str, status: int | None) -> str:
    """The run's notice for a judge that did not write the queries.

    A limit is named from the failure's status alone, the one rule the judge's own
    switch notice uses (``LIMIT_STATUSES``). A detail can mention "HTTP 429" without
    being one -- the primary's reason, quoted before a local model that failed for
    another reason -- and then no model answered, which is what the notice says.
    """
    head = LLM_LIMIT if status in LIMIT_STATUSES else LLM_UNAVAILABLE
    return f"{head} ({detail})"


def plan_queries(texts: Sequence[str], judge: Judge | None) -> QueryPlan:
    english = tuple(looks_english(text) for text in texts)
    fallback = tuple((sentence_query(text),) for text in texts)
    # An English claim is checked as written; another language only once translated.
    as_written = tuple(text if eng else None for text, eng in zip(texts, english, strict=True))
    if judge is None or not texts:
        return QueryPlan(fallback, as_written, SENTENCE_BY, None)
    written = judge.queries(texts)
    if judge.unavailable:
        return QueryPlan(
            fallback, as_written, SENTENCE_BY, fallback_notice(judge.detail, judge.status)
        )
    queries = tuple(
        (written[i].queries if i in written and written[i].queries else fallback[i])
        for i in range(len(texts))
    )
    hypotheses = tuple(
        _hypothesis(texts[i], english[i], written[i].english if i in written else None)
        for i in range(len(texts))
    )
    return QueryPlan(queries, hypotheses, judge.name, None)


def _hypothesis(text: str, english: bool, translated: str | None) -> str | None:
    """What the models check for one claim, once a judge has answered.

    The judge's English wins unless the claim is *certainly* English (round 2):
    ``looks_english`` lets a short foreign sentence through as English ("Las vacunas
    causan autismo."), and checking it as written would hand Spanish to English-only
    models. An echo, or no valid translation at all, keeps the claim when it reads as
    English and leaves nothing to check when it does not.
    """
    if translated is not None and not certainly_english(text):
        checked = _translation(text, translated)
        if checked is not None and checked.strip() != text.strip():
            return checked
    return text if english else None


# The worked examples of ``prompts/queries.md``: each example answer's English, and
# the example claim it translates. A model that copies an example into its answer
# would otherwise have the example checked as if it were the claim (ledger T11).
PROMPT_EXAMPLES = {
    "The company went bankrupt.": "Şirket battı.",  # noqa: RUF001 - real Turkish letters
    "The Eiffel Tower is in Paris.": "The Eiffel Tower is in Paris.",
}


def _folded(text: str) -> str:
    """One spelling for a comparison: case, spacing and the closing stop ignored."""
    return " ".join(text.casefold().split()).rstrip(".!?")


_EXAMPLES = {_folded(english): _folded(claim) for english, claim in PROMPT_EXAMPLES.items()}


def _translation(original: str, english: str) -> str | None:
    """The judge's English text, or ``None`` when it is not a translation at all.

    A model can copy the claim back unchanged (a local qwen with thinking off does,
    measured 2026-09-28), answer in the claim's own language, or copy one of the
    prompt's examples. Any of those handed to the English-only models would be
    checked as if it were the claim; ``None`` sends the claim down the
    no-translation path instead (``language not supported``). An echo of a claim
    that already reads as English is the right answer, and is kept.
    """
    if not english:
        return None
    if english.strip() == original.strip():
        return english if looks_english(original) else None
    if not looks_english(english):
        return None
    example = _EXAMPLES.get(_folded(english))
    if example is not None and example != _folded(original):
        return None
    return english
