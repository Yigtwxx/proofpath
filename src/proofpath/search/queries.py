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

from proofpath.search.language import looks_english

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


def fallback_notice(detail: str) -> str:
    head = LLM_LIMIT if "HTTP 429" in detail else LLM_UNAVAILABLE
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
        return QueryPlan(fallback, as_written, SENTENCE_BY, fallback_notice(judge.detail))
    queries = tuple(
        (written[i].queries if i in written and written[i].queries else fallback[i])
        for i in range(len(texts))
    )
    hypotheses = tuple(
        texts[i]
        if english[i]
        else (_translation(texts[i], written[i].english) if i in written else None)
        for i in range(len(texts))
    )
    return QueryPlan(queries, hypotheses, judge.name, None)


def _translation(original: str, english: str) -> str | None:
    """The judge's English text, or ``None`` when it is not a translation at all.

    A model can copy the claim back unchanged (a local qwen with thinking off does,
    measured 2026-09-28) or answer in the claim's own language. Either one handed to
    the English-only models would be checked as if it were English; ``None`` sends
    the claim down the no-translation path instead (``language not supported``).
    """
    if not english:
        return None
    if english.strip() == original.strip() or not looks_english(english):
        return None
    return english
