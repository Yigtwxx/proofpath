"""Phase 10 harness: measure proofpath end to end on AVeriTeC dev (spec section 14).

Usage:
    uv run python scripts/eval_averitec.py [--limit 100] [--browser | --no-browser]
        [--judge [--judge-nei] [--no-fallback]] [--fresh] [--sleep 1.0] [--resume]
        [--search] [--out docs/eval/<date>-averitec.md]

SciFact measures retrieval and entailment over abstracts we are handed. AVeriTeC
measures the whole product: a real-world claim, its real source pages, fetched over
the real ladder. That means the run is mostly network, so it is slow, it is polite
(``--sleep``), and it is resumable — every claim's row is written to the cache as
soon as it is decided, and ``--resume`` skips the ids already there.

Two accuracies are reported, never one. AVeriTeC has a fourth class,
"Conflicting Evidence/Cherrypicking", that proofpath has no verdict for: the 3-way
number leaves those rows out of the denominator, the 4-way number counts them as
wrong, and the gap between them is how much of the dataset we structurally cannot
answer. Source coverage is reported per state, never collapsed (product rule 6): a
run that reached nothing must not read like a run that reached everything.

Every source keeps the ladder step that read it and, under ``--judge``, the judge's
opinion when its verdict was in the escalation band. Two more numbers come from
those rows: the accuracy with the browser's pages dropped (what a run without it
would have read, less whatever Wayback would have rescued), and a hypothetical one
in which the judge's opinions decide the low band. The product never does the
second (spec section 11.1); the report says so.

``--judge-nei`` measures what the judge would do with the NEIs (OPEN-ITEMS 20.13):
every read source whose verdict is NEI is also sent, with the passage the models came
closest to deciding on (``pipeline.closest``), and the hypothetical column gains a
second number in which those opinions vote too. The judge's quote is checked
(OPEN-ITEMS 20.9), so the report counts the opinions that check dropped.
``--no-fallback`` keeps every opinion from the configured model: a 429 is waited out
instead of handed to the local one.

The live half below ``main`` is run by hand. Importing this module touches neither
the network nor a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from proofpath import claims as claims_mod
from proofpath import ingest, pipeline, retrieval
from proofpath import judge as judge_mod
from proofpath.config import NliProfileName, SearchConfig, load_config
from proofpath.eval import averitec
from proofpath.fetch import Fetched
from proofpath.models import Label, Passage, Verdict
from proofpath.paths import cache_dir
from proofpath.polite import ProviderError
from proofpath.providers import NO_TEXT
from proofpath.report import LANGUAGE_UNSUPPORTED, SEARCH_UNAVAILABLE
from proofpath.search import Searcher, SearchKeyError, canonical, search_claim
from proofpath.search.queries import plan_queries
from proofpath.verify import Engine, escalates

if TYPE_CHECKING:
    from proofpath.judge import Judge

# The state a claim gets when the dataset itself names no page to read. It is not a
# fetch failure and must not be counted as one.
NO_SOURCE = "no source url"
# The state a source value gets when it is not a URL at all — the dev split's literal
# "Metadata" answers. Nothing was tried, so nothing failed: reporting it as a fetch
# outcome would invent an unreachable web source out of an annotation (product rule 2).
NOT_A_URL = "not a url"
# The host whose share of the source URLs the report has to state: a third of the dev
# split's evidence is already a snapshot, and that changes what the numbers measure.
ARCHIVE_HOST = "web.archive.org"
# The state a claim gets in search mode when the product would not have searched it:
# no sentence in it with a word to search for. Scored as NEI, never dropped.
NOT_SEARCHED = "not searched (no sentence to search)"
# What the report says was measured, one line (final review, Important 3).
GOLD_PATH = "gold source URLs, no search"
SENTENCE_PATH = "sentence queries, no judge"
RESULTS_NAME = "averitec_results.json"
SEARCH_RESULTS_NAME = "averitec_search_results.json"
# The ladder step the ablation drops (``fetch.STEP_NAMES``).
BROWSER_STEP = 3
# Who decided a claim's hypothetical judged label, when it was not a judge model.
DECIDED_BY_MODELS = "models (medium or high tier)"
NO_OPINION = "models (the judge gave no opinion)"
NOTHING_ESCALATED = "nothing escalated"
# A row from a results file written before sources were kept: only its label is known.
UNRECORDED = "not recorded (row has no per-source record)"
# The two kinds of escalation a source can have been sent to the judge under: the
# product's own band (``verify.escalates``), and the ``--judge-nei`` probe of an NEI.
ESCALATED = "escalated"
NEI_PROBE = "nei"
# Why the quote check dropped a judge's opinion, keyed by the judge's own note. Read
# off ``Judge.skipped`` by exact match on the judge's public wording, never by
# substring, so a note the judge writes for another reason is never counted here.
DROP_NO_QUOTE = "no quote"
DROP_TOO_SHORT = "a quoted part under two words"
DROP_NOT_IN_PASSAGE = "quote not in the passage"
_DROP_REASONS = {
    judge_mod.QUOTE_MISSING: DROP_NO_QUOTE,
    judge_mod.QUOTE_TOO_SHORT: DROP_TOO_SHORT,
    judge_mod.QUOTE_NOT_FOUND: DROP_NOT_IN_PASSAGE,
}


@dataclass(frozen=True)
class Source:
    """One fetchable source of one claim, as the run read it.

    ``step`` is the ladder step that read the page, or the last one tried when it
    failed; ``None`` for a cache hit, which carries no step. ``label`` is the models'
    verdict, ``None`` when nothing was read. ``escalated`` says the verdict sat in the
    band the product hands the judge (``verify.escalates``); ``probed`` says it was an
    NEI sent under ``--judge-nei`` with its closest passage. ``judge_label`` and
    ``judge_model`` are the judge's opinion of it, when one came back; ``dropped`` is
    why the quote check threw one away (``DROP_NO_QUOTE``, ``DROP_TOO_SHORT``,
    ``DROP_NOT_IN_PASSAGE``).

    ``probed`` and ``dropped`` are new on 2026-10-05 and default to what an older
    results file meant without them: no probe, nothing dropped.
    """

    url: str
    state: str
    step: int | None
    label: str | None
    score: float = 0.0
    tier: str | None = None
    escalated: bool = False
    judge_label: str | None = None
    judge_model: str | None = None
    probed: bool = False
    dropped: str | None = None

    @property
    def escalation(self) -> str | None:
        """Which kind of escalation sent this source to the judge, if any.

        Derived from the two flags rather than stored beside them, so it cannot
        disagree with them, and an older file's ``escalated`` keeps its meaning."""
        if self.escalated:
            return ESCALATED
        return NEI_PROBE if self.probed else None


@dataclass(frozen=True)
class Row:
    """One scored claim: what it was, what we said, and what the sources did.

    ``predicted`` is a ``Label`` value, or ``None`` when no source was fetched at
    all. The two are not the same thing — one is a verdict, the other is the absence
    of anything to base one on — so the distinction survives into the results file
    and is only flattened where the accuracy is computed.
    """

    claim_id: int
    gold: str
    predicted: str | None
    states: tuple[str, ...]  # one honesty/fetch state per source value
    urls: tuple[str, ...]  # the fetchable URLs the states came from, in fetch order
    # One per fetchable URL. Empty in results files written before 2026-10-05.
    sources: tuple[Source, ...] = ()


def _best(sources: Sequence[Source]) -> Source:
    """The strongest source, the first one on a tie, as ``_decide_claim`` always kept."""
    best = sources[0]
    for source in sources[1:]:
        if source.score > best.score:
            best = source
    return best


def _asserts(label: str | None) -> bool:
    return label is not None and label != Label.NEI.value


def product_label(sources: Sequence[Source]) -> str | None:
    """What the product says of the claim: its strongest assertion over every source,
    NEI when something was read and nothing asserted, ``None`` when nothing was read."""
    asserted = [source for source in sources if _asserts(source.label)]
    if asserted:
        return _best(asserted).label
    return Label.NEI.value if any(source.label is not None for source in sources) else None


def judged_label(sources: Sequence[Source], *, probes: bool = False) -> str | None:
    """The claim's label if the judge's opinions decided the low band.

    The product never does this (spec section 11.1): this is the column that says
    whether it should. A confident model verdict (medium or high, not escalated)
    still wins, as the escalation rule implies. Otherwise every escalated source
    votes with the judge's label, or with the models' label when the judge gave no
    opinion. An NEI vote abstains; the majority of the asserting votes wins, and a
    tie is NEI.

    ``probes`` lets the ``--judge-nei`` probes vote too, by the same rule. A probe's
    own label is NEI, so a probe the judge did not answer abstains: it can only move
    the claim through an opinion that passed the quote check.
    """
    confident = [s for s in sources if _asserts(s.label) and not s.escalated]
    if confident:
        return _best(confident).label
    votes = [s.judge_label or s.label for s in sources if s.escalated or (probes and s.probed)]
    tally = Counter(vote for vote in votes if _asserts(vote)).most_common()
    if tally:
        if len(tally) > 1 and tally[0][1] == tally[1][1]:
            return Label.NEI.value
        return tally[0][0]
    return Label.NEI.value if any(source.label is not None for source in sources) else None


def judge_decider(sources: Sequence[Source]) -> str:
    """Who decided ``judged_label``: the models, or the judge model(s) whose vote won.

    When the vote ends in NEI (every vote abstained, or a tie) no vote won, and every
    escalated source's answerer is named: together they are what left it at NEI."""
    if any(_asserts(s.label) and not s.escalated for s in sources):
        return DECIDED_BY_MODELS
    escalated = [s for s in sources if s.escalated]
    if not escalated:
        return NOTHING_ESCALATED
    winner = judged_label(sources)
    voters = [s for s in escalated if (s.judge_label or s.label) == winner and _asserts(winner)]
    return " + ".join(sorted({s.judge_model or NO_OPINION for s in voters or escalated}))


def without_browser(sources: Sequence[Source]) -> tuple[Source, ...]:
    """The sources as a run without the browser would have them: the pages the browser
    read are unread. A cache hit has no step, so it is kept (the report counts those)."""
    return tuple(
        replace(
            source,
            label=None,
            escalated=False,
            judge_label=None,
            judge_model=None,
            probed=False,
            dropped=None,
        )
        if source.step == BROWSER_STEP and source.label is not None
        else source
        for source in sources
    )


def _is_archive(url: str) -> bool:
    """Whether a source URL is a Wayback snapshot rather than the live page."""
    return (urlsplit(url).hostname or "").lower() == ARCHIVE_HOST


def state_for(fetched: Fetched) -> str:
    """The coverage state one fetch earned.

    A fetch that reached the page and extracted nothing from it is not a clean
    ``ok``: nothing was read, so nothing could be checked against. It gets
    proofpath's own wording for that case, so the coverage table separates the pages
    we read from the pages we merely arrived at (product rule 6).
    """
    if fetched.ok and not fetched.text.strip():
        return NO_TEXT
    return fetched.outcome.value


@dataclass(frozen=True)
class Searched:
    """What the product's search did with one claim.

    ``state`` is ``None`` when it searched; otherwise it is the state the product
    would have reported instead, and the claim is scored as NEI rather than dropped.
    ``text`` is what the models check: the searched sentences, in English.
    """

    urls: tuple[str, ...]
    text: str
    state: str | None = None


def search_urls(
    searcher: Searcher,
    claim: averitec.Claim,
    limit: int,
    *,
    judge: Judge | None = None,
    max_claims: int = SearchConfig().max_claims,
) -> Searched:
    """What the product's evidence search would read for this claim, minus the answer key.

    The product's own path, step for step (final review, Important 3): the claim is
    read as pasted text, ``checkworthy`` picks the sentences, ``plan_queries`` writes
    the queries and applies the language gate, and ``search_claim`` asks the provider
    for exactly ``limit`` hits per query, as ``verify`` does.

    The one addition is the leakage exclusion: the whole fact-checking site is
    excluded, not just the one article, because a site that rated the claim quotes its
    own rating on every related page (the leakage the §17.1 gate must not measure). It
    is passed to the helper as one more excluded host, which matches across
    subdomains both ways, so a rating on ``factcheck.afp.com`` also rules out
    ``www.afp.com``. That errs towards a lower score, never a leaked one.
    """
    checker = _site(claim.fact_check_url)
    exclude = (checker,) if checker else ()
    document = ingest.from_text(claim.text, name="averitec", kind="text")
    chosen = claims_mod.checkworthy(document, limit=max_claims)
    if not chosen.claims:
        return Searched((), claim.text, NOT_SEARCHED)
    plan = plan_queries([sentence.text for sentence in chosen.claims], judge)
    urls: list[str] = []
    seen: set[str] = set()
    checked: list[str] = []
    for queries, hypothesis in zip(plan.queries, plan.hypotheses, strict=True):
        if hypothesis is None:
            continue  # not English and not translated: the product does not search it
        checked.append(hypothesis)
        try:
            hits = search_claim(queries, searcher, limit, exclude=exclude)
        except SearchKeyError:
            # Not this claim's problem: every claim would fail the same way, so the
            # run stops (``main``) rather than scoring them all unverified.
            raise
        except ProviderError:
            # Per claim: this one is unverified, the run goes on (product rule 2).
            return Searched((), claim.text, SEARCH_UNAVAILABLE)
        for hit in hits:
            key = canonical(hit.url)
            if key not in seen:
                seen.add(key)
                urls.append(hit.url)
    if not checked:
        return Searched((), claim.text, LANGUAGE_UNSUPPORTED)
    return Searched(tuple(urls), " ".join(checked))


def unsearched_row(claim: averitec.Claim, state: str) -> Row:
    """A claim the product would not have searched, as the product would report it:
    no verdict (scored as NEI), and the state that says why."""
    return Row(claim_id=claim.id, gold=claim.label, predicted=None, states=(state,), urls=())


def measured_path(judge: Judge | None) -> str:
    """The search path a run measured, in one line for the report."""
    return SENTENCE_PATH if judge is None else f"judge queries ({judge.name})"


def _site(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.")


@dataclass(frozen=True)
class Ablation:
    """The run with the browser's pages dropped. A page the browser read was never
    offered to the Wayback step, which a run without the browser tries next, so this
    is a lower bound on the pages such a run reads -- not on its accuracy, which can
    move either way (a wrong verdict dropped can turn a claim into a right NEI)."""

    read_sources: int
    browser_sources: int  # of ``read_sources``, read at ``BROWSER_STEP``
    unknown_step: int  # of ``read_sources``, cache hits: kept, since no step is known
    unrecorded: int  # rows with URLs and no per-source record: scored as the product did
    accuracy_3way: float


@dataclass(frozen=True)
class ProbedScore:
    """The hypothetical column's second number: the product's band plus the NEI probes
    (``--judge-nei``), and how it stands against the product claim by claim.

    ``wrong_way`` is (Refuted called SUPPORTED, Supported called REFUTED): the errors
    that assert the opposite of the truth, which an accuracy gain can hide. The bar in
    the spec (2026-10-05 section 2.7) is no more false claims called SUPPORTED than the
    product, so the product's own pair is carried beside it. ``fixed`` and ``broken``
    are the discordant claims of the McNemar test against the product's outcome.
    """

    accuracy_3way: float
    per_label: dict[str, tuple[int, int]]  # gold label -> (n, correct)
    probed: int  # NEI sources sent with their closest passage
    unanswered: int  # of those, how many got no opinion back
    wrong_way: tuple[int, int]
    product_wrong_way: tuple[int, int]
    fixed: int  # claims the product got wrong and this got right
    broken: int  # claims the product got right and this got wrong
    p_value: float  # exact two-sided McNemar over ``fixed`` and ``broken``


@dataclass(frozen=True)
class JudgedScore:
    """The hypothetical column: what ``judged_label`` would have scored."""

    accuracy_3way: float
    per_label: dict[str, tuple[int, int]]  # gold label -> (n, correct)
    escalated: int  # sources in the judge's band
    unanswered: int  # of those, how many got no opinion back
    # (judge model, escalation kind) -> opinions it gave: the NEI probes counted
    # apart from the product's band, so neither number borrows the other's answers.
    opinions: Counter[tuple[str, str]]
    deciders: dict[str, tuple[int, int]]  # ``judge_decider`` -> (3-way n, correct)
    unrecorded: int  # rows with URLs and no per-source record: scored as the product did
    # (escalation kind, drop reason) -> opinions the quote check dropped (OPEN-ITEMS 20.9)
    drops: Counter[tuple[str, str]] = field(default_factory=Counter)
    probed: ProbedScore | None = None  # ``None`` unless the run had ``--judge-nei``


@dataclass(frozen=True)
class AveritecResult:
    n: int
    # The 3-way denominator, carried rather than recomputed downstream: the report
    # must not derive the number it prints from a second copy of this rule.
    counted: int
    accuracy_3way: float
    accuracy_4way: float
    majority_baseline: float
    per_label: dict[str, tuple[int, int]]  # gold label -> (n, correct)
    coverage: Counter[str]  # source state -> how many values ended there
    archive_urls: int  # of ``total_urls``, how many were web.archive.org snapshots
    total_urls: int
    ablation: Ablation | None = None  # ``None`` when no row carries its sources
    judged: JudgedScore | None = None  # ``None`` unless the run had a judge


def _three_way(
    rows: Sequence[Row], label_of: Callable[[Row], str | None]
) -> tuple[float, dict[str, tuple[int, int]]]:
    """3-way accuracy and per-gold-label (n, correct) for another way of labelling the
    same rows, scored by the same rule as ``score``: no label is NEI."""
    per_label: dict[str, tuple[int, int]] = {}
    counted = correct = 0
    for row in rows:
        gold = averitec.to_label(row.gold)
        if gold is None:
            continue
        predicted = label_of(row)
        right = (Label(predicted) if predicted is not None else Label.NEI) is gold
        seen, hits = per_label.get(row.gold, (0, 0))
        per_label[row.gold] = (seen + 1, hits + int(right))
        counted += 1
        correct += int(right)
    return (correct / counted if counted else 0.0), per_label


def _unrecorded(row: Row) -> bool:
    """Written before sources were kept (a ``--resume`` of an older file): the row has
    URLs but no per-source record, so the product's own label is all that is known."""
    return not row.sources and bool(row.urls)


def _ablation(rows: Sequence[Row]) -> Ablation | None:
    if not any(row.sources for row in rows):
        return None
    read = [s for row in rows for s in row.sources if s.label is not None]
    accuracy, _ = _three_way(
        rows,
        lambda row: product_label(without_browser(row.sources)) if row.sources else row.predicted,
    )
    return Ablation(
        read_sources=len(read),
        browser_sources=sum(1 for s in read if s.step == BROWSER_STEP),
        unknown_step=sum(1 for s in read if s.step is None),
        unrecorded=sum(1 for row in rows if _unrecorded(row)),
        accuracy_3way=accuracy,
    )


def _judged_label_of(row: Row) -> str | None:
    return judged_label(row.sources) if row.sources else row.predicted


def _probes_label_of(row: Row) -> str | None:
    return judged_label(row.sources, probes=True) if row.sources else row.predicted


def _is_right(label: str | None, gold: Label) -> bool:
    return (Label(label) if label is not None else Label.NEI) is gold


def _wrong_way(rows: Sequence[Row], label_of: Callable[[Row], str | None]) -> tuple[int, int]:
    """(Refuted called SUPPORTED, Supported called REFUTED) under one way of labelling."""
    refuted_supported = supported_refuted = 0
    for row in rows:
        gold = averitec.to_label(row.gold)
        predicted = label_of(row)
        refuted_supported += int(gold is Label.REFUTED and predicted == Label.SUPPORTED.value)
        supported_refuted += int(gold is Label.SUPPORTED and predicted == Label.REFUTED.value)
    return refuted_supported, supported_refuted


def mcnemar_exact(b: int, c: int) -> float:
    """The exact two-sided McNemar p-value for ``b`` and ``c`` discordant pairs.

    Exact rather than the chi-squared approximation: a hundred claims leave a handful
    of discordant pairs, which is where the approximation is worst. Under the null
    each discordant claim is a fair coin, so ``min(b, c)`` is a Binomial(b + c, 1/2)
    tail, doubled and capped at 1. No discordant pair at all is no evidence: 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (1 << n)
    return min(1.0, 2.0 * tail)


def _probed(rows: Sequence[Row]) -> ProbedScore:
    accuracy, per_label = _three_way(rows, _probes_label_of)
    probes = [s for row in rows for s in row.sources if s.probed]
    fixed = broken = 0
    for row in rows:
        gold = averitec.to_label(row.gold)
        if gold is None:
            continue
        product = _is_right(row.predicted, gold)
        probed = _is_right(_probes_label_of(row), gold)
        fixed += int(probed and not product)
        broken += int(product and not probed)
    return ProbedScore(
        accuracy_3way=accuracy,
        per_label=per_label,
        probed=len(probes),
        unanswered=sum(1 for s in probes if s.judge_label is None),
        wrong_way=_wrong_way(rows, _probes_label_of),
        product_wrong_way=_wrong_way(rows, lambda row: row.predicted),
        fixed=fixed,
        broken=broken,
        p_value=mcnemar_exact(broken, fixed),
    )


def _judged(rows: Sequence[Row], *, probes: bool = False) -> JudgedScore:
    accuracy, per_label = _three_way(rows, _judged_label_of)
    escalated = [s for row in rows for s in row.sources if s.escalated]
    deciders: dict[str, tuple[int, int]] = {}
    for row in rows:
        gold = averitec.to_label(row.gold)
        if gold is None:
            continue
        predicted = _judged_label_of(row)
        right = (Label(predicted) if predicted is not None else Label.NEI) is gold
        decider = UNRECORDED if _unrecorded(row) else judge_decider(row.sources)
        seen, hits = deciders.get(decider, (0, 0))
        deciders[decider] = (seen + 1, hits + int(right))
    return JudgedScore(
        accuracy_3way=accuracy,
        per_label=per_label,
        escalated=len(escalated),
        unanswered=sum(1 for s in escalated if s.judge_label is None),
        opinions=Counter(
            (s.judge_model, s.escalation)
            for row in rows
            for s in row.sources
            if s.judge_model is not None and s.escalation is not None
        ),
        deciders=deciders,
        unrecorded=sum(1 for row in rows if _unrecorded(row)),
        # A dropped opinion was asked for, so its source always has an escalation.
        drops=Counter(
            (s.escalation, s.dropped)
            for row in rows
            for s in row.sources
            if s.dropped is not None and s.escalation is not None
        ),
        probed=_probed(rows) if probes else None,
    )


def score(rows: Sequence[Row], *, judge: bool = False, probes: bool = False) -> AveritecResult:
    """Turn decided rows into the numbers the report prints.

    ``probes`` adds the hypothetical column's second number (``--judge-nei``); it means
    nothing without ``judge``.

    A row with no prediction counts as NEI: having fetched nothing, NEI is the only
    thing the run honestly asserted. A row whose gold label is Conflicting can never
    be right — ``to_label`` gives it no verdict to match — so it is wrong in the
    4-way number and absent from the 3-way one.
    """
    per_label: dict[str, tuple[int, int]] = {}
    coverage: Counter[str] = Counter()
    three_way_golds: list[str] = []
    correct_3way = 0
    correct_4way = 0
    archive_urls = 0
    total_urls = 0
    for row in rows:
        coverage.update(row.states)
        total_urls += len(row.urls)
        archive_urls += sum(1 for url in row.urls if _is_archive(url))
        gold = averitec.to_label(row.gold)
        predicted = Label(row.predicted) if row.predicted is not None else Label.NEI
        correct = gold is not None and predicted is gold
        seen, right = per_label.get(row.gold, (0, 0))
        per_label[row.gold] = (seen + 1, right + int(correct))
        correct_4way += int(correct)
        if gold is not None:
            three_way_golds.append(row.gold)
            correct_3way += int(correct)
    counted = len(three_way_golds)
    majority = max(Counter(three_way_golds).values(), default=0)
    return AveritecResult(
        n=len(rows),
        counted=counted,
        accuracy_3way=correct_3way / counted if counted else 0.0,
        accuracy_4way=correct_4way / len(rows) if rows else 0.0,
        majority_baseline=majority / counted if counted else 0.0,
        per_label=per_label,
        coverage=coverage,
        archive_urls=archive_urls,
        total_urls=total_urls,
        ablation=_ablation(rows),
        judged=_judged(rows, probes=probes) if judge else None,
    )


def _md_row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_report(
    result: AveritecResult,
    *,
    date: str,
    limit: int,
    path: str = GOLD_PATH,
    browser: bool | None = None,
    fresh: bool = False,
    judge: str = "",
    judge_nei: bool = False,
    no_fallback: bool = False,
) -> str:
    """The markdown skeleton. ``## Notes`` is left for the controller to fill in.

    ``browser`` is ``None`` when the run left the browser to the config, so the report
    claims nothing about it; ``judge`` is the judge's name, empty when there was none.
    ``judge_nei`` and ``no_fallback`` are the run's ``--judge-nei`` and
    ``--no-fallback``, stated in the header so two setups' reports cannot be confused.
    """
    # ``result.counted`` is the denominator the 3-way accuracy and its baseline are
    # read over. Printed beside `n`, because "0.80 over 100 claims" would otherwise
    # claim a coverage the number does not have when some claims were never scorable.
    lines = [
        f"# AVeriTeC dev — {date}",
        "",
        f"- claims: {result.n}  (limit={limit or 'none'})",
        f"- measured path: {path}",
    ]
    if browser is not None:
        lines.append(f"- browser step: {'allowed' if browser else 'not used'}")
    lines.append(f"- fetch: {'live, cache bypassed' if fresh else 'cache allowed'}")
    if judge:
        role = (
            "its opinions are the hypothetical column below; the verdicts are the models'"
            if path == GOLD_PATH
            else "writes the search queries; its opinions are the hypothetical column below"
        )
        lines.append(f"- judge: {judge} ({role})")
        if judge_nei:
            lines.append(
                "- NEI probes: on (every read NEI source was also sent to the judge, "
                "with its closest passage)"
            )
        if no_fallback:
            lines.append("- judge fallback: off (every opinion is from the configured model)")
    lines += [
        f"- dataset: `{averitec.URL}`  (sha256 {averitec.SHA256[:12]}…)",
        "- one run of the whole product: real claims, real source pages, real fetch ladder.",
        "",
        "## Headline",
        "",
        f"3-way accuracy **{result.accuracy_3way:.3f}** vs majority baseline "
        f"{result.majority_baseline:.3f}, over {result.counted} of {result.n} claims.",
        "",
        "The 3-way number excludes the Conflicting Evidence/Cherrypicking rows, which "
        "proofpath has no verdict for; counting them as wrong gives a 4-way accuracy of "
        f"{result.accuracy_4way:.3f}.",
        "",
        "## Per label",
        "",
        "| label | n | correct | accuracy |",
        "|---|---|---|---|",
    ]
    # Known labels first, in the dataset's own order, then anything unexpected — so a
    # label the loader has never seen shows up rather than being quietly dropped.
    order = [label for label in averitec.LABELS if label in result.per_label]
    order += [label for label in result.per_label if label not in averitec.LABELS]
    for label in order:
        seen, right = result.per_label[label]
        share = f"{right / seen:.3f}" if seen else "—"
        lines.append(_md_row(label, seen, right, share))
    lines.extend(["", "## Source coverage", "", "| state | count |", "|---|---|"])
    # Every state, most common first, none collapsed into another (product rule 6).
    for state, count in result.coverage.most_common():
        lines.append(_md_row(state, count))
    if not result.coverage:
        lines.append(_md_row("no sources attempted", 0))
    # What the states were measured on. A third of AVeriTeC's evidence is already a
    # Wayback snapshot, which is a different thing to reach than the live page, so the
    # report says how much of its own coverage came from one.
    lines.append("")
    if result.total_urls:
        percent = 100.0 * result.archive_urls / result.total_urls
        lines.append(
            f"{result.archive_urls} of {result.total_urls} source URLs are "
            f"{ARCHIVE_HOST} snapshots ({percent:.1f} %)."
        )
    else:
        lines.append(f"No source URLs were measured, so no {ARCHIVE_HOST} share applies.")
    if result.ablation is not None:
        lines.extend(_ablation_lines(result.ablation))
    if result.judged is not None:
        lines.extend(_judged_lines(result.judged, product=result.accuracy_3way))
    lines.extend(["", "## Notes", "", "<!-- filled in by hand after the run -->", ""])
    return "\n".join(lines) + "\n"


def _ablation_lines(ablation: Ablation) -> list[str]:
    lines = [
        "",
        "## Without the browser",
        "",
        f"{ablation.browser_sources} of {ablation.read_sources} read sources were read by "
        f"the browser step. Treating them as unread gives a 3-way accuracy of "
        f"**{ablation.accuracy_3way:.3f}**.",
        "",
        "Those pages never reached the Wayback step, which a run without the browser "
        "tries next, so this is a lower bound on the pages such a run reads. It is not a "
        "bound on its accuracy: dropping a page can lose a right verdict or a wrong one.",
    ]
    if ablation.unknown_step:
        lines.extend(
            [
                "",
                f"{ablation.unknown_step} read sources came from the cache, so the step "
                "that read them is unknown; they are kept as read.",
            ]
        )
    if ablation.unrecorded:
        lines.extend(["", _unrecorded_line(ablation.unrecorded)])
    return lines


def _unrecorded_line(count: int) -> str:
    return (
        f"{count} rows have no per-source record (written before sources were kept); "
        "they are scored with the product's own label."
    )


def _judged_lines(judged: JudgedScore, *, product: float) -> list[str]:
    lines = [
        "",
        "## Judge (hypothetical)",
        "",
        "The product never lets the judge change a verdict (spec section 11.1). This "
        "column is what the accuracy would be if it did: a medium or high model verdict "
        "still wins; otherwise the judge's opinions on the escalated sources decide, by "
        "majority of the asserting votes (an NEI opinion abstains), a tie being NEI. An "
        "escalation the judge did not answer votes with the models' label.",
        "",
        f"3-way accuracy **{judged.accuracy_3way:.3f}**, against {product:.3f} for the "
        f"product. {judged.escalated} sources were escalated; {judged.unanswered} got "
        "no opinion back."
        + (f" {_unrecorded_line(judged.unrecorded)}" if judged.unrecorded else ""),
        "",
        "| label | n | correct | accuracy |",
        "|---|---|---|---|",
    ]
    for label in [label for label in averitec.LABELS if label in judged.per_label]:
        seen, right = judged.per_label[label]
        lines.append(_md_row(label, seen, right, f"{right / seen:.3f}" if seen else "—"))
    lines.extend(["", "| judge model | escalation | opinions |", "|---|---|---|"])
    for (model, kind), count in judged.opinions.most_common():
        lines.append(_md_row(model, kind, count))
    if not judged.opinions:
        lines.append(_md_row("none", "—", 0))
    lines.extend(["", "| decided by | n | correct | accuracy |", "|---|---|---|---|"])
    for decider, (seen, right) in sorted(judged.deciders.items()):
        lines.append(_md_row(decider, seen, right, f"{right / seen:.3f}" if seen else "—"))
    if judged.probed is not None:
        lines.extend(_probed_lines(judged.probed, product=product))
    lines.extend(_drop_lines(judged.drops))
    return lines


def _wrong_way_text(pair: tuple[int, int]) -> str:
    return f"{sum(pair)} ({pair[0]} + {pair[1]})"


def _probed_lines(probed: ProbedScore, *, product: float) -> list[str]:
    """The second hypothetical number. Reading it (spec 2026-10-05 section 2.7): it
    is a case for escalating NEIs only if the gain clears the noise (p < 0.05) and it
    calls no more false claims supported than the product does."""
    lines = [
        "",
        "### Band plus NEI probes",
        "",
        "`--judge-nei` also sent every read source whose verdict was NEI to the judge, "
        "as NEI at low tier, with the passage the models came closest to deciding on. "
        "Here those opinions vote too, by the same rule; a probe the judge did not "
        "answer abstains.",
        "",
        f"3-way accuracy **{probed.accuracy_3way:.3f}**, against {product:.3f} for the "
        f"product. {probed.probed} NEI sources were probed; {probed.unanswered} got no "
        "opinion back.",
        "",
        f"Against the product, claim by claim: {probed.fixed} fixed, {probed.broken} "
        f"broken (exact McNemar, two-sided p = {probed.p_value:.3f}).",
        "",
        "Refuted called SUPPORTED plus Supported called REFUTED, the "
        f"wrong way: {_wrong_way_text(probed.wrong_way)} with the probes, against "
        f"{_wrong_way_text(probed.product_wrong_way)} for the product.",
        "",
        "| label | n | correct | accuracy |",
        "|---|---|---|---|",
    ]
    for label in [label for label in averitec.LABELS if label in probed.per_label]:
        seen, right = probed.per_label[label]
        lines.append(_md_row(label, seen, right, f"{right / seen:.3f}" if seen else "—"))
    return lines


def _drop_lines(drops: Counter[tuple[str, str]]) -> list[str]:
    lines = [
        "",
        "### Dropped by the quote check",
        "",
        "A SUPPORTED or REFUTED opinion whose quote is not in the passage it was shown "
        "is dropped (OPEN-ITEMS 20.9), and the source then counts as unanswered above.",
        "",
        "| escalation | reason | dropped |",
        "|---|---|---|",
    ]
    for (kind, reason), count in sorted(drops.items()):
        lines.append(_md_row(kind, reason, count))
    if not drops:
        lines.append(_md_row("none", "—", 0))
    return lines


# --- live half: run by hand, not covered by tests ------------------------


def _setup_suffix(
    *, browser: bool, judge: bool, fresh: bool, judge_nei: bool = False, no_fallback: bool = False
) -> str:
    """What sets this run apart from the legacy setup, for its file names.

    ``-judge-nei`` takes the place of ``-judge``, which it requires, rather than
    following it: ``-judge-judge-nei`` would say the same thing twice."""
    judged = "-judge-nei" if judge_nei else "-judge" if judge else ""
    return (
        ("-browser" if browser else "")
        + judged
        + ("-nofallback" if no_fallback else "")
        + ("-fresh" if fresh else "")
    )


def _results_path(
    *,
    search: bool,
    browser: bool = False,
    judge: bool = False,
    fresh: bool = False,
    judge_nei: bool = False,
    no_fallback: bool = False,
) -> Path:
    """One file per run setup, so a resume does not mix rows from two setups. The
    legacy names stay the runs made without ``--browser``, ``--judge`` or ``--fresh``;
    they are shared by ``--no-browser`` and by a run that leaves the browser to the
    config, as they always were, so that older files can still be resumed."""
    name = Path(SEARCH_RESULTS_NAME if search else RESULTS_NAME)
    suffix = _setup_suffix(
        browser=browser, judge=judge, fresh=fresh, judge_nei=judge_nei, no_fallback=no_fallback
    )
    return cache_dir() / "datasets" / f"{name.stem}{suffix}{name.suffix}"


def _report_path(
    *,
    day: str,
    search: bool,
    browser: bool,
    judge: bool,
    fresh: bool,
    judge_nei: bool = False,
    no_fallback: bool = False,
) -> Path:
    """The default ``--out``: one report per run setup, so two runs on one day keep both."""
    suffix = ("-search" if search else "") + _setup_suffix(
        browser=browser, judge=judge, fresh=fresh, judge_nei=judge_nei, no_fallback=no_fallback
    )
    return Path("docs/eval") / f"{day}-averitec{suffix}.md"


def _load_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Row(
            claim_id=int(item["claim_id"]),
            gold=str(item["gold"]),
            predicted=None if item["predicted"] is None else str(item["predicted"]),
            states=tuple(str(state) for state in item["states"]),
            urls=tuple(str(url) for url in item["urls"]),
            sources=tuple(Source(**source) for source in item.get("sources", ())),
        )
        for item in raw
    ]


def _save_rows(path: Path, rows: Sequence[Row]) -> None:
    """Write the whole results file atomically, so a Ctrl-C never truncates it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        [
            {
                "claim_id": row.claim_id,
                "gold": row.gold,
                "predicted": row.predicted,
                "states": list(row.states),
                "urls": list(row.urls),
                "sources": [asdict(source) for source in row.sources],
            }
            for row in rows
        ],
        indent=1,
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _decide_claim(
    engine: Engine,
    claim: averitec.Claim,
    *,
    sleep: float,
    anonymous: bool = False,
    fresh: bool = False,
    judge: Judge | None = None,
    judge_nei: bool = False,
) -> Row:
    """Fetch every source of one claim and keep the strongest non-NEI verdict.

    ``anonymous`` is for pages the search found: fetched without the contact address,
    as the product fetches them (round 2). ``fresh`` climbs past the cache, so every
    source keeps the step that read it. ``judge`` is asked about the sources whose
    verdict sits in the product's escalation band; its opinions are recorded beside
    the verdicts and never change ``predicted`` (spec section 11.1). ``judge_nei`` also
    asks it about every read NEI source, shown the closest passage (OPEN-ITEMS 20.13)."""
    states: list[str] = []
    sources: list[Source] = []
    # What the judge is shown for each escalated source, by its index in ``sources``.
    escalated: dict[int, Verdict] = {}
    # The closest passage of each NEI source probed under ``judge_nei``, the same way.
    probes: dict[int, Passage] = {}
    for url in claim.source_urls:
        # Before every fetch, not between them: the ladder has no crawl delay of its
        # own, so this is the whole of the run's politeness and it has to hold across
        # claims too, not just within one.
        if sleep:
            time.sleep(sleep)
        fetched = engine.fetcher.fetch(url, anonymous=anonymous, use_cache=not fresh)
        state = state_for(fetched)
        states.append(state)
        step = None if fetched.from_cache else fetched.step
        if not fetched.ok or not fetched.text.strip():
            sources.append(Source(url=url, state=state, step=step, label=None))
            continue
        passages = [
            Passage(sentence, url, ordinal)
            for ordinal, sentence in enumerate(retrieval.split_sentences(fetched.text))
        ]
        # ``decide``'s verdict, unchanged, and the passage an NEI verdict drops: one
        # embedding and one scoring for both, so a probe never re-reads the page.
        verdict, nearest = pipeline.decide_closest(
            claim.text,
            passages,
            engine.get_embedder(),
            engine.get_scorer(),
            k=engine.k,
            thresholds=engine.thresholds,
        )
        index = len(sources)
        if escalates(verdict):
            escalated[index] = verdict
        elif judge is not None and judge_nei and verdict.label is Label.NEI and nearest is not None:
            # No closest passage means nothing scored above zero: there is no passage
            # to show, and asking without one is what rule 1 forbids.
            probes[index] = nearest
        # A source whose verdict is NEI was still read: it stays in the row, and
        # ``product_label`` keeps only assertions, each with the passage it rests on
        # (product rule 1, enforced by ``Verdict`` itself).
        sources.append(
            Source(
                url=url,
                state=state,
                step=step,
                label=verdict.label.value,
                score=verdict.score,
                tier=verdict.tier,
                escalated=escalates(verdict),
                probed=index in probes,
            )
        )
    if judge is not None and (escalated or probes):
        sources = _ask_judge(judge, claim.text, sources, escalated, probes)
    # One state per value the dataset gave, fetchable or not, so the coverage table
    # accounts for every annotation rather than only the ones we could try.
    states.extend([NOT_A_URL] * claim.non_urls)
    if not claim.source_urls and not claim.non_urls:
        states.append(NO_SOURCE)
    return Row(
        claim_id=claim.id,
        gold=claim.label,
        predicted=product_label(sources),
        states=tuple(states),
        urls=claim.source_urls,
        sources=tuple(sources),
    )


def _ask_judge(
    judge: Judge,
    claim: str,
    sources: list[Source],
    escalated: dict[int, Verdict],
    probes: dict[int, Passage] | None = None,
) -> list[Source]:
    """The judge's opinion on each escalated source, shown exactly what ``verify``
    shows it: the claim, the passage, the models' verdict and its tier.

    A probe is shown its closest passage as an NEI at ``low``, which is what the
    product would send were it to escalate NEIs. Both kinds go in one ``review``, and
    an opinion the quote check dropped is recorded on its source by reason."""
    shown: dict[int, judge_mod.JudgeItem] = {}
    for index, verdict in escalated.items():
        if verdict.passage is not None:  # ``escalates`` already refused the others
            shown[index] = judge_mod.JudgeItem(
                id=f"s{index}",
                claim=claim,
                passage=verdict.passage.text,
                verdict=verdict.label,
                tier=verdict.tier,
            )
    for index, passage in (probes or {}).items():
        shown[index] = judge_mod.JudgeItem(
            id=f"s{index}", claim=claim, passage=passage.text, verdict=Label.NEI, tier="low"
        )
    opinions = judge.review([shown[index] for index in sorted(shown)])
    # Read straight after the call: ``review`` resets ``skipped`` on every call, so
    # these are this claim's notes and no other's.
    dropped = _drops(judge.skipped)
    answered = list(sources)
    for index in shown:
        opinion = opinions.get(f"s{index}")
        if opinion is not None:
            answered[index] = replace(
                sources[index], judge_label=opinion.label.value, judge_model=opinion.model
            )
        elif f"s{index}" in dropped:
            answered[index] = replace(sources[index], dropped=dropped[f"s{index}"])
    return answered


def _drops(skipped: Sequence[str]) -> dict[str, str]:
    """Item id -> why the quote check dropped its opinion, from ``Judge.skipped``.

    The judge writes ``"<id>: <reason>"`` for a quote it refused; every other note is
    about something else (no JSON, an unknown id, no rationale) and is left out."""
    found: dict[str, str] = {}
    for note in skipped:
        ident, separator, reason = note.partition(": ")
        if separator and reason in _DROP_REASONS:
            found[ident] = _DROP_REASONS[reason]
    return found


NLI_PROFILE: NliProfileName = "default"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="first N dev claims")
    browser_flags = parser.add_mutually_exclusive_group()
    browser_flags.add_argument(
        "--no-browser", action="store_true", help="never use the browser step of the ladder"
    )
    browser_flags.add_argument(
        "--browser",
        action="store_true",
        help="allow the browser step whatever the config says (installs it if missing)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="ask the configured judge about the escalated verdicts (a hypothetical column)",
    )
    parser.add_argument(
        "--judge-nei",
        action="store_true",
        help="also send every read NEI source to the judge with its closest passage "
        "(requires --judge; a second hypothetical number)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="never switch the judge to its local fallback, so every opinion is from "
        "the configured model (requires --judge; sets judge.fallback = off for this run)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="fetch every page live, past the cache, so each source keeps its ladder step",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between fetches")
    parser.add_argument("--resume", action="store_true", help="skip claim ids already scored")
    parser.add_argument(
        "--search",
        action="store_true",
        help="read what the evidence search finds instead of the gold URLs",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.judge_nei and not args.judge:
        # A usage error, not a silent no-op: a run asked to probe the NEIs that had no
        # judge to probe them with would report a column it never measured.
        parser.error("--judge-nei requires --judge")
    if args.no_fallback and not args.judge:
        # The same refusal: with no judge there is no fallback to turn off, and the
        # "-nofallback" file name would label a run that is the plain one.
        parser.error("--no-fallback requires --judge")
    setup = {
        "browser": args.browser,
        "judge": args.judge,
        "fresh": args.fresh,
        "judge_nei": args.judge_nei,
        "no_fallback": args.no_fallback,
    }

    results = _results_path(search=args.search, **setup)
    if results.exists() and not args.resume:
        # The file is the only record of a run that costs hours of network; a fresh
        # run must not quietly replace it. Checked before anything is downloaded, so
        # the refusal costs nothing. The user picks: continue it, or delete it.
        print(
            f"error     {results} already holds a run.\n"
            f"          Pass --resume to continue it, or delete the file to start over.",
            file=sys.stderr,
        )
        return 2

    config = load_config()
    if args.no_fallback:
        # On the config the judge is built from, so ``build_client`` hands its
        # ``FallbackClient`` "off" and a 429 is waited out on the configured model.
        config = replace(config, judge=replace(config.judge, fallback=judge_mod.FALLBACK_OFF))
    key: judge_mod.ApiKey | None = None
    if args.judge:
        # Checked before anything is downloaded, as the CLI's ``--judge`` checks it: a
        # run told to measure the judge must not quietly measure the models alone.
        key = judge_mod.resolve_api_key(config.judge.api_key_env)
        if config.judge.api_key_env and key is None:
            print(
                f"error     {config.judge.api_key_env} is not set (put it in .env)",
                file=sys.stderr,
            )
            return 2
        if config.judge.provider == "gemini":
            print(f"note      {judge_mod.GEMINI_DATA_USE}", file=sys.stderr)

    dataset = averitec.ensure_downloaded(cache_dir())
    claims = averitec.load(dataset, limit=args.limit or None)
    print(f"dataset   averitec/dev  {len(claims)} claims  (sha256 {averitec.SHA256[:12]}…)")

    # ``stored`` is everything the file holds and everything it gets written back;
    # a resume with a smaller ``--limit`` must not delete the rows a longer run paid
    # the network for. The report is scored over ``wanted`` alone, further down.
    stored = _load_rows(results) if args.resume else []
    done = {row.claim_id for row in stored}
    if done:
        print(f"resume    {len(done)} claims already scored in {results}")

    # ``interactive=False`` so an `ask` permission is denied and reported rather than
    # prompted for (product rule 4); ``--no-browser`` forces the browser step off
    # outright and ``--browser`` on (asking for it is the consent, spec section 7.1);
    # otherwise the config's own permission decides.
    # ``nli="default"`` is pinned: the published numbers describe the default install,
    # so the user's ``models.nli`` must not change what this script measures.
    browser = True if args.browser else False if args.no_browser else None
    # Built here, not where the key is checked: ``Engine.default`` owns it from this
    # call on and closes it even when it fails, so nothing in between can leak it.
    judge = judge_mod.Judge(judge_mod.build_client(config.judge, key)) if args.judge else None
    engine = Engine.default(
        config, interactive=False, browser=browser, nli=NLI_PROFILE, judge=judge
    )
    # The configured judge's name, taken before it can switch to its local fallback:
    # the report names the judge the run was told to use, the table who answered.
    judge_name = "" if judge is None else judge.name
    print(f"nli       {NLI_PROFILE} profile (pinned; ignores models.nli)")
    searcher: Searcher | None = None
    if args.search:
        # ``Engine.default`` already resolved the configured provider onto its own
        # polite client (verify.py); reusing that instead of building a second one
        # keeps the run's searcher assembled exactly once, like everything else here.
        if engine.searcher is None:
            print(
                f"error     evidence search is not set up: "
                f"{engine.search_problem or 'search.provider is off'}",
                file=sys.stderr,
            )
            engine.close()
            return 2
        searcher = engine.searcher

    # The judge the product would write queries with: the same rule ``verify`` uses.
    # Under ``--judge`` it also gives its opinion of the escalated verdicts.
    judge = engine.judge if engine.escalate else None
    path = measured_path(judge) if searcher is not None else GOLD_PATH
    scored = 0
    try:
        for position, claim in enumerate(claims, start=1):
            if claim.id in done:
                continue
            if searcher is None:
                stored.append(
                    _decide_claim(
                        engine,
                        claim,
                        sleep=args.sleep,
                        fresh=args.fresh,
                        judge=judge,
                        judge_nei=args.judge_nei,
                    )
                )
            else:
                # The gold URLs are the answer key; search mode reads what evidence
                # search would actually find, which is the §17.1 gate this run exists
                # to measure. ``non_urls=0`` because a search hit is always a URL.
                searched = search_urls(
                    searcher,
                    claim,
                    config.search.results_per_claim,
                    judge=judge,
                    max_claims=config.search.max_claims,
                )
                if searched.state is not None:
                    stored.append(unsearched_row(claim, searched.state))
                else:
                    found = replace(
                        claim, text=searched.text, source_urls=searched.urls, non_urls=0
                    )
                    stored.append(
                        _decide_claim(
                            engine,
                            found,
                            sleep=args.sleep,
                            anonymous=True,
                            fresh=args.fresh,
                            judge=judge,
                            judge_nei=args.judge_nei,
                        )
                    )
            scored += 1
            # After every claim, not at the end: a run this long is interrupted more
            # often than it finishes, and the file has to survive that.
            _save_rows(results, stored)
            print(
                f"  {position}/{len(claims)}  claim {claim.id}  {stored[-1].predicted}",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print(f"\ninterrupted after {scored} new claims; {results} is usable", file=sys.stderr)
    except SearchKeyError as exc:
        # The message names the variable to check, never the key (round 2).
        print(
            f"error     {exc}\n"
            "          The run stopped: every claim would fail the same way.\n"
            f"          {results} keeps what was scored.",
            file=sys.stderr,
        )
        return 2
    finally:
        engine.close()
    if judge is not None and judge.switched:
        # The run's one switch notice, as the product prints it (spec section 11).
        print(f"note      {judge.switched}", file=sys.stderr)

    wanted = {claim.id for claim in claims}
    rows = [row for row in stored if row.claim_id in wanted]
    report = render_report(
        score(rows, judge=args.judge, probes=args.judge_nei),
        date=date.today().isoformat(),
        limit=args.limit,
        path=path,
        browser=browser,
        fresh=args.fresh,
        judge=judge_name,
        judge_nei=args.judge_nei,
        no_fallback=args.no_fallback,
    )
    print()
    print(report)
    out = (
        Path(args.out)
        if args.out
        else _report_path(day=date.today().isoformat(), search=args.search, **setup)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"written   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
