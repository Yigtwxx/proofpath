"""Phase 10 harness: measure proofpath end to end on AVeriTeC dev (spec section 14).

Usage:
    uv run python scripts/eval_averitec.py [--limit 100] [--no-browser] [--sleep 1.0]
        [--resume] [--search] [--out docs/eval/<date>-averitec.md]

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

The live half below ``main`` is run by hand. Importing this module touches neither
the network nor a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from proofpath import claims as claims_mod
from proofpath import ingest, pipeline, retrieval
from proofpath.config import NliProfileName, SearchConfig, load_config
from proofpath.eval import averitec
from proofpath.fetch import Fetched
from proofpath.models import Label, Passage
from proofpath.paths import cache_dir
from proofpath.polite import ProviderError
from proofpath.report import LANGUAGE_UNSUPPORTED, SEARCH_UNAVAILABLE
from proofpath.search import Searcher, SearchKeyError, canonical, search_claim
from proofpath.search.queries import plan_queries
from proofpath.verify import NO_TEXT, Engine

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


def score(rows: Sequence[Row]) -> AveritecResult:
    """Turn decided rows into the numbers the report prints.

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
    )


def _md_row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_report(result: AveritecResult, *, date: str, limit: int, path: str = GOLD_PATH) -> str:
    """The markdown skeleton. ``## Notes`` is left for the controller to fill in."""
    # ``result.counted`` is the denominator the 3-way accuracy and its baseline are
    # read over. Printed beside `n`, because "0.80 over 100 claims" would otherwise
    # claim a coverage the number does not have when some claims were never scorable.
    lines = [
        f"# AVeriTeC dev — {date}",
        "",
        f"- claims: {result.n}  (limit={limit or 'none'})",
        f"- measured path: {path}",
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
        share = 100.0 * result.archive_urls / result.total_urls
        lines.append(
            f"{result.archive_urls} of {result.total_urls} source URLs are "
            f"{ARCHIVE_HOST} snapshots ({share:.1f} %)."
        )
    else:
        lines.append(f"No source URLs were measured, so no {ARCHIVE_HOST} share applies.")
    lines.extend(["", "## Notes", "", "<!-- filled in by hand after the run -->", ""])
    return "\n".join(lines) + "\n"


# --- live half: run by hand, not covered by tests ------------------------


def _results_path(*, search: bool) -> Path:
    return cache_dir() / "datasets" / (SEARCH_RESULTS_NAME if search else RESULTS_NAME)


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
            }
            for row in rows
        ],
        indent=1,
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _decide_claim(
    engine: Engine, claim: averitec.Claim, *, sleep: float, anonymous: bool = False
) -> Row:
    """Fetch every source of one claim and keep the strongest non-NEI verdict.

    ``anonymous`` is for pages the search found: fetched without the contact address,
    as the product fetches them (round 2)."""
    states: list[str] = []
    best: pipeline.Verdict | None = None
    fetched_any = False
    for url in claim.source_urls:
        # Before every fetch, not between them: the ladder has no crawl delay of its
        # own, so this is the whole of the run's politeness and it has to hold across
        # claims too, not just within one.
        if sleep:
            time.sleep(sleep)
        fetched = engine.fetcher.fetch(url, anonymous=anonymous)
        states.append(state_for(fetched))
        if not fetched.ok or not fetched.text.strip():
            continue
        fetched_any = True
        passages = [
            Passage(sentence, url, ordinal)
            for ordinal, sentence in enumerate(retrieval.split_sentences(fetched.text))
        ]
        verdict = pipeline.decide(
            claim.text,
            passages,
            engine.get_embedder(),
            engine.get_scorer(),
            k=engine.k,
            thresholds=engine.thresholds,
        )
        # No passage, no verdict (product rule 1): an assertion without the sentence
        # it rests on is dropped rather than reported.
        if verdict.label is Label.NEI or verdict.passage is None:
            continue
        if best is None or verdict.score > best.score:
            best = verdict
    # One state per value the dataset gave, fetchable or not, so the coverage table
    # accounts for every annotation rather than only the ones we could try.
    states.extend([NOT_A_URL] * claim.non_urls)
    if not claim.source_urls and not claim.non_urls:
        states.append(NO_SOURCE)
    if best is not None:
        predicted: str | None = best.label.value
    elif fetched_any:
        predicted = Label.NEI.value
    else:
        predicted = None
    return Row(
        claim_id=claim.id,
        gold=claim.label,
        predicted=predicted,
        states=tuple(states),
        urls=claim.source_urls,
    )


NLI_PROFILE: NliProfileName = "default"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="first N dev claims")
    parser.add_argument(
        "--no-browser", action="store_true", help="never use the browser step of the ladder"
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

    results = _results_path(search=args.search)
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

    config = load_config()
    # ``interactive=False`` so an `ask` permission is denied and reported rather than
    # prompted for (product rule 4); ``--no-browser`` forces the browser step off
    # outright, otherwise the config's own permission decides.
    # ``nli="default"`` is pinned: the published numbers describe the default install,
    # so the user's ``models.nli`` must not change what this script measures.
    engine = Engine.default(
        config, interactive=False, browser=False if args.no_browser else None, nli=NLI_PROFILE
    )
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
    judge = engine.judge if engine.escalate else None
    path = measured_path(judge) if searcher is not None else GOLD_PATH
    scored = 0
    try:
        for position, claim in enumerate(claims, start=1):
            if claim.id in done:
                continue
            if searcher is None:
                stored.append(_decide_claim(engine, claim, sleep=args.sleep))
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
                    stored.append(_decide_claim(engine, found, sleep=args.sleep, anonymous=True))
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

    wanted = {claim.id for claim in claims}
    rows = [row for row in stored if row.claim_id in wanted]
    report = render_report(score(rows), date=date.today().isoformat(), limit=args.limit, path=path)
    print()
    print(report)
    suffix = "-search" if args.search else ""
    out = (
        Path(args.out)
        if args.out
        else Path("docs/eval") / f"{date.today().isoformat()}-averitec{suffix}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"written   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
