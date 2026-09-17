"""One whole report, pinned byte for byte.

Every other test in ``tests/test_verify.py`` asserts about one field at a time, so a
refactor can move a state, a note or a stage line from one place to another and still
pass all of them. This one serialises a complete :class:`~proofpath.report.Report` the
way ``--format json`` does (``ui.json_text``) and compares it against
``tests/data/verify-report-golden.json``, so that anything that changes the durable
surface of a run — a wording, an ordering, a count, a state, a stage attribution —
fails here and has to be argued for rather than noticed later.

The run covers the four shapes a bibliography actually produces: a ghost, a reference
no index covers that prints its own address, a resolved DOI with full text and a
resolved DOI with only an abstract. Everything is a stub, so the payload depends on
nothing but this repository's own code.

Two fields are wall-clock and cannot be pinned: ``Report.elapsed`` and each
``Stage.elapsed``. They are zeroed before the comparison, and every float is rounded,
so a platform's last significant bit cannot fail a test about wording.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from proofpath import oa
from proofpath.browser import ConsentGate
from proofpath.config import Config
from proofpath.fetch import Fetched, FetchStats, Outcome
from proofpath.oa import ABSTRACT_ONLY, Attempt, Evidence, Location
from proofpath.report import Report
from proofpath.resolve import Candidate, ResolveResult, Retraction, State
from proofpath.ui import json_text
from proofpath.verify import Engine, verify
from tests.fakes import WordEmbedder

GOLDEN_PATH = Path(__file__).parent / "data" / "verify-report-golden.json"

DOI = "10.1038/s41586-021-03819-2"
CLOSED_DOI = "10.5555/closed-access-2020"
PAGE = "https://example.test/air-quality.html"

# --- the four references ---------------------------------------------------------

REAL = "Vaswani, A. Attention is all you need. NeurIPS, 2017."
GHOSTLY = "Nobody, N. A study that was never written. Journal of Nothing, 2019."
WEB = f"WHO. Air quality guidelines. {PAGE} Accessed 2024."
CLOSED = "Smith, J. A closed paper. Journal of Paywalls, 2020."

BODY = (
    "Transformers improved translation quality [1]. "
    "A fabricated finding [2]. "
    "Air quality limits were tightened in 2021 [3]. "
    "The paper reports a measured effect [4]. "
    "Nothing further was measured."
)

# --- the three source texts ------------------------------------------------------

PAPER = (
    "Transformers improved translation quality on every benchmark. "
    "The rest of the paper is about tokenisers."
)
# Over ``oa.FULLTEXT_MIN_WORDS`` so the ladder's word grading calls it full text. The
# filler carries no sentence end, so it is one passage no claim's words overlap.
FILLER = " ".join(f"word{index}" for index in range(oa.FULLTEXT_MIN_WORDS + 10))
PAGE_TEXT = f"Air quality limits were tightened in 2021 across the region. {FILLER}"
ABSTRACT = "The paper reports a measured effect of unstated size."


def draft(body: str, entries: Sequence[str]) -> str:
    printed = "\n".join(f"[{number}] {raw}" for number, raw in enumerate(entries, start=1))
    return f"{body}\n\nReferences\n\n{printed}\n"


# --- stubs -----------------------------------------------------------------------


class StubResolver:
    """``resolve.Resolver``'s two methods, keyed by a substring of the raw entry."""

    def __init__(self, results: dict[str, ResolveResult]) -> None:
        self.results = results

    def resolve(self, raw: str) -> ResolveResult:
        for key, result in self.results.items():
            if key in raw:
                return result
        return ResolveResult(State.GHOST, None, [], notes=["no record anywhere"])

    def retraction(self, doi: str) -> Retraction | None:
        return None


class StubOpenAccess:
    """``oa.OpenAccess.fetch``, keyed by the identifier it is called with."""

    def __init__(self, evidence: dict[str, Evidence]) -> None:
        self.evidence = evidence

    def fetch(self, doi: str | None, arxiv_id: str | None = None) -> Evidence:
        return self.evidence[doi or arxiv_id or ""]


class StubFetcher:
    """``fetch.Fetcher``'s URL half, plus the two network attributes prepare reads."""

    network_allowed = True
    network_note = ""

    def __init__(self, pages: dict[str, Fetched]) -> None:
        self.pages = pages

    def fetch(
        self,
        url: str,
        *,
        text_kind: str = "fulltext",
        counts_as_source: bool = True,
        use_cache: bool = True,
    ) -> Fetched:
        return self.pages[url]

    def summary(self) -> FetchStats:
        return FetchStats(counts={}, browser_skipped=0)


SUPPORTED_ROW = (0.95, 0.02, 0.03)
NEI_ROW = (0.30, 0.20, 0.50)


class KeywordScorer:
    """A scorer with no table to miss: every premise gets an answer, deterministically.

    ``tests.fakes.TableScorer`` raises on a premise it does not know, which is the
    right behaviour for a test about retrieval and the wrong one for a test about the
    report: a whole-document run must not fail because a filler passage was scored.
    """

    name = "keyword"

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        rows = [
            SUPPORTED_ROW if "improved translation quality" in premise else NEI_ROW
            for premise, _ in pairs
        ]
        return np.array(rows, dtype=np.float32)


def resolved(doi: str, title: str, author: str, year: int) -> ResolveResult:
    best = Candidate(
        doi=doi, title=title, first_author=author, year=year,
        venue="NeurIPS", provider="crossref",
    )  # fmt: skip
    return ResolveResult(State.RESOLVED, best, [best])


def built_engine() -> Engine:
    return Engine(
        config=Config(),
        cache=None,
        resolver=StubResolver(
            {
                "Vaswani": resolved(DOI, "Attention is all you need", "Vaswani", 2017),
                "WHO": ResolveResult(State.NOT_INDEXED, None, [], notes=["not in any index"]),
                "Smith": resolved(CLOSED_DOI, "A closed paper", "Smith", 2020),
            }
        ),
        fetcher=StubFetcher(
            {
                PAGE: Fetched(
                    url=PAGE,
                    final_url=PAGE,
                    step=1,
                    outcome=Outcome.OK,
                    status=200,
                    content_type="text/html",
                    kind="html",
                    body=b"",
                    text=PAGE_TEXT,
                    notes=[],
                    from_cache=False,
                )
            }
        ),
        oa=StubOpenAccess(
            {
                DOI: Evidence(
                    kind="fulltext",
                    state="",
                    text=PAPER,
                    url="https://arxiv.test/paper.pdf",
                    source="arxiv",
                    attempts=[
                        Attempt(
                            Location("arxiv", "https://arxiv.test/paper.pdf", "pdf"),
                            Outcome.OK,
                            1,
                            len(PAPER.split()),
                        )
                    ],
                    notes=[],
                ),
                CLOSED_DOI: Evidence(
                    kind="abstract",
                    state=ABSTRACT_ONLY,
                    text=ABSTRACT,
                    url="https://api.test/abstract",
                    source="abstract:s2",
                    attempts=[],
                    notes=["no open-access full text"],
                ),
            }
        ),
        gate=ConsentGate("deny", interactive=False),
        embedder=WordEmbedder,
        scorer=KeywordScorer,
    )


def stable(report: Report) -> Any:
    """The report as ``--format json`` writes it, minus what a clock decides."""
    return _scrub(json.loads(json_text(report)))


def _scrub(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: (0.0 if key == "elapsed" else _scrub(value)) for key, value in node.items()}
    if isinstance(node, list):
        return [_scrub(item) for item in node]
    if isinstance(node, float):
        return round(node, 6)
    return node


def test_a_whole_report_is_the_document_it_has_always_been() -> None:
    report = verify(draft(BODY, [REAL, GHOSTLY, WEB, CLOSED]), built_engine())

    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert stable(report) == expected
