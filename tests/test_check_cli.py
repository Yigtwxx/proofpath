"""``proofpath check TARGET``: wiring, output vocabulary, exit codes (spec section 13.3).

``Engine.default`` is monkeypatched to an engine of stubs, so the command is exercised
end to end — parsing, pairing, resolving, reading, verifying, rendering — without a
socket and without an ONNX session. The stubs are the ones ``tests/test_verify.py``
uses, kept here in their smallest form so the two files can move independently.
"""

from __future__ import annotations

import json
import re
import signal
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proofpath import cli as cli_mod
from proofpath import judge as judge_mod
from proofpath import verify as verify_mod
from proofpath.browser import ConsentGate
from proofpath.cli import _interruptible, app
from proofpath.config import Config
from proofpath.events import Cancelled
from proofpath.fetch import Fetched, FetchStats, Outcome
from proofpath.judge import Completion, Judge, JudgeCost, JudgeUnavailable
from proofpath.oa import Attempt, Evidence, Location
from proofpath.resolve import Candidate, ResolveResult, Retraction, State
from proofpath.retrieval import Embedder
from proofpath.verify import Engine
from tests.fakes import TableScorer, WordEmbedder

runner = CliRunner()

DOI = "10.1038/s41586-021-03819-2"

SUPPORTING = "Transformers improved translation quality on every benchmark."
FIGURE = "The method yields a 4-8% speedup in throughput."
FILLER = "The rest of the paper is about tokenisers."
PAPER = f"{SUPPORTING} {FIGURE} {FILLER}"
SUPPORTED_ROW = (0.95, 0.02, 0.03)

REAL = "Vaswani, A. Attention is all you need. NeurIPS, 2017."
GHOSTLY = "Nobody, N. A study that was never written. Journal of Nothing, 2019."

CLEAN_BODY = "Transformers improved translation quality [1]. Nothing further was measured."
# The second sentence's number does not match the source's: a numeric mismatch, which
# is a finding without needing a ghost reference.
TROUBLED_BODY = (
    "Transformers improved translation quality [1]. The method yields a 40% speedup [1]. "
    "Nothing further was measured."
)


# --- stubs --------------------------------------------------------------------------


class StubResolver:
    """``resolve.Resolver``'s two methods, keyed by a substring of the raw entry."""

    def __init__(
        self,
        results: dict[str, ResolveResult] | None = None,
        retractions: dict[str, Retraction] | None = None,
    ) -> None:
        self.results = results or {}
        self.retractions = retractions or {}

    def resolve(self, raw: str) -> ResolveResult:
        for key, result in self.results.items():
            if key in raw:
                return result
        return ResolveResult(State.GHOST, None, [], notes=["no record anywhere"])

    def retraction(self, doi: str) -> Retraction | None:
        return self.retractions.get(doi)


class StubOpenAccess:
    """``oa.OpenAccess.fetch``, keyed by the identifier it is called with."""

    def __init__(self, evidence: dict[str, Evidence] | None = None) -> None:
        self.evidence = evidence or {}
        self.on_call: Callable[[], None] | None = None

    def fetch(self, doi: str | None, arxiv_id: str | None = None) -> Evidence:
        if self.on_call is not None:
            self.on_call()
        return self.evidence.get(doi or arxiv_id or "", none_evidence(Outcome.UNREACHABLE))


class StubFetcher:
    """``fetch.Fetcher``'s URL half, plus the two network attributes prepare reads."""

    def __init__(self, pages: dict[str, Fetched] | None = None) -> None:
        self.pages = pages or {}
        self.network_allowed = True
        self.network_note = ""

    def fetch(
        self,
        url: str,
        *,
        text_kind: str = "fulltext",
        counts_as_source: bool = True,
        use_cache: bool = True,
    ) -> Fetched:
        return self.pages.get(url, unreachable(url))

    def summary(self) -> FetchStats:
        return FetchStats(counts={}, browser_skipped=0)


def unreachable(url: str) -> Fetched:
    return Fetched(
        url=url,
        final_url=url,
        step=1,
        outcome=Outcome.UNREACHABLE,
        status=None,
        content_type="",
        kind="other",
        body=b"",
        text="",
        notes=["step 1 httpx: no response"],
        from_cache=False,
    )


def text_evidence(text: str, url: str = "https://arxiv.test/paper.pdf") -> Evidence:
    return Evidence(
        kind="fulltext",
        state="",
        text=text,
        url=url,
        source="arxiv",
        attempts=[Attempt(Location("arxiv", url, "pdf"), Outcome.OK, 1, len(text.split()))],
        notes=[],
    )


def none_evidence(outcome: Outcome) -> Evidence:
    return Evidence(
        kind="none",
        state=outcome.value,
        text="",
        url="",
        source="",
        attempts=[],
        notes=[f"every location {outcome.value}"],
    )


def resolved(doi: str = DOI) -> ResolveResult:
    best = Candidate(
        doi=doi, title="Attention is all you need", first_author="Vaswani", year=2017,
        venue="NeurIPS", provider="crossref",
    )  # fmt: skip
    return ResolveResult(State.RESOLVED, best, [best])


def built_engine(
    *,
    resolver: StubResolver | None = None,
    chain: StubOpenAccess | None = None,
    fetcher: StubFetcher | None = None,
    embedder: Embedder | None = None,
    scorer: object | None = None,
) -> Engine:
    """The engine ``Engine.default`` is replaced by: every part a stub."""
    return Engine(
        config=Config(),
        cache=None,
        resolver=resolver or StubResolver({"Vaswani": resolved()}),
        fetcher=fetcher or StubFetcher(),
        oa=chain or StubOpenAccess({DOI: text_evidence(PAPER)}),
        gate=ConsentGate("deny", interactive=False),
        embedder=lambda: embedder if embedder is not None else WordEmbedder(),
        scorer=lambda: scorer if scorer is not None else TableScorer({SUPPORTING: SUPPORTED_ROW}),
    )


def install(monkeypatch: pytest.MonkeyPatch, engine: Engine | None = None) -> dict[str, Any]:
    """Replace ``Engine.default`` and return the keyword arguments the CLI passed it."""
    seen: dict[str, Any] = {}
    prepared = engine if engine is not None else built_engine()

    def fake_default(config: Config, **kwargs: Any) -> Engine:
        seen["config"] = config
        seen.update(kwargs)
        # ``escalate`` decides whether the judging stage may spend the judge; the
        # real ``Engine.default`` puts it on the engine, so the stub does too.
        prepared.escalate = kwargs.get("escalate", True)
        judge = kwargs.get("judge")
        if judge is not None:
            # The real ``Engine.default`` takes ownership of the judge it is handed.
            # The stub engine is not the one the CLI built, so it closes it instead --
            # otherwise the socket behind it outlives the command.
            prepared._closers.append(judge.close)
        return prepared

    monkeypatch.setattr(verify_mod.Engine, "default", staticmethod(fake_default))
    return seen


def draft(body: str, entries: Sequence[str] = (REAL,)) -> str:
    printed = "\n".join(f"[{number}] {raw}" for number, raw in enumerate(entries, start=1))
    return f"{body}\n\nReferences\n\n{printed}\n"


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh config and cache dirs, and a working directory ``report.md`` can land in."""
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)


def write(tmp_path: Path, body: str, entries: Sequence[str] = (REAL,)) -> Path:
    path = tmp_path / "draft.md"
    path.write_text(draft(body, entries), encoding="utf-8")
    return path


# --- the clean and the troubled run --------------------------------------------------


def test_a_clean_draft_prints_the_stage_table_and_footer_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY))])

    assert result.exit_code == 0, result.output
    assert "error[" not in result.stdout and "warning[" not in result.stdout
    assert "  Parsing" in result.stdout and "  Verifying" in result.stdout
    assert "1 refs: 1 ok" in result.stdout
    # Rule 6: the coverage block is part of every report, clean or not.
    assert "fulltext   100%" in result.stdout
    assert "abstract   0%" in result.stdout
    assert "unverified 0%" in result.stdout
    assert "report.md written" in result.stdout


PAGE_URL = "https://example.test/news/story"
SOURCE_URL = "https://other.test/report"


def html_page(url: str, body: bytes, text: str = "") -> Fetched:
    """``body`` is what a page read as the document is cut up; ``text`` is what a
    source is graded on."""
    return Fetched(
        url=url,
        final_url=url,
        step=1,
        outcome=Outcome.OK,
        status=200,
        content_type="text/html",
        kind="html",
        body=body,
        text=text,
        notes=[],
    )


def test_check_url_on_a_page_reads_the_page_as_the_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare address on a host that is not a platform is the page at that address:
    its paragraphs are the claims and the links they carry are the sources."""
    story = f"<article><p><a href='{SOURCE_URL}'>{SUPPORTING}</a></p></article>"
    install(
        monkeypatch,
        built_engine(
            fetcher=StubFetcher(
                {
                    PAGE_URL: html_page(PAGE_URL, story.encode()),
                    # Long enough to be full text rather than "abstract only".
                    SOURCE_URL: html_page(
                        SOURCE_URL, b"", PAPER + " Appendix on tokenisers." * 600
                    ),
                }
            )
        ),
    )

    result = runner.invoke(app, ["check", "--url", PAGE_URL])

    assert result.exit_code == 0, result.output
    assert "  Parsing" in result.stdout and "fetch ladder" in result.stdout
    assert "1 claim: 1 supported" in result.stdout
    assert "1 refs: 1 ok" in result.stdout


def test_check_url_on_a_page_the_ladder_cannot_read_is_an_error_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, built_engine(fetcher=StubFetcher()))

    result = runner.invoke(app, ["check", "--url", PAGE_URL])

    assert result.exit_code == 2, result.output
    assert f"{PAGE_URL} could not be read: {Outcome.UNREACHABLE.value}" in result.output


def test_a_finding_is_printed_as_a_diagnostic_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", str(write(tmp_path, TROUBLED_BODY))])

    assert result.exit_code == 1, result.output
    assert "error[numeric-mismatch]: claim contradicts the cited source" in result.stdout
    assert "--> draft.md:" in result.stdout
    # Product rule 1: the verdict travels with the passage it rests on.
    assert FIGURE in result.stdout
    assert "1 refs: 1 unsupported, 0 ok" in result.stdout


def test_an_unreadable_source_is_reported_not_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        built_engine(
            resolver=StubResolver({"Vaswani": resolved(), "Nobody": resolved("10.5/blocked")}),
            chain=StubOpenAccess(
                {DOI: text_evidence(PAPER), "10.5/blocked": none_evidence(Outcome.BLOCKED_ROBOTS)}
            ),
        ),
    )
    body = "Transformers improved translation quality [1]. A fabricated finding [2]. Nothing else."
    result = runner.invoke(app, ["check", str(write(tmp_path, body, (REAL, GHOSTLY)))])

    assert result.exit_code == 1, result.output
    assert Outcome.BLOCKED_ROBOTS.value in result.stdout
    assert "unverified 50%" in result.stdout
    assert "coverage is weak" in result.stdout


# --- targets -------------------------------------------------------------------------


def test_a_missing_file_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "nowhere.pdf"])
    assert result.exit_code == 2
    assert "error: no such file" in result.output
    assert "nowhere.pdf" in result.output


def test_dash_reads_the_document_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "-"], input=draft(CLEAN_BODY).encode("utf-8"))
    assert result.exit_code == 0, result.output
    assert "1 refs: 1 ok" in result.stdout


def test_a_document_read_from_stdin_is_named_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not "pasted text": what the locations name is where the reader can look."""
    install(monkeypatch)
    result = runner.invoke(
        app, ["check", "-", "--format", "json"], input=draft(CLEAN_BODY).encode("utf-8")
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["document"]["name"] == "stdin"


def test_stdin_is_decoded_as_utf8_whatever_the_locale_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The document arrives as bytes, so a run is not at the mercy of the code page.

    ``sys.stdin`` on Windows decodes with the ANSI code page, which turns a UTF-8
    paper into mojibake -- and a claim nobody can quote back is a claim nobody can
    check. The bytes are read and decoded here instead.
    """
    install(monkeypatch)
    body = "Transformers improved translation quality [1]. La méthode yields a 40% speedup [1]."
    result = runner.invoke(app, ["check", "-"], input=draft(body).encode("utf-8"))

    assert result.exit_code == 1, result.output
    assert "méthode" in result.stdout


def test_a_directory_is_an_error_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", str(tmp_path)])
    assert result.exit_code == 2
    assert f"error: not a file: {tmp_path}" in result.output


def test_a_crash_inside_the_run_is_an_error_not_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 1 means "the document has findings". A bug must never be able to say that.

    Anything unforeseen -- a provider, a model, a parser -- is the tool failing (2),
    reported on one line rather than as a traceback the reader has to interpret.
    """
    install(monkeypatch)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(verify_mod, "verify", boom)
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY))])

    assert result.exit_code == 2, result.output
    assert "error: boom" in result.output
    assert "Traceback" not in result.output


# --- flags that are not built yet ----------------------------------------------------


def test_an_unknown_format_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "-", "--format", "xml"], input=draft(CLEAN_BODY))
    assert result.exit_code == 2


def test_allow_browser_and_no_browser_cannot_be_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(
        app, ["check", "-", "--allow-browser", "--no-browser"], input=draft(CLEAN_BODY)
    )
    assert result.exit_code == 2
    assert "--allow-browser" in result.output and "--no-browser" in result.output


# --- --format sarif ------------------------------------------------------------------


def test_sarif_puts_one_log_document_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What an editor opens: a SARIF 2.1.0 log and nothing else on stdout."""
    install(monkeypatch)
    write(tmp_path, TROUBLED_BODY)
    result = runner.invoke(app, ["check", "draft.md", "--format", "sarif"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["tool"]["driver"]["name"] == "proofpath"
    assert payload["runs"][0]["results"], "the troubled draft has a finding to show"


def test_sarif_points_at_the_target_as_the_caller_named_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    write(tmp_path, TROUBLED_BODY)
    result = runner.invoke(app, ["check", "draft.md", "--format", "sarif"])

    assert result.exit_code == 1, result.output
    location = json.loads(result.stdout)["runs"][0]["results"][0]["locations"][0]
    assert location["physicalLocation"]["artifactLocation"]["uri"] == "draft.md"


def test_sarif_carries_the_runs_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Product rule 6: a log opened without the terminal output still states coverage."""
    install(monkeypatch)
    write(tmp_path, CLEAN_BODY)
    result = runner.invoke(app, ["check", "draft.md", "--format", "sarif"])

    assert result.exit_code == 0, result.output
    coverage = json.loads(result.stdout)["runs"][0]["properties"]["coverage"]
    assert coverage["references"] == 1
    assert coverage["pct"]["fulltext"] == 100


def test_sarif_sends_the_human_lines_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same rule as ``--format json`` (spec section 13.3): stdout is the document."""
    install(monkeypatch)
    write(tmp_path, CLEAN_BODY)
    result = runner.invoke(app, ["check", "draft.md", "--format", "sarif"])

    assert result.exit_code == 0, result.output
    assert "Parsing" not in result.stdout
    assert "  Parsing" in result.stderr


def test_sarif_writes_no_report_md_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log *is* the output. A markdown file nobody asked for is not part of it."""
    install(monkeypatch)
    write(tmp_path, CLEAN_BODY)
    result = runner.invoke(app, ["check", "draft.md", "--format", "sarif"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "report.md").exists()


def test_sarif_out_writes_the_same_document_it_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--out draft.sarif`` is how the file reaches VS Code, so it holds the log."""
    install(monkeypatch)
    write(tmp_path, TROUBLED_BODY)
    result = runner.invoke(app, ["check", "draft.md", "--format", "sarif", "--out", "draft.sarif"])

    assert result.exit_code == 1, result.output
    written = (tmp_path / "draft.sarif").read_text(encoding="utf-8")
    assert json.loads(written) == json.loads(result.stdout)


def test_sarif_keeps_the_exit_code_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A format is a way of printing, never a second opinion about the document."""
    install(monkeypatch)
    write(tmp_path, CLEAN_BODY)
    assert runner.invoke(app, ["check", "draft.md", "--format", "sarif"]).exit_code == 0


def test_sarif_stays_refused_by_resolve_and_fetch() -> None:
    """Only ``check`` emits it; the other two say so rather than printing text."""
    for argv in (["resolve", "ref", "--format", "sarif"], ["fetch", "url", "--format", "sarif"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 2, result.output
        assert "--format sarif" in result.output


# --- the committed author-year draft -------------------------------------------------

AUTHOR_YEAR_DRAFT = Path(__file__).parent / "data" / "draft-author-year.md"
AUTHOR_YEAR_SURNAMES = ("Vaswani", "Jumper", "Harris", "Virtanen", "Wilkinson")


def test_the_author_year_draft_pairs_every_citation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v0.2 live asset (``docs/eval/2026-09-15-v0.2-live.md`` section 4), offline.

    Five APA entries, cited as ``(Surname et al., YEAR)``, one ``(ibid.)`` and one
    mixed ``(…; [5])``: every marker pairs with an entry, so the Claims row says
    ``0 unresolved`` and nothing is reported as an unsupported citation style.
    """
    dois = {name: f"10.5/{name.lower()}" for name in AUTHOR_YEAR_SURNAMES}
    # Every source is the same three-sentence paper, and every sentence of it scores
    # NEI: what this test is about is the pairing, not the verdicts.
    nei = dict.fromkeys((SUPPORTING, FIGURE, FILLER), (0.1, 0.1, 0.8))
    install(
        monkeypatch,
        built_engine(
            resolver=StubResolver({name: resolved(doi) for name, doi in dois.items()}),
            chain=StubOpenAccess({doi: text_evidence(PAPER) for doi in dois.values()}),
            scorer=TableScorer(nei),
        ),
    )
    result = runner.invoke(app, ["check", str(AUTHOR_YEAR_DRAFT), "--format", "json"])

    assert result.exit_code in (0, 1), result.output
    assert "5 refs" in result.stderr
    assert "6 citations, 0 unresolved" in result.stderr
    assert "UNSUPPORTED CITATION STYLE" not in result.output
    report = json.loads(result.stdout)
    assert report["coverage"]["references"] == 5
    assert report["coverage"]["unverified"] == 0


# --- what reaches Engine.default -----------------------------------------------------


def test_a_piped_run_is_never_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Product rule 4: no TTY, no prompt — the engine is told so, not left to sniff."""
    seen = install(monkeypatch)
    result = runner.invoke(app, ["check", "-"], input=draft(CLEAN_BODY))
    assert result.exit_code == 0, result.output
    assert seen["interactive"] is False
    assert seen["browser"] is None
    assert seen["no_cache"] is False


@pytest.mark.parametrize(("flag", "expected"), [("--allow-browser", True), ("--no-browser", False)])
def test_the_browser_override_reaches_the_engine(
    flag: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = install(monkeypatch)
    result = runner.invoke(app, ["check", "-", flag], input=draft(CLEAN_BODY))
    assert result.exit_code == 0, result.output
    assert seen["browser"] is expected


def test_no_cache_reaches_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install(monkeypatch)
    result = runner.invoke(app, ["check", "-", "--no-cache"], input=draft(CLEAN_BODY))
    assert result.exit_code == 0, result.output
    assert seen["no_cache"] is True


# --- --format json -------------------------------------------------------------------


def test_format_json_is_alone_on_stdout_and_stage_lines_go_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "--format", "json", str(write(tmp_path, TROUBLED_BODY))])
    assert result.exit_code == 1, result.output

    payload = json.loads(result.stdout)  # stdout is one JSON document, nothing else
    assert payload["document"]["name"] == "draft.md"
    assert payload["coverage"]["references"] == 1
    assert payload["coverage"]["fulltext"] == 1
    assert [f["kind"] for f in payload["findings"]] == ["numeric-mismatch"]
    assert payload["findings"][0]["verdict"]["passage"]["text"] == FIGURE
    assert payload["document"]["paragraphs"], "the paragraphs are the report's provenance"
    assert payload["stages"][0]["name"] == "Parsing"

    assert "  Parsing" in result.stderr
    assert "  Parsing" not in result.stdout


def test_format_json_writes_no_report_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "--format", "json", str(write(tmp_path, CLEAN_BODY))])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "report.md").exists()


def test_format_json_writes_the_markdown_when_out_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    out = tmp_path / "sub" / "run.md"
    out.parent.mkdir()
    result = runner.invoke(
        app, ["check", "--format", "json", "--out", str(out), str(write(tmp_path, CLEAN_BODY))]
    )
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").startswith("# proofpath report — draft.md")


# --- the written report --------------------------------------------------------------


def test_text_mode_writes_report_md_next_to_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", str(write(tmp_path, TROUBLED_BODY))])
    assert result.exit_code == 1, result.output
    written = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert written.startswith("# proofpath report — draft.md")
    assert "claim contradicts the cited source" in written
    assert FIGURE in written


def test_out_chooses_the_path_and_the_footer_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    out = tmp_path / "elsewhere.md"
    result = runner.invoke(app, ["check", "--out", str(out), str(write(tmp_path, CLEAN_BODY))])
    assert result.exit_code == 0, result.output
    assert out.exists() and not (tmp_path / "report.md").exists()
    assert f"{out} written" in result.stdout


def test_an_unwritable_out_is_an_error_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    out = tmp_path / "missing-dir" / "run.md"
    result = runner.invoke(app, ["check", "--out", str(out), str(write(tmp_path, CLEAN_BODY))])
    assert result.exit_code == 2
    assert "error:" in result.output


# --- -q ------------------------------------------------------------------------------


def test_quiet_drops_the_stages_but_never_the_findings_or_the_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 6: ``-q`` is about progress. A low-coverage run must still look like one."""
    install(monkeypatch)
    result = runner.invoke(app, ["-q", "check", str(write(tmp_path, TROUBLED_BODY))])
    assert result.exit_code == 1, result.output
    assert "  Parsing" not in result.stdout
    assert "  Verifying" not in result.stdout
    assert "note       " not in result.stdout
    assert "error[numeric-mismatch]: claim contradicts the cited source" in result.stdout
    assert "1 refs: 1 unsupported, 0 ok" in result.stdout
    assert "fulltext   100%" in result.stdout
    assert "report.md written" in result.stdout


# --- Ctrl-C --------------------------------------------------------------------------


def test_ctrl_c_is_a_cancelled_run_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = StubOpenAccess({DOI: text_evidence(PAPER)})

    def interrupt() -> None:
        raise KeyboardInterrupt

    chain.on_call = interrupt
    install(monkeypatch, built_engine(chain=chain))
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY))])

    assert result.exit_code == 2
    assert "cancelled" in result.output
    assert not (tmp_path / "report.md").exists()


# The numeric mismatch is decided by rule and never reaches the scorer, so the Ctrl-C
# below lands on the second claim, after the first has already left a finding behind.
INTERRUPTED_BODY = (
    "The method yields a 40% speedup [1]. Transformers improved translation quality [1]. "
    "The rest concerns tokenisers [1]. Nothing further was measured."
)


def test_ctrl_c_keeps_the_verdicts_the_run_had_already_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGINT sets the cancel event, so the run stops between claims and still reports.

    The third claim is never decided -- ``TableScorer`` has no row for the sentence it
    would retrieve, so scoring it would raise -- which is how the test sees that the
    run really stopped rather than finishing quietly (product rule 6).
    """

    def interrupt() -> None:
        signal.raise_signal(signal.SIGINT)  # synchronous, and never TerminateProcess

    scorer = TableScorer({SUPPORTING: SUPPORTED_ROW}, on_score=interrupt)
    install(monkeypatch, built_engine(scorer=scorer))
    result = runner.invoke(app, ["check", str(write(tmp_path, INTERRUPTED_BODY))])

    assert result.exit_code == 2, result.output
    assert "error[numeric-mismatch]: claim contradicts the cited source" in result.stdout
    assert "cancelled" in result.stdout
    written = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- cancelled: the run stopped early, so this report is partial" in written


def test_the_sigint_handler_is_installed_for_the_run_and_restored_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    original = signal.getsignal(signal.SIGINT)

    def record(signum: int, handler: Any) -> Any:
        calls.append(handler)
        return original

    monkeypatch.setattr(signal, "signal", record)
    install(monkeypatch)
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY))])

    assert result.exit_code == 0, result.output
    assert len(calls) == 2, "the handler is installed once and restored once"
    assert callable(calls[0]) and calls[0] is not original
    assert calls[1] is original


def test_a_second_ctrl_c_interrupts_instead_of_asking_again() -> None:
    """The user's way out of a slow fetch the first Ctrl-C could not stop.

    The first one asks the run to stop between units of work; the second one puts the
    previous handler back and lets ``KeyboardInterrupt`` through, so a stage stuck in
    a socket read is not a trap.
    """
    cancel = threading.Event()
    original = signal.getsignal(signal.SIGINT)

    with pytest.raises(KeyboardInterrupt), _interruptible(cancel):
        signal.raise_signal(signal.SIGINT)
        assert cancel.is_set(), "the first one only asks"
        assert signal.getsignal(signal.SIGINT) is not original
        signal.raise_signal(signal.SIGINT)

    assert signal.getsignal(signal.SIGINT) is original


def test_a_cancelled_run_says_so_and_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Cancelled`` raised by the pipeline itself, rather than by a Ctrl-C above it.

    The assertion holds whether the exception carries a partial report -- then the
    footer's ``run cancelled`` line says it -- or not, when the error line does.
    """
    chain = StubOpenAccess({DOI: text_evidence(PAPER)})

    def stop() -> None:
        raise Cancelled("run cancelled")

    chain.on_call = stop
    install(monkeypatch, built_engine(chain=chain))
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY))])

    assert result.exit_code == 2
    assert "cancelled" in result.output


# --- the committed live draft ---------------------------------------------------------


def test_the_live_draft_parses_and_is_a_finding_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tests/data/draft-live.md`` is the document the v0.1 live runs used.

    The live runs are in ``docs/eval/2026-09-12-v0.1-live.md`` and needed the network;
    this keeps the file honest offline. Only its shape is asserted -- the bibliography
    parses whole and the prose still yields claims -- because the verdicts belong to
    the real engine, and here every reference but the first is a ghost to the stub.
    """
    install(monkeypatch)
    draft_path = Path(__file__).parent / "data" / "draft-live.md"
    result = runner.invoke(app, ["check", "--format", "json", str(draft_path)])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["coverage"]["references"] >= 7
    assert payload["claims"] >= 2


# --- process teardown: exit 134 (spec section 13.3) ----------------------------------


def test_check_closes_the_engine_and_flushes_before_it_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ONNX sessions must be released, and every byte written, while the
    interpreter is still up: a session collected at shutdown aborted the process."""
    built = built_engine()
    install(monkeypatch, built)
    order: list[str] = []
    closing = built.close

    def close() -> None:
        order.append("closed")
        closing()

    monkeypatch.setattr(built, "close", close)
    monkeypatch.setattr(cli_mod, "_flush_streams", lambda: order.append("flushed"))

    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY))])

    assert result.exit_code == 0, result.output
    assert order == ["closed", "flushed"]


def test_flush_streams_flushes_both_standard_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    flushed: list[str] = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def flush(self) -> None:
            flushed.append(self.name)

    monkeypatch.setattr(cli_mod.sys, "stdout", Stream("stdout"))
    monkeypatch.setattr(cli_mod.sys, "stderr", Stream("stderr"))
    cli_mod._flush_streams()
    assert sorted(flushed) == ["stderr", "stdout"]


def test_flush_streams_survives_a_closed_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed pipe must not turn a finished run into a traceback."""

    class Broken:
        def flush(self) -> None:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(cli_mod.sys, "stdout", Broken())
    monkeypatch.setattr(cli_mod.sys, "stderr", Broken())
    assert cli_mod._flush_streams() is None


def test_a_document_with_markers_and_no_bibliography_does_not_look_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 6, end to end: the AlphaFold case of the 2026-09-12 live runs."""
    install(monkeypatch)
    path = tmp_path / "draft.md"
    path.write_text("The structures were predicted accurately [1].\n", encoding="utf-8")

    result = runner.invoke(app, ["check", str(path)])

    assert "fulltext   0%" in result.stdout
    assert "no bibliography was found; 1 citation marker could not be checked" in result.stdout
    assert "coverage is weak" not in result.stdout
    assert "no bibliography was found" in (tmp_path / "report.md").read_text(encoding="utf-8")


# --- --judge -------------------------------------------------------------------------


class FakeJudgeClient:
    """``JudgeClient`` without a socket: one agreeing answer per id it is shown."""

    provider = "fake"
    model = "judge-1"

    def __init__(self) -> None:
        self.cost = JudgeCost(model=self.model)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        ids = re.findall(r"id: (\S+)", messages[-1]["content"])
        text = json.dumps(
            {
                "opinions": [
                    {"id": i, "label": "NEI", "rationale": "the passage is silent"} for i in ids
                ]
            }
        )
        self.cost.calls += 1
        self.cost.prompt_tokens += 420
        self.cost.completion_tokens += 40
        return Completion(text=text, prompt_tokens=420, completion_tokens=40, model=self.model)

    def close(self) -> None:
        return None


# Between ``decide`` (0.45) and ``medium`` (0.457948): decided, but only just, which
# is the band the judge is asked about (spec section 9 step 8).
LOW_ROW = (0.02, 0.455, 0.525)


def test_judge_without_a_key_stops_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 2: the tool could not do what it was told, which is not a finding."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = install(monkeypatch)
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY)), "--judge"])

    assert result.exit_code == 2
    assert "error: GROQ_API_KEY is not set" in result.output
    assert "proofpath config check" in result.output
    assert seen == {}  # no engine was built, so no model was loaded


def test_judge_builds_a_judge_and_the_report_counts_its_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    client = FakeJudgeClient()
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(client)  # type: ignore[arg-type]
    install(monkeypatch, engine)
    result = runner.invoke(
        app, ["check", str(write(tmp_path, CLEAN_BODY)), "--judge", "--format", "json"]
    )

    assert result.exit_code == 1, result.output  # a low-tier REFUTED is a finding
    payload = json.loads(result.stdout)
    assert payload["api_calls"] == 1
    assert payload["judge_cost"]["prompt_tokens"] == 420
    assert payload["models"]["judge"] == "fake judge-1"
    (judged,) = [item for item in payload["results"] if item["judge"] is not None]
    assert judged["judge"]["label"] == "NEI"
    # The judge never moved the verdict it disagreed with (spec section 11.1).
    assert judged["verdict"]["label"] == "REFUTED"
    assert "gsk_not_a_real_key" not in result.output


class DownJudgeClient(FakeJudgeClient):
    """A provider that answers nothing at all -- a wrong key, or a tier that is out."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        raise JudgeUnavailable("HTTP 401 from https://api.test/chat/completions")


def test_a_quiet_run_still_says_the_judge_never_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-q`` suppresses the note and the stage row, so the summary the report keeps
    is the only thing left to tell a provider that was down from a document with
    nothing to escalate (product rules 2 and 6)."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(DownJudgeClient())  # type: ignore[arg-type]
    install(monkeypatch, engine)
    result = runner.invoke(
        app, ["-q", "check", str(write(tmp_path, CLEAN_BODY)), "--judge", "--format", "json"]
    )

    payload = json.loads(result.stdout)
    assert payload["stages"][-1]["summary"] == (
        "judge unavailable after 0 calls "
        "(HTTP 401 from https://api.test/chat/completions); local verdicts stand"
    )
    assert payload["api_calls"] == 0
    assert all(item["judge"] is None for item in payload["results"])


def test_a_quiet_text_run_prints_the_judge_line_the_stage_row_would_have_carried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ui.stage_row`` and ``ui.note`` are both dropped under ``-q``, so a text run
    whose provider was down would otherwise say nothing at all about it -- reading
    exactly like a run with nothing to escalate (product rules 2 and 6). It mirrors
    the ``summary`` line, which is unsuppressible for the same reason."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(DownJudgeClient())  # type: ignore[arg-type]
    install(monkeypatch, engine)
    written = tmp_path / "report.md"
    result = runner.invoke(
        app, ["-q", "check", str(write(tmp_path, CLEAN_BODY)), "--judge", "--out", str(written)]
    )

    assert (
        f"judge      judge unavailable after 0 calls ({DOWN_DETAIL}); local verdicts stand"
        in result.output
    )
    # The same sentence the markdown header carries, so neither surface invents one.
    text = written.read_text(encoding="utf-8")
    assert (
        f"- judge status: unavailable after 0 calls ({DOWN_DETAIL}); local verdicts stand" in text
    )


def test_a_judge_that_answered_gets_no_unavailable_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line is a state of the run, not a header: a judge that answered has none."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(FakeJudgeClient())  # type: ignore[arg-type]
    install(monkeypatch, engine)
    result = runner.invoke(
        app,
        [
            "-q",
            "check",
            str(write(tmp_path, CLEAN_BODY)),
            "--judge",
            "--out",
            str(tmp_path / "r.md"),
        ],
    )

    assert "judge unavailable" not in result.output


def test_without_judge_the_json_report_has_no_judge_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW})))
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY)), "--format", "json"])

    payload = json.loads(result.stdout)
    assert payload["api_calls"] == 0
    assert payload["judge_cost"] is None
    assert "judge" not in payload["models"]
    assert all(item["judge"] is None for item in payload["results"])


def test_choosing_gemini_says_so_once_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec section 11: Google trains on free-tier prompts, so the run says so."""
    monkeypatch.setenv("GEMINI_API_KEY", "not_a_real_key")
    (tmp_path / "conf").mkdir(parents=True, exist_ok=True)
    (tmp_path / "conf" / "config.toml").write_text(
        '[judge]\nprovider = "gemini"\nmodel = "gemini-3.8-flash"\n'
        'base_url = "https://generativelanguage.googleapis.com/v1beta/openai"\n'
        'api_key_env = "GEMINI_API_KEY"\n',
        encoding="utf-8",
    )
    client = FakeJudgeClient()
    engine = built_engine()
    engine.judge = Judge(client)  # type: ignore[arg-type]
    install(monkeypatch, engine)
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY)), "--judge"])

    assert result.exit_code == 0, result.output
    assert result.output.count(judge_mod.GEMINI_DATA_USE) == 1


# --- --summarize ---------------------------------------------------------------------


class SummarisingJudgeClient(FakeJudgeClient):
    """Reviews as ``FakeJudgeClient`` does, and writes prose when asked for prose.

    The summary call is the one that carries no JSON schema.
    """

    PARAGRAPH = "Three references do not say this."

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        if json_schema is not None:
            return super().complete(
                messages,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        self.cost.calls += 1
        self.cost.prompt_tokens += 900
        self.cost.completion_tokens += 30
        return Completion(
            text=self.PARAGRAPH, prompt_tokens=900, completion_tokens=30, model=self.model
        )


def summarising_engine() -> Engine:
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(SummarisingJudgeClient())  # type: ignore[arg-type]
    return engine


def test_summarize_without_a_key_stops_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary is a judge call too, so it needs the provider ``--judge`` needs."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = install(monkeypatch)
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY)), "--summarize"])

    assert result.exit_code == 2
    assert "error: --summarize needs --judge or a configured judge" in result.output
    assert "GROQ_API_KEY is not set" in result.output
    assert seen == {}  # no engine was built, so no model was loaded


def test_summarize_alone_costs_one_call_and_never_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec section 11.1: the summary adds exactly one call. Escalation is --judge's."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    seen = install(monkeypatch, summarising_engine())
    result = runner.invoke(app, ["check", str(write(tmp_path, CLEAN_BODY)), "--summarize"])

    assert seen["escalate"] is False
    assert f"summary    (model-written, fake judge-1) {SummarisingJudgeClient.PARAGRAPH}" in (
        result.output
    )
    assert "1 API calls" in result.output
    assert "gsk_not_a_real_key" not in result.output


def test_judge_and_summarize_together_escalate_and_then_summarise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    seen = install(monkeypatch, summarising_engine())
    result = runner.invoke(
        app,
        ["check", str(write(tmp_path, CLEAN_BODY)), "--judge", "--summarize", "--format", "json"],
    )

    assert seen["escalate"] is True
    payload = json.loads(result.stdout)
    assert payload["summary"] == SummarisingJudgeClient.PARAGRAPH
    assert payload["api_calls"] == 2
    assert payload["stages"][-1]["name"] == "Summarising"


def test_without_summarize_the_report_carries_no_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    install(monkeypatch, summarising_engine())
    result = runner.invoke(
        app, ["check", str(write(tmp_path, CLEAN_BODY)), "--judge", "--format", "json"]
    )

    payload = json.loads(result.stdout)
    assert payload["summary"] is None
    assert payload["api_calls"] == 1
    assert "model-written" not in result.output


def test_a_quiet_run_drops_the_summary_line_but_the_report_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-q`` keeps findings and drops the rest, and a summary is not a finding."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    install(monkeypatch, summarising_engine())
    written = tmp_path / "report.md"
    result = runner.invoke(
        app, ["-q", "check", str(write(tmp_path, CLEAN_BODY)), "--summarize", "--out", str(written)]
    )

    assert "model-written" not in result.stdout
    text = written.read_text(encoding="utf-8")
    assert "## Summary (model-written, fake judge-1)" in text
    assert SummarisingJudgeClient.PARAGRAPH in text


def test_a_summary_the_provider_would_not_write_is_reported_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Product rule 2: a silent provider must not read as a report with nothing to add."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(DownJudgeClient())  # type: ignore[arg-type]
    install(monkeypatch, engine)
    result = runner.invoke(
        app, ["check", str(write(tmp_path, CLEAN_BODY)), "--summarize", "--format", "json"]
    )

    payload = json.loads(result.stdout)
    assert payload["summary"] == ""  # asked, and answered with nothing
    assert payload["stages"][-1]["summary"] == (
        "summary unavailable "
        "(HTTP 401 from https://api.test/chat/completions); local verdicts stand"
    )
    assert payload["api_calls"] == 0


DOWN_DETAIL = "HTTP 401 from https://api.test/chat/completions"


def down_summary_engine() -> Engine:
    engine = built_engine(scorer=TableScorer({SUPPORTING: LOW_ROW}))
    engine.judge = Judge(DownJudgeClient())  # type: ignore[arg-type]
    return engine


def test_a_summary_that_never_came_is_printed_and_written_not_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``report.summary == ""`` is falsy, so without a line of its own the silence is
    invisible on both surfaces a text run has (product rules 2 and 6)."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    install(monkeypatch, down_summary_engine())
    written = tmp_path / "report.md"
    result = runner.invoke(
        app, ["check", str(write(tmp_path, CLEAN_BODY)), "--summarize", "--out", str(written)]
    )

    assert f"summary    judge unavailable ({DOWN_DETAIL}); local verdicts stand" in result.output
    assert "note       summary" not in result.output  # the stage and the line, not three
    text = written.read_text(encoding="utf-8")
    assert f"- summary status: unavailable ({DOWN_DETAIL}); local verdicts stand" in text
    assert "## Summary" not in text


def test_a_report_with_a_summary_says_nothing_about_one_going_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_not_a_real_key")
    install(monkeypatch, summarising_engine())
    written = tmp_path / "report.md"
    result = runner.invoke(
        app, ["check", str(write(tmp_path, CLEAN_BODY)), "--summarize", "--out", str(written)]
    )

    assert "judge unavailable" not in result.output
    assert "- summary status:" not in written.read_text(encoding="utf-8")
