"""``proofpath check TARGET``: wiring, output vocabulary, exit codes (spec section 13.3).

``Engine.default`` is monkeypatched to an engine of stubs, so the command is exercised
end to end — parsing, pairing, resolving, reading, verifying, rendering — without a
socket and without an ONNX session. The stubs are the ones ``tests/test_verify.py``
uses, kept here in their smallest form so the two files can move independently.
"""

from __future__ import annotations

import json
import signal
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from proofpath import verify as verify_mod
from proofpath.browser import ConsentGate
from proofpath.cli import _interruptible, app
from proofpath.config import Config
from proofpath.events import Cancelled
from proofpath.fetch import Fetched, FetchStats, Outcome
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
        self, url: str, *, text_kind: str = "fulltext", counts_as_source: bool = True
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
    embedder: Embedder | None = None,
    scorer: object | None = None,
) -> Engine:
    """The engine ``Engine.default`` is replaced by: every part a stub."""
    return Engine(
        config=Config(),
        cache=None,
        resolver=resolver or StubResolver({"Vaswani": resolved()}),
        fetcher=StubFetcher(),
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


def test_format_sarif_says_which_version_brings_it(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "-", "--format", "sarif"], input=draft(CLEAN_BODY))
    assert result.exit_code == 2
    assert "error: --format sarif arrives in v0.2" in result.output


def test_an_unknown_format_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "-", "--format", "xml"], input=draft(CLEAN_BODY))
    assert result.exit_code == 2


@pytest.mark.parametrize("flag", ["--judge", "--summarize"])
def test_judge_and_summarize_arrive_in_v0_3(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(app, ["check", "-", flag], input=draft(CLEAN_BODY))
    assert result.exit_code == 2
    assert f"error: {flag} arrives in v0.3" in result.output


def test_allow_browser_and_no_browser_cannot_be_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    result = runner.invoke(
        app, ["check", "-", "--allow-browser", "--no-browser"], input=draft(CLEAN_BODY)
    )
    assert result.exit_code == 2
    assert "--allow-browser" in result.output and "--no-browser" in result.output


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
