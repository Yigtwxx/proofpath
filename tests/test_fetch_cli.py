"""``proofpath fetch TARGET``: wiring, output vocabulary, exit codes (spec section 13.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from proofpath import fetch as fx
from proofpath import oa
from proofpath import resolve as rs
from proofpath.cli import app

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures" / "oa"

URL = "https://x.test/paper.html"
WAYBACK = "https://archive.org/wayback/available"
NO_TTY_REASON = "permission is set to ask but there is no interactive terminal"

DOI = "10.1038/s41586-021-03819-2"
PMCID = "PMC8371605"
PDF = "https://www.nature.com/articles/s41586-021-03819-2.pdf"
EPMC_XML = f"{oa.EUROPEPMC}/{PMCID}/fullTextXML"
ARXIV_PDF = "https://arxiv.org/pdf/2103.00020"
LANDING = f"https://doi.org/{DOI}"
S2_URL = f"{oa.S2_PAPER}/DOI:{DOI}"
CROSSREF_URL = f"{rs.CROSSREF}/{DOI}"
OPENALEX_URL = f"{rs.OPENALEX}/https://doi.org/{DOI}"


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def html_page(n_words: int) -> bytes:
    return f"<html><body><article><p>{words(n_words)}</p></article></body></html>".encode()


HTML = html_page(24)  # at least fetch.MIN_HTML_WORDS: shorter HTML is a bot wall


def curl_returning(status: int) -> Any:
    """A stand-in for ``fetch.default_curl_get`` (step 2): same signature, fixed status."""

    def get(url: str, *, timeout: float = 20.0, contact_email: str = "") -> Any:
        return status, b"", "text/html", url

    return get


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh config and cache dirs, no real waits, and no real ``curl_cffi`` call."""
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))
    now = {"t": 1000.0}

    def fake_sleep(seconds: float) -> None:
        now["t"] += seconds

    monkeypatch.setattr(fx.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(fx.time, "sleep", fake_sleep)
    monkeypatch.setattr(fx, "default_curl_get", curl_returning(403))


def serve(url: str, status: int = 200, body: bytes = HTML, ctype: str = "text/html") -> Any:
    return respx.get(url).mock(
        return_value=httpx.Response(status, content=body, headers={"content-type": ctype})
    )


def robots(*urls: str) -> None:
    """No robots.txt on these hosts: the ladder treats a 404 as "allow everything"."""
    for url in urls:
        host = httpx.URL(url).copy_with(path="/robots.txt", query=None, fragment=None)
        respx.get(str(host)).mock(return_value=httpx.Response(404))


def wayback_miss() -> Any:
    return respx.get(WAYBACK).mock(
        return_value=httpx.Response(200, json={"archived_snapshots": {}})
    )


def fixture(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def mock_doi_providers(*, crossref: bool = True) -> None:
    respx.get(S2_URL).mock(return_value=httpx.Response(200, json=fixture("s2_paper.json")))
    respx.get(CROSSREF_URL).mock(
        return_value=httpx.Response(200, json=fixture("crossref_work.json"))
        if crossref
        else httpx.Response(404)
    )
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(404))
    robots(PDF, EPMC_XML, ARXIV_PDF, LANDING)
    wayback_miss()


# --- URL targets ---------------------------------------------------------------------


@respx.mock
def test_url_ok_prints_summary_and_exits_0() -> None:
    robots(URL)
    serve(URL)
    result = runner.invoke(app, ["fetch", URL])
    assert result.exit_code == 0, result.output
    assert "outcome    ok" in result.stdout
    assert "step       1 (httpx)" in result.stdout
    assert "status     200" in result.stdout
    assert "type       text/html · html" in result.stdout
    assert "words      24" in result.stdout
    assert f"url        {URL}" in result.stdout
    assert "cached     no" in result.stdout
    assert "note       step 1 httpx: HTTP 200 (ok)" in result.stdout
    # The gate was never consulted, so the permission lines are absent.
    assert "browser    " not in result.stdout
    assert "skipped    " not in result.stdout


@respx.mock
def test_url_reached_without_text_is_a_finding() -> None:
    """A 200 with nothing to read (an image, a corrupt PDF) is not a clean run:
    the report says so and the exit code is a finding (product rule 6)."""
    robots(URL)
    serve(URL, body=b"\x89PNG", ctype="image/png")
    result = runner.invoke(app, ["fetch", URL])
    assert result.exit_code == 1, result.output
    assert "outcome    ok" in result.stdout
    assert "words      0" in result.stdout
    assert "note       reached but no text extracted" in result.stdout


@respx.mock
def test_url_corrupt_pdf_is_a_finding_in_json_too() -> None:
    robots(URL)
    serve(URL, body=b"%PDF-1.4 not really a pdf", ctype="application/pdf")
    result = runner.invoke(app, ["fetch", "--format", "json", URL])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["result"]["outcome"] == "ok"
    assert payload["result"]["text"] == ""
    assert "reached but no text extracted" in payload["result"]["notes"]


@respx.mock
def test_url_blocked_without_tty_reports_skipped_and_exits_1() -> None:
    robots(URL)
    serve(URL, 403)
    wayback_miss()
    result = runner.invoke(app, ["fetch", URL])
    assert result.exit_code == 1, result.output
    assert "outcome    UNVERIFIED (blocked, browser not permitted)" in result.stdout
    assert "step       4 (wayback)" in result.stdout
    assert "status     403" in result.stdout
    assert "note       step 2 curl_cffi: HTTP 403 (blocked)" in result.stdout
    assert "note       step 3 browser: not permitted" in result.stdout
    assert f"browser    {NO_TTY_REASON}" in result.stdout
    assert "skipped    1 source(s) because the browser was not permitted" in result.stdout
    # Product rule 4: a non-interactive run never sees the consent prompt.
    assert "Allow?" not in result.output


@respx.mock
def test_no_browser_flag_sets_reason() -> None:
    robots(URL)
    serve(URL, 403)
    wayback_miss()
    result = runner.invoke(app, ["fetch", "--no-browser", URL])
    assert result.exit_code == 1, result.output
    assert "browser    --no-browser" in result.stdout
    assert "skipped    1 source(s)" in result.stdout


def test_conflicting_flags_exit_2() -> None:
    result = runner.invoke(app, ["fetch", "--allow-browser", "--no-browser", URL])
    assert result.exit_code == 2
    assert "--allow-browser" in result.output and "--no-browser" in result.output


def test_bad_target_exits_2() -> None:
    result = runner.invoke(app, ["fetch", "not a url, doi or arxiv id"])
    assert result.exit_code == 2
    assert "error:" in result.output
    assert "not a url, doi or arxiv id" in result.output


@respx.mock
def test_no_cache_flag_skips_cache() -> None:
    robots(URL)
    page = serve(URL)
    for _ in range(2):
        result = runner.invoke(app, ["fetch", "--no-cache", URL])
        assert result.exit_code == 0, result.output
        assert "cached     no" in result.stdout
    assert page.call_count == 2
    # Without the flag the first run fills the cache and the second one reads it.
    assert runner.invoke(app, ["fetch", URL]).exit_code == 0
    result = runner.invoke(app, ["fetch", URL])
    assert page.call_count == 3
    assert "cached     yes" in result.stdout
    assert "step       0 (cache)" in result.stdout


@respx.mock
def test_show_prints_text_prefix() -> None:
    robots(URL)
    serve(URL)
    result = runner.invoke(app, ["fetch", "--show", "11", URL])
    assert result.exit_code == 0, result.output
    assert "text       word0 word1\n" in result.stdout
    assert "word2" not in result.stdout


@respx.mock
def test_show_zero_prints_no_text() -> None:
    robots(URL)
    serve(URL)
    result = runner.invoke(app, ["fetch", URL])
    assert "text       " not in result.stdout


@respx.mock
def test_quiet_hides_notes_keeps_state_and_skipped() -> None:
    robots(URL)
    serve(URL, 403)
    wayback_miss()
    result = runner.invoke(app, ["-q", "fetch", URL])
    assert result.exit_code == 1, result.output
    assert "note       " not in result.stdout
    assert "outcome    UNVERIFIED (blocked, browser not permitted)" in result.stdout
    assert f"browser    {NO_TTY_REASON}" in result.stdout
    assert "skipped    1 source(s) because the browser was not permitted" in result.stdout


# --- DOI / arXiv targets -------------------------------------------------------------


@respx.mock
def test_doi_fulltext_exits_0() -> None:
    mock_doi_providers()
    serve(PDF, body=html_page(1600))
    result = runner.invoke(app, ["fetch", DOI])
    assert result.exit_code == 0, result.output
    assert "evidence   fulltext" in result.stdout
    assert "source     s2_pdf" in result.stdout
    assert "words      1600" in result.stdout
    assert f"url        {PDF}" in result.stdout
    assert "attempt    s2_pdf" in result.stdout and "ok" in result.stdout
    assert "step 1" in result.stdout and "1600 words" in result.stdout
    assert "browser    " not in result.stdout
    assert "skipped    " not in result.stdout


@respx.mock
def test_doi_abstract_only_exits_1() -> None:
    mock_doi_providers(crossref=False)
    serve(PDF, 403)
    serve(EPMC_XML, 404)
    serve(ARXIV_PDF, 403)
    serve(LANDING, 404)
    result = runner.invoke(app, ["fetch", DOI])
    assert result.exit_code == 1, result.output
    assert "evidence   abstract — LOW CONFIDENCE (abstract only)" in result.stdout
    assert "source     abstract:s2" in result.stdout
    assert f"url        {S2_URL}" in result.stdout
    assert "attempt    s2_pdf" in result.stdout
    assert "UNVERIFIED (blocked, browser not permitted)" in result.stdout
    assert "attempt    europepmc" in result.stdout and "UNVERIFIED (unreachable)" in result.stdout
    assert f"browser    {NO_TTY_REASON}" in result.stdout
    # Two locations were walls (s2_pdf, arxiv), but they belong to one source.
    assert "skipped    1 source(s) because the browser was not permitted" in result.stdout
    assert "Allow?" not in result.output


@respx.mock
def test_arxiv_prefixed_id_is_accepted() -> None:
    arxiv_url = f"{oa.S2_PAPER}/arXiv:2103.00020"
    respx.get(arxiv_url).mock(return_value=httpx.Response(404))
    robots(ARXIV_PDF)
    serve(ARXIV_PDF, body=html_page(1600))
    result = runner.invoke(app, ["fetch", "arXiv:2103.00020"])
    assert result.exit_code == 0, result.output
    assert "evidence   fulltext" in result.stdout
    assert "source     arxiv" in result.stdout


@respx.mock
def test_quiet_hides_attempt_lines_for_doi() -> None:
    mock_doi_providers(crossref=False)
    serve(PDF, 403)
    serve(EPMC_XML, 404)
    serve(ARXIV_PDF, 403)
    serve(LANDING, 404)
    result = runner.invoke(app, ["-q", "fetch", DOI])
    assert result.exit_code == 1, result.output
    assert "attempt    " not in result.stdout
    assert "note       " not in result.stdout
    assert "evidence   abstract — LOW CONFIDENCE (abstract only)" in result.stdout
    assert "source     abstract:s2" in result.stdout
    assert f"browser    {NO_TTY_REASON}" in result.stdout
    assert "skipped    1 source(s) because the browser was not permitted" in result.stdout


@respx.mock
def test_arxiv_bare_id_is_accepted() -> None:
    arxiv_url = f"{oa.S2_PAPER}/arXiv:2103.00020"
    respx.get(arxiv_url).mock(return_value=httpx.Response(404))
    robots(ARXIV_PDF)
    serve(ARXIV_PDF, body=html_page(1600))
    result = runner.invoke(app, ["fetch", "2103.00020"])
    assert result.exit_code == 0, result.output
    assert "evidence   fulltext" in result.stdout
    assert "source     arxiv" in result.stdout


# --- --format json -------------------------------------------------------------------


@respx.mock
def test_format_json_is_parseable_and_alone_on_stdout_for_a_url() -> None:
    robots(URL)
    serve(URL, 403)
    wayback_miss()
    result = runner.invoke(app, ["fetch", "--format", "json", URL])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["target"] == URL
    assert payload["result"]["outcome"] == "UNVERIFIED (blocked, browser not permitted)"
    assert payload["result"]["step"] == 4
    assert payload["result"]["status"] == 403
    assert payload["result"]["body"] is None  # bytes are never part of a report
    assert payload["stats"]["counts"] == {"UNVERIFIED (blocked, browser not permitted)": 1}
    assert payload["stats"]["browser_skipped"] == 1
    assert payload["browser"]["decision"] == {"outcome": "deny", "reason": NO_TTY_REASON}
    assert payload["browser"]["skipped"] == 1
    assert payload["browser"]["skipped_urls"] == 1
    assert payload["browser"]["install_log"] == []
    assert "outcome    " not in result.stdout


@respx.mock
def test_format_json_is_parseable_and_alone_on_stdout_for_a_doi() -> None:
    mock_doi_providers()
    serve(PDF, body=html_page(1600))
    result = runner.invoke(app, ["fetch", "--format", "json", DOI])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["target"] == DOI
    assert payload["result"]["kind"] == "fulltext"
    assert payload["result"]["source"] == "s2_pdf"
    assert payload["result"]["attempts"][0]["location"]["label"] == "s2_pdf"
    assert payload["result"]["attempts"][0]["outcome"] == "ok"
    assert payload["stats"]["counts"] == {"ok": 1}
    assert payload["browser"]["skipped"] == 0
    assert payload["browser"]["skipped_urls"] == 0
    assert "evidence   " not in result.stdout


@respx.mock
def test_show_under_format_json_is_hinted_on_stderr_not_silently_ignored() -> None:
    robots(URL)
    serve(URL)
    result = runner.invoke(app, ["fetch", "--format", "json", "--show", "5", URL])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)  # stdout is still one JSON document
    assert payload["result"]["outcome"] == "ok"
    assert "hint       --show is ignored under --format json" in result.stderr
    assert "hint" not in result.stdout


@respx.mock
def test_format_json_doi_skipped_counts_sources_not_urls() -> None:
    mock_doi_providers(crossref=False)
    serve(PDF, 403)
    serve(EPMC_XML, 404)
    serve(ARXIV_PDF, 403)
    serve(LANDING, 403)
    result = runner.invoke(app, ["fetch", "--format", "json", DOI])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["result"]["kind"] == "abstract"
    assert payload["stats"]["browser_skipped"] == 1
    assert payload["browser"]["skipped"] == 1
    assert payload["browser"]["skipped_urls"] == 3
