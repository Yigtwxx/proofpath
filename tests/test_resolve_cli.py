from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from proofpath.cli import app

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures" / "resolve"
ALPHAFOLD = (
    "Jumper J, Evans R, et al. Highly accurate protein structure prediction with AlphaFold. "
    "Nature. 2021;596:583-589."
)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path))


@respx.mock
def test_resolve_prints_state_record_and_field_agreement() -> None:
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIX / "crossref_alphafold.json").read_text())
        )
    )
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/match").mock(
        return_value=httpx.Response(200, json=json.loads((FIX / "s2_alphafold.json").read_text()))
    )
    respx.get("https://api.crossref.org/works/10.1038/s41586-021-03819-2").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIX / "crossref_work_alphafold.json").read_text())
        )
    )
    respx.get("https://api.openalex.org/works/https://doi.org/10.1038/s41586-021-03819-2").mock(
        return_value=httpx.Response(200, json={"is_retracted": False})
    )
    result = runner.invoke(app, ["resolve", ALPHAFOLD])
    assert result.exit_code == 0, result.output
    assert "RESOLVED" in result.output
    assert "10.1038/s41586-021-03819-2" in result.output
    assert "title" in result.output and "author" in result.output and "year" in result.output
    assert "not retracted" in result.output


@respx.mock
def test_resolve_ghost_exits_with_findings_code() -> None:
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIX / "crossref_fabricated.json").read_text())
        )
    )
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/match").mock(
        return_value=httpx.Response(404, json={"error": "Title match not found"})
    )
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=(FIX / "arxiv_title_fabricated.xml").read_text())
    )
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIX / "openalex_fabricated.json").read_text())
        )
    )
    respx.get("https://openlibrary.org/search.json").mock(
        return_value=httpx.Response(
            200, json=json.loads((FIX / "openlibrary_fabricated.json").read_text())
        )
    )
    result = runner.invoke(
        app,
        [
            "resolve",
            "Zhang K, Okafor T. Neural cascade alignment for zero-shot citation grounding "
            "in scientific corpora. J Comp Verif. 2021.",
        ],
    )
    assert result.exit_code == 1
    assert "GHOST" in result.output
    assert "checked" in result.output  # candidates that were compared are listed


def _resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the resolver so the CLI layer can be exercised without provider fixtures."""
    from proofpath import resolve as rs

    best = rs.Candidate(
        doi="10.1038/s41586-021-03819-2",
        title="Highly accurate protein structure prediction with AlphaFold",
        first_author="Jumper",
        year=2021,
        venue="Nature",
        provider="crossref",
    )
    result = rs.ResolveResult(
        rs.State.RESOLVED, best, [best], notes=["a note"], match=rs.FieldMatch(1.0, True, True)
    )
    monkeypatch.setattr(rs.Resolver, "resolve", lambda self, raw: result)
    monkeypatch.setattr(
        rs.Resolver,
        "retraction",
        lambda self, doi: rs.Retraction("retraction-watch", "2024-01-02", "10.1/n", "Retraction"),
    )


def test_resolve_format_json_puts_only_the_document_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolved(monkeypatch)
    result = runner.invoke(app, ["resolve", "--format", "json", ALPHAFOLD])
    assert result.exit_code == 1, result.output  # _resolved() stubs a retraction
    payload = json.loads(result.stdout)
    assert payload["result"]["state"] == "RESOLVED"
    assert payload["result"]["best"]["doi"] == "10.1038/s41586-021-03819-2"
    assert payload["result"]["match"] == {"title": 1.0, "author": True, "year": True}
    assert payload["result"]["notes"] == ["a note"]
    assert payload["retraction"]["source"] == "retraction-watch"
    assert "state      " not in result.stdout


def test_resolve_json_retraction_is_null_when_not_retracted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from proofpath import resolve as rs

    _resolved(monkeypatch)
    monkeypatch.setattr(rs.Resolver, "retraction", lambda self, doi: None)
    result = runner.invoke(app, ["resolve", "--format", "json", ALPHAFOLD])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["retraction"] is None


def test_resolve_rejects_unknown_format(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolved(monkeypatch)
    assert runner.invoke(app, ["resolve", "--format", "yaml", ALPHAFOLD]).exit_code == 2


def test_resolve_text_shows_retraction_and_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolved(monkeypatch)
    result = runner.invoke(app, ["resolve", ALPHAFOLD])
    # RESOLVED, but retracted: a finding (spec 13.2 ``warning[retracted]``).
    assert result.exit_code == 1, result.output
    assert "state      RESOLVED" in result.stdout
    assert "retraction RETRACTED 2024-01-02" in result.stdout
    assert "note       a note" in result.stdout


def test_resolve_retracted_is_a_finding_even_when_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from proofpath import resolve as rs

    _resolved(monkeypatch)
    assert runner.invoke(app, ["resolve", ALPHAFOLD]).exit_code == 1
    monkeypatch.setattr(rs.Resolver, "retraction", lambda self, doi: None)
    assert runner.invoke(app, ["resolve", ALPHAFOLD]).exit_code == 0


def test_quiet_and_no_color_globals_keep_the_state_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolved(monkeypatch)
    result = runner.invoke(app, ["-q", "--no-color", "resolve", ALPHAFOLD])
    assert result.exit_code == 1, result.output  # the stubbed retraction is a finding
    assert "state      RESOLVED" in result.stdout
    assert "note       a note" not in result.stdout  # -q drops notes, never the state


def test_resolve_unavailable_is_a_finding_not_a_tool_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from proofpath import resolve as rs

    unavailable = rs.ResolveResult(rs.State.UNAVAILABLE, None, [], notes=["crossref unavailable"])
    monkeypatch.setattr(rs.Resolver, "resolve", lambda self, raw: unavailable)
    result = runner.invoke(app, ["resolve", ALPHAFOLD])
    assert result.exit_code == 1
    assert "UNVERIFIED (provider unavailable)" in result.stdout


@pytest.mark.parametrize(
    ("state", "code"),
    [
        ("RESOLVED", 0),
        ("RESOLVED_LOW", 1),
        ("AMBIGUOUS", 1),
        ("GHOST", 1),
        ("UNAVAILABLE", 1),
        ("NOT_INDEXED", 1),
    ],
)
def test_resolve_exit_code_follows_spec_13_3(
    monkeypatch: pytest.MonkeyPatch, state: str, code: int
) -> None:
    from proofpath import resolve as rs

    stub = rs.ResolveResult(rs.State[state], None, [])
    monkeypatch.setattr(rs.Resolver, "resolve", lambda self, raw: stub)
    assert runner.invoke(app, ["resolve", ALPHAFOLD]).exit_code == code
