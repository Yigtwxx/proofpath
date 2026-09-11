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
