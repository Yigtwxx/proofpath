"""The open-access chain (spec sections 6.1, 9): full text, labelled abstract, or honesty state."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import respx

from proofpath import fetch as fx
from proofpath import oa
from proofpath import resolve as rs
from proofpath.cache import Cache
from proofpath.config import Config, FetchConfig, Permissions
from proofpath.polite import PoliteClient

FIX = Path(__file__).parent / "fixtures" / "oa"

DOI = "10.1038/s41586-021-03819-2"
ARXIV = "2103.00020"
PMCID = "PMC8371605"
PDF = "https://www.nature.com/articles/s41586-021-03819-2.pdf"
HTML_LINK = "https://www.nature.com/articles/s41586-021-03819-2"
EPMC_XML = f"{oa.EUROPEPMC}/{PMCID}/fullTextXML"
ARXIV_PDF = f"https://arxiv.org/pdf/{ARXIV}"
LANDING = f"https://doi.org/{DOI}"

S2_URL = f"{oa.S2_PAPER}/DOI:{DOI}"
S2_ARXIV_URL = f"{oa.S2_PAPER}/arXiv:{ARXIV}"
CROSSREF_URL = f"{rs.CROSSREF}/{DOI}"
UNPAYWALL_URL = f"{oa.UNPAYWALL}/{DOI}"
EPMC_SEARCH = f"{oa.EUROPEPMC}/search"
OPENALEX_URL = f"{rs.OPENALEX}/https://doi.org/{DOI}"

S2_ABSTRACT = (
    "Proteins are essential to life, and understanding their structure can facilitate a "
    "mechanistic understanding of their function."
)


def fixture(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((FIX / name).read_text(encoding="utf-8")))


def words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def html_page(n_words: int) -> bytes:
    return f"<html><body><article><p>{words(n_words)}</p></article></body></html>".encode()


def jats_article(n_words: int) -> bytes:
    paragraphs = "".join(f"<p>{words(100)}</p>" for _ in range(n_words // 100 + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8"?><article xmlns:xlink="http://www.w3.org/1999/xlink">'
        f"<front><article-title>T</article-title></front><body><sec>{paragraphs}</sec></body>"
        "</article>"
    ).encode()


def curl_forbidden(url: str) -> tuple[int, bytes, str, str]:
    raise AssertionError(f"step 2 must not run for {url}")


def curl_returning(status: int) -> fx.CurlGet:
    return lambda url: (status, b"", "text/html", url)


def wayback_none(url: str) -> str | None:
    return None


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fake time: sleep advances the clock, so throttling and backoff never really wait."""
    now = {"t": 1000.0}
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(fx.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(fx.time, "sleep", fake_sleep)
    return slept


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(follow_redirects=True) as c:
        yield c


def make(
    client: httpx.Client,
    *,
    contact_email: str = "",
    cache: Cache | None = None,
    curl: fx.CurlGet | None = curl_forbidden,
    network: str = "allow",
    gate: fx.BrowserGate | None = None,
) -> oa.OpenAccess:
    fetcher = fx.Fetcher(
        config=Config(
            permissions=Permissions(network=network),  # type: ignore[arg-type]
            fetch=FetchConfig(respect_robots=False),
        ),
        gate=gate,
        cache=cache,
        client=client,
        curl_get=curl,
        wayback_lookup=wayback_none,
        interactive=False,
    )
    polite = PoliteClient(contact_email=contact_email, client=client)
    return oa.OpenAccess(fetcher, polite, contact_email=contact_email, cache=cache)


def json_or_404(name: str | None) -> httpx.Response:
    return httpx.Response(200, json=fixture(name)) if name else httpx.Response(404)


def mock_providers(
    *,
    s2: str | None = "s2_paper.json",
    crossref: str | None = "crossref_work.json",
    epmc: str | None = "europepmc_search.json",
    unpaywall: str | None = None,
    openalex: str | None = None,
) -> dict[str, Any]:
    return {
        "s2": respx.get(S2_URL).mock(return_value=json_or_404(s2)),
        "crossref": respx.get(CROSSREF_URL).mock(return_value=json_or_404(crossref)),
        "epmc": respx.get(EPMC_SEARCH).mock(return_value=json_or_404(epmc)),
        "unpaywall": respx.get(UNPAYWALL_URL).mock(return_value=json_or_404(unpaywall)),
        "openalex": respx.get(OPENALEX_URL).mock(return_value=json_or_404(openalex)),
    }


def serve(url: str, status: int = 200, body: bytes = b"", ctype: str = "text/html") -> Any:
    return respx.get(url).mock(
        return_value=httpx.Response(status, content=body, headers={"content-type": ctype})
    )


def labels(evidence: oa.Evidence) -> list[str]:
    return [attempt.location.label for attempt in evidence.attempts]


# --- pure helpers on provider payloads ---------------------------------------


def test_locations_from_s2_reads_pdf_ids_and_abstract() -> None:
    location, arxiv_id, pmcid, abstract = oa.locations_from_s2(fixture("s2_paper.json"))
    assert location == oa.Location("s2_pdf", PDF, "pdf")
    assert arxiv_id == ARXIV
    assert pmcid == PMCID
    assert abstract == S2_ABSTRACT
    assert oa.locations_from_s2(fixture("s2_paper_no_abstract.json")) == (None, None, None, "")


def test_locations_from_crossref_filters_tdm_links_and_strips_jats() -> None:
    links, abstract = oa.locations_from_crossref(fixture("crossref_work.json"))
    # The unspecified-type duplicate of the PDF collapses; Elsevier/Wiley TDM endpoints
    # need a key; syndication links are not for text mining.
    assert links == [
        oa.Location("crossref_link", PDF, "pdf"),
        oa.Location("crossref_link", HTML_LINK, "html"),
    ]
    assert abstract == S2_ABSTRACT
    assert "<" not in abstract
    assert oa.locations_from_crossref({"message": {}}) == ([], "")


def test_location_from_unpaywall_prefers_pdf() -> None:
    assert oa.location_from_unpaywall(fixture("unpaywall.json")) == oa.Location(
        "unpaywall", PDF, "pdf"
    )
    assert oa.location_from_unpaywall(fixture("unpaywall_landing_only.json")) == oa.Location(
        "unpaywall", "https://europepmc.org/articles/pmc8371605", "html"
    )
    assert oa.location_from_unpaywall(fixture("unpaywall_closed.json")) is None


def test_pmcid_from_europepmc() -> None:
    assert oa.pmcid_from_europepmc(fixture("europepmc_search.json")) == PMCID
    assert oa.pmcid_from_europepmc(fixture("europepmc_search_not_in_epmc.json")) is None
    assert oa.pmcid_from_europepmc(fixture("europepmc_search_empty.json")) is None


def test_abstract_from_openalex_reorders_inverted_index() -> None:
    assert oa.abstract_from_openalex(fixture("openalex_work.json")) == (
        "Proteins are essential to life, and understanding their structure informs about "
        "their function."
    )
    assert oa.abstract_from_openalex({}) == ""
    assert oa.abstract_from_openalex({"abstract_inverted_index": None}) == ""


def test_jats_body_text_extracts_paragraphs() -> None:
    text = oa.jats_body_text((FIX / "europepmc_fulltext.xml").read_bytes())
    assert text.split("\n") == [
        "Main",
        "The trial enrolled 240 patients across three sites.",
        "Structure & function were compared; accuracy exceeded 90% on the held-out set.",
        "Fig. 1",
        "Overview of the network.",
    ]
    # Front matter and the reference list are not the article.
    assert "Proteins are essential" not in text
    assert "must not appear" not in text
    # The bibliographic citation marker is dropped whole, not glued onto "sites".
    assert "sites1" not in text
    # No <body>: the whole article, rather than nothing.
    assert oa.jats_body_text(b"<article><front><p>Only front.</p></front></article>") == (
        "Only front."
    )
    assert oa.strip_tags("<p>a &amp; <b>b</b></p>\n  c") == "a & b c"


def test_strip_tags_keeps_word_and_number_boundaries() -> None:
    # A number split across a tag boundary must not glue into one number.
    assert oa.strip_tags("12<xref>3</xref>") == "12 3"
    # A structured Crossref JATS abstract: headings must not glue onto the text
    # that follows them.
    structured = (
        "<jats:title>Background</jats:title>We studied mortality in a cohort."
        "<jats:title>Results</jats:title>Mortality fell by 12%."
    )
    assert oa.strip_tags(structured) == (
        "Background We studied mortality in a cohort. Results Mortality fell by 12%."
    )


def test_evidence_words_counts_the_text_like_fetched_words() -> None:
    evidence = oa.Evidence("fulltext", "", "one two three", "https://x.test", "s2_pdf", [], [])
    assert evidence.words == 3
    empty = oa.Evidence("none", "UNVERIFIED (unreachable)", "", "", "", [], [])
    assert empty.words == 0


@pytest.mark.parametrize(
    ("doi", "expected"),
    [
        ("10.48550/arXiv.1706.03762", "1706.03762"),
        ("10.48550/ARXIV.1706.03762", "1706.03762"),  # case-insensitive
        ("10.48550/arXiv.1706.03762v5", "1706.03762v5"),  # a pinned version is kept
        ("10.48550/arXiv.hep-th/9901001", "hep-th/9901001"),
        ("10.1038/s41586-021-03819-2", None),
        ("10.48550/other.1706.03762", None),
        ("", None),
    ],
)
def test_arxiv_id_from_doi(doi: str, expected: str | None) -> None:
    assert oa.arxiv_id_from_doi(doi) == expected


# --- locate -----------------------------------------------------------------


@respx.mock
def test_locate_order_and_dedupe(client: httpx.Client) -> None:
    routes = mock_providers()
    located = make(client).locate(DOI)
    assert [(loc.label, loc.url) for loc in located.locations] == [
        ("s2_pdf", PDF),  # Crossref's PDF link is the same URL: first label wins
        ("crossref_link", HTML_LINK),
        ("europepmc", EPMC_XML),
        ("arxiv", ARXIV_PDF),
        ("landing", LANDING),
    ]
    assert located.arxiv_id == ARXIV and located.pmcid == PMCID
    assert located.abstracts == {"s2": S2_ABSTRACT, "crossref": S2_ABSTRACT}
    assert located.notes == []
    assert routes["epmc"].call_count == 0  # S2 already gave the PMCID
    assert routes["openalex"].call_count == 0  # never in locate
    assert routes["unpaywall"].call_count == 0
    s2_request = routes["s2"].calls.last.request
    assert "mailto" not in str(s2_request.url)
    assert s2_request.url.params["fields"] == "openAccessPdf,abstract,externalIds"


@respx.mock
def test_locate_searches_europepmc_when_s2_has_no_pmcid(client: httpx.Client) -> None:
    routes = mock_providers(s2="s2_paper_no_abstract.json", crossref=None)
    located = make(client).locate(DOI)
    assert routes["epmc"].call_count == 1
    assert routes["epmc"].calls.last.request.url.params["query"] == f"DOI:{DOI}"
    assert located.pmcid == PMCID
    assert [loc.label for loc in located.locations] == ["europepmc", "landing"]


@respx.mock
def test_unpaywall_only_with_contact_email(client: httpx.Client) -> None:
    routes = mock_providers(unpaywall="unpaywall_landing_only.json")
    make(client).locate(DOI)
    assert routes["unpaywall"].call_count == 0

    located = make(client, contact_email="someone@example.org").locate(DOI)
    assert routes["unpaywall"].call_count == 1
    request = routes["unpaywall"].calls.last.request
    assert request.url.params["email"] == "someone@example.org"
    assert "mailto" not in str(request.url)
    assert [loc.label for loc in located.locations] == [
        "s2_pdf", "crossref_link", "unpaywall", "europepmc", "arxiv", "landing",
    ]  # fmt: skip


@respx.mock
def test_provider_failure_is_noted_and_chain_continues(client: httpx.Client) -> None:
    routes = mock_providers()
    routes["s2"].mock(return_value=httpx.Response(500))
    located = make(client).locate(DOI)
    assert routes["s2"].call_count == 3
    assert located.notes == ["s2 unavailable (HTTP 500)"]
    assert [loc.label for loc in located.locations] == [
        "crossref_link", "crossref_link", "europepmc", "landing",
    ]  # fmt: skip
    assert located.abstracts == {"crossref": S2_ABSTRACT}
    assert located.arxiv_id is None


@respx.mock
def test_invalid_json_is_noted_and_chain_continues(client: httpx.Client) -> None:
    routes = mock_providers()
    routes["s2"].mock(
        return_value=httpx.Response(200, content=b"not json", headers={"content-type": "text/html"})
    )
    located = make(client).locate(DOI)
    assert located.notes == ["s2 unavailable (invalid JSON)"]
    assert [loc.label for loc in located.locations] == [
        "crossref_link", "crossref_link", "europepmc", "landing",
    ]  # fmt: skip


@respx.mock
def test_locate_derives_arxiv_id_from_datacite_doi(client: httpx.Client) -> None:
    """A ``10.48550/arXiv.<id>`` DOI names the arXiv id itself: the PDF at arxiv.org
    is tried even when S2 and Crossref know nothing about the paper."""
    doi = "10.48550/arXiv.1706.03762"
    respx.get(f"{oa.S2_PAPER}/DOI:{doi}").mock(return_value=httpx.Response(404))
    respx.get(f"{rs.CROSSREF}/{doi}").mock(return_value=httpx.Response(404))
    respx.get(EPMC_SEARCH).mock(return_value=json_or_404("europepmc_search_empty.json"))
    located = make(client).locate(doi)
    assert [(loc.label, loc.url) for loc in located.locations] == [
        ("arxiv", "https://arxiv.org/pdf/1706.03762"),
        ("landing", f"https://doi.org/{doi}"),
    ]
    assert located.arxiv_id == "1706.03762"


@respx.mock
def test_locate_with_arxiv_id_only(client: httpx.Client) -> None:
    respx.get(S2_ARXIV_URL).mock(return_value=json_or_404("s2_paper_no_abstract.json"))
    located = make(client).locate(None, ARXIV)
    assert [(loc.label, loc.url) for loc in located.locations] == [("arxiv", ARXIV_PDF)]
    assert located.arxiv_id == ARXIV
    assert respx.calls.call_count == 1


# --- fetch ------------------------------------------------------------------


@respx.mock
def test_chain_prefers_pdf_then_pmc_then_landing(client: httpx.Client) -> None:
    mock_providers(crossref=None)
    serve(PDF, 403)
    serve(EPMC_XML, body=jats_article(1600), ctype="application/xml")
    evidence = make(client, curl=curl_returning(403)).fetch(DOI)
    assert evidence.kind == "fulltext"
    assert evidence.state == ""
    assert evidence.source == "europepmc"
    assert evidence.url == EPMC_XML
    assert len(evidence.text.split()) >= oa.FULLTEXT_MIN_WORDS
    assert labels(evidence) == ["s2_pdf", "europepmc"]
    assert evidence.attempts[0].outcome is fx.Outcome.BLOCKED_NO_BROWSER
    assert evidence.attempts[0].words == 0
    assert evidence.attempts[1].outcome is fx.Outcome.OK
    assert evidence.attempts[1].words >= oa.FULLTEXT_MIN_WORDS
    assert not evidence.from_cache


@respx.mock
def test_abstract_only_is_labelled_low_confidence(client: httpx.Client) -> None:
    routes = mock_providers(crossref=None, openalex="openalex_work.json")
    serve(PDF, body=html_page(300))  # reachable, but abstract-grade
    serve(EPMC_XML, 404)
    serve(ARXIV_PDF, 403)
    serve(LANDING, body=html_page(100))
    evidence = make(client, curl=curl_returning(403)).fetch(DOI)
    assert evidence.kind == "abstract"
    assert evidence.state == oa.ABSTRACT_ONLY == "LOW CONFIDENCE (abstract only)"
    assert evidence.source == "abstract:s2"
    assert evidence.text == S2_ABSTRACT
    assert evidence.url == S2_URL
    assert labels(evidence) == ["s2_pdf", "europepmc", "arxiv", "landing"]
    assert [a.outcome for a in evidence.attempts] == [
        fx.Outcome.OK,
        fx.Outcome.UNREACHABLE,
        fx.Outcome.BLOCKED_NO_BROWSER,
        fx.Outcome.OK,
    ]
    assert [a.words for a in evidence.attempts] == [300, 0, 0, 100]
    assert routes["openalex"].call_count == 0


@respx.mock
def test_short_page_text_beats_openalex(client: httpx.Client) -> None:
    routes = mock_providers(
        s2="s2_paper_no_abstract.json",
        crossref="crossref_work_no_abstract.json",
        epmc="europepmc_search_empty.json",
        openalex="openalex_work.json",
    )
    serve(HTML_LINK, body=html_page(200))
    serve(LANDING, body=html_page(800))
    evidence = make(client).fetch(DOI)
    assert evidence.kind == "abstract"
    assert evidence.state == oa.ABSTRACT_ONLY
    assert evidence.source == "landing"  # the longest under-threshold page
    assert evidence.url == LANDING
    assert len(evidence.text.split()) == 800
    assert routes["openalex"].call_count == 0


@respx.mock
def test_openalex_abstract_is_last_resort(client: httpx.Client) -> None:
    routes = mock_providers(
        s2="s2_paper_no_abstract.json",
        crossref="crossref_work_no_abstract.json",
        epmc="europepmc_search_empty.json",
        openalex="openalex_work.json",
    )
    serve(HTML_LINK, 404)
    serve(LANDING, 404)
    evidence = make(client).fetch(DOI)
    assert evidence.kind == "abstract"
    assert evidence.state == oa.ABSTRACT_ONLY
    assert evidence.source == "abstract:openalex"
    assert evidence.url == OPENALEX_URL
    assert evidence.text.startswith("Proteins are essential to life")
    assert routes["openalex"].call_count == 1
    assert labels(evidence) == ["crossref_link", "landing"]


@respx.mock
def test_none_reports_most_informative_state(client: httpx.Client) -> None:
    routes = mock_providers(
        s2="s2_paper_no_abstract.json",
        crossref="crossref_work_no_abstract.json",
        epmc="europepmc_search_empty.json",
    )
    serve(HTML_LINK, 403)
    serve(LANDING, 404)
    evidence = make(client, curl=curl_returning(403)).fetch(DOI)
    assert evidence.kind == "none"
    assert evidence.state == fx.Outcome.BLOCKED_NO_BROWSER.value  # blocked beats unreachable
    assert evidence.text == "" and evidence.url == "" and evidence.source == ""
    assert [a.outcome for a in evidence.attempts] == [
        fx.Outcome.BLOCKED_NO_BROWSER,
        fx.Outcome.UNREACHABLE,
    ]
    assert routes["openalex"].call_count == 1


@respx.mock
def test_none_when_all_attempts_ok_notes_no_text_extracted(client: httpx.Client) -> None:
    mock_providers(
        s2="s2_paper_no_abstract.json",
        crossref="crossref_work_no_abstract.json",
        epmc="europepmc_search_empty.json",
    )
    # Reached (200 OK) with a content type the ladder extracts nothing from. (An
    # empty *HTML* 200 is a bot wall, not content: fetch.MIN_HTML_WORDS.)
    serve(HTML_LINK, body=b"\x89PNG", ctype="image/png")
    serve(LANDING, body=b"\x89PNG", ctype="image/png")
    evidence = make(client).fetch(DOI)
    assert evidence.kind == "none"
    assert evidence.state == fx.Outcome.UNREACHABLE.value  # the state itself is unchanged
    assert [a.outcome for a in evidence.attempts] == [fx.Outcome.OK, fx.Outcome.OK]
    assert [a.words for a in evidence.attempts] == [0, 0]
    assert "reached but no text extracted" in evidence.notes


@respx.mock
def test_none_without_any_location_is_unreachable_and_noted(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.get(OPENALEX_URL).mock(return_value=httpx.Response(404))
    access = make(client)
    empty = oa.Located([], {}, None, None, ["s2 unavailable (HTTP 500)"])
    monkeypatch.setattr(access, "locate", lambda doi, arxiv_id=None: empty)
    evidence = access.fetch(DOI)
    assert evidence.kind == "none"
    assert evidence.state == fx.Outcome.UNREACHABLE.value
    assert evidence.attempts == []
    assert evidence.notes == ["s2 unavailable (HTTP 500)", "no open-access location found"]


@respx.mock
def test_provider_notes_travel_into_evidence(client: httpx.Client) -> None:
    routes = mock_providers(s2="s2_paper_no_abstract.json", crossref=None, epmc=None)
    routes["openalex"].mock(return_value=httpx.Response(503))
    serve(LANDING, 404)
    evidence = make(client).fetch(DOI)
    assert evidence.kind == "none"
    assert evidence.notes == ["openalex unavailable (HTTP 503)"]


@respx.mock
def test_fetch_uses_and_fills_cache(client: httpx.Client, tmp_path: Path) -> None:
    mock_providers(crossref=None)
    serve(PDF, body=html_page(1600))
    with Cache(tmp_path / "c.sqlite3") as cache:
        first = make(client, cache=cache).fetch(DOI)
        assert first.kind == "fulltext" and not first.from_cache
        assert first.url == PDF
        assert cache.get_raw_text(f"doi:{DOI}") == first.text
        assert cache.text_kind(f"doi:{DOI}") == "fulltext"
        seen = respx.calls.call_count

        second = make(client, cache=cache).fetch(DOI)
        assert respx.calls.call_count == seen
        assert second.kind == "fulltext" and second.state == ""
        assert second.from_cache and second.source == "cache"
        assert second.text == first.text
        # The URL the text came from survives a cache hit, not just the text (Important 2).
        assert second.url == first.url == PDF
        assert second.attempts == []

        cache.add_source(
            f"arxiv:{ARXIV}",
            scheme="academic",
            title="",
            url="https://arxiv.org/abs/" + ARXIV,
            text_kind="abstract",
            raw_text="only the abstract",
        )
        cached = make(client, cache=cache).fetch(None, ARXIV)
    assert respx.calls.call_count == seen
    assert cached.kind == "abstract"
    assert cached.state == oa.ABSTRACT_ONLY
    assert cached.from_cache and cached.text == "only the abstract"
    assert cached.url == "https://arxiv.org/abs/" + ARXIV


@respx.mock
def test_url_rows_are_relabelled_by_measured_length(client: httpx.Client, tmp_path: Path) -> None:
    """The ladder stores every page it fetched for the chain under ``url:`` as
    ``fulltext`` (it cannot know better); once the chain has counted the words,
    an abstract-grade page is relabelled and a full-text one stays."""
    mock_providers(
        s2="s2_paper_no_abstract.json",
        crossref="crossref_work_no_abstract.json",
        epmc="europepmc_search_empty.json",
    )
    serve(HTML_LINK, body=html_page(200))
    serve(LANDING, body=html_page(1600))
    with Cache(tmp_path / "c.sqlite3") as cache:
        evidence = make(client, cache=cache).fetch(DOI)
        assert evidence.kind == "fulltext" and evidence.source == "landing"
        assert cache.text_kind(f"url:{HTML_LINK}") == "abstract"
        assert cache.text_kind(f"url:{LANDING}") == "fulltext"


@respx.mock
def test_abstract_result_is_cached_as_abstract(client: httpx.Client, tmp_path: Path) -> None:
    mock_providers(crossref=None)
    serve(PDF, 404)
    serve(EPMC_XML, 404)
    serve(ARXIV_PDF, 404)
    serve(LANDING, 404)
    with Cache(tmp_path / "c.sqlite3") as cache:
        evidence = make(client, cache=cache).fetch(DOI)
        assert evidence.kind == "abstract"
        assert cache.get_raw_text(f"doi:{DOI}") == S2_ABSTRACT
        assert cache.text_kind(f"doi:{DOI}") == "abstract"


@respx.mock
def test_one_doi_with_several_blocked_locations_is_one_skipped_source(
    client: httpx.Client,
) -> None:
    """The report's ``skipped N source(s)`` counts sources, not the URLs tried for
    them: four walls behind one DOI are one source the browser could have rescued."""
    mock_providers()
    for url in (PDF, HTML_LINK, ARXIV_PDF, LANDING):
        serve(url, 403)
    serve(EPMC_XML, 404)
    gate = fx.DenyingGate()
    access = make(client, curl=curl_returning(403), gate=gate)
    evidence = access.fetch(DOI)
    assert evidence.kind == "abstract"  # S2's abstract; every page was a wall
    assert [a.outcome for a in evidence.attempts].count(fx.Outcome.BLOCKED_NO_BROWSER) == 4
    assert gate.skipped_urls == 4
    assert gate.skipped == 1
    assert access._fetcher.summary().browser_skipped == 1

    # A DOI whose locations were merely missing skipped nothing.
    gate = fx.DenyingGate()
    for url in (PDF, HTML_LINK, ARXIV_PDF, LANDING):
        serve(url, 404)
    assert make(client, gate=gate).fetch(DOI).kind == "abstract"
    assert (gate.skipped, gate.skipped_urls) == (0, 0)


@respx.mock
def test_network_deny_makes_no_provider_request(client: httpx.Client) -> None:
    """``permissions.network = deny`` binds the providers too, not only the ladder:
    the chain answers NETWORK_DENIED without a single request (spec section 15)."""
    mock_providers(openalex="openalex_work.json")
    serve(PDF, body=html_page(1600))
    access = make(client, network="deny")

    located = access.locate(DOI)
    assert located.locations == [] and located.abstracts == {}
    assert located.notes == ["network: not permitted (permissions.network = deny)"]

    evidence = access.fetch(DOI)
    assert evidence.kind == "none"
    assert evidence.state == fx.Outcome.NETWORK_DENIED.value
    assert evidence.text == "" and evidence.url == "" and evidence.source == ""
    assert evidence.attempts == []
    assert evidence.notes == ["network: not permitted (permissions.network = deny)"]
    assert respx.calls.call_count == 0


@respx.mock
def test_network_ask_without_a_tty_makes_no_provider_request(client: httpx.Client) -> None:
    mock_providers()
    evidence = make(client, network="ask").fetch(DOI)  # make() injects interactive=False
    assert evidence.kind == "none"
    assert evidence.state == fx.Outcome.NETWORK_DENIED.value
    assert evidence.notes == [
        "network: not permitted (permission is set to ask but there is no interactive terminal)"
    ]
    assert respx.calls.call_count == 0


def test_requires_doi_or_arxiv(client: httpx.Client) -> None:
    access = make(client)
    with pytest.raises(ValueError, match="DOI or an arXiv id"):
        access.locate(None)
    with pytest.raises(ValueError, match="DOI or an arXiv id"):
        access.fetch(None, None)
    with pytest.raises(ValueError, match="DOI or an arXiv id"):
        access.fetch("")
