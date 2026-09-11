"""Reference resolution (spec section 8): candidates come from matchers, identity from fields."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from proofpath import resolve as rs

FIX = Path(__file__).parent / "fixtures" / "resolve"

ALPHAFOLD = (
    "Jumper J, Evans R, et al. Highly accurate protein structure prediction with AlphaFold. "
    "Nature. 2021;596:583-589."
)
RAG = (
    "Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive "
    "NLP tasks. Advances in Neural Information Processing Systems 33 (2020)."
)
FABRICATED = (
    "Zhang K, Okafor T, Lindqvist M. Neural cascade alignment for zero-shot citation grounding "
    "in scientific corpora. Journal of Computational Verification. 2021;14(3):211-229."
)


def fixture(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIX / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


# --- pure field checks --------------------------------------------------------


def test_tokens_are_lowercased_deaccented_and_stopword_free() -> None:
    assert rs.tokens("The Élan of Protein-Structure prediction, 2021!") == [
        "elan", "protein", "structure", "prediction", "2021",
    ]  # fmt: skip


def test_title_coverage_is_share_of_candidate_title_tokens_found_in_raw() -> None:
    assert (
        rs.title_coverage("Highly accurate protein structure prediction with AlphaFold", ALPHAFOLD)
        == 1.0
    )
    assert rs.title_coverage("Protein structure prediction review", ALPHAFOLD) == pytest.approx(
        3 / 4
    )
    assert rs.title_coverage("", ALPHAFOLD) == 0.0


def test_author_match_is_a_whole_token_and_diacritic_insensitive() -> None:
    assert rs.author_matches("Jumper", ALPHAFOLD) is True
    assert rs.author_matches("Jump", ALPHAFOLD) is False
    assert rs.author_matches("Gabaldón", "T. Gabaldon et al. 2021") is True
    assert rs.author_matches("van der Berg", "Berg, van der. 2020") is True


def test_year_match_allows_plus_minus_one() -> None:
    assert rs.year_matches(2021, ALPHAFOLD) is True
    assert rs.year_matches(2022, ALPHAFOLD) is True
    assert rs.year_matches(2019, ALPHAFOLD) is False
    assert rs.year_matches(None, ALPHAFOLD) is False


def test_find_doi_in_raw_string() -> None:
    assert (
        rs.find_doi("Smith J. Title. J Foo 2020. https://doi.org/10.1038/s41586-021-03819-2.")
        == "10.1038/s41586-021-03819-2"
    )
    assert rs.find_doi("doi:10.1016/S0140-6736(97)11096-0, 1998") == "10.1016/S0140-6736(97)11096-0"
    assert rs.find_doi(ALPHAFOLD) is None


def test_title_segments_keep_colons_quotes_and_drop_author_blocks() -> None:
    ieee = (
        '[12] A. Vaswani, N. Shazeer, and I. Polosukhin, "Attention is all you need," in '
        "Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998-6008."
    )
    assert rs.title_segments(ieee)[0] == "Attention is all you need"
    roberta = (
        "Liu Y, Ott M, Goyal N, et al. RoBERTa: A robustly optimized BERT pretraining approach. "
        "arXiv preprint arXiv:1907.11692, 2019."
    )
    assert (
        rs.title_segments(roberta)[0] == "RoBERTa: A robustly optimized BERT pretraining approach"
    )
    assert (
        rs.title_segments(RAG)[0]
        == "Retrieval-augmented generation for knowledge-intensive NLP tasks"
    )


# --- classification -----------------------------------------------------------


def cand(title: str, author: str, year: int | None, doi: str = "10.1/x") -> rs.Candidate:
    return rs.Candidate(
        doi=doi, title=title, first_author=author, year=year, venue="", provider="crossref"
    )


def test_all_three_fields_agree_is_resolved() -> None:
    c = cand("Highly accurate protein structure prediction with AlphaFold", "Jumper", 2021)
    result = rs.classify([c], ALPHAFOLD)
    assert result.state is rs.State.RESOLVED and result.best is c


def test_strong_title_with_one_field_off_is_low_confidence() -> None:
    c = cand("Highly accurate protein structure prediction with AlphaFold", "Jumper", 2018)
    assert rs.classify([c], ALPHAFOLD).state is rs.State.RESOLVED_LOW
    c = cand("Highly accurate protein structure prediction with AlphaFold", "Smith", 2021)
    assert rs.classify([c], ALPHAFOLD).state is rs.State.RESOLVED_LOW


def test_weak_title_with_author_and_year_is_ambiguous_and_lists_candidates() -> None:
    c1 = cand("Protein structure prediction with deep learning", "Jumper", 2021, "10.1/a")
    c2 = cand("Accurate structure prediction", "Jumper", 2021, "10.1/b")
    result = rs.classify([c1, c2], ALPHAFOLD)
    assert result.state is rs.State.AMBIGUOUS
    assert {c.doi for c in result.candidates} == {"10.1/a", "10.1/b"}
    assert result.best is None


def test_strong_title_with_both_other_fields_off_is_ambiguous_not_resolved() -> None:
    # Same title, different author and year: could be a different work with a common title.
    c = cand("Highly accurate protein structure prediction with AlphaFold", "Smith", 2015)
    assert rs.classify([c], ALPHAFOLD).state is rs.State.AMBIGUOUS


def test_nothing_agrees_is_ghost() -> None:
    cands = [
        cand("Exploiting prior tacit knowledge to enhance alignment", "Wang", 2025),
        cand("XeroAlign: zero-shot cross-lingual transformer alignment", "Gritta", 2021),
    ]
    result = rs.classify(cands, FABRICATED)
    assert result.state is rs.State.GHOST
    assert result.candidates == cands  # still reported, so the user can see what was checked


def test_no_candidates_at_all_is_ghost() -> None:
    assert rs.classify([], FABRICATED).state is rs.State.GHOST


def test_too_short_reference_is_ambiguous_never_ghost() -> None:
    result = rs.classify([], "Smith 2020")
    assert result.state is rs.State.AMBIGUOUS
    assert any("too short" in n for n in result.notes)


def test_best_candidate_is_the_highest_title_coverage() -> None:
    c1 = cand(
        "Highly accurate protein structure prediction with AlphaFold", "Jumper", 2021, "10.1/a"
    )
    c2 = cand(
        "Faculty Opinions recommendation of Highly accurate protein structure prediction "
        "with AlphaFold",
        "Herault",
        2021,
        "10.1/b",
    )
    assert rs.classify([c2, c1], ALPHAFOLD).best is c1


# --- candidates from recorded provider responses --------------------------------


def test_crossref_items_become_candidates() -> None:
    cands = rs.candidates_from_crossref(fixture("crossref_alphafold.json"))
    assert cands[0] == rs.Candidate(
        doi="10.1038/s41586-021-03819-2",
        title="Highly accurate protein structure prediction with AlphaFold",
        first_author="Jumper",
        year=2021,
        venue="Nature",
        provider="crossref",
    )


def test_openalex_results_become_candidates() -> None:
    cands = rs.candidates_from_openalex(fixture("openalex_rag.json"))
    assert cands and cands[0].provider == "openalex"
    # NeurIPS papers carry no DOI in OpenAlex; the record URL is the identity then.
    assert cands[0].doi == "" and cands[0].url.startswith("https://openalex.org/W")
    assert cands[0].key == cands[0].url.lower()
    assert cands[0].first_author == "Lewis" and cands[0].year == 2020
    assert any(c.doi.startswith("10.") for c in cands)


def test_retraction_is_read_from_crossref_updated_by() -> None:
    r = rs.retraction_from_crossref(fixture("crossref_retracted_wakefield.json"))
    assert r is not None
    assert r.date == "2010-02-06" and r.source == "retraction-watch"
    assert r.notice_doi == "10.1016/s0140-6736(10)60175-4"
    assert rs.retraction_from_crossref(fixture("crossref_work_alphafold.json")) is None


# --- the resolver over HTTP (recorded, no network) --------------------------------


def _mock(
    *,
    crossref: str | int = "crossref_fabricated.json",
    s2: str | int = "s2_fabricated.json",
    openalex: str | int = "openalex_fabricated.json",
    arxiv: str | int = "arxiv_title_fabricated.xml",
    openlibrary: str | int = "openlibrary_fabricated.json",
) -> dict[str, respx.Route]:
    def response(spec: str | int) -> httpx.Response:
        if isinstance(spec, int):
            return httpx.Response(spec)
        text = (FIX / spec).read_text(encoding="utf-8")
        status = 404 if spec.startswith("s2_") and "Title match not found" in text else 200
        return httpx.Response(status, text=text)

    return {
        "crossref": respx.get("https://api.crossref.org/works").mock(
            return_value=response(crossref)
        ),
        "s2": respx.get("https://api.semanticscholar.org/graph/v1/paper/search/match").mock(
            return_value=response(s2)
        ),
        "openalex": respx.get("https://api.openalex.org/works").mock(
            return_value=response(openalex)
        ),
        "arxiv": respx.get("https://export.arxiv.org/api/query").mock(return_value=response(arxiv)),
        "openlibrary": respx.get("https://openlibrary.org/search.json").mock(
            return_value=response(openlibrary)
        ),
    }


@respx.mock
def test_resolver_finds_a_real_reference_and_records_the_provider() -> None:
    _mock(crossref="crossref_alphafold.json", s2="s2_alphafold.json")
    result = rs.Resolver(contact_email="").resolve(ALPHAFOLD)
    assert result.state is rs.State.RESOLVED
    assert result.best is not None and result.best.doi == "10.1038/s41586-021-03819-2"


@respx.mock
def test_resolver_sends_polite_headers_without_hardcoding_an_email() -> None:
    routes = _mock(crossref="crossref_alphafold.json", s2="s2_alphafold.json")
    rs.Resolver(contact_email="someone@example.org").resolve(ALPHAFOLD)
    request = routes["crossref"].calls.last.request
    assert "proofpath" in request.headers["user-agent"]
    assert "mailto=someone%40example.org" in str(request.url)
    assert b"example.org" not in Path(rs.__file__).read_bytes()


@respx.mock
def test_semantic_scholar_rescues_a_paper_crossref_does_not_index() -> None:
    routes = _mock(crossref="crossref_rag.json", s2="s2_rag.json")
    result = rs.Resolver().resolve(RAG)
    assert result.state in (rs.State.RESOLVED, rs.State.RESOLVED_LOW)
    assert result.best is not None and result.best.provider == "s2"
    assert not routes["arxiv"].called and not routes["openalex"].called


@respx.mock
def test_resolver_calls_a_fabricated_reference_a_ghost_after_all_providers() -> None:
    routes = _mock()
    result = rs.Resolver().resolve(FABRICATED)
    assert result.state is rs.State.GHOST
    assert all(routes[name].called for name in ("crossref", "s2", "arxiv", "openalex"))


@respx.mock
def test_provider_outage_is_reported_never_a_ghost() -> None:
    _mock(crossref=503, s2=503, openalex=503, arxiv=503)
    result = rs.Resolver(retries=0).resolve(FABRICATED)
    assert result.state is rs.State.UNAVAILABLE
    assert any("crossref" in n for n in result.notes)


@respx.mock
def test_one_required_provider_down_still_resolves_with_a_note() -> None:
    _mock(crossref=429, s2="s2_rag.json")
    result = rs.Resolver(retries=0).resolve(RAG)
    assert result.state in (rs.State.RESOLVED, rs.State.RESOLVED_LOW)
    assert any("crossref" in n and "unavailable" in n for n in result.notes)


@respx.mock
def test_openalex_budget_exhaustion_does_not_block_a_ghost_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAlex's free tier is a daily budget (measured 2026-09-11: 1000 credits,
    # a search costs 10). Once spent it answers 429 with a Retry-After of hours.
    _mock(openalex=429)
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "53991"})
    )
    slept: list[float] = []
    monkeypatch.setattr(rs.time, "sleep", lambda s: slept.append(s))
    result = rs.Resolver().resolve(FABRICATED)
    assert result.state is rs.State.GHOST
    assert any("openalex" in n and "unavailable" in n for n in result.notes)
    assert not any(s > rs.MAX_RETRY_AFTER for s in slept)


@respx.mock
def test_doi_in_the_reference_is_resolved_directly_first() -> None:
    respx.get("https://api.crossref.org/works/10.1038/s41586-021-03819-2").mock(
        return_value=httpx.Response(200, json=fixture("crossref_work_alphafold.json"))
    )
    result = rs.Resolver().resolve(ALPHAFOLD + " https://doi.org/10.1038/s41586-021-03819-2")
    assert result.state is rs.State.RESOLVED
    assert result.best is not None and result.best.provider == "doi"


@respx.mock
def test_dead_doi_is_noted_and_search_continues() -> None:
    respx.get("https://api.crossref.org/works/10.9999/nope").mock(return_value=httpx.Response(404))
    _mock()
    result = rs.Resolver().resolve(FABRICATED + " doi:10.9999/nope")
    assert result.state is rs.State.GHOST
    assert any("10.9999/nope" in n and "not resolve" in n for n in result.notes)


@respx.mock
def test_retraction_check_uses_crossref_then_openalex() -> None:
    respx.get("https://api.crossref.org/works/10.1016/S0140-6736(97)11096-0").mock(
        return_value=httpx.Response(200, json=fixture("crossref_retracted_wakefield.json"))
    )
    r = rs.Resolver().retraction("10.1016/S0140-6736(97)11096-0")
    assert r is not None and r.date == "2010-02-06"
    respx.get("https://api.crossref.org/works/10.1038/s41586-021-03819-2").mock(
        return_value=httpx.Response(200, json=fixture("crossref_work_alphafold.json"))
    )
    respx.get("https://api.openalex.org/works/https://doi.org/10.1038/s41586-021-03819-2").mock(
        return_value=httpx.Response(200, json={"is_retracted": False})
    )
    assert rs.Resolver().retraction("10.1038/s41586-021-03819-2") is None


def test_semantic_scholar_match_becomes_a_candidate() -> None:
    (cand,) = rs.candidates_from_s2(fixture("s2_rag.json"))
    assert cand.provider == "s2" and cand.first_author == "Lewis" and cand.year == 2020
    # No DOI for the NeurIPS version, but S2 knows the arXiv id -> arXiv DOI form.
    assert cand.doi == "10.48550/arXiv.2005.11401"
    assert cand.url.startswith("https://www.semanticscholar.org/paper/")
    (cand,) = rs.candidates_from_s2(fixture("s2_roberta.json"))
    assert cand.doi == "10.48550/arXiv.1907.11692"
    assert rs.candidates_from_s2(fixture("s2_fabricated.json")) == []


# --- arXiv: the third provider, consulted before any ghost call ------------------

ROBERTA = (
    "Liu Y, Ott M, Goyal N, et al. RoBERTa: A robustly optimized BERT pretraining approach. "
    "arXiv preprint arXiv:1907.11692, 2019."
)


def test_leading_author_block_is_stripped_before_segmenting() -> None:
    assert rs.title_segments(FABRICATED)[0] == (
        "Neural cascade alignment for zero-shot citation grounding in scientific corpora"
    )
    apa = (
        "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep "
        "bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019."
    )
    assert rs.title_segments(apa)[0].startswith("BERT: Pre-training of deep bidirectional")


def test_find_arxiv_id() -> None:
    assert rs.find_arxiv_id(ROBERTA) == "1907.11692"
    assert rs.find_arxiv_id("see https://arxiv.org/abs/2312.10997v2 for details") == "2312.10997"
    assert rs.find_arxiv_id("arXiv:hep-th/9901001") == "hep-th/9901001"
    assert rs.find_arxiv_id(ALPHAFOLD) is None


def test_find_arxiv_id_accepts_a_bare_id_with_no_prefix() -> None:
    assert rs.find_arxiv_id("2103.00020") == "2103.00020"
    assert rs.find_arxiv_id("2103.00020v2") == "2103.00020"
    assert rs.find_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert rs.find_arxiv_id("  2103.00020  ") == "2103.00020"  # CLI args may carry whitespace


def test_find_arxiv_id_does_not_match_a_bare_number_inside_a_longer_reference() -> None:
    # A bare id is only accepted when it is the *entire* string; the same shape
    # inside a real reference must still require the "arxiv" marker to match.
    assert rs.find_arxiv_id("results for sample 2103.00020 were inconclusive") is None
    assert rs.find_arxiv_id(f"{ALPHAFOLD} 2103.00020") is None


def test_arxiv_entries_become_candidates() -> None:
    xml = (FIX / "arxiv_title_roberta.xml").read_text(encoding="utf-8")
    (cand,) = rs.candidates_from_arxiv(xml)
    assert cand == rs.Candidate(
        doi="10.48550/arXiv.1907.11692",
        title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        first_author="Liu",
        year=2019,
        venue="arXiv",
        provider="arxiv",
    )
    assert (
        rs.candidates_from_arxiv((FIX / "arxiv_title_fabricated.xml").read_text(encoding="utf-8"))
        == []
    )


@respx.mock
def test_arxiv_id_in_the_reference_is_resolved_directly() -> None:
    _mock(arxiv="arxiv_id_roberta.xml")
    result = rs.Resolver().resolve(ROBERTA)
    assert result.state is rs.State.RESOLVED
    assert result.best is not None and result.best.provider == "arxiv"


@respx.mock
def test_arxiv_title_search_rescues_a_paper_missing_from_the_primary_providers() -> None:
    _mock(
        crossref="crossref_roberta.json",
        openalex="openalex_roberta.json",
        arxiv="arxiv_title_roberta.xml",
    )
    raw = ROBERTA.replace(" arXiv preprint arXiv:1907.11692, 2019.", " 2019.")
    result = rs.Resolver().resolve(raw)
    assert result.state is rs.State.RESOLVED
    assert result.best is not None and result.best.provider == "arxiv"


@respx.mock
def test_ghost_requires_crossref_s2_and_arxiv_to_come_up_empty() -> None:
    result = rs.Resolver().resolve(FABRICATED) if _mock() else None
    assert result is not None and result.state is rs.State.GHOST
    assert any("arxiv" in n for n in result.notes)


@respx.mock
def test_arxiv_outage_turns_a_would_be_ghost_into_unavailable() -> None:
    _mock(arxiv=503)
    assert rs.Resolver(retries=0).resolve(FABRICATED).state is rs.State.UNAVAILABLE


# --- politeness: throttling and Retry-After ----------------------------------------


@respx.mock
def test_retry_after_is_honoured_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(rs.time, "sleep", lambda s: slept.append(s))
    route = respx.get("https://api.openalex.org/works")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json=fixture("openalex_alphafold.json")),
    ]
    resolver = rs.Resolver(retries=1)
    cands = resolver.openalex_search("Highly accurate protein structure prediction with AlphaFold")
    assert cands and route.call_count == 2
    assert 3.0 in slept


@respx.mock
def test_requests_to_one_host_are_spaced_out(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 100.0}
    slept: list[float] = []
    monkeypatch.setattr(rs.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(rs.time, "sleep", lambda s: slept.append(s))
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200, text=(FIX / "arxiv_title_roberta.xml").read_text(encoding="utf-8")
        )
    )
    resolver = rs.Resolver()
    resolver.arxiv_title("RoBERTa")
    resolver.arxiv_title("RoBERTa")  # same instant -> must wait the arXiv minimum interval
    assert slept and slept[-1] == pytest.approx(rs.MIN_INTERVAL["export.arxiv.org"])


def test_short_title_is_not_penalised_against_a_long_venue_segment() -> None:
    raw = (
        "Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. "
        "Advances in Neural Information Processing Systems 30 (2017)."
    )
    assert rs.title_score("Attention is All you Need", raw) == pytest.approx(1.0)
    c = cand("Attention is All you Need", "Vaswani", 2017)
    assert rs.classify([c], raw).state is rs.State.RESOLVED


# --- lessons from the ghost set, round 1 (false-ghost rate was 20.8 %) ---------------

ACM_FULL_NAMES = (
    "Robert C. Moore and William Lewis. 2010. Intelligent Selection of Language Model Training "
    "Data. In Proceedings of the ACL 2010 Conference Short Papers. 220-224."
)
GPT3 = (
    "Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, "
    "Arvind Neelakantan, and Dario Amodei. 2020. Language Models are Few-Shot Learners. "
    "arXiv:2005.14165 [cs.CL]"
)


def test_acm_style_author_list_with_full_names_is_stripped() -> None:
    assert rs.title_segments(ACM_FULL_NAMES)[0] == (
        "Intelligent Selection of Language Model Training Data"
    )
    assert rs.title_segments(GPT3)[0] == "Language Models are Few-Shot Learners"


def test_title_score_ignores_the_length_ratio_when_no_segment_overlaps() -> None:
    raw = (
        "Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Hanna Wallach, Hal Daumé III, and "
        "Kate Crawford. 2018. Datasheets for Datasets. arXiv:1803.09010 [cs.DB]"
    )
    assert rs.title_score("Datasheets for Datasets", raw) == pytest.approx(1.0)


def test_arxiv_id_without_the_dot_is_normalised() -> None:
    assert (
        rs.find_arxiv_id("Layer normalization. arXiv preprint arXiv:160706450. 2016;.")
        == "1607.06450"
    )


def test_two_word_titles_still_yield_a_segment() -> None:
    raw = "Ba JL, Kiros JR, Hinton GE. Layer normalization. arXiv preprint arXiv:160706450. 2016;."
    assert "Layer normalization" in rs.title_segments(raw)
    assert "Using Language" in rs.title_segments(
        "Herbert H. Clark. 1996. Using Language. Cambridge University Press, Cambridge."
    )


@pytest.mark.parametrize(
    "raw",
    [
        "World Health Organization. WHO Director-General's opening remarks at the media briefing "
        "on COVID-19 — 11 March 2020 (https://www.who.int/dg/speeches/detail/who-director-general).",
        "Leslie Kay Jones. 2020. Twitter wants you to know that you're still SOL if you get a "
        "death threat. https://medium.com/@agua.carbonica/twitter-wants-you-to-know.",
        "Food and Drug Administration. Guidance for industry: emergency use authorization for "
        "vaccines to prevent COVID-19. October 2020.",
    ],
)
def test_web_pages_and_organisation_authored_documents_are_never_ghosts(raw: str) -> None:
    result = rs.classify([], raw)
    assert result.state is rs.State.NOT_INDEXED
    assert result.state is not rs.State.GHOST


def test_author_hint_is_the_first_surname() -> None:
    assert rs.author_hint("Herbert H. Clark. 1996. Using Language. CUP.") == "Clark"
    assert rs.author_hint(ALPHAFOLD) == "Jumper"
    assert rs.author_hint("Lam, S. K., Pitrou, A. & Seibert, S. Numba. 2015.") == "Lam"
    assert rs.author_hint(ACM_FULL_NAMES) == "Moore"
    assert rs.author_hint("World Health Organization. Remarks. 2020.") == ""


def test_openlibrary_docs_become_candidates() -> None:
    cands = rs.candidates_from_openlibrary(fixture("openlibrary_using_language.json"))
    clark = next(c for c in cands if c.first_author == "Clark")
    assert clark.title == "Using language" and clark.year == 1996
    assert clark.provider == "openlibrary"
    assert clark.doi == "" and clark.url.startswith("https://openlibrary.org/works/")
    assert rs.candidates_from_openlibrary(fixture("openlibrary_fabricated.json")) == []


@respx.mock
def test_a_book_is_resolved_through_open_library_before_any_ghost_call() -> None:
    routes = _mock(openlibrary="openlibrary_using_language.json")
    result = rs.Resolver().resolve(
        "Herbert H. Clark. 1996. Using Language. Cambridge University Press, Cambridge."
    )
    assert "author=Clark" in str(routes["openlibrary"].calls.last.request.url)
    assert result.state is rs.State.RESOLVED
    assert result.best is not None and result.best.provider == "openlibrary"


@respx.mock
def test_ghost_call_consults_open_library_too() -> None:
    routes = _mock()
    assert rs.Resolver().resolve(FABRICATED).state is rs.State.GHOST
    assert routes["openlibrary"].called and routes["arxiv"].called


def test_dedupe_keeps_the_record_with_the_fuller_title() -> None:
    raw = "Lam, S. K. & Seibert, S. Numba: a LLVM-based Python JIT compiler. Proc. LLVM-HPC (2015)."
    poor = rs.Candidate("10.1145/2833157.2833162", "Numba", "Lam", 2015, "", "crossref")
    full = rs.Candidate(
        "10.1145/2833157.2833162", "Numba: a LLVM-based Python JIT compiler", "Lam", 2015, "", "s2"
    )
    assert rs.dedupe([poor, full], raw) == [full]
    assert rs.classify(rs.dedupe([poor, full], raw), raw).state is rs.State.RESOLVED


@respx.mock
def test_second_round_runs_on_ambiguous_too() -> None:
    # Crossref offers a weak-title candidate with author and year (AMBIGUOUS); the
    # book itself is only in Open Library. The second round must still run.
    routes = _mock(
        crossref="crossref_alphafold.json", openlibrary="openlibrary_using_language.json"
    )
    raw = "Herbert H. Clark. 1996. Using Language. Cambridge University Press, Cambridge."
    result = rs.Resolver().resolve(raw)
    assert routes["openlibrary"].called
    assert result.state is rs.State.RESOLVED and result.best is not None
    assert result.best.provider == "openlibrary"


# --- lessons from the ghost set, round 2 (false-ghost rate was 4.7 %) --------------


def test_trailing_publisher_parenthetical_is_dropped_from_segments() -> None:
    raw = "Van Rossum, G. & Drake, F. L. Python 3 Reference Manual (CreateSpace, 2009)."
    assert rs.title_segments(raw)[0] == "Python 3 Reference Manual"


def test_nature_style_annotation_does_not_hijack_the_year_delimiter() -> None:
    raw = (
        "Virtanen, P. et al. SciPy 1.0—fundamental algorithms for scientific computing in "
        "Python. Nat. Methods 17, 261–272 (2020). Introduces the SciPy library and includes a "  # noqa: RUF001
        "more detailed history of NumPy and SciPy."
    )
    assert rs.title_segments(raw)[0].startswith("SciPy 1.0")


@pytest.mark.parametrize(
    "raw",
    [
        "Corby Rosset. 2020. Turing-NLG: A 17-billion-parameter language model by Microsoft. "
        "Microsoft Blog (2020).",
        "Chiu, Y. H. & Dubois, P. F. Basis System, Part IV: EZD User Manual UCRL-MA-118543, "
        "Vol. 4 (Lawrence Livermore National Laboratory, 1994).",
        "Brendan Kennedy, Drew Kogon, and Morteza Dehghani. 2018. A typology and coding manual "
        "for the study of hate-based rhetoric. PsyArXiv. July 18 (2018).",
    ],
)
def test_blogs_manuals_and_unindexed_preprint_servers_are_not_ghosts(raw: str) -> None:
    assert rs.classify([], raw).state is rs.State.NOT_INDEXED


def test_paper_like_fabrications_stay_ghostable() -> None:
    for raw in (
        FABRICATED,
        'J. Demir and N. Oyelaran, "Self-supervised citation grounding for peer review," '
        "Annals of Evidence Science, vol. 40, pp. 394-1369, 2022.",
        "Kaur, K., & Duval, P. (2019). Adaptive numeric consistency checking with weak "
        "supervision. PLoS ONE, 107, 150-1436.",
    ):
        assert rs.classify([], raw).state is rs.State.GHOST, raw


def test_very_long_full_name_author_lists_are_still_stripped() -> None:
    authors = ", ".join(f"Author{i} Surname{i}" for i in range(30)) + ", and Dario Amodei"
    raw = f"{authors}. 2020. Language Models are Few-Shot Learners. In NeurIPS 33. https://x.y/z"
    assert rs.title_segments(raw)[0] == "Language Models are Few-Shot Learners"


def test_capitalised_name_lists_look_like_authors() -> None:
    assert rs._looks_like_authors(
        "Ziegler, Jeffrey Wu, Clemens Winter, Mark Chen, and Dario Amodei"
    )
    assert not rs._looks_like_authors("Language Models are Few-Shot Learners")


def test_a_candidate_matching_only_the_venue_segment_scores_zero() -> None:
    raw = (
        "Mbeki, G., Bianchi, A. D., & Tanaka, S. M. (2024). Epigenetic turnover in coral "
        "holobionts. Advances in Neural Information Processing Systems, 37, 150-1436."
    )
    proceedings = cand("Advances in Neural Information Processing Systems 37", "", 2024)
    assert rs.title_score(proceedings.title, raw) == 0.0
    assert rs.classify([proceedings], raw).state is rs.State.GHOST


def test_open_library_candidates_need_the_author_to_agree() -> None:
    raw = "World Health Organization. WHO Director-General's opening remarks. 2020."
    book = rs.Candidate("", "World Health Organization", "Lee", 2019, "", "openlibrary", url="u")
    assert rs.classify([book], raw).state is not rs.State.RESOLVED_LOW
