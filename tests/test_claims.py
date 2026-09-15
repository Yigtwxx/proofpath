"""Claims: citation markers, the sentence each one belongs to, and what is skipped.

One test per rule of the brief (spec section 9 step 2); the test name says which.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from proofpath import ingest
from proofpath.claims import (
    AUTHOR_YEAR,
    NUMERIC,
    expand,
    extract,
    find_markers,
    pair,
    strip_markers,
)
from proofpath.document import Document

EN_DASH = "\u2013"
EM_DASH = "\u2014"


def _document(body: str, *, references: int = 20, numbers: Sequence[int] | None = None) -> Document:
    """A document from inline text, with a bibliography long enough to resolve into.

    ``numbers`` prints exactly those entry numbers, so a gapped list can be built.
    """
    printed = list(numbers) if numbers is not None else list(range(1, references + 1))
    if not printed:
        return ingest.from_text(body, name="pasted.txt", kind="text")
    entries = "\n".join(
        f"[{number}] Author, A. Title number {number}. Journal, 2020." for number in printed
    )
    return ingest.from_text(f"{body}\n\nReferences\n\n{entries}\n", name="paper.txt", kind="text")


# --- `expand` and the marker patterns -------------------------------------------------


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("12", (12,)),
        ("12, 15", (12, 15)),
        ("12,15", (12, 15)),
        ("12-15", (12, 13, 14, 15)),
        (f"12{EN_DASH}15", (12, 13, 14, 15)),
        (f"12{EM_DASH}15", (12, 13, 14, 15)),
        ("3;7", (3, 7)),
        ("3; 7", (3, 7)),
        ("15-12", (12, 13, 14, 15)),
        ("7, 3, 7", (3, 7)),
        ("2, 4-6", (2, 4, 5, 6)),
    ],
)
def test_expand_table(group: str, expected: tuple[int, ...]) -> None:
    assert expand(group) == expected


def test_numeric_markers_cover_lists_ranges_and_both_dashes() -> None:
    document = _document(
        "A single source [12] and a list [12, 15] and a semicolon list [3; 7].\n"
        "\n"
        f"A hyphen range [12-15] and an en dash range [12{EN_DASH}15] follow."
    )
    markers = find_markers(document)
    assert [marker.text for marker in markers] == [
        "[12]",
        "[12, 15]",
        "[3; 7]",
        "[12-15]",
        f"[12{EN_DASH}15]",
    ]
    assert [marker.refs for marker in markers] == [
        (12,),
        (12, 15),
        (3, 7),
        (12, 13, 14, 15),
        (12, 13, 14, 15),
    ]
    assert {marker.style for marker in markers} == {"numeric"}


def test_a_section_number_or_a_table_label_is_not_a_numeric_marker() -> None:
    assert NUMERIC.search("see section [7.4] for the derivation") is None
    assert NUMERIC.search("the layout of [Table 1] is unchanged") is None
    assert NUMERIC.search("the column header [12a] is a label") is None
    document = _document("See section [7.4] and [Table 1] for the derivation.")
    assert find_markers(document) == []


def test_a_figure_reference_with_a_year_is_not_an_author_year_marker() -> None:
    assert AUTHOR_YEAR.search("the layout (Figure 2020) is unchanged") is None
    assert AUTHOR_YEAR.search("the run (Batch 1999) was discarded") is None
    document = _document("The layout is shown in the appendix (Figure 2020) and nowhere else.")
    result = extract(document)
    assert result.markers == ()
    assert result.unsupported == ()
    assert result.claims == ()


def test_find_markers_sorts_by_paragraph_and_start() -> None:
    document = _document(
        "First paragraph cites [3] and then [7] later.\n\nSecond paragraph cites [5] once."
    )
    markers = find_markers(document)
    assert [(marker.paragraph, marker.start) for marker in markers] == sorted(
        (marker.paragraph, marker.start) for marker in markers
    )
    assert [marker.text for marker in markers] == ["[3]", "[7]", "[5]"]


def test_find_markers_keeps_both_halves_of_a_mixed_citation() -> None:
    # OPEN-ITEMS 9.2: v0.1 let the numeric marker win the overlap and the author-year
    # half was reported nowhere. Both are markers now; the parenthesis is one citation
    # to a reader, so the author-year marker keeps the whole of it as its text.
    document = _document("A mixed citation (see Smith, 2020; [12]) appears here.")
    markers = find_markers(document)
    assert [(marker.style, marker.text) for marker in markers] == [
        ("author-year", "(see Smith, 2020; [12])"),
        ("numeric", "[12]"),
    ]


def test_a_parenthesis_holding_only_numbers_is_not_an_author_year_marker() -> None:
    # "(see [20])" names no author: it is the numeric marker's parenthesis, not a second
    # citation, and emitting one would report a style the passage never used.
    document = _document("Only one model reproduces the observed rate (see [20]).")
    assert [(marker.style, marker.text) for marker in find_markers(document)] == [
        ("numeric", "[20]")
    ]


# --- Pairing rules ---------------------------------------------------------------------


def test_rule_1_a_marker_belongs_to_the_sentence_that_contains_it() -> None:
    document = _document("A plain sentence with nothing. This one cites a source [2] mid-way.")
    result = extract(document)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert (claim.paragraph, claim.sentence) == (0, 1)
    assert claim.cited_refs == (2,)
    assert claim.text == "This one cites a source mid-way."


def test_rule_2_a_marker_opening_a_sentence_attaches_to_the_previous_one() -> None:
    document = _document("The effect was large. [4] The next sentence continues here.")
    result = extract(document)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert (claim.paragraph, claim.sentence) == (0, 0)
    assert claim.text == "The effect was large."
    assert claim.cited_refs == (4,)


def test_rule_2_a_marker_opening_the_first_sentence_stays_with_it() -> None:
    document = _document("[4] This paragraph opens with a marker. And then it continues.")
    result = extract(document)
    assert [claim.sentence for claim in result.claims] == [0]
    assert result.claims[0].text == "This paragraph opens with a marker."


def test_rule_2_a_marker_glued_after_the_full_stop_stays_in_its_own_sentence() -> None:
    document = _document("Alpha ends here.[4] Next thing here. A third one.")
    paragraph = document.paragraphs[0]
    # The sentence splitter needs whitespace after the stop, so this is one sentence.
    assert [sentence.text for sentence in paragraph.sentences] == [
        "Alpha ends here.[4] Next thing here.",
        "A third one.",
    ]
    result = extract(document)
    assert [claim.sentence for claim in result.claims] == [0]
    assert result.claims[0].text == "Alpha ends here. Next thing here."


def test_rule_3_two_markers_in_one_sentence_make_two_claims() -> None:
    document = _document(
        "This sentence cites two sources [2] and also [7] together. A second one follows."
    )
    result = extract(document)
    assert len(result.claims) == 2
    first, second = result.claims
    assert first.text == second.text == "This sentence cites two sources and also together."
    assert (first.cited_refs, second.cited_refs) == ((2,), (7,))
    assert first.sentence == second.sentence == 0
    assert first.marker is not None and first.marker.text == "[2]"
    assert second.marker is not None and second.marker.text == "[7]"
    assert not first.paragraph_scoped and not second.paragraph_scoped


def test_rule_4_a_paragraph_final_marker_scopes_every_sentence() -> None:
    document = _document(
        "First sentence of the paragraph. Second sentence of the paragraph. "
        "Third sentence of the paragraph [5]."
    )
    result = extract(document)
    assert len(result.claims) == 3
    assert all(claim.paragraph_scoped for claim in result.claims)
    assert [claim.sentence for claim in result.claims] == [0, 1, 2]
    assert [claim.text for claim in result.claims] == [
        "First sentence of the paragraph.",
        "Second sentence of the paragraph.",
        "Third sentence of the paragraph.",
    ]
    assert {claim.cited_refs for claim in result.claims} == {(5,)}
    marker = result.markers[0]
    assert {claim.group for claim in result.claims} == {f"p0:{marker.start}-{marker.end}"}
    assert {claim.marker for claim in result.claims} == {marker}


def test_rule_4_another_marker_in_the_carrying_sentence_prevents_scoping() -> None:
    document = _document("First sentence here. The second cites [2] and ends with [5].")
    result = extract(document)
    assert len(result.claims) == 2
    assert not any(claim.paragraph_scoped for claim in result.claims)
    assert {claim.sentence for claim in result.claims} == {1}
    assert [claim.cited_refs for claim in result.claims] == [(2,), (5,)]
    assert {claim.group for claim in result.claims} == {None}


def test_rule_5_a_single_sentence_paragraph_is_never_paragraph_scoped() -> None:
    document = _document("Only one sentence sits in this paragraph [5].")
    result = extract(document)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert not claim.paragraph_scoped
    assert claim.group is None
    assert claim.text == "Only one sentence sits in this paragraph."


def test_rule_6_a_number_beyond_the_bibliography_is_unresolved_and_makes_no_claim() -> None:
    document = _document("A sentence citing something that is not there [99].", references=5)
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["[99]"]
    assert result.unsupported == ()


def test_rule_6_a_partly_valid_group_keeps_its_valid_numbers_and_stays_unresolved() -> None:
    document = _document("A sentence citing one good and one bad source [2, 99].", references=5)
    result = extract(document)
    assert len(result.claims) == 1
    assert result.claims[0].cited_refs == (2,)
    assert [marker.text for marker in result.unresolved] == ["[2, 99]"]


def test_rule_6_without_a_bibliography_every_numeric_marker_is_kept() -> None:
    document = _document(
        "A pasted paragraph citing something [99] with no bibliography.", references=0
    )
    assert document.references == ()
    result = extract(document)
    assert len(result.claims) == 1
    assert result.claims[0].cited_refs == (99,)
    assert result.unresolved == ()


def test_rule_6_a_zero_is_a_misprint_and_never_resolves() -> None:
    # "[0]" names no entry in any bibliography, printed or positional, so it is
    # reported rather than turned into a claim against reference 0.
    for references in (5, 0):
        document = _document("A sentence citing a misprinted marker [0].", references=references)
        result = extract(document)
        assert result.claims == ()
        assert [marker.text for marker in result.unresolved] == ["[0]"]


def test_rule_6_a_gapped_bibliography_does_not_ghost_a_printed_number() -> None:
    # Three entries printed [1] [2] [4]: "[4]" is plainly there, so it must resolve even
    # though 4 is past the length of the list (constraint 3, never ghost a real reference).
    document = _document("A sentence citing the last entry [4].", numbers=(1, 2, 4))
    assert [reference.number for reference in document.references] == [1, 2, 4]
    result = extract(document)
    assert len(result.claims) == 1
    assert result.claims[0].cited_refs == (4,)
    assert result.unresolved == ()


def test_rule_6_a_gapped_bibliography_reports_the_number_it_never_prints() -> None:
    # The other half of the rule above: nothing in [1] [2] [4] prints a 3, so "[3]" is
    # reported rather than pointed at the entry that happens to sit third.
    document = _document("A sentence citing a number nobody printed [3].", numbers=(1, 2, 4))
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["[3]"]


def test_rule_6_a_bibliography_printed_from_the_middle_reports_a_number_below_it() -> None:
    # One page of a split reference list, printed 11, 12, 13. "[3]" names an entry that
    # is not on this page: reading it as the first entry would check the claim against
    # the wrong source in silence.
    document = _document("A sentence citing an entry from another page [3].", numbers=(11, 12, 13))
    assert [reference.number for reference in document.references] == [11, 12, 13]
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["[3]"]


def test_rule_6_an_unnumbered_first_entry_leaves_its_citation_unresolved() -> None:
    # Ingest drops text standing before the first printed number (it is heading residue
    # as often as it is an entry), so the list starts at [2]. "[1]" then names nothing
    # this document prints, and is reported rather than pointed at entry 2.
    body = "A sentence citing the entry above the first number [1]."
    entries = (
        "Author, A. An entry printed with no number. Journal, 2019.\n"
        "[2] Author, B. Title two. Journal, 2020.\n"
        "[3] Author, C. Title three. Journal, 2021.\n"
    )
    document = ingest.from_text(f"{body}\n\nReferences\n\n{entries}", name="paper.txt", kind="text")
    assert [reference.number for reference in document.references] == [2, 3]
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["[1]"]


def test_rule_7_an_author_year_marker_no_entry_answers_is_unresolved_never_guessed() -> None:
    # The bibliography here is "Author, A. Title number n. Journal, 2020." throughout, so
    # no surname matches. v0.1 reported these as an unsupported *style*; v0.2 pairs the
    # style and reports the marker instead, which is the difference between "this tool
    # cannot read that" and "this document names something its list does not hold".
    document = _document(
        "Earlier work (Smith et al., 2020) showed this. Jones et al. (2021) reported the same. "
        "A third group (Brown and Green, 2019) agreed."
    )
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == [
        "(Smith et al., 2020)",
        "Jones et al. (2021)",
        "(Brown and Green, 2019)",
    ]
    assert {marker.style for marker in result.unresolved} == {"author-year"}
    assert result.unsupported == ()


def test_rule_8_claim_text_drops_every_marker_and_keeps_the_sentence_locator() -> None:
    document = _document(
        "The first sentence cites [2] here. The second sentence\n"
        "wraps onto line two and cites [7] as well."
    )
    paragraph = document.paragraphs[0]
    result = extract(document)
    assert [claim.text for claim in result.claims] == [
        "The first sentence cites here.",
        "The second sentence wraps onto line two and cites as well.",
    ]
    assert [claim.locator for claim in result.claims] == [
        paragraph.sentences[0].locator,
        paragraph.sentences[1].locator,
    ]


def test_rule_9_claims_are_ordered_by_paragraph_sentence_and_marker_start() -> None:
    document = _document(
        "Second paragraph marker [5] sits here. And another one [6] after it.\n"
        "\n"
        "A later paragraph cites [2] and then [7]."
    )
    markers = find_markers(document)
    result = pair(document, tuple(reversed(markers)))
    keys = [
        (claim.paragraph, claim.sentence, claim.marker.start)
        for claim in result.claims
        if claim.marker is not None
    ]
    assert keys == sorted(keys)
    assert len(keys) == len(result.claims) == 4


# --- `strip_markers` and `extract` ------------------------------------------------------


def test_strip_markers_leaves_clean_text() -> None:
    document = _document(
        "Evidence is strong [2], and it holds. The effect persists [7], as reported."
    )
    paragraph = document.paragraphs[0]
    markers = find_markers(document)
    assert strip_markers(paragraph.text, markers) == (
        "Evidence is strong, and it holds. The effect persists, as reported."
    )
    # The spans are relative to `offset`, so a sentence is stripped with its own start.
    second = paragraph.sentences[1]
    assert second.start > 0
    assert (
        strip_markers(second.text, markers, offset=second.start)
        == "The effect persists, as reported."
    )


def test_strip_markers_drops_a_connective_left_between_two_markers() -> None:
    for body, expected in [
        ("The sources agree [1]-[3] on this point.", "The sources agree on this point."),
        (f"The sources agree [1]{EN_DASH}[3] on this point.", "The sources agree on this point."),
        ("The sources agree [1], [3] on this point.", "The sources agree on this point."),
        ("The sources agree [1]; [3] on this point.", "The sources agree on this point."),
        ("The sources agree [1] and [3] on this point.", "The sources agree on this point."),
        # A gap that carries words of its own is not a connective and stays.
        ("One source says [1] and another says [3].", "One source says and another says."),
    ]:
        document = _document(body)
        assert strip_markers(document.paragraphs[0].text, find_markers(document)) == expected


def test_strip_markers_closes_up_a_bracket_left_open_by_the_marker() -> None:
    document = _document("One sentence. Two sentence. A third one (see [5]).")
    markers = find_markers(document)
    assert strip_markers(document.paragraphs[0].text, markers) == (
        "One sentence. Two sentence. A third one (see)."
    )


def test_extract_is_find_markers_followed_by_pair() -> None:
    document = _document("A sentence citing a source [2]. Another sentence (Smith et al., 2020).")
    assert extract(document) == pair(document, find_markers(document))


def test_a_marker_left_alone_after_the_last_full_stop_makes_no_empty_claim() -> None:
    # "[29]" on its own is a sentence to the splitter but nothing to a reader: stripping
    # the marker leaves no text, and a claim with no text can carry no evidence.
    document = _document(
        "Rainfall has become more variable. The flood peak exceeded the record. "
        "Operating rules have not been revised since. [5]"
    )
    assert document.paragraphs[0].sentences[-1].text == "[5]"
    result = extract(document)
    assert [claim.text for claim in result.claims] == [
        "Rainfall has become more variable.",
        "The flood peak exceeded the record.",
        "Operating rules have not been revised since.",
    ]
    assert all(claim.paragraph_scoped for claim in result.claims)


def test_a_sentence_that_is_only_punctuation_and_a_marker_makes_no_claim() -> None:
    # ". [5]." and "([5])" strip to punctuation with no word in it: there is nothing for a
    # passage to support, so no claim is made for that sentence.
    document = _document("A first sentence here. A second one follows. ([5])")
    result = extract(document)
    assert [claim.text for claim in result.claims] == [
        "A first sentence here.",
        "A second one follows.",
    ]
    assert all(claim.paragraph_scoped for claim in result.claims)


def test_a_marker_alone_in_a_single_sentence_paragraph_makes_no_claim() -> None:
    document = _document("([5])")
    assert extract(document).claims == ()


# --- Author-year pairing (v0.2, spec section 9 step 2) ----------------------------------

# The entries are written the way an author-year bibliography prints them: no numbers, one
# entry per paragraph, so ingest's ordinal fallback numbers them 1..N in list order.
SMITH = "Smith, J. and Roe, B. Coastal subsidence in three deltas. Journal of Things, 2020."
JONES = "Jones, K. Memory consolidation during slow-wave sleep. Neuroscience, 2019."
RUIZ = "Ruiz, M. and Jones, K. A replication at larger scale. Neuroscience, 2021."
JONES_A = "Jones, K. An early note on consolidation. Neuroscience Letters, 2019."
BERG = "van der Berg, P. Soil carbon under no-till management. Soil Science, 2018."
MULLER = "Müller, H. Thermal tolerance of alpine plants. Alpine Botany, 2017."


def _author_year_document(body: str, entries: Sequence[str]) -> Document:
    """A document whose bibliography is an unnumbered, author-year reference list."""
    block = "\n\n".join(entries)
    return ingest.from_text(f"{body}\n\nReferences\n\n{block}\n", name="paper.txt", kind="text")


def test_author_year_a_single_author_parenthetical_pairs_with_its_entry() -> None:
    document = _author_year_document(
        "Subsidence outpaces sea level rise in several deltas (Smith et al., 2020).",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [(claim.cited_refs, claim.text) for claim in result.claims] == [
        ((1,), "Subsidence outpaces sea level rise in several deltas.")
    ]
    assert result.unresolved == ()
    assert result.unsupported == ()


def test_author_year_two_author_and_ampersand_forms_pair_on_the_first_surname() -> None:
    for marker in ("(Smith and Roe, 2020)", "(Smith & Roe, 2020)", "(Smith, Roe, 2020)"):
        document = _author_year_document(f"The deltas are sinking {marker}.", [SMITH, JONES])
        result = extract(document)
        assert [claim.cited_refs for claim in result.claims] == [(1,)], marker
        assert result.unresolved == (), marker


def test_author_year_a_semicolon_list_is_one_marker_naming_every_entry() -> None:
    document = _author_year_document(
        "Two independent groups agree (Smith et al., 2020; Jones, 2019).",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [marker.text for marker in result.markers] == ["(Smith et al., 2020; Jones, 2019)"]
    assert [(claim.cited_refs, claim.text) for claim in result.claims] == [
        ((1, 2), "Two independent groups agree.")
    ]
    assert result.unresolved == ()


def test_author_year_the_narrative_form_cites_the_sentence_it_opens() -> None:
    # The names are the subject of the sentence, so the citation belongs to that
    # sentence -- not to the one before it, which is where a bracketed marker would go.
    document = _author_year_document(
        "Sleep research has moved on. Jones (2019) reported the same effect in humans.",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [(claim.sentence, claim.cited_refs, claim.text) for claim in result.claims] == [
        (1, (2,), "reported the same effect in humans.")
    ]


def test_author_year_a_collision_is_resolved_by_the_printed_letter() -> None:
    document = _author_year_document(
        "The first report (Jones, 2019a) was extended a year later (Jones, 2019b).",
        [JONES, JONES_A],
    )
    result = extract(document)
    assert [claim.cited_refs for claim in result.claims] == [(1,), (2,)]
    assert result.unresolved == ()


def test_author_year_a_collision_without_a_letter_is_unresolved_never_guessed() -> None:
    # Two entries answer "Jones, 2019" equally well. Picking the first would point the
    # claim at a source the sentence may never have meant, in silence.
    document = _author_year_document(
        "An earlier report (Jones, 2019) said otherwise.", [JONES, JONES_A]
    )
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["(Jones, 2019)"]


def test_author_year_a_letter_past_the_end_of_the_collision_is_unresolved() -> None:
    document = _author_year_document(
        "A third note (Jones, 2019c) is cited but never listed.", [JONES, JONES_A]
    )
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["(Jones, 2019c)"]


def test_author_year_ibid_takes_the_refs_of_the_previous_marker() -> None:
    document = _author_year_document(
        "The deltas are sinking (Smith et al., 2020). The same survey reports salinity. Ibid.",
        [SMITH, JONES],
    )
    result = extract(document)
    # "Ibid." stands between two full stops, so it is a mark on the sentence before it
    # and, sitting at the end of the paragraph, scopes the whole paragraph (rule 4).
    assert [
        (claim.sentence, claim.cited_refs, claim.paragraph_scoped) for claim in result.claims
    ] == [
        (0, (1,), False),
        (0, (1,), True),
        (1, (1,), True),
    ]
    assert result.unresolved == ()


def test_author_year_ibid_reaches_back_one_paragraph_but_no_further() -> None:
    near = _author_year_document(
        "The deltas are sinking (Smith et al., 2020).\n\nThe rate has not changed (ibid.).",
        [SMITH, JONES],
    )
    assert [claim.cited_refs for claim in extract(near).claims] == [(1,), (1,)]

    far = _author_year_document(
        "The deltas are sinking (Smith et al., 2020).\n\nAn unrelated paragraph sits here.\n\n"
        "The rate has not changed (ibid.).",
        [SMITH, JONES],
    )
    result = extract(far)
    assert [claim.cited_refs for claim in result.claims] == [(1,)]
    assert [marker.text for marker in result.unresolved] == ["(ibid.)"]


def test_author_year_ibid_after_a_numeric_marker_takes_its_numbers() -> None:
    # "ibid." means "the citation just before this one", whatever style that one was
    # written in. A numbered list and a named citation in one paper is the normal case,
    # not an exotic one, and refusing the numeric marker lost the back-reference.
    body = (
        "The delta survey puts subsidence at eight millimetres (Smith, 2020). "
        "A second source gives a lower figure [2]. The same source reports salinity "
        "(ibid.). Neither figure has been revised."
    )
    document = ingest.from_text(
        f"{body}\n\nReferences\n\n[1] {SMITH}\n[2] {JONES}\n",
        name="paper.txt",
        kind="text",
    )
    result = extract(document)
    assert [(claim.sentence, claim.cited_refs) for claim in result.claims] == [
        (0, (1,)),
        (1, (2,)),
        (2, (2,)),
    ]
    assert result.unresolved == ()


def test_author_year_ibid_never_reaches_back_past_an_unresolved_marker() -> None:
    # The marker immediately before the "ibid." names nothing this list holds, so what
    # "ibid." means is unknown. Skipping over it to the Smith before would point the
    # claim at a source the sentence never meant, which is the guess product rule 3 is
    # about; the back-reference is reported instead.
    document = _author_year_document(
        "The delta survey puts subsidence at eight millimetres (Smith, 2020). A result "
        "nobody listed says otherwise (Okonkwo, 2022). The same source reports salinity "
        "(ibid.). Neither figure has been revised.",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [(claim.sentence, claim.cited_refs) for claim in result.claims] == [(0, (1,))]
    assert [marker.text for marker in result.unresolved] == ["(Okonkwo, 2022)", "(ibid.)"]


def test_author_year_op_cit_takes_the_last_entry_cited_under_that_surname() -> None:
    document = _author_year_document(
        "The deltas are sinking (Smith et al., 2020). Sleep is different (Jones, 2019). "
        "The subsidence figure is the one to use (Smith, op. cit.).",
        [SMITH, JONES],
    )
    result = extract(document)
    # The last marker ends the paragraph, so it also scopes it: what this test is about
    # is the ref it carries -- entry 1, the Smith already cited, not the Jones next to it.
    op_cit = [claim for claim in result.claims if claim.paragraph_scoped]
    assert {claim.cited_refs for claim in op_cit} == {(1,)}
    assert [claim.cited_refs for claim in result.claims if not claim.paragraph_scoped] == [
        (1,),
        (2,),
    ]
    assert result.unresolved == ()


def test_author_year_a_bare_op_cit_names_no_author_and_is_unresolved() -> None:
    document = _author_year_document(
        "The deltas are sinking (Smith et al., 2020). The figure stands (op. cit.).",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [claim.cited_refs for claim in result.claims] == [(1,)]
    assert [marker.text for marker in result.unresolved] == ["(op. cit.)"]


def test_author_year_a_page_tail_is_part_of_the_marker_and_not_a_second_source() -> None:
    document = _author_year_document(
        "The survey puts the figure at eight millimetres (Smith et al., 2020, p. 12).",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [marker.text for marker in result.markers] == ["(Smith et al., 2020, p. 12)"]
    assert [(claim.cited_refs, claim.text) for claim in result.claims] == [
        ((1,), "The survey puts the figure at eight millimetres.")
    ]


def test_author_year_a_surname_with_a_particle_matches_its_entry() -> None:
    document = _author_year_document(
        "No-till management raises soil carbon slowly (van der Berg, 2018).", [SMITH, BERG]
    )
    result = extract(document)
    assert [claim.cited_refs for claim in result.claims] == [(2,)]
    assert result.unresolved == ()


def test_author_year_diacritics_are_folded_before_the_surnames_are_compared() -> None:
    document = _author_year_document(
        "Alpine species tolerate frost to a point (Muller, 2017).", [SMITH, MULLER]
    )
    result = extract(document)
    assert [claim.cited_refs for claim in result.claims] == [(2,)]
    assert result.unresolved == ()


def test_author_year_a_name_no_entry_carries_is_unresolved_and_makes_no_claim() -> None:
    document = _author_year_document(
        "A result nobody listed (Okonkwo, 2022) is cited here.", [SMITH, JONES]
    )
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["(Okonkwo, 2022)"]
    assert result.unsupported == ()


def test_author_year_a_year_no_entry_prints_is_unresolved_even_when_the_name_matches() -> None:
    document = _author_year_document(
        "The survey is older than that (Smith et al., 1998).", [SMITH, JONES]
    )
    result = extract(document)
    assert result.claims == ()
    assert [marker.text for marker in result.unresolved] == ["(Smith et al., 1998)"]


def test_a_mixed_citation_yields_the_numeric_and_the_author_year_refs_on_one_sentence() -> None:
    # OPEN-ITEMS 9.2. Both markers sit in the same sentence, so neither scopes the
    # paragraph, and the claim text closes up over the whole parenthesis.
    document = _author_year_document(
        "A mixed style survives in some journals (see Smith, 2020; [2]) and confuses tools. "
        "The list itself is unnumbered.",
        [SMITH, JONES],
    )
    result = extract(document)
    assert [(claim.cited_refs, claim.text) for claim in result.claims] == [
        ((1,), "A mixed style survives in some journals and confuses tools."),
        ((2,), "A mixed style survives in some journals and confuses tools."),
    ]
    assert not any(claim.paragraph_scoped for claim in result.claims)
    assert result.unresolved == ()


def test_author_year_a_paragraph_final_marker_scopes_every_sentence() -> None:
    document = _author_year_document(
        "Soil carbon responds slowly to management. Most trials are too short to see it. "
        "The longest experiment is the exception (van der Berg, 2018).",
        [SMITH, BERG],
    )
    result = extract(document)
    assert all(claim.paragraph_scoped for claim in result.claims)
    assert [claim.cited_refs for claim in result.claims] == [(2,), (2,), (2,)]
