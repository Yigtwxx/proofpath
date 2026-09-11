"""The input-side value types: locators stay addressable, offsets stay honest."""

from __future__ import annotations

import pytest

from proofpath.document import (
    KIND_VALUES,
    CitationMarker,
    Claim,
    Document,
    Locator,
    PageError,
    Paragraph,
    Reference,
    Sentence,
)

# Three lines starting at char offsets 0, 40 and 81 of a 120-character paragraph.
LINE_OFFSETS = ((10, 0), (11, 40), (12, 81))
PARAGRAPH_TEXT = "x" * 120


def _paragraph(page: int | None = 4) -> Paragraph:
    return Paragraph(
        index=0,
        text=PARAGRAPH_TEXT,
        locator=Locator(line=10, column=1, page=page),
        sentences=(),
        lines=LINE_OFFSETS,
    )


def _document(page: int | None = 4) -> Document:
    return Document(
        name="draft.pdf",
        kind="pdf",
        paragraphs=(_paragraph(page),),
        references=(),
        pages=6,
    )


def test_label_without_a_page_is_just_the_line() -> None:
    assert Locator(line=112).label() == "L112"


def test_label_with_a_page_names_the_page() -> None:
    assert Locator(line=112, column=7, page=4).label() == "p.4 L112"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"line": 0}, "line"),
        ({"line": -3}, "line"),
        ({"line": 1, "column": 0}, "column"),
        ({"line": 1, "page": 0}, "page"),
    ],
)
def test_locator_rejects_non_positive_positions(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Locator(**kwargs)


def test_locator_page_may_be_absent() -> None:
    assert Locator(line=1).page is None


@pytest.mark.parametrize(
    ("offset", "line", "column"),
    [
        (0, 10, 1),
        (39, 10, 40),
        (40, 11, 1),
        (80, 11, 41),
        (81, 12, 1),
        (len(PARAGRAPH_TEXT), 12, 40),
    ],
)
def test_locate_maps_an_offset_to_its_line_and_column(offset: int, line: int, column: int) -> None:
    located = _document().locate(0, offset)
    assert (located.line, located.column) == (line, column)


def test_locate_carries_the_page_through() -> None:
    assert _document(page=4).locate(0, 90).page == 4


def test_locate_leaves_the_page_unset_for_non_pdf_documents() -> None:
    assert _document(page=None).locate(0, 90).page is None


@pytest.mark.parametrize("paragraph", [1, -1, 7])
def test_locate_rejects_an_unknown_paragraph(paragraph: int) -> None:
    with pytest.raises(IndexError):
        _document().locate(paragraph, 0)


@pytest.mark.parametrize("offset", [-1, len(PARAGRAPH_TEXT) + 1])
def test_locate_rejects_an_offset_outside_the_paragraph(offset: int) -> None:
    with pytest.raises(ValueError, match="offset"):
        _document().locate(0, offset)


def test_sentence_span_must_not_run_backwards() -> None:
    with pytest.raises(ValueError, match="start"):
        Sentence(text="hi", locator=Locator(line=1), start=9, end=4)


def test_sentence_span_must_not_start_before_the_paragraph() -> None:
    with pytest.raises(ValueError, match="start"):
        Sentence(text="hi", locator=Locator(line=1), start=-1, end=4)


def test_citation_marker_span_must_not_run_backwards() -> None:
    with pytest.raises(ValueError, match="start"):
        CitationMarker(style="numeric", text="[12]", refs=(12,), paragraph=0, start=9, end=4)


def test_reference_number_is_one_based() -> None:
    with pytest.raises(ValueError, match="number"):
        Reference(number=0, raw="Smith et al. 2020", locator=Locator(line=3))


def test_document_has_at_least_one_page() -> None:
    with pytest.raises(ValueError, match="pages"):
        Document(name="pasted text", kind="text", paragraphs=(), references=(), pages=0)


def test_claim_sentence_index_is_not_negative() -> None:
    with pytest.raises(ValueError, match="sentence"):
        Claim(
            text="the effect held",
            locator=Locator(line=1),
            cited_refs=(1,),
            paragraph=0,
            sentence=-1,
        )


def test_document_defaults_to_no_page_errors() -> None:
    assert _document().errors == ()


def test_page_error_records_the_page_it_failed_on() -> None:
    error = PageError(page=3, detail="no extractable text layer")
    assert (error.page, error.detail) == (3, "no extractable text layer")


def test_kind_values_are_the_five_supported_inputs() -> None:
    assert KIND_VALUES == ("pdf", "docx", "markdown", "text", "post")


def test_every_value_type_is_hashable() -> None:
    locator = Locator(line=10, column=1, page=4)
    sentence = Sentence(text="the effect held.", locator=locator, start=0, end=16)
    marker = CitationMarker(
        style="numeric", text="[12,15]", refs=(12, 15), paragraph=0, start=16, end=23
    )
    values = {
        locator,
        sentence,
        _paragraph(),
        Reference(number=12, raw="Smith et al. 2020", locator=Locator(line=300)),
        PageError(page=3, detail="no extractable text layer"),
        _document(),
        marker,
        Claim(
            text="the effect held.",
            locator=locator,
            cited_refs=(12, 15),
            paragraph=0,
            sentence=0,
            group="p3:118-122",
            marker=marker,
        ),
    }
    assert len(values) == 8
