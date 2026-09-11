"""Citation markers and the sentences they belong to.

Spec section 9 step 2. A marker is found in the flattened paragraph text, paired with
the sentence that carries it, and turned into a :class:`Claim` that still points at a
line. Nothing is guessed on the way:

* v0.1 pairs **numeric** markers only. An author-year marker is detected and reported
  as ``UNSUPPORTED CITATION STYLE`` (it lands in ``Claims.unsupported``), never resolved
  by guessing which entry it means; author-year pairing is a v0.2 task with its own test
  set (spec section 17).
* A number that no bibliography entry answers is reported in ``Claims.unresolved``
  rather than silently dropped, so a paper citing ``[99]`` against a list of five does
  not look like a paper with no citations.
* A marker at the end of a paragraph whose sentence holds no other marker cites the
  whole paragraph, so every sentence of it becomes a claim under one ``group``. A single
  sentence is never given a confident verdict on behalf of the paragraph it sits in.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from proofpath.document import CitationMarker, Claim, Document, Paragraph

# "[12]", "[12, 15]", "[3; 7]", "[12-15]" with a hyphen, en dash (\u2013) or em dash
# (\u2014) -- both are written escaped so the pattern cannot be misread on screen. The group
# has to end at the bracket, which is what keeps "[7.4]" and "[Table 1]" out: a section
# number and a figure label are not citations, and pairing one would invent a source.
NUMERIC = re.compile(r"\[(\d{1,4}(?:\s*[,;]\s*\d{1,4}|\s*[-\u2013\u2014]\s*\d{1,4})*)\]")

# Author-year, detected only. Deliberately incomplete: a missed marker costs one
# reported "unsupported style" line, while a false one ("(Figure 2020)") would report a
# citation that is not there. The year therefore has to follow a comma or "et al." --
# a capitalised word next to a bare year is a label, not an author.
_NAME = r"[A-Z][\w'\u2019-]+"
_YEAR = r"(?:19|20)\d{2}[a-z]?"
_PARENTHETICAL = (
    r"\((?:see |e\.g\.,? |cf\. )?"
    rf"{_NAME}(?: (?:and|&) {_NAME})?"
    rf"(?:,| et al\.,?) {_YEAR}"
    r"(?:; [^()]{3,80})?\)"  # "(Smith, 2020; Jones, 2021)"
)
_NARRATIVE = rf"\b{_NAME}(?: (?:and|&) {_NAME}| et al\.)? \({_YEAR}\)"
AUTHOR_YEAR = re.compile(f"{_PARENTHETICAL}|{_NARRATIVE}")

_SEPARATOR = re.compile(r"\s*[,;]\s*")
_RANGE = re.compile(r"\s*[-\u2013\u2014]\s*")
# Whitespace left in front of punctuation once a marker between the two is gone:
# "ends it [5] ." has to close up to "ends it.", and "(see [5])" to "(see)".
_BEFORE_PUNCTUATION = re.compile(r"\s+(?=[.,;:!?)])")
# A connective left stranded between two removed markers: "[1]-[3]", "[1], [3]",
# "[1] and [3]". Matched whole, so " on this point " is never mistaken for one.
_CONNECTIVE = re.compile(r"\s*(?:[-\u2013\u2014,;]\s*)?(?:and\s*)?")
# What may follow a marker and still leave it at the end of its paragraph.
_TAIL = ".)"
# A claim has to say something: punctuation left behind by a stripped marker does not.
_WORD = re.compile(r"\w")


def expand(group: str) -> tuple[int, ...]:
    """Every bibliography number a numeric marker's group names, sorted and deduplicated.

    ``"12"`` is ``(12,)``, ``"12, 15"`` is ``(12, 15)``, ``"12-15"`` is ``(12, 13, 14,
    15)``. A range printed backwards ("15-12") is read as the range it spans: a typo in
    the body is not a reason to lose the references.
    """
    numbers: set[int] = set()
    for part in _SEPARATOR.split(group.strip()):
        ends = [int(piece) for piece in _RANGE.split(part) if piece]
        if not ends:
            continue
        numbers.update(range(min(ends), max(ends) + 1))
    return tuple(sorted(numbers))


def find_markers(doc: Document) -> list[CitationMarker]:
    """Every citation marker in the body, in reading order.

    Numeric markers are found first and win any overlap: "(see Smith, 2020; [12])" holds
    a number this version can actually resolve, and reporting it as an unsupported style
    would throw that away.
    """
    markers: list[CitationMarker] = []
    for paragraph in doc.paragraphs:
        numeric = [
            CitationMarker(
                style="numeric",
                text=match.group(0),
                refs=expand(match.group(1)),
                paragraph=paragraph.index,
                start=match.start(),
                end=match.end(),
            )
            for match in NUMERIC.finditer(paragraph.text)
        ]
        spans = [(marker.start, marker.end) for marker in numeric]
        markers.extend(numeric)
        markers.extend(
            CitationMarker(
                style="author-year",
                text=match.group(0),
                refs=(),  # v0.1 never guesses which entry an author-year names
                paragraph=paragraph.index,
                start=match.start(),
                end=match.end(),
            )
            for match in AUTHOR_YEAR.finditer(paragraph.text)
            if not any(match.start() < end and start < match.end() for start, end in spans)
        )
    markers.sort(key=lambda marker: (marker.paragraph, marker.start))
    return markers


def strip_markers(text: str, markers: Sequence[CitationMarker], *, offset: int = 0) -> str:
    """``text`` with every marker span cut out, whitespace collapsed, punctuation closed up.

    Marker spans are offsets into the paragraph text, so ``offset`` says where ``text``
    starts in it; markers falling outside ``text`` are ignored.
    """
    kept: list[str] = []
    cursor = 0
    cut = False  # has a marker already been taken out, so the next gap sits between two?
    for marker in sorted(markers, key=lambda marker: marker.start):
        start = max(marker.start - offset, 0)
        end = min(marker.end - offset, len(text))
        if end <= cursor or start >= len(text):
            continue
        piece = text[cursor:start]
        # "[1]-[3]" reads as one citation, and removing both ends would leave the dash
        # joining two halves of a sentence that were never joined.
        kept.append(" " if cut and _CONNECTIVE.fullmatch(piece) else piece)
        cursor = end
        cut = True
    kept.append(text[cursor:])
    return _BEFORE_PUNCTUATION.sub("", " ".join("".join(kept).split()))


@dataclass(frozen=True)
class Claims:
    """What a document's citations came to: the claims, and everything skipped and why.

    ``markers`` is complete: every marker found in the body is in it. The other three
    fields are **not** a partition of it. A marker can appear in more than one of them (a
    partly valid ``[2, 99]`` makes a claim *and* is reported unresolved) and a marker can
    appear in none: a numeric marker alone in a single-sentence paragraph is dropped,
    because the sentence has no word left once the marker is stripped and a claim with
    nothing in it could never carry a passage (product rule 1). Coverage therefore has to
    be counted from ``markers``; counting from the three others would report a number
    larger or smaller than the document's, and a report that misstates its own coverage
    is the failure product rule 6 exists to prevent.
    """

    claims: tuple[Claim, ...]
    markers: tuple[CitationMarker, ...]
    unsupported: tuple[CitationMarker, ...]  # author-year markers (v0.1)
    unresolved: tuple[CitationMarker, ...]  # numeric markers naming a number nothing answers


def pair(doc: Document, markers: Sequence[CitationMarker]) -> Claims:
    """Turn markers into claims: one per marker, or one per sentence when paragraph-scoped."""
    ordered = sorted(markers, key=lambda marker: (marker.paragraph, marker.start))
    by_paragraph: dict[int, list[CitationMarker]] = {}
    for marker in ordered:
        by_paragraph.setdefault(marker.paragraph, []).append(marker)

    # Place every marker, author-year ones included: rule 4 asks whether the carrying
    # sentence holds another marker, and an unsupported marker is still a marker.
    placed = [
        (marker, sentence)
        for marker in ordered
        if (sentence := _carrying_sentence(doc, marker)) is not None
    ]
    occupied = Counter((marker.paragraph, sentence) for marker, sentence in placed)

    resolvable = _resolvable(doc)
    claims: list[Claim] = []
    unresolved: list[CitationMarker] = []
    for marker, sentence in placed:
        if marker.style != "numeric":
            continue
        refs = tuple(number for number in marker.refs if _resolves(number, resolvable))
        if refs != marker.refs:
            # Partly valid still counts: the numbers that do resolve keep their claim,
            # and the marker is reported so the gap is visible in the coverage summary.
            unresolved.append(marker)
        if not refs:
            continue
        paragraph = doc.paragraphs[marker.paragraph]
        scoped = (
            len(paragraph.sentences) > 1  # one sentence cannot stand for its paragraph
            and _ends_paragraph(paragraph, marker)
            and occupied[(marker.paragraph, sentence)] == 1
        )
        group = f"p{marker.paragraph}:{marker.start}-{marker.end}" if scoped else None
        targets = range(len(paragraph.sentences)) if scoped else (sentence,)
        carried = by_paragraph[marker.paragraph]
        # A "sentence" that is nothing but the marker and its punctuation -- "[29]" left on
        # its own line after the paragraph's last full stop, or "([5])" -- has no word left
        # once the marker is stripped, and a claim with nothing in it could never carry the
        # passage a verdict needs (product rule 1). It is dropped; the sentences that do
        # have words still carry the citation.
        claims.extend(
            claim
            for index in targets
            if _WORD.search(
                (
                    claim := _claim(
                        paragraph, index, carried, marker, refs, scoped=scoped, group=group
                    )
                ).text
            )
        )
    claims.sort(key=lambda claim: (claim.paragraph, claim.sentence, _start_of(claim)))
    return Claims(
        claims=tuple(claims),
        markers=tuple(ordered),
        unsupported=tuple(marker for marker in ordered if marker.style == "author-year"),
        unresolved=tuple(unresolved),
    )


def extract(doc: Document) -> Claims:
    """Find every marker in ``doc`` and pair it. The entry point for the pipeline."""
    return pair(doc, find_markers(doc))


def _claim(
    paragraph: Paragraph,
    index: int,
    carried: Sequence[CitationMarker],
    marker: CitationMarker,
    refs: tuple[int, ...],
    *,
    scoped: bool,
    group: str | None,
) -> Claim:
    """One claim: a sentence with every marker of its paragraph taken out of the text."""
    sentence = paragraph.sentences[index]
    return Claim(
        text=strip_markers(sentence.text, carried, offset=sentence.start),
        locator=sentence.locator,
        cited_refs=refs,
        paragraph=paragraph.index,
        sentence=index,
        paragraph_scoped=scoped,
        group=group,
        marker=marker,
    )


def _carrying_sentence(doc: Document, marker: CitationMarker) -> int | None:
    """Index of the sentence the marker cites, or ``None`` if there is nothing to cite.

    The sentence containing the marker, except when the marker opens one: "The effect
    was large. [4] Next" cites the sentence before it, which is where a reader looks.
    """
    if not 0 <= marker.paragraph < len(doc.paragraphs):
        return None
    paragraph = doc.paragraphs[marker.paragraph]
    sentences = paragraph.sentences
    if not sentences:
        return None
    index = next(
        (
            position
            for position, sentence in enumerate(sentences)
            if sentence.start <= marker.start < sentence.end
        ),
        None,
    )
    if index is None:
        # The marker sits in the whitespace between two sentences (or past the last
        # one); the sentence it follows is the one that can carry it.
        index = max(
            (
                position
                for position, sentence in enumerate(sentences)
                if sentence.start <= marker.start
            ),
            default=0,
        )
    opening = not any(
        character.isalnum() for character in paragraph.text[sentences[index].start : marker.start]
    )
    return index - 1 if opening and index > 0 else index


def _ends_paragraph(paragraph: Paragraph, marker: CitationMarker) -> bool:
    """Is only whitespace, a full stop or a closing bracket left after the marker?"""
    return all(
        character.isspace() or character in _TAIL for character in paragraph.text[marker.end :]
    )


def _resolvable(doc: Document) -> frozenset[int] | None:
    """Every bibliography number a marker may name, or ``None`` when there is no list.

    A pasted paragraph carries no bibliography, and a marker in one is not a dangling
    reference -- there is simply nothing yet to check it against, which Phase 6 decides
    what to do about.

    With a list, a marker resolves only against a number that list actually **prints**.
    Product rule 3 (never call a real reference a ghost) is honoured through those
    printed numbers, not through position: an ordinal-only bibliography still resolves
    because ingest numbers its entries 1..N, and a gapped list keeps every number it
    prints, however far past its own length. What position would add is a guess --
    reading "[3]" against a list printed 11, 12, 13 points the claim at the wrong source
    in silence, which is worse than saying the number is not in this document. A number
    no entry prints is therefore reported as unresolved, never resolved by counting.
    """
    if not doc.references:
        return None
    return frozenset(reference.number for reference in doc.references)


def _resolves(number: int, resolvable: frozenset[int] | None) -> bool:
    """Does the bibliography answer this number? ``[0]`` is a misprint, never an entry."""
    if number < 1:
        return False
    return resolvable is None or number in resolvable


def _start_of(claim: Claim) -> int:
    return claim.marker.start if claim.marker is not None else 0
