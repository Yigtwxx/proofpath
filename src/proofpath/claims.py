"""Citation markers and the sentences they belong to.

Spec section 9 step 2. A marker is found in the flattened paragraph text, paired with
the sentence that carries it, and turned into a :class:`Claim` that still points at a
line. Nothing is guessed on the way:

* v0.2 pairs **numeric** markers (``[12]``, ``[12, 15]``, ``[12-15]``) and
  **author-year** ones (``(Smith et al., 2020)``, ``Jones and Ruiz (2019b)``, ``ibid.``,
  ``op. cit.``). An author-year item is paired only with a bibliography entry whose
  first author's surname *and* printed year both agree with it. Two entries agreeing
  equally well, with no ``2020a``/``2020b`` disambiguator to tell them apart, resolve to
  **neither**: an ambiguous citation reported is a gap a reader can see, a guessed one
  is a verdict pointed at the wrong source.
* A marker no bibliography entry answers -- a number nothing prints, a name and year
  nothing matches, an ``ibid.`` with nothing in front of it -- is reported in
  ``Claims.unresolved`` rather than silently dropped, so a paper citing ``[99]`` against
  a list of five does not look like a paper with no citations.
* ``Claims.unsupported`` keeps markers of a style this version still cannot pair. Both
  styles the patterns below detect are paired now, so it is empty in practice; what it
  exists for is the styles nothing detects yet -- footnote-only citations, and PDF
  superscript *letters* (``ᵃ``), which the ingest superscript rule leaves alone on
  purpose. Those produce no marker at all today, so they cannot be reported here either;
  that gap is recorded in OPEN-ITEMS rather than hidden behind an empty tuple.
* A marker at the end of a paragraph whose sentence holds no other marker cites the
  whole paragraph, so every sentence of it becomes a claim under one ``group``. A single
  sentence is never given a confident verdict on behalf of the paragraph it sits in.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from proofpath import resolve
from proofpath.document import LINK_CITED, CitationMarker, Claim, Document, Paragraph

# "[12]", "[12, 15]", "[3; 7]", "[12-15]" with a hyphen, en dash (\u2013) or em dash
# (\u2014) -- both are written escaped so the pattern cannot be misread on screen. The group
# has to end at the bracket, which is what keeps "[7.4]" and "[Table 1]" out: a section
# number and a figure label are not citations, and pairing one would invent a source.
_NUMBERS = r"\d{1,4}(?:\s*[,;]\s*\d{1,4}|\s*[-\u2013\u2014]\s*\d{1,4})*"
NUMERIC = re.compile(rf"\[({_NUMBERS})\]")

# --- author-year ---------------------------------------------------------------------
#
# Deliberately incomplete: a missed marker costs one reported line, while a false one
# ("(Figure 2020)") would report a citation that is not there. Inside a parenthesis the
# year therefore has to follow a comma or "et al." -- a capitalised word next to a bare
# year is a label, not an author -- and outside one it has to sit in brackets of its own
# ("Smith et al. (2020)"). A bare "Smith 2020", with neither, is not read as a citation.
_SURNAME = resolve.SURNAME  # "Smith", "van der Berg", "al-Khalili"; shared with resolve
_YEAR = r"(?:19|20)\d{2}"
_LETTER = r"[a-z]?"  # the "a" of "2020a": one author, two entries in the same year
_ET_AL = r"(?:\set\sal\.?)"
# "Smith, Roe and Tan", "Smith, Roe, and Tan" (the Oxford comma is common enough that
# missing it loses the whole marker, not just its third author).
_NAMES = rf"{_SURNAME}(?:,\s(?:(?:and|&)\s)?{_SURNAME})*(?:\s(?:and|&)\s{_SURNAME})?"
# "(Smith, 2020, p. 12)", "(Smith, 2020, pp. 12-15)", "(Smith, 2020, ch. 4)": the tail
# names a place inside the source, not another source, so it belongs to the marker.
_LOCATOR_TAIL = r"(?:,\s(?:pp?\.|ch(?:ap)?\.|sec\.|\u00a7)\s?\w+(?:[-\u2013]\w+)?)?"
_PAREN_ITEM = rf"{_NAMES}(?:{_ET_AL},?|,)\s{_YEAR}{_LETTER}{_LOCATOR_TAIL}"
_NARRATIVE_ITEM = rf"{_NAMES}{_ET_AL}?\s\({_YEAR}{_LETTER}{_LOCATOR_TAIL}\)"
# Back-references. "ibid." names whatever the previous marker named; "op. cit." names an
# earlier source by its author. Both are citations in their own right -- the sentence
# carrying one cites something -- so both are markers, and an unmatched one is reported.
_IBID = r"[Ii]bid(?:em)?\b\.?"
_OP_CIT = rf"(?:{_SURNAME},?\s)?op\.\s?cit\.?"
# What a parenthetical may open with before its first item.
_LEAD_IN = r"(?:[Ss]ee\s(?:also\s)?|e\.g\.,?\s|cf\.\s|also\s)?"
_MEMBER = rf"{_LEAD_IN}(?:{_PAREN_ITEM}|{_IBID}|{_OP_CIT}|\[{_NUMBERS}\])"
# One parenthetical is one marker, however many items it lists. "(Smith, 2020; Jones,
# 2021)" is a single citation to a reader, its text is what a report prints back, and
# cutting it into pieces would leave half a bracket in the claim text. The items are
# split out again by `_items` when the entries behind them are looked up.
_PARENTHETICAL = rf"\({_MEMBER}(?:\s*;\s*{_MEMBER})*\)"
AUTHOR_YEAR = re.compile(rf"{_PARENTHETICAL}|{_NARRATIVE_ITEM}|\b(?:{_OP_CIT}|{_IBID})")

# The same items, matched one at a time inside a marker's own text.
_ITEM = re.compile(
    rf"(?P<narrative>{_NARRATIVE_ITEM})|(?P<paren>{_PAREN_ITEM})"
    rf"|(?P<ibid>{_IBID})|(?P<opcit>{_OP_CIT})"
)
_ITEM_YEAR = re.compile(rf"({_YEAR})({_LETTER})")
_SURNAME_RUN = re.compile(_SURNAME)
# A lead-in opening a marker's text: "See" is capitalised like a surname, and reading it
# as one would look up an author nobody cited.
_OPENING_LEAD_IN = re.compile(r"^[\s(]*(?:[Ss]ee\s(?:also\s)?|e\.g\.,?\s|cf\.\s|also\s)?")

# Styles this version can pair. A marker of any other style is reported as unsupported
# rather than paired; a style this set grows to cover stops being reported that way.
PAIRED_STYLES = frozenset({"numeric", "author-year"})

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
_POSSESSIVE = re.compile(r"'s$")


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


@dataclass(frozen=True)
class Item:
    """One source named inside an author-year marker.

    ``(Smith, 2020; Jones, 2021)`` holds two; ``(ibid.)`` holds one that names no author
    at all. ``surnames`` is what the body printed, not what the bibliography prints --
    they are compared folded, in :func:`_entries_for`.
    """

    kind: str  # "author-year", "ibid" or "op. cit."
    surnames: tuple[str, ...]
    year: int | None
    letter: str  # the disambiguator of "2020a", or ""
    text: str


def items(text: str) -> tuple[Item, ...]:
    """The sources an author-year marker's text names, in the order it prints them.

    A numeric item inside the same parenthesis ("(see Smith, 2020; [12])") is left out:
    it is a marker of its own, found and paired by the numeric half.
    """
    found: list[Item] = []
    for match in _ITEM.finditer(text):
        item = _item(match)
        if item is not None:
            found.append(item)
    return tuple(found)


def _item(match: re.Match[str]) -> Item | None:
    text = match.group(0)
    if match.group("ibid") is not None:
        return Item(kind="ibid", surnames=(), year=None, letter="", text=text)
    if match.group("opcit") is not None:
        return Item(kind="op. cit.", surnames=_surnames(text), year=None, letter="", text=text)
    year = _ITEM_YEAR.search(text)
    if year is None:  # unreachable: both remaining alternatives carry a year
        return None  # pragma: no cover
    return Item(
        kind="author-year",
        surnames=_surnames(text[: year.start()]),
        year=int(year.group(1)),
        letter=year.group(2),
        text=text,
    )


def _surnames(head: str) -> tuple[str, ...]:
    """Every surname printed before the year, the opening lead-in dropped first."""
    return tuple(match.group(0) for match in _SURNAME_RUN.finditer(_OPENING_LEAD_IN.sub("", head)))


def find_markers(doc: Document) -> list[CitationMarker]:
    """Every citation marker in the body, in reading order.

    Numeric and author-year markers are found independently and an overlap keeps both:
    "(see Smith, 2020; [12])" cites two sources in one parenthesis, and letting the
    number win it -- which is what v0.1 did -- lost the author-year half entirely, so
    the mixed style was reported nowhere (OPEN-ITEMS 9.2). ``strip_markers`` drops a
    marker already contained in one it has cut, so the claim text still closes up.
    """
    markers: list[CitationMarker] = []
    for paragraph in doc.paragraphs:
        markers.extend(
            CitationMarker(
                style="numeric",
                text=match.group(0),
                refs=expand(match.group(1)),
                paragraph=paragraph.index,
                start=match.start(),
                end=match.end(),
            )
            for match in NUMERIC.finditer(paragraph.text)
        )
        markers.extend(
            CitationMarker(
                style="author-year",
                # Which entries it names is decided against the bibliography, in
                # `pair_author_year`; `find_markers` only says where the citation is.
                refs=(),
                text=match.group(0),
                paragraph=paragraph.index,
                start=match.start(),
                end=match.end(),
            )
            for match in AUTHOR_YEAR.finditer(paragraph.text)
            # A parenthesis holding nothing but numbers ("(see [20])") matches the group
            # pattern and names no author: it is the numeric marker's, not a second one.
            if items(match.group(0))
        )
    markers.sort(key=lambda marker: (marker.paragraph, marker.start))
    return markers


def strip_markers(text: str, markers: Sequence[CitationMarker], *, offset: int = 0) -> str:
    """``text`` with every marker span cut out, whitespace collapsed, punctuation closed up.

    Marker spans are offsets into the paragraph text, so ``offset`` says where ``text``
    starts in it; markers falling outside ``text`` are ignored, and one lying inside a
    span already cut (the ``[12]`` of "(see Smith, 2020; [12])") is already gone.
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
    unsupported: tuple[CitationMarker, ...]  # styles v0.2 still cannot pair (see module docs)
    unresolved: tuple[CitationMarker, ...]  # markers naming something no entry answers


def pair(doc: Document, markers: Sequence[CitationMarker]) -> Claims:
    """Turn markers into claims: one per marker, or one per sentence when paragraph-scoped."""
    ordered = sorted(markers, key=lambda marker: (marker.paragraph, marker.start))
    numeric_claims, numeric_unresolved = pair_numeric(doc, ordered)
    author_claims, author_unresolved = pair_author_year(doc, ordered)
    claims = sorted(
        numeric_claims + author_claims,
        key=lambda claim: (claim.paragraph, claim.sentence, _start_of(claim)),
    )
    unresolved = sorted(
        numeric_unresolved + author_unresolved,
        key=lambda marker: (marker.paragraph, marker.start),
    )
    return Claims(
        claims=tuple(claims),
        markers=tuple(ordered),
        unsupported=tuple(marker for marker in ordered if marker.style not in PAIRED_STYLES),
        unresolved=tuple(unresolved),
    )


def extract(doc: Document) -> Claims:
    """Find every marker in ``doc`` and pair it. The entry point for the pipeline.

    :func:`pair_links` is taken only for a document whose ``kind`` says it cites by
    linking -- ``post`` and ``linked``, which ``ingest.from_post`` and
    ``ingest.with_link_references`` are the only two builders to produce (spec
    section 6.2). The declared kind rather than the shape of the document: a paper
    that prints a bibliography and carries no marker this version detects
    (superscript letters, footnote-only styles, anything ``headless_fallback``
    recovers) is not a paper that cites by linking, and pairing its sentences against
    its entries would give them a SUPPORTED or REFUTED verdict on a source they never
    cited (product rule 1).

    The declared kind decides it *whatever the body prints*, markers included. A
    document that cites by linking prints no numbered bibliography, so a bracketed
    number in its prose -- "[2024]", a redacted "[1]", a chess move -- names no entry
    and is not a citation. Letting one switch the document onto the numbered branch
    lost every link pairing it had, which left a post that cites its sources reading
    as a post that cites nothing (product rule 6). Nothing is reported unresolved for
    such a number either: there is no list for it to have failed to name.
    """
    if doc.kind in LINK_CITED:
        return pair_links(doc)
    return pair(doc, find_markers(doc))


def pair_links(doc: Document) -> Claims:
    """Claims for a document whose references are the links inside its paragraphs.

    Every sentence of a paragraph becomes a claim citing *all* of that paragraph's
    links, under one ``group``: the post as a whole is what its links are supposed to
    back, so no sentence of it is given a confident verdict on its own -- the rule
    ``_claims_for`` applies to a marker that ends a paragraph, applied here to a
    paragraph that ends in nothing else.

    A reference is placed by its locator's line, not by searching the paragraph for
    its address: a platform shortens the link it displays (Bluesky) or keeps it out
    of the body altogether (a Hacker News submission), and a source found by string
    match would then be attached to the wrong paragraph or to none.

    The page is part of that key. A PDF locator counts lines *within* its page, so
    line 3 of page 9 and line 3 of page 1 are two different places and a key made of
    the line alone would collide them -- pointing a claim at whatever happened to
    share its line number elsewhere in the file (product rule 1).
    """
    paragraph_of = {
        (paragraph.locator.page, line): paragraph.index
        for paragraph in doc.paragraphs
        for line, _ in paragraph.lines
    }
    cited: dict[int, list[int]] = {}
    for reference in doc.references:
        index = paragraph_of.get((reference.locator.page, reference.locator.line))
        if index is not None:
            cited.setdefault(index, []).append(reference.number)
    markers: list[CitationMarker] = []
    claims: list[Claim] = []
    for index, numbers in sorted(cited.items()):
        paragraph = doc.paragraphs[index]
        refs = tuple(sorted(set(numbers)))
        # The citation is the act of linking, and it stands at the end of what it
        # backs. An empty span: there is no marker printed in the text to point at,
        # and inventing one would put characters in the document that nobody wrote.
        marker = CitationMarker(
            style="numeric",
            text=" ".join(doc.references[number - 1].raw for number in refs),
            refs=refs,
            paragraph=index,
            start=len(paragraph.text),
            end=len(paragraph.text),
        )
        markers.append(marker)
        # One sentence *is* its paragraph, so a citation over it is not wider than
        # the sentence and the group would say something the run cannot support.
        scoped = len(paragraph.sentences) > 1
        group = f"p{index}:links" if scoped else None
        claims.extend(
            Claim(
                text=text,
                locator=sentence.locator,
                cited_refs=refs,
                paragraph=index,
                sentence=position,
                paragraph_scoped=scoped,
                group=group,
                marker=marker,
            )
            for position, sentence in enumerate(paragraph.sentences)
            if _WORD.search(text := strip_links(sentence.text))
        )
    return Claims(claims=tuple(claims), markers=tuple(markers), unsupported=(), unresolved=())


def strip_links(text: str) -> str:
    """A sentence with the addresses in it taken out, whitespace normalised.

    The counterpart of :func:`strip_markers` for a document that cites by linking:
    the address is the citation, not the assertion, and leaving it in the claim text
    would have the model score a URL as part of what the author said.
    """
    # By offset, back to front, so each cut is the address ``find_urls`` recorded.
    # ``replace`` would take the first match instead, which for a pair like
    # ``https://a.test/x/y`` and ``https://a.test/x`` is the wrong one -- it cuts
    # into the longer address and leaves the shorter one standing in the claim.
    for url, offset in reversed(resolve.find_urls(text)):
        text = f"{text[:offset]} {text[offset + len(url) :]}"
    return _BEFORE_PUNCTUATION.sub("", " ".join(text.split()))


def pair_numeric(
    doc: Document, markers: Sequence[CitationMarker]
) -> tuple[list[Claim], list[CitationMarker]]:
    """Claims for the numeric markers of ``markers``, and the ones nothing answers."""
    placed, occupied, by_paragraph = _placed(doc, markers)
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
        claims.extend(_claims_for(doc, marker, sentence, refs, occupied, by_paragraph))
    return claims, unresolved


def pair_author_year(
    doc: Document, markers: Sequence[CitationMarker]
) -> tuple[list[Claim], list[CitationMarker]]:
    """Claims for the author-year markers of ``markers``, and the ones nothing answers.

    Each item of a marker is looked up on its own: the first surname it prints has to
    equal the surname ``resolve.author_hint`` reads out of an entry (folded to ASCII and
    reduced to the last word, so ``Müller`` meets ``Muller`` and ``van der Berg`` meets a
    hint of ``Berg``), and the year has to be one the entry actually prints. Exactly one
    such entry resolves the item. Several of them resolve it only when the body printed
    the ``2020a``/``2020b`` disambiguator that says which; without one, the item is
    ambiguous and the whole marker is reported rather than pointed at a coin flip.

    ``ibid.`` takes the refs of the marker **immediately before it in reading order**,
    whatever style that one was written in: a paper that numbers half its list and names
    the other half is the normal case, and a numeric marker is as much "the previous
    citation" as an author-year one. It never reaches back *past* that marker. If the
    marker before it resolved to nothing -- or there is none, or it stands more than one
    paragraph back, where "the previous citation" is a guess about a page the reader has
    left -- the ``ibid.`` is reported rather than pointed at the citation before that.

    ``op. cit.`` is the opposite kind of back-reference: it names an author, so it takes
    the most recent entry already cited under that surname, however far back. A bare
    ``op. cit.`` names no author and so resolves to nothing.
    """
    placed, occupied, by_paragraph = _placed(doc, markers)
    index = _entries(doc)
    resolvable = _resolvable(doc)
    claims: list[Claim] = []
    unresolved: list[CitationMarker] = []
    cited: list[tuple[CitationMarker, tuple[tuple[str, int], ...]]] = []
    # The marker immediately before the one being read, and what it came to. A numeric
    # marker's refs are recomputed here rather than carried over from `pair_numeric`:
    # the two halves are independent entry points, and an `ibid.` has to see the same
    # numbers either of them would produce.
    previous: tuple[CitationMarker, tuple[int, ...]] | None = None
    for marker, sentence in placed:
        if marker.style != "author-year":
            if marker.style == "numeric":
                previous = (marker, tuple(n for n in marker.refs if _resolves(n, resolvable)))
            continue
        refs, named, complete = _refs_for(marker, index, cited, previous)
        if not complete:
            unresolved.append(marker)
        if named:
            cited.append((marker, named))
        previous = (marker, refs)
        claims.extend(_claims_for(doc, marker, sentence, refs, occupied, by_paragraph))
    return claims, unresolved


def _refs_for(
    marker: CitationMarker,
    index: Sequence[tuple[str, frozenset[int], int]],
    cited: Sequence[tuple[CitationMarker, tuple[tuple[str, int], ...]]],
    previous: tuple[CitationMarker, tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...], bool]:
    """``(refs, (surname, ref) pairs for later back-references, every item resolved?)``."""
    refs: list[int] = []
    named: list[tuple[str, int]] = []
    complete = True
    for item in items(marker.text):
        if item.kind == "ibid":
            found = _previous_refs(marker, previous)
        elif item.kind == "op. cit.":
            found = _earlier_by_author(item, cited)
        else:
            found = _entries_for(item, index)
            named.extend((_fold(item.surnames[0]), number) for number in found if item.surnames)
        if found:
            refs.extend(found)
        else:
            complete = False
    return tuple(sorted(set(refs))), tuple(named), complete


def _entries_for(item: Item, index: Sequence[tuple[str, frozenset[int], int]]) -> tuple[int, ...]:
    """The bibliography entry an author-year item names, or nothing when it is unclear."""
    surname = _fold(item.surnames[0]) if item.surnames else ""
    if not surname or item.year is None:
        return ()
    matches = [number for hint, years, number in index if hint == surname and item.year in years]
    if item.letter:
        # "2020a" is the first of the entries this author published that year, "2020b"
        # the second: the letters are assigned in bibliography order, which is the order
        # `index` keeps. A letter past the end names an entry the list does not print.
        position = ord(item.letter) - ord("a")
        return (matches[position],) if 0 <= position < len(matches) else ()
    return (matches[0],) if len(matches) == 1 else ()


def _previous_refs(
    marker: CitationMarker, previous: tuple[CitationMarker, tuple[int, ...]] | None
) -> tuple[int, ...]:
    """What ``ibid.`` points back at: the marker just before it, at most a paragraph back.

    Empty when that marker resolved to nothing. The caller then reports the ``ibid.``,
    which is the whole point of taking only the immediately preceding marker: reaching
    past an unresolved one to the last citation that happened to work would answer "which
    source?" with whichever the reader was not looking at.
    """
    if previous is None:
        return ()
    before, refs = previous
    return () if marker.paragraph - before.paragraph > 1 else refs


def _earlier_by_author(
    item: Item, cited: Sequence[tuple[CitationMarker, tuple[tuple[str, int], ...]]]
) -> tuple[int, ...]:
    """What "Smith, op. cit." points back at: the last entry cited under that surname."""
    if not item.surnames:
        return ()
    surname = _fold(item.surnames[-1])
    for _, named in reversed(cited):
        for hint, number in reversed(named):
            if hint == surname:
                return (number,)
    return ()


def _entries(doc: Document) -> tuple[tuple[str, frozenset[int], int], ...]:
    """``(folded first-author surname, printed years, entry number)`` in list order."""
    return tuple(
        (
            _fold(resolve.author_hint(reference.raw)),
            frozenset(resolve.years(reference.raw)),
            reference.number,
        )
        for reference in doc.references
    )


def _fold(name: str) -> str:
    """A surname reduced to what two spellings of it have in common.

    Diacritics go (a paper citing ``Muller`` means the ``Müller`` in its list), case
    goes, and only the last word is kept, so the body's ``van der Berg`` meets the
    ``Berg`` that ``resolve.author_hint`` reads out of "van der Berg, P.". Two different
    families sharing a last word ("Berg" and "van der Berg") therefore collide, which
    makes the citation ambiguous rather than wrong: it is reported, never guessed.
    """
    plain = unicodedata.normalize("NFKD", name.replace("\u2019", "'"))
    ascii_name = plain.encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[A-Za-z][\w'-]*", ascii_name.lower())
    # "Smith's (2020) survey" is a citation to Smith: the possessive belongs to the
    # sentence, not to the name, and keeping it would leave the entry unmatched.
    return _POSSESSIVE.sub("", words[-1]) if words else ""


def _placed(
    doc: Document, markers: Sequence[CitationMarker]
) -> tuple[
    list[tuple[CitationMarker, int]],
    Counter[tuple[int, int]],
    dict[int, list[CitationMarker]],
]:
    """Every marker on the sentence that carries it, with what shares that sentence.

    Both halves of the pairing need all three, and over *every* marker: rule 4 asks
    whether the carrying sentence holds another marker, and a marker of the other style
    is still another marker.
    """
    ordered = sorted(markers, key=lambda marker: (marker.paragraph, marker.start))
    by_paragraph: dict[int, list[CitationMarker]] = {}
    for marker in ordered:
        by_paragraph.setdefault(marker.paragraph, []).append(marker)
    placed = [
        (marker, sentence)
        for marker in ordered
        if (sentence := _carrying_sentence(doc, marker)) is not None
    ]
    occupied = Counter((marker.paragraph, sentence) for marker, sentence in placed)
    return placed, occupied, by_paragraph


def _claims_for(
    doc: Document,
    marker: CitationMarker,
    sentence: int,
    refs: tuple[int, ...],
    occupied: Counter[tuple[int, int]],
    by_paragraph: dict[int, list[CitationMarker]],
) -> list[Claim]:
    """The claims one marker makes: its own sentence, or every sentence when scoped."""
    if not refs:
        return []
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
    return [
        claim
        for index in targets
        if _WORD.search(
            (
                claim := _claim(paragraph, index, carried, marker, refs, scoped=scoped, group=group)
            ).text
        )
    ]


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
    return index - 1 if opening and index > 0 and not _narrative(paragraph, marker) else index


def _narrative(paragraph: Paragraph, marker: CitationMarker) -> bool:
    """Is the marker the subject of its own sentence rather than a mark appended to one?

    "Jones and Ruiz (2019b) agreed." opens with the authors' names: they are what the
    sentence is *about*, so the citation belongs to the sentence it opens, not to the one
    before it. "[4]" and "(Smith, 2020)" carry no subject -- a reader meeting either at
    the head of a sentence reads it as the mark on the sentence just finished -- and an
    "Ibid." standing alone between two full stops has no sentence of its own to be the
    subject of, so both keep the rule above.
    """
    if not marker.text[:1].isalpha():
        return False
    return any(
        character.isalnum() for character in paragraph.text[marker.end : _end_of(paragraph, marker)]
    )


def _end_of(paragraph: Paragraph, marker: CitationMarker) -> int:
    """Where the sentence carrying ``marker`` ends, or the end of the paragraph."""
    return next(
        (
            sentence.end
            for sentence in paragraph.sentences
            if sentence.start <= marker.start < sentence.end
        ),
        len(paragraph.text),
    )


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
