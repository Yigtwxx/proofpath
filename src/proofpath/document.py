"""The input side: a document reduced to paragraphs, sentences and references.

Spec section 9 steps 1-2. Ingest (PDF, DOCX, Markdown, plain text, posts) produces
these types and nothing else; claim extraction consumes them. Every piece of text
keeps a :class:`Locator` so a verdict can point back at the line it came from, and a
page that could not be parsed survives as a :class:`PageError` instead of vanishing
(spec section 15).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Literal, get_args


@dataclass(frozen=True)
class Locator:
    """Where a piece of text sits in the document, for reports and SARIF."""

    line: int  # 1-based: within the page for PDFs, within the file otherwise
    column: int = 1  # 1-based character offset in that line
    page: int | None = None  # PDFs only

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError(f"line must be 1-based, got {self.line}")
        if self.column < 1:
            raise ValueError(f"column must be 1-based, got {self.column}")
        if self.page is not None and self.page < 1:
            raise ValueError(f"page must be 1-based, got {self.page}")

    def label(self) -> str:
        """Human-readable position: ``"p.4 L112"`` with a page, ``"L112"`` without."""
        return f"L{self.line}" if self.page is None else f"p.{self.page} L{self.line}"


@dataclass(frozen=True)
class Sentence:
    """One sentence of a paragraph, with its char span in that paragraph's text."""

    text: str
    locator: Locator
    start: int  # inclusive
    end: int  # exclusive

    def __post_init__(self) -> None:
        _check_span(self.start, self.end)


@dataclass(frozen=True)
class Paragraph:
    """A body paragraph, flattened to one string but still mappable back to lines."""

    index: int  # 0-based position among body paragraphs
    text: str  # the paragraph as one string, lines joined by a single space
    locator: Locator  # first line of the paragraph
    sentences: tuple[Sentence, ...]
    # (line number, char offset in `text` where that line starts), ascending.
    lines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Reference:
    """One bibliography entry, kept verbatim; resolve.py does the parsing."""

    number: int  # 1-based bibliography position, what "[12]" refers to
    raw: str  # the entry verbatim, whitespace collapsed
    locator: Locator

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"reference number must be 1-based, got {self.number}")


@dataclass(frozen=True)
class PageError:
    """A page that could not be parsed. It is reported, never silently dropped."""

    page: int
    detail: str


Kind = Literal["pdf", "docx", "markdown", "text", "post"]
KIND_VALUES: tuple[str, ...] = get_args(Kind)


@dataclass(frozen=True)
class Document:
    """A parsed input document: body paragraphs, bibliography, and what failed."""

    name: str  # display name: file name, URL, or "pasted text"
    kind: Kind
    paragraphs: tuple[Paragraph, ...]  # body only; the bibliography is not in here
    references: tuple[Reference, ...]
    pages: int  # 1 for non-PDF kinds
    errors: tuple[PageError, ...] = ()

    def __post_init__(self) -> None:
        if self.pages < 1:
            raise ValueError(f"pages must be at least 1, got {self.pages}")

    def locate(self, paragraph: int, offset: int) -> Locator:
        """Map a char offset inside paragraph ``paragraph``'s text to a Locator."""
        if not 0 <= paragraph < len(self.paragraphs):
            raise IndexError(f"no paragraph {paragraph} in a document of {len(self.paragraphs)}")
        para = self.paragraphs[paragraph]
        if not 0 <= offset <= len(para.text):
            raise ValueError(f"offset {offset} is outside paragraph {paragraph}")
        # The line containing `offset` is the last one starting at or before it.
        position = bisect_right([start for _, start in para.lines], offset) - 1
        line, line_start = para.lines[position]
        return Locator(line=line, column=offset - line_start + 1, page=para.locator.page)


CitationStyle = Literal["numeric", "author-year"]


@dataclass(frozen=True)
class CitationMarker:
    """A citation as it appears in the body, and the references it points at."""

    style: CitationStyle
    text: str  # exactly as written: "[12,15]", "(Smith et al., 2020)"
    refs: tuple[int, ...]  # bibliography numbers; empty for author-year in v0.1
    paragraph: int  # Paragraph.index
    start: int  # char span within the paragraph text
    end: int

    def __post_init__(self) -> None:
        _check_span(self.start, self.end)


@dataclass(frozen=True)
class Claim:
    """A sentence that cites something, and is therefore checkable."""

    text: str  # carrying sentence, markers stripped, whitespace normalised
    locator: Locator
    cited_refs: tuple[int, ...]
    paragraph: int
    sentence: int  # index into Paragraph.sentences
    paragraph_scoped: bool = False
    # Shared by every Claim of one paragraph-scoped citation, e.g. "p3:118-122".
    group: str | None = None
    marker: CitationMarker | None = None

    def __post_init__(self) -> None:
        if self.sentence < 0:
            raise ValueError(f"sentence index must not be negative, got {self.sentence}")


def _check_span(start: int, end: int) -> None:
    if not 0 <= start <= end:
        raise ValueError(f"span must satisfy 0 <= start <= end, got ({start}, {end})")
