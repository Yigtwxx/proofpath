"""The ferret banner, drawn once at launch and pinned above the log (spec 13.1).

The animal is four lines of pure ASCII: no box drawing and no emoji, so Windows
Terminal at 80 columns renders it exactly as a Unix terminal does (assumption
3.3). Its body — the run of ``~`` from the head to the ``[PROOF]`` stamp — is the
path the name promises, and it re-flows on every resize.

This module colours nothing. It returns the stamp's column span and ``ui.py``
supplies the red, because the stamp is the banner's only coloured element and one
module owns every colour.
"""

from __future__ import annotations

from dataclasses import dataclass

STAMP = "[PROOF]"
#: Head art. Line 2 is replaced by ``eyes`` + body + stamp; line 3 carries the context.
HEAD = ("   ,_,", "  (o.o)", '   " "')
#: The eyes are the only animated part of the TUI. Every value is five columns wide.
EYES: dict[str, str] = {
    "idle": "(o.o)",
    "blink": "(-.-)",
    "busy": "(>.>)",
    "findings": "(O.O)",
    "clean": "(^.^)",
}
MIN_BODY = 12
#: Columns the drawn banner leaves free on the right. Two, not one: the spec's own
#: 80-column block ends line 2's stamp and line 3's context at column 78, which is
#: what right-aligns them with each other. The hint line is exempt because the
#: caller passes it whole (see :func:`render`).
RIGHT_MARGIN = 2
#: Width below which the stamp is dropped, and below which the body goes too.
STAMP_WIDTH_FLOOR = 60
BODY_WIDTH_FLOOR = 40
#: The head is indented two columns; the text lines hang under the body's start.
EYES_INDENT = "  "
TEXT_INDENT = " " * 10
#: Gap between ``" "`` and the version on line 3, measured off the spec block.
VERSION_GAP = " " * 4


@dataclass(frozen=True)
class Banner:
    """Four rendered lines plus where the stamp sits, so the caller can colour it."""

    lines: tuple[str, str, str, str]
    #: ``(start, end)`` column slice of :data:`STAMP` on ``lines[1]``; ``None`` when
    #: the terminal was too narrow and the stamp was dropped.
    stamp_span: tuple[int, int] | None


def render(
    width: int,
    *,
    version: str,
    context: str,
    hint: str,
    eyes: str = EYES["idle"],
) -> Banner:
    """Draw the banner for a terminal ``width`` columns wide.

    ``context`` is the run context (provider, online/offline, device) and is
    right-aligned on line 3; ``hint`` is line 4 in full — the caller has already
    joined the prompt hint and the slash-command list, because only it knows which
    commands this session offers. The version is never truncated: if line 3 does
    not fit, one space separates it from the context and the line overflows.
    """
    head = EYES_INDENT + eyes
    with_stamp = width >= STAMP_WIDTH_FLOOR
    if width < BODY_WIDTH_FLOOR:
        body_line, stamp_span = head, None
    elif with_stamp:
        body = max(width - len(head) - len(STAMP) - RIGHT_MARGIN, MIN_BODY)
        body_line = head + "~" * body + STAMP
        stamp_span = (len(body_line) - len(STAMP), len(body_line))
    else:
        body = max(width - len(head) - RIGHT_MARGIN, MIN_BODY)
        body_line, stamp_span = head + "~" * body, None

    version_line = _right_align(f"{HEAD[2]}{VERSION_GAP}proofpath v{version}", context, width)
    return Banner((HEAD[0], body_line, version_line, TEXT_INDENT + hint), stamp_span)


def _right_align(left: str, right: str, width: int) -> str:
    """Pad ``left`` so ``right`` ends at the banner's right edge, never truncating."""
    padding = width - RIGHT_MARGIN - len(left) - len(right)
    return left + " " * max(padding, 1) + right
