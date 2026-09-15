"""The two ferrets and what moves on them (TUI v2 design section 3, spec 13.1).

``RICH`` draws a seven-line ferret out of box-drawing characters: head at the
left, a body that stretches with the width, four feet on a line of their own, and
a tail that leaves the rump, wags, and ends in the ``[PROOF]`` stamp. ``PLAIN``
is the four-line ASCII animal :mod:`proofpath.tui.banner` has always drawn, and this
module delegates to it unchanged, so the golden block of spec 13.1 still holds
there.

Everything here is pure: :func:`render` returns strings and column spans, and the
animation is a state machine the app drives with a clock of its own. No Textual;
no colour either -- the stamp's span is returned and ``ui.py`` supplies the red.

The art uses only single-width characters (box drawing, ``~``, ``˘``, ASCII), so a
line's column count is its length; the tests check every glyph against
``unicodedata.east_asian_width``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from proofpath.tui import banner

STAMP = banner.STAMP

#: The RICH ferret, drawn here once so the shape can be read without running it.
#: Line 2 is the back: ``BROW`` + ``─`` run + ``╮``. Line 3 is the face, the flank,
#: the rump's mid-line ``╰`` where the tail leaves, the tail, and the stamp. Line 4
#: is the belly with four leg joints, line 5 the four feet under those joints. The
#: three rump corners (``╮ ╰ ╯``) share one column.
#:
#:     ╭╮ ╭╮
#:    ╭╯╰─╯╰──────────────────────────────╮
#:   ╸┤ o o                               ╰~~~~~~~~[PROOF]
#:    ╰─┬─┬──────────────────────────┬─┬──╯
#:      ˘ ˘                           ˘ ˘
#:       proofpath v0.3.0     academic · online · coreml
#:       paste a file path, a URL, or a claim.   /help  /config  /quit
EARS = "    ╭╮ ╭╮"
BROW = "   ╭╯╰─╯╰"
FACE = "  ╸┤ o o"
CHIN = "   ╰─┬─┬"
FEET = "     ˘ ˘"
#: The rear of the belly line: two leg joints, then the corner under the tail.
RUMP = "┬─┬──╯"
REAR_FEET = "˘ ˘"
#: The tail is ``~`` up to its tip; the tip is what wags. ``tail_offset`` modulo
#: three picks the frame: level, lifted, dropped.
TAIL_TIPS = ("~", "‾", "_")
#: The wag order: lift, level, drop, level -- the tip never jumps from lifted to dropped.
WAG_CYCLE = (1, 0, 2, 0)
#: The tail is never shorter than this with the stamp, grows with the width, and
#: shortens to :data:`TAIL_SHORT` when the stamp is dropped.
TAIL_MIN = 8
TAIL_SHORT = 4
#: One sixth of the room behind the head goes to the tail: at 80 columns ten
#: cells, at 200 thirty, never more than a third of the body.
TAIL_SHARE = 6
#: Where the eyes sit on ``FACE``: three cells, every value of :data:`EYES` is three wide.
EYES_COLUMN = FACE.index("o o")

#: RICH eyes, keyed as ``banner.EYES`` is so one animation drives both themes.
EYES: dict[str, str] = {
    "idle": "o o",
    "blink": "- -",
    "busy": "> >",
    "findings": "O O",
    "clean": "^ ^",
}

#: The back never shrinks below this many cells; the same floor as the PLAIN body.
MIN_BODY = banner.MIN_BODY
#: Below this the stamp is dropped; below the next the PLAIN pet is drawn instead.
STAMP_WIDTH_FLOOR = banner.STAMP_WIDTH_FLOOR
RICH_WIDTH_FLOOR = 48
#: The two text lines hang under the face, six in, as in the design's block.
TEXT_INDENT = " " * 6
#: The hint is passed whole (the caller joined the prompt hint and the commands);
#: RICH splits it at the first run of two or more spaces and right-aligns the rest.
HINT_GAP = re.compile(r" {2,}")

#: The animation in seconds: a blink's length, the window the next one falls in,
#: how long a finished run's expression holds (spec 13.1), and how long the tail
#: holds each wag frame while a run is busy (about 2 Hz).
BLINK_SECONDS = 0.15
BLINK_WINDOW = (6.0, 10.0)
FLASH_SECONDS = 2.0
WAG_SECONDS = 0.5

Mood = Literal["findings", "clean"]


class ThemeLike(Protocol):
    """The one field of ``theme.Theme`` the pet reads.

    Structural on purpose: ``theme.py`` is written alongside this module and the
    pet needs none of its glyphs or tones, only which ferret to draw.
    """

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class Pet:
    """The rendered lines plus where the animated and coloured parts sit."""

    #: Seven lines in RICH, four in PLAIN (``banner.Banner.lines``).
    lines: tuple[str, ...]
    #: ``(line, start, end)`` of :data:`STAMP`; ``None`` when the width dropped it.
    stamp_span: tuple[int, int, int] | None
    #: ``(line, start, end)`` of the eyes, for the app to swap expressions.
    eyes_span: tuple[int, int, int]
    #: ``(line, start, end)`` of the tail, whose last cell wags; RICH only.
    tail_span: tuple[int, int, int] | None


def render(
    width: int,
    theme: ThemeLike,
    *,
    version: str,
    context: str,
    hint: str,
    eyes: str = "idle",
    tail_offset: int = 0,
) -> Pet:
    """Draw the ferret ``theme`` calls for at ``width`` columns.

    ``eyes`` is a key of :data:`EYES` (the same keys as ``banner.EYES``), so the
    app's animation is theme-blind. ``tail_offset`` modulo three picks the tail's
    frame (:data:`TAIL_TIPS`): 0 is the tail at rest, 1 the tip lifted, 2 dropped,
    so a counter that only ever grows wags it forever. Below
    :data:`RICH_WIDTH_FLOOR` the PLAIN pet is drawn whatever the theme, because
    the RICH head alone needs the room.
    """
    if theme.name != "rich" or width < RICH_WIDTH_FLOOR:
        return _plain(width, version=version, context=context, hint=hint, eyes=eyes)
    return _rich(
        width, version=version, context=context, hint=hint, eyes=eyes, tail_offset=tail_offset
    )


def _plain(width: int, *, version: str, context: str, hint: str, eyes: str) -> Pet:
    """The PLAIN ferret is ``banner.render``, unchanged; only the spans are re-shaped."""
    drawn = banner.render(
        width, version=version, context=context, hint=hint, eyes=banner.EYES[eyes]
    )
    eyes_start = len(banner.EYES_INDENT)
    return Pet(
        lines=drawn.lines,
        stamp_span=None if drawn.stamp_span is None else (1, *drawn.stamp_span),
        eyes_span=(1, eyes_start, eyes_start + len(banner.EYES[eyes])),
        tail_span=None,
    )


def _rich(width: int, *, version: str, context: str, hint: str, eyes: str, tail_offset: int) -> Pet:
    with_stamp = width >= STAMP_WIDTH_FLOOR
    stamp = STAMP if with_stamp else ""
    # Behind the head there is room for the body, one rump corner and the tail; the
    # stamp then ends at the two-column margin, level with the two text lines.
    room = width - banner.RIGHT_MARGIN - len(stamp) - len(BROW) - 1
    tail = max(TAIL_MIN, room // TAIL_SHARE) if with_stamp else TAIL_SHORT
    body = max(room - tail, MIN_BODY)
    # The column the three rump corners share: ``╮`` back, ``╰`` mid-line, ``╯`` belly.
    edge = len(BROW) + body

    back_line = BROW + "─" * body + "╮"
    tail_cells = "~" * (tail - 1) + TAIL_TIPS[tail_offset % len(TAIL_TIPS)]
    face_line = (
        FACE[:EYES_COLUMN] + EYES[eyes] + " " * (edge - len(FACE)) + "╰" + tail_cells + stamp
    )
    belly_line = CHIN + "─" * (edge - len(CHIN) - len(RUMP) + 1) + RUMP
    # The rear feet stand under the rump's two joints, the first cell of ``RUMP``.
    feet_line = FEET + " " * (edge - len(RUMP) + 1 - len(FEET)) + REAR_FEET

    version_line = _right_align(f"{TEXT_INDENT}proofpath v{version}", context, width)
    hint_line = _hint_line(hint, width)
    tail_start = edge + 1
    return Pet(
        lines=(EARS, back_line, face_line, belly_line, feet_line, version_line, hint_line),
        stamp_span=(2, len(face_line) - len(STAMP), len(face_line)) if with_stamp else None,
        eyes_span=(2, EYES_COLUMN, EYES_COLUMN + len(EYES[eyes])),
        tail_span=(2, tail_start, tail_start + tail),
    )


def _hint_line(hint: str, width: int) -> str:
    """Prompt hint left, slash commands right; a hint with no gap is indented whole."""
    parts = HINT_GAP.split(hint, maxsplit=1)
    if len(parts) == 1:
        return TEXT_INDENT + hint
    return _right_align(TEXT_INDENT + parts[0], parts[1], width)


def _right_align(left: str, right: str, width: int) -> str:
    """Pad ``left`` so ``right`` ends at the two-column margin, never truncating."""
    padding = width - banner.RIGHT_MARGIN - len(left) - len(right)
    return left + " " * max(padding, 1) + right


# --- the animation -------------------------------------------------------------------


@dataclass(frozen=True)
class AnimState:
    """Everything the eyes and the tail need to remember between two clock readings.

    ``rng`` is the one part that is not a value: it is the seeded source of the
    blink schedule, shared (not copied) through ``replace`` and kept in the state so
    two sessions on one screen (seeded from different clocks) do not blink in
    lockstep and a test (seeded by hand) gets the same schedule every time.
    """

    rng: random.Random
    #: When the next blink starts.
    blink_at: float
    #: When the blink in progress ends; in the past when the eyes are open.
    blink_until: float = 0.0
    #: The expression a finished run left, and when it stops holding.
    mood: Mood | None = None
    mood_until: float = 0.0
    #: The tail's frame (an index into :data:`TAIL_TIPS`) and when it next steps.
    tail_frame: int = 0
    #: Position in :data:`WAG_CYCLE`; ``-1`` so the first busy reading lifts the tip.
    wag_step: int = -1
    wag_at: float = 0.0

    @classmethod
    def start(cls, now: float, *, rng: random.Random | None = None) -> AnimState:
        """Open the eyes at ``now`` with the first blink due somewhere in the window."""
        rng = random.Random() if rng is None else rng
        return cls(rng=rng, blink_at=now + _gap(rng))


def next_eyes(
    state: AnimState,
    now: float,
    *,
    running: bool,
    ended_with: Mood | None,
) -> tuple[AnimState, str]:
    """The expression ``now`` calls for, and the state that follows from it.

    ``running`` is whether any run is fetching or verifying; ``ended_with`` is set
    on the one call after a run finished. Order matters: a finished run's two
    seconds outrank everything, then a run still working, and only an idle ferret
    blinks. A blink that fell due while the eyes were busy is dropped rather than
    fired the moment the run ends, which would read as part of the result.
    """
    if ended_with is not None:
        state = replace(state, mood=ended_with, mood_until=now + FLASH_SECONDS)
    if state.mood is not None:
        if now < state.mood_until:
            return state, state.mood
        state = replace(state, mood=None, mood_until=0.0)
    if running:
        # Pushed out on every busy reading, so the first idle one is never a blink.
        state = replace(state, blink_at=now + _gap(state.rng), blink_until=0.0)
        return state, "busy"
    if now < state.blink_until:
        return state, "blink"
    if now >= state.blink_at:
        state = replace(state, blink_until=now + BLINK_SECONDS, blink_at=now + _gap(state.rng))
        return state, "blink"
    return state, "idle"


def next_tail(state: AnimState, now: float, *, running: bool) -> tuple[AnimState, int]:
    """The tail's frame ``now``, and the state that follows from it.

    The tail wags only while a run is busy: the tip lifts the instant the run
    starts, steps every :data:`WAG_SECONDS`, and falls level the instant the run
    ends. Each run's wag starts afresh, so the tip never freezes mid-air.
    """
    if not running:
        if state.tail_frame or state.wag_at or state.wag_step != -1:
            state = replace(state, tail_frame=0, wag_step=-1, wag_at=0.0)
        return state, 0
    if now >= state.wag_at:
        step = (state.wag_step + 1) % len(WAG_CYCLE)
        state = replace(state, wag_step=step, tail_frame=WAG_CYCLE[step], wag_at=now + WAG_SECONDS)
    return state, state.tail_frame


def _gap(rng: random.Random) -> float:
    return rng.uniform(*BLINK_WINDOW)


__all__ = [
    "BLINK_SECONDS",
    "BLINK_WINDOW",
    "EYES",
    "FLASH_SECONDS",
    "STAMP",
    "TAIL_TIPS",
    "WAG_SECONDS",
    "AnimState",
    "Pet",
    "ThemeLike",
    "next_eyes",
    "next_tail",
    "render",
]
