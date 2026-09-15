"""The ferret, pinned above the log and redrawn on every resize (spec section 13.1).

The art and the animation are :mod:`proofpath.tui.pet`'s; this widget only asks
it what to draw for the theme and the width, paints the stamp in the theme's red,
and drives the eyes and the tail from one timer. The timer exists at all only when
the terminal reports colour and ``-q`` was not asked for: a log being piped or read
by a screen reader gets a still banner rather than four characters rewriting
themselves forever.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from rich.text import Text
from textual.geometry import Size
from textual.timer import Timer
from textual.widgets import Static

from proofpath.tui import banner, pet
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import DEFAULT_WIDTH

#: How often the eyes and the tail are re-read, in seconds. The blink itself lasts
#: ``pet.BLINK_SECONDS``; the tick only has to be shorter than that.
EYES_TICK = 0.05


@dataclass(frozen=True)
class Drawing:
    """What the banner last drew, in the shape ``banner.Banner`` has always reported.

    ``stamp_span`` is the stamp's ``(start, end)`` columns on line ``stamp_line``, so a
    reader that knew the PLAIN banner reads the RICH one the same way.
    """

    lines: tuple[str, ...]
    stamp_span: tuple[int, int] | None
    stamp_line: int


class Banner(Static):
    """The ferret, pinned above the log and redrawn on every resize.

    The stamp is the only coloured element in it (spec section 13.1) and the red is
    the theme's ``stamp``; nothing else here names a colour.
    """

    def __init__(
        self,
        *,
        version: str,
        context: str,
        hint: str,
        theme: Theme,
        coloured: bool,
        animated: bool = False,
    ) -> None:
        super().__init__(id="banner")
        self._version = version
        self._banner_context = context
        self._hint = hint
        self._theme = theme
        self._coloured = coloured
        self._animated = animated
        #: The last drawing, so a caller (and a test) can read the exact lines back.
        self._pet: pet.Pet | None = None
        #: The timer that drives the animation (``None`` when it is off -- which is
        #: how a test, and a reader, can tell).
        self.timer: Timer | None = None
        #: Read through an attribute so a test can hand the widget a clock of its own
        #: and need not wait two real seconds for an expression to expire.
        self.clock: Callable[[], float] = time.monotonic
        # Seeded from the wall clock, so two sessions on one screen do not blink in
        # lockstep; the schedule itself is just "somewhere in the next 6-10 seconds".
        self._state = pet.AnimState.start(self.clock(), rng=random.Random(time.time()))
        self._eyes = "idle"
        self._tail = 0
        self._busy = False
        self._ended: pet.Mood | None = None

    # --- what the app tells it ------------------------------------------------------

    def start_animation(self) -> None:
        """Begin blinking, if this session animates at all. Idempotent."""
        if not self._animated or self.timer is not None:
            return
        self._state = pet.AnimState.start(self.clock(), rng=self._state.rng)
        self.timer = self.set_interval(EYES_TICK, self.tick)

    def set_context(self, context: str) -> None:
        """Line 3 changed under it — ``permissions.network``, and so online/offline."""
        if context != self._banner_context:
            self._banner_context = context
            self.refresh(layout=True)

    def set_busy(self, busy: bool) -> None:
        """Eyes down the path while any run is fetching or verifying; idle after."""
        if not self._animated or busy == self._busy:
            return
        self._busy = busy
        self.tick()

    def flash(self, mood: pet.Mood) -> None:
        """Wide eyes for two seconds on findings, the happy face for two on a clean run."""
        if not self._animated:
            return
        self._ended = mood
        self.tick()

    def tick(self) -> None:
        """Re-read the state and repaint only when the eyes or the tail moved."""
        now = self.clock()
        self._state, eyes = pet.next_eyes(
            self._state, now, running=self._busy, ended_with=self._ended
        )
        self._ended = None
        self._state, tail = pet.next_tail(self._state, now, running=self._busy)
        if (eyes, tail) != (self._eyes, self._tail):
            self._eyes, self._tail = eyes, tail
            self.refresh()

    # --- what a reader (and a test) can ask it --------------------------------------

    @property
    def eyes(self) -> str:
        """The eyes as chosen, spelled for the ferret on screen: ``(o.o)`` or ``o o``.

        Read off the state and not the drawing, because a ``tick`` only *asks* for a
        repaint: a reader that looked at the pixels would be one frame behind.
        """
        rich = self._pet is not None and self._pet.tail_span is not None
        return (pet.EYES if rich else banner.EYES)[self._eyes]

    @property
    def drawn(self) -> Drawing | None:
        drawn = self._pet
        if drawn is None:
            return None
        if drawn.stamp_span is None:
            return Drawing(drawn.lines, None, -1)
        line, start, end = drawn.stamp_span
        return Drawing(drawn.lines, (start, end), line)

    # --- drawing --------------------------------------------------------------------

    def render(self) -> Text:
        drawn = self._draw(self.size.width or self.app.size.width or DEFAULT_WIDTH)
        text = Text("\n".join(drawn.lines))
        if drawn.stamp_span is not None and self._coloured:
            line, start, end = drawn.stamp_span
            offset = sum(len(row) + 1 for row in drawn.lines[:line])
            text.stylize(self._theme.stamp, offset + start, offset + end)
        return text

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """The drawn lines, but more rows than lines once one of them wraps.

        ``pet.render`` never truncates: below about 60 columns the version and the
        run context stop fitting side by side and the PLAIN hint line is wider still,
        so the container is sized to what was actually drawn. A banner clipped to its
        line count would hide the line that says which commands exist.
        """
        return sum(max(1, -(-len(line) // max(width, 1))) for line in self._draw(width).lines)

    def _draw(self, width: int) -> pet.Pet:
        self._pet = self._draw_pet(width, self._hint)
        if self._theme.name == "rich" and len(self._pet.lines[-1]) > width:
            # The RICH hint drops its command list rather than overflow the line
            # (``/help`` lists them); the PLAIN line is the spec's own and is left
            # to wrap, as it always has.
            self._pet = self._draw_pet(width, pet.HINT_GAP.split(self._hint, maxsplit=1)[0])
        return self._pet

    def _draw_pet(self, width: int, hint: str) -> pet.Pet:
        return pet.render(
            width,
            self._theme,
            version=self._version,
            context=self._banner_context,
            hint=hint,
            eyes=self._eyes,
            tail_offset=self._tail,
        )

    def on_mount(self) -> None:
        self.start_animation()

    def on_resize(self) -> None:
        # The height is width-dependent, so a resize is a re-layout and not a repaint.
        self.refresh(layout=True)
