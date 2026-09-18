"""The wordmark, in both themes, pinned above the log and redrawn on every resize
(wordmark design §4, §8 and §8.1).

The art and the layout are :mod:`proofpath.tui.wordmark`'s; this widget only asks it
what to draw for the theme and the width and paints the colour runs it returns with
the theme's banner tones. Nothing moves: ``set_busy`` and ``flash`` are kept so the
app can keep calling them, and do nothing, until an animation is designed again.
"""

from __future__ import annotations

from typing import Literal

from rich.text import Text
from textual.geometry import Size
from textual.widgets import Static

from proofpath.tui import banner, wordmark
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import DEFAULT_WIDTH

Mood = Literal["findings", "clean"]


class Banner(Static):
    """The wordmark, in both themes, pinned above the log and redrawn on every resize."""

    def __init__(
        self,
        *,
        version: str,
        context: str,
        hint: str,
        theme: Theme,
        coloured: bool,
        # Accepted for the app's sake and unused: the banner has no animation yet.
        animated: bool = False,
    ) -> None:
        super().__init__(id="banner")
        self._version = version
        self._banner_context = context
        self._hint = hint
        self._theme = theme
        self._coloured = coloured
        #: The last drawing, so a caller (and a test) can read the exact lines back.
        self._drawn: banner.Banner | None = None

    # --- what the app tells it ------------------------------------------------------

    def set_context(self, context: str) -> None:
        """The context line changed under it -- ``permissions.network``, so online/offline."""
        if context != self._banner_context:
            self._banner_context = context
            self.refresh(layout=True)

    def set_busy(self, busy: bool) -> None:
        """Reserved: the banner does not watch the runs yet (raven design §2)."""

    def flash(self, mood: Mood) -> None:
        """Reserved: the banner does not react to a finished run yet (raven design §2)."""

    # --- what a reader (and a test) can ask it --------------------------------------

    @property
    def drawn(self) -> banner.Banner | None:
        return self._drawn

    # --- drawing --------------------------------------------------------------------

    def render(self) -> Text:
        drawn = self._draw(self.size.width or self.app.size.width or DEFAULT_WIDTH)
        text = Text("\n".join(drawn.lines))
        if self._coloured:
            offset = 0
            for line, runs in zip(drawn.lines, drawn.tones, strict=True):
                for start, end, tone in runs:
                    text.stylize(self._theme.banner[tone], offset + start, offset + end)
                offset += len(line) + 1
        return text

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """The drawn lines, but more rows than lines once one of them wraps.

        The banner is 3, 7 or 9 rows (no mark, beside, under); ``wordmark.render``
        never truncates, so a text row longer than the width wraps, and the container
        is sized to what was actually drawn rather than to the line count.
        """
        return sum(max(1, -(-len(line) // max(width, 1))) for line in self._draw(width).lines)

    def _draw(self, width: int) -> banner.Banner:
        self._drawn = self._render_banner(width, self._hint)
        where = wordmark.rows(
            width, version=self._version, context=self._banner_context, hint=self._hint
        )
        if len(self._drawn.lines[where.hint]) > width:
            # The hint drops its command list rather than overflow (``/help`` lists
            # them), in both themes: the plain banner is the same drawing now. Below
            # the floor there is no mark and the same rule applies to the bare text.
            self._drawn = self._render_banner(width, banner.split_hint(self._hint)[0])
        return self._drawn

    def _render_banner(self, width: int, hint: str) -> banner.Banner:
        return wordmark.render(
            width,
            self._theme,
            version=self._version,
            context=self._banner_context,
            hint=hint,
        )

    def on_resize(self) -> None:
        # The height is width-dependent, so a resize is a re-layout and not a repaint.
        self.refresh(layout=True)
