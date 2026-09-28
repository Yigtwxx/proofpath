"""The one-line notice above the prompt: the judge switched to a local model.

Spec section 11: the switch is never silent. The run's own block carries the line as
a note too, but that scrolls away with the log; this row sits directly above the bar,
right-aligned, from the moment the run switches until the next run starts.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from proofpath.tui.theme import Theme


class JudgeNotice(Static):
    """Hidden (``display: none``) until :meth:`show`; never an empty line.

    An empty line would still take a row out of the bottom dock, and the dock's
    height is what keeps the footer and the bar in place (rule 6).
    """

    def __init__(self, theme: Theme, *, coloured: bool) -> None:
        super().__init__(id="judge-notice")
        self._theme = theme
        self._coloured = coloured
        self.text = ""
        self.display = False

    @property
    def rows(self) -> int:
        """What the notice takes from the screen: one row when shown, none when not."""
        return 1 if self.display else 0

    def show(self, text: str) -> None:
        self.text = text
        # ``caution``: the run is still judged, by a different model than configured.
        style = self._theme.tone("caution") if self._coloured else ""
        self.update(Text(text, style=style, no_wrap=True, overflow="ellipsis"))
        self.display = True

    def hide(self) -> None:
        self.text = ""
        self.update("")
        self.display = False
