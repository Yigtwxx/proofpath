"""The input bar and the inline section 7.1 question."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from proofpath import ui
from proofpath.browser import Answer, prompt_text
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import textual_colour

#: The spec's "~15-20%" tint, at the quiet end: an accent on a bar, not a highlight.
TINT_ALPHA = 0.15
#: The four answers of spec section 7.1, in the order the prompt draws them, with the
#: label each button carries. A click is worth exactly the ``/allow`` line beside it.
ANSWER_LABELS: dict[Answer, str] = {
    "once": "allow once",
    "always": "always",
    "no": "no",
    "never": "never",
}
#: Prefix of a permission button's id, so one handler reads every one of them.
ANSWER_ID = "allow-"


class PermissionPrompt(Vertical):
    """The section 7.1 question, asked where it happened (spec section 13.1).

    Never a modal: it is mounted inside the block of the run that hit the wall, under
    the stage that hit it, and the log keeps scrolling around it. The four buttons are
    worth exactly the four ``/allow`` answers, because they resolve the same future —
    a terminal without mouse reporting loses nothing (spec section 13.1's mouse rule).

    The container is vertical although the buttons are a row: the question is the ten
    lines ``browser.prompt_text`` writes, and laying those out *beside* four buttons
    would leave neither readable at 80 columns. The row itself is the ``Horizontal``.

    Nothing here decides anything. The answer goes back to the gate, which is the only
    thing that may act on it, and the install still happens behind it (rule 5).
    """

    def __init__(self, owner: int, host: str, status: int | None, out: ui.Ui, theme: Theme) -> None:
        super().__init__(classes="permission")
        #: The run this question belongs to, or the negative id of a ``/fetch`` block.
        self.owner = owner
        #: The section 7.1 block, verbatim: the terminal and the TUI ask the same
        #: question, in the same words, about the same download.
        self.question = prompt_text(host, status)
        self._out = out
        self._theme = theme
        #: The answer once it was given, so a scrolled-back log still says what was
        #: allowed and what was not (rule 6).
        self.answer: Answer | None = None
        self._answered = Static("", classes="answered")

    def compose(self) -> ComposeResult:
        yield Static(self._question())
        with Horizontal(classes="answers"):
            for answer, label in ANSWER_LABELS.items():
                # A ``Text``, not a string: Textual reads ``[...]`` in a string label as
                # its own content markup and the brackets the spec draws would vanish.
                yield Button(Text(f"[{label}]"), id=f"{ANSWER_ID}{answer}", classes="answer")
        yield self._answered

    def on_mount(self) -> None:
        self._answered.display = False
        if not self._out.color:
            return
        # Caution is what section 13.3 gives the permission prompt (yellow): not a
        # finding. ``ui`` names the colour, the theme spells it, and this only turns
        # it into the form Textual's stylesheet takes.
        colour = textual_colour(self._theme.tones[ui.PROMPT_COLOUR])
        for button in self.query(Button):
            button.styles.color = colour

    def _question(self) -> Text:
        text = Text(self.question)
        if self._out.color:
            text.stylize(self._theme.tones[ui.PROMPT_COLOUR])
        return text

    def settle(self, answer: Answer) -> None:
        """Record the answer and take the buttons away: it is asked once per run."""
        self.answer = answer
        for button in self.query(Button):
            button.disabled = True
        self.query_one(".answers", Horizontal).display = False
        self._answered.update(
            Text(f"    answered: {ANSWER_LABELS[answer]}", style=self._theme.tone("muted"))
        )
        self._answered.display = True


class Prompt(Input):
    """The input bar. Awaiting mode tints it with the next run's accent."""

    def tint(self, accent: str) -> None:
        base = textual_colour(accent)
        # The ANSI flag is dropped on purpose: a palette entry cannot be blended, and
        # the spec asks for a tint rather than a solid bar. The *name* still comes
        # from the theme's accents; only its RGB is borrowed.
        self.styles.background = Color(base.r, base.g, base.b, TINT_ALPHA)

    def untint(self) -> None:
        self.styles.background = None
