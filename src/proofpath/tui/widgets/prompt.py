"""The input bar and the inline section 7.1 question."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from proofpath import ui
from proofpath.browser import TUI_ANSWERS, Answer, prompt_text
from proofpath.tui.history import History
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
#: What the bar shows while it holds a paste of several lines: the paste itself would
#: not fit, and Textual's ``Input`` would keep its first line and drop the rest.
HELD_SUMMARY = "pasted {sep} {lines} lines {sep} {chars} chars {sep} enter to check, esc to drop"


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
        #: question, in the same words, about the same download. Only the last line
        #: differs: it names the buttons and the ``/allow`` answers, not the keys.
        self.question = prompt_text(host, status, answers=TUI_ANSWERS)
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
    """The input bar. Awaiting mode tints it with the next run's accent.

    ``up`` and ``down`` walk the command history the way a shell does: the draft in
    the bar when the walk starts comes back one step past the newest entry, and any
    edit ends the walk. The app hands in the :class:`History` before the bar mounts.

    ``tab`` completes a slash command the way a shell does too: one candidate lands
    at once, several are cycled through on each further ``tab``, wrapping, and any
    edit ends the cycle. The app hands in ``complete`` because only it knows which
    runs are still live; the bar never decides what a line may become.

    A paste of two or more lines is *held*: the bar shows :data:`HELD_SUMMARY` and
    keeps the text whole in :attr:`held` until ``enter`` takes it or an edit drops
    it. A one-line paste is text in the bar like any other. The app hands in ``sep``
    because the bar has no theme of its own.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "history_previous", "previous command", show=False),
        Binding("down", "history_next", "next command", show=False),
        Binding("tab", "complete", "complete", show=False),
    ]

    def __init__(
        self,
        placeholder: str = "",
        *,
        id: str | None = None,  # noqa: A002 - Textual's own keyword, kept for callers
        history: History | None = None,
        complete: Callable[[str], list[str]] | None = None,
        sep: str = "·",
    ) -> None:
        # ``Input`` selects all its text on focus by default; a key handed over from a
        # log line (``Line.on_key``) must insert at the cursor, not replace a draft,
        # and a command bar keeps its cursor where it left it, the way a shell does.
        super().__init__(placeholder=placeholder, id=id, select_on_focus=False)
        self.history = history if history is not None else History()
        self._complete: Callable[[str], list[str]] = complete or (lambda text: [])
        #: The candidates of the cycle in progress and where the cycle is in them;
        #: cleared by any edit that is not the cycle's own.
        self._candidates: list[str] = []
        self._candidate = 0
        #: The lines the walk has put in the bar whose ``Changed`` has not arrived yet,
        #: oldest first. A flag raised around the assignment would not do: ``Input``
        #: *posts* ``Changed`` rather than calling the handler, so it arrives a tick
        #: later, after any such flag has been lowered. Nor would the last line alone:
        #: the app forwards keys without waiting, so a held ``up`` can be handled twice
        #: before the first ``Changed`` is read, and the first would then look like an
        #: edit of the second. Each ``Changed`` is matched against the line whose turn
        #: it is; the queue is what tells the walk's own replacements from a keystroke.
        self._pending: deque[str] = deque()
        self._sep = sep
        #: A multi-line paste, whole, while the bar shows its summary. ``take`` is the
        #: only way it leaves; ``_show`` forgets it, because whatever replaces the
        #: summary — a history step, a completion, an empty bar — is not the paste.
        self.held: str | None = None

    def action_history_previous(self) -> None:
        self.drop()  # a held paste is not a draft; the walk must not bring it back
        line = self.history.previous(self.value)
        if line is not None:
            self._show(line)

    def action_history_next(self) -> None:
        self.drop()
        line = self.history.next()
        if line is not None:
            self._show(line)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        # ``tab`` is completion only inside a slash command; anywhere else it stays
        # Textual's focus-next, which is how the keyboard reaches the log. Either
        # falsy answer lets the key pass on: ``False`` disables the binding and hides
        # it from the footer, ``None`` disables it but keeps it shown. This binding
        # has ``show=False``, so the two behave the same; ``None`` is used because it
        # is the one that reads as "off for now", and the binding comes back with
        # the next ``/``.
        if action == "complete" and not self.value.startswith("/"):
            return None
        return True

    def action_complete(self) -> None:
        # A cycle is only good for the line it was built over. A history step goes
        # through ``_show`` too, so ``on_input_changed`` never sees it as an edit and
        # never clears ``_candidates`` — without this check, a stale cycle from before
        # the step would overwrite whatever the step just put in the bar. So the cycle
        # is stale, and rebuilt, whenever the bar no longer holds the candidate it last
        # showed, for any reason, not just an edit.
        if not self._candidates or self.value != self._candidates[self._candidate]:
            self._candidates = self._complete(self.value)
            self._candidate = 0
            if not self._candidates:
                return
        else:
            self._candidate = (self._candidate + 1) % len(self._candidates)
        # Through ``_show`` like a history step: the replacement is the bar's own, so
        # its ``Changed`` is matched off the queue instead of ending the cycle.
        self._show(self._candidates[self._candidate])

    def _show(self, line: str) -> None:
        # A reactive posts no ``Changed`` for an equal assignment, so only a line that
        # actually replaces the bar's value has a ``Changed`` to wait for.
        self.held = None
        if line != self.value:
            self._pending.append(line)
            self.value = line
        self.cursor_position = len(line)

    # Neither override below calls ``super()``: Textual walks the MRO and calls
    # ``Input``'s own handler next, as the event's *default action*, unless the
    # event's ``prevent_default`` was called first. Calling it here as well would
    # run it twice — a one-line paste inserted twice, a held paste with the first
    # line typed over its summary and a second ``Changed`` that reads as an edit.

    def _on_paste(self, event: events.Paste) -> None:
        # ``Input`` keeps the first line of a paste and drops the rest, silently. A
        # post is several lines, so one of two or more is held whole behind a summary
        # instead. Trailing whitespace is cut first: a path pasted with its newline is
        # still one line and still lands in the bar as text, through ``Input``.
        text = event.text.rstrip()
        lines = text.splitlines()
        if len(lines) < 2:
            return
        counted = sum(1 for line in lines if line.strip())
        self._show(HELD_SUMMARY.format(sep=self._sep, lines=counted, chars=len(text)))
        self.held = text  # after ``_show``, which forgets any hold
        event.prevent_default()
        event.stop()

    async def _on_key(self, event: events.Key) -> None:
        # A printable key into the summary would edit the summary, which is nothing:
        # the paste is dropped first, so the key ``Input`` then types lands in an
        # empty bar. ``async`` here awaits nothing of its own -- it matches
        # ``Input._on_key``, which is itself ``async``, so the dispatcher that walks
        # the MRO (see the note above ``_on_paste``) calls the same shape on both.
        if self.held is not None and event.is_printable:
            self.drop()

    def take(self) -> str | None:
        """The held paste, whole, and the bar no longer holds it; ``None`` when none."""
        held, self.held = self.held, None
        return held

    def drop(self) -> None:
        """Forget the held paste and empty the bar. Nothing when nothing is held."""
        if self.held is not None:
            self._show("")

    def on_input_changed(self, event: Input.Changed) -> None:
        # Typing, deleting or pasting ends the walk and the cycle; the bar's own
        # replacements (a history step, a completion) do not.
        if self._pending and event.value == self._pending[0]:
            self._pending.popleft()
            return
        self._pending.clear()
        self.history.reset()
        self._candidates = []
        # An edit of the summary is an edit of nothing: the paste goes with it.
        self.drop()

    def tint(self, accent: str) -> None:
        base = textual_colour(accent)
        # The ANSI flag is dropped on purpose: a palette entry cannot be blended, and
        # the spec asks for a tint rather than a solid bar. The *name* still comes
        # from the theme's accents; only its RGB is borrowed.
        self.styles.background = Color(base.r, base.g, base.b, TINT_ALPHA)

    def untint(self) -> None:
        self.styles.background = None
