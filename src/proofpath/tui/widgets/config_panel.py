"""``/config`` as a settings panel: the file's rows, moved through with the arrow keys
and changed in place (wordmark design section 12).

The panel is a block in the log like a mirrored verb's -- inline, never a modal --
and it holds no opinion about what a setting means: every write goes through
:func:`proofpath.commands.config_set`, the function ``/config set`` and the CLI call,
so a value the panel accepts is exactly a value the file accepts. After each write
the file is read back and every row redrawn from it, because one write can change
four rows (``judge.provider`` carries its model, base URL and key variable with it).

The rows are built from the config's own dataclasses rather than written out here:
their defaults are what ``Backspace`` restores, and a default that lived in two
places would drift. The API key's *value* is never on screen; the ``key`` line says
which variable is looked up and where a value for it was found, nothing more
(``secrets.ApiKey`` will not print it either).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from rich.text import Text
from textual import events as tevents
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Static

from proofpath import ui
from proofpath.commands import ConfigView, config_set, config_view
from proofpath.config import (
    PERMISSION_VALUES,
    ConfigError,
    Contact,
    FetchConfig,
    JudgeConfig,
    Permissions,
)
from proofpath.judge import JudgeError, default_dotenv_paths, known_providers
from proofpath.secrets import resolve_api_key
from proofpath.tui.theme import Theme
from proofpath.tui.widgets._shared import panelled, textual_colour

Kind = Literal["choice", "text"]

#: The column a row's value starts at, after the marker, a space and the padded
#: name; measured off the spec's own mock, where the longest name is
#: ``install_browser``.
NAME_WIDTH = 18
#: The head's two labels (``config``, ``key``) are padded to this; the values line up.
HEAD_WIDTH = 9
#: The gap between the choices of a row (``ask   allow   deny``).
CHOICE_GAP = "   "
#: A choice row's meaning sits under it, indented past the marker and the name's start.
MEANING_INDENT = "    "
#: What an empty text value is written as, so a blank row cannot pass for a row that
#: was skipped.
UNSET = "(unset)"
#: The ink a current value's badge is written in. The five accents are the light
#: hues, so the dark ink the theme uses on its two light meaning tones reads on all
#: of them.
BADGE_INK = "black"
#: The panel's title on its border, exactly the line that opened it.
TITLE = "/config"


@dataclass(frozen=True)
class Row:
    """One setting: its dotted key, how it is changed, and what ``Backspace`` restores."""

    key: str
    kind: Kind
    values: tuple[str, ...]  # the choices; empty for a text row
    meaning: str  # one dim line under a choice row, or beside a text row; "" for none
    default: str

    @property
    def name(self) -> str:
        """The key without its section: what the row line shows."""
        return self.key.partition(".")[2]


@dataclass(frozen=True)
class Section:
    name: str
    rows: tuple[Row, ...]


def _default(value: object) -> str:
    """A dataclass default as the panel writes it: bools lowered, everything else as is."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def sections() -> tuple[Section, ...]:
    """The panel's rows in the config's own order. Built, not constant, because the
    provider list is the judge module's, and the defaults are the dataclasses' own."""
    permissions, fetch, judge, contact = Permissions(), FetchConfig(), JudgeConfig(), Contact()
    return (
        Section(
            "permissions",
            (
                Row(
                    "permissions.install_browser",
                    "choice",
                    PERMISSION_VALUES,
                    "step 3 of the fetch ladder: a ~280 MB browser engine, spec section 7.1",
                    _default(permissions.install_browser),
                ),
                Row(
                    "permissions.network",
                    "choice",
                    PERMISSION_VALUES,
                    "every fetch and every provider call; deny = offline",
                    _default(permissions.network),
                ),
            ),
        ),
        Section(
            "fetch",
            (
                Row(
                    "fetch.respect_robots",
                    "choice",
                    ("true", "false"),
                    "honour robots.txt when fetching",
                    _default(fetch.respect_robots),
                ),
            ),
        ),
        Section(
            "judge",
            (
                Row(
                    "judge.provider",
                    "choice",
                    known_providers(),
                    "switching a provider also sets its model, base_url and api_key_env",
                    _default(judge.provider),
                ),
                Row("judge.model", "text", (), "", _default(judge.model)),
                Row("judge.base_url", "text", (), "", _default(judge.base_url)),
                Row("judge.api_key_env", "text", (), "", _default(judge.api_key_env)),
            ),
        ),
        Section(
            "contact",
            (
                Row(
                    "contact.email",
                    "text",
                    (),
                    "optional, for the Crossref / OpenAlex polite pools",
                    _default(contact.email),
                ),
            ),
        ),
    )


class PanelLine(Static):
    """One line of the panel: a ``Text`` the panel redraws in place.

    A ``Static`` of its own so a reader (and a test) gets the ``Text`` back from
    ``render()`` with its spans, the way every other line in the log answers.
    """

    def __init__(self, text: Text | None = None) -> None:
        super().__init__()
        self._text = text if text is not None else Text("")

    def set(self, text: Text) -> None:
        self._text = text
        self.refresh(layout=True)

    def render(self) -> Text:
        return self._text


class ConfigPanel(Vertical):
    """The settings panel: the config's rows, one selected, changed where they stand.

    Focusable, so the arrow keys are its own; a printable key is typing and goes to
    the bar (wordmark design section 11), and the blur that comes with it folds the
    panel to its one-line record. While a text row is being edited the focused widget
    is the row's ``Input``, so keys go to the edit and the panel's own bindings wait,
    all but ``escape``, which cancels the edit.

    Nothing here decides what a value means. A write is ``config_set``, the answer is
    the file read back, and the app is told what was written so it can reload its
    config and say so in the log -- or told what went wrong, so the row keeps its
    value and the error is a line under the panel rather than a silent no-op.
    """

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move(-1)", "row up", show=False),
        Binding("down", "move(1)", "row down", show=False),
        Binding("left", "cycle(-1)", "previous value", show=False),
        Binding("right", "cycle(1)", "next value", show=False),
        Binding("enter", "activate", "change or edit", show=False),
        Binding("backspace", "reset", "default", show=False),
        Binding("escape", "close", "close", show=False),
    ]

    class Written(Message):
        """``config_set`` wrote these pairs to ``path``: the app reloads and says so."""

        def __init__(self, pairs: tuple[tuple[str, str], ...], path: Path) -> None:
            super().__init__()
            self.pairs = pairs
            self.path = path

    class Failed(Message):
        """A write was refused (``ConfigError``/``JudgeError``); the row kept its value."""

        def __init__(self, message: str) -> None:
            super().__init__()
            self.message = message

    class Closed(Message):
        """``Esc`` folded the panel: focus belongs back on the bar."""

    def __init__(self, view: ConfigView, accent: str, out: ui.Ui, theme: Theme) -> None:
        super().__init__(classes=f"run-block config-panel {theme.name}")
        self._view = view
        #: The accent the panel wears: its border, its marker and its value badges.
        self.accent = accent
        self._out = out
        self._theme = theme
        self.sections = sections()
        #: Every row in selectable order: headings and meaning lines are not rows.
        self.rows: tuple[Row, ...] = tuple(row for section in self.sections for row in section.rows)
        self.selected = 0
        #: Folded to its one-line record. A collapsed panel never opens again;
        #: ``/config`` opens a fresh one.
        self.collapsed = False
        #: A text row's ``Input`` is open and has focus.
        self.editing = False
        self._edit: Input | None = None
        self._edit_row: Horizontal | None = None
        self._head = PanelLine()
        self._key = PanelLine()
        self._lines: dict[str, PanelLine] = {row.key: PanelLine() for row in self.rows}
        self._legend = PanelLine()
        self._body = Vertical(*self._compose_body(), classes="stages")
        self._summary = PanelLine()

    def _compose_body(self) -> list[Static]:
        """The head, the sections with their rows, the legend: top to bottom."""
        lines: list[Static] = [self._head, self._key, PanelLine()]
        for section in self.sections:
            lines.append(PanelLine(Text(f" {section.name}", style=self._muted())))
            for row in section.rows:
                lines.append(self._lines[row.key])
                if row.kind == "choice" and row.meaning:
                    lines.append(PanelLine(Text(MEANING_INDENT + row.meaning, style=self._muted())))
            lines.append(PanelLine())
        lines.append(self._legend)
        return lines

    def compose(self) -> ComposeResult:
        yield self._body
        yield self._summary

    def on_mount(self) -> None:
        self._summary.display = False
        self._draw_all()
        self._fit()

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        """A bordered panel above the floor, flat rows below it or once collapsed.

        The border and its title are drawn the way a run's block draws its own: the
        theme's box in the accent, or the terminal's default without colour, which is
        still a frame. A collapsed panel is one line and a frame around one line would
        be three, so the border goes with the fold.
        """
        wide = panelled(self._theme, self.app.size.width) and not self.collapsed
        if wide:
            colour = textual_colour(self.accent if self._out.color else "")
            self.styles.border = (self._theme.glyphs.box, colour)
            self.border_title = TITLE
        else:
            self.styles.border = None
            self.border_title = ""
        self.set_class(not wide, "narrow")

    # --- what a reader (and a test) can ask it --------------------------------------

    def value(self, key: str) -> str:
        """The current value of ``key`` as the panel writes it: bools lowered."""
        section, _, name = key.partition(".")
        return _default(getattr(getattr(self._view.config, section), name))

    # --- the keys -------------------------------------------------------------------

    def action_move(self, step: int) -> None:
        """``↑``/``↓``: over rows only, and the ends stay put."""
        if self.editing or self.collapsed:
            return
        target = min(max(self.selected + step, 0), len(self.rows) - 1)
        if target == self.selected:
            return
        previous, self.selected = self.selected, target
        self._draw_row(self.rows[previous])
        self._draw_row(self.rows[target])
        self._lines[self.rows[target].key].scroll_visible(animate=False)

    def action_cycle(self, step: int) -> None:
        """``←``/``→``: the next choice, wrapping. A text row has nothing to cycle."""
        if self.editing or self.collapsed:
            return
        row = self.rows[self.selected]
        if row.kind != "choice":
            return
        current = self.value(row.key)
        # A value the file holds that is not one of the choices (a provider typed by
        # hand) starts the cycle from the first choice rather than nowhere.
        index = row.values.index(current) if current in row.values else -step
        self._write(row.key, row.values[(index + step) % len(row.values)])

    async def action_activate(self) -> None:
        """``Enter``: the next choice on a choice row, the edit on a text row."""
        if self.editing or self.collapsed:
            return
        row = self.rows[self.selected]
        if row.kind == "choice":
            self.action_cycle(1)
        else:
            await self._begin_edit(row)

    def action_reset(self) -> None:
        """``Backspace``: the row's default, written like any other value."""
        if self.editing or self.collapsed:
            return
        row = self.rows[self.selected]
        self._write(row.key, row.default)

    def action_close(self) -> None:
        """``Esc``: cancel an open edit, or fold the panel and hand focus back."""
        if self.editing:
            self._end_edit()
            return
        if self.collapsed:
            return
        self._collapse()
        self.post_message(self.Closed())

    def on_blur(self, event: tevents.Blur) -> None:
        """Focus went elsewhere -- typing reached the bar, a finding was clicked --
        and the panel folds. Not when the edit took it: that is the panel's own
        doing, and not when it is already folded. ``Closed`` is not posted: focus
        is where the user just put it, and the app must not move it to the bar.

        Not, either, when the terminal window itself lost focus: an ``AppBlur``
        clears ``screen.focused`` and delivers this same ``Blur`` to whatever was
        focused, but nothing inside the app moved -- the panel is still what would
        take input back. ``app.app_focus`` is exactly that distinction (Textual
        flips it to ``False`` for the duration of an ``AppBlur``, ``True`` for an
        ordinary blur to another widget), so it, not ``screen.focused``, is what
        this checks: by the time this handler runs, ``screen.focused`` already
        reads ``None`` either way.
        """
        if self.editing or self.collapsed or not self.app.app_focus:
            return
        self._collapse()

    def on_descendant_blur(self, event: tevents.DescendantBlur) -> None:
        """The edit lost focus to something other than its own end (a click away):
        the edit is dropped unwritten and the panel folds, as it would without one."""
        if self.editing and event.widget is self._edit:
            self._end_edit(refocus=False)
            self._collapse()

    # --- the text edit --------------------------------------------------------------

    async def _begin_edit(self, row: Row) -> None:
        """Replace the row line with its name and an ``Input`` holding the value.

        ``editing`` goes up before the input takes focus, so the panel's own blur --
        focus is moving to its child -- is not read as a departure. The input does
        not select its text on focus: ``Enter`` on ``model`` is for adding to it as
        often as for replacing it, and the cursor sits at the end, as in the bar.
        """
        self.editing = True
        line = self._lines[row.key]
        label = PanelLine(self._marker_and_name(row))
        self._edit = Input(value=self.value(row.key), select_on_focus=False)
        self._edit_row = Horizontal(label, self._edit, classes="edit")
        line.display = False
        # Waited for: focus can only land on a widget that is in the DOM.
        await self._body.mount(self._edit_row, after=line)
        self._edit.focus()

    def _end_edit(self, *, refocus: bool = True) -> None:
        """Take the input away and show the row line again. Writes nothing."""
        self.editing = False
        if self._edit_row is not None:
            self._edit_row.remove()
        self._edit = None
        self._edit_row = None
        self._lines[self.rows[self.selected].key].display = True
        if refocus:
            self.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """``Enter`` in the edit: the value is written. Stopped here, or the app would
        read it as a line submitted from the bar."""
        event.stop()
        if not self.editing:
            return
        row = self.rows[self.selected]
        self._end_edit()
        self._write(row.key, event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        # The edit's keystrokes are not the bar's: the suggestion list has no say here.
        event.stop()

    # --- writing --------------------------------------------------------------------

    def _write(self, key: str, value: str) -> None:
        """One ``config_set``, then the file read back and every row redrawn from it.

        The two calls fail differently, so they are two ``try`` blocks rather than
        one. A refused ``config_set`` changes nothing on screen: the rows still show
        the file, which is what they showed before, and a ``ConfigError`` is a line
        under the panel rather than a value that looks written. A refused re-read is
        a different failure -- the file *did* change, ``config_set`` already returned
        the pairs it wrote -- so it would be wrong to report ``Failed`` over a write
        that succeeded. That still posts ``Written`` for what actually landed, plus a
        separate note that the panel could not read the result back; the rows keep
        showing the last view this panel did manage to read, stale until the next
        successful write or a fresh ``/config``.
        """
        try:
            pairs = config_set(key, value)
        except (ConfigError, JudgeError) as exc:
            self.post_message(self.Failed(str(exc)))
            return
        try:
            self._view = config_view()
        except (ConfigError, JudgeError) as exc:
            self.post_message(self.Written(pairs, self._view.path))
            self.post_message(
                self.Failed(f"{key} was written, but the file could not be read back: {exc}")
            )
            return
        self._draw_all()
        self.post_message(self.Written(pairs, self._view.path))

    # --- folding --------------------------------------------------------------------

    def _collapse(self) -> None:
        """One line -- ``config  install_browser=ask  network=allow ...`` -- and no
        frame: a record of what the file held when the panel closed."""
        self.collapsed = True
        self._body.display = False
        self._summary.set(self._draw_summary())
        self._summary.display = True
        self._fit()

    # --- drawing --------------------------------------------------------------------

    def _muted(self) -> str:
        return self._theme.tone("muted") if self._out.color else ""

    def _draw_all(self) -> None:
        self._head.set(self._draw_head())
        self._key.set(self._draw_key())
        for row in self.rows:
            self._draw_row(row)
        self._legend.set(self._draw_legend())

    def _draw_row(self, row: Row) -> None:
        self._lines[row.key].set(self._draw_row_text(row))

    def _draw_head(self) -> Text:
        """``config   <path>``, and whether the file exists yet."""
        line = Text(f" {'config':<{HEAD_WIDTH}}{self._view.path}")
        if not self._view.exists:
            line.append("  (not written yet, showing defaults)", style=self._muted())
        return line

    def _draw_key(self) -> Text:
        """``key      GROQ_API_KEY · found in <path>``: the variable and where a value
        for it was found. Never the value (``secrets.ApiKey`` keeps it out of any
        ``str``, and this line does not ask for it)."""
        env = self._view.config.judge.api_key_env
        line = Text(f" {'key':<{HEAD_WIDTH}}")
        if not env:
            line.append("not needed", style=self._muted())
            line.append(
                f" {self._theme.glyphs.sep} judge.api_key_env is empty", style=self._muted()
            )
            return line
        line.append(env)
        paths = default_dotenv_paths()
        found = resolve_api_key(env, dotenv_paths=paths)
        if found is None:
            dash, dots = ("—", "…") if self._theme.unicode else ("-", "...")
            where = f"not set {dash} put {env}={dots} in {paths[0]}"
        elif found.source == f"environment variable {env}":
            # ``resolve_api_key`` names the environment this way and a file by its path.
            where = "found in the environment"
        else:
            where = f"found in {found.source}"
        line.append(f" {self._theme.glyphs.sep} {where}", style=self._muted())
        return line

    def _marker_and_name(self, row: Row) -> Text:
        """The marker on the selected row, a space, then the name padded to the value
        column (``install_browser`` and its three spaces in the spec's mock)."""
        selected = self.rows[self.selected] is row
        marker = self._theme.glyphs.prompt if selected else " "
        line = Text(f"{marker} {row.name:<{NAME_WIDTH}}")
        if selected and self._out.color:
            line.stylize(self.accent, 0, 1)
        return line

    def _draw_row_text(self, row: Row) -> Text:
        """A choice row: every choice, the current one as a badge; a text row: the
        value, or ``(unset)``, with its note beside it."""
        line = self._marker_and_name(row)
        current = self.value(row.key)
        if row.kind == "choice":
            return self._choices(line, row, current)
        if current:
            line.append(current)
        else:
            line.append(UNSET, style=self._muted())
        if row.meaning:
            line.append(f" {self._theme.glyphs.sep} {row.meaning}", style=self._muted())
        return line

    def _choices(self, line: Text, row: Row, current: str) -> Text:
        """``ask   allow   deny``: the current choice as a badge in the accent when the
        theme draws badges, ``[ask]`` when it does not; the others muted.

        The words keep the columns of the bare text either way: a badge is drawn over
        the space each side of its word rather than adding cells, so `` ask `` reads
        as one cell in RICH and ``ask`` sits where it sits in PLAIN. The one exception
        is a badge on the last choice, which needs a cell after the word too.

        A value the file holds that is none of the row's choices -- TOML hand-edited
        between runs, e.g. ``judge.provider = "openrouter"`` -- is not one of
        ``row.values`` and so is not among the words above; every one of them is
        drawn muted, since none is current. It is instead drawn honestly at the row's
        end, in the same badge/``[value]`` convention, so the row never claims one of
        the listed choices is current when the file says otherwise.
        """
        badge = self._theme.badge and self._out.color
        unknown = current not in row.values
        words = [
            f"[{choice}]" if choice == current and not badge else choice for choice in row.values
        ]
        position = len(line)
        line.append(CHOICE_GAP.join(words))
        if badge and not unknown and current == row.values[-1]:
            line.append(" ")
        for choice, word in zip(row.values, words, strict=True):
            if choice != current:
                line.stylize(self._muted(), position, position + len(word))
            elif badge:
                line.stylize(
                    f"{BADGE_INK} on {self.accent}", position - 1, position + len(word) + 1
                )
            position += len(word) + len(CHOICE_GAP)
        if unknown:
            line.append(CHOICE_GAP)
            word = current if badge else f"[{current}]"
            position = len(line)
            line.append(word)
            if badge:
                line.append(" ")
                line.stylize(
                    f"{BADGE_INK} on {self.accent}", position - 1, position + len(word) + 1
                )
        return line

    def _draw_legend(self) -> Text:
        """The keys, in the theme's spelling: arrows in RICH, words in PLAIN."""
        rows, change = ("↑↓", "←→") if self._theme.unicode else ("up/down", "left/right")
        return Text(
            f" {rows} row   {change} change   enter edit text   backspace default   esc close",
            style=self._muted(),
        )

    def _draw_summary(self) -> Text:
        """The one-line record: the four settings a run decides by, as the file has them."""
        config = self._view.config
        parts = (
            f"install_browser={config.permissions.install_browser}",
            f"network={config.permissions.network}",
            f"respect_robots={_default(config.fetch.respect_robots)}",
            f"judge={config.judge.provider}",
        )
        return Text("  config  " + "  ".join(parts))
