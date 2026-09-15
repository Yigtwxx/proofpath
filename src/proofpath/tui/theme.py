"""The two TUI themes and how one is chosen (TUI v2 design, section 2).

``RICH`` draws box borders, Unicode glyphs and truecolor tones; ``PLAIN`` is pure
ASCII with the ANSI-16 names ``ui.py`` has always used, so Windows conhost and
``NO_COLOR`` keep today's output exactly. Both themes carry the same four meanings
(ok, caution, finding, muted) plus one accent per run: a theme changes how a
meaning looks, never what a word means. The meaning tables stay in ``ui.py``;
this module only maps a meaning to a tone.

Detection reads an injected environment, never ``os.environ`` directly, so every
branch of the table is a unit test. ``PROOFPATH_THEME=rich|plain`` overrides it,
for screenshots and bug reports.

Pure: no Textual import, so ``pet.py`` and the tests can use it without an app.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from proofpath import ui

#: What a colour says. ``accent`` is the per-run hue; the other four are ``ui.py``'s
#: green / yellow / red / dim, renamed by meaning rather than by colour.
Meaning = Literal["ok", "caution", "finding", "muted", "accent"]
#: The ``ui`` colour name each meaning is spelled with in :attr:`Theme.tones`.
MEANING_TONES: Mapping[str, str] = {
    "ok": "green",
    "caution": "yellow",
    "finding": "red",
    "muted": "dim",
}

#: The override, checked before anything else. Only these two values count.
THEME_ENV = "PROOFPATH_THEME"
#: ``COLORTERM`` values that announce 24-bit colour.
TRUECOLOR_VALUES = frozenset({"truecolor", "24bit"})
#: Terminals that draw the RICH glyphs even without ``COLORTERM``; Terminal.app is
#: 256-colour, and Rich downgrades the tones faithfully there.
RICH_PROGRAMS = frozenset({"iTerm.app", "WezTerm", "vscode", "ghostty", "Apple_Terminal"})
#: Variables whose mere presence identifies a truecolor terminal.
TRUECOLOR_MARKERS = ("KITTY_WINDOW_ID", "ALACRITTY_LOG", "WT_SESSION")
#: Locales whose terminals commonly render "ambiguous-width" characters -- the box
#: drawing set among them -- two cells wide, which shears every RICH border.
CJK_LOCALES = ("zh", "ja", "ko")


@dataclass(frozen=True)
class Glyphs:
    """Every symbol the widgets draw, so no widget spells a glyph itself."""

    stage_pending: str
    stage_active: str
    stage_done: str
    #: A finished stage that ended with an honesty count keeps the active mark.
    stage_flag: str
    error: str
    warning: str
    note: str
    bar_full: str
    bar_empty: str
    cov_full: str
    cov_abstract: str
    cov_unverified: str
    rule: str
    copy: str
    prompt: str
    sep: str
    #: Textual border style name for a run's panel.
    box: Literal["round", "none"]


@dataclass(frozen=True)
class Theme:
    """One of the two looks. Instances are the module constants, never built ad hoc."""

    name: Literal["rich", "plain"]
    glyphs: Glyphs
    #: ``ui`` colour name (``green`` / ``yellow`` / ``red`` / ``dim``) to a Rich style.
    tones: Mapping[str, str]
    #: The five run accents, rotating. Never red, yellow or green: those mean things.
    accents: tuple[str, ...]
    #: Style of the banner's ``[PROOF]`` stamp, its only coloured element.
    stamp: str
    #: State words as badges (coloured background) or as coloured words.
    badge: bool
    unicode: bool
    #: The bare colour behind each ``ui`` name (no ``bold``), for a badge's background
    #: and for anything Textual has to parse as a colour rather than a style.
    colours: Mapping[str, str]
    #: The text colour that reads on each badge background. Empty when ``badge`` is off.
    ink: Mapping[str, str]

    def style_for(self, word: str) -> str:
        """The tone of a state word by meaning; ``""`` for an unknown word.

        ``ui.state_colour`` supplies the meaning, so the tables live in one place;
        this only swaps the ANSI name for the theme's spelling of it.
        """
        name = ui.state_colour(word)
        return self.tones[name] if name else ""

    def tone(self, meaning: Meaning) -> str:
        """The style of a meaning (``ok``, ``caution``, ``finding``, ``muted``).

        For a widget that colours something that is not a state word -- a stage's
        tick, a run's ``done`` on its border -- and so has no ``ui`` table to ask.
        ``accent`` has no single tone; ask :meth:`accent` with the run's index.
        """
        return self.tones[MEANING_TONES[meaning]]

    def badge_style(self, word: str) -> str:
        """A state word as a badge: its tone as the background, ink on top.

        ``""`` when this theme draws no badges or the word carries no meaning, so a
        caller can fall back to :meth:`style_for` in one expression.
        """
        name = ui.state_colour(word)
        if not self.badge or not name:
            return ""
        return f"{self.ink[name]} on {self.colours[name]}"

    def accent(self, index: int) -> str:
        """The accent of the ``index``-th run (zero-based), wrapping every five."""
        return self.accents[index % len(self.accents)]


_RICH_RED = "#ef4444"
_RICH_COLOURS: Mapping[str, str] = {
    "green": "#22c55e",
    "yellow": "#f59e0b",
    "red": _RICH_RED,
    "dim": "#71717a",
}
_RICH_TONES: Mapping[str, str] = {
    "green": f"bold {_RICH_COLOURS['green']}",
    "yellow": f"bold {_RICH_COLOURS['yellow']}",
    "red": f"bold {_RICH_RED}",
    "dim": _RICH_COLOURS["dim"],
}
#: What a badge's word is written in: dark on the two light tones, light on the two
#: dark ones, so every state word reads at the same weight on its own background.
_RICH_INK: Mapping[str, str] = {
    "green": "black",
    "yellow": "black",
    "red": "white",
    "dim": "white",
}
#: Five hues that stay clear of the meaning colours: cyan, violet, blue, fuchsia, teal.
_RICH_ACCENTS = ("#22d3ee", "#a78bfa", "#60a5fa", "#e879f9", "#2dd4bf")

RICH = Theme(
    name="rich",
    glyphs=Glyphs(
        stage_pending="·",
        stage_active="⏺",
        stage_done="✓",
        stage_flag="⏺",
        error="✗",
        warning="⚠",
        note="·",
        bar_full="▰",
        bar_empty="▱",
        cov_full="█",
        cov_abstract="▓",
        cov_unverified="░",
        rule="─",
        copy="⧉",
        prompt="›",  # noqa: RUF001 - the design names this glyph; not a ">"
        sep="·",
        box="round",
    ),
    tones=_RICH_TONES,
    accents=_RICH_ACCENTS,
    stamp=f"bold {_RICH_RED}",
    badge=True,
    unicode=True,
    colours=_RICH_COLOURS,
    ink=_RICH_INK,
)

PLAIN = Theme(
    name="plain",
    glyphs=Glyphs(
        stage_pending="-",
        stage_active="*",
        stage_done="+",
        stage_flag="*",
        error="x",
        warning="!",
        note="-",
        bar_full="#",
        bar_empty="-",
        cov_full="#",
        cov_abstract="=",
        cov_unverified=".",
        rule="-",
        copy="[copy]",
        prompt=">",
        sep=",",
        box="none",
    ),
    # The ANSI names map to themselves: PLAIN is exactly what ``ui.py`` prints today.
    tones={name: name for name in MEANING_TONES.values()},
    accents=ui.ACCENTS,
    stamp=ui.STAMP_COLOUR,
    badge=False,
    unicode=False,
    colours={name: name for name in MEANING_TONES.values()},
    ink={},
)


def detect(
    env: Mapping[str, str],
    *,
    no_color: bool = False,
    quiet: bool = False,
    platform: str = sys.platform,
) -> Theme:
    """Pick the theme for this terminal; first matching rule wins.

    1. ``PROOFPATH_THEME`` is ``rich`` or ``plain`` (anything else is ignored).
    2. ``PLAIN`` when colour is off: ``--no-color``, ``-q``, ``NO_COLOR`` set (its
       value does not matter, per the convention) or ``TERM=dumb``.
    3. ``PLAIN`` on Windows without ``WT_SESSION``: legacy conhost cannot draw the
       glyphs, whatever ``COLORTERM`` claims.
    4. ``PLAIN`` under a CJK locale (``LC_ALL``, ``LC_CTYPE`` or ``LANG`` starting
       ``zh``, ``ja`` or ``ko``): those terminals commonly draw box drawing two cells wide, which
       shears every border. ``PROOFPATH_THEME=rich`` is the way to say yours does not.
    5. ``RICH`` only on evidence of truecolor; ``PLAIN`` when there is none, so a
       terminal we cannot read gets the output that works everywhere.
    """
    override = env.get(THEME_ENV, "").strip().lower()
    if override == "rich":
        return RICH
    if override == "plain":
        return PLAIN
    if no_color or quiet or "NO_COLOR" in env or env.get("TERM") == "dumb":
        return PLAIN
    if platform.startswith("win") and "WT_SESSION" not in env:
        return PLAIN
    if _locale(env).startswith(CJK_LOCALES):
        return PLAIN
    if (
        env.get("COLORTERM", "").lower() in TRUECOLOR_VALUES
        or env.get("TERM_PROGRAM") in RICH_PROGRAMS
        or any(marker in env for marker in TRUECOLOR_MARKERS)
    ):
        return RICH
    return PLAIN


def _locale(env: Mapping[str, str]) -> str:
    """The character locale: ``LC_ALL`` over ``LC_CTYPE`` over ``LANG``, as POSIX says."""
    return (env.get("LC_ALL") or env.get("LC_CTYPE") or env.get("LANG") or "").strip().lower()
