"""Slash commands, and the awaiting mode a bare verb enters (spec section 13.1).

Spec section 13.3's mirror rule: every CLI verb exists in the TUI as the
same-named slash command, so ``/check``, ``/resolve``, ``/fetch``, ``/config`` and
``/cache`` are the CLI's verbs and the rest are TUI-only.

The parser is deliberately dumb. It splits a line into a verb and the rest and
says whether the verb is still waiting for an argument; it never judges the
argument. ``/allow maybe`` parses cleanly and the app rejects it, because the
parser has no idea which permission is being answered and should not guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERBS = (
    "check",
    "resolve",
    "fetch",
    "config",
    "cache",
    "allow",
    "summarize",
    "cancel",
    "help",
    "quit",
)
#: Verbs that open the awaiting mode when typed alone, with the bar's placeholder.
NEEDS_ARGUMENT = {
    "check": "paste a file path or URL",
    "resolve": "paste a reference",
    "fetch": "paste a URL, DOI or arXiv id",
    "allow": "once | always | no | never",
    "cancel": "#n",
}
#: The verbs the awaiting mode actually holds the bar for: the three spec section
#: 13.1 names, and exactly the ones that open a run. ``/cancel`` and ``/allow`` also
#: want an argument, but they are answers rather than targets -- capturing the next
#: line for them would read a pasted path as ``/cancel ~/paper.pdf``. They say what
#: they want and let the next line be the command it looks like, which is why this
#: is narrower than :data:`NEEDS_ARGUMENT` rather than equal to it.
AWAITING_VERBS = ("check", "resolve", "fetch")
ALLOW_ANSWERS = ("once", "always", "no", "never")
#: The implicit verb: a line that is not a slash command is something to check.
DEFAULT_VERB = "check"
#: Splits the verb off at the first run of whitespace, whatever the terminal sent.
_VERB_SPLIT = re.compile(r"\s")
#: What tells a pasted absolute path (``/Users/me/paper.pdf``, ``/tmp/x.md``) apart
#: from a mistyped verb (``/frobnicate``): a verb has neither a separator nor a dot.
_PATH_HINTS = ("/", "\\", ".")


@dataclass(frozen=True)
class Command:
    """A verb with everything it needs. ``arg`` is ``""`` for verbs that take none."""

    verb: str
    arg: str


@dataclass(frozen=True)
class Awaiting:
    """A verb typed without its argument: tint the bar, show ``placeholder``, wait."""

    verb: str
    placeholder: str


@dataclass(frozen=True)
class Unknown:
    """Neither a known verb nor anything to run. ``verb`` is ``""`` for a blank line."""

    verb: str


Parsed = Command | Awaiting | Unknown


def parse(line: str) -> Parsed:
    """Read one submitted input line.

    Verbs are lowercase and case-sensitive, so ``/Check`` is unknown rather than
    silently corrected — a typo that starts a run is worse than one that does not.
    An absolute path (``/Users/me/paper.pdf``) is the one exception: its first
    segment is no verb and carries a separator or a dot, so it is an implicit
    ``/check`` of the whole line rather than an unknown verb.
    """
    text = line.strip()
    if not text:
        return Unknown("")
    if not text.startswith("/"):
        return Command(DEFAULT_VERB, text)

    verb, rest = _split_verb(text[1:])
    arg = rest.strip()
    if verb not in VERBS:
        # An absolute POSIX path starts with "/" like a command does. When the first
        # segment is no verb and looks like part of a path, the line is the path.
        if any(hint in verb for hint in _PATH_HINTS):
            return Command(DEFAULT_VERB, text)
        return Unknown(verb)
    if not arg:
        placeholder = NEEDS_ARGUMENT.get(verb)
        return Awaiting(verb, placeholder) if placeholder is not None else Command(verb, "")
    if verb == "cancel":
        # The log prints runs as "#1", "#2"; typing the "#" back must not break it,
        # and a lone "#" is still a missing argument.
        arg = arg.removeprefix("#").strip()
        if not arg:
            return Awaiting(verb, NEEDS_ARGUMENT[verb])
    return Command(verb, arg)


def _split_verb(body: str) -> tuple[str, str]:
    """Split ``body`` at its first whitespace character, tab or space alike."""
    match = _VERB_SPLIT.search(body)
    return (body, "") if match is None else (body[: match.start()], body[match.end() :])
