"""Parser tests for the TUI's slash commands (spec section 13.1, section 13.3).

The parser is deliberately dumb: it splits a line into a verb and an argument and
decides whether the verb is still waiting for one. Validating the argument is the
app's job, not the parser's.
"""

from __future__ import annotations

import pytest

from proofpath.tui.commands import (
    ALLOW_ANSWERS,
    AWAITING_VERBS,
    DESCRIPTIONS,
    NEEDS_ARGUMENT,
    VERBS,
    Awaiting,
    Command,
    Unknown,
    complete,
    parse,
)

PARSE_TABLE = [
    # Empty and blank lines are not commands at all.
    ("", Unknown("")),
    ("   ", Unknown("")),
    ("\t\n", Unknown("")),
    # A bare line is a claim, a path or a URL: an implicit /check.
    ("~/Desktop/paper.pdf", Command("check", "~/Desktop/paper.pdf")),
    ("  ~/Desktop/paper.pdf  ", Command("check", "~/Desktop/paper.pdf")),
    ("the method yields a 40% speedup", Command("check", "the method yields a 40% speedup")),
    ("https://example.org/a b", Command("check", "https://example.org/a b")),
    # A verb that needs an argument and did not get one enters awaiting mode.
    ("/check", Awaiting("check", "paste a file path or URL")),
    ("  /check  ", Awaiting("check", "paste a file path or URL")),
    ("/resolve", Awaiting("resolve", "paste a reference")),
    ("/fetch", Awaiting("fetch", "paste a URL, DOI or arXiv id")),
    ("/allow", Awaiting("allow", "once | always | no | never")),
    ("/cancel", Awaiting("cancel", "#n")),
    # The same verbs with the argument on one line skip awaiting mode.
    ("/check ~/Desktop/paper.pdf", Command("check", "~/Desktop/paper.pdf")),
    ("/check   ~/Desktop/paper.pdf  ", Command("check", "~/Desktop/paper.pdf")),
    ("/resolve Zhang et al. 2021, Nature", Command("resolve", "Zhang et al. 2021, Nature")),
    ("/fetch 10.1038/s41586-021-03819-2", Command("fetch", "10.1038/s41586-021-03819-2")),
    # Verbs that are complete on their own.
    ("/config", Command("config", "")),
    ("/cache", Command("cache", "")),
    ("/help", Command("help", "")),
    ("/quit", Command("quit", "")),
    ("/summarize", Command("summarize", "")),
    ("/config set judge.provider gemini", Command("config", "set judge.provider gemini")),
    ("/cache ls", Command("cache", "ls")),
    ("/summarize #2", Command("summarize", "#2")),
    # The parser stays dumb: an answer it does not know is still the app's to reject.
    ("/allow once", Command("allow", "once")),
    ("/allow always", Command("allow", "always")),
    ("/allow no", Command("allow", "no")),
    ("/allow never", Command("allow", "never")),
    ("/allow maybe", Command("allow", "maybe")),
    # /cancel takes a run number with or without the "#" the log prints.
    ("/cancel 3", Command("cancel", "3")),
    ("/cancel #3", Command("cancel", "3")),
    ("/cancel  #12 ", Command("cancel", "12")),
    ("/cancel #", Awaiting("cancel", "#n")),
    # Unknown verbs, including case variants: verbs are lowercase and case-sensitive.
    ("/nope", Unknown("nope")),
    ("/nope with an argument", Unknown("nope")),
    ("/Check", Unknown("Check")),
    ("/CHECK ~/paper.pdf", Unknown("CHECK")),
    ("/", Unknown("")),
    ("/ check", Unknown("")),
    # An absolute POSIX path starts with "/" too, and its first segment is not a
    # verb: a path separator or a dot after that segment says it is a path to check.
    ("/Users/me/paper.pdf", Command("check", "/Users/me/paper.pdf")),
    ("/tmp/x.md", Command("check", "/tmp/x.md")),
    ("/home/me/my draft.md", Command("check", "/home/me/my draft.md")),
    ("/paper.pdf", Command("check", "/paper.pdf")),
    ("/srv/", Command("check", "/srv/")),
    # A genuinely unknown verb stays unknown: no separator, no dot.
    ("/frobnicate", Unknown("frobnicate")),
    ("/frobnicate now", Unknown("frobnicate")),
    # The first token is the whole segment: "check/paper.pdf" is not a verb.
    ("/check/paper.pdf", Command("check", "/check/paper.pdf")),
    # Non-slash paths are untouched by the rule.
    ("~/x", Command("check", "~/x")),
    ("docs/draft.md", Command("check", "docs/draft.md")),
    ("C:\\papers\\draft.pdf", Command("check", "C:\\papers\\draft.pdf")),
]


@pytest.mark.parametrize(
    ("line", "expected"), PARSE_TABLE, ids=[repr(row[0]) for row in PARSE_TABLE]
)
def test_parse(line, expected):
    assert parse(line) == expected


def test_verbs_mirror_the_cli_verbs():
    # Spec section 13.3's TUI mirror rule: every CLI verb exists as a slash command.
    for verb in ("check", "resolve", "fetch", "config", "cache"):
        assert verb in VERBS


def test_needs_argument_is_a_subset_of_verbs():
    assert set(NEEDS_ARGUMENT) <= set(VERBS)


def test_allow_answers_are_the_four_permission_replies():
    assert ALLOW_ANSWERS == ("once", "always", "no", "never")


def test_every_verb_without_an_argument_is_awaiting_or_complete():
    for verb in VERBS:
        parsed = parse(f"/{verb}")
        if verb in NEEDS_ARGUMENT:
            assert parsed == Awaiting(verb, NEEDS_ARGUMENT[verb])
        else:
            assert parsed == Command(verb, "")


def test_every_verb_with_an_argument_is_a_command():
    for verb in VERBS:
        assert parse(f"/{verb} x") == Command(verb, "x")


def test_results_are_frozen():
    parsed = parse("/check ~/paper.pdf")
    with pytest.raises(AttributeError):
        parsed.verb = "quit"  # type: ignore[misc]


# --- task 8.4: which verbs hold the bar is the parser's decision ---------------------


def test_awaiting_verbs_are_the_ones_that_open_a_run() -> None:
    """Every awaiting verb wants an argument; not every verb that does holds the bar."""
    assert set(AWAITING_VERBS) <= set(NEEDS_ARGUMENT)
    assert set(AWAITING_VERBS) == {"check", "resolve", "fetch"}
    # The two answers say what they want instead: a pasted path after them is a check.
    assert "cancel" not in AWAITING_VERBS
    assert "allow" not in AWAITING_VERBS


# --- completion ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "run_ids", "expected"),
    [
        ("/ch", (), ["/check "]),
        # VERBS order, filtered by prefix; only the verbs that take an argument get
        # the trailing space, and NEEDS_ARGUMENT is the source of truth for which.
        ("/c", (), ["/check ", "/config", "/cache", "/cancel "]),
        (
            "/",
            (),
            [f"/{verb} " if verb in NEEDS_ARGUMENT else f"/{verb}" for verb in VERBS],
        ),
        ("/help", (), ["/help"]),
        ("/allow ", (), ["/allow once", "/allow always", "/allow no", "/allow never"]),
        ("/allow n", (), ["/allow no", "/allow never"]),
        ("/cancel ", (2, 3), ["/cancel #2", "/cancel #3"]),
        ("/cancel #3", (2, 3), ["/cancel #3"]),
        ("/cancel ", (), []),
        ("/check ", (), []),  # a target is not completed
        ("paper.pdf", (), []),  # not a command
        ("", (), []),
        ("/zz", (), []),
    ],
)
def test_complete(text: str, run_ids: tuple[int, ...], expected: list[str]) -> None:
    assert complete(text, run_ids=run_ids) == expected


# --- the descriptions behind the suggestion list (wordmark design section 10) --------


def test_every_verb_has_a_short_description_and_nothing_else_does() -> None:
    """One line per verb, short enough for a row, and no verb the list cannot name."""
    assert set(DESCRIPTIONS) == set(VERBS)
    for verb, text in DESCRIPTIONS.items():
        assert 0 < len(text) <= 40 and not text.endswith("."), verb


def test_config_is_described_as_the_panel_first() -> None:
    """``/config`` alone opens the settings panel (wordmark design section 12); the
    scriptable forms come after it in the one line the list has."""
    assert DESCRIPTIONS["config"] == "settings panel, or show/set/check"
