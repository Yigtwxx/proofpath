from __future__ import annotations

import io
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from rich.text import Text

from proofpath import ui


@pytest.fixture(autouse=True)
def plain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI runners sometimes export these; the tests decide colour themselves.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def _build(
    *,
    no_color: bool = False,
    quiet: bool = False,
    force_terminal: bool | None = None,
) -> tuple[ui.Ui, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    instance = ui.build(
        stdout=out, stderr=err, no_color=no_color, quiet=quiet, force_terminal=force_terminal
    )
    return instance, out, err


# --- build ------------------------------------------------------------------------


def test_build_turns_colour_off_when_stdout_is_not_a_terminal() -> None:
    instance, _, _ = _build()
    assert instance.color is False


def test_build_turns_colour_on_for_a_forced_terminal() -> None:
    instance, _, _ = _build(force_terminal=True)
    assert instance.color is True


def test_build_honours_no_color_flag_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, _, _ = _build(force_terminal=True, no_color=True)
    assert instance.color is False
    monkeypatch.setenv("NO_COLOR", "1")
    instance, _, _ = _build(force_terminal=True)
    assert instance.color is False


# --- kv / style_state ---------------------------------------------------------------


def test_kv_pads_the_key_to_ten_columns_plus_one_space() -> None:
    instance, out, _ = _build()
    ui.kv(instance, "state", "RESOLVED")
    assert out.getvalue() == "state      RESOLVED\n"


def test_kv_does_not_wrap_long_values() -> None:
    instance, out, _ = _build()
    ui.kv(instance, "record", "x" * 200)
    assert out.getvalue() == "record     " + "x" * 200 + "\n"


def test_kv_with_state_colours_the_value_on_a_terminal() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.kv(instance, "state", "RESOLVED", state=True)
    assert "\x1b[" in out.getvalue()
    assert "RESOLVED" in out.getvalue()


def test_kv_emits_plain_text_without_colour() -> None:
    instance, out, _ = _build(force_terminal=True, no_color=True)
    ui.kv(instance, "state", "GHOST REFERENCE", state=True)
    assert out.getvalue() == "state      GHOST REFERENCE\n"


@pytest.mark.parametrize(
    ("word", "style"),
    [
        ("ok", "green"),
        ("RESOLVED", "green"),
        ("SUPPORTED", "green"),
        ("not retracted", "green"),
        ("AMBIGUOUS", "yellow"),
        ("RESOLVED (low confidence)", "yellow"),
        ("UNVERIFIED (blocked)", "yellow"),  # prefix match
        ("LOW CONFIDENCE (abstract only)", "yellow"),  # prefix match
        ("GHOST REFERENCE", "red"),
        ("REFUTED", "red"),
        ("RETRACTED", "red"),
        ("FAILED", "red"),
        ("NEI", "dim"),
        ("none", "dim"),
        ("—", "dim"),
    ],
)
def test_style_state_picks_the_meaning_colour(word: str, style: str) -> None:
    instance, _, _ = _build(force_terminal=True)
    text = ui.style_state(instance, word)
    assert isinstance(text, Text)
    assert str(text) == word
    assert [span.style for span in text.spans] == [style]


def test_style_state_leaves_unknown_words_and_colourless_output_unstyled() -> None:
    instance, _, _ = _build(force_terminal=True)
    assert ui.style_state(instance, "something else").spans == []
    plain, _, _ = _build()
    assert ui.style_state(plain, "RESOLVED").spans == []


# --- quiet ------------------------------------------------------------------------


def test_quiet_suppresses_note_stage_blank_and_rule_but_not_hint_or_kv() -> None:
    instance, out, _ = _build(quiet=True)
    ui.note(instance, "a fact")
    ui.stage(instance, "resolving")
    ui.blank(instance)
    ui.rule(instance)
    assert out.getvalue() == ""
    ui.hint(instance, "try --allow-browser")
    ui.kv(instance, "state", "RESOLVED")
    assert out.getvalue() == "hint       try --allow-browser\nstate      RESOLVED\n"


def test_hint_can_go_to_stderr_so_json_stdout_stays_one_document() -> None:
    instance, out, err = _build(quiet=True)
    ui.hint(instance, "--show is ignored under --format json", err=True)
    assert out.getvalue() == ""
    assert err.getvalue() == "hint       --show is ignored under --format json\n"


def test_note_stage_blank_and_rule_print_when_not_quiet() -> None:
    instance, out, _ = _build()
    ui.note(instance, "a fact")
    ui.stage(instance, "resolving")
    ui.blank(instance)
    ui.rule(instance)
    lines = out.getvalue().splitlines()
    assert lines[0] == "note       a fact"
    assert lines[1] == "resolving"
    assert lines[2] == ""
    assert lines[3] == "─" * 60


# --- error / json -----------------------------------------------------------------


def test_error_goes_to_stderr_with_a_prefix() -> None:
    instance, out, err = _build()
    ui.error(instance, "config is broken")
    assert out.getvalue() == ""
    assert err.getvalue() == "error: config is broken\n"


class Kind(str, Enum):
    A = "alpha"


@dataclass(frozen=True)
class Inner:
    kind: Kind
    blob: bytes
    where: Path


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner
    items: list[Inner]
    missing: Inner | None = None


def test_emit_json_round_trips_nested_dataclasses_on_stdout_only() -> None:
    instance, out, err = _build()
    inner = Inner(Kind.A, b"\x00\x01", Path("/tmp/x"))
    ui.emit_json(instance, {"result": Outer("n", inner, [inner]), "retraction": None})
    assert err.getvalue() == ""
    payload = json.loads(out.getvalue())
    assert payload["retraction"] is None
    result = payload["result"]
    assert result["name"] == "n"
    assert result["missing"] is None
    assert result["inner"] == {"kind": "alpha", "blob": None, "where": "/tmp/x"}
    assert result["items"] == [result["inner"]]
    assert out.getvalue().startswith("{\n  ")  # indented, one document


def test_kv_bold_key_does_not_bleed_into_the_value() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.kv(instance, "state", "RESOLVED", state=True)
    assert "\x1b[1mstate      \x1b[0m\x1b[32mRESOLVED\x1b[0m" in out.getvalue()


def test_rule_never_uses_a_meaning_colour_on_a_terminal() -> None:
    instance, out, err = _build(force_terminal=True)
    ui.rule(instance)
    ui.error(instance, "boom")
    assert "\x1b[32m" not in out.getvalue() and "\x1b[92m" not in out.getvalue()
    assert err.getvalue().startswith("\x1b[31merror: \x1b[0mboom")


# --- state_line ---------------------------------------------------------------------


def test_state_line_colours_the_word_but_not_the_rest() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.state_line(instance, "status", "ok", "123 ms")
    assert out.getvalue() == "\x1b[1mstatus     \x1b[0m\x1b[32mok\x1b[0m 123 ms\n"


def test_state_line_without_rest_colours_only_the_word() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.state_line(instance, "status", "FAILED")
    assert out.getvalue() == "\x1b[1mstatus     \x1b[0m\x1b[31mFAILED\x1b[0m\n"


def test_state_line_plain_without_colour() -> None:
    instance, out, _ = _build()
    ui.state_line(instance, "status", "FAILED", "401")
    assert out.getvalue() == "status     FAILED 401\n"


def test_state_line_prefix_stays_plain_and_the_word_after_it_is_styled() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.state_line(instance, "evidence", "LOW CONFIDENCE (abstract only)", prefix="abstract — ")
    assert out.getvalue() == (
        "\x1b[1mevidence   \x1b[0mabstract — \x1b[33mLOW CONFIDENCE (abstract only)\x1b[0m\n"
    )


def test_state_line_prefix_plain_without_colour() -> None:
    instance, out, _ = _build()
    ui.state_line(instance, "evidence", "UNVERIFIED (unreachable)", prefix="none — ")
    assert out.getvalue() == "evidence   none — UNVERIFIED (unreachable)\n"
