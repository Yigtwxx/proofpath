from __future__ import annotations

import io
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from rich.text import Text

from proofpath import ui
from proofpath.report import STATE_WORDS, Diagnostic, Footer


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
    inner = Inner(Kind.A, b"\x00\x01", Path("tmp") / "x")
    ui.emit_json(instance, {"result": Outer("n", inner, [inner]), "retraction": None})
    assert err.getvalue() == ""
    payload = json.loads(out.getvalue())
    assert payload["retraction"] is None
    result = payload["result"]
    assert result["name"] == "n"
    assert result["missing"] is None
    assert result["inner"] == {"kind": "alpha", "blob": None, "where": "tmp/x"}
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


# --- diagnostics ---------------------------------------------------------------------


def _diagnostic(**overrides: object) -> Diagnostic:
    fields: dict[str, object] = {
        "level": "error",
        "code": "ghost-reference",
        "title": "cited source does not exist",
        "location": "paper.pdf:4:112",
        "lines": (
            "   |",
            "   | [12] Zhang, K. et al. (2021). Neural cascade alignment",
            "   |",
            "   = note: no author, title or year agreement with any candidate",
        ),
        "state": "GHOST REFERENCE",
    }
    fields.update(overrides)
    return Diagnostic(**fields)  # type: ignore[arg-type]


def test_diagnostic_colours_the_level_word_and_nothing_else() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.diagnostic(instance, _diagnostic())
    lines = out.getvalue().splitlines()
    assert lines[0] == "\x1b[31merror\x1b[0m[ghost-reference]: cited source does not exist"
    assert lines[1] == "  --> paper.pdf:4:112"
    assert "\x1b[" not in "".join(lines[2:])


def test_diagnostic_colours_the_state_word_when_the_title_carries_it() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.diagnostic(
        instance,
        _diagnostic(
            level="note",
            code="nei",
            title="NEI - source neither supports nor contradicts",
            state="NEI",
            lines=(),
        ),
    )
    assert out.getvalue().splitlines()[0] == (
        "\x1b[2mnote\x1b[0m[nei]: \x1b[2mNEI\x1b[0m - source neither supports nor contradicts"
    )


@pytest.mark.parametrize(
    ("level", "escape"), [("error", "\x1b[31m"), ("warning", "\x1b[33m"), ("note", "\x1b[2m")]
)
def test_diagnostic_level_words_use_the_three_meaning_colours(level: str, escape: str) -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.diagnostic(instance, _diagnostic(level=level, lines=()))
    assert out.getvalue().startswith(f"{escape}{level}\x1b[0m[")


def test_diagnostic_is_plain_without_a_terminal() -> None:
    instance, out, _ = _build()
    ui.diagnostic(instance, _diagnostic())
    assert out.getvalue() == (
        "error[ghost-reference]: cited source does not exist\n"
        "  --> paper.pdf:4:112\n"
        "   |\n"
        "   | [12] Zhang, K. et al. (2021). Neural cascade alignment\n"
        "   |\n"
        "   = note: no author, title or year agreement with any candidate\n"
    )


def test_diagnostic_survives_quiet() -> None:
    # Product rule 6: a quiet run is never a silent one.
    instance, out, _ = _build(quiet=True)
    ui.diagnostic(instance, _diagnostic())
    assert out.getvalue().startswith("error[ghost-reference]:")


def test_diagnostic_indents_every_line_of_a_group_member() -> None:
    instance, out, _ = _build()
    ui.diagnostic(instance, _diagnostic(indent=2))
    for line in out.getvalue().splitlines():
        assert line.startswith("  ")


def test_diagnostic_never_interprets_brackets_as_markup() -> None:
    instance, out, _ = _build()
    ui.diagnostic(instance, _diagnostic(title="[bold]not markup[/bold]", lines=()))
    assert "[bold]not markup[/bold]" in out.getvalue()


# --- coverage -------------------------------------------------------------------------


def test_coverage_prints_the_three_keyed_lines() -> None:
    instance, out, _ = _build()
    ui.coverage(instance, 62, 21, 17, weak=False)
    assert out.getvalue() == "fulltext   62%\nabstract   21%\nunverified 17%\n"


def test_coverage_colours_full_text_green_abstract_yellow_unverified_red() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.coverage(instance, 62, 21, 17, weak=False)
    lines = out.getvalue().splitlines()
    assert lines[0].endswith("\x1b[32m62%\x1b[0m")
    assert lines[1].endswith("\x1b[33m21%\x1b[0m")
    assert lines[2].endswith("\x1b[31m17%\x1b[0m")


def test_coverage_is_never_suppressed_by_quiet() -> None:
    # Product rule 6: a low-coverage run must not look like a clean one.
    instance, out, _ = _build(quiet=True)
    ui.coverage(instance, 0, 0, 100, weak=True)
    assert out.getvalue().splitlines() == [
        "fulltext   0%",
        "abstract   0%",
        "unverified 100%",
        f"hint       {ui.WEAK_COVERAGE}",
    ]


def test_coverage_says_nothing_extra_when_it_holds() -> None:
    instance, out, _ = _build()
    ui.coverage(instance, 80, 10, 10, weak=False)
    assert "hint" not in out.getvalue()


# --- stage rows and the footer ---------------------------------------------------------


def test_stage_row_lays_out_the_spec_columns() -> None:
    instance, out, _ = _build()
    ui.stage_row(instance, "Parsing", "paper.pdf", "24 pages, 42 refs", 1.2)
    ui.stage_row(instance, "Retractions", "Retraction Watch", "1 retracted", 0.8)
    ui.stage_row(instance, "Verifying", "118 claims on mps", "", 21.4)
    assert out.getvalue().splitlines() == [
        "  Parsing      paper.pdf                         24 pages, 42 refs      1.2s",
        "  Retractions  Retraction Watch                  1 retracted            0.8s",
        "  Verifying    118 claims on mps                                       21.4s",
    ]


def test_stage_row_keeps_the_elapsed_column_when_a_field_overflows() -> None:
    instance, out, _ = _build()
    ui.stage_row(instance, "Fetching", "22 full text, 11 abstract, 9 blocked", "", 14.7)
    assert out.getvalue().rstrip("\n") == (
        "  Fetching     22 full text, 11 abstract, 9 blocked                    14.7s"
    )


def test_stage_row_is_suppressed_by_quiet() -> None:
    instance, out, _ = _build(quiet=True)
    ui.stage_row(instance, "Parsing", "paper.pdf", "24 pages, 42 refs", 1.2)
    assert out.getvalue() == ""


def test_footer_prints_counts_coverage_and_the_run_line() -> None:
    instance, out, _ = _build()
    ui.footer(
        instance,
        Footer(
            counts="42 refs: 3 ghost, 1 retracted, 6 unsupported, 32 ok",
            coverage=(62, 21, 17),
            api_calls=0,
            elapsed=38.4,
            written="report.md",
            weak=False,
            cancelled=False,
        ),
    )
    assert out.getvalue().splitlines() == [
        "42 refs: 3 ghost, 1 retracted, 6 unsupported, 32 ok",
        "fulltext   62%",
        "abstract   21%",
        "unverified 17%",
        "report.md written  ·  0 API calls  ·  38.4s",
    ]


def test_footer_survives_quiet_and_says_when_nothing_was_written() -> None:
    instance, out, _ = _build(quiet=True)
    ui.footer(
        instance,
        Footer(
            counts="4 refs: 1 ghost, 3 ok",
            coverage=(75, 0, 25),
            api_calls=2,
            elapsed=1.0,
            written=None,
            weak=True,
            cancelled=False,
        ),
    )
    lines = out.getvalue().splitlines()
    assert lines[0] == "4 refs: 1 ghost, 3 ok"
    assert lines[-1] == "no report written  ·  2 API calls  ·  1.0s"
    assert f"hint       {ui.WEAK_COVERAGE}" in lines


def test_footer_prints_the_tier_note_as_a_hint() -> None:
    instance, out, _ = _build()
    ui.footer(
        instance,
        Footer(
            counts="4 refs: 4 ok",
            coverage=(100, 0, 0),
            api_calls=0,
            elapsed=1.0,
            written=None,
            weak=False,
            cancelled=False,
            note="medium is the strongest confidence shown",
        ),
    )
    assert "hint       medium is the strongest confidence shown" in out.getvalue().splitlines()


def test_footer_announces_a_cancelled_run_in_dim() -> None:
    instance, out, _ = _build(force_terminal=True)
    ui.footer(
        instance,
        Footer(
            counts="4 refs: 4 ok",
            coverage=(100, 0, 0),
            api_calls=0,
            elapsed=1.0,
            written=None,
            weak=False,
            cancelled=True,
        ),
    )
    assert out.getvalue().splitlines()[0] == "\x1b[1mrun        \x1b[0m\x1b[2mcancelled\x1b[0m"


# --- the colour tables grow with the states the report can carry -------------------------


@pytest.mark.parametrize(
    ("word", "style"),
    [
        ("NOT SUPPORTED", "red"),
        ("PARSE ERROR", "red"),
        ("PARAGRAPH-SCOPED", "yellow"),
        ("UNSUPPORTED CITATION STYLE", "yellow"),
        ("UNRESOLVED MARKER", "yellow"),
        ("cancelled", "dim"),
    ],
)
def test_style_state_knows_every_report_state_word(word: str, style: str) -> None:
    instance, _, _ = _build(force_terminal=True)
    assert [span.style for span in ui.style_state(instance, word).spans] == [style]


def test_every_state_word_a_finding_can_carry_has_a_colour() -> None:
    instance, _, _ = _build(force_terminal=True)
    for word in STATE_WORDS.values():
        assert ui.style_state(instance, word).spans, word


# --- a document with citations but no bibliography (product rule 6) --------------------


def test_coverage_says_no_bibliography_was_found() -> None:
    # 0/0/0 over a document full of markers must not read as a clean run.
    instance, out, _ = _build(quiet=True)
    ui.coverage(instance, 0, 0, 0, weak=False, unchecked_markers=84)
    assert out.getvalue().splitlines() == [
        "fulltext   0%",
        "abstract   0%",
        "unverified 0%",
        "hint       no bibliography was found; 84 citation markers could not be checked",
    ]


def test_coverage_prefers_the_no_bibliography_hint_over_the_weak_line() -> None:
    instance, out, _ = _build()
    ui.coverage(instance, 0, 0, 100, weak=True, unchecked_markers=84)
    lines = out.getvalue().splitlines()
    assert ui.WEAK_COVERAGE not in out.getvalue()
    assert lines[-1] == (
        "hint       no bibliography was found; 84 citation markers could not be checked"
    )


def test_coverage_hint_is_silent_without_unchecked_markers() -> None:
    instance, out, _ = _build()
    ui.coverage(instance, 80, 10, 10, weak=False, unchecked_markers=0)
    assert "hint" not in out.getvalue()


def test_footer_passes_the_no_bibliography_hint_through() -> None:
    instance, out, _ = _build()
    ui.footer(
        instance,
        Footer(
            counts="0 refs: nothing to check",
            coverage=(0, 0, 0),
            api_calls=0,
            elapsed=1.0,
            written=None,
            weak=False,
            cancelled=False,
            unchecked_markers=84,
        ),
    )
    assert "hint       no bibliography was found; 84 citation markers" in out.getvalue()
