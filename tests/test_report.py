"""The output-side data model: levels, honesty states, coverage and exit codes."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from proofpath.document import Claim, Document, Locator, Reference
from proofpath.fetch import Outcome
from proofpath.models import Label, Passage, Verdict
from proofpath.oa import ABSTRACT_ONLY
from proofpath.report import (
    LEVELS,
    SNIPPET_LIMIT,
    STATE_WORDS,
    ClaimResult,
    Coverage,
    Diagnostic,
    Finding,
    Kind,
    Report,
    SourceStatus,
    Stage,
    render_diagnostics,
    render_footer,
    render_markdown,
)
from proofpath.resolve import State

DOCUMENT = Document(name="paper.pdf", kind="pdf", paragraphs=(), references=(), pages=24)
REFERENCE = Reference(number=12, raw="Zhang, K. et al. (2021).", locator=Locator(line=112, page=4))
PASSAGE = Passage(text="we observed a 4-8% improvement", source_id="doi:10.1/x", index=6)
SUPPORTED = Verdict(label=Label.SUPPORTED, score=0.91, tier="high", passage=PASSAGE)
REFUTED = Verdict(label=Label.REFUTED, score=0.88, tier="high", passage=PASSAGE)
NEI = Verdict(label=Label.NEI, score=0.4, tier="low", passage=None)

EMPTY_COVERAGE = Coverage(
    references=0,
    fulltext=0,
    abstract=0,
    unverified=0,
    reasons={},
    browser_skipped=0,
    network_denied=False,
)


def claim_at(line: int, page: int | None = None) -> Claim:
    return Claim(
        text="the method yields a 40% speedup",
        locator=Locator(line=line, page=page),
        cited_refs=(12,),
        paragraph=0,
        sentence=0,
    )


def finding(kind: Kind, line: int, page: int | None = None, column: int = 1) -> Finding:
    """A minimally valid Finding of ``kind``, positioned where the test needs it."""
    needs_passage = kind in {Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH}
    return Finding(
        kind=kind,
        level=LEVELS[kind],
        locator=Locator(line=line, column=column, page=page),
        title="cited source does not say this",
        state=STATE_WORDS[kind],
        reference=REFERENCE,
        claim=claim_at(line, page),
        verdict=REFUTED if needs_passage else None,
        source_id="doi:10.1/x",
        fetch_step=1,
        tier="high" if needs_passage else None,
    )


def report_with(*findings: Finding, cancelled: bool = False) -> Report:
    return Report(
        document=DOCUMENT,
        claims=118,
        markers=118,
        sources=(),
        results=(),
        findings=findings,
        coverage=EMPTY_COVERAGE,
        stages=(),
        models={},
        api_calls=0,
        elapsed=38.4,
        cancelled=cancelled,
    )


# --- the tables ------------------------------------------------------------------


def test_levels_covers_every_kind() -> None:
    assert set(LEVELS) == set(Kind)


def test_levels_are_only_the_three_words() -> None:
    assert set(LEVELS.values()) == {"error", "warning", "note"}


def test_the_three_error_kinds_are_the_ones_that_fail_a_run_loudly() -> None:
    errors = {kind for kind, level in LEVELS.items() if level == "error"}
    assert errors == {Kind.GHOST, Kind.NUMERIC_MISMATCH, Kind.NOT_SUPPORTED}


def test_state_words_covers_every_kind() -> None:
    assert set(STATE_WORDS) == set(Kind)
    assert all(word for word in STATE_WORDS.values())


def test_numeric_mismatch_shows_the_same_state_word_as_not_supported() -> None:
    # The reason carries "numeric mismatch"; the coloured word stays NOT SUPPORTED.
    assert STATE_WORDS[Kind.NUMERIC_MISMATCH] == STATE_WORDS[Kind.NOT_SUPPORTED] == "NOT SUPPORTED"


def test_state_words_are_bound_to_the_module_that_owns_them() -> None:
    # Product rule 2: never a copy of the string, so a stage cannot drift from the
    # report behind its back. The owning module is the single source of the wording.
    assert STATE_WORDS[Kind.GHOST] == State.GHOST.value
    assert STATE_WORDS[Kind.AMBIGUOUS] == State.AMBIGUOUS.value
    assert STATE_WORDS[Kind.PROVIDER_UNAVAILABLE] == Outcome.UNAVAILABLE.value
    assert STATE_WORDS[Kind.ABSTRACT_ONLY] == ABSTRACT_ONLY


def test_state_words_owned_here_read_as_spec_section_15_writes_them() -> None:
    # These four have no upstream owner; report.py is where they are defined.
    assert STATE_WORDS[Kind.RETRACTED] == "RETRACTED"
    assert STATE_WORDS[Kind.NEI] == "NEI"
    assert STATE_WORDS[Kind.PARAGRAPH_SCOPED] == "PARAGRAPH-SCOPED"
    assert STATE_WORDS[Kind.UNSUPPORTED_STYLE] == "UNSUPPORTED CITATION STYLE"
    assert STATE_WORDS[Kind.UNRESOLVED_MARKER] == "UNRESOLVED MARKER"
    assert STATE_WORDS[Kind.PARSE_ERROR] == "PARSE ERROR"


def test_kind_values_are_the_diagnostic_codes() -> None:
    # They print as error[ghost-reference] in the one-shot output (spec section 13.2).
    assert Kind.GHOST.value == "ghost-reference"
    assert Kind.NUMERIC_MISMATCH.value == "numeric-mismatch"
    assert Kind.UNSUPPORTED_STYLE.value == "unsupported-citation-style"


# --- Finding invariants ----------------------------------------------------------


@pytest.mark.parametrize("kind", [Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH])
def test_an_asserting_finding_without_a_verdict_raises(kind: Kind) -> None:
    with pytest.raises(ValueError, match="passage"):
        Finding(
            kind=kind,
            level=LEVELS[kind],
            locator=Locator(line=260, page=9),
            title="claim contradicts the cited source",
            state=STATE_WORDS[kind],
            reference=REFERENCE,
            claim=claim_at(260, 9),
            verdict=None,
            source_id="doi:10.1/x",
            fetch_step=1,
            tier="high",
        )


@pytest.mark.parametrize("kind", [Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH])
def test_an_asserting_finding_without_a_passage_raises(kind: Kind) -> None:
    with pytest.raises(ValueError, match="passage"):
        Finding(
            kind=kind,
            level=LEVELS[kind],
            locator=Locator(line=260, page=9),
            title="claim contradicts the cited source",
            state=STATE_WORDS[kind],
            reference=REFERENCE,
            claim=claim_at(260, 9),
            verdict=NEI,  # a verdict, but no passage behind it
            source_id="doi:10.1/x",
            fetch_step=1,
            tier="low",
        )


@pytest.mark.parametrize("kind", [Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH])
def test_an_asserting_finding_with_a_passage_is_valid(kind: Kind) -> None:
    assert finding(kind, line=260, page=9).verdict is REFUTED


def test_nei_needs_no_verdict() -> None:
    assert finding(Kind.NEI, line=12).verdict is None


def test_ghost_needs_no_verdict() -> None:
    # Nothing was read, so there is no passage to quote; the state says exactly that.
    assert finding(Kind.GHOST, line=112, page=4).verdict is None


def test_a_level_that_disagrees_with_the_kind_raises() -> None:
    with pytest.raises(ValueError, match="level"):
        Finding(
            kind=Kind.GHOST,
            level="note",  # GHOST is an error
            locator=Locator(line=112, page=4),
            title="cited source does not exist",
            state=STATE_WORDS[Kind.GHOST],
            reference=REFERENCE,
            claim=None,
            verdict=None,
            source_id=None,
            fetch_step=None,
            tier=None,
        )


def test_unverified_keeps_the_state_string_it_was_given() -> None:
    # Rule 2: the precise honesty string survives, it is not flattened to "UNVERIFIED".
    state = "UNVERIFIED (blocked, browser not permitted)"
    item = Finding(
        kind=Kind.UNVERIFIED,
        level=LEVELS[Kind.UNVERIFIED],
        locator=Locator(line=7),
        title="source could not be read",
        state=state,
        reference=REFERENCE,
        claim=None,
        verdict=None,
        source_id="url:https://example.org/a",
        fetch_step=2,
        tier=None,
    )
    assert item.state == state


@pytest.mark.parametrize(
    "state",
    ["UNVERIFIED", "UNVERIFIED (unreachable)", "UNVERIFIED (no identifier to fetch)"],
)
def test_unverified_accepts_any_state_from_that_family(state: str) -> None:
    item = Finding(
        kind=Kind.UNVERIFIED,
        level=LEVELS[Kind.UNVERIFIED],
        locator=Locator(line=7),
        title="source could not be read",
        state=state,
        reference=REFERENCE,
        claim=None,
        verdict=None,
        source_id=None,
        fetch_step=None,
        tier=None,
    )
    assert item.state == state


def test_unverified_rejects_a_state_from_another_family() -> None:
    with pytest.raises(ValueError, match="state"):
        Finding(
            kind=Kind.UNVERIFIED,
            level=LEVELS[Kind.UNVERIFIED],
            locator=Locator(line=7),
            title="source could not be read",
            state="AMBIGUOUS",
            reference=REFERENCE,
            claim=None,
            verdict=None,
            source_id=None,
            fetch_step=None,
            tier=None,
        )


def test_every_other_kind_must_carry_its_own_state_word() -> None:
    with pytest.raises(ValueError, match="state"):
        Finding(
            kind=Kind.GHOST,
            level=LEVELS[Kind.GHOST],
            locator=Locator(line=112, page=4),
            title="cited source does not exist",
            state="probably missing",  # a paraphrase is exactly what rule 2 forbids
            reference=REFERENCE,
            claim=None,
            verdict=None,
            source_id=None,
            fetch_step=None,
            tier=None,
        )


@pytest.mark.parametrize("kind", list(Kind))
def test_the_default_state_word_is_always_accepted(kind: Kind) -> None:
    assert finding(kind, line=1).state == STATE_WORDS[kind]


def test_finding_defaults() -> None:
    item = finding(Kind.NEI, line=3)
    assert item.detail == ()
    assert item.group is None


def test_finding_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding(Kind.NEI, line=3).kind = Kind.GHOST  # type: ignore[misc]


# --- Coverage --------------------------------------------------------------------


def coverage(references: int, fulltext: int, abstract: int, unverified: int) -> Coverage:
    return Coverage(
        references=references,
        fulltext=fulltext,
        abstract=abstract,
        unverified=unverified,
        reasons={},
        browser_skipped=0,
        network_denied=False,
    )


def test_pct_matches_the_spec_example() -> None:
    assert coverage(42, 26, 9, 7).pct() == (62, 21, 17)


def test_pct_sums_to_100() -> None:
    for unverified in range(0, 8):
        for abstract in range(0, 8 - unverified):
            fulltext = 7 - unverified - abstract
            assert sum(coverage(7, fulltext, abstract, unverified).pct()) == 100


def test_pct_distributes_the_rounding_leftovers() -> None:
    assert coverage(3, 1, 1, 1).pct() == (33, 33, 34)


def test_pct_is_zero_without_references() -> None:
    assert coverage(0, 0, 0, 0).pct() == (0, 0, 0)


def test_pct_is_exact_when_everything_was_read() -> None:
    assert coverage(10, 10, 0, 0).pct() == (100, 0, 0)


def test_weak_below_a_quarter_unverified() -> None:
    assert coverage(100, 76, 0, 24).weak() is False


def test_weak_at_a_quarter_unverified() -> None:
    assert coverage(100, 75, 0, 25).weak() is True


def test_weak_above_a_quarter_unverified() -> None:
    assert coverage(4, 3, 0, 1).weak() is True


def test_weak_counts_the_gap_left_by_an_under_counted_unverified() -> None:
    # 3 of 10 references were never read, but `unverified` says 0. The share comes
    # from the denominator, so the run is still weak (rule 6): a producer that
    # forgets to tally a failure cannot make its report look stronger than it was.
    assert coverage(10, 7, 0, 0).weak() is True


def test_weak_ignores_an_over_counted_unverified() -> None:
    # The mirror case: 9 of 10 read, so one reference is missing however many times
    # a buggy producer counted it. The denominator, not the tally, decides.
    assert coverage(10, 9, 0, 5).weak() is False


def test_a_run_with_no_references_is_not_weak() -> None:
    # Nothing to cover is not weak coverage: "0 of 0 sources could not be read" warns
    # about a gap that does not exist. A document with no references is reported as
    # having none (rule 6 lives in that count), not as a badly covered one.
    assert coverage(0, 0, 0, 0).weak() is False


def test_coverage_keeps_its_reasons_and_permission_counters() -> None:
    item = Coverage(
        references=9,
        fulltext=0,
        abstract=0,
        unverified=9,
        reasons={"UNVERIFIED (blocked, browser not permitted)": 3},
        browser_skipped=3,
        network_denied=True,
    )
    assert item.reasons["UNVERIFIED (blocked, browser not permitted)"] == 3
    assert item.browser_skipped == 3
    assert item.network_denied is True


# --- the composed records --------------------------------------------------------


def test_source_status_defaults() -> None:
    status = SourceStatus(
        reference=REFERENCE,
        resolve=None,
        retraction=None,
        source_id="doi:10.1/x",
        text_kind="fulltext",
        state="",
        fetch_step=1,
        url="https://example.org/a",
        from_cache=False,
    )
    assert status.notes == ()
    assert status.state == ""


def test_claim_result_keeps_supported_claims_too() -> None:
    result = ClaimResult(
        claim=claim_at(260, 9),
        reference=12,
        source_id="doi:10.1/x",
        verdict=SUPPORTED,
        from_cache=True,
    )
    assert result.verdict.label is Label.SUPPORTED
    assert result.from_cache is True


def test_stage_carries_its_attribution() -> None:
    stage = Stage(name="Resolving", by="Crossref, OpenAlex", summary="38 ok, 3 amb", elapsed=3.4)
    assert (stage.name, stage.by, stage.elapsed) == ("Resolving", "Crossref, OpenAlex", 3.4)


# --- Report ----------------------------------------------------------------------


def test_exit_code_is_zero_for_a_clean_run() -> None:
    assert report_with().exit_code() == 0


@pytest.mark.parametrize("kind", list(Kind))
def test_exit_code_is_one_for_every_state(kind: Kind) -> None:
    # Spec section 13.3: every UNVERIFIED and LOW CONFIDENCE state counts, not just errors.
    assert report_with(finding(kind, line=1)).exit_code() == 1


def test_exit_code_is_one_when_cancelled() -> None:
    assert report_with(cancelled=True).exit_code() == 1


def test_report_defaults() -> None:
    report = report_with()
    assert report.cancelled is False
    assert report.summary is None


def test_counts_tallies_each_kind() -> None:
    report = report_with(
        finding(Kind.GHOST, line=1),
        finding(Kind.GHOST, line=2),
        finding(Kind.NEI, line=3),
    )
    assert report.counts() == {Kind.GHOST: 2, Kind.NEI: 1}


def test_counts_is_empty_for_a_clean_run() -> None:
    assert report_with().counts() == {}


def test_counts_follows_the_declaration_order_of_kind() -> None:
    report = report_with(
        finding(Kind.NEI, line=3),
        finding(Kind.RETRACTED, line=2),
        finding(Kind.GHOST, line=1),
    )
    assert list(report.counts()) == [Kind.GHOST, Kind.RETRACTED, Kind.NEI]


def test_by_level_groups_errors_then_warnings_then_notes() -> None:
    report = report_with(
        finding(Kind.NEI, line=3),
        finding(Kind.GHOST, line=1),
        finding(Kind.RETRACTED, line=2),
    )
    assert list(report.by_level()) == ["error", "warning", "note"]


def test_by_level_omits_levels_with_no_findings() -> None:
    assert list(report_with(finding(Kind.NEI, line=1)).by_level()) == ["note"]
    assert report_with().by_level() == {}


def test_by_level_orders_findings_across_pages() -> None:
    report = report_with(
        finding(Kind.GHOST, line=5, page=9),
        finding(Kind.GHOST, line=112, page=4),
        finding(Kind.GHOST, line=5, page=4),
        finding(Kind.NOT_SUPPORTED, line=260, page=9),
    )
    errors = report.by_level()["error"]
    assert [(f.locator.page, f.locator.line) for f in errors] == [
        (4, 5),
        (4, 112),
        (9, 5),
        (9, 260),
    ]


def test_by_level_orders_pageless_findings_first() -> None:
    report = report_with(
        finding(Kind.GHOST, line=1, page=1),
        finding(Kind.GHOST, line=9),
    )
    errors = report.by_level()["error"]
    assert [(f.locator.page, f.locator.line) for f in errors] == [(None, 9), (1, 1)]


def test_by_level_breaks_line_ties_on_column() -> None:
    report = report_with(
        finding(Kind.GHOST, line=7, page=2, column=30),
        finding(Kind.GHOST, line=7, page=2, column=4),
    )
    errors = report.by_level()["error"]
    assert [f.locator.column for f in errors] == [4, 30]


def test_by_level_keeps_every_finding() -> None:
    findings = tuple(finding(kind, line=1) for kind in Kind)
    report = report_with(*findings)
    assert sum(len(group) for group in report.by_level().values()) == len(findings)


# --- the renderers ----------------------------------------------------------------

NUMERIC = Verdict(
    label=Label.REFUTED,
    score=1.0,
    tier="high",
    passage=PASSAGE,
    reason="numeric mismatch: claim says 40%, source says 4-8%",
)
RETRACTED_REFERENCE = Reference(
    number=28,
    raw="[28] Lee, S. (2019). Adaptive gating for efficient inference",
    locator=Locator(line=203, page=7),
)


def spec_reference(number: int = 12) -> Reference:
    return Reference(
        number=number,
        raw="[12] Zhang, K. et al. (2021). Neural cascade alignment for zero-shot",
        locator=Locator(line=112, page=4),
    )


def ghost_finding() -> Finding:
    return Finding(
        kind=Kind.GHOST,
        level="error",
        locator=Locator(line=112, column=1, page=4),
        title="cited source does not exist",
        state=STATE_WORDS[Kind.GHOST],
        reference=spec_reference(),
        claim=None,
        verdict=None,
        source_id=None,
        fetch_step=None,
        tier=None,
        detail=("no author, title or year agreement with any candidate",),
    )


def numeric_finding() -> Finding:
    claim = Claim(
        text="The method yields a 40% speedup on long-context workloads",
        locator=Locator(line=260, column=1, page=9),
        cited_refs=(12,),
        paragraph=3,
        sentence=1,
    )
    return Finding(
        kind=Kind.NUMERIC_MISMATCH,
        level="error",
        locator=claim.locator,
        title="claim contradicts the cited source",
        state=STATE_WORDS[Kind.NUMERIC_MISMATCH],
        reference=spec_reference(),
        claim=claim,
        verdict=NUMERIC,
        source_id="doi:10.1/x",
        fetch_step=1,
        tier="high",
        detail=(NUMERIC.reason,),
    )


def retracted_finding() -> Finding:
    return Finding(
        kind=Kind.RETRACTED,
        level="warning",
        locator=Locator(line=203, column=1, page=7),
        title="cited source was retracted 2023-06",
        state=STATE_WORDS[Kind.RETRACTED],
        reference=RETRACTED_REFERENCE,
        claim=None,
        verdict=None,
        source_id="doi:10.2/y",
        fetch_step=None,
        tier=None,
        detail=('"Concerns about data integrity" - Retraction Watch',),
    )


def nei_finding() -> Finding:
    claim = claim_at(40, 5)
    return Finding(
        kind=Kind.NEI,
        level="note",
        locator=claim.locator,
        title="source neither supports nor contradicts the claim",
        state=STATE_WORDS[Kind.NEI],
        reference=spec_reference(),
        claim=claim,
        verdict=NEI,  # NEI verdict, no passage: nothing may be quoted (rule 1)
        source_id="doi:10.1/x",
        fetch_step=1,
        tier="low",
    )


def unverified_finding() -> Finding:
    return Finding(
        kind=Kind.UNVERIFIED,
        level="warning",
        locator=Locator(line=7, column=1, page=2),
        title="source could not be read",
        state="UNVERIFIED (blocked, browser not permitted)",
        reference=spec_reference(),
        claim=None,
        verdict=None,
        source_id="url:https://example.org/a",
        fetch_step=2,
        tier=None,
    )


def only(report: Report) -> Diagnostic:
    rendered = render_diagnostics(report)
    assert len(rendered) == 1
    return rendered[0]


def block(diagnostic: Diagnostic) -> str:
    """The diagnostic as one string, the way ``ui.diagnostic`` lays it out."""
    pad = " " * diagnostic.indent
    head = f"{pad}{diagnostic.level}[{diagnostic.code}]: {diagnostic.title}"
    arrow = f"{pad}  --> {diagnostic.location}"
    return "\n".join([head, arrow, *(pad + line for line in diagnostic.lines)])


def test_a_ghost_renders_the_reference_line_and_its_note() -> None:
    rendered = only(report_with(ghost_finding()))
    assert rendered.level == "error"
    assert rendered.code == "ghost-reference"
    assert rendered.state == "GHOST REFERENCE"
    assert block(rendered) == (
        "error[ghost-reference]: cited source does not exist\n"
        "  --> paper.pdf:4:112\n"
        "   |\n"
        "   | [12] Zhang, K. et al. (2021). Neural cascade alignment for zero-shot\n"
        "   |\n"
        "   = note: no author, title or year agreement with any candidate"
    )


def test_a_numeric_mismatch_carets_the_figure_and_quotes_the_passage() -> None:
    rendered = only(report_with(numeric_finding()))
    assert block(rendered) == (
        "error[numeric-mismatch]: claim contradicts the cited source  (confidence: high)\n"
        "  --> paper.pdf:9:260\n"
        "   |\n"
        "   | The method yields a 40% speedup on long-context workloads\n"
        "   |                     ^^^ source reports 4-8%\n"
        "   |\n"
        '   = source: "we observed a 4-8% improvement"  ([12] passage 6)\n'
        "   = note: numeric mismatch: claim says 40%, source says 4-8%"
    )


def test_a_retraction_renders_without_a_caret_line() -> None:
    rendered = only(report_with(retracted_finding()))
    assert block(rendered) == (
        "warning[retracted]: cited source was retracted 2023-06\n"
        "  --> paper.pdf:7:203\n"
        "   |\n"
        "   | [28] Lee, S. (2019). Adaptive gating for efficient inference\n"
        "   |\n"
        '   = note: "Concerns about data integrity" - Retraction Watch'
    )


def test_nei_without_a_passage_quotes_nothing() -> None:
    # Product rule 1: no passage, no quote. Not even an empty one.
    rendered = only(report_with(nei_finding()))
    assert block(rendered) == (
        "note[nei]: source neither supports nor contradicts the claim  (confidence: low)\n"
        "  --> paper.pdf:5:40\n"
        "   |\n"
        "   | the method yields a 40% speedup"
    )
    assert not any(line.startswith("   = source:") for line in rendered.lines)


def test_an_unverified_finding_keeps_its_exact_state_string() -> None:
    # Product rule 2: the flavour of the failure is on the page, verbatim.
    rendered = only(report_with(unverified_finding()))
    assert rendered.lines[-1] == "   = state: UNVERIFIED (blocked, browser not permitted)"
    assert rendered.state == "UNVERIFIED (blocked, browser not permitted)"


def test_only_a_numeric_mismatch_gets_a_caret_line() -> None:
    for item in (ghost_finding(), retracted_finding(), nei_finding(), unverified_finding()):
        assert not any("^" in line for line in only(report_with(item)).lines)


def test_an_entailment_call_gets_no_caret_even_carrying_a_rule_shaped_reason() -> None:
    # Underlining part of a sentence claims a precision only a rule has (spec 10);
    # a NOT SUPPORTED verdict is about the whole claim, so it gets no carets.
    item = dataclasses.replace(
        numeric_finding(),
        kind=Kind.NOT_SUPPORTED,
        title="claim is not supported by the cited source",
        state=STATE_WORDS[Kind.NOT_SUPPORTED],
    )
    rendered = only(report_with(item))
    assert rendered.code == "not-supported"
    assert not any("^" in line for line in rendered.lines)


def test_a_caret_line_is_dropped_when_the_figure_is_not_in_the_claim() -> None:
    # A reworded or truncated snippet loses its carets, never its diagnostic.
    reworded = dataclasses.replace(
        NUMERIC, reason="numeric mismatch: claim says 9x, source says 2x"
    )
    item = dataclasses.replace(numeric_finding(), verdict=reworded)
    assert not any("^" in line for line in only(report_with(item)).lines)


def numeric_claiming(text: str, figure: str) -> Finding:
    """A numeric mismatch whose claim reads ``text`` and whose rule read ``figure``."""
    claim = Claim(
        text=text, locator=Locator(line=260, page=9), cited_refs=(12,), paragraph=3, sentence=1
    )
    reason = f"numeric mismatch: claim says {figure}, source says 4-8%"
    return dataclasses.replace(
        numeric_finding(),
        claim=claim,
        locator=claim.locator,
        verdict=dataclasses.replace(NUMERIC, reason=reason),
        detail=(reason,),
    )


def caret_line(item: Finding) -> str | None:
    return next((line for line in only(report_with(item)).lines if "^" in line), None)


def test_a_caret_skips_a_figure_that_is_only_part_of_a_larger_number() -> None:
    # "8%" lives inside "48%"; pointing there would accuse the wrong number.
    item = numeric_claiming("A 48% baseline and an 8% drop", "8%")
    assert caret_line(item) == "   | " + " " * 22 + "^^ source reports 4-8%"


def test_a_caret_still_points_at_a_figure_that_ends_the_sentence() -> None:
    # A full stop after the figure is punctuation, not a decimal continuation.
    item = numeric_claiming("Throughput improves by 40%.", "40%")
    assert caret_line(item) == "   | " + " " * 23 + "^^^ source reports 4-8%"


def test_a_caret_line_is_dropped_when_the_claim_names_the_figure_twice() -> None:
    # Nothing here knows which "8%" the rule read, so it points at neither.
    assert caret_line(numeric_claiming("Unlike the 8% baseline, we see 8% here", "8%")) is None


def test_a_caret_line_is_dropped_when_the_reason_is_not_a_rule_decision() -> None:
    item = dataclasses.replace(numeric_finding(), verdict=dataclasses.replace(NUMERIC, reason=""))
    assert not any("^" in line for line in only(report_with(item)).lines)


def test_a_snippet_of_exactly_the_limit_is_left_whole() -> None:
    claim = Claim(
        text="x" * SNIPPET_LIMIT,
        locator=Locator(line=1),
        cited_refs=(12,),
        paragraph=0,
        sentence=0,
    )
    item = dataclasses.replace(nei_finding(), claim=claim, locator=claim.locator)
    assert only(report_with(item)).lines[1] == "   | " + "x" * SNIPPET_LIMIT


def test_one_character_over_the_limit_is_truncated() -> None:
    claim = Claim(
        text="x" * (SNIPPET_LIMIT + 1),
        locator=Locator(line=1),
        cited_refs=(12,),
        paragraph=0,
        sentence=0,
    )
    item = dataclasses.replace(nei_finding(), claim=claim, locator=claim.locator)
    snippet = only(report_with(item)).lines[1]
    assert snippet == "   | " + "x" * (SNIPPET_LIMIT - 3) + "..."


def test_a_long_snippet_is_truncated_to_a_hundred_characters() -> None:
    claim = Claim(
        text="word " * 40,
        locator=Locator(line=1),
        cited_refs=(12,),
        paragraph=0,
        sentence=0,
    )
    item = dataclasses.replace(nei_finding(), claim=claim, locator=claim.locator)
    snippet = only(report_with(item)).lines[1]
    assert len(snippet) == len("   | ") + SNIPPET_LIMIT
    assert snippet.endswith("...")


def test_a_pageless_location_leaves_the_page_out() -> None:
    item = dataclasses.replace(nei_finding(), locator=Locator(line=112))
    assert only(report_with(item)).location == "paper.pdf:112"


ACCENTED_CLAIM = "Le modèle atteint une accélération de 40% \u2014 \u201cnettement\u201d supérieure"
ACCENTED_RAW = "[31] Kumar, A. \u2013 \u201cRésumé\u201d, Zeitschrift für Physik"
ACCENTED_NOTE = '"Concerns about data integrity" \u2014 Retraction Watch'
ACCENTED_PASSAGE = Passage(
    text="we observed a 4\u20138% improvement", source_id="doi:10.1/x", index=6
)


def test_the_layout_is_ascii_and_the_content_survives_verbatim() -> None:
    # Spec section 13.2 draws the frame in ASCII so Windows Terminal renders it
    # identically; the text a finding is about is the user's, and is never
    # transliterated -- that would corrupt the very sentence under discussion.
    accented = Reference(number=31, raw=ACCENTED_RAW, locator=Locator(line=112, page=4))
    reason = "numeric mismatch: claim says 40%, source says 4\u20138%"
    claim = Claim(
        text=ACCENTED_CLAIM,
        locator=Locator(line=260, page=9),
        cited_refs=(31,),
        paragraph=3,
        sentence=1,
    )
    report = report_with(
        dataclasses.replace(
            ghost_finding(), reference=accented, detail=(ACCENTED_NOTE,), claim=None
        ),
        dataclasses.replace(
            numeric_finding(),
            reference=accented,
            claim=claim,
            locator=claim.locator,
            verdict=dataclasses.replace(NUMERIC, passage=ACCENTED_PASSAGE, reason=reason),
            detail=(ACCENTED_NOTE,),
        ),
    )
    content = (ACCENTED_CLAIM, ACCENTED_RAW, ACCENTED_NOTE, ACCENTED_PASSAGE.text, "4\u20138%")
    printed = "\n".join(block(item) for item in render_diagnostics(report))

    for line in printed.splitlines():
        layout = line
        for piece in content:
            layout = layout.replace(piece, "")
        assert layout.isascii(), layout  # gutters, arrows, "=", carets, level[code]
    for piece in content:
        assert piece in printed, piece  # byte for byte, not normalised
    assert "   |                                       ^^^ source reports 4\u20138%" in printed


def test_diagnostics_come_out_in_document_order() -> None:
    report = report_with(numeric_finding(), ghost_finding(), retracted_finding())
    assert [item.location for item in render_diagnostics(report)] == [
        "paper.pdf:4:112",
        "paper.pdf:7:203",
        "paper.pdf:9:260",
    ]


# --- paragraph-scoped groups ------------------------------------------------------


def group_findings() -> tuple[Finding, Finding]:
    claim = Claim(
        text="the method yields a 40% speedup",
        locator=Locator(line=118, column=1, page=3),
        cited_refs=(12,),
        paragraph=3,
        sentence=0,
        paragraph_scoped=True,
        group="p3:118-122",
    )
    head = Finding(
        kind=Kind.PARAGRAPH_SCOPED,
        level="note",
        locator=claim.locator,
        title="citation supports a paragraph, not one sentence",
        state=STATE_WORDS[Kind.PARAGRAPH_SCOPED],
        reference=spec_reference(),
        claim=claim,
        verdict=None,
        source_id="doi:10.1/x",
        fetch_step=1,
        tier=None,
        detail=("3 sentences: 1 supported, 1 NEI, 1 not supported",),
        group="p3:118-122",
    )
    member = dataclasses.replace(
        nei_finding(), locator=Locator(line=122, column=1, page=3), group="p3:118-122"
    )
    return head, member


def test_a_group_is_followed_by_its_members_indented_two_more_spaces() -> None:
    head, member = group_findings()
    rendered = render_diagnostics(report_with(member, head))
    assert [item.code for item in rendered] == ["paragraph-scoped", "nei"]
    assert rendered[0].indent == 0
    assert rendered[1].indent == 2
    assert block(rendered[1]).splitlines()[0].startswith("  note[nei]:")


def test_a_group_member_block_is_shifted_whole() -> None:
    head, member = group_findings()
    rendered = render_diagnostics(report_with(member, head))
    assert block(rendered[1]) == (
        "  note[nei]: source neither supports nor contradicts the claim  (confidence: low)\n"
        "    --> paper.pdf:3:122\n"
        "     |\n"
        "     | the method yields a 40% speedup"
    )


def test_two_group_findings_sharing_a_group_do_not_print_the_members_twice() -> None:
    head, member = group_findings()
    twin = dataclasses.replace(head, locator=Locator(line=119, column=1, page=3))
    rendered = render_diagnostics(report_with(member, head, twin))
    assert [item.code for item in rendered] == ["paragraph-scoped", "nei", "paragraph-scoped"]


def test_a_member_without_its_group_finding_is_still_rendered() -> None:
    _, member = group_findings()
    assert [item.code for item in render_diagnostics(report_with(member))] == ["nei"]


# --- the footer -------------------------------------------------------------------


def covered(references: int, fulltext: int, abstract: int, unverified: int) -> Report:
    return dataclasses.replace(
        report_with(), coverage=coverage(references, fulltext, abstract, unverified)
    )


def test_footer_counts_read_like_the_spec_example() -> None:
    report = dataclasses.replace(
        covered(42, 26, 9, 7),
        findings=(
            ghost_finding(),
            retracted_finding(),
            numeric_finding(),
        ),
    )
    footer = render_footer(report, written="report.md")
    assert footer.counts == "42 refs: 1 ghost, 1 retracted, 1 unsupported, 40 ok"
    assert footer.coverage == (62, 21, 17)
    assert footer.written == "report.md"
    assert footer.weak is False


def test_footer_leaves_out_the_categories_that_did_not_happen() -> None:
    assert render_footer(covered(42, 42, 0, 0)).counts == "42 refs: 42 ok"


def test_footer_counts_a_reference_once_however_many_findings_it_has() -> None:
    report = dataclasses.replace(
        covered(4, 4, 0, 0), findings=(ghost_finding(), unverified_finding())
    )
    # Both findings point at reference 12, so three references are still ok.
    assert render_footer(report).counts == "4 refs: 1 ghost, 3 ok"


def test_footer_notes_are_not_counted_against_a_reference() -> None:
    report = dataclasses.replace(covered(4, 4, 0, 0), findings=(nei_finding(),))
    assert render_footer(report).counts == "4 refs: 4 ok"


def test_footer_carries_weak_and_cancelled_through() -> None:
    report = dataclasses.replace(covered(4, 3, 0, 1), cancelled=True)
    footer = render_footer(report)
    assert (footer.weak, footer.cancelled, footer.written) == (True, True, None)
    assert (footer.api_calls, footer.elapsed) == (0, 38.4)


# --- markdown ---------------------------------------------------------------------

WHEN = datetime(2026, 9, 11, 14, 3, 22)


def test_markdown_titles_itself_with_the_document() -> None:
    text = render_markdown(report_with(), written_at=WHEN)
    assert text.startswith("# proofpath report — paper.pdf\n")
    assert "- date: 2026-09-11 14:03:22" in text
    assert "- elapsed: 38.4s" in text
    assert "- api calls: 0" in text


def test_markdown_carries_the_model_preamble() -> None:
    report = dataclasses.replace(
        report_with(), models={"nli": "cross-encoder", "device": "mps", "thresholds": "decide=0.5"}
    )
    text = render_markdown(report, written_at=WHEN)
    assert "- nli: cross-encoder" in text
    assert "- device: mps" in text
    assert "- thresholds: decide=0.5" in text


def test_markdown_groups_findings_by_level() -> None:
    report = dataclasses.replace(
        covered(42, 26, 9, 7), findings=(ghost_finding(), retracted_finding(), nei_finding())
    )
    text = render_markdown(report, written_at=WHEN)
    assert "### Errors" in text and "### Warnings" in text and "### Notes" in text
    assert "- **p.4 L112** `[12]` GHOST REFERENCE — cited source does not exist" in text, text
    assert "  - note: no author, title or year agreement with any candidate" in text


def test_markdown_quotes_you_and_source_only_when_both_exist() -> None:
    text = render_markdown(
        dataclasses.replace(covered(42, 26, 9, 7), findings=(numeric_finding(),)),
        written_at=WHEN,
    )
    assert '  - you: "The method yields a 40% speedup on long-context workloads"' in text
    assert '  - source: "we observed a 4-8% improvement"' in text
    nei = render_markdown(
        dataclasses.replace(covered(42, 26, 9, 7), findings=(nei_finding(),)), written_at=WHEN
    )
    assert "  - source:" not in nei  # rule 1 again, on the markdown side


def test_markdown_coverage_block_is_the_spec_section_15_block() -> None:
    text = render_markdown(covered(42, 26, 9, 7), written_at=WHEN)
    assert (
        "verified against full text   62%\n"
        "abstract only                21%\n"
        "unverified                   17%\n"
    ) in text


def test_markdown_says_so_when_coverage_is_weak_at_exactly_a_quarter() -> None:
    text = render_markdown(covered(4, 3, 0, 1), written_at=WHEN)
    assert (
        "Coverage is weak: 1 of 4 sources could not be read, "
        "so the findings above are a lower bound." in text
    )


def test_markdown_omits_the_weak_sentence_when_coverage_holds() -> None:
    assert "Coverage is weak" not in render_markdown(covered(42, 32, 3, 7), written_at=WHEN)


def test_markdown_omits_the_weak_sentence_when_there_was_nothing_to_cover() -> None:
    # "0 of 0 sources could not be read" is not a coverage warning, it is noise.
    assert "Coverage is weak" not in render_markdown(covered(0, 0, 0, 0), written_at=WHEN)


def test_markdown_lists_a_supported_claim_only_under_checked() -> None:
    result = ClaimResult(
        claim=claim_at(260, 9),
        reference=12,
        source_id="doi:10.1/x",
        verdict=SUPPORTED,
        from_cache=False,
    )
    report = dataclasses.replace(covered(42, 26, 9, 7), results=(result,))
    text = render_markdown(report, written_at=WHEN)
    checked = text.split("## Checked", 1)[1].split("## Sources", 1)[0]
    assert "| p.9 L260 | [12] | high | we observed a 4-8% improvement |" in checked
    assert text.count("we observed a 4-8% improvement") == 1


def test_markdown_sources_table_keeps_every_state_verbatim() -> None:
    status = SourceStatus(
        reference=spec_reference(),
        resolve=None,
        retraction=None,
        source_id="url:https://example.org/a",
        text_kind="none",
        state="UNVERIFIED (blocked, robots.txt)",
        fetch_step=2,
        url="https://example.org/a",
        from_cache=False,
    )
    text = render_markdown(dataclasses.replace(covered(1, 0, 0, 1), sources=(status,)))
    assert "| [12] | UNVERIFIED (blocked, robots.txt) | none | 2 | https://example.org/a |" in text


def test_markdown_summary_appears_only_when_the_model_wrote_one() -> None:
    assert "## Summary" not in render_markdown(report_with(), written_at=WHEN)
    report = dataclasses.replace(report_with(), summary="Three references do not say this.")
    text = render_markdown(report, written_at=WHEN)
    assert "## Summary (model-written)" in text
    assert "Three references do not say this." in text


def test_markdown_defaults_its_date_to_now() -> None:
    assert "- date: " in render_markdown(report_with())
