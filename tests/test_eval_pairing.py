"""Offline tests for scripts/eval_pairing.py's importable functions.

No network: the whole pure half runs on ``tests/data/pairing_set.jsonl`` (the
hand-built expectation set) and on synthetic rows built here. The live half --
fetching real PDFs through the open-access chain -- is not exercised.

``test_score_on_the_hand_set_meets_the_gate`` is the Phase 5 gate: the measured
pairing rate on the hand set must stay at or above 0.95. If it fails, fix
``claims.py`` or the row -- never the bar.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "eval_pairing.py"
SET_PATH = ROOT / "tests" / "data" / "pairing_set.jsonl"

GATE = 0.95


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_pairing", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the script's dataclasses resolve their own module
    # out of sys.modules while ``@dataclass`` runs, and a file-loaded module is not
    # in there by default.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_pairing = _load_script()
Expected = eval_pairing.Expected
Row = eval_pairing.Row
Miss = eval_pairing.Miss
load_rows = eval_pairing.load_rows
score = eval_pairing.score
style_breakdown = eval_pairing.style_breakdown
render_report = eval_pairing.render_report
LiveRow = eval_pairing.LiveRow


# --- load_rows -----------------------------------------------------------


def test_load_rows_parses_every_line_of_the_hand_set() -> None:
    rows = load_rows(SET_PATH)
    lines = [line for line in SET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(lines)
    assert len(rows) >= 60
    assert all(isinstance(row, Row) for row in rows)
    assert all(row.text.strip() for row in rows)
    assert all(row.expected or row.expected_unsupported or row.expected_unresolved for row in rows)


def test_load_rows_gives_every_row_a_unique_id_and_a_known_style() -> None:
    rows = load_rows(SET_PATH)
    ids = [row.id for row in rows]
    assert len(set(ids)) == len(ids)
    assert {row.style for row in rows} <= {"numeric", "ranges", "paragraph", "mixed"}
    # Every style is actually exercised; a set that quietly lost one would still parse.
    assert {row.style for row in rows} == {"numeric", "ranges", "paragraph", "mixed"}


def test_load_rows_rejects_a_pair_expectation_with_no_refs(tmp_path: Path) -> None:
    # A marker that resolves to nothing belongs in "expected_unresolved", not in a pair.
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad-01",
                "style": "numeric",
                "text": "A sentence citing nothing resolvable [99].",
                "expected": [{"marker": "[99]", "sentence": "A sentence.", "refs": []}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refs"):
        load_rows(path)


def test_load_rows_reads_the_declared_unsupported_and_unresolved_markers(tmp_path: Path) -> None:
    path = tmp_path / "declared.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "dec-01",
                "style": "mixed",
                "text": "Earlier work (Smith et al., 2020) is cited, and so is [99].",
                "expected": [],
                "expected_unsupported": ["(Smith et al., 2020)"],
                "expected_unresolved": ["[99]"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = load_rows(path)[0]
    assert row.expected == ()
    assert row.expected_unsupported == ("(Smith et al., 2020)",)
    assert row.expected_unresolved == ("[99]",)


def test_load_rows_rejects_a_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "dupe.jsonl"
    row = {
        "id": "num-01",
        "style": "numeric",
        "text": "A sentence citing a source [2].",
        "expected": [
            {"marker": "[2]", "sentence": "A sentence citing a source.", "refs": [2]},
        ],
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_rows(path)


# --- score ---------------------------------------------------------------


def test_score_on_the_hand_set_meets_the_gate() -> None:
    rows = load_rows(SET_PATH)
    result = score(rows)
    assert result.rows == len(rows)
    assert result.pairs == sum(len(row.expected) for row in rows)
    assert result.expected == sum(row.checks for row in rows)
    detail = "\n".join(f"{m.row_id} {m.marker}: {m.reason}" for m in result.misses)
    assert result.rate >= GATE, f"pairing rate {result.rate:.3f} < {GATE}\n{detail}"


def _row(
    text: str,
    expected: tuple[Expected, ...],
    *,
    row_id: str = "syn-01",
    unsupported: tuple[str, ...] = (),
    unresolved: tuple[str, ...] = (),
) -> Row:
    return Row(
        id=row_id,
        style="numeric",
        text=text,
        expected=expected,
        expected_unsupported=unsupported,
        expected_unresolved=unresolved,
    )


def test_score_reports_a_wrong_expected_sentence_as_one_miss() -> None:
    row = _row(
        "The effect was large [4]. A second sentence follows it.",
        (Expected(marker="[4]", sentence="A second sentence follows it.", refs=(4,)),),
    )
    result = score([row])
    assert result.expected == 1
    assert result.correct == 0
    assert result.misses == (Miss(row_id="syn-01", marker="[4]", reason="wrong sentence"),)
    assert result.rate == 0.0


def test_score_counts_a_correct_expectation() -> None:
    row = _row(
        "The effect was large [4]. A second sentence follows it.",
        (Expected(marker="[4]", sentence="The effect was large.", refs=(4,)),),
    )
    result = score([row])
    assert (result.expected, result.correct, result.misses) == (1, 1, ())
    assert result.rate == 1.0


def test_score_reports_wrong_refs_when_the_sentence_is_right() -> None:
    row = _row(
        "The effect was large [4]. A second sentence follows it.",
        (Expected(marker="[4]", sentence="The effect was large.", refs=(5,)),),
    )
    assert score([row]).misses == (Miss(row_id="syn-01", marker="[4]", reason="wrong refs"),)


def test_score_reports_a_scoped_mismatch_when_only_the_flag_differs() -> None:
    row = _row(
        "One sentence here. A second one ends the paragraph [4].",
        (
            Expected(
                marker="[4]", sentence="One sentence here.", refs=(4,), paragraph_scoped=False
            ),
            Expected(
                marker="[4]",
                sentence="A second one ends the paragraph.",
                refs=(4,),
                paragraph_scoped=True,
            ),
        ),
    )
    assert score([row]).misses == (Miss(row_id="syn-01", marker="[4]", reason="scoped mismatch"),)


def test_score_reports_a_claim_nothing_expected_as_extra() -> None:
    row = _row(
        # "[7]" must not end the paragraph, or it would scope it and make two claims.
        "The effect was large [4]. A later sentence cites more [7] in passing.",
        (Expected(marker="[4]", sentence="The effect was large.", refs=(4,)),),
    )
    result = score([row])
    assert result.correct == 1
    assert result.misses == (Miss(row_id="syn-01", marker="[7]", reason="extra"),)


def test_score_reports_an_expected_marker_that_produced_nothing_as_missing() -> None:
    row = _row(
        "A paragraph with no citation at all in it.",
        (Expected(marker="[4]", sentence="A paragraph with no citation at all in it.", refs=(4,)),),
    )
    assert score([row]).misses == (Miss(row_id="syn-01", marker="[4]", reason="missing"),)


def test_score_counts_a_declared_unsupported_marker_as_an_expectation() -> None:
    row = _row(
        "Earlier work (Smith et al., 2020) established the baseline used here.",
        (),
        unsupported=("(Smith et al., 2020)",),
    )
    result = score([row])
    assert (result.expected, result.pairs, result.correct, result.misses) == (1, 0, 1, ())


def test_score_counts_a_declared_unresolved_marker_as_an_expectation() -> None:
    row = _row(
        "A sentence citing a misprinted marker [0].",
        (),
        unresolved=("[0]",),
    )
    result = score([row])
    assert (result.expected, result.correct, result.misses) == (1, 1, ())


def test_score_reports_a_declared_marker_that_was_never_reported_as_missing() -> None:
    # The numeric marker wins the overlap, so the author-year half is never reported.
    row = _row(
        "A mixed citation (see Smith, 2020; [12]) appears here, and that is all.",
        (
            Expected(
                marker="[12]",
                sentence="A mixed citation (see Smith, 2020;) appears here, and that is all.",
                refs=(12,),
            ),
        ),
        unsupported=("(see Smith, 2020; [12])",),
    )
    result = score([row])
    assert result.expected == 2
    assert result.correct == 1
    assert result.misses == (
        Miss(row_id="syn-01", marker="(see Smith, 2020; [12])", reason="missing"),
    )


def test_score_reports_an_undeclared_unsupported_marker_as_extra() -> None:
    # A row that says nothing about "(Smith et al., 2020)" must not hide it.
    row = _row(
        "Earlier work (Smith et al., 2020) established the baseline used here.",
        (),
        unresolved=("[0]",),  # something to expect, so the row is not empty
    )
    misses = score([row]).misses
    assert Miss(row_id="syn-01", marker="(Smith et al., 2020)", reason="extra") in misses


def test_score_reports_an_undeclared_unresolved_marker_as_extra() -> None:
    row = _row(
        "A sentence citing one good and one bad source [2, 99]. A second sentence here.\n"
        "\n"
        "References\n"
        "\n"
        "[1] Author, A. First entry. Journal, 2020.\n"
        "[2] Author, B. Second entry. Journal, 2021.\n",
        (
            Expected(
                marker="[2, 99]",
                sentence="A sentence citing one good and one bad source.",
                refs=(2,),
            ),
        ),
    )
    result = score([row])
    assert result.correct == 1
    assert result.misses == (Miss(row_id="syn-01", marker="[2, 99]", reason="extra"),)


def test_score_is_the_sum_of_its_rows() -> None:
    good = _row(
        "The effect was large [4]. A second sentence follows it.",
        (Expected(marker="[4]", sentence="The effect was large.", refs=(4,)),),
        row_id="syn-good",
    )
    bad = _row(
        "The effect was large [4]. A second sentence follows it.",
        (Expected(marker="[4]", sentence="Wrong.", refs=(4,)),),
        row_id="syn-bad",
    )
    result = score([good, bad])
    assert (result.rows, result.expected, result.correct) == (2, 2, 1)
    assert result.rate == 0.5
    assert [miss.row_id for miss in result.misses] == ["syn-bad"]


def test_score_of_no_rows_has_a_rate_of_zero_rather_than_dividing_by_zero() -> None:
    result = score([])
    assert (result.rows, result.expected, result.correct, result.rate) == (0, 0, 0, 0.0)


# --- by_style ------------------------------------------------------------


def test_style_breakdown_splits_pairs_and_correct_counts_per_style() -> None:
    rows = [
        Row(
            id="a",
            style="numeric",
            text="The effect was large [4]. A second sentence follows it.",
            expected=(Expected(marker="[4]", sentence="The effect was large.", refs=(4,)),),
        ),
        Row(
            id="b",
            style="ranges",
            text="Reviews [1-3] agree on this. A second sentence follows it.",
            expected=(Expected(marker="[1-3]", sentence="Wrong sentence.", refs=(1, 2, 3)),),
        ),
    ]
    assert style_breakdown(rows) == {"numeric": (1, 1), "ranges": (1, 0)}


def test_style_breakdown_of_the_hand_set_sums_to_the_overall_result() -> None:
    rows = load_rows(SET_PATH)
    breakdown = style_breakdown(rows)
    result = score(rows)
    assert sum(pairs for pairs, _ in breakdown.values()) == result.expected
    assert sum(correct for _, correct in breakdown.values()) == result.correct


# --- render_report -------------------------------------------------------


def _result(expected: int, correct: int, misses: tuple[object, ...] = ()) -> object:
    return eval_pairing.PairingResult(
        rows=60, expected=expected, correct=correct, misses=misses, pairs=expected - 3
    )


def test_render_report_has_every_heading_and_formats_the_rate() -> None:
    result = _result(100, 97, (Miss(row_id="num-05", marker="[15]", reason="wrong sentence"),))
    report = render_report(
        result,
        by_style={"numeric": (60, 59), "paragraph": (40, 38)},
        date="2026-09-11",
    )
    assert report.startswith("# Citation pairing — 2026-09-11\n")
    for heading in ("## Hand set", "## Misses", "## Real PDFs"):
        assert f"\n{heading}\n" in report
    assert "- rows: 60" in report
    assert "- expectations: 100 (97 pairs, 3 reported markers)" in report
    assert "0.97" in report
    assert "| numeric | 60 | 59 | 0.98 |" in report
    assert "| style | checks | correct | rate |" in report
    assert "| num-05 | `[15]` | wrong sentence |" in report
    # The Notes section is written by hand after the run; the harness never fakes it.
    assert "## Notes" not in report
    assert report.endswith("\n")


def test_render_report_says_none_when_nothing_missed() -> None:
    report = render_report(_result(10, 10), by_style={"numeric": (10, 10)}, date="2026-09-11")
    assert "- none" in report.split("## Misses", 1)[1]
    assert "1.00" in report


def test_render_report_renders_a_live_row_and_its_sampled_pairs() -> None:
    live = LiveRow(
        id="arXiv:1907.11692",
        source="https://arxiv.org/pdf/1907.11692",
        kind="pdf",
        pages=13,
        paragraphs=120,
        references=57,
        markers=80,
        claims=95,
        unresolved=2,
        unsupported=3,
        scoped=40,
        samples=(("[12]", "We present a replication study of BERT pretraining."),),
    )
    report = render_report(_result(10, 10), by_style={"numeric": (10, 10)}, live=(live,), date="x")
    assert "| arXiv:1907.11692 | pdf | 13 | 120 | 57 | 80 | 95 | 2 | 3 | 40 |" in report
    assert "### Sampled pairs — arXiv:1907.11692" in report
    assert "```\n[12]  We present a replication study of BERT pretraining.\n```" in report


def test_render_report_says_so_when_no_pdf_was_measured() -> None:
    report = render_report(_result(10, 10), by_style={"numeric": (10, 10)}, date="2026-09-11")
    assert "not run" in report.split("## Real PDFs", 1)[1]


# --- sample_pairs and measure_document -----------------------------------


def test_sample_pairs_returns_everything_when_there_is_little() -> None:
    pairs = [("[1]", "one"), ("[2]", "two")]
    assert eval_pairing.sample_pairs(pairs, 20) == pairs


def test_sample_pairs_spreads_the_sample_over_the_whole_document() -> None:
    pairs = [(f"[{i}]", str(i)) for i in range(100)]
    sampled = eval_pairing.sample_pairs(pairs, 5)
    assert sampled == [("[0]", "0"), ("[20]", "20"), ("[40]", "40"), ("[60]", "60"), ("[80]", "80")]


def test_measure_document_counts_what_the_real_pdf_table_prints() -> None:
    from proofpath import ingest

    text = (
        "Short opening sentence. A second sentence closes the paragraph [2].\n"
        "\n"
        "An author-year citation (Smith et al., 2020) is reported, not paired. "
        "A number nobody answers is reported too [99].\n"
        "\n"
        "References\n"
        "\n"
        "[1] Author, A. First entry. Journal, 2020.\n"
        "[2] Author, B. Second entry. Journal, 2021.\n"
    )
    doc = ingest.from_text(text, name="fake.txt", kind="text")
    row = eval_pairing.measure_document("fake", doc, source="local", count=1)
    assert (row.id, row.source, row.kind, row.pages) == ("fake", "local", "text", 1)
    assert (row.paragraphs, row.references) == (2, 2)
    assert (row.claims, row.scoped) == (2, 2)  # "[2]" ends its paragraph: both sentences
    assert (row.unresolved, row.unsupported) == (1, 1)
    assert len(row.samples) == 1


def test_measure_document_truncates_a_long_claim_in_the_sample() -> None:
    from proofpath import ingest

    long_sentence = "The measured value is stable " + "and reproducible " * 20
    doc = ingest.from_text(f"{long_sentence}[4].", name="fake.txt", kind="text")
    row = eval_pairing.measure_document("fake", doc, source="local")
    assert len(row.samples) == 1
    marker, text = row.samples[0]
    assert marker == "[4]"
    assert len(text) == eval_pairing.SAMPLE_WIDTH


# --- CLI targets ---------------------------------------------------------


def test_identifier_splits_an_arxiv_target_from_a_doi() -> None:
    assert eval_pairing.identifier("arXiv:1907.11692") == (None, "1907.11692")
    assert eval_pairing.identifier("10.1038/s41586-021-03819-2") == (
        "10.1038/s41586-021-03819-2",
        None,
    )


def test_slug_makes_a_doi_safe_as_a_file_name() -> None:
    assert eval_pairing.slug("10.1371/journal.pone.0308142") == "10.1371_journal.pone.0308142"
    assert eval_pairing.slug("arXiv:1907.11692") == "arXiv_1907.11692"


def test_render_report_says_so_instead_of_printing_an_empty_sample_block() -> None:
    live = LiveRow(
        id="10.1038/s41586-021-03819-2",
        source="https://www.nature.com/articles/s41586-021-03819-2.pdf",
        kind="pdf",
        pages=12,
        paragraphs=432,
        references=0,
        markers=9,
        claims=0,
        unresolved=0,
        unsupported=9,
        scoped=0,
        samples=(),
    )
    report = render_report(_result(10, 10), by_style={"numeric": (10, 10)}, live=(live,), date="x")
    assert "- no claims to sample" in report
    assert "```" not in report
