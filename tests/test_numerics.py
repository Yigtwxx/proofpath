"""Numeric claim layer (spec section 10). Hand-built set; every case is a real failure mode."""

from __future__ import annotations

import pytest

from proofpath import numerics as num
from proofpath.models import Passage


def q(text: str) -> list[num.Quantity]:
    return num.extract(text)


# --- extraction ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "low", "high", "unit"),
    [
        ("a 40% speedup", "percent", 40, 40, "%"),
        ("improved by 12.5 percent", "percent", 12.5, 12.5, "%"),
        ("a 4-8% improvement", "percent", 4, 8, "%"),
        ("between 4 and 8%", "percent", 4, 8, "%"),
        ("4 to 8 percent", "percent", 4, 8, "%"),
        ("a 2x speedup", "factor", 2, 2, "x"),
        ("3.5× faster", "factor", 3.5, 3.5, "x"),  # noqa: RUF001
        ("a twofold increase", "factor", 2, 2, "x"),
        ("a 10-fold reduction", "factor", 10, 10, "x"),
        ("three times higher", "factor", 3, 3, "x"),
        ("dosed at 50 mg", "unit", 50, 50, "mg"),
        ("over 1,000 genomes", "unit", 1000, 1000, "genomes"),
        ("n = 120 patients", "unit", 120, 120, "patients"),
        ("2.3 million people", "unit", 2_300_000, 2_300_000, "people"),
    ],
)
def test_extract_single_quantity(text: str, kind: str, low: float, high: float, unit: str) -> None:
    found = q(text)
    assert len(found) == 1, found
    quantity = found[0]
    assert (quantity.kind, quantity.low, quantity.high, quantity.unit) == (kind, low, high, unit)


def test_extract_reads_direction_from_nearby_words() -> None:
    assert q("a 40% increase in throughput")[0].direction == "up"
    assert q("throughput increased by 40%")[0].direction == "up"
    assert q("mortality decreased 12%")[0].direction == "down"
    assert q("a 12% reduction in mortality")[0].direction == "down"
    assert q("accuracy was 91%")[0].direction is None


def test_extract_ignores_years_citations_and_p_values() -> None:
    assert q("published in 2020 [12] with p < 0.05 and CI 95%") == [
        num.Quantity(kind="percent", low=95, high=95, unit="%", text="95%", direction=None)
    ]


def test_extract_returns_multiple_in_order() -> None:
    kinds = [x.kind for x in q("50 mg gave a 2x gain and a 4-8% drop")]
    assert kinds == ["unit", "factor", "percent"]


def test_percent_and_fold_are_not_comparable() -> None:
    a, b = q("a 40% speedup"), q("a 2x speedup")
    assert num.comparable(a[0], b[0]) is False


def test_unit_synonyms_are_comparable() -> None:
    a, b = q("50 milligrams"), q("50 mg")
    assert num.comparable(a[0], b[0]) is True


# --- comparison ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "source", "agrees"),
    [
        ("40%", "4-8%", False),  # the canonical case
        ("40%", "38%", True),  # within 10 %
        ("40%", "35%", False),
        ("6%", "4-8%", True),  # point inside range
        ("8.5%", "4-8%", True),  # range edge stretched by tolerance
        ("3-5%", "4-8%", True),  # overlapping ranges
        ("10-20%", "4-8%", False),
        ("2x", "2.1x", True),
        ("2x", "3x", False),
        ("twofold", "2x", True),
        ("50 mg", "50 milligrams", True),
        ("50 mg", "500 mg", False),
        ("1,000 genomes", "1000 genomes", True),
    ],
)
def test_agreement(claim: str, source: str, agrees: bool) -> None:
    assert num.agrees(q(claim)[0], q(source)[0]) is agrees


def test_opposite_directions_disagree_even_with_equal_values() -> None:
    assert num.agrees(q("a 12% increase")[0], q("a 12% decrease")[0]) is False


def test_direction_only_on_one_side_does_not_block_agreement() -> None:
    assert num.agrees(q("a 12% increase")[0], q("12%")[0]) is True


# --- routing ------------------------------------------------------------------

SOURCE = [
    Passage("We observed a 4-8% improvement in throughput.", "s", 0),
    Passage("The cohort had 120 patients.", "s", 1),
]


def test_check_reports_mismatch_with_both_figures() -> None:
    result = num.check("The method yields a 40% speedup.", SOURCE)
    assert result is not None
    assert result.mismatch is True
    assert result.passage is SOURCE[0]
    assert result.claim_text == "40%"
    assert result.source_text == "4-8%"


def test_check_is_none_when_claim_has_no_numbers() -> None:
    assert num.check("The method is faster.", SOURCE) is None


def test_check_is_none_when_nothing_comparable_in_source() -> None:
    assert num.check("Dosed at 50 mg.", SOURCE) is None


def test_check_is_consistent_when_any_comparable_figure_agrees() -> None:
    result = num.check("A 6% gain was seen.", SOURCE)
    assert result is not None
    assert result.mismatch is False


def test_check_is_conservative_when_one_of_several_figures_agrees() -> None:
    passages = [Passage("Gains were 12% in group A and 40% in group B.", "s", 0)]
    result = num.check("A 40% gain was seen.", passages)
    assert result is not None and result.mismatch is False


def test_check_declines_when_several_figures_and_none_agrees() -> None:
    passages = [Passage("Gains were 12% in group A and 20% in group B.", "s", 0)]
    assert num.check("A 40% gain was seen.", passages) is None


def test_check_reports_the_closest_single_figure_across_passages() -> None:
    passages = [Passage("Gains were 12%.", "s", 0), Passage("Gains were 20%.", "s", 1)]
    result = num.check("A 40% gain was seen.", passages)
    assert result is not None and result.mismatch is True
    assert result.source_text == "20%" and result.passage is passages[1]


# --- lessons from SciFact dev (2026-09-11): every early firing was wrong ------


def test_numbers_glued_to_letters_are_labels_not_quantities() -> None:
    assert q("H3.3 nucleosomes and the +1 nucleosome and CD4 cells") == []


def test_comparators_are_not_directions() -> None:
    assert q("less than 10% of children")[0].direction is None
    assert q("more than 40% of samples")[0].direction is None
    assert q("at least 3 times")[0].direction is None


def test_check_declines_when_the_passage_holds_several_comparable_figures() -> None:
    passages = [Passage("Survival increased from 43% in 1979 to 52% in 1996.", "s", 0)]
    # Neither 43% nor 52% is about the claim's figure; with two candidates the layer
    # cannot tell which one the claim refers to, so it must not refute.
    assert num.check("Incidence decreased by 10% in women.", passages) is None


def test_check_declines_when_the_claim_holds_several_figures_of_one_kind() -> None:
    passages = [Passage("Prevalence was 26.8%.", "s", 0)]
    assert num.check("Prevalence rose from 5% to 12% and then 30%.", passages) is None


def test_confidence_level_is_not_a_quantity() -> None:
    assert q("a change of -31% (95% CI, 8%-16%)") == [
        num.Quantity(kind="percent", low=31, high=31, unit="%", text="-31%", direction="down"),
        num.Quantity(kind="percent", low=8, high=16, unit="%", text="8%-16%", direction=None),
    ]


def test_a_change_is_not_refuted_by_a_level() -> None:
    # "decreased by 10%" is a change; "57% women" is a proportion. Different things.
    passages = [Passage("Patients were 4537 residents (57% women) with heart failure.", "s", 0)]
    assert num.check("Incidence decreased by 10% in women.", passages) is None
