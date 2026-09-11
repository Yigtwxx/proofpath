"""Product rule 1: never assert without a passage. Enforced by the type."""

from __future__ import annotations

import pytest

from proofpath.models import Label, Passage, Verdict

PASSAGE = Passage(text="we observed a 4-8% improvement", source_id="doc-1", index=3)


@pytest.mark.parametrize("label", [Label.SUPPORTED, Label.REFUTED])
def test_supported_and_refuted_require_a_passage(label: Label) -> None:
    with pytest.raises(ValueError, match="passage"):
        Verdict(label=label, score=0.9, tier="high", passage=None)


def test_nei_may_carry_no_passage() -> None:
    verdict = Verdict(label=Label.NEI, score=0.4, tier="low", passage=None)
    assert verdict.passage is None


def test_verdict_with_passage_is_valid() -> None:
    verdict = Verdict(label=Label.SUPPORTED, score=0.9, tier="high", passage=PASSAGE)
    assert verdict.passage is PASSAGE


def test_tier_must_be_one_of_three() -> None:
    with pytest.raises(ValueError, match="tier"):
        Verdict(label=Label.NEI, score=0.1, tier="great", passage=None)  # type: ignore[arg-type]


def test_score_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="score"):
        Verdict(label=Label.NEI, score=1.5, tier="low", passage=None)
