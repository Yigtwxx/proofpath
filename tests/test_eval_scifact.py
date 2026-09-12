"""Offline tests for scripts/eval_scifact.py's importable functions.

Only the pure half is covered here: ``tier_bands`` buckets already-scored rows by
score and counts how often the proposed label was the gold one, and ``render_tiers``
turns those buckets plus the cut-points into markdown. Neither touches a model, the
dataset tarball or the network — the rows are hand-made, the way
``tests/test_metrics.py`` makes them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from proofpath.eval import metrics
from proofpath.models import Label

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_scifact.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_scifact", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the script's dataclasses resolve their annotations
    # through ``sys.modules[__module__]``, which is not there for an unregistered spec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_scifact = _load_script()
ship_cut = eval_scifact.ship_cut
tier_bands = eval_scifact.tier_bands
render_tiers = eval_scifact.render_tiers

S, R, N = Label.SUPPORTED, Label.REFUTED, Label.NEI

# (gold, strongest non-NEI label proposed, its score); one row per band, plus one
# below any plausible decide threshold.
ROWS: list[metrics.Row] = [
    (S, S, 0.95),
    (S, S, 0.92),
    (N, S, 0.85),
    (R, R, 0.75),
    (S, R, 0.65),
    (N, S, 0.55),
    (S, S, 0.40),
]


# --- ship_cut ------------------------------------------------------------


def test_a_cut_is_rounded_up_so_pasting_it_never_loosens_the_tier() -> None:
    assert ship_cut(0.4571234) == pytest.approx(0.457124)
    assert ship_cut(0.75) == pytest.approx(0.75)


def test_a_cut_never_rounds_past_one() -> None:
    assert ship_cut(1.0) == 1.0


def test_saturated_scores_keep_their_distance_from_one() -> None:
    # Two decimals would call this 1.00 and hide a reachable high tier.
    assert ship_cut(0.9991234) < 1.0


# --- tier_bands ----------------------------------------------------------


def test_bands_run_from_decide_to_one_and_report_precision_per_band() -> None:
    bands = tier_bands(ROWS, decide=0.45)
    assert [(b.low, b.high, b.n) for b in bands] == [
        (0.45, 0.6, 1),
        (0.6, 0.7, 1),
        (0.7, 0.8, 1),
        (0.8, 0.9, 1),
        (0.9, 1.0, 2),
    ]
    assert [b.precision for b in bands] == pytest.approx([0.0, 0.0, 1.0, 0.0, 1.0])


def test_rows_below_decide_are_not_counted() -> None:
    # The 0.40 row is never decided, so it belongs to no band.
    assert sum(b.n for b in tier_bands(ROWS, decide=0.45)) == len(ROWS) - 1


def test_a_decide_above_an_edge_drops_the_bands_below_it() -> None:
    bands = tier_bands(ROWS, decide=0.75)
    assert [(b.low, b.high, b.n) for b in bands] == [(0.75, 0.8, 1), (0.8, 0.9, 1), (0.9, 1.0, 2)]


def test_the_last_band_is_closed_so_a_score_of_one_lands_in_it() -> None:
    bands = tier_bands([(R, R, 1.0)], decide=0.45)
    assert bands[-1].closed and bands[-1].n == 1
    assert not bands[0].closed


def test_an_empty_band_reports_no_precision_rather_than_zero() -> None:
    bands = tier_bands([(S, S, 0.95)], decide=0.45)
    assert [b.n for b in bands] == [0, 0, 0, 0, 1]
    assert bands[0].precision is None
    assert bands[-1].precision == pytest.approx(1.0)


def test_band_labels_name_their_interval() -> None:
    bands = tier_bands(ROWS, decide=0.45)
    assert bands[0].label() == "[0.45, 0.60)"
    assert bands[-1].label() == "[0.90, 1.00]"


# --- render_tiers --------------------------------------------------------


def test_render_lists_every_band_and_one_row_per_precision_target() -> None:
    lines = render_tiers(ROWS, k=1, decide=0.45, targets=((0.85, 0.70), (0.80, 0.65)))
    text = "\n".join(lines)
    assert "## Confidence tiers (k=1, decide=0.45)" in text
    assert "| band | n | precision |" in text
    for band in tier_bands(ROWS, decide=0.45):
        assert f"| {band.label()} |" in text
    assert "| 0.85 | 0.70 |" in text
    assert "| 0.80 | 0.65 |" in text


def test_render_counts_the_verdicts_each_cut_would_reach() -> None:
    rows: list[metrics.Row] = [(S, S, 0.95), (S, S, 0.85), (S, S, 0.75), (N, R, 0.55)]
    text = "\n".join(render_tiers(rows, k=1, decide=0.45, targets=((0.80, 0.65),)))
    # high holds down to 0.75 (3 of the 4 rows), medium down to 0.55 (all 4).
    assert "| 0.80 | 0.65 | 0.750000 | 3 | 0.550000 | 4 |" in text


def test_render_says_plainly_when_high_is_unreachable() -> None:
    # Every band holds a wrong answer, so no cut ever reaches 0.80 precision.
    rows: list[metrics.Row] = [(N, S, 0.95), (N, S, 0.75), (N, R, 0.55)]
    text = "\n".join(render_tiers(rows, k=1, decide=0.45, targets=((0.80, 0.65),)))
    assert "unreachable" in text
    assert "| 0.80 | 0.65 | 1.000000 | 0 |" in text


def test_a_cut_of_one_that_still_catches_a_verdict_is_not_called_unreachable() -> None:
    """A rule-decided verdict scores exactly 1.0 (spec section 10), so a cut of 1.0
    can be reached. Unreachable means nothing reaches it, not that the number is 1."""
    rows: list[metrics.Row] = [(R, R, 1.0), (N, S, 0.95), (N, S, 0.75)]
    text = "\n".join(render_tiers(rows, k=1, decide=0.45, targets=((0.80, 0.65),)))
    assert "| 0.80 | 0.65 | 1.000000 | 1 |" in text
    assert "unreachable" not in text


def test_render_warns_when_medium_sits_on_top_of_decide() -> None:
    # medium within a hair of decide means every asserted verdict is medium or better.
    rows: list[metrics.Row] = [(S, S, 0.95), (S, S, 0.85), (S, S, 0.46)]
    text = "\n".join(render_tiers(rows, k=1, decide=0.45, targets=((0.80, 0.65),)))
    assert "`low` is effectively empty" in text


def test_the_low_tier_warning_is_printed_once_for_two_targets_sharing_a_cut() -> None:
    rows: list[metrics.Row] = [(S, S, 0.95), (S, S, 0.85), (S, S, 0.46)]
    lines = render_tiers(rows, k=1, decide=0.45, targets=((0.85, 0.70), (0.80, 0.65)))
    assert sum("`low` is effectively empty" in line for line in lines) == 1


def test_render_does_not_warn_when_low_holds_a_real_share_of_verdicts() -> None:
    rows: list[metrics.Row] = [(S, S, 0.95), (N, R, 0.85), (S, S, 0.75), (N, R, 0.46)]
    text = "\n".join(render_tiers(rows, k=1, decide=0.45, targets=((0.80, 0.65),)))
    assert "`low` is effectively empty" not in text


def test_render_does_not_call_a_reached_cut_unreachable() -> None:
    rows: list[metrics.Row] = [(S, S, 0.95), (S, S, 0.85), (S, S, 0.75), (N, R, 0.55)]
    text = "\n".join(render_tiers(rows, k=1, decide=0.45, targets=((0.80, 0.65),)))
    assert "unreachable" not in text


def test_render_shows_an_empty_band_as_a_dash() -> None:
    text = "\n".join(render_tiers([(S, S, 0.95)], k=1, decide=0.45, targets=((0.80, 0.65),)))
    assert "| [0.45, 0.60) | 0 | — |" in text
