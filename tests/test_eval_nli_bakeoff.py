"""Offline tests for scripts/eval_nli_bakeoff.py's importable parts."""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

from proofpath.eval import bakeoff
from proofpath.models import Label
from proofpath.pipeline import Thresholds

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_nli_bakeoff.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_nli_bakeoff", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_every_candidate_is_pinned_to_a_full_revision() -> None:
    assert set(script.CANDIDATES) == {"base", "large", "large-fever"}
    for candidate in script.CANDIDATES.values():
        assert len(candidate.revision) == 40
        int(candidate.revision, 16)


def test_only_the_large_models_are_marked_large() -> None:
    assert {name for name, c in script.CANDIDATES.items() if c.large} == {"large", "large-fever"}


def test_each_candidate_names_an_onnx_file_per_machine() -> None:
    for machine in ("arm64", "x86_64"):
        for candidate in script.CANDIDATES.values():
            assert candidate.onnx_file(machine).startswith("onnx/")
            assert candidate.onnx_file(machine).endswith(".onnx")


def test_the_baseline_is_base_k1_max() -> None:
    assert (script.BASELINE.model, script.BASELINE.k, script.BASELINE.aggregation) == (
        "base",
        1,
        "max",
    )


def test_coreml_is_never_handed_to_the_int8_models() -> None:
    assert "CoreMLExecutionProvider" not in script.nli_providers()


def _row(
    model: str,
    k: int,
    agg: str,
    f1: float,
    tiers: tuple[int, int, int],
    unbacked: int = 0,
) -> object:
    sci = bakeoff.SciFactResult(
        n=340,
        accuracy=f1 + 0.01,
        macro_f1=f1,
        f1={Label.SUPPORTED: 0.7, Label.REFUTED: 0.5, Label.NEI: 0.6},
        rationale_f1=0.3,
        tiers=(
            bakeoff.TierStat("high", tiers[0], tiers[0]),
            bakeoff.TierStat("medium", tiers[1], tiers[1] // 2),
            bakeoff.TierStat("low", tiers[2], 0),
        ),
        asserted_without_passage=unbacked,
    )
    ave = bakeoff.AveritecResult(
        n=100,
        counted=89,
        accuracy_all=0.27,
        readable=61,
        accuracy_readable=0.36,
        majority_baseline=0.708,
        per_label={"Refuted": (63, 20), "Supported": (10, 7)},
        asserted_without_passage=0,
    )
    return script.ReportRow(
        bakeoff.Combo(model, k, agg),
        model != "base",
        Thresholds(decide=0.45, high=0.99933, medium=0.457948),
        sci,
        ave,
    )


def _render(
    choice: bakeoff.Choice,
    unbacked: int = 0,
    models: tuple[str, ...] = ("base", "large", "large-fever"),
) -> str:
    rows = [
        _row("base", 1, "max", 0.597, (21, 113, 1)),
        _row("large", 2, "margin", 0.650, (30, 40, 25), unbacked),
        _row("large-fever", 1, "max", 0.600, (25, 50, 20)),
    ]
    rows = [r for r in rows if r.combo.model in models]
    return script.render_report(
        rows,
        choice,
        speed={"base": 185.0, "large": 540.0, "large-fever": 540.0},
        sizes={"base": 244.4, "large": None},
        counts={"scifact-train": 919, "scifact-dev": 340, "averitec": 100},
        today="2026-10-02",
        machine="Darwin arm64",
    )


def test_the_report_prints_the_rule_verbatim_and_the_winner() -> None:
    text = _render(bakeoff.Choice(bakeoff.Combo("large", 2, "margin"), "why"))
    assert bakeoff.DECISION_RULE in text
    assert "**Winner:** `large` × k=2 × `margin`" in text  # noqa: RUF001
    assert "| base | 1 | max | 0.45 |" in text
    assert "| large | 2 | margin | 0.45 |" in text


def test_the_report_says_no_change_in_words() -> None:
    text = _render(bakeoff.Choice(None, "no change: below 0.01"))
    assert "**No change.**" in text
    assert "no change: below 0.01" in text


def test_a_tier_below_the_minimum_n_is_too_few_to_judge() -> None:
    text = _render(bakeoff.Choice(None, "no change"))
    assert "| 1 · too few to judge |" in text


def test_a_tier_at_or_above_the_minimum_n_shows_precision_and_wilson() -> None:
    text = _render(bakeoff.Choice(None, "no change"))
    low, high = bakeoff.wilson(21, 21)
    assert f"21 · 1.000 (Wilson {low:.2f}–{high:.2f})" in text  # noqa: RUF001
    assert "| base | 1 | max | 21 · 1.000" in text


def test_unscored_models_are_listed_in_the_report() -> None:
    text = _render(bakeoff.Choice(None, "x"), models=("base",))
    assert "not scored: large, large-fever" in text


def test_a_complete_run_says_all_candidates_were_scored() -> None:
    text = _render(bakeoff.Choice(None, "x"))
    assert set(script.CANDIDATES) == {"base", "large", "large-fever"}
    assert "all candidate models were scored" in text
    assert "not scored:" not in text


def test_a_missing_candidate_is_named_not_scored() -> None:
    text = _render(bakeoff.Choice(None, "x"), models=("base", "large"))
    assert "not scored: large-fever" in text
    assert "all candidate models were scored" not in text


def test_the_calibrated_cut_points_print_at_six_decimals() -> None:
    text = _render(bakeoff.Choice(None, "x"))
    assert "| 0.45 | 0.457948 | 0.999330 |" in text


def test_a_gold_label_without_a_verdict_is_not_scored_rather_than_wrong() -> None:
    rows = [_row("base", 1, "max", 0.597, (21, 113, 1))]
    rows[0].averitec.per_label["Conflicting Evidence/Cherrypicking"] = (5, 0)
    text = script.render_report(
        rows,
        bakeoff.Choice(None, "x"),
        speed={"base": 185.0},
        sizes={"base": None},
        counts={"averitec": 100},
        today="2026-10-02",
        machine="Darwin arm64",
    )
    assert "5 (not scored)" in text
    assert "0/5" not in text
    assert "not a model failure" in text


def test_the_averitec_header_states_every_denominator() -> None:
    text = _render(bakeoff.Choice(None, "x"))
    assert "100 claims, 89 answerable (3-way gold label), 61 of those with a readable" in text


def test_averitec_per_label_counts_cover_every_combination() -> None:
    text = _render(bakeoff.Choice(None, "x"))
    assert "## AVeriTeC per label" in text
    assert "| model | k | aggregation | Refuted | Supported |" in text
    assert "| base | 1 | max | 20/63 | 7/10 |" in text
    assert "| large | 2 | margin | 20/63 | 7/10 |" in text


def test_a_nonzero_without_passage_count_is_flagged() -> None:
    text = _render(bakeoff.Choice(None, "x"), unbacked=3)
    assert "**3 — rule 1 violated**" in text
    assert "| 0 |" in text


def test_a_clean_run_has_no_rule_1_flag() -> None:
    assert "rule 1 violated" not in _render(bakeoff.Choice(None, "x"))


def test_an_unknown_model_size_is_a_dash_not_zero() -> None:
    text = _render(bakeoff.Choice(None, "no change"))
    assert "| large | `cross-encoder/nli-deberta-v3-large` | — |" in text


def _touch(path: Path, mtime: float) -> Path:
    path.write_text("[]", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_items_are_stale_when_the_snapshot_is_newer(tmp_path: Path) -> None:
    items = _touch(tmp_path / "items.json", 1_000)
    snapshot = _touch(tmp_path / "snapshot.json", 2_000)
    assert script.items_stale(items, snapshot) is True


def test_items_are_fresh_when_built_after_the_snapshot(tmp_path: Path) -> None:
    items = _touch(tmp_path / "items.json", 2_000)
    snapshot = _touch(tmp_path / "snapshot.json", 1_000)
    assert script.items_stale(items, snapshot) is False


def test_report_with_a_stale_npz_names_the_model_and_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    items = _touch(tmp_path / "items.json", 2_000)
    snapshot = _touch(tmp_path / "snapshot.json", 1_000)
    (tmp_path / "base.npz").write_bytes(b"")
    monkeypatch.setattr(script, "items_path", lambda: items)
    monkeypatch.setattr(script, "snapshot_path", lambda: snapshot)
    monkeypatch.setattr(script, "bakeoff_dir", lambda: tmp_path)
    monkeypatch.setattr(script, "load_items", lambda _p: [])
    monkeypatch.setattr(script, "load_snapshot", lambda _p: [])

    def _stale(_path: Path, _items: object) -> object:
        raise ValueError("base.npz was scored on other items; run `score` again")

    monkeypatch.setattr(script, "load_probs", _stale)
    assert script.cmd_report(Namespace(out="")) == 2
    err = capsys.readouterr().err
    assert "base" in err
    assert "score --model base" in err
    assert "Traceback" not in err


def test_report_with_stale_items_does_not_advise_rebuild_items_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    items = _touch(tmp_path / "items.json", 1_000)
    snapshot = _touch(tmp_path / "snapshot.json", 2_000)
    monkeypatch.setattr(script, "items_path", lambda: items)
    monkeypatch.setattr(script, "snapshot_path", lambda: snapshot)
    assert script.cmd_report(Namespace(out="")) == 2
    err = capsys.readouterr().err
    assert "--rebuild-items" not in err
    assert "score --model <name>" in err
