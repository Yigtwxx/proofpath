"""Offline tests for scripts/eval_nli_bakeoff.py's importable parts."""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from proofpath.eval import averitec, bakeoff
from proofpath.eval.bakeoff_data import Item, SnapshotClaim, SnapshotSource
from proofpath.models import Label, Passage
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
    *,
    decide: float = 0.45,
    high: float = 0.99933,
    readable: float = 0.36,
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
        accuracy_readable=readable,
        majority_baseline=0.708,
        per_label={"Refuted": (63, 20), "Supported": (10, 7)},
        asserted_without_passage=0,
    )
    return script.ReportRow(
        bakeoff.Combo(model, k, agg),
        model != "base",
        Thresholds(decide=decide, high=high, medium=0.457948),
        sci,
        ave,
    )


NUMERIC = {
    1: script.NumericLayer(train=(4, 0), dev=(1, 1)),
    2: script.NumericLayer(train=(5, 0), dev=(1, 1)),
}


def _render(
    choice: bakeoff.Choice,
    unbacked: int = 0,
    models: tuple[str, ...] = ("base", "large", "large-fever"),
    rows: list[object] | None = None,
) -> str:
    if rows is None:
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
        numeric=NUMERIC,
        today="2026-10-02",
        machine="Darwin arm64",
    )


def _section(text: str, heading: str) -> str:
    """The body of one ``## heading`` section, up to the next one."""
    body = text.split(f"## {heading}\n", 1)[1]
    return body.split("\n## ", 1)[0]


def test_the_winner_is_named_once_when_the_reason_starts_with_it() -> None:
    winner = bakeoff.Combo("large", 2, "margin")
    reason = f"{winner.label()} is +0.053 macro-F1 over {script.BASELINE.label()}"
    text = _render(bakeoff.Choice(winner, reason))
    result = _section(text, "Result")
    assert result.count(winner.label()) == 1
    assert f"**Winner:** {reason}." in result


def test_a_reason_that_does_not_start_with_the_winner_keeps_the_label() -> None:
    winner = bakeoff.Combo("large", 2, "margin")
    text = _render(bakeoff.Choice(winner, "the baseline itself is not eligible: x"))
    assert f"**Winner:** {winner.label()} — the baseline itself is not eligible: x." in text


def test_the_result_states_the_gain_over_the_best_eligible_base_row() -> None:
    rows = [
        _row("base", 1, "max", 0.597, (21, 113, 1)),
        _row("base", 2, "max", 0.610, (1, 10, 10)),
        # Higher macro-F1, but worse on AVeriTeC readable: not eligible.
        _row("base", 3, "max", 0.640, (1, 10, 10), readable=0.30),
        # Higher macro-F1, but asserts without a passage: not eligible.
        _row("base", 2, "margin", 0.645, (1, 10, 10), unbacked=1),
        _row("large", 2, "margin", 0.650, (30, 40, 25)),
    ]
    text = _render(bakeoff.Choice(bakeoff.Combo("large", 2, "margin"), "why"), rows=rows)
    result = _section(text, "Result")
    best = bakeoff.Combo("base", 2, "max").label()
    assert f"Over the best eligible `base` row, {best} (0.610), the winner is +0.040" in result


def test_a_base_winner_that_is_the_best_base_row_says_so() -> None:
    rows = [
        _row("base", 1, "max", 0.597, (21, 113, 1)),
        _row("base", 2, "max", 0.620, (1, 10, 10)),
    ]
    text = _render(bakeoff.Choice(bakeoff.Combo("base", 2, "max"), "why"), rows=rows)
    assert "The winner is itself the best eligible `base` row." in text


def test_no_change_has_no_gain_line() -> None:
    result = _section(_render(bakeoff.Choice(None, "no change: x")), "Result")
    assert "best eligible `base` row" not in result


def test_a_decide_cut_on_the_grid_floor_is_marked() -> None:
    floor = bakeoff.GRID[0]
    rows = [
        _row("base", 1, "max", 0.597, (21, 113, 1)),
        _row("large", 1, "max", 0.600, (1, 10, 10), decide=floor),
    ]
    text = _render(bakeoff.Choice(None, "x"), rows=rows)
    assert f"| large | 1 | max | {floor:.2f} (grid floor) |" in text
    assert "| base | 1 | max | 0.45 |" in text
    assert f"1 of 2 combinations put `decide` on the grid floor ({floor:.2f})" in text


def test_no_grid_floor_cut_is_said_in_words() -> None:
    text = _render(bakeoff.Choice(None, "x"))
    assert "(grid floor) |" not in text
    assert "No combination puts `decide` on the grid floor" in text


def test_the_numeric_layer_table_has_one_row_per_k() -> None:
    section = _section(_render(bakeoff.Choice(None, "x")), "Numeric layer")
    assert "| k | SciFact train fired | train correct | SciFact dev fired | dev correct |" in (
        section
    )
    assert "| 1 | 4 | 0 | 1 | 1 |" in section
    assert "| 2 | 5 | 0 | 1 | 1 |" in section


def test_a_high_tier_made_only_of_rule_firings_is_said_to_come_from_the_rule() -> None:
    rows = [
        # high cut 1.0, one dev high verdict, one dev firing at k=1: the rule's.
        _row("base", 1, "max", 0.597, (1, 113, 1), high=1.0),
        # high cut 1.0 but two dev high verdicts against one firing: not only the rule.
        _row("base", 2, "max", 0.600, (2, 10, 10), high=1.0),
        # high cut below 1.0 with model verdicts in the tier.
        _row("large", 1, "max", 0.600, (25, 50, 20)),
    ]
    section = _section(_render(bakeoff.Choice(None, "x"), rows=rows), "Numeric layer")
    assert "The `high` cut is 1.000000 in 2 of 3 combinations." in section
    assert "In 1 of 3, every SciFact dev `high` verdict is a rule firing" in section
    assert "comes from the rule, not the model" in section


def test_the_limits_state_dev_optimism_with_the_maximum_over_all_rows() -> None:
    text = _render(bakeoff.Choice(bakeoff.Combo("large", 2, "margin"), "why"))
    limits = _section(text, "Limits")
    assert "The winner's 0.650 macro-F1 comes from the split that chose it" in limits
    assert "out of 3 combinations whose highest dev macro-F1 is 0.650" in limits


def test_the_limits_state_batch_dependence_snapshot_drift_and_grid_floor() -> None:
    limits = _section(_render(bakeoff.Choice(None, "x")), "Limits")
    assert "up to 0.197 (mean 0.008) and flipped 1 argmax" in limits
    assert "measured 2026-10-01 by the final review" in limits
    assert "reaches dev accuracy 0.607 here" in limits
    assert "against the 0.609 published on 2026-09-12" in limits
    assert "then 61 of 89 answerable claims were readable, now 61 of 89" in limits
    assert "**Grid floor.**" in limits


def test_the_old_averitec_numbers_carry_their_denominators_not_a_comparison() -> None:
    text = _render(bakeoff.Choice(None, "x"))
    assert "Published numbers to compare" not in text
    assert "on a different snapshot with different cuts" in text
    assert "over 89 answerable / 61 readable claims; this run counts 89 answerable" in text
    assert "partly a coverage change" in text


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
        numeric=NUMERIC,
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


def test_split_blocks_cuts_the_flat_scores_per_item() -> None:
    items = [_item("a", 2), _item("b", 0), _item("c", 3)]
    flat = np.arange(15, dtype=np.float32).reshape(5, 3)
    blocks = script.split_blocks(flat, items)
    assert [b.shape for b in blocks] == [(2, 3), (0, 3), (3, 3)]
    assert np.array_equal(blocks[2], flat[2:5])


def test_split_blocks_refuses_a_row_count_that_differs_from_the_passages() -> None:
    with pytest.raises(ValueError, match="score rows"):
        script.split_blocks(np.zeros((4, 3), np.float32), [_item("a", 2)])


def _item(key: str, n: int) -> Item:
    passages = tuple(Passage(f"s{i}", "doc", i) for i in range(n))
    return Item(key, "scifact-dev", key, "claim", "SUPPORTED", passages)


class _FakeEmbedder:
    """Deterministic unit vectors from the words of a text; no model."""

    name = "fake"
    dim = 16

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, sum(map(ord, word)) % self.dim] += 1.0
            out[row, 0] += 0.01
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_averitec_items_group_by_claim_skip_blank_sources_and_key_by_position() -> None:
    snapshot = [
        SnapshotClaim(
            7,
            "the claim",
            "Refuted",
            (
                SnapshotSource("https://a", "ok", "First sentence. Second sentence."),
                SnapshotSource("https://b", "blocked", ""),
                SnapshotSource("https://c", "ok", "   "),
                SnapshotSource("https://d", "ok", "Only one."),
            ),
            0,
        )
    ]
    items = script.averitec_items(snapshot, _FakeEmbedder())
    assert [i.key for i in items] == ["averitec:7:0", "averitec:7:3"]
    assert {i.group for i in items} == {"averitec:7"}
    assert {i.dataset for i in items} == {"averitec"}
    assert items[0].claim == "the claim"
    assert items[0].gold == "Refuted"
    assert {p.source_id for p in items[0].passages} == {"https://a"}
    assert [p.source_id for p in items[1].passages] == ["https://d"]
    assert len(items[0].passages) <= bakeoff.TOP_N


def test_snapshot_returns_130_after_an_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from proofpath import verify

    closed: list[bool] = []

    class _Fetcher:
        def fetch(self, _url: str) -> object:
            raise KeyboardInterrupt

    class _Engine:
        fetcher = _Fetcher()

        def close(self) -> None:
            closed.append(True)

    claim = averitec.Claim(1, "c", "Refuted", ("https://e.org",))
    monkeypatch.setattr(script, "snapshot_path", lambda: tmp_path / "snap.json")
    monkeypatch.setattr(script.averitec, "ensure_downloaded", lambda _d: tmp_path)
    monkeypatch.setattr(script.averitec, "load", lambda _p, limit=None: [claim])
    monkeypatch.setattr(verify.Engine, "default", lambda *_a, **_k: _Engine())
    monkeypatch.setattr("proofpath.config.load_config", lambda *_a, **_k: None)
    code = script.cmd_snapshot(Namespace(resume=False, limit=0, sleep=0.0))
    assert code == 130
    assert closed == [True]
