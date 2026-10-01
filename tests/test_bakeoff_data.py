"""Offline tests for the bake-off's stored inputs (no network, no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from proofpath.eval import averitec
from proofpath.eval.bakeoff_data import (
    Item,
    SnapshotClaim,
    SnapshotSource,
    items_fingerprint,
    load_items,
    load_probs,
    load_snapshot,
    pending,
    save_items,
    save_probs,
    save_snapshot,
    source_from_fetched,
)
from proofpath.fetch import Fetched, Outcome
from proofpath.models import Passage
from proofpath.providers import NO_TEXT


def _fetched(outcome: Outcome, text: str) -> Fetched:
    return Fetched(
        url="https://example.org/a",
        final_url="https://example.org/a",
        step=1,
        outcome=outcome,
        status=200 if outcome is Outcome.OK else 403,
        content_type="text/html",
        kind="html",
        body=b"",
        text=text,
        notes=[],
    )


def _item(key: str, n: int = 2, dataset: str = "scifact-dev") -> Item:
    passages = tuple(Passage(f"sentence {i} of {key}", "doc", i) for i in range(n))
    return Item(key, dataset, key, "a claim", "SUPPORTED", passages, frozenset({0}))


def test_a_read_page_keeps_its_text_and_ok_state() -> None:
    source = source_from_fetched("https://example.org/a", _fetched(Outcome.OK, "Some text."))
    assert source == SnapshotSource("https://example.org/a", "ok", "Some text.")


def test_a_reached_page_with_no_text_gets_the_no_text_state() -> None:
    source = source_from_fetched("https://example.org/a", _fetched(Outcome.OK, "   "))
    assert source.state == NO_TEXT
    assert source.text == ""


def test_a_blocked_page_keeps_its_state_and_no_text() -> None:
    blocked = _fetched(Outcome.BLOCKED_NO_BROWSER, "challenge page")
    source = source_from_fetched("https://example.org/a", blocked)
    assert source.state == Outcome.BLOCKED_NO_BROWSER.value
    assert source.text == ""


def test_snapshot_round_trips(tmp_path: Path) -> None:
    claims = [
        SnapshotClaim(
            7, "c", "Refuted", (SnapshotSource("u", "ok", "t"), SnapshotSource("v", "x", "")), 1
        )
    ]
    path = tmp_path / "snap.json"
    save_snapshot(path, claims)
    assert load_snapshot(path) == claims


def test_a_missing_snapshot_loads_as_empty(tmp_path: Path) -> None:
    assert load_snapshot(tmp_path / "absent.json") == []


def test_pending_skips_claims_already_in_the_snapshot() -> None:
    claims = [averitec.Claim(i, f"claim {i}", "Refuted", ("https://e.org",)) for i in range(3)]
    done = [SnapshotClaim(1, "claim 1", "Refuted", (), 0)]
    assert [c.id for c in pending(claims, done)] == [0, 2]


def test_items_round_trip(tmp_path: Path) -> None:
    items = [_item("a"), _item("b", 1, "averitec")]
    path = tmp_path / "items.json"
    save_items(path, items)
    assert load_items(path) == items


def test_an_unknown_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        _item("a", dataset="scifact-test")


def test_probs_round_trip(tmp_path: Path) -> None:
    items = [_item("a", 2), _item("b", 3)]
    probs = [np.full((2, 3), 0.25, np.float32), np.full((3, 3), 0.5, np.float32)]
    path = tmp_path / "base.npz"
    save_probs(path, items, probs, ms_per_pair=12.5)
    loaded, ms = load_probs(path, items)
    assert ms == pytest.approx(12.5)
    assert [p.shape for p in loaded] == [(2, 3), (3, 3)]
    assert np.allclose(loaded[1], 0.5)


def test_probs_scored_on_other_items_are_refused(tmp_path: Path) -> None:
    items = [_item("a", 2)]
    path = tmp_path / "base.npz"
    save_probs(path, items, [np.zeros((2, 3), np.float32)], ms_per_pair=1.0)
    changed = [_item("z", 2)]
    with pytest.raises(ValueError, match="other items"):
        load_probs(path, changed)


def test_probs_with_the_wrong_shape_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        save_probs(tmp_path / "x.npz", [_item("a", 2)], [np.zeros((3, 3))], ms_per_pair=1.0)


def test_the_fingerprint_changes_with_a_passage_text() -> None:
    a = _item("a")
    b = Item(a.key, a.dataset, a.group, a.claim, a.gold, (Passage("other", "doc", 0),))
    assert items_fingerprint([a]) != items_fingerprint([b])
