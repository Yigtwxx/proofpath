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
    passage_offsets,
    pending,
    save_items,
    save_probs,
    save_snapshot,
    source_from_fetched,
    split_at,
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


def test_save_probs_refuses_a_block_count_that_differs_from_the_items(tmp_path: Path) -> None:
    items = [_item("a", 2), _item("b", 1)]
    with pytest.raises(ValueError, match="one probability block per item"):
        save_probs(tmp_path / "x.npz", items, [np.zeros((2, 3))], ms_per_pair=1.0)


def test_an_empty_item_list_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    save_probs(path, [], [], ms_per_pair=3.0)
    loaded, ms = load_probs(path, [])
    assert loaded == []
    assert ms == pytest.approx(3.0)


def test_the_fingerprint_changes_with_a_claim_or_a_key() -> None:
    a = _item("a")
    other_claim = Item(a.key, a.dataset, a.group, "another claim", a.gold, a.passages)
    other_key = Item("z", a.dataset, a.group, a.claim, a.gold, a.passages)
    assert items_fingerprint([a]) != items_fingerprint([other_claim])
    assert items_fingerprint([a]) != items_fingerprint([other_key])


def _save_with_offsets(path: Path, items: list[Item], offsets: list[int], rows: int) -> None:
    np.savez(
        path,
        fingerprint=np.array(items_fingerprint(items)),
        offsets=np.array(offsets, dtype=np.int64),
        probs=np.zeros((rows, 3), dtype=np.float32),
        ms_per_pair=np.array(1.0),
    )


def test_probs_with_the_wrong_number_of_offsets_are_refused(tmp_path: Path) -> None:
    items = [_item("a", 2), _item("b", 1)]
    path = tmp_path / "bad.npz"
    _save_with_offsets(path, items, [0, 2], 2)
    with pytest.raises(ValueError, match="blocks for 2 items"):
        load_probs(path, items)


def test_probs_whose_slice_does_not_match_the_passage_count_are_refused(tmp_path: Path) -> None:
    items = [_item("a", 2), _item("b", 1)]
    path = tmp_path / "bad.npz"
    _save_with_offsets(path, items, [0, 1, 3], 3)
    with pytest.raises(ValueError, match="rows for a"):
        load_probs(path, items)


def test_passage_offsets_start_at_zero_and_end_at_the_pair_count() -> None:
    offsets = passage_offsets([_item("a", 2), _item("b", 0), _item("c", 3)])
    assert offsets.tolist() == [0, 2, 2, 5]
    assert offsets.dtype == np.int64
    assert passage_offsets([]).tolist() == [0]


def test_split_at_cuts_one_block_per_offset_pair() -> None:
    flat = np.arange(15, dtype=np.float32).reshape(5, 3)
    blocks = split_at(flat, np.array([0, 2, 2, 5]))
    assert [b.shape for b in blocks] == [(2, 3), (0, 3), (3, 3)]
    assert np.array_equal(blocks[2], flat[2:5])
    assert split_at(np.zeros((0, 3)), np.array([0])) == []
