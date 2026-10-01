# NLI Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure three NLI models × k ∈ {1,2,3} × three aggregation rules on SciFact (calibrated on train, reported on dev) and a frozen AVeriTeC snapshot, and pick a winner by a rule fixed in advance.

**Architecture:** The pure logic lives in two new modules under `src/proofpath/eval/`:
- `bakeoff_data.py`: stored inputs and their I/O;
- `bakeoff.py`: aggregation, calibration, evaluation and the decision rule.

Both are covered by `mypy --strict` and unit tests. The dev-only script `scripts/eval_nli_bakeoff.py` does the network and model work (`snapshot`, `score`) and writes the markdown (`report`). Nothing in the product pipeline changes.

**Tech Stack:** Python 3.10+, numpy, onnxruntime through `proofpath.entailment.OnnxNli`, fastembed through `proofpath.retrieval.FastEmbedder`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md`

## Global Constraints

- Python 3.10+. Every function signature carries type annotations. `ruff check`, `ruff format --check` and `mypy` (strict, `src/`) are clean.
- `pathlib` everywhere. Every text read and write passes `encoding="utf-8"`.
- Unit tests make no network calls.
- Code, identifiers and comments are in English.
- Do not change product code (`pipeline.py`, `verify.py`, `entailment.py`, defaults). The bake-off only reads it.
- Model revisions are pinned to full 40-character SHAs:
  - base `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`
  - large `bab4bc7178836f731dcfd18c06ca9def0a137712`
  - large-fever `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`
- The AVeriTeC snapshot is fetched with the browser **not permitted** and `interactive=False`.
- Third-party page text (the snapshot) stays under `<cache>/datasets/` and never goes into the repo.
- The decision rule is the spec's, verbatim. Numbers: TIE 0.01, LARGE 0.03, NO-CHANGE 0.01, MIN_TIER_N 20, precision targets 0.85 / 0.70.
- Subagents never commit. The controller stages and commits on `feat/nli-bakeoff` after a clean review.
- The code blocks in this plan are not pre-formatted. Run `uv run ruff format <new or changed files>` before every `--check`.
- When a step says "append" to a test file, put its `import` lines in the file's top import block. A mid-file import fails ruff's E402.

**Drift from the spec (binding):** the spec put the aggregation variants "in the script". They live in `src/proofpath/eval/bakeoff.py` instead, beside `metrics.py` and `scifact.py`. That way they get `mypy --strict` and clean imports in tests. That package is eval tooling, not the product pipeline, so "no product change" still holds.

## File Structure

| File | Responsibility |
|---|---|
| Create `src/proofpath/eval/bakeoff_data.py` | `SnapshotSource`, `SnapshotClaim`, `Item`; snapshot/items JSON and probability npz I/O; `pending`, `source_from_fetched`, `items_fingerprint` |
| Create `src/proofpath/eval/bakeoff.py` | `Signal`, three aggregators, `signal_for`, `decide`, `wilson`, `calibrate`, `calibration_rows`, `evaluate_scifact`, `evaluate_averitec`, `Combo`/`Outcome`/`Choice`, `choose`, `DECISION_RULE` |
| Create `scripts/eval_nli_bakeoff.py` | `snapshot`, `score` and `report` subcommands; `CANDIDATES`; `render_report` |
| Create `tests/test_bakeoff_data.py` | I/O round trips, resume, stale-probability guards |
| Create `tests/test_bakeoff.py` | aggregation, Wilson, calibration split guard, evaluation, decision rule |
| Create `tests/test_eval_nli_bakeoff.py` | the script imports without side effects; candidate pins; `render_report` |
| Create (Task 7) `docs/eval/<date>-nli-bakeoff.md` | the result |
| Modify (Task 7) `docs/superpowers/OPEN-ITEMS.md` | §20: the outcome line |

---

### Task 1: Stored inputs — `bakeoff_data.py`

**Files:**
- Create: `src/proofpath/eval/bakeoff_data.py`
- Test: `tests/test_bakeoff_data.py`

**Interfaces:**
- Consumes:
  - `proofpath.models.Passage(text: str, source_id: str, index: int)`
  - `proofpath.fetch.Fetched` (`.ok`, `.text`, `.outcome.value`)
  - `proofpath.fetch.Outcome`
  - `proofpath.providers.NO_TEXT: str`
  - `proofpath.eval.averitec.Claim(id, text, label, source_urls, non_urls)`
- Produces:
  - `DATASETS: tuple[str, ...] = ("scifact-train", "scifact-dev", "averitec")`
  - `SnapshotSource(url: str, state: str, text: str)`
  - `SnapshotClaim(claim_id: int, claim: str, gold: str, sources: tuple[SnapshotSource, ...], non_urls: int)`
  - `Item(key: str, dataset: str, group: str, claim: str, gold: str, passages: tuple[Passage, ...], rationale: frozenset[int] = frozenset())`
  - `source_from_fetched(url: str, fetched: Fetched) -> SnapshotSource`
  - `pending(claims: Sequence[averitec.Claim], done: Sequence[SnapshotClaim]) -> list[averitec.Claim]`
  - `save_snapshot(path: Path, claims: Sequence[SnapshotClaim]) -> None` / `load_snapshot(path: Path) -> list[SnapshotClaim]`
  - `save_items(path: Path, items: Sequence[Item]) -> None` / `load_items(path: Path) -> list[Item]`
  - `items_fingerprint(items: Sequence[Item]) -> str`
  - `save_probs(path: Path, items: Sequence[Item], probs: Sequence[np.ndarray], *, ms_per_pair: float) -> None`
  - `load_probs(path: Path, items: Sequence[Item]) -> tuple[list[np.ndarray], float]`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bakeoff_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'proofpath.eval.bakeoff_data'`

- [ ] **Step 3: Write the implementation**

```python
"""Stored inputs of the NLI bake-off (docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md).

Three files under ``<cache>/datasets``, written once and read by every later step:
the frozen AVeriTeC snapshot (page text per source URL), the retrieved items (top
passages per claim, shared by every model), and one npz of raw NLI probabilities per
model. Nothing here touches the network or a model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from proofpath.eval import averitec
from proofpath.fetch import Fetched
from proofpath.models import Passage
from proofpath.providers import NO_TEXT

DATASETS: tuple[str, ...] = ("scifact-train", "scifact-dev", "averitec")


@dataclass(frozen=True)
class SnapshotSource:
    """One source URL as the snapshot run found it. ``text`` is "" unless it was read."""

    url: str
    state: str
    text: str


@dataclass(frozen=True)
class SnapshotClaim:
    claim_id: int
    claim: str
    gold: str  # the AVeriTeC label, verbatim
    sources: tuple[SnapshotSource, ...]
    non_urls: int


@dataclass(frozen=True)
class Item:
    """One (claim, source) unit with its top retrieved passages, best first.

    ``group`` is the claim an item belongs to: a SciFact pair is its own group, an
    AVeriTeC claim groups every readable source it has.
    """

    key: str
    dataset: str
    group: str
    claim: str
    gold: str
    passages: tuple[Passage, ...]
    rationale: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}")


def source_from_fetched(url: str, fetched: Fetched) -> SnapshotSource:
    """The state and text one fetch earned, worded as ``eval_averitec.state_for`` does.

    A page that was reached but yielded no text is not ``ok`` (product rule 6), and
    only a read page keeps its text: a challenge page's words are not the source.
    """
    if fetched.ok and fetched.text.strip():
        return SnapshotSource(url, fetched.outcome.value, fetched.text)
    if fetched.ok:
        return SnapshotSource(url, NO_TEXT, "")
    return SnapshotSource(url, fetched.outcome.value, "")


def pending(
    claims: Sequence[averitec.Claim], done: Sequence[SnapshotClaim]
) -> list[averitec.Claim]:
    """The claims a resumed snapshot run still has to fetch, in dataset order."""
    finished = {claim.claim_id for claim in done}
    return [claim for claim in claims if claim.id not in finished]


def _write_atomic(path: Path, payload: str) -> None:
    """Write a whole file at once, so a Ctrl-C never leaves half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def save_snapshot(path: Path, claims: Sequence[SnapshotClaim]) -> None:
    payload = [
        {
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "gold": claim.gold,
            "non_urls": claim.non_urls,
            "sources": [{"url": s.url, "state": s.state, "text": s.text} for s in claim.sources],
        }
        for claim in claims
    ]
    _write_atomic(path, json.dumps(payload, indent=1))


def load_snapshot(path: Path) -> list[SnapshotClaim]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        SnapshotClaim(
            claim_id=int(row["claim_id"]),
            claim=str(row["claim"]),
            gold=str(row["gold"]),
            sources=tuple(
                SnapshotSource(str(s["url"]), str(s["state"]), str(s["text"]))
                for s in row["sources"]
            ),
            non_urls=int(row["non_urls"]),
        )
        for row in raw
    ]


def save_items(path: Path, items: Sequence[Item]) -> None:
    payload = [
        {
            "key": item.key,
            "dataset": item.dataset,
            "group": item.group,
            "claim": item.claim,
            "gold": item.gold,
            "passages": [[p.text, p.source_id, p.index] for p in item.passages],
            "rationale": sorted(item.rationale),
        }
        for item in items
    ]
    _write_atomic(path, json.dumps(payload, indent=1))


def load_items(path: Path) -> list[Item]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Item(
            key=str(row["key"]),
            dataset=str(row["dataset"]),
            group=str(row["group"]),
            claim=str(row["claim"]),
            gold=str(row["gold"]),
            passages=tuple(Passage(str(t), str(s), int(i)) for t, s, i in row["passages"]),
            rationale=frozenset(int(i) for i in row["rationale"]),
        )
        for row in raw
    ]


def items_fingerprint(items: Sequence[Item]) -> str:
    """A digest of what was scored: keys, claims and passage texts, in order.

    Stored beside the probabilities so a rebuilt item file can never be paired with
    probabilities computed on different passages.
    """
    digest = hashlib.sha256()
    for item in items:
        digest.update(item.key.encode("utf-8") + b"\0" + item.claim.encode("utf-8") + b"\0")
        for passage in item.passages:
            digest.update(passage.text.encode("utf-8") + b"\0")
    return digest.hexdigest()


def save_probs(
    path: Path, items: Sequence[Item], probs: Sequence[np.ndarray], *, ms_per_pair: float
) -> None:
    """Store one model's ``(n_passages, 3)`` probabilities per item, flattened."""
    if len(items) != len(probs):
        raise ValueError("one probability block per item is required")
    for item, block in zip(items, probs, strict=True):
        if np.asarray(block).shape != (len(item.passages), 3):
            raise ValueError(f"probability block for {item.key} has the wrong shape")
    offsets = np.cumsum([0, *(len(item.passages) for item in items)], dtype=np.int64)
    flat = (
        np.concatenate([np.asarray(b, dtype=np.float32) for b in probs])
        if probs
        else np.zeros((0, 3), dtype=np.float32)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            fingerprint=np.array(items_fingerprint(items)),
            offsets=offsets,
            probs=flat,
            ms_per_pair=np.array(ms_per_pair, dtype=np.float64),
        )
    temporary.replace(path)


def load_probs(path: Path, items: Sequence[Item]) -> tuple[list[np.ndarray], float]:
    """The stored probabilities, one block per item, and the measured ms per pair."""
    with np.load(path, allow_pickle=False) as data:
        fingerprint = str(data["fingerprint"])
        offsets = data["offsets"]
        flat = data["probs"]
        ms_per_pair = float(data["ms_per_pair"])
    if fingerprint != items_fingerprint(items):
        raise ValueError(f"{path.name} was scored on other items; run `score` again")
    return [flat[offsets[n] : offsets[n + 1]] for n in range(len(items))], ms_per_pair
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bakeoff_data.py -q`
Expected: 12 passed

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/proofpath/eval/bakeoff_data.py tests/test_bakeoff_data.py && uv run ruff format --check src tests && uv run mypy`
Expected: no errors

---

### Task 2: Signals and aggregation — `bakeoff.py` part 1

**Files:**
- Create: `src/proofpath/eval/bakeoff.py`
- Test: `tests/test_bakeoff.py`

**Interfaces:**
- Consumes:
  - `proofpath.numerics.check(claim: str, passages: Sequence[Passage]) -> NumericResult | None` (`.mismatch`, `.passage`)
  - `proofpath.entailment.LABEL_ORDER = (SUPPORTED, REFUTED, NEI)`
  - `proofpath.pipeline.Thresholds(decide, high, medium)` with `.tier(score)`
- Produces:
  - `AGGREGATIONS = ("max", "max_nei", "margin")`
  - `KS = (1, 2, 3)`
  - `TOP_N = 5`
  - `Signal(label: Label, score: float, passage: Passage | None)`
  - `NOTHING: Signal`
  - `aggregate_max` / `aggregate_max_nei` / `aggregate_margin(passages: Sequence[Passage], probs: np.ndarray) -> Signal`
  - `AGGREGATORS: Mapping[str, Aggregator]`
  - `signal_for(claim: str, passages: Sequence[Passage], probs: np.ndarray, *, k: int, aggregation: str) -> Signal`
  - `decide(signal: Signal, thresholds: Thresholds) -> Label`
  - `wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

```python
"""Offline tests for the bake-off's aggregation, calibration and decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from proofpath.eval import bakeoff
from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds

# Rows are (SUPPORTED, REFUTED, NEI), the order of entailment.LABEL_ORDER.
P = [Passage(f"passage {i}", "doc", i) for i in range(3)]


def _probs(*rows: tuple[float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_max_takes_the_loudest_assertion_over_the_hits() -> None:
    probs = _probs((0.6, 0.1, 0.3), (0.1, 0.8, 0.1))
    signal = bakeoff.aggregate_max(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.REFUTED, P[1])
    assert signal.score == pytest.approx(0.8)


def test_max_nei_ignores_an_assertion_its_own_hit_calls_neutral() -> None:
    probs = _probs((0.45, 0.05, 0.50), (0.40, 0.05, 0.30))
    signal = bakeoff.aggregate_max_nei(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.SUPPORTED, P[1])
    assert signal.score == pytest.approx(0.40)


def test_max_nei_proposes_nothing_when_every_hit_is_neutral() -> None:
    probs = _probs((0.2, 0.1, 0.7))
    assert bakeoff.aggregate_max_nei(P[:1], probs) == bakeoff.NOTHING


def test_margin_lets_a_contradicting_passage_cancel_a_supporting_one() -> None:
    probs = _probs((0.90, 0.05, 0.05), (0.05, 0.85, 0.10))
    signal = bakeoff.aggregate_margin(P[:2], probs)
    assert (signal.label, signal.passage) == (Label.SUPPORTED, P[0])
    assert signal.score == pytest.approx(0.05)


def test_margin_with_an_exact_tie_proposes_nothing() -> None:
    probs = _probs((0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    assert bakeoff.aggregate_margin(P[:2], probs) == bakeoff.NOTHING


@pytest.mark.parametrize("name", bakeoff.AGGREGATIONS)
def test_no_hits_propose_nothing(name: str) -> None:
    assert bakeoff.AGGREGATORS[name]([], np.zeros((0, 3), np.float32)) == bakeoff.NOTHING


def test_signal_for_reads_only_the_first_k_hits() -> None:
    probs = _probs((0.2, 0.1, 0.7), (0.95, 0.0, 0.05))
    first = bakeoff.signal_for("a claim", P[:2], probs, k=1, aggregation="max")
    assert first.passage == P[0]
    both = bakeoff.signal_for("a claim", P[:2], probs, k=2, aggregation="max")
    assert both.passage == P[1]


def test_a_numeric_mismatch_decides_before_any_aggregation() -> None:
    passages = [Passage("Mortality fell by 12% in the treated group.", "doc", 0)]
    probs = _probs((0.99, 0.0, 0.01))
    signal = bakeoff.signal_for(
        "Mortality fell by 40% in the treated group.", passages, probs, k=1, aggregation="max"
    )
    assert signal == bakeoff.Signal(Label.REFUTED, 1.0, passages[0])


def test_decide_needs_the_cut_and_a_passage() -> None:
    cuts = Thresholds(decide=0.5, high=0.9, medium=0.7)
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.6, P[0]), cuts) is Label.SUPPORTED
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.4, P[0]), cuts) is Label.NEI
    assert bakeoff.decide(bakeoff.Signal(Label.SUPPORTED, 0.9, None), cuts) is Label.NEI


def test_wilson_interval_matches_known_values() -> None:
    low, high = bakeoff.wilson(0, 10)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.2775, abs=1e-4)
    low, high = bakeoff.wilson(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-4)
    assert high == pytest.approx(1.0)


def test_wilson_refuses_an_empty_sample() -> None:
    with pytest.raises(ValueError):
        bakeoff.wilson(0, 0)
```

The numeric test relies on `numerics.check` refuting 40 % against a single 12 %. If the layer stays silent on this pair, change the sentences to a pair that `tests/test_numerics.py` already shows mismatching, and copy that pair verbatim.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bakeoff.py -q`
Expected: FAIL with `ImportError: cannot import name 'bakeoff'`

- [ ] **Step 3: Write the implementation**

```python
"""NLI bake-off: aggregation variants, calibration, evaluation and the decision rule.

Pure functions over stored NLI probabilities; nothing here touches the network or a
model (docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md). A variant moves into
``pipeline.py`` only if the bake-off picks it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from proofpath import numerics
from proofpath.entailment import LABEL_ORDER
from proofpath.models import Label, Passage
from proofpath.pipeline import Thresholds

_S = LABEL_ORDER.index(Label.SUPPORTED)
_R = LABEL_ORDER.index(Label.REFUTED)
_N = LABEL_ORDER.index(Label.NEI)

# Simplest first: this order is also the tie order of the decision rule.
AGGREGATIONS: tuple[str, ...] = ("max", "max_nei", "margin")
KS: tuple[int, ...] = (1, 2, 3)
# Passages retrieved and scored per item; every k is a prefix of these.
TOP_N = 5


@dataclass(frozen=True)
class Signal:
    """The strongest assertion a combination proposes for one item, before ``decide``.

    ``label`` is SUPPORTED or REFUTED, or NEI when the aggregation proposes nothing;
    ``passage`` is what the assertion rests on.
    """

    label: Label
    score: float
    passage: Passage | None


NOTHING = Signal(Label.NEI, 0.0, None)

Aggregator = Callable[[Sequence[Passage], np.ndarray], Signal]


def aggregate_max(passages: Sequence[Passage], probs: np.ndarray) -> Signal:
    """Today's ``pipeline.aggregate``: the loudest SUPPORTED or REFUTED over the hits."""
    best = NOTHING
    for passage, row in zip(passages, probs, strict=True):
        for label, column in ((Label.SUPPORTED, _S), (Label.REFUTED, _R)):
            value = float(row[column])
            if value > best.score:
                best = Signal(label, value, passage)
    return best


def aggregate_max_nei(passages: Sequence[Passage], probs: np.ndarray) -> Signal:
    """As ``max``, but a hit only counts when its assertion beats its own p(NEI)."""
    best = NOTHING
    for passage, row in zip(passages, probs, strict=True):
        neutral = float(row[_N])
        for label, column in ((Label.SUPPORTED, _S), (Label.REFUTED, _R)):
            value = float(row[column])
            if value > neutral and value > best.score:
                best = Signal(label, value, passage)
    return best


def aggregate_margin(passages: Sequence[Passage], probs: np.ndarray) -> Signal:
    """max p(S) − max p(R): a supporting and a contradicting passage cancel out."""
    if len(passages) == 0:
        return NOTHING
    s_at = int(np.argmax(probs[:, _S]))
    r_at = int(np.argmax(probs[:, _R]))
    supported = float(probs[s_at, _S])
    refuted = float(probs[r_at, _R])
    if supported > refuted:
        return Signal(Label.SUPPORTED, supported - refuted, passages[s_at])
    if refuted > supported:
        return Signal(Label.REFUTED, refuted - supported, passages[r_at])
    return NOTHING


AGGREGATORS: Mapping[str, Aggregator] = {
    "max": aggregate_max,
    "max_nei": aggregate_max_nei,
    "margin": aggregate_margin,
}


def signal_for(
    claim: str, passages: Sequence[Passage], probs: np.ndarray, *, k: int, aggregation: str
) -> Signal:
    """What one combination proposes for one item, numeric layer first (spec §10).

    A numeric mismatch decides by rule at score 1.0, as ``pipeline.decide_indexed``
    does, so no candidate model is credited with what the rule decided.
    """
    top = list(passages[:k])
    numeric = numerics.check(claim, top)
    if numeric is not None and numeric.mismatch:
        return Signal(Label.REFUTED, 1.0, numeric.passage)
    return AGGREGATORS[aggregation](top, probs[:k])


def decide(signal: Signal, thresholds: Thresholds) -> Label:
    """The verdict a signal earns: below the cut, or with no passage, it is NEI."""
    if signal.label is Label.NEI or signal.passage is None:
        return Label.NEI
    return signal.label if signal.score >= thresholds.decide else Label.NEI


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for ``correct`` successes out of ``n``."""
    if n <= 0:
        raise ValueError("a Wilson interval needs at least one observation")
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bakeoff.py -q`
Expected: 13 passed (11 test functions; the parametrised one runs three times)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy`
Expected: no errors

---

### Task 3: Calibration and evaluation — `bakeoff.py` part 2

**Files:**
- Modify: `src/proofpath/eval/bakeoff.py` (append)
- Test: `tests/test_bakeoff.py` (append)

**Interfaces:**
- Consumes:
  - Task 1: `Item`, `SnapshotClaim`
  - Task 2: `signal_for`, `decide`, `wilson`
  - `proofpath.eval.metrics`: `Row`, `LABELS`, `accuracy`, `macro_f1`, `rationale_f1`, `sweep_decide`, `tier_cutpoints`
  - `proofpath.eval.averitec.to_label(str) -> Label | None`
- Produces:
  - `GRID: tuple[float, ...]` (0.05 … 0.95)
  - `HIGH_TARGET = 0.85`, `MEDIUM_TARGET = 0.70`, `MIN_TIER_N = 20`
  - `Scored(item: Item, probs: np.ndarray)`
  - `ship_cut(value: float) -> float`
  - `calibration_rows(scored: Sequence[Scored], *, k: int, aggregation: str) -> list[metrics.Row]`, which raises `ValueError` on any item that is not `scifact-train`
  - `calibrate(rows: Sequence[metrics.Row]) -> Thresholds`
  - `TierStat(tier: str, n: int, correct: int)` with `.precision -> float | None` (None below `MIN_TIER_N`) and `.interval -> tuple[float, float] | None`
  - `SciFactResult(n, accuracy, macro_f1, f1: dict[Label, float], rationale_f1, tiers: tuple[TierStat, ...], asserted_without_passage)`
  - `evaluate_scifact(scored, thresholds, *, k, aggregation) -> SciFactResult`, which accepts only `scifact-dev`
  - `AveritecResult(n, counted, accuracy_all, readable, accuracy_readable, majority_baseline, per_label: dict[str, tuple[int, int]], asserted_without_passage)`
  - `evaluate_averitec(claims: Sequence[SnapshotClaim], scored, thresholds, *, k, aggregation) -> AveritecResult`, which accepts only `averitec`

- [ ] **Step 1: Write the failing tests (append to `tests/test_bakeoff.py`)**

```python
from proofpath.eval.bakeoff_data import Item, SnapshotClaim, SnapshotSource


def _scored(
    key: str,
    dataset: str,
    gold: str,
    row: tuple[float, float, float],
    *,
    group: str | None = None,
    rationale: frozenset[int] = frozenset({0}),
) -> bakeoff.Scored:
    item = Item(key, dataset, group or key, "a claim", gold, (Passage("p", key, 0),), rationale)
    return bakeoff.Scored(item, _probs(row))


def test_calibration_refuses_anything_but_scifact_train() -> None:
    dev = [_scored("d", "scifact-dev", "SUPPORTED", (0.9, 0.0, 0.1))]
    with pytest.raises(ValueError, match="scifact-train"):
        bakeoff.calibration_rows(dev, k=1, aggregation="max")


def test_evaluation_refuses_the_calibration_split() -> None:
    train = [_scored("t", "scifact-train", "SUPPORTED", (0.9, 0.0, 0.1))]
    cuts = Thresholds(decide=0.5, high=0.9, medium=0.7)
    with pytest.raises(ValueError, match="scifact-dev"):
        bakeoff.evaluate_scifact(train, cuts, k=1, aggregation="max")


def test_calibrate_picks_a_cut_that_separates_right_from_wrong() -> None:
    rows = [(Label.SUPPORTED, Label.SUPPORTED, 0.9)] * 5 + [(Label.NEI, Label.SUPPORTED, 0.3)] * 5
    cuts = bakeoff.calibrate(rows)
    assert 0.3 < cuts.decide <= 0.9
    assert cuts.decide <= cuts.medium <= cuts.high


def test_tier_stats_hide_precision_below_the_minimum_n() -> None:
    assert bakeoff.TierStat("high", 19, 19).precision is None
    assert bakeoff.TierStat("high", 19, 19).interval is None
    assert bakeoff.TierStat("high", 20, 15).precision == pytest.approx(0.75)


def test_evaluate_scifact_counts_verdicts_tiers_and_rationale() -> None:
    cuts = Thresholds(decide=0.5, high=0.95, medium=0.7)
    scored = [
        _scored("a", "scifact-dev", "SUPPORTED", (0.97, 0.01, 0.02)),  # high, right
        _scored("b", "scifact-dev", "REFUTED", (0.80, 0.10, 0.10)),  # medium, wrong
        _scored("c", "scifact-dev", "NEI", (0.30, 0.10, 0.60)),  # below decide
    ]
    result = bakeoff.evaluate_scifact(scored, cuts, k=1, aggregation="max")
    assert result.n == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert {t.tier: (t.n, t.correct) for t in result.tiers} == {
        "high": (1, 1),
        "medium": (1, 0),
        "low": (0, 0),
    }
    assert result.asserted_without_passage == 0
    assert 0.0 < result.rationale_f1 <= 1.0


def test_evaluate_averitec_keeps_the_strongest_source_and_both_denominators() -> None:
    cuts = Thresholds(decide=0.5, high=0.95, medium=0.7)
    claims = [
        SnapshotClaim(1, "c1", "Refuted", (SnapshotSource("u1", "ok", "t"),), 0),
        SnapshotClaim(2, "c2", "Supported", (SnapshotSource("u2", "UNVERIFIED (x)", ""),), 0),
        SnapshotClaim(3, "c3", "Conflicting Evidence/Cherrypicking", (), 0),
    ]
    scored = [
        _scored("averitec:1:0", "averitec", "Refuted", (0.6, 0.1, 0.3), group="averitec:1"),
        _scored("averitec:1:1", "averitec", "Refuted", (0.0, 0.9, 0.1), group="averitec:1"),
    ]
    result = bakeoff.evaluate_averitec(claims, scored, cuts, k=1, aggregation="max")
    assert result.n == 3
    assert result.counted == 2  # the Conflicting claim has no 3-way label
    assert result.accuracy_all == pytest.approx(0.5)  # claim 1 right, claim 2 unread → NEI
    assert result.readable == 1
    assert result.accuracy_readable == pytest.approx(1.0)
    assert result.majority_baseline == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bakeoff.py -q`
Expected: FAIL with `AttributeError: module 'proofpath.eval.bakeoff' has no attribute 'Scored'`

- [ ] **Step 3: Write the implementation (append to `bakeoff.py`; merge the imports into the top block)**

```python
from collections import Counter

from proofpath.eval import averitec, metrics
from proofpath.eval.bakeoff_data import Item, SnapshotClaim

# Wider than eval_scifact's 0.30–0.95 grid: a ``margin`` score is a difference of two
# probabilities and lives much nearer 0. Every combination is swept on the same grid.
GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
HIGH_TARGET = 0.85
MEDIUM_TARGET = 0.70
# Below this many dev verdicts a tier's precision is reported as "too few to judge".
MIN_TIER_N = 20
CUT_DECIMALS = 6


@dataclass(frozen=True)
class Scored:
    item: Item
    probs: np.ndarray  # (len(item.passages), 3) in LABEL_ORDER


def _require(scored: Sequence[Scored], dataset: str) -> None:
    wrong = sorted({s.item.dataset for s in scored if s.item.dataset != dataset})
    if wrong:
        raise ValueError(f"expected only {dataset} items, got {', '.join(wrong)}")


def ship_cut(value: float) -> float:
    """Round a cut up to ``CUT_DECIMALS`` — the rule of ``eval_scifact.ship_cut``.

    Up, never down: a cut rounded down would hand a tier to scores the calibration
    split never showed were that good.
    """
    scale = 10**CUT_DECIMALS
    return min(1.0, math.ceil(value * scale) / scale)


def calibration_rows(scored: Sequence[Scored], *, k: int, aggregation: str) -> list[metrics.Row]:
    """``(gold, proposed label, score)`` rows for calibration, SciFact train only."""
    _require(scored, "scifact-train")
    rows: list[metrics.Row] = []
    for s in scored:
        signal = signal_for(s.item.claim, s.item.passages, s.probs, k=k, aggregation=aggregation)
        rows.append((Label(s.item.gold), signal.label, signal.score))
    return rows


def calibrate(rows: Sequence[metrics.Row]) -> Thresholds:
    """Best-accuracy ``decide``, then the 0.85 / 0.70 precision cuts above it."""
    best = metrics.sweep_decide(rows, grid=GRID)
    raw_high, raw_medium = metrics.tier_cutpoints(
        rows,
        decide=best.threshold,
        high_precision=HIGH_TARGET,
        medium_precision=MEDIUM_TARGET,
    )
    high, medium = ship_cut(raw_high), ship_cut(raw_medium)
    return Thresholds(decide=best.threshold, high=max(high, medium), medium=medium)


@dataclass(frozen=True)
class TierStat:
    tier: str
    n: int
    correct: int

    @property
    def precision(self) -> float | None:
        return self.correct / self.n if self.n >= MIN_TIER_N else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.correct, self.n) if self.n >= MIN_TIER_N else None


@dataclass(frozen=True)
class SciFactResult:
    n: int
    accuracy: float
    macro_f1: float
    f1: dict[Label, float]
    rationale_f1: float
    tiers: tuple[TierStat, ...]  # high, medium, low
    asserted_without_passage: int


def per_label_f1(gold: Sequence[Label], pred: Sequence[Label]) -> dict[Label, float]:
    scores: dict[Label, float] = {}
    for label in metrics.LABELS:
        tp = sum(g is label and p is label for g, p in zip(gold, pred, strict=True))
        fp = sum(g is not label and p is label for g, p in zip(gold, pred, strict=True))
        fn = sum(g is label and p is not label for g, p in zip(gold, pred, strict=True))
        scores[label] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return scores


def _unbacked(signal: Signal, thresholds: Thresholds) -> bool:
    """An assertion over the cut with no passage under it (product rule 1)."""
    return (
        signal.label is not Label.NEI
        and signal.score >= thresholds.decide
        and signal.passage is None
    )


def evaluate_scifact(
    scored: Sequence[Scored], thresholds: Thresholds, *, k: int, aggregation: str
) -> SciFactResult:
    _require(scored, "scifact-dev")
    gold: list[Label] = []
    pred: list[Label] = []
    gold_rationale: list[frozenset[int]] = []
    pred_rationale: list[frozenset[int]] = []
    tiers = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}
    unbacked = 0
    for s in scored:
        signal = signal_for(s.item.claim, s.item.passages, s.probs, k=k, aggregation=aggregation)
        unbacked += _unbacked(signal, thresholds)
        label = decide(signal, thresholds)
        truth = Label(s.item.gold)
        gold.append(truth)
        pred.append(label)
        gold_rationale.append(s.item.rationale)
        if label is not Label.NEI and signal.passage is not None:
            pred_rationale.append(frozenset({signal.passage.index}))
            counts = tiers[thresholds.tier(signal.score)]
            counts[0] += 1
            counts[1] += label is truth
        else:
            pred_rationale.append(frozenset())
    return SciFactResult(
        n=len(scored),
        accuracy=metrics.accuracy(gold, pred),
        macro_f1=metrics.macro_f1(gold, pred),
        f1=per_label_f1(gold, pred),
        rationale_f1=metrics.rationale_f1(gold_rationale, pred_rationale),
        tiers=tuple(TierStat(name, n, c) for name, (n, c) in tiers.items()),
        asserted_without_passage=unbacked,
    )


@dataclass(frozen=True)
class AveritecResult:
    n: int
    counted: int  # claims with a 3-way gold label
    accuracy_all: float
    readable: int  # counted claims with at least one readable source
    accuracy_readable: float
    majority_baseline: float
    per_label: dict[str, tuple[int, int]]  # gold label -> (n, correct)
    asserted_without_passage: int


def evaluate_averitec(
    claims: Sequence[SnapshotClaim],
    scored: Sequence[Scored],
    thresholds: Thresholds,
    *,
    k: int,
    aggregation: str,
) -> AveritecResult:
    """Per claim, the rule of ``eval_averitec._decide_claim``.

    The strongest decided non-NEI verdict across the claim's sources wins. A claim
    with a readable source and no assertion is NEI. A claim with nothing readable is
    unanswered, and is scored as NEI.
    """
    _require(scored, "averitec")
    by_group: dict[str, list[Scored]] = {}
    for s in scored:
        by_group.setdefault(s.item.group, []).append(s)
    per_label: dict[str, tuple[int, int]] = {}
    golds: list[Label] = []
    correct_all = readable = correct_readable = unbacked = 0
    for claim in claims:
        group = by_group.get(f"averitec:{claim.claim_id}", [])
        best: Signal | None = None
        for s in group:
            signal = signal_for(
                s.item.claim, s.item.passages, s.probs, k=k, aggregation=aggregation
            )
            unbacked += _unbacked(signal, thresholds)
            if decide(signal, thresholds) is Label.NEI:
                continue
            if best is None or signal.score > best.score:
                best = signal
        predicted = best.label if best is not None else Label.NEI
        gold = averitec.to_label(claim.gold)
        correct = gold is not None and predicted is gold
        seen, right = per_label.get(claim.gold, (0, 0))
        per_label[claim.gold] = (seen + 1, right + int(correct))
        if gold is None:
            continue
        golds.append(gold)
        correct_all += correct
        if group:
            readable += 1
            correct_readable += correct
    majority = max(Counter(golds).values(), default=0)
    return AveritecResult(
        n=len(claims),
        counted=len(golds),
        accuracy_all=correct_all / len(golds) if golds else 0.0,
        readable=readable,
        accuracy_readable=correct_readable / readable if readable else 0.0,
        majority_baseline=majority / len(golds) if golds else 0.0,
        per_label=per_label,
        asserted_without_passage=unbacked,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bakeoff.py -q`
Expected: all passed

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy`
Expected: no errors

---

### Task 4: The decision rule — `bakeoff.py` part 3

**Files:**
- Modify: `src/proofpath/eval/bakeoff.py` (append)
- Test: `tests/test_bakeoff.py` (append)

**Interfaces:**
- Consumes: `AGGREGATIONS` from Task 2.
- Produces:
  - `TIE_MARGIN = 0.01`, `LARGE_MARGIN = 0.03`, `NO_CHANGE_MARGIN = 0.01`
  - `DECISION_RULE: str`
  - `Combo(model: str, k: int, aggregation: str)` with `.label() -> str`
  - `Outcome(combo: Combo, large: bool, macro_f1: float, averitec_readable: float, asserted_without_passage: int)`
  - `Choice(winner: Combo | None, reason: str)`
  - `choose(outcomes: Sequence[Outcome], *, baseline: Combo) -> Choice`

- [ ] **Step 1: Write the failing tests (append)**

```python
BASE = bakeoff.Combo("base", 1, "max")


def _o(
    model: str, k: int, agg: str, f1: float, *, readable: float = 0.40, unbacked: int = 0
) -> bakeoff.Outcome:
    return bakeoff.Outcome(bakeoff.Combo(model, k, agg), model != "base", f1, readable, unbacked)


def test_no_change_when_nothing_beats_the_baseline_by_a_hundredth() -> None:
    choice = bakeoff.choose(
        [_o("base", 1, "max", 0.597), _o("base", 2, "max", 0.604)], baseline=BASE
    )
    assert choice.winner is None
    assert "no change" in choice.reason


def test_a_base_row_that_clears_the_margin_wins() -> None:
    choice = bakeoff.choose(
        [_o("base", 1, "max", 0.597), _o("base", 3, "margin", 0.630)], baseline=BASE
    )
    assert choice.winner == bakeoff.Combo("base", 3, "margin")


def test_a_large_row_needs_three_hundredths_over_the_best_base_row() -> None:
    rows = [
        _o("base", 1, "max", 0.597),
        _o("base", 2, "max_nei", 0.620),
        _o("large", 1, "max", 0.645),
    ]
    assert bakeoff.choose(rows, baseline=BASE).winner == bakeoff.Combo("base", 2, "max_nei")
    rows.append(_o("large-fever", 1, "max", 0.651))
    assert bakeoff.choose(rows, baseline=BASE).winner == bakeoff.Combo("large-fever", 1, "max")


def test_ties_go_to_the_smaller_k_then_the_simpler_aggregation() -> None:
    rows = [
        _o("base", 1, "max", 0.597),
        _o("base", 3, "max", 0.640),
        _o("base", 2, "margin", 0.635),
        _o("base", 2, "max_nei", 0.632),
    ]
    assert bakeoff.choose(rows, baseline=BASE).winner == bakeoff.Combo("base", 2, "max_nei")


def test_a_row_that_asserts_without_a_passage_is_never_eligible() -> None:
    rows = [_o("base", 1, "max", 0.597), _o("base", 2, "max", 0.700, unbacked=1)]
    assert bakeoff.choose(rows, baseline=BASE).winner is None


def test_a_row_worse_on_averitec_readable_is_never_eligible() -> None:
    rows = [_o("base", 1, "max", 0.597, readable=0.40), _o("base", 2, "max", 0.700, readable=0.39)]
    assert bakeoff.choose(rows, baseline=BASE).winner is None


def test_the_baseline_must_be_among_the_outcomes() -> None:
    with pytest.raises(ValueError, match="baseline"):
        bakeoff.choose([_o("base", 2, "max", 0.6)], baseline=BASE)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bakeoff.py -q`
Expected: FAIL with `AttributeError: module 'proofpath.eval.bakeoff' has no attribute 'Combo'`

- [ ] **Step 3: Write the implementation (append)**

```python
TIE_MARGIN = 0.01
LARGE_MARGIN = 0.03
NO_CHANGE_MARGIN = 0.01
# Float slack, so a difference printed as 0.030 counts as 0.03.
_EPS = 1e-9

DECISION_RULE = """\
1. **Eligible:** zero assertions without a passage, and AVeriTeC readable-subset accuracy
   not below the `base` × k=1 × `max` row.
2. **Winner:** the highest SciFact dev macro-F1 among eligible rows.
3. **Size gate:** a `large*` row wins only if it beats the best eligible `base` row by
   **≥ 0.03** macro-F1. Otherwise the best `base` row wins.
4. **Tie:** two rows within 0.01 of each other go to the smaller k, then to the simpler
   aggregation, in the order `max`, `max_nei`, `margin`.
5. If nothing beats `base` × k=1 × `max` by ≥ 0.01 macro-F1, the result is "no change".
   It is written up as such, and the next package (B, coverage) starts."""


@dataclass(frozen=True)
class Combo:
    model: str
    k: int
    aggregation: str

    def label(self) -> str:
        return f"`{self.model}` × k={self.k} × `{self.aggregation}`"


@dataclass(frozen=True)
class Outcome:
    combo: Combo
    large: bool
    macro_f1: float  # SciFact dev
    averitec_readable: float
    asserted_without_passage: int


@dataclass(frozen=True)
class Choice:
    winner: Combo | None  # None means "no change"
    reason: str


def _pick(rows: Sequence[Outcome]) -> Outcome:
    top = max(row.macro_f1 for row in rows)
    contenders = [row for row in rows if top - row.macro_f1 <= TIE_MARGIN + _EPS]
    return min(
        contenders,
        key=lambda row: (
            row.combo.k,
            AGGREGATIONS.index(row.combo.aggregation),
            -row.macro_f1,
        ),
    )


def choose(outcomes: Sequence[Outcome], *, baseline: Combo) -> Choice:
    """Apply ``DECISION_RULE`` to the measured rows."""
    by_combo = {row.combo: row for row in outcomes}
    if baseline not in by_combo:
        raise ValueError(f"the baseline {baseline.label()} is not among the outcomes")
    base = by_combo[baseline]
    eligible = [
        row
        for row in outcomes
        if row.asserted_without_passage == 0
        and row.averitec_readable >= base.averitec_readable - _EPS
    ]
    small = [row for row in eligible if not row.large]
    large = [row for row in eligible if row.large]
    if not small:
        return Choice(None, "no change: no eligible base row (the baseline itself failed rule 1)")
    candidate = _pick(small)
    if large:
        best_large = _pick(large)
        if best_large.macro_f1 - candidate.macro_f1 >= LARGE_MARGIN - _EPS:
            candidate = best_large
    gain = candidate.macro_f1 - base.macro_f1
    if gain < NO_CHANGE_MARGIN - _EPS:
        return Choice(
            None,
            f"no change: the best eligible row, {candidate.combo.label()}, is "
            f"{gain:+.3f} macro-F1 over the baseline, below {NO_CHANGE_MARGIN}",
        )
    return Choice(
        candidate.combo,
        f"{candidate.combo.label()} is {gain:+.3f} macro-F1 over {baseline.label()}",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bakeoff.py -q`
Expected: all passed

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy`
Expected: no errors

---

### Task 5: The script's `snapshot` and `score` subcommands

**Files:**
- Create: `scripts/eval_nli_bakeoff.py`
- Test: `tests/test_eval_nli_bakeoff.py`

**Interfaces:**
- Consumes:
  - Tasks 1–4
  - `proofpath.retrieval.rank`, `split_sentences`, `FastEmbedder`
  - `proofpath.entailment.OnnxNli(cache_dir=, repo_id=, revision=, onnx_file=, providers=)` and `pick_onnx_file`
  - `proofpath.verify.Engine.default(config, interactive=False, browser=False)`, then `.fetcher.fetch(url)` and `.close()`
  - `proofpath.eval.scifact.ensure_downloaded/load` and `proofpath.eval.averitec.ensure_downloaded/load`
- Produces:
  - `Candidate(repo, revision, onnx_file: Callable[[str], str], large: bool)`
  - `CANDIDATES: dict[str, Candidate]`
  - `BASELINE = bakeoff.Combo("base", 1, "max")`
  - `bakeoff_dir() -> Path`, `snapshot_path() -> Path`, `items_path() -> Path`
  - `nli_providers() -> list[str]`
  - `build_items(embedder: retrieval.Embedder) -> list[Item]`
  - `main(argv: list[str] | None = None) -> int`, with subcommands `snapshot`, `score` and `report`. `report` is a stub that returns 2 until Task 6.

- [ ] **Step 1: Write the failing tests**

```python
"""Offline tests for scripts/eval_nli_bakeoff.py's importable parts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_nli_bakeoff.py -q`
Expected: FAIL with `FileNotFoundError` (the script does not exist yet)

- [ ] **Step 3: Write the script**

```python
"""NLI bake-off harness (docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md).

Usage:
    uv run python scripts/eval_nli_bakeoff.py snapshot [--limit 100] [--sleep 1.0] [--resume]
    uv run python scripts/eval_nli_bakeoff.py score --model base|large|large-fever
        [--rebuild-items]
    uv run python scripts/eval_nli_bakeoff.py report [--out docs/eval/<date>-nli-bakeoff.md]

``snapshot`` is the only network step, apart from the one-time dataset and model
downloads. ``score`` runs one NLI model over every stored (passage, claim) pair.
``report`` is offline. Importing this module touches neither the network nor a model.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from proofpath import retrieval
from proofpath.entailment import pick_onnx_file
from proofpath.eval import averitec, bakeoff, scifact
from proofpath.eval.bakeoff_data import (
    Item,
    SnapshotClaim,
    load_items,
    load_snapshot,
    pending,
    save_items,
    save_probs,
    save_snapshot,
    source_from_fetched,
)
from proofpath.models import Passage
from proofpath.paths import cache_dir, models_dir

EMBEDDER = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Candidate:
    repo: str
    revision: str
    onnx_file: Callable[[str], str]  # platform.machine() -> file inside the repo
    large: bool


def _fever_file(_machine: str) -> str:
    # The repo ships one dynamic-int8 export for every CPU.
    return "onnx/model_quantized.onnx"


CANDIDATES: dict[str, Candidate] = {
    "base": Candidate(
        "cross-encoder/nli-deberta-v3-base",
        "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        pick_onnx_file,
        large=False,
    ),
    "large": Candidate(
        "cross-encoder/nli-deberta-v3-large",
        "bab4bc7178836f731dcfd18c06ca9def0a137712",
        pick_onnx_file,
        large=True,
    ),
    "large-fever": Candidate(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
        _fever_file,
        large=True,
    ),
}
BASELINE = bakeoff.Combo("base", 1, "max")
# Pairs handed to the scorer per call: several of its 16-pair batches, so progress
# can be printed without starving the batches.
SCORE_CHUNK = 400


def bakeoff_dir() -> Path:
    return cache_dir() / "datasets" / "bakeoff"


def snapshot_path() -> Path:
    return cache_dir() / "datasets" / "averitec_snapshot.json"


def items_path() -> Path:
    return bakeoff_dir() / "items.json"


def nli_providers() -> list[str]:
    """CUDA first where present, then CPU.

    CoreML is left out for every candidate. All three files are int8, and the
    product leaves CoreML out for int8 too (``entailment.providers_for``).
    """
    from proofpath.device import onnx_providers

    return [p for p in onnx_providers() if p != "CoreMLExecutionProvider"]


def _ranked(
    claim: str, passages: list[Passage], embedder: retrieval.Embedder
) -> tuple[Passage, ...]:
    hits = retrieval.rank(claim, passages, embedder, k=bakeoff.TOP_N)
    return tuple(hit.passage for hit in hits)


def build_items(embedder: retrieval.Embedder) -> list[Item]:
    """SciFact train and dev pairs, then every readable AVeriTeC source, top 5 each.

    Passages are split exactly as the product splits them: SciFact ships its
    sentences, and a page goes through ``retrieval.split_sentences`` as in
    ``eval_averitec._decide_claim``.
    """
    items: list[Item] = []
    tarball = scifact.ensure_downloaded(cache_dir())
    for split in ("train", "dev"):
        data = scifact.load(tarball, split=split)
        for pair in data.pairs():
            doc = data.corpus[pair.doc_id]
            passages = [Passage(t, str(doc.doc_id), i) for i, t in enumerate(doc.sentences)]
            key = f"scifact-{split}:{pair.claim_id}:{pair.doc_id}"
            items.append(
                Item(
                    key,
                    f"scifact-{split}",
                    key,
                    pair.claim,
                    pair.label.value,
                    _ranked(pair.claim, passages, embedder),
                    pair.rationale,
                )
            )
    for claim in load_snapshot(snapshot_path()):
        for position, source in enumerate(claim.sources):
            sentences = retrieval.split_sentences(source.text) if source.text.strip() else []
            if not sentences:
                continue
            passages = [Passage(s, source.url, i) for i, s in enumerate(sentences)]
            items.append(
                Item(
                    f"averitec:{claim.claim_id}:{position}",
                    "averitec",
                    f"averitec:{claim.claim_id}",
                    claim.claim,
                    claim.gold,
                    _ranked(claim.claim, passages, embedder),
                )
            )
    return items


def cmd_snapshot(args: argparse.Namespace) -> int:
    from proofpath.config import load_config
    from proofpath.verify import Engine

    path = snapshot_path()
    if path.exists() and not args.resume:
        print(
            f"error     {path} already holds a snapshot.\n"
            "          Pass --resume to continue it, or delete the file to start over.",
            file=sys.stderr,
        )
        return 2
    claims = averitec.load(averitec.ensure_downloaded(cache_dir()), limit=args.limit or None)
    stored = load_snapshot(path) if args.resume else []
    todo = pending(claims, stored)
    print(f"snapshot  {len(claims)} claims, {len(stored)} done, {len(todo)} to fetch")
    # Browser not permitted and no prompts: the default install, as in the published
    # AVeriTeC number (spec: Data).
    engine = Engine.default(load_config(), interactive=False, browser=False)
    try:
        for position, claim in enumerate(todo, start=1):
            sources = []
            for url in claim.source_urls:
                if args.sleep:
                    time.sleep(args.sleep)
                sources.append(source_from_fetched(url, engine.fetcher.fetch(url)))
            stored.append(
                SnapshotClaim(claim.id, claim.text, claim.label, tuple(sources), claim.non_urls)
            )
            save_snapshot(path, stored)
            read = sum(bool(s.text) for s in sources)
            print(
                f"  {position}/{len(todo)}  claim {claim.id}  read {read}/{len(sources)}",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print(f"\ninterrupted; {path} is usable with --resume", file=sys.stderr)
    finally:
        engine.close()
    return 0


def _items(rebuild: bool) -> list[Item]:
    if items_path().exists() and not rebuild:
        return load_items(items_path())
    embedder = retrieval.FastEmbedder(EMBEDDER, cache_dir=models_dir())
    try:
        items = build_items(embedder)
    finally:
        embedder.close()
    save_items(items_path(), items)
    print(f"items     {len(items)} written to {items_path()}")
    return items


def cmd_score(args: argparse.Namespace) -> int:
    from proofpath.entailment import OnnxNli

    if not snapshot_path().exists():
        print("error     no AVeriTeC snapshot; run `snapshot` first", file=sys.stderr)
        return 2
    items = _items(args.rebuild_items)
    candidate = CANDIDATES[args.model]
    scorer = OnnxNli(
        cache_dir=models_dir(),
        repo_id=candidate.repo,
        revision=candidate.revision,
        onnx_file=candidate.onnx_file(platform.machine()),
        providers=nli_providers(),
    )
    print(f"nli       {scorer.name}  providers={scorer.providers}")
    pairs = [(p.text, item.claim) for item in items for p in item.passages]
    try:
        started = time.perf_counter()
        blocks: list[np.ndarray] = []
        for start in range(0, len(pairs), SCORE_CHUNK):
            blocks.append(scorer.score(pairs[start : start + SCORE_CHUNK]))
            done = min(start + SCORE_CHUNK, len(pairs))
            print(
                f"  {done}/{len(pairs)} pairs  {time.perf_counter() - started:.0f}s",
                file=sys.stderr,
            )
        elapsed = time.perf_counter() - started
    finally:
        scorer.close()
    flat = np.concatenate(blocks) if blocks else np.zeros((0, 3), dtype=np.float32)
    offsets = np.cumsum([0, *(len(item.passages) for item in items)])
    probs = [flat[offsets[n] : offsets[n + 1]] for n in range(len(items))]
    ms_per_pair = elapsed / max(len(pairs), 1) * 1000
    out = bakeoff_dir() / f"{args.model}.npz"
    save_probs(out, items, probs, ms_per_pair=ms_per_pair)
    print(f"scored    {len(pairs)} pairs, {ms_per_pair:.0f} ms/pair -> {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print("error     report is implemented in Task 6", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="fetch AVeriTeC gold sources once (network)")
    snap.add_argument("--limit", type=int, default=100, help="first N dev claims")
    snap.add_argument("--sleep", type=float, default=1.0, help="seconds between fetches")
    snap.add_argument("--resume", action="store_true", help="skip claims already stored")
    score = sub.add_parser("score", help="run one NLI model over every stored pair")
    score.add_argument("--model", required=True, choices=sorted(CANDIDATES))
    score.add_argument("--rebuild-items", action="store_true", help="re-run retrieval")
    report = sub.add_parser("report", help="calibrate, evaluate and choose (offline)")
    report.add_argument("--out", default="")
    args = parser.parse_args(argv)
    commands = {"snapshot": cmd_snapshot, "score": cmd_score, "report": cmd_report}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_nli_bakeoff.py -q`
Expected: 5 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff check scripts/eval_nli_bakeoff.py tests && uv run ruff format --check scripts tests`
Expected: no errors. Wrap any line over 100 characters as ruff formats it.

- [ ] **Step 6: Smoke-test that each model loads (controller-run, network, ~1.3 GB once)**

Run:
```bash
uv run python - <<'EOF'
import platform, importlib.util, sys
spec = importlib.util.spec_from_file_location("b", "scripts/eval_nli_bakeoff.py")
m = importlib.util.module_from_spec(spec); sys.modules["b"] = m; spec.loader.exec_module(m)
from proofpath.entailment import OnnxNli
from proofpath.paths import models_dir
pairs = [("The cat sat on the mat.", "A cat is sitting."), ("The cat sat on the mat.", "No animal is present.")]
for name, c in m.CANDIDATES.items():
    s = OnnxNli(cache_dir=models_dir(), repo_id=c.repo, revision=c.revision,
                onnx_file=c.onnx_file(platform.machine()), providers=m.nli_providers())
    print(name, s.score(pairs).round(3).tolist()); s.close()
EOF
```
Expected: for every model, row 0 has its largest value in column 0 (SUPPORTED) and row 1 in column 1 (REFUTED). If `large-fever` fails to load on arm64 (no int8 kernel), stop and report back; do not substitute a different file without asking.

---

### Task 6: The `report` subcommand and `render_report`

**Files:**
- Modify: `scripts/eval_nli_bakeoff.py`. Replace `cmd_report` and add `ReportRow`, `model_size_mb` and `render_report`.
- Test: `tests/test_eval_nli_bakeoff.py` (append)

**Interfaces:**
- Consumes:
  - Tasks 1–5
  - `huggingface_hub.try_to_load_from_cache(repo_id, filename, cache_dir=, revision=)`
- Produces:
  - `ReportRow(combo: bakeoff.Combo, large: bool, thresholds: Thresholds, scifact: bakeoff.SciFactResult, averitec: bakeoff.AveritecResult)`
  - `render_report(rows, choice, *, speed: Mapping[str, float], sizes: Mapping[str, float | None], counts: Mapping[str, int], today: str, machine: str) -> str`

- [ ] **Step 1: Write the failing tests (append)**

```python
from proofpath.eval import bakeoff
from proofpath.models import Label
from proofpath.pipeline import Thresholds


def _row(model: str, k: int, agg: str, f1: float, tiers: tuple[int, int, int]) -> object:
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
        asserted_without_passage=0,
    )
    ave = bakeoff.AveritecResult(
        n=100,
        counted=89,
        accuracy_all=0.27,
        readable=61,
        accuracy_readable=0.36,
        majority_baseline=0.708,
        per_label={"Refuted": (63, 20)},
        asserted_without_passage=0,
    )
    return script.ReportRow(
        bakeoff.Combo(model, k, agg),
        model != "base",
        Thresholds(decide=0.45, high=0.99933, medium=0.457948),
        sci,
        ave,
    )


def _render(choice: bakeoff.Choice) -> str:
    rows = [
        _row("base", 1, "max", 0.597, (21, 113, 1)),
        _row("large", 2, "margin", 0.650, (30, 40, 25)),
    ]
    return script.render_report(
        rows,
        choice,
        speed={"base": 185.0, "large": 540.0},
        sizes={"base": 244.4, "large": None},
        counts={"scifact-train": 919, "scifact-dev": 340, "averitec": 100},
        today="2026-10-02",
        machine="Darwin arm64",
    )


def test_the_report_prints_the_rule_verbatim_and_the_winner() -> None:
    text = _render(bakeoff.Choice(bakeoff.Combo("large", 2, "margin"), "why"))
    assert bakeoff.DECISION_RULE in text
    assert "**Winner:** `large` × k=2 × `margin`" in text
    assert "| base | 1 | max | 0.45 |" in text
    assert "| large | 2 | margin | 0.45 |" in text


def test_the_report_says_no_change_in_words() -> None:
    text = _render(bakeoff.Choice(None, "no change: below 0.01"))
    assert "**No change.**" in text
    assert "no change: below 0.01" in text


def test_a_tier_below_the_minimum_n_is_too_few_to_judge() -> None:
    text = _render(bakeoff.Choice(None, "no change"))
    assert "too few to judge" in text


def test_an_unknown_model_size_is_a_dash_not_zero() -> None:
    text = _render(bakeoff.Choice(None, "no change"))
    assert "| large | `cross-encoder/nli-deberta-v3-large` | — |" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_eval_nli_bakeoff.py -q`
Expected: FAIL with `AttributeError: module 'eval_nli_bakeoff' has no attribute 'ReportRow'`

- [ ] **Step 3: Write the implementation**

Add `from collections.abc import Callable, Mapping, Sequence`, `from datetime import date`, `from proofpath.eval.bakeoff_data import load_probs` and `from proofpath.pipeline import Thresholds` to the imports. Then replace `cmd_report`:

```python
@dataclass(frozen=True)
class ReportRow:
    combo: bakeoff.Combo
    large: bool
    thresholds: Thresholds
    scifact: bakeoff.SciFactResult
    averitec: bakeoff.AveritecResult


def model_size_mb(model: str) -> float | None:
    """Size of the ONNX file the run used, or None when it is not in the cache."""
    from huggingface_hub import try_to_load_from_cache

    candidate = CANDIDATES[model]
    found = try_to_load_from_cache(
        candidate.repo,
        candidate.onnx_file(platform.machine()),
        cache_dir=str(models_dir()),
        revision=candidate.revision,
    )
    return Path(found).stat().st_size / 1e6 if isinstance(found, str) else None


def _md(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _tier_lines(row: ReportRow) -> list[str]:
    lines = [
        f"### {row.combo.label()}  (decide {row.thresholds.decide:.2f}, "
        f"medium {row.thresholds.medium:.6f}, high {row.thresholds.high:.6f})",
        "",
        "| tier | n | precision | 95 % Wilson |",
        "|---|---|---|---|",
    ]
    for tier in row.scifact.tiers:
        if tier.precision is None or tier.interval is None:
            lines.append(_md(tier.tier, tier.n, "too few to judge", "—"))
        else:
            low, high = tier.interval
            lines.append(_md(tier.tier, tier.n, f"{tier.precision:.3f}", f"{low:.2f}–{high:.2f}"))
    lines.append("")
    return lines


def render_report(
    rows: Sequence[ReportRow],
    choice: bakeoff.Choice,
    *,
    speed: Mapping[str, float],
    sizes: Mapping[str, float | None],
    counts: Mapping[str, int],
    today: str,
    machine: str,
) -> str:
    readable = rows[0].averitec.readable if rows else 0
    baseline_ave = rows[0].averitec.majority_baseline if rows else 0.0
    lines = [
        f"# NLI bake-off — {today}",
        "",
        "- spec: `docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md`",
        f"- calibrated on SciFact train ({counts.get('scifact-train', 0)} pairs); reported on "
        f"SciFact dev ({counts.get('scifact-dev', 0)} pairs) and AVeriTeC dev "
        f"({counts.get('averitec', 0)} claims, {readable} answerable with a readable source; "
        "frozen snapshot, browser not permitted)",
        f"- embedder `{EMBEDDER}`, top {bakeoff.TOP_N} passages per item, numeric layer on",
        f"- machine: {machine}",
        "",
        "## Decision rule (fixed before any number was seen)",
        "",
        bakeoff.DECISION_RULE,
        "",
        "## Result",
        "",
    ]
    if choice.winner is None:
        lines.append(f"**No change.** {choice.reason}")
    else:
        lines.append(f"**Winner:** {choice.winner.label()} — {choice.reason}")
    lines.extend(
        ["", "## Models", "", "| model | repo | size (MB) | ms/pair |", "|---|---|---|---|"]
    )
    for model in dict.fromkeys(row.combo.model for row in rows):
        size = sizes.get(model)
        lines.append(
            _md(
                model,
                f"`{CANDIDATES[model].repo}`",
                "—" if size is None else f"{size:.0f}",
                f"{speed.get(model, 0.0):.0f}",
            )
        )
    lines.extend(
        [
            "",
            "## Every combination",
            "",
            f"AVeriTeC majority baseline: {baseline_ave:.3f}. Published numbers to compare: "
            "0.270 (all) and 0.361 (readable), 2026-09-16.",
            "",
            "| model | k | aggregation | decide | dev acc | dev macro-F1 | F1 S / R / NEI "
            "| rationale F1 | AVeriTeC all | AVeriTeC readable | w/o passage |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        f1 = row.scifact.f1
        lines.append(
            _md(
                row.combo.model,
                row.combo.k,
                row.combo.aggregation,
                f"{row.thresholds.decide:.2f}",
                f"{row.scifact.accuracy:.3f}",
                f"{row.scifact.macro_f1:.3f}",
                f"{f1[Label.SUPPORTED]:.2f} / {f1[Label.REFUTED]:.2f} / {f1[Label.NEI]:.2f}",
                f"{row.scifact.rationale_f1:.3f}",
                f"{row.averitec.accuracy_all:.3f}",
                f"{row.averitec.accuracy_readable:.3f}",
                row.scifact.asserted_without_passage + row.averitec.asserted_without_passage,
            )
        )
    lines.extend(["", "## Tiers on SciFact dev", ""])
    shown = [BASELINE] + ([choice.winner] if choice.winner and choice.winner != BASELINE else [])
    for combo in shown:
        lines.extend(_tier_lines(next(r for r in rows if r.combo == combo)))
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    if not items_path().exists() or not snapshot_path().exists():
        print("error     run `snapshot` and `score --model base` first", file=sys.stderr)
        return 2
    items = load_items(items_path())
    snapshot = load_snapshot(snapshot_path())
    rows: list[ReportRow] = []
    speed: dict[str, float] = {}
    for model, candidate in CANDIDATES.items():
        path = bakeoff_dir() / f"{model}.npz"
        if not path.exists():
            print(f"skip      {model}: not scored", file=sys.stderr)
            continue
        probs, speed[model] = load_probs(path, items)
        scored = [bakeoff.Scored(i, p) for i, p in zip(items, probs, strict=True)]
        train = [s for s in scored if s.item.dataset == "scifact-train"]
        dev = [s for s in scored if s.item.dataset == "scifact-dev"]
        ave = [s for s in scored if s.item.dataset == "averitec"]
        for k in bakeoff.KS:
            for aggregation in bakeoff.AGGREGATIONS:
                thresholds = bakeoff.calibrate(
                    bakeoff.calibration_rows(train, k=k, aggregation=aggregation)
                )
                rows.append(
                    ReportRow(
                        bakeoff.Combo(model, k, aggregation),
                        candidate.large,
                        thresholds,
                        bakeoff.evaluate_scifact(dev, thresholds, k=k, aggregation=aggregation),
                        bakeoff.evaluate_averitec(
                            snapshot, ave, thresholds, k=k, aggregation=aggregation
                        ),
                    )
                )
    if "base" not in speed:
        print(
            "error     the baseline model is not scored; run `score --model base`", file=sys.stderr
        )
        return 2
    outcomes = [
        bakeoff.Outcome(
            r.combo,
            r.large,
            r.scifact.macro_f1,
            r.averitec.accuracy_readable,
            r.scifact.asserted_without_passage + r.averitec.asserted_without_passage,
        )
        for r in rows
    ]
    choice = bakeoff.choose(outcomes, baseline=BASELINE)
    counts = {
        "scifact-train": sum(i.dataset == "scifact-train" for i in items),
        "scifact-dev": sum(i.dataset == "scifact-dev" for i in items),
        "averitec": len(snapshot),
    }
    report = render_report(
        rows,
        choice,
        speed=speed,
        sizes={model: model_size_mb(model) for model in speed},
        counts=counts,
        today=date.today().isoformat(),
        machine=f"{platform.system()} {platform.machine()}",
    )
    out = (
        Path(args.out)
        if args.out
        else Path("docs/eval") / f"{date.today().isoformat()}-nli-bakeoff.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"written   {out}")
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_nli_bakeoff.py -q`
Expected: 9 passed

- [ ] **Step 5: Full gate**

Run: `uv run ruff check src tests scripts && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Expected: no lint or type errors; the whole suite passes.

---

### Task 7: Run the bake-off and write it up (controller)

This task is long-running measurement, not code. The controller runs it, and a reviewer checks the write-up against the numbers.

- [ ] **Step 1: Snapshot (network, about 30–60 min at `--sleep 1.0`)**

Run: `uv run python scripts/eval_nli_bakeoff.py snapshot --limit 100 --sleep 1.0 --resume`
Expected: `averitec_snapshot.json` holds 100 claims. If it is interrupted, re-run the same command.

- [ ] **Step 2: Score the baseline first**

Run: `uv run python scripts/eval_nli_bakeoff.py score --model base`
Expected: `bakeoff/base.npz` is written. The ms/pair for base should be the same order as the 185 ms/pair measured on 2026-09-12; it may be lower, because batches are now filled.

- [ ] **Step 3: Sanity-check before spending hours on the large models**

Run: `uv run python scripts/eval_nli_bakeoff.py report --out /tmp/bakeoff-base.md`
Expected: the `base` × k=1 × `max` row has a dev accuracy near 0.609. It will not match exactly: the calibration now comes from train, not dev. If it is off by more than 0.05, stop and investigate with superpowers:systematic-debugging before going on.

- [ ] **Step 4: Score both large models (about 1 hour each on Apple Silicon CPU)**

Run: `uv run python scripts/eval_nli_bakeoff.py score --model large && uv run python scripts/eval_nli_bakeoff.py score --model large-fever`

- [ ] **Step 5: Write the report**

Run: `uv run python scripts/eval_nli_bakeoff.py report`
Expected: `docs/eval/<date>-nli-bakeoff.md`, with the winner or "No change" stated by the rule.

- [ ] **Step 6: Record the outcome**

Append one row to the backlog table in OPEN-ITEMS §20:

`| 20.11 | NLI bake-off result | <winner or "no change">, <one line of numbers>; docs/eval/<date>-nli-bakeoff.md. Next: <follow-up spec, or package B> |`

Change no README, spec §14 or product numbers here. That belongs to the follow-up spec.

- [ ] **Step 7: Final gate and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Then the controller commits:
- the script, the two modules, the tests, the report and OPEN-ITEMS;
- on `feat/nli-bakeoff`, with the message `feat(eval): add the NLI bake-off harness and its first result`.

Push and merge only with the user's go.
