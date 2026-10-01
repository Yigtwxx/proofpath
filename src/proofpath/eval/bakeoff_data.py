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
