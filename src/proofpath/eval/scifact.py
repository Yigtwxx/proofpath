"""SciFact (Wadden et al., 2020) as claim / cited-abstract pairs.

Loaded from the original AI2 tarball, not the Hugging Face hub: the hub entries
are script-based loaders that ``datasets >= 4`` refuses. The digest is pinned so
the evaluation cannot drift silently.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from proofpath.models import Label

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
# Recorded 2026-09-11 from a fresh download (3,115,079 bytes, Last-Modified 2021-01-26).
SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"

Split = Literal["train", "dev", "test"]

_LABELS = {"SUPPORT": Label.SUPPORTED, "CONTRADICT": Label.REFUTED}


class DatasetError(RuntimeError):
    """The dataset could not be loaded as pinned."""


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    # SciFact ships abstracts pre-split into sentences; rationale ids index this.
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Pair:
    """One (claim, cited document) unit, the abstract-level SciFact task."""

    claim_id: int
    claim: str
    doc_id: int
    label: Label
    rationale: frozenset[int]


@dataclass(frozen=True)
class SciFact:
    corpus: dict[int, Document]
    claims: list[dict[str, Any]]

    def pairs(self) -> list[Pair]:
        pairs: list[Pair] = []
        for row in self.claims:
            evidence: dict[str, list[dict[str, Any]]] = row.get("evidence") or {}
            for doc_id in row.get("cited_doc_ids") or []:
                sets = evidence.get(str(doc_id), [])
                label = Label.NEI
                rationale: set[int] = set()
                for item in sets:
                    label = _LABELS[item["label"]]
                    rationale.update(int(i) for i in item["sentences"])
                pairs.append(
                    Pair(
                        claim_id=int(row["id"]),
                        claim=str(row["claim"]),
                        doc_id=int(doc_id),
                        label=label,
                        rationale=frozenset(rationale),
                    )
                )
        return pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(tar: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    try:
        member = tar.extractfile(name)
    except KeyError:
        member = None
    if member is None:
        raise DatasetError(f"{name} not found in the SciFact tarball")
    text = member.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load(path: Path, *, expected_sha256: str = SHA256, split: Split = "dev") -> SciFact:
    """Parse the tarball at ``path`` after checking its digest."""
    actual = _sha256(path)
    if actual != expected_sha256:
        raise DatasetError(f"{path}: sha256 {actual} does not match pinned {expected_sha256}")
    with tarfile.open(path, mode="r:gz") as tar:
        corpus_rows = _read_jsonl(tar, "data/corpus.jsonl")
        claim_rows = _read_jsonl(tar, f"data/claims_{split}.jsonl")
    corpus = {
        int(row["doc_id"]): Document(
            doc_id=int(row["doc_id"]),
            title=str(row["title"]),
            sentences=tuple(str(s) for s in row["abstract"]),
        )
        for row in corpus_rows
    }
    return SciFact(corpus=corpus, claims=claim_rows)


def ensure_downloaded(cache_dir: Path) -> Path:
    """Return the local tarball, downloading it once. Verified on load."""
    target = cache_dir / "datasets" / "scifact.tar.gz"
    if target.exists() and _sha256(target) == SHA256:
        return target
    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return target
