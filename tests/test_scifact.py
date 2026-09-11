from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from proofpath.eval import scifact
from proofpath.models import Label


def _make_tarball(root: Path) -> tuple[Path, str]:
    corpus = [
        {"doc_id": 10, "title": "Doc ten", "abstract": ["S0.", "S1.", "S2."], "structured": False},
        {"doc_id": 20, "title": "Doc twenty", "abstract": ["T0.", "T1."], "structured": True},
    ]
    dev = [
        {
            "id": 1,
            "claim": "supported claim",
            "evidence": {"10": [{"sentences": [1, 2], "label": "SUPPORT"}]},
            "cited_doc_ids": [10],
        },
        {
            "id": 2,
            "claim": "refuted claim",
            "evidence": {"20": [{"sentences": [0], "label": "CONTRADICT"}]},
            "cited_doc_ids": [20],
        },
        {"id": 3, "claim": "nei claim", "evidence": {}, "cited_doc_ids": [10, 20]},
    ]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, rows in (("data/corpus.jsonl", corpus), ("data/claims_dev.jsonl", dev)):
            payload = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    path = root / "scifact.tar.gz"
    path.write_bytes(buffer.getvalue())
    return path, hashlib.sha256(buffer.getvalue()).hexdigest()


def test_load_from_tarball_yields_claim_document_pairs(tmp_path: Path) -> None:
    path, digest = _make_tarball(tmp_path)
    dataset = scifact.load(path, expected_sha256=digest, split="dev")
    assert len(dataset.corpus) == 2
    assert dataset.corpus[10].sentences == ("S0.", "S1.", "S2.")
    pairs = dataset.pairs()
    assert [(p.claim_id, p.doc_id, p.label) for p in pairs] == [
        (1, 10, Label.SUPPORTED),
        (2, 20, Label.REFUTED),
        (3, 10, Label.NEI),
        (3, 20, Label.NEI),
    ]
    assert pairs[0].rationale == frozenset({1, 2})
    assert pairs[2].rationale == frozenset()


def test_sha_mismatch_is_refused(tmp_path: Path) -> None:
    path, _ = _make_tarball(tmp_path)
    with pytest.raises(scifact.DatasetError, match="sha256"):
        scifact.load(path, expected_sha256="0" * 64, split="dev")


def test_missing_split_is_reported(tmp_path: Path) -> None:
    path, digest = _make_tarball(tmp_path)
    with pytest.raises(scifact.DatasetError, match=r"claims_test\.jsonl"):
        scifact.load(path, expected_sha256=digest, split="test")


def test_pinned_digest_is_a_real_sha256() -> None:
    assert len(scifact.SHA256) == 64
    int(scifact.SHA256, 16)
