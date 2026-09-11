"""One local SQLite file: sources, chunks + embeddings, verdicts, raw text with TTL."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from proofpath import cache as cache_mod
from proofpath.cache import Cache
from proofpath.models import Label, Passage, Verdict
from proofpath.pipeline import Thresholds


@pytest.fixture
def db(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "proofpath.sqlite3")


NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
PASSAGES = [Passage("cats purr", "doi:10.1/x", 0), Passage("dogs bark", "doi:10.1/x", 1)]
VECTORS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def test_default_path_lives_under_the_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path))
    assert Cache().path == tmp_path / "proofpath.sqlite3"


def test_schema_is_plain_sqlite_readable_without_extensions(db: Cache) -> None:
    db.close()
    conn = sqlite3.connect(db.path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "sources", "raw_text", "chunks", "verdicts"} <= tables
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("1",)


def test_chunks_round_trip_with_embeddings(db: Cache) -> None:
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="X",
        url="https://x",
        text_kind="abstract",
        raw_text="cats purr. dogs bark.",
        now=NOW,
    )
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS)
    stored = db.get_chunks("doi:10.1/x", "bge@rev")
    assert stored is not None
    passages, vectors = stored
    assert passages == PASSAGES
    assert vectors.dtype == np.float32 and np.allclose(vectors, VECTORS)
    assert db.get_chunks("doi:10.1/x", "other-model") is None


def test_verdict_round_trip_keeps_the_passage(db: Cache) -> None:
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="X",
        url="",
        text_kind="abstract",
        raw_text="t",
        now=NOW,
    )
    verdict = Verdict(Label.SUPPORTED, 0.93, "high", PASSAGES[0], reason="")
    db.put_verdict("h1", "doi:10.1/x", "nli@a+bge@b+k1+t0.5", verdict, now=NOW)
    assert db.get_verdict("h1", "doi:10.1/x", "nli@a+bge@b+k1+t0.5") == verdict
    assert db.get_verdict("h1", "doi:10.1/x", "another-model") is None


def test_schema_refuses_an_asserted_verdict_without_a_passage(db: Cache) -> None:
    db.add_source(
        "s", scheme="academic", title="", url="", text_kind="abstract", raw_text="t", now=NOW
    )
    with pytest.raises(sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO verdicts(claim_hash, source_id, model_id, label, score, tier, reason, "
            "passage_text, passage_index, created_at) "
            "VALUES ('h','s','m','SUPPORTED',0.9,'high','',NULL,NULL,'now')"
        )


def test_raw_text_expires_after_seven_days_and_chunk_text_is_dropped(db: Cache) -> None:
    db.add_source(
        "s",
        scheme="academic",
        title="",
        url="",
        text_kind="fulltext",
        raw_text="cats purr. dogs bark.",
        now=NOW,
    )
    db.put_chunks("s", "bge@rev", PASSAGES, VECTORS)
    assert db.get_raw_text("s", now=NOW + timedelta(days=6)) == "cats purr. dogs bark."
    assert db.get_raw_text("s", now=NOW + timedelta(days=8)) is None
    removed = db.expire(now=NOW + timedelta(days=8))
    assert removed == 1
    assert db.get_raw_text("s", now=NOW + timedelta(days=8)) is None
    # Embeddings stay, quotable text does not -> treated as a miss for retrieval.
    assert db.get_chunks("s", "bge@rev") is None
    row = db._conn.execute("SELECT text, length(embedding) FROM chunks LIMIT 1").fetchone()
    assert row == (None, 12)


def test_refetch_keeps_chunks_and_verdicts(db: Cache) -> None:
    """Spec section 16: a re-fetched source keeps its embeddings and the passages
    quoted on its verdicts. Re-adding the source must update the row in place,
    not replace it (which, with ON DELETE CASCADE, would wipe both)."""
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="X",
        url="https://x/v1",
        text_kind="abstract",
        raw_text="cats purr. dogs bark.",
        now=NOW,
    )
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS)
    verdict = Verdict(Label.SUPPORTED, 0.93, "high", PASSAGES[0], reason="")
    db.put_verdict("h1", "doi:10.1/x", "m", verdict, now=NOW)

    later = NOW + timedelta(days=9)
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="X, second edition",
        url="https://x/v2",
        text_kind="fulltext",
        raw_text="cats purr. dogs bark. birds sing.",
        now=later,
    )
    assert db.get_chunks("doi:10.1/x", "bge@rev") == (PASSAGES, pytest.approx(VECTORS))
    assert db.get_verdict("h1", "doi:10.1/x", "m") == verdict
    assert db.text_kind_and_url("doi:10.1/x") == ("fulltext", "https://x/v2")
    assert db.get_raw_text("doi:10.1/x", now=later) == "cats purr. dogs bark. birds sing."
    (entry,) = db.summary()
    assert (entry.title, entry.chunks, entry.verdicts) == ("X, second edition", 2, 1)
    assert entry.expires_at == (later + timedelta(days=7)).isoformat()


def test_text_kind_reports_the_stored_kind(db: Cache) -> None:
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="",
        url="",
        text_kind="abstract",
        raw_text="t",
        now=NOW,
    )
    assert db.text_kind("doi:10.1/x") == "abstract"
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="",
        url="",
        text_kind="fulltext",
        raw_text="t",
        now=NOW,
    )
    assert db.text_kind("doi:10.1/x") == "fulltext"
    assert db.text_kind("doi:10.1/missing") is None


def test_set_text_kind_relabels_a_stored_source_only(db: Cache) -> None:
    db.add_source(
        "url:https://x", scheme="web", title="", url="", text_kind="fulltext", raw_text="t", now=NOW
    )
    db.set_text_kind("url:https://x", "abstract")
    assert db.text_kind("url:https://x") == "abstract"
    db.set_text_kind("url:https://missing", "abstract")  # a no-op, never an insert
    assert db.text_kind("url:https://missing") is None


def test_text_kind_and_url_reports_the_stored_values(db: Cache) -> None:
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="",
        url="https://example.org/paper.pdf",
        text_kind="fulltext",
        raw_text="t",
        now=NOW,
    )
    assert db.text_kind_and_url("doi:10.1/x") == ("fulltext", "https://example.org/paper.pdf")
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="",
        url="",
        text_kind="abstract",
        raw_text="t",
        now=NOW,
    )
    assert db.text_kind_and_url("doi:10.1/x") == ("abstract", "")
    assert db.text_kind_and_url("doi:10.1/missing") is None


def test_summary_lists_sources_with_counts(db: Cache) -> None:
    db.add_source(
        "s", scheme="academic", title="Paper", url="", text_kind="abstract", raw_text="t", now=NOW
    )
    db.put_chunks("s", "bge@rev", PASSAGES, VECTORS)
    db.put_verdict("h", "s", "m", Verdict(Label.NEI, 0.2, "low", None), now=NOW)
    (entry,) = db.summary()
    assert (entry.source_id, entry.title, entry.chunks, entry.verdicts) == ("s", "Paper", 2, 1)
    assert entry.expires_at == (NOW + timedelta(days=7)).isoformat()


def test_clear_removes_everything_or_only_expired(db: Cache) -> None:
    db.add_source(
        "old",
        scheme="academic",
        title="",
        url="",
        text_kind="abstract",
        raw_text="t",
        now=NOW - timedelta(days=30),
    )
    db.add_source(
        "new", scheme="academic", title="", url="", text_kind="abstract", raw_text="t", now=NOW
    )
    assert db.clear(expired_only=True, now=NOW) == 1
    assert [e.source_id for e in db.summary()] == ["new"]
    assert db.clear() == 1
    assert db.summary() == []


def test_claim_hash_is_stable_and_whitespace_insensitive() -> None:
    assert cache_mod.claim_hash("The  method\nyields 40%") == cache_mod.claim_hash(
        "the method yields 40%"
    )
    assert len(cache_mod.claim_hash("x")) == 64


def test_model_id_includes_k_and_thresholds() -> None:
    a = cache_mod.model_id(nli="nli@a", embedder="bge@b", k=1, thresholds=Thresholds(0.5, 0.9, 0.7))
    b = cache_mod.model_id(
        nli="nli@a", embedder="bge@b", k=1, thresholds=Thresholds(0.45, 0.9, 0.7)
    )
    assert a != b and "k=1" in a and "nli@a" in a
