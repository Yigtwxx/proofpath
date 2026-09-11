"""One local SQLite file: sources, chunks + embeddings, verdicts, raw text with TTL."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from proofpath import cache as cache_mod
from proofpath.cache import Cache, sha256_text
from proofpath.models import Label, Passage, Verdict
from proofpath.pipeline import Thresholds


@pytest.fixture
def db(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "proofpath.sqlite3")


NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
PASSAGES = [Passage("cats purr", "doi:10.1/x", 0), Passage("dogs bark", "doi:10.1/x", 1)]
VECTORS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
RAW = "cats purr. dogs bark."
RAW_SHA = sha256_text(RAW)


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
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("2",)


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
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
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
    db.put_chunks("s", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
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
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
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
    db.put_chunks("s", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
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


def test_chunks_are_a_miss_when_the_source_text_has_changed(db: Cache) -> None:
    """Spec section 16: a re-fetched source whose text moved must be re-chunked, so
    the stored chunks only answer for the text they were cut from."""
    db.add_source(
        "doi:10.1/x",
        scheme="academic",
        title="X",
        url="",
        text_kind="abstract",
        raw_text=RAW,
        now=NOW,
    )
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
    assert db.get_chunks("doi:10.1/x", "bge@rev", text_sha256=RAW_SHA) is not None
    assert db.get_chunks("doi:10.1/x", "bge@rev", text_sha256=sha256_text("other text")) is None
    # No sha asked for, no sha checked: the caller wants whatever is stored.
    assert db.get_chunks("doi:10.1/x", "bge@rev") is not None


def _stocked(db: Cache, source_id: str = "doi:10.1/x", *, digest: str = RAW_SHA) -> Verdict:
    """A source with chunks cut from ``digest`` and one verdict quoting a passage."""
    db.add_source(
        source_id, scheme="academic", title="", url="", text_kind="abstract", raw_text=RAW, now=NOW
    )
    passages = [Passage(p.text, source_id, p.index) for p in PASSAGES]
    db.put_chunks(source_id, "bge@rev", passages, VECTORS, text_sha256=digest)
    verdict = Verdict(Label.SUPPORTED, 0.93, "high", passages[0], reason="")
    db.put_verdict("h1", source_id, "m", verdict, now=NOW)
    return verdict


def test_chunks_from_a_different_text_drop_that_source_s_verdicts(db: Cache) -> None:
    """Spec section 16 keeps verdicts across a TTL expiry, never across a different
    text: a verdict quoting a passage the source no longer carries is an assertion
    with no evidence behind it (product rule 1)."""
    kept = _stocked(db, "doi:10.1/other")
    _stocked(db)

    moved = sha256_text("the source now says something else")
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS, text_sha256=moved)

    assert db.get_verdict("h1", "doi:10.1/x", "m") is None
    # Only that source: another source's evidence has not moved anywhere.
    assert db.get_verdict("h1", "doi:10.1/other", "m") == kept


def test_chunks_from_the_same_text_keep_the_verdicts(db: Cache) -> None:
    kept = _stocked(db)
    db.put_chunks("doi:10.1/x", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
    assert db.get_verdict("h1", "doi:10.1/x", "m") == kept
    # A second embedding model over the same text is not a changed text either.
    db.put_chunks("doi:10.1/x", "other@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
    assert db.get_verdict("h1", "doi:10.1/x", "m") == kept


def test_the_first_chunks_for_a_source_keep_its_verdicts(db: Cache) -> None:
    # Nothing stored says nothing about the text, so a cached verdict (one whose
    # chunks aged out under the TTL, say) survives the first write after it.
    db.add_source(
        "s", scheme="academic", title="", url="", text_kind="abstract", raw_text=RAW, now=NOW
    )
    verdict = Verdict(Label.NEI, 0.2, "low", None)
    db.put_verdict("h1", "s", "m", verdict, now=NOW)
    db.put_chunks("s", "bge@rev", [Passage("cats purr", "s", 0)], VECTORS[:1], text_sha256=RAW_SHA)
    assert db.get_verdict("h1", "s", "m") == verdict


def test_raw_sha256_matches_the_stored_text_and_expires_with_it(db: Cache) -> None:
    db.add_source(
        "s", scheme="academic", title="", url="", text_kind="abstract", raw_text=RAW, now=NOW
    )
    assert db.raw_sha256("s", now=NOW) == RAW_SHA == sha256_text(RAW)
    assert db.raw_sha256("s", now=NOW + timedelta(days=8)) is None
    assert db.raw_sha256("missing", now=NOW) is None


def test_sha256_text_is_the_utf8_digest() -> None:
    text = "ünïcode"
    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


_V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY, scheme TEXT NOT NULL, title TEXT, url TEXT,
    text_kind TEXT NOT NULL, fetched_at TEXT NOT NULL
);
CREATE TABLE raw_text (
    source_id TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE CASCADE,
    content TEXT NOT NULL, sha256 TEXT NOT NULL, fetched_at TEXT NOT NULL, expires_at TEXT NOT NULL
);
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    embed_model TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT, embedding BLOB NOT NULL,
    UNIQUE (source_id, embed_model, ordinal)
);
CREATE TABLE verdicts (
    claim_hash TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL, label TEXT NOT NULL, score REAL NOT NULL, tier TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '', passage_text TEXT, passage_index INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (claim_hash, source_id, model_id),
    CHECK (label NOT IN ('SUPPORTED', 'REFUTED') OR passage_text IS NOT NULL)
);
CREATE INDEX chunks_by_source ON chunks (source_id, embed_model);
CREATE INDEX raw_text_expiry ON raw_text (expires_at);
"""


def _write_v1_database(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    with conn:
        conn.executescript(_V1_SCHEMA)
        conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")
        conn.execute(
            "INSERT INTO sources(source_id, scheme, title, url, text_kind, fetched_at) "
            "VALUES ('s', 'academic', 'Paper', 'https://x', 'abstract', ?)",
            (NOW.isoformat(),),
        )
        conn.execute(
            "INSERT INTO raw_text(source_id, content, sha256, fetched_at, expires_at) "
            "VALUES ('s', ?, ?, ?, ?)",
            (RAW, RAW_SHA, NOW.isoformat(), (NOW + timedelta(days=7)).isoformat()),
        )
        conn.execute(
            "INSERT INTO chunks(source_id, embed_model, ordinal, text, embedding) "
            "VALUES ('s', 'bge@rev', 0, 'cats purr', ?)",
            (VECTORS[0].tobytes(),),
        )
        conn.execute(
            "INSERT INTO verdicts(claim_hash, source_id, model_id, label, score, tier, reason, "
            "passage_text, passage_index, created_at) "
            "VALUES ('h', 's', 'm', 'SUPPORTED', 0.93, 'high', '', 'cats purr', 0, ?)",
            (NOW.isoformat(),),
        )
    conn.close()


def test_opening_a_v1_file_migrates_it_and_keeps_sources_text_and_verdicts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "proofpath.sqlite3"
    _write_v1_database(path)

    with Cache(path) as db:
        assert db._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == (
            "2",
        )
        columns = {r[1] for r in db._conn.execute("PRAGMA table_info(chunks)")}
        assert "text_sha256" in columns
        # The source, its raw text and its verdict survive; only the embeddings go,
        # because they are re-derivable from the text.
        (entry,) = db.summary()
        assert (entry.source_id, entry.title, entry.chunks, entry.verdicts) == ("s", "Paper", 0, 1)
        assert db.get_raw_text("s", now=NOW) == RAW
        assert db.raw_sha256("s", now=NOW) == RAW_SHA
        verdict = db.get_verdict("h", "s", "m")
        assert verdict is not None and verdict.label is Label.SUPPORTED
        assert verdict.passage is not None and verdict.passage.text == "cats purr"


def test_reopening_a_current_file_does_not_drop_its_chunks(tmp_path: Path) -> None:
    path = tmp_path / "proofpath.sqlite3"
    with Cache(path) as db:
        db.add_source(
            "s", scheme="academic", title="", url="", text_kind="abstract", raw_text=RAW, now=NOW
        )
        db.put_chunks("s", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
    with Cache(path) as reopened:
        assert reopened.get_chunks("s", "bge@rev", text_sha256=RAW_SHA) is not None


def test_a_newer_file_is_left_alone_and_never_downgraded(tmp_path: Path) -> None:
    """A file written by a future version records a version this code cannot produce.
    Opening it must not rewrite that version, nor touch its tables."""
    path = tmp_path / "proofpath.sqlite3"
    _write_v1_database(path)
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    conn.close()

    with Cache(path) as db:
        assert db._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == (
            "3",
        )
        assert db.migrated_from is None
        # The v1 chunks table is still there, rows and all.
        assert db._conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (1,)
        assert "text_sha256" not in {r[1] for r in db._conn.execute("PRAGMA table_info(chunks)")}


def test_a_failed_migration_leaves_the_old_version_and_self_heals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration and the version write are one transaction: a step that fails half
    way leaves nothing half-applied, so the next open migrates the file properly."""
    path = tmp_path / "proofpath.sqlite3"
    _write_v1_database(path)

    def broken_step(conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE IF EXISTS chunks")
        raise RuntimeError("migration failed half way")

    monkeypatch.setattr(cache_mod, "_MIGRATIONS", (("2", broken_step),))
    with pytest.raises(RuntimeError, match="half way"):
        Cache(path)

    conn = sqlite3.connect(str(path))
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("1",)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone() == (1,)
    conn.close()

    monkeypatch.undo()
    with Cache(path) as db:
        assert db._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == (
            "2",
        )
        assert db.migrated_from == "1"
        assert "text_sha256" in {r[1] for r in db._conn.execute("PRAGMA table_info(chunks)")}
        assert db.get_verdict("h", "s", "m") is not None


def test_migrated_from_names_the_upgraded_version_only(tmp_path: Path) -> None:
    """The CLI reports that the embeddings will be recomputed, so a plain open --
    new file or current file -- must not claim an upgrade happened."""
    fresh = tmp_path / "fresh.sqlite3"
    with Cache(fresh) as db:
        assert db.migrated_from is None
    with Cache(fresh) as db:
        assert db.migrated_from is None

    old = tmp_path / "old.sqlite3"
    _write_v1_database(old)
    with Cache(old) as db:
        assert db.migrated_from == "1"
    with Cache(old) as db:
        assert db.migrated_from is None


def test_expired_chunk_text_is_a_miss_even_when_the_sha_still_matches(db: Cache) -> None:
    db.add_source(
        "s", scheme="academic", title="", url="", text_kind="abstract", raw_text=RAW, now=NOW
    )
    db.put_chunks("s", "bge@rev", PASSAGES, VECTORS, text_sha256=RAW_SHA)
    db.expire(now=NOW + timedelta(days=8))
    # The text the sha describes is gone, so there is nothing quotable to return.
    assert db.get_chunks("s", "bge@rev", text_sha256=RAW_SHA) is None
    assert db.get_chunks("s", "bge@rev") is None
