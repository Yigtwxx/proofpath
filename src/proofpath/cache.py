"""The persistent cache: one plain SQLite file under the user cache dir.

Holds fetched source text (with a 7-day TTL), sentence chunks with their
embeddings, and verdicts keyed ``(claim_hash, source_id, model_id)`` so a re-run
of the same document costs nothing. No extension, no server: any SQLite GUI can
open the file (spec sections 5.1, 12, 16).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from proofpath.models import Label, Passage, Tier, Verdict
from proofpath.paths import cache_dir
from proofpath.pipeline import Thresholds

SCHEMA_VERSION = "2"
RAW_TEXT_TTL_DAYS = 7
DB_FILENAME = "proofpath.sqlite3"

# Kept apart from the rest of the schema, one statement per entry: a migration
# recreates this table inside a transaction (where ``executescript`` is not allowed,
# because it commits), and the two spellings must never drift.
_CHUNKS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id    INTEGER PRIMARY KEY,
        source_id   TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
        embed_model TEXT NOT NULL,
        ordinal     INTEGER NOT NULL,
        text        TEXT,                -- NULL once the raw text has expired
        embedding   BLOB NOT NULL,       -- float32 little-endian, L2-normalised
        text_sha256 TEXT NOT NULL,       -- sha256 of the text these chunks were cut from
        UNIQUE (source_id, embed_model, ordinal)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks (source_id, embed_model)",
)
_CHUNKS_SCHEMA = ";\n".join(_CHUNKS_DDL) + ";\n"

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id  TEXT PRIMARY KEY,
    scheme     TEXT NOT NULL,            -- academic | web | social
    title      TEXT,
    url        TEXT,
    text_kind  TEXT NOT NULL,            -- abstract | fulltext
    fetched_at TEXT NOT NULL             -- ISO-8601 UTC
);
CREATE TABLE IF NOT EXISTS raw_text (
    source_id  TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE CASCADE,
    content    TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    claim_hash    TEXT NOT NULL,
    source_id     TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    model_id      TEXT NOT NULL,
    label         TEXT NOT NULL,
    score         REAL NOT NULL,
    tier          TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    passage_text  TEXT,
    passage_index INTEGER,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (claim_hash, source_id, model_id),
    -- Product rule 1, enforced by the schema: no assertion without a passage.
    CHECK (label NOT IN ('SUPPORTED', 'REFUTED') OR passage_text IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS raw_text_expiry ON raw_text (expires_at);
"""

_SCHEMA = _BASE_SCHEMA + _CHUNKS_SCHEMA


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """v1 -> v2: ``chunks`` gains ``text_sha256``.

    The column is ``NOT NULL`` and the stored rows cannot say which text they were
    cut from, so the table is dropped and recreated rather than back-filled with a
    guess: embeddings are re-derivable from the source text, and ``sources``,
    ``raw_text`` and ``verdicts`` (which hold the quoted evidence) are untouched.
    """
    conn.execute("DROP TABLE IF EXISTS chunks")
    for statement in _CHUNKS_DDL:
        conn.execute(statement)


# One step per schema version, oldest first; Phase 9 appends its own.
#
# Invariant, on which the self-healing in ``Cache.__init__`` rests: every step runs
# inside the one transaction that also records the new version, so a step must issue
# its statements one at a time (``conn.execute``) and must never call
# ``executescript`` or ``commit`` -- either would commit a half-applied chain, and
# the file would come back up claiming a schema it does not have.
_MIGRATIONS: tuple[tuple[str, Callable[[sqlite3.Connection], None]], ...] = (("2", _migrate_to_v2),)


def _migrate(conn: sqlite3.Connection, from_version: str) -> None:
    """Run every step newer than ``from_version``, in order."""
    for version, step in _MIGRATIONS:
        if from_version < version:
            step(conn)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    """The digest stored for raw text and for the text a source's chunks came from."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def claim_hash(text: str) -> str:
    """Stable key for a claim: case- and whitespace-insensitive sha256."""
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def model_id(*, nli: str, embedder: str, k: int, thresholds: Thresholds) -> str:
    """Everything that can change a verdict, so stale entries never come back."""
    return (
        f"{nli}|{embedder}|k={k}"
        f"|decide={thresholds.decide:.3f}|high={thresholds.high:.3f}"
        f"|medium={thresholds.medium:.3f}"
    )


@dataclass(frozen=True)
class SourceSummary:
    source_id: str
    scheme: str
    title: str
    text_kind: str
    fetched_at: str
    expires_at: str | None
    chunks: int
    verdicts: int


@dataclass(frozen=True)
class VerdictRow:
    claim_hash: str
    model_id: str
    label: str
    score: float
    tier: str
    reason: str
    passage_text: str | None
    passage_index: int | None


@dataclass(frozen=True)
class SourceDetail:
    summary: SourceSummary
    embed_models: list[str]
    dim: int | None
    chunks: list[tuple[int, str | None]]
    verdicts: list[VerdictRow]


class Cache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or cache_dir() / DB_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        # ``CREATE TABLE IF NOT EXISTS`` leaves an older file's tables as they are, so
        # the recorded version decides what still has to change. A file created just
        # now records nothing, and its migration runs over the empty tables it just
        # got: a no-op, and one code path instead of two.
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        recorded = None if row is None else str(row[0])
        self.migrated_from: str | None = None
        if recorded is None or recorded < SCHEMA_VERSION:
            # One transaction for the whole chain and the version it records: a step
            # that fails leaves the old version in place and nothing half-applied, so
            # the next open migrates the file properly instead of trusting a version
            # its tables do not match. A file recording a newer version is left
            # exactly as it is: never downgraded, never touched.
            self._conn.execute("BEGIN")
            try:
                _migrate(self._conn, recorded or "0")
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
            except Exception:
                self._conn.rollback()
                raise
            self._conn.commit()
            # A new file was upgraded from nothing; an existing one lost its
            # embeddings, which the report may want to mention.
            self.migrated_from = recorded

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- sources and raw text ---------------------------------------------------

    def add_source(
        self,
        source_id: str,
        *,
        scheme: str,
        title: str,
        url: str,
        text_kind: str,
        raw_text: str | None,
        now: datetime | None = None,
        ttl_days: int = RAW_TEXT_TTL_DAYS,
    ) -> None:
        moment = now or _now()
        # Upserts, never INSERT OR REPLACE: a replace deletes the old row first, and
        # with foreign keys on that cascades into the source's chunks and verdicts.
        # A re-fetch must keep both (spec section 16: embeddings stay, the quoted
        # passage stays with the verdict).
        with self._conn:
            self._conn.execute(
                "INSERT INTO sources(source_id, scheme, title, url, text_kind, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_id) DO UPDATE SET scheme = excluded.scheme, "
                "title = excluded.title, url = excluded.url, text_kind = excluded.text_kind, "
                "fetched_at = excluded.fetched_at",
                (source_id, scheme, title, url, text_kind, _iso(moment)),
            )
            if raw_text is not None:
                self._conn.execute(
                    "INSERT INTO raw_text(source_id, content, sha256, fetched_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(source_id) DO UPDATE SET content = excluded.content, "
                    "sha256 = excluded.sha256, fetched_at = excluded.fetched_at, "
                    "expires_at = excluded.expires_at",
                    (
                        source_id,
                        raw_text,
                        sha256_text(raw_text),
                        _iso(moment),
                        _iso(moment + timedelta(days=ttl_days)),
                    ),
                )

    def get_raw_text(self, source_id: str, *, now: datetime | None = None) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM raw_text WHERE source_id = ? AND expires_at > ?",
            (source_id, _iso(now or _now())),
        ).fetchone()
        return None if row is None else str(row[0])

    def raw_sha256(self, source_id: str, *, now: datetime | None = None) -> str | None:
        """The digest of the stored raw text, TTL-filtered exactly like ``get_raw_text``.

        Lets a caller check its chunks against the cached text without reading the
        text back out.
        """
        row = self._conn.execute(
            "SELECT sha256 FROM raw_text WHERE source_id = ? AND expires_at > ?",
            (source_id, _iso(now or _now())),
        ).fetchone()
        return None if row is None else str(row[0])

    def set_text_kind(self, source_id: str, text_kind: str) -> None:
        """Relabel a stored source once its real grade is known (the open-access
        chain measures a page's length only after the ladder has cached it). A
        source that was never stored stays absent."""
        with self._conn:
            self._conn.execute(
                "UPDATE sources SET text_kind = ? WHERE source_id = ?", (text_kind, source_id)
            )

    def text_kind(self, source_id: str) -> str | None:
        """``abstract`` or ``fulltext`` as stored for the source, ``None`` when unknown."""
        row = self._conn.execute(
            "SELECT text_kind FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def text_kind_and_url(self, source_id: str) -> tuple[str, str] | None:
        """``(text_kind, url)`` as stored for the source, ``None`` when unknown.

        Lets a cache hit report the URL the text came from, not just its kind, in one
        read.
        """
        row = self._conn.execute(
            "SELECT text_kind, url FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return None if row is None else (str(row[0]), str(row[1] or ""))

    def expire(self, *, now: datetime | None = None) -> int:
        """Drop expired raw text and the quotable chunk text that came from it."""
        cutoff = _iso(now or _now())
        with self._conn:
            expired = [
                r[0]
                for r in self._conn.execute(
                    "SELECT source_id FROM raw_text WHERE expires_at <= ?", (cutoff,)
                )
            ]
            for source_id in expired:
                self._conn.execute("DELETE FROM raw_text WHERE source_id = ?", (source_id,))
                self._conn.execute(
                    "UPDATE chunks SET text = NULL WHERE source_id = ?", (source_id,)
                )
        return len(expired)

    # --- chunks -----------------------------------------------------------------

    def put_chunks(
        self,
        source_id: str,
        embed_model: str,
        passages: Sequence[Passage],
        vectors: np.ndarray,
        *,
        text_sha256: str,
    ) -> None:
        """Store a source's chunks. ``text_sha256`` is the digest of the text they
        were cut from, so a later read can tell whether it still describes that text.

        A digest that differs from one this source's chunks already record means the
        source now serves *different* text, and every verdict stored against it
        quotes a passage out of the old one. Section 16 keeps verdicts across a TTL
        expiry -- the raw text goes, the short quoted passage stays with the verdict
        -- but never across a different text: a verdict that cannot show the source's
        own words is an assertion with no evidence behind it (product rule 1). They
        are dropped inside the one transaction that stores the new chunks, so there
        is no moment at which the file holds new chunks and old quotations.
        """
        arr = np.asarray(vectors, dtype="<f4")
        if len(passages) != len(arr):
            raise ValueError("one vector per passage is required")
        with self._conn:
            # Across every embedding model, not just this one: the text is a property
            # of the source, and whichever model noticed first is the one that knows.
            digests = {
                str(row[0])
                for row in self._conn.execute(
                    "SELECT DISTINCT text_sha256 FROM chunks WHERE source_id = ?", (source_id,)
                )
            }
            if digests - {text_sha256}:
                self._conn.execute("DELETE FROM verdicts WHERE source_id = ?", (source_id,))
            self._conn.execute(
                "DELETE FROM chunks WHERE source_id = ? AND embed_model = ?",
                (source_id, embed_model),
            )
            self._conn.executemany(
                "INSERT INTO chunks(source_id, embed_model, ordinal, text, embedding, "
                "text_sha256) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (source_id, embed_model, p.index, p.text, arr[i].tobytes(), text_sha256)
                    for i, p in enumerate(passages)
                ],
            )

    def get_chunks(
        self, source_id: str, embed_model: str, *, text_sha256: str | None = None
    ) -> tuple[list[Passage], np.ndarray] | None:
        """The stored chunks and their vectors, or ``None`` when they cannot be used.

        A miss is: nothing stored, text dropped by the TTL (spec section 16), or --
        when ``text_sha256`` is given -- chunks cut from text that has since changed,
        which must be re-chunked rather than quoted.
        """
        rows = self._conn.execute(
            "SELECT ordinal, text, embedding, text_sha256 FROM chunks "
            "WHERE source_id = ? AND embed_model = ? ORDER BY ordinal",
            (source_id, embed_model),
        ).fetchall()
        if not rows or any(text is None for _, text, _, _ in rows):
            return None
        if text_sha256 is not None and any(stored != text_sha256 for *_, stored in rows):
            return None
        passages = [Passage(str(text), source_id, int(ordinal)) for ordinal, text, _, _ in rows]
        vectors = np.stack([np.frombuffer(blob, dtype="<f4") for _, _, blob, _ in rows])
        return passages, vectors

    # --- verdicts ---------------------------------------------------------------

    def put_verdict(
        self,
        claim_hash_: str,
        source_id: str,
        model_id_: str,
        verdict: Verdict,
        *,
        now: datetime | None = None,
    ) -> None:
        passage = verdict.passage
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO verdicts(claim_hash, source_id, model_id, label, score, "
                "tier, reason, passage_text, passage_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_hash_,
                    source_id,
                    model_id_,
                    verdict.label.value,
                    verdict.score,
                    verdict.tier,
                    verdict.reason,
                    None if passage is None else passage.text,
                    None if passage is None else passage.index,
                    _iso(now or _now()),
                ),
            )

    def get_verdict(self, claim_hash_: str, source_id: str, model_id_: str) -> Verdict | None:
        row = self._conn.execute(
            "SELECT label, score, tier, reason, passage_text, passage_index FROM verdicts "
            "WHERE claim_hash = ? AND source_id = ? AND model_id = ?",
            (claim_hash_, source_id, model_id_),
        ).fetchone()
        if row is None:
            return None
        label, score, tier, reason, passage_text, passage_index = row
        passage = (
            None if passage_text is None else Passage(passage_text, source_id, int(passage_index))
        )
        return Verdict(Label(label), float(score), _tier(tier), passage, reason=reason)

    # --- inspection and housekeeping ---------------------------------------------

    def summary(self) -> list[SourceSummary]:
        rows = self._conn.execute(
            """
            SELECT s.source_id, s.scheme, COALESCE(s.title, ''), s.text_kind, s.fetched_at,
                   r.expires_at,
                   (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.source_id),
                   (SELECT COUNT(*) FROM verdicts v WHERE v.source_id = s.source_id)
            FROM sources s LEFT JOIN raw_text r ON r.source_id = s.source_id
            ORDER BY s.fetched_at DESC, s.source_id
            """
        ).fetchall()
        return [SourceSummary(*row) for row in rows]

    def detail(self, source_id: str) -> SourceDetail | None:
        summary = next((s for s in self.summary() if s.source_id == source_id), None)
        if summary is None:
            return None
        models = [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT embed_model FROM chunks WHERE source_id = ?", (source_id,)
            )
        ]
        first = self._conn.execute(
            "SELECT length(embedding) FROM chunks WHERE source_id = ? LIMIT 1", (source_id,)
        ).fetchone()
        dim = None if first is None else int(first[0]) // 4
        chunks = [
            (int(o), t)
            for o, t in self._conn.execute(
                "SELECT ordinal, text FROM chunks WHERE source_id = ? "
                "ORDER BY embed_model, ordinal",
                (source_id,),
            )
        ]
        verdicts = [
            VerdictRow(*row)
            for row in self._conn.execute(
                "SELECT claim_hash, model_id, label, score, tier, reason, passage_text, "
                "passage_index FROM verdicts WHERE source_id = ? ORDER BY created_at",
                (source_id,),
            )
        ]
        return SourceDetail(summary, models, dim, chunks, verdicts)

    def clear(self, *, expired_only: bool = False, now: datetime | None = None) -> int:
        """Remove sources (cascading to text, chunks and verdicts). Returns the count."""
        with self._conn:
            if expired_only:
                cutoff = _iso(now or _now())
                cursor = self._conn.execute(
                    "DELETE FROM sources WHERE source_id IN "
                    "(SELECT source_id FROM raw_text WHERE expires_at <= ?)",
                    (cutoff,),
                )
            else:
                cursor = self._conn.execute("DELETE FROM sources")
        return int(cursor.rowcount)


def _tier(value: str) -> Tier:
    if value == "high":
        return "high"
    if value == "medium":
        return "medium"
    return "low"
