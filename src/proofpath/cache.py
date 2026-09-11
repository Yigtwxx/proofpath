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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from proofpath.models import Label, Passage, Tier, Verdict
from proofpath.paths import cache_dir
from proofpath.pipeline import Thresholds

SCHEMA_VERSION = "1"
RAW_TEXT_TTL_DAYS = 7
DB_FILENAME = "proofpath.sqlite3"

_SCHEMA = """
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
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    embed_model TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT,                    -- NULL once the raw text has expired
    embedding   BLOB NOT NULL,           -- float32 little-endian, L2-normalised
    UNIQUE (source_id, embed_model, ordinal)
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
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks (source_id, embed_model);
CREATE INDEX IF NOT EXISTS raw_text_expiry ON raw_text (expires_at);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


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
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()

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
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sources(source_id, scheme, title, url, text_kind, "
                "fetched_at) VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, scheme, title, url, text_kind, _iso(moment)),
            )
            if raw_text is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO raw_text(source_id, content, sha256, fetched_at, "
                    "expires_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        source_id,
                        raw_text,
                        hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
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
    ) -> None:
        arr = np.asarray(vectors, dtype="<f4")
        if len(passages) != len(arr):
            raise ValueError("one vector per passage is required")
        with self._conn:
            self._conn.execute(
                "DELETE FROM chunks WHERE source_id = ? AND embed_model = ?",
                (source_id, embed_model),
            )
            self._conn.executemany(
                "INSERT INTO chunks(source_id, embed_model, ordinal, text, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (source_id, embed_model, p.index, p.text, arr[i].tobytes())
                    for i, p in enumerate(passages)
                ],
            )

    def get_chunks(
        self, source_id: str, embed_model: str
    ) -> tuple[list[Passage], np.ndarray] | None:
        rows = self._conn.execute(
            "SELECT ordinal, text, embedding FROM chunks WHERE source_id = ? AND embed_model = ? "
            "ORDER BY ordinal",
            (source_id, embed_model),
        ).fetchall()
        if not rows or any(text is None for _, text, _ in rows):
            return None
        passages = [Passage(str(text), source_id, int(ordinal)) for ordinal, text, _ in rows]
        vectors = np.stack([np.frombuffer(blob, dtype="<f4") for _, _, blob in rows])
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
