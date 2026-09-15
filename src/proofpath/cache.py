"""The persistent cache: one plain SQLite file under the user cache dir.

Holds fetched source text (with a 7-day TTL), sentence chunks with their
embeddings, verdicts keyed ``(claim_hash, source_id, model_id)``, the optional
judge's opinions beside them, and the two
network lookups a run makes before it fetches anything -- reference resolution and
the retraction check -- so a re-run of the same document costs nothing. No
extension, no server: any SQLite GUI can open the file (spec sections 5.1, 12, 16).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from proofpath.judge import JudgeOpinion
from proofpath.models import Label, Passage, Tier, Verdict
from proofpath.paths import cache_dir
from proofpath.pipeline import CUT_DECIMALS, Thresholds
from proofpath.resolve import Candidate, FieldMatch, ResolveResult, Retraction, State, strip_marker

SCHEMA_VERSION = "4"
RAW_TEXT_TTL_DAYS = 7
# A resolution is a statement about a published record, which does not change; the
# month is there so a reference an index had not yet ingested is looked at again.
RESOLUTION_TTL_DAYS = 30
# A retraction notice, once issued, stays issued -- but a paper that is clean today
# can be retracted tomorrow, so absence is believed for a week and presence for a
# month. Caching a miss as long as a hit would hide the notice that arrives in between.
RETRACTION_HIT_TTL_DAYS = 30
RETRACTION_MISS_TTL_DAYS = 7
# Product rule 2: an outage says nothing about the reference, so it is never stored.
# Every other state is a reading of what the providers answered, and keeps.
CACHEABLE_STATES = frozenset(
    {State.RESOLVED, State.RESOLVED_LOW, State.AMBIGUOUS, State.GHOST, State.NOT_INDEXED}
)
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

# The two network lookups the resolving and retraction stages make, keyed by what
# they are about rather than by a source id: a reference has no source id until it
# resolves, and a retraction check is about a DOI, not about the text behind it.
_LOOKUPS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS resolutions (
        raw_hash        TEXT PRIMARY KEY,   -- sha256 of the marker-free, folded entry
        state           TEXT NOT NULL,      -- resolve.State value, verbatim
        best_json       TEXT,               -- the chosen Candidate, or NULL
        candidates_json TEXT NOT NULL,
        notes_json      TEXT NOT NULL,
        match_json      TEXT,               -- the FieldMatch behind the state, or NULL
        resolved_at     TEXT NOT NULL,
        expires_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retractions (
        doi             TEXT PRIMARY KEY,
        retraction_json TEXT,               -- NULL means "checked, and there was none"
        checked_at      TEXT NOT NULL,
        expires_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS resolutions_expiry ON resolutions (expires_at)",
    "CREATE INDEX IF NOT EXISTS retractions_expiry ON retractions (expires_at)",
)
_LOOKUPS_SCHEMA = ";\n".join(_LOOKUPS_DDL) + ";\n"

# The optional judge's opinions, beside the verdicts and never instead of them (spec
# section 11.1). Keyed by the judge model as well, so a run with another judge asks
# again rather than reading back an opinion that model never gave. ``model_id`` is
# deliberately absent: an opinion is about the claim and the passage, not about which
# local NLI happened to escalate it, so toggling ``--judge`` cannot invalidate a
# cached verdict and re-tuning the thresholds cannot invalidate a cached opinion.
_JUDGEMENTS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS judgements (
        claim_hash  TEXT NOT NULL,
        source_id   TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
        judge_model TEXT NOT NULL,       -- "groq openai/gpt-oss-120b"
        label       TEXT NOT NULL,
        rationale   TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (claim_hash, source_id, judge_model)
    )
    """,
)
_JUDGEMENTS_SCHEMA = ";\n".join(_JUDGEMENTS_DDL) + ";\n"

_SCHEMA = _BASE_SCHEMA + _CHUNKS_SCHEMA + _LOOKUPS_SCHEMA + _JUDGEMENTS_SCHEMA


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


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """v2 -> v3: the ``resolutions`` and ``retractions`` lookup tables.

    Purely additive -- nothing stored before v3 answers either question -- so the
    two tables are created and every existing row is left exactly where it was.
    """
    for statement in _LOOKUPS_DDL:
        conn.execute(statement)


def _migrate_to_v4(conn: sqlite3.Connection) -> None:
    """v3 -> v4: the ``judgements`` table.

    Additive, like v3: nothing stored before v4 holds a second opinion, and the
    verdicts are untouched on purpose -- their ``model_id`` says nothing about the
    judge, so a file that gains this table hands back exactly the verdicts it held
    before and a run with ``--judge`` costs no re-verification (spec section 11.1).
    """
    for statement in _JUDGEMENTS_DDL:
        conn.execute(statement)


# One step per schema version, oldest first.
#
# Invariant, on which the self-healing in ``Cache.__init__`` rests: every step runs
# inside the one transaction that also records the new version, so a step must issue
# its statements one at a time (``conn.execute``) and must never call
# ``executescript`` or ``commit`` -- either would commit a half-applied chain, and
# the file would come back up claiming a schema it does not have.
_MIGRATIONS: tuple[tuple[str, Callable[[sqlite3.Connection], None]], ...] = (
    ("2", _migrate_to_v2),
    ("3", _migrate_to_v3),
    ("4", _migrate_to_v4),
)


def _version(value: str | None) -> int:
    """A recorded schema version as a number. Versions are compared numerically, not
    as strings, so that v10 does not sort before v2 (OPEN-ITEMS 10.3). Anything this
    code cannot read counts as the oldest possible file: the chain then runs over
    tables ``CREATE TABLE IF NOT EXISTS`` has already put in place, which is a
    repair rather than a downgrade."""
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _migrate(conn: sqlite3.Connection, from_version: str) -> None:
    """Run every step newer than ``from_version``, in order."""
    for version, step in _MIGRATIONS:
        if _version(from_version) < _version(version):
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


def resolution_hash(raw: str) -> str:
    """Stable key for a bibliography entry, independent of how it was printed.

    ``Reference.raw`` keeps the marker the document put in front of it, and the same
    work is printed ``[7] `` in one paper and ``7. `` in another; the marker is
    dropped exactly as ``Resolver.resolve`` drops it, so the two share one row.
    Case is folded for the same reason a claim's is.
    """
    return hashlib.sha256(strip_marker(raw).strip().lower().encode("utf-8")).hexdigest()


def model_id(*, nli: str, embedder: str, k: int, thresholds: Thresholds) -> str:
    """Everything that can change a verdict, so stale entries never come back."""
    # ``CUT_DECIMALS``, not three: the calibrated cuts sit against 1.0 (spec section
    # 14), so a shorter id would hand verdicts decided at high=0.999330 back to a run
    # calibrated at 0.999400.
    return (
        f"{nli}|{embedder}|k={k}"
        f"|decide={thresholds.decide:.{CUT_DECIMALS}f}|high={thresholds.high:.{CUT_DECIMALS}f}"
        f"|medium={thresholds.medium:.{CUT_DECIMALS}f}"
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
class RetractionHit:
    """One cached retraction check. ``notice is None`` means the check was made and
    there was no notice, which is a different fact from "never checked" -- the miss a
    ``get_retraction`` of ``None`` reports (product rule 2)."""

    notice: Retraction | None


@dataclass(frozen=True)
class Cleared:
    """What one ``clear()`` removed, per table."""

    sources: int
    resolutions: int
    retractions: int


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
        if recorded is None or _version(recorded) < _version(SCHEMA_VERSION):
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
                # The judge quoted the same passages, so its opinions go the same way:
                # a second opinion about text the source no longer serves is evidence
                # nobody can check (product rule 1).
                self._conn.execute("DELETE FROM judgements WHERE source_id = ?", (source_id,))
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

    # --- judgements ---------------------------------------------------------------

    def put_judgement(
        self,
        claim_hash_: str,
        source_id: str,
        judge_model: str,
        opinion: JudgeOpinion,
        *,
        now: datetime | None = None,
    ) -> None:
        """Store one judge opinion. Re-asking the same judge replaces what it said."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO judgements(claim_hash, source_id, judge_model, label, rationale, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(claim_hash, source_id, judge_model) DO UPDATE SET "
                "label = excluded.label, rationale = excluded.rationale, "
                "created_at = excluded.created_at",
                (
                    claim_hash_,
                    source_id,
                    judge_model,
                    opinion.label.value,
                    opinion.rationale,
                    _iso(now or _now()),
                ),
            )

    def get_judgement(
        self, claim_hash_: str, source_id: str, judge_model: str
    ) -> JudgeOpinion | None:
        """What this judge said about this claim and source, or ``None`` if unasked.

        No TTL: an opinion is about a claim and a passage, both of which are fixed.
        The passage changing is what retires it, and ``put_chunks`` does that.
        """
        row = self._conn.execute(
            "SELECT label, rationale FROM judgements "
            "WHERE claim_hash = ? AND source_id = ? AND judge_model = ?",
            (claim_hash_, source_id, judge_model),
        ).fetchone()
        if row is None:
            return None
        label, rationale = row
        return JudgeOpinion(label=Label(label), rationale=str(rationale), model=judge_model)

    # --- resolutions and retractions ---------------------------------------------

    def get_resolution(self, raw: str, *, now: datetime | None = None) -> ResolveResult | None:
        """The stored resolution for this reference, or ``None`` when there is none
        that is still within its TTL."""
        row = self._conn.execute(
            "SELECT state, best_json, candidates_json, notes_json, match_json FROM resolutions "
            "WHERE raw_hash = ? AND expires_at > ?",
            (resolution_hash(raw), _iso(now or _now())),
        ).fetchone()
        if row is None:
            return None
        state, best_json, candidates_json, notes_json, match_json = row
        return ResolveResult(
            State(state),
            _candidate(json.loads(best_json)) if best_json else None,
            [_candidate(item) for item in json.loads(candidates_json)],
            notes=list(json.loads(notes_json)),
            match=_field_match(json.loads(match_json)) if match_json else None,
        )

    def put_resolution(
        self, raw: str, result: ResolveResult, *, now: datetime | None = None
    ) -> None:
        """Store a resolution for ``RESOLUTION_TTL_DAYS``.

        ``UNVERIFIED (provider unavailable)`` is dropped on the floor: it says that
        Crossref or Semantic Scholar was down, not anything about the reference, and
        a cached outage would repeat someone's bad afternoon for a month (rule 2).
        """
        if result.state not in CACHEABLE_STATES:
            return
        moment = now or _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO resolutions(raw_hash, state, best_json, candidates_json, "
                "notes_json, match_json, resolved_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(raw_hash) DO UPDATE SET state = excluded.state, "
                "best_json = excluded.best_json, candidates_json = excluded.candidates_json, "
                "notes_json = excluded.notes_json, match_json = excluded.match_json, "
                "resolved_at = excluded.resolved_at, expires_at = excluded.expires_at",
                (
                    resolution_hash(raw),
                    result.state.value,
                    None if result.best is None else _dump(result.best),
                    _dump([_fields(c) for c in result.candidates]),
                    _dump(list(result.notes)),
                    None if result.match is None else _dump(result.match),
                    _iso(moment),
                    _iso(moment + timedelta(days=RESOLUTION_TTL_DAYS)),
                ),
            )

    def get_retraction(self, doi: str, *, now: datetime | None = None) -> RetractionHit | None:
        """The stored retraction check for ``doi``, or ``None`` when there is none."""
        row = self._conn.execute(
            "SELECT retraction_json FROM retractions WHERE doi = ? AND expires_at > ?",
            (doi.lower(), _iso(now or _now())),
        ).fetchone()
        if row is None:
            return None
        payload = row[0]
        return RetractionHit(None if payload is None else _retraction(json.loads(payload)))

    def put_retraction(
        self, doi: str, notice: Retraction | None, *, now: datetime | None = None
    ) -> None:
        """Store a retraction check. A notice keeps for a month, its absence for a
        week: a paper that is clean today can be retracted tomorrow."""
        moment = now or _now()
        days = RETRACTION_HIT_TTL_DAYS if notice is not None else RETRACTION_MISS_TTL_DAYS
        with self._conn:
            self._conn.execute(
                "INSERT INTO retractions(doi, retraction_json, checked_at, expires_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(doi) DO UPDATE SET retraction_json = excluded.retraction_json, "
                "checked_at = excluded.checked_at, expires_at = excluded.expires_at",
                (
                    doi.lower(),
                    None if notice is None else _dump(notice),
                    _iso(moment),
                    _iso(moment + timedelta(days=days)),
                ),
            )

    def lookup_counts(self) -> tuple[int, int]:
        """``(resolutions, retraction checks)`` stored, expired rows included: the
        two numbers ``proofpath cache`` and ``cache ls`` print."""
        resolutions = self._conn.execute("SELECT COUNT(*) FROM resolutions").fetchone()
        retractions = self._conn.execute("SELECT COUNT(*) FROM retractions").fetchone()
        return (int(resolutions[0]), int(retractions[0]))

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

    def clear(self, *, expired_only: bool = False, now: datetime | None = None) -> Cleared:
        """Remove sources (cascading to text, chunks and verdicts) and the cached
        lookups. One transaction, so the file is never half cleared."""
        cutoff = _iso(now or _now())
        with self._conn:
            if expired_only:
                sources = self._conn.execute(
                    "DELETE FROM sources WHERE source_id IN "
                    "(SELECT source_id FROM raw_text WHERE expires_at <= ?)",
                    (cutoff,),
                ).rowcount
                resolutions = self._conn.execute(
                    "DELETE FROM resolutions WHERE expires_at <= ?", (cutoff,)
                ).rowcount
                retractions = self._conn.execute(
                    "DELETE FROM retractions WHERE expires_at <= ?", (cutoff,)
                ).rowcount
            else:
                sources = self._conn.execute("DELETE FROM sources").rowcount
                resolutions = self._conn.execute("DELETE FROM resolutions").rowcount
                retractions = self._conn.execute("DELETE FROM retractions").rowcount
        return Cleared(int(sources), int(resolutions), int(retractions))


def _fields(item: Any) -> dict[str, Any]:
    """A frozen value type as a plain dict, so the stored JSON stays readable in a
    SQLite GUI rather than being a pickle nobody can inspect."""
    return dict(vars(item))


def _dump(payload: Any) -> str:
    return json.dumps(_fields(payload) if hasattr(payload, "__dict__") else payload)


def _candidate(payload: dict[str, Any]) -> Candidate:
    return Candidate(
        doi=str(payload.get("doi", "")),
        title=str(payload.get("title", "")),
        first_author=str(payload.get("first_author", "")),
        year=None if payload.get("year") is None else int(payload["year"]),
        venue=str(payload.get("venue", "")),
        provider=str(payload.get("provider", "")),
        url=str(payload.get("url", "")),
    )


def _field_match(payload: dict[str, Any]) -> FieldMatch:
    return FieldMatch(
        title=float(payload["title"]), author=bool(payload["author"]), year=bool(payload["year"])
    )


def _retraction(payload: dict[str, Any]) -> Retraction:
    return Retraction(
        source=str(payload["source"]),
        date=None if payload.get("date") is None else str(payload["date"]),
        notice_doi=None if payload.get("notice_doi") is None else str(payload["notice_doi"]),
        label=str(payload["label"]),
    )


def _tier(value: str) -> Tier:
    if value == "high":
        return "high"
    if value == "medium":
        return "medium"
    return "low"
