"""Persistence: Postgres when a database URL is configured, SQLite otherwise.

The same small set of methods runs on both. Postgres is what makes more than one
server instance work (sessions, passkey challenges and the audit chain are then
shared and updated atomically); SQLite is for local development and single-host
Docker deployments.

What is stored is a deliberate privacy decision:
* Always: session state and scores (purged an hour after expiry), encrypted
  voice signatures of people who passed, passkey public keys, a hash-chained
  audit log. No IP addresses (only a keyed hash for rate limiting).
* Only with consent, or in a labelled tester session: ``samples``. These hold the
  raw measurements of a verification attempt and, where separately agreed, the
  voice recording and face snapshots. Every payload is envelope-encrypted with a
  key that is not in this database, carries an expiry date, and can be deleted
  by the person it came from.

All queries are parameterised.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .security.crypto import b64u, hmac_sha256, sha256

_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at {real} NOT NULL,
    expires_at {real} NOT NULL,
    client_hash TEXT NOT NULL,
    platform TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_client ON sessions(client_hash, created_at);
CREATE INDEX IF NOT EXISTS sessions_expires ON sessions(expires_at);
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    created_at {real} NOT NULL,
    assurance TEXT NOT NULL,
    voiceprint {blob}
);
CREATE TABLE IF NOT EXISTS webauthn_credentials (
    credential_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    public_key {blob} NOT NULL,
    sign_count BIGINT NOT NULL,
    created_at {real} NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_assertions (
    challenge TEXT PRIMARY KEY,
    expires_at {real} NOT NULL,
    purpose TEXT NOT NULL,
    rp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq {serial},
    ts {real} NOT NULL,
    event TEXT NOT NULL,
    data TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL,
    created_at {real} NOT NULL,
    expires_at {real} NOT NULL,
    checkpoint TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    attack_type TEXT,
    participant TEXT,
    source TEXT NOT NULL,
    platform TEXT NOT NULL,
    consent_version TEXT NOT NULL,
    score {real},
    decision TEXT,
    receipt_hash TEXT NOT NULL,
    subject_hash TEXT,
    payload {blob} NOT NULL
);
CREATE INDEX IF NOT EXISTS samples_capture ON samples(capture_id);
CREATE INDEX IF NOT EXISTS samples_receipt ON samples(receipt_hash);
CREATE INDEX IF NOT EXISTS samples_expires ON samples(expires_at);
CREATE INDEX IF NOT EXISTS samples_subject ON samples(subject_hash)
"""
_DIALECT = {
    "sqlite": {"real": "REAL", "blob": "BLOB", "serial": "INTEGER PRIMARY KEY AUTOINCREMENT"},
    "postgres": {"real": "DOUBLE PRECISION", "blob": "BYTEA",
                 "serial": "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"},
}
_SCHEMA_LOCK_ID = 727273  # advisory lock that serialises schema creation across cold-starting instances
_AUDIT_LOCK_ID = 727274  # advisory lock that serialises audit-chain appends across instances

SAMPLE_COLUMNS = (
    "id", "capture_id", "created_at", "expires_at", "checkpoint", "kind", "label", "attack_type",
    "participant", "source", "platform", "consent_version", "score", "decision", "receipt_hash",
    "subject_hash", "payload",
)
_INSERT_SAMPLE = (
    "INSERT INTO samples(id, capture_id, created_at, expires_at, checkpoint, kind, label, attack_type, "
    "participant, source, platform, consent_version, score, decision, receipt_hash, subject_hash, payload) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
if _INSERT_SAMPLE.count("?") != len(SAMPLE_COLUMNS) or any(c not in _INSERT_SAMPLE for c in SAMPLE_COLUMNS):
    raise RuntimeError("_INSERT_SAMPLE is out of step with SAMPLE_COLUMNS")


def is_postgres_url(target: str | Path) -> bool:
    return str(target).startswith(("postgres://", "postgresql://"))


class _Tx:
    """One transaction. ``run`` returns rows as dicts and records ``rowcount``."""

    def __init__(self, cursor, postgres: bool):
        self._cur = cursor
        self._pg = postgres
        self.rowcount = 0

    def run(self, sql: str, params: tuple = ()) -> list[dict]:
        self._cur.execute(sql.replace("?", "%s") if self._pg else sql, params)
        self.rowcount = self._cur.rowcount
        if self._cur.description is None:
            return []
        rows = self._cur.fetchall()
        return [dict(r) for r in rows]


class Store:
    def __init__(self, target: str | Path, audit_key: bytes):
        """``target`` is a ``postgres://`` URL, or a file path for SQLite."""
        self._audit_key = audit_key
        self.backend = "postgres" if is_postgres_url(target) else "sqlite"
        self._lock = threading.Lock()  # SQLite only: one writer at a time in this process
        if self.backend == "postgres":
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                str(target), min_size=1, max_size=4, timeout=15, open=True,
                # prepare_threshold=None: hosted poolers (Neon, PgBouncer) run in
                # transaction mode, where server-side prepared statements break.
                kwargs={"row_factory": dict_row, "prepare_threshold": None},
                check=ConnectionPool.check_connection,  # serverless instances wake with stale connections
            )
        else:
            self.path = Path(target)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    # ---- plumbing ------------------------------------------------------------------
    @contextmanager
    def _tx(self) -> Iterator[_Tx]:
        if self.backend == "postgres":
            with self._pool.connection() as conn, conn.cursor() as cur:  # commits on success, rolls back on error
                yield _Tx(cur, True)
            return
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield _Tx(conn.cursor(), False)
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()

    def _create_schema(self) -> None:
        ddl = _TABLES.format(**_DIALECT[self.backend])
        if self.backend == "sqlite":
            conn = sqlite3.connect(self.path, timeout=10)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            finally:
                conn.close()
        with self._tx() as tx:
            if self.backend == "postgres":
                # Several instances can cold-start at once; concurrent CREATE TABLE
                # IF NOT EXISTS can still collide in Postgres' catalogue.
                tx.run("SELECT pg_advisory_xact_lock(?)", (_SCHEMA_LOCK_ID,))
            for statement in ddl.split(";"):
                if statement.strip():
                    tx.run(statement)
            if self.backend == "sqlite":  # databases created before the version column existed
                columns = {r["name"] for r in tx.run("PRAGMA table_info(sessions)")}
                if "version" not in columns:
                    tx.run("ALTER TABLE sessions ADD COLUMN version INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        if self.backend == "postgres":
            self._pool.close()

    def execute_raw(self, sql: str, params: tuple = ()) -> list[dict]:
        """For tests and one-off maintenance only."""
        with self._tx() as tx:
            return tx.run(sql, params)

    # ---- sessions --------------------------------------------------------------
    def create_session(self, sid: str, client_hash: str, platform: str, state: dict, ttl: int) -> None:
        now = time.time()
        body = json.dumps({k: v for k, v in state.items() if k != "version"})
        with self._tx() as tx:
            tx.run(
                "INSERT INTO sessions(id, created_at, expires_at, client_hash, platform, version, state) "
                "VALUES (?,?,?,?,?,?,?)",
                (sid, now, now + ttl, client_hash, platform, int(state.get("version", 0)), body),
            )

    def get_session(self, sid: str) -> dict | None:
        with self._tx() as tx:
            rows = tx.run("SELECT * FROM sessions WHERE id = ?", (sid,))
        if not rows:
            return None
        row = rows[0]
        return {**row, "state": {**json.loads(row["state"]), "version": row["version"]}}

    def update_session_state(self, sid: str, expected_version: int, state: dict) -> bool:
        """Compare-and-swap in the database, so a checkpoint can be consumed exactly
        once even when requests land on different server instances."""
        body = json.dumps({k: v for k, v in state.items() if k != "version"})
        with self._tx() as tx:
            tx.run(
                "UPDATE sessions SET state = ?, version = version + 1 WHERE id = ? AND version = ?",
                (body, sid, expected_version),
            )
            return tx.rowcount == 1

    def count_recent_sessions(self, client_hash: str, window: float) -> int:
        with self._tx() as tx:
            rows = tx.run(
                "SELECT COUNT(*) AS n FROM sessions WHERE client_hash = ? AND created_at > ?",
                (client_hash, time.time() - window),
            )
        return int(rows[0]["n"])

    def purge_expired(self, grace: float = 3600) -> int:
        """Delete expired sessions, passkey challenges, samples past their retention
        date, and samples from attempts that were abandoned part-way: the person
        never reached the screen that shows the receipt, so nothing is kept."""
        now = time.time()
        with self._tx() as tx:
            tx.run("DELETE FROM sessions WHERE expires_at < ?", (now - grace,))
            n = tx.rowcount
            tx.run("DELETE FROM pending_assertions WHERE expires_at < ?", (now,))
            tx.run("DELETE FROM samples WHERE expires_at < ?", (now,))
            n += tx.rowcount
            tx.run("DELETE FROM samples WHERE decision IS NULL AND created_at < ?", (now - grace,))
            return n + tx.rowcount

    # ---- subjects ----------------------------------------------------------------
    def create_subject(self, sid: str, assurance: str, voiceprint: bytes | None) -> None:
        with self._tx() as tx:
            tx.run(
                "INSERT INTO subjects(id, created_at, assurance, voiceprint) VALUES (?,?,?,?)",
                (sid, time.time(), assurance, voiceprint),
            )

    def get_subject(self, sid: str) -> dict | None:
        with self._tx() as tx:
            rows = tx.run("SELECT * FROM subjects WHERE id = ?", (sid,))
        if not rows:
            return None
        row = rows[0]
        if row["voiceprint"] is not None:
            row["voiceprint"] = bytes(row["voiceprint"])
        return row

    def delete_subject(self, sid: str) -> bool:
        with self._tx() as tx:
            tx.run("DELETE FROM subjects WHERE id = ?", (sid,))
            return tx.rowcount > 0

    # ---- passkeys ------------------------------------------------------------------
    def add_credential(self, cred_id: str, subject_id: str, public_key: bytes, sign_count: int) -> None:
        with self._tx() as tx:
            tx.run(
                "INSERT INTO webauthn_credentials(credential_id, subject_id, public_key, sign_count, created_at) "
                "VALUES (?,?,?,?,?)",
                (cred_id, subject_id, public_key, sign_count, time.time()),
            )

    def get_credential(self, cred_id: str) -> dict | None:
        with self._tx() as tx:
            rows = tx.run("SELECT * FROM webauthn_credentials WHERE credential_id = ?", (cred_id,))
        if not rows:
            return None
        return {**rows[0], "public_key": bytes(rows[0]["public_key"])}

    def credentials_for(self, subject_id: str) -> list[dict]:
        with self._tx() as tx:
            return tx.run("SELECT * FROM webauthn_credentials WHERE subject_id = ?", (subject_id,))

    def update_sign_count(self, cred_id: str, count: int) -> None:
        with self._tx() as tx:
            tx.run("UPDATE webauthn_credentials SET sign_count = ? WHERE credential_id = ?", (count, cred_id))

    def put_assertion(self, challenge: str, ttl: float, purpose: str, rp: str) -> None:
        with self._tx() as tx:
            tx.run(
                "INSERT INTO pending_assertions(challenge, expires_at, purpose, rp) VALUES (?,?,?,?)",
                (challenge, time.time() + ttl, purpose, rp),
            )

    def pop_assertion(self, challenge: str) -> dict | None:
        """Fetch and delete in one transaction: a passkey challenge is single-use."""
        with self._tx() as tx:
            rows = tx.run("SELECT * FROM pending_assertions WHERE challenge = ?", (challenge,))
            if not rows:
                return None
            tx.run("DELETE FROM pending_assertions WHERE challenge = ?", (challenge,))
            if tx.rowcount != 1 or rows[0]["expires_at"] < time.time():
                return None
            return rows[0]

    # ---- collected samples (consented or labelled tester sessions) -----------------
    def add_sample(self, **fields: Any) -> None:
        row = {c: fields.get(c) for c in SAMPLE_COLUMNS}
        with self._tx() as tx:
            tx.run(_INSERT_SAMPLE, tuple(row[c] for c in SAMPLE_COLUMNS))

    def finish_capture(self, capture_id: str, decision: str, subject_hash: str | None) -> None:
        with self._tx() as tx:
            tx.run(
                "UPDATE samples SET decision = ?, subject_hash = ? WHERE capture_id = ?",
                (decision, subject_hash, capture_id),
            )

    def delete_samples(self, *, receipt_hash: str | None = None, subject_hash: str | None = None) -> int:
        if (receipt_hash is None) == (subject_hash is None):
            raise ValueError("pass exactly one of receipt_hash, subject_hash")
        with self._tx() as tx:
            if receipt_hash is not None:
                tx.run("DELETE FROM samples WHERE receipt_hash = ?", (receipt_hash,))
            else:
                tx.run("DELETE FROM samples WHERE subject_hash = ?", (subject_hash,))
            return tx.rowcount

    def iter_samples(self, batch: int = 50) -> Iterator[dict]:
        """All samples, oldest first, fetched in batches (payloads can be large)."""
        last = ("", -1.0)
        while True:
            with self._tx() as tx:
                rows = tx.run(
                    "SELECT * FROM samples WHERE (created_at > ?) OR (created_at = ? AND id > ?) "
                    "ORDER BY created_at, id LIMIT ?",
                    (last[1], last[1], last[0], batch),
                )
            if not rows:
                return
            for row in rows:
                row["payload"] = bytes(row["payload"])
                yield row
            last = (rows[-1]["id"], rows[-1]["created_at"])

    def sample_counts(self) -> list[dict]:
        with self._tx() as tx:
            return tx.run(
                "SELECT checkpoint, kind, label, source, COUNT(*) AS n, COUNT(DISTINCT capture_id) AS sessions, "
                "COUNT(DISTINCT participant) AS participants FROM samples "
                "GROUP BY checkpoint, kind, label, source ORDER BY checkpoint, kind, label, source"
            )

    # ---- audit log -----------------------------------------------------------------
    def _audit_hash(self, prev: str, ts: float, event: str, body: str) -> str:
        return b64u(hmac_sha256(self._audit_key, prev.encode(), repr(float(ts)).encode(), event.encode(), body.encode()))

    def audit(self, event: str, data: dict[str, Any]) -> None:
        """Append-only, HMAC-chained: editing or deleting a row breaks the chain."""
        body = json.dumps(data, sort_keys=True, separators=(",", ":"))
        with self._tx() as tx:
            if self.backend == "postgres":
                tx.run("SELECT pg_advisory_xact_lock(?)", (_AUDIT_LOCK_ID,))
            rows = tx.run("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1")
            prev = rows[0]["hash"] if rows else "genesis"
            ts = time.time()
            tx.run(
                "INSERT INTO audit(ts, event, data, prev_hash, hash) VALUES (?,?,?,?,?)",
                (ts, event, body, prev, self._audit_hash(prev, ts, event, body)),
            )

    def verify_audit_chain(self) -> bool:
        prev = "genesis"
        with self._tx() as tx:
            rows = tx.run("SELECT * FROM audit ORDER BY seq")
        for row in rows:
            if row["prev_hash"] != prev or row["hash"] != self._audit_hash(prev, row["ts"], row["event"], row["data"]):
                return False
            prev = row["hash"]
        return True


def client_fingerprint(secret: bytes, ip: str) -> str:
    """Keyed hash of the client IP: supports rate limiting without storing IPs."""
    return b64u(sha256(hmac_sha256(secret, b"client", ip.encode())))[:32]
