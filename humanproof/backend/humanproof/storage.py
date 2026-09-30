"""SQLite persistence.

What is stored (and what is not) is a deliberate privacy decision:
* Stored: session state and scores, encrypted voiceprint embeddings, WebAuthn
  public keys, a hash-chained audit log, and (opt-in only) encrypted feature
  vectors for model training.
* Never stored: raw video frames, face crops, raw audio, IP addresses (only a
  keyed hash), or anything that could rebuild a face or voice.

All queries are parameterised. For multi-instance deployments swap this for
Postgres; the method surface is intentionally small.
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

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    client_hash TEXT NOT NULL,
    platform TEXT NOT NULL,
    state TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_client ON sessions(client_hash, created_at);
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    assurance TEXT NOT NULL,
    voiceprint BLOB
);
CREATE TABLE IF NOT EXISTS webauthn_credentials (
    credential_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    public_key BLOB NOT NULL,
    sign_count INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    data TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_samples (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    checkpoint TEXT NOT NULL,
    consent_version TEXT NOT NULL,
    payload BLOB NOT NULL
);
"""


class Store:
    def __init__(self, path: Path, audit_key: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._audit_key = audit_key
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    # ---- sessions --------------------------------------------------------------
    def create_session(self, sid: str, client_hash: str, platform: str, state: dict, ttl: int) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions(id, created_at, expires_at, client_hash, platform, state) VALUES (?,?,?,?,?,?)",
                (sid, now, now + ttl, client_hash, platform, json.dumps(state)),
            )

    def get_session(self, sid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "state": json.loads(row["state"])}

    def update_session_state(self, sid: str, expected_version: int, state: dict) -> bool:
        """Optimistic concurrency: a checkpoint can be consumed exactly once."""
        with self._lock, self._conn() as c:
            row = c.execute("SELECT state FROM sessions WHERE id = ?", (sid,)).fetchone()
            if row is None or json.loads(row["state"]).get("version") != expected_version:
                return False
            state = {**state, "version": expected_version + 1}
            c.execute("UPDATE sessions SET state = ? WHERE id = ?", (json.dumps(state), sid))
            return True

    def count_recent_sessions(self, client_hash: str, window: float) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE client_hash = ? AND created_at > ?",
                (client_hash, time.time() - window),
            ).fetchone()
        return int(row["n"])

    def purge_expired(self, grace: float = 3600) -> int:
        with self._conn() as c:
            cur = c.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time() - grace,))
            return cur.rowcount

    # ---- subjects ----------------------------------------------------------------
    def create_subject(self, sid: str, assurance: str, voiceprint: bytes | None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO subjects(id, created_at, assurance, voiceprint) VALUES (?,?,?,?)",
                (sid, time.time(), assurance, voiceprint),
            )

    def get_subject(self, sid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM subjects WHERE id = ?", (sid,)).fetchone()
        return dict(row) if row else None

    def delete_subject(self, sid: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM subjects WHERE id = ?", (sid,)).rowcount > 0

    # ---- webauthn ------------------------------------------------------------------
    def add_credential(self, cred_id: str, subject_id: str, public_key: bytes, sign_count: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO webauthn_credentials VALUES (?,?,?,?,?)",
                (cred_id, subject_id, public_key, sign_count, time.time()),
            )

    def get_credential(self, cred_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM webauthn_credentials WHERE credential_id = ?", (cred_id,)).fetchone()
        return dict(row) if row else None

    def credentials_for(self, subject_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM webauthn_credentials WHERE subject_id = ?", (subject_id,)).fetchall()
        return [dict(r) for r in rows]

    def update_sign_count(self, cred_id: str, count: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE webauthn_credentials SET sign_count = ? WHERE credential_id = ?", (count, cred_id))

    # ---- research samples (opt-in) -----------------------------------------------
    def add_research_sample(self, rid: str, checkpoint: str, consent_version: str, payload: bytes) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO research_samples VALUES (?,?,?,?,?)",
                (rid, time.time(), checkpoint, consent_version, payload),
            )

    def iter_research_samples(self, checkpoint: str) -> Iterator[dict]:
        with self._conn() as c:
            for row in c.execute("SELECT * FROM research_samples WHERE checkpoint = ?", (checkpoint,)):
                yield dict(row)

    # ---- audit log -----------------------------------------------------------------
    def audit(self, event: str, data: dict[str, Any]) -> None:
        """Append-only, HMAC-chained: editing or deleting a row breaks the chain."""
        body = json.dumps(data, sort_keys=True, separators=(",", ":"))
        with self._lock, self._conn() as c:
            row = c.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else "genesis"
            ts = time.time()
            digest = b64u(hmac_sha256(self._audit_key, prev.encode(), repr(ts).encode(), event.encode(), body.encode()))
            c.execute(
                "INSERT INTO audit(ts, event, data, prev_hash, hash) VALUES (?,?,?,?,?)",
                (ts, event, body, prev, digest),
            )

    def verify_audit_chain(self) -> bool:
        prev = "genesis"
        with self._conn() as c:
            for row in c.execute("SELECT * FROM audit ORDER BY seq"):
                expected = b64u(
                    hmac_sha256(
                        self._audit_key, prev.encode(), repr(row["ts"]).encode(),
                        row["event"].encode(), row["data"].encode(),
                    )
                )
                if row["prev_hash"] != prev or row["hash"] != expected:
                    return False
                prev = row["hash"]
        return True


def client_fingerprint(secret: bytes, ip: str) -> str:
    """Keyed hash of the client IP: supports rate limiting without storing IPs."""
    return b64u(sha256(hmac_sha256(secret, b"client", ip.encode())))[:32]
