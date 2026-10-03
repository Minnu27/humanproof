"""Storing verification data for training, with consent.

Two ways data gets stored, and nothing is stored otherwise:

* **Opt-in** - an ordinary user ticks the optional box(es) on the consent screen.
  Their samples are stored *unlabelled*. Unlabelled data is for monitoring and
  calibration only; it never trains a model, because its only "label" would be
  the system's own verdict, and learning from your own verdicts teaches the
  system to accept whatever once slipped through.
* **Labelled tester sessions** - a trusted tester enters the private collection
  code and declares the session a real human or a specific kind of attack.
  These are the only sessions used for training.

Labels are per checkpoint, not per session: someone testing a cloned voice is
still moving a real mouse with real eyes. ``ATTACK_TARGETS`` records which
checkpoints an attack type actually fakes; the others are stored as ``unknown``
and left out of training.

Every payload is envelope-encrypted with a key that is not in the database, and
bound to its row id so a blob cannot be moved to another row.
"""
from __future__ import annotations

import json
import logging
import secrets
import time
import zlib
from typing import TYPE_CHECKING, Any

from .security.crypto import b64u, hmac_sha256, sha256

if TYPE_CHECKING:
    from .context import Context

log = logging.getLogger("humanproof.collection")

CHECKPOINTS = ("gaze", "face", "motor", "voice")

ATTACK_TARGETS: dict[str, frozenset[str]] = {
    "replay_video": frozenset({"gaze"}),            # a recording of a real person: the face itself is genuine
    "photo": frozenset({"gaze", "face"}),           # printed or on-screen still image
    "face_swap": frozenset({"face"}),               # live deepfake filter over a real person, who does follow the dot
    "synthetic_face": frozenset({"gaze", "face"}),  # fully generated face or avatar video
    "tts_voice": frozenset({"voice"}),
    "voice_clone": frozenset({"voice"}),
    "audio_replay": frozenset({"voice"}),
    "bot_pointer": frozenset({"motor"}),
    # No person at all. The numbers it sends are fabricated, which is exactly what the
    # eye and movement models must learn; the image and audio it sends could be
    # anything (noise, a stock photo), so they are not examples of a fake face or voice.
    "scripted_client": frozenset({"gaze", "motor"}),
    "other": frozenset(),                           # stored for study, never used as a training label
}


def receipt_hash(receipt: str) -> str:
    return b64u(sha256(b"hp/receipt/v1" + receipt.encode()))


def subject_hash(pairwise_secret: bytes, subject_id: str) -> str:
    return b64u(hmac_sha256(pairwise_secret, b"sample-subject", subject_id.encode()))


def new_capture(mode: str, *, media: bool, participant: str | None = None, label: str = "unlabelled",
                attack_type: str | None = None) -> tuple[dict, str]:
    """Returns (what goes in the session state, the receipt shown to the person).

    Only a hash of the receipt is kept, so the stored rows cannot be tied back to
    the receipt holder by anyone who merely reads the database.
    """
    receipt = b64u(secrets.token_bytes(18))
    capture = {
        "mode": mode, "media": media, "participant": participant, "label": label, "attack_type": attack_type,
        "capture_id": b64u(secrets.token_bytes(12)), "receipt_hash": receipt_hash(receipt),
    }
    return capture, receipt


def label_for(capture: dict, checkpoint: str) -> str:
    if capture["mode"] != "labelled":
        return "unlabelled"
    if capture["label"] == "human":
        return "human"
    return "attack" if checkpoint in ATTACK_TARGETS.get(capture["attack_type"] or "other", frozenset()) else "unknown"


def pack_json(obj: Any) -> bytes:
    return b"Z" + zlib.compress(json.dumps(obj, separators=(",", ":")).encode(), 6)


def pack_raw(data: bytes) -> bytes:
    return b"R" + data


def unpack(blob: bytes) -> Any:
    """Inverse of pack_json / pack_raw: returns the JSON value, or raw bytes."""
    if blob[:1] == b"Z":
        return json.loads(zlib.decompress(blob[1:]))
    if blob[:1] == b"R":
        return blob[1:]
    raise ValueError("unknown sample encoding")


def sample_aad(sample_id: str) -> bytes:
    return f"sample:{sample_id}".encode()


def store(c: "Context", state: dict, checkpoint: str, kind: str, packed: bytes, score: float | None) -> bool:
    """Encrypt and store one sample if this session is being captured.

    ``kind`` is ``signals`` (numbers), ``audio`` or ``image``. Audio and images
    are kept only in labelled sessions or when the person opted in to that
    separately. A storage failure is logged and never fails the verification.
    """
    capture = state.get("capture")
    if not capture or not c.collection_ready:
        return False
    if kind != "signals" and not capture["media"]:
        return False
    sample_id = b64u(secrets.token_bytes(16))
    now = time.time()
    try:
        c.store.add_sample(
            id=sample_id, capture_id=capture["capture_id"], created_at=now,
            expires_at=now + c.settings.data_retention_days * 86400,
            checkpoint=checkpoint, kind=kind, label=label_for(capture, checkpoint),
            attack_type=capture["attack_type"], participant=capture["participant"],
            source=capture["mode"], platform=state["platform"], consent_version=state["consent_version"],
            score=None if score is None else float(score), decision=None,
            receipt_hash=capture["receipt_hash"], subject_hash=None,
            payload=c.keyring.encrypt(packed, sample_aad(sample_id)),
        )
        return True
    except Exception:
        log.exception("Could not store %s/%s sample", checkpoint, kind)
        return False


def finish(c: "Context", state: dict, decision: str, subject_id: str | None) -> None:
    capture = state.get("capture")
    if not capture or not c.collection_ready:
        return
    try:
        c.store.finish_capture(
            capture["capture_id"], decision,
            subject_hash(c.pairwise_secret, subject_id) if subject_id else None,
        )
    except Exception:
        log.exception("Could not record the outcome for capture")
