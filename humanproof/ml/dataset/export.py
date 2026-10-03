"""Decrypt the stored verification samples into a training dataset on local disk.

Reads the same settings as the server (``HP_DATABASE_URL`` plus ``HP_MASTER_SECRET``
or ``HP_KEK_KEYRING``), so it must be run by someone who holds the encryption key.

    HP_DATABASE_URL=postgres://... HP_MASTER_SECRET=... python ml/dataset/export.py --out /data/hp-export

What it writes (only finished attempts; abandoned ones are skipped):

    summary.json              counts per checkpoint, label and split
    sessions.csv              one row per attempt
    gaze.jsonl                challenge + per-frame eye/head measurements + features
    motor.jsonl               challenge + pointer samples + features
    voice.jsonl               challenge words + mouth measurements + features
    face.jsonl                per-attempt face-check details
    voice.csv + audio/        path,label(bonafide|spoof),split,...  -> ml/voice/train_antispoof.py --csv
    faces/index.csv + faces/  same layout as ml/face/extract_faces.py -> ml/face/train_face.py --extra

Labels are per checkpoint (see backend/humanproof/collection.py). Only samples
labelled ``human`` or ``attack`` by a tester go into voice.csv and faces/index.csv.
Unlabelled opt-in samples are written to the .jsonl files (label "unlabelled") for
monitoring; their audio and images are written only with ``--include-unlabelled-media``.

Splits are by participant, so nobody appears in both training and test data.

The output is unencrypted biometric data. Keep it on an encrypted disk, do not
commit it, do not upload it as a CI artefact, and delete it when training is done.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from humanproof import collection  # noqa: E402
from humanproof.config import Settings  # noqa: E402
from humanproof.keys import resolve_keys  # noqa: E402
from humanproof.security.crypto import CryptoError  # noqa: E402
from humanproof.storage import Store  # noqa: E402

SPLITS = ("train", "val", "test")


def assign_splits(participants: set[str]) -> dict[str, str]:
    """Deterministic train/val/test split by participant (roughly 70/15/15).

    Ordered by a hash of the ID, so the assignment does not depend on who joined
    first. With fewer than four participants there is no validation split, and
    with one participant there is nothing to hold out at all.
    """
    order = sorted(participants, key=lambda p: hashlib.sha256(b"hp/split/v1" + p.encode()).hexdigest())
    n = len(order)
    n_test = max(1, math.ceil(0.15 * n)) if n >= 2 else 0
    n_val = max(1, math.ceil(0.15 * n)) if n >= 4 else 0
    out = {}
    for i, p in enumerate(order):
        out[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def export(store: Store, keyring, out: Path, include_unlabelled_media: bool = False) -> dict:
    os.umask(0o077)  # files readable by the current user only
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: metadata only, to fix the participant split before writing anything.
    meta_rows = store.execute_raw(
        "SELECT DISTINCT participant FROM samples WHERE participant IS NOT NULL AND decision IS NOT NULL")
    splits = assign_splits({r["participant"] for r in meta_rows})

    jsonl = {cp: open(out / f"{cp}.jsonl", "w") for cp in collection.CHECKPOINTS}
    voice_rows, face_rows = [], []
    sessions: dict[str, dict] = {}
    counts: Counter = Counter()
    skipped_unfinished = undecryptable = 0
    media_index: defaultdict[str, int] = defaultdict(int)

    try:
        for row in store.iter_samples():
            if row["decision"] is None:
                skipped_unfinished += 1
                continue
            try:
                data = collection.unpack(keyring.decrypt(row["payload"], collection.sample_aad(row["id"])))
            except (CryptoError, ValueError):
                undecryptable += 1
                continue
            cap, cp, label = row["capture_id"], row["checkpoint"], row["label"]
            split = splits.get(row["participant"], "unlabelled") if row["source"] == "labelled" else "unlabelled"
            common = {
                "capture_id": cap, "label": label, "attack_type": row["attack_type"], "participant": row["participant"],
                "source": row["source"], "platform": row["platform"], "split": split, "score": row["score"],
                "decision": row["decision"], "day": _day(row["created_at"]),
            }
            s = sessions.setdefault(cap, {
                "capture_id": cap, "day": common["day"], "source": row["source"], "participant": row["participant"] or "",
                "attack_type": row["attack_type"] or "", "platform": row["platform"], "split": split,
                "decision": row["decision"],
            })
            if row["kind"] == "signals":
                s[f"{cp}_label"], s[f"{cp}_score"] = label, row["score"]
                jsonl[cp].write(json.dumps({**common, "data": data}, separators=(",", ":")) + "\n")
                counts[(cp, "signals", label, split)] += 1
                continue

            labelled = label in ("human", "attack")
            if not labelled and not (include_unlabelled_media and label == "unlabelled"):
                continue  # "unknown": the tester attacked a different checkpoint; not usable as a label
            if row["kind"] == "audio":
                cls = {"human": "bonafide", "attack": "spoof"}.get(label, "unlabelled")
                path = out / "audio" / split / cls / f"{cap}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                if labelled:
                    voice_rows.append({"path": str(path.resolve()), "label": cls, "split": split,
                                       "participant": row["participant"], "attack_type": row["attack_type"] or "",
                                       "platform": row["platform"]})
                counts[("voice", "audio", label, split)] += 1
            elif row["kind"] == "image":
                cls = {"human": "real", "attack": "fake"}.get(label, "unlabelled")
                i = media_index[cap]
                media_index[cap] += 1
                rel = Path(split) / cls / f"{cap}_{i:02d}.jpg"
                path = out / "faces" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                if labelled:
                    face_rows.append({"path": str(rel), "label": cls, "split": split, "source": "collected",
                                      "method": row["attack_type"] or "original", "video": cap})
                counts[("face", "image", label, split)] += 1
    finally:
        for fh in jsonl.values():
            fh.close()

    with open(out / "voice.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["path", "label", "split", "participant", "attack_type", "platform"])
        w.writeheader()
        w.writerows(voice_rows)
    (out / "faces").mkdir(exist_ok=True)
    with open(out / "faces" / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["path", "label", "split", "source", "method", "video"])
        w.writeheader()
        w.writerows(face_rows)
    cols = ["capture_id", "day", "source", "participant", "attack_type", "platform", "split", "decision"]
    cols += [f"{cp}_{k}" for cp in collection.CHECKPOINTS for k in ("label", "score")]
    with open(out / "sessions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, restval="")
        w.writeheader()
        w.writerows(sessions.values())

    summary = {
        "exported_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attempts": len(sessions),
        "labelled_attempts": sum(1 for s in sessions.values() if s["source"] == "labelled"),
        "participants": len(splits),
        "participants_by_split": dict(Counter(splits.values())),
        "skipped_unfinished_samples": skipped_unfinished,
        "undecryptable_samples": undecryptable,
        "counts": [
            {"checkpoint": cp, "kind": kind, "label": label, "split": split, "n": n}
            for (cp, kind, label, split), n in sorted(counts.items())
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-unlabelled-media", action="store_true",
                    help="also write audio and images from opt-in (unlabelled) attempts")
    args = ap.parse_args()

    settings = Settings()
    if not settings.has_persistent_keys:
        sys.exit("No encryption key configured. Set HP_MASTER_SECRET (or HP_KEK_KEYRING) to the server's value.")
    keyring, _, pairwise = resolve_keys(settings)
    store = Store(settings.store_target, audit_key=pairwise)  # the audit key is unused: this only reads
    try:
        summary = export(store, keyring, args.out, args.include_unlabelled_media)
    finally:
        store.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=2))
    if summary["undecryptable_samples"]:
        print(f"WARNING: {summary['undecryptable_samples']} samples could not be decrypted "
              "(written under a different key). Check HP_MASTER_SECRET.", file=sys.stderr)
    print(f"Wrote dataset to {args.out}. It is unencrypted biometric data: delete it when you are done.")


if __name__ == "__main__":
    main()
