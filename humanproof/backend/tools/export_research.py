"""Export consented research samples (feature vectors only) to CSV for retraining.

Runs on the server with access to the key ring. Output contains no images,
audio, IPs or identifiers - only per-checkpoint feature values and a label.

    HP_ENV=prod ... python -m tools.export_research --checkpoint gaze --out gaze_human.csv
"""
from __future__ import annotations

import argparse
import csv
import json

from humanproof.config import get_settings
from humanproof.context import build_context


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=["gaze", "motor", "voice"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    c = build_context(get_settings())
    rows = []
    for s in c.store.iter_research_samples(args.checkpoint):
        payload = json.loads(c.keyring.decrypt(s["payload"], f"research:{s['id']}".encode()))
        flat = {k: v for k, v in payload["features"].items() if isinstance(v, (int, float))}
        rows.append({"label": payload["label"], **flat})
    if not rows:
        print("no samples")
        return
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
