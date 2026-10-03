"""Live end-to-end probe: drives a running API over real HTTP with real timing.

    python -m tools.e2e_probe --base http://127.0.0.1:8000 --kind bot

Useful after every deployment: a scripted bot must be rejected, and the whole
protocol (sessions, time windows, checkpoints, finalize, JWKS) must respond.
Run from backend/ (uses tests/sim.py signal generators).

With ``--tester-code`` the run is recorded as a labelled attack of type
``scripted_client`` (which is exactly what this script is), which both checks
that storage works on the deployment and adds an attack example for the eye and
movement models:

    python -m tools.e2e_probe --base https://your-app.vercel.app/api --origin https://your-app.vercel.app \
        --tester-code "$HP_COLLECTION_KEY" --delete-after
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import sim  # noqa: E402

from humanproof import challenges as C  # noqa: E402

CONSENT = {"version": "2026-10-v2", "biometric_processing": True, "research_opt_in": False}


def run(base: str, kind: str, origin: str, tester_code: str = "", participant: str = "probe",
        delete_after: bool = False) -> dict:
    human = kind == "simulated-human"
    rng = np.random.default_rng()
    h = {"Origin": origin}
    with httpx.Client(base_url=base, timeout=30, headers=h) as c:
        body = {"platform": "web", "consent": CONSENT, "relying_party": "probe.example"}
        if tester_code:
            body["collection"] = {"code": tester_code, "participant": participant, "label": "attack",
                                  "attack_type": "scripted_client"}
        s = c.post("/v1/sessions", json=body)
        if s.status_code != 200:
            sys.exit(f"Could not start a session ({s.status_code}): {s.text}")
        sid, receipt, storing = s.json()["session_id"], s.json().get("data_receipt"), s.json().get("storing")
        t = time.time()

        g = c.post(f"/v1/sessions/{sid}/checkpoints/gaze/start").json()["challenge"]
        ch = C.gaze_from_dict(g)
        time.sleep(ch.duration_ms / 1000 + 0.3)  # real time: the server rejects faster answers
        frames = sim.human_gaze_frames(ch, rng) if human else sim.replayed_gaze_frames(ch, rng)
        r_g = c.post(f"/v1/sessions/{sid}/checkpoints/gaze",
                     json={"frames": frames, "face_crops": sim.face_crops(rng), "viewport": {"w": 1280, "h": 800}})

        m = c.post(f"/v1/sessions/{sid}/checkpoints/motor/start").json()["challenge"]
        pts = m["polyline"]
        motor_ch = C.MotorChallenge(m["duration_ms"], [tuple(p) for p in pts[:: max(1, len(pts) // 5)]])
        time.sleep(1.5)
        r_m = c.post(f"/v1/sessions/{sid}/checkpoints/motor",
                     json={"pointer_type": "mouse", "viewport": {"w": 1280, "h": 800},
                           "samples": sim.trace_samples(motor_ch, rng, human=human)})

        v = c.post(f"/v1/sessions/{sid}/checkpoints/voice/start").json()["challenge"]
        x = sim.speech_like(rng, digital_silence=not human)
        time.sleep(2)
        r_v = c.post(f"/v1/sessions/{sid}/checkpoints/voice",
                     json={"audio_wav_b64": sim.wav_b64(x), "audio_offset_ms": 0,
                           "mouth_frames": sim.mouth_frames_for(x, rng, synced=human)})
        d = c.post(f"/v1/sessions/{sid}/finalize").json()
        jwks = c.get("/.well-known/jwks.json").json()
        deleted = None
        if delete_after and receipt:
            deleted = c.post("/v1/data/delete", json={"receipt": receipt}).json()["deleted"]
        return {
            "storing": storing,
            "data_stored": d.get("data_stored"),
            "data_receipt": None if deleted is not None else receipt,
            "samples_deleted": deleted,
            "words_challenge": v["words"],
            "checkpoint_status": [r_g.status_code, r_m.status_code, r_v.status_code],
            "decision": d["decision"],
            "scores": d["checkpoints"],
            "reasons": d["reasons"],
            "token_issued": d["attestation_token"] is not None,
            "jwks_keys": len(jwks["keys"]),
            "elapsed_s": round(time.time() - t, 1),
        }


if __name__ == "__main__":
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--kind", choices=["bot", "simulated-human"], default="bot")
    ap.add_argument("--origin", default="http://localhost:5173")
    ap.add_argument("--tester-code", default="", help="record the run as a labelled scripted_client attack")
    ap.add_argument("--participant", default="probe")
    ap.add_argument("--delete-after", action="store_true", help="delete what the run stored, using its receipt")
    a = ap.parse_args()
    print(json.dumps(run(a.base, a.kind, a.origin, a.tester_code, a.participant, a.delete_after), indent=2))
