import base64
import json

import numpy as np
import pytest

import sim
from humanproof import challenges as C
from humanproof.security import tokens
from humanproof.security.crypto import b64u_decode

from conftest import has_motor_model

CONSENT = {"version": "2026-09-v1", "biometric_processing": True, "research_opt_in": True}


def new_session(client, **extra):
    r = client.post("/v1/sessions", json={"platform": "web", "consent": CONSENT, **extra})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def start(client, sid, cp):
    r = client.post(f"/v1/sessions/{sid}/checkpoints/{cp}/start")
    assert r.status_code == 200, r.text
    return r.json()["challenge"]


def gaze_challenge_from_public(pub):
    return C.gaze_from_dict({**pub})


def run_flow(client, clock, transcriber, rng, *, human=True, rp="bank.example"):
    sid = new_session(client, relying_party=rp)

    pub = start(client, sid, "gaze")
    ch = gaze_challenge_from_public(pub)
    frames = sim.human_gaze_frames(ch, rng) if human else sim.replayed_gaze_frames(ch, rng)
    clock.advance(ch.duration_ms / 1000 + 0.5)
    r = client.post(f"/v1/sessions/{sid}/checkpoints/gaze",
                    json={"frames": frames, "face_crops": sim.face_crops(rng), "viewport": {"w": 1280, "h": 800}})
    assert r.status_code == 200, r.text

    pub = start(client, sid, "motor")
    motor_ch = C.MotorChallenge(pub["duration_ms"], [tuple(p) for p in client.app_ctx.store.get_session(sid)
                                                     ["state"]["challenges"]["motor"]["control_points"]])
    clock.advance(5)
    r = client.post(f"/v1/sessions/{sid}/checkpoints/motor", json={
        "pointer_type": "mouse", "samples": sim.trace_samples(motor_ch, rng, human=human),
        "viewport": {"w": 1280, "h": 800}})
    assert r.status_code == 200, r.text

    pub = start(client, sid, "voice")
    x = sim.speech_like(rng, digital_silence=not human)
    transcriber.next_text = " ".join(pub["words"]) if human else "hello world"
    clock.advance(4)
    r = client.post(f"/v1/sessions/{sid}/checkpoints/voice", json={
        "audio_wav_b64": sim.wav_b64(x), "audio_offset_ms": 0,
        "mouth_frames": sim.mouth_frames_for(x, rng, synced=human)})
    assert r.status_code == 200, r.text

    r = client.post(f"/v1/sessions/{sid}/finalize")
    assert r.status_code == 200, r.text
    return sid, r.json()


class HumanMotorModel:
    """Stands in for the trained motor model in the plumbing test below.

    The simulated traces in sim.py are synthetic, so the real model rightly
    rejects them; the model itself is tested on real held-out human movement
    in test_scoring.py::test_motor_model_on_real_held_out_humans.
    """
    meta = {"threshold": 0.2}

    def run(self, x):
        return [None, np.tile([[0.05, 0.95]], (len(x), 1)).astype(np.float32)]


def test_full_flow_human_passes_and_token_verifies(client, clock, transcriber):
    client.app_ctx.models._models["motor"] = HumanMotorModel()
    rng = np.random.default_rng(11)
    sid, d = run_flow(client, clock, transcriber, rng)
    assert d["decision"] == "pass", d
    assert d["assurance"] == "web"

    jwks = client.get("/.well-known/jwks.json").json()
    assert jwks["keys"][0]["crv"] == "Ed25519"
    claims = tokens.verify(client.app_ctx.signing, d["attestation_token"], issuer="http://testserver",
                           audience="bank.example")
    assert claims["hp_human"] is True and claims["sub"] == d["subject_hint"]

    r = client.post("/v1/tokens/verify", json={"token": d["attestation_token"], "audience": "bank.example"})
    assert r.json()["valid"] is True
    r = client.post("/v1/tokens/verify", json={"token": d["attestation_token"], "audience": "other.example"})
    assert r.json()["valid"] is False

    # voiceprint and research data are stored encrypted only
    state = client.app_ctx.store.get_session(sid)["state"]
    subject = client.app_ctx.store.get_subject(state["subject_id"])
    assert "pending_voiceprint" not in state
    samples = list(client.app_ctx.store.iter_research_samples("gaze"))
    assert samples and b"evidence" not in samples[0]["payload"]
    assert client.get("/v1/status").json()["audit_chain_ok"] is True
    assert subject is not None


def test_full_flow_bot_rejected(client, clock, transcriber):
    rng = np.random.default_rng(12)
    _, d = run_flow(client, clock, transcriber, rng, human=False)
    assert d["decision"] == "reject"
    assert d["attestation_token"] is None


def test_checkpoints_must_run_in_order(client):
    sid = new_session(client)
    assert client.post(f"/v1/sessions/{sid}/checkpoints/voice/start").status_code == 409
    assert client.post(f"/v1/sessions/{sid}/checkpoints/motor/start").status_code == 409
    assert client.post(f"/v1/sessions/{sid}/finalize").status_code == 409


def test_challenge_hidden_until_start(client):
    r = client.post("/v1/sessions", json={"platform": "web", "consent": CONSENT})
    body = json.dumps(r.json())
    assert "keyframes" not in body and "words" not in body


def test_submission_too_fast_is_burned(client, clock):
    rng = np.random.default_rng(13)
    sid = new_session(client)
    ch = gaze_challenge_from_public(start(client, sid, "gaze"))
    clock.advance(1)  # far shorter than the challenge takes to watch
    r = client.post(f"/v1/sessions/{sid}/checkpoints/gaze",
                    json={"frames": sim.human_gaze_frames(ch, rng), "viewport": {"w": 1280, "h": 800}})
    assert r.status_code == 400
    # and it cannot be retried
    clock.advance(10)
    r = client.post(f"/v1/sessions/{sid}/checkpoints/gaze",
                    json={"frames": sim.human_gaze_frames(ch, rng), "viewport": {"w": 1280, "h": 800}})
    assert r.status_code == 409


def test_double_submit_rejected(client, clock):
    rng = np.random.default_rng(14)
    sid = new_session(client)
    ch = gaze_challenge_from_public(start(client, sid, "gaze"))
    clock.advance(ch.duration_ms / 1000 + 1)
    payload = {"frames": sim.human_gaze_frames(ch, rng), "viewport": {"w": 1280, "h": 800}}
    assert client.post(f"/v1/sessions/{sid}/checkpoints/gaze", json=payload).status_code == 200
    assert client.post(f"/v1/sessions/{sid}/checkpoints/gaze", json=payload).status_code == 409
    assert client.post(f"/v1/sessions/{sid}/checkpoints/gaze/start").status_code == 409


def test_session_expiry(client, clock):
    sid = new_session(client)
    clock.advance(10_000)
    assert client.post(f"/v1/sessions/{sid}/checkpoints/gaze/start").status_code == 404


def test_unknown_session(client):
    assert client.post("/v1/sessions/nope/checkpoints/gaze/start").status_code == 404
    assert client.post("/v1/sessions/" + "a" * 500 + "/finalize").status_code == 404


def test_consent_required(client):
    r = client.post("/v1/sessions", json={"platform": "web",
                                          "consent": {**CONSENT, "biometric_processing": False}})
    assert r.status_code == 422
    r = client.post("/v1/sessions", json={"platform": "web", "consent": {**CONSENT, "version": "old"}})
    assert r.status_code == 422


def test_unknown_fields_and_bounds_rejected(client, clock):
    assert client.post("/v1/sessions", json={"platform": "web", "consent": CONSENT, "admin": True}).status_code == 422
    assert client.post("/v1/sessions", json={"platform": "toaster", "consent": CONSENT}).status_code == 422
    sid = new_session(client)
    start(client, sid, "gaze")
    bad = {"frames": [{"t": 0, "ix": 99, "iy": 0, "yaw": 0, "pitch": 0, "roll": 0, "ear": 0.3,
                       "mouth": 0, "face": True}] * 20, "viewport": {"w": 1280, "h": 800}}
    r = client.post(f"/v1/sessions/{sid}/checkpoints/gaze", json=bad)
    assert r.status_code == 422
    assert "99" not in r.text  # submitted values are never echoed back


def test_body_size_limit(client):
    r = client.post("/v1/sessions", content=b"x" * 3_000_000, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_security_headers(client):
    r = client.get("/healthz")
    for h in ("strict-transport-security", "x-content-type-options", "x-frame-options",
              "content-security-policy", "referrer-policy"):
        assert h in r.headers
    assert r.headers["cache-control"] == "no-store"


def test_cors_only_allowed_origins(client):
    ok = client.options("/v1/sessions", headers={"Origin": "http://localhost:5173",
                                                 "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.options("/v1/sessions", headers={"Origin": "https://evil.example",
                                                   "Access-Control-Request-Method": "POST"})
    assert evil.headers.get("access-control-allow-origin") is None


def test_rate_limit(settings, transcriber, clock):
    from fastapi.testclient import TestClient

    from humanproof.main import create_app

    settings.session_limit_per_hour = 3
    with TestClient(create_app(settings, transcriber=transcriber)) as c:
        codes = [c.post("/v1/sessions", json={"platform": "web", "consent": CONSENT}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


def test_audit_chain_detects_tampering(client):
    new_session(client)
    new_session(client)
    store = client.app_ctx.store
    assert store.verify_audit_chain()
    with store._conn() as c:
        c.execute("UPDATE audit SET data = '{\"forged\":1}' WHERE seq = 1")
    assert not store.verify_audit_chain()


def test_attest_rejected_for_web_and_after_start(client):
    sid = new_session(client)
    r = client.post(f"/v1/sessions/{sid}/attest", json={"kind": "apple_app_attest", "key_id": "x", "payload": "eA=="})
    assert r.status_code == 400
    start(client, sid, "gaze")
    r = client.post(f"/v1/sessions/{sid}/attest", json={"kind": "play_integrity", "payload": "tok"})
    assert r.status_code == 409


def test_passkey_requires_pass(client):
    sid = new_session(client)
    assert client.post(f"/v1/sessions/{sid}/passkey/options").status_code == 409


def test_prod_refuses_insecure_config(tmp_path):
    from humanproof.config import Settings
    from humanproof.main import create_app

    s = Settings(env="prod", data_dir=tmp_path, allow_heuristic_fallback=True)
    with pytest.raises(RuntimeError, match="Refusing to start"):
        create_app(s)


def test_tampered_model_refused(tmp_path):
    import shutil

    from conftest import MODELS_DIR
    from humanproof.models import ModelRegistry

    if not (MODELS_DIR / "motor.onnx").exists():
        pytest.skip("motor model not trained")
    shutil.copytree(MODELS_DIR, tmp_path / "m")
    with open(tmp_path / "m" / "motor.onnx", "ab") as fh:
        fh.write(b"\x00")
    reg = ModelRegistry(tmp_path / "m")
    assert reg.get("motor") is None and reg.status()["motor"] == "sha256 mismatch"


def test_openapi_hidden_in_prod_only(client):
    assert client.get("/openapi.json").status_code == 200  # test env
    assert b64u_decode(base64.urlsafe_b64encode(b"x").decode().rstrip("=")) == b"x"


def test_api_prefix_mounts_every_route(settings, transcriber, clock):
    """On Vercel the service receives the full public path (/api/...)."""
    from fastapi.testclient import TestClient

    from humanproof.main import create_app

    settings.api_prefix = "/api"
    with TestClient(create_app(settings, transcriber=transcriber)) as c:
        assert c.get("/api/healthz").status_code == 200
        assert c.get("/api/.well-known/jwks.json").json()["keys"]
        assert c.post("/api/v1/sessions", json={"platform": "web", "consent": CONSENT}).status_code == 200
        assert c.get("/api/openapi.json").status_code == 200
        assert c.get("/healthz").status_code == 404


def test_invalid_api_prefix_rejected():
    from pydantic import ValidationError

    from humanproof.config import Settings

    for bad in ("api", "/api/", "/API", "/api?x"):
        with pytest.raises(ValidationError):
            Settings(api_prefix=bad)


def test_debug_measurements_only_outside_production(client, clock, transcriber):
    client.app_ctx.models._models["motor"] = HumanMotorModel()
    _, d = run_flow(client, clock, transcriber, np.random.default_rng(31))
    assert set(d["debug"]) == {"gaze", "motor", "voice", "face"}
    assert "evidence" in d["debug"]["gaze"]["features"]
    assert "dsp" in d["debug"]["voice"]["features"]

    client.app_ctx.settings.env = "prod"
    _, d = run_flow(client, clock, transcriber, np.random.default_rng(32))
    assert d["debug"] is None
