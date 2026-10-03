"""Storage behaviours that matter once there is more than one server instance.
Runs on SQLite by default and on Postgres when HP_TEST_DATABASE_URL is set."""
import os
import threading
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: F401

from humanproof.security import tokens
from humanproof.security.crypto import keys_from_master
from humanproof.storage import Store

from test_api import CONSENT, HumanMotorModel


@pytest.fixture
def target(tmp_path, database_url):
    return database_url or (tmp_path / "s.sqlite3")


@pytest.fixture
def two_stores(target):
    a, b = Store(target, b"k" * 32), Store(target, b"k" * 32)  # two "instances", one database
    yield a, b
    a.close()
    b.close()


def test_backend_follows_configuration(target):
    s = Store(target, b"k" * 32)
    assert s.backend == ("postgres" if os.environ.get("HP_TEST_DATABASE_URL") else "sqlite")
    s.close()


def test_instances_starting_at_the_same_moment_can_all_create_the_schema(target):
    stores, errors = [], []

    def boot():
        try:
            stores.append(Store(target, b"k" * 32))
        except Exception as exc:  # noqa: BLE001 - the point is that there is none
            errors.append(exc)

    threads = [threading.Thread(target=boot) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(stores) == 6
    stores[0].create_session("s", "fp", "web", {"version": 0}, 300)
    assert stores[5].get_session("s") is not None
    for s in stores:
        s.close()


def test_session_update_is_compare_and_swap_across_instances(two_stores):
    a, b = two_stores
    a.create_session("s1", "fp", "web", {"version": 0, "n": 0}, 300)
    seen_by_a, seen_by_b = a.get_session("s1")["state"], b.get_session("s1")["state"]
    assert seen_by_b == {"n": 0, "version": 0}
    assert a.update_session_state("s1", seen_by_a["version"], {**seen_by_a, "n": 1}) is True
    # b still holds version 0: its write must lose, not overwrite a's
    assert b.update_session_state("s1", seen_by_b["version"], {**seen_by_b, "n": 99}) is False
    assert b.get_session("s1")["state"] == {"n": 1, "version": 1}


def test_passkey_challenge_is_single_use_and_expires(two_stores):
    a, b = two_stores
    a.put_assertion("c1", 120, "reverify", "bank.example")
    got = b.pop_assertion("c1")
    assert got["purpose"] == "reverify" and got["rp"] == "bank.example"
    assert a.pop_assertion("c1") is None and b.pop_assertion("c1") is None
    a.put_assertion("c2", -1, "delete", "x")
    assert b.pop_assertion("c2") is None


def test_audit_chain_survives_concurrent_writers(two_stores):
    a, b = two_stores

    def write(store, tag):
        for i in range(15):
            store.audit("event", {"tag": tag, "i": i})

    threads = [threading.Thread(target=write, args=(s, t)) for s, t in ((a, "a1"), (a, "a2"), (b, "b1"), (b, "b2"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(a.execute_raw("SELECT seq FROM audit")) == 60
    assert a.verify_audit_chain() and b.verify_audit_chain()


def test_blobs_roundtrip_as_bytes(two_stores):
    a, b = two_stores
    a.create_subject("subj", "web", b"\x00\x01voiceprint\xff")
    a.add_credential("cred", "subj", b"\x02pubkey", 0)
    assert b.get_subject("subj")["voiceprint"] == b"\x00\x01voiceprint\xff"
    assert b.get_credential("cred")["public_key"] == b"\x02pubkey"
    assert a.delete_subject("subj") and b.get_credential("cred") is None  # cascade


def test_purge_removes_only_expired(two_stores):
    a, _ = two_stores
    a.create_session("old", "fp", "web", {"version": 0}, -7200)
    a.create_session("new", "fp", "web", {"version": 0}, 300)
    a.purge_expired()
    assert a.get_session("old") is None and a.get_session("new") is not None


def test_master_secret_gives_every_instance_the_same_keys():
    m = b"0123456789abcdef0123456789abcdef"
    (r1, s1, p1), (r2, s2, p2) = keys_from_master(m), keys_from_master(m)
    assert s1.jwks() == s2.jwks() and p1 == p2
    assert r2.decrypt(r1.encrypt(b"x", b"aad"), b"aad") == b"x"
    r3, s3, p3 = keys_from_master(b"another-master-secret-0123456789")
    assert s3.jwks() != s1.jwks() and p3 != p1 and r3.active_kid != r1.active_kid
    tok = tokens.sign(s1, {"aud": "rp", "hp_human": True}, 60, "iss")
    assert tokens.verify(s2, tok, issuer="iss", audience="rp")["aud"] == "rp"


def test_one_verification_spread_over_two_server_instances(settings, transcriber, clock):
    """Each request may land on a different instance. With a shared database and
    the same master secret, the flow still completes and the token verifies anywhere."""
    import numpy as np
    from fastapi.testclient import TestClient

    import sim
    from humanproof import challenges as C
    from humanproof.main import create_app

    rng = np.random.default_rng(1)
    apps = [create_app(settings, transcriber=transcriber) for _ in range(2)]
    with TestClient(apps[0]) as a, TestClient(apps[1]) as b:
        for app in apps:
            app.state.ctx.models._models["motor"] = HumanMotorModel()
        sid = a.post("/v1/sessions", json={"platform": "web", "consent": CONSENT}).json()["session_id"]
        ch = C.gaze_from_dict(b.post(f"/v1/sessions/{sid}/checkpoints/gaze/start").json()["challenge"])
        clock.advance(ch.duration_ms / 1000 + 0.5)
        assert a.post(f"/v1/sessions/{sid}/checkpoints/gaze", json={
            "frames": sim.human_gaze_frames(ch, rng), "face_crops": sim.face_crops(rng),
            "viewport": {"w": 1280, "h": 800}}).status_code == 200
        # replaying the same checkpoint on the other instance is refused
        assert b.post(f"/v1/sessions/{sid}/checkpoints/gaze/start").status_code == 409

        b.post(f"/v1/sessions/{sid}/checkpoints/motor/start")
        mch = C.motor_from_dict(apps[0].state.ctx.store.get_session(sid)["state"]["challenges"]["motor"])
        clock.advance(5)
        assert a.post(f"/v1/sessions/{sid}/checkpoints/motor", json={
            "pointer_type": "mouse", "samples": sim.trace_samples(mch, rng), "viewport": {"w": 1280, "h": 800}
        }).status_code == 200

        words = a.post(f"/v1/sessions/{sid}/checkpoints/voice/start").json()["challenge"]["words"]
        transcriber.next_text = " ".join(words)
        x = sim.speech_like(rng)
        clock.advance(4)
        assert b.post(f"/v1/sessions/{sid}/checkpoints/voice", json={
            "audio_wav_b64": sim.wav_b64(x), "audio_offset_ms": 0, "mouth_frames": sim.mouth_frames_for(x, rng)
        }).status_code == 200

        d = a.post(f"/v1/sessions/{sid}/finalize").json()
        assert d["decision"] == "pass"
        assert b.post(f"/v1/sessions/{sid}/finalize").status_code == 409
        # the token minted by instance A verifies on instance B
        assert a.get("/.well-known/jwks.json").json() == b.get("/.well-known/jwks.json").json()
        assert b.post("/v1/tokens/verify", json={"token": d["attestation_token"], "audience": "humanproof"}).json()["valid"]
    for app in apps:
        app.state.ctx.store.close()
    assert time.time() > 0
