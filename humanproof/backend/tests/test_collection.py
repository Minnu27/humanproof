"""Data collection: what is stored, under which consent, with which labels, and how it is deleted."""
import base64
import time

import numpy as np
import pytest

from humanproof import collection
from humanproof.security.crypto import CryptoError

from conftest import TEST_COLLECTION_KEY
import sim
from test_api import CONSENT, HumanMotorModel, gaze_challenge_from_public, new_session, run_flow, start

NO_STORAGE = {**CONSENT, "research_opt_in": False}
WITH_MEDIA = {**CONSENT, "media_opt_in": True}


def as_tester(label="human", attack_type=None, participant="p01", code=TEST_COLLECTION_KEY):
    c = {"code": code, "participant": participant, "label": label}
    if attack_type:
        c["attack_type"] = attack_type
    return c


def samples(client):
    return list(client.app_ctx.store.iter_samples())


def decrypt(client, row):
    return collection.unpack(client.app_ctx.keyring.decrypt(row["payload"], collection.sample_aad(row["id"])))


def kinds(rows):
    return sorted((r["checkpoint"], r["kind"]) for r in rows)


@pytest.fixture
def human_client(client):
    client.app_ctx.models._models["motor"] = HumanMotorModel()
    return client


def test_nothing_is_stored_without_opt_in(human_client, clock, transcriber):
    _, d = run_flow(human_client, clock, transcriber, np.random.default_rng(1), consent=NO_STORAGE)
    assert human_client.last_session["storing"] == "nothing" and human_client.last_session["data_receipt"] is None
    assert d["decision"] == "pass" and d["data_stored"] is False
    assert samples(human_client) == []


def test_opt_in_stores_measurements_only_unlabelled_and_encrypted(human_client, clock, transcriber):
    sid, d = run_flow(human_client, clock, transcriber, np.random.default_rng(2))
    assert human_client.last_session["storing"] == "measurements" and d["data_stored"] is True
    rows = samples(human_client)
    assert kinds(rows) == [("face", "signals"), ("gaze", "signals"), ("motor", "signals"), ("voice", "signals")]
    assert {r["label"] for r in rows} == {"unlabelled"} and {r["source"] for r in rows} == {"opt_in"}
    assert all(r["decision"] == "pass" and r["subject_hash"] for r in rows)  # outcome recorded at finalize
    assert all(r["expires_at"] > time.time() + 300 * 86400 for r in rows)
    gaze = next(r for r in rows if r["checkpoint"] == "gaze")
    assert b"frames" not in gaze["payload"]  # ciphertext
    data = decrypt(human_client, gaze)
    assert len(data["frames"]) > 100 and len(data["challenge"]["keyframes"]) == 13 and "evidence" in data["features"]
    motor = decrypt(human_client, next(r for r in rows if r["checkpoint"] == "motor"))
    assert len(motor["submission"]["samples"]) > 50 and "control_points" in motor["challenge"]
    # the session id itself is not stored with the samples
    assert all(sid not in str(r) for r in rows)


def test_media_opt_in_also_stores_audio_and_face_snapshots(human_client, clock, transcriber):
    run_flow(human_client, clock, transcriber, np.random.default_rng(3), consent=WITH_MEDIA)
    assert human_client.last_session["storing"] == "measurements_and_media"
    rows = samples(human_client)
    assert kinds(rows).count(("face", "image")) == 4 and kinds(rows).count(("voice", "audio")) == 1
    wav = decrypt(human_client, next(r for r in rows if r["kind"] == "audio"))
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    jpeg = decrypt(human_client, next(r for r in rows if r["kind"] == "image"))
    assert jpeg[:2] == b"\xff\xd8"


def test_sample_cannot_be_decrypted_as_another_row(human_client, clock, transcriber):
    run_flow(human_client, clock, transcriber, np.random.default_rng(4))
    a, b = samples(human_client)[:2]
    with pytest.raises(CryptoError):
        human_client.app_ctx.keyring.decrypt(a["payload"], collection.sample_aad(b["id"]))


def test_tester_code_is_required_and_checked(client):
    r = client.post("/v1/sessions", json={"platform": "web", "consent": CONSENT, "collection": as_tester(code="guess")})
    assert r.status_code == 403
    bad = [{**as_tester(), "attack_type": "photo"}, as_tester(label="attack"), {**as_tester(), "participant": "Jane Doe"}]
    for body in bad:  # human+attack_type, attack without a type, a real-looking name
        assert client.post("/v1/sessions", json={"platform": "web", "consent": CONSENT, "collection": body}).status_code == 422


def test_short_tester_code_disables_tester_mode_even_in_a_demo(client):
    client.app_ctx.settings.collection_key = "short"
    assert client.get("/v1/data/policy").json()["tester_mode"] is False
    r = client.post("/v1/sessions", json={"platform": "web", "consent": CONSENT, "collection": as_tester(code="short")})
    assert r.status_code == 403 and "not enabled" in r.json()["detail"]


def test_labelled_human_session_stores_everything_and_issues_no_token(human_client, clock, transcriber):
    _, d = run_flow(human_client, clock, transcriber, np.random.default_rng(5), consent=NO_STORAGE,
                    collection=as_tester(participant="p07"))
    assert human_client.last_session["labelled"] is True
    assert human_client.last_session["storing"] == "measurements_and_media"
    # The verdict is reported, but a tester session never becomes a credential.
    assert d["decision"] == "pass" and d["attestation_token"] is None and d["subject_hint"] is None
    assert "tester session: no proof token is issued" in d["reasons"]
    rows = samples(human_client)
    assert {r["label"] for r in rows} == {"human"} and {r["participant"] for r in rows} == {"p07"}
    assert {r["source"] for r in rows} == {"labelled"} and len(rows) == 4 + 4 + 1
    assert all(r["subject_hash"] is None for r in rows)


@pytest.mark.parametrize("attack_type,attacked", [
    ("tts_voice", {"voice"}), ("bot_pointer", {"motor"}), ("face_swap", {"face"}),
    ("replay_video", {"gaze"}), ("scripted_client", {"gaze", "motor"}), ("other", set()),
])
def test_attack_labels_apply_only_to_the_checkpoints_that_were_faked(human_client, clock, transcriber,
                                                                     attack_type, attacked):
    run_flow(human_client, clock, transcriber, np.random.default_rng(6), human=False,
             collection=as_tester(label="attack", attack_type=attack_type))
    rows = samples(human_client)
    assert {r["attack_type"] for r in rows} == {attack_type}
    for r in rows:
        assert r["label"] == ("attack" if r["checkpoint"] in attacked else "unknown"), r["checkpoint"]


def test_person_can_delete_their_data_with_the_receipt(human_client, clock, transcriber):
    run_flow(human_client, clock, transcriber, np.random.default_rng(7), consent=WITH_MEDIA)
    receipt = human_client.last_session["data_receipt"]
    run_flow(human_client, clock, transcriber, np.random.default_rng(8), consent=WITH_MEDIA)  # someone else
    before = len(samples(human_client))
    assert human_client.post("/v1/data/delete", json={"receipt": "not-the-right-receipt"}).json() == {"deleted": 0}
    assert human_client.post("/v1/data/delete", json={"receipt": receipt}).json() == {"deleted": 9}
    assert len(samples(human_client)) == before - 9
    assert receipt not in str(human_client.app_ctx.store.execute_raw("SELECT receipt_hash FROM samples"))


def test_samples_expire(human_client, clock, transcriber):
    run_flow(human_client, clock, transcriber, np.random.default_rng(9))
    store = human_client.app_ctx.store
    store.execute_raw("UPDATE samples SET expires_at = ? WHERE checkpoint = ?", (time.time() - 1, "gaze"))
    store.purge_expired()
    assert ("gaze", "signals") not in kinds(samples(human_client)) and len(samples(human_client)) == 3


def test_abandoned_attempts_are_not_kept(human_client, clock, transcriber):
    """Someone who quits half-way never sees the receipt, so their samples are removed."""
    run_flow(human_client, clock, transcriber, np.random.default_rng(12))  # finished: kept
    rng = np.random.default_rng(13)
    sid = new_session(human_client, consent=WITH_MEDIA)
    ch = gaze_challenge_from_public(start(human_client, sid, "gaze"))
    clock.advance(ch.duration_ms / 1000 + 0.5)
    r = human_client.post(f"/v1/sessions/{sid}/checkpoints/gaze", json={
        "frames": sim.human_gaze_frames(ch, rng), "face_crops": sim.face_crops(rng), "viewport": {"w": 1280, "h": 800}})
    assert r.status_code == 200, r.text
    store = human_client.app_ctx.store
    assert len(samples(human_client)) == 4 + 6
    store.purge_expired()
    assert len(samples(human_client)) == 4 + 6  # still within the session's lifetime
    store.execute_raw("UPDATE samples SET created_at = ? WHERE decision IS NULL", (time.time() - 7200,))
    store.purge_expired()
    assert len(samples(human_client)) == 4 and {r["decision"] for r in samples(human_client)} == {"pass"}


def test_collection_refuses_without_a_persistent_key(settings, transcriber, clock):
    """Samples encrypted with a throwaway key could never be read again."""
    from fastapi.testclient import TestClient

    from humanproof.main import create_app

    settings.master_secret = ""
    app = create_app(settings, transcriber=transcriber)
    with TestClient(app) as c:
        st = c.get("/v1/status").json()["collection"]
        assert st["active"] is False and "persistent encryption key" in st["not_active_because"]
        assert c.get("/v1/data/policy").json() == {"sharing": False, "tester_mode": False, "retention_days": 365}
        r = c.post("/v1/sessions", json={"platform": "web", "consent": {**CONSENT, "media_opt_in": True}})
        assert r.json()["storing"] == "nothing" and r.json()["data_receipt"] is None
        r = c.post("/v1/sessions", json={"platform": "web", "consent": CONSENT, "collection": as_tester()})
        assert r.status_code == 403
    app.state.ctx.store.close()


def test_status_reports_what_is_stored(human_client, clock, transcriber):
    run_flow(human_client, clock, transcriber, np.random.default_rng(10), collection=as_tester(participant="p01"))
    run_flow(human_client, clock, transcriber, np.random.default_rng(11), collection=as_tester(participant="p02"))
    st = human_client.get("/v1/status").json()
    assert st["collection"]["active"] and st["collection"]["tester_mode"]
    assert human_client.get("/v1/data/policy").json() == {"sharing": True, "tester_mode": True, "retention_days": 365}
    gaze = next(r for r in st["collection"]["stored"] if r["checkpoint"] == "gaze")
    assert gaze["label"] == "human" and gaze["sessions"] == 2 and gaze["participants"] == 2


def test_pack_roundtrip():
    assert collection.unpack(collection.pack_json({"a": [1, 2.5, "x"]})) == {"a": [1, 2.5, "x"]}
    assert collection.unpack(collection.pack_raw(b"\x00\xff")) == b"\x00\xff"
    with pytest.raises(ValueError):
        collection.unpack(b"?abc")
    assert base64.b64encode(b"x")  # keep import used
