"""End-to-end tests for the attested (iOS) flow and passkey lifecycle."""
import base64
import hashlib
import json
import struct

import cbor2
import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

import sim
from humanproof import challenges as C
from humanproof.security import tokens
from humanproof.security.attestation import request_binding

from test_api import CONSENT, HumanMotorModel, run_flow, start
from test_attestation import TEAM, FakeApple, assertion

ORIGIN = "http://localhost:5173"


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def b64u_dec(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Authenticator:
    """Minimal platform authenticator (ES256, user verification performed)."""

    def __init__(self):
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.cred_id = b"\x01" * 16 + bytes(range(16))
        self.count = 0

    def _cose(self) -> bytes:
        nums = self.key.public_key().public_numbers()
        return cbor2.dumps({1: 2, 3: -7, -1: 1, -2: nums.x.to_bytes(32, "big"), -3: nums.y.to_bytes(32, "big")})

    def create(self, opts: dict) -> dict:
        cd = json.dumps({"type": "webauthn.create", "challenge": opts["challenge"], "origin": ORIGIN}).encode()
        auth = (hashlib.sha256(opts["rp"]["id"].encode()).digest() + b"\x45" + struct.pack(">I", 0)
                + b"\x00" * 16 + struct.pack(">H", len(self.cred_id)) + self.cred_id + self._cose())
        att = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth})
        return {"id": b64u(self.cred_id), "rawId": b64u(self.cred_id), "type": "public-key",
                "response": {"clientDataJSON": b64u(cd), "attestationObject": b64u(att)}}

    def get(self, opts: dict) -> dict:
        self.count += 1
        cd = json.dumps({"type": "webauthn.get", "challenge": opts["challenge"], "origin": ORIGIN}).encode()
        auth = hashlib.sha256(opts["rpId"].encode()).digest() + b"\x05" + struct.pack(">I", self.count)
        sig = self.key.sign(auth + hashlib.sha256(cd).digest(), ec.ECDSA(hashes.SHA256()))
        return {"id": b64u(self.cred_id), "rawId": b64u(self.cred_id), "type": "public-key",
                "response": {"clientDataJSON": b64u(cd), "authenticatorData": b64u(auth),
                             "signature": b64u(sig)}}


def test_passkey_bind_reverify_and_delete(client, clock, transcriber):
    client.app_ctx.models._models["motor"] = HumanMotorModel()
    sid, d = run_flow(client, clock, transcriber, np.random.default_rng(21))
    assert d["decision"] == "pass"

    dev = Authenticator()
    opts = client.post(f"/v1/sessions/{sid}/passkey/options").json()
    r = client.post(f"/v1/sessions/{sid}/passkey/verify", json={"credential": dev.create(opts)})
    assert r.status_code == 200, r.text
    # a second binding for the same verification is refused
    assert client.post(f"/v1/sessions/{sid}/passkey/options").status_code == 409

    # later: one-tap re-verification for a relying party
    opts = client.post("/v1/passkey/assert/options", json={"purpose": "reverify", "relying_party": "bank.example"}).json()
    r = client.post("/v1/passkey/assert/verify", json={"credential": dev.get(opts)})
    assert r.status_code == 200, r.text
    claims = tokens.verify(client.app_ctx.signing, r.json()["attestation_token"], issuer="http://testserver",
                           audience="bank.example")
    assert claims["hp_method"] == "passkey" and claims["sub"] == d["subject_hint"]

    # replaying the same assertion fails (challenge single-use)
    stale = dev.get(opts)
    assert client.post("/v1/passkey/assert/verify", json={"credential": stale}).status_code == 400

    # the user deletes their data, including anything stored for training
    assert len(list(client.app_ctx.store.iter_samples())) == 4
    opts = client.post("/v1/passkey/assert/options", json={"purpose": "delete"}).json()
    r = client.post("/v1/passkey/assert/verify", json={"credential": dev.get(opts)})
    assert r.json() == {"status": "deleted"}
    assert list(client.app_ctx.store.iter_samples()) == []
    state = client.app_ctx.store.get_session(sid)["state"]
    assert client.app_ctx.store.get_subject(state["subject_id"]) is None
    opts = client.post("/v1/passkey/assert/options", json={"purpose": "reverify"}).json()
    assert client.post("/v1/passkey/assert/verify", json={"credential": dev.get(opts)}).status_code == 400


def test_passkey_wrong_origin_rejected(client, clock, transcriber):
    client.app_ctx.models._models["motor"] = HumanMotorModel()
    sid, _ = run_flow(client, clock, transcriber, np.random.default_rng(22))
    dev = Authenticator()
    opts = client.post(f"/v1/sessions/{sid}/passkey/options").json()
    cred = dev.create(opts)
    cd = json.loads(b64u_dec(cred["response"]["clientDataJSON"]))
    cd["origin"] = "https://phishing.example"
    cred["response"]["clientDataJSON"] = b64u(json.dumps(cd).encode())
    assert client.post(f"/v1/sessions/{sid}/passkey/verify", json={"credential": cred}).status_code == 400


# ---- attested iOS session --------------------------------------------------------

@pytest.fixture
def apple(client, settings):
    fa = FakeApple()
    client.app_ctx.apple = fa.verifier()
    return fa


def test_ios_attested_flow_requires_signed_submissions(client, clock, transcriber, apple):
    r = client.post("/v1/sessions", json={"platform": "ios", "consent": CONSENT})
    sess = r.json()
    sid = sess["session_id"]
    att, kid, leaf_key = apple.attest(b64u_dec(sess["attestation_challenge"]))
    r = client.post(f"/v1/sessions/{sid}/attest", json={"kind": "apple_app_attest", "key_id": kid, "payload": att})
    assert r.status_code == 200 and r.json()["assurance"] == "device_attested"

    pub = start(client, sid, "gaze")
    ch = C.gaze_from_dict(pub)
    clock.advance(ch.duration_ms / 1000 + 0.5)
    body = json.dumps({"frames": sim.human_gaze_frames(ch, np.random.default_rng(5)),
                       "viewport": {"w": 1280, "h": 800}}).encode()
    url = f"/v1/sessions/{sid}/checkpoints/gaze"
    hdr = {"content-type": "application/json"}

    # unsigned -> rejected
    assert client.post(url, content=body, headers=hdr).status_code == 401
    # signed by a different key -> rejected
    other = ec.generate_private_key(ec.SECP256R1())
    bad = {**hdr, "x-hp-assertion": assertion(other, request_binding(sid, body), 1)}
    assert client.post(url, content=body, headers=bad).status_code == 401
    # signed over a different body -> rejected
    wrong = {**hdr, "x-hp-assertion": assertion(leaf_key, request_binding(sid, b"{}"), 1)}
    assert client.post(url, content=body, headers=wrong).status_code == 401
    # correctly signed -> accepted
    good = {**hdr, "x-hp-assertion": assertion(leaf_key, request_binding(sid, body), 1)}
    r = client.post(url, content=body, headers=good)
    assert r.status_code == 200, r.text


def test_attestation_cannot_be_reused_for_another_session(client, apple):
    s1 = client.post("/v1/sessions", json={"platform": "ios", "consent": CONSENT}).json()
    s2 = client.post("/v1/sessions", json={"platform": "ios", "consent": CONSENT}).json()
    att, kid, _ = apple.attest(b64u_dec(s1["attestation_challenge"]))
    r = client.post(f"/v1/sessions/{s2['session_id']}/attest",
                    json={"kind": "apple_app_attest", "key_id": kid, "payload": att})
    assert r.status_code == 401


def test_attestation_from_cloned_app_rejected(client, apple):
    s = client.post("/v1/sessions", json={"platform": "ios", "consent": CONSENT}).json()
    att, kid, _ = apple.attest(b64u_dec(s["attestation_challenge"]), app_id=f"{TEAM}.com.evil.clone")
    r = client.post(f"/v1/sessions/{s['session_id']}/attest",
                    json={"kind": "apple_app_attest", "key_id": kid, "payload": att})
    assert r.status_code == 401
