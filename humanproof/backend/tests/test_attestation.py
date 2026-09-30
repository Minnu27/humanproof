import base64
import datetime as dt
import hashlib
import time

import cbor2
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from humanproof.security.attestation import (
    AAGUID_DEV, AAGUID_PROD, APPLE_NONCE_OID, AppleAppAttestVerifier, AttestationError,
    PlayIntegrityVerifier,
)

TEAM, BUNDLE = "ABCDE12345", "com.humanproof.app"


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _cert(subject_key, subject, issuer_key, issuer, ca, ext=None):
    now = dt.datetime.now(dt.timezone.utc)
    b = (x509.CertificateBuilder().subject_name(_name(subject)).issuer_name(_name(issuer))
         .public_key(subject_key.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=30))
         .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
    if ext is not None:
        b = b.add_extension(x509.UnrecognizedExtension(APPLE_NONCE_OID, ext), critical=False)
    return b.sign(issuer_key, hashes.SHA256())


class FakeApple:
    def __init__(self):
        self.root_key = ec.generate_private_key(ec.SECP384R1())
        self.root = _cert(self.root_key, "Fake Root", self.root_key, "Fake Root", True)
        self.int_key = ec.generate_private_key(ec.SECP256R1())
        self.inter = _cert(self.int_key, "Fake Int", self.root_key, "Fake Root", True)

    def attest(self, challenge, app_id=f"{TEAM}.{BUNDLE}", aaguid=AAGUID_PROD, counter=0, nonce_override=None):
        leaf_key = ec.generate_private_key(ec.SECP256R1())
        point = leaf_key.public_key().public_bytes(serialization.Encoding.X962,
                                                   serialization.PublicFormat.UncompressedPoint)
        key_id = hashlib.sha256(point).digest()
        auth = (hashlib.sha256(app_id.encode()).digest() + b"\x40" + counter.to_bytes(4, "big")
                + aaguid + len(key_id).to_bytes(2, "big") + key_id + b"\xa0")
        nonce = nonce_override or hashlib.sha256(auth + hashlib.sha256(challenge).digest()).digest()
        ext = b"\x30\x24\xa1\x22\x04\x20" + nonce
        leaf = _cert(leaf_key, "leaf", self.int_key, "Fake Int", False, ext)
        obj = {"fmt": "apple-appattest", "authData": auth, "attStmt": {
            "x5c": [leaf.public_bytes(serialization.Encoding.DER), self.inter.public_bytes(serialization.Encoding.DER)],
            "receipt": b""}}
        return base64.b64encode(cbor2.dumps(obj)).decode(), base64.b64encode(key_id).decode(), leaf_key

    def verifier(self, allow_dev=False):
        return AppleAppAttestVerifier(self.root.public_bytes(serialization.Encoding.PEM), TEAM, BUNDLE, allow_dev)


def assertion(leaf_key, client_data, counter, app_id=f"{TEAM}.{BUNDLE}"):
    auth = hashlib.sha256(app_id.encode()).digest() + b"\x00" + counter.to_bytes(4, "big")
    nonce = hashlib.sha256(auth + hashlib.sha256(client_data).digest()).digest()
    sig = leaf_key.sign(nonce, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(cbor2.dumps({"signature": sig, "authenticatorData": auth})).decode()


def test_apple_attestation_valid():
    fa = FakeApple()
    att, kid, _ = fa.attest(b"challenge-1")
    key = fa.verifier().verify_attestation(att, kid, b"challenge-1")
    assert key.key_id == kid and key.counter == 0


@pytest.mark.parametrize("case", ["wrong_challenge", "wrong_app", "dev_env", "counter", "bad_nonce", "other_root"])
def test_apple_attestation_rejections(case):
    fa = FakeApple()
    if case == "wrong_challenge":
        att, kid, _ = fa.attest(b"challenge-1")
        challenge = b"challenge-2"
    elif case == "wrong_app":
        att, kid, _ = fa.attest(b"c", app_id="EVIL.com.evil")
        challenge = b"c"
    elif case == "dev_env":
        att, kid, _ = fa.attest(b"c", aaguid=AAGUID_DEV)
        challenge = b"c"
    elif case == "counter":
        att, kid, _ = fa.attest(b"c", counter=5)
        challenge = b"c"
    elif case == "bad_nonce":
        att, kid, _ = fa.attest(b"c", nonce_override=b"\x00" * 32)
        challenge = b"c"
    else:
        att, kid, _ = FakeApple().attest(b"c")  # chain to a different root
        challenge = b"c"
    with pytest.raises(AttestationError):
        fa.verifier().verify_attestation(att, kid, challenge)


def test_apple_dev_env_allowed_when_configured():
    fa = FakeApple()
    att, kid, _ = fa.attest(b"c", aaguid=AAGUID_DEV)
    fa.verifier(allow_dev=True).verify_attestation(att, kid, b"c")


def test_apple_assertions_and_replay():
    fa = FakeApple()
    v = fa.verifier()
    att, kid, leaf_key = fa.attest(b"c")
    key = v.verify_attestation(att, kid, b"c")
    body = b'{"frames": []}'
    key.counter = v.verify_assertion(key, assertion(leaf_key, body, 1), body)
    assert key.counter == 1
    with pytest.raises(AttestationError):  # replayed counter
        v.verify_assertion(key, assertion(leaf_key, body, 1), body)
    with pytest.raises(AttestationError):  # signed a different body
        v.verify_assertion(key, assertion(leaf_key, b"other", 2), body)
    other_key = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(AttestationError):  # not the attested key
        v.verify_assertion(key, assertion(other_key, body, 3), body)


# ---- Play Integrity ---------------------------------------------------------------

class FakeResp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload

    def json(self):
        return {"tokenPayloadExternal": self._p}


class FakeHttp:
    def __init__(self, payload):
        self.payload = payload

    def post(self, *a, **k):
        return FakeResp(self.payload)


class FakeCreds:
    valid = True
    token = "t"


def _payload(binding, **over):
    h = base64.urlsafe_b64encode(hashlib.sha256(binding).digest()).decode().rstrip("=")
    p = {
        "requestDetails": {"requestPackageName": "com.humanproof.app", "requestHash": h,
                           "timestampMillis": str(int(time.time() * 1000))},
        "appIntegrity": {"appRecognitionVerdict": "PLAY_RECOGNIZED"},
        "deviceIntegrity": {"deviceRecognitionVerdict": ["MEETS_DEVICE_INTEGRITY"]},
    }
    for k, v in over.items():
        p[k] = v
    return p


def _pi(payload):
    return PlayIntegrityVerifier("com.humanproof.app", None, http=FakeHttp(payload), credentials=FakeCreds())


def test_play_integrity_valid():
    _pi(_payload(b"bind")).verify("tok", b"bind")


@pytest.mark.parametrize("over", [
    {"appIntegrity": {"appRecognitionVerdict": "UNRECOGNIZED_VERSION"}},
    {"deviceIntegrity": {"deviceRecognitionVerdict": ["MEETS_BASIC_INTEGRITY"]}},
    {"deviceIntegrity": {}},
])
def test_play_integrity_bad_verdicts(over):
    with pytest.raises(AttestationError):
        _pi(_payload(b"bind", **over)).verify("tok", b"bind")


def test_play_integrity_binding_and_age():
    with pytest.raises(AttestationError):
        _pi(_payload(b"bind")).verify("tok", b"different")
    old = _payload(b"bind")
    old["requestDetails"]["timestampMillis"] = str(int((time.time() - 3600) * 1000))
    with pytest.raises(AttestationError):
        _pi(old).verify("tok", b"bind")
