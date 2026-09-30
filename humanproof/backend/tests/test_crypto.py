import base64
import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from humanproof.security import tokens
from humanproof.security.crypto import (
    CryptoError, Kek, KeyRing, SigningKey, SigningKeySet, b64u, pairwise_subject,
)


def ring(*kids):
    return KeyRing([Kek(k, AESGCM.generate_key(bit_length=256)) for k in kids])


def test_envelope_roundtrip():
    r = ring("k1")
    blob = r.encrypt(b"voiceprint", b"subject:1")
    assert b"voiceprint" not in blob
    assert r.decrypt(blob, b"subject:1") == b"voiceprint"


def test_ciphertext_bound_to_record():
    r = ring("k1")
    blob = r.encrypt(b"secret", b"subject:1")
    with pytest.raises(CryptoError):
        r.decrypt(blob, b"subject:2")


def test_tampering_detected():
    r = ring("k1")
    env = json.loads(r.encrypt(b"secret", b"a"))
    ct = bytearray(base64.urlsafe_b64decode(env["ct"] + "=="))
    ct[0] ^= 1
    env["ct"] = b64u(bytes(ct))
    with pytest.raises(CryptoError):
        r.decrypt(json.dumps(env).encode(), b"a")


def test_key_rotation():
    old = ring("k1")
    blob = old.encrypt(b"data", b"x")
    rotated = KeyRing([Kek("k2", AESGCM.generate_key(bit_length=256)), old._by_id["k1"]])
    assert rotated.decrypt(blob, b"x") == b"data"
    assert rotated.needs_rewrap(blob)
    assert not rotated.needs_rewrap(rotated.encrypt(b"data", b"x"))


def test_unknown_key_rejected():
    blob = ring("k1").encrypt(b"data", b"x")
    with pytest.raises(CryptoError):
        ring("other").decrypt(blob, b"x")


def keys():
    return SigningKeySet([SigningKey("s1", Ed25519PrivateKey.generate())])


def test_token_roundtrip():
    ks = keys()
    tok = tokens.sign(ks, {"sub": "abc", "aud": "bank.example"}, 60, "https://hp")
    claims = tokens.verify(ks, tok, issuer="https://hp", audience="bank.example")
    assert claims["sub"] == "abc" and claims["exp"] > time.time()


@pytest.mark.parametrize("mutate", ["alg_none", "wrong_typ", "sig", "payload"])
def test_token_forgeries_rejected(mutate):
    ks = keys()
    tok = tokens.sign(ks, {"sub": "abc", "aud": "rp"}, 60, "iss")
    h, p, s = tok.split(".")
    if mutate == "alg_none":
        h = b64u(json.dumps({"alg": "none", "typ": tokens.TYP, "kid": "s1"}).encode())
    elif mutate == "wrong_typ":
        h = b64u(json.dumps({"alg": "EdDSA", "typ": "JWT", "kid": "s1"}).encode())
    elif mutate == "sig":
        s = b64u(b"\x00" * 64)
    else:
        p = b64u(json.dumps({"sub": "evil", "aud": "rp", "iss": "iss", "exp": 9e9, "nbf": 0}).encode())
    with pytest.raises(tokens.TokenError):
        tokens.verify(ks, f"{h}.{p}.{s}", issuer="iss", audience="rp")


def test_token_other_key_rejected():
    tok = tokens.sign(keys(), {"aud": "rp"}, 60, "iss")
    with pytest.raises(tokens.TokenError):
        tokens.verify(keys(), tok, issuer="iss", audience="rp")


def test_token_audience_and_expiry():
    ks = keys()
    tok = tokens.sign(ks, {"aud": "rp"}, 60, "iss")
    with pytest.raises(tokens.TokenError):
        tokens.verify(ks, tok, issuer="iss", audience="other-rp")
    expired = tokens.sign(ks, {"aud": "rp"}, -120, "iss")
    with pytest.raises(tokens.TokenError):
        tokens.verify(ks, expired, issuer="iss", audience="rp")


def test_pairwise_subjects_unlinkable():
    secret = b"s" * 32
    a = pairwise_subject(secret, "human-1", "bank.example")
    b = pairwise_subject(secret, "human-1", "dating.example")
    assert a != b
    assert a == pairwise_subject(secret, "human-1", "bank.example")
