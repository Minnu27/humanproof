"""Cryptographic primitives.

* Envelope encryption (AES-256-GCM): each record gets a fresh data key, which is
  wrapped by the active key-encryption key (KEK). Record identity is bound as
  associated data, so a ciphertext cannot be swapped between records.
* Ed25519 signing keys for attestation tokens, with key IDs for rotation.
* HMAC helpers for pairwise subject IDs and audit chaining.

In production, KEKs should live in a KMS/HSM; ``KeyRing`` is the seam where a
KMS client would be plugged in.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64u_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


class CryptoError(Exception):
    pass


# ---------------------------------------------------------------------------
# Envelope encryption
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kek:
    kid: str
    key: bytes


class KeyRing:
    """Holds KEKs. The first key encrypts; all keys can decrypt (rotation)."""

    VERSION = 1

    def __init__(self, keks: list[Kek]):
        if not keks:
            raise CryptoError("KeyRing needs at least one KEK")
        for k in keks:
            if len(k.key) != 32:
                raise CryptoError(f"KEK {k.kid} must be 32 bytes")
        self._active = keks[0]
        self._by_id = {k.kid: k for k in keks}

    @classmethod
    def from_spec(cls, spec: str) -> "KeyRing":
        keks = []
        for part in filter(None, (p.strip() for p in spec.split(","))):
            kid, _, b64 = part.partition(":")
            if not kid or not b64:
                raise CryptoError("Bad KEK spec; expected kid:base64key")
            keks.append(Kek(kid, base64.b64decode(b64)))
        return cls(keks)

    @property
    def active_kid(self) -> str:
        return self._active.kid

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
        dek = AESGCM.generate_key(bit_length=256)
        n1, n2 = os.urandom(12), os.urandom(12)
        body = AESGCM(dek).encrypt(n1, plaintext, aad)
        wrapped = AESGCM(self._active.key).encrypt(n2, dek, aad + b"|dek")
        envelope = {
            "v": self.VERSION,
            "kid": self._active.kid,
            "wn": b64u(n2),
            "wk": b64u(wrapped),
            "n": b64u(n1),
            "ct": b64u(body),
        }
        return json.dumps(envelope, separators=(",", ":")).encode()

    def decrypt(self, blob: bytes, aad: bytes) -> bytes:
        try:
            env = json.loads(blob)
            if env.get("v") != self.VERSION:
                raise CryptoError("Unsupported envelope version")
            kek = self._by_id.get(env["kid"])
            if kek is None:
                raise CryptoError("Unknown key id")
            dek = AESGCM(kek.key).decrypt(b64u_decode(env["wn"]), b64u_decode(env["wk"]), aad + b"|dek")
            return AESGCM(dek).decrypt(b64u_decode(env["n"]), b64u_decode(env["ct"]), aad)
        except CryptoError:
            raise
        except Exception as exc:  # InvalidTag, KeyError, JSON errors
            raise CryptoError("Decryption failed") from exc

    def needs_rewrap(self, blob: bytes) -> bool:
        try:
            return json.loads(blob).get("kid") != self._active.kid
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Signing keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SigningKey:
    kid: str
    private: Ed25519PrivateKey

    @property
    def public(self) -> Ed25519PublicKey:
        return self.private.public_key()

    def public_jwk(self) -> dict:
        raw = self.public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return {"kty": "OKP", "crv": "Ed25519", "x": b64u(raw), "kid": self.kid, "alg": "EdDSA", "use": "sig"}


class SigningKeySet:
    def __init__(self, keys: list[SigningKey]):
        if not keys:
            raise CryptoError("Need at least one signing key")
        self.active = keys[0]
        self._by_id = {k.kid: k for k in keys}

    def get(self, kid: str) -> SigningKey | None:
        return self._by_id.get(kid)

    def jwks(self) -> dict:
        return {"keys": [k.public_jwk() for k in self._by_id.values()]}

    @classmethod
    def from_spec(cls, spec: str) -> "SigningKeySet":
        """``kid:/path/to/key.pem`` or ``kid:b64:<base64 of the PEM>`` (for hosts
        such as Vercel where secrets are environment variables, not files)."""
        keys = []
        for part in filter(None, (p.strip() for p in spec.split(","))):
            kid, _, source = part.partition(":")
            if source.startswith("b64:"):
                pem = base64.b64decode(source[4:])
            else:
                pem = Path(source).read_bytes()
            priv = serialization.load_pem_private_key(pem, password=None)
            if not isinstance(priv, Ed25519PrivateKey):
                raise CryptoError(f"Signing key {kid} is not Ed25519")
            keys.append(SigningKey(kid, priv))
        return cls(keys)


# ---------------------------------------------------------------------------
# Dev key material (never used in prod: config validation requires real keys)
# ---------------------------------------------------------------------------

def _write_secret(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def load_or_create_dev_keys(data_dir: Path) -> tuple[KeyRing, SigningKeySet, bytes]:
    kdir = data_dir / "dev-keys"
    kek_path, sig_path, pw_path = kdir / "kek.bin", kdir / "signing.pem", kdir / "pairwise.bin"
    if not kek_path.exists():
        _write_secret(kek_path, AESGCM.generate_key(bit_length=256))
    if not sig_path.exists():
        pem = Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        _write_secret(sig_path, pem)
    if not pw_path.exists():
        _write_secret(pw_path, secrets.token_bytes(32))
    ring = KeyRing([Kek("dev-1", kek_path.read_bytes())])
    priv = serialization.load_pem_private_key(sig_path.read_bytes(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise CryptoError("dev signing key is not Ed25519")
    return ring, SigningKeySet([SigningKey("dev-sig-1", priv)]), pw_path.read_bytes()


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------

def hmac_sha256(key: bytes, *parts: bytes) -> bytes:
    mac = hmac.new(key, digestmod=hashlib.sha256)
    for p in parts:
        mac.update(len(p).to_bytes(4, "big"))
        mac.update(p)
    return mac.digest()


def pairwise_subject(secret: bytes, subject: str, relying_party: str) -> str:
    """Per-relying-party pseudonym: apps cannot link the same human across services."""
    return b64u(hmac_sha256(secret, b"pairwise", subject.encode(), relying_party.encode()))


def constant_time_eq(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()
