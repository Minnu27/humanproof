"""Verify a HumanProof attestation token in your own backend (Python >= 3.9).

    from humanproof_verify import Verifier
    v = Verifier("https://api.humanproof.example", audience="bank.example")
    claims = v.verify(token)          # raises on anything invalid
    if claims["hp_assurance"] != "device_attested": require_extra_step()

Keys are fetched from /.well-known/jwks.json and cached; only EdDSA/Ed25519 is
accepted, so algorithm-confusion attacks do not apply. Depends on `cryptography`.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

TYP = "hp-attestation+jwt"


class InvalidToken(Exception):
    pass


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Verifier:
    def __init__(self, issuer: str, audience: str, *, cache_seconds: int = 3600, leeway: int = 30, jwks=None):
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.cache_seconds = cache_seconds
        self.leeway = leeway
        self._keys: dict[str, Ed25519PublicKey] = {}
        self._fetched = 0.0
        if jwks is not None:
            self._load(jwks)
            self._fetched = float("inf")

    def _load(self, jwks: dict) -> None:
        self._keys = {
            k["kid"]: Ed25519PublicKey.from_public_bytes(_b64d(k["x"]))
            for k in jwks.get("keys", [])
            if k.get("kty") == "OKP" and k.get("crv") == "Ed25519"
        }

    def _key(self, kid: str) -> Ed25519PublicKey:
        age = time.time() - self._fetched
        if (kid not in self._keys and age > 60) or age > self.cache_seconds:
            url = f"{self.issuer}/.well-known/jwks.json"
            if not url.startswith("https://") and "localhost" not in url and "127.0.0.1" not in url:
                raise InvalidToken("issuer must use https")
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 (https enforced above)
                self._load(json.load(r))
            self._fetched = time.time()
        if kid not in self._keys:
            raise InvalidToken("unknown key id")
        return self._keys[kid]

    def verify(self, token: str) -> dict:
        try:
            h64, p64, s64 = token.split(".")
            header, payload, sig = json.loads(_b64d(h64)), json.loads(_b64d(p64)), _b64d(s64)
        except Exception as exc:
            raise InvalidToken("malformed") from exc
        if header.get("alg") != "EdDSA" or header.get("typ") != TYP:
            raise InvalidToken("unexpected algorithm or type")
        try:
            self._key(str(header.get("kid"))).verify(sig, f"{h64}.{p64}".encode())
        except InvalidSignature as exc:
            raise InvalidToken("bad signature") from exc
        now = time.time()
        if payload.get("iss") != self.issuer:
            raise InvalidToken("wrong issuer")
        if payload.get("aud") != self.audience:
            raise InvalidToken("wrong audience")
        if payload.get("exp", 0) < now - self.leeway or payload.get("nbf", 0) > now + self.leeway:
            raise InvalidToken("expired or not yet valid")
        if payload.get("hp_human") is not True:
            raise InvalidToken("not a human attestation")
        return payload
