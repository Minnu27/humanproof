"""Compact JWS (EdDSA / Ed25519) for human-verification attestations.

Deliberately minimal: one algorithm, no ``alg`` negotiation, so algorithm
confusion attacks (``none``, HS256-with-public-key) are impossible by design.
"""
from __future__ import annotations

import json
import secrets
import time
from typing import Any

from cryptography.exceptions import InvalidSignature

from .crypto import SigningKeySet, b64u, b64u_decode

ALG = "EdDSA"
TYP = "hp-attestation+jwt"


class TokenError(Exception):
    pass


def _json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def sign(keys: SigningKeySet, claims: dict[str, Any], ttl_seconds: int, issuer: str) -> str:
    now = int(time.time())
    payload = {
        **claims,
        "iss": issuer,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
        "jti": b64u(secrets.token_bytes(16)),
    }
    header = {"alg": ALG, "typ": TYP, "kid": keys.active.kid}
    signing_input = f"{b64u(_json(header))}.{b64u(_json(payload))}".encode()
    sig = keys.active.private.sign(signing_input)
    return f"{signing_input.decode()}.{b64u(sig)}"


def verify(
    keys: SigningKeySet,
    token: str,
    *,
    issuer: str,
    audience: str | None = None,
    leeway: int = 30,
) -> dict[str, Any]:
    try:
        h64, p64, s64 = token.split(".")
        header = json.loads(b64u_decode(h64))
        payload = json.loads(b64u_decode(p64))
        sig = b64u_decode(s64)
    except Exception as exc:
        raise TokenError("Malformed token") from exc

    if header.get("alg") != ALG or header.get("typ") != TYP:
        raise TokenError("Unexpected token type or algorithm")
    key = keys.get(str(header.get("kid", "")))
    if key is None:
        raise TokenError("Unknown key id")
    try:
        key.public.verify(sig, f"{h64}.{p64}".encode())
    except InvalidSignature as exc:
        raise TokenError("Bad signature") from exc

    now = int(time.time())
    if payload.get("iss") != issuer:
        raise TokenError("Wrong issuer")
    if audience is not None and payload.get("aud") != audience:
        raise TokenError("Wrong audience")
    if int(payload.get("exp", 0)) < now - leeway:
        raise TokenError("Expired")
    if int(payload.get("nbf", 0)) > now + leeway:
        raise TokenError("Not yet valid")
    return payload
