"""Device attestation: proves requests come from the genuine, unmodified app on a
real device (not an emulator, a rooted/jailbroken phone, or a script).

* iOS  - Apple App Attest: attestation once per session, then an *assertion*
         (signature by the attested key over the request body hash) on every
         checkpoint submission.
* Android - Google Play Integrity: a token whose requestHash/nonce binds it to
         the session challenge or the exact submission body.

Web and desktop cannot attest; they get the lower "web" assurance level, which
relying parties can see in the token and treat accordingly.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cbor2
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

APPLE_NONCE_OID = x509.ObjectIdentifier("1.2.840.113635.100.8.2")
AAGUID_PROD = b"appattest" + b"\x00" * 7
AAGUID_DEV = b"appattestdevelop"


class AttestationError(Exception):
    pass


def _b64(data: str) -> bytes:
    data = data.strip()
    try:
        return base64.b64decode(data + "=" * (-len(data) % 4), altchars=b"-_" if ("-" in data or "_" in data) else None)
    except Exception as exc:
        raise AttestationError("bad base64") from exc


# ---------------------------------------------------------------------------
# Apple App Attest
# ---------------------------------------------------------------------------

@dataclass
class AppleAttestedKey:
    key_id: str               # base64
    public_key_pem: bytes
    counter: int


class AppleAppAttestVerifier:
    def __init__(self, root_ca_pem: bytes, team_id: str, bundle_id: str, allow_development: bool = False):
        self.root = x509.load_pem_x509_certificate(root_ca_pem)
        self.app_id = f"{team_id}.{bundle_id}"
        self.allow_development = allow_development

    @classmethod
    def from_file(cls, path: Path, team_id: str, bundle_id: str, allow_development: bool = False):
        return cls(path.read_bytes(), team_id, bundle_id, allow_development)

    def _verify_chain(self, x5c: list[bytes]) -> x509.Certificate:
        if len(x5c) < 2:
            raise AttestationError("incomplete certificate chain")
        leaf = x509.load_der_x509_certificate(x5c[0])
        intermediate = x509.load_der_x509_certificate(x5c[1])
        now = time.time()
        for cert, issuer in ((leaf, intermediate), (intermediate, self.root)):
            if not (cert.not_valid_before_utc.timestamp() <= now <= cert.not_valid_after_utc.timestamp()):
                raise AttestationError("certificate outside validity period")
            if cert.issuer != issuer.subject:
                raise AttestationError("certificate issuer mismatch")
            try:
                cert.verify_directly_issued_by(issuer)
            except (InvalidSignature, ValueError, TypeError) as exc:
                raise AttestationError("certificate signature invalid") from exc
        return leaf

    def verify_attestation(self, attestation_b64: str, key_id_b64: str, challenge: bytes) -> AppleAttestedKey:
        try:
            att = cbor2.loads(_b64(attestation_b64))
            fmt, stmt, auth_data = att["fmt"], att["attStmt"], att["authData"]
        except AttestationError:
            raise
        except Exception as exc:
            raise AttestationError("malformed attestation object") from exc
        if fmt != "apple-appattest":
            raise AttestationError("unexpected attestation format")

        leaf = self._verify_chain(stmt["x5c"])

        client_data_hash = hashlib.sha256(challenge).digest()
        nonce = hashlib.sha256(auth_data + client_data_hash).digest()
        try:
            ext = leaf.extensions.get_extension_for_oid(APPLE_NONCE_OID).value.value
        except x509.ExtensionNotFound as exc:
            raise AttestationError("nonce extension missing") from exc
        # DER: SEQUENCE { [1] { OCTET STRING (32 bytes) } }
        if len(ext) < 32 or ext[-34:-32] != b"\x04\x20" or ext[-32:] != nonce:
            raise AttestationError("nonce mismatch")

        pub = leaf.public_key()
        if not isinstance(pub, ec.EllipticCurvePublicKey):
            raise AttestationError("unexpected key type")
        raw_point = pub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        key_id = _b64(key_id_b64)
        if hashlib.sha256(raw_point).digest() != key_id:
            raise AttestationError("key id does not match certificate")

        if len(auth_data) < 55:
            raise AttestationError("authData too short")
        rp_id_hash, sign_count = auth_data[:32], int.from_bytes(auth_data[33:37], "big")
        aaguid = auth_data[37:53]
        cred_len = int.from_bytes(auth_data[53:55], "big")
        cred_id = auth_data[55:55 + cred_len]
        if rp_id_hash != hashlib.sha256(self.app_id.encode()).digest():
            raise AttestationError("app id mismatch")
        if sign_count != 0:
            raise AttestationError("counter must start at zero")
        if aaguid != AAGUID_PROD and not (self.allow_development and aaguid == AAGUID_DEV):
            raise AttestationError("wrong App Attest environment")
        if cred_id != key_id:
            raise AttestationError("credential id mismatch")

        pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        return AppleAttestedKey(base64.b64encode(key_id).decode(), pem, 0)

    def verify_assertion(self, key: AppleAttestedKey, assertion_b64: str, client_data: bytes) -> int:
        """Verify an assertion over ``client_data``; returns the new counter."""
        try:
            a = cbor2.loads(_b64(assertion_b64))
            signature, auth_data = a["signature"], a["authenticatorData"]
        except AttestationError:
            raise
        except Exception as exc:
            raise AttestationError("malformed assertion") from exc
        nonce = hashlib.sha256(auth_data + hashlib.sha256(client_data).digest()).digest()
        pub = serialization.load_pem_public_key(key.public_key_pem)
        if not isinstance(pub, ec.EllipticCurvePublicKey):
            raise AttestationError("stored key is not an EC key")
        try:
            pub.verify(signature, nonce, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature as exc:
            raise AttestationError("assertion signature invalid") from exc
        if auth_data[:32] != hashlib.sha256(self.app_id.encode()).digest():
            raise AttestationError("app id mismatch")
        counter = int.from_bytes(auth_data[33:37], "big")
        if counter <= key.counter:
            raise AttestationError("assertion counter did not increase (replay?)")
        return counter


# ---------------------------------------------------------------------------
# Google Play Integrity
# ---------------------------------------------------------------------------

class PlayIntegrityVerifier:
    """Decodes tokens via Google's server API and enforces a strict verdict."""

    ENDPOINT = "https://playintegrity.googleapis.com/v1/{package}:decodeIntegrityToken"
    SCOPE = "https://www.googleapis.com/auth/playintegrity"

    def __init__(self, package_name: str, service_account_file: Path, http=None, credentials=None):
        self.package = package_name
        self._sa_file = service_account_file
        self._http = http
        self._creds = credentials

    def _auth_header(self) -> dict:
        if self._creds is None:
            from google.oauth2 import service_account

            self._creds = service_account.Credentials.from_service_account_file(
                str(self._sa_file), scopes=[self.SCOPE]
            )
        if not self._creds.valid:
            from google.auth.transport.requests import Request

            self._creds.refresh(Request())
        return {"Authorization": f"Bearer {self._creds.token}"}

    def decode(self, token: str) -> dict:
        import httpx

        http = self._http or httpx.Client(timeout=10)
        r = http.post(
            self.ENDPOINT.format(package=self.package),
            headers=self._auth_header(),
            json={"integrity_token": token},
        )
        if r.status_code != 200:
            raise AttestationError(f"integrity API error {r.status_code}")
        return r.json()["tokenPayloadExternal"]

    def verify(self, token: str, expected_binding: bytes, max_age_s: int = 120) -> dict:
        payload = self.decode(token)
        req = payload.get("requestDetails", {})
        if req.get("requestPackageName") != self.package:
            raise AttestationError("package mismatch")
        binding = base64.urlsafe_b64encode(hashlib.sha256(expected_binding).digest()).decode().rstrip("=")
        got = (req.get("requestHash") or req.get("nonce") or "").rstrip("=")
        if got != binding:
            raise AttestationError("integrity token not bound to this request")
        ts = int(req.get("timestampMillis", "0")) / 1000
        if abs(time.time() - ts) > max_age_s:
            raise AttestationError("integrity token too old")
        app = payload.get("appIntegrity", {})
        if app.get("appRecognitionVerdict") != "PLAY_RECOGNIZED":
            raise AttestationError("app not recognised by Play")
        dev = payload.get("deviceIntegrity", {}).get("deviceRecognitionVerdict", [])
        if "MEETS_DEVICE_INTEGRITY" not in dev:
            raise AttestationError("device integrity not met")
        return payload


def request_binding(session_id: str, body: bytes) -> bytes:
    """What an attestation/assertion must sign for a submission: session + exact body."""
    return json.dumps({"s": session_id, "b": hashlib.sha256(body).hexdigest()}, separators=(",", ":")).encode()
