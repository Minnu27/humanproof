"""HTTP API.

Flow for one verification:
  POST /v1/sessions                                   consent, platform -> session
  POST /v1/sessions/{id}/attest                       (iOS/Android) device attestation
  POST /v1/sessions/{id}/checkpoints/{cp}/start       reveals that checkpoint's challenge
  POST /v1/sessions/{id}/checkpoints/{cp}             submits the response (gaze, motor, voice)
  POST /v1/sessions/{id}/finalize                     fused decision + signed attestation token
  POST /v1/sessions/{id}/passkey/options|verify       bind a device passkey to the verified human
Later:
  POST /v1/passkey/assert/options|verify              quick re-verification or data deletion
  GET  /.well-known/jwks.json                         public keys for relying parties
"""
from __future__ import annotations

import base64
import json
import logging
import secrets
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import challenges as chmod
from .context import Context
from .schemas import (
    CONSENT_VERSION, Attestation, CheckpointResult, CreateSessionRequest, CreateSessionResponse,
    Decision, GazeSubmission, MotorSubmission, StartResponse, VoiceSubmission,
)
from .scoring import face as face_scoring
from .scoring import fusion, gaze, motor, voice
from .security import tokens
from .security.attestation import AttestationError, AppleAttestedKey, request_binding
from .security.crypto import b64u, b64u_decode, pairwise_subject
from .security.http import client_ip
from .storage import client_fingerprint

log = logging.getLogger("humanproof.api")
router = APIRouter()

STEPS = ("gaze", "motor", "voice")
Checkpoint = Literal["gaze", "motor", "voice"]
SUBMIT_WINDOWS = {  # (min ms after start, max ms after start)
    "motor": (800, 90_000),
    "voice": (1_200, 60_000),
}


def ctx(request: Request) -> Context:
    return request.app.state.ctx


def _rate_limit(request: Request, c: Context, bucket: str, limit: int, window: float = 60) -> None:
    key = f"{bucket}:{client_fingerprint(c.pairwise_secret, client_ip(request))}"
    if not c.limiter.allow(key, limit, window):
        raise HTTPException(429, "Too many requests")


def _load(c: Context, sid: str) -> dict:
    if len(sid) > 64:
        raise HTTPException(404, "Session not found")
    sess = c.store.get_session(sid)
    if sess is None or sess["expires_at"] < time.time():
        raise HTTPException(404, "Session not found or expired")
    return sess


def _save(c: Context, sid: str, state: dict) -> None:
    if not c.store.update_session_state(sid, state["version"], state):
        raise HTTPException(409, "Session was modified concurrently")


def _parse(model: type[BaseModel], body: bytes):
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        # Report field locations only; never echo submitted values back.
        raise HTTPException(422, {"invalid_fields": [".".join(map(str, e["loc"])) for e in exc.errors()][:10]})


def _check_device_binding(c: Context, request: Request, sid: str, state: dict, body: bytes) -> None:
    """For attested sessions, every submission must be signed by the attested app."""
    if state["assurance"] != "device_attested":
        return
    binding = request_binding(sid, body)
    try:
        if state["platform"] == "ios":
            header = request.headers.get("x-hp-assertion", "")
            k = state["apple_key"]
            key = AppleAttestedKey(k["key_id"], k["pem"].encode(), k["counter"])
            state["apple_key"]["counter"] = c.apple.verify_assertion(key, header, binding)  # type: ignore[union-attr]
        elif state["platform"] == "android":
            c.google.verify(request.headers.get("x-hp-integrity", ""), binding)  # type: ignore[union-attr]
    except (AttestationError, AttributeError) as exc:
        c.store.audit("device_binding_failed", {"session": sid})
        raise HTTPException(401, "Device attestation failed") from exc


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@router.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(body: CreateSessionRequest, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "create", c.settings.rate_limit_per_minute)
    fp = client_fingerprint(c.pairwise_secret, client_ip(request))
    if c.store.count_recent_sessions(fp, 3600) >= c.settings.session_limit_per_hour:
        raise HTTPException(429, "Too many verification attempts; try again later")
    sid = b64u(secrets.token_bytes(32))
    state = {
        "version": 0,
        "platform": body.platform,
        "assurance": "web",
        "relying_party": body.relying_party,
        "consent_version": CONSENT_VERSION,
        "research_opt_in": body.consent.research_opt_in,
        "attest_challenge": b64u(secrets.token_bytes(32)),
        "challenges": chmod.make_challenge_set(),
        "steps": {},
        "finalized": False,
    }
    c.store.create_session(sid, fp, body.platform, state, c.settings.session_ttl_seconds)
    c.store.audit("session_created", {"session": sid[:12], "platform": body.platform, "rp": body.relying_party})
    return CreateSessionResponse(
        session_id=sid, expires_in=c.settings.session_ttl_seconds, steps=list(STEPS),
        assurance="web", attestation_challenge=state["attest_challenge"],
    )


@router.post("/v1/sessions/{sid}/attest")
def attest(sid: str, body: Attestation, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "attest", 10)
    sess = _load(c, sid)
    state = sess["state"]
    if state["steps"] or state["assurance"] != "web":
        raise HTTPException(409, "Attestation must happen once, before any checkpoint")
    challenge = b64u_decode(state["attest_challenge"])
    try:
        if body.kind == "apple_app_attest":
            if state["platform"] != "ios" or c.apple is None or not body.key_id:
                raise HTTPException(400, "App Attest not available for this session")
            key = c.apple.verify_attestation(body.payload, body.key_id, challenge)
            state["apple_key"] = {"key_id": key.key_id, "pem": key.public_key_pem.decode(), "counter": 0}
        else:
            if state["platform"] != "android" or c.google is None:
                raise HTTPException(400, "Play Integrity not available for this session")
            c.google.verify(body.payload, challenge)
    except AttestationError as exc:
        c.store.audit("attestation_failed", {"session": sid[:12], "kind": body.kind})
        raise HTTPException(401, "Device attestation failed") from exc
    state["assurance"] = "device_attested"
    _save(c, sid, state)
    c.store.audit("attested", {"session": sid[:12], "kind": body.kind})
    return {"assurance": "device_attested"}


@router.post("/v1/sessions/{sid}/checkpoints/{cp}/start", response_model=StartResponse)
def start_checkpoint(sid: str, cp: Checkpoint, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "start", c.settings.rate_limit_per_minute)
    sess = _load(c, sid)
    state = sess["state"]
    expected = next((s for s in STEPS if not state["steps"].get(s, {}).get("submitted")), None)
    if state["finalized"] or cp != expected:
        raise HTTPException(409, f"Next checkpoint is {expected}")
    if state["steps"].get(cp, {}).get("started_at"):
        raise HTTPException(409, "Checkpoint already started")
    state["steps"][cp] = {"started_at": time.time(), "submitted": False}
    _save(c, sid, state)

    ch = state["challenges"][cp]
    if cp == "gaze":
        g = chmod.gaze_from_dict(ch)
        window = (int(g.duration_ms * 0.9), g.duration_ms + 30_000)
        public = {"duration_ms": g.duration_ms, "keyframes": ch["keyframes"], "capture_ms": g.capture_ms}
    elif cp == "motor":
        m = chmod.motor_from_dict(ch)
        window = SUBMIT_WINDOWS["motor"]
        public = {"duration_ms": m.duration_ms, "polyline": [list(p) for p in m.polyline(24)]}
    else:
        window = SUBMIT_WINDOWS["voice"]
        public = {"words": ch["words"], "max_duration_ms": ch["max_duration_ms"]}
    return StartResponse(checkpoint=cp, challenge=public, submit_window_ms=window)


def _begin_submit(c: Context, sid: str, cp: str) -> tuple[dict, dict]:
    sess = _load(c, sid)
    state = sess["state"]
    step = state["steps"].get(cp)
    if not step or step.get("submitted"):
        raise HTTPException(409, "Checkpoint not started or already submitted")
    elapsed_ms = (time.time() - step["started_at"]) * 1000
    if cp == "gaze":
        dur = state["challenges"]["gaze"]["duration_ms"]
        lo, hi = int(dur * 0.9), dur + 30_000
    else:
        lo, hi = SUBMIT_WINDOWS[cp]
    if not (lo <= elapsed_ms <= hi):
        step["submitted"] = True
        step["score"] = 0.0
        step["reasons"] = ["submitted outside the allowed time window"]
        _save(c, sid, state)
        raise HTTPException(400, "Submitted outside the allowed time window")
    return state, step


def _finish_submit(c: Context, sid: str, cp: str, state: dict, s: float, reasons: list[str], feats: dict):
    step = state["steps"][cp]
    step.update(submitted=True, score=round(float(s), 4), reasons=reasons, features=_jsonable(feats))
    _save(c, sid, state)
    nxt = next((x for x in STEPS if not state["steps"].get(x, {}).get("submitted")), None)
    c.store.audit("checkpoint", {"session": sid[:12], "cp": cp, "score": round(float(s), 3)})
    return CheckpointResult(checkpoint=cp, passed=s >= 0.5, score=round(float(s), 3), next=nxt)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


@router.post("/v1/sessions/{sid}/checkpoints/gaze", response_model=CheckpointResult)
async def submit_gaze(sid: str, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "submit", c.settings.rate_limit_per_minute)
    body = await request.body()
    sub = _parse(GazeSubmission, body)
    state, _ = _begin_submit(c, sid, "gaze")
    _check_device_binding(c, request, sid, state, body)
    ch = chmod.gaze_from_dict(state["challenges"]["gaze"])
    s, reasons, feats = gaze.score(ch, sub.frames, c.models.get("gaze"))
    fs, freasons, finfo = face_scoring.score(
        sub.face_crops, c.models.get("face_deepfake"), c.settings.allow_heuristic_fallback
    )
    state["face"] = {"score": round(float(fs), 4), "reasons": freasons, "info": _jsonable(finfo)}
    return _finish_submit(c, sid, "gaze", state, s, reasons, feats)


@router.post("/v1/sessions/{sid}/checkpoints/motor", response_model=CheckpointResult)
async def submit_motor(sid: str, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "submit", c.settings.rate_limit_per_minute)
    body = await request.body()
    sub = _parse(MotorSubmission, body)
    state, _ = _begin_submit(c, sid, "motor")
    _check_device_binding(c, request, sid, state, body)
    ch = chmod.motor_from_dict(state["challenges"]["motor"])
    s, reasons, feats = motor.score(ch, sub, c.models.get("motor"))
    return _finish_submit(c, sid, "motor", state, s, reasons, feats)


@router.post("/v1/sessions/{sid}/checkpoints/voice", response_model=CheckpointResult)
async def submit_voice(sid: str, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "submit", c.settings.rate_limit_per_minute)
    body = await request.body()
    sub = _parse(VoiceSubmission, body)
    state, _ = _begin_submit(c, sid, "voice")
    _check_device_binding(c, request, sid, state, body)
    ch = chmod.voice_from_dict(state["challenges"]["voice"])
    s, reasons, feats, embedding = voice.score(
        ch, sub,
        antispoof_model=c.models.get("voice_antispoof"),
        speaker_model=c.models.get("voice_speaker"),
        transcriber=c.transcriber,
        allow_fallback=c.settings.allow_heuristic_fallback,
    )
    if embedding is not None:
        # Held only until finalize, encrypted; dropped if the session does not pass.
        state["pending_voiceprint"] = c.keyring.encrypt(embedding.tobytes(), f"vp:{sid}".encode()).decode()
    return _finish_submit(c, sid, "voice", state, s, reasons, feats)


@router.post("/v1/sessions/{sid}/finalize", response_model=Decision)
def finalize(sid: str, request: Request, c: Context = Depends(ctx)):
    _rate_limit(request, c, "finalize", c.settings.rate_limit_per_minute)
    sess = _load(c, sid)
    state = sess["state"]
    if state["finalized"]:
        raise HTTPException(409, "Session already finalized")
    if not all(state["steps"].get(s, {}).get("submitted") for s in STEPS):
        raise HTTPException(409, "All checkpoints must be completed first")

    scores = {s: float(state["steps"][s]["score"]) for s in STEPS}
    scores["face"] = float(state.get("face", {}).get("score", 0.0))
    result = fusion.fuse(scores, state["assurance"])
    reasons = result.reasons + [r for s in STEPS for r in state["steps"][s].get("reasons", [])]
    reasons += state.get("face", {}).get("reasons", [])

    token, hint = None, None
    if result.decision == "pass":
        subject = b64u(secrets.token_bytes(24))
        vp_blob = None
        if state.get("pending_voiceprint"):
            raw = c.keyring.decrypt(state["pending_voiceprint"].encode(), f"vp:{sid}".encode())
            vp_blob = c.keyring.encrypt(raw, f"voiceprint:{subject}".encode())
        c.store.create_subject(subject, state["assurance"], vp_blob)
        state["subject_id"] = subject
        token = _issue_token(c, subject, state["relying_party"], state["assurance"], scores, "full")
        hint = pairwise_subject(c.pairwise_secret, subject, state["relying_party"])
        if state.get("research_opt_in") and c.settings.data_collection_enabled:
            _store_research(c, sid, state)
    state.pop("pending_voiceprint", None)
    state["finalized"] = True
    state["decision"] = result.decision
    _save(c, sid, state)
    c.store.audit("finalized", {"session": sid[:12], "decision": result.decision, "score": round(result.score, 3)})
    return Decision(
        decision=result.decision, score=round(result.score, 3), assurance=state["assurance"],
        checkpoints={k: round(v, 3) for k, v in scores.items()}, reasons=sorted(set(reasons)),
        attestation_token=token, subject_hint=hint, debug=_debug(c, state),
    )


def _debug(c: Context, state: dict) -> dict | None:
    if c.settings.env == "prod":
        return None
    out = {
        cp: {k: state["steps"][cp].get(k) for k in ("score", "reasons", "features")}
        for cp in STEPS
    }
    out["face"] = state.get("face", {})
    return out


def _issue_token(c: Context, subject: str, rp: str, assurance: str, scores: dict, method: str) -> str:
    claims = {
        "sub": pairwise_subject(c.pairwise_secret, subject, rp),
        "aud": rp,
        "hp_human": True,
        "hp_assurance": assurance,
        "hp_method": method,  # "full" = three checkpoints, "passkey" = re-verification
        "hp_scores": {k: round(v, 2) for k, v in scores.items()},
    }
    return tokens.sign(c.signing, claims, c.settings.token_ttl_seconds, c.issuer)


def _store_research(c: Context, sid: str, state: dict) -> None:
    for cp in STEPS:
        feats = state["steps"][cp].get("features") or {}
        payload = json.dumps({"label": "passed_human", "features": feats}).encode()
        rid = b64u(secrets.token_bytes(16))
        c.store.add_research_sample(
            rid, cp, state["consent_version"], c.keyring.encrypt(payload, f"research:{rid}".encode())
        )


# ---------------------------------------------------------------------------
# Passkeys (WebAuthn): bind the verified human to a device key
# ---------------------------------------------------------------------------

class WebAuthnPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credential: dict


class AssertOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: Literal["reverify", "delete"]
    relying_party: str = Field(default="humanproof", pattern=r"^[a-z0-9.\-]{1,120}$")


def _origins(c: Context) -> list[str]:
    return list(c.settings.allowed_origins)


@router.post("/v1/sessions/{sid}/passkey/options")
def passkey_options(sid: str, request: Request, c: Context = Depends(ctx)):
    from webauthn import generate_registration_options, options_to_json
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria, ResidentKeyRequirement, UserVerificationRequirement,
    )

    _rate_limit(request, c, "passkey", 10)
    state = _load(c, sid)["state"]
    if state.get("decision") != "pass" or state.get("passkey_bound"):
        raise HTTPException(409, "Passkeys can be bound only after a passing verification")
    opts = generate_registration_options(
        rp_id=c.settings.rp_id, rp_name=c.settings.rp_name,
        user_id=b64u_decode(state["subject_id"]), user_name="Verified human",
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED,
            user_verification=UserVerificationRequirement.REQUIRED,
        ),
    )
    state["webauthn_challenge"] = b64u(opts.challenge)
    _save(c, sid, state)
    return json.loads(options_to_json(opts))


@router.post("/v1/sessions/{sid}/passkey/verify")
def passkey_verify(sid: str, body: WebAuthnPayload, request: Request, c: Context = Depends(ctx)):
    from webauthn import verify_registration_response

    _rate_limit(request, c, "passkey", 10)
    state = _load(c, sid)["state"]
    challenge = state.pop("webauthn_challenge", None)
    if state.get("decision") != "pass" or not challenge:
        raise HTTPException(409, "No passkey registration in progress")
    try:
        v = verify_registration_response(
            credential=body.credential, expected_challenge=b64u_decode(challenge),
            expected_rp_id=c.settings.rp_id, expected_origin=_origins(c), require_user_verification=True,
        )
    except Exception as exc:
        _save(c, sid, state)  # challenge is single-use even on failure
        raise HTTPException(400, "Passkey registration failed") from exc
    c.store.add_credential(b64u(v.credential_id), state["subject_id"], v.credential_public_key, v.sign_count)
    state["passkey_bound"] = True
    _save(c, sid, state)
    c.store.audit("passkey_bound", {"session": sid[:12]})
    return {"status": "bound"}


# Pending passkey assertions, keyed by challenge. Short-lived and single-use.
_ASSERTIONS: dict[str, dict] = {}


@router.post("/v1/passkey/assert/options")
def assert_options(body: AssertOptionsRequest, request: Request, c: Context = Depends(ctx)):
    from webauthn import generate_authentication_options, options_to_json
    from webauthn.helpers.structs import UserVerificationRequirement

    _rate_limit(request, c, "assert", 20)
    opts = generate_authentication_options(
        rp_id=c.settings.rp_id, user_verification=UserVerificationRequirement.REQUIRED
    )
    now = time.time()
    for k in [k for k, v in _ASSERTIONS.items() if v["exp"] < now]:
        del _ASSERTIONS[k]
    if len(_ASSERTIONS) > 10_000:
        raise HTTPException(503, "Busy")
    _ASSERTIONS[b64u(opts.challenge)] = {"exp": now + 120, "purpose": body.purpose, "rp": body.relying_party}
    return json.loads(options_to_json(opts))


@router.post("/v1/passkey/assert/verify")
def assert_verify(body: WebAuthnPayload, request: Request, c: Context = Depends(ctx)):
    from webauthn import verify_authentication_response

    _rate_limit(request, c, "assert", 20)
    try:
        client_data = json.loads(b64u_decode(body.credential["response"]["clientDataJSON"]))
        pending = _ASSERTIONS.pop(client_data["challenge"])
        cred = c.store.get_credential(body.credential["id"])
    except Exception as exc:
        raise HTTPException(400, "Unknown or expired passkey challenge") from exc
    if pending["exp"] < time.time() or cred is None:
        raise HTTPException(400, "Unknown or expired passkey challenge")
    try:
        v = verify_authentication_response(
            credential=body.credential, expected_challenge=b64u_decode(client_data["challenge"]),
            expected_rp_id=c.settings.rp_id, expected_origin=_origins(c),
            credential_public_key=cred["public_key"], credential_current_sign_count=cred["sign_count"],
            require_user_verification=True,
        )
    except Exception as exc:
        raise HTTPException(401, "Passkey verification failed") from exc
    c.store.update_sign_count(cred["credential_id"], v.new_sign_count)
    subject = c.store.get_subject(cred["subject_id"])
    if subject is None:
        raise HTTPException(401, "Passkey verification failed")
    if pending["purpose"] == "delete":
        c.store.delete_subject(subject["id"])
        c.store.audit("subject_deleted", {"by": "passkey"})
        return {"status": "deleted"}
    token = _issue_token(c, subject["id"], pending["rp"], subject["assurance"], {}, "passkey")
    c.store.audit("reverified", {"rp": pending["rp"]})
    return {"attestation_token": token}


# ---------------------------------------------------------------------------
# Relying-party endpoints and service info
# ---------------------------------------------------------------------------

@router.get("/.well-known/jwks.json")
def jwks(c: Context = Depends(ctx)):
    return c.signing.jwks()


class VerifyTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(max_length=4000)
    audience: str = Field(pattern=r"^[a-z0-9.\-]{1,120}$")


@router.post("/v1/tokens/verify")
def verify_token(body: VerifyTokenRequest, request: Request, c: Context = Depends(ctx)):
    """Convenience for relying parties; verifying locally against JWKS is preferred."""
    _rate_limit(request, c, "tokverify", 120)
    try:
        claims = tokens.verify(c.signing, body.token, issuer=c.issuer, audience=body.audience)
    except tokens.TokenError:
        return {"valid": False}
    return {"valid": True, "claims": claims}


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/v1/status")
def status(c: Context = Depends(ctx)):
    if c.settings.env == "prod":
        raise HTTPException(404)
    return {
        "env": c.settings.env,
        "models": c.models.status(),
        "asr": c.transcriber is not None,
        "heuristic_fallback": c.settings.allow_heuristic_fallback,
        "audit_chain_ok": c.store.verify_audit_chain(),
    }
