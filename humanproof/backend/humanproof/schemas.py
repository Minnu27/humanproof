"""Strict request/response schemas. Unknown fields are rejected and every
number is bounded, so malformed or oversized input fails validation early."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CONSENT_VERSION = "2026-09-v1"

Unit = Annotated[float, Field(ge=-2.0, le=2.0)]
Ms = Annotated[float, Field(ge=0, le=30_000)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=600_000)


class Viewport(Strict):
    w: int = Field(ge=200, le=10_000)
    h: int = Field(ge=200, le=10_000)


class Consent(Strict):
    version: Literal["2026-09-v1"]
    biometric_processing: Literal[True]
    research_opt_in: bool = False


class Attestation(Strict):
    kind: Literal["apple_app_attest", "play_integrity"]
    key_id: str | None = Field(default=None, max_length=200)
    payload: str = Field(max_length=20_000)  # base64 attestation object / integrity token


class CreateSessionRequest(Strict):
    platform: Literal["web", "ios", "android", "desktop"]
    consent: Consent
    # The app asking for proof (e.g. "bank.example"). Tokens are scoped to it and
    # carry a pairwise pseudonym, so different apps cannot link the same person.
    relying_party: str = Field(default="humanproof", pattern=r"^[a-z0-9.\-]{1,120}$")


class CreateSessionResponse(BaseModel):
    session_id: str
    expires_in: int
    steps: list[str]
    assurance: str
    attestation_challenge: str


class StartResponse(BaseModel):
    checkpoint: str
    challenge: dict
    submit_window_ms: tuple[int, int]


class GazeFrame(Strict):
    t: Ms
    ix: Unit           # iris offset within the eye opening, horizontal (both eyes averaged)
    iy: Unit           # vertical
    yaw: float = Field(ge=-90, le=90)
    pitch: float = Field(ge=-90, le=90)
    roll: float = Field(ge=-90, le=90)
    ear: float = Field(ge=0, le=1.5)     # eye aspect ratio (blinks)
    mouth: float = Field(ge=0, le=3)     # inner-lip opening / mouth width
    face: bool


class GazeSubmission(Strict):
    frames: list[GazeFrame] = Field(min_length=10, max_length=1500)
    face_crops: list[Annotated[str, Field(max_length=90_000)]] = Field(default_factory=list, max_length=4)
    viewport: Viewport


class MotorSample(Strict):
    t: Ms
    x: float = Field(ge=-100, le=10_000)
    y: float = Field(ge=-100, le=10_000)
    pressure: float = Field(default=0.5, ge=0, le=1)


class MotionSample(Strict):
    t: Ms
    ax: float = Field(ge=-100, le=100)
    ay: float = Field(ge=-100, le=100)
    az: float = Field(ge=-100, le=100)


class MotorSubmission(Strict):
    pointer_type: Literal["mouse", "touch", "pen", "keyboard"]
    samples: list[MotorSample] = Field(min_length=10, max_length=4000)
    viewport: Viewport
    device_motion: list[MotionSample] = Field(default_factory=list, max_length=3000)


class MouthFrame(Strict):
    t: Ms
    mouth: float = Field(ge=0, le=3)
    face: bool


class VoiceSubmission(Strict):
    audio_wav_b64: str = Field(max_length=450_000)
    audio_offset_ms: float = Field(ge=-2000, le=2000)  # audio t=0 on the mouth-frame timeline
    mouth_frames: list[MouthFrame] = Field(default_factory=list, max_length=1200)


class CheckpointResult(BaseModel):
    checkpoint: str
    passed: bool
    score: float
    next: str | None


class Decision(BaseModel):
    decision: Literal["pass", "step_up", "reject"]
    score: float
    assurance: str
    checkpoints: dict[str, float]
    reasons: list[str]
    attestation_token: str | None = None
    subject_hint: str | None = None
    # Per-checkpoint measurements behind the scores. Never sent in production:
    # it would tell an attacker exactly which signal to improve.
    debug: dict | None = None
