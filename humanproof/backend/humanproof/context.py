"""Application wiring: builds shared services once and exposes them to routes."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from .config import Settings
from .models import ModelRegistry
from .scoring.voice import Transcriber, WhisperTranscriber
from .security.attestation import AppleAppAttestVerifier, PlayIntegrityVerifier
from .security.crypto import KeyRing, SigningKeySet, hmac_sha256, load_or_create_dev_keys
from .security.http import RateLimiter
from .storage import Store

log = logging.getLogger("humanproof")

PROD_REQUIRED_MODELS = ("motor", "voice_antispoof", "voice_speaker", "face_deepfake")


@dataclass
class Context:
    settings: Settings
    keyring: KeyRing
    signing: SigningKeySet
    pairwise_secret: bytes
    store: Store
    models: ModelRegistry
    transcriber: Transcriber | None
    apple: AppleAppAttestVerifier | None
    google: PlayIntegrityVerifier | None
    limiter: RateLimiter

    @property
    def issuer(self) -> str:
        return self.settings.public_base_url.rstrip("/")


def build_context(settings: Settings, transcriber: Transcriber | None = None) -> Context:
    settings.validate_for_env()
    if settings.env == "prod":
        keyring = KeyRing.from_spec(settings.kek_keyring)
        signing = SigningKeySet.from_spec(settings.signing_keys)
        pairwise = base64.b64decode(settings.pairwise_secret)
        if len(pairwise) < 32:
            raise RuntimeError("HP_PAIRWISE_SECRET must be at least 32 bytes")
    else:
        keyring, signing, pairwise = load_or_create_dev_keys(settings.data_dir)
        if settings.kek_keyring:
            keyring = KeyRing.from_spec(settings.kek_keyring)
        if settings.signing_keys:
            signing = SigningKeySet.from_spec(settings.signing_keys)

    models = ModelRegistry(settings.models_dir)
    if settings.env == "prod":
        missing = [m for m in PROD_REQUIRED_MODELS if models.get(m) is None]
        if missing:
            raise RuntimeError(f"Refusing to start: models not loaded: {missing} ({models.status()})")

    if transcriber is None and settings.asr_model:
        transcriber = WhisperTranscriber(settings.asr_model)

    apple = None
    root = settings.data_dir / "Apple_App_Attestation_Root_CA.pem"
    if settings.apple_team_id and settings.apple_bundle_id and root.exists():
        apple = AppleAppAttestVerifier.from_file(
            root, settings.apple_team_id, settings.apple_bundle_id, settings.apple_allow_development
        )
    google = None
    if settings.android_package_name and settings.google_service_account_file:
        google = PlayIntegrityVerifier(settings.android_package_name, settings.google_service_account_file)

    store = Store(settings.db_path, audit_key=hmac_sha256(pairwise, b"audit-log"))
    return Context(settings, keyring, signing, pairwise, store, models, transcriber, apple, google, RateLimiter())
