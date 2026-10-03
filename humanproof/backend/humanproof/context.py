"""Application wiring: builds shared services once and exposes them to routes."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .config import Settings
from .models import ModelRegistry
from .scoring.voice import Transcriber, WhisperTranscriber
from .security.attestation import AppleAppAttestVerifier, PlayIntegrityVerifier
from .keys import resolve_keys
from .security.crypto import KeyRing, SigningKeySet, hmac_sha256
from .security.http import RateLimiter
from .storage import Store

log = logging.getLogger("humanproof")

MIN_COLLECTION_KEY_LENGTH = 12
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
    # Whether samples may be stored, and if not, why (shown on /v1/status).
    collection_ready: bool = False
    collection_blocker: str = ""

    @property
    def tester_mode(self) -> bool:
        """Labelled tester sessions need storage and a code that is not guessable,
        in every environment: a demo deployment is on the public internet too."""
        return self.collection_ready and len(self.settings.collection_key) >= MIN_COLLECTION_KEY_LENGTH

    @property
    def issuer(self) -> str:
        return self.settings.public_base_url.rstrip("/")


def build_context(settings: Settings, transcriber: Transcriber | None = None) -> Context:
    settings.validate_for_env()
    keyring, signing, pairwise = resolve_keys(settings)
    if not settings.master_secret and not (settings.kek_keyring and settings.signing_keys and settings.pairwise_secret):
        # Generated keys live on this machine's disk only. On hosts that run
        # several instances (Vercel), each instance would sign with its own key.
        log.warning("Using locally generated dev keys; set HP_MASTER_SECRET (or the individual keys) "
                    "for any deployment with more than one instance")

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

    store = Store(settings.store_target, audit_key=hmac_sha256(pairwise, b"audit-log"))
    ready, blocker = _collection_status(settings, store)
    if settings.data_collection_enabled and not ready:
        log.warning("Data collection is switched on but inactive: %s", blocker)
    if 0 < len(settings.collection_key) < MIN_COLLECTION_KEY_LENGTH:
        log.warning("Tester mode is off: HP_COLLECTION_KEY must be at least %d characters", MIN_COLLECTION_KEY_LENGTH)
    return Context(settings, keyring, signing, pairwise, store, models, transcriber, apple, google, RateLimiter(),
                   collection_ready=ready, collection_blocker=blocker)


def _collection_status(settings: Settings, store: Store) -> tuple[bool, str]:
    if not settings.data_collection_enabled:
        return False, "HP_DATA_COLLECTION_ENABLED is not set"
    if not settings.has_persistent_keys:
        # Samples encrypted with a key generated on one instance's temp disk could
        # never be decrypted again. Refuse rather than store unreadable biometrics.
        return False, "no persistent encryption key (set HP_MASTER_SECRET or HP_KEK_KEYRING)"
    if store.backend != "postgres" and os.environ.get("VERCEL"):
        return False, "no database (on Vercel the local disk is wiped; set HP_DATABASE_URL)"
    return True, ""
