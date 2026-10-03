"""Runtime configuration.

Every setting comes from environment variables prefixed ``HP_``. Production
mode refuses to start with insecure defaults (see ``Settings.validate_for_env``).
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    if os.environ.get("VERCEL"):
        return Path(tempfile.gettempdir()) / "humanproof"
    return Path("./var")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HP_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    # --- storage -----------------------------------------------------------
    # On Vercel the deployment filesystem is read-only; only the instance's own
    # temp directory is writable (and it is not shared or persistent).
    data_dir: Path = Field(default_factory=lambda: _default_data_dir())
    db_filename: str = "humanproof.sqlite3"
    # postgres://... When set, all state lives in Postgres (needed for more than
    # one server instance, and for anything that must survive a restart on
    # hosts without a persistent disk). When empty, SQLite under data_dir.
    database_url: str = ""

    # --- keys ----------------------------------------------------------------
    # Key-encryption keys for envelope encryption: "kid1:base64key,kid2:base64key".
    # The first entry is the active key; later ones are kept only to decrypt.
    # One secret (base64, 32+ random bytes) from which the three keys below are
    # derived when they are not set individually. Simplest way to give every
    # server instance the same keys. Separate keys (or a KMS) allow rotation.
    master_secret: str = ""
    kek_keyring: str = ""
    # Ed25519 private keys (PEM files) used to sign attestation tokens, "kid:path".
    signing_keys: str = ""
    # Secret for pairwise pseudonymous subject IDs and audit HMACs (base64, 32 bytes).
    pairwise_secret: str = ""

    # --- routing ----------------------------------------------------------------
    # Path prefix the API is mounted under. Vercel passes the original request
    # path to the service (a request to /api/v1/sessions arrives as
    # /api/v1/sessions), so the Vercel deployment sets HP_API_PREFIX=/api.
    api_prefix: str = Field(default="", pattern=r"^(/[a-z0-9\-]+)*$")

    # --- web / CORS ----------------------------------------------------------
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    public_base_url: str = "http://localhost:8000"
    max_body_bytes: int = 2_500_000  # voice clip + a few face crops
    # Header carrying the real client address, for hosts whose edge sets it and
    # strips any client-supplied value (Vercel: x-real-ip). Leave empty elsewhere:
    # a header the client can set would let anyone dodge the rate limits.
    trusted_ip_header: str = Field(default="", pattern=r"^[a-z0-9\-]*$")

    # --- WebAuthn --------------------------------------------------------------
    rp_id: str = "localhost"
    rp_name: str = "HumanProof"

    # --- device attestation ----------------------------------------------------
    apple_team_id: str = ""
    apple_bundle_id: str = ""
    apple_allow_development: bool = False
    android_package_name: str = ""
    google_service_account_file: Path | None = None

    # --- sessions / challenges ---------------------------------------------------
    session_ttl_seconds: int = 300
    token_ttl_seconds: int = 900

    # --- models --------------------------------------------------------------
    # Default is next to the package, so it resolves the same way whatever the
    # process's working directory is (Docker, uvicorn, a Vercel function).
    models_dir: Path = Path(__file__).resolve().parents[1] / "models"
    # Dev only: allow checkpoints to run on signal heuristics when a trained
    # model is missing. Production refuses to start with this enabled.
    allow_heuristic_fallback: bool = False
    asr_model: str = ""  # faster-whisper model name or path, e.g. "base.en"

    # --- rate limits -----------------------------------------------------------
    rate_limit_per_minute: int = 30
    session_limit_per_hour: int = 12

    # --- data collection for training --------------------------------------------
    # Stores encrypted samples from people who opt in, and from labelled tester
    # sessions. Needs keys that outlive the process (master secret or key ring).
    data_collection_enabled: bool = False
    # Private code that lets trusted testers record sessions labelled "human" or
    # "attack". Only labelled sessions are ever used for training.
    collection_key: str = ""
    data_retention_days: int = Field(default=365, ge=1, le=3650)

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def store_target(self) -> str | Path:
        return self.database_url or self.db_path

    @property
    def has_persistent_keys(self) -> bool:
        return bool(self.master_secret or self.kek_keyring)

    def validate_for_env(self) -> None:
        if self.env != "prod":
            return
        problems = []
        if not self.master_secret:
            if not self.kek_keyring:
                problems.append("HP_KEK_KEYRING (or HP_MASTER_SECRET) is required in prod")
            if not self.signing_keys:
                problems.append("HP_SIGNING_KEYS (or HP_MASTER_SECRET) is required in prod")
            if not self.pairwise_secret:
                problems.append("HP_PAIRWISE_SECRET (or HP_MASTER_SECRET) is required in prod")
        if self.collection_key and len(self.collection_key) < 12:
            problems.append("HP_COLLECTION_KEY must be at least 12 characters")
        if self.allow_heuristic_fallback:
            problems.append("HP_ALLOW_HEURISTIC_FALLBACK must be false in prod")
        if not self.asr_model:
            problems.append("HP_ASR_MODEL is required in prod")
        if not self.public_base_url.startswith("https://"):
            problems.append("HP_PUBLIC_BASE_URL must be https in prod")
        # App shells: Capacitor iOS uses capacitor://localhost, Android https://localhost,
        # Tauri uses tauri://localhost (macOS/Linux) and http://tauri.localhost (Windows).
        app_shell_origins = {"http://tauri.localhost"}
        if any(
            o == "*" or (o.startswith("http://") and o not in app_shell_origins)
            for o in self.allowed_origins
        ):
            problems.append("HP_ALLOWED_ORIGINS must be explicit https/app origins in prod")
        if problems:
            raise RuntimeError("Refusing to start: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    return Settings()
