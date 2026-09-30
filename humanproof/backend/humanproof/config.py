"""Runtime configuration.

Every setting comes from environment variables prefixed ``HP_``. Production
mode refuses to start with insecure defaults (see ``Settings.validate_for_env``).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HP_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("./var")
    db_filename: str = "humanproof.sqlite3"

    # --- keys ----------------------------------------------------------------
    # Key-encryption keys for envelope encryption: "kid1:base64key,kid2:base64key".
    # The first entry is the active key; later ones are kept only to decrypt.
    kek_keyring: str = ""
    # Ed25519 private keys (PEM files) used to sign attestation tokens, "kid:path".
    signing_keys: str = ""
    # Secret for pairwise pseudonymous subject IDs and audit HMACs (base64, 32 bytes).
    pairwise_secret: str = ""

    # --- web / CORS ----------------------------------------------------------
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    public_base_url: str = "http://localhost:8000"
    max_body_bytes: int = 2_500_000  # voice clip + a few face crops

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
    models_dir: Path = Path("./models")
    # Dev only: allow checkpoints to run on signal heuristics when a trained
    # model is missing. Production refuses to start with this enabled.
    allow_heuristic_fallback: bool = False
    asr_model: str = ""  # faster-whisper model name or path, e.g. "base.en"

    # --- rate limits -----------------------------------------------------------
    rate_limit_per_minute: int = 30
    session_limit_per_hour: int = 12

    # --- research data collection (opt-in, consented, features only) ------------
    data_collection_enabled: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    def validate_for_env(self) -> None:
        if self.env != "prod":
            return
        problems = []
        if not self.kek_keyring:
            problems.append("HP_KEK_KEYRING is required in prod")
        if not self.signing_keys:
            problems.append("HP_SIGNING_KEYS is required in prod")
        if not self.pairwise_secret:
            problems.append("HP_PAIRWISE_SECRET is required in prod")
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
