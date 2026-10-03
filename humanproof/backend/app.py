"""ASGI entrypoint for Vercel (service entrypoint ``app:app``).

Settings come from HP_* environment variables (see .env.example). On Vercel,
sensible defaults are derived from the platform's own variables so that a first
deployment works without configuration; anything set explicitly wins.
"""
from __future__ import annotations

import json
import os


def vercel_defaults(env: dict[str, str]) -> dict[str, str]:
    """HP_* defaults for a Vercel deployment. Returns only keys not already set."""
    if not env.get("VERCEL"):
        return {}
    # vercel.json routes /api/* to this service and passes the path through
    # unchanged, so the API has to be mounted under /api.
    out = {"HP_API_PREFIX": "/api"}
    host = env.get("VERCEL_PROJECT_PRODUCTION_URL") or env.get("VERCEL_URL")
    if host:
        out["HP_PUBLIC_BASE_URL"] = f"https://{host}/api"      # token issuer; JWKS lives under it
        out["HP_ALLOWED_ORIGINS"] = json.dumps([f"https://{host}"])  # CORS + passkey origin
        out["HP_RP_ID"] = host                                  # passkey relying-party id
    return {k: v for k, v in out.items() if k not in env}


os.environ.update(vercel_defaults(dict(os.environ)))

from humanproof.main import create_app  # noqa: E402

app = create_app()
