"""The Vercel deployment contract: vercel.json routing and the backend agree."""
import importlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_app_module():
    import app as app_module  # backend/app.py

    return importlib.reload(app_module)


def test_vercel_defaults_only_on_vercel():
    from app import vercel_defaults

    assert vercel_defaults({}) == {}
    d = vercel_defaults({"VERCEL": "1", "VERCEL_PROJECT_PRODUCTION_URL": "hp.vercel.app"})
    assert d == {
        "HP_API_PREFIX": "/api",
        "HP_PUBLIC_BASE_URL": "https://hp.vercel.app/api",
        "HP_ALLOWED_ORIGINS": '["https://hp.vercel.app"]',
        "HP_RP_ID": "hp.vercel.app",
    }


def test_vercel_defaults_never_override_explicit_settings():
    from app import vercel_defaults

    d = vercel_defaults({"VERCEL": "1", "VERCEL_URL": "x.vercel.app", "HP_RP_ID": "humanproof.example",
                         "HP_API_PREFIX": ""})
    assert "HP_RP_ID" not in d and "HP_API_PREFIX" not in d
    assert d["HP_PUBLIC_BASE_URL"] == "https://x.vercel.app/api"


def test_entrypoint_serves_under_api_on_vercel(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from humanproof.config import get_settings

    for k, v in {"VERCEL": "1", "VERCEL_PROJECT_PRODUCTION_URL": "hp.vercel.app", "HP_ENV": "dev",
                 "HP_DATA_DIR": str(tmp_path), "HP_ALLOW_HEURISTIC_FALLBACK": "true"}.items():
        monkeypatch.setenv(k, v)
    for k in ("HP_API_PREFIX", "HP_PUBLIC_BASE_URL", "HP_ALLOWED_ORIGINS", "HP_RP_ID"):
        monkeypatch.delenv(k, raising=False)
    get_settings.cache_clear()
    try:
        mod = _load_app_module()
        with TestClient(mod.app) as c:
            assert c.get("/api/healthz").json() == {"ok": True}
            assert c.get("/healthz").status_code == 404
            status = c.get("/api/v1/status").json()
            assert status["models"]["motor"] == "loaded"  # found without relying on the working directory
            r = c.post("/api/v1/sessions", json={"platform": "web", "consent": {
                "version": "2026-09-v1", "biometric_processing": True, "research_opt_in": False}})
            assert r.status_code == 200
        ctx = mod.app.state.ctx
        assert ctx.issuer == "https://hp.vercel.app/api"
        assert ctx.settings.rp_id == "hp.vercel.app"
        assert ctx.settings.allowed_origins == ["https://hp.vercel.app"]
    finally:
        get_settings.cache_clear()
        for k in ("HP_API_PREFIX", "HP_PUBLIC_BASE_URL", "HP_ALLOWED_ORIGINS", "HP_RP_ID"):
            monkeypatch.delenv(k, raising=False)


def _find_vercel_json() -> Path | None:
    for candidate in (REPO / "vercel.json", REPO.parent / "vercel.json"):
        if candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(_find_vercel_json() is None, reason="vercel.json not present")
def test_vercel_json_matches_the_code():
    path = _find_vercel_json()
    cfg = json.loads(path.read_text())
    services = cfg["services"]
    assert set(services) == {"backend", "client"}  # only web-facing pieces are Vercel services
    for name, svc in services.items():
        assert (path.parent / svc["root"]).is_dir(), f"{name} root does not exist"
    backend = path.parent / services["backend"]["root"]
    module, attr = services["backend"]["entrypoint"].split(":")
    assert (backend / f"{module}.py").exists() and attr == "app"
    assert not (backend / "pyproject.toml").exists()  # dependencies come from requirements.txt

    # /api/* -> backend must come before the catch-all, and match the prefix the backend mounts.
    rewrites = cfg["rewrites"]
    assert rewrites[0] == {"source": "/api/(.*)", "destination": {"service": "backend"}}
    assert rewrites[-1] == {"source": "/(.*)", "destination": {"service": "client"}}
    from app import vercel_defaults

    assert rewrites[0]["source"].startswith(vercel_defaults({"VERCEL": "1"})["HP_API_PREFIX"] + "/")
    # Top-level build keys are invalid in services mode.
    assert not {"buildCommand", "outputDirectory", "framework", "installCommand", "functions"} & set(cfg)
    # No service-to-service calls exist, so no bindings.
    assert all("bindings" not in s for s in services.values())
    # The web client's default API base is the same public prefix.
    api_ts = (path.parent / services["client"]["root"] / "src/lib/api.ts").read_text()
    assert '|| "/api"' in api_ts
