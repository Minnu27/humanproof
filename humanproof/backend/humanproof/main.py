"""ASGI entry point: ``uvicorn humanproof.main:app``."""
from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router
from .config import Settings, get_settings
from .context import build_context
from .scoring.voice import Transcriber
from .security.http import BodySizeLimitMiddleware, SecurityHeadersMiddleware

log = logging.getLogger("humanproof")


def create_app(settings: Settings | None = None, transcriber: Transcriber | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    context = build_context(settings, transcriber)

    is_prod = settings.env == "prod"

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async def purge_loop():
            while True:  # sessions (and their per-frame measurements) are deleted 1 h after expiry
                try:
                    n = await asyncio.to_thread(context.store.purge_expired)
                    if n:
                        log.info("Purged %d expired sessions", n)
                except Exception:
                    log.exception("Session purge failed")
                await asyncio.sleep(600)

        task = asyncio.create_task(purge_loop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(
        lifespan=lifespan,
        title="HumanProof",
        version="1.0.0",
        docs_url=None if is_prod else f"{settings.api_prefix}/docs",
        redoc_url=None,
        openapi_url=None if is_prod else f"{settings.api_prefix}/openapi.json",
    )
    app.state.ctx = context

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,  # bearer session IDs, no cookies: CSRF has nothing to ride on
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-hp-assertion", "x-hp-integrity"],
        max_age=600,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()][:10]
        return JSONResponse({"detail": {"invalid_fields": fields}}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")  # stack trace to logs only, never to clients
        return JSONResponse({"detail": "Internal error"}, status_code=500)

    app.include_router(router, prefix=settings.api_prefix)
    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    # Lazy module-level ``app`` so importing this module in tests has no side effects.
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
