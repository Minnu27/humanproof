"""HTTP hardening: security headers, body-size limits, rate limiting.

The in-memory limiter is correct for a single process. For multiple replicas,
swap ``RateLimiter`` for a Redis-backed implementation with the same interface.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    # The API only returns JSON; nothing should ever render or execute.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cache-Control": "no-store",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if "server" in response.headers:
            del response.headers["server"]
        return response


class BodySizeLimitMiddleware:
    """Rejects bodies over ``max_bytes``, whether or not Content-Length is honest."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)
                        return
                except ValueError:
                    await JSONResponse({"detail": "Bad Content-Length"}, 400)(scope, receive, send)
                    return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _TooLarge:
            await JSONResponse({"detail": "Request too large"}, 413)(scope, receive, send)


class _TooLarge(Exception):
    pass


class RateLimiter:
    """Sliding-window counter keyed by an arbitrary string (IP, device, session)."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: float) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits[key] if now - t < window_seconds]
            if len(hits) >= limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            if len(self._hits) > 100_000:  # bound memory under key-spraying
                self._evict(now, window_seconds)
            return True

    def _evict(self, now: float, window_seconds: float) -> None:
        for k in [k for k, v in self._hits.items() if not v or now - v[-1] > window_seconds]:
            del self._hits[k]


def client_ip(request: Request) -> str:
    # Behind your own reverse proxy, configure uvicorn --proxy-headers --forwarded-allow-ips
    # so request.client reflects the real peer. A forwarding header is trusted only
    # when the deployment says the platform's edge sets it (HP_TRUSTED_IP_HEADER).
    ctx = getattr(request.app.state, "ctx", None)
    header = ctx.settings.trusted_ip_header if ctx is not None else ""
    if header:
        value = request.headers.get(header, "").split(",")[0].strip()
        if value:
            return value[:64]
    return request.client.host if request.client else "unknown"
