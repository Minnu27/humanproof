import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from humanproof.config import Settings  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class FakeTranscriber:
    """Echoes whatever words the test says were spoken."""

    def __init__(self):
        self.next_text = ""

    def transcribe(self, audio):
        return self.next_text


class Clock:
    def __init__(self, start: float):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


TEST_MASTER_SECRET = "dGVzdC1tYXN0ZXItc2VjcmV0LWZvci1odW1hbnByb29mLXRlc3RzLTAwMQ=="  # test fixture only
TEST_COLLECTION_KEY = "tester-code-for-tests"


@pytest.fixture
def database_url():
    """Empty for SQLite. With HP_TEST_DATABASE_URL set, every test runs against a
    throwaway schema in that Postgres database instead."""
    base = os.environ.get("HP_TEST_DATABASE_URL", "")
    if not base:
        yield ""
        return
    import secrets as _secrets

    import psycopg

    schema = "t_" + _secrets.token_hex(6)
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in base else "?"
    yield f"{base}{sep}options=-csearch_path%3D{schema}"
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture
def settings(tmp_path, database_url):
    return Settings(
        env="test",
        database_url=database_url,
        master_secret=TEST_MASTER_SECRET,
        collection_key=TEST_COLLECTION_KEY,
        data_dir=tmp_path / "var",
        models_dir=MODELS_DIR,
        allow_heuristic_fallback=True,
        allowed_origins=["http://localhost:5173"],
        public_base_url="http://testserver",
        rate_limit_per_minute=1000,
        session_limit_per_hour=1000,
        data_collection_enabled=True,
    )


@pytest.fixture
def transcriber():
    return FakeTranscriber()


@pytest.fixture
def clock(monkeypatch):
    import time as _time

    c = Clock(_time.time())
    monkeypatch.setattr(_time, "time", c)
    return c


@pytest.fixture
def client(settings, transcriber, clock):
    from fastapi.testclient import TestClient

    from humanproof.main import create_app

    app = create_app(settings, transcriber=transcriber)
    with TestClient(app) as tc:
        tc.app_ctx = app.state.ctx
        yield tc
    app.state.ctx.store.close()


def has_motor_model() -> bool:
    return (MODELS_DIR / "motor.onnx").exists()


os.environ.setdefault("HP_ENV", "test")
