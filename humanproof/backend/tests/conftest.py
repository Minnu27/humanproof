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


@pytest.fixture
def settings(tmp_path):
    return Settings(
        env="test",
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


def has_motor_model() -> bool:
    return (MODELS_DIR / "motor.onnx").exists()


os.environ.setdefault("HP_ENV", "test")
