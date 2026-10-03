"""Signal simulators for integration tests.

These produce *plausible* human-like and bot-like submissions so the full API
flow can be exercised end to end. They are test fixtures, not evidence of
real-world accuracy (that comes from the model evaluation reports).
"""
from __future__ import annotations

import base64
import io
import wave

import numpy as np

from humanproof.challenges import GazeChallenge, MotorChallenge

# ---------------------------------------------------------------------------
# Gaze
# ---------------------------------------------------------------------------


def human_gaze_frames(ch: GazeChallenge, rng: np.random.Generator, lag_ms: float = 210, fps: float = 30):
    """Eyes follow the target after a saccadic latency, head wobbles, occasional blinks."""
    t = np.arange(0, ch.duration_ms + 150, 1000 / fps) + rng.uniform(0, 3)
    eye = np.zeros((len(t), 2))
    cur = np.array(ch.target_at(0))
    for i, ti in enumerate(t):
        tgt = np.array(ch.target_at(max(0.0, ti - lag_ms)))
        cur = cur + 0.65 * (tgt - cur)  # fast but not instant (saccade dynamics)
        eye[i] = cur
    head_yaw = np.cumsum(rng.normal(0, 0.15, len(t)))
    head_pitch = np.cumsum(rng.normal(0, 0.12, len(t)))
    blink_at = set(rng.choice(len(t), 3, replace=False).tolist())
    frames = []
    for i, ti in enumerate(t):
        ex, ey = eye[i]
        frames.append({
            "t": float(ti),
            "ix": float(0.9 * (ex - 0.5) + rng.normal(0, 0.03)),
            "iy": float(0.5 * (ey - 0.5) + rng.normal(0, 0.05)),
            "yaw": float(np.clip(12 * (ex - 0.5) + head_yaw[i], -89, 89)),
            "pitch": float(np.clip(8 * (ey - 0.5) + head_pitch[i], -89, 89)),
            "roll": float(rng.normal(0, 0.5)),
            "ear": 0.08 if i in blink_at else float(0.3 + rng.normal(0, 0.01)),
            "mouth": float(abs(rng.normal(0.05, 0.01))),
            "face": True,
        })
    return frames


def webcam_gaze_frames(ch: GazeChallenge, rng: np.random.Generator, *, path: GazeChallenge | None = None,
                       lag_ms: float = 220, fps: float = 24, x_gain: float = 0.30, noise: float = 0.04,
                       y_gain: float = 0.0, lid_gain: float = 0.0, head_gain: float = 0.0):
    """Closer to a laptop webcam than ``human_gaze_frames``: a small horizontal iris
    signal under landmark jitter, almost no vertical signal, a nearly still head.

    ``path`` simulates replaying a recording: the eyes follow *that* challenge while
    the result is scored against ``ch``. ``head_gain`` (degrees per screen width)
    models someone who turns their head toward the dot.
    """
    src = path or ch
    t = np.arange(0, ch.duration_ms + 150, 1000 / fps) + rng.uniform(0, 3)
    cur = np.array(src.target_at(0))
    frames = []
    for ti in t:
        cur = cur + 0.65 * (np.array(src.target_at(max(0.0, ti - lag_ms))) - cur)
        frames.append({
            "t": float(ti),
            "ix": float(np.clip(x_gain * (cur[0] - 0.5) + rng.normal(0, noise), -1.9, 1.9)),
            "iy": float(np.clip(y_gain * (cur[1] - 0.5) + rng.normal(0, noise * 1.5), -1.9, 1.9)),
            "yaw": float(head_gain * (cur[0] - 0.5) + rng.normal(0, 0.4)),
            "pitch": float(rng.normal(0, 0.4)),
            "roll": 0.0,
            "ear": float(np.clip(0.28 - lid_gain * (cur[1] - 0.5) + rng.normal(0, 0.012), 0.01, 1.4)),
            "mouth": 0.05,
            "face": True,
        })
    return frames


def replayed_gaze_frames(ch: GazeChallenge, rng: np.random.Generator, fps: float = 30):
    """A pre-recorded face video: eye movement unrelated to this challenge."""
    other = np.cumsum(rng.normal(0, 0.05, (int(ch.duration_ms / 1000 * fps) + 5, 2)), axis=0)
    t = np.arange(len(other)) * 1000 / fps
    return [
        {"t": float(ti), "ix": float(np.clip(o[0], -1.9, 1.9)), "iy": float(np.clip(o[1], -1.9, 1.9)),
         "yaw": float(rng.normal(0, 2)), "pitch": float(rng.normal(0, 2)), "roll": 0.0,
         "ear": 0.3, "mouth": 0.05, "face": True}
        for ti, o in zip(t, other)
    ]


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------


def trace_samples(ch: MotorChallenge, rng: np.random.Generator, w: int = 1280, h: int = 800,
                  human: bool = True):
    line = np.array(ch.polyline(60)) * [w, h]
    seg = np.hypot(*np.diff(line, axis=0).T)
    arclen = np.concatenate([[0], np.cumsum(seg)])
    dur = arclen[-1] / rng.uniform(350, 700) * 1000
    if human:
        # speed varies with curvature-like sub-movements; tremor + drift; irregular events
        dt = np.abs(rng.normal(16.7, 3, int(dur / 16.7)))
        t = np.cumsum(dt)
        u = t / t[-1]
        s = u + 0.03 * np.sin(2 * np.pi * u * rng.uniform(5, 9)) * (1 - u) * u * 4
        s = np.clip(np.maximum.accumulate(s), 0, 1) * arclen[-1]
        pos = np.stack([np.interp(s, arclen, line[:, k]) for k in (0, 1)], axis=1)
        drift = np.cumsum(rng.normal(0, 0.9, pos.shape), axis=0)
        kernel = np.ones(9) / 9
        drift = np.stack([np.convolve(drift[:, k], kernel, mode="same") for k in (0, 1)], axis=1)
        drift -= np.linspace(0, 1, len(t))[:, None] * drift[-1]
        pos = pos + drift + rng.normal(0, 0.6, pos.shape)
    else:
        t = np.arange(0, dur, 16.0)
        s = (t / t[-1]) * arclen[-1]
        pos = np.stack([np.interp(s, arclen, line[:, k]) for k in (0, 1)], axis=1)
    pos = np.rint(pos)
    return [{"t": float(a), "x": float(b[0]), "y": float(b[1])} for a, b in zip(t - t[0], pos)]


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

SR = 16_000


def speech_like(rng: np.random.Generator, seconds: float = 3.5, digital_silence: bool = False) -> np.ndarray:
    """Glottal pulses with a wandering, jittered pitch, two formant resonances, a
    4-5 Hz syllabic envelope and (for "real mic" audio) a background noise floor."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f0 = 140 * 2 ** ((np.cumsum(rng.normal(0, 0.004, n)) + 0.15 * np.sin(2 * np.pi * 0.7 * t)) / 1.0)
    f0 *= 1 + rng.normal(0, 0.01, n)
    phase = np.cumsum(f0 / SR)
    pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    src = pulses + 0.02 * rng.normal(size=n)
    from scipy.signal import lfilter

    out = np.zeros(n)
    for fc, bw in ((600, 90), (1500, 120), (2600, 180)):
        r = np.exp(-np.pi * bw / SR)
        out += lfilter([1.0], [1.0, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], src)
    syll = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3.8, 5.0) * t + rng.uniform(0, 6)))
    syll *= (1 + 0.3 * rng.normal(size=n).cumsum() / np.sqrt(n)).clip(0.3, 1.7)
    words = (np.sin(2 * np.pi * 0.9 * t) > -0.3).astype(float)
    env = syll * words
    sig = out / (np.abs(out).max() + 1e-9) * env * 0.5
    if digital_silence:
        sig[env < 0.05] = 0.0
    else:
        sig += rng.normal(0, 0.003, n)
    return sig.astype(np.float32)


def wav_b64(x: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def mouth_frames_for(x: np.ndarray, rng: np.random.Generator, fps: float = 30, synced: bool = True):
    from humanproof.scoring.dsp import rms_envelope

    et, env = rms_envelope(x)
    t = np.arange(0, len(x) / SR * 1000, 1000 / fps)
    if synced:
        mouth = np.interp(t - 60, et, env / (env.max() + 1e-9)) * 0.6 + rng.normal(0, 0.02, len(t))
    else:
        mouth = np.abs(rng.normal(0.2, 0.1, len(t)))
    return [{"t": float(a), "mouth": float(np.clip(m, 0, 2.9)), "face": True} for a, m in zip(t, mouth)]


def face_crops(rng: np.random.Generator, n: int = 4) -> list[str]:
    from PIL import Image

    out = []
    base = rng.integers(60, 200, (160, 160, 3)).astype(np.uint8)
    for _ in range(n):
        img = np.clip(base.astype(int) + rng.integers(-25, 25, base.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=85)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out
