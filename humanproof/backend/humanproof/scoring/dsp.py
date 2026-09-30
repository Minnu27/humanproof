"""Audio signal processing for the voice checkpoint (pure numpy, no model needed)."""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(ValueError):
    pass


def decode_wav(data: bytes, *, min_s: float = 1.2, max_s: float = 9.0) -> np.ndarray:
    """Strictly parse 16 kHz / mono / 16-bit PCM WAV into float32 in [-1, 1]."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                raise AudioError("Audio must be 16 kHz mono 16-bit PCM")
            n = wf.getnframes()
            if not (min_s * SAMPLE_RATE <= n <= max_s * SAMPLE_RATE):
                raise AudioError("Audio duration out of range")
            raw = wf.readframes(n)
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError("Invalid WAV") from exc
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if len(pcm) < min_s * SAMPLE_RATE:
        raise AudioError("Truncated audio")
    return pcm


def frame_signal(x: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def rms_envelope(x: np.ndarray, hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """(times_ms, rms) with 25 ms frames every ``hop_ms``."""
    hop = int(SAMPLE_RATE * hop_ms / 1000)
    frames = frame_signal(x, 400, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    t = (np.arange(len(rms)) * hop + 200) / SAMPLE_RATE * 1000.0
    return t, rms


def _pitch_track(x: np.ndarray, fmin: float = 70, fmax: float = 400) -> np.ndarray:
    """Autocorrelation pitch per 40 ms frame (0 = unvoiced)."""
    frames = frame_signal(x, 640, 160) * np.hanning(640)
    lo, hi = int(SAMPLE_RATE / fmax), int(SAMPLE_RATE / fmin)
    f0 = np.zeros(len(frames))
    energy = (frames ** 2).mean(axis=1)
    gate = np.percentile(energy, 60)
    for i, fr in enumerate(frames):
        if energy[i] < gate:
            continue
        spec = np.fft.rfft(fr, 2048)
        ac = np.fft.irfft(np.abs(spec) ** 2)[: hi + 1]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.45:
            f0[i] = SAMPLE_RATE / lag
    return f0


@dataclass
class VoiceDsp:
    speech_ratio: float        # fraction of frames above the noise gate
    digital_silence: float     # fraction of 10 ms frames that are exact zeros
    noise_floor_db: float
    pitch_std_semitones: float
    jitter: float              # mean relative frame-to-frame f0 change in voiced runs
    shimmer: float             # mean relative frame-to-frame amplitude change
    voiced_ratio: float
    syllabic_peak: float       # share of envelope modulation energy in 2-8 Hz
    clipping: float

    def liveness_score(self) -> float:
        """Heuristic 0..1 used only as a supporting signal (or dev fallback).

        Captures what synthetic/replayed pipelines commonly get wrong: digital
        silence between words, monotone pitch, missing natural micro-variation,
        and absent syllabic rhythm. It is not a substitute for the trained
        anti-spoofing model.
        """
        s = 1.0
        s *= 1.0 - min(1.0, self.digital_silence * 8)
        # Wide bands: this only has to catch flat, machine-steady pitch, and a
        # simple frame-level pitch tracker overstates natural variation.
        s *= _band(self.pitch_std_semitones, 0.5, 1.2, 9.0, 14.0)
        s *= _band(self.jitter, 0.001, 0.003, 0.2, 0.4)
        s *= _band(self.shimmer, 0.02, 0.05, 0.35, 0.6)
        s *= _band(self.syllabic_peak, 0.15, 0.3, 1.01, 1.02)
        s *= _band(self.speech_ratio, 0.15, 0.25, 0.95, 1.0)
        s *= 1.0 - min(1.0, self.clipping * 20)
        return float(max(0.0, min(1.0, s)))


def _band(v: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """Trapezoid: 0 below lo0, ramps to 1 at lo1, 1 until hi1, ramps to 0 at hi0."""
    if v <= lo0 or v >= hi0:
        return 0.05
    if v < lo1:
        return 0.05 + 0.95 * (v - lo0) / (lo1 - lo0)
    if v > hi1:
        return 0.05 + 0.95 * (hi0 - v) / (hi0 - hi1)
    return 1.0


def analyse(x: np.ndarray) -> VoiceDsp:
    frames10 = frame_signal(x, 160, 160)
    digital_silence = float(np.mean(np.all(frames10 == 0, axis=1)))
    _, rms = rms_envelope(x)
    db = 20 * np.log10(rms + 1e-9)
    floor = float(np.percentile(db, 10))
    speech = db > floor + 15
    f0 = _pitch_track(x)
    voiced = f0 > 0
    semis = 12 * np.log2(f0[voiced] / np.median(f0[voiced])) if voiced.sum() > 5 else np.zeros(1)
    both = voiced[1:] & voiced[:-1]
    jitter = float(np.mean(np.abs(np.diff(f0))[both] / f0[1:][both])) if both.sum() > 3 else 0.0
    amp = frame_signal(x, 640, 160).std(axis=1)
    shimmer = float(np.mean(np.abs(np.diff(amp))[both] / (amp[1:][both] + 1e-9))) if both.sum() > 3 else 0.0
    env = rms - rms.mean()
    spec = np.abs(np.fft.rfft(env)) ** 2
    freqs = np.fft.rfftfreq(len(env), d=0.01)
    band = (freqs >= 2) & (freqs <= 8)
    total = spec[(freqs > 0.5) & (freqs < 30)].sum() + 1e-12
    return VoiceDsp(
        speech_ratio=float(speech.mean()),
        digital_silence=digital_silence,
        noise_floor_db=floor,
        pitch_std_semitones=float(np.std(semis)),
        jitter=jitter,
        shimmer=shimmer,
        voiced_ratio=float(voiced.mean()),
        syllabic_peak=float(spec[band].sum() / total),
        clipping=float(np.mean(np.abs(x) > 0.99)),
    )
