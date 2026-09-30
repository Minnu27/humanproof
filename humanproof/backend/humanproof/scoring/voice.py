"""Checkpoint 3: live voice reading random words.

Signals:
* phrase match - ASR transcript must contain the secret random words, in order;
* anti-spoofing - trained model (``voice_antispoof.onnx``, e.g. AASIST-style,
  trained on ASVspoof) scores bona fide vs synthetic/converted/replayed speech;
* DSP liveness - digital silence, pitch/jitter/shimmer, syllabic rhythm;
* lip sync - mouth opening (from face landmarks) must track the audio envelope;
* voiceprint - speaker embedding (``voice_speaker.onnx``, ECAPA-TDNN) stored
  encrypted as the baseline for later re-verification.

Raw audio is processed in memory and discarded.
"""
from __future__ import annotations

import base64
import binascii
import difflib
import re
from dataclasses import asdict
from typing import Protocol

import numpy as np

from ..challenges import VoiceChallenge
from ..schemas import VoiceSubmission
from . import dsp

ANTISPOOF_SAMPLES = 64_600  # ~4.04 s at 16 kHz, the AASIST convention


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class WhisperTranscriber:
    """faster-whisper, loaded lazily. Model name or local path from HP_ASR_MODEL."""

    def __init__(self, model: str):
        from faster_whisper import WhisperModel  # optional dependency

        self._m = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._m.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments)


def phrase_match(expected: list[str], transcript: str) -> tuple[int, list[str]]:
    tokens = re.findall(r"[a-z]+", transcript.lower())
    matched, heard, pos = 0, [], 0
    for word in expected:
        best, best_i = 0.0, -1
        for i in range(pos, len(tokens)):
            r = difflib.SequenceMatcher(None, word, tokens[i]).ratio()
            if r > best:
                best, best_i = r, i
        if best >= 0.75:
            matched += 1
            heard.append(tokens[best_i])
            pos = best_i + 1
    return matched, heard


def _antispoof(model, x: np.ndarray) -> float:
    if len(x) < ANTISPOOF_SAMPLES:
        x = np.tile(x, int(np.ceil(ANTISPOOF_SAMPLES / len(x))))
    n_crops = 1 if len(x) < ANTISPOOF_SAMPLES + 8000 else 3
    starts = np.linspace(0, len(x) - ANTISPOOF_SAMPLES, num=n_crops)
    probs = []
    for s in starts.astype(int):
        logits = model.run(x[None, s:s + ANTISPOOF_SAMPLES].astype(np.float32))[0][0]
        e = np.exp(logits - logits.max())
        probs.append(float(e[1] / e.sum()))  # index 1 = bona fide
    return float(np.mean(probs))


def lip_sync(sub: VoiceSubmission, x: np.ndarray) -> float | None:
    frames = [f for f in sub.mouth_frames if f.face]
    if len(frames) < 30:
        return None
    ft = np.array([f.t for f in frames]) - sub.audio_offset_ms
    fm = np.array([f.mouth for f in frames])
    et, env = dsp.rms_envelope(x)
    env = np.log(env + 1e-4)
    m = (et >= ft[0]) & (et <= ft[-1])
    if m.sum() < 50:
        return None
    et, env = et[m], env[m]
    best = -1.0
    for lag in range(-100, 260, 20):  # mouth may lead the sound slightly
        mouth = np.interp(et - lag, ft, fm)
        if mouth.std() < 1e-6 or env.std() < 1e-6:
            continue
        best = max(best, float(np.corrcoef(mouth, env)[0, 1]))
    return best


def decode_audio(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise dsp.AudioError("Audio is not valid base64") from exc
    return dsp.decode_wav(raw)


def score(
    ch: VoiceChallenge,
    sub: VoiceSubmission,
    *,
    antispoof_model,
    speaker_model,
    transcriber: Transcriber | None,
    allow_fallback: bool,
) -> tuple[float, list[str], dict, np.ndarray | None]:
    reasons: list[str] = []
    try:
        x = decode_audio(sub.audio_wav_b64)
    except dsp.AudioError as exc:
        return 0.0, [str(exc)], {}, None

    feats = dsp.analyse(x)
    info: dict = {"dsp": asdict(feats), "dsp_liveness": feats.liveness_score()}

    # 1. phrase
    if transcriber is None:
        if not allow_fallback:
            return 0.0, ["speech recognition not configured"], info, None
        phrase_factor = 0.5
        reasons.append("phrase not verified (no ASR configured; dev mode)")
    else:
        matched, heard = phrase_match(ch.words, transcriber.transcribe(x))
        info.update(words_matched=matched, heard=heard)
        if matched >= 4:
            phrase_factor = 1.0
        elif matched == 3:
            phrase_factor = 0.4
            reasons.append("some words not recognised")
        else:
            return 0.0, reasons + ["did not say the requested words"], info, None

    # 2. anti-spoofing
    if antispoof_model is not None:
        live = _antispoof(antispoof_model, x)
        info["antispoof_p_bonafide"] = live
    elif allow_fallback:
        live = feats.liveness_score()
        reasons.append("anti-spoofing model missing; DSP heuristics only (dev mode)")
    else:
        return 0.0, reasons + ["anti-spoofing model not available"], info, None
    if live < 0.5:
        reasons.append("voice sounds synthetic or replayed")

    support = 0.6 + 0.4 * feats.liveness_score()

    # 3. lip sync
    ls = lip_sync(sub, x)
    info["lip_sync_r"] = ls
    if ls is None:
        lip = 0.5
        reasons.append("lip sync not measurable")
    else:
        lip = float(min(1.0, max(0.0, (ls - 0.05) / 0.35)))
        if lip < 0.3:
            reasons.append("mouth movement does not match the voice")

    # 4. voiceprint
    embedding = None
    if speaker_model is not None:
        emb = speaker_model.run(x[None, :].astype(np.float32))[0].reshape(-1)
        embedding = (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)

    s = phrase_factor * live * support * (0.5 + 0.5 * lip)
    return float(s), reasons, info, embedding
