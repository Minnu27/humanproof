import os
from pathlib import Path

import numpy as np
import pytest

import sim
from humanproof import challenges as C
from humanproof.schemas import GazeFrame, MotorSubmission, VoiceSubmission
from humanproof.scoring import dsp, fusion, gaze, motor, voice

from conftest import has_motor_model


# ---- challenge generator ---------------------------------------------------------

def test_challenges_are_unpredictable():
    a, b = C.make_challenge_set(), C.make_challenge_set()
    assert a["gaze"]["keyframes"] != b["gaze"]["keyframes"]
    assert a["motor"]["control_points"] != b["motor"]["control_points"]
    assert a["voice"]["words"] != b["voice"]["words"]


def test_gaze_challenge_shape():
    for _ in range(50):
        g = C.make_gaze_challenge()
        assert 5000 < g.duration_ms < 10000
        assert all(0 < c < g.duration_ms for c in g.capture_ms)
        assert any(k.mode == "glide" for k in g.keyframes)
        for k in g.keyframes:
            assert 0.1 <= k.x1 <= 0.9 and 0.1 <= k.y1 <= 0.9
        # the target path is continuous during glides and holds between keyframes
        for kf in g.keyframes:
            if kf.mode == "glide":
                mid = g.target_at((kf.start + kf.end) / 2)
                assert mid == pytest.approx(((kf.x0 + kf.x1) / 2, (kf.y0 + kf.y1) / 2), abs=1e-6)


def test_voice_words_unique():
    for _ in range(100):
        w = C.make_voice_challenge().words
        assert len(set(w)) == 5


# ---- gaze -------------------------------------------------------------------------

def _frames(fs):
    return [GazeFrame(**f) for f in fs]


def test_gaze_human_follower_passes():
    rng = np.random.default_rng(1)
    for _ in range(5):
        ch = C.make_gaze_challenge()
        s, reasons, feats = gaze.score(ch, _frames(sim.human_gaze_frames(ch, rng)), None)
        assert s > 0.6, reasons
        assert 100 <= feats["best_lag_ms"] <= 400


def test_gaze_replayed_video_fails():
    rng = np.random.default_rng(2)
    for _ in range(40):
        ch = C.make_gaze_challenge()
        s, _, _ = gaze.score(ch, _frames(sim.replayed_gaze_frames(ch, rng)), None)
        assert s < 0.3


def test_gaze_zero_lag_script_penalised():
    """A script that moves the 'eyes' exactly with the target (no human latency)."""
    rng = np.random.default_rng(3)
    ch = C.make_gaze_challenge()
    s_bot, reasons, feats = gaze.score(ch, _frames(sim.human_gaze_frames(ch, rng, lag_ms=0)), None)
    s_hum, _, _ = gaze.score(ch, _frames(sim.human_gaze_frames(ch, rng)), None)
    assert feats["best_lag_ms"] < 60
    assert s_bot < 0.5 < s_hum


def test_gaze_rejects_bad_timing_and_coverage():
    rng = np.random.default_rng(4)
    ch = C.make_gaze_challenge()
    frames = sim.human_gaze_frames(ch, rng)
    s, reasons, _ = gaze.score(ch, _frames(frames[: len(frames) // 2]), None)
    assert s == 0 and "cover" in reasons[0]
    shuffled = frames[:]
    shuffled[5]["t"] = shuffled[4]["t"]
    s, reasons, _ = gaze.score(ch, _frames(shuffled), None)
    assert s == 0


def test_gaze_absent_face_fails():
    rng = np.random.default_rng(5)
    ch = C.make_gaze_challenge()
    frames = sim.human_gaze_frames(ch, rng)
    for f in frames[::2]:
        f["face"] = False
    s, reasons, _ = gaze.score(ch, _frames(frames), None)
    assert s < 0.25 and any("face" in r for r in reasons)


# ---- motor ------------------------------------------------------------------------

def _motor_sub(samples, pointer="mouse"):
    return MotorSubmission(pointer_type=pointer, samples=samples, viewport={"w": 1280, "h": 800})


def test_motor_adherence_detects_wrong_curve():
    rng = np.random.default_rng(6)
    ch, other = C.make_motor_challenge(), C.make_motor_challenge()
    good, _, _ = motor.adherence(ch, _motor_sub(sim.trace_samples(ch, rng)))
    bad, _, reasons = motor.adherence(ch, _motor_sub(sim.trace_samples(other, rng)))
    assert good > 0.8 and bad < 0.3, reasons


def test_motor_machine_perfect_trace_flagged():
    rng = np.random.default_rng(7)
    ch = C.make_motor_challenge()
    s, info, reasons = motor.adherence(ch, _motor_sub(sim.trace_samples(ch, rng, human=False)))
    assert "trace is machine-perfect" in reasons and s < 0.2


def test_motor_too_fast_flagged():
    rng = np.random.default_rng(8)
    ch = C.make_motor_challenge()
    samples = sim.trace_samples(ch, rng)
    for s_ in samples:
        s_["t"] /= 10
    s, _, reasons = motor.adherence(ch, _motor_sub(samples))
    assert s < 0.2 and any("fast" in r for r in reasons)


def test_motor_without_model_fails_closed():
    rng = np.random.default_rng(9)
    ch = C.make_motor_challenge()
    s, reasons, _ = motor.score(ch, _motor_sub(sim.trace_samples(ch, rng)), None)
    assert s == 0 and "motor model not available" in reasons


@pytest.mark.skipif(not has_motor_model(), reason="motor model not trained")
def test_motor_model_separates_scripted_from_human_like():
    from humanproof.models import ModelRegistry
    from conftest import MODELS_DIR

    model = ModelRegistry(MODELS_DIR).get("motor")
    rng = np.random.default_rng(10)
    bot_scores, hum_scores = [], []
    for _ in range(8):
        ch = C.make_motor_challenge()
        bot_scores.append(motor.dynamics(_motor_sub(sim.trace_samples(ch, rng, human=False)), model)[0])
        hum_scores.append(motor.dynamics(_motor_sub(sim.trace_samples(ch, rng)), model)[0])
    assert np.mean(bot_scores) < 0.5
    assert np.mean(bot_scores) < np.mean(hum_scores)


IITKGP = Path(os.environ.get("HP_IITKGP_DATA", "__unset__"))  # clone of github.com/prativa-97/Mouse-Dynamics


@pytest.mark.skipif(not (has_motor_model() and IITKGP.exists()), reason="set HP_IITKGP_DATA to run")
def test_motor_model_on_real_held_out_humans():
    """Real movement from users the model never saw must mostly pass; scripted bots must not."""
    import itertools
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
    from motor.bot_generator import generate
    from motor.train import TEST_USERS, iitkgp_strokes

    from humanproof.models import ModelRegistry
    from conftest import MODELS_DIR

    model = ModelRegistry(MODELS_DIR).get("motor")
    thr = model.meta["threshold"]
    strokes = (s for s in iitkgp_strokes(IITKGP) if s[0] in TEST_USERS and s[1][-1] - s[1][0] > 900)
    human = []
    for _, t, x, y in itertools.islice(strokes, 300):
        f = motor.trajectory_features(t, x, y)
        if len(f):
            human.append(float(model.run(f.astype(np.float32))[1][:, 1].mean()))
    rng = np.random.default_rng(99)
    bots = []
    for fam in ("linear", "bezier", "windmouse", "spline_follow", "submovement"):
        for _ in range(30):
            _, t, x, y = generate(rng, fam)
            f = motor.trajectory_features(t, x, y)
            if len(f):
                bots.append(float(model.run(f.astype(np.float32))[1][:, 1].mean()))
    assert len(human) > 100
    assert np.mean(np.asarray(human) >= thr) > 0.9
    assert np.mean(np.asarray(bots) < thr) > 0.85


# ---- voice ------------------------------------------------------------------------

def test_wav_validation():
    x = sim.speech_like(np.random.default_rng(1), 2.0)
    assert len(dsp.decode_wav(__import__("base64").b64decode(sim.wav_b64(x)))) == 32000
    with pytest.raises(dsp.AudioError):
        dsp.decode_wav(b"RIFF....not a wav")
    with pytest.raises(dsp.AudioError):
        dsp.decode_wav(__import__("base64").b64decode(sim.wav_b64(x[:8000])))  # too short


def test_dsp_flags_digital_silence():
    rng = np.random.default_rng(2)
    real = dsp.analyse(sim.speech_like(rng)).liveness_score()
    tts = dsp.analyse(sim.speech_like(rng, digital_silence=True)).liveness_score()
    assert real > 0.7 and tts < 0.2


def test_phrase_match_in_order_and_fuzzy():
    words = ["apple", "river", "candle", "tiger", "window"]
    assert voice.phrase_match(words, "Apple, river... candel tiger window!")[0] == 5
    assert voice.phrase_match(words, "window tiger candle river apple")[0] <= 2
    assert voice.phrase_match(words, "hello there")[0] == 0


def test_lip_sync():
    rng = np.random.default_rng(3)
    x = sim.speech_like(rng)
    b64 = sim.wav_b64(x)
    synced = VoiceSubmission(audio_wav_b64=b64, audio_offset_ms=0, mouth_frames=sim.mouth_frames_for(x, rng))
    unsynced = VoiceSubmission(audio_wav_b64=b64, audio_offset_ms=0,
                               mouth_frames=sim.mouth_frames_for(x, rng, synced=False))
    assert voice.lip_sync(synced, x) > 0.5
    assert voice.lip_sync(unsynced, x) < 0.3


def test_voice_wrong_words_rejected(transcriber):
    rng = np.random.default_rng(4)
    ch = C.make_voice_challenge()
    x = sim.speech_like(rng)
    sub = VoiceSubmission(audio_wav_b64=sim.wav_b64(x), audio_offset_ms=0, mouth_frames=sim.mouth_frames_for(x, rng))
    transcriber.next_text = "completely different words here"
    s, reasons, _, _ = voice.score(ch, sub, antispoof_model=None, speaker_model=None,
                                   transcriber=transcriber, allow_fallback=True)
    assert s == 0 and "did not say the requested words" in reasons


def test_voice_fails_closed_without_models():
    rng = np.random.default_rng(5)
    ch = C.make_voice_challenge()
    x = sim.speech_like(rng)
    sub = VoiceSubmission(audio_wav_b64=sim.wav_b64(x), audio_offset_ms=0)
    s, reasons, _, _ = voice.score(ch, sub, antispoof_model=None, speaker_model=None,
                                   transcriber=None, allow_fallback=False)
    assert s == 0


# ---- fusion -----------------------------------------------------------------------

def test_fusion_decisions():
    good = {"gaze": 0.9, "face": 0.85, "motor": 0.9, "voice": 0.9}
    assert fusion.fuse(good, "device_attested").decision == "pass"
    one_fail = {**good, "voice": 0.1}
    assert fusion.fuse(one_fail, "device_attested").decision == "reject"
    borderline = {"gaze": 0.65, "face": 0.6, "motor": 0.7, "voice": 0.7}
    assert fusion.fuse(borderline, "device_attested").decision in ("step_up", "pass")
    assert fusion.fuse({"gaze": 0.9}, "web").decision == "reject"


def test_web_needs_more_evidence_than_attested_device():
    scores = {"gaze": 0.74, "face": 0.74, "motor": 0.74, "voice": 0.74}
    assert fusion.fuse(scores, "device_attested").decision == "pass"
    assert fusion.fuse(scores, "web").decision == "step_up"
