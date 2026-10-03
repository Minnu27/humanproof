import os
import random
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
        assert 8000 < g.duration_ms < 11500
        assert all(0 < c < g.duration_ms for c in g.capture_ms)
        jumps = g.keyframes[1:]
        assert len(jumps) == C.GAZE_JUMPS and all(k.mode == "jump" for k in jumps)
        pos = (0.5, 0.5)
        for prev, k in zip(g.keyframes, jumps):
            assert 0.1 <= k.x1 <= 0.9 and 0.12 <= k.y1 <= 0.88
            assert 600 <= k.start - prev.start or prev.start == 0
            assert np.hypot(k.x1 - pos[0], k.y1 - pos[1]) >= 0.3  # every jump is a visible eye movement
            assert g.target_at(k.start - 1) == pos and g.target_at(k.start) == (k.x1, k.y1)
            pos = (k.x1, k.y1)


def test_gaze_path_is_not_predictable():
    """The dot must not bounce left-right-left-right: horizontal moves of two
    independent challenges should be unrelated (this is what defeats replays)."""
    r = random.Random(3)
    corrs = []
    for _ in range(300):
        a, b = C.make_gaze_challenge(r), C.make_gaze_challenge(r)
        dx = [np.diff([k.x1 for k in c.keyframes]) for c in (a, b)]
        corrs.append(np.corrcoef(dx[0], dx[1])[0, 1])
    assert abs(np.mean(corrs)) < 0.05 and np.mean(np.abs(corrs) > 0.8) < 0.02


def test_voice_words_unique():
    for _ in range(100):
        w = C.make_voice_challenge().words
        assert len(set(w)) == 5


# ---- gaze -------------------------------------------------------------------------
# Challenges are seeded here so that the measured rates are reproducible.

def _frames(fs):
    return [GazeFrame(**f) for f in fs]


def _gaze_scores(make_frames, n, seed):
    r, rng = random.Random(seed), np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ch, other = C.make_gaze_challenge(r), C.make_gaze_challenge(r)
        out.append(gaze.score(ch, _frames(make_frames(ch, other, rng)), None))
    return out


def test_gaze_human_follower_passes():
    res = _gaze_scores(lambda ch, o, rng: sim.human_gaze_frames(ch, rng), 20, 1)
    assert all(s > 0.6 for s, _, _ in res)
    assert all(120 <= f["latency_med"] <= 400 for _, _, f in res)


def test_gaze_webcam_quality_humans_pass():
    """Small, noisy, horizontal-only eye signal, as from a laptop webcam."""
    cases = {  # name: (simulation settings, minimum share scoring 0.5 or better)
        "typical": (dict(), 0.93),
        "turns head instead of eyes": (dict(x_gain=0, head_gain=10, lag_ms=300), 0.93),
        # Worst case: slow reactions *and* a low frame rate leave few frames per
        # fixation. Measured around 83%; these users are asked to retry.
        "slow responder, 15 fps": (dict(lag_ms=380, fps=15), 0.72),
    }
    for name, (kw, floor) in cases.items():
        scores = np.array([s for s, _, _ in _gaze_scores(
            lambda ch, o, rng: sim.webcam_gaze_frames(ch, rng, **kw), 200, 2)])
        assert np.mean(scores >= 0.5) > floor, (name, float(np.mean(scores >= 0.5)))
    _, _, f = _gaze_scores(lambda ch, o, rng: sim.webcam_gaze_frames(ch, rng, x_gain=0, head_gain=10), 1, 3)[0]
    assert f["head_led"] == 1.0


def test_gaze_replayed_video_fails():
    scores = np.array([s for s, _, _ in _gaze_scores(lambda ch, o, rng: sim.replayed_gaze_frames(ch, rng), 400, 4)])
    assert np.mean(scores >= 0.25) < 0.03 and np.mean(scores >= 0.5) < 0.015


def test_gaze_recording_from_another_session_fails():
    """A genuine recording of a person doing a *different* challenge, replayed."""
    for kw in (dict(), dict(x_gain=0.9, noise=0.01, y_gain=0.5, lid_gain=0.05)):
        scores = np.array([s for s, _, _ in _gaze_scores(
            lambda ch, o, rng: sim.webcam_gaze_frames(ch, rng, path=o, **kw), 400, 5)])
        assert np.mean(scores >= 0.25) < 0.03 and np.mean(scores >= 0.5) < 0.015


def test_gaze_zero_lag_script_penalised():
    """A script that moves the 'eyes' exactly with the target (no human latency)."""
    res = _gaze_scores(lambda ch, o, rng: sim.webcam_gaze_frames(ch, rng, lag_ms=0, x_gain=0.5, noise=0.02), 20, 6)
    assert all(f["latency_med"] < 80 for _, _, f in res)
    assert all(s < 0.25 and "eye response timing not human" in r for s, r, _ in res)


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


def test_gaze_too_few_frames_is_reported():
    rng = np.random.default_rng(6)
    ch = C.make_gaze_challenge()
    frames = sim.webcam_gaze_frames(ch, rng, fps=11)
    for f in frames:  # face lost for most of the run
        f["face"] = f["t"] < 2500 or f["t"] > ch.duration_ms - 300
    s, reasons, feats = gaze.score(ch, _frames(frames), None)
    assert s == 0 and feats["n_events"] < gaze.MIN_EVENTS
    assert "not enough camera frames to measure eye movement" in reasons


def test_gaze_handles_legacy_glide_keyframes():
    """Sessions created before the 12-jump design may still carry a glide."""
    kfs = [C.GazeKeyframe(0, 0, .5, .5, .5, .5), C.GazeKeyframe(1000, 1000, .2, .3, .2, .3),
           C.GazeKeyframe(1800, 3300, .2, .3, .8, .7), C.GazeKeyframe(3900, 3900, .3, .6, .3, .6)]
    ch = C.GazeChallenge(4700, kfs, [1000])
    jumps = gaze._jumps(ch)
    assert [j[0] for j in jumps] == [1000.0, 3900.0] and jumps[1][2] == (.8, .7)
    s, reasons, _ = gaze.score(ch, _frames(sim.webcam_gaze_frames(ch, np.random.default_rng(0))), None)
    assert s == 0 and "not enough camera frames to measure eye movement" in reasons


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


def test_voice_demo_mode_without_speech_recognition():
    """No ASR in demo mode: real speech is not capped at half marks, silence still fails."""
    rng = np.random.default_rng(6)
    ch = C.make_voice_challenge()
    x = sim.speech_like(rng)
    sub = VoiceSubmission(audio_wav_b64=sim.wav_b64(x), audio_offset_ms=0, mouth_frames=sim.mouth_frames_for(x, rng))
    s, reasons, info, _ = voice.score(ch, sub, antispoof_model=None, speaker_model=None,
                                      transcriber=None, allow_fallback=True)
    assert 0.6 < s <= 0.85 and info["phrase_checked"] is False
    assert any("words not checked" in r for r in reasons)

    quiet = (rng.normal(0, 0.002, len(x))).astype(np.float32)  # room noise, nobody speaking
    sub = VoiceSubmission(audio_wav_b64=sim.wav_b64(quiet), audio_offset_ms=0)
    s, reasons, _, _ = voice.score(ch, sub, antispoof_model=None, speaker_model=None,
                                   transcriber=None, allow_fallback=True)
    assert s < 0.25 and "not enough speech in the recording" in reasons
