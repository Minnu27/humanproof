"""Checkpoint 1: face presence + eye movement following a random target.

The client sends per-frame signals derived from MediaPipe face/iris landmarks
(never video). The dot makes a series of jumps to independent random positions,
and the question is whether the eyes (or head) moved the way *this* path did.

How it is measured
------------------
For every jump, the eye signal is summarised just before the jump and again once
a saccade has had time to land (medians over each fixation). The difference is
that jump's displacement. Medians over ~6-8 frames average away landmark jitter,
which matters: single-frame iris positions from a laptop webcam are noisy.

Across the jumps we then ask: does the displacement track where the dot went?
Specifically the partial correlation between displacement and the dot's new
position, controlling for its previous position. Controlling matters because
positions are bounded, so "where it was" already predicts part of "which way it
moved"; only the unpredictable part of each jump counts as evidence.

Horizontal movement is measured from the iris offset or from head yaw (some
people turn their head instead), vertical from iris offset, head pitch or eyelid
opening (lids follow vertical gaze). Laptop webcams usually give a usable
horizontal signal and little vertical signal, so horizontal evidence alone can
pass; vertical evidence adds to it when present.

What this stops, and what it does not
-------------------------------------
* A looping or unrelated video, and a recording of a real person made during a
  *different* session, both produce displacements unrelated to this session's
  jumps (about 1 in 300 such replays reaches a passing score by chance; the
  other checkpoints still have to pass too).
* A script that moves the "eyes" at the instant the dot moves is flagged: human
  saccades start roughly 120-400 ms after the target moves.
* It does not stop software that sees the challenge and synthesises matching
  signals with human-like delay. That is what device attestation and the face
  deepfake model are for.

A trained model (``gaze.onnx``) over the same features is used when present.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..challenges import GazeChallenge
from ..schemas import GazeFrame

PRE_MS = (-200.0, 80.0)  # still the old fixation: nobody reacts within 80 ms
POST_WINDOWS_MS = {"early": (300.0, 620.0), "late": (450.0, 760.0)}  # late suits slow responders
MIN_EVENTS = 8
IX, IY, YAW, PITCH, EAR = range(5)

GAZE_FEATURE_NAMES = [
    "evidence", "r_x", "r_y", "n_events", "head_led", "late_window", "latency_med",
    "latency_n", "blink_rate_per_min", "head_micro_std", "face_ratio", "fps",
]


class GazeInputError(ValueError):
    pass


@dataclass
class GazeFeatures:
    evidence: float = 0.0        # combined z-score that the displacements follow this path
    r_x: float = 0.0             # horizontal partial correlation (best of iris, head yaw)
    r_y: float = 0.0             # vertical partial correlation (best of iris, head pitch, eyelids)
    n_events: float = 0.0        # jumps with enough frames before and after
    head_led: float = 0.0        # 1 if head yaw, not the iris, carried the horizontal signal
    late_window: float = 0.0     # 1 if the later post-jump window fitted better
    latency_med: float = -1.0    # median ms from jump to eye movement; -1 = not measurable
    latency_n: float = 0.0
    blink_rate_per_min: float = 0.0
    head_micro_std: float = 0.0
    face_ratio: float = 0.0
    fps: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, n) for n in GAZE_FEATURE_NAMES], dtype=np.float32)


def _partial_r(d: np.ndarray, cur: np.ndarray, prev: np.ndarray) -> float:
    """corr(d, cur | prev)."""
    A = np.column_stack([prev, np.ones(len(prev))])
    rd = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
    rc = cur - A @ np.linalg.lstsq(A, cur, rcond=None)[0]
    if rd.std() < 1e-9 or rc.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rd, rc)[0, 1])


def _jumps(ch: GazeChallenge) -> list[tuple[float, float, tuple[float, float], tuple[float, float]]]:
    """(time, time of next event, previous position, new position) for each jump."""
    out = []
    kfs = ch.keyframes
    pos = (kfs[0].x1, kfs[0].y1)
    for i, kf in enumerate(kfs[1:], start=1):
        nxt = kfs[i + 1].start if i + 1 < len(kfs) else ch.duration_ms
        if kf.mode == "jump":
            out.append((float(kf.start), float(nxt), pos, (kf.x1, kf.y1)))
        pos = (kf.x1, kf.y1)
    return out


def _events(t: np.ndarray, sig: np.ndarray, jumps, post: tuple[float, float]):
    rows, pre_levels, cur, prev, times = [], [], [], [], []
    for start, nxt, p0, p1 in jumps:
        a = (t >= start + PRE_MS[0]) & (t <= start + PRE_MS[1])
        b = (t >= start + post[0]) & (t <= min(nxt + PRE_MS[1], start + post[1]))
        if a.sum() < 2 or b.sum() < 2:
            continue
        pre = np.median(sig[a], axis=0)
        rows.append(np.median(sig[b], axis=0) - pre)
        pre_levels.append(pre)
        cur.append(p1)
        prev.append(p0)
        times.append(start)
    return np.array(rows), np.array(pre_levels), np.array(cur), np.array(prev), np.array(times)


def _median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x
    stacked = np.stack([np.r_[x[0], x[:-1]], x, np.r_[x[1:], x[-1]]])
    return np.median(stacked, axis=0)


def _latencies(t: np.ndarray, s: np.ndarray, times: np.ndarray, pre: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Per jump: ms until the signal has moved halfway to its new level (two frames running)."""
    s = _median3(s)
    typical = np.median(np.abs(delta)) + 1e-12
    out = []
    for t0, level, d in zip(times, pre, delta):
        if abs(d) < 0.6 * typical:
            continue  # the dot barely moved along this axis; nothing to time
        m = (t > t0 - 50) & (t <= t0 + 800)
        moved = (s[m] - level) * np.sign(d) >= 0.5 * abs(d)
        tm = t[m]
        for j in range(len(moved)):
            if moved[j] and (j + 1 >= len(moved) or moved[j + 1]):
                out.append(tm[j] - t0)
                break
    return np.asarray(out)


def extract(ch: GazeChallenge, frames: list[GazeFrame]) -> GazeFeatures:
    fr = sorted(frames, key=lambda f: f.t)
    t = np.array([f.t for f in fr])
    if np.any(np.diff(t) <= 0):
        raise GazeInputError("Frame timestamps must be strictly increasing")
    fps = 1000.0 / np.median(np.diff(t))
    if not (10 <= fps <= 125):
        raise GazeInputError("Implausible frame rate")
    if t[0] > 400 or t[-1] < 0.9 * ch.duration_ms:
        raise GazeInputError("Frames do not cover the challenge")

    face = np.array([f.face for f in fr])
    out = GazeFeatures(face_ratio=float(face.mean()), fps=float(fps))
    sig = np.array([[f.ix, f.iy, f.yaw, f.pitch, f.ear] for f in fr])[face]
    tf = t[face]
    if len(tf) < 20:
        return out

    jumps = _jumps(ch)
    best = None
    for name, post in POST_WINDOWS_MS.items():
        d, pre, cur, prev, times = _events(tf, sig, jumps, post)
        n = len(d)
        out.n_events = max(out.n_events, float(n))
        if n < MIN_EVENTS:
            continue
        rx = {c: abs(_partial_r(d[:, c], cur[:, 0], prev[:, 0])) for c in (IX, YAW)}
        ry = {c: abs(_partial_r(d[:, c], cur[:, 1], prev[:, 1])) for c in (IY, PITCH, EAR)}
        cx = max(rx, key=rx.get)
        r_x, r_y = rx[cx], max(ry.values())
        scale = np.sqrt(n - 4)  # Fisher z: roughly standard normal when there is no relationship
        zx, zy = np.arctanh(min(r_x, 0.999)) * scale, np.arctanh(min(r_y, 0.999)) * scale
        # Horizontal evidence alone, or both axes together (minus the allowance a
        # second, possibly empty, axis has to pay), whichever is stronger.
        evidence = float(max(zx, np.hypot(zx, zy) - 0.45))
        if best is None or evidence > best[0]:
            best = (evidence, r_x, r_y, n, cx, name, d, pre, times)
    if best is not None:
        evidence, r_x, r_y, n, cx, name, d, pre, times = best
        out.evidence, out.r_x, out.r_y, out.n_events = evidence, float(r_x), float(r_y), float(n)
        out.head_led, out.late_window = float(cx == YAW), float(name == "late")
        lat = _latencies(tf, sig[:, cx], times, pre[:, cx], d[:, cx])
        out.latency_n = float(len(lat))
        if len(lat) >= 4:
            out.latency_med = float(np.median(lat))

    ear = sig[:, EAR]
    closed = ear < 0.6 * (float(np.median(ear)) + 1e-6)
    out.blink_rate_per_min = float(np.sum(closed[1:] & ~closed[:-1]) / (ch.duration_ms / 60000.0))

    head = sig[:, [YAW, PITCH]]
    if len(head) > 7:
        k = np.ones(7) / 7
        smooth = np.stack([np.convolve(head[:, i], k, mode="same") for i in (0, 1)], axis=1)
        out.head_micro_std = float(np.std((head - smooth)[3:-3]))
    return out


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def heuristic_score(f: GazeFeatures) -> tuple[float, list[str]]:
    reasons = []
    face = 1.0 if f.face_ratio >= 0.85 else 0.1
    if face < 1:
        reasons.append("face not continuously visible")
    if f.n_events < MIN_EVENTS:
        reasons.append("not enough camera frames to measure eye movement")
    # Measured on simulated replays (docs/MODELS.md): about 1 in 100 unrelated
    # recordings reaches evidence 3.4 (score 0.25) and about 1 in 300 reaches 3.75
    # (score 0.5). Real webcam users typically land above 4.5.
    track = _clip01((f.evidence - 3.0) / 1.5)
    if track < 0.3 and f.n_events >= MIN_EVENTS:
        reasons.append("eyes did not follow the dot")
    timing = 1.0
    if f.latency_med >= 0 and not (80 <= f.latency_med <= 650):
        timing = 0.2  # below the checkpoint floor on its own
        reasons.append("eye response timing not human")
    micro = 1.0 if f.head_micro_std > 0.01 else 0.4
    if micro < 1:
        reasons.append("no natural head micro-movement")
    return _clip01(track * timing * micro * face), reasons


def score(ch: GazeChallenge, frames: list[GazeFrame], model) -> tuple[float, list[str], dict]:
    try:
        feats = extract(ch, frames)
    except GazeInputError as exc:
        return 0.0, [str(exc)], {}
    h, reasons = heuristic_score(feats)
    return combine(h, feats, model), reasons, asdict(feats)


def combine(h: float, feats: GazeFeatures, model) -> float:
    """Final score from the rule-based score and, when present, the trained model."""
    if model is None:
        return h
    p = float(model.run(feats.vector()[None, :])[1][0][1])
    return min(h, p) if h < 0.3 else 0.5 * h + 0.5 * p  # the model cannot override a clear heuristic fail
