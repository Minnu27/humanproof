"""Checkpoint 1: face presence + eye movement following a random target.

The client sends per-frame signals derived from MediaPipe face/iris landmarks
(never video). We test whether the eyes (plus head) *follow the secret random
path with human timing*:

* cross-validated regression of target *movements* from eye/head *movements*
  (100 ms differences, over a search of lags). Using differences matters:
  slowly drifting signals (a replayed video) correlate spuriously with any path
  when compared as levels, but their increments do not line up with the jumps
  of a secret random path. A pre-recorded or unrelated face scores ~0;
* saccade latency after each jump (humans: roughly 120-450 ms);
* natural head micro-motion (a perfectly static face is suspicious);
* face present in nearly every frame.

A trained model (``gaze.onnx``) over the same features is used when present.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..challenges import GazeChallenge
from ..schemas import GazeFrame

GRID_MS = 33.3
LAGS_MS = np.arange(0, 700, 33.3)
DIFF_STEPS = 3  # 100 ms

GAZE_FEATURE_NAMES = [
    "r2_pooled", "r2_x", "r2_y", "best_lag_ms", "saccade_ok_frac", "saccade_latency_med",
    "blink_rate_per_min", "head_micro_std", "face_ratio", "fps", "pursuit_gain",
]


class GazeInputError(ValueError):
    pass


@dataclass
class GazeFeatures:
    r2_pooled: float
    r2_x: float
    r2_y: float
    best_lag_ms: float
    saccade_ok_frac: float
    saccade_latency_med: float
    blink_rate_per_min: float
    head_micro_std: float
    face_ratio: float
    fps: float
    pursuit_gain: float

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, n) for n in GAZE_FEATURE_NAMES], dtype=np.float32)


def _cv_residuals(X: np.ndarray, y: np.ndarray, blocks: np.ndarray) -> tuple[float, float] | None:
    """Two-fold cross-validated (SS_res, SS_tot) using alternating 1 s blocks."""
    preds = np.empty_like(y)
    for fold in (0, 1):
        train, test = blocks % 2 != fold, blocks % 2 == fold
        if train.sum() < 8 or test.sum() < 8:
            return None
        coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        preds[test] = X[test] @ coef
    return float(((y - preds) ** 2).sum()), float(((y - y.mean()) ** 2).sum()) + 1e-9


def _cv_r2(X: np.ndarray, y: np.ndarray, blocks: np.ndarray) -> float:
    r = _cv_residuals(X, y, blocks)
    return 0.0 if r is None else max(-1.0, 1.0 - r[0] / r[1])


def _pooled_cv_r2(X: np.ndarray, Y: np.ndarray, blocks: np.ndarray) -> float:
    """R^2 pooled over both axes, so each axis counts in proportion to how much the
    target actually moved along it (a mostly vertical path is judged mostly on y)."""
    rx, ry = _cv_residuals(X, Y[:, 0], blocks), _cv_residuals(X, Y[:, 1], blocks)
    if rx is None or ry is None:
        return 0.0
    return max(-1.0, 1.0 - (rx[0] + ry[0]) / (rx[1] + ry[1]))


def extract(ch: GazeChallenge, frames: list[GazeFrame]) -> GazeFeatures:
    fr = sorted(frames, key=lambda f: f.t)
    t = np.array([f.t for f in fr])
    if np.any(np.diff(t) <= 0):
        raise GazeInputError("Frame timestamps must be strictly increasing")
    dt = np.median(np.diff(t))
    fps = 1000.0 / dt
    if not (10 <= fps <= 125):
        raise GazeInputError("Implausible frame rate")
    if t[0] > 400 or t[-1] < 0.9 * ch.duration_ms:
        raise GazeInputError("Frames do not cover the challenge")

    face = np.array([f.face for f in fr])
    face_ratio = float(face.mean())
    sig = np.array([[f.ix, f.iy, f.yaw, f.pitch] for f in fr])[face]
    ear = np.array([f.ear for f in fr])[face]
    tf = t[face]
    if len(tf) < 20:
        return GazeFeatures(0, 0, 0, 0, 0, 0, 0, 0, face_ratio, fps, 0)

    grid = np.arange(0, ch.duration_ms, GRID_MS)
    S = np.stack([np.interp(grid, tf, sig[:, k]) for k in range(4)], axis=1)
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-6)
    X = np.column_stack([S, np.ones(len(grid))])
    k = DIFF_STEPS
    dS = S[k:] - S[:-k]
    blocks = (grid[k:] // 1000).astype(int)

    best = (-1.0, 0.0)
    for lag in LAGS_MS:
        tgt = np.array([ch.target_at(max(0.0, g - lag)) for g in grid])
        pooled = _pooled_cv_r2(dS, tgt[k:] - tgt[:-k], blocks)
        if pooled > best[0]:
            best = (pooled, lag)
    r2_pooled, lag = best
    tgt = np.array([ch.target_at(max(0.0, g - lag)) for g in grid])
    dT = tgt[k:] - tgt[:-k]
    r2x, r2y = _cv_r2(dS, dT[:, 0], blocks), _cv_r2(dS, dT[:, 1], blocks)

    # Gaze estimate in target units (fit at best lag on everything), used for timing checks.
    coef_x, *_ = np.linalg.lstsq(X, tgt[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(X, tgt[:, 1], rcond=None)
    gx, gy = X @ coef_x, X @ coef_y

    latencies = []
    for kf_prev, kf in zip(ch.keyframes, ch.keyframes[1:]):
        if kf.mode != "jump":
            continue
        start_xy = np.array(ch.target_at(kf.start - 1))
        end_xy = np.array([kf.x1, kf.y1])
        span = end_xy - start_xy
        norm = float(span @ span) + 1e-9
        window = (grid >= kf.start) & (grid <= kf.start + 800)
        if window.sum() < 3:
            continue
        progress = ((np.column_stack([gx, gy])[window] - start_xy) @ span) / norm
        crossed = np.where(progress >= 0.5)[0]
        latencies.append(float(grid[window][crossed[0]] - kf.start) if len(crossed) else 9999.0)
    lat = np.array(latencies) if latencies else np.array([9999.0])
    sacc_ok = float(np.mean((lat >= 90) & (lat <= 500)))

    pursuit_gain = 0.0
    for kf in ch.keyframes:
        if kf.mode == "glide":
            m = (grid >= kf.start + 200) & (grid <= kf.end)
            if m.sum() > 5:
                tv = np.gradient(np.array([ch.target_at(g) for g in grid[m]]), axis=0)
                ev = np.gradient(np.column_stack([gx[m], gy[m]]), axis=0)
                denom = float((tv ** 2).sum()) + 1e-9
                pursuit_gain = float((tv * ev).sum() / denom)

    med_ear = float(np.median(ear)) + 1e-6
    closed = ear < 0.6 * med_ear
    blinks = int(np.sum(closed[1:] & ~closed[:-1]))
    blink_rate = blinks / (ch.duration_ms / 60000.0)

    head = sig[:, 2:4]
    if len(head) > 7:
        k = np.ones(7) / 7
        smooth = np.stack([np.convolve(head[:, i], k, mode="same") for i in (0, 1)], axis=1)
        micro = float(np.std((head - smooth)[3:-3]))
    else:
        micro = 0.0

    return GazeFeatures(
        r2_pooled=float(r2_pooled), r2_x=float(r2x), r2_y=float(r2y), best_lag_ms=float(lag), saccade_ok_frac=sacc_ok,
        saccade_latency_med=float(np.median(lat)), blink_rate_per_min=float(blink_rate),
        head_micro_std=micro, face_ratio=face_ratio, fps=float(fps), pursuit_gain=pursuit_gain,
    )


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def heuristic_score(f: GazeFeatures) -> tuple[float, list[str]]:
    reasons = []
    if f.face_ratio < 0.85:
        reasons.append("face not continuously visible")
    track = _clip01((f.r2_pooled - 0.12) / 0.25)
    if track < 0.3:
        reasons.append("eyes did not follow the target")
    lag_ok = 1.0 if 60 <= f.best_lag_ms <= 500 else 0.25
    if lag_ok < 1:
        reasons.append("eye response timing not human")
    sacc = 0.35 + 0.65 * _clip01(f.saccade_ok_frac / 0.6)
    micro = 1.0 if f.head_micro_std > 0.01 else 0.4
    if micro < 1:
        reasons.append("no natural head micro-movement")
    face = 1.0 if f.face_ratio >= 0.85 else 0.1
    return _clip01(np.sqrt(track) * lag_ok * sacc * micro * face), reasons


def score(ch: GazeChallenge, frames: list[GazeFrame], model) -> tuple[float, list[str], dict]:
    try:
        feats = extract(ch, frames)
    except GazeInputError as exc:
        return 0.0, [str(exc)], {}
    h, reasons = heuristic_score(feats)
    s = h
    if model is not None:
        p = float(model.run(feats.vector()[None, :])[1][0][1])
        s = min(h, p) if h < 0.3 else 0.5 * h + 0.5 * p  # the model cannot override a clear heuristic fail
    return s, reasons, asdict(feats)
