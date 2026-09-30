"""Motor-dynamics feature extraction, shared by training (ml/motor) and serving.

Keeping one implementation for both prevents train/serve skew.

Pipeline: raw pointer samples (t ms, x px, y px)
  -> canonical capture: timestamps quantised to a 16 ms clock and positions to
     whole pixels (every source - browsers, phones, OS hooks, recorded datasets -
     is reduced to the same capture characteristics, so the model cannot learn
     "which device/dataset recorded this" instead of "was this a human")
  -> resampled to a fixed 60 Hz grid
  -> fixed windows of 48 samples (0.8 s) with 50% overlap -> one feature vector each.
"""
from __future__ import annotations

import numpy as np

QUANT_MS = 16.0
RATE_HZ = 60.0
DT_MS = 1000.0 / RATE_HZ
WINDOW = 48
HOP = 24
MIN_WINDOW_PATH_PX = 100.0  # purposeful movement only (idle jitter is not informative)

FEATURE_NAMES = [
    "speed_mean", "speed_cv", "speed_max_ratio", "speed_skew",
    "accel_std_norm", "jerk_std_norm", "jerk_abs_mean_norm",
    "angle_abs_mean", "angle_std", "dir_change_frac", "angvel_sign_changes",
    "straightness", "tremor_rms_norm", "tremor_hf_ratio",
    "peak_pos", "n_speed_peaks", "pause_frac",
    "step_size_entropy", "speed_spectral_flatness",
    "curvature_p90", "accel_sign_changes",
]
N_FEATURES = len(FEATURE_NAMES)


def resample(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return an (n, 2) array of positions on a uniform 60 Hz grid."""
    order = np.argsort(t, kind="stable")
    t, x, y = np.floor(t[order] / QUANT_MS) * QUANT_MS, np.rint(x[order]), np.rint(y[order])
    # drop duplicate timestamps (keep last position)
    keep = np.append(np.diff(t) > 0, True)
    t, x, y = t[keep], x[keep], y[keep]
    if len(t) < 2:
        return np.empty((0, 2))
    grid = np.arange(t[0], t[-1], DT_MS)
    return np.stack([np.interp(grid, t, x), np.interp(grid, t, y)], axis=1)


def windows(pos: np.ndarray) -> list[np.ndarray]:
    out = []
    for start in range(0, max(0, len(pos) - WINDOW + 1), HOP):
        w = pos[start:start + WINDOW]
        if np.sum(np.hypot(*np.diff(w, axis=0).T)) >= MIN_WINDOW_PATH_PX:
            out.append(w)
    return out


def _skew(v: np.ndarray) -> float:
    s = v.std()
    return float(((v - v.mean()) ** 3).mean() / (s ** 3)) if s > 1e-9 else 0.0


def _sign_changes(v: np.ndarray, eps: float) -> float:
    s = np.sign(np.where(np.abs(v) < eps, 0, v))
    s = s[s != 0]
    return float(np.sum(s[1:] != s[:-1])) / max(1, len(v))


def _moving_average(a: np.ndarray, k: int = 5) -> np.ndarray:
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    return np.stack([np.convolve(ap[:, i], kernel, mode="valid") for i in range(a.shape[1])], axis=1)


def window_features(w: np.ndarray) -> np.ndarray:
    d = np.diff(w, axis=0)                      # px per step
    step = np.hypot(d[:, 0], d[:, 1])
    path = float(step.sum()) + 1e-9
    speed = step / (DT_MS / 1000.0)             # px/s
    mean_speed = float(speed.mean()) + 1e-9
    acc = np.diff(speed)
    jerk = np.diff(acc)

    heading = np.arctan2(d[:, 1], d[:, 0])
    moving = step > 0.5
    dtheta = np.angle(np.exp(1j * np.diff(heading)))
    dtheta = dtheta[moving[1:] & moving[:-1]] if moving.sum() > 2 else np.zeros(1)
    if dtheta.size == 0:
        dtheta = np.zeros(1)

    chord = float(np.hypot(*(w[-1] - w[0])))
    smooth = _moving_average(w, 5)
    resid = w - smooth
    tremor = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    smooth2 = _moving_average(w, 11)
    resid2 = w - smooth2
    hf_ratio = tremor / (float(np.sqrt((resid2 ** 2).sum(axis=1).mean())) + 1e-9)

    peak_pos = float(np.argmax(speed)) / max(1, len(speed) - 1)
    ds = np.diff(speed)
    n_peaks = float(np.sum((ds[:-1] > 0) & (ds[1:] <= 0) & (speed[1:-1] > 0.3 * speed.max())))

    hist, _ = np.histogram(np.clip(step, 0, 40), bins=20, range=(0, 40))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    spec = np.abs(np.fft.rfft(speed - speed.mean())) ** 2 + 1e-12
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

    curvature = np.abs(dtheta) / (step[1:][: len(dtheta)] + 1e-3) if len(dtheta) > 1 else np.zeros(1)

    feats = [
        np.log1p(mean_speed),
        float(speed.std() / mean_speed),
        float(speed.max() / mean_speed),
        _skew(speed),
        float(acc.std() / mean_speed),
        float(jerk.std() / mean_speed),
        float(np.abs(jerk).mean() / mean_speed),
        float(np.abs(dtheta).mean()),
        float(dtheta.std()),
        float(np.mean(np.abs(dtheta) > 0.3)),
        _sign_changes(dtheta, 1e-3),
        chord / path,
        tremor / (path / len(step)),
        hf_ratio,
        peak_pos,
        n_peaks,
        float(np.mean(speed < 0.1 * mean_speed)),
        entropy,
        flatness,
        float(np.percentile(curvature, 90)),
        _sign_changes(acc, 1e-3 * mean_speed),
    ]
    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def trajectory_features(t_ms, x_px, y_px) -> np.ndarray:
    """(n_windows, N_FEATURES) for one continuous trajectory."""
    pos = resample(np.asarray(t_ms, float), np.asarray(x_px, float), np.asarray(y_px, float))
    ws = windows(pos)
    if not ws:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    return np.stack([window_features(w) for w in ws])
