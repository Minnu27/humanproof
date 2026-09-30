"""Checkpoint 2: motor dynamics while tracing a random curve.

Two independent questions:
1. Did the pointer actually trace *this* secret curve (coverage, deviation,
   start/end)? Replays of recorded human movement fail here.
2. Does the movement have human dynamics (sub-movements, tremor, velocity
   profile)? Scripted and "humanised" bot paths fail here. Answered by the
   trained motor model (``motor.onnx``) over features shared with training.
"""
from __future__ import annotations

import numpy as np

from ..challenges import MotorChallenge
from ..schemas import MotorSubmission
from .features_motor import trajectory_features


def _polyline_px(ch: MotorChallenge, w: int, h: int) -> np.ndarray:
    return np.array([(x * w, y * h) for x, y in ch.polyline()])


def _nearest_dist(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest *segment* of the polyline."""
    a, b = line[:-1], line[1:]
    ab = b - a
    denom = (ab ** 2).sum(axis=1) + 1e-12
    ap = points[:, None, :] - a[None, :, :]
    u = np.clip((ap * ab[None]).sum(axis=2) / denom[None], 0, 1)
    proj = a[None] + u[..., None] * ab[None]
    return np.sqrt(((points[:, None, :] - proj) ** 2).sum(axis=2)).min(axis=1)


def adherence(ch: MotorChallenge, sub: MotorSubmission) -> tuple[float, dict, list[str]]:
    w, h = sub.viewport.w, sub.viewport.h
    scale = min(w, h)
    line = _polyline_px(ch, w, h)
    pts = np.array([(s.x, s.y) for s in sub.samples])
    t = np.array([s.t for s in sub.samples])
    reasons = []
    if np.any(np.diff(t) < 0):
        return 0.0, {}, ["timestamps not ordered"]
    dev = _nearest_dist(pts, line) / scale
    covered = _nearest_dist(line, pts) / scale < 0.05
    coverage = float(covered.mean())
    start_ok = float(np.hypot(*(pts[0] - line[0])) / scale) < 0.08
    end_ok = float(np.hypot(*(pts[-1] - line[-1])) / scale) < 0.08
    duration = float(t[-1] - t[0])
    path_px = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    speed = path_px / max(1.0, duration) * 1000.0
    info = {
        "coverage": coverage, "mean_dev": float(dev.mean()), "p95_dev": float(np.percentile(dev, 95)),
        "duration_ms": duration, "speed_px_s": speed, "start_ok": start_ok, "end_ok": end_ok,
    }
    s = 1.0
    if coverage < 0.85:
        reasons.append("curve not fully traced")
        s *= max(0.0, (coverage - 0.5) / 0.35)
    if info["p95_dev"] > 0.08:
        reasons.append("strayed from the curve")
        s *= 0.4
    if not (start_ok and end_ok):
        reasons.append("did not start and end on the curve")
        s *= 0.3
    if info["mean_dev"] < 0.0015 and sub.pointer_type != "keyboard":
        reasons.append("trace is machine-perfect")
        s *= 0.1
    if speed > 4000 or duration < 600:
        reasons.append("traced impossibly fast")
        s *= 0.1
    return float(s), info, reasons


def _calibrate(p: float, thr: float) -> float:
    """Map model probability to a score with 0.5 at the operating threshold."""
    if p >= thr:
        return 0.5 + 0.5 * (p - thr) / max(1e-6, 1 - thr)
    return 0.5 * p / max(1e-6, thr)


def dynamics(sub: MotorSubmission, model) -> tuple[float, dict, list[str]]:
    reasons = []
    t = [s.t for s in sub.samples]
    x = [s.x for s in sub.samples]
    y = [s.y for s in sub.samples]
    if sub.pointer_type == "keyboard":
        # Arrow-key tracing (accessibility path): judge key timing irregularity.
        dt = np.diff(np.asarray(t))
        dt = dt[dt > 0]
        cv = float(dt.std() / (dt.mean() + 1e-9)) if len(dt) > 5 else 0.0
        s = 0.8 if 0.15 < cv < 3 else 0.2
        if s < 0.5:
            reasons.append("key timing too regular")
        return s, {"key_interval_cv": cv}, reasons

    feats = trajectory_features(t, x, y)
    if len(feats) == 0:
        return 0.0, {}, ["not enough movement to analyse"]
    if model is None:
        return 0.0, {"model": "missing"}, ["motor model not available"]
    probs = model.run(feats.astype(np.float32))[1][:, 1]
    thr = float(model.meta.get("threshold", 0.5))
    p = float(np.mean(probs))
    s = _calibrate(p, thr)
    info = {"model_p_human": p, "threshold": thr, "windows": int(len(feats))}
    if sub.pointer_type == "touch":
        # The model was trained on mouse data; for touch it is one signal among
        # several, and real-device motion must be present.
        s = 0.5 * s + 0.25
        if sub.device_motion:
            acc = np.array([[m.ax, m.ay, m.az] for m in sub.device_motion])
            motion_std = float(acc.std(axis=0).mean())
            info["device_motion_std"] = motion_std
            if motion_std < 1e-4:
                reasons.append("device perfectly still (emulator?)")
                s *= 0.3
    if s < 0.5:
        reasons.append("movement dynamics look automated")
    return float(s), info, reasons


def score(ch: MotorChallenge, sub: MotorSubmission, model) -> tuple[float, list[str], dict]:
    a, ainfo, r1 = adherence(ch, sub)
    d, dinfo, r2 = dynamics(sub, model)
    return float(min(a, 1.0) ** 0.5 * d if a > 0 else 0.0), r1 + r2, {**ainfo, **dinfo}
