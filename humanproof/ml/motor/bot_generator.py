"""Synthetic automated-pointer trajectories: the negative class for the motor model.

Covers the families real bots use, from naive to "humanised":
  linear, eased linear, Bezier, WindMouse, minimum-jerk with noise,
  spline-following (what a bot tracing our challenge curve would do),
  and tremor-injected variants.

Every trajectory is emitted with realistic, jittered event timing at a random
capture rate, then passed through the same feature pipeline as human data.
"""
from __future__ import annotations

import math

import numpy as np


def _timestamps(rng: np.random.Generator, duration_ms: float) -> np.ndarray:
    rate = rng.choice([60, 60, 75, 90, 120, 125, 144])  # same range as real pointer capture
    dt = 1000.0 / rate
    n = max(8, int(duration_ms / dt))
    jitter = rng.normal(0, rng.uniform(0, 0.25) * dt, n)
    t = np.cumsum(np.full(n, dt) + jitter)
    return np.maximum.accumulate(t - t[0])


def _progress(rng: np.random.Generator, u: np.ndarray) -> np.ndarray:
    kind = rng.integers(0, 5)
    if kind == 0:
        return u
    if kind == 1:
        return 0.5 - 0.5 * np.cos(np.pi * u)
    if kind == 2:
        return u * u * (3 - 2 * u)
    if kind == 3:  # minimum-jerk profile: bell-shaped velocity, "human-looking" on paper
        return 10 * u**3 - 15 * u**4 + 6 * u**5
    a = rng.uniform(1.5, 3.5)  # ease-out power
    return 1 - (1 - u) ** a


def _endpoints(rng, lo=50, hi=1400):
    p0 = rng.uniform(lo, hi, 2)
    p1 = rng.uniform(lo, hi, 2)
    while np.hypot(*(p1 - p0)) < 150:
        p1 = rng.uniform(lo, hi, 2)
    return p0, p1


def linear(rng):
    p0, p1 = _endpoints(rng)
    t = _timestamps(rng, rng.uniform(250, 1500))
    s = _progress(rng, t / t[-1])[:, None]
    return t, p0 + s * (p1 - p0)


def bezier(rng):
    p0, p3 = _endpoints(rng)
    span = np.hypot(*(p3 - p0))
    p1 = p0 + (p3 - p0) * rng.uniform(0.1, 0.5) + rng.normal(0, 0.3 * span, 2)
    p2 = p0 + (p3 - p0) * rng.uniform(0.5, 0.9) + rng.normal(0, 0.3 * span, 2)
    t = _timestamps(rng, rng.uniform(300, 1800))
    s = _progress(rng, t / t[-1])[:, None]
    pts = (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * p3
    return t, pts


def windmouse(rng):
    """The widely copied WindMouse humanisation algorithm (gravity + wind)."""
    p, dst = _endpoints(rng)
    g, w = rng.uniform(7, 12), rng.uniform(2, 6)
    max_step, target_area = rng.uniform(8, 20), rng.uniform(6, 12)
    v = np.zeros(2)
    wind = np.zeros(2)
    pts = [p.copy()]
    s3, s5 = math.sqrt(3), math.sqrt(5)
    for _ in range(2000):
        dist = np.hypot(*(dst - p))
        if dist < 1:
            break
        wmag = min(w, dist)
        if dist >= target_area:
            wind = wind / s3 + (rng.uniform(-1, 1, 2) * (2 * wmag + 1) - wmag) / s5
        else:
            wind /= s3
            if max_step < 3:
                max_step = rng.uniform(3, 6)
            else:
                max_step /= s5
        v += wind + g * (dst - p) / dist
        vmag = np.hypot(*v)
        if vmag > max_step:
            v = v / vmag * (max_step / 2 + rng.uniform(0, max_step / 2))
        p = p + v
        pts.append(p.copy())
    pts.append(dst.copy())
    pts = np.asarray(pts)
    path_len = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    t = _timestamps(rng, 1000.0 * path_len / rng.uniform(400, 2500))
    idx = np.linspace(0, len(pts) - 1, len(t))
    return t, np.stack([np.interp(idx, np.arange(len(pts)), pts[:, k]) for k in (0, 1)], axis=1)


def spline_follow(rng):
    """A bot tracing a Catmull-Rom curve like our motor challenge."""
    n = rng.integers(4, 6)
    cps = rng.uniform(80, 1300, (n, 2))
    ext = np.vstack([cps[0], cps, cps[-1]])
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for u in np.linspace(0, 1, 60, endpoint=False):
            dense.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3))
    dense = np.asarray(dense)
    arclen = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])
    t = _timestamps(rng, rng.uniform(2500, 7000))
    s = _progress(rng, t / t[-1]) * arclen[-1]
    return t, np.stack([np.interp(s, arclen, dense[:, k]) for k in (0, 1)], axis=1)


def _add_noise(rng, t, pts):
    kind = rng.integers(0, 4)
    if kind == 0:
        return t, pts
    if kind == 1:  # white jitter
        return t, pts + rng.normal(0, rng.uniform(0.3, 2.5), pts.shape)
    if kind == 2:  # sinusoidal "tremor" at a single frequency
        f = rng.uniform(6, 12)
        amp = rng.uniform(0.5, 2.5)
        ph = rng.uniform(0, 2 * np.pi, 2)
        return t, pts + amp * np.stack([np.sin(2 * np.pi * f * t / 1000 + ph[0]),
                                         np.sin(2 * np.pi * f * t / 1000 + ph[1])], axis=1)
    # smoothed random walk drift (a stronger humanisation attempt)
    walk = np.cumsum(rng.normal(0, rng.uniform(0.2, 1.0), pts.shape), axis=0)
    k = 7
    kernel = np.ones(k) / k
    walk = np.stack([np.convolve(walk[:, i], kernel, mode="same") for i in (0, 1)], axis=1)
    return t, pts + walk - np.linspace(0, 1, len(t))[:, None] * walk[-1]


def _pink_noise(rng, n: int, lo_hz: float, hi_hz: float, rate: float) -> np.ndarray:
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    f = np.fft.rfftfreq(n, d=1 / rate)
    spec *= np.where((f >= lo_hz) & (f <= hi_hz), 1 / np.sqrt(np.maximum(f, 1e-3)), 0)
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-9)


def submovement(rng):
    """Sophisticated attacker: overlapping minimum-jerk sub-movements with overshoot
    and corrections (stochastic optimised-submovement model), band-limited pink
    tremor, and micro-pauses. Deliberately built to look human."""
    p0, p1 = _endpoints(rng)
    rate = 120.0
    pts_t, pts = [0.0], [p0.copy()]
    cur, t = p0.copy(), 0.0
    for k in range(rng.integers(2, 5)):
        remaining = p1 - cur
        if k == 0:
            aim = cur + remaining * rng.uniform(0.85, 1.12) + rng.normal(0, 0.04 * np.hypot(*remaining), 2)
        else:
            aim = p1 + rng.normal(0, 3, 2)
        dist = np.hypot(*(aim - cur))
        dur = 1000 * (0.12 + 0.1 * np.log2(1 + dist / 20)) * rng.uniform(0.8, 1.3)  # Fitts-like
        n = max(4, int(dur / 1000 * rate))
        u = np.linspace(0, 1, n)[1:]
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        seg = cur + s[:, None] * (aim - cur)
        start = t - (rng.uniform(0, 0.25) * dur if k > 0 else 0)  # overlap with previous
        seg_t = start + u * dur
        pts_t.extend(seg_t.tolist())
        pts.extend(seg)
        cur, t = aim, seg_t[-1] + rng.uniform(0, 120)
    order = np.argsort(pts_t)
    tt = np.asarray(pts_t)[order]
    pp = np.asarray(pts)[order]
    grid = np.arange(0, tt[-1], 1000 / rate)
    pos = np.stack([np.interp(grid, tt, pp[:, i]) for i in (0, 1)], axis=1)
    amp = rng.uniform(0.3, 1.5)
    trem = np.stack([_pink_noise(rng, len(grid), 6, 14, rate) for _ in (0, 1)], axis=1) * amp
    pos = pos + trem
    t_out = _timestamps(rng, grid[-1])
    return t_out, np.stack([np.interp(t_out, grid, pos[:, i]) for i in (0, 1)], axis=1)


GENERATORS = {
    "linear": linear, "bezier": bezier, "windmouse": windmouse,
    "spline_follow": spline_follow, "submovement": submovement,
}


def generate(rng: np.random.Generator, family: str | None = None):
    name = family or rng.choice(list(GENERATORS))
    t, pts = GENERATORS[name](rng)
    t, pts = _add_noise(rng, t, pts)
    return name, t, pts[:, 0], pts[:, 1]
