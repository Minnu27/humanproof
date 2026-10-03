"""Challenge generator.

Each checkpoint's challenge is created with the OS CSPRNG and revealed only when
the client starts that checkpoint, so nothing can be precomputed. The server
keeps the full challenge; scoring compares the response against it.

Coordinates are normalised to [0, 1] in the client's viewport; times are
milliseconds from the moment the client starts rendering the challenge.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import asdict, dataclass

_rng = secrets.SystemRandom()

# Short, common, phonetically distinct words that are easy to say in any accent.
WORDS = [
    "apple", "river", "candle", "tiger", "window", "garden", "pencil", "rocket", "silver", "orange",
    "basket", "yellow", "forest", "button", "dragon", "mirror", "pepper", "planet", "tunnel", "violin",
    "wagon", "zebra", "anchor", "bottle", "carpet", "dinner", "engine", "falcon", "guitar", "hammer",
    "island", "jacket", "kettle", "lemon", "magnet", "napkin", "oyster", "parrot", "quiet", "rabbit",
    "saddle", "turtle", "umbrella", "velvet", "walnut", "yogurt", "ladder", "marble", "noodle", "pillow",
    "cactus", "copper", "desert", "feather", "glacier", "harbor", "icicle", "jungle", "lantern", "meadow",
    "nickel", "orbit", "pebble", "puzzle", "ribbon", "sunset", "thunder", "valley", "whistle", "cookie",
    "bridge", "castle", "circle", "cotton", "crystal", "dolphin", "shadow", "summer", "winter", "spring",
    "morning", "tomato", "potato", "banana", "cherry", "coffee", "blanket", "monkey", "penguin", "ticket",
]


@dataclass(frozen=True)
class GazeKeyframe:
    """The dot moves from (x0, y0) to (x1, y1) between start and end.

    A jump (saccade target) has start == end; a glide (smooth pursuit) has end > start.
    Between keyframes the dot holds still at the last (x1, y1).
    """
    start: int  # ms
    end: int    # ms
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def mode(self) -> str:
        return "jump" if self.end == self.start else "glide"


@dataclass(frozen=True)
class GazeChallenge:
    duration_ms: int
    keyframes: list[GazeKeyframe]
    capture_ms: list[int]  # when the client captures face crops for the deepfake model

    def target_at(self, t_ms: float) -> tuple[float, float]:
        current = self.keyframes[0]
        for kf in self.keyframes:
            if kf.start <= t_ms:
                current = kf
            else:
                break
        if t_ms < current.end:
            a = (t_ms - current.start) / (current.end - current.start)
            return current.x0 + a * (current.x1 - current.x0), current.y0 + a * (current.y1 - current.y0)
        return current.x1, current.y1

    def jump_times(self) -> list[int]:
        return [k.start for k in self.keyframes[1:] if k.mode == "jump"]


@dataclass(frozen=True)
class MotorChallenge:
    duration_ms: int          # soft limit shown to the user
    control_points: list[tuple[float, float]]  # Catmull-Rom spline through these points

    def polyline(self, samples_per_segment: int = 40) -> list[tuple[float, float]]:
        pts = self.control_points
        ext = [pts[0], *pts, pts[-1]]
        out: list[tuple[float, float]] = []
        for i in range(1, len(ext) - 2):
            p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
            for s in range(samples_per_segment):
                t = s / samples_per_segment
                t2, t3 = t * t, t * t * t
                out.append(tuple(  # type: ignore[arg-type]
                    0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
                           + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
                    for k in (0, 1)
                ))
        out.append(pts[-1])
        return out


@dataclass(frozen=True)
class VoiceChallenge:
    words: list[str]
    max_duration_ms: int


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point(margin: float = 0.12) -> tuple[float, float]:
    return (round(_rng.uniform(margin, 1 - margin), 4), round(_rng.uniform(margin, 1 - margin), 4))


GAZE_JUMPS = 12


def _gaze_target(r, last: tuple[float, float]) -> tuple[float, float]:
    """Next dot position, drawn independently of where the dot has been.

    Only a minimum distance is enforced (so every jump is a visible eye movement).
    Deliberately *not* enforced: a large horizontal step on every jump. That rule
    made the dot bounce left-right-left-right, which is predictable: eye movement
    recorded during one session then correlated strongly with the path of another.
    """
    while True:
        p = (round(r.uniform(0.10, 0.90), 4), round(r.uniform(0.12, 0.88), 4))
        if _dist(p, last) >= 0.3:
            return p


def make_gaze_challenge(rng=None) -> GazeChallenge:
    """~10 s: a lead-in fixation, then 12 jumps to independent random positions.

    Scoring needs many independent events: each jump is one. ``rng`` (anything
    with the ``random.Random`` interface) is for reproducible tests only;
    production always uses the OS CSPRNG.
    """
    r = rng or _rng
    kfs = [GazeKeyframe(0, 0, 0.5, 0.5, 0.5, 0.5)]
    t = r.randint(900, 1200)
    last = (0.5, 0.5)
    for _ in range(GAZE_JUMPS):
        p = _gaze_target(r, last)
        kfs.append(GazeKeyframe(t, t, p[0], p[1], p[0], p[1]))
        last = p
        t += r.randint(600, 850)  # long enough for a slow saccade plus a measurable fixation
    duration = t
    captures = sorted(r.sample(range(800, duration - 200, 50), 4))
    return GazeChallenge(duration, kfs, captures)


def make_motor_challenge() -> MotorChallenge:
    n = _rng.randint(4, 5)
    pts = [_point(0.1)]
    while len(pts) < n:
        p = _point(0.1)
        if _dist(p, pts[-1]) > 0.25:
            pts.append(p)
    return MotorChallenge(duration_ms=12_000, control_points=pts)


def make_voice_challenge() -> VoiceChallenge:
    return VoiceChallenge(words=_rng.sample(WORDS, 5), max_duration_ms=8_000)


def make_challenge_set() -> dict:
    return {
        "gaze": asdict(make_gaze_challenge()),
        "motor": asdict(make_motor_challenge()),
        "voice": asdict(make_voice_challenge()),
    }


def gaze_from_dict(d: dict) -> GazeChallenge:
    return GazeChallenge(d["duration_ms"], [GazeKeyframe(**k) for k in d["keyframes"]], list(d["capture_ms"]))


def motor_from_dict(d: dict) -> MotorChallenge:
    return MotorChallenge(d["duration_ms"], [tuple(p) for p in d["control_points"]])


def voice_from_dict(d: dict) -> VoiceChallenge:
    return VoiceChallenge(list(d["words"]), d["max_duration_ms"])
