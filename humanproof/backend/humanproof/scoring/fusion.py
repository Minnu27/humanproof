"""Combine the checkpoint scores into one decision.

Scores are combined in log-odds space so strong evidence in one checkpoint can
lift a borderline one, but every checkpoint also has a floor: a clear failure in
any single checkpoint cannot be averaged away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

WEIGHTS = {"gaze": 1.0, "face": 0.8, "motor": 1.0, "voice": 1.3}
FLOORS = {"gaze": 0.25, "face": 0.25, "motor": 0.25, "voice": 0.25}
PASS_THRESHOLD = 0.70
STEP_UP_THRESHOLD = 0.45


@dataclass
class FusionResult:
    decision: str
    score: float
    reasons: list[str]


def _logit(p: float) -> float:
    p = min(0.995, max(0.005, p))
    return math.log(p / (1 - p))


def fuse(scores: dict[str, float], assurance: str) -> FusionResult:
    reasons = []
    for name, floor in FLOORS.items():
        if name not in scores:
            return FusionResult("reject", 0.0, [f"{name} checkpoint missing"])
        if scores[name] < floor:
            reasons.append(f"{name} checkpoint failed")
    total_w = sum(WEIGHTS.values())
    z = sum(WEIGHTS[k] * _logit(scores[k]) for k in WEIGHTS) / total_w
    combined = 1 / (1 + math.exp(-z))
    if reasons:
        return FusionResult("reject", combined, reasons)
    # Web clients cannot prove they run unmodified on a real device, so they
    # need more evidence to pass outright.
    pass_at = PASS_THRESHOLD + (0.08 if assurance == "web" else 0.0)
    if combined >= pass_at:
        return FusionResult("pass", combined, [])
    if combined >= STEP_UP_THRESHOLD:
        return FusionResult("step_up", combined, ["evidence borderline; retry required"])
    return FusionResult("reject", combined, ["overall evidence insufficient"])
