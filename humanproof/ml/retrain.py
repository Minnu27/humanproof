"""Retrain from collected sessions; change a model only when it is measurably better.

Input is a dataset written by ``ml/dataset/export.py``. Only sessions that a tester
labelled (real human / specific attack) are used for training and for judging.

    python ml/retrain.py --data /data/hp-export --iitkgp /data/Mouse-Dynamics \
        --models backend/models --report report.md

What it does
------------
* **Report** - how the live checks scored the labelled sessions (how many real
  people passed each check, how many attacks were blocked), and how much data
  there is for each model.
* **Eye check** - re-measures every real human recording against its own dot path
  (should pass) and against *other sessions'* dot paths (a real person's video
  replayed into the wrong session: should fail). That is real data for both
  sides. With enough of it, trains ``gaze.onnx`` and compares the check with and
  without it on participants the model never saw.
* **Movement check** - retrains ``motor.onnx`` on the public mouse recordings plus
  the collected tracing sessions (and collected bot sessions), and compares it
  with the current model on held-out participants, the public test users and the
  synthetic bot families.
* **Voice and face** - these are deep networks that need a GPU and the public
  anti-spoofing datasets; this script only reports whether enough has been
  collected. They are trained with ``ml/colab_train.ipynb``.

A candidate replaces the current model only if it beats it on held-out data by a
margin and makes nothing clearly worse. Otherwise the models directory is left
untouched, so "the files changed" means "there is something worth reviewing".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from humanproof import challenges as chmod  # noqa: E402
from humanproof.models import ModelRegistry  # noqa: E402
from humanproof.schemas import GazeFrame  # noqa: E402
from humanproof.scoring import gaze as gz  # noqa: E402
from humanproof.scoring.features_motor import FEATURE_NAMES, trajectory_features  # noqa: E402
from humanproof.scoring.motor import _calibrate  # noqa: E402
from common.manifest import update_manifest  # noqa: E402

CHECKPOINT_NAMES = {"gaze": "Eyes", "face": "Face", "motor": "Movement", "voice": "Voice"}

# How much labelled data a model needs before a candidate is even trained.
MIN_HUMAN_SESSIONS = {"gaze": 40, "motor": 30}
MIN_PARTICIPANTS = 5
MIN_TEST_SESSIONS = 8  # held-out real-human sessions needed to judge a candidate
MIN_TEST_ATTACKS = 5   # fewer than this and the attack-block rate is too noisy to count

# A candidate must be better by this much (in balanced accuracy over the things we
# measure), and may not make any single measure worse by more than MAX_DROP.
MIN_GAIN = {"gaze": 0.01, "motor": 0.005}
MAX_DROP = 0.02

# Rough guidance for the GPU-trained models (fine-tuning on top of public datasets).
MEDIA_GUIDE = {"voice": 200, "face": 200}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(v: float | None) -> str:
    return "n/a" if v is None or np.isnan(v) else f"{100 * v:.1f}%"


def rate(flags) -> float | None:
    flags = list(flags)
    return float(np.mean(flags)) if flags else None


# ---------------------------------------------------------------------------
# How the live system did on labelled sessions
# ---------------------------------------------------------------------------

def live_performance(data: dict[str, list[dict]]) -> list[dict]:
    out = []
    for cp, rows in data.items():
        humans = [r for r in rows if r["label"] == "human" and r["score"] is not None]
        attacks = [r for r in rows if r["label"] == "attack" and r["score"] is not None]
        by_type: dict[str, list[bool]] = defaultdict(list)
        for r in attacks:
            by_type[r["attack_type"] or "other"].append(r["score"] < 0.5)
        out.append({
            "checkpoint": cp,
            "human_sessions": len(humans),
            "participants": len({r["participant"] for r in humans}),
            "human_pass_rate": rate(r["score"] >= 0.5 for r in humans),
            "attack_sessions": len(attacks),
            "attack_block_rate": rate(r["score"] < 0.5 for r in attacks),
            "attack_block_rate_by_type": {k: {"n": len(v), "blocked": float(np.mean(v))} for k, v in sorted(by_type.items())},
            "unlabelled_sessions": sum(1 for r in rows if r["label"] == "unlabelled"),
        })
    return out


def enough(cp: str, rows: list[dict]) -> tuple[bool, str]:
    humans = [r for r in rows if r["label"] == "human"]
    people = {r["participant"] for r in humans}
    test = [r for r in humans if r["split"] == "test"]
    need = MIN_HUMAN_SESSIONS[cp]
    if len(humans) < need or len(people) < MIN_PARTICIPANTS:
        return False, (f"needs at least {need} real-human sessions from {MIN_PARTICIPANTS} different people; "
                       f"has {len(humans)} from {len(people)}")
    if len(test) < MIN_TEST_SESSIONS:
        return False, f"needs at least {MIN_TEST_SESSIONS} held-out sessions to judge a new model; has {len(test)}"
    return True, ""


def decide(name: str, current: dict, candidate: dict) -> tuple[bool, str]:
    """Promote only on a clear overall gain with no single measure clearly worse.

    Compared over the measures both sides have (a rate with too few sessions
    behind it is None and is left out for both).
    """
    keys = [k for k in candidate if k != "balanced" and candidate[k] is not None and current.get(k) is not None]
    if not keys:
        return False, "not adopted: nothing to compare on"
    worse = [k for k in keys if candidate[k] < current[k] - MAX_DROP]
    cur, cand = (float(np.mean([m[k] for k in keys])) for m in (current, candidate))
    if worse:
        return False, f"not adopted: worse on {', '.join(w.replace('_', ' ') for w in worse)}"
    if cand - cur < MIN_GAIN[name]:
        return False, f"not adopted: no clear improvement ({cand - cur:+.3f} balanced accuracy; needs {MIN_GAIN[name]:+.3f})"
    return True, f"adopted: balanced accuracy {cur:.3f} -> {cand:.3f}"


def balanced(m: dict) -> dict:
    vals = [v for k, v in m.items() if v is not None]
    return {**m, "balanced": float(np.mean(vals)) if vals else float("nan")}


def to_onnx_checked(model, probe: np.ndarray):
    """Convert a scikit-learn classifier to ONNX and confirm both give the same answers."""
    import onnxruntime as ort
    from skl2onnx import to_onnx

    probe = probe[:500].astype(np.float32)
    onx = to_onnx(model, probe[:1], options={id(model): {"zipmap": False}}, target_opset=17)
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: probe})[1][:, 1]
    diff = float(np.abs(got - model.predict_proba(probe)[:, 1]).max())
    if diff >= 1e-4:
        raise RuntimeError(f"ONNX parity failed: {diff}")
    return onx.SerializeToString(), diff


class SklearnAsServed:
    """Gives a scikit-learn model the interface the server's ONNX models have."""

    def __init__(self, model, meta: dict | None = None):
        self.model, self.meta = model, meta or {}

    def run(self, x: np.ndarray):
        return [None, self.model.predict_proba(np.asarray(x, dtype=np.float32))]


# ---------------------------------------------------------------------------
# Eye check
# ---------------------------------------------------------------------------

def gaze_items(rows: list[dict], rng: np.random.Generator, pairs_per_session: int = 4) -> list[dict]:
    """Genuine, cross-session replay and labelled-attack examples, with features.

    ``feats`` is None when the recording cannot even be measured against that
    challenge (for example it is too short); the server rejects those outright.
    """
    humans = [r for r in rows if r["label"] == "human"]
    parsed = {}
    for r in rows:
        if r["label"] in ("human", "attack"):
            parsed[r["capture_id"]] = (chmod.gaze_from_dict(r["data"]["challenge"]),
                                       [GazeFrame.model_construct(**f) for f in r["data"]["frames"]])

    def measure(frames, ch):
        try:
            return gz.extract(ch, frames)
        except gz.GazeInputError:
            return None

    items = []
    for r in rows:
        if r["label"] not in ("human", "attack"):
            continue
        ch, frames = parsed[r["capture_id"]]
        items.append({"kind": "genuine" if r["label"] == "human" else "attack", "split": r["split"],
                      "attack_type": r["attack_type"], "feats": measure(frames, ch)})
    for i, r in enumerate(humans):
        others = [j for j in range(len(humans)) if j != i]
        if not others:
            break
        for j in rng.choice(others, size=min(pairs_per_session, len(others)), replace=False):
            other_ch = parsed[humans[j]["capture_id"]][0]
            items.append({"kind": "replay", "split": r["split"], "attack_type": None,
                          "feats": measure(parsed[r["capture_id"]][1], other_ch)})
    return items


def gaze_scores(items: list[dict], model) -> np.ndarray:
    out = np.zeros(len(items))
    for i, it in enumerate(items):
        if it["feats"] is not None:
            h, _ = gz.heuristic_score(it["feats"])
            out[i] = gz.combine(h, it["feats"], model)
    return out


def gaze_metrics(items: list[dict], model) -> dict:
    s = gaze_scores(items, model)
    kinds = np.array([it["kind"] for it in items])
    attacks = s[kinds == "attack"]
    return balanced({
        "human_accept": rate(s[kinds == "genuine"] >= 0.5),
        "replay_reject": rate(s[kinds == "replay"] < 0.5),
        "attack_reject": rate(attacks < 0.5) if len(attacks) >= MIN_TEST_ATTACKS else None,
    })


def run_gaze(rows: list[dict], models_dir: Path, rng: np.random.Generator, write: bool) -> dict:
    result: dict = {"name": "gaze", "promoted": False}
    labelled = [r for r in rows if r["label"] in ("human", "attack")]
    if not any(r["label"] == "human" for r in labelled):
        result["status"] = "no labelled real-human sessions yet"
        return result
    items = gaze_items(labelled, rng)
    genuine = [it["feats"].evidence for it in items if it["kind"] == "genuine" and it["feats"] is not None]
    replay = [it["feats"].evidence for it in items if it["kind"] == "replay" and it["feats"] is not None]
    current = ModelRegistry(models_dir).get("gaze")
    result["calibration"] = {
        "note": "evidence = how strongly the eyes followed this session's dot path; the check passes at about 3.75",
        "human_sessions": sum(1 for it in items if it["kind"] == "genuine"),
        "replay_pairs": sum(1 for it in items if it["kind"] == "replay"),
        "human_evidence_p05_p25_p50": [round(float(q), 2) for q in np.quantile(genuine, [0.05, 0.25, 0.5])] if genuine else None,
        "replay_evidence_p50_p95_p99": [round(float(q), 2) for q in np.quantile(replay, [0.5, 0.95, 0.99])] if replay else None,
        "all_sessions_with_current_check": gaze_metrics(items, current),
    }
    ok, why = enough("gaze", rows)
    if not ok:
        result["status"] = f"not retrained: {why}"
        return result

    from sklearn.ensemble import GradientBoostingClassifier

    train = [it for it in items if it["split"] != "test" and it["feats"] is not None]
    test = [it for it in items if it["split"] == "test"]
    X = np.stack([it["feats"].vector() for it in train])
    y = np.array([1 if it["kind"] == "genuine" else 0 for it in train])
    w = np.where(y == 1, len(y) / (2 * max(1, y.sum())), len(y) / (2 * max(1, (1 - y).sum())))
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, subsample=0.8, min_samples_leaf=5,
                                       random_state=0).fit(X, y, sample_weight=w)
    cur_m, cand_m = gaze_metrics(test, current), gaze_metrics(test, SklearnAsServed(model))
    promote, verdict = decide("gaze", cur_m, cand_m)
    result.update(status=verdict, promoted=promote, held_out={"current": cur_m, "candidate": cand_m},
                  train_examples={"human": int(y.sum()), "replay_or_attack": int(len(y) - y.sum())},
                  current_model="gaze.onnx" if current else "rules only (no model)")
    if promote and write:
        blob, diff = to_onnx_checked(model, X)
        path = models_dir / "gaze.onnx"
        path.write_bytes(blob)
        metrics = {"held_out": result["held_out"], "train_examples": result["train_examples"],
                   "onnx_parity_max_abs_diff": diff,
                   "data": "labelled tester sessions collected by the app; replays are real recordings "
                           "measured against other sessions' dot paths"}
        (models_dir / "gaze.metrics.json").write_text(json.dumps(metrics, indent=2))
        update_manifest(models_dir, "gaze", path, {"human_class": 1, "feature_names": gz.GAZE_FEATURE_NAMES})
    return result


# ---------------------------------------------------------------------------
# Movement check
# ---------------------------------------------------------------------------

def motor_sessions(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        sub = r["data"]["submission"]
        if r["label"] not in ("human", "attack") or sub["pointer_type"] == "keyboard":
            continue  # arrow-key tracing is judged by key timing, not by this model
        f = trajectory_features([s["t"] for s in sub["samples"]], [s["x"] for s in sub["samples"]],
                                [s["y"] for s in sub["samples"]])
        if len(f):
            out.append({"label": r["label"], "split": r["split"], "windows": f, "pointer_type": sub["pointer_type"]})
    return out


def _share_weights(n_public: int, n_collected: int, share: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-window weights so collected windows carry ``share`` of their class."""
    if n_collected == 0:
        return np.ones(n_public), np.ones(0)
    return np.full(n_public, (1 - share) / n_public), np.full(n_collected, share / n_collected)


def run_motor(rows: list[dict], models_dir: Path, iitkgp: Path | None, rng_seed: int, quick: bool, write: bool) -> dict:
    result: dict = {"name": "motor", "promoted": False}
    ok, why = enough("motor", [r for r in rows if r["data"]["submission"]["pointer_type"] != "keyboard"])
    if not ok:
        result["status"] = f"not retrained: {why}"
        return result
    if iitkgp is None or not (iitkgp / "data").is_dir():
        result["status"] = "not retrained: the public mouse recordings were not provided (--iitkgp)"
        return result

    from motor import train as mt

    sessions = motor_sessions(rows)
    rng = np.random.default_rng(rng_seed)
    max_per_user, bot_traj = (800, 600) if quick else (5000, 5000)
    human, human_traj = mt.human_features(mt.iitkgp_strokes(iitkgp), rng, max_per_user)
    h_train = np.concatenate([v for u, v in human.items() if u not in mt.TEST_USERS])
    public_test_traj = [f for u, fs in human_traj.items() if u in mt.TEST_USERS for f in fs]
    b_train, _, _ = mt.bot_windows(rng, bot_traj)
    _, _, b_test_traj = mt.bot_windows(np.random.default_rng(rng_seed + 1), bot_traj // 3)

    def windows(label: str, splits: tuple[str, ...]) -> np.ndarray:
        parts = [s["windows"] for s in sessions if s["label"] == label and s["split"] in splits]
        return np.concatenate(parts) if parts else np.empty((0, len(FEATURE_NAMES)))

    c_h, c_b = windows("human", ("train",)), windows("attack", ("train", "val"))
    w_pub_h, w_col_h = _share_weights(len(h_train), len(c_h), 0.3)
    w_pub_b, w_col_b = _share_weights(len(b_train), len(c_b), 0.3 * min(1.0, len(c_b) / 50))
    model, X = mt.fit(np.vstack([h_train, c_h]), np.vstack([b_train, c_b]),
                      np.r_[w_pub_h, w_col_h], np.r_[w_pub_b, w_col_b])

    def session_p(predict, f) -> float:
        return float(np.mean(predict(f)))

    def cand_predict(f):
        return model.predict_proba(f)[:, 1]

    # Threshold: accept 98% of unseen public users' movements, and 95% of unseen
    # collected tracing sessions (validation participants), whichever is lower.
    thr = float(np.quantile([session_p(cand_predict, f) for f in public_test_traj], 0.02))
    val = [session_p(cand_predict, s["windows"]) for s in sessions if s["label"] == "human" and s["split"] == "val"]
    if len(val) >= MIN_TEST_SESSIONS:
        thr = min(thr, float(np.quantile(val, 0.05)))

    def metrics(predict, threshold: float) -> dict:
        def passed(f) -> bool:
            return _calibrate(session_p(predict, f), threshold) >= 0.5

        test_h = [s for s in sessions if s["label"] == "human" and s["split"] == "test"]
        test_a = [s for s in sessions if s["label"] == "attack" and s["split"] == "test"]
        by_family: dict[str, list[bool]] = defaultdict(list)
        for fam, f in b_test_traj:
            by_family[fam].append(not passed(f))
        return balanced({
            "collected_human_accept": rate(passed(s["windows"]) for s in test_h),
            "collected_attack_reject": rate(not passed(s["windows"]) for s in test_a) if len(test_a) >= MIN_TEST_ATTACKS else None,
            "public_human_accept": rate(passed(f) for f in public_test_traj),
            "synthetic_bot_reject": float(np.mean([np.mean(v) for v in by_family.values()])),
        })

    current = ModelRegistry(models_dir).get("motor")
    cand_m = metrics(cand_predict, thr)
    if current is None:
        cur_m = balanced({k: (0.0 if v is not None else None) for k, v in cand_m.items() if k != "balanced"})
    else:
        cur_m = metrics(lambda f: current.run(f.astype(np.float32))[1][:, 1], float(current.meta.get("threshold", 0.5)))
    promote, verdict = decide("motor", cur_m, cand_m)
    result.update(status=verdict, promoted=promote, held_out={"current": cur_m, "candidate": cand_m},
                  train_windows={"public_human": int(len(h_train)), "collected_human": int(len(c_h)),
                                 "synthetic_bot": int(len(b_train)), "collected_attack": int(len(c_b))},
                  threshold=thr, quick=quick)
    if promote and write:
        report = {
            "dataset": "IIT Kharagpur raw mouse events + synthetic bots (5 families) + labelled tracing sessions collected by the app",
            "test_users": sorted(mt.TEST_USERS), "trajectory_threshold": thr,
            "held_out": result["held_out"], "train_windows": result["train_windows"],
            "collected_sessions": dict(Counter(f"{s['label']}/{s['split']}" for s in sessions)),
            "n_features": len(FEATURE_NAMES),
        }
        mt.export_model(model, X, models_dir, report, thr)
    return result


# ---------------------------------------------------------------------------
# Voice and face (GPU-trained): readiness only
# ---------------------------------------------------------------------------

def media_readiness(data_dir: Path) -> dict:
    out = {}
    for name, path, pos in (("voice", data_dir / "voice.csv", "bonafide"), ("face", data_dir / "faces" / "index.csv", "real")):
        rows = list(csv.DictReader(open(path))) if path.exists() else []
        n_pos = sum(1 for r in rows if r["label"] == pos)
        groups = {r.get("participant") or r.get("video") for r in rows if r["label"] == pos}
        out[name] = {
            "genuine": n_pos, "fake": len(rows) - n_pos,
            "people" if name == "voice" else "sessions": len(groups),
            "enough_to_fine_tune": n_pos >= MEDIA_GUIDE[name] and len(rows) - n_pos >= MEDIA_GUIDE[name],
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render(report: dict) -> str:
    lines = ["## Retraining report", ""]
    s = report["summary"]
    lines += [f"Dataset: {s.get('labelled_attempts', 0)} labelled sessions from {s.get('participants', 0)} "
              f"participants, plus {s.get('attempts', 0) - s.get('labelled_attempts', 0)} unlabelled opt-in sessions "
              "(unlabelled data is never used for training).", ""]
    promoted = [m["name"] for m in report["models"] if m["promoted"]]
    lines += [f"**Models changed: {', '.join(promoted) if promoted else 'none'}.**", ""]

    lines += ["### How the live checks did on labelled sessions", "",
              "| Check | Real people passed | Attacks blocked |", "|---|---|---|"]
    for r in report["live"]:
        lines.append(f"| {CHECKPOINT_NAMES[r['checkpoint']]} | {pct(r['human_pass_rate'])} of {r['human_sessions']} "
                     f"| {pct(r['attack_block_rate'])} of {r['attack_sessions']} |")
    by_type = [(CHECKPOINT_NAMES[r["checkpoint"]], t, v) for r in report["live"]
               for t, v in r["attack_block_rate_by_type"].items()]
    if by_type:
        lines += ["", "| Check | Attack type | Blocked |", "|---|---|---|"]
        lines += [f"| {cp} | {t.replace('_', ' ')} | {pct(v['blocked'])} of {v['n']} |" for cp, t, v in by_type]

    titles = {"gaze": "Eye check (gaze.onnx)", "motor": "Movement check (motor.onnx)"}
    labels = {
        "human_accept": "Real people accepted", "replay_reject": "Replayed recordings rejected",
        "attack_reject": "Labelled attacks rejected", "collected_human_accept": "Real people accepted (collected)",
        "collected_attack_reject": "Labelled attacks rejected (collected)",
        "public_human_accept": "Real people accepted (public recordings)",
        "synthetic_bot_reject": "Synthetic bots rejected", "balanced": "Balanced accuracy",
    }
    for m in report["models"]:
        lines += ["", f"### {titles[m['name']]}", "", f"Result: **{m['status']}**."]
        cal = m.get("calibration")
        if cal:
            c = cal["all_sessions_with_current_check"]
            lines += ["", f"Current check over all {cal['human_sessions']} real-human recordings and "
                          f"{cal['replay_pairs']} replays into the wrong session: {pct(c['human_accept'])} of real people "
                          f"accepted, {pct(c['replay_reject'])} of replays rejected.",
                      f"Evidence score (passes at about 3.75): real people 5th/25th/50th percentile "
                      f"{cal['human_evidence_p05_p25_p50']}; replays 50th/95th/99th percentile {cal['replay_evidence_p50_p95_p99']}."]
        if "held_out" in m:
            lines += ["", "Measured on participants the candidate never trained on:", "",
                      "| | Current | Candidate |", "|---|---|---|"]
            for k, v in m["held_out"]["candidate"].items():
                cur = m["held_out"]["current"].get(k)
                lines.append(f"| {labels.get(k, k)} | {pct(cur) if k != 'balanced' else f'{cur:.3f}'} "
                             f"| {pct(v) if k != 'balanced' else f'{v:.3f}'} |")
            if m.get("quick"):
                lines += ["", "_Quick mode: trained on a reduced sample; rerun without --quick before merging._"]

    lines += ["", "### Voice and face models (trained on a GPU, not here)", "",
              "| Model | Genuine samples | Fake samples | Enough to fine-tune |", "|---|---|---|---|"]
    for name, r in report["media"].items():
        lines.append(f"| {CHECKPOINT_NAMES[name]} | {r['genuine']} | {r['fake']} | {'yes' if r['enough_to_fine_tune'] else 'not yet'} |")
    lines += ["", "Run `ml/colab_train.ipynb` to train these; it pulls the same dataset.", "",
              "_Numbers come from labelled tester sessions only. Small test sets give noisy percentages: "
              "check the session counts before trusting a difference._"]
    return "\n".join(lines) + "\n"


def run(data_dir: Path, models_dir: Path, iitkgp: Path | None = None, *, quick: bool = False, seed: int = 7,
        write: bool = True) -> dict:
    data = {cp: load_jsonl(data_dir / f"{cp}.jsonl") for cp in CHECKPOINT_NAMES}
    summary = json.loads((data_dir / "summary.json").read_text()) if (data_dir / "summary.json").exists() else {}
    rng = np.random.default_rng(seed)
    return {
        "summary": {k: summary.get(k, 0) for k in ("attempts", "labelled_attempts", "participants")},
        "live": live_performance(data),
        "models": [
            run_gaze(data["gaze"], models_dir, rng, write),
            run_motor(data["motor"], models_dir, iitkgp, seed, quick, write),
        ],
        "media": media_readiness(data_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="dataset written by ml/dataset/export.py")
    ap.add_argument("--models", type=Path, default=ROOT / "backend" / "models")
    ap.add_argument("--iitkgp", type=Path, help="clone of github.com/prativa-97/Mouse-Dynamics (for the movement model)")
    ap.add_argument("--report", type=Path, help="write the report here as Markdown")
    ap.add_argument("--json", type=Path, help="write the report here as JSON")
    ap.add_argument("--quick", action="store_true", help="smaller public sample; for trying the pipeline out")
    ap.add_argument("--dry-run", action="store_true", help="report only; never write a model")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    report = run(args.data, args.models, args.iitkgp, quick=args.quick, seed=args.seed, write=not args.dry_run)
    text = render(report)
    print(text)
    if args.report:
        args.report.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
