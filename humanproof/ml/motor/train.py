"""Train the motor (pointer-dynamics) human-vs-bot model.

Human data: IIT Kharagpur mouse-dynamics recordings (9 users, raw OS-level
            mouse events at ~125 Hz), cloned from github.com/prativa-97/Mouse-Dynamics.
Bot data:   bot_generator.py (linear, Bezier, WindMouse, spline-following,
            minimum-jerk sub-movements with pink tremor, noise variants).

Why not Balabit? Its events were captured through RDP at ~9 Hz (median 109 ms
between events). A first model trained on it scored AUC 1.0 by learning that
low-rate capture = human - it would have flagged real browser users as bots.
Balabit is therefore excluded from training; `--balabit` adds it as a clearly
labelled out-of-distribution report only.

Evaluation is honest by construction:
  * human train/test split is by *user* (test users never seen in training);
  * held-out-family runs train without one bot family and test on it;
  * everything passes through the same capture canonicalisation as serving.

    python ml/motor/train.py --iitkgp /data/Mouse-Dynamics --out backend/models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))

from humanproof.scoring.features_motor import FEATURE_NAMES, N_FEATURES, trajectory_features  # noqa: E402
from common.manifest import update_manifest  # noqa: E402
from motor.bot_generator import GENERATORS, generate  # noqa: E402

TEST_USERS = {"harsh", "nikhil", "debalina"}
MAX_GAP_MS = 300


def iitkgp_strokes(root: Path):
    """Yield (user, t_ms, x, y) per continuous movement from the raw event logs."""
    for user_dir in sorted((root / "data").iterdir()):
        if not user_dir.is_dir():
            continue
        for f in sorted(user_dir.glob("*.txt")):
            t_abs = None
            seg_t: list[float] = []
            seg_x: list[float] = []
            seg_y: list[float] = []
            for raw in open(f, errors="ignore"):
                if raw.startswith("*"):  # session header: reset the clock
                    if len(seg_t) >= 10:
                        yield user_dir.name, np.asarray(seg_t), np.asarray(seg_x), np.asarray(seg_y)
                    t_abs, seg_t, seg_x, seg_y = None, [], [], []
                    continue
                parts = [p.strip() for p in raw.split(",")]
                if parts[0] not in ("MM", "MD", "MP", "MR", "MC") or len(parts) < 3:
                    continue
                try:
                    dt = int(parts[-1])
                except ValueError:
                    continue
                t_abs = 0.0 if (t_abs is None or dt > 1e11) else t_abs + dt
                if parts[0] not in ("MM", "MD") or len(parts) != 4:
                    continue
                if seg_t and t_abs - seg_t[-1] > MAX_GAP_MS:
                    if len(seg_t) >= 10:
                        yield user_dir.name, np.asarray(seg_t), np.asarray(seg_x), np.asarray(seg_y)
                    seg_t, seg_x, seg_y = [], [], []
                seg_t.append(t_abs)
                seg_x.append(float(parts[1]))
                seg_y.append(float(parts[2]))
            if len(seg_t) >= 10:
                yield user_dir.name, np.asarray(seg_t), np.asarray(seg_x), np.asarray(seg_y)


def balabit_strokes(root: Path):
    import pandas as pd

    for split in ("training_files", "test_files"):
        for user_dir in sorted((root / split).iterdir()):
            if not user_dir.is_dir():
                continue
            for sess in sorted(user_dir.iterdir()):
                df = pd.read_csv(sess)
                df = df[df["state"].isin(["Move", "Drag"]) & (df.x < 5000) & (df.y < 5000)]
                t = df["client timestamp"].to_numpy(float) * 1000
                x, y = df.x.to_numpy(float), df.y.to_numpy(float)
                cuts = np.where((np.diff(t) > MAX_GAP_MS) | (np.diff(t) < 0))[0] + 1
                for seg in np.split(np.arange(len(t)), cuts):
                    if len(seg) >= 10:
                        yield user_dir.name, t[seg], x[seg], y[seg]


def human_features(strokes, rng, max_per_user: int):
    windows: dict[str, list[np.ndarray]] = {}
    for user, t, x, y in strokes:
        f = trajectory_features(t, x, y)
        if len(f):
            windows.setdefault(user, []).append(f)
    sampled = {}
    for user, parts in windows.items():
        arr = np.concatenate(parts)
        if len(arr) > max_per_user:
            arr = arr[rng.choice(len(arr), max_per_user, replace=False)]
        sampled[user] = arr
    return sampled, windows


def bot_windows(rng, n_traj: int, families=None):
    feats, fams, traj = [], [], []
    fam_list = families or list(GENERATORS)
    for i in range(n_traj):
        fam, t, x, y = generate(rng, fam_list[i % len(fam_list)])
        f = trajectory_features(t, x, y)
        if len(f) > 10:
            f = f[rng.choice(len(f), 10, replace=False)]
        if len(f):
            feats.append(f)
            fams.extend([fam] * len(f))
            traj.append((fam, f))
    return np.concatenate(feats), np.asarray(fams), traj


def eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def make_model():
    return GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.08, max_depth=4, subsample=0.8, min_samples_leaf=20, random_state=0,
    )


def fit(X_h, X_b, w_h=None, w_b=None):
    """Class-balanced fit. ``w_h`` / ``w_b`` optionally weight windows within a class
    (ml/retrain.py uses this so a few hundred collected sessions are not drowned out
    by tens of thousands of public windows)."""
    X = np.vstack([X_h, X_b])
    y = np.concatenate([np.ones(len(X_h)), np.zeros(len(X_b))])  # 1 = human
    w_h = np.ones(len(X_h)) if w_h is None else np.asarray(w_h, dtype=float)
    w_b = np.ones(len(X_b)) if w_b is None else np.asarray(w_b, dtype=float)
    w = np.concatenate([w_h / w_h.sum(), w_b / w_b.sum()]) * len(y) / 2
    return make_model().fit(X, y, sample_weight=w), X


def export_model(model, probe: np.ndarray, out: Path, metrics: dict, threshold: float) -> Path:
    """Write motor.onnx (checked against the sklearn model), its metrics and manifest entry."""
    import onnxruntime as ort
    from skl2onnx import to_onnx

    out.mkdir(parents=True, exist_ok=True)
    probe = probe[:500].astype(np.float32)
    onx = to_onnx(model, probe[:1], options={id(model): {"zipmap": False}}, target_opset=17)
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    diff = float(np.abs(sess.run(None, {sess.get_inputs()[0].name: probe})[1][:, 1]
                        - model.predict_proba(probe)[:, 1]).max())
    if diff >= 1e-4:
        raise RuntimeError(f"ONNX parity failed: {diff}")
    metrics["onnx_parity_max_abs_diff"] = diff
    path = out / "motor.onnx"
    path.write_bytes(onx.SerializeToString())
    (out / "motor.metrics.json").write_text(json.dumps(metrics, indent=2))
    update_manifest(out, "motor", path, {"threshold": threshold, "human_class": 1, "feature_names": FEATURE_NAMES})
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iitkgp", type=Path, required=True, help="clone of github.com/prativa-97/Mouse-Dynamics")
    ap.add_argument("--balabit", type=Path, help="optional out-of-distribution report only")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-user", type=int, default=5000)
    ap.add_argument("--bot-traj", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Extracting human windows (IIT Kharagpur) ...", flush=True)
    human, human_traj = human_features(iitkgp_strokes(args.iitkgp), rng, args.max_per_user)
    h_train = np.concatenate([v for u, v in human.items() if u not in TEST_USERS])
    h_test = np.concatenate([v for u, v in human.items() if u in TEST_USERS])
    print(f"  human windows: train={len(h_train)} test={len(h_test)} users={sorted(human)}", flush=True)

    print("Generating bot trajectories ...", flush=True)
    b_train, _, _ = bot_windows(rng, args.bot_traj)
    b_test, b_test_fam, b_test_traj = bot_windows(np.random.default_rng(args.seed + 1), args.bot_traj // 3)
    print(f"  bot windows: train={len(b_train)} test={len(b_test)}", flush=True)

    model, X = fit(h_train, b_train)
    Xt = np.vstack([h_test, b_test])
    yt = np.concatenate([np.ones(len(h_test)), np.zeros(len(b_test))])
    pt = model.predict_proba(Xt)[:, 1]
    auc, e = roc_auc_score(yt, pt), eer(yt, pt)

    # Trajectory level: serving averages window scores over one trace.
    h_scores = np.asarray([model.predict_proba(f)[:, 1].mean()
                           for u, fs in human_traj.items() if u in TEST_USERS for f in fs])
    b_scores: dict[str, list[float]] = {}
    for fam, f in b_test_traj:
        b_scores.setdefault(fam, []).append(float(model.predict_proba(f)[:, 1].mean()))
    threshold = float(np.quantile(h_scores, 0.02))  # accept 98% of unseen humans' movements
    per_family = {fam: float(np.mean(np.asarray(s) < threshold)) for fam, s in b_scores.items()}

    print("Held-out bot family runs ...", flush=True)
    held_out = {}
    for fam in GENERATORS:
        others = [f for f in GENERATORS if f != fam]
        bt, _, _ = bot_windows(np.random.default_rng(args.seed + 11), args.bot_traj // 2, others)
        m, _ = fit(h_train, bt)
        mask = b_test_fam == fam
        ph, pb = m.predict_proba(h_test)[:, 1], m.predict_proba(b_test[mask])[:, 1]
        held_out[fam] = float(roc_auc_score(np.r_[np.ones(len(ph)), np.zeros(len(pb))], np.r_[ph, pb]))

    order = np.argsort(model.feature_importances_)[::-1]
    metrics = {
        "dataset": "IIT Kharagpur raw mouse events (humans, 9 users) + synthetic bots (5 families)",
        "test_users": sorted(TEST_USERS),
        "window_auc": round(float(auc), 4),
        "window_eer": round(e, 4),
        "trajectory_threshold": threshold,
        "trajectory_human_accept_rate": 0.98,
        "trajectory_bot_reject_rate_by_family": {k: round(v, 4) for k, v in per_family.items()},
        "held_out_family_window_auc": {k: round(v, 4) for k, v in held_out.items()},
        "top_features": {FEATURE_NAMES[i]: round(float(model.feature_importances_[i]), 4) for i in order[:8]},
        "n_features": N_FEATURES,
        "caveats": [
            "Humans are free desktop mouse use, not challenge tracing, and not touch.",
            "Bots are synthetic; a novel bot family can score higher than these numbers suggest.",
            "Retrain on consented tracing data collected by the app before relying on it in production.",
        ],
    }

    if args.balabit:
        print("Out-of-distribution report on Balabit (RDP, ~9 Hz) ...", flush=True)
        bal = []
        for _, t, x, y in balabit_strokes(args.balabit):
            f = trajectory_features(t, x, y)
            if len(f):
                bal.append(model.predict_proba(f)[:, 1].mean())
        metrics["ood_balabit_human_accept_rate"] = round(float(np.mean(np.asarray(bal) >= threshold)), 4)

    print(json.dumps(metrics, indent=2))
    path = export_model(model, Xt, args.out, metrics, threshold)
    print(f"Wrote {path} (ONNX parity {metrics['onnx_parity_max_abs_diff']:.1e})")


if __name__ == "__main__":
    main()
