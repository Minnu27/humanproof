"""Train the optional gaze model (gaze.onnx) from two CSV files of gaze features.

Most people should use ``ml/retrain.py`` instead: it builds the examples from the
sessions the app has collected (including replay examples made by measuring real
recordings against other sessions' dot paths), holds out participants, and only
adopts a model that beats the current check. This script is the bare trainer for
feature tables you have assembled yourself (columns = ``GAZE_FEATURE_NAMES``).

    python ml/gaze/train_gaze.py --human gaze_human.csv --attack gaze_attack.csv --out backend/models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml"))
from humanproof.scoring.gaze import GAZE_FEATURE_NAMES  # noqa: E402
from common.manifest import update_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--attack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    h = pd.read_csv(args.human)[GAZE_FEATURE_NAMES].to_numpy(np.float32)
    a = pd.read_csv(args.attack)[GAZE_FEATURE_NAMES].to_numpy(np.float32)
    if min(len(h), len(a)) < 50:
        sys.exit("Need at least 50 human and 50 attack sessions")
    X = np.vstack([h, a])
    y = np.r_[np.ones(len(h)), np.zeros(len(a))]
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(X, y)
    metrics = {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
               "n_human": int(len(h)), "n_attack": int(len(a))}
    print(json.dumps(metrics, indent=2))
    from skl2onnx import to_onnx

    onx = to_onnx(model, X[:1], options={id(model): {"zipmap": False}}, target_opset=17)
    path = args.out / "gaze.onnx"
    path.write_bytes(onx.SerializeToString())
    (args.out / "gaze.metrics.json").write_text(json.dumps(metrics, indent=2))
    update_manifest(args.out, "gaze", path, {"human_class": 1, **metrics})


if __name__ == "__main__":
    main()
