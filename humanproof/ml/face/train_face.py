"""Train the face deepfake detector on crops from extract_faces.py.

Backbone: EfficientNet-B0 (ImageNet-pretrained via timm): small enough for fast
CPU inference on the server (~20 ms / crop), strong on FF++/Celeb-DF.

Honest evaluation:
  * identity-disjoint test split;
  * video-level AUC (mean of frame scores), which is what the product decides on;
  * cross-dataset AUC: train on one source (e.g. FF++), test on another
    (Celeb-DF) with --cross-test celebdf. Expect this number to be much lower
    than in-dataset AUC; it is the one that predicts real-world behaviour.

    python ml/face/train_face.py --faces /data/faces --epochs 12 --out backend/models --cross-test celebdf

Add snapshots collected by the app (ml/dataset/export.py) with ``--extra /data/hp-export/faces``.

Output: face_deepfake.onnx (input float32 [N,3,224,224], ImageNet-normalised;
output logits [N,2], index 1 = real) + metrics + manifest entry.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from common.manifest import update_manifest  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:  # re-compression, as camera pipelines and uploads do
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(40, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.2:  # low-resolution webcams
        s = rng.randint(96, 180)
        img = img.resize((s, s), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
    if rng.random() < 0.3:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


class Faces(Dataset):
    def __init__(self, root: Path, rows: list[dict], train: bool):
        self.root, self.rows, self.train = root, rows, train
        self.rng = random.Random()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(Path(r.get("root") or self.root) / r["path"]).convert("RGB").resize((224, 224))
        if self.train:
            img = augment(img, self.rng)
        x = ((np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), 1 if r["label"] == "real" else 0, i


def balanced_sampler(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1
    weights = [1.0 / counts[r["label"]] for r in rows]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


@torch.no_grad()
def evaluate(model, ds: Faces, device, batch=128):
    from sklearn.metrics import roc_auc_score

    model.eval()
    scores = np.zeros(len(ds))
    for x, _, idx in DataLoader(ds, batch, num_workers=4):
        logits = model(x.to(device))
        scores[idx.numpy()] = F.softmax(logits.float(), 1)[:, 1].cpu().numpy()
    labels = np.array([1 if r["label"] == "real" else 0 for r in ds.rows])
    frame_auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    by_video = defaultdict(list)
    for r, s in zip(ds.rows, scores):
        by_video[(r["video"], r["label"])].append(s)
    vl = np.array([1 if lab == "real" else 0 for (_, lab) in by_video])
    vs = np.array([np.mean(v) for v in by_video.values()])
    video_auc = roc_auc_score(vl, vs) if len(set(vl)) == 2 else float("nan")
    return float(frame_auc), float(video_auc)


def main():
    import timm

    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", type=Path, required=True)
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="more crop folders with an index.csv, e.g. faces/ from ml/dataset/export.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--cross-test", help="source held out entirely for cross-dataset testing, e.g. celebdf")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    rows = []
    for root in [args.faces, *args.extra]:
        rows += [{**r, "root": str(root)} for r in csv.DictReader(open(root / "index.csv"))]
    cross = [r for r in rows if args.cross_test and r["source"] == args.cross_test and r["split"] == "test"]
    rows = [r for r in rows if not (args.cross_test and r["source"] == args.cross_test)]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    print(f"train {len(tr)} val {len(va)} test {len(te)} cross {len(cross)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.arch, pretrained=not args.no_pretrained, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * max(1, len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    loader = DataLoader(Faces(args.faces, tr, True), args.batch, sampler=balanced_sampler(tr),
                        num_workers=8, drop_last=True, pin_memory=True)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "face_deepfake.pt"
    best = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y, _ in loader:
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
        fa, vauc = evaluate(model, Faces(args.faces, va, False), device)
        print(f"epoch {epoch + 1}: val frame AUC {fa:.4f} video AUC {vauc:.4f}", flush=True)
        if vauc > best:
            best = vauc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    metrics = {"arch": args.arch, "val_video_auc": best}
    metrics["test_frame_auc"], metrics["test_video_auc"] = evaluate(model, Faces(args.faces, te, False), device)
    if cross:
        metrics[f"cross_{args.cross_test}_frame_auc"], metrics[f"cross_{args.cross_test}_video_auc"] = evaluate(
            model, Faces(args.faces, cross, False), device)
    print(json.dumps(metrics, indent=2))
    export(model.cpu().eval(), args.out, metrics)


def export(model, out: Path, metrics: dict) -> None:
    import onnxruntime as ort

    dummy = torch.randn(4, 3, 224, 224)
    path = out / "face_deepfake.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    diff = float(np.abs(sess.run(None, {"image": dummy.numpy()})[0] - ref).max())
    assert diff < 1e-3, f"ONNX parity failed: {diff}"
    metrics["onnx_parity_max_abs_diff"] = diff
    (out / "face_deepfake.metrics.json").write_text(json.dumps(metrics, indent=2))
    update_manifest(out, "face_deepfake", path, {"input": "image[n,3,224,224] imagenet-norm", "real_index": 1, **metrics})
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
