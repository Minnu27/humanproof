"""Extract face crops from deepfake-detection video datasets, exactly the way the
app crops faces on-device (MediaPipe Face Landmarker bounding box, 25% margin,
square, 224 px, JPEG q=85). Matching the serving crop avoids train/serve skew.

Supported layouts (pass any subset):
  --ffpp  /data/FaceForensics++   original_sequences/youtube/c23/videos, manipulated_sequences/*/c23/videos
  --celebdf /data/Celeb-DF-v2      Celeb-real, YouTube-real, Celeb-synthesis, List_of_testing_videos.txt
  --csv videos.csv                 your own: columns path,label (real|fake)[,split]

Splits are by *identity* (FF++ target id / Celeb-DF subject) so the same person
never appears in both train and test.

    python ml/face/extract_faces.py --ffpp /data/FF --celebdf /data/CDF \
        --landmarker face_landmarker.task --out /data/faces --frames 16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MARGIN = 0.25
SIZE = 224


def split_for(key: str) -> str:
    b = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10
    return "train" if b < 8 else ("val" if b == 8 else "test")


def ffpp_videos(root: Path):
    for p in sorted((root / "original_sequences/youtube/c23/videos").glob("*.mp4")):
        yield p, "real", "ffpp", "original", split_for("ffpp" + p.stem)
    for method_dir in sorted((root / "manipulated_sequences").iterdir()):
        for p in sorted((method_dir / "c23/videos").glob("*.mp4")):
            target = p.stem.split("_")[0]
            yield p, "fake", "ffpp", method_dir.name, split_for("ffpp" + target)


def celebdf_videos(root: Path):
    test = set()
    listing = root / "List_of_testing_videos.txt"
    if listing.exists():
        test = {line.split()[1] for line in listing.read_text().splitlines() if line.strip()}
    for sub, label in (("Celeb-real", "real"), ("YouTube-real", "real"), ("Celeb-synthesis", "fake")):
        for p in sorted((root / sub).glob("*.mp4")):
            rel = f"{sub}/{p.name}"
            subject = p.stem.split("_")[0]
            split = "test" if rel in test else ("val" if split_for("cdf" + subject) == "val" else "train")
            yield p, label, "celebdf", sub, split


def csv_videos(path: Path):
    with open(path) as fh:
        for r in csv.DictReader(fh):
            p = Path(r["path"])
            yield p, r["label"], "custom", "custom", r.get("split") or split_for(p.stem)


_landmarker = None


def _init(model_path: str):
    global _landmarker
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    _landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1)
    )
    globals()["_mp"] = mp


def crop_face(rgb: np.ndarray):
    """Same geometry as client/src/lib/face.ts cropFace()."""
    mp = globals()["_mp"]
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    xs = np.array([p.x for p in res.face_landmarks[0]]) * w
    ys = np.array([p.y for p in res.face_landmarks[0]]) * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * MARGIN)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    from PIL import Image

    img = Image.fromarray(rgb[y0:y1, x0:x1]).resize((SIZE, SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def process(job):
    path, label, source, method, split, out, n_frames = job
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    if total <= 0:
        return rows
    vid = f"{source}_{method}_{path.stem}"
    for i, idx in enumerate(np.linspace(0, total - 1, n_frames).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        jpg = crop_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if jpg is None:
            continue
        dest = out / split / label / f"{vid}_{i:02d}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(jpg)
        rows.append([str(dest.relative_to(out)), label, split, source, method, vid])
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpp", type=Path)
    ap.add_argument("--celebdf", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--landmarker", required=True, help="face_landmarker.task (same file the app ships)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    videos = []
    if args.ffpp:
        videos += list(ffpp_videos(args.ffpp))
    if args.celebdf:
        videos += list(celebdf_videos(args.celebdf))
    if args.csv:
        videos += list(csv_videos(args.csv))
    jobs = [(*v, args.out, args.frames) for v in videos]
    print(f"{len(jobs)} videos")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "split", "source", "method", "video"])
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.landmarker,)) as pool:
            for n, rows in enumerate(pool.map(process, jobs, chunksize=4)):
                w.writerows(rows)
                if n % 200 == 0:
                    print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
