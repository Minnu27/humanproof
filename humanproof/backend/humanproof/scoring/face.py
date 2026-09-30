"""Face crops captured at secret random moments during checkpoint 1.

* ``face_deepfake.onnx`` (trained on FaceForensics++ / Celeb-DF) scores each
  crop real vs manipulated/synthetic.
* Crops must differ from each other: a static photo or frozen frame fails.

Crops are decoded in memory with strict size limits and never stored.
"""
from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000  # decompression-bomb guard

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


class CropError(ValueError):
    pass


def decode_crop(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CropError("crop is not valid base64") from exc
    try:
        img = Image.open(io.BytesIO(raw))
        if img.format != "JPEG":
            raise CropError("crop must be JPEG")
        if not (96 <= img.width <= 512 and 96 <= img.height <= 512):
            raise CropError("crop size out of range")
        img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    except CropError:
        raise
    except Exception as exc:
        raise CropError("crop could not be decoded") from exc
    return np.asarray(img, dtype=np.float32) / 255.0


def score(crops_b64: list[str], model, allow_fallback: bool) -> tuple[float, list[str], dict]:
    if not crops_b64:
        return (0.5, ["no face crops sent"], {}) if allow_fallback else (0.0, ["no face crops sent"], {})
    try:
        imgs = [decode_crop(c) for c in crops_b64]
    except CropError as exc:
        return 0.0, [str(exc)], {}
    reasons = []
    info: dict = {"n_crops": len(imgs)}
    if len(imgs) >= 2:
        diffs = [float(np.abs(a - b).mean()) for i, a in enumerate(imgs) for b in imgs[i + 1:]]
        info["min_pairwise_diff"] = min(diffs)
        if min(diffs) < 0.004:
            return 0.05, ["face frames are identical (static image?)"], info
    if model is None:
        if allow_fallback:
            return 0.6, ["face deepfake model missing (dev mode)"], info
        return 0.0, ["face deepfake model not available"], info
    batch = np.stack([((im - MEAN) / STD).transpose(2, 0, 1) for im in imgs]).astype(np.float32)
    logits = model.run(batch)[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_real = e[:, 1] / e.sum(axis=1)
    info["p_real"] = [float(p) for p in p_real]
    s = float(np.min(p_real) ** 0.5 * np.mean(p_real) ** 0.5)
    if s < 0.5:
        reasons.append("face looks synthetic or manipulated")
    return s, reasons, info
