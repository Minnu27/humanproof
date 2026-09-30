"""Export a pretrained ECAPA-TDNN speaker-embedding model to ONNX.

Uses SpeechBrain's ``speechbrain/spkrec-ecapa-voxceleb`` (trained on VoxCeleb
1+2, ~0.8% EER on VoxCeleb1-O). Feature extraction (80-d fbank + mean norm) is
folded *into* the exported graph, so the backend feeds raw 16 kHz waveform and
cannot drift from how the model was trained.

    python ml/voice/export_speaker.py --out backend/models
    python ml/voice/export_speaker.py --out backend/models --voxceleb1 /data/vox1   # + EER check

Output: voice_speaker.onnx (input float32 [1, T] waveform -> [1, 192] embedding).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from common.manifest import update_manifest  # noqa: E402


class ConvSTFT(nn.Module):
    """ONNX-friendly drop-in for SpeechBrain's STFT (torch.stft emits complex ops
    that ONNX export does not handle). Same framing, window and output layout:
    [batch, time, n_fft // 2 + 1, 2] (real, imag)."""

    def __init__(self, stft):
        super().__init__()
        n_fft, win = stft.n_fft, stft.win_length
        hop = stft.hop_length
        window = stft.window.float()
        if win < n_fft:  # torch.stft centres a shorter window inside n_fft
            left = (n_fft - win) // 2
            window = torch.nn.functional.pad(window, (left, n_fft - win - left))
        n = torch.arange(n_fft).float()
        k = torch.arange(n_fft // 2 + 1).float()[:, None]
        ang = 2 * torch.pi * k * n / n_fft
        self.register_buffer("cos_k", (torch.cos(ang) * window)[:, None, :])
        self.register_buffer("sin_k", (-torch.sin(ang) * window)[:, None, :])
        self.hop, self.pad, self.center = hop, n_fft // 2, stft.center
        if stft.pad_mode != "constant":
            raise ValueError("ConvSTFT reproduces pad_mode='constant' only")
        if stft.normalized_stft:
            raise ValueError("normalized STFT not supported")

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.center:
            x = torch.nn.functional.pad(x, (self.pad, self.pad))
        re = torch.nn.functional.conv1d(x, self.cos_k, stride=self.hop)
        im = torch.nn.functional.conv1d(x, self.sin_k, stride=self.hop)
        return torch.stack([re, im], dim=-1).transpose(1, 2)


class EcapaWaveform(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.compute_features = classifier.mods.compute_features
        self.compute_features.compute_STFT = ConvSTFT(self.compute_features.compute_STFT)
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        lens = torch.ones(wav.shape[0], device=wav.device)
        feats = self.compute_features(wav)
        feats = self.mean_var_norm(feats, lens)
        emb = self.embedding_model(feats, lens)
        return emb.squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--voxceleb1", type=Path, help="VoxCeleb1 test set root (wav/ + veri_test2.txt) for EER")
    args = ap.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier

    clf = EncoderClassifier.from_hparams(source=args.source, savedir=str(args.out / ".ecapa_cache"))
    clf.eval()

    # Reference embeddings from the untouched SpeechBrain pipeline (before patching STFT).
    torch.manual_seed(0)
    probes = [torch.randn(1, int(16000 * s)) * 0.05 for s in (1.5, 3.0, 6.0)]
    with torch.no_grad():
        refs = [clf.encode_batch(x).squeeze(1).numpy() for x in probes]

    model = EcapaWaveform(clf).eval()
    dummy = torch.randn(1, 16000 * 3) * 0.05
    path = args.out / "voice_speaker.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["wave"], output_names=["embedding"],
                      dynamic_axes={"wave": {1: "time"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for x, ref in zip(probes, refs):  # parity at several lengths (dynamic time axis)
        got = sess.run(None, {"wave": x.numpy()})[0]
        cos = float((ref * got).sum() / (np.linalg.norm(ref) * np.linalg.norm(got)))
        worst = max(worst, 1 - cos)
    assert worst < 1e-3, f"ONNX parity failed (1-cos={worst})"
    metrics = {"source": args.source, "onnx_parity_1_minus_cos": worst}

    if args.voxceleb1:
        metrics["voxceleb1_o_eer"] = voxceleb_eer(sess, args.voxceleb1)
    (args.out / "voice_speaker.metrics.json").write_text(json.dumps(metrics, indent=2))
    update_manifest(args.out, "voice_speaker", path, {"dim": 192, "match_threshold_cos": 0.25, **metrics})
    print(json.dumps(metrics, indent=2))


def voxceleb_eer(sess, root: Path) -> float:
    import soundfile as sf
    from sklearn.metrics import roc_curve

    cache: dict[str, np.ndarray] = {}

    def emb(rel: str) -> np.ndarray:
        if rel not in cache:
            x, _ = sf.read(str(root / "wav" / rel), dtype="float32")
            e = sess.run(None, {"wave": x[None, :]})[0][0]
            cache[rel] = e / np.linalg.norm(e)
        return cache[rel]

    labels, scores = [], []
    for line in (root / "veri_test2.txt").read_text().splitlines():
        lab, a, b = line.split()
        labels.append(int(lab))
        scores.append(float(emb(a) @ emb(b)))
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


if __name__ == "__main__":
    main()
