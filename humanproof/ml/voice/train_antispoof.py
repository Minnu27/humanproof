"""Train the voice anti-spoofing model (bona fide vs synthetic / converted / replayed speech).

Architecture: RawNet2-style (learnable sinc filterbank on raw waveform, residual
blocks with filter-wise feature map scaling, GRU), a standard, well-validated
ASVspoof baseline that exports cleanly to ONNX.

Data (any combination; labels come from each dataset's protocol file):
  --asvspoof2019-la /data/LA      ASVspoof 2019 Logical Access (TTS + voice conversion)
  --asvspoof2019-pa /data/PA      ASVspoof 2019 Physical Access (replay)
  --in-the-wild /data/itw         "In-the-Wild" real-world deepfakes (strongly recommended for eval)
  --csv extra.csv                 your own data: columns path,label[,split] (label: bonafide|spoof);
                                  voice.csv from ml/dataset/export.py (sessions collected by the app) fits

Robustness: RawBoost-style channel augmentation (random filtering, noise, gain,
clipping) so the model survives phone microphones and codecs.

    python ml/voice/train_antispoof.py --asvspoof2019-la /data/LA --in-the-wild /data/itw \
        --epochs 30 --out backend/models

Outputs voice_antispoof.onnx (input: float32 [batch, 64600] 16 kHz waveform;
output: logits [batch, 2], index 1 = bona fide) + metrics + manifest entry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from common.manifest import update_manifest  # noqa: E402

SR = 16_000
N_SAMPLES = 64_600


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def asvspoof_items(root: Path, kind: str, split: str) -> list[tuple[Path, int]]:
    """kind: LA or PA; split: train, dev, eval. Returns (flac path, label 1=bonafide)."""
    names = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}
    proto = next(root.glob(f"ASVspoof2019_{kind}_cm_protocols/ASVspoof2019.{kind}.cm.{names[split]}.txt"))
    audio_dir = root / f"ASVspoof2019_{kind}_{split}" / "flac"
    out = []
    for line in proto.read_text().splitlines():
        parts = line.split()
        out.append((audio_dir / f"{parts[1]}.flac", 1 if parts[-1] == "bonafide" else 0))
    return out


def in_the_wild_items(root: Path) -> list[tuple[Path, int]]:
    out = []
    with open(root / "meta.csv") as fh:
        for row in csv.DictReader(fh):
            out.append((root / row["file"], 1 if row["label"].strip().lower() in ("bona-fide", "bonafide") else 0))
    return out


def csv_items(path: Path) -> dict[str, list[tuple[Path, int]]]:
    """Rows grouped by their optional ``split`` column (train / val / test; "" if absent)."""
    groups: dict[str, list[tuple[Path, int]]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            groups.setdefault(r.get("split") or "", []).append((Path(r["path"]), 1 if r["label"] == "bonafide" else 0))
    return groups


def load_audio(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import torchaudio.functional as AF

        x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def fix_length(x: np.ndarray, train: bool) -> np.ndarray:
    if len(x) >= N_SAMPLES:
        start = random.randint(0, len(x) - N_SAMPLES) if train else 0
        return x[start:start + N_SAMPLES]
    return np.tile(x, math.ceil(N_SAMPLES / max(1, len(x))))[:N_SAMPLES]


def rawboost(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Channel/codec augmentation approximating RawBoost (Tak et al., 2022)."""
    from scipy.signal import butter, lfilter

    if rng.random() < 0.5:  # random band-pass (microphone / phone line colouration)
        lo = rng.uniform(50, 400)
        hi = rng.uniform(3000, 7800)
        b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
        x = lfilter(b, a, x).astype(np.float32)
    if rng.random() < 0.5:  # additive noise at 10-40 dB SNR
        snr = rng.uniform(10, 40)
        p = np.mean(x ** 2) + 1e-9
        x = x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(x)).astype(np.float32)
    if rng.random() < 0.3:  # mu-law companding round trip (telephony codecs)
        mu = 255.0
        y = np.sign(x) * np.log1p(mu * np.abs(np.clip(x, -1, 1))) / np.log1p(mu)
        y = np.round(y * 127) / 127
        x = (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu).astype(np.float32)
    if rng.random() < 0.3:  # gain + soft clipping
        x = np.tanh(x * rng.uniform(1, 4)).astype(np.float32)
    return x * rng.uniform(0.3, 1.0)


class AudioSet(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        x = fix_length(load_audio(path), self.train)
        if self.train:
            x = rawboost(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), label


# ---------------------------------------------------------------------------
# Model (RawNet2-style)
# ---------------------------------------------------------------------------

class SincConv(nn.Module):
    def __init__(self, out_channels=128, kernel_size=1024, sr=SR):
        super().__init__()
        self.kernel_size = kernel_size + (kernel_size % 2 == 0)
        mel = np.linspace(2595 * np.log10(1 + 30 / 700), 2595 * np.log10(1 + (sr / 2 - 100) / 700), out_channels + 1)
        hz = 700 * (10 ** (mel / 2595) - 1)
        self.low = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))
        n = (self.kernel_size - 1) / 2
        self.register_buffer("t", (torch.arange(-n, n + 1).view(1, -1) / sr))
        self.register_buffer("window", torch.hamming_window(self.kernel_size, periodic=False))

        self.frozen: torch.Tensor | None = None

    def filters(self) -> torch.Tensor:
        low = torch.abs(self.low) + 30
        high = torch.clamp(low + torch.abs(self.band), 30, SR / 2)
        f_low = 2 * low * torch.special.sinc(2 * low * self.t)
        f_high = 2 * high * torch.special.sinc(2 * high * self.t)
        filters = (f_high - f_low) * self.window
        return (filters / (filters.abs().max(dim=1, keepdim=True).values + 1e-8)).unsqueeze(1)

    def freeze(self) -> None:
        """For export: the learned band edges become a constant filter bank."""
        with torch.no_grad():
            self.frozen = self.filters().detach().clone()

    def forward(self, x):
        w = self.frozen if self.frozen is not None else self.filters()
        return F.conv1d(x, w, padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, first=False):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool1d(3)
        self.fms = nn.Linear(cout, cout)

    def forward(self, x):
        out = x if self.first else F.leaky_relu(self.bn1(x), 0.3)
        out = self.conv2(F.leaky_relu(self.bn2(self.conv1(out)), 0.3))
        out = self.pool(out + self.down(x))
        s = torch.sigmoid(self.fms(out.mean(dim=-1))).unsqueeze(-1)  # filter-wise feature map scaling
        return out * s + s


class RawNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinc = SincConv(128, 1024)
        self.bn0 = nn.BatchNorm1d(128)
        self.blocks = nn.Sequential(
            ResBlock(128, 128, first=True), ResBlock(128, 128),
            ResBlock(128, 256), ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 256),
        )
        self.bn_gru = nn.BatchNorm1d(256)
        self.gru = nn.GRU(256, 512, num_layers=3, batch_first=True)
        self.fc = nn.Linear(512, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, x):  # x: [B, T]
        x = x.unsqueeze(1)
        x = F.max_pool1d(torch.abs(self.sinc(x)), 3)
        x = F.leaky_relu(self.bn0(x), 0.3)
        x = self.blocks(x)
        x = F.leaky_relu(self.bn_gru(x), 0.3).transpose(1, 2)
        x, _ = self.gru(x)
        return self.out(self.fc(x[:, -1]))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ss.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
        ys.append(y.numpy())
    y, s = np.concatenate(ys), np.concatenate(ss)
    return compute_eer(y, s), y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asvspoof2019-la", type=Path)
    ap.add_argument("--asvspoof2019-pa", type=Path)
    ap.add_argument("--in-the-wild", type=Path)
    ap.add_argument("--csv", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    train, dev, evals = [], [], {}
    for kind, root in (("LA", args.asvspoof2019_la), ("PA", args.asvspoof2019_pa)):
        if root:
            train += asvspoof_items(root, kind, "train")
            dev += asvspoof_items(root, kind, "dev")
            evals[f"asvspoof2019_{kind.lower()}_eval"] = asvspoof_items(root, kind, "eval")
    for c in args.csv:
        groups = csv_items(c)
        if set(groups) - {""}:
            # The file says who is held out (ml/dataset/export.py splits by person, so
            # the same voice never appears in both training and test data).
            train += groups.get("train", []) + groups.get("", [])
            dev += groups.get("val", [])
            if len({y for _, y in groups.get("test", [])}) == 2:  # an error rate needs both classes
                evals[f"{c.stem}_test"] = groups["test"]
        else:
            items = groups.get("", [])
            random.Random(0).shuffle(items)
            k = int(0.9 * len(items))
            train += items[:k]
            dev += items[k:]
    if args.in_the_wild:
        evals["in_the_wild"] = in_the_wild_items(args.in_the_wild)  # held out entirely: real-world test
    if not train:
        sys.exit("No training data given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RawNet2().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # class-balanced loss: spoof usually outnumbers bona fide ~9:1 in ASVspoof
    n_bona = sum(1 for _, y in train if y == 1)
    weight = torch.tensor([n_bona / len(train), 1 - n_bona / len(train)], device=device) * 2
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    tl = DataLoader(AudioSet(train, True), args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0)
    dl = DataLoader(AudioSet(dev, False), args.batch * 2, num_workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "voice_antispoof.pt"
    best = 1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y, weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()
        eer, _, _ = evaluate(model, dl, device)
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / max(1, i + 1):.4f} dev EER {eer * 100:.2f}%", flush=True)
        if eer < best:
            best = eer
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    metrics = {"dev_eer": best}
    for name, items in evals.items():
        eer, y, s = evaluate(model, DataLoader(AudioSet(items, False), args.batch * 2, num_workers=args.workers), device)
        metrics[f"{name}_eer"] = eer
        print(f"{name}: EER {eer * 100:.2f}% on {len(items)} files")

    export(model.cpu().eval(), args.out, metrics)


def export(model: nn.Module, out: Path, metrics: dict) -> None:
    import onnxruntime as ort

    model.sinc.freeze()
    dummy = torch.randn(2, N_SAMPLES) * 0.1
    path = out / "voice_antispoof.onnx"
    torch.onnx.export(model, (dummy,), str(path), input_names=["wave"], output_names=["logits"],
                      dynamic_axes={"wave": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    diff = float(np.abs(sess.run(None, {"wave": dummy.numpy()})[0] - ref).max())
    assert diff < 1e-3, f"ONNX parity failed: {diff}"
    metrics["onnx_parity_max_abs_diff"] = diff
    (out / "voice_antispoof.metrics.json").write_text(json.dumps(metrics, indent=2))
    update_manifest(out, "voice_antispoof", path, {"input": "wave[batch,64600]@16k", "bonafide_index": 1, **metrics})
    print(f"Wrote {path} (parity diff {diff:.2e})")


if __name__ == "__main__":
    main()
