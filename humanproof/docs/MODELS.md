# Models: data, training, and honest numbers

Every model is ONNX, integrity-pinned in `backend/models/manifest.json`, and
produced by a script in `ml/` with a parity check (ONNX output must match the
training framework). Production refuses to start unless the four required
models load (`motor`, `voice_antispoof`, `voice_speaker`, `face_deepfake`).

| Checkpoint | Model | Status in this repo | Train with |
|---|---|---|---|
| Motor (pointer/touch) | Gradient-boosted trees, 21 dynamics features | **Trained**, shipped (`motor.onnx`) | `ml/motor/train.py` (CPU, ~20 min) |
| Voice liveness | RawNet2-style anti-spoofing CNN+GRU (6.6M params) | Pipeline verified end to end; needs your GPU run | `ml/voice/train_antispoof.py` |
| Voice signature | ECAPA-TDNN (SpeechBrain, VoxCeleb) exported with fbank inside the graph | Export script; needs a machine that can download the checkpoint | `ml/voice/export_speaker.py` |
| Face deepfake | EfficientNet-B0 on face crops made exactly like the app's | Pipeline verified end to end; needs your GPU run | `ml/face/extract_faces.py`, `train_face.py` |
| Gaze | Rules (always on) + optional GBT once you have real sessions | Rules shipped; model trainer ready | `ml/gaze/train_gaze.py` |

The Colab notebook `ml/colab_train.ipynb` runs the three GPU jobs in order.

## Motor model (trained here)

**Humans:** 9 people, raw OS-level mouse events at ~125 Hz (IIT Kharagpur
recordings, github.com/prativa-97/Mouse-Dynamics). Split by person: `debalina`,
`harsh`, `nikhil` are never seen in training.
**Bots:** five synthetic families: linear, Bezier, WindMouse (the widely copied
"humanised" algorithm), curve-following (what a bot tracing our challenge does),
and minimum-jerk sub-movements with overshoot, corrections and pink tremor (a
deliberately sophisticated attacker). Each is combined with jitter/tremor/drift variants.

| Metric (unseen people) | Value |
|---|---|
| Window AUC / EER | 0.9985 / 1.9% |
| Bots rejected at 98% human acceptance: linear, WindMouse, curve-following | 99–100% |
| … Bezier | 91.6% |
| … sophisticated sub-movement bot | 92.2% |
| AUC on a bot family **left out of training** (worst case: sub-movement) | 0.96 |
| Humans accepted from a different capture pipeline (Balabit, RDP ~9 Hz) | 99.0% |

**A mistake caught and fixed during training, worth knowing about.** The first
version trained on the popular Balabit dataset scored a perfect AUC of 1.0.
Investigation showed Balabit was recorded through remote desktop at ~9 events
per second, so the model had learned "sparse events = human" and would have
rejected real browser users. The fix: train on high-rate human data, generate
bots at real capture rates, and canonicalise every trace (16 ms clock, whole
pixels) in the shared feature code so the model cannot learn what device
recorded the data. Perfect scores in this field usually mean leakage.

**Known gaps:** training humans moved a mouse freely; they did not trace our
curve, and none used touch. Turn on consented data collection
(`HP_DATA_COLLECTION_ENABLED=true`), export with
`backend/tools/export_research.py`, and retrain before relying on the motor
checkpoint alone. Touch input is scored with extra caution in the meantime.

## Voice anti-spoofing (you run this)

Data: ASVspoof 2019 LA (text-to-speech and voice conversion attacks), optional
PA (replay), plus your own recordings via `--csv`. Evaluate on **In-the-Wild**
(real-world deepfakes of public figures): it is held out entirely and is the
number that predicts field performance. RawBoost-style augmentation (band-pass,
noise, companding, clipping) makes the model tolerate phone microphones.
Published RawNet2-class results: roughly 1–5% EER on ASVspoof 2019 LA eval and
far worse (often 30%+) on In-the-Wild without augmentation. Report both.

## Face deepfake detector (you run this)

Crops come from the same MediaPipe landmarker the app uses (25% margin, square,
224 px, JPEG q85). Splits are by identity. Train on FaceForensics++ c23 and use
`--cross-test celebdf` to hold Celeb-DF out: in-dataset video AUC is typically
0.95+, cross-dataset commonly 0.65–0.80. The cross-dataset number is the honest one.

## Licences

ASVspoof, In-the-Wild, FaceForensics++, Celeb-DF and VoxCeleb are research
datasets with non-commercial terms. The IIT Kharagpur mouse repository has no
licence file, so treat it as research-only as well. Models trained on them are
fine for research, demos and pilots; before charging customers, retrain on
data you have commercial rights to (your own consented collection).
