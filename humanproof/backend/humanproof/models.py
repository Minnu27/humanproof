"""Model registry: integrity-checked ONNX models.

A model is loaded only if its SHA-256 matches manifest.json, so a swapped or
tampered file is refused rather than silently used. ONNX (not pickle) is used
for every model, so loading a model file can never execute code.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("humanproof.models")

KNOWN_MODELS = ("motor", "gaze", "voice_antispoof", "voice_speaker", "face_deepfake")


@dataclass
class LoadedModel:
    name: str
    session: Any
    meta: dict
    input_name: str

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: x})


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: dict[str, LoadedModel] = {}
        self.load_errors: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        mpath = self.models_dir / "manifest.json"
        if not mpath.exists():
            log.warning("No model manifest at %s", mpath)
            return
        import onnxruntime as ort

        manifest = json.loads(mpath.read_text())
        for name, entry in manifest.items():
            if name not in KNOWN_MODELS:
                continue
            path = (self.models_dir / entry["file"]).resolve()
            if self.models_dir.resolve() not in path.parents:
                self.load_errors[name] = "path escapes models dir"
                continue
            if not path.exists():
                self.load_errors[name] = "file missing"
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                self.load_errors[name] = "sha256 mismatch"
                log.error("Refusing model %s: sha256 mismatch", name)
                continue
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            self._models[name] = LoadedModel(name, sess, entry.get("meta", {}), sess.get_inputs()[0].name)
            log.info("Loaded model %s (%s)", name, digest[:12])

    def get(self, name: str) -> LoadedModel | None:
        return self._models.get(name)

    def status(self) -> dict[str, str]:
        return {
            n: ("loaded" if n in self._models else self.load_errors.get(n, "not trained"))
            for n in KNOWN_MODELS
        }
