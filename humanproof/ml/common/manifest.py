"""Model manifest: the backend refuses to load any model whose SHA-256 differs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(models_dir: Path, name: str, model_path: Path, meta: dict) -> None:
    mpath = models_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[name] = {
        "file": model_path.name,
        "sha256": sha256_file(model_path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": meta,
    }
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
