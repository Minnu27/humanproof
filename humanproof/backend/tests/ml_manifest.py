"""Test helper mirroring ml/common/manifest.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from common.manifest import update_manifest  # noqa: E402


def write_manifest(models_dir: Path, names: list[str]) -> None:
    for n in names:
        update_manifest(models_dir, n, models_dir / f"{n}.onnx", {})
