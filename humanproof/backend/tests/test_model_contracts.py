"""I/O contract tests: the backend feeds each ONNX model exactly the tensor shape the
training scripts export (voice [N,64600], speaker [1,T]->[1,192], face [N,3,224,224]).
Tiny stand-in graphs are built with onnx.helper so the tests run without GPUs."""
import json

import numpy as np
import onnx
from onnx import TensorProto, helper

import sim
from humanproof import challenges as C
from humanproof.models import ModelRegistry
from humanproof.schemas import VoiceSubmission
from humanproof.scoring import face, voice
from ml_manifest import write_manifest


def _save(model: onnx.ModelProto, path):
    model.opset_import[0].version = 17
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def antispoof_graph(bonafide_logit: float):
    x = helper.make_tensor_value_info("wave", TensorProto.FLOAT, ["n", 64600])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [64600, 2], np.zeros((64600, 2), np.float32).ravel().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [2], [0.0, bonafide_logit])
    g = helper.make_graph([helper.make_node("Gemm", ["wave", "w", "b"], ["logits"])], "as", [x], [y], [w, b])
    return helper.make_model(g)


def speaker_graph():
    x = helper.make_tensor_value_info("wave", TensorProto.FLOAT, [1, "t"])
    y = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 192])
    rm = helper.make_node("ReduceMean", ["wave"], ["m"], keepdims=1)  # [1,1]
    ones = helper.make_tensor("ones", TensorProto.FLOAT, [1, 192], np.linspace(1, 2, 192, dtype=np.float32).tolist())
    add = helper.make_node("Add", ["m", "ones"], ["embedding"])
    return helper.make_model(helper.make_graph([rm, add], "spk", [x], [y], [ones]))


def face_graph(real_logit: float):
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 3, 224, 224])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", 2])
    gap = helper.make_node("GlobalAveragePool", ["image"], ["p"])          # [n,3,1,1]
    flat = helper.make_node("Flatten", ["p"], ["f"])                      # [n,3]
    w = helper.make_tensor("w", TensorProto.FLOAT, [3, 2], [0.0] * 6)
    b = helper.make_tensor("b", TensorProto.FLOAT, [2], [0.0, real_logit])
    gemm = helper.make_node("Gemm", ["f", "w", "b"], ["logits"])
    return helper.make_model(helper.make_graph([gap, flat, gemm], "face", [x], [y], [w, b]))


def registry(tmp_path, bonafide=4.0, real=4.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _save(antispoof_graph(bonafide), tmp_path / "voice_antispoof.onnx")
    _save(speaker_graph(), tmp_path / "voice_speaker.onnx")
    _save(face_graph(real), tmp_path / "face_deepfake.onnx")
    write_manifest(tmp_path, ["voice_antispoof", "voice_speaker", "face_deepfake"])
    return ModelRegistry(tmp_path)


def _voice_sub(rng, seconds=3.5):
    x = sim.speech_like(rng, seconds)
    return x, VoiceSubmission(audio_wav_b64=sim.wav_b64(x), audio_offset_ms=0, mouth_frames=sim.mouth_frames_for(x, rng))


def test_registry_loads_all(tmp_path):
    reg = registry(tmp_path)
    assert reg.status()["voice_antispoof"] == "loaded"
    assert reg.status()["face_deepfake"] == "loaded"
    assert reg.status()["voice_speaker"] == "loaded"


def test_voice_with_models(tmp_path, transcriber):
    reg = registry(tmp_path)
    rng = np.random.default_rng(1)
    ch = C.make_voice_challenge()
    transcriber.next_text = " ".join(ch.words)
    for seconds in (2.0, 7.0):  # short (tiled) and long (multi-crop) paths
        _, sub = _voice_sub(rng, seconds)
        s, reasons, info, emb = voice.score(ch, sub, antispoof_model=reg.get("voice_antispoof"),
                                            speaker_model=reg.get("voice_speaker"), transcriber=transcriber,
                                            allow_fallback=False)
        assert info["antispoof_p_bonafide"] > 0.95
        assert emb.shape == (192,) and abs(np.linalg.norm(emb) - 1) < 1e-5
        assert s > 0.5, reasons


def test_voice_spoof_detected_by_model(tmp_path, transcriber):
    reg = registry(tmp_path, bonafide=-4.0)
    rng = np.random.default_rng(2)
    ch = C.make_voice_challenge()
    transcriber.next_text = " ".join(ch.words)
    _, sub = _voice_sub(rng)
    s, reasons, _, _ = voice.score(ch, sub, antispoof_model=reg.get("voice_antispoof"), speaker_model=None,
                                   transcriber=transcriber, allow_fallback=False)
    assert s < 0.1 and "voice sounds synthetic or replayed" in reasons


def test_face_with_model(tmp_path):
    rng = np.random.default_rng(3)
    real = face.score(sim.face_crops(rng), registry(tmp_path / "a", real=4.0).get("face_deepfake"), False)
    fake = face.score(sim.face_crops(rng), registry(tmp_path / "b", real=-4.0).get("face_deepfake"), False)
    assert real[0] > 0.9 and fake[0] < 0.1


def test_face_rejects_static_and_bad_images(tmp_path):
    rng = np.random.default_rng(4)
    crop = sim.face_crops(rng, 1)[0]
    s, reasons, _ = face.score([crop] * 4, None, True)
    assert s < 0.1 and "identical" in reasons[0]
    s, reasons, _ = face.score(["not-base64!"], None, True)
    assert s == 0
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (160, 160)).save(buf, format="PNG")
    s, reasons, _ = face.score([base64.b64encode(buf.getvalue()).decode()], None, True)
    assert s == 0 and "JPEG" in reasons[0]


def test_manifest_json_is_valid(tmp_path):
    registry(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert set(m) == {"voice_antispoof", "voice_speaker", "face_deepfake"}
