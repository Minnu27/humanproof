"""Stored samples -> training dataset -> retraining decision (ml/dataset/export.py, ml/retrain.py)."""
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import sim
from humanproof import challenges as C

from conftest import MODELS_DIR, TEST_COLLECTION_KEY
from test_api import CONSENT, HumanMotorModel, run_flow

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import retrain  # noqa: E402
from dataset import export as ds  # noqa: E402


def as_tester(participant, label="human", attack_type=None):
    c = {"code": TEST_COLLECTION_KEY, "participant": participant, "label": label}
    return {**c, "attack_type": attack_type} if attack_type else c


@pytest.fixture
def exported(client, clock, transcriber, tmp_path):
    client.app_ctx.models._models["motor"] = HumanMotorModel()
    rng = np.random.default_rng(31)
    for p in ("p01", "p02", "p03"):
        run_flow(client, clock, transcriber, rng, collection=as_tester(p))
        run_flow(client, clock, transcriber, rng, collection=as_tester(p))
    run_flow(client, clock, transcriber, rng, human=False, collection=as_tester("p01", "attack", "tts_voice"))
    run_flow(client, clock, transcriber, rng, consent={**CONSENT, "media_opt_in": True})  # ordinary user, unlabelled
    out = tmp_path / "export"
    summary = ds.export(client.app_ctx.store, client.app_ctx.keyring, out)
    return out, summary


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_export_writes_a_labelled_dataset(exported):
    out, summary = exported
    assert summary["attempts"] == 8 and summary["labelled_attempts"] == 7 and summary["participants"] == 3
    assert summary["undecryptable_samples"] == 0

    voice = rows(out / "voice.csv")
    assert sorted(r["label"] for r in voice) == ["bonafide"] * 6 + ["spoof"]
    assert all(Path(r["path"]).read_bytes()[:4] == b"RIFF" for r in voice)
    spoof = next(r for r in voice if r["label"] == "spoof")
    assert spoof["attack_type"] == "tts_voice" and spoof["participant"] == "p01"

    faces = rows(out / "faces" / "index.csv")
    # A text-to-speech attack fakes the voice only: its face snapshots are "unknown" and left out.
    assert len(faces) == 6 * 4 and {r["label"] for r in faces} == {"real"}
    assert all((out / "faces" / r["path"]).read_bytes()[:2] == b"\xff\xd8" for r in faces)

    gaze = retrain.load_jsonl(out / "gaze.jsonl")
    assert sorted(r["label"] for r in gaze) == ["human"] * 6 + ["unknown", "unlabelled"]
    g = next(r for r in gaze if r["label"] == "human")
    assert len(g["data"]["frames"]) > 100 and g["data"]["challenge"]["keyframes"] and g["split"] in ds.SPLITS

    # Ordinary users' recordings are not written unless asked for, and never into the training lists.
    assert not (out / "audio" / "unlabelled").exists() and not (out / "faces" / "unlabelled").exists()
    sessions = rows(out / "sessions.csv")
    assert sum(1 for s in sessions if s["source"] == "opt_in") == 1
    assert (out / "gaze.jsonl").stat().st_mode & 0o077 == 0  # readable by the owner only


def test_nobody_is_in_both_training_and_test_data(exported):
    out, _ = exported
    by_person = {}
    for r in rows(out / "voice.csv"):
        by_person.setdefault(r["participant"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_person.values())
    assert "test" in {next(iter(s)) for s in by_person.values()}


def test_split_assignment():
    assert ds.assign_splits(set()) == {} and ds.assign_splits({"p01"}) == {"p01": "train"}
    two = ds.assign_splits({"a", "b"})
    assert sorted(two.values()) == ["test", "train"]
    people = {f"p{i:02d}" for i in range(20)}
    s = ds.assign_splits(people)
    assert s == ds.assign_splits(people)  # deterministic
    counts = {k: list(s.values()).count(k) for k in ds.SPLITS}
    assert counts == {"train": 14, "val": 3, "test": 3}


def test_export_with_the_wrong_key_reports_instead_of_writing_garbage(client, clock, transcriber, tmp_path):
    from humanproof.security.crypto import keys_from_master

    run_flow(client, clock, transcriber, np.random.default_rng(32), collection=as_tester("p01"))
    other, _, _ = keys_from_master(b"a-different-master-secret-000001")
    summary = ds.export(client.app_ctx.store, other, tmp_path / "x")
    assert summary["undecryptable_samples"] == 9 and summary["attempts"] == 0
    assert rows(tmp_path / "x" / "voice.csv") == []


def test_retraining_changes_nothing_without_enough_data(exported, tmp_path):
    out, _ = exported
    models = tmp_path / "models"
    shutil.copytree(MODELS_DIR, models)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir()}
    report = retrain.run(out, models, iitkgp=None)
    assert [m["promoted"] for m in report["models"]] == [False, False]
    assert all(m["status"].startswith("not retrained: needs at least") for m in report["models"])
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir()} == before
    live = {r["checkpoint"]: r for r in report["live"]}
    assert live["voice"]["human_sessions"] == 6 and live["voice"]["attack_sessions"] == 1
    assert live["voice"]["attack_block_rate_by_type"] == {"tts_voice": {"n": 1, "blocked": 1.0}}
    assert report["media"]["voice"] == {"genuine": 6, "fake": 1, "people": 3, "enough_to_fine_tune": False}
    text = retrain.render(report)
    assert "Models changed: none" in text and "p01" not in text  # the report never names participants
    json.dumps(report, default=float)


def test_real_recordings_measured_against_another_sessions_path_are_replays():
    """The eye-check training data: each real recording against its own dot path
    (genuine) and against other sessions' paths (a replay)."""
    rng = np.random.default_rng(33)
    recs = []
    for i in range(14):
        ch = C.make_gaze_challenge()
        recs.append({"capture_id": f"c{i}", "label": "human", "split": "train", "attack_type": None,
                     "participant": f"p{i % 5}", "score": 1.0,
                     "data": {"challenge": asdict(ch), "frames": sim.webcam_gaze_frames(ch, rng)}})
    items = retrain.gaze_items(recs, rng, pairs_per_session=4)
    assert sum(it["kind"] == "genuine" for it in items) == 14 and sum(it["kind"] == "replay" for it in items) == 56
    m = retrain.gaze_metrics(items, None)
    assert m["human_accept"] >= 0.8 and m["replay_reject"] >= 0.95 and m["attack_reject"] is None


def test_a_candidate_is_adopted_only_on_a_clear_gain_with_nothing_clearly_worse():
    cur = retrain.balanced({"human_accept": 0.80, "replay_reject": 0.99})
    ok, why = retrain.decide("gaze", cur, retrain.balanced({"human_accept": 0.95, "replay_reject": 0.985}))
    assert ok and why.startswith("adopted")
    ok, why = retrain.decide("gaze", cur, retrain.balanced({"human_accept": 0.99, "replay_reject": 0.90}))
    assert not ok and "worse on replay reject" in why
    ok, why = retrain.decide("gaze", cur, retrain.balanced({"human_accept": 0.805, "replay_reject": 0.99}))
    assert not ok and "no clear improvement" in why
    # A measure that could not be computed for one side is left out of the comparison.
    ok, _ = retrain.decide("motor", retrain.balanced({"a": 0.7, "b": None}), retrain.balanced({"a": 0.9, "b": 0.1}))
    assert ok
