from pathlib import Path

import torch

from xvc2_student.checkpoint import load_checkpoint, save_checkpoint
from xvc2_student.audit import audit_manifests
from xvc2_student.env_check import version_tuple
from xvc2_student.inspect_libriheavy import inspect_repository
from xvc2_student.config import ExperimentConfig
from xvc2_student.losses import valid_feature_loss
from xvc2_student.model import StreamingPhoneEncoder
from xvc2_student.smoke import tiny_config
from xvc2_student.teacher import loading_failures, remap_legacy_position_conv


def test_forward_and_feature_loss() -> None:
    model = StreamingPhoneEncoder(tiny_config())
    samples = model.receptive_field_samples + model.stride_samples * 7
    waveform = torch.randn(2, samples)
    outputs = model(waveform, torch.tensor([samples, samples]))
    loss = valid_feature_loss(
        outputs["distill_features"],
        torch.randn_like(outputs["distill_features"]),
        outputs["output_lengths"],
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = ExperimentConfig(model=tiny_config())
    model = StreamingPhoneEncoder(config.model)
    path = tmp_path / "step.pt"
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(path, model, config, step=3, scaler=scaler, data_state={"epoch": 2})
    restored = StreamingPhoneEncoder(config.model)
    payload = load_checkpoint(path, restored, scaler=scaler, restore_rng=False)
    assert payload["step"] == 3
    assert payload["data_state"] == {"epoch": 2}
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)


def test_manifest_audit_detects_split_leakage(tmp_path: Path) -> None:
    import json
    import wave

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 320)
    paths = []
    for split, utterance in (("train", "1-2-3"), ("val", "1-4-5")):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            json.dumps({"utterance_id": utterance, "audio_path": str(audio), "phone_ids": [1, 2]})
            + "\n"
        )
        paths.append((split, path))
    report = audit_manifests(paths)
    assert report["status"] == "FAIL"
    assert report["leakage"]["speaker"]["count"] == 1


def test_version_tuple() -> None:
    assert version_tuple("2.4.1+cu121") == (2, 4, 1)


def test_teacher_loading_failures_reject_incomplete_checkpoint() -> None:
    failures = loading_failures(
        {
            "missing_keys": ["encoder.position.weight"],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }
    )
    assert failures == ["teacher_missing_keys=['encoder.position.weight']"]


def test_remap_legacy_position_conv() -> None:
    old_g = "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
    old_v = "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
    state = {old_g: torch.ones(1), old_v: torch.zeros(1), "other": torch.ones(1)}
    remapped = remap_legacy_position_conv(state)
    assert old_g not in remapped
    assert old_v not in remapped
    assert (
        remapped["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"]
        is state[old_g]
    )
    assert (
        remapped["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
        is state[old_v]
    )


def test_inspect_libriheavy_lhotse_manifest(tmp_path: Path) -> None:
    import gzip
    import json

    audio = tmp_path / "librilight" / "small" / "speaker" / "book.flac"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"not-decoded")
    manifest_dir = tmp_path / "upper_no_punc" / "lhotse"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "libriheavy_cuts_small.jsonl.gz"
    row = {
        "id": "small/speaker/book_0",
        "start": 1.0,
        "duration": 2.5,
        "supervisions": [
            {
                "recording_id": "small/speaker/book",
                "speaker": "speaker",
                "custom": {"texts": ["Book text.", "BOOK TEXT"]},
            }
        ],
        "recording": {
            "sources": [
                {
                    "type": "file",
                    "source": "download/librilight/small/speaker/book.flac",
                }
            ]
        },
    }
    with gzip.open(manifest, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    report = inspect_repository(tmp_path, [tmp_path / "librilight"], 100, 10)
    assert report["status"] == "PASS"
    assert report["manifests"][0]["audio_sources_resolved"] == 1
    assert report["manifests"][0]["text_field_counts"] == {
        "book_text": 1,
        "asr_text": 1,
    }
