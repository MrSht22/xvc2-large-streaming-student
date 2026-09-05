from pathlib import Path

import torch

from xvc2_student.audit import audit_manifests
from xvc2_student.build_audio_manifest import build_manifests
from xvc2_student.checkpoint import load_checkpoint, save_checkpoint
from xvc2_student.config import ExperimentConfig
from xvc2_student.env_check import version_tuple
from xvc2_student.inspect_audio_corpora import combined_report, inspect_corpus
from xvc2_student.inspect_libriheavy import inspect_repository
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


def test_inspect_librispeech_and_librilight(tmp_path: Path) -> None:
    import json
    import wave

    def write_wav(path: Path, seconds: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(b"\x00\x00" * 16_000 * seconds)

    librispeech_root = tmp_path / "LibriSpeech"
    librispeech_audio = librispeech_root / "train-clean-100" / "84" / "121123"
    write_wav(librispeech_audio / "84-121123-0001.wav", 1)
    write_wav(librispeech_audio / "84-121123-0002.wav", 2)
    (librispeech_audio / "84-121123.trans.txt").write_text(
        "84-121123-0001 FIRST TEST\n84-121123-0002 SECOND TEST\n", encoding="utf-8"
    )

    librilight_root = tmp_path / "librilight"
    librilight_audio = librilight_root / "small" / "100" / "book"
    write_wav(librilight_audio / "recording-1.wav", 3)

    librispeech = inspect_corpus("librispeech", librispeech_root, 100, 10)
    librilight = inspect_corpus("librilight", librilight_root, 100, 10)
    report = combined_report(librispeech, librilight)

    assert report["status"] == "PASS"
    assert report["combined"]["audio_files"] == 3
    speech_group = librispeech["groups"]["train-clean-100"]
    assert speech_group["unique_speakers"] == 1
    assert speech_group["unique_chapters_or_books"] == 1
    assert speech_group["text_files"] == 1
    assert speech_group["duration"]["estimate_kind"] == "exact_metadata_sum"
    assert speech_group["duration"]["estimated_hours"] == 3 / 3600
    light_group = librilight["groups"]["small"]
    assert light_group["duration"]["sample_rate_counts"] == {16000: 1}
    assert light_group["duration"]["estimated_hours"] == 3 / 3600

    processed_root = tmp_path / "processed-librilight"
    raw = processed_root / "raw" / "large" / "100" / "book"
    vad = processed_root / "vad" / "large" / "100" / "123"
    write_wav(raw / "recording.wav", 4)
    (raw / "recording.json").write_text(
        json.dumps({"speaker": "100", "book": "book", "sample_rate": 16000}),
        encoding="utf-8",
    )
    write_wav(vad / "recording_0000.wav", 4)
    processed = inspect_corpus("librilight", processed_root, 100, 10)
    assert processed["status"] == "PASS"
    assert processed["recognized_top_level_groups"] == ["raw", "vad"]
    assert processed["groups"]["raw/large"]["unique_speakers"] == 1
    assert processed["groups"]["vad/large"]["unique_speakers"] == 1
    assert processed["groups"]["raw/large"]["json_metadata"]["top_level_key_counts"] == {
        "speaker": 1,
        "book": 1,
        "sample_rate": 1,
    }
    assert processed["representation_estimated_hours"] == {
        "raw": 4 / 3600,
        "vad": 4 / 3600,
    }
    assert processed["estimated_hours"] == 4 / 3600


def test_build_codec_audio_manifests(tmp_path: Path) -> None:
    import json
    import wave

    def write_wav(path: Path, seconds: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(b"\x00\x00" * 16_000 * seconds)

    def add_librispeech(split: str, speaker: str) -> None:
        chapter = "10"
        utterance = f"{speaker}-{chapter}-0000"
        directory = tmp_path / "LibriSpeech" / split / speaker / chapter
        write_wav(directory / f"{utterance}.wav", 1)
        (directory / f"{speaker}-{chapter}.trans.txt").write_text(
            f"{utterance} TEST TRANSCRIPT\n", encoding="utf-8"
        )

    for split, speaker in (
        ("train-clean-100", "1"),
        ("train-clean-360", "2"),
        ("train-other-500", "3"),
        ("dev-clean", "4"),
        ("dev-other", "5"),
        ("test-clean", "6"),
        ("test-other", "7"),
    ):
        add_librispeech(split, speaker)

    librilight = tmp_path / "LibriLight"
    raw = librilight / "raw" / "small" / "8" / "book"
    raw.mkdir(parents=True)
    (raw / "recording.json").write_text(
        json.dumps(
            {
                "speaker": "8",
                "snr": 12.0,
                "voice_activity": [[0.0, 4.0]],
                "book_meta": {"id": 20, "language": "English"},
            }
        ),
        encoding="utf-8",
    )
    (raw / "second.json").write_text(
        json.dumps(
            {
                "speaker": "8",
                "snr": 18.0,
                "voice_activity": [[0.0, 4.0]],
                "book_meta": {"id": 20, "language": "English"},
            }
        ),
        encoding="utf-8",
    )
    write_wav(librilight / "vad" / "small" / "8" / "20" / "recording_0000.wav", 4)
    write_wav(librilight / "vad" / "small" / "8" / "20" / "second_0000.wav", 4)

    output = tmp_path / "output"
    report = build_manifests(
        tmp_path / "LibriSpeech",
        librilight,
        output,
        target_train_hours=11 / 3600,
        librilight_subsets=("small",),
        minimum_duration=1.0,
        maximum_duration=10.0,
        minimum_snr=8.0,
        maximum_speaker_hours=1.0,
        seed=1,
        num_workers=2,
    )
    assert report["status"] == "PASS"
    assert report["splits"]["train"]["items"] == 5
    assert report["splits"]["validation"]["items"] == 2
    assert report["splits"]["test"]["items"] == 2
    assert report["target_overshoot_seconds"] == 0
    assert report["configuration"]["num_workers"] == 2
    assert report["split_integrity"]["train_heldout_speaker_leakage"] == []
    train_rows = [
        json.loads(line) for line in (output / "train_audio.jsonl").read_text().splitlines()
    ]
    light_rows = {
        row["raw_recording_id"]: row for row in train_rows if row["corpus"] == "librilight"
    }
    assert light_rows["recording"]["snr"] == 12.0
    assert light_rows["second"]["snr"] == 18.0
    assert (output / "validation_audio.jsonl").is_file()
    assert (output / "test_audio.jsonl").is_file()
