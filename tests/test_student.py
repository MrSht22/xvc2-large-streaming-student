import pickle
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from xvc2_student.audit import audit_loader, audit_manifests
from xvc2_student.build_audio_manifest import build_manifests, select_librilight
from xvc2_student.build_student_manifest import PhoneConversionPool, build_student_manifests
from xvc2_student.checkpoint import load_checkpoint, save_checkpoint
from xvc2_student.config import ExperimentConfig
from xvc2_student.data import (
    PhoneManifestDataset,
    StatefulDistributedBatchSampler,
    collate,
)
from xvc2_student.env_check import version_tuple
from xvc2_student.inspect_audio_corpora import combined_report, inspect_corpus
from xvc2_student.inspect_libriheavy import inspect_repository
from xvc2_student.losses import valid_feature_loss
from xvc2_student.model import StreamingPhoneEncoder
from xvc2_student.smoke import tiny_config
from xvc2_student.teacher import (
    configure_attention,
    loading_failures,
    optimized_teacher_targets,
    remap_legacy_position_conv,
    teacher_targets,
)
from xvc2_student.train import (
    collect_step_metrics,
    ddp_options,
    optimizer_step_due,
    override_max_steps,
)
from xvc2_student.validate import (
    collapse_ctc,
    ctc_statistics,
    edit_distance,
    feature_statistics,
    pack_validation_batches,
)


def tiny_teacher_config(
    stable_layer_norm: bool = True, attention_implementation: str = "eager"
) -> Wav2Vec2Config:
    config = Wav2Vec2Config(
        vocab_size=10,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=32,
        conv_dim=(8, 8),
        conv_stride=(2, 2),
        conv_kernel=(4, 2),
        conv_bias=True,
        feat_extract_norm="layer",
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=stable_layer_norm,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    configure_attention(config, attention_implementation)
    return config


def test_phone_manifest_dataset_random_access_and_audio_segments(tmp_path: Path) -> None:
    import json
    import struct
    import wave

    audio = tmp_path / "recording.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        samples = [1_000] * 8_000 + [-2_000] * 8_000 + [3_000] * 8_000
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    rows = [
        {
            "utterance_id": "middle",
            "audio_path": str(audio),
            "sample_rate": 8_000,
            "start_seconds": 1.0,
            "duration_seconds": 1.0,
            "phone_ids": [1, 2],
        },
        {
            "utterance_id": "last",
            "audio_path": str(audio),
            "sample_rate": 8_000,
            "start_seconds": 2.0,
            "duration_seconds": 0.5,
            "phone_ids": [3],
        },
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = PhoneManifestDataset(manifest, sample_rate=16_000)
    assert len(dataset) == 2
    assert not hasattr(dataset, "items")
    assert dataset[1]["utterance_id"] == "last"
    assert dataset[-2]["utterance_id"] == "middle"
    assert dataset[0]["waveform"].shape == (16_000,)
    assert dataset[1]["waveform"].shape == (8_000,)
    torch.testing.assert_close(
        dataset[0]["waveform"].mean(), torch.tensor(-2_000 / 32_768), atol=5e-5, rtol=0
    )
    torch.testing.assert_close(
        dataset[1]["waveform"].mean(), torch.tensor(3_000 / 32_768), atol=5e-5, rtol=0
    )
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[1]["utterance_id"] == "last"


@pytest.mark.skipif(sys.platform == "darwin", reason="sandboxed macOS disallows OpenMP SHM")
def test_phone_manifest_dataset_with_multiple_workers(tmp_path: Path) -> None:
    import json
    import wave

    audio = tmp_path / "recording.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 32_000)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "utterance_id": f"cut-{index}",
                    "audio_path": str(audio),
                    "sample_rate": 16_000,
                    "start_seconds": index * 0.25,
                    "duration_seconds": 0.25,
                    "phone_ids": [index + 1],
                }
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    loader = DataLoader(
        PhoneManifestDataset(manifest),
        batch_size=2,
        num_workers=2,
        collate_fn=collate,
    )
    batches = list(loader)
    assert [item for batch in batches for item in batch["utterance_ids"]] == [
        "cut-0",
        "cut-1",
        "cut-2",
        "cut-3",
    ]
    assert all(batch["waveform"].shape == (2, 4_000) for batch in batches)


def test_loader_audit_samples_full_audio_and_segment(tmp_path: Path) -> None:
    import json
    import wave

    audio = tmp_path / "recording.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 32_000)
    rows = [
        {
            "utterance_id": "librispeech/full",
            "corpus": "librispeech",
            "audio_path": str(audio),
            "sample_rate": 16_000,
            "duration_seconds": 2.0,
            "phone_ids": [1],
        },
        {
            "utterance_id": "libriheavy/cut",
            "corpus": "libriheavy",
            "audio_path": str(audio),
            "sample_rate": 16_000,
            "start_seconds": 0.5,
            "duration_seconds": 0.25,
            "phone_ids": [2],
        },
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = audit_loader(manifest, samples_per_region=1)
    assert report["status"] == "PASS"
    assert report["first_segment_index"] == 1
    assert report["sampled_indices"] == [0, 1]
    assert report["sample_lengths"] == [32_000, 4_000]


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


def test_student_reuses_convolution_features() -> None:
    model = StreamingPhoneEncoder(tiny_config()).eval()
    samples = model.receptive_field_samples + model.stride_samples * 7
    waveform = torch.randn(2, samples)
    lengths = torch.tensor([samples, samples])
    convolution = model.feature_extractor(waveform)
    direct = model(waveform, lengths)
    shared = model(waveform, lengths, convolution_features=convolution)
    for name in ("distill_features", "phone_logits", "output_lengths"):
        torch.testing.assert_close(direct[name], shared[name])


@pytest.mark.parametrize("stable_layer_norm", [False, True])
def test_optimized_teacher_targets_match_full_teacher(stable_layer_norm: bool) -> None:
    teacher = Wav2Vec2ForCTC(tiny_teacher_config(stable_layer_norm)).eval()
    waveform = torch.randn(2, 320)
    lengths = torch.tensor([320, 280])
    expected, expected_lengths = teacher_targets(teacher, waveform, lengths, layer=2)
    actual, actual_lengths, convolution = optimized_teacher_targets(
        teacher, waveform, lengths, layer=2
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lengths, expected_lengths)
    torch.testing.assert_close(convolution, teacher.wav2vec2.feature_extractor(waveform))
    assert not convolution.is_inference()


def test_sdpa_teacher_targets_match_eager() -> None:
    eager = Wav2Vec2ForCTC(tiny_teacher_config(attention_implementation="eager")).eval()
    sdpa = Wav2Vec2ForCTC(tiny_teacher_config(attention_implementation="sdpa")).eval()
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    waveform = torch.randn(2, 320)
    lengths = torch.tensor([320, 280])
    eager_target, eager_lengths, _ = optimized_teacher_targets(eager, waveform, lengths, 2)
    sdpa_target, sdpa_lengths, _ = optimized_teacher_targets(sdpa, waveform, lengths, 2)
    torch.testing.assert_close(sdpa_target, eager_target, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sdpa_lengths, eager_lengths)


def test_shared_teacher_convolution_supports_student_backward() -> None:
    teacher = Wav2Vec2ForCTC(tiny_teacher_config()).eval()
    student = StreamingPhoneEncoder(tiny_config())
    student.load_teacher_frontend(teacher)
    waveform = torch.randn(2, 320)
    lengths = torch.tensor([320, 280])
    _, _, convolution = optimized_teacher_targets(teacher, waveform, lengths, 2)
    output = student(waveform, lengths, convolution_features=convolution)
    output["distill_features"].square().mean().backward()
    assert student.input_projection.weight.grad is not None


def test_distributed_dynamic_batches_are_deterministic_and_balanced() -> None:
    from array import array

    sample_counts = array("I", [8, 9, 10, 11, 80, 90, 100, 110, 120, 130, 140, 150])
    samplers = [
        StatefulDistributedBatchSampler(
            sample_counts,
            rank=rank,
            world_size=2,
            seed=7,
            max_batch_samples=240,
            max_batch_items=4,
            bucket_size=6,
        )
        for rank in range(2)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert len(batches[0]) == len(batches[1])
    assert batches[0] == list(
        StatefulDistributedBatchSampler(
            sample_counts,
            rank=0,
            world_size=2,
            seed=7,
            max_batch_samples=240,
            max_batch_items=4,
            bucket_size=6,
        )
    )
    for rank_batches in batches:
        for batch in rank_batches:
            assert len(batch) <= 4
            assert (
                max(sample_counts[index] for index in batch) * len(batch) <= 240 or len(batch) == 1
            )
    samplers[0].advance()
    state = samplers[0].state_dict()
    restored = StatefulDistributedBatchSampler(
        sample_counts,
        rank=0,
        world_size=2,
        seed=7,
        max_batch_samples=240,
        max_batch_items=4,
        bucket_size=6,
    )
    restored.load_state_dict(state)
    assert list(restored) == batches[0][1:]


def test_validation_batches_cover_dataset_once_within_budget() -> None:
    sample_counts = [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41]
    batches_by_rank = [
        pack_validation_batches(
            sample_counts,
            rank=rank,
            world_size=3,
            max_batch_samples=80,
            max_batch_items=3,
        )
        for rank in range(3)
    ]
    indices = [index for batches in batches_by_rank for batch in batches for index in batch]
    assert sorted(indices) == list(range(len(sample_counts)))
    assert len(indices) == len(set(indices))
    for batches in batches_by_rank:
        for batch in batches:
            assert len(batch) <= 3
            assert max(sample_counts[index] for index in batch) * len(batch) <= 80


def test_ctc_collapse_and_edit_distance() -> None:
    assert collapse_ctc([0, 1, 1, 0, 1, 2, 2, 0]) == [1, 1, 2]
    assert edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert edit_distance([1, 2], [1, 2, 3]) == 1
    assert edit_distance([1, 2, 3], [1, 3]) == 1


def test_validation_feature_and_ctc_statistics() -> None:
    predicted = torch.randn(2, 4, 6)
    target = torch.randn(2, 4, 6)
    lengths = torch.tensor([4, 2])
    feature_sum, feature_frames = feature_statistics(predicted, target, lengths)
    expected = valid_feature_loss(predicted, target, lengths)
    assert int(feature_frames) == 6
    torch.testing.assert_close(feature_sum / feature_frames, expected)

    token_predictions = torch.tensor(
        [
            [1, 1, 0, 2],
            [1, 0, 3, 3],
        ]
    )
    logits = torch.full((2, 4, 4), -8.0)
    logits.scatter_(2, token_predictions.unsqueeze(-1), 8.0)
    ctc_sum, errors, reference_phones, exact = ctc_statistics(
        logits,
        targets=torch.tensor([1, 2, 1, 2]),
        input_lengths=torch.tensor([4, 4]),
        target_lengths=torch.tensor([2, 2]),
    )
    assert torch.isfinite(ctc_sum)
    assert errors == 1
    assert reference_phones == 4
    assert exact == 1


def test_training_runtime_helpers() -> None:
    config = ExperimentConfig()
    overridden = override_max_steps(config, 20)
    assert config.training.max_steps == 200_000
    assert overridden.training.max_steps == 20
    assert override_max_steps(config, None) is config
    assert [optimizer_step_due(index, 3) for index in range(6)] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    metrics = collect_step_metrics(
        torch.device("cpu"),
        world_size=1,
        audio_seconds=8.0,
        item_count=6,
        optimizer_steps=2,
        elapsed_seconds=2.0,
        feature=torch.tensor(1.0),
        phones=torch.tensor(2.0),
        gradient_norm=torch.tensor(3.0),
    )
    assert metrics["global_audio_seconds_per_second"] == 4.0
    assert metrics["mean_global_batch_size"] == 3.0
    assert metrics["memory_by_rank"] == []
    options = ddp_options(2)
    assert options["device_ids"] == [2]
    assert options["gradient_as_bucket_view"] is True
    assert "static_graph" not in options


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


def test_manifest_audit_counts_cut_duration(tmp_path: Path) -> None:
    import json
    import wave

    audio = tmp_path / "raw.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 160_000)
    manifest = tmp_path / "cut.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "utterance_id": "libriheavy/small/cut",
                "audio_path": str(audio),
                "speaker_id": "1",
                "chapter_or_book_id": "2",
                "start_seconds": 2.0,
                "duration_seconds": 3.0,
                "phone_ids": [1, 2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_manifests([("train", manifest)])
    assert report["status"] == "PASS"
    assert report["manifests"]["train"]["hours_scanned"] == 3 / 3600


def test_version_tuple() -> None:
    assert version_tuple("2.4.1+cu121") == (2, 4, 1)


def test_package_lazy_exports() -> None:
    import xvc2_student

    assert xvc2_student.ExperimentConfig is ExperimentConfig
    assert xvc2_student.StreamingPhoneEncoder is StreamingPhoneEncoder


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


def test_build_codec_audio_manifests(tmp_path: Path, capsys) -> None:
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
        progress=True,
        num_workers=2,
    )
    captured = capsys.readouterr()
    assert "collecting_librispeech_train num_workers=2" in captured.out
    assert "stage=librispeech/train-clean-100 status=START" in captured.err
    assert "stage=librilight/vad-metadata status=DONE" in captured.err
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

    all_output = tmp_path / "all-output"
    all_report = build_manifests(
        tmp_path / "LibriSpeech",
        librilight,
        all_output,
        target_train_hours=None,
        librilight_subsets=("small",),
        minimum_duration=1.0,
        maximum_duration=10.0,
        minimum_snr=8.0,
        maximum_speaker_hours=1.0,
        seed=1,
        num_workers=2,
    )
    assert all_report["configuration"]["selection_mode"] == "all_eligible_with_speaker_cap"
    assert all_report["target_overshoot_seconds"] is None
    assert all_report["splits"]["train"]["items"] == 5
    assert "all eligible audio" in (all_output / "report.md").read_text(encoding="utf-8")


def test_select_all_librilight_with_speaker_cap() -> None:
    candidates = [
        {"utterance_id": "a", "speaker_id": "1", "duration_seconds": 4.0},
        {"utterance_id": "b", "speaker_id": "1", "duration_seconds": 4.0},
        {"utterance_id": "c", "speaker_id": "1", "duration_seconds": 4.0},
        {"utterance_id": "d", "speaker_id": "2", "duration_seconds": 3.0},
    ]
    selected, counts = select_librilight(
        candidates,
        target_seconds=None,
        maximum_speaker_seconds=10.0,
        seed=1,
    )
    selected_speaker_one = [row for row in selected if row["speaker_id"] == "1"]
    assert len(selected_speaker_one) == 2
    assert sum(row["duration_seconds"] for row in selected_speaker_one) == 8.0
    assert {row["utterance_id"] for row in selected if row["speaker_id"] == "2"} == {"d"}
    assert counts["selection_mode"] == "all_eligible"
    assert counts["skipped_speaker_cap"] == 1


def test_build_student_manifests_from_codec_and_libriheavy(tmp_path: Path, monkeypatch) -> None:
    import gzip
    import json
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "cmudict",
        types.SimpleNamespace(
            dict=lambda: {
                "test": [["T", "EH1", "S", "T"]],
                "transcript": [["T", "R", "AE1", "N", "S", "K", "R", "IH2", "P", "T"]],
            }
        ),
    )

    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()
    audio_root = tmp_path / "LibriLight"
    raw_audio = audio_root / "raw" / "small" / "8" / "book.flac"
    raw_audio.parent.mkdir(parents=True)
    raw_audio.write_bytes(b"not-decoded")
    speech_audio = tmp_path / "speech.flac"
    speech_audio.write_bytes(b"not-decoded")

    speech_row = {
        "utterance_id": "librispeech/train-clean-100/1-2-0000",
        "corpus": "librispeech",
        "subset": "train-clean-100",
        "speaker_id": "1",
        "chapter_or_book_id": "2",
        "audio_path": str(speech_audio),
        "sample_rate": 16_000,
        "channels": 1,
        "num_frames": 16_000,
        "duration_seconds": 1.0,
        "text": "TEST TRANSCRIPT",
    }
    light_row = {
        "utterance_id": "librilight/vad/small/8/20/book_0000",
        "corpus": "librilight",
        "subset": "vad/small",
        "speaker_id": "8",
        "chapter_or_book_id": "20",
        "audio_path": str(tmp_path / "vad.flac"),
        "sample_rate": 16_000,
        "channels": 1,
        "num_frames": 32_000,
        "duration_seconds": 2.0,
        "snr": 12.0,
        "raw_recording_id": "book",
        "raw_metadata_path": str(tmp_path / "book.json"),
    }
    (codec_dir / "train_audio.jsonl").write_text(
        json.dumps(speech_row) + "\n" + json.dumps(light_row) + "\n", encoding="utf-8"
    )
    for split, subset, speaker in (
        ("validation", "dev-clean", "4"),
        ("test", "test-clean", "6"),
    ):
        row = dict(speech_row)
        row.update(
            {
                "utterance_id": f"librispeech/{subset}/{speaker}-2-0000",
                "subset": subset,
                "speaker_id": speaker,
            }
        )
        (codec_dir / f"{split}_audio.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    libriheavy_root = tmp_path / "LibriHeavy"
    libriheavy_root.mkdir()
    cut = {
        "id": "small/8/book_0000",
        "start": 1.25,
        "duration": 2.5,
        "supervisions": [
            {
                "recording_id": "small/8/book",
                "speaker": "8",
                "custom": {"texts": ["TEST TRANSCRIPT", "TEST TRANSCRIPT"]},
            }
        ],
        "recording": {
            "sampling_rate": 16_000,
            "sources": [{"type": "file", "source": "download/librilight/small/8/book.flac"}],
        },
    }
    manifest = libriheavy_root / "libriheavy_cuts_small.jsonl.gz"
    with gzip.open(manifest, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(cut) + "\n")

    output = tmp_path / "student"
    vocabulary = Path(__file__).parents[1] / "assets" / "ctc_gop_teacher_vocab.json"
    report = build_student_manifests(
        codec_dir,
        libriheavy_root,
        audio_root,
        output,
        vocabulary,
        target_train_hours=None,
        g2p_fallback=False,
    )

    assert report["status"] == "PASS"
    assert report["configuration"]["num_workers"] == 1
    assert report["codec_overlap"]["selected_recordings"] == 1
    assert report["codec_overlap"]["matched_recordings"] == 1
    assert report["codec_overlap"]["unmatched_recordings"] == 0
    assert report["codec_overlap"]["unmatched_examples"] == []
    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(train_rows) == 2
    heavy = next(row for row in train_rows if row["corpus"] == "libriheavy")
    assert heavy["audio_path"] == str(raw_audio.resolve())
    assert heavy["start_seconds"] == 1.25
    assert heavy["duration_seconds"] == 2.5
    assert heavy["normalized_text"] == "TEST TRANSCRIPT"
    assert heavy["text_source"] == "book"
    assert heavy["phone_ids"]
    assert (output / "validation.jsonl").is_file()
    assert (output / "test.jsonl").is_file()


def test_parallel_phone_conversion_preserves_order() -> None:
    vocabulary = {"<pad>": 0, "T": 1, "EH": 2, "S": 3}
    pronunciations = {
        "test": [["T", "EH1", "S", "T"]],
        "tests": [["T", "EH1", "S", "T", "S"]],
    }
    with PhoneConversionPool(
        vocabulary,
        g2p_fallback=False,
        num_workers=2,
        pronunciations=pronunciations,
    ) as pool:
        results = pool.convert_many(["TEST", "TESTS", "TEST"])
    assert [result[0] for result in results] == [
        [1, 2, 3, 1],
        [1, 2, 3, 1, 3],
        [1, 2, 3, 1],
    ]
