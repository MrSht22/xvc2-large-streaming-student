from __future__ import annotations

import argparse
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .data import PhoneManifestDataset, collate
from .model import StreamingPhoneEncoder
from .teacher import load_teacher_with_loading_info, loading_failures, teacher_targets
from .validate import load_student_checkpoint


def audio_metadata(path: Path) -> tuple[int, int]:
    try:
        metadata = torchaudio.info(path)
        return metadata.sample_rate, metadata.num_frames
    except RuntimeError:
        if path.suffix.lower() != ".wav":
            raise
        with wave.open(str(path), "rb") as stream:
            return stream.getframerate(), stream.getnframes()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name:
        raise ValueError("Manifest name cannot be empty")
    return name, Path(raw_path).expanduser().resolve()


def identity(item: dict[str, Any]) -> tuple[str | None, str | None]:
    utterance_id = str(item.get("utterance_id", ""))
    parts = utterance_id.split("-")
    speaker = item.get("speaker_id") or (parts[0] if len(parts) >= 2 else None)
    chapter = (
        item.get("chapter_or_book_id")
        or item.get("chapter_id")
        or (parts[1] if len(parts) >= 3 else None)
    )
    return (str(speaker) if speaker is not None else None, str(chapter) if chapter else None)


def audit_manifests(
    manifests: list[tuple[str, Path]], vocab_size: int = 40, max_items: int | None = None
) -> dict[str, Any]:
    failures: list[str] = []
    summaries: dict[str, Any] = {}
    seen_ids: dict[str, str] = {}
    memberships: dict[str, dict[str, set[str]]] = {
        "speaker": defaultdict(set),
        "chapter": defaultdict(set),
    }
    for name, path in manifests:
        counters: Counter[str] = Counter()
        seconds = 0.0
        sample_rates: Counter[int] = Counter()
        rows_total = 0
        with path.open(encoding="utf-8") as stream:
            rows = []
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                rows_total += 1
                if max_items is None or len(rows) < max_items:
                    rows.append((line_number, line))
        for line_number, line in rows:
            counters["rows"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                failures.append(f"{name}:{line_number}:invalid_json")
                continue
            missing = {"utterance_id", "audio_path", "phone_ids"} - item.keys()
            if missing:
                failures.append(f"{name}:{line_number}:missing={sorted(missing)}")
                continue
            utterance_id = str(item["utterance_id"])
            if utterance_id in seen_ids:
                failures.append(
                    f"{name}:{line_number}:duplicate_id={utterance_id}:first={seen_ids[utterance_id]}"
                )
            else:
                seen_ids[utterance_id] = name
            phones = item["phone_ids"]
            if not isinstance(phones, list) or not phones:
                failures.append(f"{name}:{line_number}:empty_phone_ids")
            elif any(not isinstance(phone, int) or not 1 <= phone < vocab_size for phone in phones):
                failures.append(f"{name}:{line_number}:phone_id_out_of_range")
            audio_path = Path(item["audio_path"]).expanduser()
            try:
                sample_rate, num_frames = audio_metadata(audio_path)
                sample_rates[sample_rate] += 1
                audio_seconds = num_frames / sample_rate
                start_seconds = float(item.get("start_seconds", 0.0))
                duration_seconds = float(item.get("duration_seconds", audio_seconds))
                if start_seconds < 0 or duration_seconds <= 0:
                    failures.append(f"{name}:{line_number}:invalid_segment")
                elif start_seconds + duration_seconds > audio_seconds + 0.05:
                    failures.append(f"{name}:{line_number}:segment_out_of_bounds")
                else:
                    seconds += duration_seconds
                if num_frames <= 0:
                    failures.append(f"{name}:{line_number}:empty_audio")
            except Exception as error:
                failures.append(f"{name}:{line_number}:audio={type(error).__name__}")
            speaker, chapter = identity(item)
            if speaker:
                memberships["speaker"][speaker].add(name)
            if chapter:
                memberships["chapter"][f"{speaker}/{chapter}"].add(name)
        summaries[name] = {
            "path": str(path),
            "rows_total": rows_total,
            "rows_scanned": counters["rows"],
            "hours_scanned": seconds / 3600,
            "sample_rates": dict(sorted(sample_rates.items())),
        }
    leakage = {
        kind: sorted(key for key, splits in values.items() if len(splits) > 1)
        for kind, values in memberships.items()
    }
    for kind, values in leakage.items():
        if values:
            failures.append(f"split_{kind}_leakage={len(values)}")
    return {
        "manifests": summaries,
        "leakage": {
            name: {"count": len(values), "examples": values[:20]}
            for name, values in leakage.items()
        },
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def audit_loader(
    manifest: Path,
    sample_rate: int = 16_000,
    batch_size: int = 2,
    num_workers: int = 0,
    samples_per_region: int = 4,
) -> dict[str, Any]:
    dataset = PhoneManifestDataset(manifest, sample_rate=sample_rate)
    starts = [0]
    if dataset.first_segment_index is not None:
        starts.append(dataset.first_segment_index)
    indices = sorted(
        {
            index
            for start in starts
            for index in range(start, min(start + samples_per_region, len(dataset)))
        }
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
    )
    utterance_ids: list[str] = []
    sample_lengths: list[int] = []
    for batch in loader:
        utterance_ids.extend(batch["utterance_ids"])
        sample_lengths.extend(int(value) for value in batch["sample_lengths"])
    failures: list[str] = []
    if len(utterance_ids) != len(indices):
        failures.append("loader_item_count_mismatch")
    if not sample_lengths or min(sample_lengths) <= 0:
        failures.append("empty_waveform")
    if dataset.first_segment_index is None:
        failures.append("no_segmented_row_found")
    return {
        "manifest": str(manifest),
        "manifest_items": len(dataset),
        "first_segment_index": dataset.first_segment_index,
        "sampled_indices": indices,
        "sampled_utterance_ids": utterance_ids,
        "sample_lengths": sample_lengths,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def is_lfs_pointer(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size < 1024
        and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    )


def audit_teacher(
    teacher_path: Path, config_path: Path, device: str = "cpu", seconds: float = 0.5
) -> dict[str, Any]:
    failures: list[str] = []
    weight_files = list(teacher_path.glob("*.bin")) + list(teacher_path.glob("*.safetensors"))
    if not weight_files:
        failures.append("teacher_weight_file_missing")
    pointers = [str(path) for path in weight_files if is_lfs_pointer(path)]
    if pointers:
        failures.append(f"teacher_lfs_pointers={pointers}")
    details: dict[str, Any] = {"weight_files": [str(path) for path in weight_files]}
    if not failures:
        config = load_config(config_path)
        resolved_device = torch.device(device)
        teacher, loading_info = load_teacher_with_loading_info(teacher_path)
        details["loading_info"] = {
            name: loading_info.get(name, [])
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys")
        }
        failures.extend(loading_failures(loading_info))
        teacher = teacher.to(resolved_device)
        if teacher.config.vocab_size != config.model.vocab_size:
            failures.append(f"teacher_vocab_size={teacher.config.vocab_size}")
        if failures:
            return {**details, "failures": failures, "status": "FAIL"}
        samples = max(round(seconds * 16_000), 400)
        waveform = torch.zeros(1, samples, device=resolved_device)
        lengths = torch.tensor([samples], device=resolved_device)
        hidden, teacher_lengths = teacher_targets(
            teacher, waveform, lengths, config.distillation.teacher_layer
        )
        student = StreamingPhoneEncoder(config.model)
        student_lengths = student.output_lengths(lengths.cpu()).to(resolved_device)
        details.update(
            {
                "teacher_layer_shape": list(hidden.shape),
                "teacher_lengths": teacher_lengths.tolist(),
                "student_lengths": student_lengths.tolist(),
                "teacher_vocab_size": teacher.config.vocab_size,
            }
        )
        if hidden.ndim != 3 or hidden.shape[-1] != config.model.teacher_dim:
            failures.append(f"teacher_hidden_shape={list(hidden.shape)}")
        if not torch.equal(teacher_lengths, student_lengths):
            failures.append("teacher_student_frame_length_mismatch")
    return {**details, "failures": failures, "status": "PASS" if not failures else "FAIL"}


def streaming_forward(
    model: StreamingPhoneEncoder,
    waveform: torch.Tensor,
    chunk_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    state = model.init_streaming_state(waveform.device, waveform.dtype)
    outputs: dict[str, list[torch.Tensor]] = {
        "hidden_states": [],
        "distill_features": [],
        "phone_logits": [],
    }
    emitted_before_flush = 0
    pattern = (
        max(1, chunk_samples // 3),
        chunk_samples,
        chunk_samples + model.stride_samples // 2,
        max(1, chunk_samples - model.stride_samples // 3),
    )
    offset = 0
    chunk_index = 0
    while offset < waveform.shape[1]:
        count = min(pattern[chunk_index % len(pattern)], waveform.shape[1] - offset)
        chunk_output, state = model.forward_chunk(
            waveform[:, offset : offset + count], state, is_final=False
        )
        for name in outputs:
            outputs[name].append(chunk_output[name])
        emitted_before_flush += int(chunk_output["output_lengths"][0])
        offset += count
        chunk_index += 1
    flush_output, state = model.forward_chunk(waveform[:, :0], state, is_final=True)
    for name in outputs:
        outputs[name].append(flush_output[name])
    flush_frames = int(flush_output["output_lengths"][0])
    finalized_rejected = False
    try:
        model.forward_chunk(waveform[:, :0], state, is_final=True)
    except RuntimeError:
        finalized_rejected = True
    return (
        {name: torch.cat(values, dim=1) for name, values in outputs.items()},
        {
            "chunks": chunk_index,
            "emitted_before_flush": emitted_before_flush,
            "flush_frames": flush_frames,
            "state_finalized": state.finalized,
            "waveform_buffer_samples": state.waveform_buffer.shape[1],
            "feature_buffer_frames": state.feature_buffer.shape[1],
            "finalized_state_rejected_reuse": finalized_rejected,
        },
    )


def compare_streaming(
    model: StreamingPhoneEncoder,
    waveform: torch.Tensor,
    chunk_samples: int,
    tolerance: float,
) -> dict[str, Any]:
    if waveform.ndim == 1:
        waveform = waveform[None]
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("Expected one waveform with shape [1, samples]")
    sample_lengths = torch.tensor([waveform.shape[1]], device=waveform.device)
    with torch.inference_mode():
        offline = model(waveform, sample_lengths)
        chunked, streaming_state = streaming_forward(model, waveform, chunk_samples)
        reset_chunked, reset_state = streaming_forward(model, waveform, chunk_samples)

    expected_frames = int(offline["output_lengths"][0])
    output_frames = chunked["hidden_states"].shape[1]
    comparisons: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for name in ("hidden_states", "distill_features", "phone_logits"):
        if offline[name].shape != chunked[name].shape:
            failures.append(f"{name}_shape_mismatch")
            continue
        difference = (offline[name] - chunked[name]).abs().float()
        reset_difference = (chunked[name] - reset_chunked[name]).abs().float()
        comparisons[name] = {
            "max_abs_difference": float(difference.max()) if difference.numel() else 0.0,
            "mean_abs_difference": float(difference.mean()) if difference.numel() else 0.0,
            "reset_max_abs_difference": (
                float(reset_difference.max()) if reset_difference.numel() else 0.0
            ),
        }
        if comparisons[name]["max_abs_difference"] > tolerance:
            failures.append(f"{name}_difference_exceeds_tolerance")
        if comparisons[name]["reset_max_abs_difference"] > tolerance:
            failures.append(f"{name}_reset_difference_exceeds_tolerance")
    if output_frames != expected_frames:
        failures.append("output_frame_count_mismatch")
    for name, value in {
        "state_not_finalized": not streaming_state["state_finalized"],
        "waveform_buffer_not_empty": streaming_state["waveform_buffer_samples"] != 0,
        "feature_buffer_not_empty": streaming_state["feature_buffer_frames"] != 0,
        "finalized_state_accepted_reuse": not streaming_state["finalized_state_rejected_reuse"],
        "reset_state_not_finalized": not reset_state["state_finalized"],
    }.items():
        if value:
            failures.append(name)
    return {
        "samples": waveform.shape[1],
        "expected_frames": expected_frames,
        "streaming_frames": output_frames,
        **streaming_state,
        "comparisons": comparisons,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def audit_streaming(
    checkpoint: Path,
    config_path: Path,
    manifest: Path,
    device: str = "cpu",
    num_items: int = 8,
    chunk_samples: int = 3_200,
    tolerance: float = 2e-3,
) -> dict[str, Any]:
    if num_items <= 0 or chunk_samples <= 0 or tolerance < 0:
        raise ValueError("Streaming audit limits must be positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    config = load_config(config_path)
    model, step = load_student_checkpoint(checkpoint, config, resolved_device)
    dataset = PhoneManifestDataset(manifest)
    count = min(num_items, len(dataset))
    indices = (
        [0]
        if count == 1
        else sorted({round(index * (len(dataset) - 1) / (count - 1)) for index in range(count)})
    )
    items: list[dict[str, Any]] = []
    failures: list[str] = []
    torch.set_float32_matmul_precision("highest")
    if resolved_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    for index in indices:
        item = dataset[index]
        result = compare_streaming(
            model,
            item["waveform"].to(resolved_device)[None],
            chunk_samples,
            tolerance,
        )
        result.update({"index": index, "utterance_id": item["utterance_id"]})
        items.append(result)
        failures.extend(f"{item['utterance_id']}:{failure}" for failure in result["failures"])
    maximums: dict[str, float | None] = {}
    for name in ("hidden_states", "distill_features", "phone_logits"):
        values = [
            item["comparisons"][name]["max_abs_difference"]
            for item in items
            if name in item["comparisons"]
        ]
        maximums[name] = max(values) if len(values) == len(items) else None
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "manifest": str(manifest),
        "device": str(resolved_device),
        "items_checked": len(items),
        "chunk_samples": chunk_samples,
        "chunk_milliseconds": 1_000 * chunk_samples / dataset.sample_rate,
        "tolerance": tolerance,
        "maximum_abs_differences": maximums,
        "items": items,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def emit(report: dict[str, Any], label: str) -> None:
    print(json.dumps(report, sort_keys=True))
    print(f"{label}={report['status']}")
    if report["failures"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Student data and Teacher contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--manifest", action="append", required=True)
    manifest_parser.add_argument("--vocab-size", type=int, default=40)
    manifest_parser.add_argument("--max-items", type=int)
    loader_parser = subparsers.add_parser("loader")
    loader_parser.add_argument("--manifest", type=Path, required=True)
    loader_parser.add_argument("--sample-rate", type=int, default=16_000)
    loader_parser.add_argument("--batch-size", type=int, default=2)
    loader_parser.add_argument("--num-workers", type=int, default=0)
    loader_parser.add_argument("--samples-per-region", type=int, default=4)
    teacher_parser = subparsers.add_parser("teacher")
    teacher_parser.add_argument("--teacher", type=Path, required=True)
    teacher_parser.add_argument("--config", type=Path, required=True)
    teacher_parser.add_argument("--device", default="cpu")
    teacher_parser.add_argument("--seconds", type=float, default=0.5)
    streaming_parser = subparsers.add_parser("streaming")
    streaming_parser.add_argument("--checkpoint", type=Path, required=True)
    streaming_parser.add_argument("--config", type=Path, required=True)
    streaming_parser.add_argument("--manifest", type=Path, required=True)
    streaming_parser.add_argument("--device", default="auto")
    streaming_parser.add_argument("--num-items", type=int, default=8)
    streaming_parser.add_argument("--chunk-samples", type=int, default=3_200)
    streaming_parser.add_argument("--tolerance", type=float, default=2e-3)
    args = parser.parse_args()
    if args.command == "manifest":
        emit(
            audit_manifests(
                [parse_named_path(value) for value in args.manifest],
                args.vocab_size,
                args.max_items,
            ),
            "student_manifest_audit",
        )
    elif args.command == "teacher":
        emit(
            audit_teacher(args.teacher.resolve(), args.config.resolve(), args.device, args.seconds),
            "student_teacher_audit",
        )
    elif args.command == "streaming":
        emit(
            audit_streaming(
                args.checkpoint.expanduser().resolve(),
                args.config.expanduser().resolve(),
                args.manifest.expanduser().resolve(),
                args.device,
                args.num_items,
                args.chunk_samples,
                args.tolerance,
            ),
            "student_streaming_audit",
        )
    else:
        emit(
            audit_loader(
                args.manifest.expanduser().resolve(),
                args.sample_rate,
                args.batch_size,
                args.num_workers,
                args.samples_per_region,
            ),
            "student_loader_audit",
        )


if __name__ == "__main__":
    main()
