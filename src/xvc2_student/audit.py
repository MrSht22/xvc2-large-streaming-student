from __future__ import annotations

import argparse
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .config import load_config
from .model import StreamingPhoneEncoder
from .teacher import load_teacher, teacher_targets


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
    chapter = item.get("chapter_id") or (parts[1] if len(parts) >= 3 else None)
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
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = rows[:max_items] if max_items is not None else rows
        for line_number, line in enumerate(selected, 1):
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
                seconds += num_frames / sample_rate
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
            "rows_total": len(rows),
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
        teacher = load_teacher(teacher_path).to(resolved_device)
        if teacher.config.vocab_size != config.model.vocab_size:
            failures.append(f"teacher_vocab_size={teacher.config.vocab_size}")
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
    teacher_parser = subparsers.add_parser("teacher")
    teacher_parser.add_argument("--teacher", type=Path, required=True)
    teacher_parser.add_argument("--config", type=Path, required=True)
    teacher_parser.add_argument("--device", default="cpu")
    teacher_parser.add_argument("--seconds", type=float, default=0.5)
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
    else:
        emit(
            audit_teacher(args.teacher.resolve(), args.config.resolve(), args.device, args.seconds),
            "student_teacher_audit",
        )


if __name__ == "__main__":
    main()
