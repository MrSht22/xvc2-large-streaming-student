from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import torchaudio

from .inspect_libriheavy import AUDIO_SUFFIXES


LIBRISPEECH_GROUPS = {
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
}
LIBRILIGHT_GROUPS = {"small", "medium", "large", "fine-tuning"}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audio_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = torchaudio.info(str(path))
        return {
            "sample_rate": int(metadata.sample_rate),
            "num_frames": int(metadata.num_frames),
            "channels": int(metadata.num_channels),
            "duration_seconds": float(metadata.num_frames / metadata.sample_rate),
        }
    except RuntimeError:
        if path.suffix.lower() != ".wav":
            raise
        with wave.open(str(path), "rb") as stream:
            sample_rate = stream.getframerate()
            num_frames = stream.getnframes()
            return {
                "sample_rate": sample_rate,
                "num_frames": num_frames,
                "channels": stream.getnchannels(),
                "duration_seconds": float(num_frames / sample_rate),
            }


def group_name(relative_path: Path) -> str:
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "<root>"


def identities(kind: str, relative_path: Path) -> tuple[str | None, str | None]:
    parts = relative_path.parts
    if len(parts) < 4:
        return None, None
    if kind == "librispeech":
        return parts[1], parts[2]
    if kind == "librilight":
        return parts[1], parts[2]
    raise ValueError(f"Unsupported corpus kind: {kind}")


def update_reservoir(
    paths: list[Path], candidate: Path, item_count: int, maximum: int, rng: random.Random
) -> None:
    if maximum <= 0:
        return
    if len(paths) < maximum:
        paths.append(candidate)
        return
    index = rng.randrange(item_count)
    if index < maximum:
        paths[index] = candidate


def transcript_preview(path: Path, maximum_lines: int = 2) -> list[str]:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    lines.append(line[:500])
                if len(lines) >= maximum_lines:
                    break
    except OSError as error:
        lines.append(f"{type(error).__name__}: {error}")
    return lines


def empty_group() -> dict[str, Any]:
    return {
        "files": 0,
        "bytes": 0,
        "audio_files": 0,
        "audio_bytes": 0,
        "audio_suffixes": Counter(),
        "text_files": 0,
        "speakers": set(),
        "containers": set(),
        "sample_paths": [],
        "audio_path_examples": [],
        "text_examples": [],
    }


def duration_summary(
    sample_paths: list[Path], total_audio_files: int, total_audio_bytes: int
) -> dict[str, Any]:
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    failures: list[str] = []
    examples: list[dict[str, Any]] = []
    sampled_bytes = 0
    for path in sorted(sample_paths):
        try:
            metadata = audio_metadata(path)
        except Exception as error:  # pragma: no cover - backend and file dependent
            failures.append(f"{path}: {type(error).__name__}: {error}")
            continue
        duration = metadata["duration_seconds"]
        if duration <= 0:
            failures.append(f"{path}: non_positive_duration")
            continue
        durations.append(duration)
        sample_rates[metadata["sample_rate"]] += 1
        channels[metadata["channels"]] += 1
        sampled_bytes += path.stat().st_size
        if len(examples) < 3:
            examples.append({"path": str(path), **metadata})

    estimated_hours = None
    confidence_hours = None
    estimate_kind = "unavailable"
    if durations:
        mean = statistics.fmean(durations)
        estimated_seconds = mean * total_audio_files
        estimated_hours = estimated_seconds / 3600
        estimate_kind = "sample_mean_times_file_count"
        if len(durations) == total_audio_files and not failures:
            estimate_kind = "exact_metadata_sum"
            estimated_hours = sum(durations) / 3600
            confidence_hours = [estimated_hours, estimated_hours]
        elif len(durations) > 1:
            standard_error = statistics.stdev(durations) / math.sqrt(len(durations))
            confidence_hours = [
                max(0.0, (mean - 1.96 * standard_error) * total_audio_files / 3600),
                (mean + 1.96 * standard_error) * total_audio_files / 3600,
            ]
    return {
        "metadata_samples_requested": len(sample_paths),
        "metadata_samples_valid": len(durations),
        "metadata_failures": failures[:20],
        "sample_rate_counts": dict(sorted(sample_rates.items())),
        "channel_counts": dict(sorted(channels.items())),
        "sample_duration_seconds": {
            "minimum": min(durations) if durations else None,
            "median": percentile(durations, 0.5),
            "p90": percentile(durations, 0.9),
            "maximum": max(durations) if durations else None,
            "mean": statistics.fmean(durations) if durations else None,
        },
        "sampled_audio_bytes": sampled_bytes,
        "total_audio_bytes": total_audio_bytes,
        "estimated_hours": estimated_hours,
        "estimated_hours_95pct_mean_ci": confidence_hours,
        "estimate_kind": estimate_kind,
        "examples": examples,
    }


def inspect_corpus(
    kind: str,
    root: Path,
    maximum_files: int,
    metadata_samples_per_group: int,
    seed: int = 1,
    progress: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    expected_groups = LIBRISPEECH_GROUPS if kind == "librispeech" else LIBRILIGHT_GROUPS
    groups: dict[str, dict[str, Any]] = {}
    suffixes: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    files_scanned = 0
    directories_scanned = 0
    truncated = False
    rng = random.Random(seed)

    for directory, dirnames, filenames in os.walk(root):
        directories_scanned += 1
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            if files_scanned >= maximum_files:
                truncated = True
                break
            files_scanned += 1
            path = Path(directory) / filename
            relative_path = path.relative_to(root)
            group = group_name(relative_path)
            summary = groups.setdefault(group, empty_group())
            try:
                file_bytes = path.stat().st_size
            except OSError:
                file_bytes = 0
            summary["files"] += 1
            summary["bytes"] += file_bytes
            suffix = path.suffix.lower() or "<none>"
            suffixes[suffix] += 1
            depth_counts[len(relative_path.parts)] += 1
            if suffix in AUDIO_SUFFIXES:
                summary["audio_files"] += 1
                summary["audio_bytes"] += file_bytes
                summary["audio_suffixes"][suffix] += 1
                speaker, container = identities(kind, relative_path)
                if speaker:
                    summary["speakers"].add(speaker)
                if container:
                    summary["containers"].add(f"{speaker}/{container}")
                if len(summary["audio_path_examples"]) < 3:
                    summary["audio_path_examples"].append(str(relative_path))
                update_reservoir(
                    summary["sample_paths"],
                    path,
                    summary["audio_files"],
                    metadata_samples_per_group,
                    rng,
                )
            elif suffix in {".txt", ".tsv", ".json", ".jsonl"}:
                summary["text_files"] += 1
                if len(summary["text_examples"]) < 3:
                    summary["text_examples"].append(
                        {"path": str(relative_path), "lines": transcript_preview(path)}
                    )
            if progress and files_scanned % 100_000 == 0:
                print(f"{kind}_files_scanned={files_scanned}", file=sys.stderr, flush=True)
        if truncated:
            break

    finalized_groups: dict[str, Any] = {}
    for name, summary in sorted(groups.items()):
        duration = duration_summary(
            summary["sample_paths"], summary["audio_files"], summary["audio_bytes"]
        )
        finalized_groups[name] = {
            "files": summary["files"],
            "bytes": summary["bytes"],
            "audio_files": summary["audio_files"],
            "audio_bytes": summary["audio_bytes"],
            "audio_suffix_counts": dict(summary["audio_suffixes"].most_common()),
            "text_files": summary["text_files"],
            "unique_speakers": len(summary["speakers"]),
            "unique_chapters_or_books": len(summary["containers"]),
            "audio_path_examples": summary["audio_path_examples"],
            "text_examples": summary["text_examples"],
            "duration": duration,
        }

    total_audio_files = sum(item["audio_files"] for item in finalized_groups.values())
    group_estimates = [
        item["duration"]["estimated_hours"]
        for item in finalized_groups.values()
        if item["duration"]["estimated_hours"] is not None
    ]
    recognized = sorted(set(finalized_groups) & expected_groups)
    warnings = []
    if truncated:
        warnings.append("file_scan_truncated")
    if not total_audio_files:
        warnings.append("no_audio_files_found")
    if not recognized:
        warnings.append("no_standard_top_level_groups_recognized")
    return {
        "kind": kind,
        "root": str(root),
        "expected_top_level_groups": sorted(expected_groups),
        "recognized_top_level_groups": recognized,
        "unrecognized_top_level_groups": sorted(set(finalized_groups) - expected_groups),
        "layout_hint": (
            "split/speaker/chapter/utterance.flac"
            if kind == "librispeech"
            else "subset/speaker/book/recording.flac"
        ),
        "files_scanned": files_scanned,
        "directories_scanned": directories_scanned,
        "file_scan_limit": maximum_files,
        "file_scan_truncated": truncated,
        "suffix_counts": dict(suffixes.most_common()),
        "relative_path_depth_counts": dict(sorted(depth_counts.items())),
        "audio_files": total_audio_files,
        "estimated_hours": sum(group_estimates) if group_estimates else None,
        "groups": finalized_groups,
        "warnings": warnings,
        "status": "PASS" if total_audio_files and recognized else "NEEDS_ATTENTION",
    }


def combined_report(librispeech: dict[str, Any], librilight: dict[str, Any]) -> dict[str, Any]:
    estimates = [
        item["estimated_hours"]
        for item in (librispeech, librilight)
        if item["estimated_hours"] is not None
    ]
    failures = [
        name
        for name, item in (("librispeech", librispeech), ("librilight", librilight))
        if item["status"] != "PASS"
    ]
    return {
        "schema_version": 1,
        "corpora": {"librispeech": librispeech, "librilight": librilight},
        "combined": {
            "audio_files": librispeech["audio_files"] + librilight["audio_files"],
            "estimated_hours": sum(estimates) if estimates else None,
        },
        "failures": failures,
        "status": "PASS" if not failures else "NEEDS_ATTENTION",
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# LibriSpeech and LibriLight Inspection",
        "",
        f"- Status: **{report['status']}**",
        "- Duration is estimated from deterministic per-group metadata samples unless every audio file was sampled.",
        "- No audio was decoded or modified.",
        "",
    ]
    for name, corpus in report["corpora"].items():
        hours = corpus["estimated_hours"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Root: `{corpus['root']}`",
                f"- Layout hint: `{corpus['layout_hint']}`",
                f"- Files scanned: {corpus['files_scanned']}",
                f"- Scan truncated: {corpus['file_scan_truncated']}",
                f"- Audio files: {corpus['audio_files']}",
                f"- Estimated hours: {hours:.2f}"
                if hours is not None
                else "- Estimated hours: unavailable",
                "",
                "| Group | Audio files | Speakers | Chapters/books | Estimated hours | Metadata samples |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group_name, group in corpus["groups"].items():
            duration = group["duration"]
            group_hours = duration["estimated_hours"]
            hours_text = f"{group_hours:.2f}" if group_hours is not None else "n/a"
            lines.append(
                f"| `{group_name}` | {group['audio_files']} | {group['unique_speakers']} | "
                f"{group['unique_chapters_or_books']} | {hours_text} | "
                f"{duration['metadata_samples_valid']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect LibriSpeech and LibriLight layouts and estimate their durations"
    )
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--librilight-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files-per-corpus", type=int, default=2_000_000)
    parser.add_argument("--metadata-samples-per-group", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.max_files_per_corpus, args.metadata_samples_per_group) <= 0:
        parser.error("scan limits must be positive")

    print(f"inspecting_librispeech={args.librispeech_root}", flush=True)
    librispeech = inspect_corpus(
        "librispeech",
        args.librispeech_root,
        args.max_files_per_corpus,
        args.metadata_samples_per_group,
        args.seed,
        progress=True,
    )
    print(f"inspecting_librilight={args.librilight_root}", flush=True)
    librilight = inspect_corpus(
        "librilight",
        args.librilight_root,
        args.max_files_per_corpus,
        args.metadata_samples_per_group,
        args.seed,
        progress=True,
    )
    report = combined_report(librispeech, librilight)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "librispeech_audio_files": librispeech["audio_files"],
                "librispeech_estimated_hours": librispeech["estimated_hours"],
                "librilight_audio_files": librilight["audio_files"],
                "librilight_estimated_hours": librilight["estimated_hours"],
            },
            sort_keys=True,
        )
    )
    print(f"report_json={json_path}")
    print(f"report_markdown={markdown_path}")
    print(f"speech_corpora_inspection={report['status']}")


if __name__ == "__main__":
    main()
