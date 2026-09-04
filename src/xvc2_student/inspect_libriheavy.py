from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, TextIO


AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a"}
EXPECTED_MANIFESTS = (
    "libriheavy_cuts_small.jsonl.gz",
    "libriheavy_cuts_medium.jsonl.gz",
    "libriheavy_cuts_large.jsonl.gz",
    "libriheavy_cuts_dev.jsonl.gz",
    "libriheavy_cuts_test_clean.jsonl.gz",
    "libriheavy_cuts_test_other.jsonl.gz",
)


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_rows(path: Path, maximum: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    with open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if maximum is not None and line_number > maximum:
                break
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def inventory(root: Path, maximum_files: int) -> dict[str, Any]:
    suffixes: Counter[str] = Counter()
    audio: Counter[str] = Counter()
    manifests: list[Path] = []
    kaldi: list[Path] = []
    archives: list[Path] = []
    scanned = 0
    truncated = False
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            scanned += 1
            if scanned > maximum_files:
                truncated = True
                break
            path = Path(directory) / filename
            lowered = filename.lower()
            suffix = ".jsonl.gz" if lowered.endswith(".jsonl.gz") else path.suffix.lower()
            suffixes[suffix or "<none>"] += 1
            if path.suffix.lower() in AUDIO_SUFFIXES:
                audio[path.suffix.lower()] += 1
            if lowered.endswith((".jsonl", ".jsonl.gz")):
                manifests.append(path)
            if filename in {"wav.scp", "segments", "text", "utt2spk", "spk2utt"}:
                kaldi.append(path)
            if lowered.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                archives.append(path)
        if truncated:
            break
    return {
        "files_scanned": min(scanned, maximum_files),
        "truncated": truncated,
        "suffix_counts": dict(suffixes.most_common()),
        "audio_file_counts": dict(audio.most_common()),
        "manifest_paths": [relative(path, root) for path in sorted(manifests)],
        "kaldi_paths": [relative(path, root) for path in sorted(kaldi)],
        "archive_paths": [relative(path, root) for path in sorted(archives)],
    }


def text_candidates(supervision: dict[str, Any]) -> dict[str, str | None]:
    custom = supervision.get("custom")
    custom = custom if isinstance(custom, dict) else {}
    texts = custom.get("texts")
    texts = texts if isinstance(texts, list) else []
    return {
        "text": supervision.get("text") if isinstance(supervision.get("text"), str) else None,
        "book_text": texts[0] if len(texts) > 0 and isinstance(texts[0], str) else None,
        "asr_text": texts[1] if len(texts) > 1 and isinstance(texts[1], str) else None,
    }


def source_path(row: dict[str, Any]) -> str | None:
    recording = row.get("recording")
    if not isinstance(recording, dict):
        return None
    sources = recording.get("sources")
    if not isinstance(sources, list) or not sources or not isinstance(sources[0], dict):
        return None
    source = sources[0].get("source")
    return source if isinstance(source, str) else None


def resolve_audio(source: str, root: Path, audio_roots: list[Path]) -> Path | None:
    path = Path(source).expanduser()
    if path.is_absolute() and path.is_file():
        return path.resolve()
    candidates = [root / path]
    for audio_root in audio_roots:
        candidates.append(audio_root / path)
        prefix = Path("download/librilight")
        try:
            candidates.append(audio_root / path.relative_to(prefix))
        except ValueError:
            pass
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def inspect_manifest(
    path: Path, root: Path, audio_roots: list[Path], maximum_rows: int | None
) -> dict[str, Any]:
    fields: Counter[str] = Counter()
    supervision_fields: Counter[str] = Counter()
    speakers: set[str] = set()
    recordings: set[str] = set()
    text_counts: Counter[str] = Counter()
    source_prefixes: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    rows = 0
    valid_rows = 0
    parse_failures: list[str] = []
    duration_seconds = 0.0
    resolved_audio = 0
    try:
        iterator = iter_rows(path, maximum_rows)
        for line_number, row in iterator:
            rows += 1
            fields.update(row.keys())
            duration = row.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                duration_seconds += float(duration)
            supervisions = row.get("supervisions")
            if not isinstance(supervisions, list) or not supervisions:
                parse_failures.append(f"line {line_number}: missing supervisions")
                continue
            supervision = supervisions[0]
            if not isinstance(supervision, dict):
                parse_failures.append(f"line {line_number}: invalid supervision")
                continue
            supervision_fields.update(supervision.keys())
            speaker = supervision.get("speaker")
            recording_id = supervision.get("recording_id") or row.get("recording_id")
            if speaker is not None:
                speakers.add(str(speaker))
            if recording_id is not None:
                recordings.add(str(recording_id))
            texts = text_candidates(supervision)
            text_counts.update(name for name, value in texts.items() if value)
            source = source_path(row)
            resolved = resolve_audio(source, root, audio_roots) if source else None
            if source:
                source_prefixes[source.split("/", 1)[0]] += 1
            if resolved:
                resolved_audio += 1
            valid_rows += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "cut_id": row.get("id"),
                        "recording_id": recording_id,
                        "speaker_id": speaker,
                        "start": row.get("start"),
                        "duration": duration,
                        "texts": texts,
                        "audio_source": source,
                        "resolved_audio_path": str(resolved) if resolved else None,
                    }
                )
    except (OSError, json.JSONDecodeError) as error:
        parse_failures.append(f"{type(error).__name__}: {error}")
    return {
        "path": relative(path, root),
        "compressed_bytes": path.stat().st_size,
        "rows_scanned": rows,
        "scan_limit": maximum_rows,
        "valid_rows": valid_rows,
        "sampled_hours": duration_seconds / 3600,
        "unique_speakers_sampled": len(speakers),
        "unique_recordings_sampled": len(recordings),
        "audio_sources_present": sum(source_prefixes.values()),
        "audio_sources_resolved": resolved_audio,
        "top_level_fields": dict(fields.most_common()),
        "supervision_fields": dict(supervision_fields.most_common()),
        "text_field_counts": dict(text_counts.most_common()),
        "source_prefixes": dict(source_prefixes.most_common()),
        "parse_failures": parse_failures[:20],
        "examples": examples,
    }


def markdown_report(report: dict[str, Any]) -> str:
    inventory_report = report["inventory"]
    lines = [
        "# LibriHeavy Repository Inspection",
        "",
        f"- Root: `{report['root']}`",
        f"- Files scanned: {inventory_report['files_scanned']}",
        f"- Inventory truncated: {inventory_report['truncated']}",
        f"- Manifests found: {len(report['manifests'])}",
        f"- Status: **{report['status']}**",
        "",
        "| Manifest | Rows sampled | Sampled hours | Speakers | Audio resolved | Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for manifest in report["manifests"]:
        lines.append(
            f"| `{manifest['path']}` | {manifest['rows_scanned']} | "
            f"{manifest['sampled_hours']:.2f} | {manifest['unique_speakers_sampled']} | "
            f"{manifest['audio_sources_resolved']}/{manifest['audio_sources_present']} | "
            f"{len(manifest['parse_failures'])} |"
        )
    lines.extend(
        [
            "",
            "The report is an inspection result, not a training manifest. No audio was decoded or modified.",
            "",
        ]
    )
    return "\n".join(lines)


def inspect_repository(
    root: Path,
    audio_roots: list[Path],
    maximum_files: int,
    maximum_rows: int | None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    inventory_report = inventory(root, maximum_files)
    manifest_paths = [root / path for path in inventory_report["manifest_paths"]]
    manifests = [
        inspect_manifest(path, root, audio_roots, maximum_rows)
        for path in manifest_paths
        if "libriheavy" in path.name.lower() or "cuts" in path.name.lower()
    ]
    found_names = {Path(item["path"]).name for item in manifests}
    failures = []
    if not manifests:
        failures.append("no_lhotse_cut_manifests_found")
    if manifests and not any(item["valid_rows"] for item in manifests):
        failures.append("no_valid_lhotse_cut_rows")
    return {
        "schema_version": 1,
        "root": str(root),
        "audio_roots": [str(path.expanduser().resolve()) for path in audio_roots],
        "inventory": inventory_report,
        "expected_manifest_names": list(EXPECTED_MANIFESTS),
        "missing_expected_manifest_names": sorted(set(EXPECTED_MANIFESTS) - found_names),
        "manifests": manifests,
        "failures": failures,
        "status": "PASS" if not failures else "NEEDS_ATTENTION",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LibriHeavy checkout and manifests")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=500_000)
    parser.add_argument("--max-rows-per-manifest", type=int, default=2_000)
    args = parser.parse_args()
    if min(args.max_files, args.max_rows_per_manifest) <= 0:
        parser.error("scan limits must be positive")
    report = inspect_repository(
        args.root,
        args.audio_root,
        args.max_files,
        args.max_rows_per_manifest,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "manifests": len(report["manifests"])}))
    print(f"report_json={output_dir / 'report.json'}")
    print(f"report_markdown={output_dir / 'report.md'}")
    print(f"libriheavy_repository_inspection={report['status']}")


if __name__ == "__main__":
    main()
