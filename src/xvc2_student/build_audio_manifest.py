from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from .inspect_audio_corpora import audio_metadata
from .inspect_libriheavy import AUDIO_SUFFIXES


LIBRISPEECH_TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
LIBRISPEECH_VALIDATION_SPLITS = ("dev-clean", "dev-other")
LIBRISPEECH_TEST_SPLITS = ("test-clean", "test-other")
SEGMENT_SUFFIX = re.compile(r"_\d+$")
METADATA_BATCH_SIZE = 512
PROGRESS_INTERVAL_SECONDS = 5.0


class ProgressPrinter:
    def __init__(self, stage: str, enabled: bool) -> None:
        self.stage = stage
        self.enabled = enabled
        self.started_at = time.monotonic()
        self.last_printed_at = self.started_at
        if enabled:
            self._print(0, None, "START", self.started_at)

    def update(self, processed: int, accepted: int | None = None) -> None:
        now = time.monotonic()
        if self.enabled and now - self.last_printed_at >= PROGRESS_INTERVAL_SECONDS:
            self._print(processed, accepted, "RUNNING", now)

    def finish(self, processed: int, accepted: int | None = None) -> None:
        if self.enabled:
            self._print(processed, accepted, "DONE", time.monotonic())

    def _print(self, processed: int, accepted: int | None, status: str, now: float) -> None:
        elapsed = max(now - self.started_at, 0.0)
        rate = processed / elapsed if elapsed > 0 else 0.0
        accepted_field = f" accepted={accepted}" if accepted is not None else ""
        print(
            f"progress stage={self.stage} status={status} processed={processed}"
            f"{accepted_field} elapsed_seconds={elapsed:.1f} files_per_second={rate:.1f}",
            file=sys.stderr,
            flush=True,
        )
        self.last_printed_at = now


def iter_audio_files(root: Path) -> Iterator[Path]:
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() in AUDIO_SUFFIXES:
                yield path


def batched(items: Iterator[Any], size: int = METADATA_BATCH_SIZE) -> Iterator[list[Any]]:
    while batch := list(islice(items, size)):
        yield batch


def map_audio_metadata(paths: list[Path], executor: ThreadPoolExecutor | None) -> list[Any]:
    if executor is None:
        return [audio_metadata(path) for path in paths]
    return list(executor.map(audio_metadata, paths))


def safe_audio_metadata(path: Path) -> dict[str, Any] | None:
    try:
        return audio_metadata(path)
    except Exception:  # pragma: no cover - backend and file dependent
        return None


def load_chapter_transcripts(directory: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in sorted(directory.glob("*.trans.txt")):
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
    return transcripts


def collect_librispeech_split(
    root: Path, split: str, progress: bool, num_workers: int
) -> list[dict[str, Any]]:
    split_root = root / split
    if not split_root.is_dir():
        raise NotADirectoryError(split_root)
    rows: list[dict[str, Any]] = []
    current_directory: Path | None = None
    transcripts: dict[str, str] = {}

    def items() -> Iterator[tuple[Path, str, str, str, str]]:
        nonlocal current_directory, transcripts
        for path in iter_audio_files(split_root):
            relative_path = path.relative_to(split_root)
            if len(relative_path.parts) < 3:
                raise ValueError(f"Unexpected LibriSpeech path: {path}")
            speaker_id, chapter_id = relative_path.parts[:2]
            if path.parent != current_directory:
                current_directory = path.parent
                transcripts = load_chapter_transcripts(path.parent)
            utterance_id = path.stem
            if utterance_id not in transcripts:
                raise ValueError(f"Missing transcript for {path}")
            yield path, speaker_id, chapter_id, utterance_id, transcripts[utterance_id]

    index = 0
    progress_printer = ProgressPrinter(f"librispeech/{split}", progress)
    executor_context = (
        ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else nullcontext(None)
    )
    with executor_context as executor:
        for batch in batched(items()):
            metadata_batch = map_audio_metadata([item[0] for item in batch], executor)
            for (path, speaker_id, chapter_id, utterance_id, text), metadata in zip(
                batch, metadata_batch, strict=True
            ):
                index += 1
                rows.append(
                    {
                        "utterance_id": f"librispeech/{split}/{utterance_id}",
                        "corpus": "librispeech",
                        "subset": split,
                        "speaker_id": speaker_id,
                        "chapter_or_book_id": chapter_id,
                        "audio_path": str(path),
                        "sample_rate": metadata["sample_rate"],
                        "channels": metadata["channels"],
                        "num_frames": metadata["num_frames"],
                        "duration_seconds": metadata["duration_seconds"],
                        "text": text,
                    }
                )
                if progress and index % 50_000 == 0:
                    print(f"librispeech_{split}_audio={index}", file=sys.stderr, flush=True)
            progress_printer.update(index)
    progress_printer.finish(index)
    return rows


def raw_metadata_index(
    root: Path, subsets: tuple[str, ...], progress: bool
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[tuple[str, str, str], Any]]:
    exact: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    fallback: dict[tuple[str, str, str], Any] = {}
    ambiguous = object()
    scanned = 0
    for subset in subsets:
        progress_printer = ProgressPrinter(f"librilight/raw-json/{subset}", progress)
        subset_scanned = 0
        subset_root = root / "raw" / subset
        if not subset_root.is_dir():
            raise NotADirectoryError(subset_root)
        for directory, dirnames, filenames in os.walk(subset_root):
            dirnames.sort()
            for filename in sorted(filenames):
                if not filename.endswith(".json"):
                    continue
                path = Path(directory) / filename
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object: {path}")
                speaker = str(value.get("speaker", ""))
                book_meta = value.get("book_meta")
                book_meta = book_meta if isinstance(book_meta, dict) else {}
                book_id = str(book_meta.get("id", ""))
                if not speaker:
                    raise ValueError(f"Missing speaker in {path}")
                item = {
                    "snr": float(value["snr"])
                    if isinstance(value.get("snr"), (int, float))
                    else None,
                    "raw_metadata_path": str(path),
                    "raw_recording_id": path.stem,
                    "book_id": book_id or None,
                    "language": book_meta.get("language"),
                }
                if book_id:
                    exact[(subset, speaker, book_id, path.stem)] = item
                fallback_key = (subset, speaker, path.stem)
                if fallback_key in fallback:
                    fallback[fallback_key] = ambiguous
                else:
                    fallback[fallback_key] = item
                scanned += 1
                subset_scanned += 1
                progress_printer.update(subset_scanned)
        progress_printer.finish(subset_scanned)
    return exact, {key: value for key, value in fallback.items() if value is not ambiguous}


def collect_librilight_candidates(
    root: Path,
    subsets: tuple[str, ...],
    minimum_duration: float,
    maximum_duration: float,
    minimum_snr: float,
    progress: bool,
    num_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact, fallback = raw_metadata_index(root, subsets, progress)
    candidates: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    subset_seconds: Counter[str] = Counter()
    pending: list[tuple[Path, str, str, str, dict[str, Any]]] = []
    for subset in subsets:
        progress_printer = ProgressPrinter(f"librilight/vad-index/{subset}", progress)
        subset_seen = 0
        subset_accepted = 0
        subset_root = root / "vad" / subset
        if not subset_root.is_dir():
            raise NotADirectoryError(subset_root)
        for path in iter_audio_files(subset_root):
            counters["audio_files_seen"] += 1
            subset_seen += 1
            progress_printer.update(subset_seen, subset_accepted)
            relative_path = path.relative_to(subset_root)
            if len(relative_path.parts) < 3:
                counters["rejected_path_layout"] += 1
                continue
            speaker_id, book_id = relative_path.parts[:2]
            recording_id = SEGMENT_SUFFIX.sub("", path.stem)
            source = exact.get((subset, speaker_id, book_id, recording_id))
            if source is None:
                source = fallback.get((subset, speaker_id, recording_id))
            if source is None:
                counters["rejected_missing_raw_metadata"] += 1
                continue
            snr = source["snr"]
            if snr is None:
                counters["rejected_missing_snr"] += 1
                continue
            if snr < minimum_snr:
                counters["rejected_low_snr"] += 1
                continue
            pending.append((path, subset, speaker_id, book_id, source))
            subset_accepted += 1
            progress_printer.update(subset_seen, subset_accepted)
        progress_printer.finish(subset_seen, subset_accepted)

    processed = 0
    progress_printer = ProgressPrinter("librilight/vad-metadata", progress)
    executor_context = (
        ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else nullcontext(None)
    )
    with executor_context as executor:
        for batch in batched(iter(pending)):
            paths = [item[0] for item in batch]
            if executor is None:
                metadata_batch = [safe_audio_metadata(path) for path in paths]
            else:
                metadata_batch = list(executor.map(safe_audio_metadata, paths))
            for (path, subset, speaker_id, book_id, source), metadata in zip(
                batch, metadata_batch, strict=True
            ):
                processed += 1
                if metadata is None:
                    counters["rejected_audio_metadata"] += 1
                    continue
                duration = metadata["duration_seconds"]
                if metadata["sample_rate"] != 16_000:
                    counters["rejected_sample_rate"] += 1
                    continue
                if metadata["channels"] != 1:
                    counters["rejected_channels"] += 1
                    continue
                if duration < minimum_duration:
                    counters["rejected_too_short"] += 1
                    continue
                if duration > maximum_duration:
                    counters["rejected_too_long"] += 1
                    continue
                row = {
                    "utterance_id": f"librilight/vad/{subset}/{speaker_id}/{book_id}/{path.stem}",
                    "corpus": "librilight",
                    "subset": f"vad/{subset}",
                    "speaker_id": speaker_id,
                    "chapter_or_book_id": book_id,
                    "audio_path": str(path),
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "num_frames": metadata["num_frames"],
                    "duration_seconds": duration,
                    "snr": source["snr"],
                    "raw_recording_id": source["raw_recording_id"],
                    "raw_metadata_path": source["raw_metadata_path"],
                }
                candidates.append(row)
                counters["accepted_candidates"] += 1
                subset_seconds[subset] += duration
                if progress and processed % 50_000 == 0:
                    print(
                        f"librilight_metadata_processed={processed},accepted={len(candidates)}",
                        file=sys.stderr,
                        flush=True,
                    )
            progress_printer.update(processed, len(candidates))
    progress_printer.finish(processed, len(candidates))
    return candidates, {
        "counts": dict(counters),
        "accepted_candidate_hours_by_subset": {
            key: seconds / 3600 for key, seconds in sorted(subset_seconds.items())
        },
        "raw_metadata_exact_keys": len(exact),
        "raw_metadata_fallback_keys": len(fallback),
    }


def total_seconds(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["duration_seconds"]) for row in rows)


def select_librilight(
    candidates: list[dict[str, Any]],
    target_seconds: float,
    maximum_speaker_seconds: float,
    seed: int,
    excluded_speakers: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    speaker_seconds: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    selected_seconds = 0.0
    excluded_speakers = excluded_speakers or set()
    for row in shuffled:
        duration = float(row["duration_seconds"])
        speaker = str(row["speaker_id"])
        if speaker in excluded_speakers:
            counters["skipped_heldout_librispeech_speaker"] += 1
            continue
        if speaker_seconds[speaker] + duration > maximum_speaker_seconds:
            counters["skipped_speaker_cap"] += 1
            continue
        selected.append(row)
        speaker_seconds[speaker] += duration
        selected_seconds += duration
        if selected_seconds >= target_seconds:
            break
    if selected_seconds < target_seconds:
        raise RuntimeError(
            f"Only selected {selected_seconds / 3600:.2f}h of requested "
            f"{target_seconds / 3600:.2f}h LibriLight"
        )
    counters["selected_items"] = len(selected)
    counters["selected_speakers"] = len(speaker_seconds)
    return selected, dict(counters)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    corpus_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    corpus_seconds: Counter[str] = Counter()
    subset_seconds: Counter[str] = Counter()
    speakers: set[str] = set()
    for row in rows:
        corpus = str(row["corpus"])
        subset = str(row["subset"])
        duration = float(row["duration_seconds"])
        corpus_counts[corpus] += 1
        subset_counts[subset] += 1
        corpus_seconds[corpus] += duration
        subset_seconds[subset] += duration
        speakers.add(f"{corpus}/{row['speaker_id']}")
    return {
        "items": len(rows),
        "hours": total_seconds(rows) / 3600,
        "unique_corpus_speakers": len(speakers),
        "items_by_corpus": dict(corpus_counts),
        "hours_by_corpus": {key: value / 3600 for key, value in corpus_seconds.items()},
        "items_by_subset": dict(subset_counts),
        "hours_by_subset": {key: value / 3600 for key, value in subset_seconds.items()},
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Codec Audio Manifest Selection",
        "",
        f"- Status: **{report['status']}**",
        f"- Target train hours: {report['configuration']['target_train_hours']:.2f}",
        f"- Actual train hours: {report['splits']['train']['hours']:.2f}",
        f"- Target overshoot: {report['target_overshoot_seconds']:.2f} seconds",
        f"- LibriLight subsets: {', '.join(report['configuration']['librilight_subsets'])}",
        f"- Minimum SNR: {report['configuration']['minimum_snr']:.2f} dB",
        "",
        "| Split | Items | Hours | Speakers |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in report["splits"].items():
        lines.append(
            f"| {name} | {summary['items']} | {summary['hours']:.2f} | "
            f"{summary['unique_corpus_speakers']} |"
        )
    lines.extend(
        [
            "",
            "These are source-audio selection manifests. Student hidden and speaker-target cache paths are added in the next preprocessing stage.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifests(
    librispeech_root: Path,
    librilight_root: Path,
    output_dir: Path,
    target_train_hours: float,
    librilight_subsets: tuple[str, ...],
    minimum_duration: float,
    maximum_duration: float,
    minimum_snr: float,
    maximum_speaker_hours: float,
    seed: int,
    progress: bool = False,
    num_workers: int = 1,
) -> dict[str, Any]:
    librispeech_root = librispeech_root.expanduser().resolve()
    librilight_root = librilight_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    print(f"collecting_librispeech_train num_workers={num_workers}", flush=True)
    train_librispeech = [
        row
        for split in LIBRISPEECH_TRAIN_SPLITS
        for row in collect_librispeech_split(librispeech_root, split, progress, num_workers)
    ]
    print(f"collecting_librispeech_validation num_workers={num_workers}", flush=True)
    validation = [
        row
        for split in LIBRISPEECH_VALIDATION_SPLITS
        for row in collect_librispeech_split(librispeech_root, split, progress, num_workers)
    ]
    print(f"collecting_librispeech_test num_workers={num_workers}", flush=True)
    test = [
        row
        for split in LIBRISPEECH_TEST_SPLITS
        for row in collect_librispeech_split(librispeech_root, split, progress, num_workers)
    ]
    librispeech_seconds = total_seconds(train_librispeech)
    target_seconds = target_train_hours * 3600
    if librispeech_seconds >= target_seconds:
        raise ValueError("Target train hours must exceed LibriSpeech train hours")
    print(f"collecting_librilight_candidates num_workers={num_workers}", flush=True)
    candidates, candidate_report = collect_librilight_candidates(
        librilight_root,
        librilight_subsets,
        minimum_duration,
        maximum_duration,
        minimum_snr,
        progress,
        num_workers,
    )
    heldout_speakers = {str(row["speaker_id"]) for row in validation + test}
    selected_librilight, selection_counts = select_librilight(
        candidates,
        target_seconds - librispeech_seconds,
        maximum_speaker_hours * 3600,
        seed,
        heldout_speakers,
    )
    train = train_librispeech + selected_librilight
    identifiers = [row["utterance_id"] for rows in (train, validation, test) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate utterance IDs across output manifests")
    train_speakers = {str(row["speaker_id"]) for row in train}
    leaked_speakers = sorted(train_speakers & heldout_speakers)
    if leaked_speakers:
        raise RuntimeError(f"Train/heldout speaker leakage: {leaked_speakers[:20]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_audio.jsonl",
        "validation": output_dir / "validation_audio.jsonl",
        "test": output_dir / "test_audio.jsonl",
    }
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_jsonl(paths[name], rows)
    report = {
        "schema_version": 1,
        "configuration": {
            "librispeech_root": str(librispeech_root),
            "librilight_root": str(librilight_root),
            "target_train_hours": target_train_hours,
            "librilight_subsets": list(librilight_subsets),
            "minimum_duration_seconds": minimum_duration,
            "maximum_duration_seconds": maximum_duration,
            "minimum_snr": minimum_snr,
            "maximum_librilight_hours_per_speaker": maximum_speaker_hours,
            "seed": seed,
            "num_workers": num_workers,
        },
        "candidate_filter": candidate_report,
        "selection_counts": selection_counts,
        "split_integrity": {
            "heldout_librispeech_speakers": len(heldout_speakers),
            "train_heldout_speaker_leakage": leaked_speakers,
        },
        "splits": {
            "train": split_summary(train),
            "validation": split_summary(validation),
            "test": split_summary(test),
        },
        "target_overshoot_seconds": total_seconds(train) - target_seconds,
        "output_paths": {name: str(path) for name, path in paths.items()},
        "status": "PASS",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic LibriSpeech plus LibriLight Codec audio manifests"
    )
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--librilight-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-train-hours", type=float, default=5_000)
    parser.add_argument("--librilight-subset", action="append", default=[])
    parser.add_argument("--min-duration-seconds", type=float, default=3.2)
    parser.add_argument("--max-duration-seconds", type=float, default=120.0)
    parser.add_argument("--min-snr", type=float, default=8.0)
    parser.add_argument("--max-librilight-hours-per-speaker", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    subsets = tuple(args.librilight_subset or ["small", "medium"])
    if (
        min(
            args.target_train_hours,
            args.min_duration_seconds,
            args.max_duration_seconds,
            args.max_librilight_hours_per_speaker,
        )
        <= 0
    ):
        parser.error("hours and duration limits must be positive")
    if args.min_duration_seconds > args.max_duration_seconds:
        parser.error("minimum duration cannot exceed maximum duration")
    if args.num_workers < 1:
        parser.error("num-workers must be at least 1")
    report = build_manifests(
        args.librispeech_root,
        args.librilight_root,
        args.output_dir,
        args.target_train_hours,
        subsets,
        args.min_duration_seconds,
        args.max_duration_seconds,
        args.min_snr,
        args.max_librilight_hours_per_speaker,
        args.seed,
        progress=True,
        num_workers=args.num_workers,
    )
    print(json.dumps({"status": report["status"], **report["splits"]}, sort_keys=True))
    print(f"report_json={args.output_dir.expanduser().resolve() / 'report.json'}")
    print(f"report_markdown={args.output_dir.expanduser().resolve() / 'report.md'}")
    print(f"codec_audio_manifest_build={report['status']}")


if __name__ == "__main__":
    main()
