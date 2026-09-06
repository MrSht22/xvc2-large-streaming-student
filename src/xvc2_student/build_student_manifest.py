from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterator

from .inspect_libriheavy import open_text, source_path, text_candidates


WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STRESS_PATTERN = re.compile(r"\d")
DEFAULT_TARGET_TRAIN_HOURS = 5_000.0
DEFAULT_MAXIMUM_SPEAKER_HOURS = 30.0
PHONE_BATCH_SIZE = 1_024
CODEC_FILENAMES = {
    "train": "train_audio.jsonl",
    "validation": "validation_audio.jsonl",
    "test": "test_audio.jsonl",
}


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def load_phone_vocabulary(path: Path) -> dict[str, int]:
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    result = {str(phone): int(identifier) for phone, identifier in vocabulary.items()}
    if result.get("<pad>") != 0:
        raise ValueError("Teacher vocabulary must use <pad> as CTC blank ID 0")
    identifiers = sorted(result.values())
    if identifiers != list(range(len(identifiers))):
        raise ValueError("Teacher vocabulary IDs must be contiguous from zero")
    return result


def create_g2p_fallback(enabled: bool):
    if not enabled:
        return None
    try:
        from g2p_en import G2p
    except ImportError as error:
        raise RuntimeError("Install g2p-en before enabling OOV fallback") from error
    try:
        g2p = G2p()
    except LookupError as error:
        raise RuntimeError(
            "g2p-en is missing its NLTK resources; install the offline NLTK bundle first"
        ) from error
    g2p.predict = functools.lru_cache(maxsize=None)(g2p.predict)
    return g2p


class PhoneConverter:
    def __init__(
        self,
        vocabulary: dict[str, int],
        g2p_fallback: bool,
        pronunciations: dict[str, list[list[str]]] | None = None,
    ) -> None:
        if pronunciations is None:
            try:
                import cmudict
            except ImportError as error:
                raise RuntimeError("Install cmudict before building Student manifests") from error
            pronunciations = cmudict.dict()
        self.pronunciations = pronunciations
        self.vocabulary = vocabulary
        self.g2p = create_g2p_fallback(g2p_fallback)

    def convert(self, text: str) -> tuple[list[int] | None, str | None, int]:
        words = WORD_PATTERN.findall(text.replace("’", "'"))
        if not words:
            return None, "no_words", 0
        phone_ids: list[int] = []
        fallback_words = 0
        for word in words:
            lookup_word = word.lower()
            pronunciations = self.pronunciations.get(lookup_word)
            if not pronunciations:
                if self.g2p is None:
                    return None, f"oov_word:{lookup_word}", fallback_words
                try:
                    predicted = self.g2p.predict(lookup_word)
                except LookupError as error:
                    raise RuntimeError(
                        "g2p-en could not load an NLTK resource from NLTK_DATA"
                    ) from error
                if not predicted:
                    return None, f"g2p_failed:{lookup_word}", fallback_words
                pronunciations = [predicted]
                fallback_words += 1
            for stressed_phone in pronunciations[0]:
                phone = STRESS_PATTERN.sub("", stressed_phone).upper()
                if phone not in self.vocabulary:
                    return None, f"phone_not_in_teacher_vocab:{phone}", fallback_words
                phone_ids.append(self.vocabulary[phone])
        return phone_ids, None, fallback_words


_PHONE_WORKER: PhoneConverter | None = None


def initialize_phone_worker(
    vocabulary: dict[str, int],
    g2p_fallback: bool,
    pronunciations: dict[str, list[list[str]]] | None,
) -> None:
    global _PHONE_WORKER
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _PHONE_WORKER = PhoneConverter(vocabulary, g2p_fallback, pronunciations)


def convert_phone_worker(text: str) -> tuple[list[int] | None, str | None, int]:
    if _PHONE_WORKER is None:  # pragma: no cover - executor contract
        raise RuntimeError("Phone worker was not initialized")
    return _PHONE_WORKER.convert(text)


class PhoneConversionPool:
    def __init__(
        self,
        vocabulary: dict[str, int],
        g2p_fallback: bool,
        num_workers: int,
        local_converter: PhoneConverter | None = None,
        pronunciations: dict[str, list[list[str]]] | None = None,
    ) -> None:
        self.num_workers = num_workers
        self.local_converter = local_converter or PhoneConverter(
            vocabulary, g2p_fallback, pronunciations
        )
        self.executor = (
            ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=get_context("spawn"),
                initializer=initialize_phone_worker,
                initargs=(vocabulary, g2p_fallback, pronunciations),
            )
            if num_workers > 1
            else None
        )

    def convert_many(self, texts: list[str]) -> list[tuple[list[int] | None, str | None, int]]:
        if self.executor is None:
            return [self.local_converter.convert(text) for text in texts]
        chunksize = max(1, len(texts) // (self.num_workers * 4))
        return list(self.executor.map(convert_phone_worker, texts, chunksize=chunksize))

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> PhoneConversionPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class CodecRecordingIndex:
    exact: dict[tuple[str, str, str, str], str] = field(default_factory=dict)
    fallback: dict[tuple[str, str, str], str | None] = field(default_factory=dict)
    identifiers: set[str] = field(default_factory=set)
    selected_vad_seconds: Counter[str] = field(default_factory=Counter)

    def add(self, row: dict[str, Any]) -> None:
        subset = str(row["subset"]).removeprefix("vad/")
        speaker = str(row["speaker_id"])
        book = str(row.get("chapter_or_book_id", ""))
        recording = str(row["raw_recording_id"])
        identifier = f"{subset}/{speaker}/{recording}"
        self.identifiers.add(identifier)
        self.exact[(subset, speaker, book, recording)] = identifier
        fallback_key = (subset, speaker, recording)
        previous = self.fallback.get(fallback_key, identifier)
        self.fallback[fallback_key] = identifier if previous == identifier else None
        self.selected_vad_seconds[identifier] += float(row["duration_seconds"])

    def match(self, subset: str, speaker: str, book: str | None, recording: str) -> str | None:
        if book is not None:
            exact = self.exact.get((subset, speaker, book, recording))
            if exact is not None:
                return exact
        return self.fallback.get((subset, speaker, recording))


def load_codec_selection(
    codec_manifest_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], CodecRecordingIndex, dict[str, Any]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    recording_index = CodecRecordingIndex()
    counts: Counter[str] = Counter()
    for split, filename in CODEC_FILENAMES.items():
        path = codec_manifest_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        speech_rows: list[dict[str, Any]] = []
        for line_number, row in iter_jsonl(path):
            corpus = row.get("corpus")
            if corpus == "librispeech":
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError(f"{path}:{line_number} has no LibriSpeech text")
                speech_rows.append(row)
                counts[f"{split}_librispeech_rows"] += 1
            elif split == "train" and corpus == "librilight":
                required = {
                    "subset",
                    "speaker_id",
                    "raw_recording_id",
                    "duration_seconds",
                }
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
                recording_index.add(row)
                counts["train_librilight_vad_rows"] += 1
            else:
                raise ValueError(f"Unexpected corpus={corpus!r} in {split} manifest")
        rows_by_split[split] = speech_rows
    return rows_by_split, recording_index, dict(counts)


def discover_libriheavy_manifests(root: Path, subsets: set[str]) -> list[Path]:
    paths: list[Path] = []
    for subset in sorted(subsets):
        filename = f"libriheavy_cuts_{subset}.jsonl.gz"
        preferred = (root / filename, root / "lhotse" / filename)
        path = next((candidate for candidate in preferred if candidate.is_file()), None)
        if path is None:
            candidates = sorted(root.rglob(filename), key=lambda item: (len(item.parts), str(item)))
            if not candidates:
                raise FileNotFoundError(f"Could not find {filename} under {root}")
            path = candidates[0]
        paths.append(path)
    return paths


def source_identity(source: str) -> tuple[str, str, str | None, str] | None:
    parts = list(Path(source).parts)
    for marker in ("librilight", "raw"):
        if marker in parts:
            parts = parts[parts.index(marker) + 1 :]
            break
    if len(parts) < 3:
        return None
    subset, speaker = parts[:2]
    recording = Path(parts[-1]).stem
    book = parts[2] if len(parts) >= 4 else None
    return subset, speaker, book, recording


def resolve_librilight_audio(source: str, librilight_root: Path) -> Path | None:
    source_path_value = Path(source).expanduser()
    if source_path_value.is_absolute() and source_path_value.is_file():
        return source_path_value.resolve()
    parts = list(source_path_value.parts)
    if "librilight" in parts:
        parts = parts[parts.index("librilight") + 1 :]
    if parts and parts[0] == "raw":
        parts = parts[1:]
    relative = Path(*parts)
    candidates = (librilight_root / "raw" / relative, librilight_root / relative)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def stable_priority(seed: int, utterance_id: str) -> str:
    payload = f"{seed}\0{utterance_id}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def choose_text(texts: dict[str, str | None], source: str) -> str | None:
    preferences = {
        "book": ("book_text", "text", "asr_text"),
        "asr": ("asr_text", "text", "book_text"),
        "supervision": ("text", "book_text", "asr_text"),
    }
    return next((texts[key] for key in preferences[source] if texts.get(key)), None)


def cut_payload(
    row: dict[str, Any],
    manifest_subset: str,
    recording_token: str,
    resolved_audio: Path,
    text_source: str,
) -> dict[str, Any] | None:
    supervisions = row.get("supervisions")
    if not isinstance(supervisions, list) or not supervisions:
        return None
    supervision = supervisions[0]
    if not isinstance(supervision, dict):
        return None
    texts = text_candidates(supervision)
    normalized_text = choose_text(texts, text_source)
    duration = row.get("duration")
    start = row.get("start", 0.0)
    if not normalized_text or not isinstance(duration, (int, float)) or duration <= 0:
        return None
    if not isinstance(start, (int, float)) or start < 0:
        return None
    recording_id = str(supervision.get("recording_id") or row.get("recording_id") or "")
    speaker = str(supervision.get("speaker") or recording_token.split("/", 2)[1])
    cut_id = str(row.get("id") or f"{recording_id}@{float(start):.3f}")
    recording = row.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    return {
        "utterance_id": f"libriheavy/{manifest_subset}/{cut_id}",
        "corpus": "libriheavy",
        "subset": manifest_subset,
        "speaker_id": speaker,
        "chapter_or_book_id": source_identity(source_path(row) or "")[2],
        "recording_id": recording_id,
        "codec_raw_recording": recording_token,
        "audio_path": str(resolved_audio),
        "sample_rate": recording.get("sampling_rate"),
        "start_seconds": float(start),
        "duration_seconds": float(duration),
        "normalized_text": normalized_text,
        "text_source": text_source,
        "book_text": texts["book_text"],
        "asr_text": texts["asr_text"],
    }


def index_libriheavy_candidates(
    connection: sqlite3.Connection,
    manifests: list[Path],
    recording_index: CodecRecordingIndex,
    librilight_root: Path,
    text_source: str,
    seed: int,
    progress: bool,
) -> tuple[dict[str, Any], set[str]]:
    connection.execute(
        "CREATE TABLE candidates (priority TEXT, utterance_id TEXT PRIMARY KEY, "
        "speaker_id TEXT, duration_seconds REAL, payload TEXT)"
    )
    counts: Counter[str] = Counter()
    matched_recordings: set[str] = set()
    resolved_audio_cache: dict[str, Path | None] = {}
    for manifest in manifests:
        name = manifest.name.removeprefix("libriheavy_cuts_").removesuffix(".jsonl.gz")
        batch: list[tuple[str, str, str, float, str]] = []
        for line_number, row in iter_jsonl(manifest):
            counts["cuts_seen"] += 1
            source = source_path(row)
            identity = source_identity(source) if source else None
            if identity is None:
                counts["rejected_source_identity"] += 1
                continue
            subset, source_speaker, book, recording = identity
            supervisions = row.get("supervisions")
            supervision = supervisions[0] if isinstance(supervisions, list) and supervisions else {}
            speaker = str(supervision.get("speaker") or source_speaker)
            token = recording_index.match(subset, speaker, book, recording)
            if token is None:
                counts["rejected_not_selected_by_codec"] += 1
                continue
            if source not in resolved_audio_cache:
                resolved_audio_cache[source] = resolve_librilight_audio(source, librilight_root)
            resolved = resolved_audio_cache[source]
            if resolved is None:
                counts["rejected_unresolved_audio"] += 1
                continue
            payload = cut_payload(row, name, token, resolved, text_source)
            if payload is None:
                counts["rejected_invalid_cut_or_text"] += 1
                continue
            utterance_id = payload["utterance_id"]
            batch.append(
                (
                    stable_priority(seed, utterance_id),
                    utterance_id,
                    str(payload["speaker_id"]),
                    float(payload["duration_seconds"]),
                    json.dumps(payload, sort_keys=True),
                )
            )
            matched_recordings.add(token)
            counts["accepted_candidates"] += 1
            if len(batch) >= 2_000:
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?)", batch
                )
                counts["duplicate_cut_ids"] += len(batch) - (connection.total_changes - before)
                connection.commit()
                batch.clear()
            if progress and counts["cuts_seen"] % 100_000 == 0:
                print(
                    f"libriheavy_cuts_seen={counts['cuts_seen']} "
                    f"accepted={counts['accepted_candidates']}",
                    file=sys.stderr,
                    flush=True,
                )
        if batch:
            before = connection.total_changes
            connection.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?)", batch)
            counts["duplicate_cut_ids"] += len(batch) - (connection.total_changes - before)
            connection.commit()
    stored = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(duration_seconds), 0) FROM candidates"
    ).fetchone()
    connection.execute("CREATE INDEX candidates_priority ON candidates(priority, utterance_id)")
    connection.commit()
    return {
        "counts": dict(counts),
        "stored_candidates": int(stored[0]),
        "stored_candidate_hours": float(stored[1]) / 3600,
        "manifests": [str(path) for path in manifests],
    }, matched_recordings


def convert_librispeech_row(
    row: dict[str, Any], conversion: tuple[list[int] | None, str | None, int]
) -> tuple[dict[str, Any] | None, str | None, int]:
    text = " ".join(str(row["text"]).split())
    phone_ids, error, fallback_words = conversion
    if error:
        return None, error, fallback_words
    result = dict(row)
    result.update(
        {
            "recording_id": str(row["utterance_id"]),
            "start_seconds": 0.0,
            "normalized_text": text,
            "phone_ids": phone_ids,
        }
    )
    return result, None, fallback_words


def empty_summary() -> dict[str, Any]:
    return {"items": 0, "seconds": 0.0, "speakers": set(), "corpora": Counter()}


def update_summary(summary: dict[str, Any], row: dict[str, Any]) -> None:
    summary["items"] += 1
    summary["seconds"] += float(row["duration_seconds"])
    summary["speakers"].add(f"{row['corpus']}/{row['speaker_id']}")
    summary["corpora"][str(row["corpus"])] += float(row["duration_seconds"])


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": summary["items"],
        "hours": summary["seconds"] / 3600,
        "unique_corpus_speakers": len(summary["speakers"]),
        "hours_by_corpus": {
            corpus: seconds / 3600 for corpus, seconds in sorted(summary["corpora"].items())
        },
    }


def write_student_manifests(
    connection: sqlite3.Connection,
    codec_rows: dict[str, list[dict[str, Any]]],
    converter: PhoneConversionPool,
    output_dir: Path,
    target_train_hours: float | None,
    maximum_speaker_hours: float,
    progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_paths = {split: output_dir / f"{split}.jsonl" for split in CODEC_FILENAMES}
    summaries = {split: empty_summary() for split in CODEC_FILENAMES}
    counters: Counter[str] = Counter()
    fallback_words = 0
    temporary_paths = {
        split: path.with_suffix(".jsonl.tmp") for split, path in output_paths.items()
    }
    streams = {split: path.open("w", encoding="utf-8") for split, path in temporary_paths.items()}
    try:
        for split, rows in codec_rows.items():
            for offset in range(0, len(rows), PHONE_BATCH_SIZE):
                batch = rows[offset : offset + PHONE_BATCH_SIZE]
                texts = [" ".join(str(row["text"]).split()) for row in batch]
                conversions = converter.convert_many(texts)
                for row, conversion in zip(batch, conversions, strict=True):
                    converted, error, row_fallback_words = convert_librispeech_row(row, conversion)
                    fallback_words += row_fallback_words
                    if error:
                        counters[f"{split}_librispeech_g2p_{error.split(':', 1)[0]}"] += 1
                        continue
                    streams[split].write(json.dumps(converted, sort_keys=True) + "\n")
                    update_summary(summaries[split], converted)
            if progress:
                print(
                    f"phone_labels split={split} processed={len(rows)} "
                    f"accepted={summaries[split]['items']}",
                    file=sys.stderr,
                    flush=True,
                )

        target_seconds = target_train_hours * 3600 if target_train_hours is not None else None
        maximum_speaker_seconds = maximum_speaker_hours * 3600
        speaker_seconds: Counter[str] = Counter()
        cursor = connection.execute(
            "SELECT payload FROM candidates ORDER BY priority, utterance_id"
        )
        stop = False
        while not stop:
            if target_seconds is not None and summaries["train"]["seconds"] >= target_seconds:
                break
            serialized_batch = [item[0] for item in islice(cursor, PHONE_BATCH_SIZE)]
            if not serialized_batch:
                break
            batch = [json.loads(serialized) for serialized in serialized_batch]
            conversions = converter.convert_many([row["normalized_text"] for row in batch])
            for row, (phone_ids, error, row_fallback_words) in zip(batch, conversions, strict=True):
                fallback_words += row_fallback_words
                if error:
                    counters[f"libriheavy_g2p_{error.split(':', 1)[0]}"] += 1
                    continue
                duration = float(row["duration_seconds"])
                speaker = str(row["speaker_id"])
                if speaker_seconds[speaker] + duration > maximum_speaker_seconds:
                    counters["libriheavy_skipped_speaker_cap"] += 1
                    continue
                row["phone_ids"] = phone_ids
                streams["train"].write(json.dumps(row, sort_keys=True) + "\n")
                update_summary(summaries["train"], row)
                speaker_seconds[speaker] += duration
                counters["libriheavy_selected_items"] += 1
                if target_seconds is not None and summaries["train"]["seconds"] >= target_seconds:
                    stop = True
                    break
            if progress and counters["libriheavy_selected_items"] % 50_000 < PHONE_BATCH_SIZE:
                print(
                    "phone_labels split=train corpus=libriheavy "
                    f"selected={counters['libriheavy_selected_items']} "
                    f"hours={summaries['train']['seconds'] / 3600:.2f}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for stream in streams.values():
            stream.close()
    for split, temporary_path in temporary_paths.items():
        temporary_path.replace(output_paths[split])
    counters["g2p_fallback_words"] = fallback_words
    return (
        {split: finalize_summary(summary) for split, summary in summaries.items()},
        {**dict(counters), "selected_librilight_speakers": len(speaker_seconds)},
    )


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Student Distillation Manifest",
        "",
        f"- Status: **{report['status']}**",
        f"- Target train hours: {report['configuration']['target_train_hours']}",
        f"- Actual train hours: {report['splits']['train']['hours']:.2f}",
        f"- Codec LibriLight raw recordings: {report['codec_overlap']['selected_recordings']}",
        f"- Recordings matched by LibriHeavy: {report['codec_overlap']['matched_recordings']}",
        f"- Unmatched recordings: {report['codec_overlap']['unmatched_recordings']}",
        "",
        "| Split | Items | Hours | Speakers |",
        "|---|---:|---:|---:|",
    ]
    for split, summary in report["splits"].items():
        lines.append(
            f"| {split} | {summary['items']} | {summary['hours']:.2f} | "
            f"{summary['unique_corpus_speakers']} |"
        )
    lines.extend(
        [
            "",
            "LibriHeavy rows reference LibriLight raw recordings with start_seconds and duration_seconds. The training DataLoader must read these segments rather than the full recording.",
            "",
        ]
    )
    return "\n".join(lines)


def build_student_manifests(
    codec_manifest_dir: Path,
    libriheavy_root: Path,
    librilight_root: Path,
    output_dir: Path,
    vocabulary_path: Path,
    target_train_hours: float | None = DEFAULT_TARGET_TRAIN_HOURS,
    maximum_speaker_hours: float = DEFAULT_MAXIMUM_SPEAKER_HOURS,
    text_source: str = "book",
    seed: int = 1,
    g2p_fallback: bool = True,
    num_workers: int = 1,
    explicit_libriheavy_manifests: list[Path] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    codec_manifest_dir = codec_manifest_dir.expanduser().resolve()
    libriheavy_root = libriheavy_root.expanduser().resolve()
    librilight_root = librilight_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    vocabulary_path = vocabulary_path.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    codec_rows, recording_index, codec_counts = load_codec_selection(codec_manifest_dir)
    subsets = {identifier.split("/", 1)[0] for identifier in recording_index.identifiers}
    manifests = (
        [path.expanduser().resolve() for path in explicit_libriheavy_manifests]
        if explicit_libriheavy_manifests
        else discover_libriheavy_manifests(libriheavy_root, subsets)
    )
    vocabulary = load_phone_vocabulary(vocabulary_path)
    local_converter = PhoneConverter(vocabulary, g2p_fallback)
    with tempfile.NamedTemporaryFile(
        prefix="student-candidates-", suffix=".sqlite3", dir=output_dir, delete=False
    ) as temporary_database:
        database_path = Path(temporary_database.name)
    try:
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            candidate_report, matched_recordings = index_libriheavy_candidates(
                connection,
                manifests,
                recording_index,
                librilight_root,
                text_source,
                seed,
                progress,
            )
            with PhoneConversionPool(
                vocabulary,
                g2p_fallback,
                num_workers,
                local_converter=local_converter,
            ) as conversion_pool:
                splits, selection_counts = write_student_manifests(
                    connection,
                    codec_rows,
                    conversion_pool,
                    output_dir,
                    target_train_hours,
                    maximum_speaker_hours,
                    progress,
                )
        finally:
            connection.close()
    finally:
        database_path.unlink(missing_ok=True)

    unmatched = sorted(recording_index.identifiers - matched_recordings)
    target_reached = target_train_hours is None or splits["train"]["hours"] >= target_train_hours
    failures: list[str] = []
    if not target_reached:
        failures.append("target_train_hours_not_reached")
    for split in CODEC_FILENAMES:
        if not splits[split]["items"]:
            failures.append(f"empty_{split}_split")
    if recording_index.identifiers and not matched_recordings:
        failures.append("no_codec_librilight_recordings_matched_by_libriheavy")
    selected_vad_seconds = sum(recording_index.selected_vad_seconds.values())
    matched_vad_seconds = sum(
        recording_index.selected_vad_seconds[identifier] for identifier in matched_recordings
    )
    report = {
        "schema_version": 1,
        "configuration": {
            "codec_manifest_dir": str(codec_manifest_dir),
            "libriheavy_root": str(libriheavy_root),
            "librilight_root": str(librilight_root),
            "vocabulary_path": str(vocabulary_path),
            "target_train_hours": target_train_hours,
            "maximum_librilight_hours_per_speaker": maximum_speaker_hours,
            "text_source": text_source,
            "seed": seed,
            "g2p_fallback": g2p_fallback,
            "num_workers": num_workers,
        },
        "codec_counts": codec_counts,
        "libriheavy_candidates": candidate_report,
        "selection_counts": selection_counts,
        "codec_overlap": {
            "selected_recordings": len(recording_index.identifiers),
            "selected_vad_hours": selected_vad_seconds / 3600,
            "matched_recordings": len(matched_recordings),
            "matched_vad_hours": matched_vad_seconds / 3600,
            "unmatched_recordings": len(unmatched),
            "unmatched_examples": unmatched[:20],
        },
        "teacher_vocabulary": {
            "classes": len(vocabulary),
            "sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
        },
        "splits": splits,
        "target_reached": target_reached,
        "target_overshoot_seconds": (
            (splits["train"]["hours"] - target_train_hours) * 3600
            if target_train_hours is not None and target_reached
            else None
        ),
        "failures": failures,
        "output_paths": {split: str(output_dir / f"{split}.jsonl") for split in CODEC_FILENAMES},
        "status": "PASS" if not failures else "NEEDS_ATTENTION",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def default_vocabulary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "ctc_gop_teacher_vocab.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CTC phone-label manifests from Codec selection and LibriHeavy cuts"
    )
    parser.add_argument("--codec-manifest-dir", type=Path, required=True)
    parser.add_argument("--libriheavy-root", type=Path, required=True)
    parser.add_argument("--librilight-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=default_vocabulary_path())
    parser.add_argument("--libriheavy-manifest", type=Path, action="append", default=[])
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-train-hours", type=float, default=DEFAULT_TARGET_TRAIN_HOURS
    )
    target_group.add_argument("--all-matched", action="store_true")
    parser.add_argument(
        "--max-librilight-hours-per-speaker",
        type=float,
        default=DEFAULT_MAXIMUM_SPEAKER_HOURS,
    )
    parser.add_argument("--text-source", choices=("book", "asr", "supervision"), default="book")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--disable-g2p-fallback", action="store_true")
    args = parser.parse_args()
    target_train_hours = None if args.all_matched else args.target_train_hours
    if target_train_hours is not None and target_train_hours <= 0:
        parser.error("target-train-hours must be positive")
    if args.max_librilight_hours_per_speaker <= 0:
        parser.error("max-librilight-hours-per-speaker must be positive")
    if args.num_workers < 1:
        parser.error("num-workers must be at least 1")
    report = build_student_manifests(
        args.codec_manifest_dir,
        args.libriheavy_root,
        args.librilight_root,
        args.output_dir,
        args.vocabulary,
        target_train_hours,
        args.max_librilight_hours_per_speaker,
        args.text_source,
        args.seed,
        not args.disable_g2p_fallback,
        args.num_workers,
        args.libriheavy_manifest,
        progress=True,
    )
    print(json.dumps({"status": report["status"], **report["splits"]}, sort_keys=True))
    print(f"report_json={args.output_dir.expanduser().resolve() / 'report.json'}")
    print(f"report_markdown={args.output_dir.expanduser().resolve() / 'report.md'}")
    print(f"student_manifest_build={report['status']}")


if __name__ == "__main__":
    main()
