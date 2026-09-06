from __future__ import annotations

from array import array
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


class PhoneManifestDataset(Dataset):
    """Random-access JSONL dataset with precomputed phone IDs and audio cuts."""

    def __init__(self, path: Path, sample_rate: int = 16_000) -> None:
        self.path = path.expanduser().resolve()
        self.sample_rate = sample_rate
        self.offsets = array("Q")
        self.line_numbers = array("I")
        self.sample_counts = array("I")
        self.first_segment_index: int | None = None
        self._manifest: BinaryIO | None = None
        self._manifest_pid: int | None = None
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                item = self._decode(line, line_number)
                self._validate(item, line_number)
                if self.first_segment_index is None and (
                    item.get("corpus") == "libriheavy" or float(item.get("start_seconds", 0.0)) > 0
                ):
                    self.first_segment_index = len(self.offsets)
                self.offsets.append(offset)
                self.line_numbers.append(line_number)
                duration = item.get("duration_seconds")
                self.sample_counts.append(
                    round(float(duration) * self.sample_rate) if duration is not None else 0
                )
        if not self.offsets:
            raise RuntimeError(f"No items in {self.path}")

    def _decode(self, line: bytes, line_number: int) -> dict[str, Any]:
        try:
            item = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSON at {self.path}:{line_number}: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object at {self.path}:{line_number}")
        return item

    def _validate(self, item: dict[str, Any], line_number: int) -> None:
        required = {"utterance_id", "audio_path", "phone_ids"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{self.path}:{line_number} is missing fields: {sorted(missing)}")
        if not item["phone_ids"]:
            raise ValueError(f"{self.path}:{line_number} has no phone IDs")
        start = float(item.get("start_seconds", 0.0))
        if start < 0:
            raise ValueError(f"{self.path}:{line_number} has negative start_seconds")
        if "duration_seconds" in item and float(item["duration_seconds"]) <= 0:
            raise ValueError(f"{self.path}:{line_number} has non-positive duration_seconds")
        if item.get("sample_rate") is not None and int(item["sample_rate"]) <= 0:
            raise ValueError(f"{self.path}:{line_number} has non-positive sample_rate")

    def _stream(self) -> BinaryIO:
        process_id = os.getpid()
        if self._manifest is not None and self._manifest_pid != process_id:
            self._manifest.close()
            self._manifest = None
        if self._manifest is None or self._manifest.closed:
            self._manifest = self.path.open("rb")
            self._manifest_pid = process_id
        return self._manifest

    def _item(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        stream = self._stream()
        stream.seek(self.offsets[index])
        line = stream.readline()
        return self._decode(line, self.line_numbers[index])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_manifest"] = None
        state["_manifest_pid"] = None
        return state

    def __del__(self) -> None:
        manifest = getattr(self, "_manifest", None)
        if manifest is not None:
            manifest.close()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._item(index)
        audio_path = str(item["audio_path"])
        start_seconds = float(item.get("start_seconds", 0.0))
        duration_seconds = item.get("duration_seconds")
        source_sample_rate = item.get("sample_rate")
        if source_sample_rate is None:
            source_sample_rate = torchaudio.info(audio_path).sample_rate
        source_sample_rate = int(source_sample_rate)
        frame_offset = round(start_seconds * source_sample_rate)
        num_frames = (
            round(float(duration_seconds) * source_sample_rate)
            if duration_seconds is not None
            else -1
        )
        waveform, sample_rate = torchaudio.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        if waveform.numel() == 0:
            raise RuntimeError(f"Audio cut is empty for {item['utterance_id']}")
        if num_frames > 0 and waveform.shape[-1] != num_frames:
            raise RuntimeError(
                f"Audio cut is truncated for {item['utterance_id']}: "
                f"requested_frames={num_frames}, loaded_frames={waveform.shape[-1]}"
            )
        if sample_rate != source_sample_rate:
            raise RuntimeError(
                f"Manifest sample_rate={source_sample_rate} differs from audio sample_rate="
                f"{sample_rate} for {item['utterance_id']}"
            )
        if waveform.shape[0] != 1:
            waveform = waveform.mean(0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return {
            "utterance_id": str(item["utterance_id"]),
            "waveform": waveform[0],
            "phone_ids": torch.tensor(item["phone_ids"], dtype=torch.long),
        }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = [item["waveform"] for item in items]
    phones = [item["phone_ids"] for item in items]
    return {
        "utterance_ids": [item["utterance_id"] for item in items],
        "waveform": pad_sequence(waveforms, batch_first=True),
        "sample_lengths": torch.tensor([item.numel() for item in waveforms]),
        "targets": torch.cat(phones),
        "target_lengths": torch.tensor([item.numel() for item in phones]),
    }


class StatefulDistributedSampler(Sampler[int]):
    """Deterministic shuffling with an explicit per-rank resume position."""

    def __init__(
        self,
        size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1,
        epoch: int = 0,
        position: int = 0,
    ) -> None:
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = epoch
        self.position = position

    def _indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        padding = (-len(indices)) % self.world_size
        indices.extend(indices[:padding])
        return indices[self.rank :: self.world_size]

    def __iter__(self):
        return iter(self._indices()[self.position :])

    def __len__(self) -> int:
        return len(self._indices()) - self.position

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])

    def advance(self, count: int) -> None:
        self.position += count

    def next_epoch(self) -> None:
        self.epoch += 1
        self.position = 0


class StatefulDistributedBatchSampler(Sampler[list[int]]):
    """Deterministic duration buckets with a padded-audio budget per rank."""

    def __init__(
        self,
        sample_counts: array,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1,
        max_batch_samples: int = 960_000,
        max_batch_items: int = 8,
        bucket_size: int = 2_048,
        epoch: int = 0,
        position: int = 0,
    ) -> None:
        if not sample_counts or any(count <= 0 for count in sample_counts):
            raise ValueError("Dynamic batching requires positive sample counts for every item")
        if min(world_size, max_batch_samples, max_batch_items, bucket_size) <= 0:
            raise ValueError("World size and dynamic batching limits must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("Rank must be in [0, world_size)")
        self.sample_counts = sample_counts
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.max_batch_samples = max_batch_samples
        self.max_batch_items = max_batch_items
        self.bucket_size = bucket_size
        self.epoch = epoch
        self.position = position
        self.length_order = array(
            "I", sorted(range(len(sample_counts)), key=lambda index: (sample_counts[index], index))
        )
        self._cached_epoch: int | None = None
        self._cached_batches: list[list[int]] = []
        self.padding_batches = 0

    def _global_order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        bucket_starts = list(range(0, len(self.length_order), self.bucket_size))
        bucket_order = torch.randperm(len(bucket_starts), generator=generator).tolist()
        ordered: list[int] = []
        for bucket_index in bucket_order:
            start = bucket_starts[bucket_index]
            bucket = self.length_order[start : start + self.bucket_size]
            permutation = torch.randperm(len(bucket), generator=generator).tolist()
            ordered.extend(int(bucket[index]) for index in permutation)
        padding = (-len(ordered)) % self.world_size
        ordered.extend(ordered[:padding])
        return ordered

    def _pack(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        maximum = 0
        for index in indices:
            sample_count = self.sample_counts[index]
            next_maximum = max(maximum, sample_count)
            exceeds_budget = next_maximum * (len(batch) + 1) > self.max_batch_samples
            if batch and (len(batch) >= self.max_batch_items or exceeds_budget):
                batches.append(batch)
                batch = []
                maximum = 0
                next_maximum = sample_count
            batch.append(index)
            maximum = next_maximum
        if batch:
            batches.append(batch)
        return batches

    def _batches(self) -> list[list[int]]:
        if self._cached_epoch == self.epoch:
            return self._cached_batches
        ordered = self._global_order()
        batches_by_rank = [
            self._pack(ordered[rank :: self.world_size]) for rank in range(self.world_size)
        ]
        batch_counts = [len(batches) for batches in batches_by_rank]
        target_count = max(len(batches) for batches in batches_by_rank)
        for batches in batches_by_rank:
            original_count = len(batches)
            batches.extend(
                list(batches[index % original_count])
                for index in range(target_count - original_count)
            )
        self.padding_batches = target_count - batch_counts[self.rank]
        self._cached_batches = batches_by_rank[self.rank]
        self._cached_epoch = self.epoch
        return self._cached_batches

    def __iter__(self):
        return iter(self._batches()[self.position :])

    def __len__(self) -> int:
        return len(self._batches()) - self.position

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "max_batch_samples": self.max_batch_samples,
            "max_batch_items": self.max_batch_items,
            "bucket_size": self.bucket_size,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        for name in ("max_batch_samples", "max_batch_items", "bucket_size"):
            if int(state.get(name, getattr(self, name))) != getattr(self, name):
                raise RuntimeError(f"Resume {name} differs from the current dynamic batch setting")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self._cached_epoch = None

    def advance(self) -> None:
        self.position += 1

    def next_epoch(self) -> None:
        self.epoch += 1
        self.position = 0
        self._cached_epoch = None
