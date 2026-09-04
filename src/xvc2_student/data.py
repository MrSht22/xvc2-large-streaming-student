from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


class PhoneManifestDataset(Dataset):
    """JSONL dataset with precomputed phone IDs and absolute audio paths."""

    def __init__(self, path: Path, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.items: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"utterance_id", "audio_path", "phone_ids"}
            missing = required - item.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
            if not item["phone_ids"]:
                raise ValueError(f"Line {line_number} has no phone IDs")
            self.items.append(item)
        if not self.items:
            raise RuntimeError(f"No items in {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        waveform, sample_rate = torchaudio.load(item["audio_path"])
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
