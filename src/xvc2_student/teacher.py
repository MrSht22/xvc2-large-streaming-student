from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_teacher(checkpoint_dir: Path) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir, local_files_only=True)
    model.eval().requires_grad_(False)
    return model


def load_teacher_with_loading_info(
    checkpoint_dir: Path,
) -> tuple[Wav2Vec2ForCTC, dict[str, Any]]:
    model, loading_info = Wav2Vec2ForCTC.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        output_loading_info=True,
    )
    model.eval().requires_grad_(False)
    return model, loading_info


def loading_failures(loading_info: dict[str, Any]) -> list[str]:
    failures = []
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        values = loading_info.get(name, [])
        if values:
            failures.append(f"teacher_{name}={values}")
    return failures


def load_processor(processor_dir: Path) -> Wav2Vec2Processor:
    return Wav2Vec2Processor.from_pretrained(processor_dir, local_files_only=True)


@torch.inference_mode()
def teacher_targets(
    teacher: Wav2Vec2ForCTC,
    waveform: torch.Tensor,
    sample_lengths: torch.Tensor,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.arange(waveform.shape[1], device=waveform.device)[None] < sample_lengths[:, None]
    output = teacher(
        waveform,
        attention_mask=mask.long(),
        output_hidden_states=True,
        return_dict=True,
    )
    lengths = teacher._get_feat_extract_output_lengths(sample_lengths)
    return output.hidden_states[layer], lengths
