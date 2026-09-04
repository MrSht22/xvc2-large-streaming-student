from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor


LEGACY_POSITION_CONV = {
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g": (
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    ),
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v": (
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    ),
}


def remap_legacy_position_conv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(name in state_dict for name in LEGACY_POSITION_CONV):
        return state_dict
    if not all(name in state_dict for name in LEGACY_POSITION_CONV):
        raise RuntimeError("Legacy positional convolution checkpoint is incomplete")
    remapped = dict(state_dict)
    for old_name, new_name in LEGACY_POSITION_CONV.items():
        remapped[new_name] = remapped.pop(old_name)
    return remapped


def load_legacy_weight_norm_teacher(checkpoint_dir: Path) -> Wav2Vec2ForCTC | None:
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not weights_path.is_file():
        return None
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not any(
        name in state_dict for name in LEGACY_POSITION_CONV
    ):
        return None
    config = Wav2Vec2Config.from_pretrained(checkpoint_dir, local_files_only=True)
    model = Wav2Vec2ForCTC(config)
    model.load_state_dict(remap_legacy_position_conv(state_dict), strict=True)
    return model


def load_teacher(checkpoint_dir: Path) -> Wav2Vec2ForCTC:
    model = load_legacy_weight_norm_teacher(checkpoint_dir)
    if model is None:
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir, local_files_only=True)
    model.eval().requires_grad_(False)
    return model


def load_teacher_with_loading_info(
    checkpoint_dir: Path,
) -> tuple[Wav2Vec2ForCTC, dict[str, Any]]:
    legacy_model = load_legacy_weight_norm_teacher(checkpoint_dir)
    if legacy_model is not None:
        legacy_model.eval().requires_grad_(False)
        return legacy_model, {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "legacy_weight_norm_conversion": sorted(LEGACY_POSITION_CONV),
        }
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
