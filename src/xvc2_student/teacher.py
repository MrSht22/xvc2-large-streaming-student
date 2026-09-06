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


def configure_attention(config: Wav2Vec2Config, implementation: str) -> None:
    if implementation not in {"eager", "sdpa"}:
        raise ValueError("Teacher attention implementation must be eager or sdpa")
    config._attn_implementation = implementation


def load_legacy_weight_norm_teacher(
    checkpoint_dir: Path, attention_implementation: str = "eager"
) -> Wav2Vec2ForCTC | None:
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not weights_path.is_file():
        return None
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not any(
        name in state_dict for name in LEGACY_POSITION_CONV
    ):
        return None
    config = Wav2Vec2Config.from_pretrained(checkpoint_dir, local_files_only=True)
    configure_attention(config, attention_implementation)
    model = Wav2Vec2ForCTC(config)
    model.load_state_dict(remap_legacy_position_conv(state_dict), strict=True)
    return model


def load_teacher(checkpoint_dir: Path, attention_implementation: str = "eager") -> Wav2Vec2ForCTC:
    model = load_legacy_weight_norm_teacher(checkpoint_dir, attention_implementation)
    if model is None:
        model = Wav2Vec2ForCTC.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
            attn_implementation=attention_implementation,
        )
    model.eval().requires_grad_(False)
    return model


def load_teacher_with_loading_info(
    checkpoint_dir: Path, attention_implementation: str = "eager"
) -> tuple[Wav2Vec2ForCTC, dict[str, Any]]:
    legacy_model = load_legacy_weight_norm_teacher(checkpoint_dir, attention_implementation)
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
        attn_implementation=attention_implementation,
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


def prepare_encoder_attention_mask(
    encoder: torch.nn.Module,
    attention_mask: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor | None:
    if hasattr(encoder, "_update_full_mask"):
        return encoder._update_full_mask(attention_mask, hidden)
    if getattr(encoder, "_use_flash_attention_2", False):
        return attention_mask if 0 in attention_mask else None
    expanded = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden.dtype)
    expanded = expanded * torch.finfo(hidden.dtype).min
    return expanded.expand(expanded.shape[0], 1, expanded.shape[-1], expanded.shape[-1])


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


@torch.no_grad()
def optimized_teacher_targets(
    teacher: Wav2Vec2ForCTC,
    waveform: torch.Tensor,
    sample_lengths: torch.Tensor,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact hidden-state target and reusable frozen convolution features."""

    wav2vec2 = teacher.wav2vec2
    encoder = wav2vec2.encoder
    if not 1 <= layer <= len(encoder.layers):
        raise ValueError(f"Teacher layer must be in [1, {len(encoder.layers)}]")
    raw_mask = torch.arange(waveform.shape[1], device=waveform.device)[None]
    raw_mask = raw_mask < sample_lengths[:, None]
    convolution = wav2vec2.feature_extractor(waveform)
    extract_features = convolution.transpose(1, 2)
    feature_mask = wav2vec2._get_feature_vector_attention_mask(
        extract_features.shape[1], raw_mask.long(), add_adapter=False
    )
    hidden, _ = wav2vec2.feature_projection(extract_features)
    hidden = wav2vec2._mask_hidden_states(hidden, attention_mask=feature_mask)

    expanded_mask = feature_mask.unsqueeze(-1).expand_as(hidden)
    hidden = hidden.masked_fill(~expanded_mask, 0)
    attention_mask = prepare_encoder_attention_mask(encoder, feature_mask, hidden)
    hidden = hidden + encoder.pos_conv_embed(hidden)
    if not teacher.config.do_stable_layer_norm:
        hidden = encoder.layer_norm(hidden)
    hidden = encoder.dropout(hidden)
    for encoder_layer in encoder.layers[:layer]:
        hidden = encoder_layer(
            hidden,
            attention_mask=attention_mask,
            output_attentions=False,
        )[0]

    lengths = teacher._get_feat_extract_output_lengths(sample_lengths)
    return hidden, lengths, convolution


@torch.inference_mode()
def verify_optimized_teacher(
    teacher: Wav2Vec2ForCTC,
    waveform: torch.Tensor,
    sample_lengths: torch.Tensor,
    layer: int,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    expected, expected_lengths = teacher_targets(teacher, waveform, sample_lengths, layer)
    actual, actual_lengths, convolution = optimized_teacher_targets(
        teacher, waveform, sample_lengths, layer
    )
    if not torch.equal(actual_lengths, expected_lengths):
        raise RuntimeError("Optimized Teacher frame lengths differ from the full Teacher")
    difference = (actual.float() - expected.float()).abs()
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        raise RuntimeError(
            "Optimized Teacher hidden states differ from the full Teacher: "
            f"max_abs_difference={float(difference.max())}"
        )
    return {
        "status": "PASS",
        "target_shape": list(actual.shape),
        "convolution_shape": list(convolution.shape),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
    }
