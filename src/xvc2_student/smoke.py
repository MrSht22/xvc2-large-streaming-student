from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig, load_config
from .model import StreamingPhoneEncoder, parameter_breakdown


def tiny_config() -> ModelConfig:
    return ModelConfig(
        conv_dim=(8, 8),
        conv_kernel=(4, 2),
        conv_stride=(2, 2),
        model_dim=16,
        num_heads=4,
        ffn_dim=32,
        num_layers=2,
        segment_length=4,
        left_context_length=4,
        right_context_length=2,
        dropout=0.0,
        teacher_dim=24,
        vocab_size=10,
        freeze_feature_extractor=False,
    )


def run(config_path: Path | None = None) -> dict[str, int]:
    torch.manual_seed(7)
    config = load_config(config_path).model if config_path else tiny_config()
    model = StreamingPhoneEncoder(config)
    samples = max(model.receptive_field_samples + model.stride_samples * 11, 64)
    waveform = torch.randn(2, samples)
    lengths = torch.tensor([samples, samples - model.stride_samples])
    outputs = model(waveform, lengths)
    outputs["distill_features"].square().mean().backward()
    assert outputs["phone_logits"].shape[-1] == config.vocab_size
    counts = parameter_breakdown(model)
    print(json.dumps(counts, sort_keys=True))
    print("large_streaming_student_smoke=PASS")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
