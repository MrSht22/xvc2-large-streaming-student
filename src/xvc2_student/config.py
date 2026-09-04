from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = True
    feat_extract_norm: str = "layer"
    feat_extract_activation: str = "gelu"
    model_dim: int = 768
    num_heads: int = 12
    ffn_dim: int = 3072
    num_layers: int = 12
    segment_length: int = 4
    left_context_length: int = 32
    right_context_length: int = 2
    dropout: float = 0.1
    teacher_dim: int = 1024
    vocab_size: int = 40
    freeze_feature_extractor: bool = True

    def __post_init__(self) -> None:
        if not (len(self.conv_dim) == len(self.conv_kernel) == len(self.conv_stride)):
            raise ValueError("Convolutional configuration lengths must match")
        positive = (
            *self.conv_dim,
            *self.conv_kernel,
            *self.conv_stride,
            self.model_dim,
            self.num_heads,
            self.ffn_dim,
            self.num_layers,
            self.segment_length,
            self.left_context_length,
            self.right_context_length,
            self.teacher_dim,
            self.vocab_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Model dimensions, contexts, and rates must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        values = dict(values)
        for name in ("conv_dim", "conv_kernel", "conv_stride"):
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class DistillationConfig:
    teacher_layer: int = 20
    feature_weight: float = 1.0
    ctc_weight: float = 0.1
    streaming_consistency_weight: float = 0.0
    anchor_weight: float = 0.0
    anchor_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.teacher_layer <= 24:
            raise ValueError("teacher_layer must be between 1 and 24")
        if (
            min(
                self.feature_weight,
                self.ctc_weight,
                self.streaming_consistency_weight,
                self.anchor_weight,
            )
            < 0
        ):
            raise ValueError("Distillation weights cannot be negative")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_steps: int = 200_000
    gradient_clip: float = 1.0
    amp: str = "bf16"
    seed: int = 1
    log_interval: int = 20
    save_interval: int = 5_000

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.gradient_clip) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if min(self.max_steps, self.log_interval, self.save_interval) <= 0:
            raise ValueError("Step counts must be positive")
        if self.amp not in {"bf16", "fp16", "none"}:
            raise ValueError("amp must be bf16, fp16, or none")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> ExperimentConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    return ExperimentConfig(
        schema_version=1,
        model=ModelConfig.from_dict(values["model"]),
        distillation=DistillationConfig(**values.get("distillation", {})),
        training=TrainingConfig(**values.get("training", {})),
    )
