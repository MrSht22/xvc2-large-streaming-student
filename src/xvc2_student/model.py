from __future__ import annotations

from dataclasses import dataclass
import torch
from torchaudio.models import Emformer
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder

from .config import ModelConfig


@dataclass
class StreamingState:
    waveform_buffer: torch.Tensor
    feature_buffer: torch.Tensor
    emformer_states: list[list[torch.Tensor]] | None
    finalized: bool = False


class StreamingPhoneEncoder(torch.nn.Module):
    """50 Hz Emformer Student with explicit raw-waveform streaming state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        wav2vec_config = Wav2Vec2Config(
            conv_dim=list(config.conv_dim),
            conv_kernel=list(config.conv_kernel),
            conv_stride=list(config.conv_stride),
            conv_bias=config.conv_bias,
            feat_extract_norm=config.feat_extract_norm,
            feat_extract_activation=config.feat_extract_activation,
        )
        self.feature_extractor = Wav2Vec2FeatureEncoder(wav2vec_config)
        self.input_norm = torch.nn.LayerNorm(config.conv_dim[-1])
        self.input_projection = torch.nn.Linear(config.conv_dim[-1], config.model_dim)
        self.input_dropout = torch.nn.Dropout(config.dropout)
        self.context = Emformer(
            input_dim=config.model_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            segment_length=config.segment_length,
            dropout=config.dropout,
            activation="gelu",
            left_context_length=config.left_context_length,
            right_context_length=config.right_context_length,
            max_memory_size=0,
        )
        self.output_norm = torch.nn.LayerNorm(config.model_dim)
        self.distill_projection = torch.nn.Linear(config.model_dim, config.teacher_dim)
        self.phone_head = torch.nn.Linear(config.model_dim, config.vocab_size)
        if config.freeze_feature_extractor:
            self.freeze_feature_extractor()

    @property
    def stride_samples(self) -> int:
        result = 1
        for stride in self.config.conv_stride:
            result *= stride
        return result

    @property
    def receptive_field_samples(self) -> int:
        result = 1
        accumulated_stride = 1
        for kernel, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            result += (kernel - 1) * accumulated_stride
            accumulated_stride *= stride
        return result

    def freeze_feature_extractor(self) -> None:
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()

    def train(self, mode: bool = True) -> "StreamingPhoneEncoder":
        super().train(mode)
        if self.config.freeze_feature_extractor:
            self.feature_extractor.eval()
        return self

    def load_teacher_frontend(self, teacher: torch.nn.Module) -> None:
        self.feature_extractor.load_state_dict(
            teacher.wav2vec2.feature_extractor.state_dict(), strict=True
        )

    def output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        lengths = sample_lengths.long()
        for kernel, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return lengths.clamp_min(0)

    def _project(self, convolution_features: torch.Tensor) -> torch.Tensor:
        values = convolution_features.transpose(1, 2)
        values = self.input_norm(values)
        return self.input_dropout(self.input_projection(values))

    def _heads(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.output_norm(hidden)
        return {
            "hidden_states": hidden,
            "distill_features": self.distill_projection(hidden),
            "phone_logits": self.phone_head(hidden),
        }

    def forward(
        self, input_values: torch.Tensor, input_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        convolution = self.feature_extractor(input_values)
        features = self._project(convolution)
        lengths = self.output_lengths(input_lengths)
        valid = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        features = features.masked_fill(~valid[..., None], 0.0)
        segment = self.config.segment_length
        padded_lengths = torch.div(lengths + segment - 1, segment, rounding_mode="floor") * segment
        padded_time = int(padded_lengths.max().item())
        features = torch.nn.functional.pad(features, (0, 0, 0, padded_time - features.shape[1]))
        right = features.new_zeros(
            features.shape[0], self.config.right_context_length, features.shape[-1]
        )
        hidden, _ = self.context(torch.cat((features, right), dim=1), padded_lengths)
        outputs = self._heads(hidden[:, : convolution.shape[-1]])
        outputs["output_lengths"] = lengths
        return outputs

    def init_streaming_state(
        self, device: torch.device, waveform_dtype: torch.dtype = torch.float32
    ) -> StreamingState:
        return StreamingState(
            waveform_buffer=torch.empty(1, 0, device=device, dtype=waveform_dtype),
            feature_buffer=torch.empty(
                1, 0, self.config.model_dim, device=device, dtype=self.input_projection.weight.dtype
            ),
            emformer_states=None,
        )

    def _append_features(self, chunk: torch.Tensor, state: StreamingState) -> None:
        state.waveform_buffer = torch.cat((state.waveform_buffer, chunk), dim=1)
        available = state.waveform_buffer.shape[1]
        if available < self.receptive_field_samples:
            return
        frames = (available - self.receptive_field_samples) // self.stride_samples + 1
        convolution = self.feature_extractor(state.waveform_buffer)
        if convolution.shape[-1] != frames:
            raise RuntimeError("Unexpected streaming convolution frame count")
        state.feature_buffer = torch.cat((state.feature_buffer, self._project(convolution)), dim=1)
        state.waveform_buffer = state.waveform_buffer[:, frames * self.stride_samples :]

    def _infer(
        self, segment: torch.Tensor, output_length: int, state: StreamingState
    ) -> torch.Tensor:
        required = self.config.segment_length + self.config.right_context_length
        segment = torch.nn.functional.pad(segment, (0, 0, 0, required - segment.shape[1]))
        lengths = torch.tensor(
            [output_length + self.config.right_context_length],
            device=segment.device,
            dtype=torch.long,
        )
        hidden, returned, state.emformer_states = self.context.infer(
            segment, lengths, state.emformer_states
        )
        if int(returned[0]) != output_length:
            raise RuntimeError("Unexpected Emformer streaming output length")
        return hidden[:, :output_length]

    def forward_chunk(
        self, chunk: torch.Tensor, state: StreamingState, is_final: bool = False
    ) -> tuple[dict[str, torch.Tensor], StreamingState]:
        if self.training:
            raise RuntimeError("forward_chunk requires eval mode")
        if state.finalized:
            raise RuntimeError("Streaming state has already been finalized")
        if chunk.ndim == 1:
            chunk = chunk[None]
        if chunk.ndim != 2 or chunk.shape[0] != 1:
            raise ValueError("Expected one waveform stream with shape [1, samples]")
        self._append_features(chunk, state)
        emitted: list[torch.Tensor] = []
        required = self.config.segment_length + self.config.right_context_length
        while state.feature_buffer.shape[1] >= required:
            emitted.append(
                self._infer(state.feature_buffer[:, :required], self.config.segment_length, state)
            )
            state.feature_buffer = state.feature_buffer[:, self.config.segment_length :]
        if is_final:
            while state.feature_buffer.shape[1]:
                count = min(self.config.segment_length, state.feature_buffer.shape[1])
                emitted.append(self._infer(state.feature_buffer[:, :required], count, state))
                state.feature_buffer = state.feature_buffer[:, count:]
            state.waveform_buffer = state.waveform_buffer[:, :0]
            state.finalized = True
        hidden = (
            torch.cat(emitted, dim=1)
            if emitted
            else state.feature_buffer.new_empty(1, 0, self.config.model_dim)
        )
        outputs = self._heads(hidden)
        outputs["output_lengths"] = torch.tensor([hidden.shape[1]], device=hidden.device)
        return outputs, state


def parameter_breakdown(model: StreamingPhoneEncoder) -> dict[str, int]:
    groups: dict[str, torch.nn.Module] = {
        "feature_extractor": model.feature_extractor,
        "input_projection": torch.nn.ModuleList([model.input_norm, model.input_projection]),
        "emformer": model.context,
        "heads": torch.nn.ModuleList(
            [model.output_norm, model.distill_projection, model.phone_head]
        ),
    }
    result = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
    result["total"] = sum(p.numel() for p in model.parameters())
    result["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return result
