from pathlib import Path

import torch

from xvc2_student.checkpoint import load_checkpoint, save_checkpoint
from xvc2_student.config import ExperimentConfig
from xvc2_student.losses import valid_feature_loss
from xvc2_student.model import StreamingPhoneEncoder
from xvc2_student.smoke import tiny_config


def test_forward_and_feature_loss() -> None:
    model = StreamingPhoneEncoder(tiny_config())
    samples = model.receptive_field_samples + model.stride_samples * 7
    waveform = torch.randn(2, samples)
    outputs = model(waveform, torch.tensor([samples, samples]))
    loss = valid_feature_loss(
        outputs["distill_features"],
        torch.randn_like(outputs["distill_features"]),
        outputs["output_lengths"],
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = ExperimentConfig(model=tiny_config())
    model = StreamingPhoneEncoder(config.model)
    path = tmp_path / "step.pt"
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(path, model, config, step=3, scaler=scaler, data_state={"epoch": 2})
    restored = StreamingPhoneEncoder(config.model)
    payload = load_checkpoint(path, restored, scaler=scaler, restore_rng=False)
    assert payload["step"] == 3
    assert payload["data_state"] == {"epoch": 2}
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)
