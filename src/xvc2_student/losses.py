from __future__ import annotations

import torch
import torch.nn.functional as F


def valid_feature_loss(
    predicted: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"Feature shapes differ: {predicted.shape} vs {target.shape}")
    predicted = F.layer_norm(predicted.float(), (predicted.shape[-1],))
    target = F.layer_norm(target.float(), (target.shape[-1],))
    values = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    mask = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
    return (values * mask).sum() / mask.sum().clamp_min(1)


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    return F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )


def streaming_consistency_loss(
    full_hidden: torch.Tensor, chunk_hidden: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return valid_feature_loss(chunk_hidden, full_hidden.detach(), lengths)


def anchor_loss(
    hidden: torch.Tensor, anchor_hidden: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return valid_feature_loss(hidden, anchor_hidden.detach(), lengths)
