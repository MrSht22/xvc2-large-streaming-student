from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .data import PhoneManifestDataset, collate
from .model import StreamingPhoneEncoder
from .teacher import load_teacher, optimized_teacher_targets, verify_optimized_teacher


def validation_runtime(device_arg: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed validation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device(f"cuda:{local_rank}"), rank, world_size, local_rank
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_arg), rank, world_size, local_rank


def pack_validation_batches(
    sample_counts: list[int] | torch.Tensor,
    rank: int,
    world_size: int,
    max_batch_samples: int,
    max_batch_items: int,
) -> list[list[int]]:
    if len(sample_counts) == 0 or any(int(count) <= 0 for count in sample_counts):
        raise ValueError("Validation batching requires positive sample counts")
    if min(world_size, max_batch_samples, max_batch_items) <= 0:
        raise ValueError("Validation batching limits must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("Rank must be in [0, world_size)")

    ordered = sorted(
        range(len(sample_counts)), key=lambda index: (int(sample_counts[index]), index)
    )
    local_indices = ordered[rank::world_size]
    batches: list[list[int]] = []
    batch: list[int] = []
    maximum = 0
    for index in local_indices:
        sample_count = int(sample_counts[index])
        next_maximum = max(maximum, sample_count)
        exceeds_budget = next_maximum * (len(batch) + 1) > max_batch_samples
        if batch and (len(batch) >= max_batch_items or exceeds_budget):
            batches.append(batch)
            batch = []
            maximum = 0
            next_maximum = sample_count
        batch.append(index)
        maximum = next_maximum
    if batch:
        batches.append(batch)
    return batches


def collapse_ctc(tokens: list[int], blank: int = 0) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for token in tokens:
        if token != blank and token != previous:
            collapsed.append(token)
        previous = token
    return collapsed


def edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def feature_statistics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape:
        raise ValueError(f"Feature shapes differ: {predicted.shape} vs {target.shape}")
    predicted = F.layer_norm(predicted.float(), (predicted.shape[-1],))
    target = F.layer_norm(target.float(), (target.shape[-1],))
    values = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    mask = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
    return (values * mask).sum(), mask.sum()


def ctc_statistics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> tuple[torch.Tensor, int, int, int]:
    losses = F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=True,
    )
    normalized_loss_sum = (losses / target_lengths.to(losses.dtype)).sum()

    predictions = logits.argmax(-1).cpu()
    cpu_input_lengths = input_lengths.cpu().tolist()
    cpu_targets = targets.cpu().tolist()
    cpu_target_lengths = target_lengths.cpu().tolist()
    target_offset = 0
    phone_errors = 0
    exact_sequences = 0
    reference_phones = 0
    for prediction, input_length, target_length in zip(
        predictions, cpu_input_lengths, cpu_target_lengths
    ):
        reference = cpu_targets[target_offset : target_offset + target_length]
        hypothesis = collapse_ctc(prediction[:input_length].tolist(), blank=blank)
        errors = edit_distance(reference, hypothesis)
        phone_errors += errors
        reference_phones += target_length
        exact_sequences += int(errors == 0)
        target_offset += target_length
    return normalized_loss_sum, phone_errors, reference_phones, exact_sequences


def discover_checkpoints(checkpoint_dir: Path | None, checkpoints: list[Path]) -> list[Path]:
    discovered = [path.expanduser().resolve() for path in checkpoints]
    if checkpoint_dir:
        discovered.extend(checkpoint_dir.expanduser().resolve().glob("step-*.pt"))
    unique = {path: None for path in discovered}
    paths = sorted(unique, key=lambda path: path.name)
    if not paths:
        raise ValueError("No checkpoints were provided or discovered")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint files do not exist: {missing}")
    return paths


def load_student_checkpoint(
    path: Path, config: ExperimentConfig, device: torch.device
) -> tuple[StreamingPhoneEncoder, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format_version") != 1:
        raise RuntimeError(f"Unsupported checkpoint format in {path}")
    checkpoint_config = payload.get("config", {})
    if checkpoint_config.get("model") != config.model.to_dict():
        raise RuntimeError(f"Model config differs in {path}")
    if checkpoint_config.get("distillation") != config.distillation.__dict__:
        raise RuntimeError(f"Distillation config differs in {path}")
    model = StreamingPhoneEncoder(config.model)
    model.load_state_dict(payload["model"], strict=True)
    step = int(payload["step"])
    del payload
    return model.eval().requires_grad_(False).to(device), step


def make_report(
    checkpoint_paths: list[Path],
    checkpoint_steps: list[int],
    statistics: torch.Tensor,
    config: ExperimentConfig,
    manifest: Path,
    dataset_items: int,
    world_size: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for path, step, values in zip(checkpoint_paths, checkpoint_steps, statistics.tolist()):
        feature_sum, feature_frames, ctc_sum, utterances, errors, phones, exact = values
        feature_loss = feature_sum / max(feature_frames, 1.0)
        ctc_loss = ctc_sum / max(utterances, 1.0)
        results.append(
            {
                "checkpoint": str(path),
                "step": step,
                "feature_loss": feature_loss,
                "ctc_loss": ctc_loss,
                "weighted_total_loss": (
                    config.distillation.feature_weight * feature_loss
                    + config.distillation.ctc_weight * ctc_loss
                ),
                "phone_error_rate": errors / max(phones, 1.0),
                "phone_errors": int(errors),
                "reference_phones": int(phones),
                "phone_sequence_accuracy": exact / max(utterances, 1.0),
                "exact_phone_sequences": int(exact),
                "utterances": int(utterances),
            }
        )
    return {
        "status": "PASS",
        "manifest": str(manifest),
        "manifest_items": dataset_items,
        "world_size": world_size,
        "elapsed_seconds": elapsed_seconds,
        "loss_weights": {
            "feature": config.distillation.feature_weight,
            "ctc": config.distillation.ctc_weight,
        },
        "best_by_weighted_total_loss": min(
            results, key=lambda result: result["weighted_total_loss"]
        )["checkpoint"],
        "best_by_phone_error_rate": min(results, key=lambda result: result["phone_error_rate"])[
            "checkpoint"
        ],
        "checkpoints": results,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Student Checkpoint Validation",
        "",
        f"- Status: **{report['status']}**",
        f"- Validation items: {report['manifest_items']}",
        f"- Distributed world size: {report['world_size']}",
        f"- Elapsed seconds: {report['elapsed_seconds']:.2f}",
        f"- Best total loss: `{Path(report['best_by_weighted_total_loss']).name}`",
        f"- Best phone error rate: `{Path(report['best_by_phone_error_rate']).name}`",
        "",
        "| Checkpoint | Step | Feature loss | CTC loss | Weighted total | PER | Phone exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["checkpoints"]:
        lines.append(
            f"| `{Path(result['checkpoint']).name}` | {result['step']} | "
            f"{result['feature_loss']:.6f} | {result['ctc_loss']:.6f} | "
            f"{result['weighted_total_loss']:.6f} | {result['phone_error_rate']:.4%} | "
            f"{result['phone_sequence_accuracy']:.4%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Student checkpoints")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batch-audio-seconds", type=float, default=180.0)
    parser.add_argument("--max-batch-items", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--teacher-attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()

    if (
        min(
            args.max_batch_audio_seconds,
            args.max_batch_items,
            args.prefetch_factor,
            args.progress_interval,
        )
        <= 0
    ):
        parser.error("Batching, prefetch, and progress settings must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    if args.checkpoint_dir is None and not args.checkpoint:
        parser.error("Provide --checkpoint-dir and/or at least one --checkpoint")

    config = load_config(args.config)
    if config.distillation.streaming_consistency_weight or config.distillation.anchor_weight:
        raise ValueError("Validation currently supports the active feature and CTC losses only")
    checkpoint_paths = discover_checkpoints(args.checkpoint_dir, args.checkpoint)
    device, rank, world_size, _ = validation_runtime(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = device.type == "cuda" and config.training.amp != "none"
    amp_dtype = torch.bfloat16 if config.training.amp == "bf16" else torch.float16

    dataset = PhoneManifestDataset(args.manifest)
    batches = pack_validation_batches(
        dataset.sample_counts,
        rank,
        world_size,
        round(args.max_batch_audio_seconds * dataset.sample_rate),
        args.max_batch_items,
    )
    loader_options: dict[str, Any] = {}
    if args.num_workers:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        **loader_options,
    )

    teacher = load_teacher(args.teacher, args.teacher_attention).to(device)
    verification_samples = max(400, dataset.sample_rate // 2)
    verification_waveform = torch.zeros(1, verification_samples, device=device)
    verification_lengths = torch.tensor([verification_samples], device=device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
        teacher_check = verify_optimized_teacher(
            teacher,
            verification_waveform,
            verification_lengths,
            config.distillation.teacher_layer,
        )

    models: list[StreamingPhoneEncoder] = []
    checkpoint_steps: list[int] = []
    for path in checkpoint_paths:
        model, step = load_student_checkpoint(path, config, device)
        models.append(model)
        checkpoint_steps.append(step)
        if rank == 0:
            print(json.dumps({"checkpoint_loaded": str(path), "step": step}), flush=True)

    statistics = torch.zeros(len(models), 7, device=device, dtype=torch.float64)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            waveform = batch["waveform"].to(device, non_blocking=True)
            sample_lengths = batch["sample_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                teacher_hidden, teacher_lengths, convolution = optimized_teacher_targets(
                    teacher,
                    waveform,
                    sample_lengths,
                    config.distillation.teacher_layer,
                )
                for model_index, model in enumerate(models):
                    output = model(
                        waveform,
                        sample_lengths,
                        convolution_features=convolution,
                    )
                    if not torch.equal(output["output_lengths"], teacher_lengths):
                        raise RuntimeError("Teacher and Student frame lengths differ")
                    feature_sum, feature_frames = feature_statistics(
                        output["distill_features"], teacher_hidden, teacher_lengths
                    )
                    ctc_sum, errors, reference_phones, exact = ctc_statistics(
                        output["phone_logits"],
                        targets,
                        teacher_lengths,
                        target_lengths,
                    )
                    statistics[model_index] += torch.stack(
                        (
                            feature_sum.double(),
                            feature_frames.double(),
                            ctc_sum.double(),
                            torch.tensor(float(len(batch["utterance_ids"])), device=device),
                            torch.tensor(float(errors), device=device),
                            torch.tensor(float(reference_phones), device=device),
                            torch.tensor(float(exact), device=device),
                        )
                    )
            if rank == 0 and (batch_index == 1 or batch_index % args.progress_interval == 0):
                print(
                    json.dumps(
                        {
                            "validation_batches_completed_rank0": batch_index,
                            "validation_batches_rank0": len(batches),
                        }
                    ),
                    flush=True,
                )

    elapsed = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if int(statistics[0, 3].item()) != len(dataset):
        raise RuntimeError(
            f"Validated {int(statistics[0, 3].item())} items, expected {len(dataset)}"
        )

    if rank == 0:
        report = make_report(
            checkpoint_paths,
            checkpoint_steps,
            statistics.cpu(),
            config,
            args.manifest.expanduser().resolve(),
            len(dataset),
            world_size,
            float(elapsed),
        )
        report["teacher_optimization_check"] = teacher_check
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
        for result in report["checkpoints"]:
            print(json.dumps(result), flush=True)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "best_by_weighted_total_loss": report["best_by_weighted_total_loss"],
                    "best_by_phone_error_rate": report["best_by_phone_error_rate"],
                    "report_json": str((args.output_dir / "report.json").resolve()),
                    "report_markdown": str((args.output_dir / "report.md").resolve()),
                }
            ),
            flush=True,
        )
        print("student_checkpoint_validation=PASS", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
