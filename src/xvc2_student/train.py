from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
import random
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, load_config
from .data import (
    PhoneManifestDataset,
    StatefulDistributedBatchSampler,
    StatefulDistributedSampler,
    collate,
)
from .losses import ctc_loss, valid_feature_loss
from .model import StreamingPhoneEncoder, parameter_breakdown
from .teacher import (
    load_teacher,
    optimized_teacher_targets,
    teacher_targets,
    verify_optimized_teacher,
)


def runtime(device_arg: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device(f"cuda:{local_rank}"), rank, world_size, local_rank
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    return device, rank, world_size, local_rank


def override_max_steps(config: ExperimentConfig, max_steps: int | None) -> ExperimentConfig:
    if max_steps is None:
        return config
    return replace(config, training=replace(config.training, max_steps=max_steps))


def extension_config(
    config: ExperimentConfig,
    end_step: int,
    learning_rate: float,
    warmup_ratio: float,
) -> ExperimentConfig:
    return replace(
        config,
        training=replace(
            config.training,
            max_steps=end_step,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
        ),
    )


def schedule_specification(
    start_step: int,
    end_step: int,
    learning_rate: float,
    warmup_ratio: float,
) -> dict[str, Any]:
    if not 0 <= start_step < end_step:
        raise ValueError("Schedule requires 0 <= start_step < end_step")
    if learning_rate <= 0 or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("Invalid schedule learning rate or warmup ratio")
    return {
        "kind": "linear_warmup_decay",
        "start_step": start_step,
        "end_step": end_step,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    specification: dict[str, Any],
    preserve_current_learning_rates: bool = False,
) -> torch.optim.lr_scheduler.LambdaLR:
    phase_steps = int(specification["end_step"]) - int(specification["start_step"])
    warmup = max(1, round(phase_steps * float(specification["warmup_ratio"])))
    peak_learning_rate = float(specification["learning_rate"])
    current_learning_rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = peak_learning_rate
        group["initial_lr"] = peak_learning_rate

    def multiplier(phase_step: int) -> float:
        if phase_step < warmup:
            return (phase_step + 1) / warmup
        return max(phase_steps - phase_step, 0) / max(phase_steps - warmup, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    if preserve_current_learning_rates:
        for group, learning_rate in zip(optimizer.param_groups, current_learning_rates):
            group["lr"] = learning_rate
    return scheduler


def validate_prior_config_for_extension(
    checkpoint_config: dict[str, Any], base_config: ExperimentConfig
) -> None:
    expected = base_config.to_dict()
    actual = dict(checkpoint_config)
    actual_training = dict(actual.get("training", {}))
    expected_training = dict(expected["training"])
    actual_training.pop("max_steps", None)
    expected_training.pop("max_steps", None)
    actual["training"] = actual_training
    expected["training"] = expected_training
    if actual != expected:
        raise RuntimeError("Extension checkpoint config differs from the base config")


def optimizer_step_due(micro_step: int, grad_accum: int) -> bool:
    return (micro_step + 1) % grad_accum == 0


def ddp_options(local_rank: int) -> dict[str, Any]:
    return {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
    }


def collect_step_metrics(
    device: torch.device,
    world_size: int,
    audio_seconds: float,
    item_count: int,
    optimizer_steps: int,
    elapsed_seconds: float,
    feature: torch.Tensor,
    phones: torch.Tensor,
    gradient_norm: torch.Tensor,
) -> dict[str, Any]:
    losses = torch.tensor(
        [float(feature), float(phones), float(gradient_norm)],
        device=device,
        dtype=torch.float64,
    )
    audio = torch.tensor(audio_seconds, device=device, dtype=torch.float64)
    items = torch.tensor(item_count, device=device, dtype=torch.float64)
    elapsed = torch.tensor(elapsed_seconds, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        losses /= world_size
        dist.all_reduce(audio, op=dist.ReduceOp.SUM)
        dist.all_reduce(items, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    memory_by_rank: list[dict[str, float]] = []
    if device.type == "cuda":
        local_memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / 2**30,
                torch.cuda.max_memory_reserved(device) / 2**30,
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            gathered = [torch.zeros_like(local_memory) for _ in range(world_size)]
            dist.all_gather(gathered, local_memory)
        else:
            gathered = [local_memory]
        memory_by_rank = [
            {
                "rank": rank,
                "peak_allocated_gib": float(values[0]),
                "peak_reserved_gib": float(values[1]),
            }
            for rank, values in enumerate(gathered)
        ]
    return {
        "feature_loss": float(losses[0]),
        "ctc_loss": float(losses[1]),
        "gradient_norm": float(losses[2]),
        "global_audio_seconds": float(audio),
        "global_items": int(items),
        "mean_global_batch_size": float(items / max(optimizer_steps, 1)),
        "elapsed_seconds": float(elapsed),
        "global_audio_seconds_per_second": float(audio / elapsed.clamp_min(1e-9)),
        "memory_by_rank": memory_by_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 12x768 streaming phone Student")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batch-audio-seconds", type=float)
    parser.add_argument("--max-batch-items", type=int, default=8)
    parser.add_argument("--duration-bucket-size", type=int, default=2_048)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--extension-end-step", type=int)
    parser.add_argument("--extension-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--extension-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--teacher-attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--full-teacher-forward", action="store_true")
    parser.add_argument("--no-shared-frontend", action="store_true")
    parser.add_argument("--skip-teacher-optimization-check", action="store_true")
    args = parser.parse_args()

    if (
        min(
            args.batch_size,
            args.max_batch_items,
            args.duration_bucket_size,
            args.grad_accum,
            args.prefetch_factor,
        )
        <= 0
    ):
        parser.error("Batch, bucket, accumulation, and prefetch settings must be positive")
    if args.max_batch_audio_seconds is not None and args.max_batch_audio_seconds <= 0:
        parser.error("max-batch-audio-seconds must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.extension_end_step is not None:
        if args.resume is None:
            parser.error("extension-end-step requires --resume")
        if args.max_steps is not None:
            parser.error("extension-end-step and max-steps cannot be used together")
        if args.extension_end_step <= 0 or args.extension_learning_rate <= 0:
            parser.error("Extension end step and learning rate must be positive")
        if not 0.0 <= args.extension_warmup_ratio < 1.0:
            parser.error("extension-warmup-ratio must be in [0, 1)")

    base_config = load_config(args.config)
    config = (
        extension_config(
            base_config,
            args.extension_end_step,
            args.extension_learning_rate,
            args.extension_warmup_ratio,
        )
        if args.extension_end_step is not None
        else override_max_steps(base_config, args.max_steps)
    )
    if config.distillation.streaming_consistency_weight:
        raise ValueError(
            "Streaming-state training is an experimental second-stage option, not enabled here"
        )
    device, rank, world_size, local_rank = runtime(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(config.training.seed + rank)
    torch.manual_seed(config.training.seed + rank)
    dataset = PhoneManifestDataset(args.manifest)
    dynamic_batching = args.max_batch_audio_seconds is not None
    if dynamic_batching:
        sampler: StatefulDistributedSampler | StatefulDistributedBatchSampler = (
            StatefulDistributedBatchSampler(
                dataset.sample_counts,
                rank,
                world_size,
                config.training.seed,
                max_batch_samples=round(args.max_batch_audio_seconds * dataset.sample_rate),
                max_batch_items=args.max_batch_items,
                bucket_size=args.duration_bucket_size,
            )
        )
    else:
        sampler = StatefulDistributedSampler(len(dataset), rank, world_size, config.training.seed)
    loader_options: dict[str, Any] = {}
    if args.num_workers:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    if dynamic_batching:
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
            **loader_options,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
            **loader_options,
        )
    teacher = load_teacher(args.teacher, args.teacher_attention).to(device)
    model = StreamingPhoneEncoder(config.model)
    model.load_teacher_frontend(teacher)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_options: dict[str, Any] = {}
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        **optimizer_options,
    )
    use_amp = device.type == "cuda" and config.training.amp != "none"
    amp_dtype = torch.bfloat16 if config.training.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.training.amp == "fp16")
    step = 0
    payload: dict[str, Any] | None = None
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scaler=scaler)
        sampler.load_state_dict(payload["data_state"]["sampler"])
        step = int(payload["step"])
    loaded_schedule = payload.get("data_state", {}).get("schedule") if payload else None
    if args.extension_end_step is not None:
        if step >= args.extension_end_step:
            raise RuntimeError("Extension end step must be greater than the checkpoint step")
        resuming_extension = bool(loaded_schedule and int(loaded_schedule["start_step"]) > 0)
        if resuming_extension:
            expected_schedule = schedule_specification(
                int(loaded_schedule["start_step"]),
                args.extension_end_step,
                args.extension_learning_rate,
                args.extension_warmup_ratio,
            )
            if loaded_schedule != expected_schedule or payload["config"] != config.to_dict():
                raise RuntimeError("Extension settings differ from the continuation checkpoint")
            schedule = loaded_schedule
            scheduler = build_scheduler(optimizer, schedule, preserve_current_learning_rates=True)
            scheduler.load_state_dict(payload["scheduler"])
        else:
            if step != int(payload["config"]["training"]["max_steps"]):
                raise RuntimeError("Start an extension from a completed training checkpoint")
            validate_prior_config_for_extension(payload["config"], base_config)
            schedule = schedule_specification(
                step,
                args.extension_end_step,
                args.extension_learning_rate,
                args.extension_warmup_ratio,
            )
            scheduler = build_scheduler(optimizer, schedule)
    else:
        if payload and payload["config"] != config.to_dict():
            raise RuntimeError("Resume config differs from the current config")
        if loaded_schedule and int(loaded_schedule["start_step"]) > 0:
            raise RuntimeError("Resume an extension checkpoint with --extension-end-step")
        schedule = loaded_schedule or schedule_specification(
            0,
            config.training.max_steps,
            config.training.learning_rate,
            config.training.warmup_ratio,
        )
        scheduler = build_scheduler(
            optimizer,
            schedule,
            preserve_current_learning_rates=payload is not None,
        )
        if payload:
            scheduler.load_state_dict(payload["scheduler"])
    training_model: torch.nn.Module = model
    ddp_model: DistributedDataParallel | None = None
    if world_size > 1:
        ddp_model = DistributedDataParallel(model, **ddp_options(local_rank))
        training_model = ddp_model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_optimization_report = None
    if not args.full_teacher_forward and not args.skip_teacher_optimization_check:
        verification_samples = max(model.receptive_field_samples, dataset.sample_rate // 2)
        verification_waveform = torch.zeros(1, verification_samples, device=device)
        verification_lengths = torch.tensor([verification_samples], device=device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            teacher_optimization_report = verify_optimized_teacher(
                teacher,
                verification_waveform,
                verification_lengths,
                config.distillation.teacher_layer,
            )
    batches_remaining_at_start = len(loader)
    if dynamic_batching:
        batches_per_epoch = batches_remaining_at_start + sampler.position
    else:
        items_per_epoch = len(sampler) + sampler.position
        batches_per_epoch = (items_per_epoch + args.batch_size - 1) // args.batch_size
    if rank == 0:
        print(
            json.dumps(
                {
                    "parameters": parameter_breakdown(model),
                    "config": config.to_dict(),
                    "runtime": {
                        "world_size": world_size,
                        "batch_size_per_rank": None if dynamic_batching else args.batch_size,
                        "max_batch_audio_seconds_per_rank": args.max_batch_audio_seconds,
                        "max_batch_items_per_rank": (
                            args.max_batch_items if dynamic_batching else None
                        ),
                        "duration_bucket_size": (
                            args.duration_bucket_size if dynamic_batching else None
                        ),
                        "grad_accum": args.grad_accum,
                        "effective_global_batch_size": (
                            None
                            if dynamic_batching
                            else world_size * args.batch_size * args.grad_accum
                        ),
                        "num_workers_per_rank": args.num_workers,
                        "prefetch_factor": args.prefetch_factor if args.num_workers else None,
                        "amp": config.training.amp,
                        "teacher_autocast": use_amp,
                        "teacher_attention": args.teacher_attention,
                        "optimized_teacher_forward": not args.full_teacher_forward,
                        "shared_convolution_frontend": (
                            not args.full_teacher_forward and not args.no_shared_frontend
                        ),
                        "fused_adamw": device.type == "cuda",
                        "schedule": schedule,
                        "batches_per_epoch_per_rank": batches_per_epoch,
                        "batches_remaining_at_start_per_rank": batches_remaining_at_start,
                        "dynamic_padding_batches": (
                            sampler.padding_batches if dynamic_batching else None
                        ),
                    },
                    "teacher_optimization_check": teacher_optimization_report,
                }
            )
        )
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    window_audio_seconds = 0.0
    window_items = 0
    window_optimizer_steps = 0
    window_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < config.training.max_steps:
        for batch in loader:
            window_audio_seconds += float(batch["sample_lengths"].sum()) / dataset.sample_rate
            window_items += len(batch["utterance_ids"])
            waveform = batch["waveform"].to(device, non_blocking=True)
            lengths = batch["sample_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                if args.full_teacher_forward:
                    teacher_hidden, teacher_lengths = teacher_targets(
                        teacher, waveform, lengths, config.distillation.teacher_layer
                    )
                    convolution_features = None
                else:
                    teacher_hidden, teacher_lengths, convolution_features = (
                        optimized_teacher_targets(
                            teacher,
                            waveform,
                            lengths,
                            config.distillation.teacher_layer,
                        )
                    )
                    if args.no_shared_frontend:
                        convolution_features = None
            synchronize = optimizer_step_due(micro_step, args.grad_accum)
            synchronization_context = (
                nullcontext() if ddp_model is None or synchronize else ddp_model.no_sync()
            )
            with synchronization_context:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    output = training_model(
                        waveform,
                        lengths,
                        convolution_features=convolution_features,
                    )
                    if not torch.equal(output["output_lengths"], teacher_lengths):
                        raise RuntimeError("Teacher and Student frame lengths differ")
                    feature = valid_feature_loss(
                        output["distill_features"], teacher_hidden, teacher_lengths
                    )
                    phones = ctc_loss(
                        output["phone_logits"], targets, teacher_lengths, target_lengths
                    )
                    loss = (
                        config.distillation.feature_weight * feature
                        + config.distillation.ctc_weight * phones
                    ) / args.grad_accum
                scaler.scale(loss).backward()
            if dynamic_batching:
                sampler.advance()
            else:
                sampler.advance(len(batch["utterance_ids"]))
            micro_step += 1
            if not synchronize:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            window_optimizer_steps += 1
            should_log = step == 1 or step % config.training.log_interval == 0
            if should_log:
                metrics = collect_step_metrics(
                    device,
                    world_size,
                    window_audio_seconds,
                    window_items,
                    window_optimizer_steps,
                    time.perf_counter() - window_started,
                    feature,
                    phones,
                    gradient_norm,
                )
            if rank == 0 and should_log:
                print(
                    json.dumps(
                        {
                            "step": step,
                            **metrics,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                )
            if should_log:
                window_audio_seconds = 0.0
                window_items = 0
                window_optimizer_steps = 0
                window_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            phase_step = step - int(schedule["start_step"])
            should_save = (
                phase_step % config.training.save_interval == 0 or step == config.training.max_steps
            )
            if should_save:
                if rank == 0:
                    save_checkpoint(
                        args.output_dir / f"step-{step:06d}.pt",
                        model,
                        config,
                        step,
                        optimizer,
                        scheduler,
                        scaler,
                        {"sampler": sampler.state_dict(), "schedule": schedule},
                    )
                if world_size > 1:
                    dist.barrier()
            if step >= config.training.max_steps:
                break
        sampler.next_epoch()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
