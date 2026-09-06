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
from .data import PhoneManifestDataset, StatefulDistributedSampler, collate
from .losses import ctc_loss, valid_feature_loss
from .model import StreamingPhoneEncoder, parameter_breakdown
from .teacher import load_teacher, teacher_targets


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
    device = torch.device(
        "cuda" if device_arg == "auto" and torch.cuda.is_available() else device_arg
    )
    if device_arg == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device, rank, world_size, local_rank


def override_max_steps(config: ExperimentConfig, max_steps: int | None) -> ExperimentConfig:
    if max_steps is None:
        return config
    return replace(config, training=replace(config.training, max_steps=max_steps))


def optimizer_step_due(micro_step: int, grad_accum: int) -> bool:
    return (micro_step + 1) % grad_accum == 0


def collect_step_metrics(
    device: torch.device,
    world_size: int,
    audio_seconds: float,
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
    elapsed = torch.tensor(elapsed_seconds, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        losses /= world_size
        dist.all_reduce(audio, op=dist.ReduceOp.SUM)
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
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    if min(args.batch_size, args.grad_accum, args.prefetch_factor) <= 0:
        parser.error("batch-size, grad-accum, and prefetch-factor must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")

    config = override_max_steps(load_config(args.config), args.max_steps)
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
    sampler = StatefulDistributedSampler(len(dataset), rank, world_size, config.training.seed)
    loader_options: dict[str, Any] = {}
    if args.num_workers:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        **loader_options,
    )
    teacher = load_teacher(args.teacher).to(device)
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
    warmup = max(1, round(config.training.max_steps * config.training.warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(config.training.max_steps - step, 0) / max(config.training.max_steps - warmup, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    use_amp = device.type == "cuda" and config.training.amp != "none"
    amp_dtype = torch.bfloat16 if config.training.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.training.amp == "fp16")
    step = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        if payload["config"] != config.to_dict():
            raise RuntimeError("Resume config differs from the current config")
        sampler.load_state_dict(payload["data_state"]["sampler"])
        step = int(payload["step"])
    training_model: torch.nn.Module = model
    ddp_model: DistributedDataParallel | None = None
    if world_size > 1:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        training_model = ddp_model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(
            json.dumps(
                {
                    "parameters": parameter_breakdown(model),
                    "config": config.to_dict(),
                    "runtime": {
                        "world_size": world_size,
                        "batch_size_per_rank": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "effective_global_batch_size": (
                            world_size * args.batch_size * args.grad_accum
                        ),
                        "num_workers_per_rank": args.num_workers,
                        "prefetch_factor": args.prefetch_factor if args.num_workers else None,
                        "amp": config.training.amp,
                        "teacher_autocast": use_amp,
                        "fused_adamw": device.type == "cuda",
                    },
                }
            )
        )
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    window_audio_seconds = 0.0
    window_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < config.training.max_steps:
        for batch in loader:
            window_audio_seconds += float(batch["sample_lengths"].sum()) / dataset.sample_rate
            waveform = batch["waveform"].to(device, non_blocking=True)
            lengths = batch["sample_lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                teacher_hidden, teacher_lengths = teacher_targets(
                    teacher, waveform, lengths, config.distillation.teacher_layer
                )
            synchronize = optimizer_step_due(micro_step, args.grad_accum)
            synchronization_context = (
                nullcontext() if ddp_model is None or synchronize else ddp_model.no_sync()
            )
            with synchronization_context:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    output = training_model(waveform, lengths)
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
            should_log = step == 1 or step % config.training.log_interval == 0
            if should_log:
                metrics = collect_step_metrics(
                    device,
                    world_size,
                    window_audio_seconds,
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
                window_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            should_save = (
                step % config.training.save_interval == 0 or step == config.training.max_steps
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
                        {"sampler": sampler.state_dict()},
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
