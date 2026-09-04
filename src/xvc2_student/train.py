from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
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
    args = parser.parse_args()

    config = load_config(args.config)
    if config.distillation.streaming_consistency_weight:
        raise ValueError(
            "Streaming-state training is an experimental second-stage option, not enabled here"
        )
    device, rank, world_size, local_rank = runtime(args.device)
    random.seed(config.training.seed + rank)
    torch.manual_seed(config.training.seed + rank)
    dataset = PhoneManifestDataset(args.manifest)
    sampler = StatefulDistributedSampler(len(dataset), rank, world_size, config.training.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    teacher = load_teacher(args.teacher).to(device)
    model = StreamingPhoneEncoder(config.model)
    model.load_teacher_frontend(teacher)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
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
    if world_size > 1:
        training_model = DistributedDataParallel(model, device_ids=[local_rank])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(json.dumps({"parameters": parameter_breakdown(model), "config": config.to_dict()}))
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    while step < config.training.max_steps:
        for batch in loader:
            waveform = batch["waveform"].to(device)
            lengths = batch["sample_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            teacher_hidden, teacher_lengths = teacher_targets(
                teacher, waveform, lengths, config.distillation.teacher_layer
            )
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                output = training_model(waveform, lengths)
                if not torch.equal(output["output_lengths"], teacher_lengths):
                    raise RuntimeError("Teacher and Student frame lengths differ")
                feature = valid_feature_loss(
                    output["distill_features"], teacher_hidden, teacher_lengths
                )
                phones = ctc_loss(output["phone_logits"], targets, teacher_lengths, target_lengths)
                loss = (
                    config.distillation.feature_weight * feature
                    + config.distillation.ctc_weight * phones
                ) / args.grad_accum
            scaler.scale(loss).backward()
            sampler.advance(len(batch["utterance_ids"]))
            micro_step += 1
            if micro_step % args.grad_accum:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if rank == 0 and (step == 1 or step % config.training.log_interval == 0):
                print(
                    json.dumps(
                        {
                            "step": step,
                            "feature_loss": float(feature),
                            "ctc_loss": float(phones),
                            "gradient_norm": float(gradient_norm),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                )
            if rank == 0 and (
                step % config.training.save_interval == 0 or step == config.training.max_steps
            ):
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
            if step >= config.training.max_steps:
                break
        sampler.next_epoch()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
