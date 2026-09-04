from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import load_config
from .model import StreamingPhoneEncoder, parameter_breakdown
from .smoke import tiny_config


def initialize(device_arg: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        use_cuda = torch.cuda.is_available() and device_arg != "cpu"
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if use_cuda else "gloo")
        return (
            torch.device(f"cuda:{local_rank}" if use_cuda else "cpu"),
            rank,
            world_size,
            local_rank,
        )
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_arg), rank, world_size, local_rank


def run(args: argparse.Namespace) -> dict[str, object]:
    device, rank, world_size, local_rank = initialize(args.device)
    torch.manual_seed(args.seed + rank)
    config = tiny_config() if args.tiny else load_config(args.config).model
    model = StreamingPhoneEncoder(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    training_model: torch.nn.Module = model
    if world_size > 1:
        training_model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    samples = max(
        round(args.audio_seconds * 16_000), model.receptive_field_samples + model.stride_samples * 8
    )
    waveform = torch.randn(args.batch_size, samples, device=device)
    lengths = torch.full((args.batch_size,), samples, device=device, dtype=torch.long)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    measured = []
    total_steps = args.warmup_steps + args.steps
    for index in range(total_steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = training_model(waveform, lengths)
        frame_count = int(output["output_lengths"].min())
        target_length = max(1, min(frame_count // 4, 8))
        targets = torch.randint(
            1, config.vocab_size, (args.batch_size * target_length,), device=device
        )
        target_lengths = torch.full(
            (args.batch_size,), target_length, device=device, dtype=torch.long
        )
        ctc = torch.nn.functional.ctc_loss(
            output["phone_logits"].float().log_softmax(-1).transpose(0, 1),
            targets,
            output["output_lengths"],
            target_lengths,
            zero_infinity=True,
        )
        loss = output["distill_features"].float().square().mean() + 0.1 * ctc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= args.warmup_steps:
            measured.append(elapsed)
    maximum_elapsed = sum(measured)
    if world_size > 1:
        value = torch.tensor(maximum_elapsed, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        maximum_elapsed = float(value)
    report = {
        "status": "PASS",
        "device": str(device),
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "audio_seconds_per_item": samples / 16_000,
        "steps": args.steps,
        "mean_step_seconds": maximum_elapsed / args.steps,
        "global_audio_seconds_per_second": (
            world_size * args.batch_size * args.steps * samples / 16_000 / maximum_elapsed
        ),
        "peak_memory_bytes_per_rank": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "parameters": parameter_breakdown(model),
    }
    if rank == 0:
        print(json.dumps(report, sort_keys=True))
        print("student_training_benchmark=PASS")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Student train/DDP resource benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/student_12x768.yaml"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-seconds", type=float, default=3.2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.steps) <= 0 or min(args.audio_seconds, args.warmup_steps) < 0:
        parser.error("batch size and steps must be positive; durations cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
