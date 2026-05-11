#!/usr/bin/env python3
"""
Profiling script for ONN training to identify performance bottlenecks.
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
from torch.amp import autocast, GradScaler
from onn_train import get_data_loaders, FFTConvNet
from onn_config import load_app_config_from_yaml


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    val = v.strip().lower()
    if val in {"yes", "true", "t", "1", "y"}:
        return True
    if val in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a short ONN training run with torch.profiler."
    )
    parser.add_argument("--config-file", default="configs/config_ideal.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--conv-backend",
        choices=["pytorch", "jtc_ideal", "jtc_emulation"],
        default=None,
        help="Override the convolution backend from the config.",
    )
    parser.add_argument(
        "--num-identical-layers",
        type=int,
        default=None,
        help="Override the number of repeated convolution blocks.",
    )
    parser.add_argument(
        "--enable-jtc-batched-fast-path",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Use batched JTC calls to reduce launch overhead (true/false).",
    )
    parser.add_argument(
        "--enable-jtc-ideal-fused-transfer",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Use fused ideal JTC transfer functions when applicable (true/false).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--wait-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-dir", default="profiler_logs")
    parser.add_argument("--trace-file", default="trace.json")
    parser.add_argument(
        "--no-stack",
        action="store_true",
        help="Disable Python stack collection to reduce profiler overhead.",
    )
    parser.add_argument(
        "--no-shapes",
        action="store_true",
        help="Disable tensor shape collection to reduce profiler overhead.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory profiling to reduce profiler overhead.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA autocast/GradScaler even when CUDA is available.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not write TensorBoard or Chrome trace files.",
    )
    return parser.parse_args()


def profile_training() -> None:
    """Profile a few training iterations to identify bottlenecks."""
    args = parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    config = load_app_config_from_yaml(args.config_file)
    config.batch_size = args.batch_size
    config.num_epochs = 1
    config.run_pretrain_tests = False
    if args.conv_backend is not None:
        config.conv_backend = args.conv_backend
    if args.num_identical_layers is not None:
        config.num_identical_layers = args.num_identical_layers
    if args.enable_jtc_batched_fast_path is not None:
        config.enable_jtc_batched_fast_path = args.enable_jtc_batched_fast_path
    if args.enable_jtc_ideal_fused_transfer is not None:
        config.enable_jtc_ideal_fused_transfer = args.enable_jtc_ideal_fused_transfer
    config.learning_rate = args.learning_rate

    wait_steps = max(int(args.wait_steps), 0)
    warmup_steps = max(int(args.warmup_steps), 0)
    profile_steps = max(int(args.profile_steps), 1)
    total_steps = wait_steps + warmup_steps + profile_steps

    # Get data
    trainloader, _, _ = get_data_loaders(config.batch_size)
    model = FFTConvNet(config).to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler("cuda") if use_amp else None

    # Profiler setup
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_handler = None
    if not args.no_trace:
        trace_handler = torch.profiler.tensorboard_trace_handler(args.trace_dir)

    print(
        "Profiler schedule: "
        f"wait={wait_steps}, warmup={warmup_steps}, active={profile_steps}, "
        f"total_steps={total_steps}"
    )
    print(
        "Config: "
        f"backend={config.conv_backend}, batch_size={config.batch_size}, "
        f"num_identical_layers={config.num_identical_layers}, amp={use_amp}, "
        f"jtc_fast_path={config.enable_jtc_batched_fast_path}, "
        f"jtc_ideal_fused_transfer={config.enable_jtc_ideal_fused_transfer}"
    )

    # Profile a bounded number of real training steps. The loop length must cover
    # the full profiler schedule, otherwise the active window records no steps.
    with profile(
        activities=activities,
        record_shapes=not args.no_shapes,
        profile_memory=not args.no_memory,
        with_stack=not args.no_stack,
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=profile_steps,
            repeat=1,  # Only run once
        ),
        on_trace_ready=trace_handler,
    ) as prof:
        train_iter = iter(trainloader)
        for step_idx in range(total_steps):
            with record_function("dataloader_next"):
                inputs, labels = next(train_iter)

            with record_function("copy_to_device"):
                inputs, labels = (
                    inputs.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )

            with record_function("forward_pass"):
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            with record_function("backward_pass"):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with record_function("optimizer_step"):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            prof.step()  # Signal profiler to advance to next step
            print(f"Step {step_idx + 1}/{total_steps} completed", flush=True)

    # Print summary
    print("\n" + "=" * 80)
    print("TOP CPU TIME CONSUMING OPERATIONS:")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if device.type == "cuda":
        print("\n" + "=" * 80)
        print("TOP CUDA TIME CONSUMING OPERATIONS:")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n" + "=" * 80)
    print("MEMORY USAGE:")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    # Export detailed trace
    if not args.no_trace:
        prof.export_chrome_trace(args.trace_file)
        print(f"\nDetailed trace exported to: {args.trace_file}")
        print(
            "You can view this in Chrome by going to chrome://tracing and loading the file"
        )

        print(f"\nTensorBoard logs saved to: {args.trace_dir}")
        print(
            f"Run: tensorboard --logdir={args.trace_dir} "
            "to view detailed profiling data"
        )


if __name__ == "__main__":
    profile_training()
