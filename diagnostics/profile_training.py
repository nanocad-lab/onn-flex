#!/usr/bin/env python3
"""
Profiling script for ONN training to identify performance bottlenecks.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.profiler
from torch.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, record_function

from onn_config import load_app_config_from_yaml
from onn_train import build_model, get_data_loaders


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
        "--model-arch",
        choices=["fftconvnet", "resnet11", "resnet18"],
        default=None,
        help="Override the model architecture from the config.",
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
        "--jtc-assume-nonnegative-input",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Skip signed input decomposition when all JTCConv2d inputs are nonnegative.",
    )
    parser.add_argument(
        "--jtc-max-shots",
        type=int,
        default=None,
        help="Maximum optical shots per JTCConv2d chunk.",
    )
    parser.add_argument(
        "--enable-jtc-activation-checkpointing",
        type=_str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Recompute JTCConv2d optical activations during backward to reduce memory.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--wait-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-dir", default="profiler_logs")
    parser.add_argument("--trace-file", default="trace.json")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=8,
        help="DataLoader worker processes for the profiling run.",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable DataLoader pinned host memory (true/false).",
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Keep DataLoader workers alive across epochs (true/false).",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per DataLoader worker; ignored when workers=0.",
    )
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
    if args.model_arch is not None:
        config.model_arch = args.model_arch
    if args.enable_jtc_batched_fast_path is not None:
        config.enable_jtc_batched_fast_path = args.enable_jtc_batched_fast_path
    if args.jtc_assume_nonnegative_input is not None:
        config.jtc_assume_nonnegative_input = args.jtc_assume_nonnegative_input
    if args.jtc_max_shots is not None:
        config.jtc_max_shots = args.jtc_max_shots
    if args.enable_jtc_activation_checkpointing is not None:
        config.enable_jtc_activation_checkpointing = (
            args.enable_jtc_activation_checkpointing
        )
    config.learning_rate = args.learning_rate
    config.__post_init__()

    wait_steps = max(int(args.wait_steps), 0)
    warmup_steps = max(int(args.warmup_steps), 0)
    profile_steps = max(int(args.profile_steps), 1)
    total_steps = wait_steps + warmup_steps + profile_steps

    # Get data
    trainloader, _, _ = get_data_loaders(
        config.batch_size,
        dataset=config.dataset,
        seed=config.seed if config.seed is not None else 0,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=args.dataloader_persistent_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
    )
    model = build_model(config).to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler("cuda") if use_amp else None

    # Profiler setup
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)

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
        f"model_arch={config.model_arch}, backend={config.conv_backend}, "
        f"batch_size={config.batch_size}, "
        f"num_identical_layers={config.num_identical_layers}, amp={use_amp}, "
        f"jtc_fast_path={config.enable_jtc_batched_fast_path}, "
        f"jtc_assume_nonnegative_input={config.jtc_assume_nonnegative_input}, "
        f"jtc_max_shots={config.jtc_max_shots}, "
        f"jtc_activation_checkpointing={config.enable_jtc_activation_checkpointing}, "
        f"dataloader_num_workers={args.dataloader_num_workers}, "
        f"dataloader_pin_memory={args.dataloader_pin_memory}, "
        f"dataloader_persistent_workers={args.dataloader_persistent_workers}, "
        f"dataloader_prefetch_factor={args.dataloader_prefetch_factor}"
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
        step_times: list[float] = []
        for step_idx in range(total_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_start
            step_times.append(elapsed)
            print(
                f"Step {step_idx + 1}/{total_steps} completed in {elapsed:.3f}s",
                flush=True,
            )

    # Print summary
    print("\n" + "=" * 80)
    print("WALL TIME:")
    print("=" * 80)
    active_start = wait_steps + warmup_steps
    active_times = step_times[active_start:]
    if active_times:
        mean_active = sum(active_times) / len(active_times)
        print(
            f"Profiled steps: mean={mean_active:.3f}s, "
            f"min={min(active_times):.3f}s, max={max(active_times):.3f}s"
        )
    print(f"All step times: {[round(value, 3) for value in step_times]}")

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
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
        print(
            f"\nPeak CUDA memory: allocated={peak_alloc:.2f} GiB, "
            f"reserved={peak_reserved:.2f} GiB"
        )

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
