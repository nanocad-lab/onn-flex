#!/usr/bin/env python3
"""
Profiling script for ONN training to identify performance bottlenecks.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
from onn_train import get_data_loaders, FFTConvNet
from onn_main import load_yaml_config
from onn_config import AppConfig


def profile_training() -> None:
    """Profile a few training iterations to identify bottlenecks."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    # Load config (you may need to adjust path)
    config = AppConfig()
    config = load_yaml_config("configs/config_ideal.yaml")
    config.batch_size = 32  # Smaller batch for profiling
    config.num_epochs = 1
    config.run_pretrain_tests = False

    # Get data
    trainloader, _ = get_data_loaders(config.batch_size)
    model = FFTConvNet(config).to(device)

    # Profiler setup
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Profile just a few batches
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=torch.profiler.schedule(
            wait=1,  # Skip first batch
            warmup=5,  # Warmup for 5 batches
            active=5,  # Profile 5 batches
            repeat=1,  # Only run once
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    ) as prof:
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            if batch_idx >= 5:  # Only profile first 5 batches
                break

            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            with record_function("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            with record_function("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            prof.step()  # Signal profiler to advance to next step
            print(f"Batch {batch_idx + 1} completed")

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
    prof.export_chrome_trace("trace.json")
    print("\nDetailed trace exported to: trace.json")
    print(
        "You can view this in Chrome by going to chrome://tracing and loading the file"
    )

    print("\nTensorBoard logs saved to: ./profiler_logs")
    print("Run: tensorboard --logdir=./profiler_logs to view detailed profiling data")


if __name__ == "__main__":
    profile_training()
