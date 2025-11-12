#!/usr/bin/env python3
"""
Simple comparison test to verify refactored JTC produces identical outputs.

This script is minimal and can be run quickly to verify correctness.
"""

import torch
import sys
from onn_config import AppConfig
from onn_component import JTC


def compare_outputs():
    """Compare outputs between new pipeline and old backward-compatible methods."""

    print("Loading config...")
    config = AppConfig.from_yaml("configs/config_ideal.yaml")

    print("Creating JTC instance...")
    jtc = JTC(config)
    jtc.eval()

    # Create reproducible test inputs
    torch.manual_seed(12345)
    batch, height, cout = 2, 4, 3
    signal = torch.rand(batch, height, 1, 8)
    kernel = torch.rand(cout, 8)

    print(f"\nTest inputs:")
    print(f"  Signal: {signal.shape}")
    print(f"  Kernel: {kernel.shape}")

    with torch.no_grad():
        print("\nRunning NEW refactored forward method...")
        output_new = jtc(signal, kernel)

        print("Running OLD method via backward-compatible wrappers...")
        # Replicate old pipeline using wrappers
        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size_for_jtc = (
            signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        )
        signal_reshaped = signal_full.reshape(batch_size_for_jtc, 8)
        kernel_reshaped = kernel_full.reshape(batch_size_for_jtc, 8)

        input_plane = jtc.generate_input_plane(signal_reshaped, kernel_reshaped)
        jft = jtc.post_fft(input_plane)
        jps = jtc.post_output_distortion(jft)
        inverse_output = jtc.inverse_output(jps)
        output_old = inverse_output.reshape(
            signal_full.shape[0],
            signal_full.shape[1],
            signal_full.shape[2],
            8,
        )

    # Compute differences
    diff = output_new - output_old
    max_abs_diff = torch.max(torch.abs(diff)).item()
    mean_abs_diff = torch.mean(torch.abs(diff)).item()

    print(f"\nResults:")
    print(f"  Output shape: {output_new.shape}")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")

    # Sample values
    print(f"\nSample output values (first 4 elements):")
    print(f"  New: {output_new[0, 0, 0, :4].tolist()}")
    print(f"  Old: {output_old[0, 0, 0, :4].tolist()}")

    # Pass/fail
    tolerance = 1e-5
    if max_abs_diff < tolerance:
        print(f"\n✓ PASS: Outputs match within tolerance ({tolerance:.0e})")
        print("The refactored JTC implementation is working correctly!")
        return 0
    else:
        print(f"\n✗ FAIL: Outputs differ by more than tolerance ({tolerance:.0e})")
        print("There may be an issue with the refactoring.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(compare_outputs())
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
