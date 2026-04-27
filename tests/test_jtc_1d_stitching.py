"""Test 1D stitched convolution to verify basic stitching logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from onn_config import AppConfig
from onn_layers import FTconvlayer
from jtc_cycle_planner import compute_contamination_profile


def test_1d_single_pass_matches_pytorch():
    """Verify single JTC pass matches PyTorch for 1D."""
    M, N = 8, 3
    plane_size = 32
    sep = 7

    config = AppConfig(
        input_length=M,
        kernel_length=N,
        output_length=None,
        jtc_separation=sep,
        jtc_total_field=plane_size,
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
    )

    layer = FTconvlayer(
        in_channels=1,
        out_channels=1,
        config=config,
        kernel_size=M,
        batch_size=1,
    )

    # Positive inputs
    torch.manual_seed(42)
    signal = torch.rand(M) * 0.5 + 0.5
    kernel = torch.rand(N) * 0.5 + 0.5

    # Set weights
    with torch.no_grad():
        layer.weights.data = kernel.view(1, 1, 1, N)

    # JTC output
    signal_4d = signal.view(1, 1, 1, M)
    weight_2d = kernel.unsqueeze(0)
    jtc_output = layer.fourier_conv_forward(signal_4d, weight_2d).squeeze()

    # PyTorch correlation (full padding gives M+N-1 outputs)
    signal_pt = signal.unsqueeze(0).unsqueeze(0)
    kernel_flipped = torch.flip(kernel, [0])
    kernel_pt = kernel_flipped.unsqueeze(0).unsqueeze(0)
    pytorch_corr = F.conv1d(signal_pt, kernel_pt, padding=N - 1).squeeze()

    print("\n1D Single Pass Test:")
    print(f"  JTC output length: {len(jtc_output)}")
    print(f"  PyTorch corr length: {len(pytorch_corr)}")
    print(f"  JTC:     {jtc_output}")
    print(f"  PyTorch: {pytorch_corr}")

    # First 7 should match (clean outputs)
    diff = torch.abs(jtc_output[:7] - pytorch_corr[:7])
    print(f"  Diff (first 7): {diff}")
    print(f"  Max diff (first 7): {diff.max():.6f}")

    assert diff.max() < 0.01, "Clean outputs should match PyTorch"


def test_1d_stitching_two_passes():
    """Test stitching two JTC passes together for 1D convolution."""
    # Config: M=8, stride=10, so for 16-pixel signal we need 2 passes
    M, N = 8, 3
    plane_size = 32
    sep = 7

    total, clean, stride = compute_contamination_profile(M, N, plane_size, sep)
    print(f"\nConfig: M={M}, N={N}, stride={stride}")

    config = AppConfig(
        input_length=M,
        kernel_length=N,
        output_length=None,
        jtc_separation=sep,
        jtc_total_field=plane_size,
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
    )

    layer = FTconvlayer(
        in_channels=1,
        out_channels=1,
        config=config,
        kernel_size=M,
        batch_size=1,
    )

    # Create longer signal
    torch.manual_seed(42)
    signal_length = 16
    signal = torch.rand(signal_length) * 0.5 + 0.5
    kernel = torch.rand(N) * 0.5 + 0.5

    with torch.no_grad():
        layer.weights.data = kernel.view(1, 1, 1, N)

    weight_2d = kernel.unsqueeze(0)

    # PyTorch reference (valid convolution)
    signal_pt = signal.unsqueeze(0).unsqueeze(0)
    kernel_pt = kernel.unsqueeze(0).unsqueeze(0)
    pytorch_valid = F.conv1d(signal_pt, kernel_pt, padding=0).squeeze()

    print(f"  Signal length: {signal_length}")
    print(
        f"  PyTorch valid output length: {len(pytorch_valid)}"
    )  # Should be 16 - 3 + 1 = 14

    # Stitch JTC outputs
    # Pass 1: signal[0:8], gives outputs for positions [0, 0+10)
    # Pass 2: signal[10:18] (but only goes to 16, so signal[8:16]), gives outputs for positions [10, 10+10)

    stitched_output = torch.zeros(len(pytorch_valid))

    pass_idx = 0
    out_col = 0

    while out_col < len(pytorch_valid):
        # Determine patch position
        patch_start = pass_idx * stride
        patch_end = min(patch_start + M, signal_length)

        # Extract patch
        patch = signal[patch_start:patch_end]

        # If patch is shorter than M, pad with zeros
        if len(patch) < M:
            patch_padded = torch.zeros(M)
            patch_padded[: len(patch)] = patch
            patch = patch_padded

        # Run JTC
        patch_4d = patch.view(1, 1, 1, M)
        jtc_output = layer.fourier_conv_forward(patch_4d, weight_2d).squeeze()

        print(f"\n  Pass {pass_idx}: patch[{patch_start}:{patch_end}]")
        print(f"    JTC output: {jtc_output}")

        # For valid convolution, output i corresponds to input position i + (N-1)
        # So JTC output[0] corresponds to patch[N-1], output[1] to patch[N], etc.
        # In global coordinates, output[i] corresponds to global position patch_start + i + (N-1)

        # Extract usable outputs
        num_to_use = min(stride, len(pytorch_valid) - out_col)

        for i in range(num_to_use):
            # Global input position for this output
            global_pos = patch_start + i + (N - 1)

            # Output column for valid conv is global_pos - (N - 1)
            output_col = global_pos - (N - 1)

            if 0 <= output_col < len(pytorch_valid):
                # JTC output index: i + (N-1) for valid conv starting position
                jtc_idx = i + (N - 1)
                if jtc_idx < len(jtc_output):
                    stitched_output[output_col] = jtc_output[jtc_idx]
                    print(
                        f"      output[{output_col}] = jtc_output[{jtc_idx}] = {jtc_output[jtc_idx]:.4f}"
                    )

        out_col += num_to_use
        pass_idx += 1

        if out_col >= len(pytorch_valid):
            break

    print(f"\n  Stitched output: {stitched_output}")
    print(f"  PyTorch valid:   {pytorch_valid}")

    diff = torch.abs(stitched_output - pytorch_valid)
    print(f"  Difference: {diff}")
    print(f"  Max diff: {diff.max():.6f}")

    # Should match for clean outputs
    max_diff = diff.max()
    msg = f"Stitched output should match PyTorch, got max diff {max_diff:.6f}"
    assert max_diff < 0.05, msg


if __name__ == "__main__":
    test_1d_single_pass_matches_pytorch()
    test_1d_stitching_two_passes()
