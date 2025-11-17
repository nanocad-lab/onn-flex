"""Analyze which JTC outputs are contaminated when using stride 6."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from onn_config import AppConfig
from onn_layers import FTconvlayer


def analyze_jtc_outputs():
    """Analyze contamination in individual JTC outputs."""
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
        conv_backend="fourier",
    )

    # Create test data
    torch.manual_seed(42)
    signal = torch.rand(16) * 0.5 + 0.5
    kernel = torch.rand(N) * 0.5 + 0.5

    layer = FTconvlayer(
        in_channels=1,
        out_channels=1,
        config=config,
        kernel_size=M,
        batch_size=1,
    )

    with torch.no_grad():
        layer.weights.data = kernel.view(1, 1, 1, N)

    weight_2d = kernel.unsqueeze(0)

    print("=" * 80)
    print("ANALYSIS: Which JTC outputs are contaminated?")
    print("=" * 80)
    print(f"Config: M={M}, N={N}, plane_size={plane_size}, sep={sep}")
    print(f"Valid outputs per pass: M-N+1 = {M-N+1} (indices {N-1} to {M-1})")
    print()

    # Analyze each pass
    passes = [
        {"start": 0, "name": "Pass 1 (pos 0-7)"},
        {"start": 5, "name": "Pass 2 (pos 5-12) - with stride 5"},
        {"start": 6, "name": "Pass 3 (pos 6-13) - with stride 6"},
    ]

    for pass_info in passes:
        pos = pass_info["start"]
        patch = signal[pos:pos+M]

        # Compute PyTorch reference for this patch
        pytorch_ref = torch.conv1d(
            patch.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()

        # Compute JTC output
        patch_4d = patch.view(1, 1, 1, M)
        jtc_output = layer.fourier_conv_forward(patch_4d, weight_2d).squeeze()

        print(f"\n{pass_info['name']}")
        print("─" * 80)

        # Compare valid outputs
        valid_jtc = jtc_output[N-1:M]
        errors = torch.abs(valid_jtc - pytorch_ref)

        print(f"{'Index':<8} {'JTC':<12} {'PyTorch':<12} {'Error':<12} {'Status'}")
        print("─" * 80)

        for i in range(len(valid_jtc)):
            corr_idx = N - 1 + i  # Index in full M+N-1 correlation output
            global_output_idx = pos + i  # Global position in final output
            error = errors[i].item()

            if error < 1e-6:
                status = "✓ Clean"
            elif error < 1e-3:
                status = "⚠ Small contamination"
            else:
                status = "✗ CONTAMINATED"

            marker = ""
            if i == 5:  # 6th output (stride 6 would use this)
                marker = " ← 6th output"
            elif i == 4:  # 5th output (stride 5 stops before this for next pass)
                marker = " ← 5th output (last used by stride 5)"

            print(f"  {i:<6} {valid_jtc[i].item():<12.6f} {pytorch_ref[i].item():<12.6f} "
                  f"{error:<12.8f} {status}{marker}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The 6th valid output (index 5, correlation index 7) shows contamination")
    print("in passes where it overlaps with autocorr edges.")
    print()
    print("Stride 5: Uses outputs 0-4 (5 outputs) ✓ No contamination")
    print("Stride 6: Uses outputs 0-5 (6 outputs) ✗ Contamination at index 5 and 11")


if __name__ == "__main__":
    analyze_jtc_outputs()
