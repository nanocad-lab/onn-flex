"""Compare stitching accuracy with different stride values."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from onn_config import AppConfig
from onn_layers import FTconvlayer


def test_stitching_with_stride(stride: int, signal_length: int = 16):
    """Test 1D stitching with specified stride."""
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
    signal = torch.rand(signal_length) * 0.5 + 0.5
    kernel = torch.rand(N) * 0.5 + 0.5

    # PyTorch reference
    pytorch_output = F.conv1d(
        signal.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=0
    ).squeeze()

    # Create FTconvlayer
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

    # Stitch using specified stride
    stitched = []
    pos = 0
    pass_count = 0

    while pos < signal_length:
        # Extract patch
        patch_end = min(pos + M, signal_length)
        patch = signal[pos:patch_end]

        # Pad if needed
        if len(patch) < M:
            patch_padded = torch.zeros(M)
            patch_padded[:len(patch)] = patch
            patch = patch_padded

        # Run JTC
        patch_4d = patch.view(1, 1, 1, M)
        jtc_output = layer.fourier_conv_forward(patch_4d, weight_2d).squeeze()

        # Extract valid outputs [N-1, M-1]
        valid_outputs = jtc_output[N-1:M]

        # Use stride outputs
        num_to_use = min(stride, len(valid_outputs), signal_length - pos - (N-1))
        stitched.extend(valid_outputs[:num_to_use].tolist())

        pos += stride
        pass_count += 1

    stitched = torch.tensor(stitched[:len(pytorch_output)])

    # Compare
    diff = torch.abs(stitched - pytorch_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Find positions with largest errors
    top_errors = torch.topk(diff, min(5, len(diff)))

    return {
        'stride': stride,
        'passes': pass_count,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'stitched': stitched,
        'pytorch': pytorch_output,
        'diff': diff,
        'top_error_indices': top_errors.indices.tolist(),
        'top_error_values': top_errors.values.tolist(),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STRIDE COMPARISON: M=8, N=3, Signal Length=16")
    print("=" * 70)

    for stride in [5, 6]:
        result = test_stitching_with_stride(stride)

        print(f"\n{'='*70}")
        print(f"Stride = {stride}")
        print(f"{'='*70}")
        print(f"Number of passes: {result['passes']}")
        print(f"Max difference:   {result['max_diff']:.10f}")
        print(f"Mean difference:  {result['mean_diff']:.10f}")

        if result['max_diff'] > 1e-6:
            print(f"\nTop {len(result['top_error_indices'])} error positions:")
            for idx, (pos, val) in enumerate(zip(result['top_error_indices'], result['top_error_values'])):
                print(f"  {idx+1}. Index {pos:2d}: error = {val:.10f}")
                print(f"     JTC:     {result['stitched'][pos]:.6f}")
                print(f"     PyTorch: {result['pytorch'][pos]:.6f}")

        if stride == 5:
            result_5 = result
        else:
            result_6 = result

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Stride 5: max_diff = {result_5['max_diff']:.10f}, {result_5['passes']} passes")
    print(f"Stride 6: max_diff = {result_6['max_diff']:.10f}, {result_6['passes']} passes")
    print(f"\nUsing stride 6 vs 5:")
    print(f"  Error increase: {result_6['max_diff'] - result_5['max_diff']:.10f}")
    print(f"  Passes saved:   {result_5['passes'] - result_6['passes']}")

    if result_6['max_diff'] < 1e-6:
        print(f"\n✓ Stride 6 achieves perfect accuracy (< 1e-6)")
    elif result_6['max_diff'] < 1e-3:
        print(f"\n✓ Stride 6 achieves good accuracy (< 0.1%)")
    else:
        print(f"\n✗ Stride 6 has significant errors (> 0.1%)")
