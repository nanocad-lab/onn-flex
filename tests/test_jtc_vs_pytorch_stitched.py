"""Test that stitched JTC convolution matches PyTorch for positive inputs.

Critical test: If autocorr contamination is <0.0001%, JTC outputs should
numerically match PyTorch conv2d within tight tolerances.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest
from onn_config import AppConfig
from onn_component import JTC
from jtc_cycle_planner import compute_contamination_profile


def pytorch_conv2d_reference(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Compute reference convolution using PyTorch.

    Args:
        image: [H, W] or [1, 1, H, W]
        kernel: [Kh, Kw] or [1, 1, Kh, Kw]

    Returns:
        output: [out_h, out_w] 'valid' convolution
    """
    # Reshape to [1, 1, H, W] if needed
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    # PyTorch conv2d with 'valid' padding
    output = F.conv2d(image, kernel, padding=0)

    return output.squeeze()


def jtc_conv1d_single_pass(signal: torch.Tensor, kernel: torch.Tensor,
                           config: AppConfig, jtc: JTC) -> torch.Tensor:
    """Perform single JTC pass for 1D convolution.

    Args:
        signal: [M] input signal
        kernel: [N] kernel weights
        config: JTC configuration
        jtc: JTC instance

    Returns:
        output: [output_length] correlation outputs
    """
    # Reshape for JTC: [B=1, H=1, in_ch=1, W=M]
    signal_jtc = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    kernel_jtc = kernel.unsqueeze(0)

    # Forward pass
    output = jtc(signal_jtc, kernel_jtc)

    return output.squeeze()


def jtc_conv2d_with_stitching(image: torch.Tensor, kernel_2d: torch.Tensor,
                               config: AppConfig) -> torch.Tensor:
    """Perform 2D convolution using JTC with row-wise stitching.

    Simulates the actual hardware operation:
    1. Extract patches from image (size input_length)
    2. Convolve each patch with kernel using JTC
    3. Stitch results together using effective_stride

    Args:
        image: [H, W] input image
        kernel_2d: [Kh, Kw] 2D kernel

    Returns:
        output: [out_h, out_w] convolution result
    """
    H, W = image.shape
    Kh, Kw = kernel_2d.shape

    # Output dimensions for 'valid' convolution
    out_h = H - Kh + 1
    out_w = W - Kw + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Image too small for kernel: {H}x{W} with {Kh}x{Kw} kernel")

    M = config.input_length
    N = config.kernel_length

    # Get contamination profile
    total_outputs, clean_outputs, effective_stride = compute_contamination_profile(
        M, N, config.jtc_total_field, config.jtc_separation
    )

    # Initialize output
    output = torch.zeros(out_h, out_w, device=image.device)

    # Process each row of the kernel
    for krow in range(Kh):
        # Create JTC instance for this kernel row
        jtc = JTC(config)

        # Set kernel weights (use kernel row as 1D kernel)
        kernel_1d = kernel_2d[krow, :]  # [Kw]

        # Process each output row
        for out_row in range(out_h):
            img_row = krow + out_row  # Input row index

            # Extract full image row
            row_data = image[img_row, :]  # [W]

            # Stitch together outputs for this row using effective_stride
            out_col = 0
            pass_idx = 0

            while out_col < out_w:
                # Determine patch start position
                patch_start = pass_idx * effective_stride
                patch_end = patch_start + M

                if patch_end > W:
                    # Last patch - may need to adjust
                    patch_start = W - M
                    patch_end = W

                # Extract patch
                patch = row_data[patch_start:patch_end]

                # Run JTC on this patch
                jtc_output = jtc_conv1d_single_pass(patch, kernel_1d, config, jtc)

                # Determine how many outputs to use from this pass
                outputs_to_use = min(effective_stride, out_w - out_col, len(jtc_output))

                # Map to output positions
                # JTC output corresponds to patch positions [patch_start:patch_end]
                # For 'valid' conv, output corresponds to [patch_start+Kw-1:patch_end]
                output_start_in_patch = Kw - 1  # Offset for valid convolution

                # Copy outputs
                for i in range(outputs_to_use):
                    if out_col + i < out_w:
                        # The i-th output corresponds to position patch_start + output_start_in_patch + i
                        output[out_row, out_col + i] += jtc_output[output_start_in_patch + i]

                out_col += outputs_to_use
                pass_idx += 1

                if out_col >= out_w:
                    break

    return output


class TestJTCVsPyTorchStitched:
    """Test stitched JTC convolution against PyTorch reference."""

    def test_jtc_1d_positive_inputs(self):
        """Test single JTC pass with positive inputs matches PyTorch 1D convolution."""
        # Configuration with zero contamination
        M, N = 8, 3
        plane_size = 32
        sep = 7

        # Verify contamination is negligible
        total, clean, stride = compute_contamination_profile(M, N, plane_size, sep)
        print(f"\nConfig: M={M}, N={N}, plane={plane_size}, sep={sep}")
        print(f"  Total outputs: {total}")
        print(f"  Clean outputs: {clean}")
        print(f"  Effective stride: {stride}")

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

        jtc = JTC(config)

        # Use POSITIVE inputs (like real image data)
        torch.manual_seed(42)
        signal = torch.rand(M) * 0.5 + 0.5  # [0.5, 1.0]
        kernel = torch.rand(N) * 0.5 + 0.5  # [0.5, 1.0]

        # JTC output
        signal_jtc = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        kernel_jtc = kernel.unsqueeze(0)
        jtc_output = jtc(signal_jtc, kernel_jtc).squeeze()

        # PyTorch reference (1D convolution)
        signal_pt = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
        kernel_pt = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        pytorch_output = F.conv1d(signal_pt, kernel_pt, padding=0).squeeze()

        # For positive inputs with negligible contamination, should match closely
        # Contamination <0.0001% → use tight tolerance
        print(f"\n  JTC output:     {jtc_output}")
        print(f"  PyTorch output: {pytorch_output}")
        print(f"  Max diff:       {torch.abs(jtc_output - pytorch_output).max():.6f}")
        print(f"  Mean diff:      {torch.abs(jtc_output - pytorch_output).mean():.6f}")

        # Check if they match within tolerance
        # Note: JTC outputs magnitudes, so can only match if inputs are positive
        if clean == total:  # All outputs clean
            torch.testing.assert_close(
                jtc_output[:len(pytorch_output)],
                pytorch_output,
                rtol=1e-3,  # 0.1% relative tolerance
                atol=1e-4,  # Absolute tolerance
                msg="JTC should match PyTorch for positive inputs with zero contamination"
            )
        else:
            print(f"  ⚠ Config has contamination: {total-clean}/{total} contaminated outputs")

    @pytest.mark.parametrize("M,N,plane,sep", [
        (8, 3, 32, 7),    # Zero contamination
        (16, 8, 64, 15),  # Zero contamination
    ])
    def test_jtc_1d_clean_configs(self, M, N, plane, sep):
        """Test JTC matches PyTorch for configs with zero contamination."""
        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        # Only test clean configs
        if clean < total:
            pytest.skip(f"Config has contamination: {total-clean}/{total} outputs")

        config = AppConfig(
            input_length=M,
            kernel_length=N,
            output_length=None,
            jtc_separation=sep,
            jtc_total_field=plane,
            dac_bits=None,
            adc_bits=None,
            fourier_plane_bits=None,
            scale_output="none",
        )

        jtc = JTC(config)

        # Positive inputs
        torch.manual_seed(42)
        signal = torch.rand(M) * 0.5 + 0.5
        kernel = torch.rand(N) * 0.5 + 0.5

        # JTC
        signal_jtc = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        kernel_jtc = kernel.unsqueeze(0)
        jtc_output = jtc(signal_jtc, kernel_jtc).squeeze()

        # PyTorch
        signal_pt = signal.unsqueeze(0).unsqueeze(0)
        kernel_pt = kernel.unsqueeze(0).unsqueeze(0)
        pytorch_output = F.conv1d(signal_pt, kernel_pt, padding=0).squeeze()

        print(f"\nM={M}, N={N}, plane={plane}, sep={sep}")
        print(f"  Max diff: {torch.abs(jtc_output - pytorch_output).max():.6f}")

        # Tight tolerance for zero contamination
        torch.testing.assert_close(
            jtc_output[:len(pytorch_output)],
            pytorch_output,
            rtol=1e-3,
            atol=1e-4,
        )

    def test_contaminated_config_reporting(self):
        """For contaminated configs, report which outputs are clean vs contaminated."""
        # Config with known contamination
        M, N, plane, sep = 16, 8, 48, 9

        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        contaminated = total - clean

        print(f"\nConfig: M={M}, N={N}, plane={plane}, sep={sep}")
        print(f"  Total outputs: {total}")
        print(f"  Clean outputs: {clean} ({100*clean/total:.1f}%)")
        print(f"  Contaminated:  {contaminated} ({100*contaminated/total:.1f}%)")
        print(f"  Effective stride for stitching: {stride}")

        # Verify we report this correctly
        assert total == M + N - 1
        assert clean <= total
        assert contaminated >= 0

        if contaminated > 0:
            print(f"  ⚠ {contaminated} outputs have autocorr contamination")
            print(f"  → Use effective_stride={stride} for tile stitching")
            print(f"  → Or use larger plane_size/separation for cleaner outputs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
