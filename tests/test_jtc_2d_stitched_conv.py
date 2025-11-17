"""Test 2D stitched convolution: JTC vs PyTorch end-to-end.

This test verifies that assembling a full 2D convolution from JTC passes
matches PyTorch conv2d when using only clean outputs for stitching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest
from onn_config import AppConfig
from onn_layers import FTconvlayer
from jtc_cycle_planner import compute_contamination_profile


def pytorch_conv2d_reference(image: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    """Compute reference 2D convolution using PyTorch (valid padding).

    Args:
        image: [H, W] input image
        kernel_2d: [Kh, Kw] 2D kernel

    Returns:
        output: [out_h, out_w] valid convolution
    """
    # Reshape for conv2d: [1, 1, H, W] and [1, 1, Kh, Kw]
    image_4d = image.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)

    # Valid convolution
    output = F.conv2d(image_4d, kernel_4d, padding=0)

    return output.squeeze()


def jtc_2d_stitched_conv(image: torch.Tensor, kernel_2d: torch.Tensor, config: AppConfig) -> torch.Tensor:
    """Perform 2D convolution using JTC with row-wise stitching.

    Algorithm:
    1. For each kernel row k_row in [0, Kh):
       2. For each output row out_row in [0, out_h):
          3. Extract image row at position k_row + out_row
          4. Convolve row with kernel_2d[k_row, :] using JTC passes with stride
          5. Accumulate results into output[out_row, :]

    Args:
        image: [H, W] input image (positive values)
        kernel_2d: [Kh, Kw] 2D kernel (positive values)
        config: JTC configuration

    Returns:
        output: [out_h, out_w] convolution result
    """
    H, W = image.shape
    Kh, Kw = kernel_2d.shape

    # Valid convolution output dimensions
    out_h = H - Kh + 1
    out_w = W - Kw + 1

    M = config.input_length
    N = config.kernel_length

    # Get contamination profile to determine clean outputs
    total_outputs, clean_outputs, effective_stride = compute_contamination_profile(
        M, N, config.jtc_total_field, config.jtc_separation
    )

    # Initialize output
    output = torch.zeros(out_h, out_w, device=image.device)

    # Create FTconvlayer for clean JTC physics
    layer = FTconvlayer(
        in_channels=1,
        out_channels=1,
        config=config,
        kernel_size=M,  # Patch size
        batch_size=1,
    )

    # Process each kernel row
    for k_row in range(Kh):
        # Extract 1D kernel for this row
        kernel_1d = kernel_2d[k_row, :]  # [Kw]

        # Set layer weights [out_ch=1, in_ch=1, 1, kernel_size=Kw]
        with torch.no_grad():
            layer.weights.data = kernel_1d.view(1, 1, 1, Kw)

        # Weight for fourier_conv_forward: [Cout=1, W=Kw]
        weight_2d = kernel_1d.unsqueeze(0)

        # Process each output row
        for out_row in range(out_h):
            img_row_idx = k_row + out_row
            row_data = image[img_row_idx, :]  # [W]

            # Stitch together outputs across the row using effective_stride
            out_col = 0
            pass_idx = 0

            while out_col < out_w:
                # Determine patch position
                patch_start = pass_idx * effective_stride
                patch_end = min(patch_start + M, W)

                # Extract patch [M] (may be shorter at boundary)
                patch = row_data[patch_start:patch_end]

                # Pad with zeros if patch is shorter than M
                if len(patch) < M:
                    patch_padded = torch.zeros(M, device=patch.device)
                    patch_padded[:len(patch)] = patch
                    patch = patch_padded

                # Prepare for JTC: [B=1, H=1, in_ch=1, W=M]
                patch_4d = patch.view(1, 1, 1, M)

                # Run JTC forward pass (clean physics)
                jtc_output = layer.fourier_conv_forward(patch_4d, weight_2d)
                jtc_output_1d = jtc_output.squeeze()  # [total_outputs]

                # Determine how many outputs to use from this pass
                num_to_use = min(effective_stride, out_w - out_col)

                # Extract outputs for valid convolution
                # JTC gives M+N-1 correlation outputs, we use [N-1, M-1] for valid conv
                # The i-th output we extract goes to position out_col + i
                # It corresponds to JTC output at index i + (Kw-1)
                for i in range(num_to_use):
                    output_col = out_col + i

                    if output_col < out_w:
                        # JTC output index for valid conv starting position
                        jtc_idx = i + (Kw - 1)

                        if jtc_idx < len(jtc_output_1d):
                            output[out_row, output_col] += jtc_output_1d[jtc_idx]

                # Move to next pass
                out_col += num_to_use
                pass_idx += 1

                # Break if we've covered the full width
                if out_col >= out_w:
                    break

    return output


class TestJTC2DStitchedConv:
    """Test 2D stitched convolution matches PyTorch."""

    def test_2d_small_image_clean_config(self):
        """Test 2D convolution on small image with clean configuration."""
        # Use a clean config (zero contamination)
        M, N = 8, 3
        plane_size = 32
        sep = 7

        # Get configuration profile
        total, clean, stride = compute_contamination_profile(M, N, plane_size, sep)
        print(f"\nConfig: M={M}, N={N}, plane={plane_size}, sep={sep}")
        print(f"  Total outputs: {total}, Clean valid: {clean}, Effective stride: {stride}")
        assert stride > 0, "Config should have positive stride"

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

        # Create small test image and kernel (positive values)
        torch.manual_seed(42)
        H, W = 10, 10
        Kh, Kw = 3, 3

        image = torch.rand(H, W) * 0.5 + 0.5  # [0.5, 1.0]
        kernel_2d = torch.rand(Kh, Kw) * 0.5 + 0.5

        print(f"  Image: {H}x{W}, Kernel: {Kh}x{Kw}")

        # PyTorch reference
        pytorch_output = pytorch_conv2d_reference(image, kernel_2d)
        print(f"  PyTorch output shape: {pytorch_output.shape}")

        # JTC stitched
        jtc_output = jtc_2d_stitched_conv(image, kernel_2d, config)
        print(f"  JTC output shape: {jtc_output.shape}")

        # Compare
        assert pytorch_output.shape == jtc_output.shape, "Output shapes must match"

        diff = torch.abs(pytorch_output - jtc_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_error = max_diff / pytorch_output.max().item()

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Relative error: {rel_error:.4f}")

        # For clean config, should match closely
        assert rel_error < 0.05, f"JTC should match PyTorch within 5% for clean config, got {rel_error:.4f}"

        # Print some sample values
        print(f"\n  Sample outputs (first 3x3):")
        print(f"  PyTorch:\n{pytorch_output[:3, :3]}")
        print(f"  JTC:\n{jtc_output[:3, :3]}")

    @pytest.mark.parametrize("H,W,Kh,Kw", [
        (12, 12, 3, 3),
        (16, 16, 3, 3),
    ])
    def test_2d_varying_sizes(self, H, W, Kh, Kw):
        """Test 2D convolution with varying image sizes."""
        # Clean config
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

        # Random inputs
        torch.manual_seed(42)
        image = torch.rand(H, W) * 0.5 + 0.5
        kernel_2d = torch.rand(Kh, Kw) * 0.5 + 0.5

        # Compare
        pytorch_output = pytorch_conv2d_reference(image, kernel_2d)
        jtc_output = jtc_2d_stitched_conv(image, kernel_2d, config)

        print(f"\nImage {H}x{W}, Kernel {Kh}x{Kw}:")
        print(f"  Output shape: {jtc_output.shape}")

        rel_error = (torch.abs(pytorch_output - jtc_output).max() / pytorch_output.max()).item()
        print(f"  Relative error: {rel_error:.4f}")

        assert rel_error < 0.05, f"Should match within 5%, got {rel_error:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
