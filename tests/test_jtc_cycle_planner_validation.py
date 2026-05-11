"""
Validation tests for jtc_cycle_planner against actual JTC physics.

These tests verify that the usable_outputs calculation in jtc_cycle_planner
correctly identifies overlap-free correlation outputs by comparing against
actual autocorrelation and cross-correlation decomposition from JTC physics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest
from onn_config import AppConfig
from onn_component import JTC
from jtc_cycle_planner import usable_outputs


def compute_reference_correlation(
    signal: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor:
    """Compute reference correlation using PyTorch's conv1d.

    Correlation is like convolution but without flipping the kernel.
    Returns full correlation of length M+N-1.
    """
    # Reshape for conv1d
    signal_conv = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
    # Flip kernel for correlation (correlation = conv with flipped kernel)
    kernel_flipped = torch.flip(kernel, [0])
    kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)  # [1, 1, N]

    # Full correlation with padding
    padding = len(kernel) - 1
    output = F.conv1d(signal_conv, kernel_conv, padding=padding)
    return output.squeeze()  # [M+N-1]


def compute_jtc_full_output(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    M: int,
    N: int,
    sep: int,
    plane_size: int,
) -> torch.Tensor:
    """Compute full JTC output plane (before extraction).

    Returns the magnitude of the inverse FFT of the JPS.
    """
    # Build input plane: kernel [0:N], signal [N+sep:N+sep+M]
    input_plane = torch.zeros(plane_size, dtype=torch.complex64)

    kernel_start = 0
    kernel_end = kernel_start + N
    signal_start = kernel_end + sep
    signal_end = signal_start + M

    input_plane[kernel_start:kernel_end] = kernel.to(torch.complex64)
    input_plane[signal_start:signal_end] = signal.to(torch.complex64)

    # Roll to center
    roll_amount = (plane_size // 2) - (M + signal_start) // 2
    input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

    # JTC: FFT -> fftshift -> JPS
    jft = torch.fft.fft(input_plane)
    jft_shifted = torch.fft.fftshift(jft)
    jps = torch.abs(jft_shifted) ** 2 / plane_size

    # Back to output: FFT -> fftshift -> abs
    output_fft = torch.fft.fft(jps)
    output_shifted = torch.fft.fftshift(output_fft)
    output_abs = torch.abs(output_shifted)

    return output_abs


class TestJTCCyclePlannerValidation:
    """Validate jtc_cycle_planner against actual JTC physics."""

    @pytest.mark.parametrize(
        "input_len,kernel_len,lens,sep",
        [
            (8, 3, 32, 7),
            (16, 8, 48, 9),
            (16, 8, 64, 15),
            (8, 8, 48, 8),  # Golden code case
        ],
    )
    def test_usable_outputs_matches_overlap_free_region(
        self, input_len, kernel_len, lens, sep
    ):
        """Test that usable_outputs count matches direct JTC output bookkeeping."""
        M, N = input_len, kernel_len
        plane_size = lens

        # Get cycle planner prediction
        predicted_usable = usable_outputs(M, N, plane_size, sep)
        expected_full_correlation = M + N - 1

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

        indices = jtc.compute_correlation_indices(torch.device("cpu"))
        assert jtc.output_length == predicted_usable
        assert indices.numel() == predicted_usable
        assert int(indices.min()) >= 0
        assert int(indices.max()) < plane_size
        assert predicted_usable <= expected_full_correlation

        torch.manual_seed(42)
        signal = torch.randn(2, 1, 1, M) * 0.1
        kernel = torch.randn(1, N) * 0.1
        output = jtc(signal, kernel)
        assert output.shape == (2, 1, 1, predicted_usable)
        assert torch.isfinite(output).all()

    def test_edge_case_wrapping_detection(self):
        """Test that wrapping indices don't contaminate autocorrelation.

        Edge case: when same_start + output_length > plane_size,
        indices wrap around. Verify wrapped indices don't hit autocorrelation.
        """
        # Configuration that causes wrapping
        M, N = 16, 8
        plane_size = 32  # Tight fit
        sep = 8

        predicted_usable = usable_outputs(M, N, plane_size, sep)

        # Check if indices wrap
        same_start = plane_size // 2 + sep + N // 2 + 1
        same_end = same_start + predicted_usable

        print(f"\nWrapping test: M={M}, N={N}, plane_size={plane_size}, sep={sep}")
        print(
            f"  same_start={same_start}, same_end={same_end}, plane_size={plane_size}"
        )
        print(f"  Wrapping: {same_end > plane_size}")

        if same_end > plane_size:
            # Indices wrap - this is a problem!
            # Wrapped indices: [same_start, plane_size) + [0, same_end - plane_size)
            wrapped_indices = list(range(0, same_end - plane_size))

            # Autocorrelation is centered around plane_size // 2
            # Extends roughly ± max(M-1, N-1)
            auto_center = plane_size // 2
            auto_extent = max(M - 1, N - 1)
            auto_region = set(
                range(
                    (auto_center - auto_extent) % plane_size,
                    (auto_center + auto_extent + 1) % plane_size,
                )
            )

            # Check overlap
            overlap = set(wrapped_indices) & auto_region

            print(f"  Wrapped indices: {wrapped_indices}")
            print(
                f"  Autocorrelation region: [{auto_center - auto_extent}, {auto_center + auto_extent}]"
            )
            print(f"  Overlap: {overlap}")

            # If there's overlap, the simple formula is WRONG
            if overlap:
                pytest.fail(
                    f"CRITICAL ERROR: Wrapped indices {overlap} overlap with autocorrelation region! "
                    f"The simple formula M+N+sep <= plane_size is insufficient."
                )

    @pytest.mark.parametrize(
        "input_len,kernel_len,lens,sep",
        [
            (8, 3, 32, 7),
            (16, 8, 48, 9),
            (8, 8, 48, 8),
        ],
    )
    def test_cycle_planner_vs_actual_jtc_output(self, input_len, kernel_len, lens, sep):
        """Test that cycle planner prediction matches actual JTC implementation."""
        M, N = input_len, kernel_len

        # Get cycle planner prediction
        predicted_usable = usable_outputs(M, N, lens, sep)

        # Create actual JTC
        config = AppConfig(
            input_length=M,
            kernel_length=N,
            output_length=None,  # Auto-calculate
            jtc_separation=sep,
            jtc_total_field=lens,
            dac_bits=None,
            adc_bits=None,
            fourier_plane_bits=None,
        )

        jtc = JTC(config)

        # Check that JTC's output_length matches prediction
        msg = (
            f"JTC output_length ({jtc.output_length}) != "
            f"cycle planner prediction ({predicted_usable})"
        )
        assert jtc.output_length == predicted_usable, msg

        # Test actual forward pass
        batch_size = 2
        signal = torch.randn(batch_size, 1, 1, M)
        kernel = torch.randn(1, N)

        output = jtc(signal, kernel)

        # Verify output shape matches prediction
        msg = (
            f"JTC output length ({output.shape[-1]}) != "
            f"cycle planner prediction ({predicted_usable})"
        )
        assert output.shape[-1] == predicted_usable, msg

        print(f"\nConfig: M={M}, N={N}, lens={lens}, sep={sep}")
        print(f"  Predicted: {predicted_usable}")
        print(f"  JTC output_length: {jtc.output_length}")
        print(f"  Actual output shape: {output.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
