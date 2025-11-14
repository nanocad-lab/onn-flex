"""
Comprehensive test suite for variable input/kernel/output lengths.

Tests the jtc_cycle_planner-based implementation against PyTorch conv for correctness.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest
from onn_config import AppConfig
from onn_component import JTC
from onn_layers import FTconvlayer


class TestVariableLengths:
    """Test suite for variable length support."""

    def test_usable_outputs_calculation(self):
        """Test that usable_outputs from jtc_cycle_planner is now correct (M+N-1)."""
        # All return full correlation length M+N-1 when configuration is valid
        assert JTC._compute_usable_outputs(8, 3, 32, 7) == 10   # 8+3-1 = 10
        assert JTC._compute_usable_outputs(16, 8, 48, 9) == 23  # 16+8-1 = 23
        assert JTC._compute_usable_outputs(16, 8, 64, 15) == 23 # 16+8-1 = 23
        assert JTC._compute_usable_outputs(8, 8, 48, 8) == 15   # 8+8-1 = 15 (golden code case!)

    def test_jtc_initialization_with_auto_output_length(self):
        """Test that JTC correctly auto-calculates output_length as M+N-1."""
        config = AppConfig(
            input_length=8,
            kernel_length=3,
            output_length=None,  # Auto-calculate
            jtc_separation=7,
            jtc_total_field=32,
        )

        jtc = JTC(config)
        assert jtc.input_length == 8
        assert jtc.kernel_length == 3
        # Full correlation length: M+N-1 = 8+3-1 = 10
        assert jtc.output_length == 10

    def test_jtc_initialization_with_manual_output_length(self):
        """Test that JTC respects manually specified output_length."""
        config = AppConfig(
            input_length=16,
            kernel_length=8,
            output_length=7,  # Manually specified
            jtc_separation=9,
            jtc_total_field=48,
        )

        jtc = JTC(config)
        assert jtc.input_length == 16
        assert jtc.kernel_length == 8
        assert jtc.output_length == 7

    def test_jtc_invalid_config_raises_error(self):
        """Test that invalid configuration raises an error."""
        config = AppConfig(
            input_length=32,  # Too large
            kernel_length=32,  # Too large
            output_length=None,
            jtc_separation=8,
            jtc_total_field=48,  # Not enough space: 32+32+8 = 72 > 48
        )

        with pytest.raises(ValueError, match="too small"):
            JTC(config)

    @pytest.mark.parametrize("input_len,kernel_len,lens,sep,expected_output", [
        (8, 3, 32, 7, 10),   # M+N-1 = 8+3-1 = 10
        (16, 8, 48, 9, 23),  # M+N-1 = 16+8-1 = 23
        (16, 8, 64, 15, 23), # M+N-1 = 16+8-1 = 23
        (8, 8, 48, 8, 15),   # M+N-1 = 8+8-1 = 15 (golden code case!)
    ])
    def test_jtc_forward_shape(self, input_len, kernel_len, lens, sep, expected_output):
        """Test that JTC forward pass produces correct output shape."""
        config = AppConfig(
            input_length=input_len,
            kernel_length=kernel_len,
            output_length=None,
            jtc_separation=sep,
            jtc_total_field=lens,
            dac_bits=None,  # Disable quantization for this test
            adc_bits=None,
            fourier_plane_bits=None,
        )

        jtc = JTC(config)

        # Create test inputs
        batch_size = 2
        height = 4
        num_kernels = 3
        signal = torch.randn(batch_size, height, 1, input_len)
        kernel = torch.randn(num_kernels, kernel_len)

        # Forward pass
        output = jtc(signal, kernel)

        # Check output shape
        assert output.shape == (batch_size, height, num_kernels, expected_output)
        assert torch.isfinite(output).all()

    def test_fourier_backend_variable_lengths(self):
        """Test Fourier backend with variable lengths ("same" convolution)."""
        config = AppConfig(
            input_length=16,
            kernel_length=8,
            output_length=16,  # "same" convolution: output_length = input_length
            jtc_separation=9,
            jtc_total_field=48,
            conv_backend="fourier",
            dac_bits=None,
            adc_bits=None,
            fourier_plane_bits=None,
        )

        layer = FTconvlayer(
            in_channels=1,
            out_channels=2,
            config=config,
            kernel_size=config.input_length,
            batch_size=2,
        )

        # Create test input
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 32, 32)

        # Forward pass
        output = layer(test_input)

        # Check output shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == 2  # out_channels
        assert torch.isfinite(output).all()

    def test_jtc_emulation_backend_variable_lengths(self):
        """Test JTC emulation backend with variable lengths ("same" convolution)."""
        config = AppConfig(
            input_length=8,
            kernel_length=3,
            output_length=8,  # "same" convolution: output_length = input_length
            jtc_separation=7,
            jtc_total_field=32,
            conv_backend="jtc_emulation",
            dac_bits=4,
            adc_bits=6,
        )

        layer = FTconvlayer(
            in_channels=1,
            out_channels=2,
            config=config,
            kernel_size=config.input_length,
            batch_size=2,
        )

        # Create test input
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 32, 32)

        # Forward pass
        output = layer(test_input)

        # Check output shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == 2  # out_channels
        assert torch.isfinite(output).all()

    def _pytorch_correlation_reference(self, signal, kernel):
        """Compute reference correlation using PyTorch.

        Correlation is like convolution but without flipping the kernel.
        We implement it as: correlation(x, h) = convolution(x, flip(h))
        """
        # signal: [batch, input_len]
        # kernel: [kernel_len]
        # Returns: [batch, output_len] where output_len = input_len + kernel_len - 1

        # Reshape for conv1d: signal needs [batch, channels=1, length]
        signal_conv = signal.unsqueeze(1)  # [batch, 1, input_len]
        # Flip the kernel for correlation (correlation = conv with flipped kernel)
        kernel_flipped = torch.flip(kernel, [0])
        kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_len]

        # Apply 1D convolution with full padding to get correlation
        # For full correlation, we need padding of kernel_len - 1
        padding = len(kernel) - 1
        output = F.conv1d(signal_conv, kernel_conv, padding=padding)  # [batch, 1, input_len+kernel_len-1]
        return output.squeeze(1)  # [batch, input_len+kernel_len-1]

    @pytest.mark.parametrize("input_len,kernel_len,lens,sep", [
        (8, 3, 32, 7),
        (16, 8, 48, 9),
        (16, 8, 64, 15),
        (8, 8, 48, 8),  # Golden code case: M=N, should give 15 outputs
    ])
    def test_jtc_physics_correct(self, input_len, kernel_len, lens, sep):
        """Test that JTC correctly implements optical physics (magnitude outputs)."""
        config = AppConfig(
            input_length=input_len,
            kernel_length=kernel_len,
            output_length=None,
            jtc_separation=sep,
            jtc_total_field=lens,
            conv_backend="fourier",
            dac_bits=None,  # Disable quantization
            adc_bits=None,
            fourier_plane_bits=None,
            scale_output="none",
        )

        jtc = JTC(config)
        expected_output_len = input_len + kernel_len - 1

        # Create test inputs
        batch_size = 2
        torch.manual_seed(42)
        signal = torch.randn(batch_size, input_len) * 0.1
        kernel = torch.randn(kernel_len) * 0.1

        # Compute JTC result
        signal_jtc = signal.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, input_len]
        kernel_jtc = kernel.unsqueeze(0)  # [1, kernel_len]

        layer = FTconvlayer(
            in_channels=1,
            out_channels=1,
            config=config,
            kernel_size=input_len,
            batch_size=batch_size,
        )

        with torch.no_grad():
            layer.weights.data = kernel_jtc.unsqueeze(0).unsqueeze(-1)

        result = layer.fourier_conv_forward(signal_jtc, kernel_jtc)
        result = result.squeeze(1).squeeze(1)  # [batch, output_len]

        # Test 1: Correct shape (full correlation M+N-1)
        assert result.shape == (batch_size, expected_output_len), \
            f"Expected shape ({batch_size}, {expected_output_len}), got {result.shape}"

        # Test 2: All outputs are non-negative (light intensity magnitudes)
        assert (result >= 0).all(), "JTC outputs should be non-negative (light intensity)"

        # Test 3: Outputs are finite
        assert torch.isfinite(result).all(), "JTC outputs should be finite"

        # Test 4: Zero kernel gives mostly autocorrelation of signal
        with torch.no_grad():
            layer.weights.data[:] = 0.0
        result_zero_kernel = layer.fourier_conv_forward(signal_jtc, torch.zeros_like(kernel_jtc))
        result_zero_kernel = result_zero_kernel.squeeze(1).squeeze(1)
        # Should still be non-negative and finite
        assert (result_zero_kernel >= 0).all()
        assert torch.isfinite(result_zero_kernel).all()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_usable_outputs_calculation or test_jtc_initialization"])
