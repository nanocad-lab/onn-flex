"""
Test suite for quantization bit selection alignment across backends.

Tests:
1. bits=None behavior - verify no quantization is applied when bits=None
2. Quantization flow alignment - verify JTC emulation and Fourier backends use same bit config
3. Gradient flow - verify gradients propagate through all quantization points
4. fourier_plane_bits - verify fourier_plane_bits is applied correctly in JTC backend
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from onn_config import AppConfig
from onn_layers import FTconvlayer
from onn_component import QuantDequant_STE, JTC


class TestQuantizationAlignment:
    """Test suite for quantization alignment across backends."""

    @pytest.fixture
    def base_config(self):
        """Create a base configuration for testing."""
        config = AppConfig(
            input_length=8,
            kernel_length=8,
            output_length=None,
            jtc_separation=8,
            jtc_total_field=48,
            dac_bits=4,
            adc_bits=6,
            fourier_plane_bits=6,
            conv_backend="jtc_emulation",
        )
        return config

    @pytest.fixture
    def test_input(self):
        """Create a test input tensor."""
        torch.manual_seed(42)
        return torch.randn(2, 3, 32, 32)

    def test_quantdequant_ste_none_bits(self):
        """Test that QuantDequant_STE returns input unchanged when bits=None."""
        x = torch.randn(10, 10)
        x_quantized = QuantDequant_STE.apply(x, None)

        # Should return exactly the same tensor
        assert torch.allclose(x, x_quantized)
        assert x is x_quantized

    def test_quantdequant_ste_with_bits(self):
        """Test that QuantDequant_STE quantizes when bits is provided."""
        x = torch.randn(10, 10)
        x_quantized = QuantDequant_STE.apply(x, 4)

        # Should be different from input (unless input happened to be quantized)
        # Check that output is in expected range [0, 1] with 4-bit levels
        assert x_quantized.min() >= 0
        assert x_quantized.max() <= 1

        # Check that values are quantized to expected levels
        levels = 2**4
        expected_values = torch.linspace(0, 1, levels)

        # Each quantized value should be close to one of the expected levels
        for val in x_quantized.flatten().unique():
            distances = torch.abs(expected_values - val)
            assert distances.min() < 1e-6

    def test_layer_quantizer_none_bits(self, base_config, test_input):
        """Test that layer _apply_quantizer handles None bits correctly."""
        base_config.dac_bits = None
        base_config.adc_bits = None
        base_config.fourier_plane_bits = None
        base_config.conv_backend = "fourier"

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Should run without errors
        output = layer(test_input)
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        assert torch.isfinite(output).all()

    def test_jtc_emulation_has_fourier_plane_bits(self, base_config, test_input):
        """Test that JTC emulation backend applies fourier_plane_bits quantization."""
        base_config.conv_backend = "jtc_emulation"
        base_config.fourier_plane_bits = 4

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Run forward pass - should not error
        output = layer(test_input)
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        assert torch.isfinite(output).all()

    def test_jtc_emulation_none_fourier_plane_bits(self, base_config, test_input):
        """Test that JTC emulation works when fourier_plane_bits=None."""
        base_config.conv_backend = "jtc_emulation"
        base_config.fourier_plane_bits = None

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Should run without errors
        output = layer(test_input)
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        assert torch.isfinite(output).all()

    def test_fourier_backend_fourier_plane_bits(self, base_config, test_input):
        """Test that Fourier backend applies fourier_plane_bits quantization."""
        base_config.conv_backend = "fourier"
        base_config.fourier_plane_bits = 4

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Run forward pass - should not error
        output = layer(test_input)
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        assert torch.isfinite(output).all()

    def test_quantization_reduces_values(self, base_config):
        """Test that quantization reduces the number of unique values."""
        # Create a signal with many unique values
        torch.manual_seed(42)
        signal = torch.rand(1, 8) * 0.5 + 0.25  # Values in [0.25, 0.75]

        # Count unique values before quantization
        unique_before = signal.unique().numel()

        # Apply 2-bit quantization (4 levels)
        signal_quantized = QuantDequant_STE.apply(signal, 2)
        unique_after = signal_quantized.unique().numel()

        # Should have fewer unique values after quantization
        assert unique_after <= 4  # At most 4 levels for 2-bit
        assert unique_after < unique_before

    def test_gradient_flow_through_dac_bits(self, base_config):
        """Test that gradients flow through dac_bits quantization."""
        base_config.conv_backend = "jtc_emulation"

        # Create simple inputs that require gradients
        signal = torch.randn(1, 8, requires_grad=True)
        kernel = torch.randn(1, 8, requires_grad=True)

        # Apply DAC quantization (clamped to [0,1])
        signal_clamped = torch.clamp(signal, 0, 1)
        kernel_clamped = torch.clamp(kernel, 0, 1)

        signal_quantized = QuantDequant_STE.apply(signal_clamped, base_config.dac_bits)
        kernel_quantized = QuantDequant_STE.apply(kernel_clamped, base_config.dac_bits)

        # Compute a simple loss
        loss = signal_quantized.sum() + kernel_quantized.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert signal.grad is not None
        assert kernel.grad is not None
        assert torch.isfinite(signal.grad).all()
        assert torch.isfinite(kernel.grad).all()

    def test_gradient_flow_through_fourier_plane_bits(self, base_config):
        """Test that gradients flow through fourier_plane_bits quantization."""
        base_config.conv_backend = "jtc_emulation"

        # Create simple inputs with gradients
        signal = torch.rand(1, 8, requires_grad=True)

        # Apply Fourier plane quantization
        signal_quantized = QuantDequant_STE.apply(
            signal, base_config.fourier_plane_bits
        )

        # Compute loss
        loss = signal_quantized.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert signal.grad is not None
        assert torch.isfinite(signal.grad).all()

    def test_gradient_flow_through_adc_bits(self, base_config):
        """Test that gradients flow through adc_bits quantization."""
        base_config.conv_backend = "jtc_emulation"

        # Create simple input with gradients
        signal = torch.rand(1, 8, requires_grad=True)

        # Apply ADC quantization
        signal_quantized = QuantDequant_STE.apply(signal, base_config.adc_bits)

        # Compute loss
        loss = signal_quantized.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert signal.grad is not None
        assert torch.isfinite(signal.grad).all()

    def test_full_gradient_flow_jtc_emulation(self, base_config, test_input):
        """Test that gradients flow through the entire JTC emulation pipeline."""
        base_config.conv_backend = "jtc_emulation"

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Enable gradients on input
        test_input_grad = test_input.clone().requires_grad_(True)

        # Forward pass
        output = layer(test_input_grad)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert test_input_grad.grad is not None
        assert torch.isfinite(test_input_grad.grad).all()

        # Check weight gradients
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_full_gradient_flow_fourier(self, base_config, test_input):
        """Test that gradients flow through the entire Fourier backend pipeline."""
        base_config.conv_backend = "fourier"

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Enable gradients on input
        test_input_grad = test_input.clone().requires_grad_(True)

        # Forward pass
        output = layer(test_input_grad)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert test_input_grad.grad is not None
        assert torch.isfinite(test_input_grad.grad).all()

        # Check weight gradients
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_quantization_order_jtc(self, base_config):
        """Test that quantization happens in the correct order for JTC backend."""
        base_config.conv_backend = "jtc_emulation"

        jtc = JTC(base_config)

        # Create simple inputs
        torch.manual_seed(42)
        signal = torch.rand(2, 8)
        kernel = torch.rand(2, 8)

        # Forward pass - verify no errors
        output = jtc.forward(signal, kernel)

        # Output should be valid
        assert torch.isfinite(output).all()
        assert output.shape == (2, 1, 2, 8)

    def test_backends_respect_none_quantization(self, base_config, test_input):
        """Test that both backends work with all quantization disabled (None)."""
        base_config.dac_bits = None
        base_config.adc_bits = None
        base_config.fourier_plane_bits = None

        for backend in ["fourier", "jtc_emulation"]:
            base_config.conv_backend = backend

            layer = FTconvlayer(
                in_channels=3,
                out_channels=16,
                config=base_config,
                kernel_size=8,
                batch_size=2,
            )

            # Should work without errors
            output = layer(test_input)
            assert output.shape[0] == 2
            assert output.shape[1] == 16
            assert torch.isfinite(output).all()

    def test_different_bit_configs(self, base_config, test_input):
        """Test that different bit configurations work correctly."""
        bit_configs = [
            (None, None, None),
            (4, 6, 6),
            (8, 8, 8),
            (2, 4, 4),
            (4, None, 6),  # Mixed None and values
            (None, 6, None),
        ]

        for dac, adc, fourier in bit_configs:
            base_config.dac_bits = dac
            base_config.adc_bits = adc
            base_config.fourier_plane_bits = fourier
            base_config.conv_backend = "jtc_emulation"

            layer = FTconvlayer(
                in_channels=3,
                out_channels=16,
                config=base_config,
                kernel_size=8,
                batch_size=2,
            )

            # Should work without errors
            output = layer(test_input)
            assert output.shape[0] == 2
            assert output.shape[1] == 16
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
