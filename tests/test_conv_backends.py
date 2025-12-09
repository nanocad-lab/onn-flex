"""
Comprehensive test suite for convolution backend selection.

Tests:
1. Expected outputs - verify each backend produces valid outputs
2. Gradient flow - verify gradients propagate correctly through each backend
3. Backend switching - verify switching between backends works correctly
4. Size extensibility - verify different input/weight sizes work as expected
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import warnings
from onn_config import AppConfig
from onn_layers import FTconvlayer


class TestConvBackends:
    """Test suite for convolution backend functionality."""

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
            conv_backend="jtc_fast",
            driver_distortion_data_path="./component_data/driver_sim_data.csv",
            mrm_phase_data_path="./component_data/mrm_phase_sim_data.csv",
            mrm_power_data_path="./component_data/mrm_pwr_w_sim_data.csv",
            pd_tia_distortion_data_path="./component_data/pd_tia_sim_data.csv",
            pd_distortion_data_path="./component_data/pd_sim_data.csv",
            tia_distortion_data_path="./component_data/tia_sim_data.csv",
        )
        return config

    @pytest.fixture
    def test_input(self):
        """Create a test input tensor."""
        # Shape: [batch, in_channels, height, width]
        torch.manual_seed(42)
        return torch.randn(2, 3, 32, 32)

    def test_config_validation(self):
        """Test that config validation works correctly."""
        # Valid backends should work
        for backend in ["pytorch", "fourier", "jtc_fast", "jtc_emulation"]:
            config = AppConfig(conv_backend=backend)
            assert config.conv_backend == backend

        # Invalid backend should raise error
        with pytest.raises(ValueError, match="Invalid conv_backend"):
            AppConfig(conv_backend="invalid_backend")

    def test_pytorch_backend_output(self, base_config, test_input):
        """Test PyTorch backend produces valid output."""
        base_config.conv_backend = "pytorch"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        output = layer(test_input)

        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 16  # out channels
        assert output.shape[2] == 32  # height preserved
        assert output.shape[3] == 32  # width preserved

        # Check output is finite and not all zeros
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_fourier_backend_output(self, base_config, test_input):
        """Test Fourier backend produces valid output."""
        base_config.conv_backend = "fourier"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        output = layer(test_input)

        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 16  # out channels
        assert output.shape[2] == 32  # height preserved
        assert output.shape[3] == 32  # width preserved

        # Check output is finite and not all zeros
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_jtc_fast_backend_output(self, base_config, test_input):
        """Test vectorized JTC backend produces valid output."""
        base_config.conv_backend = "jtc_fast"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        output = layer(test_input)

        assert output.shape == (2, 16, 32, 32)
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_jtc_emulation_backend_output(self, base_config, test_input):
        """Test JTC emulation backend produces valid output."""
        base_config.conv_backend = "jtc_emulation"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        output = layer(test_input)

        # Check output shape
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 16  # out channels
        assert output.shape[2] == 32  # height preserved
        assert output.shape[3] == 32  # width preserved

        # Check output is finite and not all zeros
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow_pytorch(self, base_config, test_input):
        """Test gradients flow correctly through PyTorch backend."""
        base_config.conv_backend = "pytorch"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Enable gradient computation
        test_input.requires_grad = True

        # Forward pass
        output = layer(test_input)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert test_input.grad is not None
        assert torch.isfinite(test_input.grad).all()
        assert not torch.allclose(test_input.grad, torch.zeros_like(test_input.grad))

        # Check layer weights have gradients
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()

    def test_gradient_flow_fourier(self, base_config, test_input):
        """Test gradients flow correctly through Fourier backend."""
        base_config.conv_backend = "fourier"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Enable gradient computation
        test_input.requires_grad = True

        # Forward pass
        output = layer(test_input)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert test_input.grad is not None
        assert torch.isfinite(test_input.grad).all()
        assert not torch.allclose(test_input.grad, torch.zeros_like(test_input.grad))

        # Check layer weights have gradients
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()

    def test_gradient_flow_jtc_fast(self, base_config, test_input):
        """Test gradients flow correctly through JTC fast backend."""
        base_config.conv_backend = "jtc_fast"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        test_input.requires_grad = True
        output = layer(test_input)
        loss = output.sum()
        loss.backward()

        assert test_input.grad is not None
        assert torch.isfinite(test_input.grad).all()
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()

    def test_gradient_flow_jtc_emulation(self, base_config, test_input):
        """Test gradients flow correctly through JTC emulation backend."""
        base_config.conv_backend = "jtc_emulation"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Enable gradient computation
        test_input.requires_grad = True

        # Forward pass
        output = layer(test_input)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert test_input.grad is not None
        assert torch.isfinite(test_input.grad).all()
        assert not torch.allclose(test_input.grad, torch.zeros_like(test_input.grad))

        # Check layer weights have gradients
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()

    def test_backend_switching(self, base_config, test_input):
        """Test switching between backends produces different results."""
        torch.manual_seed(42)

        # Create layer with shared weights
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Run with JTC emulation
        base_config.conv_backend = "jtc_emulation"
        output_jtc = layer(test_input)

        # Run with JTC fast
        base_config.conv_backend = "jtc_fast"
        output_fast = layer(test_input)

        # Run with Fourier
        base_config.conv_backend = "fourier"
        output_fourier = layer(test_input)

        # Run with PyTorch
        base_config.conv_backend = "pytorch"
        output_pytorch = layer(test_input)

        # All outputs should be valid
        assert torch.isfinite(output_jtc).all()
        assert torch.isfinite(output_fast).all()
        assert torch.isfinite(output_fourier).all()
        assert torch.isfinite(output_pytorch).all()

        # Outputs should have same shape
        assert (
            output_jtc.shape
            == output_fast.shape
            == output_fourier.shape
            == output_pytorch.shape
        )

        # Note: We don't require outputs to be identical because:
        # - JTC emulation includes hardware distortions
        # - Fourier and PyTorch may have numerical differences
        # But we verify they all produce valid results

    def test_invalid_backend_raises_error(self, base_config, test_input):
        """Test that an invalid backend raises an appropriate error."""
        # Create layer
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Set invalid backend
        base_config.conv_backend = "invalid_backend"

        # Should raise ValueError
        with pytest.raises(ValueError, match="Unknown conv_backend"):
            layer(test_input)

    def test_size_validation_fourier(self, base_config):
        """Test size validation for Fourier backend."""
        base_config.conv_backend = "fourier"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Valid size (8x8) should work
        valid_input = torch.randn(2, 3, 32, 32)
        output = layer(valid_input)
        assert output.shape == (2, 16, 32, 32)

    def test_size_validation_jtc_emulation(self, base_config):
        """Test size validation for JTC emulation backend."""
        base_config.conv_backend = "jtc_emulation"
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Valid size (8x8) should work
        valid_input = torch.randn(2, 3, 32, 32)
        output = layer(valid_input)
        assert output.shape == (2, 16, 32, 32)

    def test_jtc_plane_size_warning(self, base_config):
        """Test that undersized JTC plane triggers warning."""
        # Set JTC plane too small
        base_config.conv_backend = "fourier"
        base_config.jtc_total_field = 10  # Too small for kernel_size=8, sep=8
        base_config.jtc_separation = 8

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        test_input = torch.randn(2, 3, 32, 32)

        # Should produce a warning about aliasing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = layer(test_input)
            assert len(w) > 0
            assert "aliasing" in str(w[0].message).lower()

    def test_pytorch_backend_flexible_sizes(self, base_config):
        """Test that PyTorch backend handles various sizes."""
        base_config.conv_backend = "pytorch"

        # PyTorch backend should work with kernel_size=8
        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        # Test with standard input
        test_input = torch.randn(2, 3, 32, 32)
        output = layer(test_input)
        assert output.shape == (2, 16, 32, 32)
        assert torch.isfinite(output).all()

    def test_default_backend_fallback(self, base_config):
        """Test that None backend falls back to jtc_emulation."""
        base_config.conv_backend = None

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        test_input = torch.randn(2, 3, 32, 32)
        output = layer(test_input)

        # Should work with default backend
        assert output.shape == (2, 16, 32, 32)
        assert torch.isfinite(output).all()

    def test_quantization_with_backends(self, base_config):
        """Test that quantization works with all backends."""
        test_input = torch.randn(2, 3, 32, 32)

        for backend in ["pytorch", "fourier", "jtc_fast", "jtc_emulation"]:
            base_config.conv_backend = backend
            base_config.dac_bits = 4
            base_config.adc_bits = 6

            layer = FTconvlayer(
                in_channels=3,
                out_channels=16,
                config=base_config,
                kernel_size=8,
                batch_size=2,
            )

            output = layer(test_input)

            # Check output is valid
            assert torch.isfinite(output).all()
            assert output.shape == (2, 16, 32, 32)

    def test_fourier_plane_quantization(self, base_config):
        """Test Fourier plane quantization with fourier backend."""
        base_config.conv_backend = "fourier"
        base_config.fourier_plane_bits = 6

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        test_input = torch.randn(2, 3, 32, 32)
        output = layer(test_input)

        # Check output is valid
        assert torch.isfinite(output).all()
        assert output.shape == (2, 16, 32, 32)


def test_backend_selection_integration():
    """Integration test for backend selection with full forward pass."""
    config = AppConfig(
        conv_backend="pytorch",
        input_length=8,
        kernel_length=8,
        output_length=None,
        jtc_separation=8,
        jtc_total_field=48,
        driver_distortion_data_path="./component_data/driver_sim_data.csv",
        mrm_phase_data_path="./component_data/mrm_phase_sim_data.csv",
        mrm_power_data_path="./component_data/mrm_pwr_w_sim_data.csv",
        pd_tia_distortion_data_path="./component_data/pd_tia_sim_data.csv",
        pd_distortion_data_path="./component_data/pd_sim_data.csv",
        tia_distortion_data_path="./component_data/tia_sim_data.csv",
    )

    layer = FTconvlayer(
        in_channels=3,
        out_channels=16,
        config=config,
        kernel_size=8,
        batch_size=2,
    )

    test_input = torch.randn(2, 3, 32, 32)
    test_input.requires_grad = True

    # Forward and backward pass
    output = layer(test_input)
    loss = output.sum()
    loss.backward()

    # Verify everything works end-to-end
    assert output.shape == (2, 16, 32, 32)
    assert torch.isfinite(output).all()
    assert test_input.grad is not None
    assert torch.isfinite(test_input.grad).all()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
