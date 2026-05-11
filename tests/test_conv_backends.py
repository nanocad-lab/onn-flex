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
import types
from onn_config import AppConfig
from onn_layers import FTconvlayer, JTCConv2d, replace_conv2d_with_jtc


def _ideal_unquantized_equivalence_case():
    config = AppConfig(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        conv_backend="jtc_ideal",
        loss=1.0,
        driver_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        mrm_power_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        lens_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        enable_jtc_batched_fast_path=False,
        enable_jtc_ideal_fused_transfer=False,
    )
    layer = FTconvlayer(
        in_channels=1,
        out_channels=2,
        config=config,
        kernel_size=8,
        batch_size=1,
        bias=False,
    )

    torch.manual_seed(123)
    test_input = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[..., 0].copy_(torch.rand_like(layer.weight[..., 0]))
    return config, layer, test_input


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
            conv_backend="jtc_emulation",
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
        for backend in ["pytorch", "jtc_ideal", "jtc_emulation"]:
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

    def test_jtc_ideal_backend_output(self, base_config, test_input):
        """Test JTC ideal backend produces valid output."""
        base_config.conv_backend = "jtc_ideal"
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

        # Check layer weight have gradients
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_gradient_flow_jtc_ideal(self, base_config, test_input):
        """Test gradients flow correctly through JTC ideal backend."""
        base_config.conv_backend = "jtc_ideal"
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

        # Check layer weight have gradients
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

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

        # Check layer weight have gradients
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_backend_switching(self, base_config, test_input):
        """Test switching between backends produces different results."""
        torch.manual_seed(42)

        # Create layer with shared weight
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

        # Run with Fourier
        base_config.conv_backend = "jtc_ideal"
        output_jtc_ideal = layer(test_input)

        # Run with PyTorch
        base_config.conv_backend = "pytorch"
        output_pytorch = layer(test_input)

        # All outputs should be valid
        assert torch.isfinite(output_jtc).all()
        assert torch.isfinite(output_jtc_ideal).all()
        assert torch.isfinite(output_pytorch).all()

        # Outputs should have same shape
        assert output_jtc.shape == output_jtc_ideal.shape == output_pytorch.shape

        # Note: We don't require outputs to be identical because:
        # - JTC emulation includes hardware distortions
        # - JTC ideal and PyTorch may have numerical differences
        # But we verify they all produce valid results

    def test_jtc_ideal_matches_pytorch_for_unquantized_positive_inputs(self):
        """Unquantized ideal JTC backend should match PyTorch patch math."""
        config = AppConfig(
            input_length=8,
            kernel_length=8,
            output_length=8,
            jtc_separation=8,
            jtc_total_field=48,
            dac_bits=None,
            adc_bits=None,
            fourier_plane_bits=None,
            scale_output="none",
            conv_backend="jtc_ideal",
            loss=1.0,
        )
        layer = FTconvlayer(
            in_channels=2,
            out_channels=4,
            config=config,
            kernel_size=8,
            batch_size=2,
            bias=False,
        )

        torch.manual_seed(123)
        test_input = torch.rand(2, 2, 16, 16)
        with torch.no_grad():
            layer.weight.copy_(torch.rand_like(layer.weight))

        config.conv_backend = "jtc_ideal"
        output_jtc_ideal = layer(test_input)
        config.conv_backend = "pytorch"
        output_pytorch = layer(test_input)

        torch.testing.assert_close(
            output_jtc_ideal, output_pytorch, rtol=1e-5, atol=1e-5
        )

    def test_jtc_ideal_matches_pytorch_for_unquantized_ideal_config(self):
        config, layer, test_input = _ideal_unquantized_equivalence_case()

        config.conv_backend = "pytorch"
        output_pytorch = layer(test_input)
        config.conv_backend = "jtc_ideal"
        output_jtc_ideal = layer(test_input)

        torch.testing.assert_close(
            output_jtc_ideal, output_pytorch, rtol=1e-5, atol=1e-5
        )

    def test_jtc_emulation_matches_jtc_ideal_for_unquantized_ideal_config(self):
        config, layer, test_input = _ideal_unquantized_equivalence_case()

        config.conv_backend = "jtc_ideal"
        output_jtc_ideal = layer(test_input)
        config.conv_backend = "jtc_emulation"
        output_jtc_emulation = layer(test_input)

        torch.testing.assert_close(
            output_jtc_emulation, output_jtc_ideal, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize(
        "dac_bits,fourier_plane_bits,adc_bits",
        [
            (4, None, None),
            (None, 6, None),
            (None, None, 6),
            (4, 6, 6),
        ],
    )
    def test_jtc_emulation_matches_jtc_ideal_for_quantized_ideal_config(
        self, dac_bits, fourier_plane_bits, adc_bits
    ):
        config, layer, test_input = _ideal_unquantized_equivalence_case()
        config.dac_bits = dac_bits
        config.fourier_plane_bits = fourier_plane_bits
        config.adc_bits = adc_bits

        config.conv_backend = "jtc_ideal"
        output_jtc_ideal = layer(test_input)
        config.conv_backend = "jtc_emulation"
        output_jtc_emulation = layer(test_input)

        torch.testing.assert_close(
            output_jtc_emulation, output_jtc_ideal, rtol=1e-5, atol=1e-5
        )

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

    def test_size_validation_jtc_ideal(self, base_config):
        """Test size validation for JTC ideal backend."""
        base_config.conv_backend = "jtc_ideal"
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
        base_config.conv_backend = "jtc_ideal"
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
            _ = layer(test_input)
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

    def test_none_backend_raises_error(self, base_config):
        """Runtime backend routing should not silently fall back from None."""
        base_config.conv_backend = None

        layer = FTconvlayer(
            in_channels=3,
            out_channels=16,
            config=base_config,
            kernel_size=8,
            batch_size=2,
        )

        test_input = torch.randn(2, 3, 32, 32)
        with pytest.raises(ValueError, match="Unknown conv_backend"):
            layer(test_input)

    def test_quantization_with_backends(self, base_config):
        """Test that quantization works with all backends."""
        test_input = torch.randn(2, 3, 32, 32)

        for backend in ["pytorch", "jtc_ideal", "jtc_emulation"]:
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
        """Test Fourier plane quantization with jtc_ideal backend."""
        base_config.conv_backend = "jtc_ideal"
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


def test_jtc_batched_fast_path_matches_reference_accumulation():
    """Fast path must preserve the old sum-positive-minus-sum-negative order."""
    for output_length in (8, None, 10):
        config = AppConfig(
            input_length=8,
            kernel_length=8,
            output_length=output_length,
            jtc_separation=8,
            jtc_total_field=48,
            dac_bits=4,
            adc_bits=6,
            fourier_plane_bits=6,
            conv_backend="jtc_emulation",
            enable_jtc_batched_fast_path=True,
        )

        torch.manual_seed(123)
        fast_layer = FTconvlayer(
            in_channels=3,
            out_channels=8,
            config=config,
            kernel_size=8,
            batch_size=2,
        )

        ref_config = AppConfig(
            input_length=8,
            kernel_length=8,
            output_length=output_length,
            jtc_separation=8,
            jtc_total_field=48,
            dac_bits=4,
            adc_bits=6,
            fourier_plane_bits=6,
            conv_backend="jtc_emulation",
            enable_jtc_batched_fast_path=False,
        )
        ref_layer = FTconvlayer(
            in_channels=3,
            out_channels=8,
            config=ref_config,
            kernel_size=8,
            batch_size=2,
        )
        ref_layer.load_state_dict(fast_layer.state_dict())

        def reference_pseudo_forward(self, x, weight):
            return self.conv_forward(x, weight[..., 0]) - self.conv_forward(
                x, weight[..., 1]
            )

        ref_layer.pseudo_forward = types.MethodType(
            reference_pseudo_forward, ref_layer
        )

        x = torch.rand(2, 3, 16, 16)
        with torch.no_grad():
            fast_out = fast_layer(x)
            ref_out = ref_layer(x)

        assert torch.equal(fast_out, ref_out)


def test_jtc_conv2d_jtc_ideal_matches_stock_conv2d_signed_inputs():
    """JTCConv2d should be a Conv2d equivalent under the ideal JTC backend."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(7)
    stock = torch.nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=4096)
    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)


def test_jtc_conv2d_jtc_ideal_matches_stock_grouped_conv2d():
    """Grouped Conv2d semantics should be preserved by the JTC ideal wrapper."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(11)
    stock = torch.nn.Conv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=2,
        bias=False,
    )
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=4096)
    x = torch.randn(2, 4, 9, 9)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)


def test_jtc_conv2d_jtc_ideal_explicit_optical_path_matches_stock_conv2d():
    """The explicit optical-shot JTC ideal path stays equivalent."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(17)
    stock = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=512)
    x = torch.randn(1, 2, 5, 5)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)


def test_jtc_conv2d_jtc_ideal_rowwise_3x3_matches_stock_conv2d():
    """The 3x3 row-wise JTC path should stay Conv2d-equivalent."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(29)
    stock = torch.nn.Conv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
    )
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=128)
    x = torch.randn(1, 2, 32, 32)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)
    assert "row_jtc_ideal_34" in jtc._jtc_cache


def test_jtc_conv2d_jtc_ideal_packed_3x3_matches_stock_conv2d():
    """Small rows should pack multiple row correlations into each JTC shot."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(41)
    stock = torch.nn.Conv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=64)
    x = torch.randn(2, 4, 16, 16)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)
    assert "row_jtc_ideal_58" in jtc._jtc_cache


def test_jtc_conv2d_nonnegative_input_mode_matches_signed_path():
    """Known nonnegative inputs can skip the zero signal-negative optical banks."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(47)
    stock = torch.nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    signed = JTCConv2d.from_conv2d(
        stock,
        config=config,
        max_jtc_shots=128,
    )
    nonnegative = JTCConv2d.from_conv2d(
        stock,
        config=config,
        max_jtc_shots=128,
        assume_nonnegative_input=True,
    )
    x = torch.rand(2, 3, 16, 16)

    with torch.no_grad():
        signed_out = signed(x)
        nonnegative_out = nonnegative(x)

    torch.testing.assert_close(nonnegative_out, signed_out, rtol=1e-4, atol=1e-4)


def test_jtc_conv2d_rowwise_limit_recalculates_for_kernel_width():
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
    )
    conv3 = JTCConv2d(1, 1, kernel_size=3, config=config)
    conv5 = JTCConv2d(1, 1, kernel_size=5, config=config)

    assert conv3._rowwise_clean_input_limit() == 64
    assert conv5._rowwise_clean_input_limit() == 65


def test_jtc_conv2d_jtc_ideal_rowwise_5x5_matches_stock_conv2d():
    """Row-wise JTC geometry should be recalculated for wider kernels."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
    )
    torch.manual_seed(47)
    stock = torch.nn.Conv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=True,
    )
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=128)
    x = torch.randn(1, 2, 16, 16)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)
    assert "row_jtc_ideal_44" in jtc._jtc_cache


def test_jtc_conv2d_1x1_uses_native_torch_for_jtc_backend():
    """1x1 JTCConv2d is intentionally native torch until optical packing exists."""
    config = AppConfig(
        conv_backend="jtc_emulation",
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=6,
        scale_output="none",
        loss=0.9,
        driver_distortion_strength=1.0,
        pd_distortion_strength=1.0,
        tia_distortion_strength=1.0,
        mrm_power_distortion_strength=1.0,
        mrm_phase_distortion_strength=1.0,
        lens_distortion_strength=1.0,
    )
    torch.manual_seed(31)
    stock = torch.nn.Conv2d(4, 6, kernel_size=1, bias=True)
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=64)
    x = torch.randn(2, 4, 8, 8)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=0, atol=0)
    assert len(jtc._jtc_cache) == 0


def test_jtc_conv2d_asymmetric_same_padding_matches_stock_conv2d():
    """Even kernels with padding='same' require asymmetric pre-padding."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        loss=1.0,
        driver_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        mrm_power_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        lens_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(61)
    stock = torch.nn.Conv2d(
        in_channels=2,
        out_channels=3,
        kernel_size=(2, 3),
        padding="same",
        bias=True,
    )
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=256)
    x = torch.randn(2, 2, 5, 6)

    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)


def test_jtc_conv2d_rowwise_3x3_quantized_cifar_width_equivalence():
    """Quantized row-wise 3x3 should align between ideal and emulation configs."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=6,
        scale_output="none",
        loss=1.0,
        driver_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        mrm_power_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        lens_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
    )
    torch.manual_seed(37)
    stock = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=512)
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        config.conv_backend = "jtc_ideal"
        ideal = jtc(x)
        config.conv_backend = "jtc_emulation"
        emulation = jtc(x)

    torch.testing.assert_close(emulation, ideal, rtol=1e-5, atol=1e-5)
    assert "row_jtc_ideal_34" in jtc._jtc_cache
    assert "row_jtc_emulation_34" in jtc._jtc_cache


def test_jtc_conv2d_packed_3x3_quantized_equivalence():
    """Quantized packed 3x3 rows should align between ideal and emulation."""
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=6,
        scale_output="none",
        loss=1.0,
        driver_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        mrm_power_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        lens_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
    )
    torch.manual_seed(43)
    stock = torch.nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=True)
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=128)
    x = torch.randn(1, 4, 16, 16)

    with torch.no_grad():
        config.conv_backend = "jtc_ideal"
        ideal = jtc(x)
        config.conv_backend = "jtc_emulation"
        emulation = jtc(x)

    torch.testing.assert_close(emulation, ideal, rtol=1e-5, atol=1e-5)
    assert "row_jtc_ideal_58" in jtc._jtc_cache
    assert "row_jtc_emulation_58" in jtc._jtc_cache


@pytest.mark.parametrize(
    "dac_bits,fourier_plane_bits,adc_bits",
    [
        (4, None, None),
        (None, 6, None),
        (None, None, 6),
        (4, 6, 6),
    ],
)
def test_jtc_conv2d_jtc_emulation_matches_jtc_ideal_quantized(
    dac_bits, fourier_plane_bits, adc_bits
):
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=dac_bits,
        adc_bits=adc_bits,
        fourier_plane_bits=fourier_plane_bits,
        scale_output="none",
        loss=1.0,
        driver_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        mrm_power_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        lens_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(23)
    stock = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    jtc = JTCConv2d.from_conv2d(stock, config=config, max_jtc_shots=512)
    x = torch.randn(1, 2, 5, 5)

    with torch.no_grad():
        config.conv_backend = "jtc_ideal"
        ideal = jtc(x)
        config.conv_backend = "jtc_emulation"
        emulation = jtc(x)

    torch.testing.assert_close(emulation, ideal, rtol=1e-5, atol=1e-5)


def test_replace_conv2d_with_jtc_preserves_torchvision_style_module():
    config = AppConfig(
        conv_backend="jtc_ideal",
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        jtc_separation=8,
        jtc_total_field=0,
    )
    torch.manual_seed(19)
    stock = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, kernel_size=1, bias=True),
    )
    jtc = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, kernel_size=1, bias=True),
    )
    jtc.load_state_dict(stock.state_dict())
    replace_conv2d_with_jtc(jtc, config=config, max_jtc_shots=4096)

    assert isinstance(jtc[0], JTCConv2d)
    assert isinstance(jtc[2], JTCConv2d)

    x = torch.randn(1, 3, 6, 6)
    with torch.no_grad():
        stock_out = stock(x)
        jtc_out = jtc(x)

    torch.testing.assert_close(jtc_out, stock_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
