"""
Tests for the model registry and VGG implementations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from onn_config import AppConfig
from onn_layers import FTconvlayer
from onn_models import FFTConvNet, build_model


class TestModelRegistry:
    """Validate model selection and gradient flow."""

    def test_default_fftconv_selected(self):
        """Default config should instantiate FFTConvNet."""
        config = AppConfig()
        model = build_model(config)
        assert isinstance(model, FFTConvNet)

    @pytest.mark.parametrize("model_name", ["vgg3", "vgg11", "vgg16"])
    def test_vgg_gradient_flow(self, model_name: str):
        """Ensure gradients propagate through VGG variants."""
        config = AppConfig(
            model_name=model_name,
            conv_backend="pytorch",
            input_length=32,
            kernel_length=8,
            dac_bits=None,
            adc_bits=None,
            jtc_total_field=64,
        )

        model = build_model(config)
        model.train()

        inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()

        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )

        # First FT layer should have planner-optimized optics while root config stays intact
        ft_layers = [layer for layer in model.features if isinstance(layer, FTconvlayer)]
        assert ft_layers, "VGG features should include FTconvlayer blocks"
        first_cfg = ft_layers[0].config
        assert first_cfg.input_length == 22
        assert first_cfg.kernel_length == 3
        assert first_cfg.jtc_separation == 19
        # The base config is no longer mutated – it keeps the user-provided lengths
        assert config.input_length == 32
        assert config.kernel_length == 8
