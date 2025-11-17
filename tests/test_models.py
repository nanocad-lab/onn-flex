"""
Tests for the model registry and VGG implementations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from onn_config import AppConfig
from onn_models import FFTConvNet, build_model


class TestModelRegistry:
    """Validate model selection and gradient flow."""

    def test_default_fftconv_selected(self):
        """Default config should instantiate FFTConvNet."""
        config = AppConfig()
        model = build_model(config)
        assert isinstance(model, FFTConvNet)

    @pytest.mark.parametrize("model_name", ["vgg11", "vgg16"])
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

        # Verify that the cycle planner selected the expected lengths
        assert config.input_length == 22
        assert config.kernel_length == 3
        assert config.jtc_separation == 19
