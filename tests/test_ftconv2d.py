"""
Regression tests for FTConv2d to ensure parity with PyTorch Conv2d.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from onn_config import AppConfig
from onn_layers import FTConv2d


def _base_config(backend: str) -> AppConfig:
    """Create a config with clean settings and valid component data paths."""
    return AppConfig(
        input_length=8,
        kernel_length=3,
        output_length=None,
        jtc_separation=7,
        jtc_total_field=32,
        dac_bits=None,
        adc_bits=None,
        fourier_plane_bits=None,
        scale_output="none",
        conv_backend=backend,
        driver_distortion_data_path="./component_data/driver_sim_data.csv",
        mrm_phase_data_path="./component_data/mrm_phase_sim_data.csv",
        mrm_power_data_path="./component_data/mrm_pwr_w_sim_data.csv",
        pd_tia_distortion_data_path="./component_data/pd_tia_sim_data.csv",
        pd_distortion_data_path="./component_data/pd_sim_data.csv",
        tia_distortion_data_path="./component_data/tia_sim_data.csv",
    )


@pytest.mark.parametrize("backend", ["pytorch", "fourier"])
def test_ftconv2d_matches_pytorch_conv2d(backend: str):
    """FTConv2d should numerically match torch Conv2d for clean configs."""
    torch.manual_seed(0)
    batch, in_channels, out_channels = 2, 3, 4
    height = width = 16
    kernel_size = (3, 3)

    # Reference convolution
    ref_conv = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        bias=True,
    )

    # Under-test convolution
    config = _base_config(backend)
    ft_conv = FTConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        config=config,
        conv_backend=backend,
        bias=True,
    )

    with torch.no_grad():
        ft_conv.weight.copy_(ref_conv.weight)
        ft_conv.bias.copy_(ref_conv.bias)

    x = torch.randn(batch, in_channels, height, width)
    ref_out = ref_conv(x)
    test_out = ft_conv(x)

    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=1e-4,
        atol=1e-4,
        msg=f"backend={backend} failed to match PyTorch conv2d",
    )


def test_ftconv2d_jtc_emulation_backend_runs():
    """Ensure the JTC emulation backend executes and returns finite outputs."""
    config = _base_config("jtc_emulation")
    layer = FTConv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3),
        config=config,
        conv_backend="jtc_emulation",
        bias=False,
    )
    x = torch.rand(1, 1, 16, 16)
    out = layer(x)
    assert out.shape == (1, 2, 16, 16)
    assert torch.isfinite(out).all()


def test_ftconv2d_jtc_fast_backend_runs():
    """Ensure the fast JTC backend executes and returns finite outputs."""
    config = _base_config("jtc_fast")
    layer = FTConv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3),
        config=config,
        conv_backend="jtc_fast",
        bias=False,
    )
    x = torch.rand(1, 1, 16, 16)
    out = layer(x)
    assert out.shape == (1, 2, 16, 16)
    assert torch.isfinite(out).all()


def test_ftconv2d_jtc_fast_bipolar_weight_gradients_flow():
    """Negative weights should be supported via differential (+/- rail) encoding."""
    config = _base_config("jtc_fast")
    config.differential_weights = True
    config.dac_bits = 4
    config.adc_bits = 6
    config.fourier_plane_bits = None
    config.jtc_checkpoint = False

    layer = FTConv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3),
        config=config,
        conv_backend="jtc_fast",
        bias=False,
    )
    x = torch.rand(1, 1, 16, 16)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum() > 0
