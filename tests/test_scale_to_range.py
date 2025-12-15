"""
Unit tests for JTC.scale_to_range behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from onn_config import AppConfig
from onn_component import JTC


def _config() -> AppConfig:
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
        conv_backend="jtc_fast",
        driver_distortion_data_path="./component_data/driver_sim_data.csv",
        mrm_phase_data_path="./component_data/mrm_phase_sim_data.csv",
        mrm_power_data_path="./component_data/mrm_pwr_w_sim_data.csv",
        pd_tia_distortion_data_path="./component_data/pd_tia_sim_data.csv",
        pd_distortion_data_path="./component_data/pd_sim_data.csv",
        tia_distortion_data_path="./component_data/tia_sim_data.csv",
    )


def test_scale_to_range_scales_each_row_independently():
    jtc = JTC(_config())
    x = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.0, 50.0, 100.0],
        ]
    )
    y = jtc.scale_to_range(x, 1e-6, 1e-5)
    torch.testing.assert_close(y[0, 0], torch.tensor(1e-6))
    torch.testing.assert_close(y[0, -1], torch.tensor(1e-5))
    torch.testing.assert_close(y[1, 0], torch.tensor(1e-6))
    torch.testing.assert_close(y[1, -1], torch.tensor(1e-5))


def test_scale_to_range_constant_row_maps_to_min():
    jtc = JTC(_config())
    x = torch.full((2, 4), 3.14)
    y = jtc.scale_to_range(x, -2.0, 2.0)
    torch.testing.assert_close(y, torch.full_like(y, -2.0))


def test_scale_to_range_multi_dim_common_gain():
    """Scaling over multiple dims should apply a shared gain/offset."""
    jtc = JTC(_config())
    x = torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 50.0, 100.0]]]])  # (1,1,2,3)
    y = jtc.scale_to_range(x, 1e-6, 1e-5, dims=(-2, -1))

    # Global min/max over the last two dims are 0 and 100.
    torch.testing.assert_close(y.min(), torch.tensor(1e-6))
    torch.testing.assert_close(y.max(), torch.tensor(1e-5))

    # Channel 0 max (=2) should map to 1e-6 + (2/100)*(9e-6) = 1.18e-6.
    torch.testing.assert_close(y[0, 0, 0, -1], torch.tensor(1.18e-6))
