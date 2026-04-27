import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onn_config import AppConfig


def test_config_backend_validation():
    for backend in ["pytorch", "fourier", "jtc_emulation"]:
        config = AppConfig(conv_backend=backend)
        assert config.conv_backend == backend

    with pytest.raises(ValueError, match="Invalid conv_backend"):
        AppConfig(conv_backend="invalid_backend")

    config = AppConfig(conv_backend=None)
    assert config.conv_backend is None


def test_config_defaults():
    config = AppConfig()

    assert config.conv_backend == "jtc_emulation"
    assert config.input_length == 8
    assert config.kernel_length == 8
    assert config.output_length is None
    assert config.jtc_separation == 0
    assert config.jtc_total_field == 16
    assert config.dac_bits == 4
    assert config.adc_bits == 6


def test_config_explicit_values():
    config = AppConfig(
        conv_backend="fourier",
        input_length=8,
        kernel_length=8,
        output_length=None,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=6,
        quantizer="ste_clipped",
    )

    assert config.conv_backend == "fourier"
    assert config.input_length == 8
    assert config.kernel_length == 8
    assert config.output_length is None
    assert config.jtc_separation == 8
    assert config.jtc_total_field == 48


def test_backend_choices():
    for backend in ["pytorch", "fourier", "jtc_emulation"]:
        config = AppConfig(
            conv_backend=backend,
            input_length=8,
            kernel_length=8,
            output_length=None,
            jtc_separation=8,
            jtc_total_field=48,
        )
        assert config.conv_backend == backend
