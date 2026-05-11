import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onn_config import AppConfig, load_app_config_from_yaml


def test_config_backend_validation():
    for backend in ["pytorch", "jtc_ideal", "jtc_emulation"]:
        config = AppConfig(conv_backend=backend)
        assert config.conv_backend == backend

    with pytest.raises(ValueError, match="Invalid conv_backend"):
        AppConfig(conv_backend="invalid_backend")

    with pytest.raises(ValueError, match="Invalid conv_backend"):
        AppConfig(conv_backend="fourier")

    with pytest.raises(ValueError, match="Invalid conv_backend"):
        AppConfig(conv_backend=None)


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
        conv_backend="jtc_ideal",
        input_length=8,
        kernel_length=8,
        output_length=None,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=6,
    )

    assert config.conv_backend == "jtc_ideal"
    assert config.input_length == 8
    assert config.kernel_length == 8
    assert config.output_length is None
    assert config.jtc_separation == 8
    assert config.jtc_total_field == 48


def test_backend_choices():
    for backend in ["pytorch", "jtc_ideal", "jtc_emulation"]:
        config = AppConfig(
            conv_backend=backend,
            input_length=8,
            kernel_length=8,
            output_length=None,
            jtc_separation=8,
            jtc_total_field=48,
        )
        assert config.conv_backend == backend


def test_yaml_loader_rejects_unknown_fields(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("conv_backend: jtc_emulation\nlegacy_backend: fourier\n")

    with pytest.raises(ValueError, match="Unknown config field"):
        load_app_config_from_yaml(str(cfg))
