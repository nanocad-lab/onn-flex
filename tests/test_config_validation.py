import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onn_config import AppConfig, load_app_config_from_yaml


def test_cli_help_lists_training_and_physics_flags_without_output_directory(
    monkeypatch, capsys
):
    from onn_main import main

    monkeypatch.setattr(sys, "argv", ["onn_main.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--pretrained-weights",
        "--resume-checkpoint",
        "--adc-bits",
        "--dataset",
    ):
        assert option in help_text


@pytest.mark.parametrize("field", ["dac_bits", "adc_bits", "fourier_plane_bits"])
@pytest.mark.parametrize("value", [True, 6.5, "6"])
def test_converter_resolution_is_an_integer_contract(field, value):
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        AppConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "pd_noise_w",
        "driver_distortion_strength",
        "mrm_amplitude_distortion_strength",
        "mrm_phase_distortion_strength",
        "pd_distortion_strength",
        "tia_distortion_strength",
        "lens_distortion_strength",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_physics_settings_rejected(field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        AppConfig(**{field: value})


def test_zero_gain_headroom_is_not_silently_replaced():
    with pytest.raises(ValueError, match="jtc_gain_headroom"):
        AppConfig(jtc_gain_headroom=0)


@pytest.mark.parametrize(
    "field",
    [
        "driver_distortion_strength",
        "mrm_amplitude_distortion_strength",
        "mrm_phase_distortion_strength",
        "pd_distortion_strength",
        "tia_distortion_strength",
        "lens_distortion_strength",
    ],
)
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_strengths_follow_the_paper_interpolation_domain(field, value):
    with pytest.raises(ValueError, match="must be in"):
        AppConfig(**{field: value})


def test_removed_configuration_keys_are_rejected(tmp_path):
    path = tmp_path / "removed.yaml"
    path.write_text("jtc_error_injection_interval: 10\n")
    with pytest.raises(
        ValueError, match="Unknown config field.*jtc_error_injection_interval"
    ):
        load_app_config_from_yaml(path)


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
    assert config.resume_checkpoint == ""
    assert config.checkpoint_interval == 0
    assert config.checkpoint_time_interval_minutes == 0.0
    assert config.enable_ddp is True
    assert config.model_arch == "fftconvnet"
    assert config.jtc_assume_nonnegative_input is False
    assert config.jtc_max_shots == 65536
    assert config.enable_jtc_activation_checkpointing is False
    assert config.run_full_strength_inference is True
    assert config.laser_power_gain == 1.0
    assert config.converter_clamp_grad == "pwl"
    assert config.transfer_linearization == "endpoint"
    assert config.seed is None
    assert config.pd_input_clamp_min_w is None
    assert config.pd_input_clamp_max_w == 1e-5
    assert config.pd_input_clamp_mode == "soft"
    assert config.pd_input_soft_clamp_width_w == 1e-6
    assert config.pd_range_regularization_weight == 0.0
    assert config.pd_range_regularization_min_w is None
    assert config.pd_range_regularization_max_w is None
    assert config.fftconvnet_input_gain == 1.0


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


def test_model_arch_validation():
    for arch in ["resnet11", "resnet18"]:
        config = AppConfig(
            model_arch=arch,
            jtc_assume_nonnegative_input=True,
        )
        assert config.model_arch == arch
        assert config.jtc_assume_nonnegative_input is True

    with pytest.raises(ValueError, match="model_arch"):
        AppConfig(model_arch="invalid_model")


def test_yaml_loader_rejects_unknown_fields(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("conv_backend: jtc_emulation\nlegacy_backend: fourier\n")

    with pytest.raises(ValueError, match="Unknown config field"):
        load_app_config_from_yaml(str(cfg))


def test_training_checkpoint_config_validation():
    config = AppConfig(
        resume_checkpoint="runs/latest_checkpoint.pth",
        checkpoint_interval=2,
        checkpoint_time_interval_minutes=120,
        enable_ddp=False,
        jtc_max_shots=1024,
        enable_jtc_activation_checkpointing=True,
    )
    assert config.resume_checkpoint == "runs/latest_checkpoint.pth"
    assert config.checkpoint_interval == 2
    assert config.checkpoint_time_interval_minutes == 120.0
    assert config.enable_ddp is False
    assert config.jtc_max_shots == 1024
    assert config.enable_jtc_activation_checkpointing is True

    with pytest.raises(ValueError, match="checkpoint_interval"):
        AppConfig(checkpoint_interval=-1)
    with pytest.raises(ValueError, match="checkpoint_time_interval_minutes"):
        AppConfig(checkpoint_time_interval_minutes=-1)
    with pytest.raises(ValueError, match="jtc_max_shots"):
        AppConfig(jtc_max_shots=0)


def test_quant_bits_validation():
    assert AppConfig(dac_bits=None).dac_bits is None
    assert AppConfig(dac_bits=4).dac_bits == 4

    with pytest.raises(ValueError, match="dac_bits"):
        AppConfig(dac_bits=0)
    with pytest.raises(ValueError, match="adc_bits"):
        AppConfig(adc_bits=-1)
    with pytest.raises(ValueError, match="fourier_plane_bits"):
        AppConfig(fourier_plane_bits=0)


def test_converter_clamp_grad_validation():
    assert AppConfig(converter_clamp_grad="pwl").converter_clamp_grad == "pwl"
    assert AppConfig(converter_clamp_grad="MAD").converter_clamp_grad == "mad"

    with pytest.raises(ValueError, match="converter_clamp_grad"):
        AppConfig(converter_clamp_grad="ste")


def test_transfer_linearization_validation():
    assert (
        AppConfig(transfer_linearization="endpoint").transfer_linearization
        == "endpoint"
    )
    assert (
        AppConfig(transfer_linearization="LEAST_SQUARES").transfer_linearization
        == "least_squares"
    )
    assert (
        AppConfig(transfer_linearization="midpoint_ls").transfer_linearization
        == "midpoint_ls"
    )

    with pytest.raises(ValueError, match="transfer_linearization"):
        AppConfig(transfer_linearization="legacy_fit")


def test_laser_power_gain_validation():
    assert AppConfig(laser_power_gain="48").laser_power_gain == 48.0

    with pytest.raises(ValueError, match="laser_power_gain"):
        AppConfig(laser_power_gain=0)
    with pytest.raises(ValueError, match="laser_power_gain"):
        AppConfig(laser_power_gain=-1)


def test_fftconvnet_input_gain_validation():
    assert AppConfig(fftconvnet_input_gain="0.1").fftconvnet_input_gain == 0.1

    with pytest.raises(ValueError, match="fftconvnet_input_gain"):
        AppConfig(fftconvnet_input_gain=0)
    with pytest.raises(ValueError, match="fftconvnet_input_gain"):
        AppConfig(fftconvnet_input_gain=-1)


def test_pd_range_regularization_validation():
    config = AppConfig(
        pd_range_regularization_weight="0.25",
        pd_range_regularization_min_w="1e-6",
        pd_range_regularization_max_w="1e-5",
    )
    assert config.pd_range_regularization_weight == 0.25
    assert config.pd_range_regularization_min_w == 1e-6
    assert config.pd_range_regularization_max_w == 1e-5

    with pytest.raises(ValueError, match="pd_range_regularization_weight"):
        AppConfig(pd_range_regularization_weight=-1)
    with pytest.raises(ValueError, match="pd_range_regularization_max_w"):
        AppConfig(
            pd_range_regularization_min_w=1e-5,
            pd_range_regularization_max_w=1e-6,
        )


def test_pd_input_soft_clamp_validation():
    config = AppConfig(
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w="1e-5",
        pd_input_clamp_mode="SOFT",
        pd_input_soft_clamp_width_w="2e-6",
    )
    assert config.pd_input_clamp_min_w is None
    assert config.pd_input_clamp_max_w == 1e-5
    assert config.pd_input_clamp_mode == "soft"
    assert config.pd_input_soft_clamp_width_w == 2e-6

    assert AppConfig(pd_input_clamp_mode="hard").pd_input_clamp_mode == "hard"

    with pytest.raises(ValueError, match="pd_input_clamp_mode"):
        AppConfig(pd_input_clamp_mode="tangent")
    with pytest.raises(ValueError, match="pd_input_soft_clamp_width_w"):
        AppConfig(pd_input_soft_clamp_width_w=-1)
