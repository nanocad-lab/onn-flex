"""Component transfer fits, physical stage contracts, and detector arithmetic."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from onn_component import (
    JTC,
    _compute_linear_coeffs,
    _sqrt_clamped,
    apply_pd_input_guard,
    calculate_aic,
    get_coeffs,
    get_ideal_degree,
)
from onn_config import AppConfig, load_app_config_from_yaml
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import converter_quantize_ste, quantize_ste


@pytest.mark.parametrize("tag", ["mrm_amplitude", "pd"])
@pytest.mark.parametrize("linearization", ["endpoint", "least_squares"])
def test_component_plots_match_simulated_transfer_curves(tmp_path, tag, linearization):
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    from diagnostics.pretrain_tests import _sweep_and_plot
    from onn_component import MRM, PD
    from scripts.combined_plots import _plot_fit_on_axes

    config = AppConfig(transfer_linearization=linearization)
    if tag == "mrm_amplitude":
        component = MRM(config)
        path = config.mrm_amplitude_data_path
        ideal, fitted = component.ideal_field_coeffs, component.field_coeffs
    else:
        component = PD(config)
        path = config.pd_distortion_data_path
        ideal, fitted = component.ideal_coeffs, component.coeffs
    with patch("diagnostics.pretrain_tests._plot_fit") as draw:
        _sweep_and_plot(path, None, str(tmp_path), tag, linearization=linearization)
    x, samples, reference, polynomial = draw.call_args.args[:4]
    np.testing.assert_allclose(reference, np.polyval(ideal.numpy(), x), rtol=1e-5)
    np.testing.assert_allclose(
        polynomial, np.polyval(fitted.numpy(), x), rtol=2e-4, atol=1e-7
    )
    raw = pd.read_csv(path)["output"].to_numpy()
    np.testing.assert_allclose(
        samples, np.sqrt(np.maximum(raw, 0)) if tag == "mrm_amplitude" else raw
    )
    fig, ax = plt.subplots()
    try:
        assert _plot_fit_on_axes(ax, path, None, tag, linearization=linearization)
        for line, expected in zip(ax.lines, (reference, polynomial, samples)):
            np.testing.assert_allclose(line.get_ydata(), expected)
    finally:
        plt.close(fig)


def _test_config() -> AppConfig:
    return AppConfig(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=None,
        fourier_plane_bits=None,
        adc_bits=None,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        run_pretrain_tests=False,
    )


def _ideal_transfer_config(output_length: int | None = 8) -> AppConfig:
    return AppConfig(
        input_length=8,
        kernel_length=8,
        output_length=output_length,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=4,
        fourier_plane_bits=None,
        adc_bits=6,
        loss=1.0,
        scale_output="none",
        driver_distortion_strength=0.0,
        mrm_amplitude_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        run_pretrain_tests=False,
    )


def test_sqrt_nonnegative_with_finite_grad_matches_intended_surrogate():
    x = torch.tensor([-1.0, 0.0, 1e-13, 1e-6, 4.0], requires_grad=True)
    y = sqrt_nonnegative_with_finite_grad(x)

    torch.testing.assert_close(
        y,
        torch.tensor([0.0, 0.0, math.sqrt(1e-13), 1e-3, 2.0]),
        rtol=1e-6,
        atol=1e-8,
    )

    y.sum().backward()
    torch.testing.assert_close(
        x.grad,
        torch.tensor([0.0, 0.0, 0.0, 500.0, 0.25]),
        rtol=1e-6,
        atol=1e-8,
    )


def test_transfer_linearization_modes(tmp_path):
    path = tmp_path / "tf.csv"
    path.write_text("input,output\n0,0\n0.5,0.8\n1,1\n")

    endpoint = _compute_linear_coeffs(str(path), linearization="endpoint")
    least_squares = _compute_linear_coeffs(str(path), linearization="least_squares")
    midpoint = _compute_linear_coeffs(str(path), linearization="midpoint_ls")

    torch.testing.assert_close(
        torch.as_tensor(endpoint),
        torch.tensor([1.0, 0.0]),
    )
    # Regression slope for these points is 1.0, but the free fit floats the
    # intercept to the sample mean.
    torch.testing.assert_close(
        torch.as_tensor(least_squares),
        torch.tensor([1.0, 0.1]),
        rtol=1e-6,
        atol=1e-6,
    )
    # Midpoint-anchored LS keeps the regression slope but preserves the physical
    # midpoint of the measured endpoint range.
    torch.testing.assert_close(
        torch.as_tensor(midpoint),
        torch.tensor([1.0, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def _normalize_affine_response(x: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor):
    return (x - y0) / (y1 - y0).clamp_min(1e-12)


def _calibrate_input_field_for_equivalence(jtc: JTC, x: torch.Tensor) -> torch.Tensor:
    dtype = x.real.dtype if x.is_complex() else x.dtype
    device = x.device
    driver_a = jtc.driver.ideal_coeffs[0].to(device=device, dtype=dtype)
    driver_b = jtc.driver.ideal_coeffs[1].to(device=device, dtype=dtype)
    field_a = jtc.mrm.ideal_field_coeffs[0].to(device=device, dtype=dtype)
    field_b = jtc.mrm.ideal_field_coeffs[1].to(device=device, dtype=dtype)
    field_0 = field_a * driver_b + field_b
    field_1 = field_a * (driver_a + driver_b) + field_b
    magnitude = _normalize_affine_response(torch.abs(x), field_0, field_1)
    return torch.polar(magnitude.clamp_min(0.0), torch.angle(x))


def _calibrate_output_power_for_equivalence(jtc: JTC, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    device = x.device
    pd_a = jtc.pd.ideal_coeffs[0].to(device=device, dtype=dtype)
    pd_b = jtc.pd.ideal_coeffs[1].to(device=device, dtype=dtype)
    tia_a = jtc.tia.ideal_coeffs[0].to(device=device, dtype=dtype)
    tia_b = jtc.tia.ideal_coeffs[1].to(device=device, dtype=dtype)
    out_0 = tia_a * pd_b + tia_b
    out_1 = tia_a * (pd_a + pd_b) + tia_b
    return _normalize_affine_response(x, out_0, out_1)


def _test_only_output_transfer(jtc: JTC, x: torch.Tensor) -> torch.Tensor:
    pd_y = jtc.pd.ideal_coeffs[0] * x + jtc.pd.ideal_coeffs[1]
    return jtc.tia.ideal_coeffs[0] * pd_y + jtc.tia.ideal_coeffs[1]


def _test_calibrated_forward_paired(jtc: JTC, signal, kernel):
    signal_distorted = _calibrate_input_field_for_equivalence(
        jtc, jtc.input_distortion(signal)
    )
    kernel_distorted = _calibrate_input_field_for_equivalence(
        jtc, jtc.input_distortion(kernel)
    )
    input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)

    jps_field = jtc.fft_and_magnitude(input_plane) / math.sqrt(
        float(jtc.jtc_total_field)
    )
    jps = _calibrate_output_power_for_equivalence(
        jtc, _test_only_output_transfer(jtc, jps_field * jps_field)
    )
    jps = quantize_ste(jps, jtc.config.fourier_plane_bits)
    jps_distorted = _calibrate_input_field_for_equivalence(
        jtc, jtc.input_distortion(jps)
    )

    output_field = jtc.fft_and_magnitude(jps_distorted) / math.sqrt(
        float(jtc.jtc_total_field)
    )
    output_power = _calibrate_output_power_for_equivalence(
        jtc, _test_only_output_transfer(jtc, output_field * output_field)
    )
    output_plane = sqrt_nonnegative_with_finite_grad(output_power)
    return jtc.extract_correlation(output_plane)


def _ideal_forward_paired(jtc: JTC, signal, kernel):
    signal_complex = torch.complex(signal, torch.zeros_like(signal))
    kernel_complex = torch.complex(kernel, torch.zeros_like(kernel))
    input_plane = jtc.build_input_plane(signal_complex, kernel_complex)
    jps = complex_abs_squared(torch.fft.fft(input_plane)) / float(jtc.jtc_total_field)
    jps = quantize_ste(jps, jtc.config.fourier_plane_bits)
    jps_complex = torch.complex(jps, torch.zeros_like(jps))
    output_plane = torch.abs(torch.fft.fft(jps_complex)) / math.sqrt(
        float(jtc.jtc_total_field)
    )
    return jtc.extract_correlation(output_plane)


def test_forward_matches_explicit_pipeline():
    config = _test_config()
    jtc = JTC(config).eval()
    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8)
    kernel = torch.rand(3, 8)

    with torch.no_grad():
        output = jtc(signal, kernel)

        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size = signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        signal_reshaped = signal_full.reshape(batch_size, config.input_length)
        kernel_reshaped = kernel_full.reshape(batch_size, config.kernel_length)

        signal_distorted = jtc.input_distortion(signal_reshaped)
        kernel_distorted = jtc.input_distortion(kernel_reshaped)
        input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
        jps = jtc.output_distortion(
            jtc.fft_and_magnitude(input_plane) / math.sqrt(float(jtc.jtc_total_field))
        )
        jps = quantize_ste(jps, config.fourier_plane_bits)
        jps_distorted = jtc.input_distortion(jps)
        output_plane = jtc.final_detector_readout(jps_distorted)
        indices = jtc.compute_correlation_indices(output_plane.device)
        explicit = output_plane[..., indices].reshape_as(output)

    assert torch.allclose(output, explicit)


def test_same_length_indices_match_pytorch_same_window():
    for total_field in (48, 49):
        config = _test_config()
        config.jtc_total_field = total_field
        jtc = JTC(config).eval()

        same_start = (
            config.jtc_total_field // 2
            + config.jtc_separation
            + config.kernel_length // 2
            + 1
        )
        expected = torch.arange(same_start, same_start + config.output_length)
        expected = expected + (config.jtc_total_field + 1) // 2
        expected = expected % config.jtc_total_field

        assert torch.equal(jtc.compute_correlation_indices("cpu"), expected)


def test_native_fft_order_explicit_pipeline_with_lens_distortion():
    for total_field in (48, 49):
        config = AppConfig(
            input_length=8,
            kernel_length=8,
            output_length=8,
            jtc_separation=8,
            jtc_total_field=total_field,
            dac_bits=None,
            fourier_plane_bits=None,
            adc_bits=None,
            loss=1.0,
            scale_output="none",
            driver_distortion_strength=0.0,
            mrm_amplitude_distortion_strength=0.0,
            mrm_phase_distortion_strength=1.0,
            lens_distortion_strength=1.0,
            pd_distortion_strength=0.0,
            tia_distortion_strength=0.0,
            ler_std_dev=0.0,
            laser_rin_db=None,
            pd_noise_w=0.0,
            pd_input_clamp_min_w=None,
            pd_input_clamp_max_w=None,
            run_pretrain_tests=False,
        )
        jtc = JTC(config).eval()
        torch.manual_seed(7)
        signal = torch.rand(5, 8)
        kernel = torch.rand(5, 8)

        with torch.no_grad():
            actual = jtc.forward_paired(signal, kernel)

            signal_distorted = jtc.input_distortion(signal)
            kernel_distorted = jtc.input_distortion(kernel)
            input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
            jps = jtc.fft_and_magnitude(input_plane) / math.sqrt(
                float(jtc.jtc_total_field)
            )
            jps = jtc.output_distortion(jps)
            jps = quantize_ste(jps, config.fourier_plane_bits)
            jps_distorted = jtc.input_distortion(jps)
            output_plane = jtc.final_detector_readout(jps_distorted)
            expected = jtc.extract_correlation(output_plane)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_final_detector_slice_matches_full_plane_pointwise_transfer():
    config = AppConfig(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=None,
        fourier_plane_bits=None,
        adc_bits=None,
        loss=0.96,
        scale_output="none",
        driver_distortion_strength=0.0,
        mrm_amplitude_distortion_strength=0.0,
        mrm_phase_distortion_strength=1.0,
        lens_distortion_strength=1.0,
        pd_distortion_strength=1.0,
        tia_distortion_strength=1.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        run_pretrain_tests=False,
    )
    jtc = JTC(config).eval()
    assert jtc._can_slice_final_detector()

    torch.manual_seed(17)
    signal = torch.rand(4, 8)
    kernel = torch.rand(4, 8)

    with torch.no_grad():
        sliced = jtc.forward_paired(signal, kernel)

        signal_distorted = jtc.input_distortion(signal)
        kernel_distorted = jtc.input_distortion(kernel)
        input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
        jps = jtc.first_detector_readout(input_plane)
        jps = quantize_ste(jps, config.fourier_plane_bits)
        jps_distorted = jtc.input_distortion(jps)
        full_plane = jtc.final_detector_readout(jps_distorted)
        full = jtc.extract_correlation(full_plane)

    torch.testing.assert_close(sliced, full, rtol=1e-5, atol=1e-5)


def test_forward_is_deterministic_without_noise():
    jtc = JTC(_test_config()).eval()
    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8)
    kernel = torch.rand(3, 8)

    with torch.no_grad():
        output_1 = jtc(signal, kernel)
        output_2 = jtc(signal, kernel)

    assert torch.equal(output_1, output_2)


def test_input_transfer_matches_explicit_driver_mrm_chain():
    """The production transfer path should match the component module chain."""
    config = _ideal_transfer_config()
    config.mrm_phase_distortion_strength = 1.0
    jtc = JTC(config).eval()
    torch.manual_seed(123)
    x = torch.rand(4, 8)
    laser_scale = torch.ones(4, 1)

    with torch.no_grad():
        quant = converter_quantize_ste(x, config.dac_bits)
        expected = jtc.mrm(jtc.driver(quant), laser_scale=laser_scale)
        actual = jtc.input_distortion(x, laser_scale=laser_scale)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9)


def test_composed_input_transfer_matches_sequential_distorted_chain():
    """Composed input transfer should preserve distorted driver/MRM behavior."""
    config = _ideal_transfer_config()
    config.dac_bits = None
    config.driver_distortion_strength = 0.35
    config.mrm_amplitude_distortion_strength = 0.6
    config.mrm_phase_distortion_strength = 0.8
    config.laser_power_gain = 1.7
    jtc = JTC(config).eval()
    torch.manual_seed(321)
    x = torch.rand(3, 8)
    laser_scale = torch.full((3, 1), 1.25)

    with torch.no_grad():
        expected = jtc.mrm(
            jtc.driver(converter_quantize_ste(x, config.dac_bits)),
            laser_scale=laser_scale,
        )
        actual = jtc.input_distortion(x, laser_scale=laser_scale)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)


def test_output_transfer_matches_explicit_pd_tia_chain():
    """Detector readout should preserve the physical component order."""
    config = _ideal_transfer_config()
    jtc = JTC(config).eval()
    torch.manual_seed(456)
    field = torch.rand(4, 8)

    with torch.no_grad():
        power = (field * field) * jtc.loss
        expected = converter_quantize_ste(jtc.tia(jtc.pd(power)), config.adc_bits)
        actual = jtc.output_distortion(field)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pd_range_regularization_uses_detector_power_before_clamp():
    config = _ideal_transfer_config()
    config.pd_range_regularization_weight = 2.0
    jtc = JTC(config).train()
    x = torch.tensor([[0.0, 5.0e-6, 2.0e-5]], requires_grad=True)

    jtc.reset_pd_range_regularization()
    _ = jtc._pd_transfer_raw(x)
    reg = jtc.weighted_pd_range_regularization_loss()

    assert reg is not None
    assert reg.item() > 0
    reg.backward()
    assert x.grad is not None
    assert x.grad[0, 0] < 0
    assert x.grad[0, 1] == 0
    assert x.grad[0, 2] > 0


def test_pd_input_soft_clamp_is_upper_only_and_feathered():
    config = _ideal_transfer_config()
    config.pd_input_clamp_min_w = None
    config.pd_input_clamp_max_w = 1e-5
    config.pd_input_clamp_mode = "soft"
    config.pd_input_soft_clamp_width_w = 1e-6
    x = torch.tensor(
        [0.0, 5.0e-6, 1.0e-5, 1.1e-5, 1.0e-3],
        dtype=torch.float32,
        requires_grad=True,
    )

    guarded = apply_pd_input_guard(x, config)

    torch.testing.assert_close(guarded[:3], x[:3], rtol=0, atol=0)
    assert guarded[3] > config.pd_input_clamp_max_w
    assert guarded[3] < x[3]
    assert guarded[4] <= config.pd_input_clamp_max_w + (
        config.pd_input_soft_clamp_width_w * 1.001
    )

    guarded.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad[:3], torch.ones_like(x.grad[:3]))
    assert 0.0 < x.grad[3] < 1.0
    assert x.grad[4] < 1e-6


def test_mrm_amplitude_csv_loads_as_field_amplitude_transfer():
    """MRM amplitude CSV is converted to a voltage -> sqrt(W) transfer."""
    config = _ideal_transfer_config()
    config.dac_bits = None
    jtc = JTC(config).eval()
    x = torch.linspace(0.0, 1.0, steps=8).reshape(1, 8)

    with torch.no_grad():
        driver_y = jtc.driver(x)
        field_expected = (
            jtc.mrm.ideal_field_coeffs[0] * driver_y + jtc.mrm.ideal_field_coeffs[1]
        )
        field = jtc.mrm(driver_y, laser_scale=torch.ones(1, 1))

    torch.testing.assert_close(
        field.abs(),
        field_expected.clamp_min(0.0),
        rtol=1e-6,
        atol=1e-9,
    )


def test_input_distortion_returns_physical_field_amplitude_from_mrm_transfer():
    x = torch.linspace(0.0, 1.0, steps=8).reshape(1, 8)

    config = _ideal_transfer_config()
    config.dac_bits = None
    config.pd_distortion_strength = 1.0
    jtc = JTC(config).eval()

    with torch.no_grad():
        driver_y = jtc.driver(x)
        field_expected = (
            jtc.mrm.ideal_field_coeffs[0] * driver_y + jtc.mrm.ideal_field_coeffs[1]
        )
        field = jtc.input_distortion(x, laser_scale=torch.ones(1, 1))

    torch.testing.assert_close(
        field.abs(),
        field_expected.clamp_min(0.0),
        rtol=1e-6,
        atol=1e-9,
    )


def test_clean_emulation_matches_ideal_with_test_only_calibration():
    config = _ideal_transfer_config()
    config.dac_bits = None
    config.adc_bits = None
    jtc = JTC(config).eval()
    torch.manual_seed(19)
    signal = torch.rand(4, 8) * 0.2
    kernel = torch.rand(4, 8) * 0.2

    with torch.no_grad():
        raw = jtc.forward_paired(signal, kernel)
        calibrated = _test_calibrated_forward_paired(jtc, signal, kernel)
        ideal = _ideal_forward_paired(jtc, signal, kernel)

    assert raw.shape == ideal.shape
    assert torch.isfinite(raw).all()
    torch.testing.assert_close(calibrated, ideal, rtol=1e-4, atol=1e-6)


def test_identity_transfer_bypass_is_removed():
    jtc = JTC(_ideal_transfer_config()).eval()

    assert not hasattr(jtc, "_has_ideal_input_transfer")
    assert not hasattr(jtc, "_has_ideal_output_transfer")
    assert not hasattr(jtc, "_use_fused_transfer")


def test_laser_rin_intensity_scale_maps_to_field_sqrt():
    """laser_rin_db is intensity RIN, so complex field amplitude gets sqrt(scale)."""
    x = torch.linspace(0.0, 1.0, steps=8).reshape(1, 8)
    intensity_scale = torch.full((1, 1), 4.0)

    config = _ideal_transfer_config()
    config.dac_bits = None
    config.adc_bits = None
    jtc = JTC(config).eval()

    with torch.no_grad():
        base = jtc.input_distortion(x, laser_scale=torch.ones_like(intensity_scale))
        scaled = jtc.input_distortion(x, laser_scale=intensity_scale)

    torch.testing.assert_close(
        scaled.abs(),
        base.abs() * torch.sqrt(intensity_scale),
        rtol=1e-6,
        atol=1e-6,
    )


def test_laser_power_gain_maps_to_field_sqrt():
    """laser_power_gain is a power-domain source gain on field amplitude."""
    x = torch.linspace(0.0, 1.0, steps=8).reshape(1, 8)

    base_config = _ideal_transfer_config()
    base_config.dac_bits = None
    base_config.adc_bits = None

    gain_config = _ideal_transfer_config()
    gain_config.dac_bits = None
    gain_config.adc_bits = None
    gain_config.laser_power_gain = 9.0

    base_jtc = JTC(base_config).eval()
    gain_jtc = JTC(gain_config).eval()
    laser_scale = torch.ones(1, 1)

    with torch.no_grad():
        base = base_jtc.input_distortion(x, laser_scale=laser_scale)
        gained = gain_jtc.input_distortion(x, laser_scale=laser_scale)

    torch.testing.assert_close(
        gained.abs(),
        base.abs() * math.sqrt(gain_config.laser_power_gain),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gradients_flow_through_jtc():
    config = _test_config()
    # With the physically normalized final FFT, the tiny random fixture sits
    # below the detector threshold at unit laser power. Use a non-saturated
    # operating point so this test covers gradient flow through the full path.
    config.laser_power_gain = 16.0
    jtc = JTC(config).train()
    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8, requires_grad=True)
    kernel = torch.rand(3, 8, requires_grad=True)

    loss = jtc(signal, kernel).sum()
    loss.backward()

    assert signal.grad is not None
    assert kernel.grad is not None
    assert signal.grad.abs().max() > 0
    assert kernel.grad.abs().max() > 0


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.sum((y_true - y_pred) ** 2)
    centered = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - residual / centered)


def _csv_xy(path: str, transform=None) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    x = data["input"].to_numpy(dtype=np.float64)
    y = data["output"].to_numpy(dtype=np.float64)
    if transform is not None:
        y = transform(y)
    return x, y


def _aic_selected_degree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
) -> int:
    best_degree: int | None = None
    best_aic = float("inf")
    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        pred = np.polyval(coeffs, x)
        r2 = _r2_score(y, pred)
        aic = calculate_aic(y, pred, degree + 1)
        if aic < best_aic:
            best_aic = aic
            best_degree = degree
        if r2 > 0.9995:
            return degree
    return best_degree if best_degree is not None else 1


def test_endpoint_ideal_reference_preserves_measured_endpoint_range():
    cases = [
        ("component_data/driver_sim_data.csv", None),
        ("component_data/pd_sim_data.csv", None),
        ("component_data/tia_sim_data.csv", None),
        ("component_data/mrm_amp_w_sim_data.csv", _sqrt_clamped),
        ("component_data/mrm_phase_sim_data.csv", None),
    ]

    for path, transform in cases:
        x, y = _csv_xy(path, transform)
        coeffs = _compute_linear_coeffs(
            path,
            output_transform=transform,
            linearization="endpoint",
        )
        order = np.argsort(x)
        x0, x1 = x[order[0]], x[order[-1]]
        y0, y1 = y[order[0]], y[order[-1]]

        assert math.isclose(
            float(np.polyval(coeffs, x0)),
            float(y0),
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        assert math.isclose(
            float(np.polyval(coeffs, x1)),
            float(y1),
            rel_tol=1e-6,
            abs_tol=1e-7,
        )


def test_transfer_degree_selection_uses_r2_target_then_global_aic():
    cases = [
        ("component_data/driver_sim_data.csv", None),
        ("component_data/pd_sim_data.csv", None),
        ("component_data/tia_sim_data.csv", None),
        ("component_data/mrm_amp_w_sim_data.csv", _sqrt_clamped),
        ("component_data/mrm_phase_sim_data.csv", None),
    ]

    for path, transform in cases:
        x, y = _csv_xy(path, transform)
        assert get_ideal_degree(
            path, output_transform=transform
        ) == _aic_selected_degree(
            x,
            y,
        )


def test_configured_transfer_polynomials_have_expected_fit_quality():
    config = load_app_config_from_yaml("configs/config_ideal.yaml")
    jtc = JTC(config)
    cases = [
        ("driver", "component_data/driver_sim_data.csv", None, jtc.driver.degree),
        ("pd", "component_data/pd_sim_data.csv", None, jtc.pd.degree),
        ("tia", "component_data/tia_sim_data.csv", None, jtc.tia.degree),
        (
            "mrm_amplitude_field",
            "component_data/mrm_amp_w_sim_data.csv",
            _sqrt_clamped,
            jtc.mrm.field_degree,
        ),
        (
            "mrm_phase",
            "component_data/mrm_phase_sim_data.csv",
            None,
            jtc.mrm.phase_degree,
        ),
    ]
    assert jtc.driver.degree == 3

    for name, path, transform, degree in cases:
        x, y = _csv_xy(path, transform)
        coeffs = get_coeffs(path, degree, output_transform=transform)
        r2 = _r2_score(y, np.polyval(coeffs, x))

        assert r2 >= 0.999, f"{name} degree={degree} r2={r2:.6f}"


def test_pd_tia_0011_interaction_quantifies_second_order_coupling():
    base = AppConfig(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=None,
        fourier_plane_bits=None,
        adc_bits=None,
        transfer_linearization="endpoint",
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=1e-5,
        pd_input_clamp_mode="soft",
        pd_input_soft_clamp_width_w=1e-6,
        pd_noise_w=0.0,
        laser_rin_db=None,
        ler_std_dev=0.0,
        run_pretrain_tests=False,
    )
    states = {
        "00": (0.0, 0.0),
        "10": (1.0, 0.0),
        "01": (0.0, 1.0),
        "11": (1.0, 1.0),
    }
    x = torch.linspace(1e-6, 1e-5, 512).reshape(1, -1)
    curves: dict[str, torch.Tensor] = {}
    for state, (pd_strength, tia_strength) in states.items():
        cfg = AppConfig(
            **{
                **base.__dict__,
                "pd_distortion_strength": pd_strength,
                "tia_distortion_strength": tia_strength,
            }
        )
        jtc = JTC(cfg).eval()
        with torch.no_grad():
            curves[state] = jtc._tia_transfer_raw(jtc._pd_transfer_raw(x)).reshape(-1)

    interaction = curves["11"] - curves["10"] - curves["01"] + curves["00"]
    y00_range = (curves["00"].max() - curves["00"].min()).clamp_min(1e-12)
    mean_abs_rel = interaction.abs().mean() / y00_range
    max_abs_rel = interaction.abs().max() / y00_range

    assert mean_abs_rel > 0.05
    assert max_abs_rel > 0.10
    assert max_abs_rel < 0.20


def test_jtc_stage_tensors_can_keep_full_probe_batch():
    config = AppConfig(
        input_length=4,
        kernel_length=4,
        jtc_separation=4,
        jtc_total_field=24,
        dac_bits=None,
        fourier_plane_bits=None,
        adc_bits=None,
        run_pretrain_tests=False,
    )
    jtc = JTC(config).eval()
    signal = torch.rand(3, 4) * 0.2
    kernel = torch.rand(3, 4) * 0.2

    single = jtc.compute_stage_tensors(
        signal,
        kernel,
        stages=("output_tia",),
    )
    full = jtc.compute_stage_tensors(
        signal,
        kernel,
        stages=("output_tia",),
        sample_index=None,
    )

    assert single["output_tia"].shape == (24,)
    assert full["output_tia"].shape == (3, 24)
    assert torch.allclose(single["output_tia"], full["output_tia"][0])
