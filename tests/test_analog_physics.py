"""Analog physical contract, independent solver equivalence, and remodulation."""

import math
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from analog_cases import analog_config, analog_layer

import onn_inference as inference
from onn_analog_shot import AnalogJTCShot
from onn_component import JTC
from onn_config import AppConfig
from onn_jtc_conv2d import JTCConv2d
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import converter_quantize_ste
from onn_remodulation import AnalogRemodulator, RemodulationSpec
from onn_shotplan import compute_contamination_profile


def _analytic_config(**overrides) -> AppConfig:
    base = dict(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_separation=8,
        jtc_total_field=48,
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=None,
        driver_distortion_polyfit_order=3,
        loss=0.96,
        pd_input_clamp_max_w=None,
        conv_backend="jtc_analytic",
        jtc_assume_nonnegative_input=True,
        jtc_output_gain_mode="per_shot",
        driver_distortion_data_path="./component_data/driver_sim_data.csv",
        pd_distortion_data_path="./component_data/pd_sim_data.csv",
        tia_distortion_data_path="./component_data/tia_sim_data.csv",
        mrm_amplitude_data_path="./component_data/mrm_amp_w_sim_data.csv",
        mrm_phase_data_path="./component_data/mrm_phase_sim_data.csv",
    )
    base.update(overrides)
    return AppConfig(**base)


@pytest.mark.parametrize("kernel_width,row_width", [(3, 14), (5, 18)])
def test_analytic_row_corr_matches_fft_reference(kernel_width, row_width):
    cfg = _analytic_config()
    torch.manual_seed(0)
    layer = JTCConv2d(
        2,
        3,
        kernel_width,
        padding=kernel_width // 2,
        config=cfg,
        max_jtc_shots=65536,
    )
    layer.eval()
    jtc = layer._rowwise_jtc_for_width(row_width, "jtc_analytic", torch.device("cpu"))
    a_in, b_in, p_const, c1, n = layer.shot.constants(jtc)

    torch.manual_seed(3)
    signal = torch.rand(6, row_width) * 0.9
    kernel = torch.rand(6, kernel_width) * 0.7

    # FFT-based reference of the agreed physics.
    f_s = layer.shot.input_field(signal, jtc)
    f_k = layer.shot.input_field(kernel, jtc)
    sep = jtc.jtc_separation
    plane = torch.zeros(6, jtc.jtc_total_field)
    plane[:, :kernel_width] = f_k
    plane[:, kernel_width + sep : kernel_width + sep + row_width] = f_s
    jps = complex_abs_squared(
        torch.fft.fft(torch.complex(plane, torch.zeros_like(plane)))
    ) * (jtc.loss / n)
    a_pd, b_pd = (float(v) for v in jtc.pd.ideal_coeffs)
    a_t, b_t = (float(v) for v in jtc.tia.ideal_coeffs)
    field2 = a_in * (a_t * (a_pd * jps + b_pd) + b_t) + b_in
    power2 = complex_abs_squared(
        torch.fft.fft(torch.complex(field2, torch.zeros_like(field2)))
    ) * (jtc.loss / n)
    v_full = a_t * (a_pd * power2 + b_pd) + b_t
    idx = jtc.compute_correlation_indices(torch.device("cpu"))
    valid_start = (kernel_width + 1) // 2
    n_valid = row_width - kernel_width + 1
    v_valid = (v_full[:, idx] - c1)[:, valid_start : valid_start + n_valid]
    gain = 1.0 / v_valid.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    reference = sqrt_nonnegative_with_finite_grad(
        converter_quantize_ste(gain * v_valid, cfg.adc_bits)
    ) / torch.sqrt(gain)

    out, _ = layer._analytic_paired_row_corr(
        signal, kernel, jtc, layer._analytic_gain_snapshot(signal), False
    )
    out_valid = out[:, valid_start : valid_start + n_valid]
    torch.testing.assert_close(out_valid, reference, rtol=1e-4, atol=1e-7)
    assert float((out_valid > 0).float().mean()) > 0.9


def test_analytic_layer_trains_and_grads_flow():
    cfg = _analytic_config()
    torch.manual_seed(0)
    layer = JTCConv2d(3, 4, 3, padding=1, config=cfg, max_jtc_shots=1 << 20)
    layer.train()
    x = torch.rand(2, 3, 8, 8, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert float((layer.weight.grad != 0).float().mean()) > 0.5


def test_analytic_rejects_distorted_configs():
    # Fourier-plane / lens / phase nonlinearities break the closed form
    for bad in (
        dict(pd_distortion_strength=1.0),
        dict(tia_distortion_strength=1.0),
        dict(mrm_phase_distortion_strength=1.0),
        dict(lens_distortion_strength=1.0),
    ):
        layer = JTCConv2d(2, 2, 3, padding=1, config=_analytic_config(**bad))
        with pytest.raises(ValueError, match="modeling contract"):
            layer(torch.rand(1, 2, 8, 8))
    # ...but INPUT-stage nonlinearity (driver/MRM amplitude) is supported
    cfg = _analytic_config(
        driver_distortion_strength=1.0, mrm_amplitude_distortion_strength=1.0
    )
    layer = JTCConv2d(2, 2, 3, padding=1, config=cfg)
    out = layer(torch.rand(1, 2, 8, 8))
    assert torch.isfinite(out).all()


def test_analytic_gain_modes_run():
    for mode in ("per_shot", "calibrated", "calibrate_freeze", "fixed"):
        cfg = _analytic_config(jtc_output_gain_mode=mode, jtc_output_gain=1e4)
        torch.manual_seed(0)
        layer = JTCConv2d(2, 2, 3, padding=1, config=cfg, max_jtc_shots=1 << 20)
        layer.train()
        out = layer(torch.rand(2, 2, 8, 8))
        assert torch.isfinite(out).all()
        if mode in ("calibrated", "calibrate_freeze"):
            assert bool(layer._analytic_gain_calibrated)
            layer.eval()
            out_eval = layer(torch.rand(2, 2, 8, 8))
            assert torch.isfinite(out_eval).all()


@pytest.mark.parametrize("stride", [(2, 2), (2, 1), (1, 2)])
def test_analytic_strided_equals_subsampled_stride1(stride):
    """With a fixed gain the readout is pointwise, so a strided conv must
    equal the stride-1 conv subsampled at the strided output positions."""
    cfg = _analytic_config(jtc_output_gain_mode="fixed", jtc_output_gain=8e3)
    torch.manual_seed(0)
    dense = JTCConv2d(
        3,
        4,
        3,
        stride=1,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    strided = JTCConv2d(
        3,
        4,
        3,
        stride=stride,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    with torch.no_grad():
        strided.weight.copy_(dense.weight)
        strided.bias.copy_(dense.bias)
    dense.eval()
    strided.eval()
    torch.manual_seed(5)
    x = torch.rand(2, 3, 12, 12)
    with torch.no_grad():
        full = dense(x)
        sub = strided(x)
    torch.testing.assert_close(sub, full[:, :, :: stride[0], :: stride[1]])


def test_analytic_signed_direct_matches_scheduler():
    """Signed direct fused path must agree with the shot-scheduler signed
    decomposition (same physics, different batching) within one ADC code."""
    cfg = _analytic_config(jtc_output_gain_mode="fixed", jtc_output_gain=8e3)
    torch.manual_seed(0)
    direct = JTCConv2d(
        3,
        4,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=False,
    )
    torch.manual_seed(5)
    x = torch.randn(2, 3, 10, 10)  # signed input, like noised diffusion data
    direct.eval()
    with torch.no_grad():
        out_direct = direct._analytic_conv2d(x)
        out_sched = (
            direct._rowwise_conv2d(x, "jtc_analytic")
            if direct._can_use_rowwise_conv2d(x)
            else direct._optical_conv2d(x, "jtc_analytic")
        )
    lsb = 1.0 / (2**cfg.adc_bits - 1)
    # readout is sqrt(code)/sqrt(g); bound the difference by one code step
    diff = (out_direct - out_sched).abs().max()
    assert float(diff) <= (lsb / 8e3) ** 0.5 * 4 + 1e-6, float(diff)
    # and gradients flow through the signed branches
    direct.train()
    x2 = torch.randn(1, 3, 10, 10, requires_grad=True)
    direct(x2).sum().backward()
    assert torch.isfinite(x2.grad).all()
    assert float((x2.grad != 0).float().mean()) > 0.3


def _fourier_config(**overrides) -> AppConfig:
    base = dict(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_separation=8,
        jtc_total_field=48,
        jtc_rowwise_geometry="config",
        dac_bits=4,
        adc_bits=6,
        fourier_plane_bits=None,
        driver_distortion_polyfit_order=3,
        loss=0.96,
        pd_input_clamp_max_w=None,
        conv_backend="jtc_analog_fourier",
        jtc_output_gain_mode="fixed",
        jtc_output_gain=8e3,
        driver_distortion_data_path="./component_data/driver_sim_data.csv",
        pd_distortion_data_path="./component_data/pd_sim_data.csv",
        tia_distortion_data_path="./component_data/tia_sim_data.csv",
        mrm_amplitude_data_path="./component_data/mrm_amp_w_sim_data.csv",
        mrm_phase_data_path="./component_data/mrm_phase_sim_data.csv",
    )
    base.update(overrides)
    return AppConfig(**base)


def _paired_layers(field, sep, gain_mode="fixed", nonneg=True):
    """Same weights, closed-form flag on vs off."""
    cfg_closed = _fourier_config(
        jtc_total_field=field,
        jtc_separation=sep,
        jtc_output_gain_mode=gain_mode,
    )
    cfg_fft = _fourier_config(
        jtc_total_field=field,
        jtc_separation=sep,
        jtc_output_gain_mode=gain_mode,
        jtc_fourier_closed_form=False,
    )
    torch.manual_seed(0)
    closed = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg_closed,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=nonneg,
    )
    torch.manual_seed(0)
    fft = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg_fft,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=nonneg,
    )
    with torch.no_grad():
        fft.weight.copy_(closed.weight)
        fft.bias.copy_(closed.bias)
    closed.eval()
    fft.eval()
    return closed, fft


def _code_step_bound(cfg_gain: float, adc_bits: int) -> float:
    # Outputs are sqrt(code)/sqrt(g); FFT roundoff can flip an ADC code at a
    # quantization boundary, moving an output by at most ~sqrt(lsb/g).
    lsb = 1.0 / (2**adc_bits - 1)
    return (lsb / cfg_gain) ** 0.5 * 4 + 1e-6


GEOMETRIES = [(64, 20), (34, 2), (16, 1)]


@pytest.mark.parametrize("field,sep", GEOMETRIES)
def test_closed_form_matches_fft_plane(field, sep):
    closed, fft = _paired_layers(field, sep)
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        out_closed = closed(x)
        out_fft = fft(x)
    diff = float((out_closed - out_fft).abs().max())
    assert diff <= _code_step_bound(8e3, 6), (field, sep, diff)


@pytest.mark.parametrize("field,sep", [(34, 2)])
def test_closed_form_matches_fft_per_shot_gain(field, sep):
    closed, fft = _paired_layers(field, sep, gain_mode="per_shot")
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        out_closed = closed(x)
        out_fft = fft(x)
    # per-shot AGC divides by the window max; roundoff perturbs the gain
    # as well as the codes. Bounds are 3x the measured worst across seeds
    # (max abs 4.0e-7).
    torch.testing.assert_close(out_closed, out_fft, rtol=1e-5, atol=1.5e-6)


def test_closed_form_matches_fft_signed_input():
    closed, fft = _paired_layers(34, 2, nonneg=False)
    torch.manual_seed(11)
    x = torch.randn(2, 2, 10, 10)
    with torch.no_grad():
        out_closed = closed(x)
        out_fft = fft(x)
    diff = float((out_closed - out_fft).abs().max())
    # four signed branches, each within a code step
    assert diff <= 4 * _code_step_bound(8e3, 6), diff


def test_closed_form_reduces_to_analytic_when_clean():
    """At a contamination-free geometry the autocorr/mirror maps must not
    touch the selected outputs: closed fourier == analytic exactly."""
    row_width, kw, field = 12, 3, 64
    sep = next(
        s
        for s in range(row_width - kw, field - row_width - kw + 1)
        if (
            lambda prof: prof[1] == row_width - kw + 1 and prof[2] == row_width - kw + 1
        )(compute_contamination_profile(row_width, kw, field, s))
    )
    cfg_fourier = _fourier_config(jtc_total_field=field, jtc_separation=sep)
    cfg_analytic = _fourier_config(
        jtc_total_field=field, jtc_separation=sep, conv_backend="jtc_analytic"
    )
    torch.manual_seed(0)
    four = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg_fourier,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    torch.manual_seed(0)
    ana = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg_analytic,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    with torch.no_grad():
        ana.weight.copy_(four.weight)
        ana.bias.copy_(four.bias)
    four.eval()
    ana.eval()
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        torch.testing.assert_close(four(x), ana(x), rtol=1e-6, atol=1e-7)


def test_closed_form_strided_equals_subsampled():
    cfg = _fourier_config(jtc_total_field=34, jtc_separation=2)
    torch.manual_seed(0)
    dense = JTCConv2d(
        2,
        3,
        3,
        stride=1,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    strided = JTCConv2d(
        2,
        3,
        3,
        stride=(2, 2),
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    with torch.no_grad():
        strided.weight.copy_(dense.weight)
        strided.bias.copy_(dense.bias)
    dense.eval()
    strided.eval()
    torch.manual_seed(5)
    x = torch.rand(2, 2, 12, 12)
    with torch.no_grad():
        full = dense(x)
        sub = strided(x)
    torch.testing.assert_close(sub, full[:, :, ::2, ::2])


def test_closed_form_grads_flow_contaminated():
    # per-shot AGC: a fixed gain tuned for clean geometry rails the ADC here
    # (the autocorr terms inflate v), and clamp STE zeroes railed gradients.
    cfg = _fourier_config(
        jtc_total_field=34, jtc_separation=2, jtc_output_gain_mode="per_shot"
    )
    torch.manual_seed(0)
    layer = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=False,
    )
    layer.train()
    x = torch.randn(1, 2, 10, 10, requires_grad=True)
    layer(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert float((x.grad != 0).float().mean()) > 0.3
    assert torch.isfinite(layer.weight.grad).all()
    assert float(layer.weight.grad.abs().sum()) > 0


def test_infeasible_geometry_raises():
    cfg = _fourier_config(jtc_total_field=14, jtc_separation=2)
    torch.manual_seed(0)
    layer = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    layer.eval()
    x = torch.rand(1, 2, 10, 10)  # row_width 12 + kw 3 + sep 2 > field 14
    with pytest.raises(ValueError, match="infeasible|too small"):
        with torch.no_grad():
            layer(x)


def test_closed_form_matches_fft_nonlinear_input():
    """Driver/MRM-amplitude distortion at full strength acts on the input
    fields only, so the closed form must still match the FFT plane exactly
    (the Fourier-plane chain stays affine)."""
    closed_cfg = _fourier_config(
        jtc_total_field=34,
        jtc_separation=2,
        driver_distortion_strength=1.0,
        mrm_amplitude_distortion_strength=1.0,
    )
    fft_cfg = _fourier_config(
        jtc_total_field=34,
        jtc_separation=2,
        driver_distortion_strength=1.0,
        mrm_amplitude_distortion_strength=1.0,
        jtc_fourier_closed_form=False,
    )
    torch.manual_seed(0)
    closed = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=closed_cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    torch.manual_seed(0)
    fft = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=fft_cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    with torch.no_grad():
        fft.weight.copy_(closed.weight)
        fft.bias.copy_(closed.bias)
    closed.eval()
    fft.eval()
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        out_closed = closed(x)
        out_fft = fft(x)
    diff = float((out_closed - out_fft).abs().max())
    assert diff <= _code_step_bound(8e3, 6), diff
    # and the nonlinear-input layer still trains
    closed.train()
    x2 = torch.rand(1, 2, 10, 10, requires_grad=True)
    closed(x2).sum().backward()
    assert torch.isfinite(x2.grad).all()


def test_full_transfer_path_reduces_to_affine():
    """With all strengths 0 and no stop-band, the full-transfer FFT path
    must equal the affine FFT path exactly."""
    cfg = _fourier_config(
        jtc_total_field=64, jtc_separation=8, jtc_fourier_closed_form=False
    )
    torch.manual_seed(0)
    lay = JTCConv2d(
        2,
        2,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    jtc = lay._rowwise_jtc_for_width(12, "jtc_analog_fourier", torch.device("cpu"))
    torch.manual_seed(3)
    s = torch.rand(4, 12) * 0.8
    k = torch.rand(4, 3) * 0.6
    ref, _ = lay.shot.affine_fft(s, k, jtc, lay._analytic_gain_snapshot(s), False)
    full, _ = lay.shot.complex_fft(s, k, jtc, lay._analytic_gain_snapshot(s), False)
    torch.testing.assert_close(full, ref, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(pd_distortion_strength=1.0, jtc_carrier_stopband_bins=3),
        # tia at 1.0 in this toy regime maps every output below the dark
        # level (compressive poly outside its fitted domain) — legitimately
        # zero gradient; 0.5 exercises the nonlinear path with live grads.
        dict(tia_distortion_strength=0.5, jtc_carrier_stopband_bins=3),
        dict(
            lens_distortion_strength=1.0,
            lens_legendre_order=3,
            lens_coefs=[0.4, 0.2, 0.1],
        ),
        dict(mrm_phase_distortion_strength=1.0),
        dict(jtc_carrier_stopband_bins=3),  # stop-band-only control
    ],
)
def test_full_transfer_path_trains(overrides):
    cfg = _fourier_config(
        jtc_total_field=64,
        jtc_separation=8,
        jtc_fourier_closed_form=False,
        jtc_output_gain_mode="per_shot",
        **overrides,
    )
    torch.manual_seed(0)
    lay = JTCConv2d(
        2,
        2,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    lay.train()
    x = torch.rand(1, 2, 10, 10, requires_grad=True)
    out = lay(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0
    assert torch.isfinite(lay.weight.grad).all()


def test_stopband_changes_outputs():
    """The Fourier-plane DC block AC-couples the readout — it must alter
    outputs even with affine transfers (it is physics, not a no-op)."""
    base = _fourier_config(
        jtc_total_field=64, jtc_separation=8, jtc_fourier_closed_form=False
    )
    blocked = _fourier_config(
        jtc_total_field=64,
        jtc_separation=8,
        jtc_fourier_closed_form=False,
        jtc_carrier_stopband_bins=3,
    )
    torch.manual_seed(0)
    a = JTCConv2d(2, 2, 3, padding=1, config=base, assume_nonnegative_input=True)
    torch.manual_seed(0)
    b = JTCConv2d(2, 2, 3, padding=1, config=blocked, assume_nonnegative_input=True)
    with torch.no_grad():
        b.weight.copy_(a.weight)
        b.bias.copy_(a.bias)
    a.eval()
    b.eval()
    torch.manual_seed(5)
    x = torch.rand(1, 2, 10, 10)
    with torch.no_grad():
        oa, ob = a(x), b(x)
    assert torch.isfinite(ob).all()
    assert float((oa - ob).abs().max()) > 1e-6


def _paired_full_transfer_layers(overrides, lag_gemm):
    cfg_a = _fourier_config(
        jtc_total_field=64,
        jtc_separation=8,
        jtc_fourier_closed_form=False,
        jtc_fourier_lag_gemm=lag_gemm,
        adc_bits=None,
        jtc_output_gain_mode="fixed",
        jtc_output_gain=2728.0,
        **overrides,
    )
    torch.manual_seed(0)
    lay = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg_a,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    lay.eval()
    return lay


FULL_TRANSFER_CASES = [
    dict(pd_distortion_strength=1.0, jtc_carrier_stopband_bins=3),
    dict(tia_distortion_strength=0.5, jtc_carrier_stopband_bins=3),
    dict(
        pd_distortion_strength=1.0,
        tia_distortion_strength=0.5,
        jtc_carrier_stopband_bins=3,
    ),
    dict(jtc_carrier_stopband_bins=3),
    dict(
        pd_distortion_strength=1.0,
        jtc_carrier_stopband_bins=3,
        driver_distortion_strength=1.0,
        mrm_amplitude_distortion_strength=1.0,
    ),
]


@pytest.mark.parametrize("overrides", FULL_TRANSFER_CASES)
def test_lag_gemm_matches_fft_plane_strict(overrides):
    """Lag-GEMM path vs c2c FFT plane path: same physics, different
    summation order. Continuous outputs (adc off) so no code-flip slack.

    Bounds are 3x the measured worst-case across all cases and seeds
    (max abs 1.7e-5): pure fp32 path noise — at fp64 the two formulations
    agree to 1e-13 relative, and both fp32 paths are equidistant from the
    fp64 truth (see test_fp32_paths_equidistant_from_fp64).
    """
    lag = _paired_full_transfer_layers(overrides, lag_gemm=True)
    fft = _paired_full_transfer_layers(overrides, lag_gemm=False)
    with torch.no_grad():
        fft.weight.copy_(lag.weight)
        fft.bias.copy_(lag.bias)
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        out_lag = lag(x)
        out_fft = fft(x)
    torch.testing.assert_close(out_lag, out_fft, rtol=1e-4, atol=5e-5)


def test_rfft_plane_matches_c2c_strict():
    """rfft variant vs c2c on the same FFT plane path: Hermitian symmetry
    is exact for real planes, so only FFT-algorithm rounding differs."""
    ov = dict(pd_distortion_strength=1.0, jtc_carrier_stopband_bins=3)
    lay = _paired_full_transfer_layers(ov, lag_gemm=False)
    jtc = lay._rowwise_jtc_for_width(12, "jtc_analog_fourier", torch.device("cpu"))
    torch.manual_seed(3)
    s = torch.rand(6, 12) * 0.8
    k = torch.rand(6, 3) * 0.6
    with torch.no_grad():
        out_c2c, _ = lay.shot.complex_fft(
            s, k, jtc, lay._analytic_gain_snapshot(s), False
        )
        out_rfft, _ = lay.shot.real_fft(
            s, k, jtc, lay._analytic_gain_snapshot(s), False
        )
    torch.testing.assert_close(out_rfft, out_c2c, rtol=1e-5, atol=1e-7)


def test_lag_gemm_grads_match_fft_plane():
    ov = dict(pd_distortion_strength=1.0, jtc_carrier_stopband_bins=3)
    lag = _paired_full_transfer_layers(ov, lag_gemm=True)
    fft = _paired_full_transfer_layers(ov, lag_gemm=False)
    with torch.no_grad():
        fft.weight.copy_(lag.weight)
        fft.bias.copy_(lag.bias)
    lag.train()
    fft.train()
    torch.manual_seed(7)
    x1 = torch.rand(1, 2, 10, 10, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    lag(x1).sum().backward()
    fft(x2).sum().backward()
    assert torch.isfinite(x1.grad).all()
    # STE clamp masks flip for elements at quantizer boundaries when the
    # forward paths differ by fp32 rounding, so a few gradient elements
    # differ by O(1) relative at tiny absolute size. Bounds are 3x the
    # measured worst-case across all transfer cases and seeds (max abs
    # 1.95e-4, no rtol residual beyond that); direction enforced by cosine.
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-3, atol=6e-4)
    torch.testing.assert_close(lag.weight.grad, fft.weight.grad, rtol=1e-3, atol=6e-4)
    cos = torch.nn.functional.cosine_similarity(
        x1.grad.flatten(), x2.grad.flatten(), dim=0
    )
    assert float(cos) > 0.9999, float(cos)


def test_lag_gemm_with_adc_one_code():
    """With the output ADC on, summation-order noise may flip a code at a
    quantization boundary; bound by one code step like the closed-form
    tests."""
    ov = dict(pd_distortion_strength=1.0, jtc_carrier_stopband_bins=3)
    cfgs = {}
    for lag_gemm in (True, False):
        cfg = _fourier_config(
            jtc_total_field=64,
            jtc_separation=8,
            jtc_fourier_closed_form=False,
            jtc_fourier_lag_gemm=lag_gemm,
            adc_bits=8,
            jtc_output_gain_mode="fixed",
            jtc_output_gain=2728.0,
            **ov,
        )
        torch.manual_seed(0)
        cfgs[lag_gemm] = JTCConv2d(
            2,
            3,
            3,
            padding=1,
            config=cfg,
            max_jtc_shots=1 << 20,
            assume_nonnegative_input=True,
        )
    with torch.no_grad():
        cfgs[False].weight.copy_(cfgs[True].weight)
        cfgs[False].bias.copy_(cfgs[True].bias)
    cfgs[True].eval()
    cfgs[False].eval()
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        diff = float((cfgs[True](x) - cfgs[False](x)).abs().max())
    lsb = 1.0 / (2**8 - 1)
    assert diff <= (lsb / 2728.0) ** 0.5 * 4 + 1e-6, diff


def test_fp32_paths_equidistant_from_fp64():
    """Neither fp32 formulation is biased: their distances to a true fp64
    reference must be comparable (the fp32-vs-fp64 delta is dominated by
    shared quantizer-boundary flips, not by either algorithm). Guards the
    conditioning of both paths — a regression that reintroduces
    cancellation (e.g. the lag-domain form's 28,000x one-sided error)
    fails this immediately."""
    ov = dict(pd_distortion_strength=1.0, jtc_carrier_stopband_bins=3)
    lag = _paired_full_transfer_layers(ov, lag_gemm=True)
    fft = _paired_full_transfer_layers(ov, lag_gemm=False)
    ref = _paired_full_transfer_layers(ov, lag_gemm=False)
    with torch.no_grad():
        for m in (fft, ref):
            m.weight.copy_(lag.weight)
            m.bias.copy_(lag.bias)
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    with torch.no_grad():
        ref(x)  # instantiate lazy rowwise JTC caches before converting
    ref = ref.double()
    with torch.no_grad():
        o_lag = lag(x).double()
        o_fft = fft(x).double()
        o_ref = ref(x.double())
    e_lag = float((o_lag - o_ref).norm())
    e_fft = float((o_fft - o_ref).norm())
    assert e_fft > 0 and 1 / 1.5 < e_lag / e_fft < 1.5, (e_lag, e_fft)


COMPLEX_CASES = {
    "lens": dict(
        lens_distortion_strength=1.0,
        lens_legendre_order=2,
        lens_coefs=[-2.7723, -1.266519],
    ),
    "phase": dict(mrm_phase_distortion_strength=1.0),
    "lens_phase": dict(
        lens_distortion_strength=1.0,
        lens_legendre_order=2,
        lens_coefs=[-2.7723, -1.266519],
        mrm_phase_distortion_strength=1.0,
    ),
}


def _probed_gain(ov) -> float:
    """Unrailed fixed gain for equivalence tests: railed outputs compare as
    identical constants, making the check vacuous (lens@1.0 rails at the
    clean-geometry gain — window energy drops ~100x)."""
    cfg = _fourier_config(
        jtc_total_field=64,
        jtc_separation=8,
        jtc_fourier_closed_form=False,
        adc_bits=None,
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_headroom=2.0,
        **ov,
    )
    torch.manual_seed(0)
    probe = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    probe.train()
    torch.manual_seed(7)
    probe(torch.rand(2, 2, 10, 10))
    return float(probe._analytic_gain_snapshot(torch.zeros(())))


@pytest.mark.parametrize("case", sorted(COMPLEX_CASES))
def test_lag_gemm_complex_matches_fft_plane(case):
    """Lens/MRM-phase on the extended lag path vs the c2c FFT plane path.

    Lens phases fold into the DFT matrices (F diag(e^{i phi}) F^-1 unwraps
    to a plane-domain screen — verified exact at fp64 to 1e-15); complex
    fields use the 4-GEMM form. Bounds are 3x the measured worst across
    cases and seeds (fwd 1.4e-6, grad 7.1e-7).
    """
    ov = COMPLEX_CASES[case]
    g_fix = _probed_gain(ov)

    def make(lag):
        cfg = _fourier_config(
            jtc_total_field=64,
            jtc_separation=8,
            jtc_fourier_closed_form=False,
            jtc_fourier_lag_gemm=lag,
            adc_bits=None,
            jtc_output_gain_mode="fixed",
            jtc_output_gain=g_fix,
            **ov,
        )
        torch.manual_seed(0)
        return JTCConv2d(
            2,
            3,
            3,
            padding=1,
            config=cfg,
            max_jtc_shots=1 << 20,
            assume_nonnegative_input=True,
        )

    lag, fft = make(True), make(False)
    with torch.no_grad():
        fft.weight.copy_(lag.weight)
        fft.bias.copy_(lag.bias)
    lag.train()
    fft.train()
    torch.manual_seed(7)
    x = torch.rand(2, 2, 10, 10)
    x1 = x.detach().clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    o1 = lag(x1)
    o1.sum().backward()
    o2 = fft(x2)
    o2.sum().backward()
    assert float(o1.abs().max()) > 0  # unrailed, non-degenerate
    torch.testing.assert_close(o1, o2, rtol=1e-4, atol=5e-6)
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-3, atol=3e-6)
    assert torch.isfinite(lag.weight.grad).all()


def test_gain_calibration_observes_under_checkpointing():
    """Regression: rowwise paths must update calibrate_freeze gains even
    with activation checkpointing on (pure chunks return observations
    for the layer to aggregate outside checkpoint recomputation).
    Without it, gains freeze at 1/headroom and stop-band-small signals
    quantize to all-zero ADC codes — the chance-accuracy flatline."""
    cfg = _fourier_config(
        jtc_total_field=64,
        jtc_separation=8,
        jtc_fourier_closed_form=False,
        jtc_carrier_stopband_bins=3,
        adc_bits=8,
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_headroom=2.0,
        enable_jtc_activation_checkpointing=True,
    )
    torch.manual_seed(0)
    lay = JTCConv2d(
        2,
        3,
        3,
        padding=1,
        config=cfg,
        max_jtc_shots=1 << 20,
        assume_nonnegative_input=True,
    )
    lay.train()
    x = torch.rand(2, 2, 10, 10, requires_grad=True)
    lay(x).sum().backward()
    assert int(lay._analytic_gain_obs_count) > 0, "gain never observed"
    g = float(lay._analytic_gain_snapshot(torch.zeros(())))
    assert abs(g - 0.5) > 1e-6, "gain still at 1/headroom init"


def test_tia_input_bias_restores_trainability():
    """TIA operating point (2026-09-02): the PD output at correlation-plane
    intensities (~0.068 at dark) lies below the characterized TIA input
    domain (0.105–0.438), so unbiased tia@1.0 collapses the layer to a
    threshold detector (~8 distinct outputs, zero gradient) — the cause of
    the flatlined tia arms. A PD->TIA bias into the linear region restores
    full code utilization; the offset-null stage makes it a no-op at
    strength 0."""
    torch.manual_seed(0)
    x_cal = [torch.rand(2, 4, 12, 12) for _ in range(3)]

    def probe(tia, bias):
        cfg = _fourier_config(
            jtc_total_field=128,
            jtc_separation=15,
            jtc_fourier_closed_form=False,
            adc_bits=8,
            jtc_output_gain_mode="calibrate_freeze",
            jtc_gain_headroom=2.0,
            jtc_carrier_stopband_bins=3,
            tia_distortion_strength=tia,
            tia_input_bias=bias,
        )
        torch.manual_seed(1)
        layer = JTCConv2d(
            4,
            4,
            3,
            padding=1,
            config=cfg,
            max_jtc_shots=1 << 16,
            assume_nonnegative_input=True,
        )
        layer.train()
        with torch.no_grad():
            for xb in x_cal:
                layer(xb)
        x = x_cal[0].clone().requires_grad_(True)
        out = layer(x)
        out.pow(2).mean().backward()
        gain = float(layer._analytic_gain_snapshot(torch.zeros(())))
        return out.detach(), float(layer.weight.grad.norm()), gain

    o_bad, g_bad, _ = probe(1.0, 0.0)
    o_ok, g_ok, _ = probe(1.0, 0.08)
    assert o_bad.unique().numel() < 64, "expected code collapse unbiased"
    assert o_ok.unique().numel() > 0.5 * o_ok.numel(), "bias must restore codes"
    assert g_ok > 10.0 * max(g_bad, 1e-12)

    # Strength 0: the bias is an affine offset the CDS removes exactly; in
    # fp32 the fringe (~5e-5) on the shifted pedestal rounds differently at
    # ADC code boundaries, so allow one code step (same bound as the
    # closed-vs-FFT test) and no more.
    o_clean0, _, g0 = probe(0.0, 0.0)
    o_clean_b, _, _ = probe(0.0, 0.08)
    torch.testing.assert_close(
        o_clean_b, o_clean0, rtol=0.0, atol=_code_step_bound(g0, 8)
    )
    # (the calibrated gain itself moves at roundoff level, rescaling every
    # output by ~1e-6, so count only genuine code flips)
    flips = float(
        ((o_clean_b - o_clean0).abs() > 0.5 * _code_step_bound(g0, 8)).float().mean()
    )
    assert flips < 0.15, f"too many boundary flips: {flips:.3f}"


def test_analog_sensitivity_has_one_adc_and_continuous_zero_strength():
    cfg = analog_config("affine_fft", adc_bits=8, jtc_output_gain=2728.0)
    x = torch.rand(2, 2, 6, 6)
    baseline = analog_layer(cfg).eval()
    perturbed = analog_layer(replace(cfg, pd_distortion_strength=1e-10)).eval()
    # Emulation's detector includes an ADC; no analog shot may use it.
    with patch.object(
        JTC, "_detector_power_transfer", side_effect=AssertionError("extra ADC")
    ):
        with torch.no_grad():
            a, b = baseline(x), perturbed(x)
    assert a.norm() > 0
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    signal, kernel = torch.rand(4, 8), torch.rand(4, 8)
    from onn_quantization import converter_quantize_ste

    for strength in [0.0, 1e-10, 1.0]:
        model = AnalogJTCShot(replace(cfg, pd_distortion_strength=strength)).eval()
        with patch(
            "onn_analog_shot.converter_quantize_ste", wraps=converter_quantize_ste
        ) as adc:
            model(signal, kernel)
        assert sum(call.args[1] == cfg.adc_bits for call in adc.call_args_list) == 1


def test_analog_metrics_honor_noise_and_stopband_and_replay_seed():
    cfg = analog_config(jtc_output_gain_mode="per_shot")
    noisy = replace(cfg, jtc_frontend_snr_db=20.0)
    first = inference.compute_snr_enob(noisy, "jtc_frontend_snr_db", seed=13)
    torch.randn(19)
    replay = inference.compute_snr_enob(noisy, "jtc_frontend_snr_db", seed=13)
    assert first == replay
    assert math.isfinite(first[0]) and first[0] < 40
    stopped = replace(cfg, jtc_fourier_closed_form=False, jtc_carrier_stopband_bins=3)
    assert math.isfinite(inference.compute_snr_between_configs(stopped, cfg)[0])
    assert inference.compute_snr_between_configs(cfg, cfg)[0] == math.inf
    with pytest.raises(ValueError, match="trained layer"):
        inference.compute_snr_between_configs(
            replace(cfg, jtc_output_gain_mode="calibrated"), cfg
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"jtc_carrier_stopband_bins": 3},
        {"pd_distortion_strength": 1.0, "jtc_carrier_stopband_bins": 3},
        {"mrm_phase_distortion_strength": 1.0},
    ],
)
def test_float64_lag_matches_fft_without_downcasts(overrides):
    cfg = analog_config("affine_fft", dac_bits=None, **overrides)
    lag = analog_layer(cfg).double().eval()
    fft = analog_layer(replace(cfg, jtc_fourier_lag_gemm=False)).double().eval()
    x = torch.rand(2, 2, 6, 6, dtype=torch.float64)
    with torch.no_grad():
        a, b = lag(x), fft(x)
    assert b.norm() > 0
    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("shape", ["groups", "dilation", "rectangle"])
@pytest.mark.parametrize("full", [False, True])
def test_general_fourier_convolution_forward_and_gradients(shape, full):
    cfg = analog_config(
        "pixel_dft" if full else "affine_fft", dac_bits=None, jtc_output_gain=1e-4
    )
    cfg = replace(cfg, jtc_shot_mapping="row" if shape == "rectangle" else "dot")
    args = dict(in_channels=2, out_channels=4, kernel_size=3, padding=1, bias=False)
    args.update(
        {
            "groups": dict(groups=2),
            "dilation": dict(dilation=2),
            "rectangle": dict(kernel_size=(3, 5)),
        }[shape]
    )
    torch.manual_seed(20)
    lag = JTCConv2d(**args, config=cfg).double()
    fft = JTCConv2d(**args, config=replace(cfg, jtc_fourier_lag_gemm=False)).double()
    fft.load_state_dict(lag.state_dict())
    x = torch.rand(1, 2, 8, 8, dtype=torch.float64)
    a, b = lag(x), fft(x)
    assert (
        a.shape
        == torch.nn.functional.conv2d(
            x,
            lag.weight,
            padding=1,
            dilation=args.get("dilation", 1),
            groups=args.get("groups", 1),
        ).shape
    )
    assert a.norm() > 0
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-11)
    torch.testing.assert_close(lag.weight.grad, fft.weight.grad, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize(
    "settings",
    [
        {"field_coefficients": [0.5, 0.01]},
        {"field_coefficients": [0.03, 0.5, 0.01]},
        {"voltage_min_v": 0.0, "voltage_max_v": 0.15},
        {"phase_coefficients": [0.3, 0.1]},
        {"field_coefficients": [0.03, 0.5, 0.01], "phase_coefficients": [0.3, 0.1]},
    ],
)
def test_remodulation_dft_matches_fft_and_gradients(settings):
    cfg = analog_config(
        "pixel_dft", dac_bits=None, jtc_remodulation=settings, jtc_output_gain=1e-6
    )
    dft = AnalogJTCShot(cfg).double()
    fft = AnalogJTCShot(replace(cfg, jtc_fourier_lag_gemm=False)).double()
    torch.manual_seed(7)
    s = torch.rand(2, 8, dtype=torch.float64)
    k = torch.rand(2, 8, dtype=torch.float64)
    outputs, grads = [], []
    for model in (dft, fft):
        x = s.clone().requires_grad_()
        out = model(x, k)
        outputs.append(out)
        grads.append(torch.autograd.grad(out.square().sum(), x)[0])
    assert outputs[0].norm() > 0 and grads[0].norm() > 0
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-7, atol=1e-12)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-6, atol=1e-12)
    assert not list(dft.parameters())


def test_affine_remodulation_closed_matches_plane():
    cfg = analog_config(
        jtc_remodulation={"field_coefficients": [0.5, 0.01]},
        dac_bits=None,
        jtc_output_gain=1e-6,
    )
    closed = analog_layer(cfg).double().eval()
    fft = analog_layer(replace(cfg, jtc_fourier_closed_form=False)).double().eval()
    x = torch.rand(2, 2, 6, 6, dtype=torch.float64)
    with torch.no_grad():
        a, b = closed(x), fft(x)
    assert a.norm() > 0
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-8)


def test_remodulation_characterization_and_trace(tmp_path):
    table = tmp_path / "remod.csv"
    table.write_text("voltage_v,field_sqrt_w,phase_rad\n0,0,0\n1,2,0.5\n2,3,1\n")
    spec = RemodulationSpec(
        transfer_csv=str(table), voltage_min_v=-0.5, voltage_max_v=2.5
    )
    remod = AnalogRemodulator(spec).double()
    x = torch.tensor([-1.0, 0.5, 1.5, 3.0], dtype=torch.float64, requires_grad=True)
    field, trace = remod(x, (1.0, 0.0), return_trace=True)
    expected = torch.tensor([-1.0, 1.0, 2.5, 3.5], dtype=torch.float64) * torch.exp(
        1j * torch.tensor([-0.25, 0.25, 0.75, 1.25], dtype=torch.float64)
    )
    torch.testing.assert_close(field, expected)
    assert trace["clipped"].tolist() == [True, False, False, True]
    assert trace["outside_characterized_domain"].tolist() == [True, False, False, True]
    assert len(remod.description()["characterization_sha256"]) == 64
    assert remod.description()["converter_stages"] == 0
    field.abs().sum().backward()
    assert torch.isfinite(x.grad).all()
    cfg = analog_config(
        "pixel_dft",
        jtc_remodulation={"transfer_csv": str(table)},
        dac_bits=None,
        jtc_output_gain=1e-6,
    )
    a, b = (
        AnalogJTCShot(cfg).double(),
        AnalogJTCShot(replace(cfg, jtc_fourier_lag_gemm=False)).double(),
    )
    s, k = torch.rand(2, 8, dtype=torch.float64), torch.rand(2, 8, dtype=torch.float64)
    torch.testing.assert_close(a(s, k), b(s, k), rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize(
    "settings", [{}, {"phase_coefficients": [0.2, 0.0]}, {"voltage_noise_rms_v": 0.001}]
)
def test_trace_has_one_adc_and_reproduces_noisy_production(settings):
    cfg = analog_config(
        "affine_fft",
        jtc_fourier_lag_gemm=False,
        jtc_remodulation=settings,
        jtc_frontend_snr_db=30.0,
        adc_bits=8,
    )
    model = AnalogJTCShot(cfg).double()
    s, k = torch.rand(3, 8, dtype=torch.float64), torch.rand(3, 8, dtype=torch.float64)
    with patch(
        "onn_analog_shot.converter_quantize_ste", wraps=converter_quantize_ste
    ) as convert:
        torch.manual_seed(12)
        trace = model.inspect(s, k)
    assert sum(call.args[1] == cfg.adc_bits for call in convert.call_args_list) == 1
    torch.manual_seed(12)
    out = model(s, k)
    torch.testing.assert_close(out, trace["readout"]["output"], rtol=1e-9, atol=1e-12)
    assert torch.isfinite(trace["remodulation"]["field_sqrt_w"]).all()
    if settings:
        assert not model.supports_rfft()


def test_noisy_remodulator_checkpoint_replay():
    cfg = analog_config("pixel_dft", jtc_remodulation={"voltage_noise_rms_v": 0.001})
    eager, ckpt = (
        analog_layer(cfg),
        analog_layer(replace(cfg, enable_jtc_activation_checkpointing=True)),
    )
    x = torch.rand(2, 2, 6, 6)
    results = []
    for model in (eager, ckpt):
        torch.manual_seed(15)
        y = model(x)
        y.square().sum().backward()
        results.append((y, model.weight.grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_reject_unsupported_remodulation_closed_form():
    with pytest.raises(ValueError, match="non-affine remodulation"):
        analog_layer(analog_config(jtc_remodulation={"voltage_max_v": 1.0}))(
            torch.rand(1, 2, 6, 6)
        )
    with pytest.raises(ValueError, match="phase_coefficients"):
        RemodulationSpec(phase_coefficients=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler regression")
def test_compiled_remodulator_phase_backward():
    # Complex scalar exp(1j*phase) previously failed Inductor backward lowering.
    remod = AnalogRemodulator(
        RemodulationSpec(
            field_coefficients=(0.03, 0.5, 0.01), phase_coefficients=(0.3, 0.1)
        )
    ).cuda()
    compiled = torch.compile(remod, dynamic=True)
    values = []
    for model in (remod, compiled):
        voltage = torch.linspace(-0.1, 0.4, 16, device="cuda", requires_grad=True)
        field = model(voltage, (1.0, 0.0))
        # Phase-sensitive loss so the comparison exercises d(phase)/dV.
        (field.real.square().mean() + field.imag.mean()).backward()
        values.append((field.detach(), voltage.grad))
    assert values[0][1].norm() > 0
    for eager, actual in zip(*values):
        torch.testing.assert_close(actual, eager, rtol=1e-5, atol=1e-7)
