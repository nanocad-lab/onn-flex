"""Readout, calibration, spectrum reuse, checkpoint policies, and CUDA fusion."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from analog_cases import ALPHA_CASES, PATHS, analog_config, analog_layer

from onn_analog_shot import AnalogJTCShot
from onn_config import AppConfig
from onn_jtc_conv2d import JTCConv2d, configure_jtc_runtime
from onn_math import sqrt_nonnegative_with_finite_grad
from onn_quantization import converter_quantize_ste
from onn_train import build_model


@pytest.mark.parametrize("backend", ["jtc_analytic", "jtc_analog_fourier"])
def test_closed_contract_rejects_stopband(backend):
    layer = analog_layer(
        analog_config(conv_backend=backend, jtc_carrier_stopband_bins=3)
    )
    with pytest.raises(
        ValueError, match="carrier suppression.*jtc_fourier_closed_form=false"
    ):
        layer(torch.rand(1, 2, 6, 6))


@pytest.mark.parametrize("path", PATHS)
def test_frontend_noise_reaches_every_readout(path):
    quiet = analog_layer(analog_config(path)).eval()
    noisy = analog_layer(analog_config(path, jtc_frontend_snr_db=20)).eval()
    torch.manual_seed(7)
    x = torch.rand(2, 2, 6, 6)
    with torch.no_grad():
        reference = quiet(x)
        torch.manual_seed(8)
        first = noisy(x)
        second = noisy(x)
        torch.manual_seed(8)
        replay = noisy(x)
    assert float(reference.abs().max()) > 0
    assert float((first - reference).abs().max()) > 1e-5
    assert float((second - first).abs().max()) > 1e-5
    torch.testing.assert_close(first, replay, rtol=0, atol=0)


@pytest.mark.parametrize("gain", [2.0, 20.0])
def test_noise_is_adc_referred_and_excluded_from_gain_observation(gain):
    layer = analog_layer(analog_config(jtc_frontend_snr_db=40))
    torch.manual_seed(1)
    v = torch.full((50000,), 0.25 / gain)
    out, observed = layer.shot.readout(v, torch.tensor(gain), True)
    noise = out.square() * gain - 0.25
    assert abs(float(noise.mean())) < 2e-4
    assert float(noise.std()) == pytest.approx(0.01, rel=0.02)
    assert float(observed) == pytest.approx(0.25 / gain)


def test_fixed_gain_with_half_precision_scheduler_inputs():
    cfg = analog_config("pixel_dft", jtc_output_gain=1e5)
    layer = analog_layer(cfg).eval()
    torch.manual_seed(7)
    x = torch.rand(1, 2, 6, 6).half()
    with torch.no_grad():
        actual = layer(x)
        reference = layer(x.float()).half()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [None, 8])
def test_affine_bias_preserves_signal_and_matches_closed_form(bits):
    cfg = analog_config(adc_bits=bits)
    closed = analog_layer(cfg).eval()
    fft = analog_layer(
        replace(cfg, jtc_fourier_closed_form=False, tia_input_bias=0.08)
    ).eval()
    torch.manual_seed(7)
    x = torch.rand(2, 2, 6, 6)
    with torch.no_grad():
        reference, biased = closed(x), fft(x)
    assert float(reference.norm()) > 1e-3
    assert float(biased.norm()) > 1e-3
    # ADC boundary flips have the existing per-shot sqrt(code/gain) bound;
    # the unquantized comparison also prevents this being a vacuous test.
    atol = 1e-6 if bits is None else 4 * (1 / (255 * cfg.jtc_output_gain)) ** 0.5
    torch.testing.assert_close(biased, reference, rtol=1e-4, atol=atol)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("nonnegative", [True, False])
def test_calibration_is_per_forward_and_independent_of_execution(path, nonnegative):
    cfg = analog_config(
        path,
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_freeze_batches=2,
        jtc_gain_headroom=2,
    )
    layers = [
        analog_layer(
            replace(cfg, enable_jtc_activation_checkpointing=checkpointed),
            shots=shots,
            nonnegative=nonnegative,
            out_channels=10,
        )
        for checkpointed, shots in [
            (False, 65536),
            (False, 64),
            (True, 65536),
            (True, 64),
        ]
    ]
    # Exercise channels and kernel rows that the former tiny probe missed.
    with torch.no_grad():
        for layer in layers:
            layer.weight[-2:, :, -1] *= 4
    torch.manual_seed(7)
    batches = [torch.rand(2, 2, 6, 6) for _ in range(3)]
    if not nonnegative:
        batches = [2 * x - 1 for x in batches]
    batches[2] *= 5  # Challenge the frozen gain with a larger input range.
    frozen = None
    for step, batch in enumerate(batches):
        results = []
        for layer in layers:
            layer.zero_grad(set_to_none=True)
            x = batch.clone().requires_grad_()
            out = layer(x)
            observed = layer._analytic_gain_running_max.clone()
            assert int(layer._analytic_gain_obs_count) == min(step + 1, 2)
            out.square().sum().backward()
            # Recompute must use the entry gain and observation flag even
            # when this forward has just frozen calibration.
            assert int(layer._analytic_gain_obs_count) == min(step + 1, 2)
            torch.testing.assert_close(layer._analytic_gain_running_max, observed)
            results.append((out.detach(), x.grad, layer.weight.grad.clone(), observed))
        for result in results[1:]:
            for value, reference in zip(result, results[0]):
                torch.testing.assert_close(value, reference, rtol=2e-4, atol=1e-6)
        if step == 1:
            frozen = results[0][-1]
            assert float(results[0][2].abs().max()) > 0
        if step == 2:
            torch.testing.assert_close(results[0][-1], frozen, rtol=0, atol=0)
    for layer in layers:
        layer.eval()
        with torch.no_grad():
            layer(batches[0])
        assert int(layer._analytic_gain_obs_count) == 2
        layer.train()
        layer.reset_gain_calibration()
        layer(batches[0])
        assert int(layer._analytic_gain_obs_count) == 1


@pytest.mark.parametrize("path", ["pixel_dft", "rfft", "complex_fft"])
def test_noisy_checkpoint_replays_forward_and_gradients(path):
    cfg = analog_config(path, jtc_frontend_snr_db=30, adc_bits=8)
    eager = analog_layer(cfg)
    checkpointed = analog_layer(replace(cfg, enable_jtc_activation_checkpointing=True))
    torch.manual_seed(7)
    x1 = torch.rand(2, 2, 6, 6, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    torch.manual_seed(8)
    y1 = eager(x1)
    y1.square().sum().backward()
    torch.manual_seed(8)
    y2 = checkpointed(x2)
    y2.square().sum().backward()
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    torch.testing.assert_close(x1.grad, x2.grad)
    torch.testing.assert_close(eager.weight.grad, checkpointed.weight.grad)


@pytest.mark.parametrize("scheduler", ["rows", "grouped_dots"])
def test_calibration_in_fallback_schedulers(scheduler):
    cfg = analog_config(
        "analytic",
        jtc_rowwise_geometry="auto",
        jtc_output_gain_mode="calibrated",
        jtc_shot_mapping="dot" if scheduler == "grouped_dots" else "row",
    )
    layers = []
    for checkpointed, shots in [(False, 65536), (False, 32), (True, 32)]:
        torch.manual_seed(42)
        layers.append(
            JTCConv2d(
                2,
                4,
                3,
                padding=1,
                bias=False,
                groups=2 if scheduler == "grouped_dots" else 1,
                config=replace(cfg, enable_jtc_activation_checkpointing=checkpointed),
                max_jtc_shots=shots,
            )
        )
    if scheduler == "rows":
        assert layers[0]._rowwise_rows_per_shot(8, 6) == 1
    torch.manual_seed(7)
    batch = torch.rand(2, 2, 6, 6)
    for step in range(2):
        results = []
        for layer in layers:
            layer.zero_grad(set_to_none=True)
            x = batch.clone().requires_grad_()
            if scheduler == "rows":
                out = layer._rowwise_conv2d(x, "jtc_analytic")
            else:
                out = layer(x)
            out.square().sum().backward()
            assert int(layer._analytic_gain_obs_count) == step + 1
            results.append(
                (
                    out.detach(),
                    x.grad,
                    layer.weight.grad.clone(),
                    layer._analytic_gain_running_max.clone(),
                )
            )
        for result in results[1:]:
            for value, reference in zip(result, results[0]):
                torch.testing.assert_close(value, reference, rtol=2e-4, atol=1e-6)


def _distributed_calibration_worker(rank, rendezvous):
    torch.set_num_threads(1)
    cfg = analog_config(
        "pixel_dft", jtc_output_gain_mode="calibrate_freeze", jtc_gain_freeze_batches=2
    )
    reference = analog_layer(cfg)
    layer = analog_layer(cfg)
    torch.manual_seed(13)
    batch = torch.rand(2, 2, 6, 6)
    batch[0] *= 0.1  # ranks must see different local maxima
    with torch.no_grad():
        reference(batch)
        expected = reference(batch)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        with torch.no_grad():
            layer(batch[rank : rank + 1])
            actual = layer(batch[rank : rank + 1])
        torch.testing.assert_close(actual, expected[rank : rank + 1])
        torch.testing.assert_close(
            layer._analytic_gain_running_max, reference._analytic_gain_running_max
        )
        assert int(layer._analytic_gain_obs_count) == 2
    finally:
        dist.destroy_process_group()


def test_distributed_calibration_uses_global_batch_maximum(tmp_path):
    mp.spawn(
        _distributed_calibration_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def paired_reference(shot, signal, kernel, jtc, gain, need_vmax, nonnegative):
    """Expand raw apertures first and evaluate each complete joint-plane shot."""
    positions, inputs, width = signal.shape
    outputs, kw = kernel.shape[1:]
    signal = (
        signal[:, :, None].expand(positions, inputs, outputs, width).reshape(-1, width)
    )
    kernel = kernel[None].expand(positions, inputs, outputs, kw).reshape(-1, kw)
    kp, kn = kernel.clamp_min(0), (-kernel).clamp_min(0)
    sp = signal if nonnegative else signal.clamp_min(0)
    pp, vpp = shot.evaluate(sp, kp, jtc, gain, need_vmax)
    if nonnegative:
        pn, vpn = shot.evaluate(sp, kn, jtc, gain, need_vmax)
        return pp - pn, torch.maximum(vpp, vpn)
    sn = (-signal).clamp_min(0)
    nn, vnn = shot.evaluate(sn, kn, jtc, gain, need_vmax)
    pn, vpn = shot.evaluate(sp, kn, jtc, gain, need_vmax)
    np, vnp = shot.evaluate(sn, kp, jtc, gain, need_vmax)
    return (pp + nn) - (pn + np), torch.maximum(
        torch.maximum(vpp, vnn), torch.maximum(vpn, vnp)
    )


@pytest.mark.parametrize("case", ALPHA_CASES)
@pytest.mark.parametrize("dft", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nonnegative", [True, False])
def test_alpha_zero_and_one_match_joint_plane(case, dft, dtype, nonnegative):
    cfg = analog_config(
        "affine_fft",
        input_length=8,
        kernel_length=3,
        output_length=8,
        jtc_fourier_lag_gemm=dft,
        dac_bits=None,
        adc_bits=None,
        jtc_output_gain=1e-4,
        # Keep every alpha setting in its usable analog range.
        tia_input_bias=0.08,
        jtc_carrier_stopband_bins=0 if case == "ideal" else 3,
        **ALPHA_CASES[case],
    )
    shot = AnalogJTCShot(cfg).to(dtype=dtype)
    torch.manual_seed(51)
    x = torch.rand(3, 2, 8, dtype=dtype)
    if not nonnegative:
        x = 2 * x - 1
    w = torch.rand(2, 4, 3, dtype=dtype) * 2 - 1
    jtc, gain = shot._prepare(x.reshape(-1, 8), w.reshape(-1, 3)[:6], None, None)
    layer = JTCConv2d(2, 4, 3, config=cfg, assume_nonnegative_input=nonnegative).to(
        dtype=dtype
    )
    layer.shot = shot
    probe = torch.randn(3, 4, 8, dtype=dtype)
    values = []
    for reused in (False, True):
        signal = x.clone().requires_grad_()
        kernel = w.clone().requires_grad_()
        args = signal, kernel, jtc, gain, True, nonnegative
        if reused:
            with patch.object(layer, "_rowwise_jtc_for_width", return_value=jtc):
                out, vmax = layer._rowwise_chunk_contrib(
                    signal,
                    kernel,
                    cfg.conv_backend,
                    0,
                    8,
                    torch.arange(8),
                    8,
                    gain,
                    True,
                )
        else:
            out, vmax = paired_reference(shot, *args)
            out = out.reshape(3, 2, 4, 8).sum(dim=1)
        grads = torch.autograd.grad((out * probe).sum(), (signal, kernel))
        assert out.norm() > 0 and all(g.norm() > 0 for g in grads)
        values.append((out, vmax, *grads))
    rtol, atol = (2e-4, 1e-6) if dtype == torch.float32 else (2e-8, 1e-10)
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", ["ideal", "combined"])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("shots", [12, 65536])
def test_reuse_recomputes_after_updates_and_preserves_calibration(
    case, checkpointed, shots
):
    cfg = analog_config(
        "affine_fft",
        enable_jtc_activation_checkpointing=checkpointed,
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_freeze_batches=2,
        adc_bits=8,
        tia_input_bias=0.08,
        jtc_carrier_stopband_bins=0 if case == "ideal" else 3,
        **ALPHA_CASES[case],
    )
    reference = analog_layer(cfg, shots=shots, nonnegative=False)
    reused = analog_layer(cfg, shots=shots, nonnegative=False)

    # Same downstream scheduler and gain controller, independent paired shot math.
    def explicit(*args):
        return paired_reference(reference.shot, *args)

    with patch.object(reference.shot, "_compiled_rowwise", side_effect=explicit):
        for step in range(3):
            torch.manual_seed(91 + step)
            batch = torch.rand(2, 2, 6, 6) * 2 - 1
            values = []
            for layer in (reference, reused):
                layer.zero_grad(set_to_none=True)
                x = batch.clone().requires_grad_()
                out = layer(x)
                out.square().sum().backward()
                assert int(layer._analytic_gain_obs_count) == min(2, step + 1)
                values.append(
                    (
                        out,
                        x.grad,
                        layer.weight.grad.clone(),
                        layer._analytic_gain_running_max.clone(),
                    )
                )
                # Use the SAME update in both runs, isolating stale-spectrum bugs
                # from the ordinary amplification of floating point differences.
                with torch.no_grad():
                    layer.weight.add_(0.025 * (step + 1))
            for actual, expected in zip(values[1], values[0]):
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize(
    "settings",
    [
        {"jtc_remodulation": {"field_coefficients": [0.03, 0.5, 0.01]}},
        {"jtc_remodulation": {"phase_coefficients": [0.3, 0.1]}},
        {"jtc_remodulation": {"voltage_noise_rms_v": 0.001}, "jtc_frontend_snr_db": 30},
    ],
)
def test_remodulation_and_noise_remain_per_shot(settings):
    cfg = analog_config("pixel_dft", **settings)
    reference = analog_layer(cfg, nonnegative=False)
    reused = analog_layer(
        replace(cfg, enable_jtc_activation_checkpointing=True), nonnegative=False
    )
    torch.manual_seed(3)
    batch = torch.rand(2, 2, 6, 6) * 2 - 1
    values = []

    def explicit(*args):
        return paired_reference(reference.shot, *args)

    with patch.object(reference.shot, "_compiled_rowwise", side_effect=explicit):
        for layer in (reference, reused):
            x = batch.clone().requires_grad_()
            torch.manual_seed(5)
            out = layer(x)
            out.square().sum().backward()
            values.append((out, x.grad, layer.weight.grad))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-6)


def test_modulation_work_scales_with_unique_apertures():
    shot = AnalogJTCShot(analog_config("pixel_dft", input_length=8, kernel_length=3))
    signal, kernel = torch.rand(5, 2, 8), torch.randn(2, 4, 3)
    jtc, gain = shot._prepare(
        signal.reshape(-1, 8)[:8], kernel.reshape(-1, 3), None, None
    )
    with patch.object(
        jtc, "input_distortion", wraps=jtc.input_distortion
    ) as modulation:
        shot.rowwise(signal, kernel, jtc, gain, False, True)
    apertures = sum(
        call.args[0].numel() // call.args[0].shape[-1]
        for call in modulation.call_args_list
    )
    assert apertures == 2 * (2 * 4) + 5 * 2
    # Explicit signed shots previously modulated both apertures for each pair.
    assert apertures < 2 * 2 * (5 * 2 * 4)


@pytest.mark.parametrize("case", ALPHA_CASES)
@pytest.mark.parametrize("dft", [True, False])
def test_fft_routes_preserve_joint_transform(case, dft):
    cfg = analog_config("affine_fft", jtc_fourier_lag_gemm=dft, **ALPHA_CASES[case])
    layer = analog_layer(cfg).eval()
    expected_reuse = dft and layer.shot.needs_full_transfers()
    with patch.object(
        layer.shot, "_compiled_rowwise", wraps=layer.shot.rowwise
    ) as reused:
        with torch.no_grad():
            layer(torch.rand(1, 2, 6, 6))
    assert bool(reused.call_count) == expected_reuse


def test_single_position_does_not_expand_a_kernel_spectrum_bank():
    layer = analog_layer(analog_config("pixel_dft"), shots=6).eval()
    with patch.object(
        layer.shot, "_compiled_rowwise", side_effect=AssertionError("no kernel reuse")
    ):
        with torch.no_grad():
            result = layer(torch.rand(1, 2, 6, 6))
    assert result.shape == (1, 3, 6, 6)


@pytest.mark.parametrize("case", ALPHA_CASES)
@pytest.mark.parametrize("nonnegative", [False, True])
def test_spectra_checkpoint_matches_recompute(case, nonnegative):
    cfg = analog_config(
        "affine_fft",
        enable_jtc_activation_checkpointing=True,
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_freeze_batches=2,
        adc_bits=8,
        tia_input_bias=0.08,
        jtc_carrier_stopband_bins=0 if case == "ideal" else 3,
        **ALPHA_CASES[case],
    )
    layers = [
        analog_layer(
            replace(cfg, jtc_checkpoint_policy=policy), nonnegative=nonnegative
        )
        for policy in ("recompute", "spectra")
    ]
    for step in range(3):
        torch.manual_seed(50 + step)
        batch = torch.rand(2, 2, 6, 6)
        if not nonnegative:
            batch = 2 * batch - 1
        values = []
        for layer in layers:
            x = batch.clone().requires_grad_()
            y = layer(x)
            grads = torch.autograd.grad(y.square().sum(), (x, layer.weight))
            values.append((y, layer._analytic_gain_running_max.clone(), *grads))
            with torch.no_grad():
                layer.weight.add_(0.025)
        for actual, expected in zip(values[1], values[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["recompute", "spectra"])
def test_only_aperture_preparation_is_retained(policy):
    cfg = analog_config(
        "pixel_dft",
        enable_jtc_activation_checkpointing=True,
        jtc_checkpoint_policy=policy,
        jtc_output_gain=1e-4,
    )
    layer = analog_layer(cfg)
    with patch.object(
        layer.shot, "prepare_rowwise", wraps=layer.shot.prepare_rowwise
    ) as prep:
        y = layer(torch.rand(2, 2, 6, 6, requires_grad=True))
        forward_calls = prep.call_count
        assert forward_calls > 0
        y.sum().backward()
        assert prep.call_count == forward_calls * (2 if policy == "recompute" else 1)
    assert layer.weight.grad.norm() > 0


@pytest.mark.parametrize("shots", [6, 24, 65536])
def test_noisy_spectra_checkpoint_replays_without_extra_rng_draws(shots):
    cfg = analog_config(
        "pixel_dft",
        jtc_output_gain=1e-4,
        jtc_frontend_snr_db=45,
        jtc_remodulation={"voltage_noise_rms_v": 0.0001},
    )
    layers = [
        analog_layer(cfg, shots=shots, nonnegative=False),
        analog_layer(
            replace(
                cfg,
                enable_jtc_activation_checkpointing=True,
                jtc_checkpoint_policy="spectra",
            ),
            shots=shots,
            nonnegative=False,
        ),
    ]
    batch = torch.rand(2, 2, 6, 6) * 2 - 1
    values = []
    for layer in layers:
        torch.manual_seed(71)
        x = batch.clone().requires_grad_()
        y = layer(x)
        grads = torch.autograd.grad(y.sum(), (x, layer.weight))
        values.append((y, *grads, torch.get_rng_state()))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"conv1": {"max_shots": 0}},
        {"conv1": {"max_shots": 1.5}},
        {"conv1": {"max_shots": True}},
        {"conv1": {"checkpoint_policy": "all"}},
        {"conv1": {"shots": 12}},
        {"conv1": {}},
        [],
    ],
)
def test_invalid_overrides_rejected(overrides):
    with pytest.raises(ValueError):
        AppConfig(jtc_runtime_overrides=overrides)


def test_named_overrides_apply_without_changing_shotplan():
    cfg = analog_config("pixel_dft", model_arch="resnet18")
    model = build_model(cfg)
    layer = model.layer1[0].conv1
    before = layer.shot_plan((2, 64, 8, 8))
    configure_jtc_runtime(
        model,
        replace(
            cfg,
            jtc_runtime_overrides={
                "layer1.0.conv1": {"max_shots": 1234, "checkpoint_policy": "spectra"},
            },
        ),
    )
    assert layer.max_jtc_shots == 1234 and layer.checkpoint_policy == "spectra"
    assert model.layer1[0].conv2.max_jtc_shots == cfg.jtc_max_shots
    assert layer.shot_plan((2, 64, 8, 8)) == before
    with pytest.raises(ValueError, match="unknown JTC layers"):
        configure_jtc_runtime(
            model,
            replace(
                cfg,
                jtc_runtime_overrides={
                    "layer1.misspelled": {"max_shots": 1234},
                },
            ),
        )


@pytest.mark.parametrize("field", [31, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"pd_distortion_strength": 1.0},
        {"tia_distortion_strength": 1.0, "tia_input_bias": 0.08},
        {"tia_distortion_strength": 0.5, "tia_input_bias": 0.08},
        {"jtc_remodulation": {"field_coefficients": [0.03, 0.5, 0.01]}},
    ],
)
def test_folded_dft_all_bins_match_full_dft_and_fft(field, dtype, settings):
    # Extract the entire circle: catches lost DC/Nyquist and odd-length weights,
    # even though ordinary convolution usually selects only correlation lags.
    cfg = analog_config(
        "pixel_dft",
        input_length=8,
        kernel_length=3,
        output_length=field,
        jtc_total_field=field,
        jtc_separation=2,
        dac_bits=None,
        adc_bits=None,
        jtc_output_gain=1e-8,
        **settings,
    )
    folded = AnalogJTCShot(cfg).to(dtype=dtype)
    full = AnalogJTCShot(cfg).to(dtype=dtype)
    fft = AnalogJTCShot(replace(cfg, jtc_fourier_lag_gemm=False)).to(dtype=dtype)
    torch.manual_seed(29)
    signal = torch.rand(3, 8, dtype=dtype)
    kernel = torch.rand(3, 3, dtype=dtype)
    probe = torch.randn(3, field, dtype=dtype)
    values = []
    # Disabling the symmetry predicate also forces the independent complex FFT
    # reference; it leaves the physical configuration unchanged.
    with (
        patch.object(full, "supports_rfft", return_value=False),
        patch.object(fft, "supports_rfft", return_value=False),
    ):
        # The independent full-circle FFT check uses double precision. In
        # FP32, both DFT versions share a detector dark-subtraction floor near
        # zero bins; sqrt magnifies it relative to the FFT reference. Compare
        # old/full and folded arithmetic directly there, without widening the
        # existing FFT tolerances for normal correlation windows.
        shots = (folded, full, fft) if dtype == torch.float64 else (folded, full)
        for shot in shots:
            x = signal.clone().requires_grad_()
            w = kernel.clone().requires_grad_()
            out = shot(x, w)
            gradients = torch.autograd.grad((out * probe).sum(), (x, w))
            assert out.norm() > 0 and all(g.norm() > 0 for g in gradients)
            values.append((out, *gradients))
    for other in values[1:]:
        for index, (actual, expected) in enumerate(zip(values[0], other)):
            if dtype == torch.float64:
                rtol, atol = (1e-7, 1e-10) if index == 0 else (1e-6, 1e-9)
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            elif index == 0:
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-8)
            else:
                relative_l2 = (actual - expected).norm() / expected.norm()
                assert relative_l2 < 3e-6


@pytest.mark.parametrize(
    "settings",
    [
        {"mrm_phase_distortion_strength": 1.0},
        {"lens_distortion_strength": 1.0},
        {"jtc_remodulation": {"phase_coefficients": [0.3, 0.0]}},
        {"jtc_remodulation": {"voltage_noise_rms_v": 0.001}},
        {"pd_range_regularization_weight": 0.1},
    ],
)
def test_asymmetric_or_pixel_regularized_shots_keep_full_dft(settings):
    cfg = analog_config("pixel_dft", **settings)
    shot = AnalogJTCShot(cfg)
    signal = torch.rand(2, 8)
    kernel = torch.rand(2, 8)
    jtc, _ = shot._prepare(signal, kernel, None, None)
    maps = shot.dft_maps(8, jtc, signal.device)
    assert maps[0].shape[-1] == cfg.jtc_total_field
    assert maps[4] is not None


@pytest.mark.parametrize("field", [31, 32])
def test_symmetric_maps_retain_only_unique_bins(field):
    cfg = analog_config("pixel_dft", jtc_total_field=field)
    shot = AnalogJTCShot(cfg)
    signal = torch.rand(2, 8)
    jtc, _ = shot._prepare(signal, signal, None, None)
    maps = shot.dft_maps(8, jtc, signal.device)
    assert maps[0].shape[-1] == field // 2 + 1
    assert maps[3].shape[0] == field // 2 + 1
    assert maps[4] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree", [1, 3, 10])
def test_horner_matches_eager(dtype, degree):
    from onn_fused import horner

    torch.manual_seed(29)
    # Noncontiguous inputs and a nonconstant upstream gradient.
    raw = torch.rand(71, 19, device="cuda", dtype=dtype).T
    coeffs = torch.randn(degree + 1, device="cuda", dtype=dtype)
    probe = torch.randn_like(raw)
    values = []
    for fused in (False, True):
        x = raw.detach().requires_grad_()
        if fused:
            y = horner(x, coeffs)
        else:
            y = coeffs[0] * x + coeffs[1]
            for c in coeffs[2:]:
                y = y * x + c
        (dx,) = torch.autograd.grad(y, x, probe)
        values.append((y, dx))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("bits", [None, 4, 8, 12])
@pytest.mark.parametrize("mode", ["pwl", "mad"])
def test_adc_matches_eager_at_codes_rails_and_zero(bits, mode):
    from onn_fused import adc_readout

    torch.manual_seed(35)
    levels = 255 if bits is None else 2**bits - 1
    edges = (torch.arange(levels, device="cuda") + 0.5) / levels
    raw = torch.cat(
        (
            torch.randn(1701, device="cuda"),
            edges,
            torch.nextafter(edges, torch.full_like(edges, float("inf"))),
            torch.nextafter(edges, torch.full_like(edges, -float("inf"))),
            torch.tensor([-1.0, 0.0, 1e-14, 1e-12, 1.0, 2.0], device="cuda"),
        )
    )
    gain = torch.tensor(137.0, device="cuda")
    probe = torch.randn_like(raw)
    values = []
    for fused in (False, True):
        x = raw.detach().requires_grad_()
        if fused:
            out = adc_readout(x, gain, 0 if bits is None else levels, mode == "mad")
        else:
            y = converter_quantize_ste(x, bits, mode)
            out = sqrt_nonnegative_with_finite_grad(y) / gain.sqrt()
        (dx,) = torch.autograd.grad(out, x, probe)
        values.append((out, dx))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("case", ALPHA_CASES)
@pytest.mark.parametrize("policy", ["off", "recompute", "spectra"])
def test_fusion_alpha_zero_and_one_outputs_gains_gradients(case, policy):
    cfg = analog_config(
        "affine_fft",
        enable_jtc_activation_checkpointing=policy != "off",
        jtc_checkpoint_policy="recompute" if policy == "off" else policy,
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_freeze_batches=2,
        adc_bits=8,
        tia_input_bias=0.08,
        jtc_carrier_stopband_bins=0 if case == "ideal" else 3,
        **ALPHA_CASES[case],
    )
    layers = [
        analog_layer(replace(cfg, jtc_fuse_pointwise=fused), nonnegative=False).cuda()
        for fused in (False, True)
    ]
    for step in range(3):
        torch.manual_seed(80 + step)
        batch = torch.rand(2, 2, 6, 6, device="cuda") * 2 - 1
        values = []
        for layer in layers:
            layer.zero_grad(set_to_none=True)
            x = batch.detach().requires_grad_()
            out = layer(x)
            grads = torch.autograd.grad(out.square().sum(), (x, layer.weight))
            values.append((out, layer._analytic_gain_running_max.clone(), *grads))
            with torch.no_grad():
                layer.weight.add_(0.0125)
        for index, (actual, expected) in enumerate(zip(values[1], values[0])):
            if index < 2:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
def test_fused_ops_compile_with_exact_arithmetic():
    from onn_fused import adc_readout, affine_plane, horner

    x = torch.rand(17, 19, device="cuda", requires_grad=True)
    coefficients = torch.tensor([0.2, 0.3, 0.1], device="cuda")
    gain = torch.tensor(7.0, device="cuda")
    for op, args in (
        (horner, (x, coefficients)),
        (adc_readout, (x, gain, 255, False)),
        (affine_plane, (x, [937.2, 0.04, 3.1, -0.03, 0.08, 0.01, 0.001])),
    ):
        expected = op(*args)
        actual = torch.compile(op, fullgraph=True, dynamic=True)(*args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (g_actual,) = torch.autograd.grad(actual.sum(), x)
        (g_expected,) = torch.autograd.grad(expected.sum(), x)
        torch.testing.assert_close(g_actual, g_expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("bias", [0.0, 0.08])
def test_affine_plane_preserves_each_rounding_and_gradient(bias):
    from onn_fused import affine_plane

    torch.manual_seed(31)
    raw = torch.rand(35, 41, device="cuda") * 1e-5
    coefficients = [937.2, 0.04, 3.1, -0.03, bias, 0.01, 0.001]
    probe = torch.randn_like(raw)
    values = []
    for fused in (False, True):
        x = raw.clone().requires_grad_()
        if fused:
            y = affine_plane(x, coefficients)
        else:
            ap, bp, at, bt, bias, am, bm = coefficients
            y = am * (at * (ap * x + bp + bias) + bt) + bm
        (dx,) = torch.autograd.grad(y, x, probe)
        values.append((y, dx))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("path", ["affine_fft", "pixel_dft", "rfft"])
@pytest.mark.parametrize("amp", [False, True])
def test_fusion_preserves_noise_and_amp_checkpoint_replay(path, amp):
    cfg = analog_config(
        path,
        jtc_output_gain=1e-4,
        jtc_frontend_snr_db=35,
        enable_jtc_activation_checkpointing=True,
        jtc_checkpoint_policy="spectra",
    )
    if path != "affine_fft":
        cfg = replace(
            cfg,
            tia_distortion_strength=1.0,
            tia_input_bias=0.08,
            jtc_remodulation={"voltage_noise_rms_v": 0.0001},
        )
    layers = [
        analog_layer(replace(cfg, jtc_fuse_pointwise=fused), nonnegative=False).cuda()
        for fused in (False, True)
    ]
    batch = torch.rand(2, 2, 6, 6, device="cuda") * 2 - 1
    values = []
    for layer in layers:
        torch.manual_seed(23)
        x = batch.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            y = layer(x)
        y.float().sum().backward()
        grads = x.grad, layer.weight.grad
        values.append((y, *grads, torch.cuda.get_rng_state()))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 2, 3, 19), (7, 3, 5, 33), (31, 8, 16, 65)])
def test_intensity_preserves_broadcast_gradients(dtype, shape):
    from onn_fused import spectrum_intensity

    torch.manual_seed(37)
    positions, inputs, outputs, bins = shape
    sr = torch.randn(positions, inputs, bins, device="cuda", dtype=torch.float64)
    si = torch.randn_like(sr)
    kr = torch.randn(inputs, outputs, bins, device="cuda", dtype=torch.float64)
    ki = torch.randn_like(kr)
    mask = (torch.rand(bins, device="cuda") > 0.2).float()
    probe = torch.randn(positions * inputs * outputs, bins, device="cuda", dtype=dtype)
    results = []
    for fused in (False, True):
        args = [x.detach().requires_grad_() for x in (sr, si, kr, ki)]
        if fused:
            y = spectrum_intensity(*args, mask, dtype)
        else:
            re = args[0][:, :, None] + args[2][None]
            im = args[1][:, :, None] + args[3][None]
            y = ((re * re + im * im).to(dtype) * mask).reshape(-1, bins)
        results.append((y, *torch.autograd.grad(y, args, probe)))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
@pytest.mark.parametrize("degrees", [(1, 1), (1, 10), (3, 1), (3, 10)])
@pytest.mark.parametrize("bias", [0.0, 0.08])
@pytest.mark.parametrize("remodulation", [None, (0.37, 0.02)])
def test_detector_chain_preserves_intermediate_rounding(degrees, bias, remodulation):
    from onn_fused import detector_chain

    torch.manual_seed(41)
    raw = torch.randn(33, 41, device="cuda").T * 0.1
    pd = torch.randn(degrees[0] + 1, device="cuda") * 0.1
    tia = torch.randn(degrees[1] + 1, device="cuda") * 0.1
    probe = torch.randn_like(raw)
    values = []
    for fused in (False, True):
        x = raw.detach().requires_grad_()
        if fused:
            y = detector_chain(x, pd, tia, bias, remodulation)
        else:
            y = x
            for coefficients, offset in ((pd, 0.0), (tia, bias)):
                z = y + offset if offset else y
                y = coefficients[0] * z + coefficients[1]
                for c in coefficients[2:]:
                    y = y * z + c
            if remodulation is not None:
                y = remodulation[0] * y + remodulation[1]
        values.append((y, *torch.autograd.grad(y, x, probe)))
    for actual, expected in zip(values[1], values[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")
def test_chain_and_intensity_compile_preserve_outputs_and_gradients():
    from onn_fused import detector_chain, spectrum_intensity

    torch.manual_seed(43)
    sr = torch.randn(3, 2, 17, device="cuda", dtype=torch.float64, requires_grad=True)
    kr = torch.randn(2, 5, 17, device="cuda", dtype=torch.float64, requires_grad=True)
    mask = torch.ones(17, device="cuda")
    pd = torch.tensor([0.21, 0.17, 0.04], device="cuda")
    tia = torch.tensor([0.13, 0.12, 0.11, 0.1], device="cuda")

    def call(s, k):
        power = spectrum_intensity(s, s, k, k, mask, torch.float32)
        return detector_chain(power, pd, tia, 0.08, (0.37, 0.02))

    a = call(sr, kr)
    b = torch.compile(call, fullgraph=True)(sr, kr)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for actual, expected in zip(
        torch.autograd.grad(b.sum(), (sr, kr)), torch.autograd.grad(a.sum(), (sr, kr))
    ):
        # Inductor changes the FP64 broadcast-reduction tree; pointwise
        # rounding stays exact. Require a tight nonvacuous reduction bound.
        assert expected.norm() > 0
        assert (actual - expected).norm() / expected.norm() < 1e-12

    x = torch.randn(17, 19, device="cuda", requires_grad=True)

    def chain(z):
        return detector_chain(z, pd, tia, 0.08, (0.37, 0.02))

    a = chain(x)
    b = torch.compile(chain, fullgraph=True)(x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(a.sum(), x)[0],
        torch.autograd.grad(b.sum(), x)[0],
        rtol=0,
        atol=0,
    )
