import sys
from pathlib import Path
import math

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from onn_component import JTC
from onn_quantization import quantize_ste
from onn_config import AppConfig


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
        run_pretrain_tests=False,
    )


def _ideal_transfer_config(
    output_length: int | None = 8, fused: bool = False
) -> AppConfig:
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
        mrm_power_distortion_strength=0.0,
        mrm_phase_distortion_strength=0.0,
        pd_distortion_strength=0.0,
        tia_distortion_strength=0.0,
        ler_std_dev=0.0,
        laser_rin_db=None,
        pd_noise_w=0.0,
        pd_input_clamp_min_w=None,
        pd_input_clamp_max_w=None,
        run_pretrain_tests=False,
        enable_jtc_ideal_fused_transfer=fused,
    )


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
            jtc.fft_and_magnitude(input_plane)
            / math.sqrt(float(jtc.jtc_total_field))
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


def test_native_fft_order_matches_shifted_pipeline_with_lens_distortion():
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
            mrm_power_distortion_strength=0.0,
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

        def shifted_magnitude(x):
            x = torch.fft.fftshift(torch.fft.fft(x), dim=-1)
            if not jtc.lens.is_identity():
                x = jtc.lens(x)
            return torch.abs(x)

        with torch.no_grad():
            actual = jtc.forward_paired(signal, kernel)

            signal_distorted = jtc.input_distortion(signal)
            kernel_distorted = jtc.input_distortion(kernel)
            input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
            jps = shifted_magnitude(input_plane) / math.sqrt(float(jtc.jtc_total_field))
            jps = jtc.output_distortion(jps)
            jps = quantize_ste(jps, config.fourier_plane_bits)
            jps_distorted = jtc.input_distortion(jps)
            output_plane = shifted_magnitude(jps_distorted)
            output_plane = torch.sqrt(jtc.output_distortion(output_plane).clamp_min(0.0))
            shifted_indices = torch.arange(
                jtc._correlation_start,
                jtc._correlation_start + jtc.output_length,
                device=output_plane.device,
            )
            shifted_indices = shifted_indices % jtc.jtc_total_field
            expected = output_plane.index_select(-1, shifted_indices)

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
        mrm_power_distortion_strength=0.0,
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


def test_ideal_fused_transfer_matches_reference_outputs():
    """The fused ideal transfer path must preserve JTC outputs exactly."""
    for output_length in (8, None, 10):
        ref_jtc = JTC(_ideal_transfer_config(output_length, fused=False)).eval()
        fused_jtc = JTC(_ideal_transfer_config(output_length, fused=True)).eval()

        assert not ref_jtc._use_fused_transfer
        assert fused_jtc._use_fused_transfer

        torch.manual_seed(123)
        paired_signal = torch.rand(5, 8)
        paired_kernel = torch.rand(5, 8)
        signal = torch.rand(2, 3, 1, 8)
        kernel = torch.rand(4, 8)

        with torch.no_grad():
            ref_paired = ref_jtc.forward_paired(paired_signal, paired_kernel)
            fused_paired = fused_jtc.forward_paired(paired_signal, paired_kernel)
            ref_out = ref_jtc(signal, kernel)
            fused_out = fused_jtc(signal, kernel)

        assert torch.equal(fused_paired, ref_paired)
        assert torch.equal(fused_out, ref_out)


def test_fused_transfer_keeps_complex_phase_path_exact():
    """Fused transfer should still use the complex MRM path when phase is active."""
    ref_config = _ideal_transfer_config(fused=False)
    ref_config.mrm_phase_distortion_strength = 1.0
    fused_config = _ideal_transfer_config(fused=True)
    fused_config.mrm_phase_distortion_strength = 1.0

    ref_jtc = JTC(ref_config).eval()
    fused_jtc = JTC(fused_config).eval()

    assert not ref_jtc._use_fused_transfer
    assert fused_jtc._use_fused_transfer

    torch.manual_seed(456)
    signal = torch.rand(3, 8)
    kernel = torch.rand(3, 8)

    with torch.no_grad():
        ref_out = ref_jtc.forward_paired(signal, kernel)
        fused_out = fused_jtc.forward_paired(signal, kernel)

    assert torch.equal(fused_out, ref_out)


def test_laser_rin_intensity_scale_maps_to_field_sqrt():
    """laser_rin_db is intensity RIN, so complex field amplitude gets sqrt(scale)."""
    x = torch.linspace(0.0, 1.0, steps=8).reshape(1, 8)
    intensity_scale = torch.full((1, 1), 4.0)

    for fused in (False, True):
        config = _ideal_transfer_config(fused=fused)
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


def test_ideal_fused_transfer_matches_reference_gradients():
    """Training gradients should stay aligned with the unfused ideal path."""
    ref_jtc = JTC(_ideal_transfer_config(fused=False)).train()
    fused_jtc = JTC(_ideal_transfer_config(fused=True)).train()

    torch.manual_seed(321)
    signal = torch.rand(4, 8)
    kernel = torch.rand(4, 8)
    ref_signal = signal.clone().requires_grad_(True)
    ref_kernel = kernel.clone().requires_grad_(True)
    fused_signal = signal.clone().requires_grad_(True)
    fused_kernel = kernel.clone().requires_grad_(True)

    ref_loss = ref_jtc.forward_paired(ref_signal, ref_kernel).sum()
    fused_loss = fused_jtc.forward_paired(fused_signal, fused_kernel).sum()
    ref_loss.backward()
    fused_loss.backward()

    assert torch.equal(fused_loss.detach(), ref_loss.detach())
    torch.testing.assert_close(fused_signal.grad, ref_signal.grad, rtol=0, atol=0)
    torch.testing.assert_close(fused_kernel.grad, ref_kernel.grad, rtol=0, atol=0)


def test_gradients_flow_through_jtc():
    jtc = JTC(_test_config()).train()
    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8, requires_grad=True)
    kernel = torch.rand(3, 8, requires_grad=True)

    loss = jtc(signal, kernel).sum()
    loss.backward()

    assert signal.grad is not None
    assert kernel.grad is not None
    assert signal.grad.abs().max() > 0
    assert kernel.grad.abs().max() > 0
