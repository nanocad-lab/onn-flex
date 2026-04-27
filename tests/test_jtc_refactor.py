import torch

from onn_component import JTC, QuantDequant_STE
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
        jps = jtc.output_distortion(jtc.fft_and_magnitude(input_plane))
        jps = QuantDequant_STE.apply(jps, config.fourier_plane_bits)
        jps_distorted = jtc.input_distortion(jps)
        output_plane = jtc.output_distortion(jtc.fft_and_magnitude(jps_distorted))
        indices = jtc.compute_correlation_indices(output_plane.device)
        explicit = output_plane[..., indices].reshape_as(output)

    assert torch.allclose(output, explicit)


def test_forward_is_deterministic_without_noise():
    jtc = JTC(_test_config()).eval()
    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8)
    kernel = torch.rand(3, 8)

    with torch.no_grad():
        output_1 = jtc(signal, kernel)
        output_2 = jtc(signal, kernel)

    assert torch.equal(output_1, output_2)


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
