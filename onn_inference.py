from contextlib import contextmanager
from dataclasses import replace

import torch

from onn_analog_shot import AnalogJTCShot
from onn_component import JTC
from onn_config import AppConfig
from onn_train import build_model, evaluate, get_data_loaders, load_model_state


def run_inference(config: AppConfig, weights_path: str) -> float:
    """Run inference using *weights_path* and return accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, testloader = get_data_loaders(config.batch_size, dataset=config.dataset)
    model = build_model(config).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    load_model_state(model, ckpt["model_state_dict"])
    acc = evaluate(model, testloader, device, max_batches=config.max_eval_batches)
    return acc


def _ideal_param_value(param: str) -> float | None:
    """Return the 'no distortion/noise' value for *param*.

    Most distortion parameters are ideal at 0.0. Some parameters (like
    optional noise terms) use `None` to disable the effect entirely.
    """
    if param in {"laser_rin_db", "jtc_frontend_snr_db"}:
        return None
    return 0.0


def _build_ideal_reference_cfg(config: AppConfig, *, disable_quant: bool) -> AppConfig:
    """Build an "ideal" reference config for SQNDR-like comparisons.

    The reference disables all distortion/noise terms; optionally disables
    all quantization stages as well.
    """
    overrides: dict[str, float | None] = {
        "driver_distortion_strength": 0.0,
        "pd_distortion_strength": 0.0,
        "tia_distortion_strength": 0.0,
        "mrm_amplitude_distortion_strength": 0.0,
        "mrm_phase_distortion_strength": 0.0,
        "ler_std_dev": 0.0,
        "lens_distortion_strength": 0.0,
        "laser_rin_db": None,
        "pd_noise_w": 0.0,
        "jtc_frontend_snr_db": None,
    }
    if disable_quant:
        overrides.update(
            {
                "dac_bits": None,
                "fourier_plane_bits": None,
                "adc_bits": None,
            }
        )
    return replace(config, **overrides)


_SNR_CHUNK_SIZE = 1024


def _snr_enob(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    noise = out - ref
    snr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())
    enob = (snr - 1.76) / 6.02
    return snr.item(), enob.item()


def _random_jtc_inputs(config: AppConfig) -> tuple[torch.Tensor, torch.Tensor]:
    signal = torch.rand(1, 1, 1, config.input_length)
    kernel = torch.rand(1, config.kernel_length)
    return signal, kernel


def _jps_output(jtc: JTC, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    signal = signal.reshape(signal.shape[0], jtc.input_length)
    kernel = kernel.reshape(kernel.shape[0], jtc.kernel_length)
    laser_scale = jtc.mrm.make_laser_scale(
        torch.empty(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)
    )
    signal_distorted = jtc.input_distortion(signal, laser_scale=laser_scale)
    kernel_distorted = jtc.input_distortion(kernel, laser_scale=laser_scale)
    input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
    if jtc.config.conv_backend in {"jtc_analytic", "jtc_analog_fourier"}:
        power = jtc.fft_and_power(input_plane) * (jtc.loss / jtc.jtc_total_field)
        stop = jtc.config.jtc_carrier_stopband_bins
        if stop:
            mask = torch.ones_like(power)
            mask[..., : stop + 1] = 0
            mask[..., -stop:] = 0
            power = power * mask
        return jtc._tia_transfer_raw(jtc._pd_transfer_raw(power))
    return jtc.first_detector_readout(input_plane)


@contextmanager
def _metric_rng(seed, device):
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def _shot_model(config: AppConfig, device: torch.device):
    if config.conv_backend in {"jtc_analytic", "jtc_analog_fourier"}:
        if config.jtc_output_gain_mode not in {"fixed", "per_shot"}:
            raise ValueError(
                "Standalone SNR needs fixed or per_shot gain; a config does not "
                "contain a trained layer's calibrated gain. Set the measured fixed gain."
            )
        # Use the exact physical plane for metrics, including contamination.
        cfg = replace(
            config, conv_backend="jtc_analog_fourier", jtc_fourier_closed_form=False
        )
        model = AnalogJTCShot(cfg).to(device).eval()
        model.validate()
        return model
    return JTC(config).to(device).eval()


def compute_snr_between_configs(
    config: AppConfig, ref_config: AppConfig, num_tests: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Compute SNR and ENOB between outputs of two configurations.

    Test vectors are processed in batched chunks on the available device so
    large sample counts (e.g. 100k) stay tractable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with _metric_rng(seed, device):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        if num_tests <= 0:
            raise ValueError("num_tests must be positive")
        if (config.input_length, config.kernel_length) != (
            ref_config.input_length,
            ref_config.kernel_length,
        ):
            raise ValueError("SNR configurations must use matching aperture lengths")
        jtc = _shot_model(config, device)
        jtc_ref = _shot_model(ref_config, device)

        ref_power = torch.zeros((), dtype=torch.float64, device=device)
        err_power = torch.zeros((), dtype=torch.float64, device=device)
        remaining = num_tests
        with torch.no_grad():
            while remaining > 0:
                n = min(_SNR_CHUNK_SIZE, remaining)
                signal = torch.rand(
                    n, config.input_length, generator=generator, device=device
                )
                kernel = torch.rand(
                    n, config.kernel_length, generator=generator, device=device
                )
                out = jtc(signal, kernel).double()
                ref = jtc_ref(signal, kernel).double()
                ref_power += ref.pow(2).sum()
                err_power += (out - ref).pow(2).sum()
                remaining -= n
        if ref_power == 0:
            raise ValueError(
                "SNR reference has zero power; check gain and detector operating point"
            )
        snr = 10.0 * torch.log10(ref_power / err_power)
        enob = (snr - 1.76) / 6.02
        return snr.item(), enob.item()


def compute_snr_enob(
    config: AppConfig, param: str, num_tests: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Compute SNR and ENOB for a specific distortion parameter."""
    ref_cfg = replace(config, **{param: _ideal_param_value(param)})
    return compute_snr_between_configs(config, ref_cfg, num_tests=num_tests, seed=seed)


def compute_snqr_enob(
    config: AppConfig,
    quantization_stages: tuple[str, ...] = ("dac", "fourier_plane", "adc"),
    num_tests: int = 16,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute SNQR and ENOB due to quantization.

    Quantization is modeled by (possibly) enabling DAC, Fourier-plane, and ADC
    stages (via their bit-width settings). This helper estimates the
    signal-to-quantization-noise ratio (SNQR) by comparing the JTC output from
    the current *config* against an otherwise-identical configuration where the
    requested stages are disabled (bit-width set to None).

    Args:
        config: Configuration to evaluate.
        quantization_stages: Subset of {"dac", "fourier_plane", "adc"} to
            disable in the reference configuration.
        num_tests: Number of random trials used for the estimate.
        seed: RNG seed for reproducibility.

    Returns:
        (snqr_db, enob) where enob is computed via the common ENOB heuristic:
        (SNQR-1.76)/6.02.
    """
    ref_overrides: dict[str, None] = {}
    for q in quantization_stages:
        if q == "dac":
            ref_overrides["dac_bits"] = None
        elif q == "fourier_plane":
            ref_overrides["fourier_plane_bits"] = None
        elif q == "adc":
            ref_overrides["adc_bits"] = None
        else:
            raise ValueError(
                f"Unknown quantization stage '{q}'. Expected one of: dac,fourier_plane,adc."
            )

    ref_cfg = replace(config, **ref_overrides)
    return compute_snr_between_configs(config, ref_cfg, num_tests=num_tests, seed=seed)


def compute_sqndr_enob(
    config: AppConfig, num_tests: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Compute SQNDR (signal-to-quantization-noise-and-distortion) and ENOB.

    This metric captures the *combined* effect of:
      - Transfer-function distortions (driver, PD, TIA, MRM, lens, LER, etc.)
      - Additive noise terms (laser RIN, PD noise)
      - Quantization (DAC / Fourier-plane / ADC)

    The estimate is formed by comparing the JTC output from the current
    configuration against an otherwise-identical "ideal" reference that:
      - Disables all quantization stages (bit-widths set to None)
      - Sets all distortion strengths to their ideal values (typically 0)
      - Disables noise terms (laser_rin_db=None, pd_noise_w=0)

    Args:
        config: Configuration to evaluate.
        num_tests: Number of random trials used for the estimate.
        seed: RNG seed for reproducibility.

    Returns:
        (sqndr_db, enob) with ENOB computed via (SQNDR-1.76)/6.02.
    """
    ref_cfg = _build_ideal_reference_cfg(config, disable_quant=True)
    return compute_snr_between_configs(config, ref_cfg, num_tests=num_tests, seed=seed)


def compute_sndr_vs_quantized_ideal_enob(
    config: AppConfig, num_tests: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Compute SNDR vs a quantized-ideal reference (distortions off, quant on)."""
    ref_cfg = _build_ideal_reference_cfg(config, disable_quant=False)
    return compute_snr_between_configs(config, ref_cfg, num_tests=num_tests, seed=seed)


def compute_snr_jps(
    config: AppConfig, param: str, num_tests: int = 16, seed: int = 0
) -> float:
    """Compute the SNR at the JPS (Fourier plane) for a given distortion parameter.

    The methodology mirrors ``compute_snr_enob`` but measures the signal after
    first-pass Fourier-plane output distortion instead of the final detector
    output. Only the SNR is returned because ENOB is less meaningful at the
    optical plane.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with _metric_rng(seed, device):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        jtc = JTC(config).to(device).eval()
        ref_cfg = replace(config, **{param: _ideal_param_value(param)})
        jtc_ref = JTC(ref_cfg).to(device).eval()

        ref_power = torch.zeros((), dtype=torch.float64, device=device)
        err_power = torch.zeros((), dtype=torch.float64, device=device)
        remaining = num_tests
        with torch.no_grad():
            while remaining > 0:
                n = min(_SNR_CHUNK_SIZE, remaining)
                signal = torch.rand(
                    n, config.input_length, generator=generator, device=device
                )
                kernel = torch.rand(
                    n, config.kernel_length, generator=generator, device=device
                )
                jps = _jps_output(jtc, signal, kernel).double()
                ref_jps = _jps_output(jtc_ref, signal, kernel).double()
                ref_power += ref_jps.pow(2).sum()
                err_power += (jps - ref_jps).pow(2).sum()
                remaining -= n
        if ref_power == 0:
            raise ValueError(
                "SNR reference has zero power; check gain and detector operating point"
            )
        snr = 10.0 * torch.log10(ref_power / err_power)
        return snr.item()
