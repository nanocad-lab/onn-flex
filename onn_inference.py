from dataclasses import replace

import torch

from onn_config import AppConfig, load_app_config_from_yaml
from onn_component import JTC
from onn_train import FFTConvNet, evaluate, get_data_loaders


def load_config_from_yaml(path: str) -> AppConfig:
    """Load AppConfig from a YAML file."""
    return load_app_config_from_yaml(path)


def run_inference(config: AppConfig, weights_path: str) -> float:
    """Run inference using *weights_path* and return accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, testloader = get_data_loaders(config.batch_size)
    model = FFTConvNet(config).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    max_batches = getattr(config, "max_eval_batches", None)
    acc = evaluate(model, testloader, device, max_batches=max_batches)
    return acc


def _build_jtc(config: AppConfig) -> JTC:
    return JTC(config)


def _ideal_param_value(param: str) -> float | None:
    """Return the 'no distortion/noise' value for *param*.

    Most distortion parameters are ideal at 0.0. Some parameters (like
    optional noise terms) use `None` to disable the effect entirely.
    """
    if param == "laser_rin_db":
        return None
    return 0.0


def _build_ideal_reference_cfg(
    config: AppConfig, *, disable_quant: bool
) -> AppConfig:
    """Build an "ideal" reference config for SQNDR-like comparisons.

    The reference disables all distortion/noise terms; optionally disables
    all quantizers as well.
    """
    overrides: dict[str, float | None] = {
        "driver_distortion_strength": 0.0,
        "pd_distortion_strength": 0.0,
        "tia_distortion_strength": 0.0,
        "mrm_power_distortion_strength": 0.0,
        "mrm_phase_distortion_strength": 0.0,
        "ler_std_dev": 0.0,
        "lens_distortion_strength": 0.0,
        "laser_rin_db": None,
        "pd_noise_w": 0.0,
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


def _snr_enob(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    noise = out - ref
    snr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())
    enob = (snr - 1.76) / 6.02
    return snr.item(), enob.item()


def _random_jtc_inputs(config: AppConfig) -> tuple[torch.Tensor, torch.Tensor]:
    signal = torch.rand(1, 1, 1, config.input_length)
    kernel = torch.rand(1, config.kernel_length)
    return signal, kernel


def _jtc_output(jtc: JTC, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return jtc(signal, kernel)


def _jps_output(jtc: JTC, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    signal = signal.reshape(signal.shape[0], jtc.input_length)
    kernel = kernel.reshape(kernel.shape[0], jtc.kernel_length)
    laser_scale = jtc.mrm.make_laser_scale(
        torch.empty(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)
    )
    signal_distorted = jtc.input_distortion(signal, laser_scale=laser_scale)
    kernel_distorted = jtc.input_distortion(kernel, laser_scale=laser_scale)
    input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
    return jtc.output_distortion(jtc.fft_and_magnitude(input_plane))


def compute_snr_between_configs(
    config: AppConfig, ref_config: AppConfig, num_tests: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Compute SNR and ENOB between outputs of two configurations."""
    torch.manual_seed(seed)

    jtc = _build_jtc(config)
    jtc_ref = _build_jtc(ref_config)

    out_list = []
    ref_list = []
    for _ in range(num_tests):
        signal, kernel = _random_jtc_inputs(config)
        out_list.append(_jtc_output(jtc, signal, kernel))
        ref_list.append(_jtc_output(jtc_ref, signal, kernel))

    out = torch.stack(out_list)
    ref = torch.stack(ref_list)
    return _snr_enob(out, ref)


def compute_snr_enob(
    config: AppConfig, param: str, num_tests: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Compute SNR and ENOB for a specific distortion parameter."""
    ref_cfg = replace(config, **{param: _ideal_param_value(param)})
    return compute_snr_between_configs(config, ref_cfg, num_tests=num_tests, seed=seed)


def compute_snqr_enob(
    config: AppConfig,
    quantizers: tuple[str, ...] = ("dac", "fourier_plane", "adc"),
    num_tests: int = 16,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute SNQR and ENOB due to quantization.

    Quantization is modeled by (possibly) enabling DAC, Fourier-plane, and ADC
    quantizers (via their bit-width settings). This helper estimates the
    signal-to-quantization-noise ratio (SNQR) by comparing the JTC output from
    the current *config* against an otherwise-identical configuration where the
    requested quantizers are disabled (bit-width set to None).

    Args:
        config: Configuration to evaluate.
        quantizers: Subset of {"dac", "fourier_plane", "adc"} to disable in the
            reference configuration.
        num_tests: Number of random trials used for the estimate.
        seed: RNG seed for reproducibility.

    Returns:
        (snqr_db, enob) where enob is computed via the common ENOB heuristic:
        (SNQR-1.76)/6.02.
    """
    torch.manual_seed(seed)

    ref_overrides: dict[str, None] = {}
    for q in quantizers:
        if q == "dac":
            ref_overrides["dac_bits"] = None
        elif q == "fourier_plane":
            ref_overrides["fourier_plane_bits"] = None
        elif q == "adc":
            ref_overrides["adc_bits"] = None
        else:
            raise ValueError(
                f"Unknown quantizer '{q}'. Expected one of: dac,fourier_plane,adc."
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
      - Disables all quantizers (bit-widths set to None)
      - Sets all distortion strengths to their ideal values (typically 0)
      - Disables noise terms (laser_rin_db=None, pd_noise_w=0)

    Args:
        config: Configuration to evaluate.
        num_tests: Number of random trials used for the estimate.
        seed: RNG seed for reproducibility.

    Returns:
        (sqndr_db, enob) with ENOB computed via (SQNDR-1.76)/6.02.
    """
    torch.manual_seed(seed)

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
    torch.manual_seed(seed)

    jtc = _build_jtc(config)
    ref_cfg = replace(config, **{param: _ideal_param_value(param)})
    jtc_ref = _build_jtc(ref_cfg)

    jps_list = []
    ref_jps_list = []
    for _ in range(num_tests):
        signal, kernel = _random_jtc_inputs(config)
        jps_list.append(_jps_output(jtc, signal, kernel))
        ref_jps_list.append(_jps_output(jtc_ref, signal, kernel))

    jps = torch.stack(jps_list)
    ref_jps = torch.stack(ref_jps_list)

    snr, _ = _snr_enob(jps, ref_jps)
    return snr
