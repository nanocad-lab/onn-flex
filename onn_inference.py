from dataclasses import replace
from typing import Tuple, Any, Dict

import torch
import yaml

from onn_config import AppConfig
from onn_component import JTC
from onn_train import FFTConvNet, evaluate, get_data_loaders


def load_config_from_yaml(path: str) -> AppConfig:
    """Load AppConfig from a YAML file."""
    with open(path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    valid = {k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__}
    return AppConfig(**valid)


def run_inference(config: AppConfig, weights_path: str) -> float:
    """Run inference using *weights_path* and return accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = get_data_loaders(config.batch_size)
    model = FFTConvNet(config).to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    acc = evaluate(model, testloader, device)
    return acc


def _build_jtc(config: AppConfig) -> JTC:
    return JTC(config)


def compute_snr_enob(
    config: AppConfig, param: str, num_tests: int = 16, seed: int = 0
) -> Tuple[float, float]:
    """Compute SNR and ENOB for a specific distortion parameter."""
    torch.manual_seed(seed)

    jtc = _build_jtc(config)
    ref_cfg = replace(config, **{param: 0.0})
    jtc_ref = _build_jtc(ref_cfg)

    out_list = []
    ref_list = []
    for _ in range(num_tests):
        signal = torch.rand(1, 1, 1, config.jtc_half_size)
        kernel = torch.rand(1, config.jtc_half_size)
        out_list.append(jtc(signal, kernel))
        ref_list.append(jtc_ref(signal, kernel))

    out = torch.stack(out_list)
    ref = torch.stack(ref_list)
    noise = out - ref
    snr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())
    enob = (snr - 1.76) / 6.02
    return snr.item(), enob.item()


# NEW ------------------------------------------------------------------
#  SNR helper for the JPS stage
# ----------------------------------------------------------------------


def compute_snr_jps(
    config: AppConfig, param: str, num_tests: int = 16, seed: int = 0
) -> float:
    """Compute the SNR at the JPS (Fourier plane) for a given distortion parameter.

    The methodology mirrors *compute_snr_enob* but measures the signal after
    the *post_output_distortion* stage (the JPS) instead of the final inverse
    propagation.  Only the SNR is returned because ENOB is typically defined
    for ADC-level metrics and is less meaningful at the optical plane.
    """
    torch.manual_seed(seed)

    jtc = _build_jtc(config)
    ref_cfg = replace(config, **{param: 0.0})
    jtc_ref = _build_jtc(ref_cfg)

    jps_list = []
    ref_jps_list = []
    for _ in range(num_tests):
        # Random 1-D test vectors (same dimensions used in *compute_snr_enob*)
        signal = torch.rand(1, config.jtc_half_size)
        kernel = torch.rand(1, config.jtc_half_size)

        # Build input plane and propagate to JPS for the distorted JTC
        input_plane = jtc.generate_input_plane(signal, kernel)
        jft = jtc.post_fft(input_plane)
        jps_list.append(jtc.post_output_distortion(jft))

        # Same for the reference (ideal) JTC
        input_plane_ref = jtc_ref.generate_input_plane(signal, kernel)
        jft_ref = jtc_ref.post_fft(input_plane_ref)
        ref_jps_list.append(jtc_ref.post_output_distortion(jft_ref))

    jps = torch.stack(jps_list)
    ref_jps = torch.stack(ref_jps_list)

    noise = jps - ref_jps
    snr = 10.0 * torch.log10(ref_jps.pow(2).mean() / noise.pow(2).mean())
    return snr.item()
