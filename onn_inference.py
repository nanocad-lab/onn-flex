from dataclasses import replace
from typing import Tuple

import torch
import yaml

from onn_config import AppConfig
from onn_component import Driver, MRM, PD_TIA, JTC
from onn_train import FFTConvNet, evaluate, get_data_loaders


def load_config_from_yaml(path: str) -> AppConfig:
    """Load AppConfig from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    valid = {k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__}
    return AppConfig(**valid)


def run_inference(config: AppConfig, weights_path: str) -> float:
    """Run inference using *weights_path* and return accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader = get_data_loaders(config.batch_size)
    model = FFTConvNet(config).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    acc = evaluate(model, testloader, device)
    return acc


def _build_jtc(config: AppConfig) -> JTC:
    driver = Driver(config)
    pd_tia = PD_TIA(config)
    mrm = MRM(config)
    return JTC(config, driver, mrm, pd_tia)


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
        signal = torch.rand(config.jtc_half_size) * 2 - 1
        kernel = torch.rand(config.jtc_half_size) * 2 - 1
        out_list.append(jtc(signal, kernel))
        ref_list.append(jtc_ref(signal, kernel))

    out = torch.stack(out_list)
    ref = torch.stack(ref_list)
    noise = out - ref
    snr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())
    enob = (snr - 1.76) / 6.02
    return snr.item(), enob.item()
