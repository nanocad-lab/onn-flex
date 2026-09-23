"""
Run a single (mode, case) experiment for the "quantization as a parameter" protocol.

This is intentionally single-run so that Slurm can parallelize across cases.

Protocol summary:
  - all-zeros: all distortions off, quantization off
  - one-hot distortion cases: one distortion on, quantization off
  - onehot-quant: all distortions off, quantization on
  - all-ones: all distortions on, quantization on
  - one-cold distortion cases: one distortion off, all other distortions on,
    quantization on
  - onecold-quant: all distortions on, quantization off
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path

import yaml

# Ensure repository root is importable when run directly
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from onn_config import AppConfig, load_app_config_from_yaml
from onn_inference import run_inference
from onn_train import train_onn_model

DISTORTION_KEYS = [
    "driver_distortion_strength",
    "pd_distortion_strength",
    "tia_distortion_strength",
    "mrm_amplitude_distortion_strength",
    "mrm_phase_distortion_strength",
    "lens_distortion_strength",
]
DISTORTION_CASES = {
    "driver": "driver_distortion_strength",
    "pd": "pd_distortion_strength",
    "tia": "tia_distortion_strength",
    "mrm-amplitude": "mrm_amplitude_distortion_strength",
    "mrm-phase": "mrm_phase_distortion_strength",
    "lens": "lens_distortion_strength",
}
ConfigOverride = float | int | str | bool | None
UNSET = object()


def _optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single onehot/quant case")
    p.add_argument(
        "--base-run",
        required=True,
        help="Baseline run directory (must contain final_config.yaml and fftconv_checkpoint.pth).",
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["infer", "finetune", "retrain"],
        help="Which experiment type to run.",
    )
    p.add_argument(
        "--case",
        required=True,
        choices=(
            ["all-zeros", "onehot-quant"]
            + [f"onehot-{name}" for name in DISTORTION_CASES]
            + ["all-ones", "onecold-quant"]
            + [f"onecold-{name}" for name in DISTORTION_CASES]
        ),
        help="Which case to run.",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for the run.")
    p.add_argument(
        "--quant-bits",
        type=int,
        nargs=3,
        metavar=("DAC", "FOURIER", "ADC"),
        default=(4, 4, 6),
        help="Bitwidths used for the quant and all-ones cases (default: 4 4 6).",
    )
    p.add_argument(
        "--laser-power-gain",
        type=float,
        default=None,
        help="Optional laser source power gain override. Defaults to the base config.",
    )
    p.add_argument(
        "--converter-clamp-grad",
        choices=["pwl", "mad"],
        default=None,
        help="Optional converter clamp backward surrogate override.",
    )
    p.add_argument(
        "--transfer-linearization",
        choices=["endpoint", "least_squares", "midpoint_ls"],
        default=None,
        help="Optional ideal transfer-function linearization override.",
    )
    p.add_argument(
        "--pd-input-clamp-min-w",
        type=_optional_float,
        default=UNSET,
        help="Optional lower PD input clamp override in Watts; use 'none' to disable.",
    )
    p.add_argument(
        "--pd-input-clamp-max-w",
        type=_optional_float,
        default=UNSET,
        help="Optional upper PD input clamp override in Watts; use 'none' to disable.",
    )
    p.add_argument(
        "--pd-input-clamp-mode",
        choices=["hard", "soft"],
        default=None,
        help="Optional PD input clamp mode override.",
    )
    p.add_argument(
        "--pd-input-soft-clamp-width-w",
        type=float,
        default=None,
        help="Optional soft PD input clamp feather width in Watts.",
    )
    p.add_argument(
        "--pd-range-regularization-weight",
        type=float,
        default=None,
        help="Optional training penalty weight for PD input power leaving the fit range.",
    )
    p.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="Epochs to run for finetune mode (default: 5).",
    )
    p.add_argument(
        "--finetune-lr",
        type=float,
        default=5e-4,
        help="Learning rate for finetune mode (default: 5e-4).",
    )
    p.add_argument(
        "--retrain-epochs",
        type=int,
        default=None,
        help="Epochs to run for retrain mode (default: use config.num_epochs).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override for training/evaluation.",
    )
    return p.parse_args()


def _read_base_config(base_run_dir: str) -> AppConfig:
    final_cfg = os.path.join(base_run_dir, "final_config.yaml")
    if not os.path.exists(final_cfg):
        raise FileNotFoundError(f"Missing {final_cfg}")
    return load_app_config_from_yaml(final_cfg)


def _case_overrides(args: argparse.Namespace) -> dict[str, ConfigOverride]:
    dac_b, fp_b, adc_b = (
        int(args.quant_bits[0]),
        int(args.quant_bits[1]),
        int(args.quant_bits[2]),
    )

    ov: dict[str, ConfigOverride] = {k: 0.0 for k in DISTORTION_KEYS}
    ov.update({"dac_bits": None, "fourier_plane_bits": None, "adc_bits": None})
    if args.laser_power_gain is not None:
        ov["laser_power_gain"] = float(args.laser_power_gain)
    if args.converter_clamp_grad is not None:
        ov["converter_clamp_grad"] = str(args.converter_clamp_grad)
    if args.transfer_linearization is not None:
        ov["transfer_linearization"] = str(args.transfer_linearization)
    if args.pd_input_clamp_min_w is not UNSET:
        ov["pd_input_clamp_min_w"] = args.pd_input_clamp_min_w
    if args.pd_input_clamp_max_w is not UNSET:
        ov["pd_input_clamp_max_w"] = args.pd_input_clamp_max_w
    if args.pd_input_clamp_mode is not None:
        ov["pd_input_clamp_mode"] = str(args.pd_input_clamp_mode)
    if args.pd_input_soft_clamp_width_w is not None:
        ov["pd_input_soft_clamp_width_w"] = float(args.pd_input_soft_clamp_width_w)
    if args.pd_range_regularization_weight is not None:
        ov["pd_range_regularization_weight"] = float(
            args.pd_range_regularization_weight
        )
    if args.seed is not None:
        ov["seed"] = int(args.seed)

    case = str(args.case)
    if case == "all-zeros":
        return ov
    if case == "onehot-quant":
        ov.update({"dac_bits": dac_b, "fourier_plane_bits": fp_b, "adc_bits": adc_b})
        return ov
    if case == "all-ones":
        for k in DISTORTION_KEYS:
            ov[k] = 1.0
        ov.update({"dac_bits": dac_b, "fourier_plane_bits": fp_b, "adc_bits": adc_b})
        return ov
    if case == "onecold-quant":
        for k in DISTORTION_KEYS:
            ov[k] = 1.0
        return ov

    if case.startswith("onehot-"):
        name = case.removeprefix("onehot-")
        if name not in DISTORTION_CASES:
            raise ValueError(f"Unknown one-hot case: {case}")
        ov[DISTORTION_CASES[name]] = 1.0
        return ov

    if case.startswith("onecold-"):
        name = case.removeprefix("onecold-")
        if name not in DISTORTION_CASES:
            raise ValueError(f"Unknown one-cold case: {case}")
        for k in DISTORTION_KEYS:
            ov[k] = 1.0
        ov[DISTORTION_CASES[name]] = 0.0
        ov.update({"dac_bits": dac_b, "fourier_plane_bits": fp_b, "adc_bits": adc_b})
        return ov

    raise ValueError(f"Unknown case: {case}")


def _write_cfg(cfg: AppConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def main() -> None:
    args = _parse_args()

    base_run = str(args.base_run)
    weights_path = os.path.join(base_run, "fftconv_checkpoint.pth")
    if args.mode in {"infer", "finetune"} and not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path}")

    base_cfg = _read_base_config(base_run)
    ov = _case_overrides(args)

    out_dir = str(args.output_dir)
    cfg = replace(
        base_cfg,
        **ov,
        output_dir=out_dir,
        run_pretrain_tests=False,
        pretrain_tests_only=False,
        run_full_strength_inference=False,
    )

    if args.mode == "infer":
        _write_cfg(cfg, out_dir)
        acc = run_inference(cfg, weights_path)
        with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
            f.write(f"accuracy: {acc:.4f}\n")
        return

    if args.mode == "finetune":
        cfg = replace(
            cfg,
            eval_only=False,
            pretrained_weights=weights_path,
            num_epochs=int(args.finetune_epochs),
            learning_rate=float(args.finetune_lr),
        )
        _write_cfg(cfg, out_dir)
        train_onn_model(cfg)
        return

    if args.mode == "retrain":
        epochs = (
            base_cfg.num_epochs
            if args.retrain_epochs is None
            else int(args.retrain_epochs)
        )
        cfg = replace(
            cfg,
            eval_only=False,
            pretrained_weights="",
            num_epochs=int(epochs),
        )
        _write_cfg(cfg, out_dir)
        train_onn_model(cfg)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
