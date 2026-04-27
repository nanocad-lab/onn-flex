"""
Run a single (mode, case) experiment for the "quantization as a parameter" protocol.

This is intentionally single-run so that Slurm can parallelize across cases.

Protocol summary:
  - baseline + one-hot cases: clamp-only, no quantization noise (dac/fourier/adc bits=None)
  - quant-only case: distortions off, quant enabled (default 4/4/6)
  - all-ones case: all distortions on (alpha=1), quant enabled (default 4/4/6)
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

from onn_config import load_app_config_from_yaml, AppConfig
from onn_inference import run_inference
from onn_train import train_onn_model


DISTORTION_KEYS = [
    "driver_distortion_strength",
    "pd_distortion_strength",
    "tia_distortion_strength",
    "mrm_power_distortion_strength",
    "mrm_phase_distortion_strength",
    "lens_distortion_strength",
]
ConfigOverride = float | int | str | bool | None


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
        choices=[
            "all-zeros",
            "quant",
            "driver",
            "pd",
            "tia",
            "mrm-power",
            "mrm-phase",
            "lens",
            "all-ones",
        ],
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
        "--mrm-power-gain",
        type=float,
        default=1.0,
        help="MRM power gain (only used for mrm-power case).",
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
    ov.update(
        {
            "dac_bits": None,
            "fourier_plane_bits": None,
            "adc_bits": None,
            "mrm_power_gain": 1.0,
        }
    )

    case = str(args.case)
    if case == "all-zeros":
        return ov
    if case == "quant":
        ov.update({"dac_bits": dac_b, "fourier_plane_bits": fp_b, "adc_bits": adc_b})
        return ov
    if case == "all-ones":
        for k in DISTORTION_KEYS:
            ov[k] = 1.0
        ov.update({"dac_bits": dac_b, "fourier_plane_bits": fp_b, "adc_bits": adc_b})
        return ov

    mapping = {
        "driver": "driver_distortion_strength",
        "pd": "pd_distortion_strength",
        "tia": "tia_distortion_strength",
        "mrm-power": "mrm_power_distortion_strength",
        "mrm-phase": "mrm_phase_distortion_strength",
        "lens": "lens_distortion_strength",
    }
    if case not in mapping:
        raise ValueError(f"Unknown case: {case}")
    ov[mapping[case]] = 1.0
    if case == "mrm-power":
        ov["mrm_power_gain"] = float(args.mrm_power_gain)
    return ov


def _write_cfg(cfg: AppConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def main() -> None:
    args = _parse_args()

    base_run = str(args.base_run)
    weights_path = os.path.join(base_run, "fftconv_checkpoint.pth")
    if not os.path.exists(weights_path):
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
