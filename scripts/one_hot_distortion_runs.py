"""
Run inference and training by toggling one distortion strength to 1.0 at a time,
then all strengths to 1.0, starting from a pretrained ideal run directory.

Usage example (one line):

  python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --output-root runs/runs_ideal_0825_onehots --do-infer --do-train --epochs 20

Fine-tune example (set additional epochs and LR):

  python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --do-finetune --finetune-additional-epochs 5 --finetune-lr 5e-4

Notes:
- Inference uses the checkpoint in --base-run (fftconv_checkpoint.pth).
- Training starts from scratch using the modified configs (no finetune path
  implemented in the current training API).
"""

import argparse
import os
import sys
from dataclasses import replace, asdict
from pathlib import Path

# Ensure repository root is importable when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml

from onn_config import AppConfig
from onn_config import load_app_config_from_yaml
from onn_inference import load_config_from_yaml, run_inference
from onn_train import train_onn_model


DISTORTION_STRENGTH_KEYS: list[str] = [
    "driver_distortion_strength",
    "pd_distortion_strength",
    "tia_distortion_strength",
    "mrm_power_distortion_strength",
    "mrm_phase_distortion_strength",
    "lens_distortion_strength",
]


def _read_base_config(base_run_dir: str) -> AppConfig:
    """Load the base AppConfig from the run directory.

    Prefers `final_config.yaml` (plain YAML). Falls back to `config.yaml`.
    """
    final_cfg = os.path.join(base_run_dir, "final_config.yaml")
    if os.path.exists(final_cfg):
        return load_config_from_yaml(final_cfg)

    cfg_path = os.path.join(base_run_dir, "config.yaml")
    try:
        return load_app_config_from_yaml(cfg_path)
    except Exception:
        pass
    raise FileNotFoundError(
        f"Could not load a usable config from {base_run_dir}. Expected final_config.yaml or config.yaml"
    )


def _save_config(cfg: AppConfig, out_dir: str, filename: str = "config.yaml") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False)


def _build_cfg_with_strengths(base: AppConfig, kv: dict[str, float]) -> AppConfig:
    return replace(base, **kv)


def _infer_case(cfg: AppConfig, weights: str, out_dir: str) -> float:
    os.makedirs(out_dir, exist_ok=True)
    _save_config(cfg, out_dir)
    acc = run_inference(cfg, weights)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
    return acc


def _train_case(cfg: AppConfig, out_dir: str) -> float:
    os.makedirs(out_dir, exist_ok=True)
    # Ensure training flags are set appropriately
    cfg = replace(
        cfg,
        eval_only=False,
        pretrained_weights="",
        output_dir=out_dir,
        run_pretrain_tests=False,
    )
    _save_config(cfg, out_dir)
    final_test_acc = train_onn_model(cfg)
    return final_test_acc


def _finetune_case(
    cfg: AppConfig,
    out_dir: str,
    weights: str,
    additional_epochs: int = 0,
    finetune_lr: float | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Enable fine-tuning from checkpoint
    epochs_for_finetune = (
        cfg.num_epochs + int(additional_epochs)
        if additional_epochs > 0
        else cfg.num_epochs
    )
    lr = finetune_lr if finetune_lr is not None else cfg.learning_rate
    cfg = replace(
        cfg,
        eval_only=False,
        pretrained_weights=weights,
        output_dir=out_dir,
        run_pretrain_tests=False,
        num_epochs=epochs_for_finetune,
        learning_rate=lr,
    )
    _save_config(cfg, out_dir)
    train_onn_model(cfg)


def _format_case_name(kv: dict[str, float]) -> str:
    # Create short, filesystem-friendly name for the case, e.g. driver1.0
    if len(kv) == 1:
        k, v = next(iter(kv.items()))
        return f"{k.replace('_distortion_strength', '').replace('_', '-')}{v:.1f}"
    return "all-ones"


def run_one_hot_sweeps(
    base_run_dir: str,
    output_root: str,
    do_infer: bool,
    do_train: bool,
    do_finetune: bool,
    epochs_override: int | None,
    finetune_additional_epochs: int,
    finetune_lr: float | None,
    no_quant_non_all_ones: bool,
    include_quant_onehot: bool,
    all_ones_dac_bits: int | None,
    all_ones_fourier_plane_bits: int | None,
    all_ones_adc_bits: int | None,
    include_keys: list[str],
    onehot_keys: list[str] | None = None,
    skip_all_zeros: bool = False,
    skip_all_ones: bool = False,
    batch_size_override: int | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> tuple[AppConfig, str]:
    base_cfg = _read_base_config(base_run_dir)
    weights_path = os.path.join(base_run_dir, "fftconv_checkpoint.pth")
    if not os.path.exists(weights_path) and do_infer:
        raise FileNotFoundError(
            f"Pretrained checkpoint not found for inference: {weights_path}"
        )

    # Optionally override epochs for training runs
    if epochs_override is not None:
        base_cfg = replace(base_cfg, num_epochs=int(epochs_override))
    if batch_size_override is not None:
        base_cfg = replace(base_cfg, batch_size=int(batch_size_override))
    if max_train_batches is not None:
        base_cfg = replace(base_cfg, max_train_batches=int(max_train_batches))
    if max_eval_batches is not None:
        base_cfg = replace(base_cfg, max_eval_batches=int(max_eval_batches))

    # Ensure output layout exists
    infer_root = os.path.join(output_root, "infer")
    train_root = os.path.join(output_root, "train")
    os.makedirs(output_root, exist_ok=True)

    summary_lines: list[str] = []

    def _run_case(case_name: str, case_cfg: AppConfig) -> None:
        if do_infer:
            out_dir = os.path.join(infer_root, case_name)
            acc = _infer_case(case_cfg, weights_path, out_dir)
            summary_lines.append(f"infer {case_name}: {acc:.3f}%")

        if do_train:
            out_dir = os.path.join(train_root, case_name)
            final_test_acc = _train_case(case_cfg, out_dir)
            summary_lines.append(f"train {case_name}: {final_test_acc:.3f}%")

        if do_finetune:
            out_dir = os.path.join(output_root, "finetune", case_name)
            _finetune_case(
                case_cfg,
                out_dir,
                weights_path,
                additional_epochs=finetune_additional_epochs,
                finetune_lr=finetune_lr,
            )
            summary_lines.append(f"finetune {case_name}: done")

    # 0) Base case: all parameters at 0.0 (inference, training, and finetune)
    if not skip_all_zeros:
        all_zeros = {k: 0.0 for k in include_keys}
        zero_case_cfg = _build_cfg_with_strengths(base_cfg, all_zeros)
        if no_quant_non_all_ones:
            zero_case_cfg = replace(
                zero_case_cfg, dac_bits=None, fourier_plane_bits=None, adc_bits=None
            )
        _run_case("all-zeros", zero_case_cfg)

    def _resolve_bits_for_quant_cases() -> tuple[int | None, int | None, int | None]:
        dac = all_ones_dac_bits if all_ones_dac_bits is not None else base_cfg.dac_bits
        fp = (
            all_ones_fourier_plane_bits
            if all_ones_fourier_plane_bits is not None
            else base_cfg.fourier_plane_bits
        )
        adc = all_ones_adc_bits if all_ones_adc_bits is not None else base_cfg.adc_bits
        return dac, fp, adc

    # 0b) Quantization-only case (distortions off, quant on)
    if include_quant_onehot:
        dac_b, fp_b, adc_b = _resolve_bits_for_quant_cases()
        quant_case_cfg = replace(
            base_cfg,
            driver_distortion_strength=0.0,
            pd_distortion_strength=0.0,
            tia_distortion_strength=0.0,
            mrm_power_distortion_strength=0.0,
            mrm_phase_distortion_strength=0.0,
            lens_distortion_strength=0.0,
            dac_bits=dac_b,
            fourier_plane_bits=fp_b,
            adc_bits=adc_b,
        )
        quant_case_name = f"quant{dac_b}-{fp_b}-{adc_b}"

        _run_case(quant_case_name, quant_case_cfg)

    # 1) Single-parameter at 1.0 (others at 0.0)
    keys_for_onehot = onehot_keys if onehot_keys is not None else list(include_keys)
    unknown = sorted(set(keys_for_onehot) - set(include_keys))
    if unknown:
        raise ValueError(f"Requested one-hot keys not in include_keys: {unknown}")

    for key in keys_for_onehot:
        kv = {k: (1.0 if k == key else 0.0) for k in include_keys}
        case_cfg = _build_cfg_with_strengths(base_cfg, kv)
        if no_quant_non_all_ones:
            case_cfg = replace(
                case_cfg, dac_bits=None, fourier_plane_bits=None, adc_bits=None
            )
        case_name = _format_case_name({key: 1.0})

        _run_case(case_name, case_cfg)

    # 2) All parameters at 1.0
    if not skip_all_ones:
        all_ones = {k: 1.0 for k in include_keys}
        case_cfg = _build_cfg_with_strengths(base_cfg, all_ones)
        # Optionally override quantization only for the all-ones case.
        # This is useful when the baseline/one-hot cases are run with clamp-only
        # (no quantization noise), but the full "all" system includes quant at
        # specific bitwidths (e.g. 4/4/6).
        all_ones_overrides = {}
        if all_ones_dac_bits is not None:
            all_ones_overrides["dac_bits"] = int(all_ones_dac_bits)
        if all_ones_fourier_plane_bits is not None:
            all_ones_overrides["fourier_plane_bits"] = int(all_ones_fourier_plane_bits)
        if all_ones_adc_bits is not None:
            all_ones_overrides["adc_bits"] = int(all_ones_adc_bits)
        if all_ones_overrides:
            case_cfg = replace(case_cfg, **all_ones_overrides)
        case_name = _format_case_name(all_ones)

        _run_case(case_name, case_cfg)

    # Write a brief summary file
    summary_path = os.path.join(output_root, "summary.txt")
    with open(summary_path, "a") as f:
        f.write("\n".join(summary_lines) + "\n")

    return base_cfg, weights_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate one-hot distortion strengths (1.0 each, then all at 1.0) "
            "for inference and training, using a pretrained ideal run."
        )
    )
    parser.add_argument(
        "--base-run",
        required=True,
        help="Path to the pretrained ideal run directory (contains final_config.yaml and fftconv_checkpoint.pth)",
    )
    parser.add_argument(
        "--output-root",
        required=False,
        help="Root directory to write results under (defaults next to base run)",
    )
    parser.add_argument(
        "--do-infer",
        action="store_true",
        help="Run inference cases (requires checkpoint in --base-run)",
    )
    parser.add_argument(
        "--do-train",
        action="store_true",
        help="Run training cases (from scratch with modified configs)",
    )
    parser.add_argument(
        "--do-finetune",
        action="store_true",
        help="Run fine-tuning cases initialized from the base checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs for training cases",
    )
    parser.add_argument(
        "--finetune-additional-epochs",
        type=int,
        default=0,
        help="Number of additional epochs to run for fine-tuning (added to base num_epochs)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Learning rate to use during fine-tuning (overrides config value)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for all cases (inference/training/finetune).",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug: cap training batches per epoch for all training/finetune cases.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: cap eval batches for inference and testing.",
    )
    parser.add_argument(
        "--only-onehot-keys",
        type=str,
        default="",
        help=(
            "Comma-separated subset of distortion keys to run as one-hot 1.0 cases. "
            "All-zeros/all-ones cases still use the full include set."
        ),
    )
    parser.add_argument(
        "--skip-all-zeros",
        action="store_true",
        help="Skip the baseline all-zeros case.",
    )
    parser.add_argument(
        "--skip-all-ones",
        action="store_true",
        help="Skip the all-ones case.",
    )
    parser.add_argument(
        "--no-quant-non-all-ones",
        action="store_true",
        help=(
            "Force baseline + one-hot cases to run with clamp-only (dac/fourier/adc bits set to None). "
            "This disables quantization noise while keeping the [0,1] range limiter."
        ),
    )
    parser.add_argument(
        "--include-quantization-onehot",
        action="store_true",
        help=(
            "Add an extra one-hot case where *only* quantization is enabled "
            "(distortions off, bits inherited from YAML unless --all-ones-*-bits overrides are set)."
        ),
    )
    parser.add_argument(
        "--all-ones-dac-bits",
        type=int,
        default=None,
        help="Override dac_bits only for the all-ones case (e.g. 4).",
    )
    parser.add_argument(
        "--all-ones-fourier-plane-bits",
        type=int,
        default=None,
        help="Override fourier_plane_bits only for the all-ones case (e.g. 4).",
    )
    parser.add_argument(
        "--all-ones-adc-bits",
        type=int,
        default=None,
        help="Override adc_bits only for the all-ones case (e.g. 6).",
    )

    args = parser.parse_args()

    base_run_dir = args.base_run
    # Default output root next to base run
    if args.output_root:
        output_root = args.output_root
    else:
        base_name = os.path.basename(os.path.normpath(base_run_dir))
        parent = os.path.dirname(os.path.normpath(base_run_dir))
        output_root = os.path.join(parent, f"{base_name}_onehots")

    include_keys = list(DISTORTION_STRENGTH_KEYS)

    onehot_keys: list[str] | None = None
    if args.only_onehot_keys:
        onehot_keys = [k.strip() for k in args.only_onehot_keys.split(",") if k.strip()]

    if not args.do_infer and not args.do_train and not args.do_finetune:
        # Default to all three if none selected
        args.do_infer = True
        args.do_train = True
        args.do_finetune = True

    run_one_hot_sweeps(
        base_run_dir=base_run_dir,
        output_root=output_root,
        do_infer=args.do_infer,
        do_train=args.do_train,
        do_finetune=args.do_finetune,
        epochs_override=args.epochs,
        finetune_additional_epochs=args.finetune_additional_epochs,
        finetune_lr=args.finetune_lr,
        no_quant_non_all_ones=bool(args.no_quant_non_all_ones),
        include_quant_onehot=bool(args.include_quantization_onehot),
        all_ones_dac_bits=args.all_ones_dac_bits,
        all_ones_fourier_plane_bits=args.all_ones_fourier_plane_bits,
        all_ones_adc_bits=args.all_ones_adc_bits,
        include_keys=include_keys,
        onehot_keys=onehot_keys,
        skip_all_zeros=bool(args.skip_all_zeros),
        skip_all_ones=bool(args.skip_all_ones),
        batch_size_override=args.batch_size,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()
