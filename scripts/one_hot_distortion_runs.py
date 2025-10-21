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
from typing import Dict, List, Tuple, Optional

# Ensure repository root is importable when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import yaml

from onn_config import AppConfig
from onn_inference import load_config_from_yaml, run_inference
from onn_train import train_onn_model


# Distortion strength fields to toggle. Excludes pd_tia_distortion_strength
# because the current JTC pipeline uses separate PD and TIA stages.
DISTORTION_STRENGTH_KEYS: List[str] = [
    "driver_distortion_strength",
    "pd_distortion_strength",
    "tia_distortion_strength",
    "mrm_power_distortion_strength",
    "mrm_phase_distortion_strength",
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
        # In case `config.yaml` contains tagged object YAML, we parse minimally
        # by loading raw and extracting only AppConfig fields.
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            valid = {
                k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__
            }
            return AppConfig(**valid)
    except Exception:
        pass
    raise FileNotFoundError(
        f"Could not load a usable config from {base_run_dir}. Expected final_config.yaml or config.yaml"
    )


def _save_config(cfg: AppConfig, out_dir: str, filename: str = "config.yaml") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False)


def _build_cfg_with_strengths(base: AppConfig, kv: Dict[str, float]) -> AppConfig:
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
    finetune_lr: Optional[float] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Enable fine-tuning from checkpoint
    epochs_for_finetune = (
        int(additional_epochs) if additional_epochs > 0 else cfg.num_epochs
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


def _format_case_name(kv: Dict[str, float]) -> str:
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
    epochs_override: Optional[int],
    finetune_additional_epochs: int,
    finetune_lr: Optional[float],
    include_keys: List[str],
) -> Tuple[AppConfig, str]:
    base_cfg = _read_base_config(base_run_dir)
    weights_path = os.path.join(base_run_dir, "fftconv_checkpoint.pth")
    if not os.path.exists(weights_path) and do_infer:
        raise FileNotFoundError(
            f"Pretrained checkpoint not found for inference: {weights_path}"
        )

    # Optionally override epochs for training runs
    if epochs_override is not None:
        base_cfg = replace(base_cfg, num_epochs=int(epochs_override))

    # Ensure output layout exists
    infer_root = os.path.join(output_root, "infer")
    train_root = os.path.join(output_root, "train")
    os.makedirs(output_root, exist_ok=True)

    summary_lines: List[str] = []

    # 0) Base case: all parameters at 0.0 (inference, training, and finetune)
    all_zeros = {k: 0.0 for k in include_keys}
    zero_case_cfg = _build_cfg_with_strengths(base_cfg, all_zeros)
    zero_case_name = "all-zeros"

    if do_infer:
        out_dir = os.path.join(infer_root, zero_case_name)
        acc = _infer_case(zero_case_cfg, weights_path, out_dir)
        summary_lines.append(f"infer {zero_case_name}: {acc:.3f}%")

    if do_train:
        out_dir = os.path.join(train_root, zero_case_name)
        final_test_acc = _train_case(zero_case_cfg, out_dir)
        summary_lines.append(f"train {zero_case_name}: {final_test_acc:.3f}%")

    if do_finetune:
        out_dir = os.path.join(output_root, "finetune", zero_case_name)
        _finetune_case(
            zero_case_cfg,
            out_dir,
            weights_path,
            additional_epochs=finetune_additional_epochs,
            finetune_lr=finetune_lr,
        )
        summary_lines.append(f"finetune {zero_case_name}: done")

    # 1) Single-parameter at 1.0 (others at 0.0)
    for key in include_keys:
        kv = {k: (1.0 if k == key else 0.0) for k in include_keys}
        case_cfg = _build_cfg_with_strengths(base_cfg, kv)
        case_name = _format_case_name({key: 1.0})

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

    # 2) All parameters at 1.0
    all_ones = {k: 1.0 for k in include_keys}
    case_cfg = _build_cfg_with_strengths(base_cfg, all_ones)
    case_name = _format_case_name(all_ones)

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
        "--include-pd-tia",
        action="store_true",
        help=(
            "Also iterate pd_tia_distortion_strength (not used in current pipeline)."
        ),
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
    if args.include_pd_tia:
        include_keys.append("pd_tia_distortion_strength")

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
        include_keys=include_keys,
    )


if __name__ == "__main__":
    main()
