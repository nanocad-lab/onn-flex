import argparse
import os
import yaml
from dataclasses import replace
from onn_train import train_onn_model
from onn_config import AppConfig, load_app_config_from_yaml


def parse_initial_args() -> tuple[str | None, str | None]:
    """Parse just the config file and output directory from command line."""
    parser = argparse.ArgumentParser(
        description="Parse config location and output directory"
    )
    parser.add_argument(
        "-c", "--config-file", type=str, help="Path to YAML config file"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory for saving config"
    )

    args, _ = parser.parse_known_args()
    return args.config_file, args.output_dir


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    val = v.strip().lower()
    if val in {"yes", "true", "t", "1", "y"}:
        return True
    if val in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _optional_int(v: str | int | None) -> int | None:
    if v is None or isinstance(v, int):
        return v
    val = v.strip().lower()
    if val in {"none", "null"}:
        return None
    return int(v)


def parse_cli_args(yaml_config: AppConfig) -> AppConfig:
    """Parse command line arguments that can override YAML values."""
    parser = argparse.ArgumentParser(description="Application with YAML config")

    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default=yaml_config.config_file,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=yaml_config.output_dir,
        help="Output directory for saving config",
    )

    # JTC parameters
    parser.add_argument(
        "--input-length",
        type=int,
        default=yaml_config.input_length,
        help="Length of input signal",
    )
    parser.add_argument(
        "--kernel-length",
        type=int,
        default=yaml_config.kernel_length,
        help="Length of kernel/weight",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=yaml_config.output_length,
        help="Length of output (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--jtc-separation",
        type=int,
        default=yaml_config.jtc_separation,
        help="Separation between kernel and signal",
    )
    parser.add_argument(
        "--jtc-total-field",
        type=int,
        default=yaml_config.jtc_total_field,
        help="Total size of the JTC plane (lens size)",
    )
    parser.add_argument(
        "--lens-distortion-strength",
        type=float,
        default=yaml_config.lens_distortion_strength,
        help="Lens distortion strength in [0,1] (0=ideal, 1=full lens_coefs).",
    )
    parser.add_argument(
        "--lens-legendre-order",
        type=int,
        default=yaml_config.lens_legendre_order,
        help="Max Legendre order for the lens model (uses orders 1..N).",
    )
    parser.add_argument(
        "--lens-coefs",
        type=float,
        nargs="+",
        default=yaml_config.lens_coefs,
        help="Lens Legendre coefficients (for orders 1..lens_legendre_order).",
    )

    # Quantization parameters
    parser.add_argument(
        "--dac-bits",
        type=_optional_int,
        default=yaml_config.dac_bits,
        help="DAC bits for quantization; use 'none' to disable DAC quantization",
    )
    parser.add_argument(
        "--adc-bits",
        type=_optional_int,
        default=yaml_config.adc_bits,
        help="ADC bits for quantization; use 'none' to disable ADC quantization",
    )
    parser.add_argument(
        "--scale-output",
        type=str,
        default=yaml_config.scale_output,
        help="Where to scale output ['none', 'pd', 'adc']",
    )

    # Driver parameters
    parser.add_argument(
        "--driver-distortion-strength",
        type=float,
        default=yaml_config.driver_distortion_strength,
        help="Driver distortion strength",
    )
    parser.add_argument(
        "--driver-distortion-data-path",
        type=str,
        default=yaml_config.driver_distortion_data_path,
        help="Path to driver distortion data CSV",
    )
    parser.add_argument(
        "--driver-distortion-polyfit-order",
        type=int,
        default=yaml_config.driver_distortion_polyfit_order,
        help="Polynomial fit order for driver distortion",
    )

    # PD parameters
    parser.add_argument(
        "--pd-distortion-strength",
        type=float,
        default=yaml_config.pd_distortion_strength,
        help="PD distortion strength",
    )
    parser.add_argument(
        "--pd-noise-w",
        type=float,
        default=yaml_config.pd_noise_w,
        help="Additive PD input-referred noise (Watts), sampled per input sample.",
    )
    parser.add_argument(
        "--pd-distortion-data-path",
        type=str,
        default=yaml_config.pd_distortion_data_path,
        help="Path to PD distortion data CSV",
    )
    parser.add_argument(
        "--pd-distortion-polyfit-order",
        type=int,
        default=yaml_config.pd_distortion_polyfit_order,
        help="Polynomial fit order for PD distortion",
    )

    # TIA parameters
    parser.add_argument(
        "--tia-distortion-strength",
        type=float,
        default=yaml_config.tia_distortion_strength,
        help="TIA distortion strength",
    )
    parser.add_argument(
        "--tia-distortion-data-path",
        type=str,
        default=yaml_config.tia_distortion_data_path,
        help="Path to TIA distortion data CSV",
    )
    parser.add_argument(
        "--tia-distortion-polyfit-order",
        type=int,
        default=yaml_config.tia_distortion_polyfit_order,
        help="Polynomial fit order for TIA distortion",
    )

    # MRM power parameters
    parser.add_argument(
        "--mrm-power-distortion-strength",
        type=float,
        default=yaml_config.mrm_power_distortion_strength,
        help="MRM power distortion strength",
    )
    parser.add_argument(
        "--mrm-power-data-path",
        type=str,
        default=yaml_config.mrm_power_data_path,
        help="Path to MRM power data CSV",
    )
    parser.add_argument(
        "--mrm-power-polyfit-order",
        type=int,
        default=yaml_config.mrm_power_polyfit_order,
        help="Polynomial fit order for MRM power",
    )
    parser.add_argument(
        "--mrm-power-gain",
        type=float,
        default=yaml_config.mrm_power_gain,
        help="Gain applied to the MRM power output in Watts",
    )

    # MRM phase parameters
    parser.add_argument(
        "--mrm-phase-distortion-strength",
        type=float,
        default=yaml_config.mrm_phase_distortion_strength,
        help="MRM phase distortion strength",
    )
    parser.add_argument(
        "--mrm-phase-data-path",
        type=str,
        default=yaml_config.mrm_phase_data_path,
        help="Path to MRM phase data CSV",
    )
    parser.add_argument(
        "--mrm-phase-polyfit-order",
        type=int,
        default=yaml_config.mrm_phase_polyfit_order,
        help="Polynomial fit order for MRM phase",
    )
    parser.add_argument(
        "--laser-rin-db",
        type=float,
        default=yaml_config.laser_rin_db,
        help="Per-shot global laser relative intensity noise (RIN) in dB.",
    )

    # Conv backend selection
    parser.add_argument(
        "--conv-backend",
        type=str,
        default=yaml_config.conv_backend,
        choices=["pytorch", "jtc_ideal", "jtc_emulation"],
        help=(
            "Convolution backend: 'pytorch' (PyTorch conv2d), "
            "'jtc_ideal' (ideal JTC optical model), or "
            "'jtc_emulation' (full hardware JTC emulation)"
        ),
    )
    parser.add_argument(
        "--fourier-plane-bits",
        type=_optional_int,
        default=yaml_config.fourier_plane_bits,
        help="Quantization bits used in the Fourier plane; use 'none' to disable it",
    )

    # Optional activation normalization inside identical blocks
    parser.add_argument(
        "--normalize-blocks",
        dest="normalize_blocks",
        action="store_true",
        default=yaml_config.normalize_blocks,
        help="Normalize activations after each identical block",
    )
    parser.add_argument(
        "--enable-jtc-batched-fast-path",
        dest="enable_jtc_batched_fast_path",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.enable_jtc_batched_fast_path,
        help=(
            "Enable batched JTC calls to reduce launch overhead (true/false)"
        ),
    )
    parser.add_argument(
        "--enable-jtc-ideal-fused-transfer",
        dest="enable_jtc_ideal_fused_transfer",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.enable_jtc_ideal_fused_transfer,
        help=(
            "Use fused ideal JTC transfer functions when the configured "
            "transfer curves are ideal (true/false)"
        ),
    )
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.show_progress,
        help="Show tqdm training progress bars (true/false)",
    )

    # ------------------------------------------------------------------
    #  Training / model-related CLI overrides
    # ------------------------------------------------------------------
    parser.add_argument(
        "--num-identical-layers",
        dest="num_identical_layers",
        type=int,
        default=yaml_config.num_identical_layers,
        help="Number of identical conv blocks after the 2nd layer",
    )
    parser.add_argument(
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=yaml_config.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=yaml_config.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=yaml_config.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-train-batches",
        dest="max_train_batches",
        type=int,
        default=yaml_config.max_train_batches,
        help="Debug: limit number of training batches per epoch",
    )
    parser.add_argument(
        "--max-eval-batches",
        dest="max_eval_batches",
        type=int,
        default=yaml_config.max_eval_batches,
        help="Debug: limit number of eval/inference batches",
    )
    parser.add_argument(
        "--eval-only",
        dest="eval_only",
        action="store_true",
        default=yaml_config.eval_only,
        help="Skip training and only run evaluation using --pretrained-weights",
    )
    parser.add_argument(
        "--pretrained-weights",
        dest="pretrained_weights",
        type=str,
        default=yaml_config.pretrained_weights,
        help="Path to pretrained weights (.pth) for evaluation or resume",
    )
    parser.add_argument(
        "--run-pretrain-tests",
        dest="run_pretrain_tests",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.run_pretrain_tests,
        help="Run pretrain tests (true/false)",
    )
    parser.add_argument(
        "--pretrain-tests-only",
        dest="pretrain_tests_only",
        action="store_true",
        default=yaml_config.pretrain_tests_only,
        help="Run only the pretrain tests/plots and exit",
    )

    args = parser.parse_args()

    return replace(yaml_config, **vars(args))


def save_config(config: AppConfig, output_dir: str) -> str:
    """Save the final configuration to output_dir/config.yaml."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "config.yaml")

    with open(output_path, "w") as f:
        yaml.safe_dump(vars(config), f, default_flow_style=False)

    return output_path


def main():
    config_path, output_dir = parse_initial_args()

    if output_dir is None:
        raise ValueError("Output directory is required")

    # Step 2: Load config from YAML if provided, otherwise use defaults
    yaml_config = AppConfig()
    if config_path:
        yaml_config = load_app_config_from_yaml(config_path)
        yaml_config.config_file = config_path

    # Step 3: Parse CLI args using YAML values as defaults
    final_config = parse_cli_args(yaml_config)

    # Step 4: Save the final config to output_dir/config.yaml
    saved_path = save_config(final_config, final_config.output_dir)

    print(f"Saved config to output directory: {saved_path}")

    train_onn_model(final_config)


if __name__ == "__main__":
    main()
