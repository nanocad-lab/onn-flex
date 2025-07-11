import argparse
import os
import yaml
from typing import Optional
from onn_train import train_onn_model
from onn_config import AppConfig


def parse_initial_args() -> tuple[Optional[str], Optional[str]]:
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

    args, _ = (
        parser.parse_known_args()
    )  # Use known_args to ignore other arguments for now
    return args.config_file, args.output_dir


def load_yaml_config(config_path: str) -> AppConfig:
    """Load application config from YAML file."""
    try:
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        # Filter out keys that don't exist in AppConfig
        valid_config = {
            k: v
            for k, v in yaml_data.items()
            if k in [field.name for field in AppConfig.__dataclass_fields__.values()]
        }

        return AppConfig(**valid_config)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return AppConfig()
    except Exception as e:
        print(f"Error loading config: {str(e)}. Using defaults.")
        return AppConfig()


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

    # # Simulation parameters
    # parser.add_argument(
    #     "--num-samples",
    #     type=int,
    #     default=yaml_config.num_samples,
    #     help="Number of samples",
    # )
    # parser.add_argument(
    #     "--num-iterations",
    #     type=int,
    #     default=yaml_config.num_iterations,
    #     help="Number of iterations",
    # )
    # parser.add_argument(
    #     "--random-seed",
    #     type=int,
    #     default=yaml_config.random_seed,
    #     help="Random seed for reproducibility",
    # )

    # JTC parameters
    parser.add_argument(
        "--jtc-half-size",
        type=int,
        default=yaml_config.jtc_half_size,
        help="Half of the total size of the JTC",
    )
    parser.add_argument(
        "--jtc-separation",
        type=int,
        default=yaml_config.jtc_separation,
        help="Separation between kernel and weight",
    )
    parser.add_argument(
        "--jtc-total-field",
        type=int,
        default=yaml_config.jtc_total_field,
        help="Total size of the JTC plane",
    )

    # Quantization parameters
    parser.add_argument(
        "--dac-bits",
        type=int,
        default=yaml_config.dac_bits,
        help="DAC bits for quantization",
    )
    parser.add_argument(
        "--adc-bits",
        type=int,
        default=yaml_config.adc_bits,
        help="ADC bits for quantization",
    )
    parser.add_argument(
        "--adc-scale-input",
        type=bool,
        default=yaml_config.adc_scale_input,
        help="Whether to scale ADC input",
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

    # PD/TIA parameters
    parser.add_argument(
        "--pd-tia-distortion-strength",
        type=float,
        default=yaml_config.pd_tia_distortion_strength,
        help="PD/TIA distortion strength",
    )
    parser.add_argument(
        "--pd-tia-distortion-data-path",
        type=str,
        default=yaml_config.pd_tia_distortion_data_path,
        help="Path to PD/TIA distortion data CSV",
    )
    parser.add_argument(
        "--pd-tia-distortion-polyfit-order",
        type=int,
        default=yaml_config.pd_tia_distortion_polyfit_order,
        help="Polynomial fit order for PD/TIA distortion",
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

    args = parser.parse_args()

    return AppConfig(**vars(args))


def save_config(config: AppConfig, output_dir: str) -> str:
    """Save the final configuration to output_dir/config.yaml."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "config.yaml")

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def main():
    config_path, output_dir = parse_initial_args()

    if output_dir is None:
        raise ValueError("Output directory is required")

    # Step 2: Load config from YAML if provided, otherwise use defaults
    yaml_config = AppConfig()  # Start with defaults
    if config_path:
        yaml_config = load_yaml_config(config_path)
        # Update config_file to preserve the path
        yaml_config.config_file = config_path

    # Step 3: Parse CLI args using YAML values as defaults
    final_config = parse_cli_args(yaml_config)

    # Ensure we preserve the config_file path from initial parsing
    if config_path and not final_config.config_file:
        final_config.config_file = config_path

    # Step 4: Save the final config to output_dir/config.yaml
    saved_path = save_config(final_config, final_config.output_dir)

    print(f"Saved config to output directory: {saved_path}")

    train_onn_model(final_config)


if __name__ == "__main__":
    main()
