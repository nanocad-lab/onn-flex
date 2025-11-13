"""quantization_sweep.py

A script to train the ONN model across different quantization bit configurations
and backends to compare performance.

This script sweeps through various combinations of:
- dac_bits: Input quantization (DAC)
- adc_bits: Output quantization (ADC)
- fourier_plane_bits: Fourier plane quantization
- conv_backend: fourier or jtc_emulation

Results are saved to separate directories for analysis and comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import yaml
import json
from datetime import datetime

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from onn_config import AppConfig
from onn_train import train_onn_model

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs" / "quantization_sweep"

# Quantization configurations to test
# Format: (dac_bits, adc_bits, fourier_plane_bits)
QUANTIZATION_CONFIGS: List[Tuple[int | None, int | None, int | None]] = [
    # No quantization baseline
    (None, None, None),
    # Low precision
    (2, 4, 4),
    (3, 4, 4),
    # Medium precision
    (4, 6, 6),
    (4, 8, 6),
    # High precision
    (8, 8, 8),
    # Mixed precision experiments
    (4, 6, 8),
    (8, 6, 4),
    # Extreme low precision
    (1, 2, 2),
]

# Backends to test
BACKENDS: List[str] = ["fourier", "jtc_emulation"]

# Default base configuration
DEFAULT_BASE_CONFIG = {
    "jtc_half_size": 8,
    "jtc_separation": 8,
    "jtc_total_field": 48,
    "scale_output": "adc",
    "driver_distortion_data_path": "./component_data/driver_sim_data.csv",
    "driver_distortion_polyfit_order": 3,
    "driver_distortion_strength": 0.0,
    "mrm_phase_data_path": "./component_data/mrm_phase_sim_data.csv",
    "mrm_phase_distortion_strength": 0.0,
    "mrm_phase_polyfit_order": None,
    "mrm_power_data_path": "./component_data/mrm_pwr_w_sim_data.csv",
    "mrm_power_distortion_strength": 0.0,
    "mrm_power_polyfit_order": None,
    "pd_tia_distortion_data_path": "./component_data/pd_tia_corr_sim_data.csv",
    "pd_tia_distortion_polyfit_order": None,
    "pd_tia_distortion_strength": 0.0,
    "pd_distortion_data_path": "./component_data/pd_sim_data.csv",
    "pd_distortion_polyfit_order": None,
    "pd_distortion_strength": 0.0,
    "tia_distortion_data_path": "./component_data/tia_sim_data.csv",
    "tia_distortion_polyfit_order": None,
    "tia_distortion_strength": 0.0,
    "ler_std_dev": 0.0,
    "num_identical_layers": 1,
    "num_epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 128,
    "eval_only": False,
    "pretrained_weights": "",
    "run_pretrain_tests": False,
    "loss": 0.96,
    "quantizer": "ste_clipped",
}


# -----------------------------------------------------------------------------
#  Helper utilities
# -----------------------------------------------------------------------------


def format_bits(bits: int | None) -> str:
    """Format bit value for display and filenames."""
    return "none" if bits is None else str(bits)


def create_config(
    backend: str,
    dac_bits: int | None,
    adc_bits: int | None,
    fourier_plane_bits: int | None,
    output_dir: Path,
    base_config: Dict = None,
) -> AppConfig:
    """Create an AppConfig with the specified quantization settings."""
    if base_config is None:
        base_config = DEFAULT_BASE_CONFIG.copy()

    # Override quantization and backend settings
    config_dict = base_config.copy()
    config_dict.update(
        {
            "conv_backend": backend,
            "dac_bits": dac_bits,
            "adc_bits": adc_bits,
            "fourier_plane_bits": fourier_plane_bits,
            "output_dir": str(output_dir),
        }
    )

    return AppConfig(**config_dict)


def run_name(
    backend: str, dac_bits: int | None, adc_bits: int | None, fourier_plane_bits: int | None
) -> str:
    """Generate a descriptive name for this run."""
    dac_str = format_bits(dac_bits)
    adc_str = format_bits(adc_bits)
    fourier_str = format_bits(fourier_plane_bits)
    return f"{backend}_dac{dac_str}_adc{adc_str}_fourier{fourier_str}"


# -----------------------------------------------------------------------------
#  Main entry-point
# -----------------------------------------------------------------------------


def main() -> None:
    """Run quantization sweep across backends and bit configurations."""

    parser = argparse.ArgumentParser(
        description="Train ONN model across different quantization configurations"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        help="Path to base YAML config file (optional, uses defaults if not provided)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["fourier", "jtc_emulation"],
        default=BACKENDS,
        help="Backends to test (default: both)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with fewer epochs (5 instead of 20)",
    )
    parser.add_argument(
        "--pretrain-tests",
        action="store_true",
        help="Run pretrain tests for each configuration",
    )
    parser.add_argument(
        "--configs",
        type=str,
        help="Comma-separated list of config indices to run (e.g., '0,1,2' for first 3 configs)",
    )
    args = parser.parse_args()

    # Load base config if provided
    base_config = DEFAULT_BASE_CONFIG.copy()
    if args.base_config:
        config_path = Path(args.base_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Base config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            yaml_cfg = yaml.safe_load(fh)
            base_config.update(yaml_cfg)

    # Override epochs if quick mode
    if args.quick:
        base_config["num_epochs"] = 5

    # Enable pretrain tests if requested
    if args.pretrain_tests:
        base_config["run_pretrain_tests"] = True

    # Filter configs if specified
    configs_to_run = QUANTIZATION_CONFIGS
    if args.configs:
        indices = [int(i) for i in args.configs.split(",")]
        configs_to_run = [QUANTIZATION_CONFIGS[i] for i in indices]

    # Ensure the root output directory exists
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Create summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RUNS_DIR / f"summary_{timestamp}.json"
    results = []

    print("\n" + "=" * 80)
    print("QUANTIZATION SWEEP")
    print("=" * 80)
    print(f"Backends: {args.backends}")
    print(f"Configurations: {len(configs_to_run)}")
    print(f"Total runs: {len(args.backends) * len(configs_to_run)}")
    print(f"Output directory: {RUNS_DIR}")
    print("=" * 80 + "\n")

    # Run experiments
    total_runs = len(args.backends) * len(configs_to_run)
    current_run = 0

    for backend in args.backends:
        for dac_bits, adc_bits, fourier_plane_bits in configs_to_run:
            current_run += 1

            # Create run name and output directory
            run_id = run_name(backend, dac_bits, adc_bits, fourier_plane_bits)
            output_dir = RUNS_DIR / run_id
            output_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Run {current_run}/{total_runs}: {run_id}")
            print("=" * 80)
            print(f"Backend: {backend}")
            print(f"DAC bits: {format_bits(dac_bits)}")
            print(f"ADC bits: {format_bits(adc_bits)}")
            print(f"Fourier plane bits: {format_bits(fourier_plane_bits)}")
            print(f"Output directory: {output_dir}")
            print("=" * 80 + "\n")

            # Create config
            config = create_config(
                backend, dac_bits, adc_bits, fourier_plane_bits, output_dir, base_config
            )

            # Save config to output directory
            config_save_path = output_dir / "config.yaml"
            config_dict = {
                f.name: getattr(config, f.name)
                for f in config.__dataclass_fields__.values()
            }
            with config_save_path.open("w", encoding="utf-8") as fh:
                yaml.dump(config_dict, fh, default_flow_style=False)

            # Train model
            try:
                train_result = train_onn_model(config)

                # Record results
                result = {
                    "run_id": run_id,
                    "backend": backend,
                    "dac_bits": dac_bits,
                    "adc_bits": adc_bits,
                    "fourier_plane_bits": fourier_plane_bits,
                    "output_dir": str(output_dir),
                    "status": "success",
                }
                results.append(result)

                print("\n" + "-" * 80)
                print(f"✓ Completed run {current_run}/{total_runs}: {run_id}")
                print("-" * 80 + "\n")

            except Exception as e:
                print(f"\n✗ Error in run {run_id}: {e}\n")
                result = {
                    "run_id": run_id,
                    "backend": backend,
                    "dac_bits": dac_bits,
                    "adc_bits": adc_bits,
                    "fourier_plane_bits": fourier_plane_bits,
                    "output_dir": str(output_dir),
                    "status": "failed",
                    "error": str(e),
                }
                results.append(result)

            # Save intermediate results
            with summary_file.open("w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("QUANTIZATION SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total runs: {total_runs}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
