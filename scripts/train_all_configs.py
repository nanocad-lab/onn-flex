"""train_all_configs.py

A convenience script to sequentially train the ONN model using the four
pre-defined YAML configurations located under the `configs/` directory::

    configs/config_ideal.yaml
    configs/config_pytorch_conv.yaml
    configs/config_worst.yaml
    configs/config_worst_corr.yaml

Each run writes its outputs to a dedicated sub-directory under `runs/`
that matches the config’s base-name (e.g. `runs/config_ideal/`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import argparse
import yaml

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from onn_config import AppConfig
from onn_train import train_onn_model

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "runs"

# Explicit list to guarantee execution order.
CONFIG_FILES: List[str] = [
    "config_ideal.yaml",
    "config_pytorch_conv.yaml",
    "config_worst.yaml",
    "config_worst_corr.yaml",
]


# -----------------------------------------------------------------------------
#  Helper utilities
# -----------------------------------------------------------------------------


def _load_app_config(path: Path) -> AppConfig:
    """Load YAML at *path* and convert it into an :class:`AppConfig`."""

    with path.open("r", encoding="utf-8") as fh:
        yaml_cfg = yaml.safe_load(fh)

    # The YAML may omit optional fields; ** unpacking handles this gracefully.
    return AppConfig(**yaml_cfg)


# -----------------------------------------------------------------------------
#  Main entry-point
# -----------------------------------------------------------------------------


def main() -> None:
    """Iterate over the configs and run training (or pretrain tests only)."""

    parser = argparse.ArgumentParser(
        description="Train all configured runs or run only pretrain tests"
    )
    parser.add_argument(
        "--pretrain-tests-only",
        action="store_true",
        help="Run only the pretrain tests/plots for each config and exit",
    )
    args = parser.parse_args()

    # Ensure the root output directory exists.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_name in CONFIG_FILES:
        cfg_path = CONFIG_DIR / cfg_name
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Expected config file not found: {cfg_path}")

        # ------------------------------------------------------------------
        #  Construct the application configuration for this run
        # ------------------------------------------------------------------
        cfg = _load_app_config(cfg_path)
        cfg.config_file = str(cfg_path)

        # Place outputs under runs/<config_base>/
        out_dir = RUNS_DIR / cfg_path.stem
        cfg.output_dir = str(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Training with configuration: {cfg_name}")
        print(f"Output directory: {out_dir}")
        print("=" * 80 + "\n")

        # ------------------------------------------------------------------
        #  Configure mode and kick off training
        # ------------------------------------------------------------------
        if args.pretrain_tests_only:
            cfg.pretrain_tests_only = True
            cfg.run_pretrain_tests = True
        train_onn_model(cfg)

        print("\n" + "-" * 80)
        print(f"Finished training {cfg_name}\n")


if __name__ == "__main__":
    main()

