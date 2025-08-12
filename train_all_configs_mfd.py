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

# Standard library
from pathlib import Path
from typing import List

import yaml

from onn_config import AppConfig
from onn_train import train_onn_model

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
RUNS_DIR = Path(__file__).resolve().parent / "runs_mfd"

# Explicit list to guarantee execution order.
CONFIG_FILES: List[str] = [
    "config_ideal.yaml",
    # "config_pytorch_conv.yaml",
    # "config_worst.yaml",
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
    """Iterate over the four configs and train a model for each one."""

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
        #  Kick off training
        # ------------------------------------------------------------------
        train_onn_model(cfg)

        print("\n" + "-" * 80)
        print(f"Finished training {cfg_name}\n")


if __name__ == "__main__":
    main()
