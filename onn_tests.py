from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from onn_config import AppConfig
from onn_component import get_ideal_degree, get_coeffs, Driver, PD_TIA, MRM, JTC


def _plot_fit(x: np.ndarray, y: np.ndarray, y_lin: np.ndarray, y_poly: np.ndarray, title: str, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, "k.", label="CSV data")
    plt.plot(x, y_lin, "b-", label="Linear fit")
    plt.plot(x, y_poly, "r--", label="Poly fit")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _sweep_and_plot(csv_path: str, degree: int, output_dir: str, tag: str) -> None:
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}. Skipping {tag} plot.")
        return

    data = pd.read_csv(csv_path)
    x = data["input"].values
    y = data["output"].values

    # Linear behaviour (degree 1 fit)
    lin_coeff = np.polyfit(x, y, 1)
    y_lin = np.polyval(lin_coeff, x)

    # Polynomial fit
    poly_coeff = get_coeffs(csv_path, degree)
    y_poly = np.polyval(poly_coeff, x)

    plot_path = os.path.join(output_dir, f"{tag}_fit.png")
    _plot_fit(x, y, y_lin, y_poly, f"{tag} distortion fit", plot_path)
    print(f"[TEST] Saved plot {plot_path}")


def _range_check_jtc(config: AppConfig, output_dir: str) -> None:
    """Run a quick JTC forward pass to ensure outputs are finite and within a sensible range."""
    # Make dummy components (using CSV-driven poly fits)
    driver = Driver(config)
    pd_tia = PD_TIA(config)
    mrm = MRM(config)
    jtc = JTC(config, driver, mrm, pd_tia)

    # Random batch of signals/kernels in [-1, 1]
    torch.manual_seed(0)
    signal = torch.rand(8) * 2 - 1
    kernel = torch.rand(8) * 2 - 1
    out = jtc(signal, kernel)

    if not torch.isfinite(out).all():
        raise ValueError("JTC output contains NaNs/Infs")
    out_min, out_max = out.min().item(), out.max().item()
    print(f"[TEST] JTC output range: {out_min:.3f} .. {out_max:.3f}")

    # Save to disk for reference
    np.save(os.path.join(output_dir, "jtc_test_output.npy"), out.cpu().numpy())


def run_pretrain_tests(config: AppConfig) -> None:
    """Generate plots + basic checks before starting (or instead of) training."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Driver
    if config.driver_distortion_data_path and os.path.exists(config.driver_distortion_data_path):
        deg = config.driver_distortion_polyfit_order or get_ideal_degree(config.driver_distortion_data_path)
        _sweep_and_plot(config.driver_distortion_data_path, deg, config.output_dir, "driver")

    # PD/TIA
    if config.pd_tia_distortion_data_path and os.path.exists(config.pd_tia_distortion_data_path):
        deg = config.pd_tia_distortion_polyfit_order or get_ideal_degree(config.pd_tia_distortion_data_path)
        _sweep_and_plot(config.pd_tia_distortion_data_path, deg, config.output_dir, "pd_tia")

    # MRM power
    if config.mrm_power_data_path and os.path.exists(config.mrm_power_data_path):
        deg = config.mrm_power_polyfit_order or get_ideal_degree(config.mrm_power_data_path)
        _sweep_and_plot(config.mrm_power_data_path, deg, config.output_dir, "mrm_power")

    # MRM phase
    if config.mrm_phase_data_path and os.path.exists(config.mrm_phase_data_path):
        deg = config.mrm_phase_polyfit_order or get_ideal_degree(config.mrm_phase_data_path)
        _sweep_and_plot(config.mrm_phase_data_path, deg, config.output_dir, "mrm_phase")

    # Quick JTC sanity check
    try:
        _range_check_jtc(config, config.output_dir)
    except Exception as e:
        print(f"[ERROR] JTC range check failed: {e}") 