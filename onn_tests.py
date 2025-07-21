from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from onn_config import AppConfig
from onn_component import get_ideal_degree, get_coeffs, Driver, PD_TIA, MRM, JTC


def _plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_ref: np.ndarray,
    y_poly: np.ndarray,
    ref_order: int,
    poly_order: int,
    title: str,
    save_path: str,
) -> None:
    """Plot raw CSV data against a reference polynomial fit and the higher-order distortion fit.

    Args:
        x: Input values from the CSV.
        y: Output values from the CSV.
        y_ref: Values predicted by the reference fit (typically linear).
        y_poly: Values predicted by the polynomial distortion fit.
        ref_order: Order of the reference polynomial used for *y_ref*.
        poly_order: Order of the distortion polynomial used for *y_poly*.
        title: Title for the plot.
        save_path: Where to save the PNG.
    """
    # Compute R^2 for the distortion polynomial fit
    ss_res = np.sum((y - y_poly) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, "k.", label="CSV data")
    plt.plot(x, y_ref, "b-", label=f"Ref fit (deg {ref_order})")
    plt.plot(
        x,
        y_poly,
        "r--",
        label=f"Poly fit (deg {poly_order}, R²={r_squared:.5f})",
    )
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _sweep_and_plot(
    csv_path: str,
    degree: int,
    output_dir: str,
    tag: str,
    ref_degree: int = 1,
) -> None:
    """Load a CSV I/O sweep, fit polynomials, and save a comparison plot.

    Args:
        csv_path: Path to the CSV containing *input* and *output* columns.
        degree: Order of the main polynomial used to model distortion.
        output_dir: Where to write the PNG plot.
        tag: Descriptive tag inserted into the file name and plot title.
        ref_degree: Order of the reference polynomial (defaults to 1 → linear).
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}. Skipping {tag} plot.")
        return

    data = pd.read_csv(csv_path)
    x = data["input"].values
    y = data["output"].values

    # Reference behaviour (could be linear or higher order)
    ref_coeff = np.polyfit(x, y, ref_degree)
    y_ref = np.polyval(ref_coeff, x)

    # Distortion polynomial fit
    poly_coeff = get_coeffs(csv_path, degree)
    y_poly = np.polyval(poly_coeff, x)

    # Save as PDF rather than PNG
    plot_path = os.path.join(output_dir, f"{tag}_fit.pdf")
    _plot_fit(
        x,
        y,
        y_ref,
        y_poly,
        ref_degree,
        degree,
        f"{tag} distortion fit",
        plot_path,
    )
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


def _plot_array(arr: torch.Tensor, title: str, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    arr_np = arr.detach().cpu().numpy()
    plt.plot(arr_np)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _stage_plots_jtc(config: AppConfig, output_dir: str) -> None:
    """Plot intermediate JTC stages and compare with PyTorch conv."""
    driver = Driver(config)
    pd_tia = PD_TIA(config)
    mrm = MRM(config)
    jtc = JTC(config, driver, mrm, pd_tia)

    torch.manual_seed(1)
    signal = torch.rand(8)
    kernel = torch.rand(8)

    input_plane, _, N = jtc.generate_input_plane(signal, kernel)
    jft = jtc.post_fft(input_plane)
    jps = jtc.post_output_distortion(jft)
    out = jtc.final_output(jps, N)

    # --- Combine stage plots into a single multi-panel PDF ---
    conv_out = torch.nn.functional.conv1d(
        signal.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel.shape[0] // 2
    )[0, 0, : out.shape[0]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top-left: Input plane magnitude
    axes[0, 0].plot(torch.abs(input_plane).detach().cpu().numpy())
    axes[0, 0].set_title("Input plane")

    # Top-right: After FFT magnitude
    axes[0, 1].plot(torch.abs(jft).detach().cpu().numpy())
    axes[0, 1].set_title("Post FFT")

    # Bottom-left: After output distortion
    axes[1, 0].plot(jps.detach().cpu().numpy())
    axes[1, 0].set_title("Post output distortion")

    # Bottom-right: Final output vs. PyTorch conv reference
    axes[1, 1].plot(out.detach().cpu().numpy(), label="jtc")
    axes[1, 1].plot(conv_out.detach().cpu().numpy(), label="torch_conv")
    axes[1, 1].set_title("Final output comparison")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stage_plots.pdf"))
    plt.close(fig)


def run_pretrain_tests(config: AppConfig) -> None:
    """Generate plots + basic checks before starting (or instead of) training."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Driver
    if config.driver_distortion_data_path and os.path.exists(
        config.driver_distortion_data_path
    ):
        deg = config.driver_distortion_polyfit_order or get_ideal_degree(
            config.driver_distortion_data_path
        )
        _sweep_and_plot(
            config.driver_distortion_data_path, deg, config.output_dir, "driver"
        )

    # PD/TIA (use 2nd-order reference instead of linear)
    if config.pd_tia_distortion_data_path and os.path.exists(
        config.pd_tia_distortion_data_path
    ):
        deg = config.pd_tia_distortion_polyfit_order or get_ideal_degree(
            config.pd_tia_distortion_data_path
        )
        _sweep_and_plot(
            config.pd_tia_distortion_data_path,
            deg,
            config.output_dir,
            "pd_tia",
            ref_degree=2,
        )

    # MRM power
    if config.mrm_power_data_path and os.path.exists(config.mrm_power_data_path):
        deg = config.mrm_power_polyfit_order or get_ideal_degree(
            config.mrm_power_data_path
        )
        _sweep_and_plot(config.mrm_power_data_path, deg, config.output_dir, "mrm_power")

    # MRM phase
    if config.mrm_phase_data_path and os.path.exists(config.mrm_phase_data_path):
        deg = config.mrm_phase_polyfit_order or get_ideal_degree(
            config.mrm_phase_data_path
        )
        _sweep_and_plot(config.mrm_phase_data_path, deg, config.output_dir, "mrm_phase")

    # Quick JTC sanity check and stage plots
    try:
        # TODO: Fix stage plots to work with batched JTC
        # _stage_plots_jtc(config, config.output_dir)
        _range_check_jtc(config, config.output_dir)
    except Exception as e:
        print(f"[ERROR] JTC range check failed: {e}")
