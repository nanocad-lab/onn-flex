from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import (
    apply_global_plot_style,
    DEFAULT_TITLE_FONTSIZE,
    SHOW_TITLES,
)
import torch

from onn_config import AppConfig
from onn_component import get_ideal_degree, get_coeffs, JTC

# Apply shared Matplotlib style (labels/ticks/titles)
apply_global_plot_style()


def _plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_ref: np.ndarray,
    y_poly: np.ndarray,
    ref_label: str,
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
        ref_label: Legend label describing the reference curve.
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
    plt.plot(x, y_ref, "b-", label=ref_label)
    plt.plot(
        x,
        y_poly,
        "r--",
        label=f"Poly fit (deg {poly_order}, R²={r_squared:.5f})",
    )
    plt.xlabel("Input")
    plt.ylabel("Output")
    if SHOW_TITLES:
        plt.title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
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
        ref_degree: Order of the reference polynomial (defaults to 1 → linear; ignored for mrm_phase).
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}. Skipping {tag} plot.")
        return

    data = pd.read_csv(csv_path)
    x = data["input"].values
    y = data["output"].values

    # Reference behaviour (linear fit unless overridden per component)
    if tag == "mrm_phase":
        y_ref = np.zeros_like(x)
        ref_label = "Ideal (0.0)"
    else:
        ref_coeff = np.polyfit(x, y, ref_degree)
        y_ref = np.polyval(ref_coeff, x)
        ref_label = f"Ideal (deg {ref_degree})"

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
        ref_label,
        degree,
        f"{tag} distortion fit",
        plot_path,
    )
    print(f"[TEST] Saved plot {plot_path}")


def _range_check_jtc(config: AppConfig, output_dir: str) -> None:
    """Run a quick JTC forward pass to ensure outputs are finite and within a sensible range."""
    # Make dummy components (using CSV-driven poly fits)
    jtc = JTC(config)

    # Random batch of signals/kernels in [-1, 1]
    torch.manual_seed(0)
    signal = torch.randn(1, 1, 1, config.jtc_half_size) / 4
    kernel = torch.randn(1, config.jtc_half_size) / 4
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
    if SHOW_TITLES:
        plt.title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _stage_plots_jtc(config: AppConfig, output_dir: str) -> None:
    """Plot intermediate JTC stages and compare with PyTorch conv."""
    jtc = JTC(config)

    torch.manual_seed(1)
    signal = torch.randn(1, 1, 1, config.jtc_half_size) / 4
    kernel = torch.randn(1, config.jtc_half_size) / 4

    input_plane = jtc.generate_input_plane(signal, kernel)
    jft = jtc.post_fft(input_plane)
    jps = jtc.post_output_distortion(jft)
    print(f"jps: {jps.shape}")
    print(f"jps: {jps[0, :]}")
    out = jtc.inverse_output(jps)
    print(f"out: {out.shape}")
    print(f"out: {out[0, :]}")

    # --- Combine stage plots into a single multi-panel PDF ---
    conv_out = torch.nn.functional.conv1d(
        signal.view(1, 1, -1), kernel.view(1, 1, -1), padding="same"
    )[0, :]

    print(f"conv_out: {conv_out.shape}")
    print(f"conv_out: {conv_out[0, :]}")
    # input("Press Enter to continue...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top-left: Input plane magnitude
    axes[0, 0].plot(torch.abs(input_plane[0, :]).detach().cpu().numpy())
    if SHOW_TITLES:
        axes[0, 0].set_title("Input plane", fontsize=DEFAULT_TITLE_FONTSIZE)

    # Top-right: After FFT magnitude
    axes[0, 1].plot(torch.abs(jft[0, :]).detach().cpu().numpy())
    if SHOW_TITLES:
        axes[0, 1].set_title("Post FFT", fontsize=DEFAULT_TITLE_FONTSIZE)

    # Bottom-left: After output distortion
    axes[1, 0].plot(jps[0, :].detach().cpu().numpy())
    if SHOW_TITLES:
        axes[1, 0].set_title("Post output distortion", fontsize=DEFAULT_TITLE_FONTSIZE)

    # Bottom-right: Final output vs. PyTorch conv reference
    axes[1, 1].plot(out[0, :].detach().cpu().numpy(), label="jtc")
    axes[1, 1].plot(conv_out[0, :].detach().cpu().numpy(), label="torch_conv")
    if SHOW_TITLES:
        axes[1, 1].set_title("Final output comparison", fontsize=DEFAULT_TITLE_FONTSIZE)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stage_plots.pdf"))
    plt.close(fig)


def _stage_plots_detailed(config: AppConfig, output_dir: str) -> None:
    """Compute and plot detailed JTC pipeline stages into a single multi-panel PDF.

    Uses `JTC.compute_stage_tensors` to retrieve the following stages (when available):
    input_plane, input_plane_quant, input_plane_driver, input_plane_mrm_phase,
    input_plane_mrm_pwr, jps_raw, jps_pd, jps_tia, jps_scale, jps_quant,
    jps_driver, jps_mrm_phase, jps_mrm_pwr, output_raw, output_pd, output_tia, output_scale,
    output_quant, output_slice.
    """
    jtc = JTC(config)

    torch.manual_seed(2)
    signal = torch.randn(1, config.jtc_half_size) / 4
    kernel = torch.randn(1, config.jtc_half_size) / 4

    with torch.no_grad():
        stage_data = jtc.compute_stage_tensors(signal, kernel)

    ordered_names = [
        name
        for name in getattr(jtc, "stage_order", list(stage_data.keys()))
        if name in stage_data
    ]
    num_stages = len(ordered_names)
    if num_stages == 0:
        print("[WARNING] No stages produced for detailed plot.")
        return

    cols = 3
    rows = (num_stages + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 2, 2.5 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, name in enumerate(ordered_names):
        r = idx // cols
        c = idx % cols
        arr = stage_data[name].detach().cpu().numpy()
        axes[r][c].plot(arr)
        if SHOW_TITLES:
            axes[r][c].set_title(name, fontsize=DEFAULT_TITLE_FONTSIZE)

    # Hide any unused subplots
    for k in range(num_stages, rows * cols):
        r = k // cols
        c = k % cols
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "stage_plots_detailed.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[TEST] Saved detailed stage plot {out_path}")


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

    # PD
    if config.pd_distortion_data_path and os.path.exists(
        config.pd_distortion_data_path
    ):
        deg = config.pd_distortion_polyfit_order or get_ideal_degree(
            config.pd_distortion_data_path
        )
        _sweep_and_plot(
            config.pd_distortion_data_path, deg, config.output_dir, "pd", ref_degree=2
        )

    # TIA
    if config.tia_distortion_data_path and os.path.exists(
        config.tia_distortion_data_path
    ):
        deg = config.tia_distortion_polyfit_order or get_ideal_degree(
            config.tia_distortion_data_path
        )
        _sweep_and_plot(config.tia_distortion_data_path, deg, config.output_dir, "tia")

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
        _stage_plots_jtc(config, config.output_dir)
    except Exception as e:
        print(f"[ERROR] JTC stage plots failed: {e}")

    try:
        _range_check_jtc(config, config.output_dir)
    except Exception as e:
        print(f"[ERROR] JTC range check failed: {e}")

    # Detailed stage plots (non-fatal)
    try:
        _stage_plots_detailed(config, config.output_dir)
    except Exception as e:
        print(f"[ERROR] Detailed JTC stage plots failed: {e}")


__all__ = ["run_pretrain_tests"]
