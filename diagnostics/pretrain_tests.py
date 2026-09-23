from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from onn_component import (
    JTC,
    _compute_linear_coeffs,
    _sqrt_clamped,
    get_coeffs,
    get_ideal_degree,
)
from onn_config import AppConfig
from plot_style import (
    DEFAULT_TITLE_FONTSIZE,
    SHOW_TITLES,
    apply_global_plot_style,
)

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
    ylabel: str = "Output",
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
    plt.ylabel(ylabel)
    if SHOW_TITLES:
        plt.title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _sweep_and_plot(
    csv_path: str,
    degree: int | None,
    output_dir: str,
    tag: str,
    *,
    linearization: str,
) -> None:
    """Load a CSV I/O sweep, fit polynomials, and save a comparison plot.

    Args:
        csv_path: Path to the CSV containing *input* and *output* columns.
        degree: Order of the main polynomial used to model distortion.
        output_dir: Where to write the PNG plot.
        tag: Descriptive tag inserted into the file name and plot title.
        linearization: Ideal reference used by the simulated component.
    """
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV file not found: {csv_path}. Skipping {tag} plot.")
        return

    data = pd.read_csv(csv_path)
    x = data["input"].values
    transform = _sqrt_clamped if tag == "mrm_amplitude" else None
    y = data["output"].values
    if transform is not None:
        y = transform(y)

    # Use the same ideal reference and output units as the component model.
    if tag == "mrm_phase":
        y_ref = np.zeros_like(x)
        ref_label = "Ideal (0.0)"
    else:
        ref_coeff = _compute_linear_coeffs(
            csv_path, output_transform=transform, linearization=linearization
        )
        y_ref = np.polyval(ref_coeff, x)
        ref_label = f"Ideal ({linearization})"

    # Distortion polynomial fit
    if degree is None:
        degree = get_ideal_degree(csv_path, output_transform=transform)
    poly_coeff = get_coeffs(csv_path, degree, output_transform=transform)
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
        ylabel="Field amplitude (√W)" if transform else "Output",
    )
    print(f"[TEST] Saved plot {plot_path}")


def _range_check_jtc(config: AppConfig, output_dir: str) -> None:
    """Run a quick JTC forward pass to ensure outputs are finite and within a sensible range."""
    # Make dummy components (using CSV-driven poly fits)
    jtc = JTC(config)

    # Random batch of signals/kernels in [-1, 1]
    torch.manual_seed(0)
    signal = torch.randn(1, 1, 1, config.input_length) / 4
    kernel = torch.randn(1, config.kernel_length) / 4
    out = jtc(signal, kernel)

    if not torch.isfinite(out).all():
        raise ValueError("JTC output contains NaNs/Infs")
    out_min, out_max = out.min().item(), out.max().item()
    print(f"[TEST] JTC output range: {out_min:.3f} .. {out_max:.3f}")

    # Save to disk for reference
    np.save(os.path.join(output_dir, "jtc_test_output.npy"), out.cpu().numpy())


def _stage_plots_detailed(config: AppConfig, output_dir: str) -> None:
    """Compute and plot detailed JTC pipeline stages into a single multi-panel PDF.

    Uses `JTC.compute_stage_tensors` to retrieve the following stages (when available):
    input_plane, input_plane_quant, input_plane_driver, input_plane_mrm_phase,
    input_plane_mrm_amp, jps_raw, jps_pd_input, jps_pd, jps_tia, jps_scale, jps_quant,
    jps_driver, jps_mrm_phase, jps_mrm_amp, output_raw, output_pd, output_tia, output_scale,
    output_scale_slice, output_quant, output_quant_slice, output_slice.
    """
    jtc = JTC(config)

    torch.manual_seed(2)
    signal = torch.randn(1, config.input_length) / 4
    kernel = torch.randn(1, config.kernel_length) / 4

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

    components = (
        (
            "driver",
            config.driver_distortion_data_path,
            config.driver_distortion_polyfit_order,
        ),
        ("pd", config.pd_distortion_data_path, config.pd_distortion_polyfit_order),
        ("tia", config.tia_distortion_data_path, config.tia_distortion_polyfit_order),
        (
            "mrm_amplitude",
            config.mrm_amplitude_data_path,
            config.mrm_amplitude_polyfit_order,
        ),
        ("mrm_phase", config.mrm_phase_data_path, config.mrm_phase_polyfit_order),
    )
    for tag, csv_path, degree in components:
        if csv_path:
            _sweep_and_plot(
                csv_path,
                degree,
                config.output_dir,
                tag,
                linearization=config.transfer_linearization,
            )

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
