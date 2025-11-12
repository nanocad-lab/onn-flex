import os
import sys
from pathlib import Path
import argparse
from typing import Optional, Tuple
from matplotlib.axes import Axes

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import (
    apply_global_plot_style,
    DEFAULT_AXIS_LABEL_FONTSIZE,
    DEFAULT_TICK_LABEL_FONTSIZE,
    DEFAULT_TITLE_FONTSIZE,
    SHOW_TITLES,
)
from onn_config import AppConfig
from onn_component import get_ideal_degree, get_coeffs
from onn_inference import load_config_from_yaml
from scripts.distortion_sweep import REF_DIGITAL_ACC, ACC_YLIM, SNDR_YLIM

# Apply shared Matplotlib style (labels/ticks/titles)
apply_global_plot_style()


def _load_sweep_results(
    sweep_dir: str, param: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load a parameter sweep results file saved by distortion_sweep.

    Returns tuple: (strengths, accs, sndrs, jps_sndrs)
    """
    data_file = os.path.join(sweep_dir, f"{param}_results.txt")
    if not os.path.exists(data_file):
        print(f"[WARN] Sweep data file not found: {data_file}")
        return None

    results = np.loadtxt(data_file)
    if results.ndim == 1:
        results = results.reshape(1, -1)

    strengths = results[:, 0]
    accs = results[:, 1]
    sndrs = results[:, 2]
    if results.shape[1] >= 5:
        jps_sndrs = results[:, 4]
    else:
        jps_sndrs = sndrs
    return strengths, accs, sndrs, jps_sndrs


def _plot_sweep_on_axes(
    ax_left: Axes,
    param: str,
    strengths: np.ndarray,
    accs: np.ndarray,
    sndrs: np.ndarray,
    jps_sndrs: np.ndarray,
    compact: bool = False,
) -> None:
    """Render the sweep plot onto the provided axes (left axis + twin y)."""
    ax1 = ax_left
    ms = 3 if compact else 5
    lw = 1.0 if compact else 1.5
    label_fs = 9 if compact else DEFAULT_AXIS_LABEL_FONTSIZE
    acc_line = ax1.plot(
        strengths, accs, "bo-", label="Inference Accuracy", markersize=ms, linewidth=lw
    )[0]
    # Reference digital accuracy (from distortion_sweep)
    ref_acc = REF_DIGITAL_ACC
    ref_line = ax1.axhline(
        y=ref_acc,
        color="k",
        linestyle="--",
        alpha=0.7,
        label=f"Digital Reference ({ref_acc:.1f}%)",
        linewidth=lw,
    )
    if param == "ler_std_dev":
        ax1.set_xlabel("Splitter Ratio Std. Dev.", fontsize=label_fs)
    else:
        ax1.set_xlabel("Distortion Ratio (α)", fontsize=label_fs)
    ax1.set_ylabel("Inference Accuracy (%)", color="b", fontsize=label_fs)
    ax1.set_ylim(ACC_YLIM)
    if SHOW_TITLES and not compact:
        ax1.set_title(f"{param} sweep", fontsize=DEFAULT_TITLE_FONTSIZE)

    ax2 = ax1.twinx()
    sndr_line = ax2.plot(
        strengths, sndrs, "r^-", label="SNDR (pJTC output)", markersize=ms, linewidth=lw
    )[0]
    jps_line = ax2.plot(
        strengths, jps_sndrs, "gs--", label="SNDR (JPS)", markersize=ms, linewidth=lw
    )[0]
    ax2.set_ylabel("SNDR (dB)", color="r", fontsize=label_fs)
    ax2.set_ylim(SNDR_YLIM)

    # Legend: top-right in specific order
    legend_fs = 8 if compact else DEFAULT_TICK_LABEL_FONTSIZE
    ordered_handles = [ref_line, acc_line, sndr_line, jps_line]
    ordered_labels = [
        f"Digital Reference ({ref_acc:.1f}%)",
        "Inference Accuracy",
        "SNDR (pJTC output)",
        "SNDR (JPS)",
    ]
    ax1.legend(
        ordered_handles,
        ordered_labels,
        loc="upper right",
        fontsize=legend_fs,
        ncol=1,
    )
    if compact:
        ax1.tick_params(axis="both", labelsize=7, length=3, width=0.8)
        ax2.tick_params(axis="both", labelsize=7, length=3, width=0.8)
        # Reduce scientific notation offset text (e.g., 1e-5)
        ax1.xaxis.offsetText.set_fontsize(7)
        ax1.yaxis.offsetText.set_fontsize(7)
        ax2.xaxis.offsetText.set_fontsize(7)
        ax2.yaxis.offsetText.set_fontsize(7)


def _plot_fit_on_axes(
    ax: Axes,
    csv_path: str,
    poly_order: Optional[int],
    tag: str,
    ref_degree: int = 1,
    compact: bool = False,
) -> bool:
    """Render the fit comparison plot onto the provided axes.

    Returns True if plotted, False if CSV missing.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found for {tag}: {csv_path}")
        return False

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
    degree = poly_order or get_ideal_degree(csv_path)
    poly_coeff = get_coeffs(csv_path, degree)
    y_poly = np.polyval(poly_coeff, x)

    # R^2
    ss_res = np.sum((y - y_poly) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    lw = 1.0 if compact else 1.5
    ms = 3 if compact else 5
    label_fs = 9 if compact else DEFAULT_AXIS_LABEL_FONTSIZE
    # Draw lines and capture handles
    (line_ideal,) = ax.plot(x, y_ref, "b-", label=ref_label, linewidth=lw)
    (line_poly,) = ax.plot(
        x,
        y_poly,
        "r--",
        label=f"Poly fit (deg {degree}, R²={r_squared:.5f})",
        linewidth=lw,
    )
    (line_data,) = ax.plot(x, y, "k.", label="CSV data", markersize=ms)
    ax.set_xlabel("Input", fontsize=label_fs)
    ax.set_ylabel("Output", fontsize=label_fs)
    if SHOW_TITLES and not compact:
        ax.set_title(f"{tag} distortion fit", fontsize=DEFAULT_TITLE_FONTSIZE)
    legend_fs = 8 if compact else DEFAULT_TICK_LABEL_FONTSIZE
    # Legend: top-left in specific order
    ax.legend(
        [line_ideal, line_poly, line_data],
        [ref_label, f"Poly fit (deg {degree}, R²={r_squared:.5f})", "CSV data"],
        loc="upper left",
        fontsize=legend_fs,
    )
    if compact:
        ax.tick_params(axis="both", labelsize=7, length=3, width=0.8)
        # Reduce scientific notation offset text (e.g., 1e-5)
        ax.xaxis.offsetText.set_fontsize(7)
        ax.yaxis.offsetText.set_fontsize(7)
    return True


def generate_combined_component_plots(
    config: AppConfig,
    sweep_output_dir: str,
    output_dir: Optional[str] = None,
    include_ler: bool = False,
    ieee_compact: bool = True,
) -> None:
    """Generate side-by-side plots combining distortion sweep (left) and CSV fit (right) for each component.

    Args:
        config: App configuration holding CSV paths and polyfit overrides.
        sweep_output_dir: Directory containing "*_results.txt" files from distortion_sweep.
        output_dir: Where to write combined PDFs. Defaults to config.output_dir.
        include_ler: Whether to include LER parameter (no CSV fit; right panel will be empty).
    """
    out_dir = output_dir or config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    components = [
        (
            "driver_distortion_strength",
            "driver",
            config.driver_distortion_data_path,
            config.driver_distortion_polyfit_order,
            1,
        ),
        (
            "pd_distortion_strength",
            "pd",
            config.pd_distortion_data_path,
            config.pd_distortion_polyfit_order,
            2,
        ),
        (
            "tia_distortion_strength",
            "tia",
            config.tia_distortion_data_path,
            config.tia_distortion_polyfit_order,
            1,
        ),
        (
            "mrm_power_distortion_strength",
            "mrm_power",
            config.mrm_power_data_path,
            config.mrm_power_polyfit_order,
            1,
        ),
        (
            "mrm_phase_distortion_strength",
            "mrm_phase",
            config.mrm_phase_data_path,
            config.mrm_phase_polyfit_order,
            1,
        ),
    ]

    if include_ler:
        components.append(("ler_std_dev", "ler", "", None, 1))

    for param, tag, csv_path, poly_order, ref_degree in components:
        sweep = _load_sweep_results(sweep_output_dir, param)
        if sweep is None:
            print(f"[SKIP] No sweep data for {tag} ({param})")
            continue

        strengths, accs, sndrs, jps_sndrs = sweep

        # Two-column IEEE half-page: ~7.2in width x ~3.4in height
        figsize = (7.2, 3.4) if ieee_compact else (12, 4)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # Right panel: distortion sweep; Left panel: CSV fit
        _plot_sweep_on_axes(
            axes[1], param, strengths, accs, sndrs, jps_sndrs, compact=ieee_compact
        )

        # Fit panel may be absent if CSV not available (e.g., LER)
        have_fit = False
        if csv_path:
            have_fit = _plot_fit_on_axes(
                axes[0], csv_path, poly_order, tag, ref_degree, compact=ieee_compact
            )

        if not have_fit:
            axes[0].axis("off")
            if SHOW_TITLES:
                axes[0].set_title(
                    "No fit data available", fontsize=DEFAULT_TITLE_FONTSIZE
                )

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{tag}_combined.pdf")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"[COMBINED] Saved {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined distortion sweep + fit plots per component"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--sweep-dir",
        required=True,
        dest="sweep_dir",
        help="Directory containing *_results.txt from distortion_sweep",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directory to save combined PDFs (defaults to config.output_dir)",
    )
    parser.add_argument(
        "--include-ler",
        action="store_true",
        dest="include_ler",
        help="Include LER sweep (no fit panel)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config_from_yaml(args.config)
    generate_combined_component_plots(
        cfg, args.sweep_dir, args.output_dir, args.include_ler
    )


if __name__ == "__main__":
    main()
