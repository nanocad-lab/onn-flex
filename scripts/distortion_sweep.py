import os
import sys
from pathlib import Path
import argparse
from dataclasses import replace

import matplotlib.pyplot as plt

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from plot_style import (
    apply_global_plot_style,
    DEFAULT_TICK_LABEL_FONTSIZE,
    DEFAULT_TITLE_FONTSIZE,
    SHOW_TITLES,
)
import numpy as np
import torch
from typing import Iterable

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from onn_inference import _build_jtc
from onn_config import AppConfig
from onn_inference import (
    compute_snr_enob,
    compute_snr_jps,
    load_config_from_yaml,
    run_inference,
)

# Apply shared Matplotlib style (labels/ticks/titles)
apply_global_plot_style()

PARAMS = [
    "driver_distortion_strength",
    "pd_distortion_strength",
    "tia_distortion_strength",
    "mrm_power_distortion_strength",
    "mrm_phase_distortion_strength",
    "ler_std_dev",
]

REF_DIGITAL_ACC = 60.13

# Fixed y-axis limits for consistency across components
ACC_YLIM = (0.0, 65.0)  # Accuracy in %
SNDR_YLIM = (-10.0, 40.0)  # SNDR in dB


def _plot_param_sweep_from_data(param: str, out_dir: str) -> None:
    """Generate plots from previously saved sweep data."""
    data_file = os.path.join(out_dir, f"{param}_results.txt")

    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Skipping plot generation for {param}.")
        return

    # Load previously saved results
    results = np.loadtxt(data_file)
    if results.ndim == 1:
        results = results.reshape(1, -1)  # Handle single row case

    strengths = results[:, 0]
    accs = results[:, 1]
    sndrs = results[:, 2]

    # Handle backward compatibility - old files may not have jps_sndr column
    if results.shape[1] >= 5:
        jps_sndrs = results[:, 4]
    else:
        # Use the same SNDR values for both if JPS SNDR is not available
        jps_sndrs = sndrs
        print("Note: Using output SNDR values for JPS SNDR (old format compatibility)")

    print(f"Loaded {len(strengths)} data points for {param}")

    # Create plot
    fig, ax1 = plt.subplots()
    ax1.plot(strengths, accs, "bo-", label="Inference Accuracy")
    # Add reference digital accuracy line
    ax1.axhline(
        y=REF_DIGITAL_ACC,
        color="k",
        linestyle="--",
        alpha=0.7,
        label=f"Digital Reference ({REF_DIGITAL_ACC:.1f}%)",
    )
    # Set xlabel based on parameter type
    if param == "ler_std_dev":
        ax1.set_xlabel("Splitter Ratio Std. Dev.")
    else:
        ax1.set_xlabel("Distortion Ratio (α)")
    ax1.set_ylabel("Inference Accuracy (%)", color="b")
    # Fix y-axis limits for comparability
    ax1.set_ylim(ACC_YLIM)
    if SHOW_TITLES:
        ax1.set_title(f"{param} sweep", fontsize=DEFAULT_TITLE_FONTSIZE)
    ax2 = ax1.twinx()
    ax2.plot(strengths, sndrs, "r^-", label="SNDR (pJTC output)")
    ax2.plot(strengths, jps_sndrs, "gs--", label="SNDR (JPS)")
    ax2.set_ylabel("SNDR (dB)", color="r")
    ax2.set_ylim(SNDR_YLIM)
    # Combine legends from both axes into a single legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{param}_sweep.pdf"))
    plt.close(fig)

    print(f"Generated plot: {param}_sweep.pdf")


def _plot_jtc_2d_from_data(out_dir: str) -> None:
    """Generate 2D JTC plots from previously saved sweep data."""
    acc_file = os.path.join(out_dir, "jtc_2d_accuracy.npy")
    sndr_file = os.path.join(out_dir, "jtc_2d_sndr_output.npy")

    if not os.path.exists(acc_file) or not os.path.exists(sndr_file):
        print("JTC 2D sweep data files not found. Skipping 2D plot generation.")
        return

    # Load previously saved results
    acc_matrix = np.load(acc_file)
    sndr_out_matrix = np.load(sndr_file)

    print(f"Loaded JTC 2D sweep data: {acc_matrix.shape}")

    # We need to reconstruct the field and sep values based on the data shape
    # This assumes the same ranges as in _sweep_jtc_2d
    sep_values = np.arange(0, acc_matrix.shape[0])
    field_values = np.arange(16, 16 + acc_matrix.shape[1])  # Assuming min_field = 16

    # Heat-maps --------------------------------------------------------
    def _plot_heat(
        data: np.ndarray,
        filename: str,
        cbar_label: str,
        add_ref_line: bool = False,
    ) -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(data, origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_xticks(np.arange(len(field_values)))
        ax.set_xticklabels(field_values)
        ax.set_yticks(np.arange(len(sep_values)))
        ax.set_yticklabels(sep_values)
        ax.set_xlabel("jtc_total_field")
        ax.set_ylabel("jtc_separation")

        fig.colorbar(im, ax=ax, label=cbar_label)
        if SHOW_TITLES:
            ax.set_title(
                f"JTC 2D Sweep - {cbar_label}", fontsize=DEFAULT_TITLE_FONTSIZE
            )

        # Add reference line for accuracy plots
        if add_ref_line:
            # Add a contour line at the reference digital accuracy level
            contour = ax.contour(
                data,
                levels=[REF_DIGITAL_ACC],
                colors="white",
                linewidths=2,
                linestyles="--",
            )
            ax.clabel(
                contour, inline=True, fontsize=DEFAULT_TICK_LABEL_FONTSIZE, fmt="%.1f%%"
            )
            # Add text annotation for clarity
            ax.text(
                0.02,
                0.98,
                f"Digital Reference: {REF_DIGITAL_ACC:.1f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)

    _plot_heat(
        acc_matrix,
        "jtc_2d_sweep_accuracy.pdf",
        "Inference Accuracy (%)",
        add_ref_line=True,
    )
    _plot_heat(sndr_out_matrix, "jtc_2d_sweep_output_sndr.pdf", "SNDR (dB)")


def _sweep_param(
    base_cfg: AppConfig,
    weights: str,
    param: str,
    out_dir: str,
    strengths: Iterable[float],
) -> None:
    accs = []
    sndrs = []
    enobs = []
    jps_sndrs = []
    for val in strengths:
        cfg = replace(base_cfg, **{param: float(val)})
        acc = run_inference(cfg, weights)
        sndr, enob = compute_snr_enob(cfg, param, num_tests=10000)
        sndr_jps = compute_snr_jps(cfg, param, num_tests=10000)
        accs.append(acc)
        sndrs.append(sndr)
        enobs.append(enob)
        jps_sndrs.append(sndr_jps)
        print(
            f"{param}={val:.3f} -> acc {acc:.3f}% sndr {sndr:.2f}dB jps_sndr {sndr_jps:.2f}dB"
        )

    results: np.ndarray = np.stack([strengths, accs, sndrs, enobs, jps_sndrs], axis=1)
    np.savetxt(
        os.path.join(out_dir, f"{param}_results.txt"),
        results,
        header="strength accuracy sndr enob jps_sndr",
    )

    fig, ax1 = plt.subplots()
    ax1.plot(strengths, accs, "bo-", label="Inference Accuracy")
    # Add reference digital accuracy line
    ax1.axhline(
        y=REF_DIGITAL_ACC,
        color="k",
        linestyle="--",
        alpha=0.7,
        label=f"Digital Reference ({REF_DIGITAL_ACC:.1f}%)",
    )
    if param == "ler_std_dev":
        ax1.set_xlabel("Splitter Ratio Std. Dev.")
    else:
        ax1.set_xlabel("Distorted TF Ratio (α)")
    ax1.set_ylabel("Inference Accuracy (%)", color="b")
    # Fix y-axis limits for comparability
    ax1.set_ylim(ACC_YLIM)
    if SHOW_TITLES:
        ax1.set_title(f"{param} sweep", fontsize=DEFAULT_TITLE_FONTSIZE)
    ax2 = ax1.twinx()
    ax2.plot(strengths, sndrs, "r^-", label="SNDR (pJTC output)")
    ax2.plot(strengths, jps_sndrs, "gs--", label="SNDR (JPS)")
    ax2.set_ylabel("SNDR (dB)", color="r")
    ax2.set_ylim(SNDR_YLIM)
    # Combine legends from both axes into a single legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{param}_sweep.pdf"))
    plt.close(fig)


# -----------------------------------------------------------------------------
#  2-D sweep for JTC separation and total field
# -----------------------------------------------------------------------------


def _compute_jtc_sndrs(cfg: AppConfig, num_tests: int = 1000, seed: int = 0):
    """Compute SNDR between two JTC *forward* outputs.

    The comparison is between:
      • The **current** configuration (*cfg*).
      • A **reference** configuration with the same parameters except
        ``jtc_separation = cfg.jtc_half_size`` and
        ``jtc_total_field = 6 * cfg.jtc_half_size``.

    Distortion-strength parameters are **not** zeroed – the goal is to
    isolate the effect of geometry (separation / total-field).

    For backward compatibility with the existing caller, the function
    returns a *pair* where the second value is a duplicate of the first
    (the caller expects two numbers).
    """

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Current JTC
    jtc_cur = _build_jtc(cfg).to(device)

    # Reference geometry (same distortion settings)
    ref_cfg = replace(
        cfg,
        jtc_separation=cfg.jtc_half_size,
        jtc_total_field=cfg.jtc_half_size * 6,
    )
    jtc_ref = _build_jtc(ref_cfg).to(device)

    out_list = []
    ref_list = []

    for _ in range(num_tests):
        signal = torch.rand(1, 1, 1, cfg.jtc_half_size, device=device)
        kernel = torch.rand(1, cfg.jtc_half_size, device=device)

        out_list.append(jtc_cur(signal, kernel))
        ref_list.append(jtc_ref(signal, kernel))

    out = torch.stack(out_list)
    ref = torch.stack(ref_list)
    noise = out - ref

    sndr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())

    # Return duplicate values to keep (sndr_out, sndr_jps) signature unchanged
    return sndr.item()


def _sweep_jtc_2d(base_cfg: AppConfig, weights: str, out_dir: str) -> None:
    """Run a 2-D sweep over (*jtc_separation*, *jtc_total_field*).

    A nested loop explores a grid of valid separation / total-field
    combinations, measuring the classification accuracy.  Results are saved
    as a NumPy matrix + a PDF heat-map for quick visualisation.
    """

    # Define sweep ranges (feel free to adjust as needed)
    sep_values = np.arange(0, base_cfg.jtc_half_size + 1 + 4)  # 0 … 8 for default cfg
    # Ensure the smallest field is the minimal valid value
    min_field = 2 * base_cfg.jtc_half_size
    field_values = np.arange(min_field, min_field + 41, 4)  # 16,20,24,28,32

    acc_matrix = np.full((len(sep_values), len(field_values)), np.nan)
    sndr_out_matrix = np.full_like(acc_matrix, np.nan, dtype=float)

    for i, sep in enumerate(sep_values):
        for j, field in enumerate(field_values):
            # Skip invalid combos (not enough plane length)
            if field < min_field + sep:
                continue
            cfg = replace(
                base_cfg,
                jtc_separation=int(sep),
                jtc_total_field=int(field),
            )
            acc = run_inference(cfg, weights)
            acc_matrix[i, j] = acc
            sndr = _compute_jtc_sndrs(cfg, num_tests=200)
            sndr_out_matrix[i, j] = sndr

    # Save matrices for later reuse / plotting
    np.save(os.path.join(out_dir, "jtc_2d_accuracy.npy"), acc_matrix)
    np.save(os.path.join(out_dir, "jtc_2d_sndr_output.npy"), sndr_out_matrix)

    # Also export as text for quick inspection
    np.savetxt(
        os.path.join(out_dir, "jtc_2d_accuracy.txt"),
        acc_matrix,
        fmt="%0.2f",
        header=(
            "Rows: jtc_separation values {}\nCols: jtc_total_field values {}".format(
                list(sep_values), list(field_values)
            )
        ),
    )
    np.savetxt(
        os.path.join(out_dir, "jtc_2d_sndr_output.txt"),
        sndr_out_matrix,
        fmt="%0.2f",
        header=(
            "Rows: jtc_separation values {}\nCols: jtc_total_field values {}".format(
                list(sep_values), list(field_values)
            )
        ),
    )

    # Heat-maps --------------------------------------------------------
    def _plot_heat(data, filename, cbar_label, add_ref_line=False):
        fig, ax = plt.subplots()
        im = ax.imshow(data, origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_xticks(np.arange(len(field_values)))
        ax.set_xticklabels(field_values)
        ax.set_yticks(np.arange(len(sep_values)))
        ax.set_yticklabels(sep_values)
        ax.set_xlabel("jtc_total_field")
        ax.set_ylabel("jtc_separation")

        fig.colorbar(im, ax=ax, label=cbar_label)
        if SHOW_TITLES:
            ax.set_title(
                f"JTC 2D Sweep - {cbar_label}", fontsize=DEFAULT_TITLE_FONTSIZE
            )

        # Add reference line for accuracy plots
        if add_ref_line:
            # Add a contour line at the reference digital accuracy level
            contour = ax.contour(
                data,
                levels=[REF_DIGITAL_ACC],
                colors="white",
                linewidths=2,
                linestyles="--",
            )
            ax.clabel(
                contour, inline=True, fontsize=DEFAULT_TICK_LABEL_FONTSIZE, fmt="%.1f%%"
            )
            # Add text annotation for clarity
            ax.text(
                0.02,
                0.98,
                f"Digital Reference: {REF_DIGITAL_ACC:.1f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)

    _plot_heat(
        acc_matrix,
        "jtc_2d_sweep_accuracy.pdf",
        "Inference Accuracy (%)",
        add_ref_line=True,
    )
    _plot_heat(sndr_out_matrix, "jtc_2d_sweep_output_sndr.pdf", "SNDR (dB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distortion sweep inference")
    parser.add_argument("--config", help="config yaml (required for new sweeps)")
    parser.add_argument(
        "--weights", help="trained weights path (required for new sweeps)"
    )
    parser.add_argument("--output-dir", default="sweep_results")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from previously saved data, skip inference",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_only:
        # Only generate plots from existing data
        print("Plot-only mode: generating plots from previously saved data...")

        # Generate 1D parameter sweep plots
        for param in PARAMS:
            _plot_param_sweep_from_data(param, args.output_dir)

        # Generate 2D JTC sweep plots
        _plot_jtc_2d_from_data(args.output_dir)

        print("Plot generation completed.")

    else:
        # Run full sweep with inference
        if not args.config or not args.weights:
            parser.error(
                "--config and --weights are required when not using --plot-only"
            )

        base_cfg = load_config_from_yaml(args.config)

        default_strengths = np.arange(0, 1.1, 0.1)
        for param in PARAMS:
            _sweep_param(
                base_cfg, args.weights, param, args.output_dir, default_strengths
            )
            if param == "ler_std_dev":
                _sweep_param(
                    base_cfg,
                    args.weights,
                    param,
                    args.output_dir,
                    np.arange(0, 0.05, 0.005),
                )
        # 2-D sweep (after 1-D distortion sweeps)
        _sweep_jtc_2d(base_cfg, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
