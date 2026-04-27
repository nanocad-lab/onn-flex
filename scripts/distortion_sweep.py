import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from plot_style import (
    apply_global_plot_style,
    DEFAULT_TICK_LABEL_FONTSIZE,
    DEFAULT_TITLE_FONTSIZE,
    SHOW_TITLES,
)
from onn_config import AppConfig
from onn_inference import (
    _build_jtc,
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
    "lens_distortion_strength",
    "laser_rin_db",
    "pd_noise_w",
]

REF_DIGITAL_ACC = 60.13

# Fixed y-axis limits for consistency across components
ACC_YLIM = (0.0, 65.0)  # Accuracy in %
SNDR_YLIM = (-10.0, 55.0)  # SNDR in dB

# Noise sweeps (chosen to be meaningful relative to the internal unit ranges):
# - Laser RIN: interpreted as 20*log10(rms_fraction), so:
#   -60 dB -> 0.1% RMS, -40 dB -> 1% RMS, -30 dB -> 3.16% RMS.
#   Include an explicit "off" point (None) so the baseline is deterministic.
#
# Use a coarser grid for readability in the paper figures.
LASER_RIN_DB_MIN = -80
LASER_RIN_DB_MAX = -30
LASER_RIN_DB_STEP = 5
LASER_RIN_DB_SWEEP: list[float | None] = [None] + [
    float(x)
    for x in np.arange(LASER_RIN_DB_MIN, LASER_RIN_DB_MAX + 1, LASER_RIN_DB_STEP)
]
# - PD noise: PD input is clamped to ~[1e-6, 1e-5] W, so sweep ~0–1e-7 W RMS,
#   plus a few stress-test points. Use a coarser grid for readability.
PD_NOISE_W_LOG10_MIN = -10.0
PD_NOISE_W_LOG10_MAX = -6.0
PD_NOISE_W_LOG10_STEP = 0.5
PD_NOISE_W_SWEEP = np.concatenate(
    (
        np.array([0.0]),
        10.0
        ** np.arange(
            PD_NOISE_W_LOG10_MIN,
            PD_NOISE_W_LOG10_MAX + 0.5 * PD_NOISE_W_LOG10_STEP,
            PD_NOISE_W_LOG10_STEP,
        ),
    )
)


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
    strengths_plot = strengths.copy()
    rin_off_x: float | None = None
    rin_major_ticks: list[float] | None = None
    if param == "laser_rin_db" and np.isnan(strengths_plot).any():
        non_nan = strengths_plot[~np.isnan(strengths_plot)]
        rin_off_x = -100.0
        if non_nan.size > 0 and np.nanmin(non_nan) <= rin_off_x:
            rin_off_x = float(np.nanmin(non_nan)) - 10.0
        strengths_plot[np.isnan(strengths_plot)] = rin_off_x
        # Avoid setting a tick label for every data point; show "off" and
        # major dB ticks for readability when the sweep is dense.
        if non_nan.size > 0:
            tick_step = 10.0
            tick_start = tick_step * np.floor(float(np.nanmin(non_nan)) / tick_step)
            tick_end = float(np.nanmax(non_nan))
            major_ticks = np.arange(tick_start, tick_end + 0.5 * tick_step, tick_step)
            rin_major_ticks = [
                float(x)
                for x in major_ticks
                if x >= float(np.nanmin(non_nan)) - 1e-12
                and x <= float(np.nanmax(non_nan)) + 1e-12
            ]

    if results.shape[1] < 5:
        raise ValueError(
            f"{data_file} must contain columns: strength accuracy sndr enob jps_sndr"
        )
    jps_sndrs = results[:, 4]

    print(f"Loaded {len(strengths)} data points for {param}")

    # Create plot
    fig, ax1 = plt.subplots()
    if param == "laser_rin_db" and rin_off_x is not None:
        ax1.set_xticks([rin_off_x] + (rin_major_ticks or []))
        ax1.set_xticklabels(["off"] + [f"{x:.0f}" for x in (rin_major_ticks or [])])
    ax1.plot(strengths_plot, accs, "bo-", label="Inference Accuracy")
    # Set xlabel based on parameter type
    if param == "ler_std_dev":
        ax1.set_xlabel("Splitter Ratio Std. Dev.")
    elif param == "lens_distortion_strength":
        ax1.set_xlabel("Lens distortion strength (α)")
    elif param == "laser_rin_db":
        ax1.set_xlabel("Laser RIN (dB, RMS fraction)")
    elif param == "pd_noise_w":
        ax1.set_xlabel("PD input noise (W RMS)")
    else:
        ax1.set_xlabel("Distortion Ratio (α)")
    if param == "pd_noise_w":
        ax1.set_xscale("symlog", linthresh=1e-10)
    ax1.set_ylabel("Inference Accuracy (%)", color="b")
    # Fix y-axis limits for comparability
    ax1.set_ylim(ACC_YLIM)
    if SHOW_TITLES:
        ax1.set_title(f"{param} sweep", fontsize=DEFAULT_TITLE_FONTSIZE)
    ax2 = ax1.twinx()
    ax2.plot(strengths_plot, sndrs, "r^-", label="SNDR (pJTC output)")
    ax2.plot(strengths_plot, jps_sndrs, "gs--", label="SNDR (JPS)")
    ax2.set_ylabel("SNDR (dB)", color="r")
    ax2.set_ylim(SNDR_YLIM)
    # Combine legends from both axes into a single legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_loc = (
        "lower left" if param in {"laser_rin_db", "pd_noise_w"} else "upper right"
    )
    ax1.legend(handles1 + handles2, labels1 + labels2, loc=legend_loc)
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
    strengths: Iterable[float | None],
    snr_tests: int,
) -> None:
    accs = []
    sndrs = []
    enobs = []
    jps_sndrs = []
    strengths_saved: list[float] = []
    for val in strengths:
        if param == "laser_rin_db" and val is None:
            cfg = replace(base_cfg, laser_rin_db=None)
            strengths_saved.append(float("nan"))
        else:
            cfg = replace(base_cfg, **{param: float(val)})
            strengths_saved.append(float(val))
        acc = run_inference(cfg, weights)
        sndr, enob = compute_snr_enob(cfg, param, num_tests=snr_tests)
        sndr_jps = compute_snr_jps(cfg, param, num_tests=snr_tests)
        accs.append(acc)
        sndrs.append(sndr)
        enobs.append(enob)
        jps_sndrs.append(sndr_jps)
        if param == "laser_rin_db" and val is None:
            val_str = "off"
        elif param == "pd_noise_w":
            val_str = f"{val:.3e}"
        elif param == "laser_rin_db":
            val_str = f"{val:.1f}"
        else:
            val_str = f"{val:.3f}"
        print(
            f"{param}={val_str} -> acc {acc:.3f}% sndr {sndr:.2f}dB jps_sndr {sndr_jps:.2f}dB"
        )

    strengths_arr = np.asarray(strengths_saved, dtype=float)
    results: np.ndarray = np.stack(
        [strengths_arr, accs, sndrs, enobs, jps_sndrs], axis=1
    )
    np.savetxt(
        os.path.join(out_dir, f"{param}_results.txt"),
        results,
        header="strength accuracy sndr enob jps_sndr",
    )

    fig, ax1 = plt.subplots()
    strengths_plot = strengths_arr.copy()
    if param == "laser_rin_db" and np.isnan(strengths_plot).any():
        non_nan = strengths_plot[~np.isnan(strengths_plot)]
        off_x = -100.0
        if non_nan.size > 0 and np.nanmin(non_nan) <= off_x:
            off_x = float(np.nanmin(non_nan)) - 10.0
        strengths_plot[np.isnan(strengths_plot)] = off_x
        # Avoid setting a tick label for every data point; show "off" and
        # major dB ticks for readability when the sweep is dense.
        if non_nan.size > 0:
            tick_step = 10.0
            tick_start = tick_step * np.floor(float(np.nanmin(non_nan)) / tick_step)
            tick_end = float(np.nanmax(non_nan))
            major_ticks = np.arange(tick_start, tick_end + 0.5 * tick_step, tick_step)
            major_ticks = [
                float(x)
                for x in major_ticks
                if x >= float(np.nanmin(non_nan)) - 1e-12
                and x <= float(np.nanmax(non_nan)) + 1e-12
            ]
            ax1.set_xticks([off_x] + major_ticks)
            ax1.set_xticklabels(["off"] + [f"{x:.0f}" for x in major_ticks])
    ax1.plot(strengths_plot, accs, "bo-", label="Inference Accuracy")
    if param == "ler_std_dev":
        ax1.set_xlabel("Splitter Ratio Std. Dev.")
    elif param == "lens_distortion_strength":
        ax1.set_xlabel("Lens distortion strength (α)")
    elif param == "laser_rin_db":
        ax1.set_xlabel("Laser RIN (dB, RMS fraction)")
    elif param == "pd_noise_w":
        ax1.set_xlabel("PD input noise (W RMS)")
    else:
        ax1.set_xlabel("Distorted TF Ratio (α)")
    if param == "pd_noise_w":
        ax1.set_xscale("symlog", linthresh=1e-10)
    ax1.set_ylabel("Inference Accuracy (%)", color="b")
    # Fix y-axis limits for comparability
    ax1.set_ylim(ACC_YLIM)
    if SHOW_TITLES:
        ax1.set_title(f"{param} sweep", fontsize=DEFAULT_TITLE_FONTSIZE)
    ax2 = ax1.twinx()
    ax2.plot(strengths_plot, sndrs, "r^-", label="SNDR (pJTC output)")
    ax2.plot(strengths_plot, jps_sndrs, "gs--", label="SNDR (JPS)")
    ax2.set_ylabel("SNDR (dB)", color="r")
    ax2.set_ylim(SNDR_YLIM)
    # Combine legends from both axes into a single legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_loc = (
        "lower left" if param in {"laser_rin_db", "pd_noise_w"} else "upper right"
    )
    ax1.legend(handles1 + handles2, labels1 + labels2, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{param}_sweep.pdf"))
    plt.close(fig)


# -----------------------------------------------------------------------------
#  2-D sweep for JTC separation and total field
# -----------------------------------------------------------------------------


def _compute_jtc_sndrs(cfg: AppConfig, num_tests: int = 1000, seed: int = 0) -> float:
    """Compute SNDR between two JTC *forward* outputs.

    The comparison is between:
      • The **current** configuration (*cfg*).
      • A **reference** configuration with the same parameters except
        ``jtc_separation = max(cfg.input_length, cfg.kernel_length)`` and
        ``jtc_total_field = 6 * max(cfg.input_length, cfg.kernel_length)``.

    Distortion-strength parameters are **not** zeroed – the goal is to
    isolate the effect of geometry (separation / total-field).

    """

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Current JTC
    jtc_cur = _build_jtc(cfg).to(device)

    # Reference geometry (same distortion settings)
    ref_len = int(max(cfg.input_length, cfg.kernel_length))
    ref_cfg = replace(
        cfg,
        jtc_separation=ref_len,
        jtc_total_field=ref_len * 6,
    )
    jtc_ref = _build_jtc(ref_cfg).to(device)

    out_list = []
    ref_list = []

    for _ in range(num_tests):
        signal = torch.rand(1, 1, 1, cfg.input_length, device=device)
        kernel = torch.rand(1, cfg.kernel_length, device=device)

        out_list.append(jtc_cur(signal, kernel))
        ref_list.append(jtc_ref(signal, kernel))

    out = torch.stack(out_list)
    ref = torch.stack(ref_list)
    noise = out - ref

    sndr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())

    return sndr.item()


def _sweep_jtc_2d(base_cfg: AppConfig, weights: str, out_dir: str) -> None:
    """Run a 2-D sweep over (*jtc_separation*, *jtc_total_field*).

    A nested loop explores a grid of valid separation / total-field
    combinations, measuring the classification accuracy.  Results are saved
    as a NumPy matrix + a PDF heat-map for quick visualisation.
    """

    # Define sweep ranges (feel free to adjust as needed)
    max_sep = int(max(base_cfg.input_length, base_cfg.kernel_length))
    sep_values = np.arange(0, max_sep + 1 + 4)  # 0 … 8 for default cfg
    # Ensure the smallest field is the minimal valid value
    min_field = int(base_cfg.input_length + base_cfg.kernel_length)
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
        "--params",
        type=str,
        default="",
        help=(
            "Comma-separated list of params to sweep (default: all). "
            f"Options: {','.join(PARAMS)}"
        ),
    )
    parser.add_argument(
        "--skip-jtc-2d",
        action="store_true",
        help="Skip the JTC geometry 2D sweep.",
    )
    parser.add_argument(
        "--snr-tests",
        type=int,
        default=10000,
        help="Number of random trials to use for SNDR/ENOB estimates.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: limit test batches during inference accuracy evaluation.",
    )
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
        if args.params:
            sweep_params = [p.strip() for p in args.params.split(",") if p.strip()]
        else:
            sweep_params = list(PARAMS)
        unknown = sorted(set(sweep_params) - set(PARAMS))
        if unknown:
            raise ValueError(f"Unknown params requested: {unknown}. Options: {PARAMS}")
        for param in sweep_params:
            _plot_param_sweep_from_data(param, args.output_dir)

        # Generate 2D JTC sweep plots
        if not args.skip_jtc_2d:
            _plot_jtc_2d_from_data(args.output_dir)

        print("Plot generation completed.")

    else:
        # Run full sweep with inference
        if not args.config or not args.weights:
            parser.error(
                "--config and --weights are required when not using --plot-only"
            )

        base_cfg = load_config_from_yaml(args.config)
        if args.max_eval_batches is not None:
            base_cfg = replace(base_cfg, max_eval_batches=int(args.max_eval_batches))

        if args.params:
            sweep_params = [p.strip() for p in args.params.split(",") if p.strip()]
        else:
            sweep_params = list(PARAMS)
        unknown = sorted(set(sweep_params) - set(PARAMS))
        if unknown:
            raise ValueError(f"Unknown params requested: {unknown}. Options: {PARAMS}")

        default_strengths = np.arange(0, 1.1, 0.1)
        for param in sweep_params:
            if param == "ler_std_dev":
                strengths = np.arange(0, 0.05, 0.005)
            elif param == "laser_rin_db":
                strengths = LASER_RIN_DB_SWEEP
            elif param == "pd_noise_w":
                strengths = PD_NOISE_W_SWEEP
            else:
                strengths = default_strengths

            _sweep_param(
                base_cfg,
                args.weights,
                param,
                args.output_dir,
                strengths,
                snr_tests=int(args.snr_tests),
            )
        # 2-D sweep (after 1-D distortion sweeps)
        if not args.skip_jtc_2d:
            _sweep_jtc_2d(base_cfg, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
