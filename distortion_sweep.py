import argparse
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from onn_inference import _build_jtc
from typing import Iterable

from onn_config import AppConfig
from onn_inference import (
    compute_snr_enob,
    compute_snr_jps,
    load_config_from_yaml,
    run_inference,
)

PARAMS = [
    "driver_distortion_strength",
    "pd_tia_distortion_strength",
    "mrm_power_distortion_strength",
    "mrm_phase_distortion_strength",
]


def _sweep_param(base_cfg: AppConfig, weights: str, param: str, out_dir: str, strengths: Iterable[int]) -> None:
    accs = []
    snrs = []
    enobs = []
    jps_snrs = []
    for val in strengths:
        cfg = replace(base_cfg, **{param: float(val)})
        acc = run_inference(cfg, weights)
        snr, enob = compute_snr_enob(cfg, param, num_tests=10000)
        snr_jps = compute_snr_jps(cfg, param, num_tests=10000)
        accs.append(acc)
        snrs.append(snr)
        enobs.append(enob)
        jps_snrs.append(snr_jps)
        print(f"{param}={val:.3f} -> acc {acc:.3f}% snr {snr:.2f}dB jps_snr {snr_jps:.2f}dB")

    results = np.stack([strengths, accs, snrs, enobs, jps_snrs], axis=1)
    np.savetxt(
        os.path.join(out_dir, f"{param}_results.txt"),
        results,
        header="strength accuracy snr enob jps_snr",
    )

    fig, ax1 = plt.subplots()
    ax1.plot(strengths, accs, "bo-", label="Accuracy")
    ax1.set_xlabel("distortion_strength")
    ax1.set_ylabel("Accuracy (%)", color="b")
    ax2 = ax1.twinx()
    ax2.plot(strengths, snrs, "r^-", label="SNR (output)")
    ax2.plot(strengths, jps_snrs, "gs--", label="SNR (JPS)")
    ax2.set_ylabel("SNR (dB)", color="r")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{param}_sweep.pdf"))
    plt.close(fig)


# -----------------------------------------------------------------------------
#  2-D sweep for JTC separation and total field
# -----------------------------------------------------------------------------


def _compute_jtc_snrs(cfg: AppConfig, num_tests: int = 1000, seed: int = 0):
    """Compute SNR between two JTC *forward* outputs.

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

    snr = 10.0 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())

    # Return duplicate values to keep (snr_out, snr_jps) signature unchanged
    return snr.item()


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
    field_values = np.arange(min_field, min_field + 41, 1)  # 16,20,24,28,32

    acc_matrix = np.full((len(sep_values), len(field_values)), np.nan)
    snr_out_matrix = np.full_like(acc_matrix, np.nan, dtype=float)

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

            try:
                acc = run_inference(cfg, weights)
                snr = _compute_jtc_snrs(cfg, num_tests=1000)

                acc_matrix[i, j] = acc
                snr_out_matrix[i, j] = snr

                print(
                    f"jtc_sep={sep}, jtc_total_field={field} -> acc {acc:.2f}% "
                    f"SNR_out {snr:.2f}dB"
                )
            except ValueError as e:
                print(f"[SKIP] sep={sep}, field={field}: {e}")

    # Persist results
    np.save(os.path.join(out_dir, "jtc_2d_accuracy.npy"), acc_matrix)
    np.save(os.path.join(out_dir, "jtc_2d_snr_output.npy"), snr_out_matrix)

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
        os.path.join(out_dir, "jtc_2d_snr_output.txt"),
        snr_out_matrix,
        fmt="%0.2f",
        header=(
            "Rows: jtc_separation values {}\nCols: jtc_total_field values {}".format(
                list(sep_values), list(field_values)
            )
        ),
    )

    # Heat-maps --------------------------------------------------------
    def _plot_heat(data, title, filename, cbar_label):
        fig, ax = plt.subplots()
        im = ax.imshow(data, origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_xticks(np.arange(len(field_values)))
        ax.set_xticklabels(field_values)
        ax.set_yticks(np.arange(len(sep_values)))
        ax.set_yticklabels(sep_values)
        ax.set_xlabel("jtc_total_field")
        ax.set_ylabel("jtc_separation")
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)

    _plot_heat(acc_matrix, "Accuracy", "jtc_2d_sweep_accuracy.pdf", "Accuracy (%)")
    _plot_heat(snr_out_matrix, "Output SNR", "jtc_2d_sweep_output_snr.pdf", "SNR (dB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distortion sweep inference")
    parser.add_argument("--config", required=True, help="config yaml")
    parser.add_argument("--weights", required=True, help="trained weights path")
    parser.add_argument("--output-dir", default="sweep_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_cfg = load_config_from_yaml(args.config)

    default_strengths = np.arange(0, 1.1, 0.1)
    for param in PARAMS:
        _sweep_param(base_cfg, args.weights, param, args.output_dir, default_strengths)

    # 2-D sweep (after 1-D distortion sweeps)
    _sweep_jtc_2d(base_cfg, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
