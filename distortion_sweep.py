import argparse
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

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


def _sweep_param(base_cfg: AppConfig, weights: str, param: str, out_dir: str) -> None:
    strengths = np.arange(0.0, 1.1, 0.1)
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
        print(f"{param}={val:.1f} -> acc {acc:.3f}% snr {snr:.2f}dB jps_snr {snr_jps:.2f}dB")

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


def _sweep_jtc_2d(base_cfg: AppConfig, weights: str, out_dir: str) -> None:
    """Run a 2-D sweep over (*jtc_separation*, *jtc_total_field*).

    A nested loop explores a grid of valid separation / total-field
    combinations, measuring the classification accuracy.  Results are saved
    as a NumPy matrix + a PDF heat-map for quick visualisation.
    """

    # Define sweep ranges (feel free to adjust as needed)
    sep_values = np.arange(0, base_cfg.jtc_half_size + 1)  # 0 … 8 for default cfg
    # Ensure the smallest field is the minimal valid value
    min_field = 2 * base_cfg.jtc_half_size
    field_values = np.arange(min_field, min_field + 17, 4)  # 16,20,24,28,32

    acc_matrix = np.full((len(sep_values), len(field_values)), np.nan)

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
                acc_matrix[i, j] = acc
                print(f"jtc_sep={sep}, jtc_total_field={field} -> acc {acc:.2f}%")
            except ValueError as e:
                print(f"[SKIP] sep={sep}, field={field}: {e}")

    # Persist results
    np.save(os.path.join(out_dir, "jtc_2d_accuracy.npy"), acc_matrix)
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

    # Heat-map
    fig, ax = plt.subplots()
    im = ax.imshow(acc_matrix, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(field_values)))
    ax.set_xticklabels(field_values)
    ax.set_yticks(np.arange(len(sep_values)))
    ax.set_yticklabels(sep_values)
    ax.set_xlabel("jtc_total_field")
    ax.set_ylabel("jtc_separation")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "jtc_2d_sweep_accuracy.pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distortion sweep inference")
    parser.add_argument("--config", required=True, help="config yaml")
    parser.add_argument("--weights", required=True, help="trained weights path")
    parser.add_argument("--output-dir", default="sweep_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_cfg = load_config_from_yaml(args.config)

    for param in PARAMS:
        _sweep_param(base_cfg, args.weights, param, args.output_dir)

    # 2-D sweep (after 1-D distortion sweeps)
    _sweep_jtc_2d(base_cfg, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
