import argparse
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from onn_config import AppConfig
from onn_inference import (
    compute_snr_enob,
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
    for val in strengths:
        cfg = replace(base_cfg, **{param: float(val)})
        acc = run_inference(cfg, weights)
        snr, enob = compute_snr_enob(cfg, param)
        accs.append(acc)
        snrs.append(snr)
        enobs.append(enob)
        print(f"{param}={val:.1f} -> acc {acc:.2f}% snr {snr:.2f}dB enob {enob:.2f}")

    results = np.stack([strengths, accs, snrs, enobs], axis=1)
    np.savetxt(
        os.path.join(out_dir, f"{param}_results.txt"),
        results,
        header="strength accuracy snr enob",
    )

    fig, ax1 = plt.subplots()
    ax1.plot(strengths, accs, "bo-", label="Accuracy")
    ax1.set_xlabel("distortion_strength")
    ax1.set_ylabel("Accuracy (%)", color="b")
    ax2 = ax1.twinx()
    ax2.plot(strengths, snrs, "r^-", label="SNR")
    ax2.set_ylabel("SNR (dB)", color="r")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{param}_sweep.pdf"))
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


if __name__ == "__main__":
    main()
