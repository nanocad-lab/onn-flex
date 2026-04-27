"""
Quick random-vector SNR checks for the analog JTC pipeline.

This script complements `scripts/distortion_sweep.py` by providing a fast way to
inspect SNDR (relative to an "idealized" reference config for a given parameter)
and SNQR (quantization-only) without running full model inference.

Examples:

  # SNDR of PD noise / laser RIN at the current config values
  python scripts/check_sndr_snqr.py --config runs/runs_ideal_0825/final_config.yaml --params pd_noise_w,laser_rin_db

  # Quantization SNQR (overall + per-quantizer)
  python scripts/check_sndr_snqr.py --config runs/runs_ideal_0825/final_config.yaml --snqr
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is importable when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from onn_inference import (
    compute_sqndr_enob,
    compute_sndr_vs_quantized_ideal_enob,
    compute_snqr_enob,
    compute_snr_enob,
    compute_snr_jps,
    load_config_from_yaml,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-vector SNDR/SNQR checks")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help=(
            "Override config fields (repeatable), e.g. --set laser_rin_db=-30 "
            "--set pd_noise_w=1e-8 --set dac_bits=4. Use 'null' to set None."
        ),
    )
    parser.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated distortion params to report SNDR for (e.g. pd_noise_w,laser_rin_db).",
    )
    parser.add_argument(
        "--snqr",
        action="store_true",
        help="Also report SNQR for quantization (overall + per-quantizer).",
    )
    parser.add_argument(
        "--sqndr",
        action="store_true",
        help="Also report SQNDR (combined quant + distortion) vs an ideal reference.",
    )
    parser.add_argument(
        "--sndr-vs-quantized-ideal",
        action="store_true",
        help="Also report SNDR vs a quantized-ideal reference (distortions off, quant on).",
    )
    parser.add_argument(
        "--snr-tests",
        type=int,
        default=10000,
        help="Number of random trials for the estimate.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _parse_overrides(items: list[str]) -> dict:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE in --set, got: {item!r}")
        key, value_raw = item.split("=", 1)
        key = key.strip()
        value_raw = value_raw.strip()
        low = value_raw.lower()
        if low in {"none", "null"}:
            value: object = None
        elif low in {"true", "false"}:
            value = low == "true"
        else:
            try:
                value = int(value_raw)
            except ValueError:
                try:
                    value = float(value_raw)
                except ValueError:
                    value = value_raw
        overrides[key] = value
    return overrides


def main() -> None:
    args = _parse_args()

    cfg = load_config_from_yaml(args.config)
    overrides = _parse_overrides(list(args.set or []))
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)

    print("Config:", args.config)
    print(
        "Quant bits:",
        f"dac_bits={cfg.dac_bits} fourier_plane_bits={cfg.fourier_plane_bits} adc_bits={cfg.adc_bits}",
    )
    print("Noise:", f"laser_rin_db={cfg.laser_rin_db} pd_noise_w={cfg.pd_noise_w}")
    print(
        "Lens:",
        f"lens_distortion_strength={cfg.lens_distortion_strength} lens_legendre_order={cfg.lens_legendre_order}",
    )

    params = [p.strip() for p in (args.params or "").split(",") if p.strip()]
    for param in params:
        sndr, enob = compute_snr_enob(
            cfg, param, num_tests=args.snr_tests, seed=args.seed
        )
        sndr_jps = compute_snr_jps(cfg, param, num_tests=args.snr_tests, seed=args.seed)
        print(
            f"SNDR({param}): output={sndr:.2f} dB (ENOB~{enob:.2f}) | JPS={sndr_jps:.2f} dB"
        )

    if args.sqndr:
        sqndr, enob = compute_sqndr_enob(cfg, num_tests=args.snr_tests, seed=args.seed)
        print(f"SQNDR(total): {sqndr:.2f} dB (ENOB~{enob:.2f})")

    if args.sndr_vs_quantized_ideal:
        sndr_q, enob_q = compute_sndr_vs_quantized_ideal_enob(
            cfg, num_tests=args.snr_tests, seed=args.seed
        )
        print(f"SNDR(vs quantized ideal): {sndr_q:.2f} dB (ENOB~{enob_q:.2f})")

    if args.snqr:
        # Total quantization effect (disable all quantizers in the reference)
        snqr_all, enob_all = compute_snqr_enob(
            cfg,
            quantizers=("dac", "fourier_plane", "adc"),
            num_tests=args.snr_tests,
            seed=args.seed,
        )
        print(f"SNQR(all quant): {snqr_all:.2f} dB (ENOB~{enob_all:.2f})")

        for q in ("dac", "fourier_plane", "adc"):
            snqr_q, enob_q = compute_snqr_enob(
                cfg, quantizers=(q,), num_tests=args.snr_tests, seed=args.seed
            )
            print(f"SNQR({q}): {snqr_q:.2f} dB (ENOB~{enob_q:.2f})")


if __name__ == "__main__":
    main()
