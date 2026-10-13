"""Run the preliminary 2D packaging study and write a markdown report + CSV.

Usage (repository root):

    python -m packaging2d.study --output-dir runs/packaging2d
    python -m packaging2d.study --report packaging2d/RESULTS.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import replace
from pathlib import Path

from packaging2d import model as m

PIXEL_OPTIONS = {
    # label: (mode, in-pixel pitch)
    "ideal-5um": ("ideal", 5.0),
    "in-pixel-10um": ("in_pixel", 10.0),
    "in-pixel-15um": ("in_pixel", 15.0),
    "fan-out": ("fanout", 15.0),
}
VIA_OPTIONS = {
    "TGV AR10": {"via__kind": "TGV", "via__aspect_ratio": 10.0},
    "TGV AR20": {"via__kind": "TGV", "via__aspect_ratio": 20.0},
    "TSV AR10": {"via__kind": "TSV", "via__aspect_ratio": 10.0},
    "TSV AR20": {"via__kind": "TSV", "via__aspect_ratio": 20.0},
}
COMPRESSIONS = (1.0, 2.0, 4.0, 8.0)
N_VALUES = (16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 100, 112, 128)
F_MOD = (1e9, 1e10)


def _fmt(value, digits=3):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if value == 0:
            return "0"
        mag = abs(value)
        if mag >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{max(digits - 1 - int(math.floor(math.log10(mag))), 0)}f}"
    return str(value)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _params(pixel: str, **changes) -> m.StackParams:
    mode, pitch = PIXEL_OPTIONS[pixel]
    return m.with_overrides(
        m.StackParams(), pixel_mode=mode, optics__pixel_pitch_um=pitch, **changes
    )


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------


def optics_section() -> str:
    rows = []
    for label in PIXEL_OPTIONS:
        p = _params(label)
        for n in (32, 50, 100):
            dense, sparse = m.pixel_pitches_um(n, p.optics, p.pixel_mode)
            f_eff = m.required_focal_length_um(n, p.optics, p.pixel_mode)
            rows.append(
                [
                    label,
                    n,
                    f"{dense:g} x {sparse:g}",
                    f_eff / 1e3,
                    f_eff / 500.0,
                    m.fourier_fill_factor(n, p.optics, p.pixel_mode),
                ]
            )
    nmax = []
    for label, (mode, _) in PIXEL_OPTIONS.items():
        optics = _params(label).optics
        nmax.append(
            [label]
            + [
                m.max_n_for_gap(gap, replace(optics, compression=g), mode)
                for gap, g in ((500, 1), (500, 4), (1000, 4), (3000, 1), (3000, 4))
            ]
        )
    return "\n\n".join(
        [
            "## 1. Optics sets the gap",
            "DFT-matched Fourier sampling requires `p_in * p_F = (lambda/n) f / N`"
            " per axis, so `f_eff = n N p_in p_F / lambda` (lambda = 1.31 um,"
            " n = 1.45). Compression G = f_eff / physical gap.",
            _table(
                [
                    "pixel option",
                    "N",
                    "pitch (um)",
                    "f_eff (mm)",
                    "G for 500 um",
                    "Fourier fill",
                ],
                rows,
            ),
            "Largest N per dimension that fits a gap at compression G:",
            _table(
                [
                    "pixel option",
                    "0.5 mm, G=1",
                    "0.5 mm, G=4",
                    "1 mm, G=4",
                    "3 mm, G=1",
                    "3 mm, G=4",
                ],
                nmax,
            ),
        ]
    )


def via_section() -> str:
    rows = []
    link = m.LinkParams()
    for kind, ar, gap in itertools.product(
        ("TGV", "TSV"), (10.0, 20.0), (500.0, 1000.0, 2000.0, 3000.0)
    ):
        lm = m.link_metrics(gap, m.ViaTech(kind=kind, aspect_ratio=ar), link)
        rows.append(
            [
                kind,
                ar,
                gap / 1e3,
                f"{lm['diameter_um']:g}/{lm['pitch_um']:g}",
                lm["c_via_ff"],
                lm["f3db_ghz"],
                lm["analog_gsps"],
                lm["nrz_gbps"],
                lm["energy_pj_per_sample"],
                lm["lanes_per_mm2"],
                lm["analog_gsps"] >= 0.5,
                lm["analog_gsps"] >= 5.0,
            ]
        )
    return "\n\n".join(
        [
            "## 2. TSV/TGV lane bandwidth",
            "Series R_drv (50 ohm) + via R/L into via C + pads + load; f3dB is the"
            " second-order -3 dB point. Analog rate = 6-bit settling in half a"
            " sample; NRZ = 2 x f3dB. TSV liner = 2% of diameter (1 um at 50 um)"
            " with 0.7 depletion factor; TGV in glass (er 4.8) with G-S-G returns."
            " Pass columns compare against the A/B Fourier-loop sample rate per"
            " lane at 1 GHz (0.5 GS/s) and 10 GHz (5 GS/s) modulation.",
            _table(
                [
                    "via",
                    "AR",
                    "gap (mm)",
                    "d/pitch (um)",
                    "C (fF)",
                    "f3dB (GHz)",
                    "analog GS/s",
                    "NRZ Gb/s",
                    "pJ/sample",
                    "lanes/mm2",
                    "1 GHz ok",
                    "10 GHz ok",
                ],
                rows,
            ),
        ]
    )


def crossing_section(n: int = 100) -> str:
    """Ways to carry the A/B Fourier loop (n^2 samples per pass) across the gap."""
    link, pw = m.LinkParams(), m.PowerParams()
    e_digital = (
        pw.adc_fom_fj * 2 ** (link.analog_bits - 1) * 1e-3
        + pw.dac_pj
        + link.analog_bits * pw.serdes_pj_per_bit
    )
    rows = []
    for f_mod in F_MOD:
        f_shot = f_mod / 2
        rate = n * n * f_shot
        for kind, gap in (("TGV", 500.0), ("TSV", 500.0), ("TGV", 1000.0)):
            via = m.ViaTech(kind=kind)
            lm = m.link_metrics(gap, via, link)
            lane_area = lm["lane_area_um2"] * 1e-6
            direct_ok = lm["analog_gsps"] * 1e9 >= f_shot
            tdm = int(min(lm["analog_gsps"], link.tdm_max_gsps) * 1e9 // f_shot)
            tdm = min(tdm, link.tdm_max)
            digital_lanes = math.ceil(rate * link.analog_bits / (lm["nrz_gbps"] * 1e9))
            options = [
                (
                    "analog 1:1",
                    n * n if direct_ok else None,
                    lm["energy_pj_per_sample"],
                ),
                (
                    f"analog TDM x{tdm}" if tdm >= 2 else "analog TDM",
                    math.ceil(n * n / tdm) if tdm >= 2 else None,
                    lm["energy_pj_per_sample"] + link.tdm_pj_per_sample,
                ),
                ("digital ADC-SerDes-DAC", digital_lanes, e_digital),
            ]
            for name, lanes, e_pj in options:
                rows.append(
                    [
                        f_mod / 1e9,
                        f"{kind} {gap / 1e3:g} mm",
                        name,
                        "rate-limited" if lanes is None else lanes,
                        "-" if lanes is None else lanes * lane_area,
                        "-" if lanes is None else rate * e_pj * 1e-12,
                    ]
                )
    return "\n\n".join(
        [
            f"## 3. Crossing the gap with the Fourier loop (A/B, N={n})",
            "A and B time-share one lens, so each shot is two passes and the"
            f" {n * n:,} Fourier-plane samples cross the gap once per shot"
            " (shot rate = f_mod / 2). Analog TDM is capped at 4 GS/s per lane by"
            " the S/H + mux circuits. Digital = 6-bit ADC + SerDes + DAC per"
            " sample. C closes this loop on the bottom die and needs none of it.",
            _table(
                ["f_mod (GHz)", "via", "option", "lanes", "area (mm2)", "power (W)"],
                rows,
            ),
        ]
    )


ARCH_KEYS = [
    "gap_um",
    "compression_needed",
    "f_shot_ghz",
    "limiter",
    "speedup_workload",
    "speedup_worst_layer",
    "tmacs",
    "analog_lanes",
    "tdm_ratio",
    "digital_lanes",
    "power_vias",
    "via_field_mm2",
    "p_top_w",
    "p_eic_w",
    "p_bottom_w",
    "p_gap_lanes_w",
    "p_cross_gap_w",
    "p_stack_w",
    "p_laser_w",
    "pj_per_mac",
    "footprint_mm2",
    "footprint_thermal_mm2",
    "t_top_c",
    "t_eic_c",
    "t_bottom_c",
    "mem_gbytes_s",
    "mem_devices",
    "failing",
]


def baseline_section() -> tuple[str, list[dict]]:
    """Architectures at the stated assumptions: 500 um gap, 50/100 um TGV."""
    records = []
    for f_mod, kind in itertools.product(F_MOD, ("TGV", "TSV")):
        for key in m.ARCHITECTURES:
            p = _params("ideal-5um", gap_um=500.0, f_mod_hz=f_mod, via__kind=kind)
            records.append(m.evaluate(key, p))
    sections = ["## 4. Architectures at the stated assumptions"]
    sections.append(
        "N = 100, 500 um gap, AR 10 (50 um vias on 100 um pitch), 5 um GC pitch"
        " (best-case optics; still needs 5.5x compression). Thermal footprint ="
        " smallest stack area that keeps photonic dies <= 85 C and the EIC <="
        " 105 C (cap 400 mm2). Speedup = MACs/s vs the 1D F=256 JTC at the same"
        " modulation rate over CIFAR ResNet-18 stages (equal MACs per stage, so"
        " a harmonic mean); speedup_worst_layer is the 4x4-map stage alone."
    )
    for f_mod, kind in itertools.product(F_MOD, ("TGV", "TSV")):
        subset = [
            r
            for r in records
            if r["f_mod_ghz"] == f_mod / 1e9 and r["via_kind"] == kind
        ]
        rows = [[k] + [r[k] for r in subset] for k in ARCH_KEYS]
        sections.append(f"### {f_mod / 1e9:g} GHz, {kind}")
        sections.append(_table(["metric", "A", "B", "C"], rows))
    for key, arch in m.ARCHITECTURES.items():
        sections.append(f"- **{key}** - {arch.name}: {arch.organization}")
    return "\n\n".join(sections), records


def sweep() -> list[dict]:
    records = []
    for pixel, via_label, g, n, f_mod in itertools.product(
        PIXEL_OPTIONS, VIA_OPTIONS, COMPRESSIONS, N_VALUES, F_MOD
    ):
        p = _params(
            pixel, n=n, f_mod_hz=f_mod, optics__compression=g, **VIA_OPTIONS[via_label]
        )
        for key in m.ARCHITECTURES:
            r = m.evaluate(key, p)
            r["pixel_option"] = pixel
            r["via_option"] = via_label
            records.append(r)
    return records


def feasibility_section(records: list[dict]) -> str:
    """Largest feasible N per arch / rate / pixel / compression (best via)."""
    header = ["pixel option", "G"] + [
        f"{a} @ {f / 1e9:g} GHz" for a in m.ARCHITECTURES for f in F_MOD
    ]
    rows = []
    for pixel, g in itertools.product(PIXEL_OPTIONS, COMPRESSIONS):
        row = [pixel, g]
        for a, f in itertools.product(m.ARCHITECTURES, F_MOD):
            ok = [
                r
                for r in records
                if r["pixel_option"] == pixel
                and r["compression"] == g
                and r["arch"] == a
                and r["f_mod_ghz"] == f / 1e9
                and r["feasible"]
            ]
            row.append(max(r["n"] for r in ok) if ok else "-")
        rows.append(row)

    best_rows = []
    for a, f in itertools.product(m.ARCHITECTURES, F_MOD):
        ok = [
            r
            for r in records
            if r["arch"] == a and r["f_mod_ghz"] == f / 1e9 and r["feasible"]
        ]
        ok.sort(
            key=lambda r: (-r["tmacs"], r["compression"], r["footprint_thermal_mm2"])
        )
        for r in ok[:2]:
            best_rows.append(
                [
                    a,
                    f / 1e9,
                    r["n"],
                    r["pixel_option"],
                    r["compression"],
                    r["via_option"],
                    r["gap_um"],
                    r["speedup_workload"],
                    r["tmacs"],
                    r["p_stack_w"],
                    r["footprint_thermal_mm2"],
                    r["mem_devices"],
                ]
            )
        if not ok:
            best_rows.append([a, f / 1e9] + ["none"] + ["-"] * 9)

    min_n = []
    for a, f in itertools.product(m.ARCHITECTURES, F_MOD):
        hits = [
            r["n"]
            for r in records
            if r["arch"] == a
            and r["f_mod_ghz"] == f / 1e9
            and r["ok_target"]
            and r["ok_rate"]
        ]
        min_n.append([a, f / 1e9, min(hits) if hits else "-"])
    return "\n\n".join(
        [
            "## 5. Feasible design points (optics-derived gap, cap 3 mm)",
            "Feasible = optics fits the gap cap, thermal fits within 400 mm2, and"
            " the workload speedup over 1D is >= 10x (after any gap-lane rate"
            " limit). Cells give the largest feasible N per dimension over the"
            f" swept N {list(N_VALUES)} and via options.",
            _table(header, rows),
            "Smallest N reaching 10x at full shot rate (geometry only):",
            _table(["arch", "f_mod (GHz)", "N_min"], min_n),
            "Highest-throughput feasible points:",
            _table(
                [
                    "arch",
                    "f_mod",
                    "N",
                    "pixel",
                    "G",
                    "via",
                    "gap (um)",
                    "speedup",
                    "TMAC/s",
                    "stack W",
                    "footprint mm2",
                    "memory",
                ],
                best_rows,
            ),
        ]
    )


def target_section(records: list[dict], n: int = 100) -> str:
    """Least optical compression that makes the N x N target feasible."""
    rows = []
    for a, f, pixel in itertools.product(m.ARCHITECTURES, F_MOD, PIXEL_OPTIONS):
        ok = [
            r
            for r in records
            if r["n"] == n
            and r["arch"] == a
            and r["f_mod_ghz"] == f / 1e9
            and r["pixel_option"] == pixel
            and r["feasible"]
        ]
        if not ok:
            rows.append([a, f / 1e9, pixel, f"> {max(COMPRESSIONS):g}"] + ["-"] * 8)
            continue
        r = min(ok, key=lambda r: (r["compression"], r["footprint_thermal_mm2"]))
        rows.append(
            [
                a,
                f / 1e9,
                pixel,
                r["compression"],
                r["via_option"],
                r["gap_um"],
                r["speedup_workload"],
                r["tmacs"],
                r["p_stack_w"],
                r["p_laser_w"],
                r["footprint_thermal_mm2"],
                r["mem_devices"],
            ]
        )
    return "\n\n".join(
        [
            f"## 6. Design points for the {n} x {n} target",
            "Least swept compression G that makes each option feasible, then the"
            " smallest thermal footprint among via options.",
            _table(
                [
                    "arch",
                    "f_mod",
                    "pixel",
                    "G",
                    "via",
                    "gap (um)",
                    "speedup",
                    "TMAC/s",
                    "stack W",
                    "laser W",
                    "footprint mm2",
                    "memory",
                ],
                rows,
            ),
        ]
    )


SENSITIVITY_CASES = (
    ("baseline", {}),
    ("heaters 0.5 mW/ring", {"power__heater_mw_per_ring": 0.5}),
    ("heaters 3 mW/ring", {"power__heater_mw_per_ring": 3.0}),
    (
        "TIA + MRM driver energy / 2",
        {"power__tia_pj": 0.2, "power__mrm_drive_pj": 0.125},
    ),
    ("analog accumulation x8 before ADC", {"workload__analog_accum_depth": 8}),
    (
        "both levers above",
        {
            "power__tia_pj": 0.2,
            "power__mrm_drive_pj": 0.125,
            "workload__analog_accum_depth": 8,
        },
    ),
    ("cold plate 0.15 K cm2/W", {"thermal__r_sink_kcm2_w": 0.15}),
    ("gap via budget 250 mm2", {"via_area_budget_mm2": 250.0}),
)


def sensitivity_section() -> str:
    rows = []
    for label, changes in SENSITIVITY_CASES:
        row = [label]
        for f, a in itertools.product(F_MOD, m.ARCHITECTURES):
            p = _params("ideal-5um", gap_um=500.0, f_mod_hz=f, **changes)
            r = m.evaluate(a, p)
            cell = f"{r['p_stack_w']:.0f} W, {r['footprint_thermal_mm2']:.0f} mm2"
            if not r["ok_thermal"]:
                cell += " HOT"
            if not r["ok_target"]:
                cell += " <10x"
            row.append(cell)
        rows.append(row)
    header = ["case"] + [
        f"{a} @ {f / 1e9:g} GHz" for f in F_MOD for a in m.ARCHITECTURES
    ]
    return "\n\n".join(
        [
            "## 7. Power and thermal sensitivity (N = 100, 500 um TGV gap)",
            "Cells: stack power (excludes the off-stack laser) and the thermal"
            " footprint. HOT = exceeds limits even at 400 mm2; <10x = gap lanes"
            " throttle the shot rate below the target.",
            _table(header, rows),
        ]
    )


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(records[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def build_report() -> tuple[str, list[dict], list[dict]]:
    baseline_md, baseline = baseline_section()
    records = sweep()
    parts = [
        "# 2D pJTC packaging study (generated)",
        "Generated by `python -m packaging2d.study`. First-order placeholders;"
        " see `packaging2d/README.md` for assumptions.",
        optics_section(),
        via_section(),
        crossing_section(),
        baseline_md,
        feasibility_section(records),
        target_section(records),
        sensitivity_section(),
    ]
    return "\n\n".join(parts) + "\n", baseline, records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="runs/packaging2d")
    parser.add_argument("--report", default=None, help="also write report here")
    args = parser.parse_args(argv)
    report, baseline, records = build_report()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report)
    write_csv(baseline, out / "baseline.csv")
    write_csv(records, out / "sweep.csv")
    if args.report:
        Path(args.report).write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
