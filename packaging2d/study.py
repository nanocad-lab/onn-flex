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
from pathlib import Path

from packaging2d import model as m

VIA_OPTIONS = {
    "TGV AR10": {"via__kind": "TGV", "via__aspect_ratio": 10.0},
    "TGV AR20": {"via__kind": "TGV", "via__aspect_ratio": 20.0},
    "TSV AR10": {"via__kind": "TSV", "via__aspect_ratio": 10.0},
    "TSV AR20": {"via__kind": "TSV", "via__aspect_ratio": 20.0},
}
VIA_BUDGETS = (100.0, 200.0, 400.0)
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


def _params(**changes) -> m.StackParams:
    return m.with_overrides(m.StackParams(), **changes)


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------


def assumptions_section() -> str:
    p = m.StackParams()
    d_um, pitch_um = m.via_geometry_um(p.gap_um, p.via)
    return "\n\n".join(
        [
            "## 1. Assumptions",
            f"N = {p.n} per side ({p.n * p.n:,} channels), "
            f"{p.optics.pixel_pitch_um:g} um pixel pitch, "
            f"{p.gap_um:g} um between emitting and receiving grating couplers. "
            "The optics are taken as given; the separation is the middle-layer "
            f"thickness, so every vertical via is {p.gap_um:g} um long "
            f"({d_um:g} um diameter on {pitch_um:g} um pitch at AR "
            f"{p.via.aspect_ratio:g}:1). Rates: 1 and 10 GHz modulation. 1D "
            f"baseline: F = {p.baseline_1d_field}, "
            f"{m.macs_per_shot_1d(p.baseline_1d_field, p.kernel):g} MAC per shot.",
        ]
    )


def via_section() -> str:
    rows = []
    for kind, ar, r_drv in itertools.product(
        ("TGV", "TSV"), (10.0, 20.0), (50.0, 100.0)
    ):
        lm = m.link_metrics(
            500.0,
            m.ViaTech(kind=kind, aspect_ratio=ar),
            m.LinkParams(r_drv_ohm=r_drv),
        )
        rows.append(
            [
                kind,
                ar,
                r_drv,
                f"{lm['diameter_um']:g}/{lm['pitch_um']:g}",
                lm["c_via_ff"],
                lm["l_nh"],
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
            "## 2. TSV/TGV lane bandwidth (500 um gap)",
            "Series driver resistance + via R/L into via C + 2 x 20 fF pads + 20 fF"
            " load; f3dB is the second-order -3 dB point. Analog rate = 6-bit"
            " settling in half a sample; NRZ = 2 x f3dB. TSV liner = 2% of the"
            " diameter with a 0.7 depletion factor; TGV in glass (er 4.8) with"
            " G-S-G returns. The ok columns compare against the A/B Fourier-loop"
            " rate per lane: 0.5 GS/s at 1 GHz and 5 GS/s at 10 GHz modulation.",
            _table(
                [
                    "via",
                    "AR",
                    "R_drv",
                    "d/pitch (um)",
                    "C (fF)",
                    "L (nH)",
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


def crossing_options(n: int, f_mod: float, via: m.ViaTech) -> list[dict]:
    """Ways to carry the A/B Fourier loop (n^2 samples per shot) across the gap."""
    link, pw = m.LinkParams(), m.PowerParams()
    lm = m.link_metrics(500.0, via, link)
    f_shot = f_mod / 2
    rate = n * n * f_shot
    lane_area = lm["lane_area_um2"] * 1e-6
    tdm = int(min(lm["analog_gsps"], link.tdm_max_gsps) * 1e9 // f_shot)
    tdm = min(tdm, link.tdm_max)
    e_digital = (
        pw.adc_fom_fj * 2 ** (link.analog_bits - 1) * 1e-3
        + pw.dac_pj
        + link.analog_bits * pw.serdes_pj_per_bit
    )
    options = [
        (
            "analog 1:1",
            n * n if lm["analog_gsps"] * 1e9 >= f_shot else None,
            lm["energy_pj_per_sample"],
        ),
        (
            f"analog TDM x{tdm}" if tdm >= 2 else "analog TDM",
            math.ceil(n * n / tdm) if tdm >= 2 else None,
            lm["energy_pj_per_sample"] + link.tdm_pj_per_sample,
        ),
        (
            "digital ADC-SerDes-DAC",
            math.ceil(rate * link.analog_bits / (lm["nrz_gbps"] * 1e9)),
            e_digital,
        ),
    ]
    return [
        {
            "option": name,
            "lanes": lanes,
            "area_mm2": None if lanes is None else lanes * lane_area,
            "power_w": None if lanes is None else rate * e_pj * 1e-12,
        }
        for name, lanes, e_pj in options
    ]


def crossing_section(n: int = 100) -> str:
    rows = []
    for f_mod in F_MOD:
        for label in ("TGV AR10", "TGV AR20", "TSV AR10"):
            via = m.ViaTech(
                kind=VIA_OPTIONS[label]["via__kind"],
                aspect_ratio=VIA_OPTIONS[label]["via__aspect_ratio"],
            )
            for opt in crossing_options(n, f_mod, via):
                rows.append(
                    [
                        f_mod / 1e9,
                        label,
                        opt["option"],
                        "rate-limited" if opt["lanes"] is None else opt["lanes"],
                        "-" if opt["area_mm2"] is None else opt["area_mm2"],
                        "-" if opt["power_w"] is None else opt["power_w"],
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
    """Architectures at the stated assumptions: 500 um gap, 50/100 um vias."""
    records = []
    for f_mod, kind in itertools.product(F_MOD, ("TGV", "TSV")):
        for key in m.ARCHITECTURES:
            records.append(m.evaluate(key, _params(f_mod_hz=f_mod, via__kind=kind)))
    sections = ["## 4. Architectures at the stated assumptions"]
    sections.append(
        "N = 100, 5 um pitch, 500 um gap, AR 10 (50 um vias on 100 um pitch),"
        " 200 mm2 gap-via budget. Thermal footprint = smallest stack area that"
        " keeps photonic dies <= 85 C and the EIC <= 105 C (cap 400 mm2)."
        " Speedup = MACs/s vs the 1D F=256 JTC at the same modulation rate over"
        " CIFAR ResNet-18 stages (equal MACs per stage, so a harmonic mean);"
        " speedup_worst_layer is the 4x4-map stage alone."
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
    for via_label, budget, n, f_mod in itertools.product(
        VIA_OPTIONS, VIA_BUDGETS, N_VALUES, F_MOD
    ):
        p = _params(
            n=n, f_mod_hz=f_mod, via_area_budget_mm2=budget, **VIA_OPTIONS[via_label]
        )
        for key in m.ARCHITECTURES:
            r = m.evaluate(key, p)
            r["via_option"] = via_label
            r["via_budget_mm2"] = budget
            records.append(r)
    return records


def feasibility_section(records: list[dict]) -> str:
    """Largest feasible N per via option and budget, plus N = 100 design points."""
    header = ["via", "budget (mm2)"] + [
        f"{a} @ {f / 1e9:g} GHz" for a in m.ARCHITECTURES for f in F_MOD
    ]
    rows = []
    for via_label, budget in itertools.product(VIA_OPTIONS, VIA_BUDGETS):
        row = [via_label, budget]
        for a, f in itertools.product(m.ARCHITECTURES, F_MOD):
            ok = [
                r["n"]
                for r in records
                if r["via_option"] == via_label
                and r["via_budget_mm2"] == budget
                and r["arch"] == a
                and r["f_mod_ghz"] == f / 1e9
                and r["feasible"]
                and r["ok_rate"]
            ]
            row.append(max(ok) if ok else "-")
        rows.append(row)

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

    points = []
    for a, f, via_label in itertools.product(m.ARCHITECTURES, F_MOD, VIA_OPTIONS):
        cands = [
            r
            for r in records
            if r["n"] == 100
            and r["arch"] == a
            and r["f_mod_ghz"] == f / 1e9
            and r["via_option"] == via_label
        ]
        full = [r for r in cands if r["feasible"] and r["ok_rate"]]
        feas = full or [r for r in cands if r["feasible"]]
        r = min(feas or cands, key=lambda r: r["via_budget_mm2"])
        points.append(
            [
                a,
                f / 1e9,
                via_label,
                r["via_budget_mm2"] if feas else "-",
                "full rate" if full else ("throttled" if feas else "no"),
                r["failing"],
                r["speedup_workload"],
                r["tmacs"],
                r["via_field_mm2"],
                r["p_stack_w"],
                r["footprint_thermal_mm2"],
                r["mem_devices"],
            ]
        )
    return "\n\n".join(
        [
            "## 5. Feasible design points",
            "Feasible = thermal fits within 400 mm2 and the workload speedup over"
            " 1D is >= 10x after any gap-lane or memory rate limit. Cells give"
            " the largest N per side that is feasible at full shot rate (gap"
            f" lanes not throttling) over the swept N {list(N_VALUES)}.",
            _table(header, rows),
            "Smallest N reaching 10x at full shot rate (any via option):",
            _table(["arch", "f_mod (GHz)", "N_min"], min_n),
            "N = 100 design points (smallest gap-via budget that runs at full"
            " rate; else the smallest feasible but throttled one):",
            _table(
                [
                    "arch",
                    "f_mod",
                    "via",
                    "budget mm2",
                    "feasible",
                    "failing",
                    "speedup",
                    "TMAC/s",
                    "via field mm2",
                    "stack W",
                    "footprint mm2",
                    "memory",
                ],
                points,
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
    ("TGV AR 20 (25/50 um)", {"via__aspect_ratio": 20.0}),
)


def sensitivity_section() -> str:
    rows = []
    for label, changes in SENSITIVITY_CASES:
        row = [label]
        for f, a in itertools.product(F_MOD, m.ARCHITECTURES):
            r = m.evaluate(a, _params(f_mod_hz=f, **changes))
            cell = f"{r['p_stack_w']:.0f} W, {r['footprint_thermal_mm2']:.0f} mm2"
            if not r["ok_thermal"]:
                cell += " HOT"
            if not r["ok_target"]:
                cell += " <10x"
            elif not r["ok_rate"]:
                cell += " throttled"
            row.append(cell)
        rows.append(row)
    header = ["case"] + [
        f"{a} @ {f / 1e9:g} GHz" for f in F_MOD for a in m.ARCHITECTURES
    ]
    return "\n\n".join(
        [
            "## 6. Power and thermal sensitivity (N = 100, 500 um TGV gap)",
            "Cells: stack power (excludes the off-stack laser) and the thermal"
            " footprint. HOT = exceeds limits even at 400 mm2; throttled = gap"
            " lanes hold the shot rate below f_mod / passes; <10x = below target.",
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
        assumptions_section(),
        via_section(),
        crossing_section(),
        baseline_md,
        feasibility_section(records),
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
