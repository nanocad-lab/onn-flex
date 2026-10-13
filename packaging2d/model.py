"""First-order packaging model for a 2D pJTC 3-layer stack.

Stack (top to bottom):

* top die: GF 45SPCLO PIC, half driver/MRM, half grating-coupler (GC) emitters;
* middle layer: 12LP EIC (converters, accumulation, memory PHY) beside a glass
  substrate carrying the 2D metalens; its thickness is the optical gap;
* bottom die: GF 45SPCLO PIC, half GC receivers, half PD/TIA.

The three architectures differ in where memory, power and the analog
Fourier-plane loop (PD/TIA -> remodulating MRM) cross that gap:

* A - single-sided package + vertical interconnect (organization 1);
* B - dual-sided packaging, top die also packaged with memory (organization 2);
* C - single-sided electronics + optical loopback: two optical columns, the
  second inverted, so the Fourier loop closes laterally on the bottom die
  (organization 3).

The optical design is taken as given: 5 um pixel pitch on both optical planes
and ~500 um between the emitting and receiving grating couplers. That
separation is the middle-layer thickness, so it sets the via length.

Every number here is a placeholder for sizing studies. Component energies,
thermal resistances and via electrical models are textbook first-order forms;
replace them with PDK, package and measurement data before drawing
conclusions beyond relative trends.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass, field, replace

import onn_shotplan

EPS0 = 8.854e-12
MU0 = 4e-7 * math.pi
C0 = 2.998e8
RHO_CU = 1.72e-8
LPDDR5X_GBYTES_S = 34.1  # x32 package at 8533 MT/s
HBM3_GBYTES_S = 819.0  # 1024 bits at 6.4 Gb/s


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OpticsParams:
    pixel_pitch_um: float = 5.0  # GC pitch on both optical planes (given)
    gc_size_um: float = 5.0  # mini grating coupler


@dataclass(frozen=True)
class ViaTech:
    kind: str = "TGV"  # "TGV" (glass) or "TSV" (silicon spacer / EIC)
    aspect_ratio: float = 10.0
    pitch_to_diameter: float = 2.0
    eps_r_glass: float = 4.8
    eps_r_liner: float = 3.9
    liner_frac: float = 0.02  # liner thickness / via diameter (1 um at 50 um)
    depletion_factor: float = 0.7  # MOS depletion lowers TSV liner C
    returns_per_signal: int = 2  # adjacent return vias (G-S-G)
    grounds_per_signal: float = 1.0  # via-field area overhead
    j_max_a_cm2: float = 5e4  # Cu via current density limit for power vias


@dataclass(frozen=True)
class LinkParams:
    r_drv_ohm: float = 50.0
    c_pad_ff: float = 20.0  # per via end
    c_load_ff: float = 20.0  # far-end buffer / MRM driver input
    v_swing: float = 0.8
    activity: float = 0.5
    analog_bits: int = 6
    settle_fraction: float = 0.5  # fraction of a sample period for settling
    nrz_bits_per_hz: float = 2.0  # NRZ bit rate per Hz of -3 dB bandwidth
    e_rx_buffer_pj: float = 0.05
    tdm_pj_per_sample: float = 0.5  # analog S/H + mux/demux when TDM > 1
    tdm_max: int = 32
    tdm_max_gsps: float = 4.0  # per-lane S/H + analog mux limit when TDM > 1


@dataclass(frozen=True)
class PowerParams:
    heater_mw_per_ring: float = 1.0
    mrm_drive_pj: float = 0.25  # per analog MRM update
    mrm_hold_mw: float = 0.05  # static per held input MRM
    tia_pj: float = 0.4  # per detected sample
    tia_static_mw: float = 0.1
    dac_pj: float = 0.2
    adc_fom_fj: float = 20.0  # Walden FoM per conversion step
    adc_enob: float = 7.0
    accum_pj: float = 1.0  # digital add + SRAM per output sample
    eic_static_w: float = 1.0
    mem_pj_per_bit: float = 5.0  # DRAM + PHY
    serdes_pj_per_bit: float = 1.0
    laser_pd_uw: float = 10.0  # average optical power per detector
    laser_loss_db: float = 10.0  # MRM, GCs, lens, splitter tree
    laser_wpe: float = 0.15
    vdd: float = 0.9


@dataclass(frozen=True)
class AreaParams:
    mrm_cell_um2: float = 900.0  # ring + heater + driver + bond pad
    pd_cell_um2: float = 625.0  # PD + TIA + bond pad
    adc_um2_per_gsps: float = 1500.0
    acc_um2: float = 200.0
    dac_um2: float = 50.0
    lane_driver_um2: float = 100.0
    sram_mb: float = 8.0
    sram_mb_per_mm2: float = 0.5
    eic_fixed_mm2: float = 5.0  # PHY, control, clocking
    aperture_margin: float = 0.2


@dataclass(frozen=True)
class ThermalParams:
    t_ambient_c: float = 40.0
    t_max_photonic_c: float = 85.0
    t_max_eic_c: float = 105.0
    r_sink_kcm2_w: float = 0.3  # TIM + cold plate on an exposed backside
    r_top_pkg_kcm2_w: float = 3.0  # extra: top package substrate (B only)
    r_pkg_board_kcm2_w: float = 10.0  # die -> organic package -> board
    r_bond_kcm2_w: float = 0.1  # micro-bump + underfill layer
    k_si_w_mk: float = 120.0
    k_glass_w_mk: float = 1.1


@dataclass(frozen=True)
class WorkloadParams:
    c_in: int = 128
    c_out: int = 128
    branches: int = 4  # signed input/weight branches
    analog_accum_depth: int = 1  # shots integrated before each ADC sample
    bits_weight: int = 4
    bits_input: int = 4
    bits_output: int = 8
    # padded CIFAR ResNet feature-map widths (32, 16, 8, 4 + 2)
    widths: tuple[int, ...] = (34, 18, 10, 6)
    max_lpddr5x: int = 4
    max_hbm3: int = 2


@dataclass(frozen=True)
class StackParams:
    n: int = 100  # 2D field per dimension: n x n channels
    kernel: int = 3
    f_mod_hz: float = 1e9
    gap_um: float = 500.0  # GC-to-GC separation = middle-layer thickness (given)
    via_area_budget_mm2: float = 200.0
    tdm_policy: str = "min_area"  # or "min_power" (1:1 lanes when they fit)
    footprint_max_mm2: float = 400.0  # thermal spreading limit
    baseline_1d_field: int = 256
    target_speedup: float = 10.0
    optics: OpticsParams = field(default_factory=OpticsParams)
    via: ViaTech = field(default_factory=ViaTech)
    link: LinkParams = field(default_factory=LinkParams)
    power: PowerParams = field(default_factory=PowerParams)
    area: AreaParams = field(default_factory=AreaParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    workload: WorkloadParams = field(default_factory=WorkloadParams)


@dataclass(frozen=True)
class Architecture:
    key: str
    name: str
    organization: str
    serial_passes: int  # lens passes that share one GC/MRM/PD array
    optical_columns: int
    fourier_loop_crosses_gap: bool
    input_crosses_gap: bool
    top_power_crosses_gap: bool
    eic_power_crosses_gap: bool
    memory_crosses_gap: bool
    top_package: bool
    eic_faces: str  # die the EIC is face-to-face bonded with


ARCHITECTURES = {
    "A": Architecture(
        "A",
        "Single-sided package + vertical interconnect",
        "Org 1: LPDDR5/HBM beside the stack on the bottom package",
        serial_passes=2,
        optical_columns=1,
        fourier_loop_crosses_gap=True,
        input_crosses_gap=True,
        top_power_crosses_gap=True,
        eic_power_crosses_gap=False,
        memory_crosses_gap=False,
        top_package=False,
        eic_faces="bottom",
    ),
    "B": Architecture(
        "B",
        "Dual-sided packaging",
        "Org 2: top die also packaged with its own LPDDR5/HBM and power",
        serial_passes=2,
        optical_columns=1,
        fourier_loop_crosses_gap=True,
        input_crosses_gap=False,
        top_power_crosses_gap=False,
        eic_power_crosses_gap=False,
        memory_crosses_gap=False,
        top_package=True,
        eic_faces="bottom",
    ),
    "C": Architecture(
        "C",
        "Single-sided electronics + optical loopback",
        "Org 3: two optical columns, second inverted; Fourier loop on bottom die",
        serial_passes=1,
        optical_columns=2,
        fourier_loop_crosses_gap=False,
        input_crosses_gap=False,
        top_power_crosses_gap=True,
        eic_power_crosses_gap=True,
        memory_crosses_gap=True,
        top_package=False,
        eic_faces="top",
    ),
}


# --------------------------------------------------------------------------
# Optics (given)
# --------------------------------------------------------------------------


def fourier_fill_factor(optics: OpticsParams) -> float:
    """Fraction of each pixel a GC-sized receiver covers (1 at 5 um / 5 um)."""
    return min(1.0, (optics.gc_size_um / optics.pixel_pitch_um) ** 2)


# --------------------------------------------------------------------------
# Vertical interconnect through the optical gap
# --------------------------------------------------------------------------


def via_geometry_um(gap_um: float, via: ViaTech) -> tuple[float, float]:
    """(diameter, pitch) of an aspect-ratio-limited via spanning the gap."""
    diameter = gap_um / via.aspect_ratio
    return diameter, diameter * via.pitch_to_diameter


def via_parasitics(gap_um: float, via: ViaTech, freq_hz: float = 1e10) -> dict:
    """Lumped R (ohm), loop L (H) and C (F) for one signal via with returns."""
    length = gap_um * 1e-6
    d_um, p_um = via_geometry_um(gap_um, via)
    radius = 0.5 * d_um * 1e-6
    spacing = math.acosh(p_um / d_um)
    l_pair = MU0 * length / math.pi * spacing
    inductance = l_pair / via.returns_per_signal
    if via.kind == "TGV":
        c_pair = math.pi * EPS0 * via.eps_r_glass * length / spacing
        capacitance = c_pair * via.returns_per_signal
    elif via.kind == "TSV":
        r_out = radius * (1 + 2 * via.liner_frac)
        c_ox = 2 * math.pi * EPS0 * via.eps_r_liner * length / math.log(r_out / radius)
        capacitance = c_ox * via.depletion_factor
    else:
        raise ValueError(f"unknown via kind {via.kind!r}")
    skin = math.sqrt(RHO_CU / (math.pi * freq_hz * MU0))
    r_dc = RHO_CU * length / (math.pi * radius**2)
    r_ac = RHO_CU * length / (2 * math.pi * radius * min(skin, radius))
    return {"r_ohm": max(r_dc, r_ac), "l_h": inductance, "c_f": capacitance}


def _second_order_f3db(r: float, l_h: float, c: float) -> float:
    """-3 dB frequency of a series R-L driving shunt C (highest crossing)."""

    def gain2(f):
        w = 2 * math.pi * f
        return 1.0 / ((1 - w * w * l_h * c) ** 2 + (w * r * c) ** 2)

    lo, hi = 1e6, 1e6
    while gain2(hi) >= 0.5 or hi < 1e8:
        lo, hi = hi, hi * 1.25
        if hi > 1e14:
            return hi
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if gain2(mid) >= 0.5:
            lo = mid
        else:
            hi = mid
    return lo


def link_metrics(gap_um: float, via: ViaTech, link: LinkParams) -> dict:
    """Bandwidth, rates and energy of one signal lane across the gap."""
    par = via_parasitics(gap_um, via)
    c_total = par["c_f"] + (2 * link.c_pad_ff + link.c_load_ff) * 1e-15
    f3db = _second_order_f3db(link.r_drv_ohm + par["r_ohm"], par["l_h"], c_total)
    analog_rate = (
        link.settle_fraction
        * 2
        * math.pi
        * f3db
        / ((link.analog_bits + 1) * math.log(2))
    )
    eps_eff = via.eps_r_glass if via.kind == "TGV" else 11.9
    d_um, p_um = via_geometry_um(gap_um, via)
    lane_area_um2 = p_um**2 * (1 + via.grounds_per_signal)
    energy_pj = link.activity * c_total * link.v_swing**2 * 1e12 + link.e_rx_buffer_pj
    return {
        "gap_um": gap_um,
        "via_kind": via.kind,
        "diameter_um": d_um,
        "pitch_um": p_um,
        "r_ohm": par["r_ohm"],
        "l_nh": par["l_h"] * 1e9,
        "c_via_ff": par["c_f"] * 1e15,
        "f3db_ghz": f3db / 1e9,
        "analog_gsps": analog_rate / 1e9,
        "nrz_gbps": link.nrz_bits_per_hz * f3db / 1e9,
        "energy_pj_per_sample": energy_pj,
        "lane_area_um2": lane_area_um2,
        "lanes_per_mm2": 1e6 / lane_area_um2,
        "quarter_wave_10ghz_um": C0 / (1e10 * math.sqrt(eps_eff)) / 4 * 1e6,
    }


def power_via_count(current_a: float, gap_um: float, via: ViaTech) -> int:
    """Power + ground vias for a current, limited by Cu current density."""
    if current_a <= 0:
        return 0
    d_um, _ = via_geometry_um(gap_um, via)
    i_max = via.j_max_a_cm2 * math.pi * (0.5 * d_um * 1e-4) ** 2
    return 2 * math.ceil(current_a / i_max)


# --------------------------------------------------------------------------
# JTC geometry and throughput (reuses the repo's contamination rule)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def clean_input_limit(field_len: int, kernel: int) -> int:
    """Largest clean signal length with S = M - N (jtc_geometry_constraints.md)."""
    best = 0
    for m in range(kernel, field_len + 1):
        sep = m - kernel
        if m + kernel + sep > field_len:
            break
        _, clean, stride = onn_shotplan.compute_contamination_profile(
            m, kernel, field_len, sep
        )
        valid = m - kernel + 1
        if clean == valid and stride == valid:
            best = m
    return best


def geometry_2d(field_len: int, kernel: int) -> tuple[int, int]:
    """Clean 2D signal box with kernel/signal separated along x.

    Box overlap needs overlap on both axes, so a 1D-clean x layout is clean
    in 2D; y only needs the cross term not to wrap onto itself.
    """
    return clean_input_limit(field_len, kernel), field_len - kernel + 1


def outputs_along(width: int, capacity: int, kernel: int) -> float:
    """Average valid outputs per shot along one axis (packing or stitching)."""
    valid = width - kernel + 1
    per_tile = capacity - kernel + 1
    if valid <= 0 or per_tile <= 0:
        return 0.0
    if width <= capacity:
        guard = kernel - 1
        return (capacity + guard) // (width + guard) * valid
    return valid / math.ceil(valid / per_tile)


def macs_per_shot_1d(field_len: int, kernel: int, width: int | None = None) -> float:
    m = clean_input_limit(field_len, kernel)
    return outputs_along(m if width is None else width, m, kernel) * kernel


def macs_per_shot_2d(field_len: int, kernel: int, width: int | None = None) -> float:
    mx, my = geometry_2d(field_len, kernel)
    out_x = outputs_along(mx if width is None else width, mx, kernel)
    out_y = outputs_along(my if width is None else width, my, kernel)
    return out_x * out_y * kernel * kernel


# --------------------------------------------------------------------------
# Thermal: three-node network (top die, EIC, bottom die)
# --------------------------------------------------------------------------


def _solve3(g: list[list[float]], q: list[float]) -> list[float]:
    a = [row[:] + [rhs] for row, rhs in zip(g, q)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(3):
            if r != col:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [a[i][3] / a[i][i] for i in range(3)]


def stack_temperatures(
    arch: Architecture,
    powers_w: dict,
    footprint_cm2: float,
    si_area_cm2: float,
    gap_um: float,
    th: ThermalParams,
) -> dict:
    """Steady-state die temperatures from per-area thermal resistances."""
    gap_si = gap_um * 1e-6 / th.k_si_w_mk * 1e4  # K cm^2 / W
    gap_glass = gap_um * 1e-6 / th.k_glass_w_mk * 1e4
    glass_cm2 = max(footprint_cm2 - si_area_cm2, 0.0)
    r_top = th.r_sink_kcm2_w + (th.r_top_pkg_kcm2_w if arch.top_package else 0.0)
    g_t_amb = footprint_cm2 / r_top
    g_b_amb = footprint_cm2 / th.r_pkg_board_kcm2_w
    g_bond = si_area_cm2 / th.r_bond_kcm2_w
    g_spacer = si_area_cm2 / (th.r_bond_kcm2_w + gap_si)
    g_tm, g_mb = (g_bond, g_spacer) if arch.eic_faces == "top" else (g_spacer, g_bond)
    g_tb = glass_cm2 / (2 * th.r_bond_kcm2_w + gap_glass)
    amb = th.t_ambient_c
    g = [
        [g_t_amb + g_tm + g_tb, -g_tm, -g_tb],
        [-g_tm, g_tm + g_mb, -g_mb],
        [-g_tb, -g_mb, g_b_amb + g_mb + g_tb],
    ]
    q = [
        powers_w["top"] + g_t_amb * amb,
        powers_w["eic"],
        powers_w["bottom"] + g_b_amb * amb,
    ]
    t_top, t_eic, t_bot = _solve3(g, q)
    return {
        "t_top_c": t_top,
        "t_eic_c": t_eic,
        "t_bottom_c": t_bot,
        "heat_to_sink_w": g_t_amb * (t_top - amb),
        "heat_to_board_w": g_b_amb * (t_bot - amb),
    }


# --------------------------------------------------------------------------
# Architecture evaluation
# --------------------------------------------------------------------------


def _die_powers(
    arch: Architecture,
    p: StackParams,
    f_shot: float,
    n_out: float,
    tile_updates: float,
    mem_bits: dict,
    lane_pj: float,
    samples_per_lane: float,
) -> dict:
    """Per-die power (W) at shot rate f_shot; lane_pj covers gap-lane I/O."""
    pw, wl = p.power, p.workload
    f2 = p.n * p.n
    pj = 1e-12
    heaters = f2 * pw.heater_mw_per_ring * 1e-3
    adc = n_out * f_shot / wl.analog_accum_depth * pw.adc_fom_fj * 2**pw.adc_enob
    adc *= 1e-15
    accum = n_out * f_shot * pw.accum_pj * pj
    dac = tile_updates * f_shot * pw.dac_pj * pj
    tia_static = pw.tia_static_mw * 1e-3
    mem_total = (mem_bits["in"] + mem_bits["out"]) * f_shot * pw.mem_pj_per_bit * pj
    gap_lanes = 0.0
    if arch.fourier_loop_crosses_gap:
        gap_lanes = f2 * samples_per_lane * f_shot * lane_pj * pj
    if arch.key == "C":
        top = (
            heaters
            + f2 * pw.mrm_hold_mw * 1e-3
            + tile_updates * f_shot * pw.mrm_drive_pj * pj
            + n_out * (f_shot * pw.tia_pj * pj + tia_static)
        )
        serdes = (mem_bits["in"] + mem_bits["out"]) * f_shot * pw.serdes_pj_per_bit
        eic = pw.eic_static_w + dac + adc + accum + mem_total + serdes * pj
        bottom = heaters + f2 * (f_shot * (pw.tia_pj + pw.mrm_drive_pj) * pj)
        bottom += f2 * tia_static
    else:
        # one MRM/GC/PD array time-shared by the input and remodulation passes;
        # the gap-lane drivers sit in the EIC, receivers on the top die
        top = heaters + f2 * 2 * f_shot * pw.mrm_drive_pj * pj + 0.5 * gap_lanes
        bottom = f2 * (2 * f_shot * pw.tia_pj * pj + tia_static)
        eic = pw.eic_static_w + adc + accum + 0.5 * gap_lanes
        if arch.top_package:
            mem_top = mem_bits["in"] * f_shot * pw.mem_pj_per_bit * pj
            handoff = mem_bits["out"] * f_shot * pw.serdes_pj_per_bit * pj
            top += dac + mem_top
            eic += mem_total - mem_top + handoff
        else:
            eic += dac + mem_total
    return {"top": top, "eic": eic, "bottom": bottom, "gap_lanes": gap_lanes}


def _thermal_spread(arch, powers, footprint_mm2, si_mm2, gap, p) -> tuple:
    """Smallest footprint scale (>= 1) that meets die temperature limits."""
    th = p.thermal

    def temps(scale):
        return stack_temperatures(
            arch, powers, footprint_mm2 * scale / 100, si_mm2 * scale / 100, gap, th
        )

    def ok(t):
        return (
            max(t["t_top_c"], t["t_bottom_c"]) <= th.t_max_photonic_c
            and t["t_eic_c"] <= th.t_max_eic_c
        )

    lo, hi = 1.0, max(p.footprint_max_mm2 / footprint_mm2, 1.0)
    if ok(temps(lo)):
        return lo, temps(lo), True
    if not ok(temps(hi)):
        return hi, temps(hi), False
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(temps(mid)) else (mid, hi)
    return hi, temps(hi), True


def evaluate(arch_key: str, p: StackParams) -> dict:
    """Evaluate one architecture at one design point; returns a flat record."""
    arch = ARCHITECTURES[arch_key]
    wl, ar, op, ln = p.workload, p.area, p.optics, p.link
    f2 = p.n * p.n
    k = p.kernel

    # JTC geometry and the 1D baseline
    mx, my = geometry_2d(p.n, k)
    n_out = (mx - k + 1) * (my - k + 1) if mx >= k else 0
    macs_2d = macs_per_shot_2d(p.n, k)
    macs_1d = macs_per_shot_1d(p.baseline_1d_field, k)
    ratios = [
        macs_per_shot_2d(p.n, k, w) / macs_per_shot_1d(p.baseline_1d_field, k, w)
        for w in wl.widths
    ]
    # ResNet-18 CIFAR stages carry equal MACs, so time adds harmonically
    workload_ratio = len(ratios) / sum(1.0 / r for r in ratios)

    # optics are given: the GC separation is the gap every via must span
    gap = p.gap_um
    aperture_mm = p.n * op.pixel_pitch_um * 1e-3
    fill = fourier_fill_factor(op)

    # traffic per shot
    lm = link_metrics(gap, p.via, ln)
    f_shot_opt = p.f_mod_hz / arch.serial_passes
    tile_updates = mx * my / wl.c_out + k * k
    mem_bits = {
        "in": k * k * wl.bits_weight + mx * my * wl.bits_input / wl.c_out,
        "out": n_out * wl.bits_output / (wl.c_in * wl.branches),
    }
    bits_per_shot = mem_bits["in"] + mem_bits["out"]
    spl = 0.0  # analog samples per Fourier-loop lane per shot
    if arch.fourier_loop_crosses_gap:
        spl = 1.0 + (tile_updates / f2 if arch.input_crosses_gap else 0.0)
    mem_cap_gbs = max(wl.max_lpddr5x * LPDDR5X_GBYTES_S, wl.max_hbm3 * HBM3_GBYTES_S)
    mem_cap_gbs *= 2 if arch.top_package else 1
    f_shot_mem = mem_cap_gbs * 8e9 / bits_per_shot

    def cross_power(powers):
        return (powers["top"] if arch.top_power_crosses_gap else 0.0) + (
            powers["eic"] if arch.eic_power_crosses_gap else 0.0
        )

    # reserve power vias from the upper-bound power at f_shot_opt
    lane_pj_max = lm["energy_pj_per_sample"] + ln.tdm_pj_per_sample
    p_opt = _die_powers(
        arch, p, f_shot_opt, n_out, tile_updates, mem_bits, lane_pj_max, spl
    )
    n_power_vias = power_via_count(cross_power(p_opt) / p.power.vdd, gap, p.via)
    power_via_area = n_power_vias * lm["pitch_um"] ** 2 * 1e-6
    lanes_avail = int(
        max(p.via_area_budget_mm2 - power_via_area, 0.0) * 1e6 // lm["lane_area_um2"]
    )

    # Fourier-loop lanes: analog, optionally time-multiplexed (TDM) per via
    lane_rate = lm["analog_gsps"] * 1e9
    mux, analog_lanes, f_shot_io = 1, 0, math.inf
    if arch.fourier_loop_crosses_gap:
        tdm_rate = min(lane_rate, ln.tdm_max_gsps * 1e9)
        mux_rate = max(1, min(ln.tdm_max, int(tdm_rate // (f_shot_opt * spl))))
        mux = mux_rate if p.tdm_policy == "min_area" else 1
        if lanes_avail <= 0:
            mux, f_shot_io = math.inf, 0.0
        else:
            mux = max(mux, math.ceil(f2 / lanes_avail))
            analog_lanes = math.ceil(f2 / mux)
            f_shot_io = (lane_rate if mux == 1 else tdm_rate) / (mux * spl)
    digital_lanes = 0
    if arch.memory_crosses_gap:
        digital_lanes = math.ceil(bits_per_shot * f_shot_opt / (lm["nrz_gbps"] * 1e9))
    f_shot = min(f_shot_opt, f_shot_io, f_shot_mem)
    lane_pj = lm["energy_pj_per_sample"]
    if mux != 1:
        lane_pj += ln.tdm_pj_per_sample

    # power, area, thermal at the achieved shot rate
    pw = _die_powers(arch, p, f_shot, n_out, tile_updates, mem_bits, lane_pj, spl)
    stack_w = pw["top"] + pw["eic"] + pw["bottom"]
    laser_w = (
        f2
        * arch.optical_columns
        * p.power.laser_pd_uw
        * 1e-6
        * 10 ** (p.power.laser_loss_db / 10)
        / fill
        / p.power.laser_wpe
    )
    f_adc = f_shot / wl.analog_accum_depth
    eic_mm2 = (
        (
            n_out * (ar.adc_um2_per_gsps * max(1.0, f_adc / 1e9) + ar.acc_um2)
            + f2 * ar.dac_um2
            + analog_lanes * ar.lane_driver_um2
        )
        * 1e-6
        + ar.sram_mb / ar.sram_mb_per_mm2
        + ar.eic_fixed_mm2
    )
    if arch.key == "C":
        top_cells = f2 * (ar.mrm_cell_um2 + ar.pd_cell_um2) * 1e-6
        bottom_cells = top_cells
    else:
        top_cells = f2 * ar.mrm_cell_um2 * 1e-6
        bottom_cells = f2 * ar.pd_cell_um2 * 1e-6
    electronics_mm2 = max(top_cells, bottom_cells, eic_mm2)
    optics_mm2 = arch.optical_columns * aperture_mm**2 * (1 + ar.aperture_margin)
    lane_area = lm["lane_area_um2"] * 1e-6
    via_field_mm2 = (analog_lanes + digital_lanes) * lane_area + power_via_area
    footprint_mm2 = electronics_mm2 + optics_mm2 + via_field_mm2
    si_mm2 = eic_mm2 + (via_field_mm2 if p.via.kind == "TSV" else 0.0)
    scale, temps, thermal_ok = _thermal_spread(arch, pw, footprint_mm2, si_mm2, gap, p)

    # throughput relative to the 1D baseline at the same modulation rate
    tmacs = macs_2d * f_shot / 1e12
    speedup_workload = workload_ratio * f_shot / p.f_mod_hz
    mem_gbs = bits_per_shot * f_shot / 8e9
    checks = {
        "rate": f_shot >= f_shot_opt * (1 - 1e-9),
        "thermal": thermal_ok,
        "target": speedup_workload >= p.target_speedup,
    }
    if not checks["rate"]:
        limiter = "gap lanes" if f_shot_io <= f_shot_mem else "memory"
    else:
        limiter = "modulation rate"
    return {
        "arch": arch.key,
        "n": p.n,
        "channels": f2,
        "f_mod_ghz": p.f_mod_hz / 1e9,
        "pixel_pitch_um": op.pixel_pitch_um,
        "gap_um": gap,
        "fourier_fill": fill,
        "aperture_mm": aperture_mm,
        "via_kind": p.via.kind,
        "aspect_ratio": p.via.aspect_ratio,
        "via_d_um": lm["diameter_um"],
        "via_pitch_um": lm["pitch_um"],
        "via_c_ff": lm["c_via_ff"],
        "lane_f3db_ghz": lm["f3db_ghz"],
        "lane_analog_gsps": lm["analog_gsps"],
        "lane_nrz_gbps": lm["nrz_gbps"],
        "lumped_ok_10ghz": gap <= 0.4 * lm["quarter_wave_10ghz_um"],
        "analog_lanes": analog_lanes,
        "tdm_ratio": mux,
        "digital_lanes": digital_lanes,
        "power_vias": n_power_vias,
        "via_field_mm2": via_field_mm2,
        "f_shot_ghz": f_shot / 1e9,
        "limiter": limiter,
        "mx": mx,
        "my": my,
        "outputs_per_shot": n_out,
        "macs_per_shot": macs_2d,
        "tmacs": tmacs,
        "speedup_peak": macs_2d * f_shot / (macs_1d * p.f_mod_hz),
        "speedup_workload": speedup_workload,
        "speedup_worst_layer": min(ratios) * f_shot / p.f_mod_hz,
        "p_top_w": pw["top"],
        "p_eic_w": pw["eic"],
        "p_bottom_w": pw["bottom"],
        "p_gap_lanes_w": pw["gap_lanes"],
        "p_stack_w": stack_w,
        "p_laser_w": laser_w,
        "pj_per_mac": (stack_w + laser_w) / (tmacs * 1e12) * 1e12 if tmacs else None,
        "p_cross_gap_w": cross_power(pw),
        "eic_mm2": eic_mm2,
        "footprint_mm2": footprint_mm2,
        "footprint_thermal_mm2": footprint_mm2 * scale,
        "power_density_w_cm2": stack_w / (footprint_mm2 * scale / 100),
        "t_top_c": temps["t_top_c"],
        "t_eic_c": temps["t_eic_c"],
        "t_bottom_c": temps["t_bottom_c"],
        "heat_to_board_frac": temps["heat_to_board_w"] / stack_w if stack_w else 0.0,
        "mem_gbytes_s": mem_gbs,
        "mem_devices": memory_devices(mem_gbs, wl),
        **{f"ok_{name}": ok for name, ok in checks.items()},
        "feasible": checks["thermal"] and checks["target"],
        "failing": ",".join(name for name, ok in checks.items() if not ok) or "-",
    }


def memory_devices(gbytes_s: float, wl: WorkloadParams) -> str:
    """Smallest LPDDR5X or HBM3 configuration that meets a bandwidth."""
    lp = math.ceil(gbytes_s / LPDDR5X_GBYTES_S)
    if lp <= wl.max_lpddr5x:
        return f"{max(lp, 1)}x LPDDR5X"
    hbm = math.ceil(gbytes_s / HBM3_GBYTES_S)
    return f"{hbm}x HBM3" + ("" if hbm <= wl.max_hbm3 else " (over budget)")


def with_overrides(p: StackParams, **changes) -> StackParams:
    """Replace top-level or dotted nested fields, e.g. ``via__kind="TSV"``."""
    nested: dict[str, dict] = {}
    top: dict = {}
    for key, value in changes.items():
        if "__" in key:
            group, name = key.split("__", 1)
            nested.setdefault(group, {})[name] = value
        else:
            top[key] = value
    for group, values in nested.items():
        top[group] = replace(getattr(p, group), **values)
    return replace(p, **top)


def params_dict(p: StackParams) -> dict:
    return asdict(p)
