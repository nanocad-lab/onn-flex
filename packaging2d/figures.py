"""Render the slide figures for the 2D pJTC packaging study.

Usage (repository root):

    python -m packaging2d.figures --output-dir packaging2d/figures

Every number is computed from `packaging2d.model` and `packaging2d.study` at
run time, except the feasibility-matrix verdicts in FEASIBILITY.
"""

from __future__ import annotations

import argparse
import functools
import math
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Rectangle,
)
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullLocator

from packaging2d import model as m
from packaging2d import study

FIGSIZE = (10.4, 6.6)
DPI = 150
PX = 72 / DPI  # points per output pixel
FS = 17
INK = "#1a1a1a"
INK2 = "#4a5563"
GRID = "#e3e6e9"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
NEUTRAL = "#6b747d"
FILL = {
    "pic": ("#c8d6e5", "#5b7086"),
    "eic": ("#d8d0ec", "#6a5a9e"),
    "glass": ("#dcefea", "#5f9c8c"),
    "si": ("#d5d9dd", "#7d858c"),
    "pkg": ("#e3d3b4", "#9a8159"),
    "mem": ("#f2c894", "#a86b1f"),
    "sink": ("#9aa4ae", "#6b747d"),
    "bump": ("#b9bfc5", "#7d858c"),
}
PATHS = {  # colour, legend label, line style
    "light": ("#6250d6", "Light", "-"),
    "loop": ("#eb6834", "Fourier loop", "-"),
    "power": ("#1a1a1a", "Power", "-"),
    "memory": ("#1c5cab", "Memory", (0, (4, 2))),
}
OPAQUE = {"frameon": True, "facecolor": "white", "edgecolor": "none", "framealpha": 1}
STATUS = {
    "Pass": ("#dcefe3", "#1e6b3a"),
    "Marginal": ("#fbefd5", "#7a4f00"),
    "Fail": ("#f8deda", "#a12c2c"),
}
RC = {
    "font.size": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK2,
    "axes.linewidth": 2 * PX,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.width": 2 * PX,
    "ytick.major.width": 2 * PX,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 2 * PX,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "lines.linewidth": 3 * PX,
    "lines.markersize": 9 * PX,
    "hatch.linewidth": 2 * PX,
}

# Verdicts per row for A 1/10 GHz, B 1/10 GHz, C 1/10 GHz ("status: reason; ...").
FEASIBILITY = {
    "Fourier loop across 500 µm": (
        "Pass: TDM x7, 29 mm²; Marginal: 1:1, 200 mm²; Pass: TDM x8, 25 mm²; "
        "Marginal: 1:1, 200 mm²; Pass: on bottom die; Pass: on bottom die"
    ),
    "TSV instead of TGV": (
        "Marginal: 100 mm², TDM x2; Fail: 1 GS/s per via; Marginal: 100 mm², TDM x2; "
        "Fail: 1 GS/s per via; Pass: power, memory; Pass: power, memory"
    ),
    "Thermal footprint": (
        "Pass: 55 mm²; Marginal: 241 mm²; Marginal: 144 mm²; Fail: > 400 mm²; "
        "Pass: 36 mm²; Marginal: 177 mm²"
    ),
    "≥ 10x over 1D, N = 100": (
        "Pass: 37x; Pass: 37x; Pass: 37x; Pass: 37x; Pass: 73x; Pass: 73x"
    ),
    "Memory": (
        "Pass: 1 LPDDR5X; Pass: 3 LPDDR5X; Pass: split top/bottom; "
        "Marginal: two systems; Pass: 1 LPDDR5X; Marginal: HBM3 across gap"
    ),
}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _styled(func):
    """Build the figure under the deck's rcParams."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(RC):
            return func(*args, **kwargs)

    return wrapper


def _figure(**kwargs) -> Figure:
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor="white", **kwargs)
    FigureCanvasAgg(fig)
    return fig


def _chart(ncols: int = 1):
    fig = _figure(layout="constrained")
    return fig, fig.subplots(1, ncols)


def _log_y(ax, lo: float, hi: float) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _note(ax, text, xy, dx=0.0, dy=4.0, ha="center", va="bottom", **kwargs) -> None:
    """Label offset from xy in points, on a white pad that hides gridlines."""
    pad = {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"}
    ax.annotate(
        text,
        xy,
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        bbox=pad,
        **kwargs,
    )


def _ref_line(ax, y: float, text: str, x_text: float) -> None:
    """Dashed horizontal reference labelled above its right end."""
    ax.axhline(y, color=INK2, ls=(0, (5, 4)), lw=2 * PX, zorder=1.5)
    _note(ax, text, (x_text, y), dx=-2, dy=3, ha="right", color=INK2)


def _evaluate_rates(**changes) -> dict:
    """evaluate() for every architecture at 1 and 10 GHz, keyed (arch, f_mod)."""
    return {
        (key, f): m.evaluate(
            key, m.with_overrides(m.StackParams(), f_mod_hz=f, **changes)
        )
        for key in m.ARCHITECTURES
        for f in study.F_MOD
    }


# --------------------------------------------------------------------------
# Cross-section primitives (axes units of 0.1 inch)
# --------------------------------------------------------------------------


def _canvas():
    fig = _figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIGSIZE[0] * 10)
    ax.set_ylim(0, FIGSIZE[1] * 10)
    ax.axis("off")
    return fig, ax


def _label(ax, x: float, y: float, text: str, **kwargs) -> None:
    style = {"ha": "center", "va": "center", "color": INK, "linespacing": 1.15}
    ax.text(x, y, text, zorder=6, **{**style, **kwargs})


def _block(ax, box, kind: str, label=None, at=None) -> None:
    """Filled block over box = (x0, y0, x1, y1), labelled at its centre or `at`."""
    x0, y0, x1, y1 = box
    face, edge = FILL[kind]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=face, ec=edge, lw=2 * PX))
    if label:
        _label(ax, *(at or ((x0 + x1) / 2, (y0 + y1) / 2)), label)


def _arrow(ax, p0, p1, color, lw, scale=14, style="-|>", zorder=4) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=scale,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _die(ax, xs, y0, y1, labels, label_y=None, split_from=None, at=None) -> None:
    """One PIC die over xs[0]..xs[-1], dashed region splits, one label per region."""
    _block(ax, (xs[0], y0, xs[-1], y1), "pic")
    for x in xs[1:-1]:
        y_split = (y0 if split_from is None else split_from, y1)
        ax.plot([x, x], y_split, color=FILL["pic"][1], lw=2 * PX, ls=(0, (4, 3)))
    centres = at or [(xa + xb) / 2 for xa, xb in zip(xs, xs[1:])]
    for x, text in zip(centres, labels):
        _label(ax, x, (y0 + y1) / 2 if label_y is None else label_y, text)


def _bumps(ax, x0: float, x1: float, y: float, pitch: float = 2.2) -> None:
    n = int((x1 - x0) // pitch)
    start = x0 + (x1 - x0 - (n - 1) * pitch) / 2
    face, edge = FILL["bump"]
    for i in range(n):
        ax.add_patch(Circle((start + i * pitch, y), 0.55, fc=face, ec=edge, lw=PX))


def _memory(ax, x0: float, x1: float, y: float, dies: int = 3) -> None:
    """Stacked DRAM dies with their label above."""
    pitch = 2.7
    for i in range(dies):
        _block(ax, (x0, y + i * pitch, x1, y + i * pitch + 2.2), "mem")
    _label(ax, (x0 + x1) / 2, y + dies * pitch + 0.3, "LPDDR5/HBM", va="bottom")


def _heat_sink(ax, x0: float, x1: float, y: float, h: float, fins=2.5) -> None:
    _block(ax, (x0, y, x1, y + h), "sink", "Heat sink")
    face, edge = FILL["sink"]
    for fx in np.arange(x0 + 1.0, x1 - 1.0, 3.2):
        ax.add_patch(Rectangle((fx, y + h), 1.4, fins, fc=face, ec=edge, lw=PX))


def _metasurface(ax, x0: float, x1: float, y: float) -> None:
    for xb in np.arange(x0, x1 - 0.5, 1.25):
        ax.add_patch(
            Rectangle((xb, y), 0.55, 1.3, fc=PATHS["light"][0], lw=0, zorder=3)
        )


def _light(ax, x0: float, x1: float, y_wide: float, y_focus: float) -> None:
    """Converging cone edges from the lens (x0..x1 at y_wide) to a focus, with arrow."""
    color = PATHS["light"][0]
    xc = (x0 + x1) / 2
    for xa, xb in ((x0, xc - 0.5), (x1, xc + 0.5)):
        ax.plot([xa, xb], [y_wide, y_focus], color=color, lw=2 * PX)
    step = math.copysign(1.5, y_focus - y_wide)
    _arrow(ax, (xc, y_wide + step), (xc, y_focus - step), color, 5 * PX, scale=24)


def _vertical(ax, x: float, y0: float, y1: float, kind: str) -> None:
    """Fourier-loop lane (arrow toward y1) or power via between y0 and y1."""
    color = PATHS[kind][0]
    if kind == "loop":
        _arrow(ax, (x, y0), (x, y1), color, 4 * PX)
    else:
        ax.plot([x, x], [y0, y1], color=color, lw=6 * PX, solid_capstyle="butt")


def _route(ax, xs: list[float], ys: list[float], zorder: float = 4) -> None:
    color, _, style = PATHS["memory"]
    ax.plot(xs, ys, color=color, ls=style, lw=5 * PX, zorder=zorder)


def _dimension(ax, x: float, y0: float, y1: float, x_edge: float) -> None:
    """Double arrow across the middle layer labelled with the gap."""
    _arrow(ax, (x, y0), (x, y1), INK2, 3 * PX, scale=16, style="<|-|>")
    for y in (y0, y1):
        ax.plot([x - 0.8, x_edge], [y, y], color=INK2, lw=2 * PX)
    _label(ax, x - 1.4, (y0 + y1) / 2, f"{m.StackParams().gap_um:g} µm", ha="right")


def _path_legend(ax) -> None:
    handles = [
        Line2D([], [], color=c, ls=s, lw=6 * PX, label=t) for c, t, s in PATHS.values()
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.008, 0.995),
        ncol=len(handles),
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.4,
        borderaxespad=0.0,
        borderpad=0.2,
    )


# --------------------------------------------------------------------------
# 1-3. Cross-sections
# --------------------------------------------------------------------------


def _xsec_ab(dual: bool) -> Figure:
    """A (single-sided) or B (dual-sided): one optical column, loop crosses."""
    fig, ax = _canvas()
    xl, xm, xr = 13.0, 42.0, 77.0  # stack left, electronics | optics, stack right
    xm0, xm1 = 82.0, 99.0  # memory beside the stack
    x_mem = (xm0 + xm1) / 2  # memory path drops here, rises at x_via
    # power vias on bump centres, never crossing a memory path
    x_via, x_power = (38.4, (14.2, 16.4)) if dual else (15.5, (38.4, 40.6))
    h_pkg, h_bump, h_die, h_mid, h_eic, h_sink = (
        (7.0, 1.4, 6.5, 21.0, 7.2, 5.5) if dual else (8.5, 1.5, 7.0, 24.0, 7.5, 6.5)
    )
    y_pkg = 1.5 if dual else 2.0
    y_bot = y_pkg + h_pkg + h_bump
    y_mid = y_bot + h_die
    y_sp = y_mid + h_eic
    y_top = y_mid + h_mid
    y_up = y_top + h_die
    y_bus = y_pkg + h_pkg - 1.7

    # bottom package with memory beside the stack, bottom die
    name = "Bottom package substrate" if dual else "Package substrate"
    at = (56.0, (y_pkg + y_bus) / 2)
    _block(ax, (1.0, y_pkg, 103.0, y_pkg + h_pkg), "pkg", name, at)
    for x0, x1 in ((xl, xr), (xm0, xm1)):
        _bumps(ax, x0, x1, y_pkg + h_pkg + h_bump / 2)
    _memory(ax, xm0, xm1, y_bot)
    _die(ax, (xl, xm, xr), y_bot, y_mid, ("PD + TIA", "GC receivers"))

    # middle layer: EIC face-down under a Si spacer, glass + metalens beside it
    eic_at = (30.0, (y_mid + y_sp) / 2)
    _block(ax, (xl, y_mid, xm, y_sp), "eic", "EIC (12LP,\nface-down)", eic_at)
    _block(ax, (xl, y_sp, xm, y_top), "si", "Si spacer", (30.0, (y_sp + y_top) / 2))
    glass_at = (50.5, (y_mid + y_top) / 2)
    _block(ax, (xm, y_mid, xr, y_top), "glass", "Glass +\n2D metalens", glass_at)
    _metasurface(ax, 59.0, 76.5, y_top - 1.7)
    _light(ax, 60.0, 75.0, y_top - 2.0, y_mid)
    for x in (17.0, 18.6, 20.2, 21.8):
        _vertical(ax, x, y_mid, y_top, "loop")
    if not dual:
        for x in x_power:
            _vertical(ax, x, y_bus + 1.0, y_top + 1.5, "power")
    _route(ax, [x_mem, x_mem, x_via, x_via], [y_bot, y_bus, y_bus, y_mid])

    # top die; B adds a top package with its own memory and power feed
    _die(ax, (xl, xm, xr), y_top, y_up, ("MRM + driver", "GC emitters"))
    y_sink = y_up
    if dual:
        y_tp = y_up + h_bump
        y_sink = y_tp + h_pkg - 0.5
        _bumps(ax, xl, xr, y_up + h_bump / 2)
        at = (56.0, y_tp + 4.3)
        _block(ax, (12.0, y_tp, 100.0, y_sink), "pkg", "Top package substrate", at)
        _bumps(ax, xm0, xm1, y_sink + h_bump / 2)
        _memory(ax, xm0, xm1, y_sink + h_bump)
        y_tbus = y_tp + 1.5
        _route(
            ax, [x_mem, x_mem, x_via, x_via], [y_sink + h_bump, y_tbus, y_tbus, y_up]
        )
        for x in x_power:
            _vertical(ax, x, y_tp + 5.0, y_up - 1.5, "power")
            _vertical(ax, x, y_pkg + 1.0, y_bot + 1.5, "power")
    _heat_sink(ax, xl, xr, y_sink, h_sink)

    _dimension(ax, 11.5, y_mid, y_top, xl)
    _path_legend(ax)
    return fig


@_styled
def xsec_a() -> Figure:
    """Architecture A: single-sided package + vertical interconnect."""
    return _xsec_ab(dual=False)


@_styled
def xsec_b() -> Figure:
    """Architecture B: dual-sided packaging, no power across the gap."""
    return _xsec_ab(dual=True)


@_styled
def xsec_c() -> Figure:
    """Architecture C: two optical columns, Fourier loop on the bottom die."""
    fig, ax = _canvas()
    xs = (13.0, 34.0, 63.0, 84.0)  # lens 1 | electronics | lens 2
    xc = 47.0  # centre-column labels, left of the vias
    centres = (23.5, xc, 73.5)
    xm0, xm1 = 87.5, 102.0
    x_mem, x_via, x_power = (xm0 + xm1) / 2, 62.0, (58.2, 60.1)
    y_pkg, h_pkg, h_bump = 2.0, 8.0, 1.5
    y_bot = y_pkg + h_pkg + h_bump
    y_mid = y_bot + 11.0
    y_eic = y_mid + 14.5
    y_top = y_mid + 22.0
    y_up = y_top + 7.5
    y_bus = y_pkg + h_pkg - 1.7
    y_glass = (y_mid + y_top) / 2

    at = (40.0, (y_pkg + y_bus) / 2)
    _block(ax, (1.0, y_pkg, 103.0, y_pkg + h_pkg), "pkg", "Package substrate", at)
    for x0, x1 in ((xs[0], xs[-1]), (xm0, xm1)):
        _bumps(ax, x0, x1, y_pkg + h_pkg + h_bump / 2)
    _memory(ax, xm0, xm1, y_bot)

    # bottom die: detect, then remodulate in place; the loop never leaves it
    labels = ("GC receivers", "PD + TIA → MRM", "GC emitters")
    y_loop = y_bot + 5.1
    _die(ax, xs, y_bot, y_mid, labels, y_bot + 8.5, y_bot + 6.5, centres)
    loop = ((15.5, y_loop), (82.0, y_loop))
    _arrow(ax, *loop, PATHS["loop"][0], 5 * PX, 22, zorder=5)
    _label(ax, 15.5, y_bot + 2.1, "Fourier loop stays on the bottom die", ha="left")

    # middle layer: lens 1 (down), EIC face-up over a Si spacer, lens 2 (up)
    _block(
        ax, (xs[0], y_mid, xs[1], y_top), "glass", "Glass +\nlens 1", (18.6, y_glass)
    )
    _block(
        ax, (xs[2], y_mid, xs[3], y_top), "glass", "Glass +\nlens 2", (78.6, y_glass)
    )
    si_at, eic_at = (xc, (y_mid + y_eic) / 2), (xc, (y_eic + y_top) / 2)
    _block(ax, (xs[1], y_mid, xs[2], y_eic), "si", "Si spacer", si_at)
    _block(ax, (xs[1], y_eic, xs[2], y_top), "eic", "EIC\n(12LP, face-up)", eic_at)
    _metasurface(ax, 24.0, 33.6, y_top - 1.7)
    _light(ax, 24.5, 33.0, y_top - 2.0, y_mid)
    _metasurface(ax, 64.0, 73.6, y_mid + 0.4)
    _light(ax, 64.5, 73.0, y_mid + 2.0, y_top)
    # power (package -> EIC -> top die) and memory hop over the Fourier loop
    lo, hi = y_loop - 0.4, y_loop + 0.4
    for x in x_power:
        _vertical(ax, x, y_bus + 1.0, lo, "power")
        _vertical(ax, x, hi, y_top + 1.5, "power")
    _route(ax, [x_mem, x_mem, x_via, x_via], [y_bot, y_bus, y_bus, lo], zorder=3)
    _route(ax, [x_via, x_via], [hi, y_eic], zorder=3)

    labels = ("GC emitters", "MRM + driver,\noutput PD + TIA", "GC receivers")
    _die(ax, xs, y_top, y_up, labels, at=centres)
    _heat_sink(ax, xs[0], xs[-1], y_up, 5.5)

    _dimension(ax, 11.5, y_mid, y_top, xs[0])
    _path_legend(ax)
    return fig


# --------------------------------------------------------------------------
# 4-8. Charts
# --------------------------------------------------------------------------


@_styled
def via_bw() -> Figure:
    """Analog lane rate per via across the gap, TGV vs TSV and driver R."""
    fig, ax = _chart()
    gap = m.StackParams().gap_um
    cases = [("TGV", 50.0), ("TGV", 100.0), ("TSV", 50.0), ("TSV", 100.0)]
    rates = [
        m.link_metrics(gap, m.ViaTech(kind=k), m.LinkParams(r_drv_ohm=r))["analog_gsps"]
        for k, r in cases
    ]
    colors = {"TGV": SERIES[0], "TSV": SERIES[1]}
    x = np.arange(len(cases))
    ax.bar(x, rates, width=0.6, color=[colors[k] for k, _ in cases])
    for xi, v in zip(x, rates):
        _note(ax, f"{v:.3g}", (xi, v))
    x_max = 5.3
    passes = m.ARCHITECTURES["A"].serial_passes
    for f in study.F_MOD:
        rate = f / passes / 1e9
        _ref_line(ax, rate, f"{f / 1e9:g} GHz loop: {rate:g} GS/s", x_max)
    ax.set_xticks(x, [f"{k}, {r:g} Ω" for k, r in cases])
    ax.set_xlim(-0.5, x_max)
    _log_y(ax, 0.1, 40)
    ax.set_ylabel("Analog rate per via (GS/s)")
    handles = [
        Patch(color=colors["TGV"], label="TGV (glass)"),
        Patch(color=colors["TSV"], label="TSV (silicon)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    return fig


def _loop_area(via: m.ViaTech, n: int, f_ghz: float) -> float | None:
    """Smallest analog (1:1 or TDM) Fourier-loop via area, None if too slow."""
    areas = [
        o["area_mm2"]
        for o in study.crossing_options(n, f_ghz * 1e9, via)
        if o["option"].startswith("analog") and o["area_mm2"] is not None
    ]
    return min(areas) if areas else None


def _staircase(fn, grid, tol: float = 1e-3):
    """Corners of a non-decreasing step function sampled on grid, jumps bisected.

    Stops where fn first returns None; returns (xs, ys, that rate or None).
    """
    pts = [(grid[0], fn(grid[0]))]

    def refine(a, b, yb):
        if pts[-1][1] == yb:
            return
        if b - a < tol:
            pts.append((b, yb))
            return
        mid = 0.5 * (a + b)
        refine(a, mid, fn(mid))
        refine(mid, b, yb)

    for a, b in zip(grid, grid[1:]):
        refine(a, b, fn(b))
        if pts[-1][1] is None:
            break
    end = pts.pop()[0] if pts[-1][1] is None else None
    pts.append((grid[-1] if end is None else end, pts[-1][1]))
    xs, ys = zip(*pts)
    return list(xs), list(ys), end


@_styled
def loop_area() -> Figure:
    """A/B Fourier-loop via area versus modulation rate for three via options."""
    fig, ax = _chart()
    p0 = m.StackParams()
    f_lo, f_hi = 0.5, 10.0
    grid = np.linspace(f_lo, f_hi, 39)
    x_lo, x_hi = f_lo - 0.25, 11.9
    budget = p0.via_area_budget_mm2
    for color, label in zip(SERIES, ("TGV AR10", "TSV AR10", "TGV AR20")):
        opt = study.VIA_OPTIONS[label]
        via = m.ViaTech(kind=opt["via__kind"], aspect_ratio=opt["via__aspect_ratio"])
        area = functools.partial(_loop_area, via, p0.n)
        fs, areas, limit = _staircase(area, grid)
        ax.plot(fs, areas, color=color, drawstyle="steps-post", label=label)
        if limit is None and areas[-1] >= budget:  # under the budget label
            _note(ax, label, (fs[-1], areas[-1]), dx=0, dy=-6, ha="right", va="top")
        elif limit is None:
            _note(ax, label, (fs[-1], areas[-1]), dx=6, dy=0, ha="left", va="center")
        else:
            end = {"ms": 14 * PX, "mew": 4 * PX, "zorder": 4}
            ax.plot([limit], [areas[-1]], "x", color=color, **end)
            text = f"{via.kind} too slow beyond ~{limit:.1f} GHz"
            _note(ax, text, (f_lo - 0.1, areas[-1]), ha="left")
    # over the series that sit on the budget, and alone past the last rate
    ax.hlines(budget, x_lo, x_hi, color=INK2, ls=(0, (5, 4)), lw=2 * PX, zorder=3)
    text = f"{budget:g} mm² via budget"
    _note(ax, text, (x_hi, budget), dx=-2, ha="right", color=INK2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(np.arange(0, 11, 2))
    _log_y(ax, 2, 600)
    ax.set_xlabel("Modulation rate (GHz)")
    ax.set_ylabel("Fourier-loop via area (mm²)")
    ax.legend(loc="lower right")
    return fig


@_styled
def speedup() -> Figure:
    """Workload speedup over the 1D JTC versus field size N."""
    fig, ax = _chart()
    p0 = m.StackParams()
    ns = sorted(set(range(24, 129, 8)) | {p0.n})
    target = p0.target_speedup
    ax.axhline(target, color=INK2, ls=(0, (5, 4)), lw=2 * PX)
    label_at = (ns[-1], target)
    _note(ax, f"{target:g}x target", label_at, dy=-4, ha="right", va="top", color=INK2)
    series = (
        ("A", "A and B (one lens, two passes)", (10, -10, "left", "top")),
        ("C", "C (two lenses, pipelined)", (-10, 12, "right", "bottom")),
    )
    for color, (key, label, (dx, dy, ha, va)) in zip(SERIES, series):
        vals = [
            m.evaluate(key, m.with_overrides(p0, n=n, via_area_budget_mm2=1e6))[
                "speedup_workload"
            ]
            for n in ns
        ]
        ax.plot(ns, vals, color=color, marker="o", label=label)
        n_hit, v_hit = next((n, v) for n, v in zip(ns, vals) if v >= target)
        v_n = vals[ns.index(p0.n)]
        ring = {"ms": 16 * PX, "mfc": "white", "mec": color, "mew": 3 * PX}
        ax.plot([n_hit, p0.n], [v_hit, v_n], "o", **ring)
        _note(ax, f"N = {n_hit}", (n_hit, v_hit), dx=dx, dy=dy, ha=ha, va=va)
        _note(ax, f"{v_n:.0f}x", (p0.n, v_n), dy=10)
    ax.set_xlim(20, 132)
    _log_y(ax, 1.5, 200)
    ax.set_xlabel("N (channels per side)")
    ax.set_ylabel("Speedup over 1D JTC")
    ax.legend(loc="upper left", **OPAQUE)
    return fig


def _rate_axis(ax, x: list[float], keys: list[tuple]) -> None:
    """Rate tick under each bar, architecture under each pair, then the title."""
    ax.set_xticks(x, [f"{f / 1e9:g}" for _, f in keys])
    ax.tick_params(axis="x", length=0)
    for i in range(0, len(x), 2):
        xy = ((x[i] + x[i + 1]) / 2, 0)
        _note(ax, keys[i][0], xy, dy=-30, va="top", xycoords=("data", "axes fraction"))
    ax.set_xlabel("Rate (GHz) / architecture", labelpad=34)  # rows, top to bottom


@_styled
def power_thermal() -> Figure:
    """Stack power by die and thermal footprint, A/B/C at 1 and 10 GHz."""
    fig, (axp, axt) = _chart(ncols=2)
    recs = _evaluate_rates(via_area_budget_mm2=250.0)
    keys = list(recs)
    x = [i + 0.5 * (i // 2) for i in range(len(keys))]

    dies = (("p_bottom_w", "Bottom die"), ("p_eic_w", "EIC"), ("p_top_w", "Top die"))
    bottom = np.zeros(len(keys))
    for color, (field, label) in zip(SERIES, dies):
        vals = np.array([recs[k][field] for k in keys])
        axp.bar(x, vals, 0.8, bottom, color=color, label=label, ec="white", lw=2 * PX)
        bottom += vals
    for xi, total in zip(x, bottom):
        _note(axp, f"{total:.0f}", (xi, total))
    axp.set_ylim(0, max(bottom) * 1.15)
    axp.yaxis.set_major_locator(MultipleLocator(50))
    axp.set_ylabel("Stack power (W)")
    axp.legend(loc="upper left", reverse=True, **OPAQUE)  # rows in stack order

    cap = m.StackParams().footprint_max_mm2
    for xi, k in zip(x, keys):
        if recs[k]["ok_thermal"]:
            axt.bar(xi, recs[k]["footprint_thermal_mm2"], 0.8, color=NEUTRAL)
        else:
            axt.bar(xi, cap, 0.8, color="white", ec=NEUTRAL, hatch="//", lw=2 * PX)
            _note(axt, f"> {cap:g}", (xi, cap))
    _ref_line(axt, cap, f"{cap:g} mm² cap", x[2] - 0.2)
    axt.set_ylim(0, cap * 1.2)
    axt.set_ylabel("Thermal footprint (mm²)")

    for ax in (axp, axt):
        _rate_axis(ax, x, keys)
    return fig


@_styled
def memory_bw() -> Figure:
    """Memory bandwidth per architecture against LPDDR5X and HBM3 parts."""
    fig, ax = _chart()
    recs = _evaluate_rates(via_area_budget_mm2=250.0)
    archs = list(m.ARCHITECTURES)
    width = 0.34
    for j, (color, f) in enumerate(zip(SERIES, study.F_MOD)):
        xs = np.arange(len(archs)) + (j - 0.5) * width
        vals = [recs[(a, f)]["mem_gbytes_s"] for a in archs]
        ax.bar(xs, vals, width * 0.92, color=color, label=f"{f / 1e9:g} GHz")
        for xi, v in zip(xs, vals):
            _note(ax, f"{v:.3g}", (xi, v), dy=3)
    x_max = 3.95
    lp_max = m.WorkloadParams().max_lpddr5x
    refs = (
        (m.LPDDR5X_GBYTES_S, "1 LPDDR5X"),
        (lp_max * m.LPDDR5X_GBYTES_S, f"{lp_max} LPDDR5X"),
        (m.HBM3_GBYTES_S, "1 HBM3"),
    )
    for value, name in refs:
        _ref_line(ax, value, f"{name} ({value:.0f} GB/s)", x_max)
    ax.set_xticks(np.arange(len(archs)), archs)
    ax.set_xlim(-0.5, x_max)
    _log_y(ax, 6, 1300)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Memory bandwidth (GB/s)")
    fig.legend(loc="outside upper center", ncol=2)
    return fig


# --------------------------------------------------------------------------
# 9. Feasibility matrix
# --------------------------------------------------------------------------

_GLUE = re.compile(r"^([>≥=]|[\d.:]+)$")  # never break right after these


def _tokens(text: str) -> list[str]:
    """Words, keeping numbers with their units and "N = 100" together."""
    out: list[str] = []
    for word in text.split(" "):
        if out and (_GLUE.match(out[-1].split(" ")[-1]) or word == "="):
            out[-1] += " " + word
        else:
            out.append(word)
    return out


def _wrap(text: str, width_in: float, measure) -> str:
    """Greedy wrap to width_in; two-line results are balanced."""
    words, lines = _tokens(text), []
    for word in words:
        if lines and measure(f"{lines[-1]} {word}") <= width_in:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    if len(lines) == 2:
        splits = [
            (" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))
        ]
        lines = list(min(splits, key=lambda s: max(map(measure, s))))
    return "\n".join(lines)


@_styled
def feasibility() -> Figure:
    """Pass / marginal / fail matrix for A, B, C at 1 and 10 GHz."""
    fig = _figure()
    ax = fig.add_axes((0, 0, 1, 1))
    width, height = FIGSIZE
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(size=FS)

    def measure(s: str) -> float:
        return renderer.get_text_width_height_descent(s, prop, ismath=False)[0] / DPI

    rows = {
        row: [cell.split(": ", 1) for cell in cells.split("; ")]
        for row, cells in FEASIBILITY.items()
    }
    columns = [f"{a} {f / 1e9:g} GHz" for a in m.ARCHITECTURES for f in study.F_MOD]
    # cells fit their widest unbreakable word; padding gives way to the row
    # headers until those reach head_pref, but never drops below 0.04 in
    words = [w for cells in rows.values() for c in cells for w in _tokens(c[1])]
    inner = max(map(measure, words + list(STATUS)))
    margin, head_h, head_pref, gap = 0.05, 0.5, 1.45, 0.06
    spare = width - 2 * margin - head_pref - len(columns) * (inner + gap)
    pad = min(0.08, max(0.04, spare / (2 * len(columns))))
    col_w = inner + 2 * pad + gap
    head_w = width - 2 * margin - len(columns) * col_w
    row_h = (height - 2 * margin - head_h) / len(rows)
    x0, y0 = width - margin - len(columns) * col_w, height - margin - head_h
    for j, name in enumerate(columns):
        ax.text(x0 + (j + 0.5) * col_w, y0 + head_h / 2, name, ha="center", va="center")
    for i, (row, cells) in enumerate(rows.items()):
        y_mid = y0 - (i + 0.5) * row_h
        header = _wrap(row, head_w - 0.04, measure)
        ax.text(margin, y_mid, header, va="center", linespacing=1.15)
        for j, (status, reason) in enumerate(cells):
            face, ink = STATUS[status]
            corner = (x0 + j * col_w + gap / 2, y_mid - (row_h - gap) / 2)
            box = "round,pad=0,rounding_size=0.08"
            ax.add_patch(
                FancyBboxPatch(corner, col_w - gap, row_h - gap, box, fc=face, lw=0)
            )
            text = f"{status}\n{_wrap(reason, col_w - gap - 0.08, measure)}"
            ax.text(
                x0 + (j + 0.5) * col_w,
                y_mid,
                text,
                ha="center",
                va="center",
                color=ink,
                linespacing=1.15,
            )
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

FIGURES = {
    "xsec_a.png": xsec_a,
    "xsec_b.png": xsec_b,
    "xsec_c.png": xsec_c,
    "via_bw.png": via_bw,
    "loop_area.png": loop_area,
    "speedup.png": speedup,
    "power_thermal.png": power_thermal,
    "memory_bw.png": memory_bw,
    "feasibility.png": feasibility,
}


def render(out_dir: Path, names=None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in FIGURES.items():
        if names and name not in names:
            continue
        path = out_dir / name
        build().savefig(path, dpi=DPI, facecolor="white", pil_kwargs={"optimize": True})
        paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="packaging2d/figures")
    parser.add_argument("--only", nargs="*", help="file names to render (default all)")
    args = parser.parse_args(argv)
    for path in render(Path(args.output_dir), args.only):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
