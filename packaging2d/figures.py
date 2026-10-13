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
from matplotlib.transforms import offset_copy

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
    "lines.linewidth": 4.5 * PX,
    "lines.markersize": 12 * PX,
    "hatch.linewidth": 2 * PX,
}
# chart margins (figure fractions); grouped charts need two rows of x labels
MARGINS = {"left": 0.105, "right": 0.985, "bottom": 0.135, "top": 0.88}
GROUPED_BOTTOM = 0.165
DIGIT_H = 26.0  # px: ink height of digits and capitals at FS
CLEAR = 6.0  # px: least gap between a label's ink and a gridline
# figures 7-9 raise the gap-via budget (power + loop vias) from the 200 mm2
# default in loop_area, so A keeps 1:1 lanes at 10 GHz (its TGV field is
# 201 mm2 with power vias); the note says "raised" so the two numbers agree
BUDGET_MM2 = 250.0
BUDGET_NOTE = f"Via budget raised to {BUDGET_MM2:g} mm²"
# the widest y tick label in the deck (memory_bw); every y-axis title clears
# it, so the titles share one x and do not jump between slides
Y_TITLE_TICK = "1000"
# bar legends use square swatches, compact enough to sit within one panel
SWATCH = dict(handlelength=0.7, handleheight=0.7, handletextpad=0.4, columnspacing=0.8)

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


def _figure() -> Figure:
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _chart(ncols: int = 1, bottom: float = MARGINS["bottom"]):
    """Figure with the deck's fixed chart margins (equal panels)."""
    fig = _figure()
    axes = fig.subplots(1, ncols)
    fig.subplots_adjust(**{**MARGINS, "bottom": bottom}, wspace=0.3)
    return fig, axes


def _legend(fig, ax, handles=None, **style) -> None:
    """Frameless one-row legend above the axes, centred on it."""
    handles = handles or ax.get_legend_handles_labels()[0]
    box = ax.get_position()
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=((box.x0 + box.x1) / 2, 0.995),
        ncol=len(handles),
        frameon=False,
        **{"handlelength": 1.8, "columnspacing": 1.6, **style},
    )


def _setting_note(fig, text: str = BUDGET_NOTE) -> None:
    """Model setting in secondary ink, top right on the legend row."""
    probe = fig.legend(
        handles=[Patch(label="probe")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    box = probe.get_window_extent(fig.canvas.get_renderer())
    probe.remove()
    y = (box.y0 + box.y1) / 2 / fig.bbox.height
    fig.text(0.987, y, text, ha="right", va="center", color=INK2)


def _text_px(fig, text: str) -> float:
    """Width (px) of the widest line of text at FS."""
    renderer, prop = fig.canvas.get_renderer(), FontProperties(size=FS)
    return max(
        renderer.get_text_width_height_descent(line, prop, ismath=False)[0]
        for line in text.split("\n")
    )


def _y_title(ax, text: str) -> None:
    """Y-axis title at one distance from the spine in every chart.

    The distance clears the widest tick label of the deck (Y_TITLE_TICK) as
    matplotlib would, so all titles share one x however wide the ticks are.
    """
    ax.set_ylabel(text)
    rc = mpl.rcParams
    pts = rc["ytick.major.size"] + rc["ytick.major.pad"] + rc["axes.labelpad"]
    dx = pts + _text_px(ax.figure, Y_TITLE_TICK) * PX
    at = offset_copy(ax.transAxes, fig=ax.figure, x=-dx, units="points")
    ax.yaxis.set_label_coords(0, 0.5, transform=at)


def _label_room(ax, x0: float, x_end: float, texts, gap: float = 30.0) -> float:
    """Right x limit leaving gap px plus the widest label line right of x_end.

    x0 is the left x limit; labels end 2 pt inside the right limit (_ref_line).
    """
    width = ax.get_position().width * ax.figure.bbox.width
    k = (gap + max(_text_px(ax.figure, t) for t in texts) + 2 / PX) / width
    return (x_end - k * x0) / (1 - k)


def _log_y(ax, lo: float, hi: float, subs=(1.0, 2.0, 5.0)) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _value(
    ax, text, xy, lift=10.0, below=False, min_lift=5.0, dx=0.0, **kwargs
) -> None:
    """Label lift px above (or below) xy, its ink kept CLEAR px off gridlines.

    A label that would touch a gridline tucks in front of it when it can stay
    min_lift px off xy, and otherwise moves past it. Needs the final axes
    position and limits. Labels have no pad, so nothing is cut.
    """
    lo, hi = sorted(ax.get_ylim())
    ys = [t for t in ax.get_yticks() if lo <= t <= hi]
    y0 = ax.transData.transform(xy)[1]
    sign = -1.0 if below else 1.0
    dists = sorted(sign * (ax.transData.transform((0, y))[1] - y0) for y in ys)
    near = lift  # px from xy to the near edge of the ink
    for d in (d for d in dists if d > -CLEAR):
        if near + DIGIT_H + CLEAR <= d:
            break
        if near >= d + CLEAR:
            continue
        tuck = d - CLEAR - DIGIT_H
        near = tuck if tuck >= min_lift else d + CLEAR
    dy = near if not below else -(near + DIGIT_H)
    style = {"ha": "center", "va": "baseline", **kwargs}
    ax.annotate(
        text, xy, xytext=(dx * PX, dy * PX), textcoords="offset points", **style
    )


def _ref_line(
    ax,
    y: float,
    text: str,
    x_text: float,
    side: str = "above",
    ha: str = "right",
    zorder: float = 1.5,
) -> None:
    """Dashed horizontal reference labelled above (or below) x_text.

    The label ends at x_text (ha="right") or starts there (ha="left"). Bar
    charts pass a zorder under the bars' (1), so bars rise over their line.
    """
    ax.axhline(y, color=INK2, ls=(0, (5, 4)), lw=2 * PX, zorder=zorder)
    if not text:
        return
    dy, va = (3, "bottom") if side == "above" else (-3, "top")
    ax.annotate(
        text,
        (x_text, y),
        xytext=(-2 if ha == "right" else 3, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=INK2,
        multialignment=ha,
    )


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
# Cross-section layout shared by A, B and C (axes units of 0.1 inch, so one
# unit is 15 px) and its primitives
# --------------------------------------------------------------------------

XL = 12.0  # stack left edge; the 500 um dimension sits in the gutter left of it
XR = 84.0  # stack right edge, shared by A, B and C
UNIT_PX = DPI / 10  # output px per layout unit
Y_PKG, H_PKG, H_BUMP, H_DIE, H_MID = 1.5, 7.0, 1.4, 6.5, 21.0
H_EIC, H_SINK, H_MEM = 7.5, 5.5, 4.6
Y_BUMP = Y_PKG + H_PKG + H_BUMP / 2  # package-to-die bump row
Y_BOT = Y_PKG + H_PKG + H_BUMP  # underside of the bottom die
Y_BUS = Y_PKG + H_PKG - 1.7  # memory bus inside the package
PITCH = 2.2  # bump pitch; vias that meet a bump row sit on bump centres
HOP = 1.1  # half-gap where a via hops over an arrow
BEAM = 6.0  # width of the light beam (emitter span = receiver span)
N_BARS = 13  # metalens bars and GC marks across the beam; arrows sit on bars
BAR_W, GC_W, GC_H = 0.34, 0.22, 0.6  # bar width; GC mark width and height
X_MEM = (86.0, 103.5)  # memory beside the stack
X_PKG = 73.0  # package labels, clear of every via that enters a package
X_NOTE = 102.7  # right edge of the "not to scale" note


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


def _die(ax, xs, y0, y1, labels, at, label_y=None, split_from=None) -> None:
    """One PIC die over xs[0]..xs[-1], dashed region splits, a label per region."""
    _block(ax, (xs[0], y0, xs[-1], y1), "pic")
    for x in xs[1:-1]:
        y_split = (y0 if split_from is None else split_from, y1)
        ax.plot([x, x], y_split, color=FILL["pic"][1], lw=2 * PX, ls=(0, (4, 3)))
    for x, text in zip(at, labels):
        if text:
            _label(ax, x, (y0 + y1) / 2 if label_y is None else label_y, text)


def _grid(anchor: float, x0: float, x1: float) -> list[float]:
    """Bump centres on PITCH through anchor, inside x0..x1."""
    k0 = math.ceil((x0 + 0.7 - anchor) / PITCH)
    k1 = math.floor((x1 - 0.7 - anchor) / PITCH)
    return [anchor + k * PITCH for k in range(k0, k1 + 1)]


def _bumps(ax, xs, y: float) -> None:
    face, edge = FILL["bump"]
    for x in xs:
        ax.add_patch(Circle((x, y), 0.55, fc=face, ec=edge, lw=PX, zorder=2))


def _memory(ax, y: float) -> tuple[float, float]:
    """LPDDR5/HBM on bumps on a package whose top is y; returns its bump centre."""
    x0, x1 = X_MEM
    xc, y_bump = (x0 + x1) / 2, y + H_BUMP / 2
    _bumps(ax, _grid(xc, x0, x1), y_bump)
    _block(ax, (x0, y + H_BUMP, x1, y + H_BUMP + H_MEM), "mem", "LPDDR5/HBM")
    return xc, y_bump


def _tgv_field(ax, x0: float, x1: float, y0: float, y1: float, x_label) -> None:
    """Full-height glass strip that carries every via crossing the gap."""
    _block(ax, (x0, y0, x1, y1), "glass")
    _label(ax, x_label, (y0 + y1) / 2, "TGV field", rotation=90)


def _optics(ax, x0: float, y_die: float, y_to: float, label_side: str) -> None:
    """GC emitters -> metalens -> GC receivers as one constant-width beam.

    Both GC arrays are rows of marks just inside the die faces, as wide as the
    beam; the arrows run face to face and pass under the metalens bars.
    """
    color = PATHS["light"][0]
    x1, s = x0 + BEAM, math.copysign(1.0, y_to - y_die)
    xs = np.linspace(x0, x1, N_BARS)
    mark = {"fc": FILL["pic"][1], "lw": 0, "zorder": 3}
    for y_face, inward in ((y_die, -s), (y_to, s)):
        y_mark = min(y_face, y_face + inward * GC_H)
        for x in xs:
            ax.add_patch(Rectangle((x - GC_W / 2, y_mark), GC_W, GC_H, **mark))
    y_near, y_far = y_die + 1.2 * s, y_die + 2.3 * s
    y_bar = min(y_near, y_far)
    bar = {"fc": color, "lw": 0, "zorder": 5}  # over the arrows
    for x in xs:
        ax.add_patch(Rectangle((x - BAR_W / 2, y_bar), BAR_W, 1.1, **bar))
    for x in np.linspace(x0, x1, 4):
        _arrow(ax, (x, y_die), (x, y_to), color, 4 * PX, scale=18)
    y_text = (y_near + y_far) / 2
    if label_side == "left":
        _label(ax, x0 - BAR_W / 2 - 0.8, y_text, "metalens", ha="right")
    else:
        _label(ax, x1 + BAR_W / 2 + 0.8, y_text, "metalens", ha="left")


def _via(ax, x: float, y0: float, y1: float, kind: str, hops=()) -> None:
    """Vertical path from y0 to y1, broken by +/-HOP around each y in hops.

    Loop lanes are drawn like the other vias, with one small arrowhead. A
    dashed piece has its pattern stretched (a few %) to start and end on a
    dash, so it visibly reaches the die face, bump or hop it ends at.
    """
    color, _, style = PATHS[kind]
    ys = [y0]
    for y in sorted(hops, reverse=y1 < y0):
        ys += [y - math.copysign(HOP, y1 - y0), y + math.copysign(HOP, y1 - y0)]
    ys.append(y1)
    pieces = list(zip(ys[::2], ys[1::2]))
    lw = 6 * PX if kind == "power" else 5 * PX
    for i, (a, b) in enumerate(pieces):
        if kind == "loop" and i == len(pieces) - 1:
            _arrow(ax, (x, a), (x, b), color, lw, scale=11)
            continue
        ls = style
        if isinstance(style, tuple):  # dashes scale with lw (points)
            on, off = style[1]
            span = abs(b - a) * UNIT_PX * PX / lw  # in linewidths
            k = max(round((span - on) / (on + off)), 0)
            s = span / (k * (on + off) + on)
            ls = (0, (on * s, off * s))
        ax.plot([x, x], [a, b], color=color, ls=ls, lw=lw, solid_capstyle="butt")


def _route(ax, xs: list[float], ys: list[float], hops=()) -> None:
    """Dashed memory path; its last (vertical) leg hops over arrows at hops."""
    color, _, style = PATHS["memory"]
    if hops:
        _via(ax, xs[-1], ys[-2], ys[-1], "memory", hops)
        xs, ys = xs[:-1], ys[:-1]
    ax.plot(xs, ys, color=color, ls=style, lw=5 * PX, zorder=4)


def _dimension(ax, y0: float, y1: float) -> None:
    """Double arrow across the middle layer labelled with the gap."""
    x = XL - 1.2
    _arrow(ax, (x, y0), (x, y1), INK2, 3 * PX, scale=16, style="<|-|>")
    for y in (y0, y1):
        ax.plot([x - 0.8, XL], [y, y], color=INK2, lw=2 * PX)
    _label(ax, x - 1.2, (y0 + y1) / 2, f"{m.StackParams().gap_um:g} µm", ha="right")


def _frame(ax, y_mid: float, y_top: float, loop_label: str) -> None:
    """Gap dimension, path legend (top-left row), "not to scale" at its right."""
    _dimension(ax, y_mid, y_top)
    labels = {**{k: v[1] for k, v in PATHS.items()}, "loop": loop_label}
    handles = [
        Line2D([], [], color=c, ls=s, lw=6 * PX, label=labels[k])
        for k, (c, _, s) in PATHS.items()
    ]
    legend = ax.legend(
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
    # on the legend entries' baseline (legend text is baseline-anchored)
    ax.figure.canvas.draw()
    entry = legend.get_texts()[0]
    y_px = entry.get_transform().transform(entry.get_position())[1]
    y = ax.transData.inverted().transform((0, y_px))[1]
    _label(ax, X_NOTE, y, "not to scale", ha="right", va="baseline", color=INK2)


# --------------------------------------------------------------------------
# 1-3. Cross-sections
# --------------------------------------------------------------------------


def _xsec_ab(dual: bool) -> Figure:
    """A (single-sided) or B (dual-sided): one optical column, lanes cross."""
    fig, ax = _canvas()
    xl, xm, xr = XL, 47.0, XR  # stack left, electronics | optics, stack right
    x_pwr = (27.0, 27.0 + PITCH)  # power vias; the bump grid runs through them
    lanes = [x_pwr[0] + k * PITCH for k in range(-5, -2)]  # analog lanes
    tgv = (xl, x_pwr[1] + 1.4)
    x_tgv = (lanes[-1] + x_pwr[0]) / 2  # TGV field label between the groups
    x_via = x_pwr[0] + 8 * PITCH  # memory into the EIC, clear of the die label
    x_beam = 63.5  # leaves room for the passes note right of the beam
    x_eic = (tgv[1] + xm) / 2
    x_opt = x_beam + BEAM / 2
    y_mid = Y_BOT + H_DIE
    y_top = y_mid + H_MID
    y_up = y_top + H_DIE
    grid = _grid(x_pwr[0], xl, xr)

    # package with memory beside the stack, bottom die
    name = "Bottom package substrate" if dual else "Package substrate"
    at = (X_PKG, (Y_PKG + Y_BUS) / 2)
    _block(ax, (0.8, Y_PKG, 104.0 - 0.4, Y_PKG + H_PKG), "pkg", name, at)
    _bumps(ax, grid, Y_BUMP)
    x_mem, y_mem = _memory(ax, Y_PKG + H_PKG)
    labels, at = ("PD + TIA", "GC receivers"), ((x_pwr[1] + x_via) / 2, x_opt)
    _die(ax, (xl, xm, xr), Y_BOT, y_mid, labels, at)

    # middle layer: TGV field | EIC face-down under a Si spacer | glass + metalens
    _tgv_field(ax, *tgv, y_mid, y_top, x_tgv)
    y_sp = y_mid + H_EIC
    _block(ax, (tgv[1], y_mid, xm, y_sp), "eic", "EIC (12LP,\nface-down)")
    _block(ax, (tgv[1], y_sp, xm, y_top), "si", "Si spacer", (x_eic, y_sp + 6.0))
    glass_at = ((xm + x_beam - 1.6) / 2, y_mid + 7.0)
    _block(ax, (xm, y_mid, xr, y_top), "glass", "Glass", glass_at)
    _optics(ax, x_beam, y_top, y_mid, "left")
    x_note = x_beam + BEAM + BAR_W / 2 + 1.5
    _label(ax, x_note, (y_mid + y_top) / 2, "2 passes\nper shot", ha="left", color=INK2)
    for x in lanes:
        _via(ax, x, y_mid, y_top, "loop")
    if not dual:  # power enters from the package, as deep as B's package feeds
        for x in x_pwr:
            _via(ax, x, Y_PKG + 1.0, y_top, "power")
    _route(ax, [x_mem, x_mem, x_via, x_via], [y_mem, Y_BUS, Y_BUS, y_mid])

    # top die; B adds a top package with its own memory and power feed
    labels, at = ("MRM + driver", "GC emitters"), ((xl + xm) / 2, x_opt)
    _die(ax, (xl, xm, xr), y_top, y_up, labels, at)
    y_sink = y_up
    if dual:
        y_tb = y_up + H_BUMP / 2
        y_tp = y_up + H_BUMP
        y_sink = y_tp + H_PKG
        _bumps(ax, grid, y_tb)
        at = (X_PKG, y_tp + 4.4)
        box = (0.8, y_tp, 104.0 - 0.4, y_sink)  # same edges as the bottom one
        _block(ax, box, "pkg", "Top package substrate", at)
        x_mem, y_mem = _memory(ax, y_sink)
        y_tbus = y_tp + 1.7
        _route(ax, [x_mem, x_mem, x_via, x_via], [y_mem, y_tbus, y_tbus, y_tb])
        for x in x_pwr:
            _via(ax, x, y_sink - 1.0, y_tb, "power")
            _via(ax, x, Y_PKG + 1.0, Y_BUMP, "power")
    _block(ax, (xl, y_sink, xr, y_sink + H_SINK), "sink", "Heat sink")

    loop = "Fourier loop" if dual else "Analog lanes (loop + input)"
    _frame(ax, y_mid, y_top, loop)
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
    xs = (XL, 32.5, 63.5, XR)  # lens 1 | electronics + TGV field | lens 2
    # TGV field, left to right: label | power vias | memory via. The memory via
    # is rightmost so the package bus from the memory crosses no power via.
    x_mv = xs[2] - 1.8  # memory via, 25 px clear of the lens-2 column
    x_pwr = (x_mv - 2 * PITCH, x_mv - PITCH)
    x_tgv = x_pwr[0] - 2.7  # label >= 25 px from the power via and the EIC
    tgv = (x_tgv - 2.6, xs[2])
    centres = ((xs[0] + xs[1]) / 2, (xs[1] + xs[2]) / 2, (xs[2] + xs[3]) / 2)
    h_bot = 10.5  # room for the loop arrow and its label
    y_mid = Y_BOT + h_bot
    y_top = y_mid + H_MID
    y_up = y_top + H_DIE
    y_loop = Y_BOT + 4.6

    at = (X_PKG, (Y_PKG + Y_BUS) / 2)
    _block(ax, (0.8, Y_PKG, 104.0 - 0.4, Y_PKG + H_PKG), "pkg", "Package substrate", at)
    _bumps(ax, _grid(x_mv, xs[0], xs[-1]), Y_BUMP)
    x_mem, y_mem = _memory(ax, Y_PKG + H_PKG)

    # bottom die: detect, then remodulate in place; the loop never leaves it
    labels = ("GC receivers", "PD + TIA → MRM", "GC emitters")
    at = (centres[0], (xs[1] + x_pwr[0]) / 2, centres[2])  # clear of the vias
    _die(ax, xs, Y_BOT, y_mid, labels, at, Y_BOT + 7.9, Y_BOT + 6.3)
    _arrow(ax, (XL + 2.0, y_loop), (xs[-1] - 2.0, y_loop), PATHS["loop"][0], 5 * PX, 22)
    _label(ax, XL + 0.9, Y_BOT + 2.0, "Fourier loop stays on the bottom die", ha="left")

    # middle layer: lens 1 (down) | EIC face-up over a Si spacer | TGV | lens 2 (up)
    edge = 1.67 + BAR_W / 2  # beam (with its bars) >= 25 px from the column edges
    x_b1 = xs[1] - edge - BEAM
    x_b2 = xs[2] + edge
    at1 = ((XL + x_b1 - 1.2) / 2, y_mid + 7.0)
    at2 = ((x_b2 + BEAM + 1.2 + xs[3]) / 2, y_top - 7.0)
    _block(ax, (xs[0], y_mid, xs[1], y_top), "glass", "Glass +\nlens 1", at1)
    _block(ax, (xs[2], y_mid, xs[3], y_top), "glass", "Glass +\nlens 2", at2)
    y_eic = y_top - H_EIC
    _block(ax, (xs[1], y_mid, tgv[0], y_eic), "si", "Si spacer")
    _block(ax, (xs[1], y_eic, tgv[0], y_top), "eic", "EIC\n(12LP, face-up)")
    _tgv_field(ax, *tgv, y_mid, y_top, x_tgv)
    _optics(ax, x_b1, y_top, y_mid, "left")
    _optics(ax, x_b2, y_mid, y_top, "right")
    # power and serialized memory: package -> bottom die -> TGV field -> top die
    for x in x_pwr:
        _via(ax, x, Y_PKG + 1.0, y_top, "power", hops=[y_loop])
    _route(ax, [x_mem, x_mem, x_mv, x_mv], [y_mem, Y_BUS, Y_BUS, y_top], [y_loop])

    labels = ("GC emitters", "", "GC receivers")
    _die(ax, xs, y_top, y_up, labels, centres)
    # tight leading keeps the two-line label >= 12 px off both die faces
    text = "MRM + driver,\noutput PD + TIA"
    _label(ax, centres[1], (y_top + y_up) / 2, text, linespacing=0.95)
    _block(ax, (xs[0], y_up, xs[-1], y_up + H_SINK), "sink", "Heat sink")

    _frame(ax, y_mid, y_top, "Fourier loop")
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
    width = 0.7
    ax.bar(x, rates, width=width, color=[colors[k] for k, _ in cases])
    # both loop lines are labelled alike: two lines, above the line, ending
    # at the right edge of a strip right of the bars just wide enough for them
    passes = m.ARCHITECTURES["A"].serial_passes
    refs = [(f / passes / 1e9, f"{f / 1e9:g} GHz loop:") for f in study.F_MOD]
    texts = [f"{name}\n{v:g} GS/s" for v, name in refs]
    x_max = _label_room(ax, -0.6, x[-1] + width / 2, texts)
    ax.set_xlim(-0.6, x_max)
    _log_y(ax, 0.1, 30)
    for xi, v in zip(x, rates):
        _value(ax, f"{v:.3g}", (xi, v))
    for (v, _), text in zip(refs, texts):
        _ref_line(ax, v, text, x_max, zorder=0.9)  # under the bars
    ax.set_xticks(x, [f"{k}, {r:g} Ω" for k, r in cases])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlabel(f"Via type ({gap:g} µm long), driver resistance")
    _y_title(ax, "Analog rate per via (GS/s)")
    handles = [
        Patch(color=colors["TGV"], label="TGV (glass)"),
        Patch(color=colors["TSV"], label="TSV (silicon)"),
    ]
    _legend(fig, ax, handles, **SWATCH)
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
    """Fourier-loop via area versus modulation rate for three via options.

    These are B's n^2 loop lanes. A also sends input-plane updates over them
    (about 0.3% more samples), which at exactly 1 GHz, a TDM step edge, drops
    A from TDM x8 to x7: 29 mm2 instead of 25 mm2 (see feasibility.png).
    """
    fig, ax = _chart()
    p0 = m.StackParams()
    f_lo, f_hi = 0.5, 10.0
    grid = np.linspace(f_lo, f_hi, 39)
    budget = p0.via_area_budget_mm2
    ax.set_xlim(0, f_hi + 0.5)
    _log_y(ax, 2, 600)
    # under the series (zorder), across the whole axis from the spine; its
    # label sits below it at the right end, in the empty band above the 100
    # gridline, clear of the TSV staircase that reaches 200 mm2 at the left
    _ref_line(ax, budget, f"{budget:g} mm² via budget", f_hi + 0.5, side="below")
    style = {"textcoords": "offset points", "va": "bottom", "color": INK}
    # TSV is orange here as in via_bw, so with the fixed colour order it is
    # the second series (and legend entry)
    for color, label in zip(SERIES, ("TGV AR10", "TSV AR10", "TGV AR20")):
        opt = study.VIA_OPTIONS[label]
        via = m.ViaTech(kind=opt["via__kind"], aspect_ratio=opt["via__aspect_ratio"])
        area = functools.partial(_loop_area, via, p0.n)
        fs, areas, limit = _staircase(area, grid)
        ax.plot(fs, areas, color=color, drawstyle="steps-post", label=label)
        if limit is None:
            _value(ax, label, (fs[-1], areas[-1]), dx=-3, ha="right")
        else:
            mark = {"ms": 14 * PX, "mew": 4.5 * PX, "zorder": 4}
            ax.plot([limit], [areas[-1]], "x", color=color, **mark)
            text = f"{via.kind} too slow\nbeyond ~{limit:.1f} GHz"
            xy = (limit, areas[-1])
            ax.annotate(text, xy, xytext=(9, 3), ha="left", **style)
    ax.set_xticks(np.arange(0, f_hi + 1, 2))
    ax.set_xlabel("Modulation rate (GHz)")
    _y_title(ax, "Fourier-loop via area (mm²)")
    _legend(fig, ax)
    return fig


@_styled
def speedup() -> Figure:
    """Workload speedup over the 1D JTC versus field size N."""
    fig, ax = _chart()
    p0 = m.StackParams()
    ns = sorted(set(range(24, 129, 8)) | {p0.n})
    target = p0.target_speedup
    x_max = 132
    ax.set_xlim(16, x_max)  # first x tick clear of the y tick in the corner
    _log_y(ax, 1, 200)
    _ref_line(ax, target, f"{target:g}x target", x_max)
    # the lower curve is labelled below its points, the upper one above
    series = (
        ("A", "A and B (one lens, two passes)", (20, True, "left")),
        ("C", "C (two lenses, pipelined)", (-20, False, "right")),
    )
    ring_r = 11.0  # px
    for color, (key, label, (dx, below, ha)) in zip(SERIES, series):
        vals = [
            m.evaluate(key, m.with_overrides(p0, n=n, via_area_budget_mm2=1e6))[
                "speedup_workload"
            ]
            for n in ns
        ]
        ax.plot(ns, vals, color=color, marker="o", label=label)
        n_hit, v_hit = next((n, v) for n, v in zip(ns, vals) if v >= target)
        v_n = vals[ns.index(p0.n)]
        ring = {"ms": 2 * ring_r * PX, "mfc": "white", "mec": color, "mew": 4 * PX}
        ax.plot([n_hit, p0.n], [v_hit, v_n], "o", **ring)
        near = {"lift": ring_r + 12, "min_lift": ring_r + 4, "below": below}
        _value(ax, f"N = {n_hit}", (n_hit, v_hit), dx=dx, ha=ha, **near)
        _value(ax, f"{v_n:.0f}x", (p0.n, v_n), **near)
    ax.set_xlabel("N (channels per side)")
    _y_title(ax, "Speedup over 1D JTC")
    _legend(fig, ax)
    return fig


def _grouped_x(fig, axes, keys: list[tuple]) -> list[float]:
    """Rate under each bar, architecture under each pair, one shared x label."""
    x = [i + 0.5 * (i // 2) for i in range(len(keys))]
    for ax in axes:
        ax.set_xticks(x, [f"{f / 1e9:g}" for _, f in keys])
        ax.tick_params(axis="x", length=0, pad=6)
        for i in range(0, len(x), 2):
            xy = ((x[i] + x[i + 1]) / 2, 0)
            ax.annotate(
                keys[i][0],
                xy,
                xycoords=("data", "axes fraction"),
                xytext=(0, -30),
                textcoords="offset points",
                ha="center",
                va="top",
            )
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    text = "Modulation rate (GHz) / architecture"
    fig.text((left + right) / 2, 0.012, text, ha="center", va="bottom")
    return x


@_styled
def power_thermal() -> Figure:
    """Stack power by die and thermal footprint, A/B/C at 1 and 10 GHz."""
    # equal panels with equal x limits draw the six cases at one scale
    fig, (axp, axt) = _chart(ncols=2, bottom=GROUPED_BOTTOM)
    recs = _evaluate_rates(via_area_budget_mm2=BUDGET_MM2)
    keys = list(recs)
    x = _grouped_x(fig, (axp, axt), keys)
    x_lim = (x[0] - 0.6, x[-1] + 0.6)

    dies = (("p_bottom_w", "Bottom die"), ("p_eic_w", "EIC"), ("p_top_w", "Top die"))
    bottom = np.zeros(len(keys))
    for color, (field, label) in zip(SERIES, dies):
        vals = np.array([recs[k][field] for k in keys])
        axp.bar(x, vals, 0.8, bottom, color=color, label=label, ec="white", lw=2 * PX)
        bottom += vals
    axp.set_xlim(*x_lim)
    axp.set_ylim(0, max(bottom) * 1.15)
    axp.yaxis.set_major_locator(MultipleLocator(50))
    for xi, total in zip(x, bottom):
        _value(axp, f"{total:.0f}", (xi, total))
    _y_title(axp, "Stack power (W)")
    # bottom die, EIC, top die (the stacking order), within the left panel
    _legend(fig, axp, **SWATCH)
    _setting_note(fig)

    cap = m.StackParams().footprint_max_mm2
    axt.set_xlim(*x_lim)
    axt.set_ylim(0, cap * 1.2)
    # the cap label goes above its line at the left end, over the short A
    # bars and well apart from the "> 400" label over B 10 GHz
    _ref_line(axt, cap, f"{cap:g} mm² cap", x_lim[0], ha="left")
    for xi, k in zip(x, keys):
        area = recs[k]["footprint_thermal_mm2"]
        if recs[k]["ok_thermal"]:
            axt.bar(xi, area, 0.8, color=NEUTRAL)
            _value(axt, f"{area:.0f}", (xi, area))
        else:
            axt.bar(xi, cap, 0.8, color="white", ec=NEUTRAL, hatch="//", lw=2 * PX)
            _value(axt, f"> {cap:g}", (xi, cap))
    _y_title(axt, "Thermal footprint (mm²)")
    return fig


@_styled
def memory_bw() -> Figure:
    """Memory bandwidth per case against LPDDR5X and HBM3 parts."""
    # One-line reference labels at the right end, in a strip right of the
    # bars, each on the side of its line with room before the next gridline
    # (~51 px above 34 and 136, ~66 px below 819). The floor stays at 5: from
    # 1, a decade is only 236 px tall and those gaps shrink below a label.
    lp_max = m.WorkloadParams().max_lpddr5x
    refs = (
        (m.LPDDR5X_GBYTES_S, "1 LPDDR5X", "above"),
        (lp_max * m.LPDDR5X_GBYTES_S, f"{lp_max} LPDDR5X", "above"),
        (m.HBM3_GBYTES_S, "1 HBM3", "below"),
    )
    texts = [f"{name} ({value:.0f} GB/s)" for value, name, _ in refs]
    fig, ax = _chart(bottom=GROUPED_BOTTOM)
    recs = _evaluate_rates(via_area_budget_mm2=BUDGET_MM2)
    keys = list(recs)
    x = _grouped_x(fig, (ax,), keys)
    vals = [recs[k]["mem_gbytes_s"] for k in keys]
    width = 0.8
    ax.bar(x, vals, width, color=SERIES[0])
    x_max = _label_room(ax, x[0] - 0.6, x[-1] + width / 2, texts)
    ax.set_xlim(x[0] - 0.6, x_max)
    _log_y(ax, 5, 1000)
    for (value, _, side), text in zip(refs, texts):
        _ref_line(ax, value, text, x_max, side=side, zorder=0.9)  # under bars
    # values inside the bar tops (every bar is >= 80 px tall), so none sits
    # between a gridline and a reference line
    for xi, v in zip(x, vals):
        ax.annotate(
            f"{v:.3g}",
            (xi, v),
            xytext=(0, -8 * PX),
            textcoords="offset points",
            ha="center",
            va="top",
            color="white",
        )
    _y_title(ax, "Memory bandwidth (GB/s)")
    _setting_note(fig)
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
    """Greedy wrap to width_in; two-line results are balanced, without orphans."""
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
        width = lambda s: max(map(measure, s))  # noqa: E731
        # a last line of one short word ("die") reads as a leftover
        good = [
            s for s in splits if (" " in s[1] or len(s[1]) > 3) and width(s) <= width_in
        ]
        lines = list(min(good or splits, key=width))
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
    left, right, top, bottom = 0.11, 0.06, 0.04, 0.05  # figure margins (in)
    head_h, head_pref, gap = 0.68, 1.39, 0.045
    spare = width - left - right - head_pref - len(columns) * (inner + gap)
    pad = min(0.12, max(0.04, spare / (2 * len(columns))))
    col_w = inner + 2 * pad + gap
    head_w = width - left - right - len(columns) * col_w
    row_h = (height - top - bottom - head_h) / len(rows)
    x0, y0 = width - right - len(columns) * col_w, height - top - head_h
    # each cell line sits on a row-wide baseline, so the status words line up;
    # a probe line gives the font's ascent and descent as matplotlib boxes it
    probe = ax.text(0, 0, "lp", va="baseline")
    box = probe.get_window_extent(renderer).transformed(ax.transData.inverted())
    probe.remove()
    ascent, descent = box.y1, -box.y0
    pitch = 1.15 * (ascent + descent)
    for j, name in enumerate(columns):
        ax.text(x0 + (j + 0.5) * col_w, y0 + head_h / 2, name, ha="center", va="center")
    # the A 10 GHz verdicts hold only with this budget (TDM x2 at 200 mm2);
    # the corner fits two short lines, so "raised to" (said on figures 7-8)
    # is left out here
    note = f"Via budget\n{BUDGET_MM2:g} mm²"
    ax.text(left, y0 + head_h / 2, note, va="center", color=INK2, linespacing=1.15)
    for i, (row, cells) in enumerate(rows.items()):
        y_mid = y0 - (i + 0.5) * row_h
        header = _wrap(row, head_w - 0.04, measure)
        ax.text(left, y_mid, header, va="center", linespacing=1.15)
        lines = [
            [status, *_wrap(reason, col_w - gap - 0.08, measure).split("\n")]
            for status, reason in cells
        ]
        # top-anchored blocks: the tallest is centred, the rest share its top
        n = max(map(len, lines))
        y_base = y_mid + ((n - 1) * pitch + ascent + descent) / 2 - ascent
        for j, ((status, _), cell) in enumerate(zip(cells, lines)):
            face, ink = STATUS[status]
            corner = (x0 + j * col_w + gap / 2, y_mid - (row_h - gap) / 2)
            box = "round,pad=0,rounding_size=0.08"
            ax.add_patch(
                FancyBboxPatch(corner, col_w - gap, row_h - gap, box, fc=face, lw=0)
            )
            for k, line in enumerate(cell):
                xy = (x0 + (j + 0.5) * col_w, y_base - k * pitch)
                ax.text(*xy, line, ha="center", va="baseline", color=ink)
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
