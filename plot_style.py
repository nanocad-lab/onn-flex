"""Shared plotting style configuration.

Use `apply_global_plot_style()` at module import time to ensure consistent
axis label, tick, title, and legend font sizes across all figures, and to
optionally set the global font family.

Exports:
- DEFAULT_AXIS_LABEL_FONTSIZE: int – font size for axes labels (x/y labels)
- DEFAULT_TICK_LABEL_FONTSIZE: int – font size for tick labels
- DEFAULT_TITLE_FONTSIZE: int – font size for axes titles
- DEFAULT_LEGEND_FONTSIZE: int – font size for legend text
- DEFAULT_FONT_FAMILY: str – global Matplotlib font family (e.g. "sans-serif", "serif")
- SHOW_TITLES: bool – master toggle to enable or disable titles
"""

from __future__ import annotations

from typing import Dict, Any
import seaborn as sns

# Single global knobs for plot styling
DEFAULT_AXIS_LABEL_FONTSIZE: int = 16
DEFAULT_TICK_LABEL_FONTSIZE: int = 12
DEFAULT_TITLE_FONTSIZE: int = 18
DEFAULT_LEGEND_FONTSIZE: int = 12
DEFAULT_FONT_FAMILY: str = "sans-serif"
SHOW_TITLES: bool = False


def apply_global_plot_style(
    axis_label_fontsize: int | None = None,
    tick_label_fontsize: int | None = None,
    title_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    font_family: str | None = None,
) -> None:
    """Apply a consistent matplotlib style for axis labels, ticks, titles, and legend.

    Args:
        axis_label_fontsize: Optional override for the axis label font size.
            If None, uses DEFAULT_AXIS_LABEL_FONTSIZE.
        tick_label_fontsize: Optional override for tick label font size.
            If None, uses DEFAULT_TICK_LABEL_FONTSIZE.
        title_fontsize: Optional override for title font size.
            If None, uses DEFAULT_TITLE_FONTSIZE.
        legend_fontsize: Optional override for legend font size.
            If None, uses DEFAULT_LEGEND_FONTSIZE.
        font_family: Optional global font family. Can be a generic family
            (e.g., "sans-serif", "serif", "monospace") or a specific font
            name available on the system (e.g., "DejaVu Sans", "Times New Roman").
            If None, uses DEFAULT_FONT_FAMILY.
    """
    try:
        import matplotlib as _mpl  # Local import to avoid hard dependency at import time
    except Exception:
        return
    sns.set_theme(context="paper", style="whitegrid", font="serif")
    axis_size: int = (
        int(axis_label_fontsize)
        if axis_label_fontsize is not None
        else DEFAULT_AXIS_LABEL_FONTSIZE
    )
    tick_size: int = (
        int(tick_label_fontsize)
        if tick_label_fontsize is not None
        else DEFAULT_TICK_LABEL_FONTSIZE
    )
    title_size: int = (
        int(title_fontsize) if title_fontsize is not None else DEFAULT_TITLE_FONTSIZE
    )
    legend_size: int = (
        int(legend_fontsize) if legend_fontsize is not None else DEFAULT_LEGEND_FONTSIZE
    )
    family: str = font_family if font_family is not None else DEFAULT_FONT_FAMILY

    rc_updates: Dict[str, Any] = {
        # Global font family
        "font.family": family,
        # Axis labels (set_xlabel / set_ylabel)
        "axes.labelsize": axis_size,
        # Tick labels
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        # Title size
        "axes.titlesize": title_size,
        # Legend size
        "legend.fontsize": legend_size,
    }

    _mpl.rcParams.update(rc_updates)
