"""Shared plotting style configuration.

Use `apply_global_plot_style()` at module import time to ensure consistent
axis label, tick, and title font sizes across all figures.

Exports:
- DEFAULT_AXIS_FONTSIZE: int – base font size for axes labels and tick labels
- DEFAULT_TITLE_FONTSIZE: int – base font size for titles
- SHOW_TITLES: bool – master toggle to enable or disable titles
"""

from __future__ import annotations

from typing import Dict, Any


# Single global knobs for plot styling
DEFAULT_AXIS_FONTSIZE: int = 16
DEFAULT_TITLE_FONTSIZE: int = 18
SHOW_TITLES: bool = True


def apply_global_plot_style(
    axis_fontsize: int | None = None,
    title_fontsize: int | None = None,
) -> None:
    """Apply a consistent matplotlib style for axis labels, ticks, and titles.

    Args:
        axis_fontsize: Optional override for the axis/tick font size.
            If None, uses DEFAULT_AXIS_FONTSIZE.
        title_fontsize: Optional override for title font size.
            If None, uses DEFAULT_TITLE_FONTSIZE.
    """
    try:
        import matplotlib as _mpl  # Local import to avoid hard dependency at import time
    except Exception:
        return

    size: int = (
        int(axis_fontsize) if axis_fontsize is not None else DEFAULT_AXIS_FONTSIZE
    )
    tsize: int = (
        int(title_fontsize) if title_fontsize is not None else DEFAULT_TITLE_FONTSIZE
    )

    rc_updates: Dict[str, Any] = {
        # Axis labels (set_xlabel / set_ylabel)
        "axes.labelsize": size,
        # Tick labels
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        # Title size
        "axes.titlesize": tsize,
    }

    _mpl.rcParams.update(rc_updates)
