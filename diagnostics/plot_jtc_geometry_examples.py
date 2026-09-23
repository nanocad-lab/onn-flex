from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onn_shotplan import compute_contamination_profile


def _mask(field: int, start: int, stop: int) -> np.ndarray:
    out = np.zeros(field, dtype=bool)
    if stop <= start:
        return out
    if stop - start >= field:
        out[:] = True
        return out
    for lag in range(start, stop):
        out[lag % field] = True
    return out


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - start))
            start = None
    return segments


def _plot_mask(ax, mask: np.ndarray, y: float, color: str, label: str) -> None:
    for start, width in _segments(mask):
        ax.broken_barh([(start, width)], (y - 0.32, 0.64), facecolors=color)
    ax.text(-7, y, label, ha="right", va="center", fontsize=9)


def _case_masks(field: int, signal_len: int, kernel_len: int, sep: int):
    d = kernel_len + sep
    autocorr = _mask(field, -(signal_len - 1), signal_len)
    autocorr |= _mask(field, -(kernel_len - 1), kernel_len)
    desired = _mask(field, d - (kernel_len - 1), d + signal_len)
    mirror = _mask(field, -(d + signal_len - 1), -(d - (kernel_len - 1)) + 1)

    valid = np.zeros(field, dtype=bool)
    clean_valid = np.zeros(field, dtype=bool)
    dirty_valid = np.zeros(field, dtype=bool)
    valid_start = 0 if kernel_len == 1 else (kernel_len + 1) // 2
    same_length_shift = 1 if kernel_len == 1 else 0
    for idx in range(valid_start, valid_start + signal_len - kernel_len + 1):
        lag = (sep + kernel_len // 2 + same_length_shift + idx) % field
        valid[lag] = True
        if desired[lag] and not autocorr[lag] and not mirror[lag]:
            clean_valid[lag] = True
        else:
            dirty_valid[lag] = True
    return autocorr, desired, mirror, clean_valid, dirty_valid


def main() -> None:
    field = 256
    cases = [
        ("N=3, M=64, S=61: full valid window clean", 64, 3, 61),
        ("N=9, M=66, S=57: larger M but fewer valid outputs", 66, 9, 57),
        ("N=3, M=86, S=83: mirrored lobe contaminates valid window", 86, 3, 83),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(12, 7.8), sharex=True)

    for ax, (title, signal_len, kernel_len, sep) in zip(axes, cases, strict=True):
        total, clean, stride = compute_contamination_profile(
            signal_len, kernel_len, field, sep
        )
        autocorr, desired, mirror, clean_valid, dirty_valid = _case_masks(
            field, signal_len, kernel_len, sep
        )
        _plot_mask(ax, autocorr, 4, "#777777", "autocorr")
        _plot_mask(ax, desired, 3, "#4c78a8", "desired full")
        _plot_mask(ax, mirror, 2, "#f58518", "mirrored")
        _plot_mask(ax, clean_valid, 1, "#54a24b", "valid clean")
        _plot_mask(ax, dirty_valid, 0, "#e45756", "valid dirty")
        valid_count = signal_len - kernel_len + 1
        ax.set_title(
            f"{title} | valid={valid_count}, clean={clean}, stride={stride}, total={total}",
            loc="left",
            fontsize=10,
        )
        ax.set_ylim(-0.8, 4.8)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.25)

    axes[-1].set_xlabel("circular lag index modulo F=256")
    axes[-1].set_xlim(0, field)
    fig.tight_layout()
    out = Path("diagnostics/jtc_geometry_examples.png")
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
