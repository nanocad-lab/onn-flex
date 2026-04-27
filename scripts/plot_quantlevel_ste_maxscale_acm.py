from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure repository root is on sys.path when run directly
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

DEFAULT_FIG_WIDTH_IN = 6.5
DEFAULT_FIG_HEIGHT_IN = 4.0


def set_fig_theme() -> None:
    sns.set_theme(context="paper", style="whitegrid", font="serif")
    plt.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            # Axes
            "axes.linewidth": 0.5,
            "axes.grid": True,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.5,
            # Lines & markers
            "lines.linewidth": 0.8,
            "lines.markersize": 3,
            # Colors (black + grayscale friendly)
            "axes.prop_cycle": plt.cycler(color=["k", "0.3", "0.6", "0.9"]),
            # Savefig
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def _load_best_per_bits(csv_path: Path, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in [
        "bits",
        "returncode",
        "best_test_acc",
        "final_test_acc",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "returncode" in df.columns:
        df = df[df["returncode"] == 0]

    df = df.dropna(subset=["bits"])
    df["bits"] = df["bits"].astype(int)

    if metric_col not in df.columns:
        raise ValueError(
            f"Requested metric '{metric_col}' not found in CSV columns: {sorted(df.columns)}"
        )

    df = df.dropna(subset=[metric_col])
    df = df.sort_values(["bits", metric_col], ascending=[True, True])
    df_best = df.groupby("bits", as_index=False).tail(1).sort_values("bits")
    return df_best.reset_index(drop=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Plot quantlevel_ste_maxscale sweep results in ACM TODAES single-column format."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(repo_root / "sweep_results" / "quantlevel_ste_maxscale.csv"),
        help="Path to quantlevel_ste_maxscale.csv.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            repo_root / "sweep_results" / "quantlevel_ste_maxscale_acm_singlecol.pdf"
        ),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--width-in",
        type=float,
        default=DEFAULT_FIG_WIDTH_IN,
        help="Figure width (inches).",
    )
    parser.add_argument(
        "--height-in",
        type=float,
        default=DEFAULT_FIG_HEIGHT_IN,
        help="Figure height (inches).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="best_test_acc",
        choices=["best_test_acc", "final_test_acc"],
        help="Which test accuracy to plot (only one curve will be plotted).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo_root / csv_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    set_fig_theme()
    df = _load_best_per_bits(csv_path, metric_col=str(args.metric))
    df = df.sort_values("bits", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(float(args.width_in), float(args.height_in)))

    x = df["bits"].tolist()
    ax.plot(
        x,
        df[str(args.metric)],
        marker="o",
        linestyle="-",
        color="b",
        label="Test accuracy",
    )

    ax.set_xlabel("Fourier-plane Quantization (bits)")
    ax.set_ylabel("Accuracy (%)")

    bits_min = int(df["bits"].min())
    bits_max = int(df["bits"].max())
    tick_start = ((bits_min + 3) // 4) * 4  # ceil to multiple of 4
    tick_end = (bits_max // 4) * 4  # floor to multiple of 4
    if tick_start <= tick_end:
        ax.set_xticks(list(range(tick_start, tick_end + 1, 4)))
    else:
        ax.set_xticks(sorted(set(x)))
    ax.grid(True, which="major", axis="x")
    ax.invert_xaxis()  # high bits on the left
    #ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
