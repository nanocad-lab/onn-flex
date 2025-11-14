import argparse
import csv
import math
import sys
from typing import Iterable, Optional, Sequence

#CIFAR10 size
HEIGHT = 32
WIDTH = 32
KERNEL_HEIGHT = 3

# Contamination threshold for considering outputs "clean"
CONTAMINATION_THRESHOLD_PCT = 10.0  # 10% max autocorrelation contamination


def compute_contamination_profile(input_len: int, kernel_len: int, lens_size: int, sep: int) -> tuple[int, int, int]:
    """Compute contamination profile for JTC configuration.

    Physics: JTC output = autocorr(signal) + autocorr(kernel) + cross-correlation
    - Autocorr region centered at lens_size//2, length 2*max(M,N)-1
    - Cross-corr extracted starting at lens_size//2 + sep + N//2, length M+N-1
    - For valid convolution stitching: use correlation indices [N-1, M-1] → M-N+1 outputs
    - Contamination = (autocorr_at_index) / (total_at_index)

    Returns:
        total_outputs: Total M+N-1 correlation outputs
        clean_valid_outputs: Number of clean valid convolution outputs (subset of M-N+1)
        effective_stride: Stride for tile stitching (clean valid outputs per pass)
    """
    M, N = input_len, kernel_len

    if input_len + kernel_len + sep > lens_size:
        return 0, 0, 0

    # Physics-based analysis
    autocorr_center = lens_size // 2
    autocorr_length = 2 * max(M, N) - 1
    autocorr_start = autocorr_center - (autocorr_length // 2)
    autocorr_end = autocorr_start + autocorr_length

    # Extraction start (formula without +1, as analysis shows)
    extraction_start = lens_size // 2 + sep + N // 2
    total_outputs = M + N - 1

    # For valid convolution stitching, we only use correlation outputs [N-1, M-1]
    # This gives M-N+1 valid outputs per patch
    valid_start_idx = N - 1
    valid_end_idx = M  # exclusive
    num_valid_outputs = M - N + 1

    # Check which valid outputs are clean (outside autocorr region)
    clean_valid_count = 0

    for i in range(valid_start_idx, valid_end_idx):
        idx = (extraction_start + i) % lens_size

        # Distance from autocorr center
        dist_from_center = min(
            abs(idx - autocorr_center),
            abs(idx - autocorr_center + lens_size),
            abs(idx - autocorr_center - lens_size)
        )

        # Clean if distance > half autocorr length
        if dist_from_center > autocorr_length // 2:
            clean_valid_count += 1

    # Effective stride for stitching valid convolution:
    # Use conservative estimate to avoid autocorr edge effects
    # Empirically, the last valid output often has some contamination even when
    # geometric analysis suggests it's clean

    if clean_valid_count == num_valid_outputs:
        # All valid outputs appear clean - use M-N+1 but be conservative for small configs
        if num_valid_outputs <= 6:
            effective_stride = max(num_valid_outputs - 1, 1)  # Conservative: exclude last output
        else:
            effective_stride = num_valid_outputs  # Large enough to trust
    else:
        # Some contamination detected - use clean count
        effective_stride = clean_valid_count if clean_valid_count > 0 else 0

    return total_outputs, clean_valid_count, effective_stride


def usable_outputs(input_len: int, kernel_len: int, lens_size: int, sep: int) -> int:
    """Calculate number of usable correlation outputs for given JTC configuration.

    This returns the TOTAL number of outputs (M+N-1), which includes both
    clean and contaminated outputs. For cycle planning with stitching,
    use compute_contamination_profile() to get the effective stride.

    Args:
        input_len: Length of input signal (M)
        kernel_len: Length of kernel (N)
        lens_size: Total size of JTC plane
        sep: Separation between kernel and signal

    Returns:
        Number of total outputs (M+N-1 if valid, 0 otherwise)
    """
    # Validity checks
    if input_len <= 0 or kernel_len <= 0 or lens_size <= 0:
        return 0
    if input_len < kernel_len:
        return 0
    if sep < 0:
        return 0

    # Check if configuration fits in lens plane
    # Need space for: kernel (N) + separation (sep) + signal (M)
    if input_len + kernel_len + sep > lens_size:
        return 0

    # Total correlation length is M + N - 1
    # Note: Some outputs may have autocorr contamination
    return input_len + kernel_len - 1


def cycles_for_config(
    input_len: int,
    kernel_len: int,
    lens_size: int,
    sep: int,
) -> Optional[tuple[int, int, int, int]]:
    """Compute cycles needed for image convolution with tile stitching.

    For a 32x32 image with 3x3 kernel:
    - Output dimensions: (32-3+1) x (32-3+1) = 30x30
    - Each JTC pass produces M+N-1 outputs, but some may be contaminated
    - We use effective_stride (clean outputs) for stitching adjacent tiles
    - Each row requires ceil(out_w / effective_stride) passes
    - Total cycles = passes_per_width * out_h * kernel_height

    Returns:
        (passes_per_width, total_cycles, effective_stride, total_outputs)
        or None if configuration is invalid
    """
    total_outputs, clean_outputs, effective_stride = compute_contamination_profile(
        input_len, kernel_len, lens_size, sep
    )

    if total_outputs <= 0 or effective_stride <= 0:
        return None

    # Output dimensions for "same" convolution
    out_w = WIDTH - kernel_len + 1
    out_h = HEIGHT - KERNEL_HEIGHT + 1

    if out_w <= 0 or out_h <= 0:
        return None

    # Number of passes needed per row, using effective stride for stitching
    # Each pass produces `effective_stride` usable outputs for stitching
    passes_per_width = math.ceil(out_w / effective_stride)

    # Total cycles: passes_per_width * num_rows * kernel_height
    total_cycles = passes_per_width * out_h * KERNEL_HEIGHT

    return passes_per_width, total_cycles, effective_stride, total_outputs


def sweep(
    input_lengths: Iterable[int],
    kernel_lengths: Iterable[int],
    lens_sizes: Iterable[int],
    separations: Optional[Iterable[int]],
    output_path: str,
) -> None:
    fieldnames = [
        "input_length",
        "kernel_length",
        "lens_size",
        "separation",
        "delta",
        "total_outputs",
        "effective_stride",
        "passes_per_width",
        "total_cycles",
    ]
    rows = []
    for lens in lens_sizes:
        best_row = None
        best_cycles = None
        for kernel_len in kernel_lengths:
            if kernel_len <= 0:
                continue
            for input_len in input_lengths:
                if input_len < kernel_len or input_len <= 0:
                    continue
                output_len = input_len - kernel_len + 1
                max_sep = lens - (input_len + kernel_len)
                if max_sep < 0:
                    continue
                if separations is None:
                    sep_iter: Iterable[int] = range(0, max_sep + 1)
                else:
                    sep_iter = [s for s in separations if 0 <= s <= max_sep]
                for sep in sep_iter:
                    passes_cycles = cycles_for_config(input_len, kernel_len, lens, sep)
                    if passes_cycles is None:
                        continue
                    passes, cycles, effective_stride, total_outputs = passes_cycles
                    delta = sep + 0.5 * (input_len + kernel_len)
                    row = {
                        "input_length": input_len,
                        "kernel_length": kernel_len,
                        "lens_size": lens,
                        "separation": sep,
                        "delta": delta,
                        "total_outputs": total_outputs,
                        "effective_stride": effective_stride,
                        "passes_per_width": passes,
                        "total_cycles": cycles,
                    }
                    if best_cycles is None or cycles < best_cycles or (
                        cycles == best_cycles and effective_stride > (best_row or {}).get("effective_stride", 0)
                    ):
                        best_cycles = cycles
                        best_row = row
        if best_row is None:
            raise ValueError(f"No valid configuration found for lens size {lens}")
        rows.append(best_row)
    if output_path == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _expand_lengths(values: Optional[Sequence[int]], minimum: int, maximum: int) -> list[int]:
    if values:
        return sorted(set(int(v) for v in values if v >= minimum and v <= maximum))
    return list(range(minimum, maximum + 1))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep JTC lens parameters")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=list(range(3, 33)),
                        help="Candidate input lengths (defaults to 3..32)")
    parser.add_argument("--kernel-lengths", type=int, nargs="+", default=[3],
                        help="Candidate kernel lengths (defaults to 3)")
    parser.add_argument("--lens-sizes", type=int, nargs="+", default=list(range(32, 65)))
    parser.add_argument("--separations", type=int, nargs="+", default=None,
                        help="Candidate separations (defaults to 0..max feasible for each lens)")
    parser.add_argument("--output", type=str, default="jtc_cycles.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_lengths = _expand_lengths(args.input_lengths, 3, WIDTH)
    kernel_lengths = _expand_lengths(args.kernel_lengths, 3, WIDTH)
    sweep(input_lengths, kernel_lengths, args.lens_sizes, args.separations, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
