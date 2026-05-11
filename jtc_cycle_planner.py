import argparse
import csv
import math
import sys
from collections.abc import Iterable, Sequence

HEIGHT = 32
WIDTH = 32
KERNEL_HEIGHT = 3


def compute_contamination_profile(
    input_len: int, kernel_len: int, lens_size: int, sep: int
) -> tuple[int, int, int]:
    """Compute contamination profile for JTC configuration.

    Physics: JTC output contains autocorrelation terms plus two mirrored
    cross-correlation lobes. A valid output is clean only if the extracted lag
    is in the desired cross-correlation support and is not overlapped by either
    autocorrelation or the mirrored cross-correlation support.

    Returns:
        total_outputs: Total M+N-1 correlation outputs
        clean_valid_outputs: Number of clean valid convolution outputs (subset of M-N+1)
        effective_stride: Contiguous clean prefix usable for tile stitching
    """
    M, N = input_len, kernel_len

    if M <= 0 or N <= 0 or lens_size <= 0:
        return 0, 0, 0
    if M < N:
        return 0, 0, 0
    if sep < 0:
        return 0, 0, 0
    if M + N + sep > lens_size:
        return 0, 0, 0

    total_outputs = M + N - 1

    def lag_range(start: int, stop: int) -> set[int]:
        width = stop - start
        if width <= 0:
            return set()
        if width >= lens_size:
            return set(range(lens_size))
        return {lag % lens_size for lag in range(start, stop)}

    # Autocorrelation support is centered at zero lag before fftshift. We work
    # in lag coordinates modulo L, so zero lag is 0 rather than L//2.
    autocorr = lag_range(-(M - 1), M) | lag_range(-(N - 1), N)

    # The signal starts D samples after the kernel start in the input plane.
    # Desired cross support is signal-position minus kernel-position. The JTC
    # also produces the mirrored lobe at the negative of those lags.
    d = N + sep
    desired_cross = lag_range(d - (N - 1), d + M)
    mirrored_cross = lag_range(-(d + M - 1), -(d - (N - 1)) + 1)

    clean_flags: list[bool] = []
    valid_start = 0 if N == 1 else (N + 1) // 2
    valid_end = valid_start + (M - N + 1)
    same_length_shift = 1 if N == 1 else 0
    for i in range(valid_start, valid_end):
        # This matches JTC._build_correlation_start() with the fftshift center
        # removed: physical_index = L//2 + sep + N//2 + i.
        lag = (sep + N // 2 + same_length_shift + i) % lens_size
        clean_flags.append(
            lag in desired_cross
            and lag not in autocorr
            and lag not in mirrored_cross
        )

    clean_valid_count = sum(clean_flags)
    effective_stride = 0
    for is_clean in clean_flags:
        if not is_clean:
            break
        effective_stride += 1

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
) -> tuple[int, int, int, int] | None:
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
    total_outputs, _, effective_stride = compute_contamination_profile(
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
    separations: Iterable[int] | None,
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
                    if (
                        best_cycles is None
                        or cycles < best_cycles
                        or (
                            cycles == best_cycles
                            and effective_stride
                            > (best_row or {}).get("effective_stride", 0)
                        )
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


def _expand_lengths(
    values: Sequence[int] | None, minimum: int, maximum: int
) -> list[int]:
    if values:
        return sorted(set(int(v) for v in values if v >= minimum and v <= maximum))
    return list(range(minimum, maximum + 1))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep JTC lens parameters")
    parser.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=list(range(3, 33)),
        help="Candidate input lengths (defaults to 3..32)",
    )
    parser.add_argument(
        "--kernel-lengths",
        type=int,
        nargs="+",
        default=[3],
        help="Candidate kernel lengths (defaults to 3)",
    )
    parser.add_argument(
        "--lens-sizes", type=int, nargs="+", default=list(range(32, 65))
    )
    parser.add_argument(
        "--separations",
        type=int,
        nargs="+",
        default=None,
        help="Candidate separations (defaults to 0..max feasible for each lens)",
    )
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
