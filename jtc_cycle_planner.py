import argparse
import csv
import math
import sys
from typing import Iterable, Optional, Sequence

#CIFAR10 size
HEIGHT = 32
WIDTH = 32
KERNEL_HEIGHT = 3

def usable_outputs(input_len: int, kernel_len: int, lens_size: int, sep: int) -> int:
    """Calculate number of usable correlation outputs for given JTC configuration.

    Based on golden code: with proper separation, ALL M+N-1 correlation outputs
    are overlap-free with autocorrelation terms.

    Args:
        input_len: Length of input signal (M)
        kernel_len: Length of kernel (N)
        lens_size: Total size of JTC plane
        sep: Separation between kernel and signal

    Returns:
        Number of usable outputs (M+N-1 if valid, 0 otherwise)
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

    # With proper separation, all correlation outputs are usable
    # Full correlation length is M + N - 1
    return input_len + kernel_len - 1


def cycles_for_config(
    input_len: int,
    kernel_len: int,
    lens_size: int,
    sep: int,
) -> Optional[tuple[int, int]]:
    usable = usable_outputs(input_len, kernel_len, lens_size, sep)
    if usable <= 0:
        return None
    out_w = WIDTH - kernel_len + 1
    out_h = HEIGHT - KERNEL_HEIGHT + 1
    if out_w <= 0 or out_h <= 0:
        return None
    passes_per_width = math.ceil(out_w / usable)
    total_cycles = passes_per_width * out_h * KERNEL_HEIGHT
    return passes_per_width, total_cycles


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
        "output_length",
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
                    usable = usable_outputs(input_len, kernel_len, lens, sep)
                    if usable <= 0:
                        continue
                    passes_cycles = cycles_for_config(input_len, kernel_len, lens, sep)
                    if passes_cycles is None:
                        continue
                    passes, cycles = passes_cycles
                    delta = sep + 0.5 * (input_len + kernel_len)
                    row = {
                        "input_length": input_len,
                        "kernel_length": kernel_len,
                        "lens_size": lens,
                        "separation": sep,
                        "delta": delta,
                        "output_length": usable,
                        "passes_per_width": passes,
                        "total_cycles": cycles,
                    }
                    if best_cycles is None or cycles < best_cycles or (
                        cycles == best_cycles and usable > (best_row or {}).get("usable_outputs", 0)
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
