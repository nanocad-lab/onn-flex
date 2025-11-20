import argparse
import csv
import math
import sys
from functools import lru_cache
from typing import Optional, Sequence

#CIFAR10 size
HEIGHT = 32
WIDTH = 32
KERNEL_HEIGHT = 3

# Contamination threshold for considering outputs "clean"
CONTAMINATION_THRESHOLD_PCT = 10.0  # 10% max autocorrelation contamination
DEFAULT_GEOMETRY_SEARCH_LIMIT = 64


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
    # Autocorrelation (DC term) is centered at frequency 0
    autocorr_center = 0
    autocorr_length = 2 * max(M, N) - 1
    # Autocorr region wraps around 0: [0, len//2] and [size-len//2, size]
    
    # Extraction start (formula without +1, as analysis shows)
    extraction_start = (sep + N // 2) % lens_size
    total_outputs = M + N - 1

    # For valid convolution stitching, we only use correlation outputs [N-1, M-1]
    # This gives M-N+1 valid outputs per patch
    valid_start_idx = N - 1
    valid_end_idx = M  # exclusive
    num_valid_outputs = M - N + 1
    if num_valid_outputs <= 0:
        return total_outputs, 0, 0

    # Check which valid outputs are clean (outside autocorr region)
    clean_valid_count = 0
    
    half_autocorr = autocorr_length // 2

    for i in range(valid_start_idx, valid_end_idx):
        idx = (extraction_start + i) % lens_size

        # Distance from autocorr center (0)
        dist_from_center = min(
            idx,
            lens_size - idx
        )

        # Clean if distance > half autocorr length
        if dist_from_center > half_autocorr:
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
        effective_stride = clean_valid_count

    if total_outputs > 0 and effective_stride <= 0:
        # Degenerate-but-valid config (e.g., zero separation). Allow stride=1 so
        # higher-level planners can still run even though performance will be poor.
        effective_stride = 1

    return total_outputs, clean_valid_count, effective_stride


@lru_cache(maxsize=1024)
def _find_clean_geometry_cached(
    input_len: int,
    kernel_len: int,
    max_lens_size: int,
) -> Optional[tuple[int, int, int, int]]:
    """Internal helper that performs the exhaustive geometry search."""

    min_lens = max(kernel_len + input_len, 1)
    best: Optional[tuple[int, int, int, int]] = None
    best_stride = -1

    for lens in range(min_lens, max_lens_size + 1):
        max_sep = lens - (input_len + kernel_len)
        if max_sep < 0:
            continue
        for sep in range(max_sep + 1):
            _, clean, stride = compute_contamination_profile(
                input_len, kernel_len, lens, sep
            )
            if clean <= 0:
                continue
            if stride > best_stride:
                best_stride = stride
                best = (sep, lens, clean, stride)
    return best


def find_clean_geometry(
    input_len: int,
    kernel_len: int,
    max_lens_size: Optional[int] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Search for a JTC geometry with non-zero clean outputs.

    Args:
        input_len: Signal length (M)
        kernel_len: Kernel length (N)
        max_lens_size: Optional cap for the lens size scan. When omitted,
            DEFAULT_GEOMETRY_SEARCH_LIMIT (currently 64) is used. Pass a larger
            value to permit bigger JTC planes or <=0 to skip searching.

    Returns:
        (sep, lens_size, clean_valid_outputs, effective_stride) for the
        configuration that yields the highest effective stride. If multiple
        configurations share the same stride, the first encountered (smallest
        lens / separation) is returned.
    """

    limit = DEFAULT_GEOMETRY_SEARCH_LIMIT if max_lens_size is None else max_lens_size
    if limit is None or limit <= 0:
        return None

    limit = int(limit)
    return _find_clean_geometry_cached(int(input_len), int(kernel_len), limit)


# Expose cache controls for tests/debuggers
find_clean_geometry.cache_clear = _find_clean_geometry_cached.cache_clear  # type: ignore[attr-defined]


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

    if total_outputs <= 0:
        return None

    effective_stride = max(1, effective_stride)

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


VGG_STAGE_CHANNELS = [64, 128, 256, 512, 512]
VGG_VARIANTS = {
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}
VARIANT_ALIASES = {f"ft{name}": name for name in VGG_VARIANTS}


def _normalize_variant(name: str) -> str:
    key = name.lower()
    if key in VGG_VARIANTS:
        return key
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]
    raise ValueError(f"Unsupported VGG variant '{name}'")


def _vgg_spatial_schedule(variant: str, input_size: int = WIDTH) -> list[int]:
    """Return the spatial dimension for each conv layer in the specified variant."""
    norm = _normalize_variant(variant)
    size = input_size
    schedule: list[int] = []
    for convs in VGG_VARIANTS[norm]:
        for _ in range(convs):
            schedule.append(size)
        size = max(1, size // 2)
    return schedule


def compute_vgg_forward_cycles(
    effective_stride: int,
    variant: str = "vgg11",
    kernel_height: int = KERNEL_HEIGHT,
) -> int:
    """Aggregate FTConv row cycles across every conv in the VGG stack."""
    if effective_stride <= 0:
        return 0
    total_cycles = 0
    for spatial in _vgg_spatial_schedule(variant):
        passes_per_width = math.ceil(spatial / effective_stride)
        total_cycles += passes_per_width * spatial * kernel_height
    return total_cycles


def plan_vgg_geometry(
    lens_size: int,
    *,
    variant: str = "vgg11",
    kernel_lengths: Sequence[int] | None = None,
    input_min: int = 3,
    input_max: int = WIDTH,
    separations: Optional[Sequence[int]] = None,
) -> dict[str, int]:
    """Find the geometry that minimizes total forward cycles for VGG on CIFAR-10."""
    norm_variant = _normalize_variant(variant)
    kernels = kernel_lengths or [KERNEL_HEIGHT]
    best: Optional[dict[str, int]] = None

    for kernel_len in kernels:
        if kernel_len <= 0 or kernel_len > WIDTH:
            continue
        min_input = max(kernel_len, input_min)
        max_input = min(input_max, WIDTH, lens_size - kernel_len)
        if max_input < min_input:
            continue
        for input_len in range(min_input, max_input + 1):
            max_sep = lens_size - (input_len + kernel_len)
            if max_sep < 0:
                continue
            if separations is None:
                sep_values: Sequence[int] = range(0, max_sep + 1)
            else:
                sep_values = [s for s in separations if 0 <= s <= max_sep]
                if not sep_values:
                    continue
            for sep in sep_values:
                total, clean, stride = compute_contamination_profile(
                    input_len, kernel_len, lens_size, sep
                )
                if total <= 0:
                    continue
                eff_stride = max(1, stride)
                cycles = compute_vgg_forward_cycles(eff_stride, variant=norm_variant)
                candidate = {
                    "lens_size": lens_size,
                    "variant": norm_variant,
                    "input_length": input_len,
                    "kernel_length": kernel_len,
                    "jtc_separation": sep,
                    "total_outputs": total,
                    "clean_outputs": clean,
                    "effective_stride": eff_stride,
                    "total_cycles": cycles,
                }
                if best is None:
                    best = candidate
                else:
                    if cycles < best["total_cycles"]:
                        best = candidate
                    elif cycles == best["total_cycles"]:
                        if eff_stride > best["effective_stride"]:
                            best = candidate
                        elif (
                            eff_stride == best["effective_stride"]
                            and input_len > best["input_length"]
                        ):
                            best = candidate
                        elif (
                            eff_stride == best["effective_stride"]
                            and input_len == best["input_length"]
                            and sep < best["jtc_separation"]
                        ):
                            best = candidate

    if best is None:
        raise ValueError(f"No valid geometry found for lens size {lens_size}")
    return best


def plan_vgg_for_lenses(
    lens_sizes: Sequence[int],
    *,
    variant: str = "vgg11",
    kernel_lengths: Sequence[int] | None = None,
    input_min: int = 3,
    input_max: int = WIDTH,
    separations: Optional[Sequence[int]] = None,
) -> list[dict[str, int]]:
    results = []
    for lens in lens_sizes:
        results.append(
            plan_vgg_geometry(
                lens,
                variant=variant,
                kernel_lengths=kernel_lengths,
                input_min=input_min,
                input_max=input_max,
                separations=separations,
            )
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan VGG forward cycles for CIFAR-10 given lens constraints"
    )
    parser.add_argument(
        "--lens-sizes",
        type=int,
        nargs="+",
        default=[32, 48, 64, 96],
        help="Lens sizes to evaluate",
    )
    parser.add_argument(
        "--kernel-lengths",
        type=int,
        nargs="+",
        default=[3],
        help="Candidate kernel widths (default: 3)",
    )
    parser.add_argument(
        "--input-min",
        type=int,
        default=3,
        help="Minimum input length to consider",
    )
    parser.add_argument(
        "--input-max",
        type=int,
        default=WIDTH,
        help="Maximum input length to consider",
    )
    parser.add_argument(
        "--separations",
        type=int,
        nargs="+",
        default=None,
        help="Candidate separations (default: all feasible)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vgg11",
        choices=sorted(set(VGG_VARIANTS) | set(VARIANT_ALIASES)),
        help="VGG variant to target (vgg11/13/16/19 or ft-prefixed names)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output CSV path or '-' for stdout",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = plan_vgg_for_lenses(
        args.lens_sizes,
        variant=args.variant,
        kernel_lengths=args.kernel_lengths,
        input_min=args.input_min,
        input_max=args.input_max,
        separations=args.separations,
    )
    fieldnames = [
        "lens_size",
        "variant",
        "input_length",
        "kernel_length",
        "jtc_separation",
        "effective_stride",
        "total_outputs",
        "clean_outputs",
        "total_cycles",
    ]
    if args.output == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
