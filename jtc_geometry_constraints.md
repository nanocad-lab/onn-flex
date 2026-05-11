# JTC Geometry Constraints

This note documents the geometry constraints for choosing JTC optical signal
length, kernel length, separation, and row-packing parameters. It is written for
the current `JTC` / `JTCConv2d` implementation, where a 1-D row correlation is
run inside a fixed-length optical field.

## Symbols

- `F`: total JTC optical field / lens size, `jtc_total_field`
- `M`: optical signal length for one JTC shot
- `N`: optical kernel length for one JTC shot
- `S`: separation between kernel and signal, `jtc_separation`
- `C`: autocorrelation center in the output plane, `F // 2`
- `W`: padded row width for a Conv2d row
- `R`: number of image rows packed into one JTC shot
- `G`: zero guard samples between packed rows

For a 3x3 Conv2d row correlation, `N = 3`. The 2-D kernel height is handled by
running three independent row correlations and summing them.

## Basic Feasibility

The input plane layout is:

```text
[ kernel length N ][ gap length S ][ signal length M ][ remaining zeros ]
```

The basic field-fit constraints are:

```text
M > 0
N > 0
M >= N                 # required for valid row convolution
S >= 0
M + N + S <= F
```

These conditions only prove that the input plane fits. They do not prove that
the extracted correlation samples are clean.

## Output Window

The full JTC correlation length is:

```text
M + N - 1
```

For standard full correlation, valid convolution corresponds to lags:

```text
D, D + 1, ..., D + M - N
where D = N + S
```

In the current `JTC.extract_correlation()` output coordinates, gather indices:

```text
valid_start, ..., valid_start + M - N
where valid_start = 0 if N == 1 else ceil(N / 2)
```

That gives:

```text
M - N + 1
```

valid row outputs per unstrided shot. For Conv2d stride `stride_w`, gather
columns:

```text
0, stride_w, 2 * stride_w, ...
```

from that valid window.

## Autocorrelation Avoidance

The planner now checks support overlap in lag space. This is the pseudocode for
the current rule:

```text
function lag_range(start, stop, F):
    if stop <= start:
        return {}
    if stop - start >= F:
        return {0, 1, ..., F - 1}
    return {k mod F for k in [start, stop)}

function contamination_profile(M, N, F, S):
    if M <= 0 or N <= 0 or F <= 0:
        return total=0, clean=0, stride=0
    if M < N or S < 0:
        return total=0, clean=0, stride=0
    if M + N + S > F:
        return total=0, clean=0, stride=0

    total = M + N - 1
    D = N + S

    autocorr =
        lag_range(-(M - 1), M, F)
        union lag_range(-(N - 1), N, F)

    desired_cross =
        lag_range(D - (N - 1), D + M, F)

    mirrored_cross =
        lag_range(-(D + M - 1), -(D - (N - 1)) + 1, F)

    clean_flags = []
    valid_start = 0 if N == 1 else ceil(N / 2)
    valid_end = valid_start + (M - N + 1)
    same_length_shift = 1 if N == 1 else 0
    for i in [valid_start, valid_end):
        lag = (S + floor(N / 2) + same_length_shift + i) mod F
        clean =
            lag in desired_cross
            and lag not in autocorr
            and lag not in mirrored_cross
        append clean to clean_flags

    clean_count = sum(clean_flags)
    effective_stride = count_contiguous_true_prefix(clean_flags)
    return total, clean_count, effective_stride
```

For each extracted JTC output index `i`, the physical output index is
equivalent to the lag:

```text
lag_i = (S + floor(N / 2) + same_length_shift + i) mod F
where same_length_shift = 1 if N == 1 else 0
```

The corresponding fft-shifted output-plane index is:

```text
idx_i = (F // 2 + lag_i) mod F
```

The autocorrelation support is:

```text
[-(M - 1), M - 1] union [-(N - 1), N - 1]  modulo F
```

The desired cross-correlation support is:

```text
[D - (N - 1), D + M - 1] modulo F
where D = N + S
```

The mirrored cross-correlation support is:

```text
[-(D + M - 1), -(D - (N - 1))] modulo F
```

A valid output is clean only if it is inside the desired support and outside
both undesired supports.

## Conservative 3-Wide Rule

For the current 3-wide row path, use:

```text
N = 3
S = M - N
M <= clean_input_limit(F, N)
```

For `F = 256, N = 3`, this gives:

```text
M <= 64
S = M - 3
```

The limit is not hard-coded. It is recalculated from `contamination_profile`
for the selected field and kernel width:

```text
function clean_input_limit(F, N):
    best = 0
    for M in [N, F]:
        S = M - N
        total, clean, stride = contamination_profile(M, N, F, S)
        valid = M - N + 1
        if clean == valid and stride == valid:
            best = M
    return best
```

For `F = 256`, the current planner gives:

```text
N = 1 -> M_clean_max = 64
N = 3 -> M_clean_max = 64
N = 5 -> M_clean_max = 65
N = 7 -> M_clean_max = 65
N = 9 -> M_clean_max = 66
```

Larger 3-wide values near `M = 86` can look clean in an autocorrelation-only
planner but are not clean because of the mirrored cross-correlation term.

Current examples for `F = 256, N = 3`:

```text
W = 34  -> M = 34, S = 31, clean, no row packing
W = 18  -> M can pack to 58, S = 55, clean
W = 10  -> M can pack to 58, S = 55, clean
W = 6   -> M can pack to 62, S = 59, clean
```

For other lens sizes or kernel widths, call `clean_input_limit(F, N)` instead
of reusing the 3-wide `64` result.

## Row Packing Rule

Packing means concatenating multiple padded image rows into one optical signal,
with zero guards between rows:

```text
[ row 0 length W ][ guard G ][ row 1 length W ][ guard G ] ...
```

Use:

```text
G = N - 1
```

For `R` packed rows, the packed signal length is:

```text
M_pack = R * W + (R - 1) * G
```

The conservative packed-row constraint is:

```text
M_pack <= M_clean_max
```

Solving for `R`:

```text
R_max = floor((M_clean_max + G) / (W + G))
```

Then cap by the number of output rows available:

```text
rows_per_shot = min(out_h, max(1, R_max))
```

For the current `F = 256, N = 3` implementation:

```text
M_clean_max = 64
G = 2
rows_per_shot = floor((64 + 2) / (W + 2))
```

Examples:

```text
W = 34: floor(66 / 36) = 1 row per shot
W = 18: floor(66 / 20) = 3 rows per shot
W = 10: floor(66 / 12) = 5 rows per shot
W = 6:  floor(66 / 8)  = 8 rows per shot
```

The current implementation pads the last row pack with zeros if `out_h` is not
divisible by `rows_per_shot`.

Pseudocode:

```text
function rows_per_shot(W, N, F, out_h):
    M_clean_max = clean_input_limit(F, N)
    G = N - 1
    R = floor((M_clean_max + G) / (W + G))
    return min(out_h, max(1, R))

function packed_geometry(W, N, F, out_h):
    R = rows_per_shot(W, N, F, out_h)
    G = N - 1
    M = R * W + (R - 1) * G
    S = M - N
    require M <= clean_input_limit(F, N)
    require M + N + S <= F
    return M, N, S, R
```

## Selection Procedure

For a 3x3 Conv2d row path:

1. Set `N = 3`.
2. Compute the padded row width `W = input_width + left_padding + right_padding`.
3. Set `M_clean_max = clean_input_limit(F, N)`.
4. If `W > M_clean_max`, do not use the current full-row rowwise path. Use
   tiling/stitching or a different planned geometry.
5. Choose row packing:

   ```text
   G = N - 1
   rows_per_shot = floor((M_clean_max + G) / (W + G))
   ```

6. For the actual packed shot:

   ```text
   M = rows_per_shot * W + (rows_per_shot - 1) * G
   S = M - N
   ```

7. Verify the backend with regression tests:
   - unquantized `jtc_ideal` vs stock PyTorch Conv2d
   - quantized clean `jtc_emulation` vs `jtc_ideal`

For candidate geometries outside this conservative rule, run a direct numerical
verification against PyTorch correlation. Do not rely only on the simple
autocorrelation planner near the lens capacity limit.
