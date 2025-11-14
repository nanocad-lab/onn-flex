"""Test the old complex formula vs current simple formula."""

import torch

def compute_old_formula_indices(M, N, plane_size, sep):
    """The old complex formula from _compute_valid_output_indices."""
    delta = sep + 0.5 * (M + N)
    conv_len = M + N - 1
    half_conv = 0.5 * (conv_len - 1)
    auto_right = max(M - 1, N - 1)
    lens_right = 0.5 * (plane_size - 1)

    # Find valid run
    start_j = None
    run_length = 0
    for j in range(N - 1, M):
        x = delta + (j - half_conv)
        if x <= auto_right:
            continue
        if x > lens_right:
            break
        if start_j is None:
            start_j = j
        run_length += 1

    if start_j is None:
        return None, None, 0

    # Compute indices
    base_center = plane_size // 2 + sep + N // 2
    start_idx = base_center + 1 - (conv_len // 2) + start_j
    indices = list(range(start_idx, start_idx + run_length))
    indices = [i % plane_size for i in indices]

    return indices, start_idx, run_length

# Test configs
configs = [
    (8, 3, 32, 7),
    (16, 8, 48, 9),
    (16, 8, 64, 15),
    (8, 8, 48, 8),
]

print("="*90)
print("COMPARING FORMULAS")
print("="*90)

for M, N, plane_size, sep in configs:
    print(f"\nConfig: M={M}, N={N}, plane={plane_size}, sep={sep}")

    # Old formula
    old_indices, old_start, old_len = compute_old_formula_indices(M, N, plane_size, sep)

    # Current formula (with +1)
    current_start = plane_size // 2 + sep + N // 2 + 1
    current_indices = [(current_start + i) % plane_size for i in range(M + N - 1)]

    # Proposed formula (no +1)
    proposed_start = plane_size // 2 + sep + N // 2
    proposed_indices = [(proposed_start + i) % plane_size for i in range(M + N - 1)]

    print(f"  Old complex:     start={old_start}, length={old_len}")
    if old_indices:
        print(f"                   indices={old_indices[:5]}{'...' if old_len > 5 else ''}")
    print(f"  Current (+1):    start={current_start}, length={M+N-1}")
    print(f"                   indices={current_indices[:5]}...")
    print(f"  Proposed (no+1): start={proposed_start}, length={M+N-1}")
    print(f"                   indices={proposed_indices[:5]}...")

print("\n" + "="*90)
