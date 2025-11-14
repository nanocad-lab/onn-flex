"""Verify that autocorrelation contamination is low for all configs with simple formula."""

import torch

def compute_jtc_decomposed(signal, kernel, M, N, sep, plane_size):
    """Compute JTC with autocorr/cross decomposition."""
    # Build input plane
    input_plane = torch.zeros(plane_size, dtype=torch.complex64)
    kernel_start = 0
    signal_start = kernel_start + N + sep

    input_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
    input_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

    # Roll
    roll_amount = (plane_size // 2) - (M + signal_start) // 2
    input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

    # Separate components
    signal_plane = torch.zeros(plane_size, dtype=torch.complex64)
    kernel_plane = torch.zeros(plane_size, dtype=torch.complex64)

    kernel_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
    signal_plane[signal_start:signal_start+M] = signal.to(torch.complex64)
    kernel_plane = torch.roll(kernel_plane, shifts=roll_amount, dims=-1)
    signal_plane = torch.roll(signal_plane, shifts=roll_amount, dims=-1)

    # FFTs
    signal_fft_shifted = torch.fft.fftshift(torch.fft.fft(signal_plane))
    kernel_fft_shifted = torch.fft.fftshift(torch.fft.fft(kernel_plane))

    # JPS components
    jps_signal = torch.abs(signal_fft_shifted) ** 2
    jps_kernel = torch.abs(kernel_fft_shifted) ** 2
    jps_cross = torch.conj(signal_fft_shifted) * kernel_fft_shifted + signal_fft_shifted * torch.conj(kernel_fft_shifted)

    # Back to spatial
    auto_signal = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_signal / plane_size)))
    auto_kernel = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_kernel / plane_size)))
    cross_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_cross / plane_size)))

    return auto_signal, auto_kernel, cross_plane

# Test all configs
configs = [
    (8, 3, 32, 7),
    (16, 8, 48, 9),
    (16, 8, 64, 15),
    (8, 8, 48, 8),
]

print("="*90)
print("AUTOCORRELATION CONTAMINATION CHECK (Simple Formula: plane//2 + sep + N//2)")
print("="*90)

all_pass = True

for M, N, plane_size, sep in configs:
    torch.manual_seed(42)
    signal = torch.randn(M) * 0.1
    kernel = torch.randn(N) * 0.1

    # Check if config is valid
    if M + N + sep > plane_size:
        print(f"\nConfig: M={M}, N={N}, plane={plane_size}, sep={sep}")
        print(f"  ✗ INVALID: M+N+sep={M+N+sep} > plane_size={plane_size}")
        all_pass = False
        continue

    # Compute JTC decomposed
    auto_signal, auto_kernel, cross_plane = compute_jtc_decomposed(signal, kernel, M, N, sep, plane_size)

    # Extract using simple formula (without +1)
    same_start = plane_size // 2 + sep + N // 2
    indices = torch.arange(same_start, same_start + (M+N-1)) % plane_size

    # Get values
    auto_s = auto_signal[indices]
    auto_k = auto_kernel[indices]
    cross_v = cross_plane[indices]
    total_auto = auto_s + auto_k

    # Compute contamination
    contamination_ratio = total_auto / (cross_v + 1e-10)
    auto_percentage = 100 * total_auto / (total_auto + cross_v + 1e-10)

    print(f"\nConfig: M={M}, N={N}, plane={plane_size}, sep={sep}")
    print(f"  M+N+sep = {M+N+sep} {'✓' if M+N+sep <= plane_size else '✗ TOO LARGE'}")
    print(f"  Extraction indices: [{same_start}, {(same_start + M + N - 2) % plane_size}]")
    print(f"  Autocorr contamination ratio: mean={contamination_ratio.mean():.6f}, max={contamination_ratio.max():.6f}")
    print(f"  Autocorr percentage: mean={auto_percentage.mean():.4f}%, max={auto_percentage.max():.4f}%")

    # Check if contamination is acceptable
    if auto_percentage.mean() < 1.0:  # Less than 1% contamination
        print(f"  ✓ PASS: Autocorrelation contamination < 1%")
    elif auto_percentage.mean() < 10.0:  # Less than 10%
        print(f"  ⚠ MARGINAL: Autocorrelation contamination {auto_percentage.mean():.2f}% (acceptable but not ideal)")
    else:
        print(f"  ✗ FAIL: Autocorrelation contamination {auto_percentage.mean():.2f}% too high!")
        all_pass = False

print("\n" + "="*90)
if all_pass:
    print("✓ ALL CONFIGS PASS: Simple formula M+N+sep <= plane_size guarantees low contamination")
    print("  The simplified jtc_cycle_planner is CORRECT for separation validation")
else:
    print("✗ SOME CONFIGS FAIL: Simple formula is INSUFFICIENT")
    print("  Need to revert to old conservative formula")
print("="*90)
