"""Verify the correct extraction formula for all test configurations."""

import torch
import torch.nn.functional as F

def compute_reference_corr(signal, kernel):
    """Compute reference using PyTorch conv1d."""
    signal_conv = signal.unsqueeze(0).unsqueeze(0)
    kernel_flipped = torch.flip(kernel, [0])
    kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)
    return F.conv1d(signal_conv, kernel_conv, padding=len(kernel)-1).squeeze()

def compute_jtc_cross_corr(signal, kernel, M, N, sep, plane_size):
    """Compute JTC cross-correlation component."""
    signal_plane = torch.zeros(plane_size, dtype=torch.complex64)
    kernel_plane = torch.zeros(plane_size, dtype=torch.complex64)

    kernel_start = 0
    signal_start = kernel_start + N + sep

    kernel_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
    signal_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

    roll_amount = (plane_size // 2) - (M + signal_start) // 2
    kernel_plane = torch.roll(kernel_plane, shifts=roll_amount, dims=-1)
    signal_plane = torch.roll(signal_plane, shifts=roll_amount, dims=-1)

    kernel_fft = torch.fft.fftshift(torch.fft.fft(kernel_plane))
    signal_fft = torch.fft.fftshift(torch.fft.fft(signal_plane))

    cross_freq = torch.conj(signal_fft) * kernel_fft + signal_fft * torch.conj(kernel_fft)
    cross_spatial = torch.abs(torch.fft.fftshift(torch.fft.fft(cross_freq / plane_size)))

    return cross_spatial

# Test configurations from test_variable_lengths.py
configs = [
    (8, 3, 32, 7),
    (16, 8, 48, 9),
    (16, 8, 64, 15),
    (8, 8, 48, 8),  # Golden code case
]

print("="*90)
print("EXTRACTION FORMULA VERIFICATION")
print("="*90)

for M, N, plane_size, sep in configs:
    torch.manual_seed(42)
    signal = torch.randn(M) * 0.1
    kernel = torch.randn(N) * 0.1

    ref_corr = compute_reference_corr(signal, kernel)
    cross_plane = compute_jtc_cross_corr(signal, kernel, M, N, sep, plane_size)

    # Test old formula (with +1)
    same_start_old = plane_size // 2 + sep + N // 2 + 1
    indices_old = torch.arange(same_start_old, same_start_old + (M+N-1)) % plane_size
    extracted_old = cross_plane[indices_old]

    # Test new formula (without +1)
    same_start_new = plane_size // 2 + sep + N // 2
    indices_new = torch.arange(same_start_new, same_start_new + (M+N-1)) % plane_size
    extracted_new = cross_plane[indices_new]

    # Find best match
    best_corr = -1
    best_start = None
    for start in range(plane_size):
        test_indices = torch.arange(start, start + (M+N-1)) % plane_size
        test_extracted = cross_plane[test_indices]
        if test_extracted.std() > 1e-6 and torch.abs(ref_corr).std() > 1e-6:
            corr = torch.corrcoef(torch.stack([
                torch.abs(ref_corr) / torch.abs(ref_corr).std(),
                test_extracted / test_extracted.std()
            ]))[0, 1].item()
            if corr > best_corr:
                best_corr = corr
                best_start = start

    # Compute correlations
    if extracted_old.std() > 1e-6:
        corr_old = torch.corrcoef(torch.stack([
            torch.abs(ref_corr) / torch.abs(ref_corr).std(),
            extracted_old / extracted_old.std()
        ]))[0, 1].item()
    else:
        corr_old = 0.0

    if extracted_new.std() > 1e-6:
        corr_new = torch.corrcoef(torch.stack([
            torch.abs(ref_corr) / torch.abs(ref_corr).std(),
            extracted_new / extracted_new.std()
        ]))[0, 1].item()
    else:
        corr_new = 0.0

    print(f"\nConfig: M={M}, N={N}, plane_size={plane_size}, sep={sep}")
    print(f"  Old formula (with +1): start={same_start_old}, correlation={corr_old:.4f}")
    print(f"  New formula (no +1):   start={same_start_new}, correlation={corr_new:.4f}")
    print(f"  Best possible:         start={best_start}, correlation={best_corr:.4f}")

    if corr_new > 0.95:
        print(f"  ✓ New formula works! (correlation > 0.95)")
    elif corr_old > 0.95:
        print(f"  ⚠ Old formula works for this config")
    else:
        print(f"  ✗ Neither formula works well - best is {best_start}")

print("\n" + "="*90)
print("CONCLUSION:")
print("If new formula (without +1) consistently gives correlation > 0.95,")
print("then we should remove the +1 from the extraction formula.")
print("="*90)
