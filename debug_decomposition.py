"""Decompose JTC output into autocorrelation and cross-correlation components."""

import torch
import torch.nn.functional as F
import numpy as np

# Test configuration
M, N = 8, 3
plane_size = 32
sep = 7

torch.manual_seed(42)
signal = torch.randn(M) * 0.1
kernel = torch.randn(N) * 0.1

print("="*80)
print("JPS DECOMPOSITION: |S+K|^2 = |S|^2 + |K|^2 + S*conj(K) + conj(S)*K")
print("="*80)

# Build input plane
input_plane = torch.zeros(plane_size, dtype=torch.complex64)
kernel_start = 0
signal_start = kernel_start + N + sep

input_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
input_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

# Roll to center
roll_amount = (plane_size // 2) - (M + signal_start) // 2
input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

print(f"\n1. Combined input (after roll):")
print(f"   Nonzero regions: kernel and signal")

# FFT -> fftshift
combined_fft = torch.fft.fft(input_plane)
combined_fft_shifted = torch.fft.fftshift(combined_fft)

# Now compute components separately
# Signal only
signal_plane = torch.zeros(plane_size, dtype=torch.complex64)
signal_plane[signal_start:signal_start+M] = signal.to(torch.complex64)
signal_plane = torch.roll(signal_plane, shifts=roll_amount, dims=-1)
signal_fft = torch.fft.fft(signal_plane)
signal_fft_shifted = torch.fft.fftshift(signal_fft)

# Kernel only
kernel_plane = torch.zeros(plane_size, dtype=torch.complex64)
kernel_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
kernel_plane = torch.roll(kernel_plane, shifts=roll_amount, dims=-1)
kernel_fft = torch.fft.fft(kernel_plane)
kernel_fft_shifted = torch.fft.fftshift(kernel_fft)

# Compute JPS components
jps_signal = torch.abs(signal_fft_shifted) ** 2
jps_kernel = torch.abs(kernel_fft_shifted) ** 2
jps_cross = torch.conj(signal_fft_shifted) * kernel_fft_shifted + signal_fft_shifted * torch.conj(kernel_fft_shifted)
jps_full = torch.abs(combined_fft_shifted) ** 2

# Verify decomposition
jps_reconstructed = jps_signal + jps_kernel + jps_cross
error = torch.abs(jps_full - jps_reconstructed).max()
print(f"\n2. JPS decomposition verification:")
print(f"   |S+K|^2 vs |S|^2 + |K|^2 + 2*Re(S*conj(K))")
print(f"   Max error: {error:.6e}")

# Now IFFT each component
auto_signal_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_signal / plane_size)))
auto_kernel_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_kernel / plane_size)))
cross_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_cross / plane_size)))
full_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_full / plane_size)))

print(f"\n3. Output plane components:")
print(f"   autocorr(signal) max: {auto_signal_plane.max():.6f} at {auto_signal_plane.argmax()}")
print(f"   autocorr(kernel) max: {auto_kernel_plane.max():.6f} at {auto_kernel_plane.argmax()}")
print(f"   cross-corr       max: {cross_plane.max():.6f} at {cross_plane.argmax()}")
print(f"   full output      max: {full_plane.max():.6f} at {full_plane.argmax()}")

# Extract at cross-correlation indices
same_start = plane_size // 2 + sep + N // 2 + 1
indices = torch.arange(same_start, same_start + (M+N-1)) % plane_size

print(f"\n4. Values at extraction indices [{same_start}, {same_start + M + N - 2}]:")
print(f"   Indices: {indices.tolist()}")
print(f"   autocorr(signal): {auto_signal_plane[indices]}")
print(f"   autocorr(kernel): {auto_kernel_plane[indices]}")
print(f"   cross-corr:       {cross_plane[indices]}")
print(f"   FULL (sum):       {full_plane[indices]}")

# Compute contamination ratio
total_auto = auto_signal_plane[indices] + auto_kernel_plane[indices]
contamination_ratio = total_auto / (cross_plane[indices] + 1e-10)

print(f"\n5. Contamination analysis:")
print(f"   autocorr/cross ratio: {contamination_ratio}")
print(f"   Min ratio: {contamination_ratio.min():.4f}")
print(f"   Max ratio: {contamination_ratio.max():.4f}")
print(f"   Mean ratio: {contamination_ratio.mean():.4f}")

# What percentage of the extracted signal is autocorrelation?
auto_percentage = 100 * total_auto / (full_plane[indices] + 1e-10)
print(f"\n6. Autocorrelation percentage of extracted signal:")
print(f"   {auto_percentage}")
print(f"   Min: {auto_percentage.min():.1f}%")
print(f"   Max: {auto_percentage.max():.1f}%")
print(f"   Mean: {auto_percentage.mean():.1f}%")

# Reference correlation for comparison
signal_conv = signal.unsqueeze(0).unsqueeze(0)
kernel_flipped = torch.flip(kernel, [0])
kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)
ref_corr = F.conv1d(signal_conv, kernel_conv, padding=N-1).squeeze()

print(f"\n7. Comparison with reference:")
print(f"   Reference corr:  {ref_corr}")
print(f"   Ref (abs):       {torch.abs(ref_corr)}")
print(f"   Cross-corr out:  {cross_plane[indices]}")
print(f"   Full JTC out:    {full_plane[indices]}")

# Does cross_plane match abs(ref_corr)?
if cross_plane[indices].std() > 1e-6:
    corr_cross = torch.corrcoef(torch.stack([
        torch.abs(ref_corr) / torch.abs(ref_corr).std(),
        cross_plane[indices] / cross_plane[indices].std()
    ]))[0, 1].item()
    print(f"\n   Correlation(|ref|, cross_component): {corr_cross:.4f}")

    corr_full = torch.corrcoef(torch.stack([
        torch.abs(ref_corr) / torch.abs(ref_corr).std(),
        full_plane[indices] / full_plane[indices].std()
    ]))[0, 1].item()
    print(f"   Correlation(|ref|, full_JTC): {corr_full:.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
if contamination_ratio.mean() < 0.1:
    print("✓ Autocorrelation contamination is LOW (< 10% on average)")
    print("  The simple formula M+N+sep <= plane_size appears sufficient")
elif contamination_ratio.mean() < 0.5:
    print("⚠ Autocorrelation contamination is MODERATE (10-50% on average)")
    print("  The outputs are usable but not ideal - consider larger separation")
else:
    print("✗ Autocorrelation contamination is HIGH (> 50% on average)")
    print("  The simple formula is INSUFFICIENT - revert to old formula")
print("="*80)
