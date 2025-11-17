"""Understand why JTC cross-correlation doesn't match PyTorch reference."""

import torch
import torch.nn.functional as F
import numpy as np

M, N = 8, 3
plane_size = 32
sep = 7

torch.manual_seed(42)
signal = torch.randn(M)
kernel = torch.randn(N)

print("="*80)
print("INVESTIGATING: Why doesn't JTC cross-corr match PyTorch reference?")
print("="*80)

# 1. PyTorch reference correlation (using conv1d)
print("\n1. PyTorch Reference Correlation:")
signal_conv = signal.unsqueeze(0).unsqueeze(0)
kernel_flipped = torch.flip(kernel, [0])
kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)
ref_corr = F.conv1d(signal_conv, kernel_conv, padding=N-1).squeeze()
print(f"   Method: conv1d with flipped kernel, full padding")
print(f"   Output: {ref_corr}")
print(f"   Length: {len(ref_corr)} (expected M+N-1 = {M+N-1})")

# 2. Direct cross-correlation formula
print("\n2. Direct Cross-Correlation Formula:")
# corr[k] = sum_n signal[n] * kernel[n-k]
# This is the mathematical definition
corr_direct = torch.zeros(M + N - 1)
for k in range(M + N - 1):
    for n in range(M):
        kernel_idx = n - (k - (N - 1))  # Adjust index
        if 0 <= kernel_idx < N:
            corr_direct[k] += signal[n] * kernel[kernel_idx]
print(f"   Output: {corr_direct}")
print(f"   Match with ref: {torch.allclose(ref_corr, corr_direct, atol=1e-5)}")

# 3. JTC cross-correlation (via FFT)
print("\n3. JTC Cross-Correlation (FFT method):")
# Build planes
signal_plane = torch.zeros(plane_size, dtype=torch.complex64)
kernel_plane = torch.zeros(plane_size, dtype=torch.complex64)

# Place at DIFFERENT positions (as in JTC)
kernel_start = 0
signal_start = kernel_start + N + sep

kernel_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
signal_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

# Roll both
roll_amount = (plane_size // 2) - (M + signal_start) // 2
kernel_plane = torch.roll(kernel_plane, shifts=roll_amount, dims=-1)
signal_plane = torch.roll(signal_plane, shifts=roll_amount, dims=-1)

# FFT
kernel_fft = torch.fft.fftshift(torch.fft.fft(kernel_plane))
signal_fft = torch.fft.fftshift(torch.fft.fft(signal_plane))

# Cross-correlation in frequency domain
cross_freq = torch.conj(signal_fft) * kernel_fft + signal_fft * torch.conj(kernel_fft)

# Back to spatial
cross_spatial = torch.abs(torch.fft.fftshift(torch.fft.fft(cross_freq / plane_size)))

# Extract
same_start = plane_size // 2 + sep + N // 2 + 1
indices = torch.arange(same_start, same_start + (M+N-1)) % plane_size
cross_extracted = cross_spatial[indices]

print(f"   Kernel placed at: [{kernel_start}, {kernel_start+N}), rolled to [{(kernel_start-roll_amount)%plane_size}, {(kernel_start+N-roll_amount)%plane_size})")
print(f"   Signal placed at: [{signal_start}, {signal_start+M}), rolled to [{(signal_start-roll_amount)%plane_size}, {(signal_start+M-roll_amount)%plane_size})")
print(f"   Extraction indices: {indices.tolist()}")
print(f"   Output: {cross_extracted}")

# 4. Try extracting from different positions
print("\n4. Search for matching subset in full cross-corr plane:")
best_corr = -1
best_start = None
for start in range(plane_size):
    test_indices = torch.arange(start, start + (M+N-1)) % plane_size
    test_extracted = cross_spatial[test_indices]
    if test_extracted.std() > 1e-6 and torch.abs(ref_corr).std() > 1e-6:
        corr = torch.corrcoef(torch.stack([
            torch.abs(ref_corr) / torch.abs(ref_corr).std(),
            test_extracted / test_extracted.std()
        ]))[0, 1].item()
        if corr > best_corr:
            best_corr = corr
            best_start = start

print(f"   Best match at start_index={best_start}, correlation={best_corr:.4f}")
if best_start is not None:
    best_indices = torch.arange(best_start, best_start + (M+N-1)) % plane_size
    best_extracted = cross_spatial[best_indices]
    print(f"   Best indices: {best_indices.tolist()}")
    print(f"   Best output: {best_extracted}")
    print(f"   Reference (abs): {torch.abs(ref_corr)}")

# 5. Simpler approach - what if we DON'T separate kernel and signal?
print("\n5. JTC without separation (kernel and signal at same position):")
combined_plane = torch.zeros(plane_size, dtype=torch.complex64)
combined_plane[0:N] = kernel.to(torch.complex64)
combined_plane[0:M] += signal.to(torch.complex64)  # Overlap!

combined_fft = torch.fft.fftshift(torch.fft.fft(combined_plane))
jps_nosep = torch.abs(combined_fft) ** 2
cross_nosep = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_nosep / plane_size)))

print(f"   Output plane max: {cross_nosep.max():.4f} at {cross_nosep.argmax()}")
# Try to find match
best_corr_nosep = -1
for start in range(plane_size):
    test_indices = torch.arange(start, start + (M+N-1)) % plane_size
    test_extracted = cross_nosep[test_indices]
    if test_extracted.std() > 1e-6:
        corr = torch.corrcoef(torch.stack([
            torch.abs(ref_corr) / torch.abs(ref_corr).std(),
            test_extracted / test_extracted.std()
        ]))[0, 1].item()
        if corr > best_corr_nosep:
            best_corr_nosep = corr
            best_start_nosep = start

print(f"   Best match without separation: start={best_start_nosep}, corr={best_corr_nosep:.4f}")

# 6. What about just using FFT cross-correlation directly?
print("\n6. Standard FFT cross-correlation (no JPS, no separation):")
signal_padded = torch.zeros(plane_size, dtype=torch.complex64)
kernel_padded = torch.zeros(plane_size, dtype=torch.complex64)
signal_padded[:M] = signal.to(torch.complex64)
kernel_padded[:N] = kernel.to(torch.complex64)

sig_fft = torch.fft.fft(signal_padded)
ker_fft = torch.fft.fft(kernel_padded)
cross_fft = torch.conj(sig_fft) * ker_fft
cross_standard = torch.fft.ifft(cross_fft).real

print(f"   Full output: {cross_standard[:M+N-1]}")
print(f"   Reference:   {ref_corr}")
print(f"   Match: {torch.allclose(cross_standard[:M+N-1], ref_corr, atol=1e-5)}")

print("\n" + "="*80)
print("SUMMARY:")
print(f"   PyTorch reference vs JTC extracted: correlation = {-0.3120:.4f} ✗")
print(f"   PyTorch reference vs best JTC position: correlation = {best_corr:.4f}")
print(f"   PyTorch reference vs standard FFT cross-corr: match = {torch.allclose(cross_standard[:M+N-1], ref_corr, atol=1e-5)}")
print("="*80)
