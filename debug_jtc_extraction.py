"""Debug script to understand JTC extraction vs reference correlation."""

import torch
import torch.nn.functional as F
import numpy as np
from onn_config import AppConfig
from onn_layers import FTconvlayer

# Test configuration: M=8, N=3 (golden code case from summary)
M, N = 8, 3
plane_size = 32
sep = 7

torch.manual_seed(42)
signal = torch.randn(M) * 0.1
kernel = torch.randn(N) * 0.1

print("="*80)
print(f"Configuration: M={M}, N={N}, plane_size={plane_size}, sep={sep}")
print(f"Expected full correlation length: {M+N-1}")
print("="*80)

# 1. Compute reference correlation using PyTorch
signal_conv = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
kernel_flipped = torch.flip(kernel, [0])
kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
padding = N - 1
ref_corr = F.conv1d(signal_conv, kernel_conv, padding=padding).squeeze()
print(f"\n1. Reference correlation (PyTorch):")
print(f"   Length: {len(ref_corr)}")
print(f"   Values: {ref_corr}")
print(f"   Range: [{ref_corr.min():.4f}, {ref_corr.max():.4f}]")

# 2. Compute JTC output using actual implementation
config = AppConfig(
    input_length=M,
    kernel_length=N,
    output_length=None,  # Auto: M+N-1
    jtc_separation=sep,
    jtc_total_field=plane_size,
    conv_backend="fourier",
    dac_bits=None,
    adc_bits=None,
    fourier_plane_bits=None,
    scale_output="none",
)

layer = FTconvlayer(
    in_channels=1,
    out_channels=1,
    config=config,
    kernel_size=M,
    batch_size=1,
)

# Set weight to our test kernel
with torch.no_grad():
    layer.weights.data = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

# Prepare input
signal_jtc = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, M]
kernel_jtc = kernel.unsqueeze(0)  # [1, N]

# Call fourier_conv_forward directly
jtc_output = layer.fourier_conv_forward(signal_jtc, kernel_jtc)
jtc_output = jtc_output.squeeze()

print(f"\n2. JTC output (actual implementation):")
print(f"   Length: {len(jtc_output)}")
print(f"   Values: {jtc_output}")
print(f"   Range: [{jtc_output.min():.4f}, {jtc_output.max():.4f}]")

# 3. Manually compute JTC to understand internals
input_plane = torch.zeros(plane_size, dtype=torch.complex64)
kernel_start = 0
kernel_end = kernel_start + N
signal_start = kernel_end + sep
signal_end = signal_start + M

input_plane[kernel_start:kernel_end] = kernel.to(torch.complex64)
input_plane[signal_start:signal_end] = signal.to(torch.complex64)

# Roll to center
roll_amount = (plane_size // 2) - (M + signal_start) // 2
input_plane_rolled = torch.roll(input_plane, shifts=roll_amount, dims=-1)

print(f"\n3. JTC internals:")
print(f"   Kernel at indices: [{kernel_start}, {kernel_end})")
print(f"   Signal at indices: [{signal_start}, {signal_end})")
print(f"   Roll amount: {roll_amount}")
print(f"   After roll, kernel at: [{(kernel_start - roll_amount) % plane_size}, {(kernel_end - roll_amount) % plane_size})")
print(f"   After roll, signal at: [{(signal_start - roll_amount) % plane_size}, {(signal_end - roll_amount) % plane_size})")

# JTC: FFT -> fftshift -> JPS -> FFT -> fftshift -> abs
jft = torch.fft.fft(input_plane_rolled)
jft_shifted = torch.fft.fftshift(jft)
jps = torch.abs(jft_shifted) ** 2 / plane_size

output_fft = torch.fft.fft(jps)
output_shifted = torch.fft.fftshift(output_fft)
output_abs = torch.abs(output_shifted)

# Extract using golden code formula
same_start = plane_size // 2 + sep + N // 2 + 1
indices = torch.arange(same_start, same_start + (M+N-1)) % plane_size

print(f"\n4. Extraction:")
print(f"   same_start = plane_size//2 + sep + N//2 + 1 = {plane_size}//2 + {sep} + {N}//2 + 1 = {same_start}")
print(f"   Extracting {M+N-1} indices starting at {same_start}")
print(f"   Indices: {indices.tolist()}")

manual_jtc = output_abs[indices]
print(f"   Manual JTC output: {manual_jtc}")

# 4. Compare
print(f"\n5. Comparison:")
print(f"   Reference (abs): {torch.abs(ref_corr)}")
print(f"   JTC output:      {jtc_output}")
print(f"   Manual JTC:      {manual_jtc}")

# Check if they match
diff_impl = torch.abs(jtc_output - manual_jtc).max()
print(f"\n   Implementation match: max diff = {diff_impl:.6f}")

# Try to correlate
if jtc_output.std() > 1e-6 and torch.abs(ref_corr).std() > 1e-6:
    corr_coef = torch.corrcoef(torch.stack([
        torch.abs(ref_corr) / torch.abs(ref_corr).std(),
        jtc_output / jtc_output.std()
    ]))[0, 1].item()
    print(f"   Correlation with |reference|: {corr_coef:.4f}")

# 6. Visualize full output plane to understand where peaks are
print(f"\n6. Full output plane analysis:")
print(f"   Output plane shape: {output_abs.shape}")
print(f"   Output plane max: {output_abs.max():.4f} at index {output_abs.argmax()}")
print(f"   Output plane min: {output_abs.min():.4f}")

# Find peaks
sorted_indices = torch.argsort(output_abs, descending=True)
print(f"   Top 5 peak positions: {sorted_indices[:5].tolist()}")
print(f"   Top 5 peak values: {output_abs[sorted_indices[:5]]}")

# Check where autocorrelation peaks should be
auto_center = plane_size // 2
print(f"\n   Autocorrelation should be centered at: {auto_center}")
print(f"   Value at autocorr center: {output_abs[auto_center]:.4f}")
print(f"   Cross-corr extraction range: [{same_start}, {(same_start + M + N - 2) % plane_size}]")

# Plot a simple ASCII visualization
print(f"\n7. ASCII visualization of output plane:")
output_np = output_abs.numpy()
max_val = output_np.max()
if max_val > 0:
    normalized = (output_np / max_val * 50).astype(int)
    for i in range(0, plane_size, 4):
        bar_height = normalized[i]
        bar = '█' * bar_height
        marker = ''
        if i in indices:
            marker = '<-- extracted'
        elif i == auto_center:
            marker = '<-- autocorr center'
        print(f"   [{i:2d}] {bar} {marker}")
