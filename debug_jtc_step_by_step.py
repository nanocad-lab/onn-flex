"""Step-by-step comparison of manual JTC vs implementation."""

import torch
from onn_config import AppConfig
from onn_layers import FTconvlayer

# Config
M, N = 8, 3
plane_size = 32
sep = 7

# Inputs
torch.manual_seed(42)
signal = torch.rand(M) * 0.5 + 0.5
kernel = torch.rand(N) * 0.5 + 0.5

print("="*80)
print(f"Signal: {signal}")
print(f"Kernel: {kernel}")

# Manual simulation
print("\n" + "="*80)
print("MANUAL JTC SIMULATION:")

input_plane = torch.zeros(plane_size, dtype=torch.complex64)
kernel_start = 0
signal_start = kernel_start + N + sep

input_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
input_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

print(f"Input plane (before roll):")
print(f"  Kernel at [{kernel_start}, {kernel_start+N})")
print(f"  Signal at [{signal_start}, {signal_start+M})")

roll_amount = (plane_size // 2) - (M + signal_start) // 2
input_plane_rolled = torch.roll(input_plane, shifts=roll_amount, dims=-1)

print(f"Roll amount: {roll_amount}")

# JTC physics
jft = torch.fft.fft(input_plane_rolled)
jft_shifted = torch.fft.fftshift(jft)
jps = torch.abs(jft_shifted) ** 2
print(f"JPS max (before /plane_size): {jps.max():.4f}")

jps_scaled = jps / plane_size
print(f"JPS max (after /plane_size): {jps_scaled.max():.4f}")

output_fft = torch.fft.fft(jps_scaled)
output_shifted = torch.fft.fftshift(output_fft)
output_abs = torch.abs(output_shifted)

print(f"Output plane max: {output_abs.max():.4f}")

# Extract
same_start = plane_size // 2 + sep + N // 2
indices = torch.arange(same_start, same_start + (M+N-1)) % plane_size
manual_output = output_abs[indices]

print(f"Extraction start: {same_start}")
print(f"Indices: {indices.tolist()}")
print(f"Manual output: {manual_output}")

# Now run through actual implementation
print("\n" + "="*80)
print("ACTUAL IMPLEMENTATION (via JTC class):")

config = AppConfig(
    input_length=M,
    kernel_length=N,
    output_length=None,
    jtc_separation=sep,
    jtc_total_field=plane_size,
    dac_bits=None,
    adc_bits=None,
    fourier_plane_bits=None,
    scale_output="none",
)

from onn_component import JTC
jtc = JTC(config)

# Prepare inputs with correct shapes
# signal: [batch_size=1, height=1, in_ch=1, input_len=8]
# kernel: [num_kernels=1, kernel_len=3]
signal_jtc = signal.view(1, 1, 1, M)
kernel_jtc = kernel.view(1, N)

print(f"Signal shape: {signal_jtc.shape}")
print(f"Kernel shape: {kernel_jtc.shape}")

# Forward pass through JTC
output_jtc = jtc(signal_jtc, kernel_jtc)

print(f"Output shape: {output_jtc.shape}")
print(f"Output: {output_jtc.squeeze()}")

print("\n" + "="*80)
print("COMPARISON:")
print(f"Manual:         {manual_output}")
print(f"Implementation: {output_jtc.squeeze()}")
print(f"Match: {torch.allclose(manual_output, output_jtc.squeeze(), atol=1e-4)}")

if not torch.allclose(manual_output, output_jtc.squeeze(), atol=1e-4):
    print("\n⚠ MISMATCH DETECTED!")
    diff = torch.abs(manual_output - output_jtc.squeeze())
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    print(f"Ratio (manual/impl): {(manual_output / (output_jtc.squeeze() + 1e-10)).mean():.2f}x")
