"""Debug why JTC doesn't match PyTorch even for positive inputs."""

import torch
import torch.nn.functional as F
from onn_config import AppConfig
from onn_component import JTC

# Simple config
M, N = 8, 3
plane_size = 32
sep = 7

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

jtc = JTC(config)

# Positive inputs
torch.manual_seed(42)
signal = torch.rand(M) * 0.5 + 0.5  # [0.5, 1.0]
kernel = torch.rand(N) * 0.5 + 0.5  # [0.5, 1.0]

print("="*80)
print("INPUTS:")
print(f"Signal: {signal}")
print(f"Kernel: {kernel}")

# JTC forward
signal_jtc = signal.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 8]
kernel_jtc = kernel.unsqueeze(0)  # [1, 3]
jtc_output = jtc(signal_jtc, kernel_jtc).squeeze()

print("\n" + "="*80)
print("JTC OUTPUT:")
print(f"Shape: {jtc_output.shape}")
print(f"Values: {jtc_output}")
print(f"Range: [{jtc_output.min():.4f}, {jtc_output.max():.4f}]")

# PyTorch conv1d (valid padding)
signal_pt = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
kernel_pt = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
pytorch_conv_valid = F.conv1d(signal_pt, kernel_pt, padding=0).squeeze()

print("\n" + "="*80)
print("PYTORCH CONV1D (valid padding):")
print(f"Shape: {pytorch_conv_valid.shape}")
print(f"Values: {pytorch_conv_valid}")
print(f"Range: [{pytorch_conv_valid.min():.4f}, {pytorch_conv_valid.max():.4f}]")

# PyTorch conv1d (full padding for correlation)
pytorch_conv_full = F.conv1d(signal_pt, kernel_pt, padding=N-1).squeeze()

print("\n" + "="*80)
print("PYTORCH CONV1D (full padding, M+N-1 outputs):")
print(f"Shape: {pytorch_conv_full.shape}")
print(f"Values: {pytorch_conv_full}")
print(f"Range: [{pytorch_conv_full.min():.4f}, {pytorch_conv_full.max():.4f}]")

# Manual convolution to verify
print("\n" + "="*80)
print("MANUAL CONVOLUTION (for verification):")
manual_output = []
for i in range(M - N + 1):
    patch = signal[i:i+N]
    result = torch.sum(patch * kernel)
    manual_output.append(result.item())
    print(f"  Position {i}: patch={patch} * kernel={kernel} = {result:.4f}")

manual_output = torch.tensor(manual_output)
print(f"\nManual output: {manual_output}")
print(f"Matches PyTorch valid: {torch.allclose(manual_output, pytorch_conv_valid)}")

# Check correlation vs convolution
print("\n" + "="*80)
print("CORRELATION vs CONVOLUTION:")
print("Convolution flips the kernel, correlation doesn't")

# Correlation (flip kernel for conv1d to get correlation)
kernel_flipped = torch.flip(kernel, [0])
kernel_flipped_pt = kernel_flipped.unsqueeze(0).unsqueeze(0)
pytorch_corr_full = F.conv1d(signal_pt, kernel_flipped_pt, padding=N-1).squeeze()

print(f"\nPyTorch CORRELATION (full, flipped kernel):")
print(f"  Values: {pytorch_corr_full}")
print(f"  Shape: {pytorch_corr_full.shape}")

# Manual JTC simulation to understand scaling
print("\n" + "="*80)
print("MANUAL JTC SIMULATION:")

input_plane = torch.zeros(plane_size, dtype=torch.complex64)
kernel_start = 0
signal_start = kernel_start + N + sep

input_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
input_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

roll_amount = (plane_size // 2) - (M + signal_start) // 2
input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

# JTC physics
jft = torch.fft.fft(input_plane)
jft_shifted = torch.fft.fftshift(jft)
jps = torch.abs(jft_shifted) ** 2
jps_scaled = jps / plane_size  # This scaling might be the issue

print(f"JPS max (before scaling): {jps.max():.4f}")
print(f"JPS max (after /plane_size): {jps_scaled.max():.4f}")

output_fft = torch.fft.fft(jps_scaled)
output_shifted = torch.fft.fftshift(output_fft)
output_abs = torch.abs(output_shifted)

same_start = plane_size // 2 + sep + N // 2
indices = torch.arange(same_start, same_start + (M+N-1)) % plane_size
manual_jtc_output = output_abs[indices]

print(f"\nManual JTC output: {manual_jtc_output}")
print(f"Implementation JTC: {jtc_output}")
print(f"Match: {torch.allclose(manual_jtc_output, jtc_output, atol=1e-5)}")

# Try without the /plane_size scaling
output_fft_no_scale = torch.fft.fft(jps)
output_shifted_no_scale = torch.fft.fftshift(output_fft_no_scale)
output_abs_no_scale = torch.abs(output_shifted_no_scale)
manual_jtc_no_scale = output_abs_no_scale[indices]

print(f"\nJTC without /plane_size scaling: {manual_jtc_no_scale}")
print(f"PyTorch correlation:             {pytorch_corr_full}")

# Check if it matches PyTorch correlation
if len(manual_jtc_no_scale) == len(pytorch_corr_full):
    print(f"\nDoes unscaled JTC match PyTorch correlation?")
    print(f"  Max diff: {torch.abs(manual_jtc_no_scale - pytorch_corr_full).max():.4f}")
    print(f"  Close: {torch.allclose(manual_jtc_no_scale, pytorch_corr_full, rtol=0.1)}")

print("\n" + "="*80)
print("CONCLUSION:")
print("JTC is computing correlation, but the scaling factor /plane_size")
print("reduces magnitudes significantly. Without this scaling, JTC should")
print("match PyTorch correlation for positive inputs.")
print("="*80)
