"""Test clean JTC physics (fourier_conv_forward) vs PyTorch."""

import torch
import torch.nn.functional as F
from onn_config import AppConfig
from onn_layers import FTconvlayer

# Config
M, N = 8, 3
plane_size = 32
sep = 7

# Positive inputs
torch.manual_seed(42)
signal = torch.rand(M) * 0.5 + 0.5
kernel = torch.rand(N) * 0.5 + 0.5

print("="*80)
print(f"Signal: {signal}")
print(f"Kernel: {kernel}")

# Setup FTconvlayer (uses clean JTC physics)
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
    conv_backend="fourier",
)

layer = FTconvlayer(
    in_channels=1,
    out_channels=1,
    config=config,
    kernel_size=M,  # This is the patch size
    batch_size=1,
)

# Set kernel weights
with torch.no_grad():
    # Layer expects [out_ch, in_ch, 1, kernel_size]
    layer.weights.data = kernel.view(1, 1, 1, N)

# Prepare input [B=1, H=1, in_ch=1, W=M]
input_tensor = signal.view(1, 1, 1, M)

# Call fourier_conv_forward directly (clean JTC physics)
# It expects x: [B, H, 1, W], weight: [Cout, W]
weight_2d = kernel.unsqueeze(0)  # [1, N]
jtc_output = layer.fourier_conv_forward(input_tensor, weight_2d)

print("\n" + "="*80)
print("JTC OUTPUT (clean physics):")
print(f"  Shape: {jtc_output.shape}")
print(f"  Values: {jtc_output.squeeze()}")

# PyTorch correlation (full padding)
signal_pt = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
kernel_flipped = torch.flip(kernel, [0])
kernel_pt = kernel_flipped.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
pytorch_corr = F.conv1d(signal_pt, kernel_pt, padding=N-1).squeeze()

print("\n" + "="*80)
print("PYTORCH CORRELATION (full):")
print(f"  Shape: {pytorch_corr.shape}")
print(f"  Values: {pytorch_corr}")

# Compare
print("\n" + "="*80)
print("COMPARISON:")

jtc_squeezed = jtc_output.squeeze()
if len(jtc_squeezed) == len(pytorch_corr):
    diff = torch.abs(jtc_squeezed - pytorch_corr)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Relative error: {max_diff / pytorch_corr.max():.4f}")

    # Check if they match within tolerance
    matches = torch.allclose(jtc_squeezed, pytorch_corr, rtol=1e-3, atol=1e-4)
    print(f"\nMatch within rtol=1e-3, atol=1e-4: {matches}")

    if matches:
        print("✓ JTC clean physics MATCHES PyTorch correlation!")
    else:
        print("✗ Mismatch detected")
        print(f"\nJTC:     {jtc_squeezed}")
        print(f"PyTorch: {pytorch_corr}")
else:
    print(f"✗ Length mismatch: JTC={len(jtc_squeezed)}, PyTorch={len(pytorch_corr)}")
    print(f"JTC:     {jtc_squeezed}")
    print(f"PyTorch: {pytorch_corr}")
