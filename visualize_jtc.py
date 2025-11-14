"""Visualization function similar to golden code to verify JTC correctness."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def visualize_jtc_output(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    M: int,
    N: int,
    sep: int,
    plane_size: int,
    save_path: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Visualize JTC output plane with autocorr and cross-corr components.

    Returns:
        auto_signal: Autocorrelation of signal
        auto_kernel: Autocorrelation of kernel
        cross_plane: Cross-correlation plane
        info: Dictionary with analysis info (peaks, contamination, etc.)
    """
    # Build input plane
    input_plane = torch.zeros(plane_size, dtype=torch.complex64)
    kernel_start = 0
    signal_start = kernel_start + N + sep

    input_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
    input_plane[signal_start:signal_start+M] = signal.to(torch.complex64)

    # Roll to center
    roll_amount = (plane_size // 2) - (M + signal_start) // 2
    input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

    # Separate components for decomposition
    signal_plane = torch.zeros(plane_size, dtype=torch.complex64)
    kernel_plane = torch.zeros(plane_size, dtype=torch.complex64)

    kernel_plane[kernel_start:kernel_start+N] = kernel.to(torch.complex64)
    signal_plane[signal_start:signal_start+M] = signal.to(torch.complex64)
    kernel_plane = torch.roll(kernel_plane, shifts=roll_amount, dims=-1)
    signal_plane = torch.roll(signal_plane, shifts=roll_amount, dims=-1)

    # FFTs
    signal_fft = torch.fft.fftshift(torch.fft.fft(signal_plane))
    kernel_fft = torch.fft.fftshift(torch.fft.fft(kernel_plane))
    combined_fft = torch.fft.fftshift(torch.fft.fft(input_plane))

    # JPS components
    jps_signal = torch.abs(signal_fft) ** 2
    jps_kernel = torch.abs(kernel_fft) ** 2
    jps_cross = torch.conj(signal_fft) * kernel_fft + signal_fft * torch.conj(kernel_fft)
    jps_full = torch.abs(combined_fft) ** 2

    # Back to spatial domain
    auto_signal = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_signal / plane_size)))
    auto_kernel = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_kernel / plane_size)))
    cross_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_cross / plane_size)))
    full_plane = torch.abs(torch.fft.fftshift(torch.fft.fft(jps_full / plane_size)))

    # Analysis
    autocorr_center = plane_size // 2
    autocorr_length = 2 * max(M, N) - 1
    crosscorr_length = M + N - 1

    # Expected positions
    autocorr_start = autocorr_center - (autocorr_length // 2)
    autocorr_end = autocorr_start + autocorr_length

    # Cross-correlation peak position (empirical - find it)
    cross_peak_idx = cross_plane.argmax().item()
    cross_peak_val = cross_plane.max().item()

    # Autocorrelation peak
    auto_peak_idx = full_plane.argmax().item()
    auto_peak_val = full_plane.max().item()

    # Test extraction formulas
    formula_old = plane_size // 2 + sep + N // 2 + 1
    formula_new = plane_size // 2 + sep + N // 2

    info = {
        'M': M,
        'N': N,
        'sep': sep,
        'plane_size': plane_size,
        'autocorr_center': autocorr_center,
        'autocorr_length': autocorr_length,
        'autocorr_range': (autocorr_start, autocorr_end),
        'crosscorr_length': crosscorr_length,
        'cross_peak_idx': cross_peak_idx,
        'cross_peak_val': cross_peak_val,
        'auto_peak_idx': auto_peak_idx,
        'auto_peak_val': auto_peak_val,
        'formula_old_start': formula_old,
        'formula_new_start': formula_new,
    }

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    x = np.arange(plane_size)

    # Plot 1: Full output plane
    ax = axes[0]
    ax.plot(x, full_plane.numpy(), 'b-', linewidth=2, label='Full JTC Output')
    ax.axvline(autocorr_center, color='r', linestyle='--', alpha=0.5, label=f'Autocorr Center ({autocorr_center})')
    ax.axvspan(autocorr_start, autocorr_end, alpha=0.2, color='red', label=f'Autocorr Region ({autocorr_length})')
    ax.axvline(cross_peak_idx, color='g', linestyle='--', alpha=0.5, label=f'Cross Peak ({cross_peak_idx})')
    ax.set_title(f'JTC Output Plane (M={M}, N={N}, sep={sep}, plane={plane_size})')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Decomposition
    ax = axes[1]
    ax.plot(x, auto_signal.numpy(), 'r-', alpha=0.7, label='Autocorr(signal)')
    ax.plot(x, auto_kernel.numpy(), 'orange', alpha=0.7, label='Autocorr(kernel)')
    ax.plot(x, cross_plane.numpy(), 'g-', linewidth=2, label='Cross-correlation')
    ax.set_title('JPS Decomposition: auto_s + auto_k + cross')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Extraction comparison
    ax = axes[2]

    # Test different extraction positions
    for offset in [-2, -1, 0, 1, 2]:
        test_start = formula_new + offset
        indices = [(test_start + i) % plane_size for i in range(crosscorr_length)]
        ax.scatter(indices, full_plane[indices].numpy(), alpha=0.6, s=30,
                  label=f'Start={test_start} (new{offset:+d})')

    # Old formula
    indices_old = [(formula_old + i) % plane_size for i in range(crosscorr_length)]
    ax.scatter(indices_old, full_plane[indices_old].numpy(), marker='x', s=100,
              color='red', linewidth=2, label=f'Old formula (start={formula_old})')

    ax.plot(x, full_plane.numpy(), 'k-', alpha=0.3, linewidth=1)
    ax.axvspan(autocorr_start, autocorr_end, alpha=0.2, color='red')
    ax.set_title('Extraction Formula Comparison')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Contamination analysis
    ax = axes[3]

    # Compute contamination for each possible starting position
    contamination_mean = []
    contamination_max = []
    starts = []

    for start_idx in range(plane_size):
        indices = torch.tensor([(start_idx + i) % plane_size for i in range(crosscorr_length)])
        auto_s = auto_signal[indices]
        auto_k = auto_kernel[indices]
        cross_v = cross_plane[indices]
        total_auto = auto_s + auto_k

        # Contamination percentage
        cont_pct = 100 * total_auto / (total_auto + cross_v + 1e-10)

        starts.append(start_idx)
        contamination_mean.append(cont_pct.mean().item())
        contamination_max.append(cont_pct.max().item())

    ax.plot(starts, contamination_mean, 'b-', linewidth=2, label='Mean Contamination %')
    ax.plot(starts, contamination_max, 'r--', linewidth=1, alpha=0.7, label='Max Contamination %')
    ax.axvline(formula_old, color='red', linestyle=':', linewidth=2, label=f'Old formula ({formula_old})')
    ax.axvline(formula_new, color='green', linestyle=':', linewidth=2, label=f'New formula ({formula_new})')
    ax.axhline(10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_title('Contamination vs Extraction Start Index')
    ax.set_xlabel('Start Index')
    ax.set_ylabel('Autocorrelation Contamination (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(100, max(contamination_max) * 1.1))

    # Find best start (minimum mean contamination)
    best_idx = np.argmin(contamination_mean)
    best_start = starts[best_idx]
    info['best_start'] = best_start
    info['best_contamination_mean'] = contamination_mean[best_idx]
    info['best_contamination_max'] = contamination_max[best_idx]
    info['old_contamination_mean'] = contamination_mean[formula_old]
    info['old_contamination_max'] = contamination_max[formula_old]
    info['new_contamination_mean'] = contamination_mean[formula_new]
    info['new_contamination_max'] = contamination_max[formula_new]

    ax.axvline(best_start, color='purple', linestyle='-', linewidth=2,
              label=f'Best ({best_start}, mean={contamination_mean[best_idx]:.1f}%)')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.savefig('jtc_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to jtc_visualization.png")

    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("JTC VISUALIZATION SUMMARY")
    print("="*80)
    print(f"Config: M={M}, N={N}, sep={sep}, plane_size={plane_size}")
    print(f"Autocorr region: [{autocorr_start}, {autocorr_end}) (length={autocorr_length})")
    print(f"Crosscorr length: {crosscorr_length}")
    print(f"Cross-corr peak at index: {cross_peak_idx}")
    print(f"Autocorr peak at index: {auto_peak_idx}")
    print(f"\nExtraction Formula Analysis:")
    print(f"  Old formula (plane//2 + sep + N//2 + 1): start={formula_old}")
    print(f"    Mean contamination: {contamination_mean[formula_old]:.2f}%")
    print(f"    Max contamination:  {contamination_max[formula_old]:.2f}%")
    print(f"  New formula (plane//2 + sep + N//2): start={formula_new}")
    print(f"    Mean contamination: {contamination_mean[formula_new]:.2f}%")
    print(f"    Max contamination:  {contamination_max[formula_new]:.2f}%")
    print(f"  Optimal start: {best_start}")
    print(f"    Mean contamination: {contamination_mean[best_idx]:.2f}%")
    print(f"    Max contamination:  {contamination_max[best_idx]:.2f}%")
    print("="*80)

    return auto_signal, auto_kernel, cross_plane, info


if __name__ == "__main__":
    # Test all configurations
    configs = [
        (8, 3, 32, 7),
        (16, 8, 48, 9),
        (16, 8, 64, 15),
        (8, 8, 48, 8),  # Golden code case
    ]

    for M, N, plane_size, sep in configs:
        print(f"\n{'='*80}")
        print(f"Testing Config: M={M}, N={N}, plane_size={plane_size}, sep={sep}")
        print(f"{'='*80}")

        torch.manual_seed(42)
        signal = torch.randn(M) * 0.1
        kernel = torch.randn(N) * 0.1

        save_path = f"jtc_viz_M{M}_N{N}_p{plane_size}_s{sep}.png"
        auto_s, auto_k, cross, info = visualize_jtc_output(
            signal, kernel, M, N, sep, plane_size, save_path
        )
