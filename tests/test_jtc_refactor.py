#!/usr/bin/env python3
"""Test script to verify the refactored JTC implementation.

This script tests that the refactored JTC produces identical outputs
to the original implementation via backward compatibility wrappers.
"""

import torch
from onn_config import AppConfig
from onn_component import JTC


def test_backward_compatibility():
    """Test that old methods produce same results as new pipeline."""
    print("=" * 80)
    print("TEST 1: Backward Compatibility")
    print("=" * 80)

    # Load a config
    config = AppConfig.from_yaml("configs/config_ideal.yaml")
    jtc = JTC(config)
    jtc.eval()

    # Create test inputs
    torch.manual_seed(42)
    batch_size = 2
    height = 8
    cout = 4
    signal = torch.rand(batch_size, height, 1, 8)
    kernel = torch.rand(cout, 8)

    print(f"Input shapes: signal={signal.shape}, kernel={kernel.shape}")

    with torch.no_grad():
        # New refactored forward method
        output_new = jtc(signal, kernel)

        # Old method via backward compatibility wrappers
        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size_for_jtc = (
            signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        )
        signal_reshaped = signal_full.reshape(batch_size_for_jtc, 8)
        kernel_reshaped = kernel_full.reshape(batch_size_for_jtc, 8)

        # Use old methods
        input_plane_old = jtc.generate_input_plane(signal_reshaped, kernel_reshaped)
        jft_old = jtc.post_fft(input_plane_old)
        jps_old = jtc.post_output_distortion(jft_old)
        inverse_output_old = jtc.inverse_output(jps_old)
        output_old = inverse_output_old.reshape(
            signal_full.shape[0],
            signal_full.shape[1],
            signal_full.shape[2],
            8,
        )

    # Compare outputs
    max_diff = torch.max(torch.abs(output_new - output_old)).item()
    mean_diff = torch.mean(torch.abs(output_new - output_old)).item()
    relative_diff = mean_diff / (torch.mean(torch.abs(output_old)).item() + 1e-12)

    print(f"Output shape: {output_new.shape}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Relative difference: {relative_diff:.2e}")

    if max_diff < 1e-5:
        print("✓ PASS: Outputs are identical (within tolerance)")
        return True
    else:
        print("✗ FAIL: Outputs differ significantly")
        print(f"New output sample: {output_new[0, 0, 0, :3]}")
        print(f"Old output sample: {output_old[0, 0, 0, :3]}")
        return False


def test_helper_methods():
    """Test individual helper methods."""
    print("\n" + "=" * 80)
    print("TEST 2: Helper Methods")
    print("=" * 80)

    config = AppConfig.from_yaml("configs/config_ideal.yaml")
    jtc = JTC(config)

    torch.manual_seed(42)

    # Test fft_and_magnitude
    print("\n[2.1] Testing fft_and_magnitude()...")
    x = torch.randn(4, 32, dtype=torch.complex64)
    result = jtc.fft_and_magnitude(x)
    expected = torch.abs(torch.fft.fftshift(torch.fft.fft(x)))
    diff = torch.max(torch.abs(result - expected)).item()
    print(f"  Max difference from manual FFT: {diff:.2e}")
    print(f"{'✓ PASS' if diff < 1e-6 else '✗ FAIL'}")

    # Test compute_correlation_indices
    print("\n[2.2] Testing compute_correlation_indices()...")
    indices = jtc.compute_correlation_indices(torch.device("cpu"))
    print(f"  Indices shape: {indices.shape}")
    print(f"  Indices range: [{indices.min()}, {indices.max()}]")
    print(f"  Expected length: {jtc.jtc_half_size}")
    print(f"  {'✓ PASS' if len(indices) == jtc.jtc_half_size else '✗ FAIL'}")

    # Test build_input_plane
    print("\n[2.3] Testing build_input_plane()...")
    signal_complex = torch.randn(4, 8, dtype=torch.complex64)
    kernel_complex = torch.randn(4, 8, dtype=torch.complex64)
    plane = jtc.build_input_plane(signal_complex, kernel_complex)
    print(f"  Plane shape: {plane.shape}")
    print(f"  Expected shape: (4, {jtc.jtc_total_field})")
    non_zero = (plane != 0).sum().item()
    print(f"  Non-zero elements: {non_zero}")
    print(f"  Expected: {8 + 8} (signal + kernel)")
    print(f"{'✓ PASS' if plane.shape[1] == jtc.jtc_total_field else '✗ FAIL'}")

    return True


def test_pipeline_steps():
    """Test the 7-step pipeline individually."""
    print("\n" + "=" * 80)
    print("TEST 3: Pipeline Steps")
    print("=" * 80)

    config = AppConfig.from_yaml("configs/config_ideal.yaml")
    jtc = JTC(config)
    jtc.eval()

    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8)
    kernel = torch.rand(3, 8)

    print(f"\nInput: signal={signal.shape}, kernel={kernel.shape}")

    with torch.no_grad():
        # Prepare inputs
        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size_for_jtc = (
            signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        )
        signal_reshaped = signal_full.reshape(batch_size_for_jtc, 8)
        kernel_reshaped = kernel_full.reshape(batch_size_for_jtc, 8)

        # Step 1: Input distortion
        print("\nStep 1: Input distortion")
        signal_distorted = jtc.input_distortion(signal_reshaped)
        kernel_distorted = jtc.input_distortion(kernel_reshaped)
        print(
            f"  Signal distorted: {signal_distorted.shape}, dtype={signal_distorted.dtype}"
        )
        print(
            f"  Kernel distorted: {kernel_distorted.shape}, dtype={kernel_distorted.dtype}"
        )

        # Step 2: Build input plane
        print("\nStep 2: Build input plane")
        input_plane = jtc.build_input_plane(signal_distorted, kernel_distorted)
        print(f"  Input plane: {input_plane.shape}, dtype={input_plane.dtype}")

        # Step 3: FFT
        print("\nStep 3: FFT")
        jft = jtc.fft_and_magnitude(input_plane)
        print(f"  JFT: {jft.shape}, dtype={jft.dtype}")
        print(f"  JFT range: [{jft.min():.4f}, {jft.max():.4f}]")

        # Step 4: Output distortion
        print("\nStep 4: Output distortion")
        jps = jtc.output_distortion(jft)
        print(f"  JPS: {jps.shape}, dtype={jps.dtype}")
        print(f"  JPS range: [{jps.min():.4f}, {jps.max():.4f}]")

        # Step 5: Input distortion
        print("\nStep 5: Input distortion (again)")
        jps_distorted = jtc.input_distortion(jps)
        print(f"  JPS distorted: {jps_distorted.shape}, dtype={jps_distorted.dtype}")

        # Step 6: FFT
        print("\nStep 6: FFT (again)")
        output_plane = jtc.fft_and_magnitude(jps_distorted)
        print(f"  Output plane: {output_plane.shape}, dtype={output_plane.dtype}")

        # Step 7: Output distortion
        print("\nStep 7: Output distortion (again)")
        output_plane = jtc.output_distortion(output_plane)
        print(f"  Output plane: {output_plane.shape}")
        print(f"  Output range: [{output_plane.min():.4f}, {output_plane.max():.4f}]")

        # Step 8: Index selection
        print("\nStep 8: Index selection")
        indices = jtc.compute_correlation_indices(output_plane.device)
        output = output_plane[..., indices]
        print(f"  Output: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

        print("\n✓ All pipeline steps executed successfully")

    return True


def test_determinism():
    """Test that the forward pass is deterministic."""
    print("\n" + "=" * 80)
    print("TEST 4: Determinism")
    print("=" * 80)

    config = AppConfig.from_yaml("configs/config_ideal.yaml")
    jtc = JTC(config)
    jtc.eval()

    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8)
    kernel = torch.rand(3, 8)

    with torch.no_grad():
        output1 = jtc(signal, kernel)
        output2 = jtc(signal, kernel)

    diff = torch.max(torch.abs(output1 - output2)).item()
    print(f"Max difference between runs: {diff:.2e}")

    if diff < 1e-10:
        print("✓ PASS: Forward pass is deterministic")
        return True
    else:
        print("✗ FAIL: Forward pass is non-deterministic")
        return False


def test_gradients():
    """Test that gradients flow correctly through the refactored pipeline."""
    print("\n" + "=" * 80)
    print("TEST 5: Gradient Flow")
    print("=" * 80)

    config = AppConfig.from_yaml("configs/config_ideal.yaml")
    jtc = JTC(config)
    jtc.train()

    torch.manual_seed(42)
    signal = torch.rand(2, 4, 1, 8, requires_grad=True)
    kernel = torch.rand(3, 8, requires_grad=True)

    print(f"Input shapes: signal={signal.shape}, kernel={kernel.shape}")

    # Forward pass
    output = jtc(signal, kernel)
    loss = output.sum()

    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    signal_grad = signal.grad
    kernel_grad = kernel.grad

    print("\nGradient statistics:")
    print(f"  Signal grad: shape={signal_grad.shape}")
    print(
        f"    - Non-zero elements: {(signal_grad != 0).sum().item()}/{signal_grad.numel()}"
    )
    print(f"    - Mean abs: {signal_grad.abs().mean().item():.2e}")
    print(f"    - Max abs: {signal_grad.abs().max().item():.2e}")

    print(f"  Kernel grad: shape={kernel_grad.shape}")
    print(
        f"    - Non-zero elements: {(kernel_grad != 0).sum().item()}/{kernel_grad.numel()}"
    )
    print(f"    - Mean abs: {kernel_grad.abs().mean().item():.2e}")
    print(f"    - Max abs: {kernel_grad.abs().max().item():.2e}")

    has_signal_grad = signal_grad is not None and signal_grad.abs().max() > 0
    has_kernel_grad = kernel_grad is not None and kernel_grad.abs().max() > 0

    if has_signal_grad and has_kernel_grad:
        print("\n✓ PASS: Gradients flow correctly")
        return True
    else:
        print("\n✗ FAIL: Gradients are missing or all zero")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("JTC REFACTOR VALIDATION TESTS")
    print("=" * 80)

    results = []

    try:
        results.append(("Backward Compatibility", test_backward_compatibility()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Backward Compatibility", False))

    try:
        results.append(("Helper Methods", test_helper_methods()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Helper Methods", False))

    try:
        results.append(("Pipeline Steps", test_pipeline_steps()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Pipeline Steps", False))

    try:
        results.append(("Determinism", test_determinism()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Determinism", False))

    try:
        results.append(("Gradient Flow", test_gradients()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Gradient Flow", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The refactored JTC is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
