#!/usr/bin/env python3
"""Test and validate memory optimization features.

This script tests the transfer function simplification and gradient checkpointing
features to measure memory usage improvements during training.
"""

import torch
import torch.nn as nn
from onn_config import AppConfig
from onn_component import JTC
from simplified_transfer_functions import SimplifiedJTCTransferFunctions
import tracemalloc
import gc


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_transfer_function_simplification():
    """Test transfer function simplification and measure fidelity."""
    print("\n" + "="*70)
    print("TEST 1: Transfer Function Simplification")
    print("="*70)

    # Create config
    config = AppConfig()
    config.simplify_transfer_functions = True
    config.tf_simplification_max_error = 1e-4

    # Create simplified transfer functions
    print("\nCreating simplified transfer functions...")
    tf_dict = SimplifiedJTCTransferFunctions.create_simplified_functions(
        config,
        enable_simplification=True,
        driver_max_error=1e-4,
        pd_tia_max_error=1e-4
    )

    # Validate simplification
    print("\nValidating simplification fidelity...")
    metrics = SimplifiedJTCTransferFunctions.validate_simplification(
        tf_dict['original_driver'],
        tf_dict['original_pd'],
        tf_dict['original_tia'],
        tf_dict.get('driver_double'),  # Not implemented yet
        tf_dict['pd_tia'],
        num_samples=10000
    )

    print("\nPD-TIA Composition Metrics:")
    print(f"  Original order: {metrics['pd_tia']['order_original']}")
    print(f"  Simplified order: {metrics['pd_tia']['order_simplified']}")
    print(f"  Max error: {metrics['pd_tia']['max_error']:.2e}")
    print(f"  Mean error: {metrics['pd_tia']['mean_error']:.2e}")
    print(f"  RMSE: {metrics['pd_tia']['rmse']:.2e}")

    return metrics


def test_jtc_memory_with_simplification():
    """Test JTC memory usage with and without simplification."""
    print("\n" + "="*70)
    print("TEST 2: JTC Memory Usage Comparison")
    print("="*70)

    batch_size = 128
    num_iterations = 10

    # Test without simplification
    print(f"\nTesting WITHOUT transfer function simplification...")
    config_baseline = AppConfig()
    config_baseline.simplify_transfer_functions = False
    config_baseline.checkpoint_jtc = False

    jtc_baseline = JTC(config_baseline)
    jtc_baseline.train()

    # Warm-up
    signal = torch.randn(batch_size, 1, 1, config_baseline.input_length)
    kernel = torch.randn(8, config_baseline.kernel_length)
    _ = jtc_baseline(signal, kernel)

    # Measure
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before_baseline = get_gpu_memory_usage()

    for _ in range(num_iterations):
        signal = torch.randn(batch_size, 1, 1, config_baseline.input_length)
        kernel = torch.randn(8, config_baseline.kernel_length)
        output = jtc_baseline(signal, kernel)
        loss = output.sum()
        loss.backward()

    mem_peak_baseline = get_gpu_memory_usage()
    if torch.cuda.is_available():
        mem_peak_baseline = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Peak memory: {mem_peak_baseline:.2f} MB")

    # Test with simplification
    print(f"\nTesting WITH transfer function simplification...")
    config_optimized = AppConfig()
    config_optimized.simplify_transfer_functions = True
    config_optimized.tf_simplification_max_error = 1e-4
    config_optimized.checkpoint_jtc = False

    jtc_optimized = JTC(config_optimized)
    jtc_optimized.train()

    # Warm-up
    _ = jtc_optimized(signal, kernel)

    # Measure
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    mem_before_optimized = get_gpu_memory_usage()

    for _ in range(num_iterations):
        signal = torch.randn(batch_size, 1, 1, config_optimized.input_length)
        kernel = torch.randn(8, config_optimized.kernel_length)
        output = jtc_optimized(signal, kernel)
        loss = output.sum()
        loss.backward()

    mem_peak_optimized = get_gpu_memory_usage()
    if torch.cuda.is_available():
        mem_peak_optimized = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Peak memory: {mem_peak_optimized:.2f} MB")

    # Calculate improvement
    if mem_peak_baseline > 0:
        improvement = (mem_peak_baseline - mem_peak_optimized) / mem_peak_baseline * 100
        print(f"\nMemory reduction: {improvement:.1f}%")
    else:
        print("\nNote: Running on CPU, GPU memory stats not available")

    return {
        'baseline_memory': mem_peak_baseline,
        'optimized_memory': mem_peak_optimized
    }


def test_gradient_checkpointing():
    """Test gradient checkpointing memory savings."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Checkpointing")
    print("="*70)

    batch_size = 128
    num_iterations = 5

    # Test without checkpointing
    print(f"\nTesting WITHOUT gradient checkpointing...")
    config_no_cp = AppConfig()
    config_no_cp.simplify_transfer_functions = True
    config_no_cp.checkpoint_jtc = False

    jtc_no_cp = JTC(config_no_cp)
    jtc_no_cp.train()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(num_iterations):
        signal = torch.randn(batch_size, 1, 1, config_no_cp.input_length)
        kernel = torch.randn(8, config_no_cp.kernel_length)
        output = jtc_no_cp(signal, kernel)
        loss = output.sum()
        loss.backward()

    mem_no_cp = get_gpu_memory_usage()
    if torch.cuda.is_available():
        mem_no_cp = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Peak memory: {mem_no_cp:.2f} MB")

    # Test with checkpointing
    print(f"\nTesting WITH gradient checkpointing...")
    config_with_cp = AppConfig()
    config_with_cp.simplify_transfer_functions = True
    config_with_cp.checkpoint_jtc = True

    jtc_with_cp = JTC(config_with_cp)
    jtc_with_cp.train()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(num_iterations):
        signal = torch.randn(batch_size, 1, 1, config_with_cp.input_length)
        kernel = torch.randn(8, config_with_cp.kernel_length)
        output = jtc_with_cp(signal, kernel)
        loss = output.sum()
        loss.backward()

    mem_with_cp = get_gpu_memory_usage()
    if torch.cuda.is_available():
        mem_with_cp = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Peak memory: {mem_with_cp:.2f} MB")

    # Calculate improvement
    if mem_no_cp > 0:
        improvement = (mem_no_cp - mem_with_cp) / mem_no_cp * 100
        print(f"\nMemory reduction from checkpointing: {improvement:.1f}%")
    else:
        print("\nNote: Running on CPU, GPU memory stats not available")

    return {
        'no_checkpoint_memory': mem_no_cp,
        'checkpoint_memory': mem_with_cp
    }


def test_combined_optimizations():
    """Test all optimizations combined."""
    print("\n" + "="*70)
    print("TEST 4: Combined Optimizations")
    print("="*70)

    batch_size = 128
    num_iterations = 10

    # Baseline: No optimizations
    print(f"\nBaseline (no optimizations)...")
    config_baseline = AppConfig()
    config_baseline.simplify_transfer_functions = False
    config_baseline.checkpoint_jtc = False

    jtc_baseline = JTC(config_baseline)
    jtc_baseline.train()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(num_iterations):
        signal = torch.randn(batch_size, 1, 1, config_baseline.input_length)
        kernel = torch.randn(8, config_baseline.kernel_length)
        output = jtc_baseline(signal, kernel)
        loss = output.sum()
        loss.backward()

    mem_baseline = get_gpu_memory_usage()
    if torch.cuda.is_available():
        mem_baseline = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Peak memory: {mem_baseline:.2f} MB")

    # All optimizations
    print(f"\nAll optimizations enabled...")
    config_all = AppConfig()
    config_all.simplify_transfer_functions = True
    config_all.tf_simplification_max_error = 1e-4
    config_all.checkpoint_jtc = True

    jtc_all = JTC(config_all)
    jtc_all.train()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(num_iterations):
        signal = torch.randn(batch_size, 1, 1, config_all.input_length)
        kernel = torch.randn(8, config_all.kernel_length)
        output = jtc_all(signal, kernel)
        loss = output.sum()
        loss.backward()

    mem_all = get_gpu_memory_usage()
    if torch.cuda.is_available():
        mem_all = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Peak memory: {mem_all:.2f} MB")

    # Calculate total improvement
    if mem_baseline > 0:
        total_improvement = (mem_baseline - mem_all) / mem_baseline * 100
        print(f"\nTotal memory reduction: {total_improvement:.1f}%")
        print(f"Absolute memory saved: {mem_baseline - mem_all:.2f} MB")
    else:
        print("\nNote: Running on CPU, GPU memory stats not available")

    return {
        'baseline_memory': mem_baseline,
        'optimized_memory': mem_all,
        'improvement_percent': total_improvement if mem_baseline > 0 else 0
    }


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MEMORY OPTIMIZATION VALIDATION TESTS")
    print("="*70)

    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\nRunning on: {device_name}")

    # Run tests
    try:
        tf_metrics = test_transfer_function_simplification()
        jtc_memory = test_jtc_memory_with_simplification()
        cp_memory = test_gradient_checkpointing()
        combined = test_combined_optimizations()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n1. Transfer Function Simplification:")
        print(f"   PD-TIA composition RMSE: {tf_metrics['pd_tia']['rmse']:.2e}")
        print(f"   Polynomial order reduction: {tf_metrics['pd_tia']['order_original']} → {tf_metrics['pd_tia']['order_simplified']}")

        if torch.cuda.is_available():
            print(f"\n2. Memory Improvements:")
            print(f"   Baseline memory: {combined['baseline_memory']:.2f} MB")
            print(f"   Optimized memory: {combined['optimized_memory']:.2f} MB")
            print(f"   Total reduction: {combined['improvement_percent']:.1f}%")
        else:
            print(f"\n2. Memory Improvements:")
            print(f"   (GPU required for accurate memory measurements)")

        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
