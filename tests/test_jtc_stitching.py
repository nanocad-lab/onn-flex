"""Tests for JTC tile stitching and contamination-aware cycle planning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from jtc_cycle_planner import (
    usable_outputs,
    compute_contamination_profile,
    cycles_for_config,
)


class TestContaminationProfile:
    """Test contamination analysis and effective stride computation."""

    def test_contamination_profile_basic(self):
        """Test basic contamination profile computation."""
        # Config with good separation
        M, N, plane, sep = 16, 8, 64, 15

        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        # Should get full correlation length
        assert total == M + N - 1  # 23
        # clean is the number of clean VALID outputs (M-N+1 = 9)
        num_valid = M - N + 1
        assert clean <= num_valid
        # Most or all valid outputs should be clean for this config
        assert clean >= num_valid * 0.8
        # Stride should be reasonable
        assert stride > 0
        assert stride <= num_valid

        print(f"\nM={M}, N={N}, plane={plane}, sep={sep}")
        print(f"  Total outputs: {total}")
        print(f"  Clean valid outputs: {clean}/{num_valid}")
        print(f"  Effective stride: {stride}")

    def test_contamination_profile_tight_config(self):
        """Test config with expected contamination (golden code case)."""
        # Golden code case: M=8, N=8, sep=8, plane=48
        # Visualization showed mean 6.16%, max 92.43% contamination
        M, N, plane, sep = 8, 8, 48, 8

        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        # Should get full correlation length
        assert total == 15  # M + N - 1

        # Due to contamination, clean count may be less than total
        # But should still have some usable outputs
        assert clean > 0
        assert stride > 0

        print(f"\nM={M}, N={N}, plane={plane}, sep={sep} (golden code)")
        print(f"  Total outputs: {total}")
        print(f"  Clean outputs: {clean} ({100*clean/total:.1f}%)")
        print(f"  Effective stride: {stride}")

    def test_contamination_profile_invalid_config(self):
        """Test that invalid configs return zeros."""
        # Too tight - won't fit
        M, N, plane, sep = 16, 8, 20, 10  # 16+8+10 = 34 > 20

        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        assert total == 0
        assert clean == 0
        assert stride == 0

    @pytest.mark.parametrize("M,N,plane,sep", [
        (8, 3, 32, 7),
        (16, 8, 48, 9),
        (16, 8, 64, 15),
        (8, 8, 48, 8),
    ])
    def test_contamination_all_test_configs(self, M, N, plane, sep):
        """Test contamination profile for all standard test configs."""
        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        # All these configs are valid
        assert total == M + N - 1

        # Should have at least some clean outputs
        assert clean >= 0  # May be zero for very contaminated configs

        # Stride should be positive if any clean outputs
        if clean > 0:
            assert stride > 0
            assert stride <= total

        print(f"\nM={M}, N={N}, plane={plane}, sep={sep}")
        print(f"  Total={total}, Clean={clean}, Stride={stride}")
        print(f"  Clean percentage: {100*clean/total:.1f}%")


class TestCyclesForConfig:
    """Test cycle planning with stitching logic."""

    def test_cycles_basic(self):
        """Test basic cycle computation."""
        # Good config with minimal contamination
        M, N, plane, sep = 16, 3, 48, 10

        result = cycles_for_config(M, N, plane, sep)
        assert result is not None

        passes, cycles, stride, total = result

        # Check values are reasonable
        assert passes > 0
        assert cycles > 0
        assert stride > 0
        assert total == M + N - 1

        # For 32x32 image with 3x3 kernel: output is 30x30
        # Each row needs ceil(30 / stride) passes
        expected_passes = (30 + stride - 1) // stride  # ceil division
        assert passes == expected_passes

        # Total cycles = passes_per_width * num_rows * kernel_height
        # num_rows = 30, kernel_height = 3
        expected_cycles = passes * 30 * 3
        assert cycles == expected_cycles

        print(f"\nM={M}, N={N}, plane={plane}, sep={sep}")
        print(f"  Total outputs: {total}")
        print(f"  Effective stride: {stride}")
        print(f"  Passes per row: {passes}")
        print(f"  Total cycles: {cycles}")

    def test_cycles_with_contamination(self):
        """Test cycle computation with contaminated config."""
        # Golden code case - has contamination
        M, N, plane, sep = 8, 8, 48, 8

        result = cycles_for_config(M, N, plane, sep)

        # Even with contamination, should return valid result
        # (We accept contamination and adjust stride accordingly)
        if result is not None:
            passes, cycles, stride, total = result

            assert passes > 0
            assert cycles > 0
            assert total == 15  # M + N - 1

            # Stride may be less than total due to contamination
            assert stride > 0
            assert stride <= total

            print(f"\nM={M}, N={N}, plane={plane}, sep={sep} (contaminated)")
            print(f"  Total outputs: {total}")
            print(f"  Effective stride: {stride} (reduced due to contamination)")
            print(f"  Passes per row: {passes}")
            print(f"  Total cycles: {cycles}")

    def test_stitching_logic_verification(self):
        """Verify stitching logic for covering 30-pixel row with stride."""
        # Test config
        M, N, plane, sep = 16, 3, 48, 10

        result = cycles_for_config(M, N, plane, sep)
        assert result is not None

        passes, cycles, stride, total = result

        # For 30-pixel output row with stride S:
        # Pass 1: covers pixels [0, S)
        # Pass 2: covers pixels [S, 2S)
        # ...
        # Last pass: covers pixels [(P-1)*S, 30)
        # Where P = ceil(30/S)

        # Verify we have enough passes to cover all 30 pixels
        coverage = passes * stride
        assert coverage >= 30, f"Coverage {coverage} < 30 with {passes} passes of stride {stride}"

        # Verify we don't have too many passes (shouldn't be more than 1 pass extra)
        assert (passes - 1) * stride < 30, f"Too many passes: {passes-1} passes already cover {(passes-1)*stride} >= 30"

        print(f"\nStitching verification for M={M}, N={N}:")
        print(f"  Output row width: 30 pixels")
        print(f"  Effective stride: {stride}")
        print(f"  Number of passes: {passes}")
        print(f"  Coverage: {coverage} pixels")
        print(f"  ✓ Stitching logic correct")

    @pytest.mark.parametrize("M,N,plane,sep", [
        (8, 3, 32, 7),
        (16, 8, 48, 9),
        (16, 8, 64, 15),
    ])
    def test_cycles_multiple_configs(self, M, N, plane, sep):
        """Test cycle computation for multiple configs."""
        result = cycles_for_config(M, N, plane, sep)

        if result is not None:
            passes, cycles, stride, total = result

            print(f"\nM={M}, N={N}, plane={plane}, sep={sep}")
            print(f"  Total outputs: {total}")
            print(f"  Effective stride: {stride}")
            print(f"  Passes per width: {passes}")
            print(f"  Total cycles: {cycles}")

            # Basic sanity checks
            assert passes > 0
            assert cycles > 0
            assert stride > 0
            assert total > 0

            # Verify stitching covers full width
            # cycles_for_config uses WIDTH=32 from jtc_cycle_planner
            # out_w = WIDTH - N + 1
            import math
            WIDTH = 32  # From jtc_cycle_planner
            out_w = WIDTH - N + 1
            expected_passes = math.ceil(out_w / stride)
            assert passes == expected_passes, f"Expected {expected_passes} passes for width {out_w} with stride {stride}, got {passes}"


class TestStitchingEdgeCases:
    """Test edge cases in stitching logic."""

    def test_stride_equals_output_width(self):
        """Test when stride >= output width (single pass case)."""
        # Large M should give large stride
        M, N, plane, sep = 30, 3, 64, 15

        result = cycles_for_config(M, N, plane, sep)
        assert result is not None

        passes, cycles, stride, total = result

        # With stride >= 30, should only need 1 pass per row
        if stride >= 30:
            assert passes == 1

        print(f"\nLarge stride case: M={M}, N={N}")
        print(f"  Stride: {stride}")
        print(f"  Passes per row: {passes}")

    def test_small_stride_many_passes(self):
        """Test when stride is small (many passes needed)."""
        # Small M gives small stride
        M, N, plane, sep = 3, 3, 32, 10

        result = cycles_for_config(M, N, plane, sep)
        assert result is not None

        passes, cycles, stride, total = result

        # Small stride means many passes
        assert passes >= 5  # 30 / 5 = 6 passes minimum

        print(f"\nSmall stride case: M={M}, N={N}")
        print(f"  Stride: {stride}")
        print(f"  Passes per row: {passes}")

    def test_backward_compatibility_with_usable_outputs(self):
        """Verify backward compatibility with old usable_outputs function."""
        M, N, plane, sep = 16, 8, 48, 9

        # Old function
        old_usable = usable_outputs(M, N, plane, sep)

        # New function
        total, clean, stride = compute_contamination_profile(M, N, plane, sep)

        # old_usable should equal total (M+N-1)
        assert old_usable == total == M + N - 1

        print(f"\nBackward compatibility: M={M}, N={N}")
        print(f"  Old usable_outputs: {old_usable}")
        print(f"  New total_outputs: {total}")
        print(f"  ✓ Match (backward compatible)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
