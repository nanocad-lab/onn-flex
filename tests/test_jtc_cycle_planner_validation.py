"""Validate aperture support, physical shot counts, and selected readout groups."""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from analog_cases import analog_config, analog_layer

from jtc_cycle_planner import usable_outputs
from onn_component import JTC
from onn_config import AppConfig
from onn_jtc_conv2d import JTCConv2d
from onn_shotplan import (
    ApertureGeometry,
    compute_contamination_profile,
    plan_linear_readout,
)
from onn_shotreport import observed_shot_plans, plan_resnet


def _execute_linear_reference(x, weight, plan):
    """Independent FFT oracle for phase shots, ADC groups and digital copies.

    The two-branch optical reference requires nonnegative inputs. Its 128-point
    zero-padded FFTs reproduce full linear correlation and do not specify the
    physical intermediate sampling count. The default aperture is 21/dark22/21.
    """
    if tuple(x.shape) != plan.input_shape or weight.shape[:2] != (
        plan.output_shape[1],
        x.shape[1],
    ):
        raise ValueError("Tensor shapes do not match the plan")
    if (x < 0).any() or not plan.nonnegative:
        raise ValueError("Two-branch reference requires nonnegative inputs")
    if (plan.signal_slots, plan.kernel_slots, plan.dark_positions) != (
        21,
        21,
        22,
    ):
        raise ValueError("Optical reference expects the 21/dark22/21 aperture")
    top, bottom, left, right = plan.padding
    padded = F.pad(x, (left, right, top, bottom))
    kh, _ = plan.kernel_size
    sh, _ = plan.stride
    results = []
    for tile in plan.tiles:
        indices = (
            tile.input_start
            + torch.arange(tile.input_count, device=x.device) * tile.input_stride
        )
        signal_tile = padded.index_select(-1, indices)
        signal_tile = F.pad(signal_tile, (0, plan.signal_slots - tile.input_count))
        rows = []
        for iy in range(plan.output_shape[2]):
            accumulated = None
            for ci in range(x.shape[1]):
                for ky in range(kh):
                    signal = signal_tile[:, ci, iy * sh + ky]
                    kernels = weight[:, ci, ky, tile.kernel_phase :: tile.input_stride]
                    kernels = F.pad(
                        kernels, (0, plan.kernel_slots - tile.active_kernel_taps)
                    )
                    decoded = []
                    for sign in (1, -1):
                        plane = F.pad(signal[:, None], (0, 107)) + F.pad(
                            (sign * kernels).clamp_min(0)[None], (43, 64)
                        )
                        correlation = torch.fft.ifft(
                            torch.fft.fft(plane).abs().square()
                        ).real
                        decoded.append(sign * correlation[..., 23:64].flip(-1))
                    full = sum(decoded)
                    # One ADC value per group; repeat digitally to all its columns.
                    values = [
                        full[..., list(group)]
                        .mean(-1, keepdim=True)
                        .expand(*full.shape[:-1], len(output))
                        for group, output in zip(
                            tile.correlation_groups, tile.output_groups
                        )
                    ]
                    selected = torch.cat(values, dim=-1)
                    accumulated = (
                        selected if accumulated is None else accumulated + selected
                    )
            rows.append(accumulated)
        tile_result = torch.stack(rows, dim=2)
        first, last = tile.output_groups[0][0], tile.output_groups[-1][-1]
        results.append(F.pad(tile_result, (first, plan.output_shape[-1] - 1 - last)))
    return sum(results)


def compute_reference_correlation(
    signal: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor:
    """Compute reference correlation using PyTorch's conv1d.

    Correlation is like convolution but without flipping the kernel.
    Returns full correlation of length M+N-1.
    """
    # Reshape for conv1d
    signal_conv = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
    # Flip kernel for correlation (correlation = conv with flipped kernel)
    kernel_flipped = torch.flip(kernel, [0])
    kernel_conv = kernel_flipped.unsqueeze(0).unsqueeze(0)  # [1, 1, N]

    # Full correlation with padding
    padding = len(kernel) - 1
    output = F.conv1d(signal_conv, kernel_conv, padding=padding)
    return output.squeeze()  # [M+N-1]


def compute_jtc_full_output(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    M: int,
    N: int,
    sep: int,
    plane_size: int,
) -> torch.Tensor:
    """Compute full JTC output plane (before extraction).

    Returns the magnitude of the inverse FFT of the JPS.
    """
    # Build input plane: kernel [0:N], signal [N+sep:N+sep+M]
    input_plane = torch.zeros(plane_size, dtype=torch.complex64)

    kernel_start = 0
    kernel_end = kernel_start + N
    signal_start = kernel_end + sep
    signal_end = signal_start + M

    input_plane[kernel_start:kernel_end] = kernel.to(torch.complex64)
    input_plane[signal_start:signal_end] = signal.to(torch.complex64)

    # Roll to center
    roll_amount = (plane_size // 2) - (M + signal_start) // 2
    input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

    # JTC: FFT -> fftshift -> JPS
    jft = torch.fft.fft(input_plane)
    jft_shifted = torch.fft.fftshift(jft)
    jps = torch.abs(jft_shifted) ** 2 / plane_size

    # Back to output: FFT -> fftshift -> abs
    output_fft = torch.fft.fft(jps)
    output_shifted = torch.fft.fftshift(output_fft)
    output_abs = torch.abs(output_shifted)

    return output_abs


class TestJTCCyclePlannerValidation:
    """Validate jtc_cycle_planner against actual JTC physics."""

    @pytest.mark.parametrize(
        "input_len,kernel_len,lens,sep",
        [
            (8, 3, 32, 7),
            (16, 8, 48, 9),
            (16, 8, 64, 15),
            (8, 8, 48, 8),  # Golden code case
        ],
    )
    def test_usable_outputs_matches_overlap_free_region(
        self, input_len, kernel_len, lens, sep
    ):
        """Test that usable_outputs count matches direct JTC output bookkeeping."""
        M, N = input_len, kernel_len
        plane_size = lens

        # Get cycle planner prediction
        predicted_usable = usable_outputs(M, N, plane_size, sep)
        expected_full_correlation = M + N - 1

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

        indices = jtc.compute_correlation_indices(torch.device("cpu"))
        assert jtc.output_length == predicted_usable
        assert indices.numel() == predicted_usable
        assert int(indices.min()) >= 0
        assert int(indices.max()) < plane_size
        assert predicted_usable <= expected_full_correlation

        torch.manual_seed(42)
        signal = torch.randn(2, 1, 1, M) * 0.1
        kernel = torch.randn(1, N) * 0.1
        output = jtc(signal, kernel)
        assert output.shape == (2, 1, 1, predicted_usable)
        assert torch.isfinite(output).all()

    def test_edge_case_wrapping_detection(self):
        """Test that wrapping indices don't contaminate autocorrelation.

        Edge case: when same_start + output_length > plane_size,
        indices wrap around. Verify wrapped indices don't hit autocorrelation.
        """
        # Configuration that causes wrapping
        M, N = 16, 8
        plane_size = 32  # Tight fit
        sep = 8

        predicted_usable = usable_outputs(M, N, plane_size, sep)

        # Check if indices wrap
        same_start = plane_size // 2 + sep + N // 2 + 1
        same_end = same_start + predicted_usable

        print(f"\nWrapping test: M={M}, N={N}, plane_size={plane_size}, sep={sep}")
        print(
            f"  same_start={same_start}, same_end={same_end}, plane_size={plane_size}"
        )
        print(f"  Wrapping: {same_end > plane_size}")

        if same_end > plane_size:
            # Indices wrap - this is a problem!
            # Wrapped indices: [same_start, plane_size) + [0, same_end - plane_size)
            wrapped_indices = list(range(0, same_end - plane_size))

            # Autocorrelation is centered around plane_size // 2
            # Extends roughly ± max(M-1, N-1)
            auto_center = plane_size // 2
            auto_extent = max(M - 1, N - 1)
            auto_region = set(
                range(
                    (auto_center - auto_extent) % plane_size,
                    (auto_center + auto_extent + 1) % plane_size,
                )
            )

            # Check overlap
            overlap = set(wrapped_indices) & auto_region

            print(f"  Wrapped indices: {wrapped_indices}")
            print(
                f"  Autocorrelation region: [{auto_center - auto_extent}, {auto_center + auto_extent}]"
            )
            print(f"  Overlap: {overlap}")

            # If there's overlap, the simple formula is WRONG
            if overlap:
                pytest.fail(
                    f"CRITICAL ERROR: Wrapped indices {overlap} overlap with autocorrelation region! "
                    f"The simple formula M+N+sep <= plane_size is insufficient."
                )

    @pytest.mark.parametrize(
        "input_len,kernel_len,lens,sep",
        [
            (8, 3, 32, 7),
            (16, 8, 48, 9),
            (8, 8, 48, 8),
        ],
    )
    def test_cycle_planner_vs_actual_jtc_output(self, input_len, kernel_len, lens, sep):
        """Test that cycle planner prediction matches actual JTC implementation."""
        M, N = input_len, kernel_len

        # Get cycle planner prediction
        predicted_usable = usable_outputs(M, N, lens, sep)

        # Create actual JTC
        config = AppConfig(
            input_length=M,
            kernel_length=N,
            output_length=None,  # Auto-calculate
            jtc_separation=sep,
            jtc_total_field=lens,
            dac_bits=None,
            adc_bits=None,
            fourier_plane_bits=None,
        )

        jtc = JTC(config)

        # Check that JTC's output_length matches prediction
        msg = (
            f"JTC output_length ({jtc.output_length}) != "
            f"cycle planner prediction ({predicted_usable})"
        )
        assert jtc.output_length == predicted_usable, msg

        # Test actual forward pass
        batch_size = 2
        signal = torch.randn(batch_size, 1, 1, M)
        kernel = torch.randn(1, N)

        output = jtc(signal, kernel)

        # Verify output shape matches prediction
        msg = (
            f"JTC output length ({output.shape[-1]}) != "
            f"cycle planner prediction ({predicted_usable})"
        )
        assert output.shape[-1] == predicted_usable, msg

        print(f"\nConfig: M={M}, N={N}, lens={lens}, sep={sep}")
        print(f"  Predicted: {predicted_usable}")
        print(f"  JTC output_length: {jtc.output_length}")
        print(f"  Actual output shape: {output.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


@pytest.mark.parametrize(
    "path", ["analytic", "closed", "pixel_dft", "rfft", "complex_fft"]
)
@pytest.mark.parametrize("mapping", ["row", "dot"])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_plan_counts_actual_shots_and_adc_samples(path, mapping, nonnegative):
    cfg = analog_config(path, jtc_shot_mapping=mapping, jtc_rowwise_geometry="auto")
    layer = JTCConv2d(
        2,
        3,
        3,
        stride=2,
        padding=1,
        bias=False,
        config=cfg,
        max_jtc_shots=64,
        assume_nonnegative_input=nonnegative,
    ).eval()
    x = torch.rand(1, 2, 5, 6)
    plan = layer.shot_plan(x.shape)
    observed = []
    original = layer.shot.readout

    def readout(v, *args, **kwargs):
        observed.append((v.numel() // v.shape[-1], v.numel()))
        return original(v, *args, **kwargs)

    with patch.object(layer.shot, "readout", side_effect=readout), torch.no_grad():
        y = layer(x)
    assert tuple(y.shape) == plan.output_shape
    assert sum(item[0] for item in observed) == plan.shots
    assert sum(item[1] for item in observed) == plan.adc_samples
    assert plan.to_dict()["adc_stages_per_shot"] == 1
    assert (
        json.loads(json.dumps(observed_shot_plans(layer)))["layers"][0]["plans"][0][
            "plan_id"
        ]
        == plan.to_dict()["plan_id"]
    )


def test_plan_solver_independence_and_no_mapping_fallback():
    cfg = analog_config()
    plans = [
        analog_layer(
            replace(cfg, jtc_fourier_closed_form=closed, jtc_fourier_lag_gemm=lag)
        ).shot_plan((2, 2, 6, 6))
        for closed, lag in [(True, True), (False, True), (False, False)]
    ]
    assert plans[0] == plans[1] == plans[2]
    grouped = JTCConv2d(2, 4, 3, groups=2, config=cfg)
    with pytest.raises(ValueError, match="explicitly select"):
        grouped(torch.rand(1, 2, 6, 6))
    bypass = JTCConv2d(2, 4, 1, config=cfg).shot_plan((2, 2, 6, 6))
    assert bypass.mapping == "electronic" and bypass.shots == bypass.adc_samples == 0


def test_plan_contamination_uses_selected_bins_and_shared_geometry():
    geometry = ApertureGeometry(8, 3, 24, 3)
    profile = geometry.contamination(range(2, 8))
    _, clean, _ = compute_contamination_profile(8, 3, 24, 3)
    assert sum(item["clean"] for item in profile) == clean
    assert 0 < clean < len(profile)
    cfg = analog_config(jtc_total_field=20, jtc_separation=3)
    plan = JTCConv2d(2, 3, 3, stride=2, padding=1, config=cfg).shot_plan((1, 2, 6, 6))
    assert plan.selected_offsets == (2, 4, 6)
    assert plan.to_dict()["accuracy_impact"] == "requires workload evaluation"


def test_model_report_records_calls_and_electronic_bypass():
    cfg = analog_config(model_arch="resnet11", jtc_total_field=256, jtc_separation=31)
    report = plan_resnet(cfg, (2, 3, 32, 32))
    assert report["output_shape"] == (2, 10)
    assert len(report["layers"]) == 14
    assert any(call["mapping"] == "electronic" for call in report["layers"])
    assert report["total_shots"] == sum(call["shots"] for call in report["layers"])
    assert report["total_adc_samples"] > report["total_shots"] > 0
    assert json.loads(json.dumps(report))["remodulation"]["converter_stages"] == 0


@pytest.mark.parametrize(
    "width,stride", [(32, 1), (17, 1), (32, 2), (17, 2), (4, 2), (1, 1)]
)
@pytest.mark.parametrize("readout", ["individual", "pair_mean_hold"])
def test_planned_optical_groups_match_native_operator_values_and_gradients(
    width, stride, readout
):
    torch.manual_seed(width + stride)
    x = torch.rand(1, 2, 3, width, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 2, 3, 3, dtype=torch.float64, requires_grad=True)
    plan = plan_linear_readout(
        input_shape=x.shape,
        out_channels=2,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=(1, 1, 1, 1),
        readout=readout,
    )
    direct = F.conv2d(x, weight, stride=stride, padding=1)
    if readout == "pair_mean_hold":
        # Non-overlapping adjacent means, repeated into the same output positions.
        pooled = F.avg_pool2d(
            direct, (1, 2), (1, 2), ceil_mode=True, count_include_pad=False
        )
        direct = pooled.repeat_interleave(2, dim=-1)[..., : direct.shape[-1]]
    tiled = _execute_linear_reference(x, weight, plan)
    torch.testing.assert_close(tiled, direct, rtol=1e-10, atol=1e-10)
    probe = torch.randn_like(direct)
    expected = torch.autograd.grad(
        (direct * probe).sum(), (x, weight), retain_graph=True
    )
    actual = torch.autograd.grad((tiled * probe).sum(), (x, weight))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    for tile in plan.tiles:
        assert tile.input_count <= 21
        for group in tile.correlation_groups:
            assert len(group) <= 2
            if len(group) == 2:
                assert group[1] == group[0] + 1
    assert len(plan.tiles) > 1 if width == 32 else True
