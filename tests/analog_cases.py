"""Shared analog solver cases and deterministic layer construction."""

import torch

from onn_config import AppConfig
from onn_jtc_conv2d import JTCConv2d

PATHS = {
    "analytic": dict(conv_backend="jtc_analytic"),
    "closed": {},
    "affine_fft": dict(jtc_fourier_closed_form=False),
    "pixel_dft": dict(jtc_fourier_closed_form=False, jtc_carrier_stopband_bins=3),
    "rfft": dict(
        jtc_fourier_closed_form=False,
        jtc_carrier_stopband_bins=3,
        jtc_fourier_lag_gemm=False,
    ),
    "complex_fft": dict(
        jtc_fourier_closed_form=False,
        mrm_phase_distortion_strength=1.0,
        jtc_fourier_lag_gemm=False,
    ),
}


def analog_config(path="closed", **overrides):
    args = dict(
        input_length=8,
        kernel_length=8,
        output_length=8,
        jtc_total_field=64,
        jtc_separation=8,
        jtc_rowwise_geometry="config",
        conv_backend="jtc_analog_fourier",
        fourier_plane_bits=None,
        pd_input_clamp_max_w=None,
        adc_bits=None,
        dac_bits=4,
        driver_distortion_polyfit_order=3,
        jtc_output_gain_mode="fixed",
        jtc_output_gain=1000.0,
    )
    args.update(PATHS[path])
    args.update(overrides)
    return AppConfig(**args)


def analog_layer(config, shots=65536, nonnegative=True, out_channels=3):
    torch.manual_seed(42)
    return JTCConv2d(
        2,
        out_channels,
        3,
        padding=1,
        bias=False,
        config=config,
        max_jtc_shots=shots,
        assume_nonnegative_input=nonnegative,
    )


STRENGTHS = (
    "driver_distortion_strength",
    "mrm_amplitude_distortion_strength",
    "mrm_phase_distortion_strength",
    "lens_distortion_strength",
    "pd_distortion_strength",
    "tia_distortion_strength",
)

ALPHA_CASES = {
    "ideal": {},
    **{name: {name: 1.0} for name in STRENGTHS},
    "combined": {name: 1.0 for name in STRENGTHS},
}
