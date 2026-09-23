"""Continuous interstage voltage-to-field response; no DAC or ADC here.

This is a steady-state component. Temporal settling and lane state require
an explicit clock/lane schedule and are intentionally not inferred from GPU chunks.
"""

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class RemodulationSpec:
    # Descending polynomial coefficients, V -> signed sqrt(W). None uses the
    # existing ideal driver/MRM composition as the baseline affine response.
    field_coefficients: tuple[float, ...] | None = None
    phase_coefficients: tuple[float, ...] = (0.0,)  # V -> radians
    voltage_min_v: float | None = None
    voltage_max_v: float | None = None
    voltage_noise_rms_v: float = 0.0
    # Optional characterized steady-state table: voltage_v,field_sqrt_w[,phase_rad].
    transfer_csv: str | None = None
    extrapolation: str = "linear"  # explicit table policy: linear or clamp

    def __post_init__(self):
        if self.phase_coefficients is None:
            raise ValueError(
                "phase_coefficients must contain finite polynomial coefficients"
            )
        for name in ["field_coefficients", "phase_coefficients"]:
            values = getattr(self, name)
            if values is not None:
                values = tuple(float(v) for v in values)
                if not values or not all(math.isfinite(v) for v in values):
                    raise ValueError(
                        f"{name} must contain finite polynomial coefficients"
                    )
                object.__setattr__(self, name, values)
        if self.transfer_csv and self.field_coefficients is not None:
            raise ValueError("Select transfer_csv or field_coefficients, not both")
        if self.extrapolation not in {"linear", "clamp"}:
            raise ValueError("remodulation extrapolation must be 'linear' or 'clamp'")
        if not math.isfinite(self.voltage_noise_rms_v) or self.voltage_noise_rms_v < 0:
            raise ValueError(
                "remodulation voltage_noise_rms_v must be finite and nonnegative"
            )
        for value in [self.voltage_min_v, self.voltage_max_v]:
            if value is not None and not math.isfinite(value):
                raise ValueError("Remodulator voltage rails must be finite")
        if (
            self.voltage_min_v is not None
            and self.voltage_max_v is not None
            and self.voltage_min_v >= self.voltage_max_v
        ):
            raise ValueError("Remodulator lower rail must be below upper rail")

    @property
    def affine(self):
        return (
            self.transfer_csv is None
            and (self.field_coefficients is None or len(self.field_coefficients) <= 2)
            and self.voltage_min_v is None
            and self.voltage_max_v is None
            and self.voltage_noise_rms_v == 0
            and not any(self.phase_coefficients)
        )

    @property
    def real_symmetric(self):
        # Independent spatial noise breaks JPS symmetry even for real apertures.
        return (
            self.transfer_csv is None
            and not any(self.phase_coefficients)
            and self.voltage_noise_rms_v == 0
        )


class AnalogRemodulator(nn.Module):
    def __init__(self, spec: RemodulationSpec):
        super().__init__()
        self.spec = spec
        self.characterization_sha256 = None
        if spec.transfer_csv:
            self.characterization_sha256 = hashlib.sha256(
                Path(spec.transfer_csv).read_bytes()
            ).hexdigest()
            with open(spec.transfer_csv, newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) < 2:
                raise ValueError(
                    "A remodulation table requires at least two voltage samples"
                )
            table = torch.tensor(
                [
                    [
                        float(row["voltage_v"]),
                        float(row["field_sqrt_w"]),
                        float(row.get("phase_rad") or 0.0),
                    ]
                    for row in rows
                ],
                dtype=torch.float64,
            )
            if (
                not torch.isfinite(table).all()
                or not (table[1:, 0] > table[:-1, 0]).all()
            ):
                raise ValueError(
                    "Remodulation table must be finite with strictly increasing voltages"
                )
            # Characterization data, not a trainable parameter or stochastic state.
            self.register_buffer("table", table, persistent=False)
        else:
            self.register_buffer("table", None, persistent=False)

    def affine_coefficients(self, baseline):
        if not self.spec.affine:
            raise ValueError(
                "Closed-form solver requires affine, noiseless remodulation without voltage rails"
            )
        coeffs = self.spec.field_coefficients
        if coeffs is None:
            return baseline
        return (0.0, coeffs[0]) if len(coeffs) == 1 else coeffs

    @staticmethod
    def _poly(x, coefficients):
        y = torch.zeros_like(x) + coefficients[0]
        for coefficient in coefficients[1:]:
            y = y * x + coefficient
        return y

    def forward(self, voltage, baseline, *, return_trace=False):
        if self.spec.affine and not return_trace:
            a, b = self.affine_coefficients(baseline)
            return a * voltage + b
        requested = voltage
        if self.spec.voltage_noise_rms_v:
            voltage = (
                voltage + torch.randn_like(voltage) * self.spec.voltage_noise_rms_v
            )
        lo, hi = self.spec.voltage_min_v, self.spec.voltage_max_v
        drive = (
            torch.clamp(voltage, min=lo, max=hi)
            if lo is not None or hi is not None
            else voltage
        )
        outside = torch.zeros_like(drive, dtype=torch.bool)
        if self.table is None:
            coefficients = self.spec.field_coefficients or baseline
            # Retain the exact baseline affine evaluation order.
            field = (
                coefficients[0] * drive + coefficients[1]
                if len(coefficients) == 2
                else self._poly(drive, coefficients)
            )
            phase = self._poly(drive, self.spec.phase_coefficients)
            complex_output = any(self.spec.phase_coefficients)
        else:
            table = self.table.to(dtype=drive.dtype)
            v = table[:, 0].contiguous()
            outside = (drive < v[0]) | (drive > v[-1])
            x = (
                drive.clamp(v[0], v[-1])
                if self.spec.extrapolation == "clamp"
                else drive
            )
            index = torch.searchsorted(v, x.contiguous()).clamp(1, len(v) - 1)
            fraction = (x - v[index - 1]) / (v[index] - v[index - 1])
            field = table[index - 1, 1] + fraction * (
                table[index, 1] - table[index - 1, 1]
            )
            phase = table[index - 1, 2] + fraction * (
                table[index, 2] - table[index - 1, 2]
            )
            phase = phase + self._poly(drive, self.spec.phase_coefficients)
            complex_output = True
        if complex_output:
            # Keep the phase derivative in real arithmetic; complex scalar
            # multiplication breaks Inductor's compiled backward lowering.
            field = torch.complex(field * torch.cos(phase), field * torch.sin(phase))
        if return_trace:
            return field, {
                "requested_voltage_v": requested,
                "drive_voltage_v": drive,
                "clipped": voltage != drive,
                "outside_characterized_domain": outside,
                "field_sqrt_w": field,
                "phase_rad": phase,
            }
        return field

    def description(self):
        return {
            **asdict(self.spec),
            "model": "steady_state_voltage_to_field",
            "input_units": "V",
            "output_units": "sqrt(W)",
            "converter_stages": 0,
            "baseline_source": (
                "ideal input driver/MRM composition"
                if self.spec.field_coefficients is None and not self.spec.transfer_csv
                else "independent remodulation characterization"
            ),
            "characterization_sha256": self.characterization_sha256,
            "temporal_settling": "not modeled; requires timed lane schedule",
        }
