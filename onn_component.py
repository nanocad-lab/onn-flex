import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from onn_config import AppConfig
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import (
    converter_quantize_ste,
    quantized_transfer_lut,
)
from onn_shotplan import ApertureGeometry, compute_contamination_profile


def _legendre_torch(n: int, x: torch.Tensor) -> torch.Tensor:
    """Compute the n-th Legendre polynomial P_n(x) for a 1-D tensor x."""
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x.clone()
    p0 = torch.ones_like(x)
    p1 = x.clone()
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1


class LensLegendre(nn.Module):
    """Unitary lens operator parameterized by a 1-D Legendre basis."""

    def __init__(self, config: AppConfig, length: int):
        super().__init__()
        self.config = config
        self.length = int(length)
        self.order = int(config.lens_legendre_order or 0)

        if self.order <= 0 or self.length <= 0:
            self.register_buffer(
                "basis",
                torch.empty(0, self.length, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "base_coefs", torch.empty(0, dtype=torch.float32), persistent=False
            )
            return

        raw = config.lens_coefs or []
        if isinstance(raw, (int, float, str)):
            raw = [raw]
        base = [float(value) for value in raw]
        if len(base) < self.order:
            base += [0.0] * (self.order - len(base))
        elif len(base) > self.order:
            base = base[: self.order]

        self.register_buffer(
            "base_coefs", torch.as_tensor(base, dtype=torch.float32), persistent=False
        )

        rho = torch.linspace(-1.0, 1.0, self.length, dtype=torch.float32)
        terms: list[torch.Tensor] = []
        for ord_k in range(1, self.order + 1):
            p = _legendre_torch(ord_k, rho)
            p = p - p.mean()
            nrm = torch.linalg.vector_norm(p).item()
            if nrm > 0:
                p = p / nrm
            terms.append(p)
        basis = torch.stack(terms, dim=0)
        self.register_buffer("basis", basis, persistent=False)

    def _phase_diag(self, x: torch.Tensor) -> torch.Tensor | None:
        strength = float(self.config.lens_distortion_strength or 0.0)
        if strength == 0.0 or self.base_coefs.numel() == 0 or self.basis.numel() == 0:
            return None

        real_dtype = x.real.dtype
        coefs = (self.base_coefs.to(dtype=real_dtype) * strength).reshape(-1, 1)
        phase_profile = (self.basis.to(dtype=real_dtype) * coefs).sum(dim=0)
        phase_diag = torch.polar(torch.ones_like(phase_profile), phase_profile)
        return phase_diag.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase_diag = self._phase_diag(x)
        if phase_diag is None:
            return x

        n = x.shape[-1]
        scale = math.sqrt(float(n))
        # Apply the unitary similarity transform F diag(e^{iφ}) F^{-1},
        # where F is the (unitary) forward DFT. This matches the MATLAB
        # construction used to fit the Legendre coefficients.
        x_ifft = torch.fft.ifft(x, dim=-1) * scale  # unitary inverse DFT
        x_ifft = x_ifft * phase_diag
        return torch.fft.fft(x_ifft, dim=-1) / scale  # unitary forward DFT

    def is_identity(self) -> bool:
        return (
            float(self.config.lens_distortion_strength or 0.0) == 0.0
            or self.base_coefs.numel() == 0
            or self.basis.numel() == 0
        )


def _sqrt_clamped(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(values, 0.0, None))


def _compute_linear_coeffs(
    csv_file: str,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    linearization: str = "endpoint",
) -> np.ndarray:
    """Compute coefficients a, b for the ideal linear transfer y = a * x + b."""
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    if output_transform is not None:
        y = output_transform(y)
    linearization = str(linearization or "endpoint").lower()

    if linearization == "least_squares":
        return np.polyfit(x, y, 1).astype(np.float32)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    x_first, x_last = x_sorted[0], x_sorted[-1]
    y_first, y_last = y_sorted[0], y_sorted[-1]
    if x_last == x_first:
        raise ValueError("Input points for ideal linear interpolation are identical.")

    if linearization == "midpoint_ls":
        a = np.polyfit(x, y, 1)[0]
        x_mid = 0.5 * (x_first + x_last)
        y_mid = 0.5 * (y_first + y_last)
        b = y_mid - a * x_mid
        return np.array([a, b], dtype=np.float32)

    if linearization != "endpoint":
        raise ValueError(
            "linearization must be 'endpoint', 'least_squares', or 'midpoint_ls'"
        )

    a = (y_last - y_first) / (x_last - x_first)
    b = y_first - a * x_first
    return np.array([a, b], dtype=np.float32)


def _blend_poly_coeffs(
    poly_coeffs: np.ndarray,
    ideal_coeffs: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Return high-to-low coefficients for the blended transfer curve."""
    strength = float(strength)
    if strength <= 0.0:
        return np.asarray(ideal_coeffs, dtype=np.float32)
    if strength >= 1.0:
        return np.asarray(poly_coeffs, dtype=np.float32)

    poly = np.asarray(poly_coeffs, dtype=np.float64)
    ideal = np.asarray(ideal_coeffs, dtype=np.float64)
    if poly.size > ideal.size:
        ideal = np.pad(ideal, (poly.size - ideal.size, 0))
    elif ideal.size > poly.size:
        poly = np.pad(poly, (ideal.size - poly.size, 0))
    return (strength * poly + (1.0 - strength) * ideal).astype(np.float32)


def _compose_poly_coeffs(
    outer_coeffs: np.ndarray,
    inner_coeffs: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Compose high-to-low polynomial coefficients as outer(inner(x))."""
    outer = np.poly1d(np.asarray(outer_coeffs, dtype=np.float64))
    inner = np.poly1d(np.asarray(inner_coeffs, dtype=np.float64))
    composed = outer(inner) * float(scale)
    return np.asarray(composed.c, dtype=np.float32)


def _poly_derivative_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """Return high-to-low coefficients of the polynomial's derivative."""
    derivative = np.polyder(np.poly1d(np.asarray(coeffs, dtype=np.float64)))
    return np.asarray(derivative.c, dtype=np.float32)


def raise_dynamo_recompile_limit(minimum: int = 64) -> None:
    """Give shared compiled code objects enough recompile budget.

    Every module instance traces its own variant of a shared method code
    object (different self, channel counts, plane widths). Dynamo's default
    budget of 8 silently falls back to eager once exceeded.
    """
    import torch._dynamo

    torch._dynamo.config.recompile_limit = max(
        int(torch._dynamo.config.recompile_limit), int(minimum)
    )


def calculate_aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Calculate AIC for polynomial regression"""
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = 2 * n_params - 2 * log_likelihood
    return aic


def get_ideal_degree(
    csv_file: str,
    max_degree: int = 10,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> int:
    """
    Fit polynomials to CSV data and print results

    Parameters:
    csv_file: path to CSV file with 'input' and 'output' columns
    max_degree: maximum polynomial degree to test
    """
    # Read CSV data
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    if output_transform is not None:
        y = output_transform(y)

    best_degree: int | None = None
    best_aic = float("inf")

    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        n_params = degree + 1  # coefficients + intercept
        aic = calculate_aic(y, y_pred, n_params)
        if aic < best_aic:
            best_aic = aic
            best_degree = degree

        if r2 > 0.9995:
            return degree

    # AIC can be non-monotonic across polynomial degree; scan the full candidate
    # range before falling back to the global best AIC fit.
    return best_degree if best_degree is not None else 1


def get_io_ranges(csv_file: str) -> tuple[float, float, float, float]:
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    return x.min(), x.max(), y.min(), y.max()


def _apply_soft_upper_cap(
    x: torch.Tensor,
    upper: float,
    width: float,
) -> torch.Tensor:
    delta = x - float(upper)
    feather = -torch.expm1(-delta.clamp_min(0.0) / float(width))
    capped = float(upper) + float(width) * feather
    return torch.where(delta > 0, capped, x)


def _apply_soft_lower_cap(
    x: torch.Tensor,
    lower: float,
    width: float,
) -> torch.Tensor:
    delta = float(lower) - x
    feather = -torch.expm1(-delta.clamp_min(0.0) / float(width))
    capped = float(lower) - float(width) * feather
    return torch.where(delta > 0, capped, x)


def apply_pd_input_guard(x: torch.Tensor, config: AppConfig) -> torch.Tensor:
    clamp_min = config.pd_input_clamp_min_w
    clamp_max = config.pd_input_clamp_max_w
    if clamp_min is None and clamp_max is None:
        return x

    mode = str(config.pd_input_clamp_mode or "hard").lower()
    if mode == "hard":
        if clamp_min is None:
            return torch.clamp_max(x, float(clamp_max))
        if clamp_max is None:
            return torch.clamp_min(x, float(clamp_min))
        return torch.clamp(x, float(clamp_min), float(clamp_max))

    if mode != "soft":
        raise ValueError("pd_input_clamp_mode must be 'hard' or 'soft'")

    width = float(config.pd_input_soft_clamp_width_w or 0.0)
    if width <= 0.0:
        if clamp_min is None:
            return torch.clamp_max(x, float(clamp_max))
        if clamp_max is None:
            return torch.clamp_min(x, float(clamp_min))
        return torch.clamp(x, float(clamp_min), float(clamp_max))

    # Feather the polynomial input just outside the measured range. The response
    # is exactly identity inside the configured range, then smoothly approaches
    # a bounded shoulder one `width` past the rail instead of extrapolating a
    # high-order polynomial arbitrarily far.
    if clamp_min is not None:
        x = _apply_soft_lower_cap(x, float(clamp_min), width)
    if clamp_max is not None:
        x = _apply_soft_upper_cap(x, float(clamp_max), width)
    return x


def get_coeffs(
    csv_file: str,
    degree: int,
    output_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    if output_transform is not None:
        y = output_transform(y)
    coeffs = np.polyfit(x, y, degree)
    return coeffs


class Driver(nn.Module):
    def __init__(self, config: AppConfig):
        super(Driver, self).__init__()
        self.config = config
        self.degree: int = 0
        if self.config.driver_distortion_data_path is None:
            raise ValueError("Driver distortion data path is not set")
        if self.config.driver_distortion_polyfit_order is None:
            self.degree = get_ideal_degree(self.config.driver_distortion_data_path)
        else:
            self.degree = self.config.driver_distortion_polyfit_order
        coeffs = get_coeffs(self.config.driver_distortion_data_path, self.degree)
        coeff_tensor = torch.as_tensor(coeffs, dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)

        # Ideal (reference) linear coefficients a, b where y = a * x + b
        ideal_coeffs = _compute_linear_coeffs(
            self.config.driver_distortion_data_path,
            linearization=self.config.transfer_linearization,
        )
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength (0 -> ideal, 1 -> fitted polynomial)
        self.strength: float = float(self.config.driver_distortion_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strength = float(self.strength)
        if strength <= 0.0:
            return self.ideal_coeffs[0] * x + self.ideal_coeffs[1]

        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y.mul_(x).add_(a)

        if strength >= 1.0:
            return poly_y

        # Ideal linear response
        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
        # Blend in-place to reduce peak memory
        poly_y.mul_(strength).add_(ideal_y, alpha=(1.0 - strength))
        return poly_y


class PD(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.degree: int = 0
        if self.config.pd_distortion_data_path is None:
            raise ValueError("PD distortion data path is not set")
        if self.config.pd_distortion_polyfit_order is None:
            self.degree = get_ideal_degree(self.config.pd_distortion_data_path)
        else:
            self.degree = self.config.pd_distortion_polyfit_order
        self.input_min, self.input_max, self.output_min, self.output_max = (
            get_io_ranges(self.config.pd_distortion_data_path)
        )
        coeffs = get_coeffs(self.config.pd_distortion_data_path, self.degree)
        coeff_tensor = torch.as_tensor(coeffs, dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)

        # Ideal (reference) linear coefficients a, b where y = a * x + b.
        # PD.forward receives optical power in Watts; the square-law
        # field-amplitude -> power conversion happens at the detector boundary.
        ideal_coeffs = _compute_linear_coeffs(
            self.config.pd_distortion_data_path,
            linearization=self.config.transfer_linearization,
        )
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength
        self.strength: float = float(self.config.pd_distortion_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the photodetector transfer curve to optical power in Watts."""
        pd_noise_w = float(self.config.pd_noise_w or 0.0)
        if pd_noise_w > 0:
            x = x + torch.randn_like(x) * pd_noise_w
        x = apply_pd_input_guard(x, self.config)
        strength = float(self.strength)
        if strength <= 0.0:
            ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
            return ideal_y

        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y.mul_(x).add_(a)

        if strength >= 1.0:
            return poly_y

        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
        # Blend in-place to reduce peak memory
        poly_y.mul_(strength).add_(ideal_y, alpha=(1.0 - strength))
        return poly_y


class TIA(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.degree: int = 0
        if self.config.tia_distortion_data_path is None:
            raise ValueError("TIA distortion data path is not set")
        if self.config.tia_distortion_polyfit_order is None:
            self.degree = get_ideal_degree(self.config.tia_distortion_data_path)
        else:
            self.degree = self.config.tia_distortion_polyfit_order
        coeffs = get_coeffs(self.config.tia_distortion_data_path, self.degree)
        coeff_tensor = torch.as_tensor(coeffs, dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)

        # Ideal (reference) linear coefficients a, b where y = a * x + b
        ideal_coeffs = _compute_linear_coeffs(
            self.config.tia_distortion_data_path,
            linearization=self.config.transfer_linearization,
        )
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength
        self.strength: float = float(self.config.tia_distortion_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strength = float(self.strength)
        if strength <= 0.0:
            ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
            return ideal_y

        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y.mul_(x).add_(a)

        if strength >= 1.0:
            return poly_y

        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
        # Blend in-place to reduce peak memory
        poly_y.mul_(strength).add_(ideal_y, alpha=(1.0 - strength))
        return poly_y


class MRM(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.phase_degree: int = 0
        self.field_degree: int = 0
        self.ler_variation = LER_variation(config)
        if self.config.mrm_amplitude_data_path is None:
            raise ValueError("MRM amplitude data path is not set")
        if self.config.mrm_amplitude_polyfit_order is None:
            self.field_degree = get_ideal_degree(
                self.config.mrm_amplitude_data_path,
                output_transform=_sqrt_clamped,
            )
        else:
            self.field_degree = self.config.mrm_amplitude_polyfit_order

        self.ph_in_min, self.ph_in_max, self.ph_out_min, self.ph_out_max = (
            get_io_ranges(self.config.mrm_phase_data_path)
        )
        if max(self.ph_out_min, self.ph_out_max) > 2 * np.pi:
            raise ValueError(
                "part of the MRM phase output is greater than 2*pi, please check the data to make sure it is in radians"
            )

        field_coeffs = get_coeffs(
            self.config.mrm_amplitude_data_path,
            self.field_degree,
            output_transform=_sqrt_clamped,
        )
        field_coeff_tensor = torch.as_tensor(field_coeffs, dtype=torch.float32)
        self.register_buffer("field_coeffs", field_coeff_tensor)

        # Ideal MRM amplitude coefficients. The CSV stores power in Watts, but
        # optical propagation uses field amplitude, so the ideal transfer is a
        # linear fit from voltage to sqrt(W).
        ideal_field_coeffs = _compute_linear_coeffs(
            self.config.mrm_amplitude_data_path,
            output_transform=_sqrt_clamped,
            linearization=self.config.transfer_linearization,
        )
        self.register_buffer(
            "ideal_field_coeffs",
            torch.as_tensor(ideal_field_coeffs, dtype=torch.float32),
        )

        # Distortion strength for the MRM amplitude fit. The source CSV stores
        # measured optical power, which is converted to field amplitude on load.
        self.field_strength: float = float(
            self.config.mrm_amplitude_distortion_strength
        )
        if self.config.mrm_phase_data_path is None:
            raise ValueError("MRM phase data path is not set")
        if self.config.mrm_phase_polyfit_order is None:
            self.phase_degree = get_ideal_degree(self.config.mrm_phase_data_path)
        else:
            self.phase_degree = self.config.mrm_phase_polyfit_order
        phase_coeffs = get_coeffs(self.config.mrm_phase_data_path, self.phase_degree)
        phase_coeff_tensor = torch.as_tensor(phase_coeffs, dtype=torch.float32)
        self.register_buffer("phase_coeffs", phase_coeff_tensor)

        # Distortion strength for phase
        self.phase_strength: float = float(self.config.mrm_phase_distortion_strength)

    def make_laser_scale(self, x: torch.Tensor) -> torch.Tensor | None:
        """Sample a per-shot (batch-wise) laser intensity scale factor.

        A single laser source is shared across all channels in a shot, so the
        sampled scale is constant across the channel dimension(s) and varies
        only across the batch dimension.

        The scale factor is modeled as log-normal with mean 1.0. The noise
        magnitude is specified by `config.laser_rin_db`, interpreted as a
        fractional RMS intensity noise in dB:
            `laser_rin_db = 20 * log10(rms_fraction)`.
        The returned value is an intensity scale. MRM transfer outputs field
        amplitude, so the field receives sqrt(scale).
        """
        rin_db = self.config.laser_rin_db
        if rin_db is None:
            return None
        rin_db = float(rin_db)
        if x.dim() < 1:
            return None
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        rms_fraction = 10 ** (rin_db / 20.0)
        sigma_log = math.sqrt(math.log1p(rms_fraction**2))
        eps = torch.randn(shape, device=x.device, dtype=x.dtype)
        return torch.exp(eps * sigma_log - 0.5 * (sigma_log**2))

    def apply_laser_power_gain(self, field_y: torch.Tensor) -> torch.Tensor:
        """Apply deterministic laser power gain to field amplitude."""
        gain = float(
            1.0
            if self.config.laser_power_gain is None
            else self.config.laser_power_gain
        )
        if gain == 1.0:
            return field_y
        return field_y * math.sqrt(gain)

    def forward(
        self, x: torch.Tensor, laser_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Field-amplitude response (Horner's rule). The configured MRM amplitude
        # data is converted from W to sqrt(W) when coefficients are loaded.
        field_strength = float(self.field_strength)
        if field_strength <= 0.0:
            field_y = self.ideal_field_coeffs[0] * x + self.ideal_field_coeffs[1]
        else:
            field_poly_y = torch.zeros_like(
                x, dtype=self.field_coeffs.dtype, device=x.device
            )
            for a in self.field_coeffs:
                field_poly_y.mul_(x).add_(a)
            if field_strength >= 1.0:
                field_y = field_poly_y
            else:
                field_ideal_y = (
                    self.ideal_field_coeffs[0] * x + self.ideal_field_coeffs[1]
                )
                field_poly_y.mul_(field_strength).add_(
                    field_ideal_y, alpha=(1.0 - field_strength)
                )
                field_y = field_poly_y

        if laser_scale is None:
            laser_scale = self.make_laser_scale(field_y)
        if laser_scale is not None:
            # RIN is defined on intensity/power; the field amplitude gets
            # sqrt(intensity scale).
            scale = laser_scale.to(device=field_y.device, dtype=field_y.dtype)
            field_y = field_y * torch.sqrt(scale)

        if self.config.ler_std_dev > 0:
            # LER variation is also a power-domain scale.
            ler_scale = self.ler_variation.generate_ler_matrix(
                field_y.shape[0],
                field_y.shape[1],
                device=field_y.device,
            ).to(dtype=field_y.dtype)
            field_y = field_y * torch.sqrt(ler_scale)

        field_y = self.apply_laser_power_gain(field_y)

        # Phase response
        phase_strength = float(self.phase_strength)
        if phase_strength <= 0.0:
            phase_y = torch.zeros_like(field_y)
        else:
            phase_poly_y = torch.zeros_like(
                x, dtype=self.phase_coeffs.dtype, device=x.device
            )
            for a in self.phase_coeffs:
                phase_poly_y.mul_(x).add_(a)
            phase_y = phase_poly_y.mul_(phase_strength)

        return torch.polar(field_y.clamp_min(0.0), phase_y)

    # Note: separate power/phase can be obtained as abs/angle of forward(x)


class LER_variation(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.dim = self.config.jtc_total_field
        self.ler_std_dev = self.config.ler_std_dev

    def generate_ler_matrix(self, batch: int, length: int, device=None) -> torch.Tensor:
        """
        Balanced splitter tree (Gaussian i.i.d. ratios) that:
        • Handles non-powers of two by building to the next power-of-two (m)
            and center-cropping the m leaves down to n.
        • Supports an arbitrary batch dimension.
        • Returns a tensor of shape (batch, n) whose rows sum to n.
        """
        m = 1 << (length - 1).bit_length()  # smallest 2^k ≥ n
        levels = int(math.log2(m))

        powers = m * torch.ones((batch, 1), device=device)  # start with 1 W

        for _ in range(levels):
            k = powers.size(1)

            ratios = torch.normal(
                0.5, self.ler_std_dev, size=(batch, k), device=device
            ).clamp(0, 1)

            left = ratios * powers
            right = (1.0 - ratios) * powers

            # Interleave: L1,R1,L2,R2,…  — works for any batch size, including 1
            new_powers = torch.empty((batch, k * 2), device=device)
            new_powers[:, 0::2] = left
            new_powers[:, 1::2] = right
            powers = new_powers  # (batch, 2k)

        # Center-crop from m leaves down to n leaves
        if length < m:
            start = (m - length) // 2
            powers = powers[:, start : start + length]

        return powers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ler_matrix = self.generate_ler_matrix(x.shape[0], x.shape[1], device=x.device)
        return torch.mul(x, ler_matrix)


class JTC(nn.Module):
    # Minimum plane size for the rfft readout path. The half-spectrum FFT
    # saving must outweigh the mirror/fold gathers it adds; measured on A100:
    # 256-wide planes gain ~12%, 48-wide planes lose ~15%.
    _RFFT_MIN_FIELD = 128

    def __init__(self, config: AppConfig):
        super(JTC, self).__init__()
        self.config = config
        self.driver = Driver(config)
        self.mrm = MRM(config)
        self.pd = PD(config)
        self.tia = TIA(config)
        self.lens = LensLegendre(config, length=config.jtc_total_field)
        self._pd_range_regularization_terms: list[torch.Tensor] = []
        driver_coeffs = _blend_poly_coeffs(
            self.driver.coeffs.cpu().numpy(),
            self.driver.ideal_coeffs.cpu().numpy(),
            self.driver.strength,
        )
        mrm_field_coeffs = _blend_poly_coeffs(
            self.mrm.field_coeffs.cpu().numpy(),
            self.mrm.ideal_field_coeffs.cpu().numpy(),
            self.mrm.field_strength,
        )
        input_field_coeffs = _compose_poly_coeffs(mrm_field_coeffs, driver_coeffs)
        self.register_buffer(
            "input_field_coeffs",
            torch.as_tensor(input_field_coeffs, dtype=torch.float32),
            persistent=False,
        )
        if float(self.mrm.phase_strength) > 0.0:
            input_phase_coeffs = _compose_poly_coeffs(
                self.mrm.phase_coeffs.cpu().numpy(),
                driver_coeffs,
                scale=self.mrm.phase_strength,
            )
        else:
            input_phase_coeffs = np.asarray([0.0], dtype=np.float32)
        self.register_buffer(
            "input_phase_coeffs",
            torch.as_tensor(input_phase_coeffs, dtype=torch.float32),
            persistent=False,
        )
        self._build_input_transfer_luts(input_field_coeffs, input_phase_coeffs)
        self.input_length = config.input_length
        self.kernel_length = config.kernel_length
        self.jtc_separation = config.jtc_separation
        self.jtc_total_field = config.jtc_total_field
        self.loss = float(config.loss)

        # Calculate output_length if not specified
        # Default: full correlation length (M+N-1)
        # Note: Some outputs may have autocorrelation contamination depending on config
        # Use effective_stride (computed below) for clean stitching
        if config.output_length is None:
            self.output_length = self.input_length + self.kernel_length - 1
        else:
            self.output_length = config.output_length

        # Validate that configuration is feasible
        if (
            self.input_length + self.kernel_length + self.jtc_separation
            > self.jtc_total_field
        ):
            raise ValueError(
                f"JTC total field ({self.jtc_total_field}) is too small for "
                f"input_length ({self.input_length}) + kernel_length ({self.kernel_length}) + "
                f"separation ({self.jtc_separation}) = {self.input_length + self.kernel_length + self.jtc_separation}"
            )

        # Analyze contamination profile using cycle planner
        total_outputs, clean_valid_outputs, effective_stride = (
            compute_contamination_profile(
                self.input_length,
                self.kernel_length,
                self.jtc_total_field,
                self.jtc_separation,
            )
        )
        self.total_correlation_outputs = total_outputs
        self.clean_valid_outputs = clean_valid_outputs
        self.effective_stride = effective_stride
        self._correlation_start = self._build_correlation_start()
        self.register_buffer(
            "_correlation_indices",
            self._build_correlation_indices(),
            persistent=False,
        )
        # For real input planes |FFT[k]| == |FFT[N-k]|, so the full-plane
        # power spectrum is a gather of the rfft half-spectrum.
        idx = torch.arange(self.jtc_total_field)
        self.register_buffer(
            "_rfft_mirror_indices",
            torch.minimum(idx, self.jtc_total_field - idx),
            persistent=False,
        )

        # Report contamination status (only if significant)
        # Note: clean_valid_outputs is the number of clean outputs in the valid convolution region
        # For valid conv, we use M-N+1 outputs from the M+N-1 correlation
        num_valid_outputs = self.input_length - self.kernel_length + 1
        if num_valid_outputs > 0:
            valid_contamination_percent = 100 * (
                1 - clean_valid_outputs / num_valid_outputs
            )
            if valid_contamination_percent > 10:
                import warnings

                warnings.warn(
                    f"JTC config has {valid_contamination_percent:.1f}% contamination in valid outputs: "
                    f"M={self.input_length}, N={self.kernel_length}, "
                    f"plane={self.jtc_total_field}, sep={self.jtc_separation}. "
                    f"Clean valid outputs: {clean_valid_outputs}/{num_valid_outputs}, "
                    f"Effective stride: {effective_stride}",
                    UserWarning,
                    stacklevel=2,
                )

        # Ordered list of available stage names
        self.stage_order = [
            "input_plane",
            "input_plane_quant",
            "input_plane_driver",
            "input_plane_mrm_phase",
            "input_plane_mrm_amp",
            "jps_raw",
            "jps_pd_input",
            "jps_pd",
            "jps_tia",
            "jps_scale",
            "jps_quant",
            "jps_driver",
            "jps_mrm_phase",
            "jps_mrm_amp",
            "output_raw",
            "output_pd",
            "output_tia",
            "output_scale",
            "output_scale_slice",
            "output_quant",
            "output_quant_slice",
            "output_slice",
        ]

        # Optionally fuse the paired-shot pipeline. The pipeline is dozens of
        # small elementwise ops between two FFTs, so kernel fusion dominates
        # the training step time when enabled.
        self._paired_pipeline = self._make_paired_pipeline()

    def _make_paired_pipeline(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if bool(getattr(self.config, "compile_jtc", False)):
            raise_dynamo_recompile_limit()
            return torch.compile(self._paired_shot_pipeline, dynamic=True)
        return self._paired_shot_pipeline

    def __getstate__(self):
        # A torch.compile-wrapped bound method is not picklable; rebuild it on
        # load instead of serializing it (full-model torch.save / deepcopy).
        state = dict(self.__dict__)
        state.pop("_paired_pipeline", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._paired_pipeline = self._make_paired_pipeline()

    def _build_input_transfer_luts(
        self,
        input_field_coeffs: np.ndarray,
        input_phase_coeffs: np.ndarray,
    ) -> None:
        """Precompute per-DAC-level transfer tables for the input modulators.

        With a `dac_bits`-bit converter the modulator input takes only
        `2**dac_bits` values, so the composed driver->MRM polynomial (and its
        derivative, needed for the STE backward) collapses to a table lookup.

        For low-degree transfers (ideal linear responses) the Horner loop is
        already cheaper than the index/gather sequence, so the LUT only kicks
        in once the composed polynomial degree crosses the breakeven point.
        """
        field_degree = len(input_field_coeffs) - 1
        phase_degree = (
            len(input_phase_coeffs) - 1 if float(self.mrm.phase_strength) > 0.0 else 0
        )
        self._input_lut_enabled = (
            self.config.dac_bits is not None and max(field_degree, phase_degree) >= 3
        )
        if not self._input_lut_enabled:
            return
        num_levels = 1 << int(self.config.dac_bits)
        levels = torch.arange(num_levels, dtype=torch.float32) / float(
            max(num_levels - 1, 1)
        )

        def eval_table(coeffs: np.ndarray) -> torch.Tensor:
            return self._eval_poly(torch.as_tensor(coeffs, dtype=torch.float32), levels)

        self.register_buffer(
            "input_field_lut", eval_table(input_field_coeffs), persistent=False
        )
        self.register_buffer(
            "input_field_lut_grad",
            eval_table(_poly_derivative_coeffs(input_field_coeffs)),
            persistent=False,
        )
        if float(self.mrm.phase_strength) > 0.0:
            self.register_buffer(
                "input_phase_lut", eval_table(input_phase_coeffs), persistent=False
            )
            self.register_buffer(
                "input_phase_lut_grad",
                eval_table(_poly_derivative_coeffs(input_phase_coeffs)),
                persistent=False,
            )

    def pd_range_bounds(self) -> tuple[float | None, float | None]:
        lower = self.config.pd_range_regularization_min_w
        upper = self.config.pd_range_regularization_max_w
        if lower is None:
            lower = float(self.pd.input_min)
        if upper is None:
            upper = float(self.pd.input_max)
        return lower, upper

    def reset_pd_range_regularization(self) -> None:
        self._pd_range_regularization_terms = []

    def pd_range_regularization_loss(self) -> torch.Tensor | None:
        if not self._pd_range_regularization_terms:
            return None
        total = self._pd_range_regularization_terms[0]
        for term in self._pd_range_regularization_terms[1:]:
            total = total + term
        return total

    def _accumulate_pd_range_regularization(self, x: torch.Tensor) -> None:
        weight = float(self.config.pd_range_regularization_weight or 0.0)
        if weight <= 0.0 or not torch.is_grad_enabled() or not x.requires_grad:
            return

        lower, upper = self.pd_range_bounds()
        if lower is None and upper is None:
            return

        violation = torch.zeros_like(x)
        if lower is not None:
            violation = violation + torch.relu(float(lower) - x)
        if upper is not None:
            violation = violation + torch.relu(x - float(upper))

        if lower is not None and upper is not None:
            span = abs(float(upper) - float(lower))
        else:
            bound = float(upper if upper is not None else lower)
            span = max(abs(bound), 1.0)
        span = max(span, 1e-24)
        self._pd_range_regularization_terms.append((violation / span).pow(2).mean())

    def weighted_pd_range_regularization_loss(self) -> torch.Tensor | None:
        loss = self.pd_range_regularization_loss()
        if loss is None:
            return None
        return loss * float(self.config.pd_range_regularization_weight or 0.0)

    def _eval_poly(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n = coeffs.numel()
        x_eval = x if x.dtype == coeffs.dtype else x.to(dtype=coeffs.dtype)
        if n == 1:
            return torch.full_like(x_eval, coeffs[0], dtype=coeffs.dtype)
        if (
            self.config.jtc_fuse_pointwise
            and x_eval.is_cuda
            and x_eval.dtype in (torch.float32, torch.float64)
            and not coeffs.requires_grad
        ):
            from onn_fused import horner

            return horner(x_eval, coeffs)
        y = coeffs[0] * x_eval + coeffs[1]
        for a in coeffs[2:]:
            y = y * x_eval + a
        return y

    def _pd_transfer_raw(self, x: torch.Tensor) -> torch.Tensor:
        pd_noise_w = float(self.config.pd_noise_w or 0.0)
        if pd_noise_w > 0:
            x = x + torch.randn_like(x) * pd_noise_w
        self._accumulate_pd_range_regularization(x)
        x = apply_pd_input_guard(x, self.config)

        pd_strength = float(self.pd.strength)
        if pd_strength <= 0.0:
            ideal_y = self.pd.ideal_coeffs[0] * x + self.pd.ideal_coeffs[1]
            return ideal_y

        pd_poly_y = self._eval_poly(self.pd.coeffs, x)
        if pd_strength >= 1.0:
            return pd_poly_y

        pd_ideal_y = self.pd.ideal_coeffs[0] * x + self.pd.ideal_coeffs[1]
        pd_poly_y.mul_(pd_strength).add_(pd_ideal_y, alpha=(1.0 - pd_strength))
        return pd_poly_y

    def _tia_transfer_raw(self, x: torch.Tensor) -> torch.Tensor:
        # PD->TIA bias stage: shifts the operating point into the TIA's
        # characterized input domain (see AppConfig.tia_input_bias).
        bias = float(getattr(self.config, "tia_input_bias", 0.0) or 0.0)
        if bias != 0.0:
            x = x + bias
        tia_strength = float(self.tia.strength)
        if tia_strength <= 0.0:
            ideal_y = self.tia.ideal_coeffs[0] * x + self.tia.ideal_coeffs[1]
            return ideal_y

        tia_poly_y = self._eval_poly(self.tia.coeffs, x)
        if tia_strength >= 1.0:
            return tia_poly_y

        tia_ideal_y = self.tia.ideal_coeffs[0] * x + self.tia.ideal_coeffs[1]
        tia_poly_y.mul_(tia_strength).add_(tia_ideal_y, alpha=(1.0 - tia_strength))
        return tia_poly_y

    def _modulate_field(
        self,
        field_y: torch.Tensor,
        phase_y: torch.Tensor | None,
        laser_scale: torch.Tensor | None,
        as_complex: bool = True,
    ) -> torch.Tensor:
        """Apply laser/LER scaling and combine field amplitude with phase.

        With zero phase the modulated field is purely real; callers that can
        consume a real tensor (the rfft fast path) pass ``as_complex=False``
        to skip materializing the imaginary half.
        """
        if laser_scale is None:
            laser_scale = self.mrm.make_laser_scale(field_y)
        if laser_scale is not None:
            scale = laser_scale.to(device=field_y.device, dtype=field_y.dtype)
            field_y = field_y * torch.sqrt(scale)

        if self.config.ler_std_dev > 0:
            ler_scale = self.mrm.ler_variation.generate_ler_matrix(
                field_y.shape[0],
                field_y.shape[1],
                device=field_y.device,
            ).to(dtype=field_y.dtype)
            field_y = field_y * torch.sqrt(ler_scale)

        field_y = self.mrm.apply_laser_power_gain(field_y)

        field_y = field_y.clamp_min(0.0)
        if phase_y is None:
            if not as_complex:
                return field_y
            # Avoid polar/PolarBackward only for the exact zero-phase case.
            return torch.complex(field_y, torch.zeros_like(field_y))
        return torch.polar(field_y, phase_y)

    def _output_transfer(self, x: torch.Tensor) -> torch.Tensor:
        """Apply field-amplitude -> power -> PD -> TIA -> ADC transfer."""
        # Detector input is field amplitude. Square-law detection converts it
        # to optical power in Watts before PD noise/clamp/transfer functions.
        return self._detector_power_transfer((x * x) * self.loss)

    def _detector_power_transfer(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PD -> TIA -> ADC transfer to optical power in Watts."""
        if self.config.scale_output == "pd":
            x = self.scale_to_range(x, 1e-6, 1e-5)

        # PD/TIA transfer curves are analog responses, not normalized rails.
        # DAC/ADC quantizers are responsible for code-range clipping.
        pd_y = self._pd_transfer_raw(x)
        x = self._tia_transfer_raw(pd_y)
        if self.config.scale_output == "adc":
            x = x / x.max().clamp_min(1e-12)
        x = converter_quantize_ste(
            x,
            self.config.adc_bits,
            self.config.converter_clamp_grad,
        )
        return x

    def input_distortion(
        self,
        x: torch.Tensor,
        laser_scale: torch.Tensor | None = None,
        as_complex: bool = True,
    ) -> torch.Tensor:
        """Apply DAC quantization, driver response, and MRM modulation."""
        if self._input_lut_enabled:
            field_y = quantized_transfer_lut(
                x,
                self.input_field_lut,
                self.input_field_lut_grad,
                self.config.converter_clamp_grad,
            )
            if float(self.mrm.phase_strength) > 0.0:
                phase_y = quantized_transfer_lut(
                    x,
                    self.input_phase_lut,
                    self.input_phase_lut_grad,
                    self.config.converter_clamp_grad,
                )
            else:
                phase_y = None
            return self._modulate_field(
                field_y, phase_y, laser_scale, as_complex=as_complex
            )
        x = converter_quantize_ste(
            x,
            self.config.dac_bits,
            self.config.converter_clamp_grad,
        )
        field_y = self._eval_poly(self.input_field_coeffs, x)
        if float(self.mrm.phase_strength) > 0.0:
            phase_y = self._eval_poly(self.input_phase_coeffs, x)
        else:
            phase_y = None
        return self._modulate_field(
            field_y, phase_y, laser_scale, as_complex=as_complex
        )

    def output_distortion(self, x: torch.Tensor) -> torch.Tensor:
        """Convert field amplitude to power, then apply PD, TIA, and ADC."""
        return self._output_transfer(x)

    def fft_and_magnitude(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Apply FFT, optional lens distortion, then take magnitude.

        Tensors stay in native FFT order. Correlation extraction maps the old
        centered-plane indices into native order, avoiding full-plane rolls.
        """
        x = torch.fft.fft(x)
        if not self.lens.is_identity():
            x = self.lens(x)
        if indices is not None:
            if isinstance(indices, slice):
                x = x[..., indices]
            else:
                x = x.index_select(-1, indices)
        x = torch.abs(x)
        if x.dtype == torch.float16:
            x = x.float()
        return x

    def fft_and_power(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Apply FFT, optional lens distortion, then take squared magnitude.

        Square-law detection needs |FFT|^2; computing it directly avoids the
        sqrt in `torch.abs` that the detector would immediately square again.

        A real input plane (zero MRM phase) with an ideal lens uses rfft and
        recovers the full plane through DFT conjugate symmetry — exact, and
        it halves the FFT work while skipping all complex-dtype glue. Complex
        modulation (phase distortion or lens) takes the full c2c path.
        """
        if (
            not torch.is_complex(x)
            and self.lens.is_identity()
            and self.jtc_total_field >= self._RFFT_MIN_FIELD
        ):
            power = complex_abs_squared(torch.fft.rfft(x))
            if indices is None:
                folded = self._rfft_mirror_indices
            else:
                if isinstance(indices, slice):
                    indices = torch.arange(indices.start, indices.stop, device=x.device)
                folded = torch.minimum(indices, self.jtc_total_field - indices)
            return power.index_select(-1, folded)
        x = torch.fft.fft(x)
        if not self.lens.is_identity():
            x = self.lens(x)
        if indices is not None:
            if isinstance(indices, slice):
                x = x[..., indices]
            else:
                x = x.index_select(-1, indices)
        return complex_abs_squared(x)

    def first_detector_readout(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the first detector/JPS readout."""
        power = self.fft_and_power(x) * (self.loss / float(self.jtc_total_field))
        return self._detector_power_transfer(power)

    def _ideal_sqrt_readout(self, x: torch.Tensor) -> torch.Tensor:
        return sqrt_nonnegative_with_finite_grad(x)

    def final_detector_readout(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Compute final detector readout and recover ideal field amplitude."""
        power = self.fft_and_power(x, indices=indices) * (
            self.loss / float(self.jtc_total_field)
        )
        x = self._detector_power_transfer(power)
        return self._ideal_sqrt_readout(x)

    def _build_correlation_start(self) -> int:
        plane_size = self.jtc_total_field
        sep = self.jtc_separation
        N = self.kernel_length

        same_start = plane_size // 2 + sep + N // 2
        if self.output_length == self.input_length:
            same_start += 1
        return int(same_start)

    def _build_correlation_indices(self) -> torch.Tensor:
        """Compute native FFT-order indices for extracting convolution output.

        The physical extraction formula is expressed in the historical centered
        plane layout. The tensors now remain in native FFT order, so convert the
        centered indices by half a field before indexing.
        """
        geometry = ApertureGeometry(
            self.input_length,
            self.kernel_length,
            self.jtc_total_field,
            self.jtc_separation,
            self.output_length,
        )
        return torch.tensor(geometry.extraction_indices, dtype=torch.long)

    def compute_correlation_indices(self, device) -> torch.Tensor:
        return self._correlation_indices.to(device=device)

    def _correlation_indices_for_length(
        self,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if length == self.output_length:
            return self.compute_correlation_indices(device)
        start = self._correlation_start
        shifted_indices = torch.arange(
            start,
            start + length,
            device=device,
        )
        native_offset = (self.jtc_total_field + 1) // 2
        return (shifted_indices + native_offset) % self.jtc_total_field

    def _correlation_slice_for_length(self, length: int) -> slice | None:
        start = self._correlation_start
        native_offset = (self.jtc_total_field + 1) // 2
        native_start = (start + native_offset) % self.jtc_total_field
        native_end = native_start + int(length)
        if native_end <= self.jtc_total_field:
            return slice(native_start, native_end)
        return None

    def _can_slice_final_detector(self) -> bool:
        return (
            self.config.scale_output == "none"
            and float(self.config.pd_noise_w or 0.0) <= 0.0
        )

    def extract_correlation(
        self,
        output_plane: torch.Tensor,
        output_length: int | None = None,
    ) -> torch.Tensor:
        length = self.output_length if output_length is None else int(output_length)
        selector = self._correlation_slice_for_length(length)
        if selector is not None:
            return output_plane[..., selector]
        indices = self._correlation_indices_for_length(length, output_plane.device)
        return output_plane.index_select(-1, indices)

    def forward_paired(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Run paired JTC shots.

        `signal[i]` is correlated with `kernel[i]`. This avoids constructing a
        Cartesian product inside `forward()` when the caller already has the
        exact shot list.
        """
        if signal.dim() != 2 or kernel.dim() != 2:
            raise ValueError("forward_paired expects signal and kernel to be 2D")
        if signal.shape[0] != kernel.shape[0]:
            raise ValueError("signal and kernel must have the same shot count")
        if signal.shape[-1] != self.input_length:
            raise ValueError(
                f"signal width must be {self.input_length}, got {signal.shape[-1]}"
            )
        if kernel.shape[-1] != self.kernel_length:
            raise ValueError(
                f"kernel width must be {self.kernel_length}, got {kernel.shape[-1]}"
            )

        return self._paired_pipeline(signal, kernel)

    def _paired_shot_pipeline(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Distort, correlate, and read out a batch of paired shots.

        This is the training hot path; `compile_jtc` wraps it in torch.compile.
        """
        laser_scale = self.mrm.make_laser_scale(
            torch.empty(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)
        )
        # as_complex=False keeps zero-phase planes real so the readouts can
        # use the rfft fast path; with phase distortion on, the modulated
        # plane comes back complex and the full c2c path runs as before.
        signal_distorted = self.input_distortion(
            signal, laser_scale=laser_scale, as_complex=False
        )
        kernel_distorted = self.input_distortion(
            kernel, laser_scale=laser_scale, as_complex=False
        )
        input_plane = self.build_input_plane(signal_distorted, kernel_distorted)

        jps = self.first_detector_readout(input_plane)
        jps = converter_quantize_ste(
            jps,
            self.config.fourier_plane_bits,
            self.config.converter_clamp_grad,
        )

        laser_scale_2 = self.mrm.make_laser_scale(
            torch.empty(jps.shape[0], 1, device=jps.device, dtype=jps.dtype)
        )
        jps_distorted = self.input_distortion(
            jps, laser_scale=laser_scale_2, as_complex=False
        )

        if self._can_slice_final_detector():
            selector = self._correlation_slice_for_length(self.output_length)
            if selector is None:
                selector = self.compute_correlation_indices(jps_distorted.device)
            return self.final_detector_readout(jps_distorted, indices=selector)

        output_plane = self.final_detector_readout(jps_distorted)
        return self.extract_correlation(output_plane)

    def build_input_plane(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Build JTC input plane from distorted signal and kernel.

        Args:
            signal: Distorted signal tensor (already passed through input_distortion)
            kernel: Distorted kernel tensor (already passed through input_distortion)

        Returns:
            Complex input plane with signal and kernel placed at correct positions
        """
        B = signal.shape[0]
        M = signal.shape[-1]
        N = kernel.shape[-1]

        # Validation
        if M > self.input_length:
            raise ValueError(
                f"Signal length ({M}) is greater than configured input_length ({self.input_length})"
            )
        if N > self.kernel_length:
            raise ValueError(
                f"Kernel length ({N}) is greater than configured kernel_length ({self.kernel_length})"
            )
        if M + N + self.jtc_separation > self.jtc_total_field:
            raise ValueError(
                f"Not enough JTC field: {M} + {N} + {self.jtc_separation} > {self.jtc_total_field}"
            )

        # Calculate positions
        kernel_start = 0
        kernel_end = kernel_start + N
        signal_start = kernel_end + self.jtc_separation
        signal_end = signal_start + M

        plane_dtype = torch.promote_types(signal.dtype, kernel.dtype)
        kernel = kernel.to(dtype=plane_dtype)
        signal = signal.to(dtype=plane_dtype)
        gap = torch.zeros(
            B,
            self.jtc_separation,
            dtype=plane_dtype,
            device=signal.device,
        )
        tail_length = self.jtc_total_field - signal_end
        if tail_length > 0:
            tail = torch.zeros(B, tail_length, dtype=plane_dtype, device=signal.device)
            return torch.cat((kernel, gap, signal, tail), dim=-1)
        return torch.cat((kernel, gap, signal), dim=-1)

    def scale_to_range(
        self, tensor: torch.Tensor, min_val: float = -30, max_val: float = -20
    ) -> torch.Tensor:
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        denom = (tensor_max - tensor_min).clamp_min(1e-12)
        scaled = (tensor - tensor_min) / denom  # Scale to [0,1]
        return scaled * (max_val - min_val) + min_val  # Scale to [min_val, max_val]

    def compute_stage_tensors(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        stages: tuple[str, ...] | None = None,
        sample_index: int | None = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute intermediate tensors for the requested pipeline stages.

        By default returns one 1D tensor per stage, suitable for plotting. Pass
        ``sample_index=None`` to keep the full batch for distributional stats.
        Complex tensors are converted to magnitudes where needed.
        """
        if stages is None:
            stages = tuple(self.stage_order)

        results: dict[str, torch.Tensor] = {}

        def stage_tensor(x: torch.Tensor) -> torch.Tensor:
            detached = x.detach()
            if sample_index is None:
                return detached
            return detached[int(sample_index), :]

        B = signal.shape[0]
        M = signal.shape[-1]
        N = kernel.shape[-1]
        if M > self.input_length:
            raise ValueError(
                f"Signal length ({M}) is greater than configured input_length ({self.input_length})"
            )
        if N > self.kernel_length:
            raise ValueError(
                f"Kernel length ({N}) is greater than configured kernel_length ({self.kernel_length})"
            )
        if M + N + self.jtc_separation > self.jtc_total_field:
            raise ValueError(
                f"Not enough JTC field: {M} + {N} + {self.jtc_separation} > {self.jtc_total_field}"
            )

        # Indices for placement
        kernel_start = 0
        kernel_end = kernel_start + N
        signal_start = kernel_end + self.jtc_separation
        signal_end = signal_start + M

        # 1) Quantize (DAC) and place into full input field
        kernel_quant = converter_quantize_ste(
            kernel,
            self.config.dac_bits,
            self.config.converter_clamp_grad,
        )
        signal_quant = converter_quantize_ste(
            signal,
            self.config.dac_bits,
            self.config.converter_clamp_grad,
        )
        plane_quant = torch.zeros(
            B, self.jtc_total_field, dtype=torch.float32, device=signal.device
        )
        plane_quant[..., kernel_start:kernel_end] = kernel_quant
        plane_quant[..., signal_start:signal_end] = signal_quant
        if "input_plane_quant" in stages:
            results["input_plane_quant"] = stage_tensor(plane_quant)

        # 2) Driver on the full field
        plane_driver = self.driver(plane_quant)
        if "input_plane_driver" in stages:
            results["input_plane_driver"] = stage_tensor(plane_driver)

        # 3) MRM on the full, driver-processed field
        plane_mrm = self.mrm(plane_driver)
        # Store overall input plane magnitude
        if "input_plane" in stages:
            results["input_plane"] = stage_tensor(torch.abs(plane_mrm))
        # Store separate amplitude and phase components (real-valued)
        if "input_plane_mrm_amp" in stages:
            results["input_plane_mrm_amp"] = stage_tensor(torch.abs(plane_mrm))
        if "input_plane_mrm_phase" in stages:
            results["input_plane_mrm_phase"] = stage_tensor(torch.angle(plane_mrm))

        B = signal.shape[0]
        M = signal.shape[-1]
        N = kernel.shape[-1]
        kernel_start = 0
        kernel_end = kernel_start + N
        signal_start = kernel_end + self.jtc_separation
        signal_end = signal_start + M
        # For propagation, reuse the complex input plane computed above
        input_plane_full = plane_mrm

        # Fourier plane (native-order FFT + optional lens distortion)
        jps_mag = self.fft_and_magnitude(input_plane_full)
        if "jps_raw" in stages:
            results["jps_raw"] = stage_tensor(jps_mag)

        # Output distortion (first pass). The FFT produces field amplitude;
        # square-law detection converts it to optical power in Watts.
        jps_field = jps_mag / math.sqrt(float(self.jtc_total_field))
        jps_base = (jps_field * jps_field) * self.loss

        if self.config.scale_output == "pd":
            jps_base = self.scale_to_range(jps_base, 1e-6, 1e-5)
        if "jps_pd_input" in stages:
            results["jps_pd_input"] = stage_tensor(jps_base)

        jps_pd = self._pd_transfer_raw(jps_base)
        if "jps_pd" in stages:
            results["jps_pd"] = stage_tensor(jps_pd)

        jps_tia = self._tia_transfer_raw(jps_pd)
        if "jps_tia" in stages:
            results["jps_tia"] = stage_tensor(jps_tia)

        if self.config.scale_output == "adc":
            jps_scaled = jps_tia / jps_tia.max().clamp_min(1e-12)
        else:
            jps_scaled = jps_tia
        if "jps_scale" in stages:
            results["jps_scale"] = stage_tensor(jps_scaled)

        # Quantize (Fourier plane bits)
        jps_quant = converter_quantize_ste(
            jps_scaled,
            self.config.fourier_plane_bits,
            self.config.converter_clamp_grad,
        )
        if "jps_quant" in stages:
            results["jps_quant"] = stage_tensor(jps_quant)

        # Input distortion again before inverse FFT (second pass). DAC
        # quantization is applied first for this JTC pipeline.
        jps_dac = converter_quantize_ste(
            jps_quant,
            self.config.dac_bits,
            self.config.converter_clamp_grad,
        )
        jps_driver = self.driver(jps_dac)
        if "jps_driver" in stages:
            results["jps_driver"] = stage_tensor(jps_driver)
        jps_complex = self.mrm(jps_driver)
        jps_amp = torch.abs(jps_complex)
        jps_phase = torch.angle(jps_complex)
        if "jps_mrm_amp" in stages:
            results["jps_mrm_amp"] = stage_tensor(jps_amp)
        if "jps_mrm_phase" in stages:
            results["jps_mrm_phase"] = stage_tensor(jps_phase)
        # jps_complex already includes both components

        # Back to detector plane
        output_mag = self.fft_and_magnitude(jps_complex)
        output_field = output_mag / math.sqrt(float(self.jtc_total_field))
        output_raw = (output_field * output_field) * self.loss
        if "output_raw" in stages:
            results["output_raw"] = stage_tensor(output_raw)

        if self.config.scale_output == "pd":
            output_raw = self.scale_to_range(output_raw, 1e-6, 1e-5)

        out_pd = self._pd_transfer_raw(output_raw)
        if "output_pd" in stages:
            results["output_pd"] = stage_tensor(out_pd)

        out_tia = self._tia_transfer_raw(out_pd)
        if "output_tia" in stages:
            results["output_tia"] = stage_tensor(out_tia)

        if self.config.scale_output == "adc":
            out_scaled = out_tia / out_tia.max().clamp_min(1e-12)
        else:
            out_scaled = out_tia

        same_indices = self.compute_correlation_indices(jps_complex.device)
        if "output_scale" in stages:
            results["output_scale"] = stage_tensor(out_scaled)
        if "output_scale_slice" in stages:
            results["output_scale_slice"] = stage_tensor(out_scaled[..., same_indices])

        out_quant = converter_quantize_ste(
            out_scaled,
            self.config.adc_bits,
            self.config.converter_clamp_grad,
        )
        if "output_quant" in stages:
            results["output_quant"] = stage_tensor(out_quant)
        if "output_quant_slice" in stages:
            results["output_quant_slice"] = stage_tensor(out_quant[..., same_indices])

        output_slice = sqrt_nonnegative_with_finite_grad(out_quant)[..., same_indices]
        if "output_slice" in stages:
            results["output_slice"] = stage_tensor(output_slice)

        return results

    def forward(self, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Joint Transform Correlator forward pass.

        Pipeline:
        1. Input distortion (DAC quant + driver + MRM + optional laser_rin_db)
        2. FFT to Fourier plane
        3. Detector readout (field amplitude -> power + PD + TIA + ADC quant)
        4. Fourier plane quantization (fourier_plane_bits)
        5. Input distortion again (DAC quant + driver + MRM + optional laser_rin_db)
        6. FFT to detector plane
        7. Detector readout again (field amplitude -> power + PD + TIA + ADC quant)
        8. Index selection

        Args:
            signal: Input signal tensor (B, H, 1, W)
            kernel: Kernel weights tensor (Cout, W)

        Returns:
            Correlation output tensor (B, H, Cout, W)
        """
        direct_2d_signal = signal.dim() == 2
        if direct_2d_signal:
            signal = signal.unsqueeze(1).unsqueeze(2)
        if signal.dim() != 4:
            raise ValueError("JTC.forward expects signal shape (B, H, 1, W) or (B, W)")
        if kernel.dim() != 2:
            raise ValueError("JTC.forward expects kernel shape (Cout, W)")

        B, H = int(signal.shape[0]), int(signal.shape[1])
        Cout = int(kernel.shape[0])
        reps_per_batch = int(H * Cout)
        # Reshape inputs for batch processing
        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size_for_jtc = (
            signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        )
        signal_reshaped = signal_full.reshape(batch_size_for_jtc, self.input_length)
        kernel_reshaped = kernel_full.reshape(batch_size_for_jtc, self.kernel_length)

        # Step 1: Input distortion (signal & kernel)
        # Laser noise is global per shot: sample once per batch element and
        # broadcast across all MRM channels (including all output channels and
        # spatial rows in this JTC call).
        laser_scale = self.mrm.make_laser_scale(
            torch.empty(
                B, 1, device=signal_reshaped.device, dtype=signal_reshaped.dtype
            )
        )
        if laser_scale is not None:
            laser_scale = laser_scale.reshape(B, 1).repeat_interleave(
                reps_per_batch, dim=0
            )
        signal_distorted = self.input_distortion(
            signal_reshaped, laser_scale=laser_scale
        )
        kernel_distorted = self.input_distortion(
            kernel_reshaped, laser_scale=laser_scale
        )
        input_plane = self.build_input_plane(signal_distorted, kernel_distorted)

        # Step 2/3: FFT to Fourier plane and first detector readout
        jps = self.first_detector_readout(input_plane)

        # Step 4: Fourier plane quantization
        jps = converter_quantize_ste(
            jps,
            self.config.fourier_plane_bits,
            self.config.converter_clamp_grad,
        )

        # Step 5: Input distortion (DAC quantization + driver/MRM)
        # Second pass occurs at a different time, so re-sample laser noise from
        # the same distribution (do not reuse the first-pass sample).
        laser_scale_2 = self.mrm.make_laser_scale(
            torch.empty(B, 1, device=jps.device, dtype=jps.dtype)
        )
        if laser_scale_2 is not None:
            laser_scale_2 = laser_scale_2.reshape(B, 1).repeat_interleave(
                reps_per_batch, dim=0
            )
        jps_distorted = self.input_distortion(jps, laser_scale=laser_scale_2)

        # Step 6: FFT to detector plane
        output_length = self.output_length
        if direct_2d_signal and self.config.output_length is None:
            output_length = self.input_length
        if self._can_slice_final_detector():
            indices = self._correlation_indices_for_length(
                output_length,
                jps_distorted.device,
            )
            output = self.final_detector_readout(jps_distorted, indices=indices)
        else:
            output_plane = self.final_detector_readout(jps_distorted)
            output = self.extract_correlation(output_plane, output_length=output_length)

        # Reshape output
        output_reshaped = output.reshape(
            signal_full.shape[0],
            signal_full.shape[1],
            signal_full.shape[2],
            output_length,
        )
        return output_reshaped
