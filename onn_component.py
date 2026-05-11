import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from jtc_cycle_planner import compute_contamination_profile
from onn_config import AppConfig
from onn_math import sqrt_nonnegative_with_finite_grad
from onn_quantization import quantize_ste


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


def _compute_linear_coeffs(csv_file: str) -> np.ndarray:
    """Compute coefficients a, b for y = a * x + b using first and last data points."""
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    # Sort by x to get true first and last in domain
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    x_first, x_last = x_sorted[0], x_sorted[-1]
    y_first, y_last = y_sorted[0], y_sorted[-1]
    if x_last == x_first:
        raise ValueError("Input points for ideal linear interpolation are identical.")
    a = (y_last - y_first) / (x_last - x_first)
    b = y_first - a * x_first
    return np.array([a, b], dtype=np.float32)


def _compute_quadratic_coeffs(csv_file: str) -> np.ndarray:
    """Compute coefficients a, b for y = a * x**2 + b using first and last data points."""
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    # Sort by x to get true first and last in domain
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    x_first, x_last = x_sorted[0], x_sorted[-1]
    y_first, y_last = y_sorted[0], y_sorted[-1]
    if x_last**2 == x_first**2:
        raise ValueError(
            "Input points for ideal quadratic interpolation are identical."
        )
    a = (y_last - y_first) / (x_last**2 - x_first**2)
    b = y_first - a * x_first**2
    return np.array([a, b], dtype=np.float32)


def calculate_aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Calculate AIC for polynomial regression"""
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = 2 * n_params - 2 * log_likelihood
    return aic


def get_ideal_degree(csv_file: str, max_degree: int = 10) -> int:
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

    last_aic = float("inf")
    best_degree: int | None = None
    best_aic = float("inf")

    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        n_params = degree + 1  # coefficients + intercept
        aic = calculate_aic(y, y_pred, n_params)
        # print(f"degree: {degree}, r2: {r2}, aic: {aic}")
        if aic < best_aic:
            best_aic = aic
            best_degree = degree

        if r2 > 0.9995:
            return degree
        elif aic > last_aic:
            # Stop if AIC worsens; prefer previous degree if available
            return max(1, degree - 1)
        else:
            last_aic = aic

    # Fallback to the best degree seen instead of returning a non-int sentinel
    return best_degree if best_degree is not None else 1


def get_io_ranges(csv_file: str) -> tuple[float, float, float, float]:
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    return x.min(), x.max(), y.min(), y.max()


def get_coeffs(csv_file: str, degree: int) -> np.ndarray:
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
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
        ideal_coeffs = _compute_linear_coeffs(self.config.driver_distortion_data_path)
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
        coeffs = get_coeffs(self.config.pd_distortion_data_path, self.degree)
        coeff_tensor = torch.as_tensor(coeffs, dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)

        # Ideal (reference) quadratic coefficients a, b where y = a * x**2 + b
        ideal_coeffs = _compute_quadratic_coeffs(self.config.pd_distortion_data_path)
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength
        self.strength: float = float(self.config.pd_distortion_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pd_noise_w = float(self.config.pd_noise_w or 0.0)
        if pd_noise_w > 0:
            x = x + torch.randn_like(x) * pd_noise_w
        clamp_min = self.config.pd_input_clamp_min_w
        clamp_max = self.config.pd_input_clamp_max_w
        if clamp_min is not None or clamp_max is not None:
            if clamp_min is None:
                x = torch.clamp_max(x, float(clamp_max))
            elif clamp_max is None:
                x = torch.clamp_min(x, float(clamp_min))
            else:
                x = torch.clamp(x, float(clamp_min), float(clamp_max))
        strength = float(self.strength)
        if strength <= 0.0:
            x_squared = x * x
            ideal_y = self.ideal_coeffs[0] * x_squared + self.ideal_coeffs[1]
            return torch.clamp(ideal_y, 0, 1)

        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y.mul_(x).add_(a)

        if strength >= 1.0:
            return torch.clamp(poly_y, 0, 1)

        x_squared = x * x
        ideal_y = self.ideal_coeffs[0] * x_squared + self.ideal_coeffs[1]
        # Blend in-place to reduce peak memory
        poly_y.mul_(strength).add_(ideal_y, alpha=(1.0 - strength))
        poly_y.clamp_(0, 1)
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
        ideal_coeffs = _compute_linear_coeffs(self.config.tia_distortion_data_path)
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength
        self.strength: float = float(self.config.tia_distortion_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        strength = float(self.strength)
        if strength <= 0.0:
            ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
            return torch.clamp(ideal_y, 0, 1)

        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y.mul_(x).add_(a)

        if strength >= 1.0:
            return torch.clamp(poly_y, 0, 1)

        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
        # Blend in-place to reduce peak memory
        poly_y.mul_(strength).add_(ideal_y, alpha=(1.0 - strength))
        poly_y.clamp_(0, 1)
        return poly_y


class MRM(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.phase_degree: int = 0
        self.pwr_degree: int = 0
        self.ler_variation = LER_variation(config)
        if self.config.mrm_power_data_path is None:
            raise ValueError("MRM power data path is not set")
        if self.config.mrm_power_polyfit_order is None:
            self.pwr_degree = get_ideal_degree(self.config.mrm_power_data_path)
        else:
            self.pwr_degree = self.config.mrm_power_polyfit_order

        self.ph_in_min, self.ph_in_max, self.ph_out_min, self.ph_out_max = (
            get_io_ranges(self.config.mrm_phase_data_path)
        )
        if max(self.ph_out_min, self.ph_out_max) > 2 * np.pi:
            raise ValueError(
                "part of the MRM phase output is greater than 2*pi, please check the data to make sure it is in radians"
            )

        pwr_coeffs = get_coeffs(self.config.mrm_power_data_path, self.pwr_degree)
        pwr_coeff_tensor = torch.as_tensor(pwr_coeffs, dtype=torch.float32)
        self.register_buffer("pwr_coeffs", pwr_coeff_tensor)

        # Ideal power coefficients (linear)
        ideal_pwr_coeffs = _compute_linear_coeffs(self.config.mrm_power_data_path)
        self.register_buffer(
            "ideal_pwr_coeffs", torch.as_tensor(ideal_pwr_coeffs, dtype=torch.float32)
        )

        # Distortion strength for power
        self.pwr_strength: float = float(self.config.mrm_power_distortion_strength)
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
        The returned value is an intensity scale; field-amplitude paths apply
        its square root.
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

    def _laser_field_scale(
        self, laser_scale: torch.Tensor | float, reference: torch.Tensor
    ) -> torch.Tensor:
        """Convert laser intensity RIN scale to field-amplitude scale."""
        if torch.is_tensor(laser_scale):
            laser_scale = laser_scale.to(device=reference.device, dtype=reference.dtype)
        else:
            laser_scale = torch.as_tensor(
                laser_scale, device=reference.device, dtype=reference.dtype
            )
        return torch.sqrt(laser_scale)

    def forward(
        self, x: torch.Tensor, laser_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Power response (Horner's rule). Use branch-shortcuts and in-place ops
        # to reduce peak memory during training.
        pwr_strength = float(self.pwr_strength)
        if pwr_strength <= 0.0:
            pwr_y = self.ideal_pwr_coeffs[0] * x + self.ideal_pwr_coeffs[1]
        else:
            pwr_poly_y = torch.zeros_like(
                x, dtype=self.pwr_coeffs.dtype, device=x.device
            )
            for a in self.pwr_coeffs:
                pwr_poly_y.mul_(x).add_(a)
            if pwr_strength >= 1.0:
                pwr_y = pwr_poly_y
            else:
                pwr_ideal_y = self.ideal_pwr_coeffs[0] * x + self.ideal_pwr_coeffs[1]
                pwr_poly_y.mul_(pwr_strength).add_(
                    pwr_ideal_y, alpha=(1.0 - pwr_strength)
                )
                pwr_y = pwr_poly_y

        # Per-shot global laser noise (shared across channels), applied before splitting/LER.
        if laser_scale is None:
            laser_scale = self.make_laser_scale(pwr_y)
        if laser_scale is not None:
            # RIN is defined on optical intensity/power. The MRM output is used
            # as a complex field amplitude, so the field gets sqrt(intensity).
            pwr_y = pwr_y * self._laser_field_scale(laser_scale, pwr_y)

        # Apply LER variation to MRM power only
        if self.config.ler_std_dev > 0:
            pwr_y = self.ler_variation(pwr_y)

        gain = float(self.config.mrm_power_gain or 1.0)
        if gain != 1.0:
            pwr_y = pwr_y * gain

        # Phase response
        phase_strength = float(self.phase_strength)
        if phase_strength <= 0.0:
            phase_y = torch.zeros_like(pwr_y)
        else:
            phase_poly_y = torch.zeros_like(
                x, dtype=self.phase_coeffs.dtype, device=x.device
            )
            for a in self.phase_coeffs:
                phase_poly_y.mul_(x).add_(a)
            phase_y = phase_poly_y.mul_(phase_strength)

        return torch.polar(pwr_y, phase_y)

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
    def __init__(self, config: AppConfig):
        super(JTC, self).__init__()
        self.config = config
        self.driver = Driver(config)
        self.mrm = MRM(config)
        self.pd = PD(config)
        self.tia = TIA(config)
        self.lens = LensLegendre(config, length=config.jtc_total_field)
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
                )

        # Ordered list of available stage names
        self.stage_order = [
            "input_plane",
            "input_plane_quant",
            "input_plane_driver",
            "input_plane_mrm_phase",
            "input_plane_mrm_pwr",
            "jps_raw",
            "jps_pd",
            "jps_tia",
            "jps_scale",
            "jps_quant",
            "jps_driver",
            "jps_mrm_phase",
            "jps_mrm_pwr",
            "output_raw",
            "output_pd",
            "output_tia",
            "output_scale",
            "output_quant",
            "output_slice",
        ]
        self._use_fused_transfer = self._can_use_fused_transfer()

    def _to_complex_field(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_complex():
            return x.to(dtype=torch.complex64)
        real = x.to(dtype=torch.float32)
        return torch.complex(real, torch.zeros_like(real))

    def _can_use_fused_transfer(self) -> bool:
        return bool(self.config.enable_jtc_ideal_fused_transfer)

    def _has_ideal_input_transfer(self) -> bool:
        return (
            float(self.driver.strength) <= 0.0
            and float(self.mrm.pwr_strength) <= 0.0
            and float(self.mrm.phase_strength) <= 0.0
            and float(self.config.ler_std_dev or 0.0) <= 0.0
            and self.config.laser_rin_db is None
            and float(self.config.mrm_power_gain or 1.0) == 1.0
        )

    def _has_ideal_output_transfer(self) -> bool:
        return (
            float(self.loss) == 1.0
            and float(self.pd.strength) <= 0.0
            and float(self.tia.strength) <= 0.0
            and float(self.config.pd_noise_w or 0.0) <= 0.0
            and self.config.pd_input_clamp_min_w is None
            and self.config.pd_input_clamp_max_w is None
            and self.config.scale_output == "none"
        )

    def _eval_poly(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x, dtype=coeffs.dtype, device=x.device)
        for a in coeffs:
            y.mul_(x).add_(a)
        return y

    def _input_distortion_fused(
        self, x: torch.Tensor, laser_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fused DAC -> driver -> complex MRM path."""
        driver_strength = float(self.driver.strength)
        if driver_strength <= 0.0:
            driver_y = self.driver.ideal_coeffs[0] * x + self.driver.ideal_coeffs[1]
        else:
            driver_poly_y = self._eval_poly(self.driver.coeffs, x)
            if driver_strength >= 1.0:
                driver_y = driver_poly_y
            else:
                driver_ideal_y = (
                    self.driver.ideal_coeffs[0] * x + self.driver.ideal_coeffs[1]
                )
                driver_poly_y.mul_(driver_strength).add_(
                    driver_ideal_y, alpha=(1.0 - driver_strength)
                )
                driver_y = driver_poly_y

        pwr_strength = float(self.mrm.pwr_strength)
        if pwr_strength <= 0.0:
            pwr_y = (
                self.mrm.ideal_pwr_coeffs[0] * driver_y
                + self.mrm.ideal_pwr_coeffs[1]
            )
        else:
            pwr_poly_y = self._eval_poly(self.mrm.pwr_coeffs, driver_y)
            if pwr_strength >= 1.0:
                pwr_y = pwr_poly_y
            else:
                pwr_ideal_y = (
                    self.mrm.ideal_pwr_coeffs[0] * driver_y
                    + self.mrm.ideal_pwr_coeffs[1]
                )
                pwr_poly_y.mul_(pwr_strength).add_(
                    pwr_ideal_y, alpha=(1.0 - pwr_strength)
                )
                pwr_y = pwr_poly_y

        if laser_scale is None:
            laser_scale = self.mrm.make_laser_scale(pwr_y)
        if laser_scale is not None:
            # RIN is defined on optical intensity/power. The MRM output is used
            # as a complex field amplitude, so the field gets sqrt(intensity).
            pwr_y = pwr_y * self.mrm._laser_field_scale(laser_scale, pwr_y)

        if self.config.ler_std_dev > 0:
            pwr_y = self.mrm.ler_variation(pwr_y)

        gain = float(self.config.mrm_power_gain or 1.0)
        if gain != 1.0:
            pwr_y = pwr_y * gain

        phase_strength = float(self.mrm.phase_strength)
        if phase_strength <= 0.0:
            # Avoid polar/PolarBackward only for the exact zero-phase case.
            return torch.complex(pwr_y, torch.zeros_like(pwr_y))

        phase_y = self._eval_poly(self.mrm.phase_coeffs, driver_y)
        phase_y = phase_y.mul(phase_strength)
        return torch.polar(pwr_y, phase_y)

    def _output_distortion_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused loss -> PD -> TIA -> ADC path."""
        x = x * self.loss
        if self.config.scale_output == "pd":
            x = self.scale_to_range(x, 1e-6, 1e-5)

        pd_noise_w = float(self.config.pd_noise_w or 0.0)
        if pd_noise_w > 0:
            x = x + torch.randn_like(x) * pd_noise_w
        clamp_min = self.config.pd_input_clamp_min_w
        clamp_max = self.config.pd_input_clamp_max_w
        if clamp_min is not None or clamp_max is not None:
            if clamp_min is None:
                x = torch.clamp_max(x, float(clamp_max))
            elif clamp_max is None:
                x = torch.clamp_min(x, float(clamp_min))
            else:
                x = torch.clamp(x, float(clamp_min), float(clamp_max))

        pd_strength = float(self.pd.strength)
        if pd_strength <= 0.0:
            x_squared = x * x
            pd_y = self.pd.ideal_coeffs[0] * x_squared + self.pd.ideal_coeffs[1]
        else:
            pd_poly_y = self._eval_poly(self.pd.coeffs, x)
            if pd_strength >= 1.0:
                pd_y = pd_poly_y
            else:
                x_squared = x * x
                pd_ideal_y = (
                    self.pd.ideal_coeffs[0] * x_squared + self.pd.ideal_coeffs[1]
                )
                pd_poly_y.mul_(pd_strength).add_(
                    pd_ideal_y, alpha=(1.0 - pd_strength)
                )
                pd_y = pd_poly_y
        pd_y = torch.clamp(pd_y, 0, 1)

        tia_strength = float(self.tia.strength)
        if tia_strength <= 0.0:
            tia_y = self.tia.ideal_coeffs[0] * pd_y + self.tia.ideal_coeffs[1]
        else:
            tia_poly_y = self._eval_poly(self.tia.coeffs, pd_y)
            if tia_strength >= 1.0:
                tia_y = tia_poly_y
            else:
                tia_ideal_y = self.tia.ideal_coeffs[0] * pd_y + self.tia.ideal_coeffs[1]
                tia_poly_y.mul_(tia_strength).add_(
                    tia_ideal_y, alpha=(1.0 - tia_strength)
                )
                tia_y = tia_poly_y
        x = torch.clamp(tia_y, 0, 1)
        if self.config.scale_output == "adc":
            x = x / x.max().clamp_min(1e-12)
        x = quantize_ste(x, self.config.adc_bits)
        return x

    def input_distortion(
        self, x: torch.Tensor, laser_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply DAC quantization, driver response, and MRM modulation."""
        x = quantize_ste(x, self.config.dac_bits)
        if (
            laser_scale is None
            and self._has_ideal_input_transfer()
            and self._has_ideal_output_transfer()
        ):
            return self._to_complex_field(x)
        if self._use_fused_transfer:
            return self._to_complex_field(
                self._input_distortion_fused(x, laser_scale=laser_scale)
            )
        x = self.driver(x)
        x = self.mrm(x, laser_scale=laser_scale)
        return self._to_complex_field(x)

    def output_distortion(self, x: torch.Tensor) -> torch.Tensor:
        """Apply loss, square-law PD, TIA, scaling, then ADC quantization."""
        if self._has_ideal_output_transfer():
            x = x * x
            x = quantize_ste(x, self.config.adc_bits)
            return x
        if self._use_fused_transfer:
            return self._output_distortion_fused(x)
        x = x * self.loss
        if self.config.scale_output == "pd":
            x = self.scale_to_range(x, 1e-6, 1e-5)
        x = self.pd(x)
        x = self.tia(x)
        if self.config.scale_output == "adc":
            x = x / x.max().clamp_min(1e-12)
        x = quantize_ste(x, self.config.adc_bits)
        return x

    def fft_and_magnitude(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply FFT, optional lens distortion, then take magnitude.

        Tensors stay in native FFT order. Correlation extraction maps the old
        centered-plane indices into native order, avoiding full-plane rolls.
        """
        x = torch.fft.fft(x)
        if not self.lens.is_identity():
            x = self.lens(x)
        if indices is not None:
            x = x.index_select(-1, indices)
        x = torch.abs(x)
        if x.dtype == torch.float16:
            x = x.float()
        return x

    def first_detector_readout(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the first detector/JPS readout."""
        x = self.fft_and_magnitude(x)
        x = x / math.sqrt(float(self.jtc_total_field))
        return self.output_distortion(x)

    def _ideal_sqrt_readout(self, x: torch.Tensor) -> torch.Tensor:
        return sqrt_nonnegative_with_finite_grad(x)

    def final_detector_readout(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute final detector readout and recover ideal field amplitude."""
        x = self.fft_and_magnitude(x, indices=indices)
        x = self.output_distortion(x)
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
        plane_size = self.jtc_total_field
        same_start = self._correlation_start
        shifted_indices = torch.arange(same_start, same_start + self.output_length)
        native_offset = (plane_size + 1) // 2
        indices = (shifted_indices + native_offset) % plane_size
        return indices

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

        laser_scale = self.mrm.make_laser_scale(
            torch.empty(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)
        )
        signal_distorted = self.input_distortion(signal, laser_scale=laser_scale)
        kernel_distorted = self.input_distortion(kernel, laser_scale=laser_scale)
        input_plane = self.build_input_plane(signal_distorted, kernel_distorted)

        jps = self.first_detector_readout(input_plane)
        jps = quantize_ste(jps, self.config.fourier_plane_bits)

        laser_scale_2 = self.mrm.make_laser_scale(
            torch.empty(jps.shape[0], 1, device=jps.device, dtype=jps.dtype)
        )
        jps_distorted = self.input_distortion(jps, laser_scale=laser_scale_2)

        if self._can_slice_final_detector():
            indices = self.compute_correlation_indices(jps_distorted.device)
            return self.final_detector_readout(jps_distorted, indices=indices)

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
    ) -> dict[str, torch.Tensor]:
        """Compute intermediate tensors for the requested pipeline stages.

        Returns a dict of 1D tensors per stage name, suitable for plotting.
        Complex tensors are converted to magnitudes.
        """
        if stages is None:
            stages = tuple(self.stage_order)

        results: dict[str, torch.Tensor] = {}

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
        kernel_quant = quantize_ste(kernel, self.config.dac_bits)
        signal_quant = quantize_ste(signal, self.config.dac_bits)
        plane_quant = torch.zeros(
            B, self.jtc_total_field, dtype=torch.float32, device=signal.device
        )
        plane_quant[..., kernel_start:kernel_end] = kernel_quant
        plane_quant[..., signal_start:signal_end] = signal_quant
        if "input_plane_quant" in stages:
            results["input_plane_quant"] = plane_quant[0, :].detach()

        # 2) Driver on the full field
        plane_driver = self.driver(plane_quant)
        if "input_plane_driver" in stages:
            results["input_plane_driver"] = plane_driver[0, :].detach()

        # 3) MRM on the full, driver-processed field
        plane_mrm = self.mrm(plane_driver)
        # Store overall input plane magnitude
        if "input_plane" in stages:
            results["input_plane"] = torch.abs(plane_mrm)[0, :].detach()
        # Store separate power and phase components (real-valued)
        if "input_plane_mrm_pwr" in stages:
            results["input_plane_mrm_pwr"] = torch.abs(plane_mrm)[0, :].detach()
        if "input_plane_mrm_phase" in stages:
            results["input_plane_mrm_phase"] = torch.angle(plane_mrm)[0, :].detach()

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
            results["jps_raw"] = jps_mag[0, :].detach()

        # Output distortion (first pass)
        # Apply loss, then PD and TIA.
        jps_base = (jps_mag / math.sqrt(float(self.jtc_total_field))) * self.loss

        if self.config.scale_output == "pd":
            jps_base = self.scale_to_range(jps_base, 1e-6, 1e-5)

        jps_pd = self.pd(jps_base)
        if "jps_pd" in stages:
            results["jps_pd"] = jps_pd[0, :].detach()

        jps_tia = self.tia(jps_pd)
        if "jps_tia" in stages:
            results["jps_tia"] = jps_tia[0, :].detach()

        # Scale (ADC pre-scale)
        if self.config.scale_output == "adc":
            jps_scaled = jps_tia / jps_tia.max().clamp_min(1e-12)
        else:
            jps_scaled = jps_tia
        if "jps_scale" in stages:
            results["jps_scale"] = jps_scaled[0, :].detach()

        # Quantize (Fourier plane bits)
        jps_quant = quantize_ste(
            jps_scaled, self.config.fourier_plane_bits
        )
        if "jps_quant" in stages:
            results["jps_quant"] = jps_quant[0, :].detach()

        # Input distortion again before inverse FFT (second pass). DAC
        # quantization is applied first for this JTC pipeline.
        jps_dac = quantize_ste(jps_quant, self.config.dac_bits)
        jps_driver = self.driver(jps_dac)
        if "jps_driver" in stages:
            results["jps_driver"] = jps_driver[0, :].detach()
        jps_complex = self.mrm(jps_driver)
        jps_pwr = torch.abs(jps_complex)
        jps_phase = torch.angle(jps_complex)
        if "jps_mrm_pwr" in stages:
            results["jps_mrm_pwr"] = jps_pwr[0, :].detach()
        if "jps_mrm_phase" in stages:
            results["jps_mrm_phase"] = jps_phase[0, :].detach()
        # jps_complex already includes both components

        # Back to detector plane
        output_mag = self.fft_and_magnitude(jps_complex)
        output_raw = output_mag * self.loss
        if "output_raw" in stages:
            results["output_raw"] = output_raw[0, :].detach()

        if self.config.scale_output == "pd":
            output_raw = self.scale_to_range(output_raw, 1e-6, 1e-5)

        out_pd = self.pd(output_raw)
        if "output_pd" in stages:
            results["output_pd"] = out_pd[0, :].detach()

        out_tia = self.tia(out_pd)
        if "output_tia" in stages:
            results["output_tia"] = out_tia[0, :].detach()

        if self.config.scale_output == "adc":
            out_scaled = out_tia / out_tia.max().clamp_min(1e-12)
        else:
            out_scaled = out_tia

        if "output_scale" in stages:
            results["output_scale"] = out_scaled[0, :].detach()

        out_quant = quantize_ste(out_scaled, self.config.adc_bits)
        if "output_quant" in stages:
            results["output_quant"] = out_quant[0, :].detach()

        same_indices = self.compute_correlation_indices(jps_complex.device)
        output_slice = sqrt_nonnegative_with_finite_grad(out_quant)[..., same_indices]
        if "output_slice" in stages:
            results["output_slice"] = output_slice[0, :].detach()

        return results

    def forward(self, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Joint Transform Correlator forward pass.

        Pipeline:
        1. Input distortion (DAC quant + driver + MRM + optional laser_rin_db)
        2. FFT to Fourier plane
        3. Output distortion (loss + PD + optional pd_noise_w + TIA + scale + ADC quant)
        4. Fourier plane quantization (fourier_plane_bits)
        5. Input distortion again (DAC quant + driver + MRM + optional laser_rin_db)
        6. FFT to detector plane
        7. Output distortion again (loss + PD + optional pd_noise_w + TIA + scale + ADC quant)
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
            raise ValueError(
                "JTC.forward expects signal shape (B, H, 1, W) or (B, W)"
            )
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
        jps = quantize_ste(jps, self.config.fourier_plane_bits)

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
