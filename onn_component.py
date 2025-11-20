from typing import Tuple, Optional

from onn_config import AppConfig
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import math
import warnings
from jtc_cycle_planner import compute_contamination_profile

# NEW: Helper functions to compute ideal (reference) transfer function coefficients


class QuantDequant_STE(torch.autograd.Function):
    # version of the MRR LUT but implemented with straight-thourgh estimator to help training
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int | None) -> torch.Tensor:
        if bits is None:
            return input
        levels = 2**bits
        input_clamped = torch.clamp(input, 0, 1)
        return torch.round(input_clamped * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Return gradients for (input, bits)
        return grad_output, None


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


def get_io_ranges(csv_file: str) -> Tuple[float, float, float, float]:
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


def _fit_minimal_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float,
    max_degree: int,
    min_r2: Optional[float] = None,
) -> tuple[np.ndarray, int, float, float]:
    """Fit the lowest-order polynomial whose relative RMSE/R² criteria pass."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if max_degree < 1:
        raise ValueError("max_degree must be >= 1")
    y_range = float(np.max(y) - np.min(y))
    denom = y_range if y_range > 0 else 1.0

    best_coeffs: np.ndarray | None = None
    best_degree = 1
    best_error = float("inf")
    best_r2 = float("-inf")
    y_var = float(np.var(y))

    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        approx = np.polyval(coeffs, x)
        rmse = float(np.sqrt(np.mean((approx - y) ** 2)) / denom)
        if y_var > 0:
            r2 = 1.0 - float(np.sum((approx - y) ** 2) / (len(y) * y_var))
        else:
            r2 = 1.0

        meets_tol = rmse <= tolerance
        meets_r2 = min_r2 is None or r2 >= min_r2

        if rmse < best_error or (abs(rmse - best_error) < 1e-12 and r2 > best_r2):
            best_error = rmse
            best_coeffs = coeffs
            best_degree = degree
            best_r2 = r2
        if meets_tol and meets_r2:
            break

    if best_coeffs is None:
        raise RuntimeError("Polynomial fitting failed to converge")
    return best_coeffs.astype(np.float32), best_degree, best_error, best_r2


_POLY_FN_CACHE: dict[int, type[torch.autograd.Function]] = {}


def _get_poly_function(degree: int) -> type[torch.autograd.Function]:
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative")
    fn = _POLY_FN_CACHE.get(degree)
    if fn is not None:
        return fn

    class _PolyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
            if torch.is_complex(x):
                x = x.real
            dtype = coeffs.dtype
            input_dtype = x.dtype
            x_eval = x if input_dtype == dtype else x.to(dtype)
            ctx.save_for_backward(x_eval, coeffs)
            ctx.input_dtype = input_dtype
            ctx.degree = degree

            if coeffs.numel() != degree + 1:
                raise ValueError(
                    f"Expected {degree + 1} coefficients, got {coeffs.numel()}"
                )

            out = torch.zeros_like(x_eval)
            out.add_(coeffs[0])
            for idx in range(1, degree + 1):
                out.mul_(x_eval)
                out.add_(coeffs[idx])
            return out

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
            x_eval, coeffs = ctx.saved_tensors
            dtype = coeffs.dtype
            grad_output = grad_output.to(dtype)
            degree = ctx.degree
            if degree == 0:
                grad_x = grad_output.new_zeros(x_eval.shape)
            else:
                powers = torch.arange(
                    degree, 0, -1, device=coeffs.device, dtype=dtype
                )
                deriv_coeffs = coeffs[:-1] * powers
                deriv = torch.zeros_like(x_eval)
                deriv.add_(deriv_coeffs[0])
                for idx in range(1, deriv_coeffs.numel()):
                    deriv.mul_(x_eval)
                    deriv.add_(deriv_coeffs[idx])
                grad_x = grad_output * deriv
            if ctx.input_dtype != dtype:
                grad_x = grad_x.to(ctx.input_dtype)
            return grad_x, None

    _PolyFunction.__name__ = f"EvalPolyDeg{degree}"
    _POLY_FN_CACHE[degree] = _PolyFunction
    return _PolyFunction


def _eval_static_poly(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    degree = coeffs.numel() - 1
    fn = _get_poly_function(degree)
    return fn.apply(x, coeffs)


class PolynomialTransfer1D(nn.Module):
    """Single real-valued polynomial mapping with optional input clipping."""

    def __init__(
        self,
        coeffs: torch.Tensor,
        clip_range: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("coeffs", coeffs)
        if clip_range is not None:
            lo, hi = clip_range
            if lo > hi:
                raise ValueError("clip_range lower bound exceeds upper bound")
            self.clip_lo = float(lo)
            self.clip_hi = float(hi)
        else:
            self.clip_lo = None
            self.clip_hi = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_eval = x
        if self.clip_lo is not None and self.clip_hi is not None:
            x_eval = torch.clamp(x_eval, self.clip_lo, self.clip_hi)
        return _eval_static_poly(x_eval, self.coeffs)


class ComplexPolynomialTransfer(nn.Module):
    """Polynomial cascade that predicts magnitude+phase and returns complex output."""

    def __init__(
        self,
        mag_coeffs: torch.Tensor,
        phase_coeffs: torch.Tensor,
        clip_range: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.magnitude = PolynomialTransfer1D(mag_coeffs, clip_range)
        self.phase = PolynomialTransfer1D(phase_coeffs, clip_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.clamp(self.magnitude(x), min=0.0)
        phase = self.phase(x)
        return torch.polar(mag, phase)


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
        poly_y = _eval_static_poly(x, self.coeffs)

        # Ideal linear response
        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
        # Blend based on strength
        combined_y = self.strength * poly_y + (1.0 - self.strength) * ideal_y
        # print(f"ideal_y_shape: {ideal_y.shape}")
        # print(f"ideal_y: {ideal_y[0]}")
        # print(f"poly_y: {poly_y[0]}")
        # input("Press Enter to continue...")
        return combined_y


class PD_TIA(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.degree: int = 0
        if self.config.pd_tia_distortion_data_path is None:
            raise ValueError("PD-TIA distortion data path is not set")
        if self.config.pd_tia_distortion_polyfit_order is None:
            self.degree = get_ideal_degree(self.config.pd_tia_distortion_data_path)
        else:
            self.degree = self.config.pd_tia_distortion_polyfit_order
        coeffs = get_coeffs(self.config.pd_tia_distortion_data_path, self.degree)
        coeff_tensor = torch.as_tensor(coeffs, dtype=torch.float32)
        self.register_buffer("coeffs", coeff_tensor)

        # Ideal (reference) quadratic coefficients a, b where y = a * x**2 + b
        ideal_coeffs = _compute_quadratic_coeffs(
            self.config.pd_tia_distortion_data_path
        )
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength
        self.strength: float = float(self.config.pd_tia_distortion_strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Polynomial evaluation using Horner's rule
        x = torch.clamp(x, 1e-6, 1e-5)
        poly_y = _eval_static_poly(x, self.coeffs)

        # Ideal quadratic response
        ideal_y = self.ideal_coeffs[0] * torch.pow(x, 2) + self.ideal_coeffs[1]

        combined_y = self.strength * poly_y + (1.0 - self.strength) * ideal_y
        combined_y = torch.clamp(combined_y, 0, 1)
        return combined_y


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
        x = torch.clamp(x, 1e-6, 1e-5)
        poly_y = _eval_static_poly(x, self.coeffs)

        ideal_y = self.ideal_coeffs[0] * torch.pow(x, 2) + self.ideal_coeffs[1]
        combined_y = self.strength * poly_y + (1.0 - self.strength) * ideal_y
        combined_y = torch.clamp(combined_y, 0, 1)
        return combined_y


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
        poly_y = _eval_static_poly(x, self.coeffs)

        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]
        combined_y = self.strength * poly_y + (1.0 - self.strength) * ideal_y
        combined_y = torch.clamp(combined_y, 0, 1)
        return combined_y


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

        # Get IO ranges for power
        # self.pwr_in_min, self.pwr_in_max, self.pwr_out_min, self.pwr_out_max = get_io_ranges(self.config.mrm_power_data_path)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pwr_poly_y = _eval_static_poly(x, self.pwr_coeffs)

        # Ideal linear response for power
        pwr_ideal_y = self.ideal_pwr_coeffs[0] * x + self.ideal_pwr_coeffs[1]

        pwr_y = self.pwr_strength * pwr_poly_y + (1.0 - self.pwr_strength) * pwr_ideal_y

        # Apply LER variation to MRM power only
        if self.config.ler_std_dev > 0:
            pwr_y = self.ler_variation(pwr_y)

        phase_poly_y = _eval_static_poly(x, self.phase_coeffs)

        phase_y = self.phase_strength * phase_poly_y

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
    @staticmethod
    def _compute_usable_outputs(input_len: int, kernel_len: int, lens_size: int, sep: int) -> int:
        """Calculate number of usable correlation outputs for given JTC configuration.

        Uses contamination-aware cycle planner to determine clean valid outputs.
        Accounts for autocorrelation contamination and edge effects.

        Args:
            input_len: Length of input signal (M)
            kernel_len: Length of kernel (N)
            lens_size: Total size of JTC plane
            sep: Separation between kernel and signal

        Returns:
            Number of clean valid outputs for stitching (effective stride)
        """
        # Validity checks
        if input_len <= 0 or kernel_len <= 0 or lens_size <= 0:
            return 0
        if input_len < kernel_len:
            return 0
        if sep < 0:
            return 0

        # Check if configuration fits in lens plane
        # Need space for: kernel (N) + separation (sep) + signal (M)
        if input_len + kernel_len + sep > lens_size:
            return 0

        # Use contamination-aware cycle planner
        # Returns: (total_outputs, clean_valid_outputs, effective_stride)
        _, clean_valid, effective_stride = compute_contamination_profile(
            input_len, kernel_len, lens_size, sep
        )

        return effective_stride

    def __init__(self, config: AppConfig):
        super(JTC, self).__init__()
        self.config = config
        self.driver = Driver(config)
        self.mrm = MRM(config)
        self.pd = PD(config)
        self.tia = TIA(config)
        self.input_length = config.input_length
        self.kernel_length = config.kernel_length
        self.jtc_separation = config.jtc_separation
        self.jtc_total_field = config.jtc_total_field
        self.loss = float(config.loss)

        # Persistent workspaces to avoid per-call allocations
        self.register_buffer(
            "_input_plane_buffer",
            torch.zeros(1, self.jtc_total_field, dtype=torch.complex64),
        )
        self.register_buffer(
            "_fourier_plane_buffer",
            torch.zeros(1, self.jtc_total_field, dtype=torch.float32),
        )
        self.register_buffer(
            "_detector_plane_buffer",
            torch.zeros(1, self.jtc_total_field, dtype=torch.float32),
        )

        # Calculate output_length if not specified.
        # Default: match the input patch width so downstream stitching stays consistent.
        if config.output_length is None:
            self.output_length = self.input_length
        else:
            self.output_length = config.output_length

        # Validate that configuration is feasible
        if self.input_length + self.kernel_length + self.jtc_separation > self.jtc_total_field:
            raise ValueError(
                f"JTC total field ({self.jtc_total_field}) is too small for "
                f"input_length ({self.input_length}) + kernel_length ({self.kernel_length}) + "
                f"separation ({self.jtc_separation}) = {self.input_length + self.kernel_length + self.jtc_separation}"
            )

        # Analyze contamination profile using cycle planner
        total_outputs, clean_valid_outputs, effective_stride = compute_contamination_profile(
            self.input_length, self.kernel_length, self.jtc_total_field, self.jtc_separation
        )
        self.total_correlation_outputs = total_outputs
        self.clean_valid_outputs = clean_valid_outputs
        self.effective_stride = effective_stride

        # Report contamination status (only if significant)
        # Note: clean_valid_outputs is the number of clean outputs in the valid convolution region
        # For valid conv, we use M-N+1 outputs from the M+N-1 correlation
        num_valid_outputs = self.input_length - self.kernel_length + 1
        if num_valid_outputs > 0:
            valid_contamination_percent = 100 * (1 - clean_valid_outputs / num_valid_outputs)
            if valid_contamination_percent > 10:
                import warnings
                warnings.warn(
                    f"JTC config has {valid_contamination_percent:.1f}% contamination in valid outputs: "
                    f"M={self.input_length}, N={self.kernel_length}, "
                    f"plane={self.jtc_total_field}, sep={self.jtc_separation}. "
                    f"Clean valid outputs: {clean_valid_outputs}/{num_valid_outputs}, "
                    f"Effective stride: {effective_stride}",
                    UserWarning
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

        self.pre_lens_poly: Optional[ComplexPolynomialTransfer] = None
        self.post_lens_poly: Optional[PolynomialTransfer1D] = None
        self.transfer_fit_stats: dict[str, dict[str, float]] = {}
        if getattr(self.config, "simplify_transfer_functions", False):
            self._maybe_initialize_transfer_simplification()

    def _maybe_initialize_transfer_simplification(self) -> None:
        num_samples = max(16, int(getattr(self.config, "transfer_fit_samples", 256)))
        tolerance = float(getattr(self.config, "transfer_fit_tolerance", 1e-4))
        max_degree = max(1, int(getattr(self.config, "transfer_fit_max_degree", 10)))

        min_r2 = getattr(self.config, "transfer_fit_min_r2", None)

        try:
            module, stats = self._fit_pre_lens_transfer(
                num_samples, tolerance, max_degree, min_r2
            )
            self.pre_lens_poly = module
            self.transfer_fit_stats["pre_lens"] = stats
            # print(
            #     "[SIMPLIFY] Pre-lens magnitude deg "
            #     f"{stats['mag_degree']} (rel_rmse={stats['mag_error']:.2e}, R^2={stats['mag_r2']:.4f}), "
            #     f"phase deg {stats['phase_degree']} (rel_rmse={stats['phase_error']:.2e}, R^2={stats['phase_r2']:.4f})"
            # )
        except Exception as exc:  # pragma: no cover - debug helper
            warnings.warn(
                f"Pre-lens transfer simplification failed: {exc}", RuntimeWarning
            )

        scale_mode = (self.config.scale_output or "none").lower()
        if scale_mode != "none":
            warnings.warn(
                "Post-lens transfer simplification skipped because scale_output != 'none'",
                RuntimeWarning,
            )
            return

        try:
            module, stats = self._fit_post_lens_transfer(
                num_samples, tolerance, max_degree, min_r2
            )
            self.post_lens_poly = module
            self.transfer_fit_stats["post_lens"] = stats
            #print(
            #    "[SIMPLIFY] Post-lens degree "
            #    f"{stats['degree']} (rel_rmse={stats['error']:.2e}, R^2={stats['r2']:.4f})"
            #)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Post-lens transfer simplification failed: {exc}", RuntimeWarning
            )

    def _fit_pre_lens_transfer(
        self,
        num_samples: int,
        tolerance: float,
        max_degree: int,
        min_r2: Optional[float],
    ) -> tuple[ComplexPolynomialTransfer, dict[str, float]]:
        driver_path = getattr(self.config, "driver_distortion_data_path", None)
        if not driver_path:
            raise ValueError("driver_distortion_data_path must be provided")
        in_lo, in_hi, _, _ = get_io_ranges(driver_path)
        if not math.isfinite(in_lo) or not math.isfinite(in_hi) or in_lo == in_hi:
            raise ValueError("Driver input range is degenerate; cannot fit polynomial.")

        samples = torch.linspace(
            float(in_lo), float(in_hi), steps=num_samples, dtype=torch.float32
        )
        with torch.no_grad():
            driver_out = self.driver(samples)
            cascade = self.mrm(driver_out)

        mags = cascade.abs().cpu().numpy()
        phases = np.unwrap(torch.angle(cascade).cpu().numpy())
        x_np = samples.cpu().numpy()

        mag_coeffs, mag_deg, mag_err, mag_r2 = _fit_minimal_polynomial(
            x_np, mags, tolerance, max_degree, min_r2
        )
        phase_coeffs, phase_deg, phase_err, phase_r2 = _fit_minimal_polynomial(
            x_np, phases, tolerance, max_degree, min_r2
        )

        module = ComplexPolynomialTransfer(
            torch.as_tensor(mag_coeffs, dtype=torch.float32),
            torch.as_tensor(phase_coeffs, dtype=torch.float32),
            clip_range=(float(in_lo), float(in_hi)),
        )
        stats = {
            "mag_degree": float(mag_deg),
            "mag_error": float(mag_err),
            "phase_degree": float(phase_deg),
            "phase_error": float(phase_err),
            "mag_r2": float(mag_r2),
            "phase_r2": float(phase_r2),
        }
        return module, stats

    def _fit_post_lens_transfer(
        self,
        num_samples: int,
        tolerance: float,
        max_degree: int,
        min_r2: Optional[float],
    ) -> tuple[PolynomialTransfer1D, dict[str, float]]:
        pd_path = getattr(self.config, "pd_distortion_data_path", None)
        if not pd_path:
            raise ValueError("pd_distortion_data_path must be provided")
        in_lo, in_hi, _, _ = get_io_ranges(pd_path)
        if not math.isfinite(in_lo) or not math.isfinite(in_hi) or in_lo == in_hi:
            raise ValueError("PD input range is degenerate; cannot fit polynomial.")

        samples = torch.linspace(
            float(in_lo), float(in_hi), steps=num_samples, dtype=torch.float32
        )
        with torch.no_grad():
            targets = self.output_distortion(samples.clone())

        coeffs, degree, error, r2 = _fit_minimal_polynomial(
            samples.cpu().numpy(),
            targets.cpu().numpy(),
            tolerance,
            max_degree,
            min_r2,
        )
        module = PolynomialTransfer1D(
            torch.as_tensor(coeffs, dtype=torch.float32),
            clip_range=(float(in_lo), float(in_hi)),
        )
        stats = {"degree": float(degree), "error": float(error), "r2": float(r2)}
        return module, stats

    def input_distortion(self, x: torch.Tensor) -> torch.Tensor:
        """Apply driver and MRM distortion without quantization.

        Note: Quantization should be applied separately before calling this method.
        """
        if self.pre_lens_poly is not None:
            return self.pre_lens_poly(x)
        x = self.driver(x)
        x = self.mrm(x)
        return x

    def output_distortion(self, x: torch.Tensor) -> torch.Tensor:
        """Apply loss, PD, TIA, and scaling without quantization.

        Note: Quantization should be applied separately after calling this method.
        """
        if self.post_lens_poly is not None:
            return self.post_lens_poly(x)

        x = x * self.loss
        if self.config.scale_output == "pd":
            x = self.scale_to_range(x, 1e-6, 1e-5)
        x = self.pd(x)
        x = self.tia(x)
        if self.config.scale_output == "adc":
            x = x / x.max().clamp_min(1e-12)
        return x

    def prepare_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """Quantize and apply input distortion to a kernel tensor."""
        quantized = QuantDequant_STE.apply(kernel, self.config.dac_bits)
        return self.input_distortion(quantized)

    def fft_and_magnitude(
        self, x: torch.Tensor, *, buffer_name: str | None = None
    ) -> torch.Tensor:
        """Apply FFT and take magnitude, optionally reusing a workspace buffer."""
        freq = torch.fft.fft(x)
        if buffer_name is None or freq.requires_grad:
            return torch.abs(freq)

        mag_dtype = torch.float32 if freq.is_complex() else freq.dtype
        buf = self._reserve_workspace(
            buffer_name,
            freq.shape[0],
            freq.shape[-1],
            device=freq.device,
            dtype=mag_dtype,
        )
        torch.abs(freq, out=buf)
        return buf

    def compute_correlation_indices(self, device) -> torch.Tensor:
        """Compute the indices for extracting convolution output.

        Uses extraction formula: same_start = sep + N//2
        Extracts output_length indices starting from same_start (mod plane size).
        Note: Original formula had +1, removed based on empirical analysis.
        """
        plane_size = self.jtc_total_field
        sep = self.jtc_separation
        N = self.kernel_length

        # Extraction formula for correlation output indices
        same_start = (sep + N // 2) % plane_size
        indices = torch.arange(
            same_start,
            same_start + self.output_length,
            device=device
        ) % plane_size
        return indices

    def _reserve_workspace(
        self,
        name: str,
        batch: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Always allocate fresh tensor to avoid graph issues with checkpointing/buffer reuse
        return torch.zeros(max(1, batch), length, dtype=dtype, device=device)

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
            raise ValueError(f"Signal length ({M}) is greater than configured input_length ({self.input_length})")
        if N > self.kernel_length:
            raise ValueError(f"Kernel length ({N}) is greater than configured kernel_length ({self.kernel_length})")
        if M + N + self.jtc_separation > self.jtc_total_field:
            raise ValueError(f"Not enough JTC field: {M} + {N} + {self.jtc_separation} > {self.jtc_total_field}")

        # Calculate positions
        kernel_start = 0
        kernel_end = kernel_start + N
        signal_start = kernel_end + self.jtc_separation
        signal_end = signal_start + M

        plane = self._reserve_workspace(
            "_input_plane_buffer",
            B,
            self.jtc_total_field,
            device=signal.device,
            dtype=signal.dtype,
        )
        plane.zero_()
        plane[..., kernel_start:kernel_end].copy_(kernel)
        plane[..., signal_start:signal_end].copy_(signal)
        return plane

    # Backward compatibility wrappers (deprecated - use new methods instead)
    def generate_input_plane(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Apply input distortion and build the JTC input plane.

        DEPRECATED: This method is kept for backward compatibility.
        Use input_distortion() and build_input_plane() separately instead.

        Note: This wrapper applies DAC quantization to maintain backward compatibility.
        """
        kernel_quantized = QuantDequant_STE.apply(kernel, self.config.dac_bits)
        signal_quantized = QuantDequant_STE.apply(signal, self.config.dac_bits)
        kernel_distorted = self.input_distortion(kernel_quantized)
        signal_distorted = self.input_distortion(signal_quantized)
        return self.build_input_plane(signal_distorted, kernel_distorted)

    def post_fft(self, input_plane: torch.Tensor) -> torch.Tensor:
        """Perform FFT and return frequency-domain representation.

        DEPRECATED: This method is kept for backward compatibility.
        Use fft_and_magnitude() or inline torch.fft operations instead.
        """
        jft = torch.fft.fft(input_plane)
        return jft

    def post_output_distortion(self, jft: torch.Tensor) -> torch.Tensor:
        """Apply output distortion after the Fourier plane.

        DEPRECATED: This method is kept for backward compatibility.
        Use output_distortion(torch.abs(jft)) instead.

        Note: This wrapper applies ADC quantization to maintain backward compatibility.
        """
        output = self.output_distortion(torch.abs(jft))
        return QuantDequant_STE.apply(output, self.config.adc_bits)

    def inverse_output(self, jps: torch.Tensor) -> torch.Tensor:
        """Propagate back to the detector plane and crop the result.

        DEPRECATED: This method is kept for backward compatibility.
        The forward method now implements this logic using unified helper methods.

        Note: This wrapper does NOT apply quantization. Use forward() for full pipeline.
        """
        jps_distorted = self.input_distortion(jps)
        output_plane = self.fft_and_magnitude(
            jps_distorted, buffer_name="_detector_plane_buffer"
        )
        output_plane = self.output_distortion(output_plane)
        output_plane = QuantDequant_STE.apply(output_plane, self.config.adc_bits)
        indices = self.compute_correlation_indices(output_plane.device)
        return output_plane[..., indices]

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
        stages: Tuple[str, ...] | None = None,
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
            raise ValueError(f"Signal length ({M}) is greater than configured input_length ({self.input_length})")
        if N > self.kernel_length:
            raise ValueError(f"Kernel length ({N}) is greater than configured kernel_length ({self.kernel_length})")
        if M + N + self.jtc_separation > self.jtc_total_field:
            raise ValueError(f"Not enough JTC field: {M} + {N} + {self.jtc_separation} > {self.jtc_total_field}")

        # Indices for placement
        kernel_start = 0
        kernel_end = kernel_start + N
        signal_start = kernel_end + self.jtc_separation
        signal_end = signal_start + M

        # 1) Quantize (DAC) and place into full input field
        kernel_quant = QuantDequant_STE.apply(kernel, self.config.dac_bits)
        signal_quant = QuantDequant_STE.apply(signal, self.config.dac_bits)
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

        # Fourier plane (FFT)
        jft = torch.fft.fft(input_plane_full)
        if "jps_raw" in stages:
            results["jps_raw"] = torch.abs(jft)[0, :].detach()

        # Output distortion (first pass)
        # Apply loss then PD/TIA
        jps_base = torch.abs(jft) * self.loss

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
        jps_quant = QuantDequant_STE.apply(jps_scaled, self.config.fourier_plane_bits)
        if "jps_quant" in stages:
            results["jps_quant"] = jps_quant[0, :].detach()

        # Input distortion again before inverse FFT (second pass)
        # No additional quantization - just driver and MRM
        jps_driver = self.driver(jps_quant)
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
        output_plane = torch.fft.fft(jps_complex)
        output_raw = torch.abs(output_plane) * self.loss
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

        out_quant = QuantDequant_STE.apply(out_scaled, self.config.adc_bits)
        if "output_quant" in stages:
            results["output_quant"] = out_quant[0, :].detach()

        same_indices = self.compute_correlation_indices(jps_complex.device)
        output_slice = out_quant[..., same_indices]
        if "output_slice" in stages:
            results["output_slice"] = output_slice[0, :].detach()

        return results

    def forward(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        *,
        kernel_pre_distorted: bool = False,
    ) -> torch.Tensor:
        """Joint Transform Correlator forward pass.
        """
        # Determine chunk size for output channels
        # Aim for batch_size_for_jtc < ~65536 to keep memory low
        B, H, _, _ = signal.shape
        C_out = kernel.shape[0]
        
        # Calculate per-channel batch multiplier (B * H)
        # Total batch size = B * H * C_out
        # We want B * H * chunk_size <= threshold
        # e.g. 128 * 32 * chunk = 4096 * chunk <= 65536 -> chunk <= 16
        
        # A100 80GB can handle much larger chunks.
        limit = 20_000_000 
        
        batch_multiplier = B * H
        chunk_size = max(1, limit // batch_multiplier)
        
        outputs = []
        
        for c_start in range(0, C_out, chunk_size):
            c_end = min(c_start + chunk_size, C_out)
            kernel_chunk = kernel[c_start:c_end]
            
            # Process this chunk
            out_chunk = self._forward_chunk(signal, kernel_chunk, kernel_pre_distorted)
            outputs.append(out_chunk)
            
        if len(outputs) == 1:
            return outputs[0]
        
        return torch.cat(outputs, dim=2)

    def _forward_chunk(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        kernel_pre_distorted: bool,
    ) -> torch.Tensor:
        # Reshape inputs for batch processing
        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size_for_jtc = (
            signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        )
        
        # Don't print warning inside chunk processing
        # if batch_size_for_jtc > 100000:
        #      print(f"JTC Batch Size: {batch_size_for_jtc} (Sig: {signal_full.shape})")
             
        signal_reshaped = signal_full.reshape(batch_size_for_jtc, self.input_length)
        kernel_reshaped = kernel_full.reshape(batch_size_for_jtc, self.kernel_length)

        # Step 1: DAC quantization
        signal_quantized = QuantDequant_STE.apply(signal_reshaped, self.config.dac_bits)
        if kernel_pre_distorted:
            kernel_distorted = kernel_reshaped
        else:
            kernel_quantized = QuantDequant_STE.apply(
                kernel_reshaped, self.config.dac_bits
            )
            kernel_distorted = self.input_distortion(kernel_quantized)

        # Step 2: Input distortion (signal & kernel)
        signal_distorted = self.input_distortion(signal_quantized)
        input_plane = self.build_input_plane(signal_distorted, kernel_distorted)

        # Step 3: FFT to Fourier plane
        jft = self.fft_and_magnitude(
            input_plane, buffer_name="_fourier_plane_buffer"
        )

        # Step 4: Output distortion
        jps = self.output_distortion(jft)

        # Step 5: Fourier plane quantization
        jps = QuantDequant_STE.apply(jps, self.config.fourier_plane_bits)

        # Step 6: Input distortion (no quantization before this)
        jps_distorted = self.input_distortion(jps)

        # Step 7: FFT to detector plane
        output_plane = self.fft_and_magnitude(
            jps_distorted, buffer_name="_detector_plane_buffer"
        )

        # Step 8: Output distortion
        output_plane = self.output_distortion(output_plane)

        # Step 9: ADC quantization
        output_plane = QuantDequant_STE.apply(output_plane, self.config.adc_bits)

        # Step 10: Index selection
        indices = self.compute_correlation_indices(output_plane.device)
        output = output_plane[..., indices]

        # Reshape output
        output_reshaped = output.reshape(
            signal_full.shape[0],
            signal_full.shape[1],
            signal_full.shape[2],
            self.output_length,
        )
        return output_reshaped
