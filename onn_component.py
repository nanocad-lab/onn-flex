from typing import Tuple

from onn_config import AppConfig
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# NEW: Helper functions to compute ideal (reference) transfer function coefficients

def _compute_linear_coeffs(csv_file: str):
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


def _compute_quadratic_coeffs(csv_file: str):
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
    if x_last ** 2 == x_first ** 2:
        raise ValueError("Input points for ideal quadratic interpolation are identical.")
    a = (y_last - y_first) / (x_last ** 2 - x_first ** 2)
    b = y_first - a * x_first ** 2
    return np.array([a, b], dtype=np.float32)


def calculate_aic(y_true, y_pred, n_params):
    """Calculate AIC for polynomial regression"""
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = 2 * n_params - 2 * log_likelihood
    return aic


def get_ideal_degree(csv_file, max_degree: int = 10):
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

    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        n_params = degree + 1  # coefficients + intercept
        aic = calculate_aic(y, y_pred, n_params)
        if r2 > 0.999:
            return degree
        elif aic > last_aic:
            return degree - 1
        else:
            last_aic = aic

    return "fail"


def get_coeffs(csv_file, degree: int):
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
        self.register_buffer("ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32))

        # Distortion strength (0 -> ideal, 1 -> fitted polynomial)
        self.strength: float = float(self.config.driver_distortion_strength)

    def forward(self, x):
        # Polynomial evaluation using Horner's rule
        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype)
        for a in self.coeffs:
            poly_y = poly_y * x + a

        # Ideal linear response
        ideal_y = self.ideal_coeffs[0] * x + self.ideal_coeffs[1]

        # Blend based on strength
        return self.strength * poly_y + (1.0 - self.strength) * ideal_y


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
        ideal_coeffs = _compute_quadratic_coeffs(self.config.pd_tia_distortion_data_path)
        self.register_buffer("ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32))

        # Distortion strength
        self.strength: float = float(self.config.pd_tia_distortion_strength)

    def forward(self, x):
        # Polynomial evaluation using Horner's rule
        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype)
        for a in self.coeffs:
            poly_y = poly_y * x + a

        # Ideal quadratic response
        ideal_y = self.ideal_coeffs[0] * torch.pow(x, 2) + self.ideal_coeffs[1]

        return self.strength * poly_y + (1.0 - self.strength) * ideal_y


class MRM(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.phase_degree: int = 0
        self.pwr_degree: int = 0
        if self.config.mrm_power_data_path is None:
            raise ValueError("MRM power data path is not set")
        if self.config.mrm_power_polyfit_order is None:
            self.pwr_degree = get_ideal_degree(self.config.mrm_power_data_path)
        else:
            self.pwr_degree = self.config.mrm_power_polyfit_order
        pwr_coeffs = get_coeffs(self.config.mrm_power_data_path, self.pwr_degree)
        pwr_coeff_tensor = torch.as_tensor(pwr_coeffs, dtype=torch.float32)
        self.register_buffer("pwr_coeffs", pwr_coeff_tensor)

        # Ideal power coefficients (linear)
        ideal_pwr_coeffs = _compute_linear_coeffs(self.config.mrm_power_data_path)
        self.register_buffer("ideal_pwr_coeffs", torch.as_tensor(ideal_pwr_coeffs, dtype=torch.float32))

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

        # Ideal phase coefficients (linear)
        ideal_phase_coeffs = _compute_linear_coeffs(self.config.mrm_phase_data_path)
        self.register_buffer("ideal_phase_coeffs", torch.as_tensor(ideal_phase_coeffs, dtype=torch.float32))

        # Distortion strength for phase
        self.phase_strength: float = float(self.config.mrm_phase_distortion_strength)

    def forward(self, x):
        # Polynomial evaluation for power
        pwr_poly_y = torch.zeros_like(x, dtype=self.pwr_coeffs.dtype)
        for a in self.pwr_coeffs:
            pwr_poly_y = pwr_poly_y * x + a

        # Ideal linear response for power
        pwr_ideal_y = self.ideal_pwr_coeffs[0] * x + self.ideal_pwr_coeffs[1]

        pwr_y = self.pwr_strength * pwr_poly_y + (1.0 - self.pwr_strength) * pwr_ideal_y

        # Polynomial evaluation for phase
        phase_poly_y = torch.zeros_like(x, dtype=self.phase_coeffs.dtype)
        for a in self.phase_coeffs:
            phase_poly_y = phase_poly_y * x + a

        # Ideal linear response for phase
        phase_ideal_y = self.ideal_phase_coeffs[0] * x + self.ideal_phase_coeffs[1]

        phase_y = self.phase_strength * phase_poly_y + (1.0 - self.phase_strength) * phase_ideal_y

        return torch.polar(pwr_y, phase_y)


class JTC(nn.Module):
    def __init__(self, config: AppConfig, driver: Driver, mrm: MRM, pd_tia: PD_TIA):
        super(JTC, self).__init__()
        self.config = config
        self.driver = driver
        self.mrm = mrm
        self.pd_tia = pd_tia

        self.jtc_half_size = config.jtc_half_size
        self.jtc_separation = config.jtc_separation
        self.jtc_total_field = config.jtc_total_field

    def input_distortion(self, x):
        x = self.driver(x)
        x = self.mrm(x)
        return x

    def output_distortion(self, x):
        x = self.pd_tia(x)
        return x

    def generate_input_plane(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """Apply input distortion and build the JTC input plane."""
        M = signal.shape[0]
        N = kernel.shape[0]

        if M > self.jtc_half_size:
            raise ValueError("Signal length is greater than JTC half size")
        if N > self.jtc_half_size:
            raise ValueError("Kernel length is greater than JTC half size")
        if M + N + self.jtc_separation > self.jtc_total_field:
            raise ValueError("Not enough JTC field")

        kernel_distorted = self.input_distortion(kernel)
        signal_distorted = self.input_distortion(signal)

        kernel_start = 0
        kernel_end = kernel_start + N

        signal_start = kernel_end + self.jtc_separation
        signal_end = signal_start + M

        input_plane = torch.zeros(self.jtc_total_field, dtype=torch.complex64)
        input_plane[kernel_start:kernel_end] = kernel_distorted
        input_plane[signal_start:signal_end] = signal_distorted

        return input_plane, M, N

    def post_fft(self, input_plane: torch.Tensor) -> torch.Tensor:
        """Perform FFT and shift the result."""
        jft = torch.fft.fft(input_plane)
        jft = torch.fft.fftshift(jft)  # DC in center for optical lens
        return jft

    def post_output_distortion(self, jft: torch.Tensor) -> torch.Tensor:
        """Apply output distortion after the Fourier plane."""
        jps = self.output_distortion(torch.abs(jft) ** 2)
        jps = jps / self.jtc_total_field  # fft normalization by L
        return jps

    def final_output(self, jps: torch.Tensor, N: int) -> torch.Tensor:
        """Propagate back to the detector plane and crop the result."""
        jps = self.input_distortion(jps)

        output_plane = torch.fft.fft(jps)
        output_plane = torch.fft.fftshift(output_plane)  # DC in center for optical lens
        output_plane = torch.abs(output_plane)
        output_plane = self.output_distortion(output_plane)

        same_indices = (
            torch.arange(
                self.jtc_total_field // 2 + self.jtc_separation + N // 2 + 1,
                self.jtc_total_field // 2 + self.jtc_separation + N // 2 + 1 + 8,
            )
            % self.jtc_total_field
        )
        return output_plane[same_indices]

    def forward(self, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        input_plane, _, N = self.generate_input_plane(signal, kernel)
        jft = self.post_fft(input_plane)
        jps = self.post_output_distortion(jft)
        return self.final_output(jps, N)
