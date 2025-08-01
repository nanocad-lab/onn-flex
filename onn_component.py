from typing import Tuple

from onn_config import AppConfig
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import math

# NEW: Helper functions to compute ideal (reference) transfer function coefficients


class QuantDequant_STE(torch.autograd.Function):
    # version of the MRR LUT but implemented with straight-thourgh estimator to help training
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int) -> torch.Tensor:
        levels = 2**bits
        input_clamped = torch.clamp(input, 0, 1)
        return torch.round(input_clamped * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


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
    if x_last**2 == x_first**2:
        raise ValueError(
            "Input points for ideal quadratic interpolation are identical."
        )
    a = (y_last - y_first) / (x_last**2 - x_first**2)
    b = y_first - a * x_first**2
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
        # print(f"degree: {degree}, r2: {r2}, aic: {aic}")
        if r2 > 0.9995:
            return degree
        elif aic > last_aic:
            return degree - 1
        else:
            last_aic = aic

    return "fail"


def get_io_ranges(csv_file: str):
    data = pd.read_csv(csv_file)
    x = data["input"].values
    y = data["output"].values
    return x.min(), x.max(), y.min(), y.max()


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
        self.register_buffer(
            "ideal_coeffs", torch.as_tensor(ideal_coeffs, dtype=torch.float32)
        )

        # Distortion strength (0 -> ideal, 1 -> fitted polynomial)
        self.strength: float = float(self.config.driver_distortion_strength)

    def forward(self, x):
        # Polynomial evaluation using Horner's rule
        # print(f"x shape: {x.shape}")
        # print(f"x: {x[0]}")
        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y = poly_y * x + a

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

    def forward(self, x):
        # Polynomial evaluation using Horner's rule
        x = torch.clamp(x, 1e-6, 1e-5)
        # print(f"pd_tia x shape: {x.shape}")
        # print(f"pd_tia x: {x[0]}")
        # input("Press Enter to continue...")
        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y = poly_y * x + a

        # Ideal quadratic response
        ideal_y = self.ideal_coeffs[0] * torch.pow(x, 2) + self.ideal_coeffs[1]

        # print(f"ideal_y_shape: {ideal_y.shape}")
        # print(f"ideal_y: {ideal_y[0]}")
        # print(f"poly_y: {poly_y[0]}")
        # input("Press Enter to continue...")

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

    def forward(self, x):
        x = torch.clamp(x, 1e-6, 1e-5)
        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y = poly_y * x + a

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

    def forward(self, x):
        poly_y = torch.zeros_like(x, dtype=self.coeffs.dtype, device=x.device)
        for a in self.coeffs:
            poly_y = poly_y * x + a

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

    def forward(self, x):
        # Polynomial evaluation for power
        pwr_poly_y = torch.zeros_like(x, dtype=self.pwr_coeffs.dtype, device=x.device)
        for a in self.pwr_coeffs:
            pwr_poly_y = pwr_poly_y * x + a

        # Ideal linear response for power
        pwr_ideal_y = self.ideal_pwr_coeffs[0] * x + self.ideal_pwr_coeffs[1]

        pwr_y = self.pwr_strength * pwr_poly_y + (1.0 - self.pwr_strength) * pwr_ideal_y

        # Polynomial evaluation for phase
        phase_poly_y = torch.zeros_like(
            x, dtype=self.phase_coeffs.dtype, device=x.device
        )
        for a in self.phase_coeffs:
            phase_poly_y = phase_poly_y * x + a

        phase_y = self.phase_strength * phase_poly_y

        return torch.polar(pwr_y, phase_y)


class LER_variation(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.dim = self.config.jtc_total_field
        self.ler_std_dev = self.config.ler_std_dev

    def generate_ler_matrix(self, batch: int, length: int):
        """
        Balanced splitter tree (Gaussian i.i.d. ratios) that:
        • Handles non-powers of two by building to the next power-of-two (m)
            and center-cropping the m leaves down to n.
        • Supports an arbitrary batch dimension.
        • Returns a tensor of shape (batch, n) whose rows sum to n.
        """
        m = 1 << (length - 1).bit_length()  # smallest 2^k ≥ n
        levels = int(math.log2(m))

        powers = m * torch.ones((batch, 1))  # start with 1 W

        for _ in range(levels):
            k = powers.size(1)

            ratios = torch.normal(0.5, self.ler_std_dev, size=(batch, k)).clamp(0, 1)

            left = ratios * powers
            right = (1.0 - ratios) * powers

            # Interleave: L1,R1,L2,R2,…  — works for any batch size, including 1
            new_powers = torch.empty((batch, k * 2))
            new_powers[:, 0::2] = left
            new_powers[:, 1::2] = right
            powers = new_powers  # (batch, 2k)

        # Center-crop from m leaves down to n leaves
        if length < m:
            start = (m - length) // 2
            powers = powers[:, start : start + length]

        return powers

    def forward(self, x):
        ler_matrix = self.generate_ler_matrix(x.shape[0], x.shape[1])
        return torch.mul(x, ler_matrix)


class JTC(nn.Module):
    def __init__(self, config: AppConfig):
        super(JTC, self).__init__()
        self.config = config
        self.driver = Driver(config)
        self.mrm = MRM(config)
        self.pd_tia = PD_TIA(config)
        self.ler_variation = LER_variation(config)
        self.jtc_half_size = config.jtc_half_size
        self.jtc_separation = config.jtc_separation
        self.jtc_total_field = config.jtc_total_field
        self.loss = float(config.loss)

    def input_distortion(self, x):
        x = QuantDequant_STE.apply(x, self.config.dac_bits)
        x = self.driver(x)
        x = self.mrm(x)
        if self.config.ler_std_dev > 0:
            x = self.ler_variation(x)
        return x

    def output_distortion(self, x):
        x = x * self.loss
        x = self.pd_tia(x)
        if self.config.adc_scale_input:
            x = x / x.max()
        x = QuantDequant_STE.apply(x, self.config.adc_bits)
        return x

    def generate_input_plane(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """Apply input distortion and build the JTC input plane."""
        B = signal.shape[0]
        M = signal.shape[-1]
        N = kernel.shape[-1]
        # signal/kernel shapes are Bx32, 8
        # print(f"signal shape: {signal.shape}")
        # print(f"kernel shape: {kernel.shape}")

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

        input_plane = torch.zeros(
            B,
            self.jtc_total_field,
            dtype=torch.complex64,
            device=kernel_distorted.device,
        )
        input_plane[..., kernel_start:kernel_end] = kernel_distorted
        input_plane[..., signal_start:signal_end] = signal_distorted
        return input_plane

    def post_fft(self, input_plane: torch.Tensor) -> torch.Tensor:
        """Perform FFT and shift the result."""
        jft = torch.fft.fft(input_plane)
        jft = torch.fft.fftshift(jft)  # DC in center for optical lens
        return jft

    def post_output_distortion(self, jft: torch.Tensor) -> torch.Tensor:
        """Apply output distortion after the Fourier plane."""
        jps = self.output_distortion(torch.abs(jft))
        return jps

    def inverse_output(self, jps: torch.Tensor, N: int) -> torch.Tensor:
        """Propagate back to the detector plane and crop the result."""
        jps = self.input_distortion(jps)

        output_plane = torch.fft.fft(jps)
        output_plane = torch.fft.fftshift(output_plane)  # DC in center for optical lens
        output_plane = torch.abs(output_plane)
        output_plane = self.output_distortion(output_plane)

        same_indices = (
            torch.arange(
                self.jtc_total_field // 2
                + self.jtc_separation
                + self.jtc_half_size // 2
                + 1,
                self.jtc_total_field // 2
                + self.jtc_separation
                + self.jtc_half_size // 2
                + 1
                + self.jtc_half_size,
                device=jps.device,
            )
            % self.jtc_total_field
        )
        output_slice = output_plane[..., same_indices]
        return output_slice

    def forward(self, signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        # signal B H 1 W
        # kernel Cout W
        # print(f" JTC signal shape: {signal.shape}")
        # print(f" JTC kernel shape: {kernel.shape}")
        signal_full = signal.repeat(1, 1, kernel.shape[0], 1)
        kernel_full = kernel.repeat(signal.shape[0], signal.shape[1], 1, 1)
        batch_size_for_jtc = (
            signal_full.shape[0] * signal_full.shape[1] * signal_full.shape[2]
        )
        signal_reshaped = signal_full.reshape(batch_size_for_jtc, self.jtc_half_size)
        kernel_reshaped = kernel_full.reshape(batch_size_for_jtc, self.jtc_half_size)
        input_plane = self.generate_input_plane(signal_reshaped, kernel_reshaped)
        jft = self.post_fft(input_plane)
        jps = self.post_output_distortion(jft)
        # print(f"jps shape: ", jps.shape)
        # print("jps: ", jps[..., :8])
        # print(f"max: {jps.max()}, min: {jps.min()}")
        # input("Press Enter to continue...")
        inverse_output = self.inverse_output(jps, self.jtc_half_size)

        output_reshaped = inverse_output.reshape(
            signal_full.shape[0],
            signal_full.shape[1],
            signal_full.shape[2],
            self.jtc_half_size,
        )
        return output_reshaped
