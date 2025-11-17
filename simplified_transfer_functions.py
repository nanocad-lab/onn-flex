"""Simplified transfer functions using algebraic composition.

This module provides memory-efficient transfer function implementations that compose
cascaded functions into simplified polynomials, reducing computation and memory usage.
"""

import torch
import torch.nn as nn
from typing import Optional
from onn_config import AppConfig
from transfer_function_composer import (
    compose_driver_cascade,
    compose_pd_tia_cascade,
    evaluate_polynomial,
    reduce_polynomial_order,
    measure_fidelity
)
from onn_component import Driver, MRM, PD, TIA
import warnings


class ComposedDriverDouble(nn.Module):
    """Composed Driver(Driver(x)) as a single polynomial.

    This replaces two sequential Driver applications with a single polynomial
    evaluation, reducing computation and memory.
    """

    def __init__(
        self,
        driver: Driver,
        enable_simplification: bool = True,
        max_error: float = 1e-4
    ):
        """Initialize composed driver.

        Args:
            driver: Original Driver instance to compose with itself
            enable_simplification: If True, reduce polynomial order based on fidelity
            max_error: Maximum acceptable RMSE for order reduction
        """
        super().__init__()
        self.config = driver.config
        self.original_degree = driver.degree
        self.strength = driver.strength

        # Compose Driver(Driver(x))
        composed_coeffs = compose_driver_cascade(
            driver.coeffs,
            driver.ideal_coeffs,
            driver.strength
        )

        # Optionally reduce order
        if enable_simplification and max_error > 0:
            # Create reference function for fidelity testing
            def reference_fn(x):
                return driver(driver(x))

            # Reduce order while maintaining fidelity
            reduced_coeffs, reduced_order, metrics = reduce_polynomial_order(
                composed_coeffs,
                reference_fn,
                max_error=max_error,
                test_range=(0.0, 1.0),
                min_order=1,
                device="cpu"
            )

            self.register_buffer("coeffs", reduced_coeffs)
            self.final_order = reduced_order

            if reduced_order < len(composed_coeffs) - 1:
                print(f"ComposedDriverDouble: Reduced order from {len(composed_coeffs)-1} to {reduced_order} "
                      f"(RMSE: {metrics['rmse']:.2e})")
        else:
            self.register_buffer("coeffs", composed_coeffs)
            self.final_order = len(composed_coeffs) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply composed Driver(Driver(x)) transformation.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        return evaluate_polynomial(self.coeffs, x)


class ComposedPDTIA(nn.Module):
    """Composed TIA(PD(x)) as a single polynomial.

    This replaces sequential PD and TIA applications with a single polynomial
    evaluation, reducing computation and memory.
    """

    def __init__(
        self,
        pd: PD,
        tia: TIA,
        enable_simplification: bool = True,
        max_error: float = 1e-4
    ):
        """Initialize composed PD-TIA.

        Args:
            pd: Original PD instance
            tia: Original TIA instance
            enable_simplification: If True, reduce polynomial order based on fidelity
            max_error: Maximum acceptable RMSE for order reduction
        """
        super().__init__()
        self.config = pd.config
        self.pd_degree = pd.degree
        self.tia_degree = tia.degree

        # Compose TIA(PD(x))
        composed_coeffs = compose_pd_tia_cascade(
            pd.coeffs,
            pd.ideal_coeffs,
            pd.strength,
            tia.coeffs,
            tia.ideal_coeffs,
            tia.strength
        )

        # Optionally reduce order
        if enable_simplification and max_error > 0:
            # Create reference function for fidelity testing
            # Note: PD and TIA have clamping, so we need to include that
            def reference_fn(x):
                x_clamped = torch.clamp(x, 1e-6, 1e-5)
                y_pd = pd(x_clamped)
                y_tia = tia(y_pd)
                return y_tia

            # Test range matches PD input clamping
            reduced_coeffs, reduced_order, metrics = reduce_polynomial_order(
                composed_coeffs,
                reference_fn,
                max_error=max_error,
                test_range=(1e-6, 1e-5),
                min_order=1,
                device="cpu"
            )

            self.register_buffer("coeffs", reduced_coeffs)
            self.final_order = reduced_order

            if reduced_order < len(composed_coeffs) - 1:
                print(f"ComposedPDTIA: Reduced order from {len(composed_coeffs)-1} to {reduced_order} "
                      f"(RMSE: {metrics['rmse']:.2e})")
        else:
            self.register_buffer("coeffs", composed_coeffs)
            self.final_order = len(composed_coeffs) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply composed TIA(PD(x)) transformation.

        Note: Input clamping is applied to match original PD behavior.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor, clamped to [0, 1]
        """
        x = torch.clamp(x, 1e-6, 1e-5)
        result = evaluate_polynomial(self.coeffs, x)
        return torch.clamp(result, 0, 1)


class SimplifiedJTCTransferFunctions:
    """Factory for creating simplified transfer function combinations for JTC.

    This class analyzes the JTC pipeline and creates optimized transfer function
    implementations based on the usage pattern.
    """

    @staticmethod
    def create_simplified_functions(
        config: AppConfig,
        enable_simplification: bool = True,
        driver_max_error: float = 1e-4,
        pd_tia_max_error: float = 1e-4
    ):
        """Create simplified transfer functions for JTC.

        Analyzes the standard JTC pipeline where:
        1. Driver is applied twice: once to input, once to Fourier plane
        2. PD->TIA is applied twice: once to Fourier plane, once to output

        Args:
            config: Application configuration
            enable_simplification: Enable polynomial order reduction
            driver_max_error: Max RMSE for driver simplification
            pd_tia_max_error: Max RMSE for PD-TIA simplification

        Returns:
            Dictionary with simplified transfer functions:
            - 'driver_double': ComposedDriverDouble instance
            - 'pd_tia': ComposedPDTIA instance
            - 'mrm': Original MRM instance (complex, not simplified yet)
        """
        # Create original transfer functions
        original_driver = Driver(config)
        original_mrm = MRM(config)
        original_pd = PD(config)
        original_tia = TIA(config)

        # Create simplified versions
        simplified_driver_double = ComposedDriverDouble(
            original_driver,
            enable_simplification=enable_simplification,
            max_error=driver_max_error
        )

        simplified_pd_tia = ComposedPDTIA(
            original_pd,
            original_tia,
            enable_simplification=enable_simplification,
            max_error=pd_tia_max_error
        )

        return {
            'driver_double': simplified_driver_double,
            'pd_tia': simplified_pd_tia,
            'mrm': original_mrm,  # MRM is complex, keep as-is for now
            'original_driver': original_driver,  # Keep for fallback
            'original_pd': original_pd,
            'original_tia': original_tia
        }

    @staticmethod
    def validate_simplification(
        original_driver: Driver,
        original_pd: PD,
        original_tia: TIA,
        simplified_driver_double: ComposedDriverDouble,
        simplified_pd_tia: ComposedPDTIA,
        num_samples: int = 10000
    ):
        """Validate that simplified functions match original cascades.

        Args:
            original_driver: Original Driver instance
            original_pd: Original PD instance
            original_tia: Original TIA instance
            simplified_driver_double: Simplified Driver(Driver(x))
            simplified_pd_tia: Simplified TIA(PD(x))
            num_samples: Number of test samples

        Returns:
            Dictionary with validation metrics
        """
        # Test Driver(Driver(x))
        x_driver = torch.linspace(0.0, 1.0, num_samples)
        with torch.no_grad():
            y_original_driver = original_driver(original_driver(x_driver))
            y_simplified_driver = simplified_driver_double(x_driver)

        driver_error = torch.abs(y_original_driver - y_simplified_driver)

        # Test TIA(PD(x))
        x_pd_tia = torch.linspace(1e-6, 1e-5, num_samples)
        with torch.no_grad():
            y_pd = original_pd(x_pd_tia)
            y_original_pd_tia = original_tia(y_pd)
            y_simplified_pd_tia = simplified_pd_tia(x_pd_tia)

        pd_tia_error = torch.abs(y_original_pd_tia - y_simplified_pd_tia)

        metrics = {
            'driver_double': {
                'max_error': driver_error.max().item(),
                'mean_error': driver_error.mean().item(),
                'rmse': torch.sqrt(torch.mean(driver_error ** 2)).item(),
                'order_original': '2x' + str(original_driver.degree),
                'order_simplified': simplified_driver_double.final_order
            },
            'pd_tia': {
                'max_error': pd_tia_error.max().item(),
                'mean_error': pd_tia_error.mean().item(),
                'rmse': torch.sqrt(torch.mean(pd_tia_error ** 2)).item(),
                'order_original': f"PD({original_pd.degree}) + TIA({original_tia.degree})",
                'order_simplified': simplified_pd_tia.final_order
            }
        }

        return metrics
