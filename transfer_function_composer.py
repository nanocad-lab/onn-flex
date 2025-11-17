"""Transfer function polynomial composition and simplification utilities.

This module provides tools to algebraically compose cascaded polynomial transfer functions
into simplified forms, reducing computation and memory during training.

Key capabilities:
1. Polynomial composition: f(g(x)) -> h(x) as a single polynomial
2. Fidelity measurement: compare composed vs cascaded transfer functions
3. Order reduction: find minimum polynomial order with acceptable fidelity loss
"""

import torch
import numpy as np
from typing import Tuple, Optional
import warnings


def compose_polynomials(
    outer_coeffs: torch.Tensor,
    inner_coeffs: torch.Tensor
) -> torch.Tensor:
    """Compose two polynomials f(g(x)) into a single polynomial h(x).

    Given:
    - f(y) = outer_coeffs[0] * y^n + ... + outer_coeffs[n]
    - g(x) = inner_coeffs[0] * x^m + ... + inner_coeffs[m]

    Computes h(x) = f(g(x)) as a polynomial in x.

    Args:
        outer_coeffs: Coefficients of outer polynomial f (highest degree first)
        inner_coeffs: Coefficients of inner polynomial g (highest degree first)

    Returns:
        Coefficients of composed polynomial h (highest degree first)

    Example:
        >>> # f(y) = 2y + 1, g(x) = x^2
        >>> outer = torch.tensor([2.0, 1.0])  # 2y + 1
        >>> inner = torch.tensor([1.0, 0.0, 0.0])  # x^2
        >>> h = compose_polynomials(outer, inner)  # 2x^2 + 1
    """
    # Convert to numpy for easier polynomial arithmetic
    outer = outer.cpu().numpy()
    inner = inner.cpu().numpy()

    # Start with the constant term (highest index in numpy poly representation)
    result = np.array([outer[-1]])

    # Build up the composition using Horner's rule in reverse
    # For each term in outer polynomial (starting from second-to-last)
    for i in range(len(outer) - 2, -1, -1):
        # Multiply current result by inner polynomial
        result = np.polymul(result, inner)
        # Add the next coefficient
        if len(result) < 1:
            result = np.array([outer[i]])
        else:
            result[-1] += outer[i]

    return torch.as_tensor(result, dtype=torch.float32)


def compose_linear_with_polynomial(
    linear_coeffs: torch.Tensor,
    poly_coeffs: torch.Tensor
) -> torch.Tensor:
    """Compose linear function f(y) = a*y + b with polynomial g(x).

    Optimized version for when outer function is linear.

    Args:
        linear_coeffs: [a, b] where f(y) = a*y + b
        poly_coeffs: Coefficients of g(x) (highest degree first)

    Returns:
        Coefficients of f(g(x)) = a*g(x) + b
    """
    a, b = linear_coeffs[0].item(), linear_coeffs[1].item()

    # f(g(x)) = a*g(x) + b
    # Just scale the polynomial and add constant
    result = poly_coeffs.clone()
    result = result * a
    result[-1] += b

    return result


def compose_quadratic_with_polynomial(
    quad_coeffs: torch.Tensor,
    poly_coeffs: torch.Tensor
) -> torch.Tensor:
    """Compose quadratic function f(y) = a*y^2 + b with polynomial g(x).

    Optimized version for when outer function is quadratic.

    Args:
        quad_coeffs: [a, b] where f(y) = a*y^2 + b
        poly_coeffs: Coefficients of g(x) (highest degree first)

    Returns:
        Coefficients of f(g(x)) = a*g(x)^2 + b
    """
    a, b = quad_coeffs[0].item(), quad_coeffs[1].item()

    # f(g(x)) = a * (g(x))^2 + b
    poly_np = poly_coeffs.cpu().numpy()
    squared = np.polymul(poly_np, poly_np)  # g(x)^2
    result = squared * a
    result[-1] += b

    return torch.as_tensor(result, dtype=torch.float32)


def evaluate_polynomial(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate polynomial using Horner's rule.

    Args:
        coeffs: Polynomial coefficients (highest degree first)
        x: Input tensor

    Returns:
        Polynomial evaluated at x
    """
    result = torch.zeros_like(x, dtype=coeffs.dtype, device=x.device)
    for a in coeffs:
        result = result * x + a
    return result


def measure_fidelity(
    original_fn,
    simplified_coeffs: torch.Tensor,
    test_range: Tuple[float, float] = (0.0, 1.0),
    num_samples: int = 10000,
    device: str = "cpu"
) -> Tuple[float, float, float]:
    """Measure fidelity of simplified polynomial vs original function cascade.

    Args:
        original_fn: Callable that takes tensor input and returns output (cascaded TFs)
        simplified_coeffs: Coefficients of simplified polynomial
        test_range: (min, max) range for test inputs
        num_samples: Number of test points
        device: Device for computation

    Returns:
        Tuple of (max_abs_error, mean_abs_error, rmse)
    """
    # Generate test inputs
    x = torch.linspace(test_range[0], test_range[1], num_samples, device=device)

    # Evaluate original cascaded function
    with torch.no_grad():
        y_original = original_fn(x)

        # Evaluate simplified polynomial
        y_simplified = evaluate_polynomial(simplified_coeffs.to(device), x)

    # Compute error metrics
    abs_error = torch.abs(y_original - y_simplified)
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    rmse = torch.sqrt(torch.mean((y_original - y_simplified) ** 2)).item()

    return max_abs_error, mean_abs_error, rmse


def reduce_polynomial_order(
    full_coeffs: torch.Tensor,
    original_fn,
    max_error: float = 1e-3,
    test_range: Tuple[float, float] = (0.0, 1.0),
    min_order: int = 1,
    device: str = "cpu"
) -> Tuple[torch.Tensor, int, dict]:
    """Find minimum polynomial order that maintains fidelity within threshold.

    Uses iterative order reduction with fidelity testing to find the simplest
    polynomial approximation that meets the error criteria.

    Args:
        full_coeffs: Full composed polynomial coefficients
        original_fn: Original cascaded transfer function for comparison
        max_error: Maximum acceptable RMSE
        test_range: Range for fidelity testing
        min_order: Minimum polynomial order to try
        device: Device for computation

    Returns:
        Tuple of:
        - Reduced coefficients
        - Selected order
        - Dict with error metrics for selected order
    """
    full_order = len(full_coeffs) - 1

    if full_order <= min_order:
        # Already at minimum, return as-is
        max_err, mean_err, rmse = measure_fidelity(
            original_fn, full_coeffs, test_range, device=device
        )
        return full_coeffs, full_order, {
            "max_error": max_err,
            "mean_error": mean_err,
            "rmse": rmse
        }

    # Generate test data
    x_test = torch.linspace(test_range[0], test_range[1], 10000, device=device)
    with torch.no_grad():
        y_target = original_fn(x_test)

    # Try reducing order from full down to min
    best_coeffs = full_coeffs
    best_order = full_order
    best_metrics = {}

    for order in range(full_order, min_order - 1, -1):
        # Fit polynomial of this order to the target function
        # Use least squares polynomial fitting
        x_np = x_test.cpu().numpy()
        y_np = y_target.cpu().numpy()

        # Fit polynomial of degree 'order'
        fitted_coeffs_np = np.polyfit(x_np, y_np, order)
        fitted_coeffs = torch.as_tensor(fitted_coeffs_np, dtype=torch.float32)

        # Measure fidelity
        max_err, mean_err, rmse = measure_fidelity(
            original_fn, fitted_coeffs, test_range, device=device
        )

        if rmse <= max_error:
            # This order meets the criteria
            best_coeffs = fitted_coeffs
            best_order = order
            best_metrics = {
                "max_error": max_err,
                "mean_error": mean_err,
                "rmse": rmse
            }
            # Continue trying lower orders
        else:
            # This order exceeds error threshold, stop reduction
            break

    if not best_metrics:
        # Even full order doesn't meet criteria - use it anyway with warning
        max_err, mean_err, rmse = measure_fidelity(
            original_fn, full_coeffs, test_range, device=device
        )
        warnings.warn(
            f"Cannot reduce polynomial order while maintaining error < {max_error}. "
            f"Full order {full_order} has RMSE {rmse:.2e}",
            UserWarning
        )
        best_coeffs = full_coeffs
        best_order = full_order
        best_metrics = {
            "max_error": max_err,
            "mean_error": mean_err,
            "rmse": rmse
        }

    return best_coeffs, best_order, best_metrics


def compose_driver_cascade(
    driver_coeffs: torch.Tensor,
    driver_ideal_coeffs: torch.Tensor,
    strength: float
) -> torch.Tensor:
    """Compose Driver(Driver(x)) into single polynomial.

    Driver applies: strength * poly(x) + (1-strength) * (a*x + b)
    Composing Driver(Driver(x)) gives a single polynomial.

    Args:
        driver_coeffs: Polynomial coefficients of driver distortion
        driver_ideal_coeffs: [a, b] for ideal linear response
        strength: Distortion strength [0, 1]

    Returns:
        Coefficients of Driver(Driver(x))
    """
    # First, create the blended polynomial for a single Driver application
    # y = strength * poly(x) + (1-strength) * (a*x + b)
    poly_coeffs = driver_coeffs.clone()
    ideal_coeffs = driver_ideal_coeffs.clone()

    # Single driver polynomial
    single_driver = poly_coeffs * strength

    # Add ideal component to last two coefficients (linear term)
    a, b = ideal_coeffs[0].item(), ideal_coeffs[1].item()
    if len(single_driver) >= 2:
        single_driver[-2] += (1 - strength) * a  # linear term
        single_driver[-1] += (1 - strength) * b  # constant term
    else:
        # If polynomial is just constant, extend it
        single_driver = torch.cat([
            torch.zeros(1, dtype=single_driver.dtype),
            single_driver
        ])
        single_driver[-2] = (1 - strength) * a
        single_driver[-1] += (1 - strength) * b

    # Compose: Driver(Driver(x)) = Driver(single_driver(x))
    composed = compose_polynomials(single_driver, single_driver)

    return composed


def compose_pd_tia_cascade(
    pd_coeffs: torch.Tensor,
    pd_ideal_coeffs: torch.Tensor,
    pd_strength: float,
    tia_coeffs: torch.Tensor,
    tia_ideal_coeffs: torch.Tensor,
    tia_strength: float
) -> torch.Tensor:
    """Compose TIA(PD(x)) into single polynomial.

    PD applies: strength * poly(x) + (1-strength) * (a*x^2 + b)
    TIA applies: strength * poly(x) + (1-strength) * (c*x + d)

    Args:
        pd_coeffs: PD polynomial coefficients
        pd_ideal_coeffs: [a, b] for PD ideal quadratic
        pd_strength: PD distortion strength
        tia_coeffs: TIA polynomial coefficients
        tia_ideal_coeffs: [c, d] for TIA ideal linear
        tia_strength: TIA distortion strength

    Returns:
        Coefficients of TIA(PD(x))
    """
    # Build PD blended polynomial
    pd_poly = pd_coeffs * pd_strength
    a, b = pd_ideal_coeffs[0].item(), pd_ideal_coeffs[1].item()

    # Ensure pd_poly has at least 3 coefficients for quadratic
    while len(pd_poly) < 3:
        pd_poly = torch.cat([torch.zeros(1, dtype=pd_poly.dtype), pd_poly])

    pd_poly[-3] += (1 - pd_strength) * a  # quadratic term
    pd_poly[-1] += (1 - pd_strength) * b  # constant term

    # Build TIA blended polynomial
    tia_poly = tia_coeffs * tia_strength
    c, d = tia_ideal_coeffs[0].item(), tia_ideal_coeffs[1].item()

    while len(tia_poly) < 2:
        tia_poly = torch.cat([torch.zeros(1, dtype=tia_poly.dtype), tia_poly])

    tia_poly[-2] += (1 - tia_strength) * c  # linear term
    tia_poly[-1] += (1 - tia_strength) * d  # constant term

    # Compose: TIA(PD(x))
    composed = compose_polynomials(tia_poly, pd_poly)

    return composed
