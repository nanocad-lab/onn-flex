import math

import torch


def complex_abs_squared(x: torch.Tensor) -> torch.Tensor:
    """Return |x|^2 for a complex tensor without computing sqrt(|x|^2)."""
    parts = torch.view_as_real(x)
    if parts.dtype == torch.float16:
        parts = parts.float()
    return (parts * parts).sum(dim=-1)


class _SqrtNonnegativeFiniteGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = torch.sqrt(input.clamp_min(0.0))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output,) = ctx.saved_tensors
        eps = max(torch.finfo(output.dtype).tiny, 1e-12)
        sqrt_eps = math.sqrt(eps)
        active = output >= sqrt_eps
        grad = grad_output / (2.0 * output.clamp_min(sqrt_eps))
        return torch.where(active, grad, torch.zeros_like(grad))


def sqrt_nonnegative_with_finite_grad(x: torch.Tensor) -> torch.Tensor:
    """Return exact sqrt(max(x, 0)) with a finite surrogate gradient at zero."""
    if not x.requires_grad:
        return torch.sqrt(x.clamp_min(0.0))
    return _SqrtNonnegativeFiniteGrad.apply(x)
