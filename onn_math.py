import torch


def complex_abs_squared(x: torch.Tensor) -> torch.Tensor:
    """Return |x|^2 for a complex tensor without computing sqrt(|x|^2)."""
    parts = torch.view_as_real(x)
    if parts.dtype == torch.float16:
        parts = parts.float()
    return (parts * parts).sum(dim=-1)


def sqrt_nonnegative_with_finite_grad(x: torch.Tensor) -> torch.Tensor:
    """Return exact sqrt(max(x, 0)) with a finite surrogate gradient at zero."""
    exact = torch.sqrt(x.clamp_min(0.0))
    if not x.requires_grad:
        return exact
    eps = max(torch.finfo(x.dtype).tiny, 1e-12)
    safe = torch.sqrt(x.clamp_min(eps))
    return exact.detach() + safe - safe.detach()
