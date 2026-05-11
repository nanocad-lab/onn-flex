import torch


class STEQuantize(torch.autograd.Function):
    """Uniform [0, 1] quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int | None) -> torch.Tensor:
        if bits is None:
            return input
        levels_minus_one = float((2**int(bits)) - 1)
        if input.is_cuda and _COMPILED_QUANTIZE_FORWARD is not None:
            try:
                return _COMPILED_QUANTIZE_FORWARD(input, levels_minus_one)
            except Exception:
                _disable_compiled_quantize()
        return _ste_quantize_forward(input, levels_minus_one)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def quantize_ste(input: torch.Tensor, bits: int | None) -> torch.Tensor:
    return STEQuantize.apply(input, bits)


def _ste_quantize_forward(
    input: torch.Tensor, levels_minus_one: float
) -> torch.Tensor:
    input_clamped = torch.clamp(input, 0, 1)
    return torch.round(input_clamped * levels_minus_one) / levels_minus_one


def _disable_compiled_quantize() -> None:
    global _COMPILED_QUANTIZE_FORWARD
    _COMPILED_QUANTIZE_FORWARD = None


try:
    _COMPILED_QUANTIZE_FORWARD = torch.compile(_ste_quantize_forward)
except Exception:
    _COMPILED_QUANTIZE_FORWARD = None
