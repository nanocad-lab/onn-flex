import torch

_VALID_CLAMP_GRAD_MODES = {"pwl", "mad"}


class STEQuantize(torch.autograd.Function):
    """Uniform [0, 1] quantization with STE only for rounding.

    The converter clamp is not straight-through: values outside the valid code
    range get zero gradient, matching the saturating DAC/ADC boundary.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int | None) -> torch.Tensor:
        if bits is None:
            ctx.quantized = False
            return input
        bits = int(bits)
        if bits < 1:
            raise ValueError("quantization bits must be >= 1 or None")
        ctx.quantized = True
        ctx.save_for_backward(input)
        levels_minus_one = float((2**bits) - 1)
        if input.is_cuda and _COMPILED_QUANTIZE_FORWARD is not None:
            try:
                return _COMPILED_QUANTIZE_FORWARD(input, levels_minus_one)
            except Exception:
                _disable_compiled_quantize()
        return _ste_quantize_forward(input, levels_minus_one)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        if not ctx.quantized:
            return grad_output, None
        (input,) = ctx.saved_tensors
        clamp_mask = (input >= 0) & (input <= 1)
        return grad_output * clamp_mask.to(dtype=grad_output.dtype), None


def quantize_ste(input: torch.Tensor, bits: int | None) -> torch.Tensor:
    return STEQuantize.apply(input, bits)


class ConverterClamp(torch.autograd.Function):
    """Hard [0, 1] converter clamp with selectable surrogate gradient."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, grad_mode: str) -> torch.Tensor:
        mode = str(grad_mode).lower()
        if mode not in _VALID_CLAMP_GRAD_MODES:
            raise ValueError(
                "converter clamp gradient mode must be one of "
                f"{sorted(_VALID_CLAMP_GRAD_MODES)}, got {grad_mode!r}"
            )
        ctx.grad_mode = mode
        ctx.save_for_backward(input)
        return torch.clamp(input, 0, 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (input,) = ctx.saved_tensors
        grad_scale = _converter_clamp_grad_scale(input, ctx.grad_mode)
        return grad_output * grad_scale.to(dtype=grad_output.dtype), None


def clamp_unit_interval(input: torch.Tensor, grad_mode: str = "pwl") -> torch.Tensor:
    """Saturate a converter-domain signal to normalized [0, 1] rails."""
    return ConverterClamp.apply(input, grad_mode)


def _converter_clamp_grad_scale(input: torch.Tensor, grad_mode: str) -> torch.Tensor:
    inside = (input >= 0) & (input <= 1)
    if grad_mode == "pwl":
        return inside.to(dtype=input.dtype)
    if grad_mode == "mad":
        distance = torch.abs(input - 0.5)
        eps = torch.finfo(input.dtype).eps
        outside_scale = 0.5 / distance.clamp_min(eps)
        outside_scale = outside_scale.clamp_max(1.0)
        return torch.where(
            inside,
            torch.ones_like(outside_scale),
            outside_scale,
        )
    raise RuntimeError(f"Unsupported converter clamp mode: {grad_mode}")


class ConverterQuantizeSTE(torch.autograd.Function):
    """Fused converter saturation plus optional uniform quantization."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int | None,
        clamp_grad: str,
    ) -> torch.Tensor:
        mode = str(clamp_grad).lower()
        if mode not in _VALID_CLAMP_GRAD_MODES:
            raise ValueError(
                "converter clamp gradient mode must be one of "
                f"{sorted(_VALID_CLAMP_GRAD_MODES)}, got {clamp_grad!r}"
            )
        if bits is not None:
            bits = int(bits)
            if bits < 1:
                raise ValueError("quantization bits must be >= 1 or None")

        ctx.grad_mode = mode
        if mode == "pwl":
            ctx.save_for_backward((input >= 0) & (input <= 1))
        else:
            ctx.save_for_backward(input)

        clamped = torch.clamp(input, 0, 1)
        if bits is None:
            return clamped
        levels_minus_one = float((2**bits) - 1)
        return torch.round(clamped * levels_minus_one) / levels_minus_one

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (saved,) = ctx.saved_tensors
        if ctx.grad_mode == "pwl":
            grad_scale = saved.to(dtype=grad_output.dtype)
        else:
            grad_scale = _converter_clamp_grad_scale(saved, ctx.grad_mode)
        return grad_output * grad_scale.to(dtype=grad_output.dtype), None, None


def converter_quantize_ste(
    input: torch.Tensor,
    bits: int | None,
    clamp_grad: str = "pwl",
) -> torch.Tensor:
    """Apply converter saturation, then optional STE quantization.

    `bits=None` means no quantization noise, but the physical converter still
    has normalized full-scale rails. Gradients through the saturation are the
    standard clamp gradients; the STE only bypasses quantization rounding.
    """
    return ConverterQuantizeSTE.apply(input, bits, clamp_grad)


class QuantizedTransferLUT(torch.autograd.Function):
    """Fused converter quantization plus transfer function via table lookup.

    A DAC-quantized signal only takes `2**bits` values, so any pointwise
    transfer function of it is a lookup into a per-level table instead of a
    long polynomial evaluation. `values[i]` must hold `transfer(i / L)` and
    `grad_values[i]` must hold `transfer'(i / L)` for `L = len(values) - 1`,
    reproducing the gradient of transfer-after-STE-quantization: the transfer
    slope at the quantized level times the converter clamp surrogate.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        values: torch.Tensor,
        grad_values: torch.Tensor,
        clamp_grad: str,
    ) -> torch.Tensor:
        mode = str(clamp_grad).lower()
        if mode not in _VALID_CLAMP_GRAD_MODES:
            raise ValueError(
                "converter clamp gradient mode must be one of "
                f"{sorted(_VALID_CLAMP_GRAD_MODES)}, got {clamp_grad!r}"
            )
        levels_minus_one = values.numel() - 1
        idx = (
            torch.clamp(input, 0, 1).mul(float(levels_minus_one)).round().to(torch.long)
        )
        ctx.grad_mode = mode
        if mode == "pwl":
            ctx.save_for_backward(idx, (input >= 0) & (input <= 1), grad_values)
        else:
            ctx.save_for_backward(idx, input, grad_values)
        return values[idx]

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        idx, saved, grad_values = ctx.saved_tensors
        if ctx.grad_mode == "pwl":
            clamp_scale = saved.to(dtype=grad_output.dtype)
        else:
            clamp_scale = _converter_clamp_grad_scale(saved, ctx.grad_mode).to(
                dtype=grad_output.dtype
            )
        return grad_output * grad_values[idx] * clamp_scale, None, None, None


def quantized_transfer_lut(
    input: torch.Tensor,
    values: torch.Tensor,
    grad_values: torch.Tensor,
    clamp_grad: str = "pwl",
) -> torch.Tensor:
    return QuantizedTransferLUT.apply(input, values, grad_values, clamp_grad)


def _ste_quantize_forward(input: torch.Tensor, levels_minus_one: float) -> torch.Tensor:
    input_clamped = torch.clamp(input, 0, 1)
    return torch.round(input_clamped * levels_minus_one) / levels_minus_one


def _disable_compiled_quantize() -> None:
    global _COMPILED_QUANTIZE_FORWARD
    _COMPILED_QUANTIZE_FORWARD = None


try:
    _COMPILED_QUANTIZE_FORWARD = torch.compile(_ste_quantize_forward)
except Exception:
    _COMPILED_QUANTIZE_FORWARD = None
