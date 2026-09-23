"""CUDA pointwise kernels with explicit rounding and first-order autograd.

Horner multiply/adds use round-to-nearest intrinsics, preventing FMA even
when an outer torch.compile rebuilds the Triton launch options. Fit
coefficients and readout gains are constants; input/weight gradients remain
connected through explicit backward kernels. Imported only for CUDA work.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _horner(
    X, C, G, Y, N, DEGREE: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    c = tl.load(C)
    y = libdevice.add_rn(libdevice.mul_rn(c, x), tl.load(C + 1))
    prefixes = (c,)
    for i in tl.static_range(2, DEGREE + 1):
        prefixes += (y,)
        y = libdevice.add_rn(libdevice.mul_rn(y, x), tl.load(C + i))
    if BACKWARD:
        g = tl.load(G + index, index < N, other=0)
        dx = libdevice.mul_rn(g, prefixes[DEGREE - 1])
        for i in tl.static_range(DEGREE - 2, -1, -1):
            g = libdevice.mul_rn(g, x)
            dx = libdevice.add_rn(dx, libdevice.mul_rn(g, prefixes[i]))
        y = dx
    tl.store(Y + index, y, index < N)


def _horner_forward(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    coefficients = coefficients.contiguous()
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _horner[triton.cdiv(x.numel(), 256),](
            x,
            coefficients,
            x,
            y,
            x.numel(),
            coefficients.numel() - 1,
            False,
            BLOCK=256,
            enable_fp_fusion=False,
        )
    return y


def _horner_backward(
    x: torch.Tensor, coefficients: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    x, gradient = (x.contiguous(), gradient.contiguous())
    coefficients = coefficients.contiguous()
    dx = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _horner[triton.cdiv(x.numel(), 256),](
            x,
            coefficients,
            gradient,
            dx,
            x.numel(),
            coefficients.numel() - 1,
            True,
            BLOCK=256,
            enable_fp_fusion=False,
        )
    return dx


@triton.jit
def _adc(
    X,
    GAIN,
    GRAD,
    Y,
    N,
    LEVELS: tl.constexpr,
    MAD: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    g = tl.load(GAIN)
    y = tl.minimum(tl.maximum(x, 0), 1)
    if LEVELS > 0:
        y = libdevice.nearbyint(y * LEVELS) * (1.0 / LEVELS)
    root = libdevice.sqrt_rn(y)
    root_gain = libdevice.sqrt_rn(g)
    if BACKWARD:
        dy = tl.load(GRAD + index, index < N, other=0)
        dy = tl.div_rn(dy, root_gain)
        dy = tl.div_rn(dy, 2.0 * tl.maximum(root, 1e-06))
        dy = tl.where(root >= 1e-06, dy, 0.0)
        inside = (x >= 0) & (x <= 1)
        if MAD:
            scale = tl.minimum(tl.div_rn(0.5, tl.abs(x - 0.5)), 1.0)
            scale = tl.where(inside, 1.0, scale)
        else:
            scale = inside.to(x.dtype)
        y = dy * scale
    else:
        y = tl.div_rn(root, root_gain)
    tl.store(Y + index, y, index < N)


def _adc_forward(
    x: torch.Tensor, gain: torch.Tensor, levels: int, mad: bool
) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _adc[triton.cdiv(x.numel(), 256),](
            x,
            gain,
            x,
            y,
            x.numel(),
            levels,
            mad,
            False,
            BLOCK=256,
            enable_fp_fusion=False,
        )
    return y


def _adc_backward(
    x: torch.Tensor, gain: torch.Tensor, gradient: torch.Tensor, levels: int, mad: bool
) -> torch.Tensor:
    x, gradient = (x.contiguous(), gradient.contiguous())
    dx = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _adc[triton.cdiv(x.numel(), 256),](
            x,
            gain,
            gradient,
            dx,
            x.numel(),
            levels,
            mad,
            True,
            BLOCK=256,
            enable_fp_fusion=False,
        )
    return dx


@triton.jit
def _affine_plane(
    X,
    Y,
    N,
    AP: tl.constexpr,
    BP: tl.constexpr,
    AT: tl.constexpr,
    BT: tl.constexpr,
    BIAS: tl.constexpr,
    AM: tl.constexpr,
    BM: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    if BACKWARD:
        y = libdevice.mul_rn(libdevice.mul_rn(libdevice.mul_rn(x, AM), AT), AP)
    else:
        y = libdevice.add_rn(libdevice.mul_rn(x, AP), BP)
        if BIAS != 0:
            y = libdevice.add_rn(y, BIAS)
        y = libdevice.add_rn(libdevice.mul_rn(y, AT), BT)
        y = libdevice.add_rn(libdevice.mul_rn(y, AM), BM)
    tl.store(Y + index, y, index < N)


def _affine_plane_forward(x: torch.Tensor, coefficients: list[float]) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _affine_plane[triton.cdiv(x.numel(), 256),](
            x, y, x.numel(), *coefficients, False, BLOCK=256, enable_fp_fusion=False
        )
    return y


def _affine_plane_backward(
    gradient: torch.Tensor, coefficients: list[float]
) -> torch.Tensor:
    gradient = gradient.contiguous()
    dx = torch.empty_like(gradient)
    with torch.cuda.device(gradient.device):
        _affine_plane[triton.cdiv(gradient.numel(), 256),](
            gradient,
            dx,
            gradient.numel(),
            *coefficients,
            True,
            BLOCK=256,
            enable_fp_fusion=False,
        )
    return dx


class _Horner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficients):
        ctx.save_for_backward(x, coefficients)
        return _horner_forward(x, coefficients)

    @staticmethod
    def backward(ctx, gradient):
        x, coefficients = ctx.saved_tensors
        return _horner_backward(x, coefficients, gradient), None


def horner(x, coefficients):
    return _Horner.apply(x, coefficients)


class _ADC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain, levels, mad):
        ctx.save_for_backward(x, gain)
        ctx.levels, ctx.mad = levels, mad
        return _adc_forward(x, gain, levels, mad)

    @staticmethod
    def backward(ctx, gradient):
        x, gain = ctx.saved_tensors
        return _adc_backward(x, gain, gradient, ctx.levels, ctx.mad), None, None, None


def adc_readout(x, gain, levels, mad):
    return _ADC.apply(x, gain, levels, mad)


class _Affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficients):
        ctx.coefficients = list(coefficients)
        return _affine_plane_forward(x, coefficients)

    @staticmethod
    def backward(ctx, gradient):
        return _affine_plane_backward(gradient, ctx.coefficients), None


def affine_plane(x, coefficients):
    return _Affine.apply(x, coefficients)


@triton.jit
def _intensity_kernel(
    SR,
    SI,
    KR,
    KI,
    MASK,
    GRAD,
    Y,
    N,
    INPUTS: tl.constexpr,
    OUTPUTS: tl.constexpr,
    BINS: tl.constexpr,
    FP32: tl.constexpr,
    DERIVATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    bin_index = index % BINS
    output = index // BINS % OUTPUTS
    channel = index // (BINS * OUTPUTS) % INPUTS
    position = index // (BINS * OUTPUTS * INPUTS)
    signal_index = (position * INPUTS + channel) * BINS + bin_index
    kernel_index = (channel * OUTPUTS + output) * BINS + bin_index
    mask = tl.load(MASK + bin_index)
    if DERIVATIVE != 2:
        re = libdevice.add_rn(
            tl.load(SR + signal_index, valid, other=0),
            tl.load(KR + kernel_index, valid, other=0),
        )
    if DERIVATIVE != 1:
        im = libdevice.add_rn(
            tl.load(SI + signal_index, valid, other=0),
            tl.load(KI + kernel_index, valid, other=0),
        )
    if DERIVATIVE == 0:
        power = libdevice.add_rn(libdevice.mul_rn(re, re), libdevice.mul_rn(im, im))
        if FP32:
            power = power.to(tl.float32)
        value = libdevice.mul_rn(power, mask.to(power.dtype))
    else:
        grad = tl.load(GRAD + index, valid, other=0)
        grad = libdevice.mul_rn(grad, mask.to(grad.dtype)).to(tl.float64)
        field = re if DERIVATIVE == 1 else im
        term = libdevice.mul_rn(grad, field)
        # Autograd adds the two contributions of x*x, in this order.
        value = libdevice.add_rn(term, term)
    tl.store(Y + index, value, valid)


class _Intensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sr, si, kr, ki, mask, dtype):
        sr, si, kr, ki = (t.contiguous() for t in (sr, si, kr, ki))
        positions, inputs, bins = sr.shape
        outputs = kr.shape[1]
        shape = (positions, inputs, outputs, bins)
        y = torch.empty(shape, device=sr.device, dtype=dtype)
        ctx.save_for_backward(sr, si, kr, ki, mask)
        ctx.shape = shape
        with torch.cuda.device(sr.device):
            _intensity_kernel[triton.cdiv(y.numel(), 256),](
                sr,
                si,
                kr,
                ki,
                mask,
                y,
                y,
                y.numel(),
                inputs,
                outputs,
                bins,
                dtype == torch.float32,
                0,
                BLOCK=256,
                enable_fp_fusion=False,
            )
        return y.reshape(-1, bins)

    @staticmethod
    def backward(ctx, gradient):
        sr, si, kr, ki, mask = ctx.saved_tensors
        gradient = gradient.contiguous()
        _, inputs, outputs, bins = ctx.shape
        # Reconstruct one expanded derivative at a time. Keep PyTorch's
        # original broadcast reductions instead of introducing atomic sums.
        values = []
        with torch.cuda.device(sr.device):
            for derivative in (1, 2):
                expanded = torch.empty(ctx.shape, device=sr.device, dtype=sr.dtype)
                _intensity_kernel[triton.cdiv(expanded.numel(), 256),](
                    sr,
                    si,
                    kr,
                    ki,
                    mask,
                    gradient,
                    expanded,
                    expanded.numel(),
                    inputs,
                    outputs,
                    bins,
                    gradient.dtype == torch.float32,
                    derivative,
                    BLOCK=256,
                    enable_fp_fusion=False,
                )
                values.append((expanded.sum(dim=2), expanded.sum(dim=0)))
                del expanded
        return values[0][0], values[1][0], values[0][1], values[1][1], None, None


def spectrum_intensity(sr, si, kr, ki, mask, dtype):
    """Broadcast aperture fields and detect power without saving expanded fields."""
    return _Intensity.apply(sr, si, kr, ki, mask, dtype)


@triton.jit
def _chain_poly(
    x, coefficients, gradient, DEGREE: tl.constexpr, BACKWARD: tl.constexpr
):
    c = tl.load(coefficients)
    y = libdevice.add_rn(libdevice.mul_rn(c, x), tl.load(coefficients + 1))
    prefixes = (c,)
    for i in tl.static_range(2, DEGREE + 1):
        prefixes += (y,)
        y = libdevice.add_rn(libdevice.mul_rn(y, x), tl.load(coefficients + i))
    if BACKWARD:
        y = libdevice.mul_rn(gradient, prefixes[DEGREE - 1])
        for i in tl.static_range(DEGREE - 2, -1, -1):
            gradient = libdevice.mul_rn(gradient, x)
            y = libdevice.add_rn(y, libdevice.mul_rn(gradient, prefixes[i]))
    return y


@triton.jit
def _detector_chain_kernel(
    X,
    PD,
    TIA,
    G,
    Y,
    N,
    PD_DEGREE: tl.constexpr,
    TIA_DEGREE: tl.constexpr,
    BIAS: tl.constexpr,
    AM: tl.constexpr,
    BM: tl.constexpr,
    REMOD: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + index, index < N, other=0)
    pd = _chain_poly(x, PD, x, PD_DEGREE, False)
    if BIAS != 0:
        pd = libdevice.add_rn(pd, BIAS)
    if BACKWARD:
        grad = tl.load(G + index, index < N, other=0)
        if REMOD:
            grad = libdevice.mul_rn(grad, AM)
        grad = _chain_poly(pd, TIA, grad, TIA_DEGREE, True)
        y = _chain_poly(x, PD, grad, PD_DEGREE, True)
    else:
        y = _chain_poly(pd, TIA, pd, TIA_DEGREE, False)
        if REMOD:
            y = libdevice.add_rn(libdevice.mul_rn(y, AM), BM)
    tl.store(Y + index, y, index < N)


class _DetectorChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pd, tia, bias, remodulation):
        x, pd, tia = (v.contiguous() for v in (x, pd, tia))
        y = torch.empty_like(x)
        ctx.save_for_backward(x, pd, tia)
        ctx.bias, ctx.remodulation = bias, remodulation
        am, bm = remodulation or (1.0, 0.0)
        with torch.cuda.device(x.device):
            _detector_chain_kernel[triton.cdiv(x.numel(), 256),](
                x,
                pd,
                tia,
                x,
                y,
                x.numel(),
                pd.numel() - 1,
                tia.numel() - 1,
                bias,
                am,
                bm,
                remodulation is not None,
                False,
                BLOCK=256,
                enable_fp_fusion=False,
            )
        return y

    @staticmethod
    def backward(ctx, gradient):
        x, pd, tia = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx = torch.empty_like(x)
        am, bm = ctx.remodulation or (1.0, 0.0)
        with torch.cuda.device(x.device):
            _detector_chain_kernel[triton.cdiv(x.numel(), 256),](
                x,
                pd,
                tia,
                gradient,
                dx,
                x.numel(),
                pd.numel() - 1,
                tia.numel() - 1,
                ctx.bias,
                am,
                bm,
                ctx.remodulation is not None,
                True,
                BLOCK=256,
                enable_fp_fusion=False,
            )
        return dx, None, None, None, None


def detector_chain(x, pd, tia, bias, remodulation=None):
    """PD→TIA and optional affine remodulation, preserving each rounding step."""
    return _DetectorChain.apply(x, pd, tia, bias, remodulation)
