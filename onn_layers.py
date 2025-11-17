import math
from typing import Any, Tuple
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from onn_config import AppConfig
from onn_component import JTC
from jtc_cycle_planner import compute_contamination_profile

__all__ = ["FTconvlayer"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_8(x: int, name: str):
    if x != 8:
        raise ValueError(f"{name} length must be 8 for this implementation. Got {x}.")


'''
class PIC(nn.Module):
    """In–memory implementation of the joint transform correlator used in the
    original template. The interface is kept identical so that the training
    script can be reused without changes."""

    def __init__(self, plane_size: int, sep: int):
        super().__init__()
        self.plane_size = plane_size
        self.sep = sep

    def _perform_jtc_correlation_batch(
        self, signal_batch: torch.Tensor, kernel_batch: torch.Tensor
    ) -> torch.Tensor:
        B = signal_batch.shape[0]
        M = signal_batch.shape[-1]
        N = kernel_batch.shape[-1]
        _check_8(M, "Signal")
        _check_8(N, "Kernel")

        kernel_complex_batch = kernel_batch.to(torch.complex64)
        signal_complex_batch = signal_batch.to(torch.complex64)

        plane_size = self.plane_size
        sep = self.sep

        input_plane_batch = torch.zeros(
            B, plane_size, dtype=torch.complex64, device=signal_batch.device
        )

        kernel_start = 0
        kernel_end = kernel_start + M
        signal_start = kernel_end + sep
        signal_end = signal_start + N

        input_plane_batch[:, kernel_start:kernel_end] = kernel_complex_batch
        input_plane_batch[:, signal_start:signal_end] = signal_complex_batch

        roll_amount = (plane_size // 2) - (M + signal_start) // 2
        input_plane_batch = torch.roll(input_plane_batch, shifts=roll_amount, dims=-1)

        jft_batch = torch.fft.fft(input_plane_batch, dim=-1)
        jft_batch = torch.fft.fftshift(jft_batch, dim=-1)
        jps_batch = torch.abs(jft_batch) ** 2 / plane_size

        output_plane_fft_batch = torch.fft.fft(jps_batch, dim=-1)
        output_plane_shifted_batch = torch.fft.fftshift(output_plane_fft_batch, dim=-1)
        output_plane_abs_batch = torch.abs(output_plane_shifted_batch)

        same_indices = (
            torch.arange(
                plane_size // 2 + sep + N // 2 + 1,
                plane_size // 2 + sep + N // 2 + 1 + 8,
                device=signal_batch.device,
            )
            % plane_size
        )
        return output_plane_abs_batch[:, same_indices]

    def forward(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ins = input.shape
        wes = weights.shape
        input_full = input.repeat(1, 1, wes[0], 1)
        weight_full = weights.repeat(ins[0], ins[1], 1, 1)

        batch_size_for_jtc = (
            input_full.shape[0] * input_full.shape[1] * input_full.shape[2]
        )
        signal_reshaped = input_full.reshape(batch_size_for_jtc, 8)
        kernel_reshaped = weight_full.reshape(batch_size_for_jtc, 8)

        correlation_output_batched = self._perform_jtc_correlation_batch(
            signal_reshaped, kernel_reshaped
        )
        output_reshaped = correlation_output_batched.reshape(
            input_full.shape[0], input_full.shape[1], input_full.shape[2], 8
        )
        return output_reshaped
'''


class _ConvNd(Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        batch_size: int,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        transposed: bool,
        output_padding: tuple[int, int],
        groups: int,
        bias: bool,
        padding_mode: str,
        kernel_length: int = None,  # Added for variable length support
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # Use kernel_length if provided, otherwise fall back to kernel_size
        weight_dim = kernel_length if kernel_length is not None else kernel_size
        self.weights = Parameter(
            torch.Tensor(in_channels, out_channels // groups, weight_dim, 2)
        )
        self.cout_per_cin = out_channels // groups
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class FTconvlayer(_ConvNd):
    """Photonic in–memory convolution layer that internally uses the PIC module
    to compute 1-D correlations across patches. Only the subset of features
    needed by the training script are retained."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: AppConfig,
        kernel_size: int = 8,
        batch_size: int = 128,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        vertical: bool = False,
        hv_concat: bool = False,
    ):
        # Store config first so we can access it
        self.config = config

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            batch_size,
            (stride, stride),
            (padding, padding),
            (dilation, dilation),
            False,
            (0, 0),
            groups,
            bias,
            padding_mode,
            kernel_length=config.kernel_length,  # Pass kernel_length from config
        )
        self.vertical = vertical
        self.hv_concat = hv_concat
        self.PIC_CONV = JTC(config)
        # Persist only the quantizer name (avoid lambdas for pickle safety)
        self.quantizer_name = (
            getattr(self.config, "quantizer", "ste_clipped") or "ste_clipped"
        )

    def _apply_quantizer(
        self, tensor: torch.Tensor, bits: int | None, domain: str
    ) -> torch.Tensor:
        """Apply configured quantizer by name for the given domain.
        domain in {activation, weight, output, fourier} controls signedness.
        If bits is None, no quantization is applied.
        """
        if bits is None:
            return tensor

        name = self.quantizer_name
        signed_for_domain = domain in ("weight",)
        is_weight_flag = domain in ("weight",)

        if name == "ste_clipped":
            return QAT_STE.apply(tensor, int(bits))
        if name == "ste_maxscale":
            return QAT_STE_maxscale.apply(tensor, int(bits))
        if name == "ios":
            return QAT_IOS.apply(tensor, int(bits), 1.0, bool(signed_for_domain))
        if name == "mad":
            return QAT_MAD.apply(tensor, int(bits), 1.0, bool(signed_for_domain))
        if name == "mph":
            return QAT_MPH.apply(tensor, int(bits), 1.0, bool(is_weight_flag))
        if name == "pwl":
            return QAT_PWL.apply(tensor, int(bits), 1.0, bool(signed_for_domain))
        # default fallback
        return QAT_STE.apply(tensor, int(bits))

    # ---------------- Internal helpers ------------------
    def _validate_backend_sizes(
        self, x: torch.Tensor, weight: torch.Tensor, backend: str
    ) -> None:
        """Validate tensor sizes for the specified backend.

        Size constraints:
        - pytorch: No specific size constraints, works with any input/weight size
        - fourier: Input width must match configured input_length, weight width must match kernel_length
        - jtc_emulation: Same as fourier, JTC parameters must be valid
        """
        import warnings

        match backend:
            case "pytorch":
                # PyTorch conv2d is flexible with sizes
                pass
            case "fourier" | "jtc_emulation":
                # Check that sizes match configuration
                if x.shape[-1] != self.config.input_length:
                    raise ValueError(
                        f"{backend.replace('_', ' ').title()} backend requires input width to be {self.config.input_length}. "
                        f"Got input width={x.shape[-1]}"
                    )
                if weight.shape[-1] != self.config.kernel_length:
                    raise ValueError(
                        f"{backend.replace('_', ' ').title()} backend requires weight width to be {self.config.kernel_length}. "
                        f"Got weight width={weight.shape[-1]}"
                    )
                # Check JTC plane sizing
                required_size = self.config.input_length + self.config.kernel_length + self.config.jtc_separation
                if self.config.jtc_total_field < required_size:
                    warnings.warn(
                        f"JTC total field ({self.config.jtc_total_field}) is smaller than "
                        f"required size ({required_size} = input_length + kernel_length + separation). "
                        f"This may cause errors or aliasing artifacts.",
                        UserWarning
                    )

    def jtc_emulation_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Full hardware JTC emulation pipeline with distortions."""
        return self.PIC_CONV(x, weight)

    def pytorch_conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch conv2d with 'same' padding as alternative to JTC."""
        # x shape: B H 1 W (batch, height, channels=1, width)
        # weight shape: Cout W (output_channels, width)

        # Reshape weight for conv2d: (out_channels, in_channels, kernel_height, kernel_width)
        # We use kernel_height=1 since we're doing 1D convolution along width
        weight_conv = weight.unsqueeze(1).unsqueeze(2)  # Cout W -> Cout 1 1 W

        # Apply conv2d with 'same' padding
        # x is B H 1 W, we need to transpose to B 1 H W for conv2d
        x_conv = x.transpose(1, 2)  # B H 1 W -> B 1 H W

        # Quantize inputs (extra bit for sign)
        if self.config.dac_bits is not None:
            x_conv = self._apply_quantizer(
                x_conv, self.config.dac_bits, domain="activation"
            )
            weight_conv = self._apply_quantizer(
                weight_conv, self.config.dac_bits, domain="weight"
            )

        # Apply convolution with same padding
        output = F.conv2d(x_conv, weight_conv, padding="same")  # B Cout H W

        # Transpose back to match expected output format: B Cout H W -> B H Cout W
        output = output.transpose(1, 2)  # B Cout H W -> B H Cout W

        # Optionally scale before ADC quantization
        max_val = output.max()
        if self.config.scale_output == "adc" and max_val.item() > 0:
            output = output / max_val

        # Apply output ADC quantization
        if self.config.adc_bits is not None:
            output = self._apply_quantizer(
                output, self.config.adc_bits, domain="output"
            )
        return output

    def fourier_conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor, preserve_phase: bool = False
    ) -> torch.Tensor:
        """Software JTC-style correlation via FFT with optional JPS quantization.

        Implements the provided block using plane_size=config.jtc_total_field and
        sep=config.jtc_separation, and quantizes at jps_batch if enabled.

        Shapes:
        - x: B H 1 W
        - weight: Cout W
        Returns: B H Cout W
        """
        if x.dim() != 4 or weight.dim() != 2:
            raise ValueError("Unexpected shapes for fourier_conv_forward")

        batch_size, height, in_ch, width = x.shape
        if in_ch != 1:
            raise ValueError(
                "fourier_conv_forward expects in_ch == 1 for local patch conv"
            )
        if width != self.kernel_size:
            raise ValueError("Patch width must equal kernel_size for fourier path")

        cout = weight.shape[0]

        if self.config.dac_bits is not None:
            x = self._apply_quantizer(x, self.config.dac_bits, domain="activation")
            weight = self._apply_quantizer(
                weight, self.config.dac_bits, domain="weight"
            )

        # Repeat to pair each signal with each kernel (per-output channel)
        input_full = x.repeat(1, 1, cout, 1)  # B H Cout W
        weight_full = (
            weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, height, 1, 1)
        )  # B H Cout W

        # Save shapes
        B = input_full.shape[0]
        H = input_full.shape[1]
        C = input_full.shape[2]
        M = input_full.shape[-1]  # input length
        N = weight_full.shape[-1]  # kernel length

        if input_full.shape[:-1] != weight_full.shape[:-1]:
            raise ValueError(
                "Input signal and kernel_weights must have matching batch dimensions."
            )

        # Flatten for batch JTC processing: [B*H*C, 8]
        batch_size_for_jtc = B * H * C
        signal_reshaped = input_full.reshape(batch_size_for_jtc, M)
        kernel_reshaped = weight_full.reshape(batch_size_for_jtc, N)

        # Parameters from config
        plane_size = int(self.config.jtc_total_field)
        sep = int(self.config.jtc_separation)

        # JTC (Joint Transform Correlator) simulation - emulating real optical physics
        # JTC computes correlation via Joint Power Spectrum, outputs are magnitudes (always positive)

        # Build input plane: place kernel [0:N], signal [N+sep:N+sep+M]
        input_plane = torch.zeros(
            batch_size_for_jtc, plane_size, dtype=torch.complex64, device=x.device
        )

        kernel_complex = kernel_reshaped.to(torch.complex64)
        signal_complex = signal_reshaped.to(torch.complex64)

        kernel_start = 0
        kernel_end = kernel_start + N
        signal_start = kernel_end + sep
        signal_end = signal_start + M

        input_plane[:, kernel_start:kernel_end] = kernel_complex
        input_plane[:, signal_start:signal_end] = signal_complex

        # Roll to center the input pattern
        roll_amount = (plane_size // 2) - (M + signal_start) // 2
        input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

        # JTC physics: FFT -> fftshift -> Joint Power Spectrum (JPS)
        jft = torch.fft.fft(input_plane, dim=-1)
        jft_shifted = torch.fft.fftshift(jft, dim=-1)
        jps = torch.abs(jft_shifted) ** 2
        jps = jps / plane_size

        # Quantize at JPS if enabled (Fourier plane quantization)
        if self.config.fourier_plane_bits is not None:
            jps = self._apply_quantizer(
                jps, self.config.fourier_plane_bits, domain="fourier"
            )

        # Back to output plane: FFT -> fftshift -> magnitude
        # Output is magnitude (light intensity), always positive
        output_plane_fft = torch.fft.fft(jps, dim=-1)
        output_plane_shifted = torch.fft.fftshift(output_plane_fft, dim=-1)
        if preserve_phase:
            correlation_plane = output_plane_shifted.real
        else:
            correlation_plane = torch.abs(output_plane_shifted)

        # Extract correlation output
        # Formula: same_start = plane_size//2 + sep + N//2
        # Note: Original formula had +1, but empirical analysis shows it should be removed
        same_start = plane_size // 2 + sep + N // 2

        # Extract full correlation (M+N-1 outputs) if config allows
        # For properly sized planes with adequate separation, full correlation is overlap-free
        if self.config.output_length is not None:
            output_length = self.config.output_length
        else:
            # Default: extract full correlation length (M+N-1)
            output_length = M + N - 1

        # Extract indices, wrapping around plane_size
        output_indices = torch.arange(
            same_start,
            same_start + output_length,
            device=x.device
        ) % plane_size

        convolution_output_batched = correlation_plane[:, output_indices]

        # Reshape back to B H Cout output_length
        out = convolution_output_batched.reshape(B, H, C, output_length)

        # Optional output scaling then ADC quantization
        max_val = out.max()
        if self.config.scale_output == "adc" and max_val.item() > 0:
            out = out / max_val

        if self.config.adc_bits is not None:
            out = self._apply_quantizer(out, self.config.adc_bits, domain="output")
        return out

    def _run_backend_patch(
        self, patch: torch.Tensor, weight: torch.Tensor, backend: str
    ) -> torch.Tensor:
        """Run the configured backend on a 1-D patch and return [B, Cout, L]."""
        self._validate_backend_sizes(patch, weight, backend)

        match backend:
            case "pytorch":
                out = self.pytorch_conv_forward(patch, weight)
            case "fourier":
                out = self.fourier_conv_forward(patch, weight)
            case "jtc_emulation":
                out = self.jtc_emulation_forward(patch, weight)
            case _:
                raise ValueError(
                    f"Unknown conv_backend: {backend}. "
                    f"Must be one of: 'pytorch', 'fourier', 'jtc_emulation'"
                )

        # Output shape: [B, H=1, Cout, L] → squeeze height dimension
        return out[:, 0, :, :]

    def conv_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Perform row-wise stitched convolution with contamination-aware stride."""
        backend = self.config.conv_backend or "jtc_emulation"

        batch_size, in_channels, height, width = x.shape
        patch_len = int(self.config.input_length)
        kernel_len = int(self.config.kernel_length)
        if patch_len <= 0 or kernel_len <= 0:
            raise ValueError("input_length and kernel_length must be positive")
        if patch_len < kernel_len:
            raise ValueError("input_length must be >= kernel_length")

        _, _, effective_stride = compute_contamination_profile(
            patch_len, kernel_len, self.config.jtc_total_field, self.config.jtc_separation
        )
        max_valid = max(1, patch_len - kernel_len + 1)
        stride = effective_stride if effective_stride and effective_stride > 0 else max_valid
        stride = min(stride, max_valid)

        total_outputs = (
            patch_len + kernel_len - 1
            if self.config.output_length is None
            else int(self.config.output_length)
        )
        pad_total = max(kernel_len - 1, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        valid_start = max(0, kernel_len - 1 - pad_left)
        x_padded = F.pad(x, (pad_left, pad_right, 0, 0))
        out_width = width

        output = torch.zeros(
            batch_size, self.out_channels, height, out_width, device=x.device, dtype=x.dtype
        )

        for c_in in range(in_channels):
            channel_data = x_padded[:, c_in, :, :]  # [B, H, W_pad]
            weight_c = weight[c_in, ...]  # [cout_per_cin, patch_len]
            c_out_start = (c_in % self.groups) * self.cout_per_cin
            c_out_end = c_out_start + self.cout_per_cin

            for row in range(height):
                row_data = channel_data[:, row, :]  # [B, W_pad]
                out_col = 0
                pass_idx = 0

                while out_col < out_width:
                    patch_start = pass_idx * stride
                    patch = row_data[:, patch_start : patch_start + patch_len]
                    if patch.shape[-1] < patch_len:
                        patch = F.pad(patch, (0, patch_len - patch.shape[-1]))
                    patch_4d = patch.view(batch_size, 1, 1, patch_len)

                    patch_out = self._run_backend_patch(patch_4d, weight_c, backend)
                    end_idx = min(total_outputs, valid_start + stride)
                    if end_idx <= valid_start:
                        break
                    valid_slice = patch_out[:, :, valid_start:end_idx]
                    usable = min(valid_slice.shape[-1], out_width - out_col)
                    if usable <= 0:
                        break

                    output[:, c_out_start:c_out_end, row, out_col : out_col + usable] += valid_slice[:, :, :usable]
                    out_col += usable
                    pass_idx += 1

        return output

    def pseudo_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight_p = weight[..., 0]
        weight_n = weight[..., 1]
        output_p = self.conv_forward(x, weight_p)
        output_n = self.conv_forward(x, weight_n)
        return output_p - output_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.hv_concat:
            conv_h = self.pseudo_forward(x, self.weights)
            conv_v = self.pseudo_forward(x.permute(0, 1, 3, 2), self.weights).permute(
                0, 1, 3, 2
            )
            conv_stacked = torch.stack([conv_h, conv_v], dim=2)
            return conv_stacked.view(conv_h.size(0), -1, conv_h.size(2), conv_h.size(3))
        if self.vertical:
            return self.pseudo_forward(x.permute(0, 1, 3, 2), self.weights).permute(
                0, 1, 3, 2
            )
        return self.pseudo_forward(x, self.weights)


def _uniform_quantize(x, bits: int, s: float = 1.0, signed: bool = False):
    """
    Forward: clip to [0,s] if unsigned else [-s,s], then uniform quantize.
    Returns (q, lo, hi, delta)
    """
    L = 2**bits
    if signed:
        lo, hi = -s, s
        delta = (hi - lo) / (L - 1)  # = 2*s/(L-1)
        xq = torch.clamp(x, lo, hi)
        q = torch.round((xq - lo) / delta) * delta + lo
    else:
        lo, hi = 0.0, s
        delta = (hi - lo) / (L - 1)  # = s/(L-1)
        xq = torch.clamp(x, lo, hi)
        q = torch.round((xq - lo) / delta) * delta + lo
    return q, lo, hi, delta


class QAT_IOS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, bits: int, s: float = 1.0, signed: bool = False
    ) -> torch.Tensor:
        q, lo, hi, _ = _uniform_quantize(x, bits, s, signed)
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return q

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        lo, hi = ctx.lo, ctx.hi
        inside = (x >= lo) & (x <= hi)
        pull_left = (x > hi) & (g < 0)  # decreasing x moves it inward
        pull_right = (x < lo) & (g > 0)  # increasing x moves it inward
        mask = (inside | pull_left | pull_right).to(g.dtype)
        return g * mask, None, None, None


class QAT_MAD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, bits: int, s: float = 1.0, signed: bool = True
    ) -> torch.Tensor:
        # MAD is intended for symmetric (signed) weight quantization.
        q, lo, hi, _ = _uniform_quantize(x, bits, s, signed=True if signed else False)
        ctx.save_for_backward(x)
        ctx.signed = signed
        ctx.s = s
        ctx.lo, ctx.hi = lo, hi
        return q

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        s = ctx.s
        lo, hi = ctx.lo, ctx.hi
        inside = (x >= lo) & (x <= hi)
        if ctx.signed:
            scale = (s / (x.abs() + 1e-12)).clamp_max(1.0)  # only used outside
            outside = (~inside).to(g.dtype)
            w = inside.to(g.dtype) + outside * scale.to(g.dtype)
        else:
            # unsigned MAD (right-clipping emphasis): 1 for x<=s, s/x for x>s, 0 for x<0
            right = x > hi
            left = x < lo
            center = ~right & ~left
            w = torch.zeros_like(g, dtype=g.dtype)
            w = torch.where(center, torch.ones_like(w), w)
            w = torch.where(right, (s / (x + 1e-12)).clamp_max(1.0).to(g.dtype), w)
            # left region stays 0 (acts like PWL on the left)
        return g * w, None, None, None


class QAT_MPH(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, bits: int, s: float = 1.0, is_weight: bool = True
    ) -> torch.Tensor:
        signed = bool(is_weight)  # weights → symmetric; activations → unsigned
        q, lo, hi, _ = _uniform_quantize(x, bits, s, signed)
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        ctx.s = s
        ctx.is_weight = is_weight
        return q

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        lo, hi, s = ctx.lo, ctx.hi, ctx.s
        if ctx.is_weight:
            # MAD (signed)
            inside = (x >= lo) & (x <= hi)
            scale = (s / (x.abs() + 1e-12)).clamp_max(1.0)
            outside = (~inside).to(g.dtype)
            w = inside.to(g.dtype) + outside * scale.to(g.dtype)
        else:
            # PWL (unsigned activations)
            w = ((x >= lo) & (x <= hi)).to(g.dtype)
        return g * w, None, None, None


class QAT_PWL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, bits: int, s: float = 1.0, signed: bool = False
    ) -> torch.Tensor:
        q, lo, hi, _ = _uniform_quantize(x, bits, s, signed)
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return q

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        lo, hi = ctx.lo, ctx.hi
        mask = ((x >= lo) & (x <= hi)).to(g.dtype)
        return g * mask, None, None, None


class QAT_STE_maxscale(torch.autograd.Function):
    # Quantization and dequantization with straight-thourgh estimator to help training
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int) -> torch.Tensor:
        levels = 2**bits

        # normalize to 0-1
        denom = (input.max() - input.min()).clamp_min(1e-12)
        input_norm = (input - input.min()) / denom
        # quantize
        quantized = torch.round(input_norm * (levels - 1)) / (levels - 1)
        # return to original range
        quantized = quantized * denom + input.min()
        return quantized

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Return gradients for (input, bits)
        return grad_output, None


class QAT_STE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int,
        s: float = 1.0,
    ) -> torch.Tensor:
        # save original input to know who was saturated
        ctx.save_for_backward(input)
        ctx.bits = bits  # not used in backward, but harmless to keep
        levels = 2**bits
        return torch.round(torch.clamp(input, 0.0, s) * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        return grad_output, None, None
