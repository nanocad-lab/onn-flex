import math
import contextlib
import inspect
import warnings
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from onn_config import AppConfig
from onn_component import JTC
from jtc_cycle_planner import compute_contamination_profile

__all__ = ["FTconvlayer", "FTConv2d"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    _CHECKPOINT_SUPPORTS_KWARGS = "use_reentrant" in inspect.signature(
        _torch_checkpoint
    ).parameters
except (TypeError, ValueError):  # pragma: no cover
    _CHECKPOINT_SUPPORTS_KWARGS = False


def _checkpoint(fn, *args: torch.Tensor) -> torch.Tensor:
    if _CHECKPOINT_SUPPORTS_KWARGS:
        return _torch_checkpoint(
            fn, *args, use_reentrant=False, preserve_rng_state=True
        )
    return _torch_checkpoint(fn, *args)


def _check_8(x: int, name: str):
    if x != 8:
        raise ValueError(f"{name} length must be 8 for this implementation. Got {x}.")


def _apply_quantizer_by_name(
    tensor: torch.Tensor, bits: int | None, quantizer_name: str, domain: str
) -> torch.Tensor:
    """Utility to apply a named quantizer with domain-aware signedness."""
    if bits is None:
        return tensor

    signed_for_domain = domain in ("weight",)
    is_weight_flag = domain in ("weight",)

    if quantizer_name == "ste_clipped":
        if signed_for_domain:
            return QAT_STE_Signed.apply(tensor, int(bits))
        return QAT_STE.apply(tensor, int(bits))
    if quantizer_name == "ste_maxscale":
        return QAT_STE_maxscale.apply(tensor, int(bits))
    if quantizer_name == "ios":
        return QAT_IOS.apply(tensor, int(bits), 1.0, bool(signed_for_domain))
    if quantizer_name == "mad":
        return QAT_MAD.apply(tensor, int(bits), 1.0, bool(signed_for_domain))
    if quantizer_name == "mph":
        return QAT_MPH.apply(tensor, int(bits), 1.0, bool(is_weight_flag))
    if quantizer_name == "pwl":
        return QAT_PWL.apply(tensor, int(bits), 1.0, bool(signed_for_domain))

    # default fallback
    return QAT_STE.apply(tensor, int(bits))


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
        backend_for_init = (config.conv_backend or "jtc_fast")
        self.PIC_CONV: Optional[JTC] = (
            JTC(config) if backend_for_init in ("jtc_fast", "jtc_emulation") else None
        )
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
        return _apply_quantizer_by_name(tensor, bits, self.quantizer_name, domain)

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
            case "fourier" | "jtc_emulation" | "jtc_fast":
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

    def _build_shifted_input_plane(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        plane_size: int,
        sep: int,
    ) -> torch.Tensor:
        """Place kernel and signal into the single JTC plane and center the pattern."""
        batch = signal.shape[0]
        M = signal.shape[-1]
        N = kernel.shape[-1]

        input_plane = torch.zeros(
            batch, plane_size, dtype=torch.complex64, device=signal.device
        )
        input_plane[:, 0:N] = kernel
        input_plane[:, N + sep : N + sep + M] = signal

        roll_amount = (plane_size // 2) - (M + N + sep) // 2
        return torch.roll(input_plane, shifts=roll_amount, dims=-1)

    def jtc_emulation_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Full hardware JTC emulation pipeline with distortions."""
        if self.PIC_CONV is None:
            self.PIC_CONV = JTC(self.config)
        return self.PIC_CONV(x, weight)

    def jtc_fast_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """High-throughput JTC path (vectorized but physics-faithful via JTC core)."""
        if self.PIC_CONV is None:
            self.PIC_CONV = JTC(self.config)
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
        self, x: torch.Tensor, weight: torch.Tensor
        ) -> torch.Tensor:
        """Ideal JTC formulation: |FFT(signal + kernel plane)|^2 -> IFFT (no distortions)."""
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
        plane_size = int(self.config.jtc_total_field)
        sep = int(self.config.jtc_separation)

        # Optional DAC quantization to match hardware bit depth
        x = self._apply_quantizer(x, self.config.dac_bits, domain="activation")
        weight = self._apply_quantizer(weight, self.config.dac_bits, domain="weight")

        input_full = x.repeat(1, 1, cout, 1)  # B H Cout W
        weight_full = weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, height, 1, 1)

        B = input_full.shape[0]
        H = input_full.shape[1]
        C = input_full.shape[2]
        M = input_full.shape[-1]
        N = weight_full.shape[-1]

        signal_reshaped = input_full.reshape(B * H * C, M).to(torch.complex64)
        kernel_reshaped = weight_full.reshape(B * H * C, N).to(torch.complex64)

        required_size = N + sep + M
        if required_size > plane_size:
            warnings.warn(
                f"JTC total field ({plane_size}) is smaller than required ({required_size}); "
                "results may exhibit aliasing.",
                UserWarning,
            )
            plane_size = required_size

        plane = self._build_shifted_input_plane(
            signal_reshaped, kernel_reshaped, plane_size=plane_size, sep=sep
        )

        jft = torch.fft.fftshift(torch.fft.fft(plane, dim=-1), dim=-1)
        jps = torch.abs(jft) ** 2 / plane_size
        if self.config.fourier_plane_bits is not None:
            jps = self._apply_quantizer(jps, self.config.fourier_plane_bits, domain="fourier")
        corr_plane = torch.fft.ifft(torch.fft.ifftshift(jps, dim=-1), dim=-1).real

        output_length = self.config.output_length or (M + N - 1)
        start = plane_size // 2 + sep + N // 2
        output_indices = torch.arange(
            start, start + output_length, device=x.device
        ) % plane_size

        out = corr_plane[:, output_indices].reshape(B, H, C, output_length)

        if self.config.adc_bits is not None:
            out = self._apply_quantizer(out, self.config.adc_bits, domain="output")
        return out

    def conv_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Original JTC implementation with variable length support
        x_shape = x.shape
        w = x.shape[2]
        output = torch.zeros(x_shape[0], self.out_channels, w, w, device=x.device)
        x = x.permute(0, 3, 1, 2)

        # Use configured input_length for patching
        patch_size = self.config.input_length

        for c_in in range(x.shape[2]):
            x_c = x[:, :, c_in : c_in + 1, ...]
            weight_c = weight[c_in, ...]
            n_patch = int(x.shape[3] / patch_size)
            c_out_start = (c_in % self.groups) * self.cout_per_cin
            c_out_end = c_out_start + self.cout_per_cin

            for i_p in range(n_patch):
                patch = x_c[..., patch_size * i_p : patch_size * i_p + patch_size]

                # Select backend based on conv_backend parameter
                backend = self.config.conv_backend or "jtc_fast"

                # Validate sizes for the selected backend
                self._validate_backend_sizes(patch, weight_c, backend)

                # Route to appropriate backend using match/case
                match backend:
                    case "pytorch":
                        system_out = self.pytorch_conv_forward(patch, weight_c).permute(
                            0, 2, 3, 1
                        )
                    case "jtc_fast":
                        system_out = self.jtc_fast_forward(patch, weight_c).permute(
                            0, 2, 3, 1
                        )
                    case "fourier":
                        system_out = self.fourier_conv_forward(patch, weight_c).permute(
                            0, 2, 3, 1
                        )
                    case "jtc_emulation":
                        system_out = self.jtc_emulation_forward(patch, weight_c).permute(
                            0, 2, 3, 1
                        )
                    case _:
                        raise ValueError(
                            f"Unknown conv_backend: {backend}. "
                            f"Must be one of: 'pytorch', 'fourier', 'jtc_fast', 'jtc_emulation'"
                        )
                # Get the actual output length
                actual_out_len = system_out.shape[2]
                usable_len = min(actual_out_len, patch_size)
                if usable_len < actual_out_len:
                    system_out = system_out[:, :, :usable_len, :]
                output[
                    :,
                    c_out_start:c_out_end,
                    patch_size * i_p : patch_size * i_p + usable_len,
                    :,
                ] += system_out
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


class FTConv2d(Module):
    """Row-wise photonic convolution layer with shared variable-length logic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        config: AppConfig,
        stride: int | tuple[int, int] = 1,
        padding: str = "same",
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_backend: Optional[str] = None,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise ValueError("FTConv2d currently supports groups=1 only.")
        stride_pair = _pair(stride)
        dilation_pair = _pair(dilation)
        if stride_pair != (1, 1) or dilation_pair != (1, 1):
            raise ValueError("Only stride=1 and dilation=1 are supported.")
        if padding != "same":
            raise ValueError("Only padding='same' is supported.")

        k_h, k_w = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.config = config
        self.conv_backend = conv_backend or config.conv_backend or "pytorch"
        if self.conv_backend not in ("pytorch", "fourier", "jtc_fast", "jtc_emulation"):
            raise ValueError(
                "FTConv2d backend must be 'pytorch', 'fourier', 'jtc_fast', or 'jtc_emulation', "
                f"got {self.conv_backend}"
            )

        self.quantizer_name = getattr(config, "quantizer", "ste_clipped") or "ste_clipped"
        self.differential_weights = bool(getattr(config, "differential_weights", True))
        self.patch_length = int(config.input_length)
        self.kernel_length = int(config.kernel_length)
        if self.kernel_length != k_w:
            raise ValueError(
                f"kernel_width ({k_w}) must match config.kernel_length ({self.kernel_length})"
            )
        if self.patch_length < self.kernel_length:
            raise ValueError(
                f"input_length ({self.patch_length}) must be >= kernel_length ({self.kernel_length})"
            )

        total_out, _, eff_stride = compute_contamination_profile(
            self.patch_length,
            self.kernel_length,
            int(config.jtc_total_field),
            int(config.jtc_separation),
        )
        valid_per_patch = self.patch_length - self.kernel_length + 1
        if valid_per_patch <= 0:
            raise ValueError("Patch configuration yields no valid outputs per pass.")
        if total_out == 0:
            raise ValueError("Invalid JTC geometry: no usable outputs.")
        self.valid_per_patch = valid_per_patch
        self.effective_stride = eff_stride if eff_stride > 0 else valid_per_patch

        pad_h_total = self.kernel_size[0] - 1
        pad_w_total = self.kernel_size[1] - 1
        self.pad_h = (pad_h_total // 2, pad_h_total - pad_h_total // 2)
        self.pad_w = (pad_w_total // 2, pad_w_total - pad_w_total // 2)

        self.weight = Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.jtc: Optional[JTC] = (
            JTC(config) if self.conv_backend in ("jtc_emulation", "jtc_fast") else None
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("FTConv2d expects input of shape [B, C, H, W].")
        x_padded = self._pad_input(x)
        batch_size, _, _, _ = x_padded.shape
        _, _, height, width = x.shape
        out = x.new_zeros(batch_size, self.out_channels, height, width)

        for k_row in range(self.kernel_size[0]):
            rows = x_padded[:, :, k_row : k_row + height, :]
            kernels = self.weight[:, :, k_row, :]
            out += self._row_convolution(rows, kernels, width)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.pad_w
        top, bottom = self.pad_h
        if left == right == top == bottom == 0:
            return x
        return F.pad(x, (left, right, top, bottom))

    def _row_convolution(
        self, rows: torch.Tensor, kernels: torch.Tensor, output_width: int
    ) -> torch.Tensor:
        batch_size, _, height, _ = rows.shape
        out = None
        for c_in in range(self.in_channels):
            signal = rows[:, c_in, :, :]
            kernel = kernels[:, c_in, :]
            contrib = self._single_channel_row_conv(signal, kernel, output_width)
            out = contrib if out is None else out + contrib
        if out is None:
            return rows.new_zeros(batch_size, self.out_channels, height, output_width)
        return out

    def _single_channel_row_conv(
        self, signal_rows: torch.Tensor, kernel: torch.Tensor, output_width: int
    ) -> torch.Tensor:
        batch_size, height, row_width = signal_rows.shape
        cout = kernel.shape[0]
        stride = max(1, min(self.effective_stride, self.valid_per_patch))
        max_start = max(row_width - self.patch_length, 0)
        out = None
        out_col = 0
        pass_idx = 0

        # Prepare kernel once per row-conv call (avoid recomputing bipolar split per patch).
        apply_bipolar = self.differential_weights and self.conv_backend in (
            "jtc_fast",
            "jtc_emulation",
        )
        if apply_bipolar:
            kernel_pos = torch.clamp(kernel, min=0.0)
            kernel_neg = torch.clamp(-kernel, min=0.0)
            kernel_prepared = torch.cat([kernel_pos, kernel_neg], dim=0)

            def _apply_patch_conv_prepared(p: torch.Tensor) -> torch.Tensor:
                if self.conv_backend == "jtc_fast":
                    out_cat = self._jtc_fast_patch_conv_unipolar(
                        p, kernel_prepared, weight_domain="activation"
                    )
                else:
                    out_cat = self._jtc_emulation_patch_conv_unipolar(p, kernel_prepared)
                out_pos, out_neg = out_cat.split(cout, dim=1)
                return out_pos - out_neg
        else:

            def _apply_patch_conv_prepared(p: torch.Tensor) -> torch.Tensor:
                return self._apply_patch_conv(p, kernel)

        while out_col < output_width:
            patch_start = min(pass_idx * stride, max_start)
            patch = signal_rows[..., patch_start : patch_start + self.patch_length]
            if patch.size(-1) < self.patch_length:
                patch = F.pad(patch, (0, self.patch_length - patch.size(-1)))
            patch_out = _apply_patch_conv_prepared(patch)
            valid_slice = patch_out[
                ..., self.kernel_length - 1 : self.kernel_length - 1 + self.valid_per_patch
            ]
            usable = min(stride, self.valid_per_patch, output_width - out_col)
            slice_offset = out_col - patch_start
            max_offset = self.valid_per_patch - usable
            if slice_offset < 0:
                slice_offset = 0
            if slice_offset > max_offset:
                slice_offset = max_offset
            slice_part = valid_slice[..., slice_offset : slice_offset + usable]
            # pad along width only: (left, right, top, bottom)
            padded = F.pad(slice_part, (out_col, output_width - out_col - usable, 0, 0))
            out = padded if out is None else out + padded
            out_col += usable
            pass_idx += 1

        if out is None:
            return signal_rows.new_zeros(batch_size, self.out_channels, height, output_width)
        return out

    def _apply_patch_conv(self, patch: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        if self.conv_backend == "pytorch":
            return self._pytorch_patch_conv(patch, kernel)
        if self.conv_backend == "fourier":
            return self._fourier_patch_conv(patch, kernel)
        if self.conv_backend == "jtc_fast":
            return self._jtc_fast_patch_conv_unipolar(patch, kernel, weight_domain="weight")
        if self.conv_backend == "jtc_emulation":
            return self._jtc_patch_conv(patch, kernel)
        raise RuntimeError(f"Unsupported backend {self.conv_backend}")

    def _pytorch_patch_conv(self, patch: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        batch_size, height, _ = patch.shape
        patch_flat = patch.reshape(batch_size * height, 1, self.patch_length)
        weight = kernel.unsqueeze(1)
        conv = F.conv1d(
            patch_flat,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_length - 1,
        )
        conv = conv.view(batch_size, height, self.out_channels, -1)
        return conv.permute(0, 2, 1, 3)

    def _jtc_vectorized_patch_conv(
        self,
        patch: torch.Tensor,
        kernel: torch.Tensor,
        *,
        apply_distortions: bool,
        apply_quantization: bool,
        use_cross_spectrum: bool = False,
        weight_domain: str = "weight",
    ) -> torch.Tensor:
        autocast_ctx = (
            torch.autocast(device_type=patch.device.type, enabled=False)
            if apply_distortions
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            patch = patch.float()
            kernel = kernel.float()

            batch_size, height, _ = patch.shape
            cout = kernel.shape[0]
            plane_size = int(self.config.jtc_total_field)
            sep = int(self.config.jtc_separation)

            patch_full = patch.unsqueeze(2).repeat(1, 1, cout, 1)  # B H Cout M
            kernel_full = kernel.unsqueeze(0).unsqueeze(0).repeat(batch_size, height, 1, 1)

            BHC = batch_size * height * cout
            M = patch_full.shape[-1]
            N = kernel_full.shape[-1]

            signal = patch_full.reshape(BHC, M)
            kernel_r = kernel_full.reshape(BHC, N)

            quant_fn = lambda t, b, d: _apply_quantizer_by_name(
                t, b, self.quantizer_name, d
            )

            if apply_quantization and self.config.dac_bits is not None:
                signal = quant_fn(signal, self.config.dac_bits, "activation")
                kernel_r = quant_fn(kernel_r, self.config.dac_bits, weight_domain)

            if apply_distortions:
                if self.jtc is None:
                    self.jtc = JTC(self.config)
                signal = self.jtc.mrm(self.jtc.driver(signal))
                kernel_r = self.jtc.mrm(self.jtc.driver(kernel_r))
            else:
                signal = signal.to(torch.complex64)
                kernel_r = kernel_r.to(torch.complex64)

            if use_cross_spectrum:
                kernel_pad = F.pad(kernel_r, (0, plane_size - N))
                signal_pad = F.pad(
                    signal, (N + sep, plane_size - (N + sep + M))
                )
                kernel_plane = kernel_pad
                signal_plane = signal_pad

                signal_fft = torch.fft.fft(signal_plane, dim=-1)
                kernel_fft = torch.fft.fft(kernel_plane, dim=-1)
                cross_spectrum = signal_fft * torch.conj(kernel_fft)
                corr_plane = torch.fft.ifft(cross_spectrum, dim=-1).real
            else:
                kernel_pad = F.pad(kernel_r, (0, plane_size - N))
                signal_pad = F.pad(
                    signal, (N + sep, plane_size - (N + sep + M))
                )
                plane = kernel_pad + signal_pad
                plane = torch.roll(
                    plane, shifts=(plane_size // 2) - (M + N + sep) // 2, dims=-1
                )

                freq = torch.fft.fft(plane, dim=-1)
                freq = torch.fft.fftshift(freq, dim=-1)
                jps = torch.abs(freq) ** 2 / plane_size

                if apply_quantization and self.config.fourier_plane_bits is not None:
                    jps = quant_fn(jps, self.config.fourier_plane_bits, "fourier")

                corr_plane = torch.fft.ifft(torch.fft.ifftshift(jps, dim=-1), dim=-1).real

            # Apply the optical loss multiplier whenever requested, even if we skip
            # other distortions for the ideal path.
            if apply_distortions or abs(float(self.config.loss) - 1.0) > 1e-6:
                corr_plane = corr_plane * float(self.config.loss)

            if apply_distortions:
                # Keep amplitudes inside PD/TIA operating window to avoid hard clamp.
                # Scale per (batch, height) across all output channels so differential
                # (+/-) rails share a common gain.
                corr_plane = corr_plane.reshape(batch_size, height, cout, plane_size)
                corr_plane = self.jtc.scale_to_range(corr_plane, 1e-6, 1e-5, dims=(-2, -1))
                corr_plane = self.jtc.pd(corr_plane)
                corr_plane = self.jtc.tia(corr_plane)
                if self.config.scale_output == "adc":
                    corr_plane = corr_plane / corr_plane.amax(dim=-1, keepdim=True).clamp_min(1e-12)
                corr_plane = corr_plane.reshape(BHC, plane_size)

            if apply_quantization and self.config.adc_bits is not None:
                corr_plane = quant_fn(corr_plane, self.config.adc_bits, "output")

            output_length = self.config.output_length or (M + N - 1)
            same_start = sep + 1 if use_cross_spectrum else plane_size // 2 + sep + N // 2
            idx = torch.arange(
                same_start, same_start + output_length, device=patch.device
            ) % plane_size

            out = corr_plane[:, idx].reshape(batch_size, height, cout, output_length)
            return out.permute(0, 2, 1, 3)

    def _fourier_patch_conv(self, patch: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return self._jtc_vectorized_patch_conv(
            patch,
            kernel,
            apply_distortions=False,
            apply_quantization=False,
            use_cross_spectrum=True,
        )

    def _jtc_fast_patch_conv(self, patch: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return self._jtc_fast_patch_conv_unipolar(patch, kernel, weight_domain="weight")

    def _jtc_fast_patch_conv_unipolar(
        self, patch: torch.Tensor, kernel: torch.Tensor, *, weight_domain: str
    ) -> torch.Tensor:
        def _run(p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            return self._jtc_vectorized_patch_conv(
                p,
                k,
                apply_distortions=True,
                apply_quantization=True,
                weight_domain=weight_domain,
            )

        if (
            self.training
            and torch.is_grad_enabled()
            and bool(getattr(self.config, "jtc_checkpoint", False))
        ):
            return _checkpoint(_run, patch, kernel)
        return _run(patch, kernel)

    def _jtc_emulation_patch_conv_unipolar(
        self, patch: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        if self.jtc is None:
            self.jtc = JTC(self.config)
        patch_4d = patch.unsqueeze(2)  # B, H, 1, W
        return self.jtc(patch_4d, kernel).permute(0, 2, 1, 3)

    def _jtc_patch_conv(self, patch: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        if self.jtc is None:
            self.jtc = JTC(self.config)
        patch_4d = patch.unsqueeze(2)  # B, H, 1, W
        kernel_pos = torch.clamp(kernel, min=0.0)
        kernel_neg = torch.clamp(-kernel, min=0.0)

        result_pos: Optional[torch.Tensor] = None
        result_neg: Optional[torch.Tensor] = None

        if torch.any(kernel_pos):
            result_pos = self.jtc(patch_4d, kernel_pos).permute(0, 2, 1, 3)
        if torch.any(kernel_neg):
            result_neg = self.jtc(patch_4d, kernel_neg).permute(0, 2, 1, 3)

        if result_pos is None and result_neg is None:
            return patch.new_zeros(
                patch.size(0),
                self.out_channels,
                patch.size(1),
                self.patch_length + self.kernel_length - 1,
            )
        if result_pos is None:
            return -result_neg  # type: ignore[return-value]
        if result_neg is None:
            return result_pos
        return result_pos - result_neg


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


class QAT_STE_Signed(torch.autograd.Function):
    """Signed, symmetric STE quantizer with per-tensor clipping scale.

    Matches the "ste_clipped" behavior used for activations, but supports signed
    tensors (e.g. weights) and preserves gradient flow via a straight-through
    estimator.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, bits: int) -> torch.Tensor:
        ctx.save_for_backward(input)
        levels = 2**bits
        input_fp32 = input.float()
        s = input_fp32.detach().abs().max().clamp_min(1e-6)
        step = 2 * s / (levels - 1)
        quantized = torch.round(torch.clamp(input_fp32, -s, s) / step) * step
        return quantized.to(input.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
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
