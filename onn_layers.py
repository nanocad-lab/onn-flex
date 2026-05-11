import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from onn_config import AppConfig
from onn_component import JTC
from onn_jtc_conv2d import JTCConv2d, replace_conv2d_with_jtc
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import quantize_ste

__all__ = ["FTconvlayer", "JTCConv2d", "replace_conv2d_with_jtc"]


class _ConvNd(nn.Module):
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
        kernel_length: int | None = None,
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
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, weight_dim, 2)
        )
        self.cout_per_cin = out_channels // groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
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
        self.PIC_CONV = None
        if self.config.conv_backend == "jtc_emulation":
            self.PIC_CONV = JTC(config)

    # ---------------- Internal helpers ------------------
    def _validate_backend_sizes(
        self, x: torch.Tensor, weight: torch.Tensor, backend: str
    ) -> None:
        """Validate tensor sizes for the specified backend.

        Size constraints:
        - pytorch: No specific size constraints, works with any input/weight size
        - jtc_ideal: Input width must match configured input_length, weight width must match kernel_length
        - jtc_emulation: Same as jtc_ideal, JTC parameters must be valid
        """
        import warnings

        match backend:
            case "pytorch":
                # PyTorch conv2d is flexible with sizes
                pass
            case "jtc_ideal" | "jtc_emulation":
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
                required_size = (
                    self.config.input_length
                    + self.config.kernel_length
                    + self.config.jtc_separation
                )
                if self.config.jtc_total_field < required_size:
                    warnings.warn(
                        f"JTC total field ({self.config.jtc_total_field}) is smaller than "
                        f"required size ({required_size} = input_length + kernel_length + separation). "
                        f"This may cause errors or aliasing artifacts.",
                        UserWarning,
                    )

    def jtc_emulation_forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Full hardware JTC emulation pipeline with distortions."""
        return self._pic_conv(x.device)(x, weight)

    def jtc_emulation_forward_paired(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Full hardware JTC emulation for explicit signal/kernel shot pairs."""
        return self._pic_conv(signal.device).forward_paired(signal, kernel)

    def _pic_conv(self, device: torch.device) -> JTC:
        if self.PIC_CONV is None:
            self.PIC_CONV = JTC(self.config).to(device=device)
        return self.PIC_CONV

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
            x_conv = quantize_ste(x_conv, self.config.dac_bits)
            weight_conv = quantize_ste(weight_conv, self.config.dac_bits)

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
            output = quantize_ste(output, self.config.adc_bits)
        return output

    def jtc_ideal_forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Ideal JTC-style correlation via FFT with optional JPS quantization.

        Implements the provided block using plane_size=config.jtc_total_field and
        sep=config.jtc_separation, and quantizes at jps_batch if enabled.

        Shapes:
        - x: B H 1 W
        - weight: Cout W
        Returns: B H Cout W
        """
        if x.dim() != 4 or weight.dim() != 2:
            raise ValueError("Unexpected shapes for jtc_ideal_forward")

        batch_size, height, in_ch, width = x.shape
        if in_ch != 1:
            raise ValueError(
                "jtc_ideal_forward expects in_ch == 1 for local patch conv"
            )
        if width != self.kernel_size:
            raise ValueError("Patch width must equal kernel_size for jtc_ideal path")

        cout = weight.shape[0]

        if self.config.dac_bits is not None:
            x = quantize_ste(x, self.config.dac_bits)
            weight = quantize_ste(weight, self.config.dac_bits)

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
        sep = int(self.config.jtc_separation)
        plane_size = max(int(self.config.jtc_total_field), M + N + sep)

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
        jps = complex_abs_squared(jft_shifted) / plane_size

        if self.config.adc_bits is not None:
            jps = quantize_ste(jps, self.config.adc_bits)

        # Quantize at JPS if enabled (Fourier plane quantization)
        if self.config.fourier_plane_bits is not None:
            jps = quantize_ste(jps, self.config.fourier_plane_bits)

        if self.config.dac_bits is not None:
            jps = quantize_ste(jps, self.config.dac_bits)

        # Back to output plane: FFT -> fftshift -> magnitude, then final
        # square-law detection with ideal sqrt amplitude recovery.
        output_plane_fft = torch.fft.fft(jps, dim=-1)
        output_plane_shifted = torch.fft.fftshift(output_plane_fft, dim=-1)
        output_plane_abs = torch.abs(output_plane_shifted)
        output_plane_power = output_plane_abs * output_plane_abs
        if self.config.scale_output == "adc":
            output_plane_power = output_plane_power / output_plane_power.max().clamp_min(
                1e-12
            )
        output_plane_power = quantize_ste(output_plane_power, self.config.adc_bits)
        output_plane = sqrt_nonnegative_with_finite_grad(output_plane_power)

        # Extract full correlation (M+N-1 outputs) if config allows
        # For properly sized planes with adequate separation, full correlation is overlap-free
        if self.config.output_length is not None:
            output_length = self.config.output_length
        else:
            # Default: extract full correlation length (M+N-1)
            output_length = M + N - 1

        same_start = plane_size // 2 + sep + N // 2
        if output_length == M:
            # PyTorch's padding="same" returns the centered M samples from the
            # full linear correlation. Under this JTC placement the extracted
            # cross-correlation term is one sample earlier, so advance the
            # output window to make the ideal JTC backend match PyTorch.
            same_start += 1

        # Extract indices, wrapping around plane_size
        output_indices = (
            torch.arange(same_start, same_start + output_length, device=x.device)
            % plane_size
        )

        convolution_output_batched = output_plane[:, output_indices]

        # Reshape back to B H Cout output_length
        out = convolution_output_batched.reshape(B, H, C, output_length)

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
                backend = self.config.conv_backend

                # Validate sizes for the selected backend
                self._validate_backend_sizes(patch, weight_c, backend)

                # Route to appropriate backend using match/case
                match backend:
                    case "pytorch":
                        system_out = self.pytorch_conv_forward(patch, weight_c).permute(
                            0, 2, 3, 1
                        )
                    case "jtc_ideal":
                        system_out = self.jtc_ideal_forward(patch, weight_c).permute(
                            0, 2, 3, 1
                        )
                    case "jtc_emulation":
                        system_out = self.jtc_emulation_forward(
                            patch, weight_c
                        ).permute(0, 2, 3, 1)
                    case _:
                        raise ValueError(
                            f"Unknown conv_backend: {backend}. "
                            f"Must be one of: 'pytorch', 'jtc_ideal', 'jtc_emulation'"
                        )
                # Get the actual output length
                actual_out_len = system_out.shape[2]
                start = patch_size * i_p
                end = min(start + actual_out_len, output.shape[2])
                usable_len = end - start
                if usable_len <= 0:
                    continue
                output[
                    :,
                    c_out_start:c_out_end,
                    start:end,
                    :,
                ] += system_out[:, :, :usable_len, :]
        return output

    def _batched_patch_input(
        self, x_c: torch.Tensor, patch_size: int, n_patch: int
    ) -> torch.Tensor:
        patches = x_c[..., : n_patch * patch_size].unfold(
            dimension=-1, size=patch_size, step=patch_size
        )
        patches = patches.squeeze(2).permute(0, 2, 1, 3).contiguous()
        return patches.reshape(-1, x_c.shape[1], 1, patch_size)

    def _accumulate_batched_patch_output(
        self,
        output: torch.Tensor,
        system_out: torch.Tensor,
        c_out_start: int,
        c_out_end: int,
        patch_size: int,
        n_patch: int,
    ) -> None:
        batch_size = output.shape[0]
        cout = c_out_end - c_out_start
        output_length = system_out.shape[-1]
        rows = system_out.shape[1]
        out = system_out.reshape(batch_size, n_patch, rows, cout, output_length)
        out = out.permute(0, 3, 1, 4, 2).contiguous()
        for i_p in range(n_patch):
            start = patch_size * i_p
            end = min(start + output_length, output.shape[2])
            usable_len = end - start
            if usable_len <= 0:
                continue
            output[:, c_out_start:c_out_end, start:end, :] += out[
                :, :, i_p, :usable_len, :
            ]

    def _conv_forward_jtc_batched_signed(
        self,
        x: torch.Tensor,
        weight_p: torch.Tensor,
        weight_n: torch.Tensor,
    ) -> torch.Tensor:
        x_shape = x.shape
        w = x.shape[2]
        x = x.permute(0, 3, 1, 2)

        patch_size = self.config.input_length
        n_patch = int(x.shape[3] / patch_size)
        if n_patch <= 0:
            return torch.zeros(x_shape[0], self.out_channels, w, w, device=x.device)

        batch_size = x.shape[0]
        rows = x.shape[1]
        in_channels = x.shape[2]
        cout = self.cout_per_cin
        out_len = self._pic_conv(x.device).output_length

        patches = x[..., : n_patch * patch_size].unfold(
            dimension=-1, size=patch_size, step=patch_size
        )
        patches = patches.permute(0, 3, 1, 2, 4).contiguous()

        weight_c = torch.stack([weight_p, weight_n], dim=1)
        weight_c = weight_c.reshape(in_channels, 2 * cout, self.config.kernel_length)

        signal_pairs = (
            patches.unsqueeze(4)
            .expand(batch_size, n_patch, rows, in_channels, 2 * cout, patch_size)
            .reshape(-1, patch_size)
        )
        kernel_pairs = (
            weight_c.reshape(1, 1, 1, in_channels, 2 * cout, self.config.kernel_length)
            .expand(batch_size, n_patch, rows, in_channels, 2 * cout, -1)
            .reshape(-1, self.config.kernel_length)
        )

        system_out = self.jtc_emulation_forward_paired(signal_pairs, kernel_pairs)
        system_out = system_out.reshape(
            batch_size, n_patch, rows, in_channels, 2, cout, out_len
        )

        # Differential signed weights are modeled as two accumulated optical
        # banks followed by one subtraction: sum(pos) - sum(neg).
        # This preserves the old arithmetic order and avoids quantization-sensitive
        # float32 differences from sum(pos - neg).
        if out_len == patch_size and n_patch * patch_size == w:
            pos_channels = system_out[:, :, :, :, 0, :, :].unbind(dim=3)
            neg_channels = system_out[:, :, :, :, 1, :, :].unbind(dim=3)
            group_outputs = []
            for group_idx in range(self.groups):
                channel_indices = range(group_idx, in_channels, self.groups)
                first_c_in = next(iter(channel_indices))
                output_p = pos_channels[first_c_in]
                output_n = neg_channels[first_c_in]
                for c_in in range(first_c_in + self.groups, in_channels, self.groups):
                    output_p = output_p + pos_channels[c_in]
                    output_n = output_n + neg_channels[c_in]
                group_outputs.append(
                    (output_p - output_n)
                    .permute(0, 3, 1, 4, 2)
                    .reshape(batch_size, cout, w, rows)
                )
            return torch.cat(group_outputs, dim=1)

        output_p = torch.zeros(x_shape[0], self.out_channels, w, w, device=x.device)
        output_n = torch.zeros_like(output_p)
        for c_in in range(in_channels):
            c_out_start = (c_in % self.groups) * cout
            c_out_end = c_out_start + cout
            system_out_p = system_out[:, :, :, c_in, 0, :, :]
            system_out_n = system_out[:, :, :, c_in, 1, :, :]
            system_out_p = system_out_p.reshape(batch_size * n_patch, rows, cout, out_len)
            system_out_n = system_out_n.reshape(batch_size * n_patch, rows, cout, out_len)
            self._accumulate_batched_patch_output(
                output_p,
                system_out_p,
                c_out_start,
                c_out_end,
                patch_size,
                n_patch,
            )
            self._accumulate_batched_patch_output(
                output_n,
                system_out_n,
                c_out_start,
                c_out_end,
                patch_size,
                n_patch,
            )
        return output_p - output_n

    def pseudo_forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight_p = weight[..., 0]
        weight_n = weight[..., 1]
        backend = self.config.conv_backend
        if backend == "jtc_emulation" and self.config.enable_jtc_batched_fast_path:
            return self._conv_forward_jtc_batched_signed(x, weight_p, weight_n)
        output_p = self.conv_forward(x, weight_p)
        output_n = self.conv_forward(x, weight_n)
        return output_p - output_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hv_concat:
            conv_h = self.pseudo_forward(x, self.weight)
            conv_v = self.pseudo_forward(x.permute(0, 1, 3, 2), self.weight).permute(
                0, 1, 3, 2
            )
            conv_stacked = torch.stack([conv_h, conv_v], dim=2)
            return conv_stacked.view(conv_h.size(0), -1, conv_h.size(2), conv_h.size(3))
        if self.vertical:
            return self.pseudo_forward(x.permute(0, 1, 3, 2), self.weight).permute(
                0, 1, 3, 2
            )
        return self.pseudo_forward(x, self.weight)
