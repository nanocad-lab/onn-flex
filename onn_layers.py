import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

__all__ = ["PIC", "FTconvlayer"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_8(x: int, name: str):
    if x != 8:
        raise ValueError(f"{name} length must be 8 for this implementation. Got {x}.")


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
        self.weights = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size, 2)
        )
        self.cout_per_cin = out_channels // groups
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
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
        plane_size: int = 48,
        sep: int = 8,
    ):
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
        )
        self.vertical = vertical
        self.hv_concat = hv_concat
        self.PIC_CONV = PIC(plane_size, sep)

    # ---------------- Internal helpers ------------------
    def hardware_forward(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return self.PIC_CONV(input, weight)

    def conv_forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        w = input.shape[2]
        output = torch.zeros(
            input_shape[0], self.out_channels, w, w, device=input.device
        )
        input = input.permute(0, 3, 1, 2)
        for c_in in range(input.shape[2]):
            input_c = input[:, :, c_in : c_in + 1, ...]
            weight_c = weight[c_in, ...]
            n_patch = int(input.shape[3] / 8)
            c_out_start = (c_in % self.groups) * self.cout_per_cin
            c_out_end = c_out_start + self.cout_per_cin
            for i_p in range(n_patch):
                patch = input_c[..., 8 * i_p : 8 * i_p + 8]
                system_out = self.hardware_forward(patch, weight_c).permute(0, 2, 3, 1)
                output[:, c_out_start:c_out_end, 8 * i_p : 8 * i_p + 8, :] += system_out
        return output

    def pseudo_forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight_p = weight[..., 0]
        weight_n = weight[..., 1]
        output_p = self.conv_forward(input, weight_p)
        output_n = self.conv_forward(input, weight_n)
        return output_p - output_n

    def forward(self, input: torch.Tensor):  # type: ignore[override]
        if self.hv_concat:
            conv_h = self.pseudo_forward(input, self.weights)
            conv_v = self.pseudo_forward(
                input.permute(0, 1, 3, 2), self.weights
            ).permute(0, 1, 3, 2)
            conv_stacked = torch.stack([conv_h, conv_v], dim=2)
            return conv_stacked.view(conv_h.size(0), -1, conv_h.size(2), conv_h.size(3))
        if self.vertical:
            return self.pseudo_forward(input.permute(0, 1, 3, 2), self.weights).permute(
                0, 1, 3, 2
            )
        return self.pseudo_forward(input, self.weights)
